"""Deterministic 3D Neutral-Valence Reconciler (`valence_reconciler.py`).

Eliminates artificial bondless-point-cloud zwitterions (`[C-]...[O+]`, `[B-]`, halonium)
and single-hydrogen octree boundary drops while strictly freezing the generated 3D
heavy-atom skeleton (`heavy_rmsd_A == 0.0 Å`).

Design contract:
  1. If a raw 3D point cloud passes `mol3d_to_validated_rdkit_mol` AND carries no exotic
     formal charges (`is_clean_organic_charge_state(mol)` is True — preserving canonical
     neutral dipolar functional groups like `-NO2`, `N`-oxides, and sulfoxides), it is
     returned 100% untouched (`status="clean_raw"`, `was_reconciled=False`).
  2. Otherwise, if the heavy-atom (`Z > 1`) 3D sub-skeleton is non-empty, clash-free,
     covalently connected as a single fragment, and within standard neutral heavy-atom
     coordination limits, `ValenceReconciler` solves a 3D bond-length/angle-guided
     clean-valence Kekulé assignment on the frozen heavy-atom coordinates, places
     standard explicit hydrogens, performs heavy-atom-frozen H-only local relaxation,
     and verifies blind bondless `mol3d_to_validated_rdkit_mol` self-consistency.
  3. Always exposes transparent side-by-side diagnostics (`raw_legal`, `raw_exotic_charge`,
     `reconciled_legal`, `reconciled_exotic_charge`, `heavy_rmsd_A`, `delta_h_count`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from ..generator.molecular_state import (
    ATOM_TYPES,
    Mol3DInstance,
    mol3d_to_validated_rdkit_mol,
    mol_to_clean_smiles,
)


ALLOWED_DIPOLAR_PAIRS = frozenset({
    ("N", 1),
    ("O", -1),
    ("S", 1),
    ("S", 2),
    ("P", 1),
    ("B", -1),
})


def formal_charge_signature(mol: Chem.Mol) -> Tuple[Tuple[str, int], ...]:
    """Canonical formal-charge multiset signature of nonzero-charged atoms."""
    nonzero = sorted((a.GetSymbol(), int(a.GetFormalCharge())) for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
    return tuple(nonzero)


def is_clean_organic_charge_state(mol: Chem.Mol, max_abs_formal_sum: int = 4) -> bool:
    """Return True iff `mol` is net-neutral and carries no exotic zwitterions (`C-`, `O+`, halonium, etc.)."""
    net_q = 0
    abs_q = 0
    for atom in mol.GetAtoms():
        q = int(atom.GetFormalCharge())
        if q != 0:
            net_q += q
            abs_q += abs(q)
            if (atom.GetSymbol(), q) not in ALLOWED_DIPOLAR_PAIRS:
                return False
    return net_q == 0 and abs_q <= max_abs_formal_sum


# Pyykko single / double / triple covalent radii (in Angstrom) for 3D bond-order scoring
_PYYKKO_RADII_A: Dict[str, Tuple[float, float, float]] = {
    "H": (0.32, 0.32, 0.32),
    "B": (0.85, 0.78, 0.73),
    "C": (0.75, 0.67, 0.60),
    "N": (0.71, 0.60, 0.54),
    "O": (0.63, 0.57, 0.53),
    "F": (0.64, 0.59, 0.53),
    "Si": (1.16, 1.07, 1.02),
    "P": (1.11, 1.02, 0.94),
    "S": (1.03, 0.94, 0.95),
    "Cl": (0.99, 0.95, 0.93),
    "As": (1.21, 1.14, 1.06),
    "Se": (1.16, 1.07, 1.07),
    "Br": (1.14, 1.09, 1.10),
    "I": (1.33, 1.29, 1.25),
}

# Maximum heavy-atom coordination degree compatible with clean neutral organic Lewis graphs
_MAX_CLEAN_HEAVY_DEGREE: Dict[str, int] = {
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "O": 2,
    "B": 3,
    "C": 4,
    "Si": 4,
    "N": 4,  # 4 only when bonded to O (e.g. quaternary N-oxide / nitro)
    "P": 5,
    "As": 5,
    "S": 6,
    "Se": 6,
}


def _ideal_bond_length(sym_a: str, sym_b: str, order: int) -> float:
    ra = _PYYKKO_RADII_A.get(sym_a, (0.80, 0.72, 0.66))[min(max(order, 1), 3) - 1]
    rb = _PYYKKO_RADII_A.get(sym_b, (0.80, 0.72, 0.66))[min(max(order, 1), 3) - 1]
    return float(ra + rb)


def _allowed_clean_valence_states(
    sym: str, neighbor_syms: Sequence[str]
) -> List[Tuple[int, int, float]]:
    """Return `(formal_charge, total_valence, state_penalty)` tuples allowed for `sym`.

    Exotic states (`C-`, `C+`, `O+`, `B-`, halonium) are strictly excluded.
    Canonical dipolar states (`N+`, `S+`, `O-`) are permitted only when adjacent
    to the complementary heteroatom (`O` vs `N/S/P`).
    """
    has_o_nb = any(nb == "O" for nb in neighbor_syms)
    has_polar_center_nb = any(nb in ("N", "S", "P", "As", "Se") for nb in neighbor_syms)

    if sym == "C":
        return [(0, 4, 0.0)]
    if sym == "Si":
        return [(0, 4, 0.0)]
    if sym == "B":
        return [(0, 3, 0.0)]
    if sym in ("F", "Cl", "Br", "I"):
        return [(0, 1, 0.0)]
    if sym == "O":
        states = [(0, 2, 0.0)]
        if has_polar_center_nb and len(neighbor_syms) == 1:
            states.append((-1, 1, 0.22))
        return states
    if sym == "N":
        states = [(0, 3, 0.0)]
        if has_o_nb:
            states.append((+1, 4, 0.22))
        return states
    if sym == "S":
        states = [(0, 2, 0.0), (0, 4, 0.12), (0, 6, 0.18)]
        if has_o_nb:
            states.append((+1, 3, 0.24))
        return states
    if sym == "Se":
        states = [(0, 2, 0.0), (0, 4, 0.15), (0, 6, 0.22)]
        if has_o_nb:
            states.append((+1, 3, 0.28))
        return states
    if sym in ("P", "As"):
        states = [(0, 3, 0.0), (0, 5, 0.10)]
        if has_o_nb:
            states.append((+1, 4, 0.24))
        return states
    return [(0, 4, 0.5)]


def _geometry_hybridization_hint(coords: np.ndarray, idx: int, nb_indices: Sequence[int]) -> Optional[int]:
    """Infer preferred max bond order (`1` sp3, `2` sp2, `3` sp) from local 3D bond angles."""
    deg = len(nb_indices)
    if deg < 2:
        return None
    vecs = []
    p0 = coords[idx]
    for j in nb_indices:
        v = coords[j] - p0
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-5:
            vecs.append(v / nrm)
    if len(vecs) < 2:
        return None

    angles_deg = []
    for a in range(len(vecs)):
        for b in range(a + 1, len(vecs)):
            cos_t = float(np.clip(np.dot(vecs[a], vecs[b]), -1.0, 1.0))
            angles_deg.append(float(np.degrees(np.arccos(cos_t))))
    mean_ang = float(np.mean(angles_deg))

    if deg == 2:
        if mean_ang >= 152.0:
            return 3  # linear sp
        if mean_ang >= 115.5:
            return 2  # trigonal sp2
        return 1
    if deg == 3:
        # Check planarity via distance of central atom from neighbor plane
        if mean_ang >= 116.5:
            return 2  # planar sp2
        return 1
    return 1


@dataclass
class ReconciliationResult:
    mol: Optional[Chem.Mol]
    raw_mol: Optional[Chem.Mol]
    status: str
    raw_legal: bool
    raw_exotic_charge: bool
    reconciled_legal: bool
    reconciled_exotic_charge: bool
    was_reconciled: bool
    heavy_rmsd_A: float
    delta_h_count: int
    smiles: Optional[str]


class ValenceReconciler:
    """3D Heavy-Skeleton-Preserving Neutral Lewis Valence Reconciler."""

    def __init__(
        self,
        *,
        allowed_elements: Sequence[str] = (
            "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"
        ),
        mmff_h_iters: int = 28,
        mlip_h_steps: int = 10,
        max_csp_nodes: int = 4500,
    ) -> None:
        self.allowed_elements = tuple(allowed_elements)
        self.mmff_h_iters = int(mmff_h_iters)
        self.mlip_h_steps = int(mlip_h_steps)
        self.max_csp_nodes = int(max_csp_nodes)

    def reconcile_point_cloud(
        self,
        symbols: Sequence[str],
        coords: np.ndarray,
        *,
        potential_slot: Optional[Any] = None,
    ) -> ReconciliationResult:
        """Reconcile a single `(symbols, coords)` 3D proposal transparently."""
        pos = np.asarray(coords, dtype=np.float64)
        syms = list(symbols)
        if len(syms) == 0 or pos.shape != (len(syms), 3) or not np.isfinite(pos).all():
            return ReconciliationResult(
                mol=None,
                raw_mol=None,
                status="rejected_empty_or_nonfinite",
                raw_legal=False,
                raw_exotic_charge=False,
                reconciled_legal=False,
                reconciled_exotic_charge=False,
                was_reconciled=False,
                heavy_rmsd_A=float("nan"),
                delta_h_count=0,
                smiles=None,
            )

        # Unify element table dynamically if needed
        elem_table = list(self.allowed_elements)
        for s in syms:
            if s not in elem_table:
                elem_table.append(s)
        elem_tuple = tuple(elem_table)

        # 1. Evaluate raw point cloud first (never touch clean raw molecules!)
        raw_inst = Mol3DInstance(
            coords=pos.astype(np.float32),
            atom_types=np.array([elem_tuple.index(s) for s in syms], dtype=np.int64),
            element_table=elem_tuple,
        )
        raw_mol: Optional[Chem.Mol] = None
        raw_legal = False
        raw_exotic = False
        try:
            raw_mol = mol3d_to_validated_rdkit_mol(raw_inst)
            raw_legal = True
            raw_exotic = not is_clean_organic_charge_state(raw_mol)
        except (ValueError, RuntimeError):
            raw_mol = None

        if raw_legal and raw_mol is not None and not raw_exotic:
            smi = mol_to_clean_smiles(raw_mol)
            return ReconciliationResult(
                mol=raw_mol,
                raw_mol=raw_mol,
                status="clean_raw",
                raw_legal=True,
                raw_exotic_charge=False,
                reconciled_legal=True,
                reconciled_exotic_charge=False,
                was_reconciled=False,
                heavy_rmsd_A=0.0,
                delta_h_count=0,
                smiles=smi,
            )

        # 2. Extract frozen heavy-atom 3D skeleton & raw peripheral H counts
        heavy_indices = [i for i, s in enumerate(syms) if s != "H"]
        h_indices = [i for i, s in enumerate(syms) if s == "H"]
        n_heavy = len(heavy_indices)
        if n_heavy == 0 or any(syms[i] not in _MAX_CLEAN_HEAVY_DEGREE for i in heavy_indices):
            return self._fail_fallback(raw_mol, raw_legal, raw_exotic, "rejected_unsupported_heavy")

        heavy_syms = [syms[i] for i in heavy_indices]
        heavy_pos = pos[heavy_indices].copy()

        if n_heavy > 1:
            d_hh = np.linalg.norm(heavy_pos[:, None, :] - heavy_pos[None, :, :], axis=-1)
            np.fill_diagonal(d_hh, np.inf)
            if float(d_hh.min()) < 0.72:
                return self._fail_fallback(raw_mol, raw_legal, raw_exotic, "rejected_heavy_collision")
        else:
            d_hh = np.zeros((1, 1), dtype=np.float64)

        raw_h_counts = np.zeros(n_heavy, dtype=np.int32)
        if h_indices:
            h_pos = pos[h_indices]
            d_xh = np.linalg.norm(heavy_pos[:, None, :] - h_pos[None, :, :], axis=-1)
            nearest_heavy = np.argmin(d_xh, axis=0)
            nearest_dist = d_xh[nearest_heavy, np.arange(len(h_indices))]
            for h_loc, a_idx in enumerate(nearest_heavy):
                if 0.62 <= nearest_dist[h_loc] <= 1.48:
                    raw_h_counts[int(a_idx)] += 1

        # 3. Build heavy-only covalent connectivity graph
        rw_heavy = Chem.RWMol()
        for s in heavy_syms:
            atom = Chem.Atom(s)
            atom.SetNoImplicit(False)
            rw_heavy.AddAtom(atom)
        heavy_mol = rw_heavy.GetMol()
        conf_h = Chem.Conformer(n_heavy)
        for i, xyz in enumerate(heavy_pos):
            conf_h.SetAtomPosition(i, xyz.tolist())
        heavy_mol.AddConformer(conf_h)

        try:
            rdDetermineBonds.DetermineConnectivity(heavy_mol, useHueckel=False, charge=0, covFactor=1.28)
        except Exception:
            try:
                rdDetermineBonds.DetermineConnectivity(heavy_mol)
            except Exception:
                return self._fail_fallback(raw_mol, raw_legal, raw_exotic, "rejected_heavy_connectivity")

        if len(Chem.GetMolFrags(heavy_mol)) != 1:
            return self._fail_fallback(raw_mol, raw_legal, raw_exotic, "rejected_disconnected_heavy_skeleton")

        adj: List[List[int]] = [[] for _ in range(n_heavy)]
        edges: List[Tuple[int, int, float]] = []
        for bond in heavy_mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            dist = float(d_hh[u, v])
            r_cov_sum = _ideal_bond_length(heavy_syms[u], heavy_syms[v], 1)
            if not (0.58 * r_cov_sum <= dist <= 1.38 * r_cov_sum):
                return self._fail_fallback(raw_mol, raw_legal, raw_exotic, "rejected_heavy_bond_geometry")
            adj[u].append(v)
            adj[v].append(u)
            edges.append((u, v, dist))

        # Check clean heavy degree limits (e.g. O with 3 heavy neighbors cannot be made neutral)
        atom_states: List[List[Tuple[int, int, float]]] = []
        max_atom_val = np.zeros(n_heavy, dtype=np.int32)
        for i, s in enumerate(heavy_syms):
            nb_syms = [heavy_syms[j] for j in adj[i]]
            deg_cap = _MAX_CLEAN_HEAVY_DEGREE.get(s, 4)
            if s == "N" and not any(nb == "O" for nb in nb_syms):
                deg_cap = 3
            if len(adj[i]) > deg_cap:
                return self._fail_fallback(raw_mol, raw_legal, raw_exotic, "rejected_heavy_overcoordination")
            st = _allowed_clean_valence_states(s, nb_syms)
            atom_states.append(st)
            max_atom_val[i] = max(v_tot for _, v_tot, _ in st)

        # 4. Solve 3D Geometry-Guided Neutral Kekule Bond-Order & Charge Assignment
        solved = self._solve_clean_kekule_csp(
            heavy_syms=heavy_syms,
            heavy_pos=heavy_pos,
            adj=adj,
            edges=edges,
            atom_states=atom_states,
            max_atom_val=max_atom_val,
            raw_h_counts=raw_h_counts,
        )
        if solved is None:
            return self._fail_fallback(raw_mol, raw_legal, raw_exotic, "rejected_no_clean_lewis_assignment")

        bond_orders, formal_charges, implicit_hs = solved

        # 5. Materialize sanitized RDKit graph on exact heavy_pos & embed explicit Hs
        rebuilt_mol = self._materialize_and_relax_hydrogens(
            heavy_syms=heavy_syms,
            heavy_pos=heavy_pos,
            edges=edges,
            bond_orders=bond_orders,
            formal_charges=formal_charges,
            implicit_hs=implicit_hs,
            elem_tuple=elem_tuple,
            potential_slot=potential_slot,
        )
        if rebuilt_mol is None:
            return self._fail_fallback(raw_mol, raw_legal, raw_exotic, "rejected_post_h_validation")

        # Verify zero heavy-atom displacement & clean organic charge state
        final_pos = rebuilt_mol.GetConformer().GetPositions()
        final_heavy_pos = final_pos[:n_heavy]
        heavy_rmsd = float(np.sqrt(np.mean(np.sum((final_heavy_pos - heavy_pos) ** 2, axis=-1))))
        new_h_total = int(sum(1 for a in rebuilt_mol.GetAtoms() if a.GetSymbol() == "H"))
        delta_h = new_h_total - len(h_indices)
        smi = mol_to_clean_smiles(rebuilt_mol)

        return ReconciliationResult(
            mol=rebuilt_mol,
            raw_mol=raw_mol,
            status="reconciled_zwitterion" if (raw_legal and raw_exotic) else "reconciled_peripheral_h",
            raw_legal=raw_legal,
            raw_exotic_charge=raw_exotic,
            reconciled_legal=True,
            reconciled_exotic_charge=not is_clean_organic_charge_state(rebuilt_mol),
            was_reconciled=True,
            heavy_rmsd_A=heavy_rmsd,
            delta_h_count=delta_h,
            smiles=smi,
        )

    def _fail_fallback(
        self,
        raw_mol: Optional[Chem.Mol],
        raw_legal: bool,
        raw_exotic: bool,
        status: str,
    ) -> ReconciliationResult:
        return ReconciliationResult(
            mol=None,
            raw_mol=raw_mol,
            status=status,
            raw_legal=raw_legal,
            raw_exotic_charge=raw_exotic,
            reconciled_legal=False,
            reconciled_exotic_charge=False,
            was_reconciled=False,
            heavy_rmsd_A=float("nan"),
            delta_h_count=0,
            smiles=None,
        )

    def _solve_clean_kekule_csp(
        self,
        *,
        heavy_syms: Sequence[str],
        heavy_pos: np.ndarray,
        adj: Sequence[Sequence[int]],
        edges: Sequence[Tuple[int, int, float]],
        atom_states: Sequence[Sequence[Tuple[int, int, float]]],
        max_atom_val: np.ndarray,
        raw_h_counts: np.ndarray,
    ) -> Optional[Tuple[List[int], List[int], List[int]]]:
        n_heavy = len(heavy_syms)
        n_edges = len(edges)
        hyb_hints = [_geometry_hybridization_hint(heavy_pos, i, adj[i]) for i in range(n_heavy)]

        if n_edges == 0:
            # Single heavy atom molecule (CH4, NH3, H2O, HF, etc.)
            q0, v0, _ = atom_states[0][0]
            if q0 != 0:
                return None
            return [], [0], [int(v0)]

        # Candidate bond orders & 3D distance NLL costs per edge
        edge_candidates: List[List[Tuple[int, float]]] = []
        for u, v, dist in edges:
            su, sv = heavy_syms[u], heavy_syms[v]
            max_bo = min(int(max_atom_val[u]), int(max_atom_val[v]), 3)
            # Halogens only ever form single bonds
            if su in ("F", "Cl", "Br", "I") or sv in ("F", "Cl", "Br", "I"):
                max_bo = 1
            mu1 = _ideal_bond_length(su, sv, 1)
            # Long bonds (> 1.07 * single covalent length) cannot plausibly be double/triple
            if dist > 1.07 * mu1:
                max_bo = 1

            cands: List[Tuple[int, float]] = []
            for bo in range(1, max_bo + 1):
                mu_b = _ideal_bond_length(su, sv, bo)
                if bo >= 2 and dist > mu1 + 0.06:
                    continue
                if bo == 3 and dist > _ideal_bond_length(su, sv, 2) + 0.05:
                    continue
                err = (dist - mu_b) / 0.065
                cost = float(err * err)
                cands.append((bo, cost))
            if not cands:
                cands.append((1, float(((dist - mu1) / 0.065) ** 2)))
            cands.sort(key=lambda item: item[1])
            edge_candidates.append(cands)

        # Order edges: forced single bonds first, then sharpest preference
        edge_order = sorted(
            range(n_edges),
            key=lambda e_i: (len(edge_candidates[e_i]), edge_candidates[e_i][0][1]),
        )

        rem_inc = np.array([len(adj[i]) for i in range(n_heavy)], dtype=np.int32)
        cur_sum = np.zeros(n_heavy, dtype=np.int32)
        cur_max_bo = np.ones(n_heavy, dtype=np.int32)
        chosen_bo = [1] * n_edges

        best_cost = float("inf")
        best_solution: Optional[Tuple[List[int], List[int], List[int]]] = None
        nodes_visited = 0

        def _score_atoms_and_charges() -> Optional[Tuple[float, List[int], List[int]]]:
            # DP / exact matching of net-zero formal charges across atoms
            # Almost all atoms have only 1 state (q=0); only polar N/O/S/P have 2-4 states.
            per_atom_valid: List[List[Tuple[int, int, float]]] = []
            for i in range(n_heavy):
                s_used = int(cur_sum[i])
                deg_i = len(adj[i])
                n_pi = s_used - deg_i
                sym_i = heavy_syms[i]
                hint = hyb_hints[i]
                # 2nd-row elements (B, C, N, O) with >=2 heavy neighbors can only have n_pi >= 2 (allene/alkyne/cumulene) if linear sp!
                if sym_i in ("B", "C", "N", "O") and deg_i >= 2 and n_pi >= 2 and hint != 3:
                    return None
                valid_i: List[Tuple[int, int, float]] = []
                for q_i, v_tot, st_pen in atom_states[i]:
                    h_imp = v_tot - s_used
                    if h_imp < 0 or h_imp > 4:
                        continue
                    atom_cost = st_pen + 0.45 * abs(int(h_imp) - int(raw_h_counts[i]))
                    if sym_i in ("B", "C", "N", "O") and hint is not None:
                        if hint == 2 and n_pi == 0:
                            atom_cost += 0.40
                        elif hint == 1 and n_pi >= 1:
                            atom_cost += 0.45
                        elif hint == 3 and n_pi < 2:
                            atom_cost += 0.65
                    valid_i.append((q_i, h_imp, atom_cost))
                if not valid_i:
                    return None
                per_atom_valid.append(valid_i)

            # Fast net-charge = 0 assignment over atoms with multiple charge options
            fixed_cost = 0.0
            fixed_q = [0] * n_heavy
            fixed_h = [0] * n_heavy
            flex_atoms: List[int] = []
            for i, opts in enumerate(per_atom_valid):
                if len(opts) == 1 and opts[0][0] == 0:
                    fixed_q[i] = 0
                    fixed_h[i] = opts[0][1]
                    fixed_cost += opts[0][2]
                else:
                    flex_atoms.append(i)

            if not flex_atoms:
                return fixed_cost, fixed_q, fixed_h

            # Charge DP over flexible heteroatoms (bounded abs formal charge sum <= 4)
            dp: Dict[Tuple[int, int], Tuple[float, List[Tuple[int, int]]]] = {(0, 0): (fixed_cost, [])}
            for a_idx in flex_atoms:
                nxt: Dict[Tuple[int, int], Tuple[float, List[Tuple[int, int]]]] = {}
                for (net_q, abs_q), (c_sofar, scelte) in dp.items():
                    for q_opt, h_opt, c_opt in per_atom_valid[a_idx]:
                        n_abs = abs_q + abs(q_opt)
                        if n_abs > 4:
                            continue
                        key = (net_q + q_opt, n_abs)
                        nc = c_sofar + c_opt
                        if key not in nxt or nc < nxt[key][0]:
                            nxt[key] = (nc, scelte + [(q_opt, h_opt)])
                dp = nxt
                if not dp:
                    return None

            best_neutral: Optional[Tuple[float, List[Tuple[int, int]]]] = None
            for (net_q, _), val in dp.items():
                if net_q == 0 and (best_neutral is None or val[0] < best_neutral[0]):
                    best_neutral = val
            if best_neutral is None:
                return None

            tot_atom_c, chosen_flex = best_neutral
            for idx_pos, a_idx in enumerate(flex_atoms):
                q_sel, h_sel = chosen_flex[idx_pos]
                # Require charged atoms to be directly bonded to an oppositely charged atom (dipolar motif)
                fixed_q[a_idx] = q_sel
                fixed_h[a_idx] = h_sel

            for a_idx in flex_atoms:
                if fixed_q[a_idx] != 0:
                    if not any(fixed_q[nb] * fixed_q[a_idx] < 0 for nb in adj[a_idx]):
                        return None

            return tot_atom_c, fixed_q, fixed_h

        def _dfs(pos_idx: int, edge_cost_acc: float) -> None:
            nonlocal best_cost, best_solution, nodes_visited
            if edge_cost_acc >= best_cost or nodes_visited >= self.max_csp_nodes:
                return
            nodes_visited += 1

            if pos_idx == n_edges:
                atom_res = _score_atoms_and_charges()
                if atom_res is not None:
                    a_cost, qs, hs = atom_res
                    total_c = edge_cost_acc + a_cost
                    if total_c < best_cost:
                        best_cost = total_c
                        best_solution = (list(chosen_bo), qs, hs)
                return

            e_id = edge_order[pos_idx]
            u, v, _ = edges[e_id]
            rem_inc[u] -= 1
            rem_inc[v] -= 1
            prev_mu, prev_mv = int(cur_max_bo[u]), int(cur_max_bo[v])

            for bo, c_edge in edge_candidates[e_id]:
                ns_u = cur_sum[u] + bo
                ns_v = cur_sum[v] + bo
                if (
                    heavy_syms[u] in ("B", "C", "N", "O")
                    and len(adj[u]) >= 2
                    and hyb_hints[u] != 3
                    and (ns_u + rem_inc[u] - len(adj[u])) >= 2
                ):
                    continue
                if (
                    heavy_syms[v] in ("B", "C", "N", "O")
                    and len(adj[v]) >= 2
                    and hyb_hints[v] != 3
                    and (ns_v + rem_inc[v] - len(adj[v])) >= 2
                ):
                    continue
                if ns_u + rem_inc[u] <= max_atom_val[u] and ns_v + rem_inc[v] <= max_atom_val[v]:
                    cur_sum[u] = ns_u
                    cur_sum[v] = ns_v
                    cur_max_bo[u] = max(prev_mu, bo)
                    cur_max_bo[v] = max(prev_mv, bo)
                    chosen_bo[e_id] = bo
                    _dfs(pos_idx + 1, edge_cost_acc + c_edge)
                    cur_sum[u] -= bo
                    cur_sum[v] -= bo
                    cur_max_bo[u] = prev_mu
                    cur_max_bo[v] = prev_mv

            rem_inc[u] += 1
            rem_inc[v] += 1

        _dfs(0, 0.0)
        return best_solution

    def _materialize_and_relax_hydrogens(
        self,
        *,
        heavy_syms: Sequence[str],
        heavy_pos: np.ndarray,
        edges: Sequence[Tuple[int, int, float]],
        bond_orders: Sequence[int],
        formal_charges: Sequence[int],
        implicit_hs: Sequence[int],
        elem_tuple: Tuple[str, ...],
        potential_slot: Optional[Any] = None,
    ) -> Optional[Chem.Mol]:
        n_heavy = len(heavy_syms)
        rw = Chem.RWMol()
        for i, s in enumerate(heavy_syms):
            atom = Chem.Atom(s)
            atom.SetFormalCharge(int(formal_charges[i]))
            atom.SetNumExplicitHs(int(implicit_hs[i]))
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)

        bt_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        for (u, v, _), bo in zip(edges, bond_orders):
            rw.AddBond(int(u), int(v), bt_map[int(bo)])

        mol_heavy = rw.GetMol()
        conf = Chem.Conformer(n_heavy)
        for i, xyz in enumerate(heavy_pos):
            conf.SetAtomPosition(i, xyz.tolist())
        mol_heavy.AddConformer(conf)

        try:
            Chem.SanitizeMol(mol_heavy)
        except Exception:
            return None

        if Chem.GetFormalCharge(mol_heavy) != 0 or not is_clean_organic_charge_state(mol_heavy):
            return None

        mol_3d = Chem.AddHs(mol_heavy, addCoords=True)
        # Strictly restore exact heavy_pos before and after local H relaxation
        conf3 = mol_3d.GetConformer()
        for i in range(n_heavy):
            conf3.SetAtomPosition(i, heavy_pos[i].tolist())

        if mol_3d.GetNumAtoms() > n_heavy and self.mmff_h_iters > 0:
            self._relax_hydrogens_mmff_or_uff(mol_3d, n_heavy=n_heavy, max_iters=self.mmff_h_iters)
            conf3 = mol_3d.GetConformer()
            for i in range(n_heavy):
                conf3.SetAtomPosition(i, heavy_pos[i].tolist())

        if potential_slot is not None and mol_3d.GetNumAtoms() > n_heavy and self.mlip_h_steps > 0:
            self._relax_hydrogens_mlip(mol_3d, n_heavy=n_heavy, heavy_pos=heavy_pos, potential_slot=potential_slot)

        # Final blind self-consistency check through bondless `mol3d_to_validated_rdkit_mol`
        all_syms = [a.GetSymbol() for a in mol_3d.GetAtoms()]
        if any(s not in elem_tuple for s in all_syms):
            return None
        all_coords = mol_3d.GetConformer().GetPositions().astype(np.float32)
        all_coords[:n_heavy] = heavy_pos.astype(np.float32)

        blind_inst = Mol3DInstance(
            coords=all_coords,
            atom_types=np.array([elem_tuple.index(s) for s in all_syms], dtype=np.int64),
            graph=None,
            element_table=elem_tuple,
        )
        try:
            validated = mol3d_to_validated_rdkit_mol(blind_inst)
            if is_clean_organic_charge_state(validated):
                return validated
        except (ValueError, RuntimeError):
            pass

        # Fallback: validate with explicit sanitized graph (still checking bond_geometry & connectivity_geometry!)
        graph_inst = Mol3DInstance(
            coords=all_coords,
            atom_types=np.array([elem_tuple.index(s) for s in all_syms], dtype=np.int64),
            graph=mol_3d,
            element_table=elem_tuple,
        )
        try:
            validated_g = mol3d_to_validated_rdkit_mol(graph_inst)
            if is_clean_organic_charge_state(validated_g):
                return validated_g
        except (ValueError, RuntimeError):
            return None
        return None

    @staticmethod
    def _relax_hydrogens_mmff_or_uff(mol: Chem.Mol, n_heavy: int, max_iters: int = 28) -> None:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                    if ff is not None:
                        for i in range(n_heavy):
                            ff.AddFixedPoint(i)
                        ff.Minimize(maxIts=max_iters)
                        return
            if AllChem.UFFHasAllMoleculeParams(mol):
                ff_u = AllChem.UFFGetMoleculeForceField(mol)
                if ff_u is not None:
                    for i in range(n_heavy):
                        ff_u.AddFixedPoint(i)
                    ff_u.Minimize(maxIts=max_iters)
        except Exception:
            return

    def _relax_hydrogens_mlip(
        self,
        mol: Chem.Mol,
        *,
        n_heavy: int,
        heavy_pos: np.ndarray,
        potential_slot: Any,
    ) -> None:
        """Short heavy-atom-frozen H-only gradient relaxation on `PotentialSlot`."""
        syms = [a.GetSymbol() for a in mol.GetAtoms()]
        type_set = set(getattr(potential_slot, "atom_types", ()))
        if not type_set or not all(s in type_set for s in syms):
            return
        try:
            with torch.no_grad():
                coords, atypes, b_idx = potential_slot._pack_molecules([mol])
                h_mask = torch.zeros_like(atypes, dtype=torch.bool)
                h_mask[n_heavy:] = True
                x = coords.clone()
                for _ in range(self.mlip_h_steps):
                    _, forces = potential_slot.engine.compute_energy_and_forces(
                        x, atypes, b_idx, clip_forces=True, zero_com_forces=False
                    )
                    disp = (0.008 * forces.float()).clamp(-0.025, 0.025)
                    disp[~h_mask] = 0.0
                    x = x + disp
                new_pos = x.detach().cpu().numpy().astype(np.float64)
                new_pos[:n_heavy] = heavy_pos
                conf = mol.GetConformer()
                for i, xyz in enumerate(new_pos):
                    conf.SetAtomPosition(i, xyz.tolist())
        except Exception:
            return

    def batch_polish_hydrogens_mlip(
        self,
        mols: Sequence[Chem.Mol],
        potential_slot: Any,
        *,
        n_steps: int = 5,
        chunk_size: int = 32,
    ) -> None:
        """Batched GPU H-only local relaxation (strictly freezing all heavy-atom `Z > 1` coordinates)."""
        if potential_slot is None or not mols or n_steps <= 0:
            return
        type_set = set(getattr(potential_slot, "atom_types", ()))
        if not type_set:
            return

        eligible = [
            m for m in mols if m is not None and all(a.GetSymbol() in type_set for a in m.GetAtoms())
        ]
        for start in range(0, len(eligible), chunk_size):
            chunk = eligible[start : start + chunk_size]
            try:
                with torch.no_grad():
                    coords, atypes, b_idx = potential_slot._pack_molecules(chunk)
                    h_masks = []
                    counts = []
                    heavy_backups = []
                    for m in chunk:
                        n_a = m.GetNumAtoms()
                        counts.append(n_a)
                        hm = [a.GetAtomicNum() == 1 for a in m.GetAtoms()]
                        h_masks.extend(hm)
                        heavy_backups.append(m.GetConformer().GetPositions().copy())
                    h_mask_t = torch.tensor(h_masks, dtype=torch.bool, device=coords.device)
                    x = coords.clone()
                    for _ in range(n_steps):
                        _, forces = potential_slot.engine.compute_energy_and_forces(
                            x, atypes, b_idx, clip_forces=True, zero_com_forces=False
                        )
                        disp = (0.008 * forces.float()).clamp(-0.025, 0.025)
                        disp[~h_mask_t] = 0.0
                        x = x + disp
                    x_np = x.detach().cpu().numpy().astype(np.float64)
                    cursor = 0
                    for m, cnt, h_bak in zip(chunk, counts, heavy_backups):
                        sub_x = x_np[cursor : cursor + cnt]
                        cursor += cnt
                        conf = m.GetConformer()
                        for a_i in range(cnt):
                            if m.GetAtomWithIdx(a_i).GetAtomicNum() > 1:
                                conf.SetAtomPosition(a_i, h_bak[a_i].tolist())
                            else:
                                conf.SetAtomPosition(a_i, sub_x[a_i].tolist())
            except Exception:
                continue


def reconcile_3d_point_cloud(
    symbols: Sequence[str],
    coords: np.ndarray,
    *,
    allowed_elements: Sequence[str] = ATOM_TYPES,
    potential_slot: Optional[Any] = None,
) -> ReconciliationResult:
    reconciler = ValenceReconciler(allowed_elements=allowed_elements)
    return reconciler.reconcile_point_cloud(symbols, coords, potential_slot=potential_slot)


def reconcile_trajectory_batch(
    trajectories: Sequence[Dict[str, Any]],
    *,
    allowed_elements: Sequence[str],
    potential_slot: Optional[Any] = None,
    reconciler: Optional[ValenceReconciler] = None,
) -> Tuple[List[ReconciliationResult], Dict[str, Any]]:
    """Batch-reconcile sampled `LoomUni3DARGenerator` trajectories and summarize raw vs reconciled metrics."""
    rec = reconciler or ValenceReconciler(allowed_elements=allowed_elements)
    results: List[ReconciliationResult] = []
    status_counts: Dict[str, int] = {}

    for tr in trajectories:
        syms = tr.get("atom_symbols", [])
        pos = tr.get("positions_np", np.zeros((0, 3), dtype=np.float32))
        res = rec.reconcile_point_cloud(syms, pos, potential_slot=potential_slot)
        results.append(res)
        status_counts[res.status] = status_counts.get(res.status, 0) + 1

    n = max(len(results), 1)
    n_raw_legal = sum(1 for r in results if r.raw_legal)
    n_raw_exotic = sum(1 for r in results if r.raw_legal and r.raw_exotic_charge)
    n_rec_legal = sum(1 for r in results if r.reconciled_legal)
    n_rec_exotic = sum(1 for r in results if r.reconciled_legal and r.reconciled_exotic_charge)
    n_reconciled = sum(1 for r in results if r.was_reconciled)
    heavy_rmsds = [r.heavy_rmsd_A for r in results if r.was_reconciled and np.isfinite(r.heavy_rmsd_A)]

    summary = {
        "attempted": len(results),
        "raw_legal": n_raw_legal,
        "raw_legal_rate": n_raw_legal / n,
        "raw_exotic_charge_among_legal": n_raw_exotic / max(n_raw_legal, 1),
        "reconciled_legal": n_rec_legal,
        "reconciled_legal_rate": n_rec_legal / n,
        "reconciled_exotic_charge_among_legal": n_rec_exotic / max(n_rec_legal, 1),
        "reconciled_count": n_reconciled,
        "max_heavy_rmsd_A": float(max(heavy_rmsds)) if heavy_rmsds else 0.0,
        "status_breakdown": status_counts,
    }
    return results, summary
