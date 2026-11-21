"""Unified full-periodic-table chemistry vocabulary, physical descriptors, and RDKit utilities for FlowMol."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from rdkit import Chem
import torch

# Full 118-element periodic table vocabulary (H..Og, index = atomic_number - 1),
# matching Uni-3DAR (`loom.vendor.uni3dar.data.atom_data.atom_list`) slot-for-slot.
# Dedicated [MASK] index is strictly N_ATOM_TYPES (= 118).
FLOWMOL_ATOM_TYPES: Tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
ORGANIC_16_ATOM_TYPES: Tuple[str, ...] = (
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I", "Li"
)
ATOM_TO_IDX: Dict[str, int] = {sym: idx for idx, sym in enumerate(FLOWMOL_ATOM_TYPES)}
N_ATOM_TYPES: int = len(FLOWMOL_ATOM_TYPES)
ATOM_MASK_IDX: int = N_ATOM_TYPES

# Full-atom formal charge range (-4..+6) covering organic zwitterions, main-group hypervalency,
# and transition-metal / lanthanide oxidation states; dedicated [MASK] index = 11.
FLOWMOL_FORMAL_CHARGES: Tuple[int, ...] = (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)
LEGACY_6_FORMAL_CHARGES: Tuple[int, ...] = (-2, -1, 0, 1, 2, 3)
CHARGE_TO_IDX: Dict[int, int] = {q: idx for idx, q in enumerate(FLOWMOL_FORMAL_CHARGES)}
N_CHARGES: int = len(FLOWMOL_FORMAL_CHARGES)
CHARGE_MASK_IDX: int = N_CHARGES

# 0 = no bond, 1 = single, 2 = double, 3 = triple, 4 = aromatic, 5 = coordinate/dative; 6 = [MASK]
FLOWMOL_BOND_TYPES: Tuple[str, ...] = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "COORDINATE")
LEGACY_5_BOND_TYPES: Tuple[str, ...] = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
N_BOND_TYPES: int = len(FLOWMOL_BOND_TYPES)
BOND_MASK_IDX: int = N_BOND_TYPES

RDKIT_BOND_TO_IDX: Dict[Chem.rdchem.BondType, int] = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.QUADRUPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
    Chem.rdchem.BondType.DATIVE: 5,
    Chem.rdchem.BondType.DATIVEONE: 5,
    Chem.rdchem.BondType.ZERO: 5,
}
IDX_TO_RDKIT_BOND: Dict[int, Chem.rdchem.BondType] = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
    5: Chem.rdchem.BondType.DATIVE,
}

# Strict non-metals / metalloids whose standard covalent valences RDKit checks cleanly;
# all other Z in 1..118 are metals (s/d/f/post-transition) that act as coordination acceptors.
NON_METAL_ATOMIC_NUMS = frozenset({
    1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 33, 34, 35, 36, 52, 53, 54, 85, 86, 117, 118
})

PERIODIC_FEATURE_DIM: int = 34


def _element_period_group_block(z: int) -> Tuple[int, int, int]:
    """Return `(period 1..7, iupac_group 0..18 [0=f-block], block 0=s,1=p,2=d,3=f)` for Z in 1..118."""
    if z <= 2:
        return 1, (1 if z == 1 else 18), 0
    if z <= 10:
        g = (z - 2) if z <= 4 else (z + 8)
        return 2, g, (0 if z <= 4 else 1)
    if z <= 18:
        g = (z - 10) if z <= 12 else (z + 0)
        return 3, g, (0 if z <= 12 else 1)
    if z <= 36:
        g = z - 18
        blk = 0 if g <= 2 else (2 if g <= 12 else 1)
        return 4, g, blk
    if z <= 54:
        g = z - 36
        blk = 0 if g <= 2 else (2 if g <= 12 else 1)
        return 5, g, blk
    if z <= 86:
        if z <= 56:
            return 6, z - 54, 0
        if z <= 71:
            return 6, 0, 3  # Lanthanides (f-block)
        g = z - 68
        return 6, g, (2 if g <= 12 else 1)
    if z <= 88:
        return 7, z - 86, 0
    if z <= 103:
        return 7, 0, 3  # Actinides (f-block)
    g = z - 100
    return 7, g, (2 if g <= 12 else 1)


@lru_cache(maxsize=1)
def _periodic_feature_matrix_np() -> np.ndarray:
    """Build `(119, 34)` float32 periodic-table invariant descriptor table (`row 118 = [MASK] = zeros`).

    Columns:
      [0]      : Z / 118.0
      [1..8]   : period 1..7 one-hot (7)
      [8..27]  : group 0..18 one-hot (19; 0 = lanthanide/actinide f-block)
      [27..31] : block s/p/d/f one-hot (4)
      [31]     : covalent radius / 2.5 A
      [32]     : van der Waals radius / 3.0 A
      [33]     : outer valence electrons / 8.0
    """
    pt = Chem.GetPeriodicTable()
    mat = np.zeros((N_ATOM_TYPES + 1, PERIODIC_FEATURE_DIM), dtype=np.float32)
    for idx in range(N_ATOM_TYPES):
        z = idx + 1
        period, group, blk = _element_period_group_block(z)
        rcov = float(pt.GetRcovalent(z))
        rvdw = float(pt.GetRvdw(z))
        n_outer = float(max(pt.GetNOuterElecs(z), 0))
        mat[idx, 0] = float(z) / 118.0
        mat[idx, 1 + (period - 1)] = 1.0
        mat[idx, 8 + group] = 1.0
        mat[idx, 27 + blk] = 1.0
        mat[idx, 31] = (rcov if rcov > 0 else 1.5) / 2.5
        mat[idx, 32] = (rvdw if rvdw > 0 else 2.0) / 3.0
        mat[idx, 33] = min(n_outer, 12.0) / 8.0
    return mat


def build_periodic_feature_tensor(atom_types: Sequence[str] = FLOWMOL_ATOM_TYPES) -> torch.Tensor:
    """Return `(len(atom_types) + 1, 34)` float32 periodic-table descriptor tensor (last row = [MASK] zeros)."""
    full_np = _periodic_feature_matrix_np()
    if tuple(atom_types) == FLOWMOL_ATOM_TYPES:
        return torch.from_numpy(full_np.copy())
    pt = Chem.GetPeriodicTable()
    out = np.zeros((len(atom_types) + 1, PERIODIC_FEATURE_DIM), dtype=np.float32)
    for i, sym in enumerate(atom_types):
        z = int(pt.GetAtomicNumber(str(sym)))
        if 1 <= z <= N_ATOM_TYPES:
            out[i] = full_np[z - 1]
    return torch.from_numpy(out)


def build_allowed_element_mask(
    allowed_elements: Optional[Sequence[str]],
    atom_types: Sequence[str] = FLOWMOL_ATOM_TYPES,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Return a boolean tensor `(len(atom_types),)` with True for allowed elements."""
    if allowed_elements is None:
        return None
    allowed_set = set(allowed_elements)
    mask = torch.tensor([sym in allowed_set for sym in atom_types], dtype=torch.bool, device=device)
    if not mask.any():
        raise ValueError(f"No allowed elements {allowed_elements} match vocabulary {atom_types}")
    return mask


def build_allowed_charge_mask(
    allowed_charges: Optional[Sequence[int]],
    charge_vocab: Sequence[int] = FLOWMOL_FORMAL_CHARGES,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Return a boolean tensor `(len(charge_vocab),)` with True for allowed formal charges."""
    if allowed_charges is None:
        return None
    allowed_set = {int(q) for q in allowed_charges}
    mask = torch.tensor([int(q) in allowed_set for q in charge_vocab], dtype=torch.bool, device=device)
    if not mask.any():
        raise ValueError(f"No allowed charges {allowed_charges} match vocabulary {charge_vocab}")
    return mask


def _is_metal_atomic_num(z: int) -> bool:
    return z not in NON_METAL_ATOMIC_NUMS


def _orient_coordinate_pair(u: int, v: int, z_u: int, z_v: int) -> Tuple[int, int]:
    """Return `(donor_idx, acceptor_idx)` for an undirected `COORDINATE` (`DATIVE`) bond."""
    m_u = _is_metal_atomic_num(z_u)
    m_v = _is_metal_atomic_num(z_v)
    if m_v and not m_u:
        return u, v  # Ligand -> Metal
    if m_u and not m_v:
        return v, u  # Ligand -> Metal
    # Main-group Lewis base (N, P, As, O, S, Se, F, Cl, Br, I, C) -> Lewis acid (B, Si, Ge, Te)
    lewis_acids = {5, 14, 32, 52}
    if z_v in lewis_acids and z_u not in lewis_acids:
        return u, v
    if z_u in lewis_acids and z_v not in lewis_acids:
        return v, u
    return (u, v) if z_u >= z_v else (v, u)


def mol_to_sparse_record(
    mol: Optional[Chem.Mol],
    *,
    atom_to_idx: Dict[str, int] = ATOM_TO_IDX,
    charge_to_idx: Dict[int, int] = CHARGE_TO_IDX,
    max_atoms: int = 128,
    min_atoms: int = 2,
    max_diameter: float = 50.0,
    min_interatomic_dist: float = 0.45,
) -> Optional[Dict[str, np.ndarray]]:
    """Convert an RDKit 3D molecule (organic or organometallic, explicit Hs) into a zero-COM sparse record."""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    n_atoms = int(mol.GetNumAtoms())
    if n_atoms < min_atoms or n_atoms > max_atoms or n_atoms > 255:
        return None

    atom_idx = np.empty(n_atoms, dtype=np.uint8)
    charge_idx = np.empty(n_atoms, dtype=np.uint8)
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        if sym not in atom_to_idx:
            return None
        q = int(atom.GetFormalCharge())
        if q not in charge_to_idx:
            return None
        atom_idx[i] = atom_to_idx[sym]
        charge_idx[i] = charge_to_idx[q]

    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    if pos.shape != (n_atoms, 3) or not np.isfinite(pos).all():
        return None
    pos = pos - pos.mean(axis=0, keepdims=True)

    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    if float(np.sqrt(dist_sq.max())) > max_diameter:
        return None
    np.fill_diagonal(dist_sq, np.inf)
    if float(np.sqrt(dist_sq.min())) < min_interatomic_dist:
        return None

    bond_u: List[int] = []
    bond_v: List[int] = []
    bond_t: List[int] = []
    for bond in mol.GetBonds():
        bt = RDKIT_BOND_TO_IDX.get(bond.GetBondType())
        if bt is None:
            return None
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())
        if u == v:
            continue
        if u > v:
            u, v = v, u
        bond_u.append(u)
        bond_v.append(v)
        bond_t.append(bt)

    return {
        "pos": pos.astype(np.float32, copy=False),
        "atom_type": atom_idx,
        "charge": charge_idx,
        "bond_u": np.asarray(bond_u, dtype=np.uint8),
        "bond_v": np.asarray(bond_v, dtype=np.uint8),
        "bond_type": np.asarray(bond_t, dtype=np.uint8),
    }


def _sanitize_organometallic_fallback(mol: Chem.Mol) -> bool:
    """Validate an organometallic complex when RDKit's main-group valence table rejects a d/f-metal center.

    Requires:
      1. At least one metal atom is present and has <= 16 total bonds (covering up to lanthanocene Cp3LnL);
      2. Every non-metal ligand atom passes strict RDKit valence/aromaticity sanitization when metal
         coordination bonds are detached!
    """
    has_metal = False
    metal_indices = set()
    for atom in mol.GetAtoms():
        z = int(atom.GetAtomicNum())
        if _is_metal_atomic_num(z):
            has_metal = True
            metal_indices.add(atom.GetIdx())
            if len(atom.GetBonds()) > 16:
                return False
    if not has_metal:
        return False

    # Verify all non-metal ligand subgraphs strictly pass standard RDKit sanitization!
    rw_lig = Chem.RWMol(mol)
    for m_idx in sorted(metal_indices, reverse=True):
        rw_lig.RemoveAtom(m_idx)
    if rw_lig.GetNumAtoms() > 0:
        try:
            lig_mol = rw_lig.GetMol()
            for a in lig_mol.GetAtoms():
                a.SetNoImplicit(True)
            Chem.SanitizeMol(lig_mol)
        except Exception:
            return False

    # Sanitize rings/aromaticity/symmSSSR on the full complex without main-group property overflow
    flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
    Chem.SanitizeMol(mol, flags)
    mol.UpdatePropertyCache(strict=False)
    return True


def tensors_to_rdkit_mol(
    positions: np.ndarray | torch.Tensor,
    atom_types: np.ndarray | torch.Tensor,
    charges: np.ndarray | torch.Tensor,
    upper_bond_types: np.ndarray | torch.Tensor,
    *,
    atom_vocab: Sequence[str] = FLOWMOL_ATOM_TYPES,
    charge_vocab: Sequence[int] = FLOWMOL_FORMAL_CHARGES,
    sanitize: bool = True,
) -> Tuple[Optional[Chem.Mol], Dict[str, object]]:
    """Build an RDKit Mol (organic or organometallic) from sampled tensors and return diagnostics."""
    pos_np = np.asarray(positions.detach().cpu() if isinstance(positions, torch.Tensor) else positions, dtype=np.float64)
    a_np = np.asarray(atom_types.detach().cpu() if isinstance(atom_types, torch.Tensor) else atom_types, dtype=np.int64)
    c_np = np.asarray(charges.detach().cpu() if isinstance(charges, torch.Tensor) else charges, dtype=np.int64)
    e_np = np.asarray(
        upper_bond_types.detach().cpu() if isinstance(upper_bond_types, torch.Tensor) else upper_bond_types,
        dtype=np.int64,
    )

    n_atoms = int(pos_np.shape[0])
    diag: Dict[str, object] = {
        "n_atoms": n_atoms,
        "has_mask_token": False,
        "valid": False,
        "connected": False,
        "min_dist": float("nan"),
        "smiles": None,
    }
    if n_atoms < 2 or not np.isfinite(pos_np).all():
        return None, diag

    dmat = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    diag["min_dist"] = float(dmat.min())

    if (a_np >= len(atom_vocab)).any() or (a_np < 0).any() or (c_np >= len(charge_vocab)).any() or (c_np < 0).any():
        diag["has_mask_token"] = True
        return None, diag
    if (e_np >= N_BOND_TYPES).any() or (e_np < 0).any():
        diag["has_mask_token"] = True
        return None, diag

    pt = Chem.GetPeriodicTable()
    rw = Chem.RWMol()
    atomic_nums: List[int] = []
    for i in range(n_atoms):
        sym = str(atom_vocab[int(a_np[i])])
        atom = Chem.Atom(sym)
        atom.SetFormalCharge(int(charge_vocab[int(c_np[i])]))
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
        atomic_nums.append(int(pt.GetAtomicNumber(sym)))

    tri_u, tri_v = np.triu_indices(n_atoms, k=1)
    for k in range(len(tri_u)):
        bt_id = int(e_np[k])
        if bt_id == 0:
            continue
        u, v = int(tri_u[k]), int(tri_v[k])
        if bt_id == 5:
            donor, acceptor = _orient_coordinate_pair(u, v, atomic_nums[u], atomic_nums[v])
            rw.AddBond(donor, acceptor, Chem.rdchem.BondType.DATIVE)
        else:
            rd_bt = IDX_TO_RDKIT_BOND.get(bt_id)
            if rd_bt is not None:
                if bt_id == 4:
                    rw.GetAtomWithIdx(u).SetIsAromatic(True)
                    rw.GetAtomWithIdx(v).SetIsAromatic(True)
                rw.AddBond(u, v, rd_bt)

    mol = rw.GetMol()
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, (float(pos_np[i, 0]), float(pos_np[i, 1]), float(pos_np[i, 2])))
    mol.AddConformer(conf, assignId=True)

    if not sanitize:
        diag["valid"] = True
        return mol, diag

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            mol_fb = Chem.Mol(mol)
            if not _sanitize_organometallic_fallback(mol_fb):
                return None, diag
            mol = mol_fb
        except Exception:
            return None, diag

    try:
        frags = Chem.GetMolFrags(mol)
        diag["valid"] = True
        diag["connected"] = len(frags) == 1
        diag["has_metal"] = bool(any(_is_metal_atomic_num(z) for z in atomic_nums))
        diag["has_coordinate_bond"] = bool((e_np == 5).any())
        diag["elements"] = sorted({str(atom_vocab[int(x)]) for x in a_np.tolist()})
        diag["smiles"] = Chem.MolToSmiles(mol, canonical=True)
        return mol, diag
    except Exception:
        return None, diag


def evaluate_sampled_molecules(
    samples: Iterable[Tuple[Optional[Chem.Mol], Dict[str, object]]],
) -> Dict[str, float]:
    """Summarize RDKit validity, connectivity, uniqueness, element diversity, and min-distance distribution."""
    diags = []
    valid_smiles = []
    min_dists = []
    distinct_elems = set()
    for mol, d in samples:
        diags.append(d)
        if np.isfinite(d["min_dist"]):
            min_dists.append(float(d["min_dist"]))
        if d["valid"] and d["smiles"]:
            valid_smiles.append(str(d["smiles"]))
            for sym in d.get("elements", ()):
                distinct_elems.add(str(sym))

    total = max(len(diags), 1)
    n_valid = sum(1 for d in diags if d["valid"])
    n_conn = sum(1 for d in diags if d["valid"] and d["connected"])
    n_masked = sum(1 for d in diags if d["has_mask_token"])
    n_metal = sum(1 for d in diags if d.get("valid") and d.get("has_metal"))
    n_coord = sum(1 for d in diags if d.get("valid") and d.get("has_coordinate_bond"))
    n_unique = len(set(valid_smiles))
    min_d_arr = np.asarray(min_dists, dtype=np.float64) if min_dists else np.asarray([float("nan")])
    return {
        "n_sampled": float(len(diags)),
        "valid_rate": float(n_valid / total),
        "valid_connected_rate": float(n_conn / total),
        "mask_leak_rate": float(n_masked / total),
        "unique_valid_rate": float(n_unique / max(n_valid, 1)) if n_valid > 0 else 0.0,
        "valid_metal_complex_rate": float(n_metal / total),
        "valid_coordinate_bond_rate": float(n_coord / total),
        "n_distinct_valid_elements": float(len(distinct_elems)),
        "min_dist_p05": float(np.nanpercentile(min_d_arr, 5)),
        "min_dist_p50": float(np.nanpercentile(min_d_arr, 50)),
        "min_dist_p95": float(np.nanpercentile(min_d_arr, 95)),
    }
