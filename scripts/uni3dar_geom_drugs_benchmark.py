"""Standalone `Uni-3DAR-Mol` Unconditional & Large-Molecule Scaling Benchmark on `GEOM-Drugs`.

Designed for Paper 2 (`Uni-3DAR-Mol`: Next-Generation 3D Octree Autoregressive Generation &
Speculative Tree Decoding), strictly enforcing all scientific red lines:
  1. Zero PCA canonicalization (`random_rotation_prob = 0.80`, full SO(3)).
  2. Zero `n_atoms` in `LoomConditioner` (100% unconditional `active_mask = False`; size scale
     injected solely via non-trapping 6-D `ScaleFreeBudgetEmbedding`).
  3. Zero inference-time MLIP (`DPA-4`) relaxation or scoring (`potential_slot=None`).
     `DPA-4` is used ONLY offline during post-processor training and blind post-hoc force audit.

Phases (`--mode`):
  - `train_all` (or `train_sft`, `fit_post`):
      1) Lightweight `ScaleFreeBudgetEmbedding` + `HeavyAtomValenceHead` + `LoRA (r=32)` SFT
         with expected-XYZ 1-2/1-3 bonded distance supervision on `geom_drugs_octree_train_25k_fast.sdf`
         (`merge_level = 7`, 14 main-group elements).
      2) 4-Layer Kabsch `PostTreeGraphRefiner` (`< 0.5 ms` seam healing + bilinear bond-order head)
         distilled on equilibrium `GEOM-Drugs` conformers + real on-policy octree rollouts.
      3) Unconditional 3-layer GNN `TwoStageOctreeProcessCritic` trained with grouped ranking loss
         over `(chemical_validity, neutral_closed_shell, training_time_F_phys)` (`0` external property
         oracles), saved into a single self-contained `uni3dar_mol_geom_bundled.pt`.
  - `eval_10k`:
      Standard `10,000`-molecule unconditional `GEOM-Drugs` benchmark (`eval_10k_atom_budgets.json`)
      comparing:
        * `Vanilla_Uni3DAR_Pretrained_M1` (`drug.pt`, raw point cloud, no refiner/reconciler/budget)
        * `Uni3DAR_Mol_Base_M1` (`1->1->1`, `0` critic calls, `+ Budget + ValenceGuard + Refiner + Reconciler`)
        * `Uni3DAR_Mol_SpecTree_M2` (`low`, continuous `~16->4->2` internal speculative tree, `0` external calls)
        * `Uni3DAR_Mol_SpecTree_M4` (`medium`, continuous `~32->8->4` internal speculative tree, `0` external calls)
      plus published `EDM`, `GeoLDM`, `MiDi`, `EQGAT-diff`, `FlowMol`, and `SemlaFlow` reference baselines.
  - `eval_scaling`:
      Controlled large-molecule asymptotic complexity benchmark (`N in {30, 45, 60, 80, 100}`,
      `250` molecules/bucket) on the exact same A800 GPU comparing sparse `O(N log N)` `Uni-3DAR-Mol`
      against dense `O(N^2)` 100-step CTMC Flow Matching (`FlowMol / PropMolFlow`).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
import torch
from torch.utils.data import DataLoader, Dataset

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state
from loom.generator.octree_backend import OctreeDataset
from loom.weaving import (
    BuiltInRuleAsset,
    LoomFlowMolGenerator,
    LoomUni3DARGenerator,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    PostTreeGraphRefiner,
    PotentialSlot,
    TwoStageOctreeProcessCritic,
    fit_octree_element_process_critic,
    fit_octree_process_critic,
    fit_post_tree_graph_refiner,
    zero_step_physical_reward,
)
from loom.weaving.lora import iter_lora_parameters
from loom.weaving.service import _analytic_conformer_strain_penalty


GEOM_ALLOWED_ELEMENTS: Tuple[str, ...] = (
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"
)

# Standard Hoogeboom et al. (EDM, ICML 2022) & MiDi (Vignac et al., 2023) lookup-table bond lengths (in Angstrom)
_EDM_BOND1_A: Dict[str, Dict[str, float]] = {
    "H": {"H": 0.74, "B": 1.19, "C": 1.09, "N": 1.01, "O": 0.96, "F": 0.92, "Si": 1.48, "P": 1.44, "S": 1.34, "Cl": 1.27, "As": 1.52, "Se": 1.46, "Br": 1.41, "I": 1.61},
    "B": {"H": 1.19, "C": 1.56, "N": 1.44, "O": 1.36, "F": 1.31, "Cl": 1.75, "Br": 1.89},
    "C": {"H": 1.09, "B": 1.56, "C": 1.54, "N": 1.47, "O": 1.43, "F": 1.35, "Si": 1.85, "P": 1.84, "S": 1.82, "Cl": 1.77, "As": 1.96, "Se": 1.94, "Br": 1.94, "I": 2.14},
    "N": {"H": 1.01, "B": 1.44, "C": 1.47, "N": 1.45, "O": 1.40, "F": 1.36, "Si": 1.72, "P": 1.77, "S": 1.68, "Cl": 1.75, "Br": 2.14},
    "O": {"H": 0.96, "B": 1.36, "C": 1.43, "N": 1.40, "O": 1.48, "F": 1.42, "Si": 1.63, "P": 1.63, "S": 1.51, "Cl": 1.64, "As": 1.78, "Se": 1.72, "Br": 1.72, "I": 1.94},
    "F": {"H": 0.92, "B": 1.31, "C": 1.35, "N": 1.36, "O": 1.42, "F": 1.42, "Si": 1.60, "P": 1.56, "S": 1.58},
    "Si": {"H": 1.48, "C": 1.85, "N": 1.72, "O": 1.63, "F": 1.60, "Si": 2.33, "Cl": 2.02},
    "P": {"H": 1.44, "C": 1.84, "N": 1.77, "O": 1.63, "F": 1.56, "S": 2.10, "Cl": 2.03},
    "S": {"H": 1.34, "C": 1.82, "N": 1.68, "O": 1.51, "F": 1.58, "P": 2.10, "S": 2.04, "Cl": 2.07},
    "Cl": {"H": 1.27, "B": 1.75, "C": 1.77, "N": 1.75, "O": 1.64, "Si": 2.02, "P": 2.03, "S": 2.07, "Cl": 1.99},
    "As": {"H": 1.52, "C": 1.96, "O": 1.78},
    "Se": {"H": 1.46, "C": 1.94, "O": 1.72},
    "Br": {"H": 1.41, "B": 1.89, "C": 1.94, "N": 2.14, "O": 1.72, "Br": 2.28},
    "I": {"H": 1.61, "C": 2.14, "O": 1.94, "I": 2.66},
}
_EDM_BOND2_A: Dict[str, Dict[str, float]] = {
    "C": {"C": 1.34, "N": 1.29, "O": 1.20, "S": 1.60},
    "N": {"C": 1.29, "N": 1.25, "O": 1.21},
    "O": {"C": 1.20, "N": 1.21, "O": 1.21, "P": 1.50, "S": 1.43},
    "P": {"O": 1.50, "S": 1.86},
    "S": {"C": 1.60, "O": 1.43, "P": 1.86},
}
_EDM_BOND3_A: Dict[str, Dict[str, float]] = {
    "C": {"C": 1.20, "N": 1.16, "O": 1.13},
    "N": {"C": 1.16, "N": 1.10},
    "O": {"C": 1.13},
}
_ALLOWED_NEUTRAL_VALENCES: Dict[str, Tuple[int, ...]] = {
    "H": (1,),
    "B": (3,),
    "C": (4,),
    "N": (3,),
    "O": (2,),
    "F": (1,),
    "Si": (4,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "Cl": (1,),
    "As": (3, 5),
    "Se": (2, 4, 6),
    "Br": (1,),
    "I": (1,),
}

# Published GEOM-Drugs unconditional baselines (Hoogeboom 2022; Xu 2023; Vignac 2023; Le 2023; Dunn 2024; Irwin 2025)
PUBLISHED_GEOM_DRUGS_BASELINES: Dict[str, Dict[str, Any]] = {
    "EDM (ICML 2022)": {
        "paradigm": "E(3) Diffusion (1000 steps)",
        "atom_stability_pct": 81.3,
        "mol_stability_pct": 0.2,
        "valid_pct": 91.9,
        "valid_connected_pct": None,
        "nfe_steps": 1000,
    },
    "GeoLDM (ICML 2023)": {
        "paradigm": "Latent E(3) Diffusion (1000 steps)",
        "atom_stability_pct": 84.4,
        "mol_stability_pct": 5.2,
        "valid_pct": 99.3,
        "valid_connected_pct": None,
        "nfe_steps": 1000,
    },
    "MiDi-Adaptive (ECML 2023)": {
        "paradigm": "Joint 2D+3D Diffusion (500 steps)",
        "atom_stability_pct": 99.8,
        "mol_stability_pct": 91.6,
        "valid_pct": 77.8,
        "valid_connected_pct": 77.8,
        "nfe_steps": 500,
    },
    "EQGAT-diff (NeurIPS 2023)": {
        "paradigm": "Joint Equivariant Diffusion (500 steps)",
        "atom_stability_pct": 99.8,
        "mol_stability_pct": 93.4,
        "valid_pct": 94.6,
        "valid_connected_pct": 87.5,
        "nfe_steps": 500,
    },
    "FlowMol (2024)": {
        "paradigm": "Joint CTMC + Euclidean Flow (100 steps)",
        "atom_stability_pct": 99.7,
        "mol_stability_pct": 96.2,
        "valid_pct": 97.1,
        "valid_connected_pct": 95.3,
        "nfe_steps": 100,
    },
    "SemlaFlow (ICLR 2025)": {
        "paradigm": "Equivariant OT Flow Matching (100 steps)",
        "atom_stability_pct": 99.8,
        "mol_stability_pct": 97.7,
        "valid_pct": 98.4,
        "valid_connected_pct": 97.2,
        "nfe_steps": 100,
    },
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _GeomOctreeDataset(Dataset):
    def __init__(self, mols: List[Chem.Mol], backbone: Any, training: bool, seed: int) -> None:
        self.octree_ds = OctreeDataset(
            mols, backbone, training=training, seed=seed, random_rotation_prob=(0.80 if training else 0.0)
        )

    def set_epoch(self, epoch: int) -> None:
        self.octree_ds.set_grid_epoch(epoch)

    def __len__(self) -> int:
        return len(self.octree_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.octree_ds.grid[int(idx)]


class _CollateGeomOctree:
    def __init__(self, collater_fn: Any) -> None:
        self.collater_fn = collater_fn

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.collater_fn(batch)


def _edm_lookup_atom_stability(syms: Sequence[str], pos: np.ndarray) -> Tuple[int, int, bool]:
    """Compute Hoogeboom et al. (2022) lookup-table `(n_stable_atoms, n_total_atoms, is_mol_stable)`."""
    n = len(syms)
    if n < 2 or pos.shape[0] != n:
        return 0, max(n, 1), False
    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    valences = np.zeros(n, dtype=np.int32)
    for i in range(n):
        si = syms[i]
        b1_i = _EDM_BOND1_A.get(si, {})
        b2_i = _EDM_BOND2_A.get(si, {})
        b3_i = _EDM_BOND3_A.get(si, {})
        for j in range(i + 1, n):
            sj = syms[j]
            dij = float(dmat[i, j])
            t1 = b1_i.get(sj)
            if t1 is None or dij >= (t1 + 0.10):
                continue
            t3 = b3_i.get(sj)
            t2 = b2_i.get(sj)
            if t3 is not None and dij < (t3 + 0.05):
                order = 3
            elif t2 is not None and dij < (t2 + 0.05):
                order = 2
            else:
                order = 1
            valences[i] += order
            valences[j] += order

    n_stable = 0
    for i, si in enumerate(syms):
        if int(valences[i]) in _ALLOWED_NEUTRAL_VALENCES.get(si, ()):
            n_stable += 1
    return n_stable, n, (n_stable == n)


def _explicit_graph_atom_stability(mol: Optional[Chem.Mol]) -> Tuple[int, int, bool]:
    """Compute explicit-graph atom & molecule valence stability (MiDi / SemlaFlow / FlowMol protocol)."""
    if mol is None or mol.GetNumAtoms() < 2:
        return 0, 1, False
    n = mol.GetNumAtoms()
    n_stable = 0
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        q = int(a.GetFormalCharge())
        ev = int(round(a.GetExplicitValence()))
        allowed = _ALLOWED_NEUTRAL_VALENCES.get(sym, ())
        if q == 0 and ev in allowed:
            n_stable += 1
        elif sym == "N" and q == 1 and ev == 4:
            n_stable += 1
        elif sym == "O" and q == -1 and ev == 1:
            n_stable += 1
    return n_stable, n, (n_stable == n)


def _raw_point_cloud_to_rdkit_mol(syms: Sequence[str], pos: np.ndarray) -> Optional[Chem.Mol]:
    """Standard raw point-cloud bond perception (`rdDetermineBonds`) for Vanilla Uni-3DAR baseline."""
    if len(syms) < 2 or pos.shape[0] != len(syms):
        return None
    ptable = Chem.GetPeriodicTable()
    # Closed-shell neutral Lewis assignment (charge=0) requires an even total electron count.
    if sum(ptable.GetAtomicNumber(s) for s in syms) % 2 != 0:
        return None
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(syms))
    for i, s in enumerate(syms):
        rw.AddAtom(Chem.Atom(s))
        conf.SetAtomPosition(i, (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    try:
        rdDetermineBonds.DetermineConnectivity(mol)
        if any(a.GetDegree() > 6 for a in mol.GetAtoms()):
            return None
        rdDetermineBonds.DetermineBondOrders(mol, charge=0, maxIterations=250)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _js_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64)
    sp = p.sum()
    sq = q.sum()
    if sp <= 0 or sq <= 0:
        return float("nan")
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)
    mask_p = p > 1e-12
    mask_q = q > 1e-12
    kl_pm = float(np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p])))
    kl_qm = float(np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q])))
    return max(0.0, 0.5 * (kl_pm + kl_qm))


def _extract_geom_distributions(mols: Sequence[Optional[Chem.Mol]]) -> Dict[str, Any]:
    """Extract bond-length, bond-angle, and ring-size histograms across valid connected 3D molecules."""
    bond_keys = (
        ("C", "C", "SINGLE"),
        ("C", "C", "AROMATIC"),
        ("C", "H", "SINGLE"),
        ("C", "N", "SINGLE"),
        ("C", "N", "AROMATIC"),
        ("C", "O", "SINGLE"),
        ("C", "O", "DOUBLE"),
        ("C", "S", "SINGLE"),
    )
    bond_vals: Dict[Tuple[str, str, str], List[float]] = {k: [] for k in bond_keys}
    angles_deg: List[float] = []
    ring_counts = np.zeros(16, dtype=np.float64)  # ring sizes 3..15

    for m in mols:
        if m is None or m.GetNumConformers() == 0 or m.GetNumAtoms() < 2:
            continue
        pos = np.asarray(m.GetConformer().GetPositions(), dtype=np.float64)
        syms = [a.GetSymbol() for a in m.GetAtoms()]
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            s1, s2 = (syms[i], syms[j]) if syms[i] <= syms[j] else (syms[j], syms[i])
            bt = "AROMATIC" if b.GetIsAromatic() else str(b.GetBondType()).split(".")[-1]
            key = (s1, s2, bt)
            if key in bond_vals:
                bond_vals[key].append(float(np.linalg.norm(pos[i] - pos[j])))

        # Heavy-atom bond angles
        for a in m.GetAtoms():
            if a.GetSymbol() == "H":
                continue
            nbrs = [nb.GetIdx() for nb in a.GetNeighbors()]
            if len(nbrs) < 2:
                continue
            p_c = pos[a.GetIdx()]
            for u_i in range(len(nbrs)):
                v1 = pos[nbrs[u_i]] - p_c
                n1 = float(np.linalg.norm(v1))
                if n1 < 1e-4:
                    continue
                for v_i in range(u_i + 1, len(nbrs)):
                    v2 = pos[nbrs[v_i]] - p_c
                    n2 = float(np.linalg.norm(v2))
                    if n2 < 1e-4:
                        continue
                    cos_t = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    angles_deg.append(math.degrees(math.acos(cos_t)))

        try:
            for ring in m.GetRingInfo().AtomRings():
                rsz = len(ring)
                if 3 <= rsz < 16:
                    ring_counts[rsz] += 1.0
        except Exception:
            pass

    b_bins = np.linspace(0.85, 2.15, 131)
    a_bins = np.linspace(60.0, 180.0, 121)
    bond_hists = {
        f"{k[0]}-{k[1]}:{k[2]}": np.histogram(vals, bins=b_bins)[0].astype(np.float64)
        for k, vals in bond_vals.items()
    }
    bond_means = {
        f"{k[0]}-{k[1]}:{k[2]}": (round(float(np.mean(vals)), 4) if vals else float("nan"))
        for k, vals in bond_vals.items()
    }
    angle_hist = np.histogram(angles_deg, bins=a_bins)[0].astype(np.float64)
    return {
        "bond_hists": bond_hists,
        "bond_means_A": bond_means,
        "angle_hist": angle_hist,
        "ring_counts": ring_counts,
    }


def _evaluate_geom_cohort(
    raw_clouds: Sequence[Tuple[List[str], np.ndarray]],
    mols: Sequence[Optional[Chem.Mol]],
    requested_budgets: Optional[np.ndarray],
    train_smiles_set: set[str],
    ref_dist: Dict[str, Any],
    pot_slot: Optional[PotentialSlot],
    wall_sec: float,
    *,
    fmax_eval_cap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    n_total = len(mols)
    edm_atom_ok = 0
    edm_atom_tot = 0
    edm_mol_ok = 0

    exp_atom_ok = 0
    exp_atom_tot = 0
    exp_mol_ok = 0

    n_valid = 0
    n_connected = 0
    n_clean_neutral = 0
    valid_connected_mols: List[Chem.Mol] = []
    valid_smiles: List[str] = []
    gen_atom_counts: List[int] = []
    budget_errs: List[float] = []

    for idx, ((syms, pos), mol) in enumerate(zip(raw_clouds, mols)):
        if len(syms) >= 2:
            sa, ta, ms = _edm_lookup_atom_stability(syms, pos)
            edm_atom_ok += sa
            edm_atom_tot += ta
            if ms:
                edm_mol_ok += 1

        if mol is not None and mol.GetNumAtoms() >= 2:
            n_valid += 1
            n_gen_i = int(mol.GetNumAtoms())
            gen_atom_counts.append(n_gen_i)
            if requested_budgets is not None and float(requested_budgets[idx]) > 0:
                budget_errs.append(abs(float(n_gen_i) - float(requested_budgets[idx])))

            esa, eta, ems = _explicit_graph_atom_stability(mol)
            exp_atom_ok += esa
            exp_atom_tot += eta

            frags = Chem.GetMolFrags(mol)
            is_conn = len(frags) == 1
            if is_conn:
                n_connected += 1
                valid_connected_mols.append(mol)
                if ems:
                    exp_mol_ok += 1
                if is_clean_organic_charge_state(mol) and sum(abs(int(a.GetFormalCharge())) for a in mol.GetAtoms()) == 0:
                    n_clean_neutral += 1
                try:
                    smi = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True)
                    if smi:
                        valid_smiles.append(smi)
                except Exception:
                    pass

    uniq_smiles = set(valid_smiles)
    uniq_rate = float(len(uniq_smiles)) / max(len(valid_smiles), 1)
    novel_cnt = sum(1 for s in uniq_smiles if s not in train_smiles_set)
    novel_rate = float(novel_cnt) / max(len(uniq_smiles), 1)

    gen_dist = _extract_geom_distributions(valid_connected_mols)
    bond_jsds: Dict[str, float] = {}
    for bkey, ref_h in ref_dist["bond_hists"].items():
        if ref_h.sum() >= 50:
            jsd_v = _js_divergence(gen_dist["bond_hists"][bkey], ref_h)
            if math.isfinite(jsd_v):
                bond_jsds[bkey] = round(jsd_v, 5)
    mean_bond_jsd = float(np.mean(list(bond_jsds.values()))) if bond_jsds else float("nan")
    angle_jsd = _js_divergence(gen_dist["angle_hist"], ref_dist["angle_hist"])
    ring_jsd = _js_divergence(gen_dist["ring_counts"], ref_dist["ring_counts"])
    tot_rings = float(gen_dist["ring_counts"].sum())
    strained_34_ring_frac = (
        float(gen_dist["ring_counts"][3] + gen_dist["ring_counts"][4]) / tot_rings if tot_rings > 0 else 0.0
    )

    fmax_med = float("nan")
    fmax_p90 = float("nan")
    frms_med = float("nan")
    if pot_slot is not None and valid_connected_mols:
        rng = np.random.default_rng(seed)
        if len(valid_connected_mols) > fmax_eval_cap:
            sub_idx = np.sort(rng.choice(len(valid_connected_mols), size=fmax_eval_cap, replace=False))
            sub_f_mols = [valid_connected_mols[int(i)] for i in sub_idx]
        else:
            sub_f_mols = valid_connected_mols
        pot_out = pot_slot.evaluate_molecules(sub_f_mols, chunk_size=48)
        fm = np.asarray(pot_out["f_max_eV_A"], dtype=np.float64)
        fr = np.asarray(pot_out["f_rms_eV_A"], dtype=np.float64)
        fin = np.isfinite(fm) & np.isfinite(fr)
        if fin.any():
            fmax_med = float(np.median(fm[fin]))
            fmax_p90 = float(np.percentile(fm[fin], 90))
            frms_med = float(np.median(fr[fin]))

    mol_per_s = float(n_total) / max(wall_sec, 1e-6)
    return {
        "n_requested": n_total,
        "wall_sec": round(wall_sec, 3),
        "throughput_mol_per_sec": round(mol_per_s, 3),
        "gpu_sec_per_1k_molecules": round(1000.0 / max(mol_per_s, 1e-6), 2),
        "valid_rate": round(n_valid / max(n_total, 1), 4),
        "valid_and_connected_rate": round(n_connected / max(n_total, 1), 4),
        "neutral_closed_shell_rate": round(n_clean_neutral / max(n_total, 1), 4),
        "explicit_atom_stability": round(exp_atom_ok / max(exp_atom_tot, 1), 4),
        "explicit_mol_stability": round(exp_mol_ok / max(n_total, 1), 4),
        "edm_lookup_atom_stability": round(edm_atom_ok / max(edm_atom_tot, 1), 4),
        "edm_lookup_mol_stability": round(edm_mol_ok / max(n_total, 1), 4),
        "uniqueness_among_connected": round(uniq_rate, 4),
        "novelty_among_unique": round(novel_rate, 4),
        "mean_atoms_generated": round(float(np.mean(gen_atom_counts)), 2) if gen_atom_counts else 0.0,
        "mean_natoms_abs_err": round(float(np.mean(budget_errs)), 2) if budget_errs else float("nan"),
        "bond_length_jsd_mean": round(mean_bond_jsd, 5) if math.isfinite(mean_bond_jsd) else None,
        "bond_length_jsd_per_type": bond_jsds,
        "bond_length_means_A": gen_dist["bond_means_A"],
        "bond_angle_jsd": round(angle_jsd, 5) if math.isfinite(angle_jsd) else None,
        "ring_size_jsd": round(ring_jsd, 5) if math.isfinite(ring_jsd) else None,
        "strained_3_4_ring_fraction": round(strained_34_ring_frac, 4),
        "zero_step_dpa4_f_max_median_eV_A": round(fmax_med, 3) if math.isfinite(fmax_med) else None,
        "zero_step_dpa4_f_max_p90_eV_A": round(fmax_p90, 3) if math.isfinite(fmax_p90) else None,
        "zero_step_dpa4_f_rms_median_eV_A": round(frms_med, 3) if math.isfinite(frms_med) else None,
    }


def _select_top1_unconditional(cands_row: Sequence[Optional[Chem.Mol]]) -> Optional[Chem.Mol]:
    """Select Top-1 unconditional candidate using ONLY internal `process_utility` + CPU bond geometry (`0` MLIP/oracle calls)."""
    best_mol: Optional[Chem.Mol] = None
    best_sc = -1e18
    for m in cands_row:
        if m is None:
            continue
        if len(Chem.GetMolFrags(m)) != 1:
            continue
        proc_u = float(m.GetDoubleProp("_loom_process_utility")) if m.HasProp("_loom_process_utility") else 0.0
        disp_rms = float(m.GetDoubleProp("_loom_refiner_disp_rms_A")) if m.HasProp("_loom_refiner_disp_rms_A") else 0.0
        strain_p = _analytic_conformer_strain_penalty(m)
        neutral_bonus = 0.25 if sum(abs(int(a.GetFormalCharge())) for a in m.GetAtoms()) == 0 else 0.0
        sc = proc_u + neutral_bonus - 3.0 * strain_p - 1.2 * max(0.0, disp_rms - 0.10)
        if sc > best_sc:
            best_sc = sc
            best_mol = m
    if best_mol is None:
        for m in cands_row:
            if m is not None:
                return m
    return best_mol


def _load_sdfs_clean(sdf_path: Path, max_count: Optional[int] = None) -> List[Chem.Mol]:
    out: List[Chem.Mol] = []
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    for m in suppl:
        if m is None or m.GetNumConformers() == 0 or m.GetNumAtoms() < 4:
            continue
        out.append(m)
        if max_count is not None and len(out) >= max_count:
            break
    return out


def run_train_geom_uni3dar_mol(args: argparse.Namespace) -> Path:
    """Train `Uni-3DAR-Mol` on `GEOM-Drugs`: Budget+Valence+LoRA SFT -> Refiner -> Unconditional ProcessCritic."""
    device = torch.device(args.device)
    out_dir = args.work_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bundled_ckpt = out_dir / "uni3dar_mol_geom_bundled.pt"
    if args.reuse_checkpoints and bundled_ckpt.exists():
        print(f"[GEOM Train] Reusing existing {bundled_ckpt}", flush=True)
        return bundled_ckpt

    sft_ckpt = out_dir / "uni3dar_mol_geom_sft.pt"
    train_mols = _load_sdfs_clean(args.geom_root / "processed" / "geom_drugs_octree_train_25k_fast.sdf")
    print(f"[GEOM Train] Loaded {len(train_mols)} training conformers from train_25k_fast.sdf", flush=True)

    if args.reuse_checkpoints and sft_ckpt.exists():
        gen, sft_payload = LoomUni3DARGenerator.load_checkpoint(sft_ckpt, device=args.device, batch_size=64)
        sft_gpuh = float((sft_payload.get("metadata") or {}).get("sft_gpuh", 0.0))
    else:
        gen = LoomUni3DARGenerator.from_parent_checkpoint(
            args.geom_root / "pretrained" / "uni3dar_drug.pt",
            target_cols=["dummy"],
            target_units=["u"],
            mean=[0.0],
            scale=[1.0],
            allowed_elements=GEOM_ALLOWED_ELEMENTS,
            device=args.device,
            batch_size=int(args.sft_batch_size),
            lora_rank=32,
            lora_alpha=64.0,
            num_prefix_tokens=0,
            use_valence_head=True,
            non_trapping_budget=True,
            random_rotation_prob=0.80,
        )
        for p in gen.parameters():
            p.requires_grad_(False)
        lora_params = [p for _, p in iter_lora_parameters(gen.backbone)]
        for p in lora_params:
            p.requires_grad_(True)
        head_params: List[torch.nn.Parameter] = []
        if gen.budget_emb is not None:
            for p in gen.budget_emb.parameters():
                p.requires_grad_(True)
                head_params.append(p)
        if gen.valence_head is not None:
            for p in gen.valence_head.parameters():
                p.requires_grad_(True)
                head_params.append(p)

        opt = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": 1.8e-4, "weight_decay": 1e-4},
                {"params": head_params, "lr": 4.2e-4, "weight_decay": 1e-5},
            ],
            betas=(0.9, 0.98),
        )
        total_steps = int(args.sft_steps)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1.5e-5)

        ds_tr = _GeomOctreeDataset(train_mols, gen.backbone, training=True, seed=20261119)
        loader = DataLoader(
            ds_tr,
            batch_size=int(args.sft_batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=_CollateGeomOctree(ds_tr.octree_ds.grid.collater),
        )

        gen.train()
        _sync(device)
        t_sft0 = time.perf_counter()
        step = 0
        epoch = 0
        running_loss = 0.0
        while step < total_steps:
            epoch += 1
            ds_tr.set_epoch(epoch)
            for packed in loader:
                if step >= total_steps:
                    break
                bsz = int(packed["decoder_input_len"].numel()) - 1
                t_zero = torch.zeros((bsz, 1), dtype=torch.float32, device=device)
                m_false = torch.zeros((bsz, 1), dtype=torch.bool, device=device)

                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss, info = gen.compute_sft_loss(
                        packed,
                        t_zero,
                        m_false,
                        cfg_dropout_prob=0.0,
                        contrast_weight=0.0,
                        xyz_loss_weight=0.15,
                        valence_loss_weight=0.20,
                        geom_dist_loss_weight=0.25,
                        budget_dropout_prob=0.12,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                opt.step()
                sched.step()
                step += 1
                running_loss += float(info["loss_gen"])
                if step % 200 == 0 or step == total_steps:
                    avg_l = running_loss / float(200 if step % 200 == 0 else (step % 200))
                    running_loss = 0.0
                    el = time.perf_counter() - t_sft0
                    print(
                        f"[GEOM SFT step {step:4d}/{total_steps}] loss={avg_l:.4f} "
                        f"(tree={info['loss_tree']:.3f}, atom={info['loss_atom']:.3f}, "
                        f"xyz={info['loss_xyz']:.3f}, val={info['loss_valence']:.3f}, geom={info['loss_geom']:.3f}) | "
                        f"wall={el:.1f}s",
                        flush=True,
                    )

        _sync(device)
        sft_wall = time.perf_counter() - t_sft0
        sft_gpuh = round(sft_wall / 3600.0, 4)
        gen.eval()
        gen.save_checkpoint(sft_ckpt, extra_metadata={"sft_steps": total_steps, "sft_gpuh": sft_gpuh})
        print(f"[GEOM SFT SAVED] {sft_ckpt} (sft_gpuh={sft_gpuh})", flush=True)

    # ---------------- Stage 2: Fit 4-Layer Kabsch `PostTreeGraphRefiner` ----------------
    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.dpa4_weights), device=args.device)
    reconciler = ValenceReconciler(allowed_elements=GEOM_ALLOWED_ELEMENTS)
    rng = np.random.default_rng(20261120)

    _sync(device)
    t_ref0 = time.perf_counter()
    # Collect 512 real on-policy rollouts & relax offline with DPA-4 to capture exact merge_level=7 seam statistics
    emp_nats = np.array([m.GetNumAtoms() for m in train_mols], dtype=np.float32)
    b_onpol = rng.choice(emp_nats, size=512, replace=True)
    t_z = torch.zeros((512, 1), dtype=torch.float32, device=device)
    m_f = torch.zeros((512, 1), dtype=torch.bool, device=device)
    raw_trajs: List[Dict[str, Any]] = []
    for s in range(0, 512, 128):
        e = min(s + 128, 512)
        raw_trajs.extend(
            gen.sample(
                t_z[s:e],
                m_f[s:e],
                cfg_scale=1.0,
                xyz_cfg_scale=1.0,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=7000 + s,
                use_bf16=True,
                electron_parity=0,
                leaf_atom_budget=b_onpol[s:e],
                budget_guidance_scale=0.35,
            )
        )
    cand_mols_onpol: List[Chem.Mol] = []
    raw_pos_onpol: List[np.ndarray] = []
    for tr in raw_trajs:
        res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
        if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
            cand_mols_onpol.append(res.mol)
            raw_pos_onpol.append(np.asarray(tr["positions_np"], dtype=np.float32))

    paired_onpol: List[Tuple[np.ndarray, Chem.Mol]] = []
    if cand_mols_onpol:
        rel_on = pot_slot.relax_molecules(cand_mols_onpol, n_steps=16, chunk_size=48)
        for rp, rm, gp, ffin in zip(
            raw_pos_onpol, rel_on["relaxed_mols"], rel_on["graph_preserved"], rel_on["final_f_max_eV_A"]
        ):
            if rm is not None and bool(gp) and math.isfinite(float(ffin)) and float(ffin) <= 2.4:
                paired_onpol.append((rp, rm))

    ref_sub_idx = rng.choice(len(train_mols), size=min(14000, len(train_mols)), replace=False)
    ref_train_mols = [train_mols[int(i)] for i in ref_sub_idx]
    refiner = PostTreeGraphRefiner(
        num_atom_classes=int(gen.backbone.num_atom),
        d_model=192,
        n_heads=4,
        n_layers=4,
        n_rbf=24,
        d_pair=128,
        max_disp_A=0.25,
    ).to(device)
    ref_stats = fit_post_tree_graph_refiner(
        refiner,
        ref_train_mols,
        paired_trajectories=paired_onpol,
        epochs=int(args.refiner_epochs),
        batch_size=96,
        lr=1.2e-3,
        seed=202699,
    )
    refiner.eval()
    _sync(device)
    ref_gpuh = round((time.perf_counter() - t_ref0) / 3600.0, 4)
    refiner.save_checkpoint(out_dir / "uni3dar_mol_geom_refiner.pt", extra=ref_stats.__dict__)
    print(f"[GEOM Refiner DONE] stats={ref_stats.__dict__} | paired={len(paired_onpol)} | gpuh={ref_gpuh}", flush=True)

    # ---------------- Stage 3: Unconditional `TwoStageOctreeProcessCritic` ----------------
    _sync(device)
    t_cr0 = time.perf_counter()
    n_roll = int(args.critic_rollouts)
    g_sz = 8
    n_groups = n_roll // g_sz
    grp_nats = rng.choice(emp_nats, size=n_groups, replace=True)
    rep_nats = np.repeat(np.clip(np.round(grp_nats), 18.0, 110.0), g_sz).astype(np.float32)
    group_ids = np.repeat(np.arange(n_groups, dtype=np.int64), g_sz)

    s1_h_l, s1_g_l, s2_h_l, s2_c_l = [], [], [], []
    s2_e_sl, s2_p_sl = [], []
    roll_mols: List[Optional[Chem.Mol]] = []
    is_leg_arr = np.zeros(n_roll, dtype=bool)

    for s in range(0, n_roll, 192):
        e = min(s + 192, n_roll)
        nb = e - s
        tz = torch.zeros((nb, 1), dtype=torch.float32, device=device)
        mf = torch.zeros((nb, 1), dtype=torch.bool, device=device)
        roll = gen.sample_rollouts_with_two_stage_features(
            tz,
            mf,
            cfg_scale=1.0,
            xyz_cfg_scale=1.0,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=910000 + s,
            use_bf16=True,
            electron_parity=0,
            leaf_atom_budget=rep_nats[s:e],
            budget_guidance_scale=0.35,
        )
        s1_h_l.append(roll["s1_h_concat"])
        s1_g_l.append(roll["s1_geom_feats"])
        s2_h_l.append(roll["s2_h_stage2"])
        s2_c_l.append(roll["s2_cloud_feats"])
        tr_sub = roll["trajectories"]
        refiner.refine_trajectories_batch(tr_sub, batch_size=192)
        for loc_i, tr in enumerate(tr_sub):
            s2_e_sl.append(tr["stage2_elem_ids"])
            s2_p_sl.append(tr["stage2_leaf_pos"])
            res = reconciler.reconcile_point_cloud(
                tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
            )
            if (
                res.reconciled_legal
                and (not res.reconciled_exotic_charge)
                and res.mol is not None
                and len(Chem.GetMolFrags(res.mol)) == 1
            ):
                roll_mols.append(res.mol)
                is_leg_arr[s + loc_i] = True
            else:
                roll_mols.append(None)

    valid_roll_mols = [m for m in roll_mols if m is not None]
    pot_roll = pot_slot.evaluate_molecules(valid_roll_mols, chunk_size=48) if valid_roll_mols else None
    rule_asset = BuiltInRuleAsset(asset_id="geom_rules")
    r_out = rule_asset.predict_mols(roll_mols, batch_size=256)
    sa_roll = r_out["mean"][:, rule_asset.channel_index("sa_score")].astype(np.float32)

    phys_all = np.zeros(n_roll, dtype=np.float32)
    fmax_all = np.full(n_roll, np.nan, dtype=np.float32)
    if pot_roll is not None:
        r_p = zero_step_physical_reward(pot_roll["f_rms_eV_A"], pot_roll["f_max_eV_A"]).numpy()
        cp = 0
        for g_i, m in enumerate(roll_mols):
            if m is not None:
                fmax_all[g_i] = float(pot_roll["f_max_eV_A"][cp])
                if math.isfinite(float(r_p[cp])):
                    phys_all[g_i] = float(r_p[cp])
                cp += 1

    neutral_ok = np.array(
        [
            (m is not None and sum(abs(int(a.GetFormalCharge())) for a in m.GetAtoms()) == 0)
            for m in roll_mols
        ],
        dtype=bool,
    )
    low_strain = (~np.isfinite(fmax_all)) | (fmax_all <= 3.0)
    is_constr = is_leg_arr & neutral_ok & np.isfinite(sa_roll) & (sa_roll <= 4.2) & low_strain

    max_k_r = max(1, max((int(es.numel()) for es in s2_e_sl), default=1))
    pad_e = torch.zeros((n_roll, max_k_r), dtype=torch.long)
    pad_p = torch.zeros((n_roll, max_k_r, 3), dtype=torch.float32)
    pad_v = torch.zeros((n_roll, max_k_r), dtype=torch.bool)
    for i, (es, ps) in enumerate(zip(s2_e_sl, s2_p_sl)):
        ki = int(es.numel())
        if ki > 0:
            pad_e[i, :ki] = es
            pad_p[i, :ki] = ps
            pad_v[i, :ki] = True

    emb_d = int(gen.args.emb_dim)
    s1_c = OctreeProcessCritic(emb_dim=emb_d, num_targets=1, hidden_dim=512).to(device)
    s1_c.set_normalization([0.0], [1.0])
    t_zeros_cpu = torch.zeros((n_roll, 1), dtype=torch.float32)
    act_false_cpu = torch.zeros((n_roll, 1), dtype=torch.bool)
    st1 = fit_octree_process_critic(
        s1_c,
        h_concat=torch.cat(s1_h_l, dim=0),
        geom_feats=torch.cat(s1_g_l, dim=0),
        targets_norm=t_zeros_cpu,
        active_mask=act_false_cpu,
        y_surrogate_norm=t_zeros_cpu,
        y_valid_mask=torch.from_numpy(is_leg_arr),
        is_legal=torch.from_numpy(is_leg_arr),
        is_constr_met=torch.from_numpy(is_constr),
        phys_scores=torch.from_numpy(phys_all),
        phys_weight=0.85,
        group_ids=torch.from_numpy(group_ids),
        ranking_weight=0.65,
        epochs=24,
        batch_size=256,
        lr=1.3e-3,
        seed=202691,
    )
    s2_c = OctreeElementProcessCritic(
        emb_dim=emb_d,
        num_targets=1,
        num_atom_classes=int(gen.backbone.num_atom),
        d_node=192,
        n_rbf=24,
        n_gnn_layers=3,
        hidden_dim=512,
    ).to(device)
    s2_c.set_normalization([0.0], [1.0])
    st2 = fit_octree_element_process_critic(
        s2_c,
        h_stage2=torch.cat(s2_h_l, dim=0),
        elem_ids=pad_e,
        leaf_pos=pad_p,
        leaf_valid=pad_v,
        cloud_feats=torch.cat(s2_c_l, dim=0),
        targets_norm=t_zeros_cpu,
        active_mask=act_false_cpu,
        y_surrogate_norm=t_zeros_cpu,
        y_valid_mask=torch.from_numpy(is_leg_arr),
        is_legal=torch.from_numpy(is_leg_arr),
        is_constr_met=torch.from_numpy(is_constr),
        phys_scores=torch.from_numpy(phys_all),
        phys_weight=0.90,
        group_ids=torch.from_numpy(group_ids),
        ranking_weight=0.70,
        epochs=28,
        batch_size=256,
        lr=1.1e-3,
        seed=202692,
    )
    ts_critic = TwoStageOctreeProcessCritic(s1_c, s2_c).to(device)
    ts_critic.eval()
    _sync(device)
    crit_gpuh = round((time.perf_counter() - t_cr0) / 3600.0, 4)

    gen.attach_post_processors(post_tree_refiner=refiner, process_critic=ts_critic)
    tot_gpuh = round(sft_gpuh + ref_gpuh + crit_gpuh, 4)
    meta_bundle = {
        "dataset": "GEOM-Drugs (merge_level=7, 14 elements, full SO(3))",
        "sft_gpuh": sft_gpuh,
        "refiner_gpuh": ref_gpuh,
        "critic_gpuh": crit_gpuh,
        "total_post_training_gpuh": tot_gpuh,
        "rollout_clean_connected_rate": round(float(is_leg_arr.mean()), 4),
        "rollout_fmax_median_eV_A": round(float(np.nanmedian(fmax_all)), 3),
        "refiner_stats": ref_stats.__dict__,
        "stage1_critic_stats": st1.__dict__,
        "stage2_critic_stats": st2.__dict__,
    }
    gen.save_checkpoint(bundled_ckpt, extra_metadata=meta_bundle)
    print(f"[GEOM Bundled Checkpoint SAVED] {bundled_ckpt} | meta={meta_bundle}", flush=True)
    return bundled_ckpt


def run_eval_10k_geom(args: argparse.Namespace, bundled_ckpt: Path) -> None:
    """Run standard `10,000`-molecule `GEOM-Drugs` unconditional benchmark across 4 arms."""
    device = torch.device(args.device)
    proc_dir = args.geom_root / "processed"
    budgets_10k = np.asarray(json.loads((proc_dir / "eval_10k_atom_budgets.json").read_text()), dtype=np.float32)
    n_eval = int(args.eval_count)
    budgets = budgets_10k[:n_eval]

    with gzip.open(proc_dir / "geom_drugs_train_canonical_smiles.txt.gz", "rt") as gz:
        train_smiles_set = {line.strip() for line in gz if line.strip()}
    ref_mols = _load_sdfs_clean(proc_dir / "geom_drugs_octree_test_10k_ref.sdf", max_count=10000)
    ref_dist = _extract_geom_distributions(ref_mols)
    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.dpa4_weights), device=args.device)

    arms_results: Dict[str, Any] = {}
    if args.eval_10k_json.exists():
        try:
            arms_results = dict(json.loads(args.eval_10k_json.read_text()).get("arms") or {})
        except Exception:
            arms_results = {}
    only_set = set(args.only_arms) if args.only_arms else None

    tz_w = torch.zeros((16, 1), dtype=torch.float32, device=device)
    mf_w = torch.zeros((16, 1), dtype=torch.bool, device=device)

    mol_gen, mol_payload = LoomUni3DARGenerator.load_checkpoint(bundled_ckpt, device=args.device, batch_size=512)
    mol_gen.eval()
    mol_meta = mol_payload.get("metadata") or {}

    def _flush_10k() -> None:
        out_payload = {
            "experiment": "GEOM-Drugs 10,000-Molecule Unconditional 3D Molecular Generation Benchmark — Uni-3DAR-Mol",
            "n_molecules": n_eval,
            "merge_level": 7,
            "grid_len_A": 0.24,
            "allowed_elements": list(GEOM_ALLOWED_ELEMENTS),
            "post_training_metadata": mol_meta,
            "published_reference_baselines": PUBLISHED_GEOM_DRUGS_BASELINES,
            "arms": arms_results,
        }
        args.eval_10k_json.parent.mkdir(parents=True, exist_ok=True)
        args.eval_10k_json.write_text(json.dumps(out_payload, indent=2) + "\n")

    # 1. Vanilla Pretrained `Uni-3DAR (drug.pt)` baseline (M=1, raw point cloud, zero refiner/reconciler)
    if (only_set is None or "Vanilla_Uni3DAR_Pretrained_M1" in only_set) and not (
        args.reuse_checkpoints and "Vanilla_Uni3DAR_Pretrained_M1" in arms_results
    ):
        vanilla_gen = LoomUni3DARGenerator.from_parent_checkpoint(
            args.geom_root / "pretrained" / "uni3dar_drug.pt",
            target_cols=["dummy"],
            target_units=["u"],
            mean=[0.0],
            scale=[1.0],
            allowed_elements=GEOM_ALLOWED_ELEMENTS,
            device=args.device,
            batch_size=256,
            lora_rank=0,
            num_prefix_tokens=0,
            use_valence_head=False,
            non_trapping_budget=True,
        )
        vanilla_gen.eval()
        vanilla_gen.sample(tz_w, mf_w, cfg_scale=1.0, temperatures=(1.0, 1.0, 1.0), seed=3, use_bf16=True, electron_parity=None)
        _sync(device)

        t0_v = time.perf_counter()
        van_trajs: List[Dict[str, Any]] = []
        for s in range(0, n_eval, 512):
            e = min(s + 512, n_eval)
            nb = e - s
            van_trajs.extend(
                vanilla_gen.sample(
                    torch.zeros((nb, 1), dtype=torch.float32, device=device),
                    torch.zeros((nb, 1), dtype=torch.bool, device=device),
                    cfg_scale=1.0,
                    xyz_cfg_scale=1.0,
                    geometric_guard=False,
                    temperatures=(1.0, 1.0, 1.0),
                    seed=50000 + s,
                    use_bf16=True,
                    electron_parity=None,
                )
            )
        van_clouds = [(tr.get("atom_symbols", []), tr.get("positions_np", np.zeros((0, 3), dtype=np.float32))) for tr in van_trajs]
        with ThreadPoolExecutor(max_workers=min(8, max(2, (os.cpu_count() or 4) // 2))) as pool:
            van_mols = list(pool.map(lambda cp: _raw_point_cloud_to_rdkit_mol(cp[0], cp[1]), van_clouds))
        _sync(device)
        van_wall = time.perf_counter() - t0_v
        del vanilla_gen
        torch.cuda.empty_cache()

        arms_results["Vanilla_Uni3DAR_Pretrained_M1"] = _evaluate_geom_cohort(
            van_clouds, van_mols, None, train_smiles_set, ref_dist, pot_slot, van_wall
        )
        _flush_10k()
        print(f"[GEOM 10k Vanilla M=1] {arms_results['Vanilla_Uni3DAR_Pretrained_M1']}", flush=True)

    # 2. Evaluate `Uni-3DAR-Mol` bundled checkpoint (`Base M=1`, `SpecTree M=2`, `SpecTree M=4`)
    mol_gen.generate_candidates(
        tz_w, mf_w, atom_budget=np.full(16, 45.0, dtype=np.float32), effort="low", cfg_scale=1.0, seed=5
    )
    _sync(device)

    for arm_name, effort_tier in (
        ("Uni3DAR_Mol_Base_M1", "base"),
        ("Uni3DAR_Mol_SpecTree_M2", "low"),
        ("Uni3DAR_Mol_SpecTree_M4", "medium"),
    ):
        if only_set is not None and arm_name not in only_set:
            continue
        if args.reuse_checkpoints and arm_name in arms_results:
            continue
        _sync(device)
        t_a0 = time.perf_counter()
        tz_all = torch.zeros((n_eval, 1), dtype=torch.float32, device=device)
        mf_all = torch.zeros((n_eval, 1), dtype=torch.bool, device=device)
        c_grid = mol_gen.generate_candidates(
            tz_all,
            mf_all,
            atom_budget=budgets,
            effort=effort_tier,
            cfg_scale=1.0,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.0,
            seed=610000 + len(arm_name) * 97,
        )
        sel_mols = [_select_top1_unconditional(row) for row in c_grid]
        _sync(device)
        a_wall = time.perf_counter() - t_a0

        sel_clouds: List[Tuple[List[str], np.ndarray]] = []
        for m in sel_mols:
            if m is not None and m.GetNumConformers() > 0:
                sel_clouds.append(
                    ([a.GetSymbol() for a in m.GetAtoms()], np.asarray(m.GetConformer().GetPositions(), dtype=np.float32))
                )
            else:
                sel_clouds.append(([], np.zeros((0, 3), dtype=np.float32)))

        res_arm = _evaluate_geom_cohort(
            sel_clouds, sel_mols, budgets, train_smiles_set, ref_dist, pot_slot, a_wall
        )
        _, ms1, ms2, mk = mol_gen.resolve_internal_funnel(effort=effort_tier, atom_budget=budgets)
        res_arm["effort"] = effort_tier
        res_arm["internal_funnel_median"] = {"m_stage1": ms1, "m_stage2": ms2, "m_keep": mk}
        arms_results[arm_name] = res_arm
        _flush_10k()
        print(f"[GEOM 10k {arm_name}] {res_arm}", flush=True)

    _flush_10k()
    print(f"[GEOM 10k SAVED] {args.eval_10k_json}", flush=True)


def run_eval_scaling_geom(args: argparse.Namespace, bundled_ckpt: Path) -> None:
    """Controlled large-molecule scaling (`N in {30, 45, 60, 80, 100}`, 250 molecules/bucket) on same A800 GPU."""
    device = torch.device(args.device)
    proc_dir = args.geom_root / "processed"
    with gzip.open(proc_dir / "geom_drugs_train_canonical_smiles.txt.gz", "rt") as gz:
        train_smiles_set = {line.strip() for line in gz if line.strip()}
    ref_mols = _load_sdfs_clean(proc_dir / "geom_drugs_octree_test_10k_ref.sdf", max_count=5000)
    ref_dist = _extract_geom_distributions(ref_mols)
    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.dpa4_weights), device=args.device)

    mol_gen, _ = LoomUni3DARGenerator.load_checkpoint(bundled_ckpt, device=args.device, batch_size=256)
    mol_gen.eval()

    buckets = (30, 45, 60, 80, 100)
    n_per_b = int(args.per_bucket)
    scaling_out: Dict[str, Any] = {str(nb): {"target_n_atoms": nb, "n_prompts": n_per_b, "arms": {}} for nb in buckets}

    tz_w = torch.zeros((16, 1), dtype=torch.float32, device=device)
    mf_w = torch.zeros((16, 1), dtype=torch.bool, device=device)
    mol_gen.generate_candidates(tz_w, mf_w, atom_budget=np.full(16, 45.0, dtype=np.float32), effort="low", cfg_scale=1.0, seed=9)
    _sync(device)

    for nb in buckets:
        b_vec = np.full(n_per_b, float(nb), dtype=np.float32)
        tz_b = torch.zeros((n_per_b, 1), dtype=torch.float32, device=device)
        mf_b = torch.zeros((n_per_b, 1), dtype=torch.bool, device=device)
        for arm_key, eff_t in (("Uni3DAR_Mol_Base_M1", "base"), ("Uni3DAR_Mol_SpecTree_M2", "low")):
            _sync(device)
            t0 = time.perf_counter()
            c_grid = mol_gen.generate_candidates(
                tz_b, mf_b, atom_budget=b_vec, effort=eff_t, cfg_scale=1.0, xyz_cfg_scale=1.0, seed=82000 + nb * 31
            )
            sel_mols = [_select_top1_unconditional(row) for row in c_grid]
            _sync(device)
            w_s = time.perf_counter() - t0
            clouds = [
                ([a.GetSymbol() for a in m.GetAtoms()], np.asarray(m.GetConformer().GetPositions(), dtype=np.float32))
                if (m is not None and m.GetNumConformers() > 0)
                else ([], np.zeros((0, 3), dtype=np.float32))
                for m in sel_mols
            ]
            r_b = _evaluate_geom_cohort(clouds, sel_mols, b_vec, train_smiles_set, ref_dist, pot_slot, w_s, fmax_eval_cap=250)
            _, ms1, ms2, mk = mol_gen.resolve_internal_funnel(effort=eff_t, atom_budget=float(nb))
            r_b["internal_funnel"] = {"m_stage1": ms1, "m_stage2": ms2, "m_keep": mk}
            scaling_out[str(nb)]["arms"][arm_key] = r_b
            print(
                f"[Scaling N={nb:3d} {arm_key:24s}] {r_b['throughput_mol_per_sec']:.2f} mol/s "
                f"({r_b['gpu_sec_per_1k_molecules']:.1f} s/1k) | atoms={r_b['mean_atoms_generated']} | "
                f"valid_conn={r_b['valid_and_connected_rate']*100:.1f}% | Fmax={r_b['zero_step_dpa4_f_max_median_eV_A']}",
                flush=True,
            )

    del mol_gen
    torch.cuda.empty_cache()

    # Same-GPU Dense O(N^2) 100-Step DGL CTMC Flow Matching (`FlowMol` architecture) across N = 30..100
    if args.flowmol_ckpt is not None and args.flowmol_ckpt.exists():
        flow_gen = LoomFlowMolGenerator.from_propmolflow_checkpoint(
            args.flowmol_ckpt,
            device=args.device,
            data_dir=args.flowmol_data_dir,
            property_name="homo",
            sample_timesteps=100,
        )
        flow_gen.max_atoms = 120
        flow_gen.eval()
        flow_gen.generate_candidates(
            torch.zeros((8, 1), dtype=torch.float32, device=device),
            torch.ones((8, 1), dtype=torch.bool, device=device),
            atom_budget=np.full(8, 30.0, dtype=np.float32),
            effort="base",
            seed=13,
        )
        _sync(device)

        for nb in buckets:
            b_vec = np.full(n_per_b, float(nb), dtype=np.float32)
            # Chunk dense DGL graphs safely at N=80,100 (`O(N^2)` edges per graph!)
            f_chunk = 125 if nb <= 60 else 50
            sel_f: List[Optional[Chem.Mol]] = []
            _sync(device)
            t0 = time.perf_counter()
            for s in range(0, n_per_b, f_chunk):
                e = min(s + f_chunk, n_per_b)
                c_g = flow_gen.generate_candidates(
                    torch.zeros((e - s, 1), dtype=torch.float32, device=device),
                    torch.ones((e - s, 1), dtype=torch.bool, device=device),
                    atom_budget=b_vec[s:e],
                    effort="base",
                    seed=93000 + nb * 37 + s,
                )
                sel_f.extend([row[0] for row in c_g])
            _sync(device)
            w_f = time.perf_counter() - t0
            clouds_f = [
                ([a.GetSymbol() for a in m.GetAtoms()], np.asarray(m.GetConformer().GetPositions(), dtype=np.float32))
                if (m is not None and m.GetNumConformers() > 0)
                else ([], np.zeros((0, 3), dtype=np.float32))
                for m in sel_f
            ]
            r_f = _evaluate_geom_cohort(clouds_f, sel_f, b_vec, train_smiles_set, ref_dist, pot_slot, w_f, fmax_eval_cap=250)
            scaling_out[str(nb)]["arms"]["FlowMol_Dense_ODE100_M1"] = r_f
            print(
                f"[Scaling N={nb:3d} FlowMol_Dense_ODE100_M1 ] {r_f['throughput_mol_per_sec']:.2f} mol/s "
                f"({r_f['gpu_sec_per_1k_molecules']:.1f} s/1k) | valid_conn={r_f['valid_and_connected_rate']*100:.1f}% | "
                f"Fmax={r_f['zero_step_dpa4_f_max_median_eV_A']}",
                flush=True,
            )

    payload = {
        "experiment": "Large-Molecule Asymptotic Complexity & 3D Quality Scaling (N in {30, 45, 60, 80, 100}) — Sparse Octree Uni-3DAR-Mol vs Dense O(N^2) Flow Matching",
        "buckets_atoms": list(buckets),
        "n_molecules_per_bucket": n_per_b,
        "buckets": scaling_out,
    }
    args.scaling_json.parent.mkdir(parents=True, exist_ok=True)
    args.scaling_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[GEOM Scaling SAVED] {args.scaling_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["all", "train", "eval_10k", "eval_scaling"], default="all")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--reuse-checkpoints", action="store_true")
    p.add_argument("--geom-root", type=Path, default=Path("/sjtu-combustion/wjw/chensq/geom_drugs_20261119"))
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="uni3dar_mol_geom_drugs")
    p.add_argument("--sft-steps", type=int, default=2400)
    p.add_argument("--sft-batch-size", type=int, default=32)
    p.add_argument("--refiner-epochs", type=int, default=12)
    p.add_argument("--critic-rollouts", type=int, default=8192)
    p.add_argument("--eval-count", type=int, default=10000)
    p.add_argument("--only-arms", type=str, nargs="*", default=None)
    p.add_argument("--per-bucket", type=int, default=250)
    p.add_argument("--dpa4-weights", type=Path, default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"))
    p.add_argument(
        "--flowmol-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/propmolflow_oe62_52h/checkpoints/last.ckpt"),
    )
    p.add_argument(
        "--flowmol-data-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/propmolflow_oe62/data"),
    )
    p.add_argument(
        "--eval-10k-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/geom_drugs_uni3dar_mol_10k_eval.json"),
    )
    p.add_argument(
        "--scaling-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/geom_drugs_large_molecule_scaling.json"),
    )
    args = p.parse_args()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    bundled = args.work_dir / args.run_name / "uni3dar_mol_geom_bundled.pt"
    if args.mode in ("all", "train"):
        bundled = run_train_geom_uni3dar_mol(args)
    if args.mode in ("all", "eval_10k"):
        run_eval_10k_geom(args, bundled)
    if args.mode in ("all", "eval_scaling"):
        run_eval_scaling_geom(args, bundled)


if __name__ == "__main__":
    main()
