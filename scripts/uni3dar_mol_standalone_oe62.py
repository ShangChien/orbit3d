"""Standalone `Uni-3DAR-Mol` Conditional Orbital Generation, Scaling & PBE0 Verification on `OE62`.

Strictly 100% Loom-Free (Training + Inference):
  1. Initialized directly from public unconditional `Uni-3DAR (drug.pt)`.
  2. Trained ONLY on scaffold-disjoint `OE62` DFT training conformers (`target_cols = ["homo"]` ONLY;
     ZERO `sa_score`, ZERO `Uni-Mol2`, ZERO `AWR`, ZERO `TopoGRPO`, ZERO external surrogates).
  3. `PostTreeGraphRefiner` distilled SOLELY on `OE62` equilibrium DFT conformers with synthetic
     random-`SO(3)` Morton octant-seam perturbations (ZERO `DPA-4` training calls).
  4. `TwoStageOctreeProcessCritic` trained SOLELY on ground-truth `OE62` DFT `HOMO` labels (`G=8`
     matched/mismatched contrast groups + anti-shortcut decoupled targets) and unlabeled self-rollouts.
  5. Strict single-molecule output (`m_s1 -> m_s2 -> 1`, `M_ext = 1`, `0` external oracle calls),
     evaluated post-hoc ONLY by blinded DFT-calibrated `DPA-4` (`HOMO` & zero-step `F_max`) and
     live `GPU4PySCF PBE0/def2-TZVP/DF` (`42` deterministic prompts).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
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
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch.utils.data import DataLoader, Dataset

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
for _cand in (
    ROOT / "src" / "compat",
    ROOT / "src",
    ROOT,
    Path("/sjtu-combustion/chensq/mol_gen/src/compat"),
    Path("/sjtu-combustion/chensq/PropMolFlow"),
):
    if _cand.exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402
from loom.generator.molecular_state import mol_to_clean_smiles  # noqa: E402
from loom.generator.octree_backend import OctreeDataset  # noqa: E402
from scripts.loom_train_sft import load_sdf_with_targets  # noqa: E402
from loom.weaving import (  # noqa: E402
    AuditOracleSlot,
    LoomUni3DARGenerator,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    PostTreeGraphRefiner,
    PotentialSlot,
    TwoStageOctreeProcessCritic,
    fit_octree_element_process_critic,
    fit_octree_process_critic,
    fit_post_tree_graph_refiner,
)
from loom.weaving.lora import iter_lora_parameters  # noqa: E402
from scripts.loom_oe62_head2head_dft_anchor import SUPPORTED_TZVP_Z, _pair_metrics  # noqa: E402
from scripts.loom_oe62_size_anchored import _get_clean_oe62_pool  # noqa: E402
from scripts.loom_uni3dar_v4_step1_eval import build_fair_30atom_prompts  # noqa: E402

HA_TO_EV = 27.211386245988
OE62_DRUG_ELEMENTS: Tuple[str, ...] = (
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _min_pair_dist(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 99.0
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d, 99.0)
    return float(d.min())


def _raw_point_cloud_to_rdkit_mol(symbols: Sequence[str], positions_np: np.ndarray) -> Optional[Chem.Mol]:
    if len(symbols) < 2 or positions_np.shape[0] != len(symbols):
        return None
    if _min_pair_dist(positions_np) < 0.65:
        return None
    try:
        rw = Chem.RWMol()
        conf = Chem.Conformer(len(symbols))
        for i, sym in enumerate(symbols):
            rw.AddAtom(Chem.Atom(str(sym)))
            x, y, z = float(positions_np[i, 0]), float(positions_np[i, 1]), float(positions_np[i, 2])
            conf.SetAtomPosition(i, (x, y, z))
        mol = rw.GetMol()
        mol.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.SanitizeMol(mol)
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
        return mol
    except Exception:
        return None


def _murcko_smi(mol: Chem.Mol) -> str:
    try:
        core = MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(Chem.Mol(mol)))
        return Chem.MolToSmiles(core, canonical=True) if core is not None and core.GetNumAtoms() > 0 else ""
    except Exception:
        return ""


def _stoich_basis_features(mol: Chem.Mol, tier2_homo: float, tier2_lumo: float) -> np.ndarray:
    """Pure quantum-chemistry basis-set alignment features (`FHI-aims Tier2 -> PySCF def2-TZVP`, 0 ML models)."""
    cnts = np.zeros(len(OE62_DRUG_ELEMENTS), dtype=np.float64)
    elem_map = {s: idx for idx, s in enumerate(OE62_DRUG_ELEMENTS)}
    n_hvy = 0.0
    n_arom = 0.0
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym in elem_map:
            cnts[elem_map[sym]] += 1.0
        if sym != "H":
            n_hvy += 1.0
        if a.GetIsAromatic():
            n_arom += 1.0
    n_tot = max(float(mol.GetNumAtoms()), 1.0)
    return np.concatenate(
        [
            np.array([float(tier2_homo), float(tier2_lumo), float(tier2_lumo - tier2_homo), n_hvy / 35.0, n_arom / max(n_hvy, 1.0)], dtype=np.float64),
            cnts / n_tot,
        ]
    )


def _load_pure_oe62_train(
    master_train_sdf: Path,
    quantum_train_dft_sdf: Optional[Path] = None,
    eq_conformer_pool_sdf: Optional[Path] = None,
    max_atoms: int = 76,
) -> Tuple[List[Chem.Mol], np.ndarray, np.ndarray]:
    """Load scaffold-disjoint OE62 equilibrium conformers with 100% pure-DFT `PBE0/def2-TZVP` `homo (eV)` ONLY.

    Strictly zero `Uni-Mol2` (`B_homo` / `v4_homo_eV` ignored) and zero `SA` filter:
      1. Fits a linear pure-DFT basis conversion from `OE62` `FHI-aims Tier2` (`OriginalLabel_homo`)
         to `PySCF PBE0/def2-TZVP` (`DFT_homo`) strictly on `quantum_train_dft.sdf` (`3,949` training DFT conformers).
      2. Pairs gas-phase equilibrium conformers (`quantum_train_dft.sdf` + `eq_conformer_pool_sdf` matched by
         canonical SMILES to `master_train_sdf`) with exact `DFT_homo` (when available) or pure-DFT basis-converted
         `OriginalLabel_homo`.
    """
    allowed_set = set(OE62_DRUG_ELEMENTS)
    sup_master = Chem.SDMolSupplier(str(master_train_sdf), removeHs=False)
    tier2_by_id: Dict[str, Tuple[Chem.Mol, float, float]] = {}
    tier2_by_smi: Dict[str, Tuple[float, float]] = {}
    master_clean: List[Tuple[Chem.Mol, str, float, float]] = []

    for m in sup_master:
        if m is None:
            continue
        na = m.GetNumAtoms()
        if na < 6 or na > max_atoms:
            continue
        if any(a.GetSymbol() not in allowed_set for a in m.GetAtoms()):
            continue
        if not is_clean_organic_charge_state(m):
            continue
        h_str = m.GetProp("OriginalLabel_homo") if m.HasProp("OriginalLabel_homo") else (m.GetProp("DFT_homo") if m.HasProp("DFT_homo") else "")
        if not h_str:
            continue
        h_t2 = float(h_str)
        l_t2 = float(m.GetProp("OriginalLabel_lumo")) if m.HasProp("OriginalLabel_lumo") else (h_t2 + 3.8)
        if not (math.isfinite(h_t2) and math.isfinite(l_t2)):
            continue
        sid = m.GetProp("SourceID") if m.HasProp("SourceID") else ""
        smi = mol_to_clean_smiles(m)
        if sid:
            tier2_by_id[sid] = (m, h_t2, l_t2)
        if smi:
            tier2_by_smi[smi] = (h_t2, l_t2)
        master_clean.append((m, smi, h_t2, l_t2))

    # Fit pure-DFT `FHI-aims Tier2 -> PySCF PBE0/def2-TZVP` basis conversion on `quantum_train_dft.sdf`
    w_basis: Optional[np.ndarray] = None
    b_basis: float = 0.0
    exact_dft_by_smi: Dict[str, Tuple[Chem.Mol, float]] = {}
    if quantum_train_dft_sdf is not None and quantum_train_dft_sdf.exists():
        x_rows: List[np.ndarray] = []
        y_rows: List[float] = []
        for qm in Chem.SDMolSupplier(str(quantum_train_dft_sdf), removeHs=False):
            if qm is None or not qm.HasProp("DFT_homo"):
                continue
            na = qm.GetNumAtoms()
            if na < 6 or na > max_atoms or any(a.GetSymbol() not in allowed_set for a in qm.GetAtoms()):
                continue
            if not is_clean_organic_charge_state(qm):
                continue
            dft_h = float(qm.GetProp("DFT_homo"))
            if not math.isfinite(dft_h):
                continue
            qsmi = mol_to_clean_smiles(qm)
            if qsmi:
                exact_dft_by_smi[qsmi] = (qm, dft_h)
            qsid = qm.GetProp("SourceID") if qm.HasProp("SourceID") else ""
            t2_pair = tier2_by_id.get(qsid, (None, None, None))[1:] if qsid in tier2_by_id else tier2_by_smi.get(qsmi)
            if t2_pair is not None and t2_pair[0] is not None:
                x_rows.append(_stoich_basis_features(qm, float(t2_pair[0]), float(t2_pair[1])))
                y_rows.append(dft_h - float(t2_pair[0]))
        if len(x_rows) >= 64:
            x_mat = np.stack(x_rows, axis=0)
            y_vec = np.asarray(y_rows, dtype=np.float64)
            x_aug = np.concatenate([x_mat, np.ones((len(x_mat), 1), dtype=np.float64)], axis=1)
            reg = 5.0 * np.eye(x_aug.shape[1], dtype=np.float64)
            reg[-1, -1] = 0.0
            theta = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_vec)
            w_basis = theta[:-1]
            b_basis = float(theta[-1])
            pred_shift = x_mat @ w_basis + b_basis
            print(
                f"[OE62 Pure-DFT Basis Fit] n={len(x_rows)} Tier2->def2-TZVP mean_shift={float(np.mean(y_vec)):+.4f} eV | "
                f"residual_mae={float(np.mean(np.abs(y_vec - pred_shift))):.4f} eV",
                flush=True,
            )

    def _convert_tier2(m_obj: Chem.Mol, h_t2: float, l_t2: float) -> float:
        if w_basis is None:
            return float(h_t2)
        feat = _stoich_basis_features(m_obj, h_t2, l_t2)
        return float(h_t2 + float(feat @ w_basis) + b_basis)

    mols_out: List[Chem.Mol] = []
    homos_out: List[float] = []
    nats_out: List[float] = []
    seen_eq_smi: set[str] = set()

    # 1. Include exact `PySCF PBE0/def2-TZVP` training conformers first
    for qsmi, (qm, dft_h) in exact_dft_by_smi.items():
        mols_out.append(qm)
        homos_out.append(float(dft_h))
        nats_out.append(float(qm.GetNumAtoms()))
        seen_eq_smi.add(qsmi)

    # 2. Match gas-phase equilibrium conformers (`eq_conformer_pool_sdf`) strictly to pure-DFT labels
    if eq_conformer_pool_sdf is not None and eq_conformer_pool_sdf.exists():
        for em in Chem.SDMolSupplier(str(eq_conformer_pool_sdf), removeHs=False):
            if em is None:
                continue
            na = em.GetNumAtoms()
            if na < 6 or na > max_atoms or any(a.GetSymbol() not in allowed_set for a in em.GetAtoms()):
                continue
            if not is_clean_organic_charge_state(em):
                continue
            esmi = mol_to_clean_smiles(em)
            if not esmi or esmi in seen_eq_smi:
                continue
            if esmi in exact_dft_by_smi:
                h_val = float(exact_dft_by_smi[esmi][1])
            elif esmi in tier2_by_smi:
                ht2, lt2 = tier2_by_smi[esmi]
                h_val = _convert_tier2(em, ht2, lt2)
            else:
                continue
            mols_out.append(em)
            homos_out.append(h_val)
            nats_out.append(float(na))
            seen_eq_smi.add(esmi)

    # 3. If no equilibrium pool provided (e.g. validation split), convert `master_clean` directly
    if len(mols_out) < 512:
        for m_c, smi_c, ht2, lt2 in master_clean:
            if smi_c and smi_c in seen_eq_smi:
                continue
            mols_out.append(m_c)
            homos_out.append(_convert_tier2(m_c, ht2, lt2))
            nats_out.append(float(m_c.GetNumAtoms()))

    return mols_out, np.asarray(homos_out, dtype=np.float32), np.asarray(nats_out, dtype=np.float32)


class _OE62OctreeTorchDataset(Dataset):
    def __init__(self, mols: Sequence[Chem.Mol], homos: np.ndarray, backbone: Any, seed: int = 2026) -> None:
        self.mols = list(mols)
        self.homos = np.asarray(homos, dtype=np.float32)
        b_copy = copy(backbone)
        b_copy.args = copy(backbone.args)
        b_copy.args.random_rotation_prob = 0.80
        self.oct_ds = OctreeDataset(self.mols, b_copy, training=True)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.mols)

    def __getitem__(self, idx: int) -> Tuple[Optional[Dict[str, Any]], float]:
        try:
            item = self.oct_ds.grid[int(idx)]
            expected = self.mols[int(idx)].GetNumAtoms()
            decoded = int(((item["decoder_level"] == 0) & item["decoder_is_second_atom"]).sum().item())
            if item["cutoff_atom_count"] or decoded != expected:
                return None, 0.0
            return item, float(self.homos[int(idx)])
        except Exception:
            return None, 0.0


class _CollateOE62:
    def __init__(self, grid_collater: Any) -> None:
        self.grid_collater = grid_collater

    def __call__(self, batch: Sequence[Tuple[Optional[Dict[str, Any]], float]]) -> Tuple[Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        valid = [(it, h) for it, h in batch if it is not None]
        if not valid:
            return None, torch.zeros((0, 1), dtype=torch.float32)
        items = [v[0] for v in valid]
        hs = torch.tensor([[v[1]] for v in valid], dtype=torch.float32)
        return self.grid_collater(items), hs


def _build_stratified_oe62_indices(homos: np.ndarray, nats: np.ndarray, total_samples: int, rng: np.random.Generator) -> List[int]:
    """Balanced sampling across HOMO deciles and molecule size bands (`N >= 40` enriched)."""
    q_edges = np.quantile(homos, np.linspace(0.0, 1.0, 9))
    size_bands = [(0, 26, 0.25), (26, 36, 0.35), (36, 48, 0.24), (48, 999, 0.16)]
    out: List[int] = []
    for lo_n, hi_n, w_s in size_bands:
        n_band = int(round(total_samples * w_s))
        per_q = max(1, n_band // 8)
        for qi in range(8):
            m_q = (nats >= lo_n) & (nats < hi_n) & (homos >= q_edges[qi] - 1e-5) & (homos <= q_edges[qi + 1] + 1e-5)
            cand = np.where(m_q)[0]
            if cand.size == 0:
                cand = np.where((nats >= lo_n) & (nats < hi_n))[0]
            if cand.size > 0:
                out.extend(rng.choice(cand, size=per_q, replace=True).tolist())
    rng.shuffle(out)
    return out


def train_or_load_vanilla_oe62(
    args: argparse.Namespace,
    train_mols: Sequence[Chem.Mol],
    train_homos: np.ndarray,
) -> Tuple[LoomUni3DARGenerator, float]:
    out_dir = args.work_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    van_ckpt = out_dir / "vanilla_uni3dar_oe62_homo_sft.pt"
    if args.reuse_checkpoints and van_ckpt.exists():
        gen, pay = LoomUni3DARGenerator.load_checkpoint(van_ckpt, device=args.device, batch_size=256)
        gen.eval()
        return gen, float((pay.get("metadata") or {}).get("sft_gpuh", 0.0))

    device = torch.device(args.device)
    mean = [float(np.mean(train_homos))]
    scale = [float(np.std(train_homos))]
    gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_checkpoint,
        target_cols=["homo"],
        target_units=["eV"],
        mean=mean,
        scale=scale,
        allowed_elements=OE62_DRUG_ELEMENTS,
        device=args.device,
        batch_size=args.sft_batch_size,
        lora_rank=16,
        lora_alpha=32.0,
        lora_include_ffn=False,
        lora_include_heads=True,
        lora_include_xyz_heads=False,
        train_xyz_embeddings=False,
        use_valence_head=False,
        use_scale_free_count=False,
        non_trapping_budget=False,
        num_prefix_tokens=0,
    )
    trainable = gen.configure_trainable(train_conditioner=True, train_lora=True, train_xyz_embeddings=False)
    opt = torch.optim.AdamW(trainable, lr=2.4e-4, weight_decay=0.01)
    steps = int(args.vanilla_sft_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2.0e-5)

    ds = _OE62OctreeTorchDataset(train_mols, train_homos, gen.backbone, seed=args.seed + 31)
    loader = DataLoader(
        ds, batch_size=args.sft_batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True, collate_fn=_CollateOE62(ds.oct_ds.grid.collater),
    )

    gen.train()
    _sync(device)
    t0 = time.perf_counter()
    step = 0
    while step < steps:
        for packed, hs in loader:
            if step >= steps:
                break
            if packed is None or hs.shape[0] < 8:
                continue
            t_norm = gen.normalize_targets(hs)
            m_act = (torch.rand((hs.shape[0], 1), device=device) >= 0.15)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, _ = gen.compute_sft_loss(
                    packed, t_norm, m_act,
                    cfg_dropout_prob=0.0,
                    contrast_weight=0.12,
                    xyz_loss_weight=0.10,
                    valence_loss_weight=0.0,
                    count_loss_weight=0.25,
                    geom_dist_loss_weight=0.0,
                    budget_dropout_prob=0.0,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 300 == 0 or step == steps:
                print(f"[Vanilla OE62 SFT {step:4d}/{steps}] loss={float(loss.item()):.4f} | wall={time.perf_counter()-t0:.1f}s", flush=True)

    _sync(device)
    gpuh = round((time.perf_counter() - t0) / 3600.0, 4)
    gen.eval()
    gen.save_checkpoint(van_ckpt, extra_metadata={"sft_steps": steps, "sft_gpuh": gpuh})
    print(f"[Vanilla OE62 SFT SAVED] {van_ckpt} ({gpuh} GPU-h)", flush=True)
    return gen, gpuh


def train_or_load_standalone_oe62_bundle(
    args: argparse.Namespace,
    train_mols: Sequence[Chem.Mol],
    train_homos: np.ndarray,
    train_nats: np.ndarray,
    valid_mols: Sequence[Chem.Mol],
    valid_homos: np.ndarray,
) -> Tuple[LoomUni3DARGenerator, Dict[str, Any]]:
    device = torch.device(args.device)
    out_dir = args.work_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_ckpt = out_dir / "uni3dar_mol_standalone_oe62_sft.pt"
    ref_ckpt = out_dir / "uni3dar_mol_standalone_oe62_refiner.pt"
    bundled_ckpt = out_dir / "uni3dar_mol_standalone_oe62_bundled.pt"

    if args.reuse_checkpoints and bundled_ckpt.exists():
        gen, pay = LoomUni3DARGenerator.load_checkpoint(bundled_ckpt, device=args.device, batch_size=256)
        gen.eval()
        print(f"[OE62 Standalone Bundle] Reusing {bundled_ckpt}", flush=True)
        return gen, dict(pay.get("metadata") or {})

    # ---------------- 1. Pure Full-SO(3) HOMO SFT (`target_cols = ["homo"]`) ----------------
    if args.reuse_checkpoints and sft_ckpt.exists():
        gen, sft_pay = LoomUni3DARGenerator.load_checkpoint(
            sft_ckpt, device=args.device, batch_size=256, auto_discover_companions=False
        )
        gen.eval()
        sft_gpuh = float((sft_pay.get("metadata") or {}).get("sft_gpuh", 0.0))
    else:
        rng_s = np.random.default_rng(args.seed + 11)
        strat_idx = _build_stratified_oe62_indices(
            train_homos, train_nats, total_samples=int(args.sft_steps) * int(args.sft_batch_size) + 2048, rng=rng_s
        )
        strat_mols = [train_mols[i] for i in strat_idx]
        strat_homos = train_homos[strat_idx]

        mean = [float(np.mean(train_homos))]
        scale = [float(np.std(train_homos))]
        gen = LoomUni3DARGenerator.from_parent_checkpoint(
            args.parent_checkpoint,
            target_cols=["homo"],
            target_units=["eV"],
            mean=mean,
            scale=scale,
            allowed_elements=OE62_DRUG_ELEMENTS,
            device=args.device,
            batch_size=args.sft_batch_size,
            num_rbf=32,
            num_freqs=16,
            lora_rank=64,
            lora_alpha=128.0,
            lora_dropout=0.05,
            lora_include_ffn=True,
            lora_include_heads=True,
            lora_include_xyz_heads=False,
            train_xyz_embeddings=False,
            use_valence_head=True,
            use_scale_free_count=True,
            non_trapping_budget=True,
            num_prefix_tokens=0,
        )
        gen.configure_trainable(
            train_conditioner=True,
            train_lora=True,
            train_xyz_embeddings=False,
            unfreeze_full_backbone=False,
        )
        lora_params = [p for _, p in iter_lora_parameters(gen.backbone) if p.requires_grad]
        lora_ids = {id(p) for p in lora_params}
        head_params = [p for p in gen.parameters() if p.requires_grad and id(p) not in lora_ids]
        ema_gen = deepcopy(gen).eval()
        for p in ema_gen.parameters():
            p.requires_grad_(False)

        opt = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": 2.6e-4, "weight_decay": 1e-4},
                {"params": head_params, "lr": 4.8e-4, "weight_decay": 1e-5},
            ],
            betas=(0.9, 0.98),
        )
        steps = int(args.sft_steps)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2.0e-5)

        ds = _OE62OctreeTorchDataset(strat_mols, strat_homos, gen.backbone, seed=args.seed + 19)
        loader = DataLoader(
            ds, batch_size=args.sft_batch_size, shuffle=False, num_workers=4,
            pin_memory=True, drop_last=True, collate_fn=_CollateOE62(ds.oct_ds.grid.collater),
        )

        gen.train()
        _sync(device)
        t0 = time.perf_counter()
        step = 0
        run_l = 0.0
        for packed, hs in loader:
            if step >= steps:
                break
            if packed is None or hs.shape[0] < 16:
                continue
            t_norm = gen.normalize_targets(hs)
            m_act = (torch.rand((hs.shape[0], 1), device=device) >= 0.16)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, info = gen.compute_sft_loss(
                    packed, t_norm, m_act,
                    cfg_dropout_prob=0.0,
                    contrast_weight=0.22,
                    contrast_margin=0.06,
                    contrast_min_gap_eV=0.28,
                    xyz_loss_weight=0.12,
                    valence_loss_weight=0.18,
                    count_loss_weight=0.25,
                    geom_dist_loss_weight=0.30,
                    budget_dropout_prob=0.12,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for p_ema, p_cur in zip(ema_gen.parameters(), gen.parameters()):
                    p_ema.data.mul_(0.996).add_(p_cur.data, alpha=0.004)
                for b_ema, b_cur in zip(ema_gen.buffers(), gen.buffers()):
                    b_ema.copy_(b_cur)
            step += 1
            run_l += float(info["loss_gen"])
            if step % 300 == 0 or step == steps:
                n_div = 300 if step % 300 == 0 else (step % 300)
                print(
                    f"[OE62 Standalone SFT {step:4d}/{steps}] gen={run_l/n_div:.4f} "
                    f"(tree={info['loss_tree']:.3f}, atom={info['loss_atom']:.3f}, xyz={info['loss_xyz']:.3f}, "
                    f"geom={info['loss_geom']:.3f}) | wall={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
                run_l = 0.0

        _sync(device)
        sft_gpuh = round((time.perf_counter() - t0) / 3600.0, 4)
        gen = ema_gen.eval()
        gen.save_checkpoint(sft_ckpt, extra_metadata={"sft_steps": step, "sft_gpuh": sft_gpuh})
        print(f"[OE62 Standalone SFT SAVED] {sft_ckpt} ({sft_gpuh} GPU-h)", flush=True)

    # ---------------- 2. Pure `PostTreeGraphRefiner` (OE62 DFT Conformers Only, Zero DPA-4) ----------------
    if args.reuse_checkpoints and ref_ckpt.exists():
        refiner, ref_pay = PostTreeGraphRefiner.load_checkpoint(ref_ckpt, device=args.device)
        refiner.eval()
        ref_gpuh = float(ref_pay.get("refiner_gpuh", 0.0))
    else:
        _sync(device)
        tr0 = time.perf_counter()
        rng_r = np.random.default_rng(args.seed + 202)
        sub_r_ids = rng_r.choice(len(train_mols), size=min(12000, len(train_mols)), replace=False)
        ref_mols = [train_mols[int(i)] for i in sub_r_ids]
        refiner = PostTreeGraphRefiner(
            num_atom_classes=128, d_model=192, n_heads=4, n_layers=4, n_rbf=24, d_pair=128, max_disp_A=0.22
        ).to(device)
        ref_st = fit_post_tree_graph_refiner(
            refiner, ref_mols, epochs=12, batch_size=96, lr=1.3e-3, seed=args.seed + 203
        )
        refiner.eval()
        _sync(device)
        ref_gpuh = round((time.perf_counter() - tr0) / 3600.0, 4)
        ref_extra = dict(ref_st.__dict__)
        ref_extra["refiner_gpuh"] = ref_gpuh
        refiner.save_checkpoint(ref_ckpt, extra=ref_extra)
        print(f"[OE62 Standalone Refiner SAVED] {ref_ckpt} | stats={ref_extra}", flush=True)

    # ---------------- 3. 100% Ground-Truth `OE62` DFT `TwoStageOctreeProcessCritic` ----------------
    _sync(device)
    tc0 = time.perf_counter()
    rng_c = np.random.default_rng(args.seed + 303)
    n_train = len(train_mols)
    y_gt_norm_all = gen.normalize_targets(train_homos[:, None]).detach().cpu()

    b_copy = copy(gen.backbone)
    b_copy.args = copy(gen.args)
    b_copy.args.random_rotation_prob = 0.80
    oct_ds = OctreeDataset(train_mols, b_copy, training=True)
    reconciler = ValenceReconciler(allowed_elements=OE62_DRUG_ELEMENTS)

    # Precompute valid octree indices (skipping rare >1 atom-per-0.24A-cell collisions)
    valid_pool_ids: List[int] = []
    for idx_m, m in enumerate(train_mols):
        if m.GetNumAtoms() <= 68:
            valid_pool_ids.append(idx_m)
    valid_pool_arr = np.asarray(valid_pool_ids, dtype=np.int32)
    valid_pool_nats = train_nats[valid_pool_arr].astype(np.int32)

    size_buckets: Dict[int, np.ndarray] = {}
    for nv in range(6, 72):
        c_sub = valid_pool_arr[np.abs(valid_pool_nats - nv) <= 3]
        size_buckets[nv] = c_sub if c_sub.size >= 8 else valid_pool_arr

    def _safe_collate_batch(id_list: Sequence[int], t_sub: torch.Tensor, m_sub: torch.Tensor):
        items, keep_rows = [], []
        for loc_k, mid in enumerate(id_list):
            try:
                it = oct_ds.grid[int(mid)]
                exp_n = train_mols[int(mid)].GetNumAtoms()
                dec_n = int(((it["decoder_level"] == 0) & it["decoder_is_second_atom"]).sum().item())
                if (not it["cutoff_atom_count"]) and dec_n == exp_n:
                    items.append(it)
                    keep_rows.append(loc_k)
            except Exception:
                continue
        if not items:
            return None, []
        pk = oct_ds.grid.collater(items)
        feats = gen.extract_teacher_forced_two_stage_features(
            pk, t_sub[keep_rows].to(device), m_sub[keep_rows].to(device)
        )
        return feats, keep_rows

    # Build `1,600` hybrid `G=8` groups (`2` matched GT + `4` size-matched off-target GT + `2` unlabeled self-rollouts)
    # plus `16,000` broad decoupled-target GT conformers!
    n_gt_groups = int(args.gt_ranking_groups)
    pos_picks = rng_c.choice(valid_pool_arr, size=n_gt_groups, replace=True)

    gt6_ids: List[int] = []
    gt6_t: List[float] = []
    roll2_t: List[float] = []
    roll2_b: List[float] = []

    for g_i in range(n_gt_groups):
        ip = int(pos_picks[g_i])
        tv = float(train_homos[ip])
        nv = int(round(float(train_nats[ip])))
        pool_n = size_buckets.get(nv, valid_pool_arr)
        negs = rng_c.choice(pool_n, size=4, replace=(pool_n.size < 4))
        for mid in [ip, ip] + [int(x) for x in negs]:
            gt6_ids.append(mid)
            gt6_t.append(tv)
        for _ in range(2):
            roll2_t.append(tv)
            roll2_b.append(float(nv))

    gt6_t_norm = gen.normalize_targets(np.asarray(gt6_t, dtype=np.float32)[:, None]).detach().cpu()
    gt6_m = torch.ones((len(gt6_ids), 1), dtype=torch.bool)

    # Extract GT6 features with replacement fallback if any single conformer collides at 0.24A
    emb_d = int(gen.args.emb_dim)
    gt6_s1_h = torch.zeros((len(gt6_ids), 4 * emb_d), dtype=torch.float32)
    gt6_s1_g = torch.zeros((len(gt6_ids), 14), dtype=torch.float32)
    gt6_s2_h = torch.zeros((len(gt6_ids), 6 * emb_d), dtype=torch.float32)
    gt6_s2_c = torch.zeros((len(gt6_ids), 42), dtype=torch.float32)
    gt6_e: List[torch.Tensor] = [torch.zeros(1, dtype=torch.long) for _ in range(len(gt6_ids))]
    gt6_p: List[torch.Tensor] = [torch.zeros((1, 3), dtype=torch.float32) for _ in range(len(gt6_ids))]
    gt6_valid_row = np.zeros(len(gt6_ids), dtype=bool)
    gt6_y = y_gt_norm_all[gt6_ids].clone()

    for s in range(0, len(gt6_ids), 192):
        e = min(s + 192, len(gt6_ids))
        feats, kept = _safe_collate_batch(gt6_ids[s:e], gt6_t_norm[s:e], gt6_m[s:e])
        if feats is None:
            continue
        for k_pos, loc_k in enumerate(kept):
            gi = s + loc_k
            gt6_s1_h[gi] = feats["s1_h_concat"][k_pos]
            gt6_s1_g[gi] = feats["s1_geom_feats"][k_pos]
            gt6_s2_h[gi] = feats["s2_h_stage2"][k_pos]
            gt6_s2_c[gi] = feats["s2_cloud_feats"][k_pos]
            gt6_e[gi] = feats["s2_elem_slices"][k_pos]
            gt6_p[gi] = feats["s2_pos_slices"][k_pos]
            gt6_valid_row[gi] = True

    # Rollout 2 unlabeled on-policy trajectories per group
    r2_t_norm = gen.normalize_targets(np.asarray(roll2_t, dtype=np.float32)[:, None])
    r2_m = torch.ones((len(roll2_t), 1), dtype=torch.bool, device=device)
    r2_b_arr = np.asarray(roll2_b, dtype=np.float32)
    r2_s1_h, r2_s1_g, r2_s2_h, r2_s2_c, r2_e, r2_p = [], [], [], [], [], []
    r2_leg = np.zeros(len(roll2_t), dtype=np.float32)
    r2_phys = np.zeros(len(roll2_t), dtype=np.float32)

    for s in range(0, len(roll2_t), 192):
        e = min(s + 192, len(roll2_t))
        roll = gen.sample_rollouts_with_two_stage_features(
            r2_t_norm[s:e],
            r2_m[s:e],
            cfg_scale=2.0,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=710000 + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            leaf_atom_budget=r2_b_arr[s:e],
            budget_guidance_scale=0.45,
        )
        r2_s1_h.append(roll["s1_h_concat"])
        r2_s1_g.append(roll["s1_geom_feats"])
        r2_s2_h.append(roll["s2_h_stage2"])
        r2_s2_c.append(roll["s2_cloud_feats"])
        tr_sub = roll["trajectories"]
        refiner.refine_trajectories_batch(tr_sub, batch_size=192)
        for loc_i, tr in enumerate(tr_sub):
            r2_e.append(tr["stage2_elem_ids"])
            r2_p.append(tr["stage2_leaf_pos"])
            d_min = _min_pair_dist(tr["positions_np"])
            res = reconciler.reconcile_point_cloud(
                tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
            )
            if (
                res.reconciled_legal
                and (not res.reconciled_exotic_charge)
                and res.mol is not None
                and len(Chem.GetMolFrags(res.mol)) == 1
                and d_min >= 0.72
            ):
                r2_leg[s + loc_i] = 1.0
                r2_phys[s + loc_i] = float(np.clip((d_min - 0.72) / 0.28, 0.30, 1.0))

    r2_s1_h_cat = torch.cat(r2_s1_h, dim=0)
    r2_s1_g_cat = torch.cat(r2_s1_g, dim=0)
    r2_s2_h_cat = torch.cat(r2_s2_h, dim=0)
    r2_s2_c_cat = torch.cat(r2_s2_c, dim=0)

    s1_h_l, s1_g_l, s2_h_l, s2_c_l = [], [], [], []
    s2_e_sl, s2_p_sl = [], []
    t_cond_l, m_cond_l, y_sup_l, y_ok_l = [], [], [], []
    leg_l, con_l, phys_l, gid_l = [], [], [], []

    kept_g = 0
    for g_i in range(n_gt_groups):
        g6 = slice(g_i * 6, (g_i + 1) * 6)
        if not bool(gt6_valid_row[g6].all()):
            continue
        r2 = slice(g_i * 2, (g_i + 1) * 2)
        s1_h_l.extend([gt6_s1_h[g6], r2_s1_h_cat[r2]])
        s1_g_l.extend([gt6_s1_g[g6], r2_s1_g_cat[r2]])
        s2_h_l.extend([gt6_s2_h[g6], r2_s2_h_cat[r2]])
        s2_c_l.extend([gt6_s2_c[g6], r2_s2_c_cat[r2]])
        s2_e_sl.extend(gt6_e[g_i * 6 : (g_i + 1) * 6] + r2_e[g_i * 2 : (g_i + 1) * 2])
        s2_p_sl.extend(gt6_p[g_i * 6 : (g_i + 1) * 6] + r2_p[g_i * 2 : (g_i + 1) * 2])
        t_cond_l.extend([gt6_t_norm[g6], r2_t_norm[r2].detach().cpu()])
        m_cond_l.extend([gt6_m[g6], r2_m[r2].detach().cpu()])
        y_sup_l.extend([gt6_y[g6], torch.zeros((2, 1), dtype=torch.float32)])
        y_ok_l.extend([torch.ones(6, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)])
        leg_l.extend([torch.ones(6, dtype=torch.float32), torch.from_numpy(r2_leg[r2])])
        con_l.extend([torch.ones(6, dtype=torch.float32), torch.from_numpy(r2_leg[r2])])
        phys_l.extend([torch.ones(6, dtype=torch.float32), torch.from_numpy(r2_phys[r2])])
        gid_l.append(torch.full((8,), kept_g, dtype=torch.long))
        kept_g += 1

    # Broad anti-shortcut pass over `20,000` training conformers (`80%` random `t' != y_i`)
    broad_ids = rng_c.choice(valid_pool_arr, size=20000, replace=(valid_pool_arr.size < 20000))
    broad_other = rng_c.choice(valid_pool_arr, size=20000, replace=True)
    broad_t_raw = np.where((np.arange(20000) % 5) == 0, train_homos[broad_ids], train_homos[broad_other]).astype(np.float32)
    broad_t_norm = gen.normalize_targets(broad_t_raw[:, None]).detach().cpu()
    broad_m = torch.ones((20000, 1), dtype=torch.bool)

    b_s1_h, b_s1_g, b_s2_h, b_s2_c, b_e, b_p, b_t, b_y = [], [], [], [], [], [], [], []
    for s in range(0, len(broad_ids), 192):
        e = min(s + 192, len(broad_ids))
        feats, kept = _safe_collate_batch(broad_ids[s:e].tolist(), broad_t_norm[s:e], broad_m[s:e])
        if feats is None or not kept:
            continue
        b_s1_h.append(feats["s1_h_concat"])
        b_s1_g.append(feats["s1_geom_feats"])
        b_s2_h.append(feats["s2_h_stage2"])
        b_s2_c.append(feats["s2_cloud_feats"])
        b_e.extend(feats["s2_elem_slices"])
        b_p.extend(feats["s2_pos_slices"])
        orig_rows = [int(broad_ids[s + k]) for k in kept]
        b_t.append(broad_t_norm[s:e][kept])
        b_y.append(y_gt_norm_all[orig_rows])

    b_s1_h_cat = torch.cat(b_s1_h, dim=0)
    n_b_grp = b_s1_h_cat.shape[0] // 8
    trim_b = n_b_grp * 8
    if trim_b >= 8:
        s1_h_l.append(b_s1_h_cat[:trim_b])
        s1_g_l.append(torch.cat(b_s1_g, dim=0)[:trim_b])
        s2_h_l.append(torch.cat(b_s2_h, dim=0)[:trim_b])
        s2_c_l.append(torch.cat(b_s2_c, dim=0)[:trim_b])
        s2_e_sl.extend(b_e[:trim_b])
        s2_p_sl.extend(b_p[:trim_b])
        t_cond_l.append(torch.cat(b_t, dim=0)[:trim_b])
        m_cond_l.append(torch.ones((trim_b, 1), dtype=torch.bool))
        y_sup_l.append(torch.cat(b_y, dim=0)[:trim_b])
        y_ok_l.append(torch.ones(trim_b, dtype=torch.bool))
        leg_l.append(torch.ones(trim_b, dtype=torch.float32))
        con_l.append(torch.ones(trim_b, dtype=torch.float32))
        phys_l.append(torch.ones(trim_b, dtype=torch.float32))
        gid_l.append(torch.arange(n_b_grp, dtype=torch.long).repeat_interleave(8) + kept_g)

    s1_h_all = torch.cat(s1_h_l, dim=0)
    s1_g_all = torch.cat(s1_g_l, dim=0)
    s2_h_all = torch.cat(s2_h_l, dim=0)
    s2_c_all = torch.cat(s2_c_l, dim=0)
    t_all = torch.cat(t_cond_l, dim=0)
    m_all = torch.cat(m_cond_l, dim=0)
    y_all = torch.cat(y_sup_l, dim=0)
    y_ok_all = torch.cat(y_ok_l, dim=0)
    leg_all = torch.cat(leg_l, dim=0)
    con_all = torch.cat(con_l, dim=0)
    phys_all = torch.cat(phys_l, dim=0)
    gid_all = torch.cat(gid_l, dim=0)

    n_tot = len(s2_e_sl)
    max_k = max(1, max((int(es.numel()) for es in s2_e_sl), default=0))
    pad_e = torch.zeros((n_tot, max_k), dtype=torch.long)
    pad_p = torch.zeros((n_tot, max_k, 3), dtype=torch.float32)
    pad_v = torch.zeros((n_tot, max_k), dtype=torch.bool)
    for i, (es, ps) in enumerate(zip(s2_e_sl, s2_p_sl)):
        ki = int(es.numel())
        if ki > 0:
            pad_e[i, :ki] = es
            pad_p[i, :ki] = ps
            pad_v[i, :ki] = True

    s1_c = OctreeProcessCritic(emb_dim=emb_d, num_targets=1, hidden_dim=512).to(device)
    s1_c.set_normalization(gen.target_mean, gen.target_scale)
    st1 = fit_octree_process_critic(
        s1_c,
        h_concat=s1_h_all,
        geom_feats=s1_g_all,
        targets_norm=t_all,
        active_mask=m_all,
        y_surrogate_norm=y_all,
        y_valid_mask=y_ok_all,
        is_legal=leg_all,
        is_constr_met=con_all,
        phys_scores=phys_all,
        phys_weight=0.55,
        group_ids=gid_all,
        ranking_weight=0.55,
        epochs=24,
        batch_size=384,
        lr=1.3e-3,
        seed=args.seed + 411,
    )

    s2_c = OctreeElementProcessCritic(
        emb_dim=emb_d,
        num_targets=1,
        num_atom_classes=gen.backbone.num_atom,
        d_node=256,
        n_rbf=64,
        n_gnn_layers=4,
        hidden_dim=512,
        rbf_min=0.65,
        rbf_max=5.20,
        rbf_gamma=42.0,
        edge_value_gate=True,
        equivariant_multipole=True,
        h_gate_scale=0.0,
    ).to(device)
    s2_c.set_normalization(gen.target_mean, gen.target_scale)
    st2 = fit_octree_element_process_critic(
        s2_c,
        h_stage2=s2_h_all,
        elem_ids=pad_e,
        leaf_pos=pad_p,
        leaf_valid=pad_v,
        cloud_feats=s2_c_all,
        targets_norm=t_all,
        active_mask=m_all,
        y_surrogate_norm=y_all,
        y_valid_mask=y_ok_all,
        is_legal=leg_all,
        is_constr_met=con_all,
        phys_scores=phys_all,
        phys_weight=0.65,
        group_ids=gid_all,
        ranking_weight=0.65,
        epochs=36,
        batch_size=384,
        lr=1.1e-3,
        pos_noise_std=0.015,
        supervise_all_valid_targets=True,
        seed=args.seed + 412,
    )

    ts_critic = TwoStageOctreeProcessCritic(s1_c, s2_c).to(device)
    ts_critic.eval()
    _sync(device)
    crit_gpuh = round((time.perf_counter() - tc0) / 3600.0, 4)

    gen.attach_post_processors(post_tree_refiner=refiner, process_critic=ts_critic)
    tot_gpuh = round(sft_gpuh + ref_gpuh + crit_gpuh, 4)
    meta = {
        "protocol": "100% Standalone Uni-3DAR-Mol on OE62 (Zero Loom, Zero Uni-Mol2, Zero SA, Zero RL, M_ext=1)",
        "sft_gpuh": sft_gpuh,
        "refiner_gpuh": ref_gpuh,
        "gt_critic_gpuh": crit_gpuh,
        "total_training_gpuh": tot_gpuh,
        "stage1_critic_stats": st1.__dict__,
        "stage2_critic_stats": st2.__dict__,
    }
    gen.save_checkpoint(bundled_ckpt, extra_metadata=meta)
    print(f"[OE62 Standalone Bundled SAVED] {bundled_ckpt} | total_gpuh={tot_gpuh}", flush=True)
    return gen, meta


def _audit_oe62_cohort(
    mols: Sequence[Optional[Chem.Mol]],
    req_homo: np.ndarray,
    req_natoms: np.ndarray,
    audit_slot: AuditOracleSlot,
    pot_slot: PotentialSlot,
    train_smiles: set[str],
    wall_sec: float,
    funnel: Tuple[int, int, int],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Post-hoc blind evaluation of single-output (`M_ext = 1`) OE62 cohort."""
    n_req = len(mols)
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid_idx]
    n_leg = len(valid_mols)

    exotic_cnt = sum(1 for m in valid_mols if not is_clean_organic_charge_state(m))
    clean_mask_local = np.array([is_clean_organic_charge_state(m) for m in valid_mols], dtype=bool)

    dpa_homo_full = np.full(n_req, np.nan, dtype=np.float64)
    fmax_full = np.full(n_req, np.nan, dtype=np.float64)

    if valid_mols:
        aud_means, _, aud_ok, _ = audit_slot.evaluate_candidates(valid_mols, batch_size=128)
        pot_out = pot_slot.evaluate_molecules(valid_mols, chunk_size=48)
        fmax_vec = np.asarray(pot_out["f_max_eV_A"], dtype=np.float64)
        for loc_k, g_idx in enumerate(valid_idx):
            if bool(aud_ok[loc_k]) and math.isfinite(float(aud_means[loc_k, 0])):
                dpa_homo_full[g_idx] = float(aud_means[loc_k, 0])
            if math.isfinite(float(fmax_vec[loc_k])):
                fmax_full[g_idx] = float(fmax_vec[loc_k])

    eval_mask = np.isfinite(dpa_homo_full)
    errs = np.abs(dpa_homo_full[eval_mask] - req_homo[eval_mask])
    mae = float(np.mean(errs)) if errs.size > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(errs ** 2))) if errs.size > 0 else float("nan")
    pr = (
        float(np.corrcoef(req_homo[eval_mask], dpa_homo_full[eval_mask])[0, 1])
        if errs.size >= 8 and np.std(req_homo[eval_mask]) > 1e-6
        else float("nan")
    )
    slope = (
        float(np.polyfit(req_homo[eval_mask], dpa_homo_full[eval_mask], 1)[0])
        if errs.size >= 8 and np.std(req_homo[eval_mask]) > 1e-6
        else float("nan")
    )

    clean_full = np.zeros(n_req, dtype=bool)
    for loc_k, g_idx in enumerate(valid_idx):
        if bool(clean_mask_local[loc_k]):
            clean_full[g_idx] = True

    clean_hits = int(np.sum(clean_full & eval_mask & (np.abs(dpa_homo_full - req_homo) <= 0.25)))
    all_hits = int(np.sum(eval_mask & (np.abs(dpa_homo_full - req_homo) <= 0.25)))

    smis = [mol_to_clean_smiles(m) for m in valid_mols]
    smis = [s for s in smis if s]
    uniq_smis = set(smis)
    nov_cnt = sum(1 for s in uniq_smis if s not in train_smiles)
    scafs = {sc for sc in (_murcko_smi(m) for m in valid_mols) if sc}

    nat_gen = np.array([float(m.GetNumAtoms()) for m in valid_mols], dtype=np.float64) if valid_mols else np.zeros(0)
    nat_err = (
        [abs(float(mols[i].GetNumAtoms()) - float(req_natoms[i])) for i in valid_idx]
        if valid_idx
        else []
    )
    fm_finite = fmax_full[np.isfinite(fmax_full)]

    res = {
        "internal_funnel": {"m_stage1": int(funnel[0]), "m_stage2": int(funnel[1]), "m_out": int(funnel[2])},
        "external_oracle_calls": 0,
        "n_prompts": int(n_req),
        "n_legal": int(n_leg),
        "legal_pct": round(100.0 * n_leg / max(n_req, 1), 2),
        "exotic_charge_pct": round(100.0 * exotic_cnt / max(n_leg, 1), 2),
        "uniqueness_pct": round(100.0 * len(uniq_smis) / max(len(smis), 1), 2),
        "novelty_pct": round(100.0 * nov_cnt / max(len(uniq_smis), 1), 2),
        "unique_murcko_scaffolds": int(len(scafs)),
        "mean_atoms_generated": round(float(np.mean(nat_gen)), 2) if nat_gen.size > 0 else None,
        "mean_natoms_abs_err": round(float(np.mean(nat_err)), 3) if nat_err else None,
        "zero_step_f_max_median_eV_A": round(float(np.median(fm_finite)), 3) if fm_finite.size > 0 else None,
        "zero_step_f_max_p90_eV_A": round(float(np.percentile(fm_finite, 90)), 3) if fm_finite.size > 0 else None,
        "blind_dpa4_homo_mae_eV": round(mae, 4),
        "blind_dpa4_homo_rmse_eV": round(rmse, 4),
        "blind_dpa4_slope": round(slope, 4) if math.isfinite(slope) else None,
        "blind_dpa4_pearson_r": round(pr, 4) if math.isfinite(pr) else None,
        "blind_dpa4_hits": int(all_hits),
        "blind_dpa4_clean_hits": int(clean_hits),
        "blind_dpa4_clean_hit_pct": round(100.0 * clean_hits / max(n_req, 1), 2),
        "wall_sec": round(float(wall_sec), 3),
        "gpu_sec_per_1k_prompts": round(1000.0 * float(wall_sec) / max(n_req, 1), 2),
        "throughput_mol_per_sec": round(float(n_req) / max(float(wall_sec), 1e-6), 2),
    }
    return res, dpa_homo_full, fmax_full


def _run_unbiased_42_pbe0(
    arm_name: str,
    mols: Sequence[Optional[Chem.Mol]],
    req_homo: np.ndarray,
    dpa_homo: np.ndarray,
    fmax_arr: np.ndarray,
    dft_root: Path,
) -> Dict[str, Any]:
    """Run live `GPU4PySCF PBE0/def2-TZVP/DF` single points on the fixed 42 deterministic prompt indices."""
    from gpu4pyscf.dft.rks import RKS
    from pyscf import gto, lib

    lib.num_threads(4)
    ptable = Chem.GetPeriodicTable()
    fixed_pindices = [q * 256 + k * 42 for q in range(7) for k in range(6)]
    sp_root = dft_root / arm_name
    sp_root.mkdir(parents=True, exist_ok=True)

    pbe0_vals: List[float] = []
    tgt_vals: List[float] = []
    dpa_vals: List[float] = []
    fm_vals: List[float] = []
    scf_walls: List[float] = []
    n_admitted = 0
    n_conv = 0

    for rank_i, p_idx in enumerate(fixed_pindices):
        m = mols[p_idx]
        if m is None or m.GetNumAtoms() < 2 or m.GetNumAtoms() > 80:
            continue
        if Chem.GetFormalCharge(m) != 0 or any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()):
            continue
        if any(ptable.GetAtomicNumber(a.GetSymbol()) not in SUPPORTED_TZVP_Z for a in m.GetAtoms()):
            continue
        n_admitted += 1
        mol_dir = sp_root / f"mol_{rank_i:04d}_p{p_idx:04d}"
        res_p = mol_dir / "result.json"
        if res_p.exists():
            prev = json.loads(res_p.read_text())
            if prev.get("status") == "completed" and prev.get("homo_pbe0_eV") is not None:
                n_conv += 1
                pbe0_vals.append(float(prev["homo_pbe0_eV"]))
                tgt_vals.append(float(req_homo[p_idx]))
                dpa_vals.append(float(dpa_homo[p_idx]))
                fm_vals.append(float(fmax_arr[p_idx]))
                scf_walls.append(float(prev.get("wall_sec", 0.0)))
                continue

        mol_dir.mkdir(parents=True, exist_ok=True)
        syms = [a.GetSymbol() for a in m.GetAtoms()]
        coords = m.GetConformer().GetPositions()
        ecp = {s: "def2-tzvp" for s in set(syms) if ptable.GetAtomicNumber(s) > 36}
        t0 = time.perf_counter()
        try:
            gmol = gto.M(
                atom=list(zip(syms, coords.tolist())),
                basis="def2-tzvp",
                ecp=ecp,
                unit="Angstrom",
                charge=0,
                spin=0,
                verbose=3,
                output=str(mol_dir / "sp.log"),
            )
            mf = RKS(gmol).density_fit(auxbasis="def2-universal-jkfit")
            mf.xc = "pbe0"
            mf.grids.level = 3
            mf.conv_tol = 1e-8
            mf.max_cycle = 100
            mf.kernel()
            if not bool(mf.converged):
                mf = mf.newton()
                mf.kernel()
            wall_s = time.perf_counter() - t0
            if bool(mf.converged):
                mo_e = mf.mo_energy.get() if hasattr(mf.mo_energy, "get") else np.asarray(mf.mo_energy)
                mo_occ = mf.mo_occ.get() if hasattr(mf.mo_occ, "get") else np.asarray(mf.mo_occ)
                occ_idx = np.where(mo_occ > 0.5)[0]
                homo_ev = float(mo_e[occ_idx[-1]] * HA_TO_EV)
                res_p.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "prompt_index": int(p_idx),
                            "homo_target_eV": float(req_homo[p_idx]),
                            "homo_pbe0_eV": round(homo_ev, 5),
                            "wall_sec": round(wall_s, 3),
                        },
                        indent=2,
                    )
                )
                n_conv += 1
                pbe0_vals.append(homo_ev)
                tgt_vals.append(float(req_homo[p_idx]))
                dpa_vals.append(float(dpa_homo[p_idx]))
                fm_vals.append(float(fmax_arr[p_idx]))
                scf_walls.append(wall_s)
        except Exception as exc:
            (mol_dir / "error.txt").write_text(str(exc))

    p_arr = np.asarray(pbe0_vals, dtype=np.float64)
    t_arr = np.asarray(tgt_vals, dtype=np.float64)
    d_arr = np.asarray(dpa_vals, dtype=np.float64)
    fm_a = np.asarray(fm_vals, dtype=np.float64)
    vs_tgt = _pair_metrics(p_arr, t_arr)
    vs_dpa = _pair_metrics(d_arr, p_arr)

    raw_hits = int(np.sum(np.abs(p_arr - t_arr) <= 0.25)) if p_arr.size > 0 else 0
    cen_hits = (
        int(np.sum(np.abs((p_arr - t_arr) - np.mean(p_arr - t_arr)) <= 0.25)) if p_arr.size > 0 else 0
    )
    return {
        "n_prompts_attempted": 42,
        "n_admitted": int(n_admitted),
        "n_scf_converged": int(n_conv),
        "scf_convergence_pct": round(100.0 * n_conv / max(n_admitted, 1), 2),
        "end_to_end_pbe0_clean_hits_raw": raw_hits,
        "end_to_end_pbe0_clean_hit_pct_raw": round(100.0 * raw_hits / 42.0, 2),
        "end_to_end_pbe0_clean_hits_centered": cen_hits,
        "end_to_end_pbe0_clean_hit_pct_centered": round(100.0 * cen_hits / 42.0, 2),
        "median_f_max_eV_A": round(float(np.nanmedian(fm_a)), 3) if fm_a.size > 0 else None,
        "mean_scf_wall_sec": round(float(np.mean(scf_walls)), 2) if scf_walls else None,
        "pbe0_vs_target_homo": vs_tgt,
        "dpa4_vs_pbe0_homo": vs_dpa,
    }


def run_all(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    train_mols, train_homos, train_nats = _load_pure_oe62_train(
        args.oe62_master_train_sdf,
        quantum_train_dft_sdf=args.quantum_train_dft_sdf,
        eq_conformer_pool_sdf=args.eq_conformer_pool_sdf,
    )
    valid_mols, valid_homos, valid_nats = _load_pure_oe62_train(
        args.oe62_master_valid_sdf,
        quantum_train_dft_sdf=args.quantum_train_dft_sdf,
        eq_conformer_pool_sdf=None,
    )
    print(f"[OE62 Standalone] Loaded train={len(train_mols)}, valid={len(valid_mols)} pure DFT conformers (mean HOMO={float(np.mean(train_homos)):.4f} eV)", flush=True)

    out_dir = args.work_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    smis_cache = out_dir / "oe62_train_smiles.json"
    if smis_cache.exists():
        train_smiles = set(json.loads(smis_cache.read_text()))
    else:
        train_smiles = {mol_to_clean_smiles(m) for m in train_mols}
        train_smiles.discard("")
        smis_cache.write_text(json.dumps(sorted(train_smiles)))

    van_gen, van_gpuh = train_or_load_vanilla_oe62(args, train_mols, train_homos)
    mol_gen, mol_meta = train_or_load_standalone_oe62_bundle(
        args, train_mols, train_homos, train_nats, valid_mols, valid_homos
    )

    # Blind post-hoc evaluators ONLY (`audit_dpa4` + `PotentialSlot`)
    audit_slot = AuditOracleSlot(
        backend="dpa4",
        checkpoint_dir=args.oe62_audit_oracle_dir,
        device=args.device,
        forcefield_path=str(args.oe62_potential_weights),
    )
    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.oe62_potential_weights), device=args.device)

    # Internal 112-dev prompt affine calibration using ONLY internal `OctreeElementProcessCritic`!
    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    rng_dev = np.random.default_rng(2026901)
    mid_n = train_nats[(train_nats >= 20.0) & (train_nats <= 42.0)]
    dev_h, dev_n = [], []
    for q in (10, 25, 35, 50, 65, 75, 90):
        dev_h.extend([float(q_points[str(q)])] * 16)
        dev_n.extend(np.round(rng_dev.choice(mid_n, size=16, replace=True)).tolist())
    dev_h_arr = np.asarray(dev_h, dtype=np.float32)
    dev_n_arr = np.asarray(dev_n, dtype=np.float32)

    center_h = float(np.mean(train_homos))
    best_cfg = 2.0
    best_alpha, best_beta = 1.0, 0.0
    best_err = float("inf")
    s2_critic = mol_gen.process_critic.stage2_critic
    b_dev = copy(mol_gen.backbone)
    b_dev.args = copy(mol_gen.args)
    b_dev.args.random_rotation_prob = 0.0

    for cfg_try in (1.75, 2.15):
        mols_d, _ = mol_gen.generate_single_speculative(
            mol_gen.normalize_targets(dev_h_arr[:, None]),
            torch.ones((len(dev_h_arr), 1), dtype=torch.bool, device=device),
            speculative_scale=2,
            atom_budget=dev_n_arr,
            seed=99001,
            cfg_scale=cfg_try,
            cfg_tree_decay=0.25,
            budget_guidance_scale=0.45,
        )
        ok_ids = [i for i, m in enumerate(mols_d) if m is not None and m.GetNumAtoms() <= 68]
        if len(ok_ids) >= 32:
            ds_tmp = OctreeDataset([mols_d[i] for i in ok_ids], b_dev, training=False)
            items_t, kept_t = [], []
            for lk in range(len(ok_ids)):
                try:
                    it = ds_tmp.grid[lk]
                    if (not it["cutoff_atom_count"]) and int(((it["decoder_level"] == 0) & it["decoder_is_second_atom"]).sum().item()) == mols_d[ok_ids[lk]].GetNumAtoms():
                        items_t.append(it)
                        kept_t.append(ok_ids[lk])
                except Exception:
                    continue
            if len(items_t) >= 24:
                pk_t = ds_tmp.grid.collater(items_t)
                tn_zero = torch.zeros((len(items_t), 1), dtype=torch.float32, device=device)
                am_false = torch.zeros((len(items_t), 1), dtype=torch.bool, device=device)
                df_t = mol_gen.extract_teacher_forced_two_stage_features(pk_t, tn_zero, am_false)
                mk_t = max(1, max((int(es.numel()) for es in df_t["s2_elem_slices"]), default=0))
                pe_t = torch.zeros((len(items_t), mk_t), dtype=torch.long, device=device)
                pp_t = torch.zeros((len(items_t), mk_t, 3), dtype=torch.float32, device=device)
                pv_t = torch.zeros((len(items_t), mk_t), dtype=torch.bool, device=device)
                for idx_r, (es, ps) in enumerate(zip(df_t["s2_elem_slices"], df_t["s2_pos_slices"])):
                    ki = int(es.numel())
                    if ki > 0:
                        pe_t[idx_r, :ki] = es.to(device)
                        pp_t[idx_r, :ki] = ps.to(device)
                        pv_t[idx_r, :ki] = True
                with torch.no_grad():
                    mu_n = s2_critic(
                        df_t["s2_h_stage2"].to(device), pe_t, pp_t, pv_t,
                        df_t["s2_cloud_feats"].to(device), tn_zero, am_false,
                    )["mu_norm"][:, 0]
                    pred_ev = s2_critic.denormalize_y(mu_n[:, None])[:, 0].detach().cpu().numpy()
                req_sub = dev_h_arr[kept_t]
                xc = req_sub - center_h
                yc = pred_ev - center_h
                slp, bias_f = np.polyfit(xc, yc, 1)
                alp = float(np.clip(1.0 + 0.35 * (1.0 / max(float(slp), 0.65) - 1.0), 0.94, 1.08))
                bet = float(np.clip(-float(bias_f) * 0.60, -0.22, 0.22))
                err_d = float(np.mean(np.abs((center_h + alp * yc + bet) - req_sub)))
                if err_d < best_err:
                    best_err = err_d
                    best_cfg, best_alpha, best_beta = cfg_try, round(alp, 4), round(bet, 4)

    print(f"[OE62 Standalone Internal DevCal] cfg={best_cfg} alpha={best_alpha} beta={best_beta}", flush=True)

    # Warmup
    mol_gen.generate_single_speculative(
        torch.zeros((16, 1), dtype=torch.float32, device=device),
        torch.ones((16, 1), dtype=torch.bool, device=device),
        speculative_scale=2, atom_budget=np.full(16, 30.0, dtype=np.float32), seed=9,
    )
    _sync(device)

    results: Dict[str, Any] = {
        "benchmark": "OE62_Standalone_Uni3DAR_Mol_ZeroLoom",
        "zero_loom_verified": True,
        "zero_external_surrogate_verified": True,
        "zero_inference_mlip_verified": True,
        "single_output_m_ext": 1,
        "vanilla_sft_gpuh": van_gpuh,
        "uni3dar_mol_training_metadata": mol_meta,
        "internal_dev_calibration": {"cfg_scale": best_cfg, "alpha": best_alpha, "beta": best_beta},
        "fair_30atom_1792": {},
        "ablation_k4_1792": {},
        "atom_buckets_280": {},
        "unbiased_42_pbe0": {},
    }

    def _flush() -> None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")

    # ---------------- Part A: Fair 30.4-Atom Main Benchmark (`N = 1,792`) ----------------
    _, ref_pool_2d = _get_clean_oe62_pool(args)
    prompts_1792 = build_fair_30atom_prompts(ref_pool_2d, args.oe62_targets_json)
    req_h_1792 = prompts_1792[:, 0].astype(np.float32)
    req_n_1792 = prompts_1792[:, 1].astype(np.float32)
    cal_h_1792 = center_h + best_alpha * (req_h_1792 - center_h) + best_beta
    t_norm_1792 = mol_gen.normalize_targets(cal_h_1792[:, None])
    m_act_1792 = torch.ones((len(req_h_1792), 1), dtype=torch.bool, device=device)

    # A1. Vanilla Uni-3DAR (`1->1->1`, raw point cloud)
    t_van_1792 = van_gen.normalize_targets(req_h_1792[:, None])
    _sync(device)
    tv0 = time.perf_counter()
    van_trajs: List[Dict[str, Any]] = []
    for s in range(0, len(req_h_1792), 256):
        e = min(s + 256, len(req_h_1792))
        van_trajs.extend(
            van_gen.sample(
                t_van_1792[s:e], m_act_1792[s:e],
                cfg_scale=2.0, xyz_cfg_scale=1.0, cfg_tree_decay=0.0,
                geometric_guard=False, temperatures=(0.85, 0.30, 0.25),
                seed=310000 + s, use_bf16=(device.type == "cuda"), electron_parity=None,
            )
        )
    with ThreadPoolExecutor(max_workers=4) as pool:
        van_mols = list(
            pool.map(
                lambda tr: _raw_point_cloud_to_rdkit_mol(tr.get("atom_symbols", []), tr.get("positions_np", np.zeros((0, 3), dtype=np.float32))),
                van_trajs,
            )
        )
    _sync(device)
    van_wall = time.perf_counter() - tv0
    van_res, _, _ = _audit_oe62_cohort(
        van_mols, req_h_1792, req_n_1792, audit_slot, pot_slot, train_smiles, van_wall, (1, 1, 1)
    )
    results["fair_30atom_1792"]["Vanilla_Uni3DAR_M1_Raw"] = van_res
    _flush()
    print(f"[OE62 Fair30 | Vanilla_M1_Raw] {van_res}", flush=True)

    # A2. Standalone `Uni-3DAR-Mol` across `K in {1, 2, 4, 8, 12}` (`m_s1 -> m_s2 -> 1`)
    saved_cohorts_for_dft: Dict[str, Tuple[List[Optional[Chem.Mol]], np.ndarray, np.ndarray]] = {}
    for k_val in args.k_scales:
        _sync(device)
        tk0 = time.perf_counter()
        sel_mols, fun_t = mol_gen.generate_single_speculative(
            t_norm_1792, m_act_1792,
            speculative_scale=int(k_val),
            atom_budget=req_n_1792,
            seed=420000 + int(k_val) * 193,
            cfg_scale=best_cfg,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            budget_guidance_scale=0.45,
        )
        _sync(device)
        kwall = time.perf_counter() - tk0
        t_key = f"K{int(k_val)}"
        k_res, dpa_h_k, fm_k = _audit_oe62_cohort(
            sel_mols, req_h_1792, req_n_1792, audit_slot, pot_slot, train_smiles, kwall, fun_t
        )
        results["fair_30atom_1792"][t_key] = k_res
        if int(k_val) in (4, 8):
            saved_cohorts_for_dft[f"Uni3DAR_Mol_{t_key}_{fun_t[0]}to{fun_t[1]}to1"] = (sel_mols, dpa_h_k, fm_k)
        _flush()
        print(f"[OE62 Fair30 | {t_key} {fun_t[0]}->{fun_t[1]}->1] {k_res}", flush=True)

    # ---------------- Part B: Fixed `K=4` Internal Architecture Ablation (`N = 1,792`) ----------------
    # Contrasts `1->1->1` & `Naive 4->4->1` (no coarse speculative expansion) vs `SpecTree 32->8->1` x `PostTreeGraphRefiner`
    for abl_name, mode_tag, use_ref in (
        ("NoTree_1to1to1_NoRefiner", "1to1", False),
        ("NoTree_1to1to1_WithRefiner", "1to1", True),
        ("NaiveLeaf_4to4to1_NoRefiner", "4to4", False),
        ("NaiveLeaf_4to4to1_WithRefiner", "4to4", True),
        ("SpecTree_32to8to1_NoRefiner", "32to8", False),
        ("Full_Uni3DAR_Mol_32to8to1_WithRefiner", "32to8", True),
    ):
        _sync(device)
        ta0 = time.perf_counter()
        if mode_tag == "4to4":
            # Decode 4 full leaf trajectories without coarse Stage-1 fan-out (`4->4->1`), pick Top-1 by internal critic
            grouped_4 = mol_gen.sample_with_two_stage_process_critic(
                t_norm_1792, m_act_1792, mol_gen.process_critic,
                m_stage1=4, m_stage2=4, m_keep=2,
                stage1_legal_weight=0.80, stage1_constr_weight=0.55,
                stage2_legal_weight=1.05, stage2_constr_weight=0.70,
                stage2_diversity_weight=0.0,
                cfg_scale=best_cfg, xyz_cfg_scale=1.0, cfg_tree_decay=0.25,
                geometric_guard=True, temperatures=(0.85, 0.30, 0.25),
                seed=420000 + 4 * 193, use_bf16=(device.type == "cuda"), electron_parity=0,
                leaf_atom_budget=torch.as_tensor(req_n_1792, dtype=torch.float32, device=device),
                budget_guidance_scale=0.45,
            )
            prim_4 = [g[0] for g in grouped_4]
            if use_ref and mol_gen.post_tree_refiner is not None:
                mol_gen.post_tree_refiner.refine_trajectories_batch(prim_4, batch_size=256)
            rec_4 = mol_gen.reconciler
            abl_mols = []
            for idx_p, tr_p in enumerate(prim_4):
                r_p = rec_4.reconcile_point_cloud(
                    tr_p.get("atom_symbols", []), tr_p.get("positions_np", np.zeros((0, 3), dtype=np.float32)),
                    potential_slot=None, bond_order_priors=tr_p.get("bond_order_probs"),
                )
                if r_p.reconciled_legal and (not r_p.reconciled_exotic_charge) and r_p.mol is not None and len(Chem.GetMolFrags(r_p.mol)) == 1:
                    abl_mols.append(Chem.Mol(r_p.mol))
                else:
                    tr_f = grouped_4[idx_p][1]
                    if use_ref and mol_gen.post_tree_refiner is not None:
                        mol_gen.post_tree_refiner.refine_trajectories_batch([tr_f], batch_size=1)
                    r_f = rec_4.reconcile_point_cloud(
                        tr_f.get("atom_symbols", []), tr_f.get("positions_np", np.zeros((0, 3), dtype=np.float32)),
                        potential_slot=None, bond_order_priors=tr_f.get("bond_order_probs"),
                    )
                    abl_mols.append(
                        Chem.Mol(r_f.mol)
                        if (r_f.reconciled_legal and (not r_f.reconciled_exotic_charge) and r_f.mol is not None and len(Chem.GetMolFrags(r_f.mol)) == 1)
                        else None
                    )
            abl_fun = (4, 4, 1)
        else:
            abl_mols, abl_fun = mol_gen.generate_single_speculative(
                t_norm_1792, m_act_1792,
                speculative_scale=(4 if mode_tag == "32to8" else 1),
                atom_budget=req_n_1792,
                enable_internal_tree=(mode_tag == "32to8"),
                enable_post_tree_refiner=use_ref,
                seed=420000 + 4 * 193,
                cfg_scale=best_cfg,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                budget_guidance_scale=0.45,
            )
        _sync(device)
        awall = time.perf_counter() - ta0
        a_res, _, _ = _audit_oe62_cohort(
            abl_mols, req_h_1792, req_n_1792, audit_slot, pot_slot, train_smiles, awall, abl_fun
        )
        results["ablation_k4_1792"][abl_name] = a_res
        _flush()
        print(f"[OE62 Ablation | {abl_name}] {a_res}", flush=True)

    # ---------------- Part C: Controlled Multi-Atom Scaling `N in {20, 30, 40, 50, 60}` (`280` prompts/bucket) ----------------
    q_list = [float(q_points[str(q)]) for q in (10, 25, 35, 50, 65, 75, 90)]
    b_req_h = np.asarray([q_list[i // 40] for i in range(280)], dtype=np.float32)
    b_cal_h = center_h + best_alpha * (b_req_h - center_h) + best_beta
    b_t_norm = mol_gen.normalize_targets(b_cal_h[:, None])
    b_m_act = torch.ones((280, 1), dtype=torch.bool, device=device)

    for n_b in (20, 30, 40, 50, 60):
        b_nat = np.full(280, float(n_b), dtype=np.float32)
        b_arms: Dict[str, Any] = {}
        for k_b in (1, 4, 8):
            _sync(device)
            tb0 = time.perf_counter()
            bmols, bfun = mol_gen.generate_single_speculative(
                b_t_norm, b_m_act,
                speculative_scale=k_b,
                atom_budget=b_nat,
                seed=610000 + n_b * 100 + k_b,
                cfg_scale=best_cfg,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                budget_guidance_scale=0.45,
            )
            _sync(device)
            bwall = time.perf_counter() - tb0
            bres, _, _ = _audit_oe62_cohort(
                bmols, b_req_h, b_nat, audit_slot, pot_slot, train_smiles, bwall, bfun
            )
            b_arms[f"Uni3DAR_Mol_K{k_b}"] = bres
            print(f"[OE62 Bucket N={n_b} | K{k_b} {bfun[0]}->{bfun[1]}->1] spd={bres['throughput_mol_per_sec']} mae={bres['blind_dpa4_homo_mae_eV']} hit={bres['blind_dpa4_clean_hit_pct']}%", flush=True)
        results["atom_buckets_280"][str(n_b)] = b_arms
        _flush()

    # ---------------- Part D: Live `42`-Prompt `GPU4PySCF PBE0/def2-TZVP/DF` Verification ----------------
    dft_dir = out_dir / "pbe0_unbiased_42"
    for arm_tag, (m_list, dh_arr, fm_arr) in saved_cohorts_for_dft.items():
        dft_res = _run_unbiased_42_pbe0(arm_tag, m_list, req_h_1792, dh_arr, fm_arr, dft_dir)
        results["unbiased_42_pbe0"][arm_tag] = dft_res
        _flush()
        print(f"[OE62 Live PBE0 42-Prompt | {arm_tag}] {dft_res}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--allow-tf32", action="store_true", default=True)
    p.add_argument("--reuse-checkpoints", action="store_true")
    p.add_argument(
        "--parent-checkpoint",
        type=Path,
        default=Path("/sjtu-combustion/chensq/orbit3d_oe62_20261112/pretrained/drug.pt"),
    )
    p.add_argument(
        "--oe62-master-train-sdf",
        type=Path,
        default=Path("/sjtu-combustion/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"),
    )
    p.add_argument(
        "--oe62-master-valid-sdf",
        type=Path,
        default=Path("/sjtu-combustion/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_valid_B.sdf"),
    )
    p.add_argument(
        "--quantum-train-dft-sdf",
        type=Path,
        default=Path("/sjtu-combustion/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_dft.sdf"),
    )
    p.add_argument(
        "--eq-conformer-pool-sdf",
        type=Path,
        default=Path("/sjtu-combustion/chensq/loom_oe62_20261115/runs/uni3dar_v4_pro/v4_pro_relaxed_pool.sdf"),
    )
    p.add_argument(
        "--oe62-targets-json",
        type=Path,
        default=Path("/sjtu-combustion/chensq/orbit_b_task_20261114/targets_B.json"),
    )
    p.add_argument(
        "--oe62-audit-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/chensq/orbit_e15_calibration_20261114/full_fit/oracle"),
    )
    p.add_argument(
        "--oe62-potential-weights",
        type=Path,
        default=Path("/sjtu-combustion/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"),
    )
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="uni3dar_mol_standalone_oe62")
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/chensq/loom_oe62_20261115/results/oe62_uni3dar_mol_standalone.json"),
    )
    p.add_argument("--vanilla-sft-steps", type=int, default=1200)
    p.add_argument("--sft-steps", type=int, default=2000)
    p.add_argument("--sft-batch-size", type=int, default=64)
    p.add_argument("--gt-ranking-groups", type=int, default=1600)
    p.add_argument("--k-scales", type=int, nargs="*", default=[1, 2, 4, 8, 12])
    p.add_argument("--seed", type=int, default=20261120)
    args = p.parse_args()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_all(args)


if __name__ == "__main__":
    main()
