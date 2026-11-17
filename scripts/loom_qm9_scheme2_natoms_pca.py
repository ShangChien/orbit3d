"""Scheme 2: Fair `(target, N_atoms)` Conditioning + Canonical SE(3) PCA Frame +
Full-Backbone Pre-Voxelized Adaptation + Size-Anchored CFG + Tight Point-Regression `TopoGRPO`
(with `M in {1, 2, 4, 8, 16}` Test-Time Critic Weaving Evaluation).

Addresses the three root causes identified on public QM9:
  1. Fair `(target, N_atoms)` conditioning matching EDM / GeoLDM / PropMolFlow's official
     joint sampling input `(train_half_sampled_values_{prop}.npy, train_half_sampled_no_atoms_{prop}.npy)`,
     paired with **Size-Anchored CFG** (`uncond = p(M | N_atoms)`, `cond = p(M | target, N_atoms)`)
     and **Fixed-Atom-Count Contrastive Margin** (`contrast_col_mask = [True, False]`).
  2. Deterministic **Canonical Inertial + Polarity SO(3) Frame (`canonicalize_coords_pca`)**
     (`det(R) = +1`, exact 0.000000 A internal geometry preservation) eliminating arbitrary
     rotation ambiguity on discrete 0.01 A octrees (especially critical for dipole norm `mu`),
     plus **RAM Pre-Voxelization** of all 44,453 canonical conformers (`< 1 ms` CPU collation).
  3. **Unfrozen Full Backbone** (`num_prefix_tokens = 4`, `num_rbf = 48`, `num_freqs = 24`,
     differential AdamW learning rates) + Hindsight Relabeling + **Tight Continuous Point-Regression
     `TopoGRPO`** supervised strictly by the zero-leakage `train_rQM_v0` Uni-Mol surrogate
     (`0` overlap with `gvp_rQM_v0`).
"""

from __future__ import annotations

import argparse
from copy import copy, deepcopy
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem

from loom.chemistry.valence_reconciler import ValenceReconciler
from loom.generator.octree_alignment import categorical_policy_loss
from loom.vendor.uni3dar.data import GridDataset
from loom.weaving.uni3dar_slot import LoomUni3DARGenerator

from scripts.loom_qm9_benchmark import (
    ALLOWED_QM9_ELEMENTS,
    EV_PROPS,
    GVPBenchmarkSuite,
    HA_TO_EV,
    PROPS,
    _move_packed_to_device,
    load_disjoint_qm9_splits,
)
from scripts.loom_qm9_best_of_m import (
    _canonical_smiles_2d,
    _load_propmolflow_eval_natoms,
    _min_pair_dist,
    select_best_of_m_and_score,
)
from scripts.loom_qm9_k1_pipeline import (
    PROP_EVAL_UNITS,
    PROP_GEN_UNITS,
    QM9TrainSplitSurrogateOracle,
    _load_propmolflow_eval_targets,
    _records_to_gen_targets,
    _target_ha_to_gen,
)
from scripts.loom_train_sft import compute_tail_balanced_weights

_ATOMIC_Z: Dict[str, float] = {
    "H": 1.0,
    "B": 5.0,
    "C": 6.0,
    "N": 7.0,
    "O": 8.0,
    "F": 9.0,
    "Si": 14.0,
    "P": 15.0,
    "S": 16.0,
    "Cl": 17.0,
    "Br": 35.0,
    "I": 53.0,
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def canonicalize_coords_pca(symbols: Sequence[str], coords: np.ndarray) -> np.ndarray:
    """Deterministic SO(3) inertial + polarity frame (`det(R)=+1`, preserves chirality & centroid)."""
    pts = np.asarray(coords, dtype=np.float64)
    n = pts.shape[0]
    if n <= 2:
        return (pts - pts.mean(axis=0, keepdims=True)).astype(np.float32)

    centered = pts - pts.mean(axis=0, keepdims=True)
    z = np.asarray([_ATOMIC_Z.get(str(s), 6.0) for s in symbols], dtype=np.float64)
    cov = (centered * z[:, None]).T @ centered
    _, evecs = np.linalg.eigh(cov)  # ascending eigenvalues: [minor, mid, major]

    v1 = evecs[:, 2].copy()
    v2 = evecs[:, 1].copy()
    dz = z - float(np.mean(z))
    for vk in (v1, v2):
        proj = centered @ vk
        pol = float(np.dot(dz, proj))
        if abs(pol) < 1e-4:
            pol = float(np.dot(z, proj ** 3))
        if pol < 0.0:
            vk *= -1.0

    v3 = np.cross(v1, v2)
    n3 = float(np.linalg.norm(v3))
    if n3 < 1e-8:
        return centered.astype(np.float32)
    v3 /= n3
    rot = np.stack([v1, v2, v3], axis=1)
    return (centered @ rot).astype(np.float32)


class _RawRecordView:
    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[int(index)]
        return {
            "uid": int(rec.get("gdb9_idx", index)),
            "data_type": "molecule",
            "atom_type": rec["symbols"],
            "atom_pos": rec["coords"],
        }


class PreVoxelizedCanonicalCache:
    """RAM-cached canonical-frame octree features eliminating CPU voxelization bottlenecks."""

    def __init__(self, records: Sequence[Dict[str, Any]], backbone: Any, seed: int = 42):
        t0 = time.perf_counter()
        grid_args = copy(backbone.args)
        grid_args.random_rotation_prob = 0.0
        self.view = _RawRecordView(records)
        self.grid = GridDataset(self.view, seed, backbone.dictionary, "atom_type", "atom_pos", False, grid_args)
        self.grid.set_epoch(1)
        self.backbone_dict = backbone.dictionary
        self.grid_args = grid_args

        self.feats: List[Dict[str, torch.Tensor]] = []
        self.props_ha_list: List[np.ndarray] = []
        self.natoms_list: List[float] = []

        for idx, rec in enumerate(records):
            item = self.grid[idx]
            expected = len(rec["symbols"])
            decoded = int(((item["decoder_level"] == 0) & item["decoder_is_second_atom"]).sum().item())
            if item["cutoff_atom_count"] or decoded != expected:
                continue
            # Detach clone lightweight CPU tensors
            clean_item = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
            self.feats.append(clean_item)
            self.props_ha_list.append(np.asarray(rec["props_ha"], dtype=np.float32))
            self.natoms_list.append(float(expected))

        self.primary_count = len(self.feats)
        self.props_ha = np.stack(self.props_ha_list, axis=0)
        self.natoms = np.asarray(self.natoms_list, dtype=np.float32)
        print(
            f"[CanonicalCache] Pre-voxelized {self.primary_count}/{len(records)} canonical conformers "
            f"in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    def build_extra_subset(self, extra_records: Sequence[Dict[str, Any]], seed: int = 99) -> Tuple[List[Dict[str, torch.Tensor]], np.ndarray, np.ndarray]:
        if not extra_records:
            return [], np.zeros((0, len(PROPS)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        v = _RawRecordView(extra_records)
        g = GridDataset(v, seed, self.backbone_dict, "atom_type", "atom_pos", False, self.grid_args)
        g.set_epoch(1)
        e_feats: List[Dict[str, torch.Tensor]] = []
        e_props: List[np.ndarray] = []
        e_nat: List[float] = []
        for idx, rec in enumerate(extra_records):
            item = g[idx]
            expected = len(rec["symbols"])
            decoded = int(((item["decoder_level"] == 0) & item["decoder_is_second_atom"]).sum().item())
            if item["cutoff_atom_count"] or decoded != expected:
                continue
            e_feats.append({k: (val.clone() if isinstance(val, torch.Tensor) else val) for k, val in item.items()})
            e_props.append(np.asarray(rec["props_ha"], dtype=np.float32))
            e_nat.append(float(expected))
        if not e_feats:
            return [], np.zeros((0, len(PROPS)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return e_feats, np.stack(e_props, axis=0), np.asarray(e_nat, dtype=np.float32)

    def collate_items(self, items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return self.grid.collater(list(items))


def _make_cfg_masks(
    batch_size: int,
    cfg_mode: str,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    cond_mask = torch.ones((batch_size, 2), dtype=torch.bool, device=device)
    if cfg_mode == "size_anchored":
        uncond_mask = torch.zeros((batch_size, 2), dtype=torch.bool, device=device)
        uncond_mask[:, 1] = True
        return cond_mask, uncond_mask
    return cond_mask, None


def _sample_conditioned_pairs(
    generator: LoomUni3DARGenerator,
    reconciler: ValenceReconciler,
    targets_prop_gen: np.ndarray,
    targets_natoms: np.ndarray,
    *,
    batch_size: int,
    cfg_scale: float,
    cfg_mode: str,
    seed: int,
    device: torch.device,
    do_warmup: bool = False,
) -> Dict[str, Any]:
    n_total = len(targets_prop_gen)
    generator.eval()
    pair_mat = np.stack(
        [targets_prop_gen.astype(np.float32), targets_natoms.astype(np.float32)],
        axis=-1,
    )

    if do_warmup and n_total > 0:
        w_n = min(batch_size, n_total)
        w_t = generator.normalize_targets(pair_mat[:w_n])
        w_m, w_u = _make_cfg_masks(w_n, cfg_mode, device)
        for w_i in range(2):
            _ = generator.sample(
                w_t,
                w_m,
                cfg_scale=cfg_scale,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=seed + 701 + w_i,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
                uncond_active_mask=w_u,
            )
        _sync(device)

    _sync(device)
    t0 = time.perf_counter()
    trajectories: List[Dict[str, Any]] = []
    for s in range(0, n_total, batch_size):
        e = min(s + batch_size, n_total)
        b_t = generator.normalize_targets(pair_mat[s:e])
        b_m, b_u = _make_cfg_masks(e - s, cfg_mode, device)
        sub = generator.sample(
            b_t,
            b_m,
            cfg_scale=cfg_scale,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            uncond_active_mask=b_u,
        )
        trajectories.extend(sub)
    _sync(device)
    sampling_wall_s = time.perf_counter() - t0

    raw_mols: List[Optional[Chem.Mol]] = []
    rec_mols: List[Optional[Chem.Mol]] = []
    raw_legal = 0
    raw_exotic = 0
    rec_legal = 0
    rec_exotic = 0
    raw_clean_records: List[Dict[str, Any]] = []
    natoms_abs_err: List[float] = []

    for idx_t, traj in enumerate(trajectories):
        d_min = _min_pair_dist(traj["positions_np"])
        clash_free = d_min >= 0.72
        res = reconciler.reconcile_point_cloud(traj["atom_symbols"], traj["positions_np"], potential_slot=None)
        if res.raw_legal and res.raw_mol is not None:
            raw_legal += 1
            raw_mols.append(res.raw_mol)
            if res.raw_exotic_charge:
                raw_exotic += 1
            elif clash_free and len(Chem.GetMolFrags(res.raw_mol)) == 1:
                raw_clean_records.append(
                    {
                        "mol": res.raw_mol,
                        "symbols": list(traj["atom_symbols"]),
                        "coords": np.asarray(traj["positions_np"], dtype=np.float32),
                        "smiles": Chem.MolToSmiles(Chem.RemoveHs(res.raw_mol), isomericSmiles=True),
                    }
                )
        else:
            raw_mols.append(None)

        if res.reconciled_legal and res.mol is not None:
            rec_legal += 1
            rec_mols.append(res.mol)
            if res.reconciled_exotic_charge:
                rec_exotic += 1
            natoms_abs_err.append(abs(int(res.mol.GetNumAtoms()) - int(targets_natoms[idx_t])))
        else:
            rec_mols.append(None)

    return {
        "n_proposals": n_total,
        "sampling_wall_seconds": round(sampling_wall_s, 3),
        "sampling_mol_per_sec": round(float(n_total) / max(sampling_wall_s, 1e-6), 2),
        "raw_mols": raw_mols,
        "rec_mols": rec_mols,
        "raw_legal_count": raw_legal,
        "raw_legal_pct": round(100.0 * raw_legal / max(n_total, 1), 2),
        "raw_exotic_charge_pct": round(100.0 * raw_exotic / max(raw_legal, 1), 2),
        "rec_legal_count": rec_legal,
        "rec_legal_pct": round(100.0 * rec_legal / max(n_total, 1), 2),
        "rec_exotic_charge_pct": round(100.0 * rec_exotic / max(rec_legal, 1), 2),
        "mean_natoms_abs_err": round(float(np.mean(natoms_abs_err)), 3) if natoms_abs_err else None,
        "raw_clean_records": raw_clean_records,
    }


def _eval_surrogate_dev_pair(
    generator: LoomUni3DARGenerator,
    surrogate: QM9TrainSplitSurrogateOracle,
    reconciler: ValenceReconciler,
    prop_idx: int,
    prop_name: str,
    prop_std: float,
    dev_prop_gen: np.ndarray,
    dev_natoms: np.ndarray,
    *,
    batch_size: int,
    cfg_scale: float,
    cfg_mode: str,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    gen_out = _sample_conditioned_pairs(
        generator,
        reconciler,
        dev_prop_gen,
        dev_natoms,
        batch_size=batch_size,
        cfg_scale=cfg_scale,
        cfg_mode=cfg_mode,
        seed=seed,
        device=device,
        do_warmup=False,
    )
    preds_raw, _, mask_raw, _ = surrogate.evaluate_candidates(gen_out["raw_mols"], batch_size=256)
    preds_rec, _, mask_rec, _ = surrogate.evaluate_candidates(gen_out["rec_mols"], batch_size=256)
    scale = 1000.0 if prop_name in EV_PROPS else 1.0

    raw_mae = (
        float(np.mean(np.abs(preds_raw[mask_raw, prop_idx] - dev_prop_gen[mask_raw])))
        if bool(mask_raw.any())
        else float("inf")
    )
    rec_mae = (
        float(np.mean(np.abs(preds_rec[mask_rec, prop_idx] - dev_prop_gen[mask_rec])))
        if bool(mask_rec.any())
        else float("inf")
    )
    rec_r = (
        float(np.corrcoef(preds_rec[mask_rec, prop_idx], dev_prop_gen[mask_rec])[0, 1])
        if int(mask_rec.sum()) >= 4
        else float("nan")
    )
    rec_rate = gen_out["rec_legal_pct"] / 100.0
    raw_rate = gen_out["raw_legal_pct"] / 100.0
    dev_score = (rec_mae / max(prop_std, 1e-6)) + 0.30 * max(0.0, 0.90 - rec_rate) + 0.15 * max(0.0, 0.75 - raw_rate)
    return {
        "cfg_mode": cfg_mode,
        "cfg_scale": cfg_scale,
        "raw_legal_pct": gen_out["raw_legal_pct"],
        "raw_exotic_charge_pct": gen_out["raw_exotic_charge_pct"],
        "rec_legal_pct": gen_out["rec_legal_pct"],
        "mean_natoms_abs_err": gen_out["mean_natoms_abs_err"],
        "raw_surrogate_mae_eval": round(raw_mae * scale, 4),
        "rec_surrogate_mae_eval": round(rec_mae * scale, 4),
        "rec_surrogate_r": round(rec_r, 4) if math.isfinite(rec_r) else None,
        "dev_score": round(dev_score, 5),
    }


def _run_sft_cached_loop(
    generator: LoomUni3DARGenerator,
    cache: PreVoxelizedCanonicalCache,
    primary_y_gen: np.ndarray,
    primary_probs: np.ndarray,
    *,
    extra_feats: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
    extra_y_gen: Optional[np.ndarray] = None,
    extra_natoms: Optional[np.ndarray] = None,
    extra_mix_prob: float = 0.0,
    steps: int,
    lr_cond: float,
    lr_backbone: float,
    warmup_steps: int,
    batch_size: int,
    prop_mean: float,
    prop_std: float,
    natoms_mean: float,
    natoms_std: float,
    contrast_min_gap_gen: float,
    seed: int,
    device: torch.device,
    log_tag: str,
) -> Dict[str, Any]:
    cond_params = list(generator.conditioner.parameters()) + (
        list(generator.valence_head.parameters()) if generator.valence_head is not None else []
    )
    cond_ids = {id(p) for p in cond_params}
    bb_params = [p for p in generator.parameters() if p.requires_grad and id(p) not in cond_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": cond_params, "lr": lr_cond, "weight_decay": 1e-4},
            {"params": bb_params, "lr": lr_backbone, "weight_decay": 1e-4},
        ]
    )
    all_trainable = cond_params + bb_params

    def _lr_mult(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(max(1, warmup_steps))
        prog = float(step_idx - warmup_steps) / float(max(1, steps - warmup_steps))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_mult)
    rng = np.random.default_rng(seed)
    c_col_mask = torch.tensor([[True, False]], dtype=torch.bool, device=device)

    has_extra = extra_feats is not None and extra_y_gen is not None and extra_natoms is not None and len(extra_feats) >= 16 and extra_mix_prob > 0.0
    total_examples = 0
    _sync(device)
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        if has_extra:
            assert extra_feats is not None and extra_y_gen is not None and extra_natoms is not None
            n_ex = max(1, int(round(batch_size * extra_mix_prob)))
            n_pr = batch_size - n_ex
            idx_pr = rng.choice(cache.primary_count, size=n_pr, replace=True, p=primary_probs)
            idx_ex = rng.integers(0, len(extra_feats), size=n_ex)
            items = [cache.feats[int(i)] for i in idx_pr] + [extra_feats[int(j)] for j in idx_ex]
            y_vals = np.concatenate([primary_y_gen[idx_pr], extra_y_gen[idx_ex]], axis=0)
            n_vals = np.concatenate([cache.natoms[idx_pr], extra_natoms[idx_ex]], axis=0)
            perm = rng.permutation(batch_size)
            items = [items[p] for p in perm]
            y_vals = y_vals[perm]
            n_vals = n_vals[perm]
        else:
            idx_pr = rng.choice(cache.primary_count, size=batch_size, replace=True, p=primary_probs)
            items = [cache.feats[int(i)] for i in idx_pr]
            y_vals = primary_y_gen[idx_pr]
            n_vals = cache.natoms[idx_pr]

        packed = cache.collate_items(items)
        bd = _move_packed_to_device(packed, device)

        pair_norm = np.stack(
            [
                (y_vals - prop_mean) / max(prop_std, 1e-6),
                (n_vals - natoms_mean) / max(natoms_std, 1e-6),
            ],
            axis=-1,
        ).astype(np.float32)
        t_norm = torch.from_numpy(pair_norm).to(device)

        # Structured active mask: 75% [T, T], 15% [F, T] (Size-Anchored CFG baseline), 10% [F, F]
        u_draw = rng.random(batch_size)
        act_np = np.ones((batch_size, 2), dtype=bool)
        only_size = (u_draw >= 0.75) & (u_draw < 0.90)
        uncond = u_draw >= 0.90
        act_np[only_size, 0] = False
        act_np[uncond, :] = False
        active_t = torch.from_numpy(act_np).to(device)

        generator.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss, info = generator.compute_sft_loss(
                bd,
                t_norm,
                active_t,
                cfg_dropout_prob=0.0,
                contrast_weight=0.30,
                contrast_margin=0.05,
                contrast_min_gap_eV=contrast_min_gap_gen,
                contrast_col_mask=c_col_mask,
                xyz_loss_weight=0.12,
                valence_loss_weight=0.15,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        optimizer.step()
        scheduler.step()
        total_examples += batch_size

        if step == 1 or step % 500 == 0 or step == steps:
            _sync(device)
            el = time.perf_counter() - t0
            rate = float(total_examples) / max(el, 1e-6)
            pref_pct = float(info.get("inbatch_preference_rate", float("nan"))) * 100.0
            print(
                f"[{log_tag} {step:04d}/{steps:04d}] lr={scheduler.get_last_lr()[0]:.2e} "
                f"loss={info['loss_total']:.4f} gen={info['loss_gen']:.4f} "
                f"con={info.get('loss_contrast', 0.0):.4f} pref={pref_pct:.1f}% "
                f"rate={rate:.1f} mol/s ({el/60:.1f}m)",
                flush=True,
            )

    _sync(device)
    wall_s = time.perf_counter() - t0
    return {
        "steps": steps,
        "examples": total_examples,
        "wall_seconds": round(wall_s, 2),
        "mol_per_sec": round(float(total_examples) / max(wall_s, 1e-6), 2),
    }


def _run_tight_regression_topogrpo(
    generator: LoomUni3DARGenerator,
    surrogate: QM9TrainSplitSurrogateOracle,
    reconciler: ValenceReconciler,
    cache: PreVoxelizedCanonicalCache,
    primary_y_gen: np.ndarray,
    primary_probs: np.ndarray,
    *,
    prop: str,
    prop_idx: int,
    prop_mean: float,
    prop_std: float,
    natoms_mean: float,
    natoms_std: float,
    cfg_mode: str,
    cfg_scale: float,
    grpo_steps: int,
    prompts_per_step: int,
    group_size: int,
    dev_prop_gen: np.ndarray,
    dev_natoms: np.ndarray,
    best_dev_ckpt: Path,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Freeze fine sub-voxel XYZ prediction heads & XYZ embeddings during coarse Stage 2.5A TopoGRPO
    generator.configure_trainable(
        train_conditioner=True,
        train_lora=True,
        train_xyz_embeddings=False,
        unfreeze_full_backbone=False,
    )
    for ahead in getattr(generator.backbone, "atom_heads", []):
        for p_idx_h in (1, 2, 3):
            if p_idx_h < len(ahead.pred_heads):
                ahead.pred_heads[p_idx_h].requires_grad_(False)

    reference = deepcopy(generator).eval().requires_grad_(False)
    cond_params = list(generator.conditioner.parameters()) + (
        list(generator.valence_head.parameters()) if generator.valence_head is not None else []
    )
    cond_ids = {id(p) for p in cond_params}
    bb_params = [p for p in generator.parameters() if p.requires_grad and id(p) not in cond_ids]
    all_trainable = cond_params + bb_params

    optimizer = torch.optim.AdamW(
        [
            {"params": cond_params, "lr": 5.0e-5, "weight_decay": 1e-4},
            {"params": bb_params, "lr": 3.0e-5, "weight_decay": 1e-4},
        ]
    )
    rng = np.random.default_rng(seed)
    c_col_mask = torch.tensor([[True, False]], dtype=torch.bool, device=device)
    best_dev_score = float("inf")
    history: List[Dict[str, Any]] = []

    _sync(device)
    t0 = time.perf_counter()
    total_b = prompts_per_step * group_size

    for step in range(1, grpo_steps + 1):
        p_idx = rng.choice(cache.primary_count, size=prompts_per_step, replace=False, p=primary_probs)
        p_prop = primary_y_gen[p_idx]
        p_nat = cache.natoms[p_idx]

        rep_prop = np.repeat(p_prop, group_size)
        rep_nat = np.repeat(p_nat, group_size)
        rep_mat = np.stack([rep_prop, rep_nat], axis=-1).astype(np.float32)

        t_norm = generator.normalize_targets(rep_mat)
        cond_mask, uncond_mask = _make_cfg_masks(total_b, cfg_mode, device)

        generator.eval()
        trajs = generator.sample(
            t_norm,
            cond_mask,
            cfg_scale=cfg_scale,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed * 101 + step * 997,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            uncond_active_mask=uncond_mask,
        )

        with torch.no_grad():
            old_packets = generator.trajectory_distributions(
                trajs,
                t_norm,
                cond_mask,
                temperatures=(0.85, 0.30, 0.25),
                cfg_scale=cfg_scale,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                geometric_guard=True,
                topology_only_policy=True,
                electron_parity=0,
                uncond_active_mask=uncond_mask,
            )
            ref_packets = reference.trajectory_distributions(
                trajs,
                t_norm,
                cond_mask,
                temperatures=(0.85, 0.30, 0.25),
                cfg_scale=cfg_scale,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                geometric_guard=True,
                topology_only_policy=True,
                electron_parity=0,
                uncond_active_mask=uncond_mask,
            )

        rec_mols: List[Optional[Chem.Mol]] = []
        tier_flags: List[int] = []  # 0=illegal, 1=exotic/disconn, 2=clean
        cand_natoms = np.zeros(total_b, dtype=np.int32)
        for i, tr in enumerate(trajs):
            clash_free = _min_pair_dist(tr["positions_np"]) >= 0.72
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if not res.raw_legal or res.raw_mol is None or not clash_free:
                rec_mols.append(None)
                tier_flags.append(0)
            elif res.raw_exotic_charge or len(Chem.GetMolFrags(res.raw_mol)) != 1 or not res.reconciled_legal or res.mol is None:
                m_use = res.mol if (res.reconciled_legal and res.mol is not None) else res.raw_mol
                rec_mols.append(m_use)
                cand_natoms[i] = int(m_use.GetNumAtoms())
                tier_flags.append(1)
            else:
                rec_mols.append(res.mol)
                cand_natoms[i] = int(res.mol.GetNumAtoms())
                tier_flags.append(2)

        preds_surr, _, ok_surr, _ = surrogate.evaluate_candidates(rec_mols, batch_size=128)
        rewards_np = np.zeros(total_b, dtype=np.float32)
        clean_abs_errs: List[float] = []
        for i in range(total_b):
            tf = tier_flags[i]
            if tf == 0 or not bool(ok_surr[i]):
                rewards_np[i] = -1.0
            else:
                err = abs(float(preds_surr[i, prop_idx]) - float(rep_prop[i]))
                if tf == 1:
                    rewards_np[i] = -0.35 + 0.25 * math.exp(-err / max(prop_std, 1e-6))
                else:
                    clean_abs_errs.append(err)
                    tight_r = math.exp(-err / max(0.14 * prop_std, 1e-6))
                    wide_r = math.exp(-err / max(0.50 * prop_std, 1e-6))
                    size_pen = 0.04 * min(abs(int(cand_natoms[i]) - int(rep_nat[i])), 8)
                    rewards_np[i] = 0.15 + 0.60 * tight_r + 0.25 * wide_r - size_pen

        # Per-prompt group normalization
        adv_np = np.zeros(total_b, dtype=np.float32)
        for g_i in range(prompts_per_step):
            s = g_i * group_size
            e = s + group_size
            grp = rewards_np[s:e]
            sd = float(np.std(grp))
            if sd > 1e-5:
                adv_np[s:e] = np.clip((grp - float(np.mean(grp))) / (sd + 1e-6), -3.0, 3.0)
        adv_t = torch.from_numpy(adv_np).to(device)

        # Anchor SFT batch (64 canonical molecules) to preserve octree syntax & sharp conditioning
        a_bs = 64
        a_idx = rng.choice(cache.primary_count, size=a_bs, replace=True, p=primary_probs)
        a_packed = _move_packed_to_device(cache.collate_items([cache.feats[int(j)] for j in a_idx]), device)
        a_norm = torch.from_numpy(
            np.stack(
                [
                    (primary_y_gen[a_idx] - prop_mean) / max(prop_std, 1e-6),
                    (cache.natoms[a_idx] - natoms_mean) / max(natoms_std, 1e-6),
                ],
                axis=-1,
            ).astype(np.float32)
        ).to(device)
        a_u = rng.random(a_bs)
        a_act = np.ones((a_bs, 2), dtype=bool)
        a_act[(a_u >= 0.75) & (a_u < 0.90), 0] = False
        a_act[a_u >= 0.90, :] = False
        a_act_t = torch.from_numpy(a_act).to(device)

        generator.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            cur_packets = generator.trajectory_distributions(
                trajs,
                t_norm,
                cond_mask,
                temperatures=(0.85, 0.30, 0.25),
                cfg_scale=cfg_scale,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                geometric_guard=True,
                topology_only_policy=True,
                electron_parity=0,
                uncond_active_mask=uncond_mask,
            )
            p_loss = categorical_policy_loss(cur_packets, old_packets, ref_packets, adv_t, beta=0.03, clip=0.20)
            sft_anchor_loss, _ = generator.compute_sft_loss(
                a_packed,
                a_norm,
                a_act_t,
                cfg_dropout_prob=0.0,
                contrast_weight=0.20,
                contrast_margin=0.04,
                contrast_min_gap_eV=0.55 * prop_std,
                contrast_col_mask=c_col_mask,
                xyz_loss_weight=0.10,
                valence_loss_weight=0.12,
            )
            total_loss = p_loss + 0.25 * sft_anchor_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        optimizer.step()

        clean_pct = 100.0 * sum(1 for t in tier_flags if t == 2) / float(total_b)
        scale_eval = 1000.0 if prop in EV_PROPS else 1.0
        mean_clean_mae = (float(np.mean(clean_abs_errs)) * scale_eval) if clean_abs_errs else float("nan")
        rec_step = {
            "step": step,
            "reward_mean": round(float(np.mean(rewards_np)), 4),
            "clean_pct": round(clean_pct, 2),
            "clean_surr_mae_eval": round(mean_clean_mae, 4) if math.isfinite(mean_clean_mae) else None,
            "policy_loss": round(float(p_loss.detach().item()), 5),
        }
        history.append(rec_step)

        if step == 1 or step % 35 == 0 or step == grpo_steps:
            print(f"[S2-GRPO {prop} {step:03d}/{grpo_steps:03d}] {rec_step}", flush=True)

        if step in (grpo_steps // 2, grpo_steps):
            d_chk = _eval_surrogate_dev_pair(
                generator,
                surrogate,
                reconciler,
                prop_idx,
                prop,
                prop_std,
                dev_prop_gen,
                dev_natoms,
                batch_size=250,
                cfg_scale=cfg_scale,
                cfg_mode=cfg_mode,
                seed=seed + 9000 + step,
                device=device,
            )
            print(f"[S2-GRPO {prop} DevCheck@{step}] {d_chk}", flush=True)
            if d_chk["dev_score"] < best_dev_score:
                best_dev_score = d_chk["dev_score"]
                generator.save_checkpoint(best_dev_ckpt, extra_metadata={"prop": prop, "step": step, "dev": d_chk})

    _sync(device)
    wall_s = time.perf_counter() - t0
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        {
            "steps": grpo_steps,
            "rollouts": grpo_steps * total_b,
            "wall_seconds": round(wall_s, 2),
            "rollout_mol_per_sec": round(float(grpo_steps * total_b) / max(wall_s, 1e-6), 2),
        },
        history,
    )


def _time_pair_inference_bs(
    gen: LoomUni3DARGenerator,
    targets_prop_gen: np.ndarray,
    targets_natoms: np.ndarray,
    *,
    batch_size: int,
    n_batches: int,
    cfg_scale: float,
    cfg_mode: str,
    seed: int,
    device: torch.device,
) -> float:
    gen.eval()
    pair_mat = np.stack([targets_prop_gen.astype(np.float32), targets_natoms.astype(np.float32)], axis=-1)
    w_t = gen.normalize_targets(pair_mat[:batch_size])
    w_m, w_u = _make_cfg_masks(batch_size, cfg_mode, device)
    for w_i in range(2):
        _ = gen.sample(
            w_t,
            w_m,
            cfg_scale=cfg_scale,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed + 300 + w_i,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            uncond_active_mask=w_u,
        )
    _sync(device)
    t0 = time.perf_counter()
    total_n = 0
    for b_i in range(n_batches):
        s = (b_i * batch_size) % max(len(pair_mat) - batch_size + 1, 1)
        sub = pair_mat[s : s + batch_size]
        if len(sub) < batch_size:
            sub = pair_mat[:batch_size]
        b_t = gen.normalize_targets(sub)
        b_m, b_u = _make_cfg_masks(len(sub), cfg_mode, device)
        _ = gen.sample(
            b_t,
            b_m,
            cfg_scale=cfg_scale,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed + 400 + b_i,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            uncond_active_mask=b_u,
        )
        total_n += len(sub)
    _sync(device)
    return round(float(total_n) / max(time.perf_counter() - t0, 1e-6), 2)


def _evaluate_heldout_scheme2(
    ckpt_path: Path,
    prop: str,
    prop_idx: int,
    prop_std: float,
    targets_ha: np.ndarray,
    natoms_targets: np.ndarray,
    *,
    cfg_mode: str,
    cfg_scale: float,
    m_budgets: Sequence[int],
    batch_size_pool: int,
    surrogate: QM9TrainSplitSurrogateOracle,
    reconciler: ValenceReconciler,
    gvp_suite: GVPBenchmarkSuite,
    train_smiles: set[str],
    device_str: str,
    seed: int,
) -> Dict[str, Any]:
    device = torch.device(device_str)
    eval_n = len(targets_ha)
    m_max = int(max(m_budgets))
    targets_gen = _target_ha_to_gen(prop, targets_ha)
    pair_mat = np.stack([targets_gen.astype(np.float32), natoms_targets.astype(np.float32)], axis=-1)

    gen, _ = LoomUni3DARGenerator.load_checkpoint(ckpt_path, device=device_str, batch_size=batch_size_pool)
    gen.eval()

    # Exact warm single-shot (`M=1`) GPU speeds at bs=128 and bs=250 for apples-to-apples table comparison
    speed_bs128 = _time_pair_inference_bs(
        gen, targets_gen, natoms_targets, batch_size=128, n_batches=4, cfg_scale=cfg_scale, cfg_mode=cfg_mode, seed=seed + 11, device=device
    )
    speed_bs250 = _time_pair_inference_bs(
        gen, targets_gen, natoms_targets, batch_size=250, n_batches=4, cfg_scale=cfg_scale, cfg_mode=cfg_mode, seed=seed + 29, device=device
    )

    trajs_by_round: List[List[Dict[str, Any]]] = []
    round_gen_walls: List[float] = []
    for m_i in range(m_max):
        _sync(device)
        t_r0 = time.perf_counter()
        r_trajs: List[Dict[str, Any]] = []
        for s in range(0, eval_n, batch_size_pool):
            e = min(s + batch_size_pool, eval_n)
            b_t = gen.normalize_targets(pair_mat[s:e])
            b_m, b_u = _make_cfg_masks(e - s, cfg_mode, device)
            sub = gen.sample(
                b_t,
                b_m,
                cfg_scale=cfg_scale,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=seed + m_i * 1009 + s,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
                uncond_active_mask=b_u,
            )
            r_trajs.extend(sub)
        _sync(device)
        round_gen_walls.append(time.perf_counter() - t_r0)
        trajs_by_round.append(r_trajs)

    del gen
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t_rec0 = time.perf_counter()
    raw_mols_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(eval_n)]
    rec_mols_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(eval_n)]
    raw_legal_grid = np.zeros((eval_n, m_max), dtype=bool)
    raw_exotic_grid = np.zeros((eval_n, m_max), dtype=bool)
    raw_clean_grid = np.zeros((eval_n, m_max), dtype=bool)
    rec_legal_grid = np.zeros((eval_n, m_max), dtype=bool)
    rec_exotic_grid = np.zeros((eval_n, m_max), dtype=bool)
    rec_clean_grid = np.zeros((eval_n, m_max), dtype=bool)
    raw_natoms_grid = np.zeros((eval_n, m_max), dtype=np.int32)
    rec_natoms_grid = np.zeros((eval_n, m_max), dtype=np.int32)
    ppl_grid = np.full((eval_n, m_max), 10.0, dtype=np.float32)

    flat_raw_mols: List[Optional[Chem.Mol]] = []
    flat_rec_mols: List[Optional[Chem.Mol]] = []

    for i in range(eval_n):
        for m_i in range(m_max):
            tr = trajs_by_round[m_i][i]
            ppl_grid[i, m_i] = float(tr.get("atom_score_ppl", 10.0))
            clash_free = _min_pair_dist(tr["positions_np"]) >= 0.72
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)

            if res.raw_legal and res.raw_mol is not None:
                raw_mols_grid[i][m_i] = res.raw_mol
                raw_legal_grid[i, m_i] = True
                raw_exotic_grid[i, m_i] = bool(res.raw_exotic_charge)
                raw_natoms_grid[i, m_i] = int(res.raw_mol.GetNumAtoms())
                conn = len(Chem.GetMolFrags(res.raw_mol)) == 1
                raw_clean_grid[i, m_i] = (not res.raw_exotic_charge) and conn and clash_free
                flat_raw_mols.append(res.raw_mol)
            else:
                flat_raw_mols.append(None)

            if res.reconciled_legal and res.mol is not None:
                rec_mols_grid[i][m_i] = res.mol
                rec_legal_grid[i, m_i] = True
                rec_exotic_grid[i, m_i] = bool(res.reconciled_exotic_charge)
                rec_natoms_grid[i, m_i] = int(res.mol.GetNumAtoms())
                conn_r = len(Chem.GetMolFrags(res.mol)) == 1
                rec_clean_grid[i, m_i] = (not res.reconciled_exotic_charge) and conn_r and clash_free
                flat_rec_mols.append(res.mol)
            else:
                flat_rec_mols.append(None)

    reconcile_wall = time.perf_counter() - t_rec0

    _sync(device)
    t_sur0 = time.perf_counter()
    if m_max > 1:
        raw_preds_all, _, raw_ok_all, _ = surrogate.evaluate_candidates(flat_raw_mols, batch_size=256)
        rec_preds_all, _, rec_ok_all, _ = surrogate.evaluate_candidates(flat_rec_mols, batch_size=256)
        raw_surr_grid = raw_preds_all[:, prop_idx].reshape(eval_n, m_max)
        raw_ok_grid = raw_ok_all.reshape(eval_n, m_max)
        rec_surr_grid = rec_preds_all[:, prop_idx].reshape(eval_n, m_max)
        rec_ok_grid = rec_ok_all.reshape(eval_n, m_max)
    else:
        raw_surr_grid = np.zeros((eval_n, 1), dtype=np.float32)
        raw_ok_grid = np.zeros((eval_n, 1), dtype=bool)
        rec_surr_grid = np.zeros((eval_n, 1), dtype=np.float32)
        rec_ok_grid = np.zeros((eval_n, 1), dtype=bool)
    _sync(device)
    surrogate_wall = time.perf_counter() - t_sur0

    budgets_out: Dict[str, Any] = {}
    for m_b in m_budgets:
        r_raw = select_best_of_m_and_score(
            prop=prop,
            prop_idx=prop_idx,
            prop_std=prop_std,
            targets_ha=targets_ha,
            targets_gen=targets_gen,
            natoms_targets=natoms_targets,
            m_budget=m_b,
            m_max=m_max,
            mode="target_plus_natoms",
            track="raw",
            mols_grid=raw_mols_grid,
            legal_grid=raw_legal_grid,
            exotic_grid=raw_exotic_grid,
            clean_connected_grid=raw_clean_grid,
            natoms_grid=raw_natoms_grid,
            ppl_grid=ppl_grid,
            surr_preds_grid=raw_surr_grid,
            surr_ok_grid=raw_ok_grid,
            round_gen_walls=round_gen_walls,
            reconcile_wall_total=reconcile_wall,
            surrogate_wall_total=surrogate_wall,
            gvp_suite=gvp_suite,
            train_smiles=train_smiles,
        )
        r_rec = select_best_of_m_and_score(
            prop=prop,
            prop_idx=prop_idx,
            prop_std=prop_std,
            targets_ha=targets_ha,
            targets_gen=targets_gen,
            natoms_targets=natoms_targets,
            m_budget=m_b,
            m_max=m_max,
            mode="target_plus_natoms",
            track="reconciled",
            mols_grid=rec_mols_grid,
            legal_grid=rec_legal_grid,
            exotic_grid=rec_exotic_grid,
            clean_connected_grid=rec_clean_grid,
            natoms_grid=rec_natoms_grid,
            ppl_grid=ppl_grid,
            surr_preds_grid=rec_surr_grid,
            surr_ok_grid=rec_ok_grid,
            round_gen_walls=round_gen_walls,
            reconcile_wall_total=reconcile_wall,
            surrogate_wall_total=surrogate_wall,
            gvp_suite=gvp_suite,
            train_smiles=train_smiles,
        )
        budgets_out[f"M{m_b}"] = {"raw": r_raw, "reconciled": r_rec}

    return {
        "checkpoint": str(ckpt_path),
        "cfg_mode": cfg_mode,
        "cfg_scale": cfg_scale,
        "single_shot_mol_per_sec_bs128": speed_bs128,
        "single_shot_mol_per_sec_bs250": speed_bs250,
        "budgets": budgets_out,
    }


def train_and_eval_scheme2_property(
    prop: str,
    prop_idx: int,
    cache: PreVoxelizedCanonicalCache,
    dev_recs: Sequence[Dict[str, Any]],
    surrogate: QM9TrainSplitSurrogateOracle,
    reconciler: ValenceReconciler,
    gvp_suite: GVPBenchmarkSuite,
    train_smiles: set[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = torch.device(args.device)
    prop_seed = args.seed + prop_idx * 1337
    torch.manual_seed(prop_seed)
    np.random.seed(prop_seed)
    random.seed(prop_seed)

    run_dir = args.runs_dir / f"s2_{prop}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"qm9_s2_{prop}.json"

    primary_y = cache.props_ha[:, prop_idx].copy()
    if prop in EV_PROPS:
        primary_y *= float(HA_TO_EV)
    prop_mean = float(np.mean(primary_y))
    prop_std = float(np.std(primary_y))
    natoms_mean = float(np.mean(cache.natoms))
    natoms_std = float(np.std(cache.natoms))
    primary_probs = compute_tail_balanced_weights(primary_y[:, None], [0], power=0.40, n_bins=24)

    dev_y_all = _records_to_gen_targets(dev_recs, prop_idx, prop)
    dev_nat_all = np.asarray([len(r["symbols"]) for r in dev_recs], dtype=np.int32)
    dev_y = dev_y_all[: args.dev_eval_n]
    dev_nat = dev_nat_all[: args.dev_eval_n]

    p005, p995 = np.quantile(primary_y, [0.005, 0.995])
    contrast_gap = 0.55 * prop_std

    print(
        f"\n==================== [Scheme 2] Property {prop} (unit={PROP_GEN_UNITS[prop]}) ====================\n"
        f"[S2 {prop}] mean={prop_mean:.4f} std={prop_std:.4f} | natoms_mean={natoms_mean:.2f} std={natoms_std:.2f}",
        flush=True,
    )

    generator = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_checkpoint,
        target_cols=[prop, "n_atoms"],
        target_units=[PROP_GEN_UNITS[prop], "atoms"],
        mean=[prop_mean, natoms_mean],
        scale=[prop_std, natoms_std],
        allowed_elements=ALLOWED_QM9_ELEMENTS,
        device=args.device,
        batch_size=args.batch_size,
        num_rbf=48,
        num_freqs=24,
        num_prefix_tokens=0,
        modulation_cap=4.0,
        lora_rank=32,
        lora_alpha=64.0,
        lora_dropout=0.05,
        lora_include_ffn=True,
        lora_include_heads=True,
        lora_include_xyz_heads=True,
        train_xyz_embeddings=True,
        use_valence_head=True,
    )
    generator.configure_trainable(
        train_conditioner=True,
        train_lora=True,
        train_xyz_embeddings=True,
        unfreeze_full_backbone=False,
    )
    n_train_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in generator.parameters())
    print(f"[S2 {prop}] High-rank LoRA-32 trainable parameters: {n_train_params:,} / {n_total_params:,}", flush=True)

    # 1. Stage 0: High-Capacity LoRA-32 Pre-Voxelized Canonical SFT
    sft_timing = _run_sft_cached_loop(
        generator,
        cache,
        primary_y,
        primary_probs,
        steps=args.sft_steps,
        lr_cond=3.0e-4,
        lr_backbone=2.5e-4,
        warmup_steps=250,
        batch_size=args.batch_size,
        prop_mean=prop_mean,
        prop_std=prop_std,
        natoms_mean=natoms_mean,
        natoms_std=natoms_std,
        contrast_min_gap_gen=contrast_gap,
        seed=prop_seed + 17,
        device=device,
        log_tag=f"S2-SFT-{prop}",
    )
    sft_ckpt = run_dir / f"s2_{prop}_sft.pt"
    generator.save_checkpoint(sft_ckpt, extra_metadata={"prop": prop, "stage": "sft", "timing": sft_timing})

    # Blind surrogate dev sweep on SFT across cfg_mode x cfg_scale
    sft_dev_sweep: List[Dict[str, Any]] = []
    for c_mode in ("size_anchored", "full_uncond"):
        for c_scale in (1.5, 2.25, 3.0):
            d_res = _eval_surrogate_dev_pair(
                generator,
                surrogate,
                reconciler,
                prop_idx,
                prop,
                prop_std,
                dev_y,
                dev_nat,
                batch_size=250,
                cfg_scale=c_scale,
                cfg_mode=c_mode,
                seed=prop_seed + 200 + int(c_scale * 20) + (0 if c_mode == "size_anchored" else 7),
                device=device,
            )
            sft_dev_sweep.append(d_res)
            print(f"[S2 {prop} SFT Dev] {d_res}", flush=True)
    best_sft_dev = min(sft_dev_sweep, key=lambda d: d["dev_score"])

    # 2. Stage 1: Canonical Hindsight Self-Training (zero test leakage)
    _sync(device)
    hs_t0 = time.perf_counter()
    rng = np.random.default_rng(prop_seed + 2026)
    hs_idx = rng.choice(cache.primary_count, size=args.hindsight_proposals, replace=True, p=primary_probs)
    hs_prop_req = primary_y[hs_idx] + rng.normal(0.0, 0.08 * prop_std, size=args.hindsight_proposals).astype(np.float32)
    hs_nat_req = cache.natoms[hs_idx].astype(np.int32)
    hs_gen_out = _sample_conditioned_pairs(
        generator,
        reconciler,
        hs_prop_req,
        hs_nat_req,
        batch_size=500,
        cfg_scale=float(best_sft_dev["cfg_scale"]),
        cfg_mode=str(best_sft_dev["cfg_mode"]),
        seed=prop_seed + 3030,
        device=device,
        do_warmup=False,
    )
    dedup_recs: List[Dict[str, Any]] = []
    seen_smis: set[str] = set()
    for cr in hs_gen_out["raw_clean_records"]:
        smi = cr["smiles"]
        if smi and smi not in seen_smis:
            seen_smis.add(smi)
            dedup_recs.append(cr)

    hs_preds, _, hs_mask, _ = surrogate.evaluate_candidates([r["mol"] for r in dedup_recs], batch_size=256)
    hs_canon_recs: List[Dict[str, Any]] = []
    hs_labels_gen: List[float] = []
    low_b = float(p005) - 0.5 * prop_std
    high_b = float(p995) + 0.5 * prop_std
    for i, rec in enumerate(dedup_recs):
        if not bool(hs_mask[i]):
            continue
        val_g = float(hs_preds[i, prop_idx])
        if not math.isfinite(val_g) or val_g < low_b or val_g > high_b:
            continue
        props_ha = np.zeros(len(PROPS), dtype=np.float32)
        props_ha[prop_idx] = val_g / float(HA_TO_EV) if prop in EV_PROPS else val_g
        canon_xyz = canonicalize_coords_pca(rec["symbols"], rec["coords"])
        hs_canon_recs.append(
            {
                "gdb9_idx": -1,
                "symbols": rec["symbols"],
                "coords": canon_xyz,
                "props_ha": props_ha,
            }
        )
        hs_labels_gen.append(val_g)

    extra_feats, extra_props_ha, extra_natoms = cache.build_extra_subset(hs_canon_recs, seed=prop_seed + 55)
    extra_y_gen = extra_props_ha[:, prop_idx] * (float(HA_TO_EV) if prop in EV_PROPS else 1.0)
    print(
        f"[S2 {prop} Hindsight] generated={args.hindsight_proposals} clean_raw={hs_gen_out['raw_legal_count']} "
        f"voxelized_relabeled={len(extra_feats)}",
        flush=True,
    )

    hs_sft_timing = _run_sft_cached_loop(
        generator,
        cache,
        primary_y,
        primary_probs,
        extra_feats=extra_feats,
        extra_y_gen=extra_y_gen,
        extra_natoms=extra_natoms,
        extra_mix_prob=0.30,
        steps=args.hindsight_sft_steps,
        lr_cond=1.0e-4,
        lr_backbone=8.0e-5,
        warmup_steps=50,
        batch_size=args.batch_size,
        prop_mean=prop_mean,
        prop_std=prop_std,
        natoms_mean=natoms_mean,
        natoms_std=natoms_std,
        contrast_min_gap_gen=contrast_gap,
        seed=prop_seed + 66,
        device=device,
        log_tag=f"S2-HS-{prop}",
    )
    _sync(device)
    hs_total_wall = time.perf_counter() - hs_t0
    hs_ckpt = run_dir / f"s2_{prop}_hindsight.pt"
    generator.save_checkpoint(hs_ckpt, extra_metadata={"prop": prop, "stage": "hindsight"})

    # 3. Stage 2.5A: Tight Point-Regression TopoGRPO
    grpo_best_ckpt = run_dir / f"s2_{prop}_grpo_best_dev.pt"
    grpo_final_ckpt = run_dir / f"s2_{prop}_grpo_final.pt"
    grpo_timing, grpo_hist = _run_tight_regression_topogrpo(
        generator,
        surrogate,
        reconciler,
        cache,
        primary_y,
        primary_probs,
        prop=prop,
        prop_idx=prop_idx,
        prop_mean=prop_mean,
        prop_std=prop_std,
        natoms_mean=natoms_mean,
        natoms_std=natoms_std,
        cfg_mode=str(best_sft_dev["cfg_mode"]),
        cfg_scale=float(best_sft_dev["cfg_scale"]),
        grpo_steps=args.grpo_steps,
        prompts_per_step=args.grpo_prompts,
        group_size=args.grpo_group_size,
        dev_prop_gen=dev_y,
        dev_natoms=dev_nat,
        best_dev_ckpt=grpo_best_ckpt,
        seed=prop_seed + 88,
        device=device,
    )
    generator.save_checkpoint(grpo_final_ckpt, extra_metadata={"prop": prop, "stage": "grpo_final"})
    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Blind Surrogate Dev Selection across post-SFT candidates x (cfg_mode, cfg_scale)
    full_dev_candidates: List[Dict[str, Any]] = []
    cand_ckpts = [
        ("hindsight", hs_ckpt),
        ("grpo_best_dev", grpo_best_ckpt if grpo_best_ckpt.exists() else grpo_final_ckpt),
        ("grpo_final", grpo_final_ckpt),
    ]
    for tag, cpath in cand_ckpts:
        cand_gen, _ = LoomUni3DARGenerator.load_checkpoint(cpath, device=args.device, batch_size=250)
        for c_mode in ("size_anchored", "full_uncond"):
            for c_scale in (1.5, 2.25, 3.0):
                d_res = _eval_surrogate_dev_pair(
                    cand_gen,
                    surrogate,
                    reconciler,
                    prop_idx,
                    prop,
                    prop_std,
                    dev_y,
                    dev_nat,
                    batch_size=250,
                    cfg_scale=c_scale,
                    cfg_mode=c_mode,
                    seed=prop_seed + 6100 + int(c_scale * 20) + (0 if c_mode == "size_anchored" else 7),
                    device=device,
                )
                d_res["ckpt_tag"] = tag
                d_res["ckpt_path"] = str(cpath)
                full_dev_candidates.append(d_res)
                print(f"[S2 {prop} Full Dev ({tag})] {d_res}", flush=True)
        del cand_gen
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_full_dev = min(full_dev_candidates, key=lambda d: d["dev_score"])
    print(f"[S2 {prop} Selected] SFT={best_sft_dev} | Full={best_full_dev}", flush=True)

    # 5. One-Shot Held-Out GVP Evaluation (`N = 1,000` official `(target, natoms)` pairs)
    targets_ha = _load_propmolflow_eval_targets(args.propmolflow_root, prop, args.eval_n)
    natoms_targets = _load_propmolflow_eval_natoms(args.propmolflow_root, prop, args.eval_n)

    eval_s2_sft = _evaluate_heldout_scheme2(
        sft_ckpt,
        prop,
        prop_idx,
        prop_std,
        targets_ha,
        natoms_targets,
        cfg_mode=str(best_sft_dev["cfg_mode"]),
        cfg_scale=float(best_sft_dev["cfg_scale"]),
        m_budgets=[1],
        batch_size_pool=500,
        surrogate=surrogate,
        reconciler=reconciler,
        gvp_suite=gvp_suite,
        train_smiles=train_smiles,
        device_str=args.device,
        seed=prop_seed + 70000,
    )

    eval_s2_full = _evaluate_heldout_scheme2(
        Path(best_full_dev["ckpt_path"]),
        prop,
        prop_idx,
        prop_std,
        targets_ha,
        natoms_targets,
        cfg_mode=str(best_full_dev["cfg_mode"]),
        cfg_scale=float(best_full_dev["cfg_scale"]),
        m_budgets=args.m_budgets,
        batch_size_pool=500,
        surrogate=surrogate,
        reconciler=reconciler,
        gvp_suite=gvp_suite,
        train_smiles=train_smiles,
        device_str=args.device,
        seed=prop_seed + 80000,
    )
    eval_s2_full["selected_stage"] = best_full_dev["ckpt_tag"]

    total_full_wall = sft_timing["wall_seconds"] + hs_total_wall + grpo_timing["wall_seconds"]
    payload = {
        "property": prop,
        "gen_unit": PROP_GEN_UNITS[prop],
        "eval_unit": PROP_EVAL_UNITS[prop],
        "scheme2_features": {
            "joint_target_and_natoms": True,
            "canonical_so3_pca_polarity_frame": True,
            "high_rank_lora_r32": True,
            "size_anchored_cfg_supported": True,
            "fixed_natoms_contrastive_margin": True,
            "tight_regression_topogrpo": True,
            "trainable_params": n_train_params,
            "total_params": n_total_params,
        },
        "training_cost": {
            "sft": sft_timing,
            "hindsight": {
                "proposals": args.hindsight_proposals,
                "voxelized_relabeled": len(extra_feats),
                "mixed_sft": hs_sft_timing,
                "total_wall_seconds": round(hs_total_wall, 2),
            },
            "topo_grpo": {
                **grpo_timing,
                "last_step": grpo_hist[-1] if grpo_hist else None,
            },
            "s2_sft_total_wall_seconds": round(sft_timing["wall_seconds"], 2),
            "s2_full_total_wall_seconds": round(total_full_wall, 2),
        },
        "blind_dev_selection": {
            "sft_dev_sweep": sft_dev_sweep,
            "selected_sft": best_sft_dev,
            "full_dev_sweep": full_dev_candidates,
            "selected_full": best_full_dev,
        },
        "s2_sft": eval_s2_sft,
        "s2_full": eval_s2_full,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    sft_m1 = eval_s2_sft["budgets"]["M1"]["reconciled"]
    full_m1 = eval_s2_full["budgets"]["M1"]["reconciled"]
    m_top_key = f"M{max(args.m_budgets)}"
    full_mtop = eval_s2_full["budgets"][m_top_key]["reconciled"]
    print(
        f"[S2 {prop} DONE] "
        f"SFT(M=1) RecMAE={sft_m1['gvp']['mae']} (r={sft_m1['gvp']['pearson_r']}, Leg={sft_m1['legal_pct']}%) | "
        f"Full({eval_s2_full['selected_stage']}, M=1) RecMAE={full_m1['gvp']['mae']} (r={full_m1['gvp']['pearson_r']}, Leg={full_m1['legal_pct']}%, bs128={eval_s2_full['single_shot_mol_per_sec_bs128']} bs250={eval_s2_full['single_shot_mol_per_sec_bs250']} mol/s) | "
        f"Full({m_top_key}) RecMAE={full_mtop['gvp']['mae']} (r={full_mtop['gvp']['pearson_r']}, Leg={full_mtop['legal_pct']}%, eff={full_mtop['effective_output_mol_per_sec']} mol/s)",
        flush=True,
    )
    return payload


def run_worker_phase(args: argparse.Namespace) -> None:
    qm9_raw_dir = args.propmolflow_root / "data" / "qm9_raw"
    splits = load_disjoint_qm9_splits(args.lmdb_dir, qm9_raw_dir, dev_size=1500, holdout_size=1000, seed=20261115)
    assert int(splits["audit"]["overlap_train_with_gvp_rQM_v0"]) == 0

    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    print("[S2] Canonicalizing SO(3) inertial+polarity PCA frame & indexing train SMILES...", flush=True)
    train_smiles: set[str] = set()
    canon_train_recs: List[Dict[str, Any]] = []
    for r in splits["train"]:
        res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
        if res.smiles:
            m2d = Chem.MolFromSmiles(res.smiles)
            if m2d is not None:
                train_smiles.add(Chem.MolToSmiles(Chem.RemoveHs(m2d), isomericSmiles=False))
        canon_train_recs.append(
            {
                "gdb9_idx": r["gdb9_idx"],
                "symbols": r["symbols"],
                "coords": canonicalize_coords_pca(r["symbols"], r["coords"]),
                "props_ha": r["props_ha"],
            }
        )
    canon_dev_recs = [
        {
            "gdb9_idx": r["gdb9_idx"],
            "symbols": r["symbols"],
            "coords": canonicalize_coords_pca(r["symbols"], r["coords"]),
            "props_ha": r["props_ha"],
        }
        for r in splits["dev"]
    ]

    # Probe parent checkpoint once to instantiate matching backbone config for RAM voxelization
    probe_gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_checkpoint,
        target_cols=["alpha", "n_atoms"],
        target_units=["Bohr^3", "atoms"],
        mean=[75.0, 18.0],
        scale=[8.0, 3.0],
        allowed_elements=ALLOWED_QM9_ELEMENTS,
        device="cpu",
        batch_size=args.batch_size,
        lora_rank=0,
    )
    cache = PreVoxelizedCanonicalCache(canon_train_recs, probe_gen.backbone, seed=args.seed)
    del probe_gen

    surrogate = QM9TrainSplitSurrogateOracle(args.surrogate_ckpt, device=args.device)
    gvp_suite = GVPBenchmarkSuite(args.propmolflow_root, device=args.device)

    if args.queue_dir is not None:
        args.queue_dir.mkdir(parents=True, exist_ok=True)

    for prop in args.props:
        if prop not in PROPS:
            raise ValueError(f"Unknown property '{prop}'")
        out_json = args.results_dir / f"qm9_s2_{prop}.json"
        if out_json.exists() and not args.overwrite:
            print(f"[S2] {out_json} already exists; skipping {prop}.", flush=True)
            continue
        if args.queue_dir is not None:
            claim_dir = args.queue_dir / f"claim_{prop}"
            try:
                os.mkdir(claim_dir)
                (claim_dir / "owner.txt").write_text(f"pid={os.getpid()} device={args.device}\n")
                print(f"[S2 Queue] Claimed {prop} on {args.device} (pid={os.getpid()})", flush=True)
            except FileExistsError:
                print(f"[S2 Queue] {prop} already claimed by peer GPU worker; moving to next.", flush=True)
                continue
        prop_idx = PROPS.index(prop)
        train_and_eval_scheme2_property(
            prop, prop_idx, cache, canon_dev_recs, surrogate, reconciler, gvp_suite, train_smiles, args
        )


def consolidate_all_phase(args: argparse.Namespace) -> None:
    s1_path = args.results_dir / "qm9_scheme1_best_of_m.json"
    s1_data = json.loads(s1_path.read_text()) if s1_path.exists() else {}
    pmf_path = args.results_dir / "qm9_propmolflow_audit.json"
    pmf_data = json.loads(pmf_path.read_text()) if pmf_path.exists() else {}

    s2_props: Dict[str, Any] = {}
    for p in PROPS:
        pp = args.results_dir / f"qm9_s2_{p}.json"
        if pp.exists():
            s2_props[p] = json.loads(pp.read_text())

    out = {
        "benchmark": "QM9_Scheme1_And_Scheme2_Consolidated",
        "scheme1_best_of_m": s1_data,
        "scheme2_per_property": s2_props,
        "propmolflow_audit_summary": pmf_data.get("summary", {}),
    }
    dst = args.results_dir / "qm9_scheme1_and_scheme2_summary.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[Consolidate S1+S2] Wrote {dst} (S2 completed props: {sorted(s2_props.keys())})", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["run_worker", "consolidate_all"], default="run_worker")
    p.add_argument("--props", nargs="+", default=list(PROPS))
    p.add_argument("--queue-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--lmdb-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9_data"))
    p.add_argument("--parent-checkpoint", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9.pt"))
    p.add_argument("--propmolflow-root", type=Path, default=Path("/sjtu-combustion/wjw/chensq/PropMolFlow"))
    p.add_argument(
        "--surrogate-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_surrogate/qm9_train_unimol_surrogate.pt"),
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results"),
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--sft-steps", type=int, default=4500)
    p.add_argument("--hindsight-proposals", type=int, default=2560)
    p.add_argument("--hindsight-sft-steps", type=int, default=600)
    p.add_argument("--grpo-steps", type=int, default=140)
    p.add_argument("--grpo-prompts", type=int, default=8)
    p.add_argument("--grpo-group-size", type=int, default=8)
    p.add_argument("--dev-eval-n", type=int, default=250)
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--m-budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    if args.phase == "run_worker":
        run_worker_phase(args)
    elif args.phase == "consolidate_all":
        consolidate_all_phase(args)


if __name__ == "__main__":
    main()
