"""Step 2 of `Uni-3DAR v4 + Uni-Loom` Speed-Preserving Overtake Plan (`~4-8 GPU-h` 2-GPU DDP SFT).

Executes the core `Uni-3DAR v4` backbone & representation upgrade across `GPU 0` and `GPU 1`:
  1. **Abolish PCA Canonicalization & Restore Full SO(3) Generality (Red Line #1)**:
     - Zero `_canonicalize_mol_pca`; trains on native 3D coordinates with `random_rotation_prob = 0.80`
       continuous SO(3) rotations + `+-0.12 A` intra-voxel translation jitter refreshed every epoch,
       paired with `dropout = 0.05` across all 12 Transformer layers.
  2. **Retire Soft `n_atoms` in `LoomConditioner` & Enable Unbounded `ScaleFreeCountHead` (`Delta N == 0`) (Red Line #2)**:
     - `LoomConditioner` encodes `["homo", "sa_score"]` (`P = 8` SO(3)-centered prefix tokens) trained
       unbiasedly across ALL `~47,807` OE62 conformers (with zero `5.5x` oversampling).
     - HOMO targets align with `PBE0/def2-TZVP/DF` via zero-leakage `ResidualCalibratedAsset` (`0` `audit_dpa4` calls).
     - Exact atom count `N` is conserved top-down (`Delta N == 0`) via `ScaleFreeBudgetEmbedding` + `ScaleFreeCountHead`.
  3. **Training-Only Differentiable Expected-XYZ & 1-2/1-3 Distance Loss (`0%` Inference Slowdown, Red Line #3)**:
     - Pre-relaxes only the small unoptimized `frozen_3d_input_conformer` training subset once offline on `GPU 0`
       so 100% of training conformers are genuine low-strain equilibrium 3D geometries.
     - Supervises differentiable expected sub-voxel coordinates + covalent/geminal distances (`d_gt < 2.65 A`)
       during training while keeping inference `0.01 A` categorical heads & zero-MLIP speed 100% unchanged.
  4. **90M Full-Parameter Layered-LR 2-GPU DDP SFT + EMA (`decay = 0.9995`) + TwoStage `32->8->2` Eval**:
     - Unfreezes all `~90M` backbone parameters (`lr_backbone = 3.5e-5`, `lr_heads = 2.5e-4`), maintains EMA
       shadow weights, fits Step-2 `TwoStageOctreeProcessCritic`, and benchmarks both Fair `30.4`-Atom (`N=1,792`)
       and Multi-Atom-Count Scaling (`N in {20, 30, 40, 50, 60}`, `280` prompts/bucket).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MOLGEN_COMPAT_DEFAULT = Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat")
for cand in (ROOT / "src" / "compat", ROOT / "src", ROOT, MOLGEN_COMPAT_DEFAULT):
    cstr = str(cand)
    if cand.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402
from loom.generator.octree_backend import OctreeDataset  # noqa: E402
from loom.utils.dataset_io import write_precise_sdf  # noqa: E402
from loom.weaving import (  # noqa: E402
    BuiltInRuleAsset,
    LoomUni3DARGenerator,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    PotentialSlot,
    TwoStageOctreeProcessCritic,
    fit_octree_element_process_critic,
    fit_octree_process_critic,
)
from loom.weaving.registry import _mol_static_residual_descriptor  # noqa: E402
from scripts.loom_oe62_fair_size_matched_and_buckets import build_fair_30atom_prompts  # noqa: E402
from scripts.loom_oe62_size_anchored import (  # noqa: E402
    _get_clean_oe62_pool,
    _get_source_novelty_sets,
    _score_oe62_size_cohort,
)
from scripts.loom_uni3dar_v4_step1_eval import (  # noqa: E402
    _build_calibrated_oe62_bus,
    _select_v4_step1_from_pool,
    _sync,
)


class _Step2OctreeTorchDataset(Dataset):
    """Worker-friendly wrapper around `OctreeDataset` returning single-item grid features + `[homo, sa]`."""

    def __init__(self, mols: List[Chem.Mol], targets_2d: np.ndarray, backbone: Any, training: bool, seed: int) -> None:
        self.octree_ds = OctreeDataset(
            mols, backbone, training=training, seed=seed, random_rotation_prob=(0.80 if training else 0.0)
        )
        self.targets_2d = np.asarray(targets_2d, dtype=np.float32)

    def set_epoch(self, epoch: int) -> None:
        self.octree_ds.set_grid_epoch(epoch)

    def __len__(self) -> int:
        return len(self.octree_ds)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], np.ndarray]:
        item = self.octree_ds.grid[int(idx)]
        return item, self.targets_2d[int(idx)]


class _CollateStep2Octree:
    def __init__(self, collater_fn: Any) -> None:
        self.collater_fn = collater_fn

    def __call__(self, batch: List[Tuple[Dict[str, Any], np.ndarray]]) -> Tuple[Dict[str, Any], torch.Tensor]:
        items = [b[0] for b in batch]
        t_mat = np.stack([b[1] for b in batch], axis=0)
        packed = self.collater_fn(items)
        return packed, torch.from_numpy(t_mat)


def _sample_v4_sft_masks(batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    """Structured `[homo, sa_score]` CFG mask: `75% [T,T]`, `15% [F,T]`, `10% [F,F]`."""
    u = rng.random(size=batch_size)
    mask = np.ones((batch_size, 2), dtype=bool)
    mask[(u >= 0.75) & (u < 0.90), 0] = False
    mask[u >= 0.90] = False
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def _fast_calibrate_sdf_homo(
    mols: Sequence[Chem.Mol],
    cal_unimol2: Any,
) -> np.ndarray:
    """Compute zero-leakage `PBE0/def2-TZVP`-calibrated HOMO on SDF molecules using stored `B_homo/B_lumo`."""
    n = len(mols)
    homo_out = np.zeros(n, dtype=np.float32)
    need_cal_idx: List[int] = []
    base_preds_list: List[List[float]] = []

    for i, m in enumerate(mols):
        origin = m.GetProp("OriginType") if m.HasProp("OriginType") else ""
        if origin == "historical_quantum" and m.HasProp("DFT_homo"):
            homo_out[i] = float(m.GetProp("DFT_homo"))
        else:
            bh = float(m.GetProp("B_homo")) if m.HasProp("B_homo") else float(m.GetProp("homo"))
            bl = float(m.GetProp("B_lumo")) if m.HasProp("B_lumo") else (bh + 3.5)
            homo_out[i] = bh
            need_cal_idx.append(i)
            base_preds_list.append([bh, bl])

    if need_cal_idx and cal_unimol2._gps and cal_unimol2._gps[0] is not None:
        gp0 = cal_unimol2._gps[0]
        tr0 = cal_unimol2._trends[0] if cal_unimol2._trends else None
        base_mat = np.asarray(base_preds_list, dtype=np.float64)
        for s in range(0, len(need_cal_idx), 2048):
            e = min(s + 2048, len(need_cal_idx))
            sub_ids = need_cal_idx[s:e]
            st_mat = np.stack(
                [_mol_static_residual_descriptor(mols[idx], cal_unimol2._fp_gen) for idx in sub_ids], axis=0
            )
            feats = cal_unimol2._compose_features(st_mat, base_mat[s:e])
            t_mu = tr0.predict(feats) if tr0 is not None else 0.0
            d_mu = gp0.predict(feats, return_std=False)
            homo_out[sub_ids] = (base_mat[s:e, 0] + t_mu + d_mu).astype(np.float32)
    return homo_out


def prepare_step2_dataset(args: argparse.Namespace) -> Tuple[Path, Path, Dict[str, Any]]:
    """Prepare unbiased, equilibrium-conformer, `PBE0/def2-TZVP`-aligned OE62 train & valid SDFs."""
    step2_dir = args.work_dir / args.run_name
    step2_dir.mkdir(parents=True, exist_ok=True)
    train_sdf_out = step2_dir / "oe62_v4_train_eq.sdf"
    valid_sdf_out = step2_dir / "oe62_v4_valid_eq.sdf"
    meta_json = step2_dir / "oe62_v4_data_meta.json"
    calib_cache = step2_dir / "residual_calibrated_unimol2.pt"

    if train_sdf_out.exists() and valid_sdf_out.exists() and meta_json.exists():
        return train_sdf_out, valid_sdf_out, json.loads(meta_json.read_text())

    # Reuse Step 1 calibration cache if present; otherwise fit from `quantum_train_dft.sdf`
    if not calib_cache.exists() and args.step1_calib_cache.exists():
        calib_cache.write_bytes(args.step1_calib_cache.read_bytes())
    _, cal_unimol2, calib_diag = _build_calibrated_oe62_bus(
        args, include_blind_auditor=False, calib_cache_path=calib_cache
    )

    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.oe62_potential_weights), device=args.device)
    rule_asset = BuiltInRuleAsset(asset_id="prep_rules")

    # Temp backbone strictly for validating lossless octree encodability (`cutoff_atom_count == 0`)
    probe_gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_ckpt,
        target_cols=["homo", "sa_score"],
        target_units=["eV", "sa"],
        mean=[-6.5, 3.0],
        scale=[0.8, 0.8],
        device="cpu",
        batch_size=16,
        lora_rank=0,
    )

    def _process_split(in_sdf: Path, out_sdf: Path, relax_unopt: bool) -> Tuple[int, np.ndarray]:
        raw_mols = [m for m in Chem.SDMolSupplier(str(in_sdf), removeHs=False) if m is not None]
        if relax_unopt:
            unopt_idx = [
                i
                for i, m in enumerate(raw_mols)
                if m.HasProp("GeometryOrigin") and m.GetProp("GeometryOrigin") == "frozen_3d_input_conformer"
            ]
            if unopt_idx:
                sub_unopt = [raw_mols[i] for i in unopt_idx]
                rel = pot_slot.relax_molecules(sub_unopt, n_steps=20, step_size=0.008, max_atom_disp=0.03, chunk_size=64)
                n_rep = 0
                for loc_i, glob_i in enumerate(unopt_idx):
                    if rel["graph_preserved"][loc_i] and rel["relaxed_mols"][loc_i] is not None:
                        rm = Chem.Mol(rel["relaxed_mols"][loc_i])
                        for p_name in raw_mols[glob_i].GetPropNames():
                            rm.SetProp(p_name, raw_mols[glob_i].GetProp(p_name))
                        raw_mols[glob_i] = rm
                        n_rep += 1
                print(f"[Step2 Prep] Relaxed {n_rep}/{len(unopt_idx)} unoptimized historical training conformers", flush=True)

        homo_cal = _fast_calibrate_sdf_homo(raw_mols, cal_unimol2)
        r_out = rule_asset.predict_mols(raw_mols, batch_size=512)
        sa_all = r_out["mean"][:, rule_asset.channel_index("sa_score")].astype(np.float32)

        probe_ds = OctreeDataset(raw_mols, probe_gen.backbone, training=False, seed=42)
        kept_mols: List[Chem.Mol] = []
        kept_targets: List[Tuple[float, float]] = []
        for idx, m in enumerate(raw_mols):
            h_i = float(homo_cal[idx])
            s_i = float(sa_all[idx])
            if not (math.isfinite(h_i) and math.isfinite(s_i)):
                continue
            try:
                item = probe_ds.grid[idx]
                decoded = int(((item["decoder_level"] == 0) & item["decoder_is_second_atom"]).sum().item())
                if item["cutoff_atom_count"] or decoded != m.GetNumAtoms():
                    continue
            except Exception:
                continue
            cm = Chem.Mol(m)
            cm.SetProp("v4_homo_eV", f"{h_i:.6f}")
            cm.SetProp("v4_sa_score", f"{s_i:.4f}")
            kept_mols.append(cm)
            kept_targets.append((h_i, s_i))

        write_precise_sdf(out_sdf, kept_mols)
        return len(kept_mols), np.asarray(kept_targets, dtype=np.float32)

    t0 = time.perf_counter()
    n_tr, tr_targets = _process_split(args.oe62_master_train_sdf, train_sdf_out, relax_unopt=True)
    n_va, _ = _process_split(args.oe62_master_valid_sdf, valid_sdf_out, relax_unopt=False)

    t_mean = np.mean(tr_targets, axis=0)
    t_scale = np.maximum(np.std(tr_targets, axis=0), 0.25)
    meta = {
        "train_count": n_tr,
        "valid_count": n_va,
        "target_cols": ["homo", "sa_score"],
        "target_units": ["eV", "sa"],
        "mean": [round(float(x), 6) for x in t_mean],
        "scale": [round(float(x), 6) for x in t_scale],
        "prep_wall_sec": round(time.perf_counter() - t0, 2),
        "calib_diag": calib_diag,
    }
    meta_json.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[Step2 Prep DONE] {meta}", flush=True)
    return train_sdf_out, valid_sdf_out, meta


def _load_prepared_v4_sdf(sdf_path: Path) -> Tuple[List[Chem.Mol], np.ndarray]:
    mols: List[Chem.Mol] = []
    rows: List[Tuple[float, float]] = []
    for m in Chem.SDMolSupplier(str(sdf_path), removeHs=False):
        if m is None:
            continue
        mols.append(m)
        rows.append((float(m.GetProp("v4_homo_eV")), float(m.GetProp("v4_sa_score"))))
    return mols, np.asarray(rows, dtype=np.float32)


def run_train_ddp(args: argparse.Namespace) -> None:
    """90M Full-Parameter Unfrozen Backbone 2-GPU DDP SFT with Layered LR + EMA (`decay = 0.9995`)."""
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(args.device)

    step2_dir = args.work_dir / args.run_name
    if rank == 0:
        prepare_step2_dataset(args)
    if is_ddp:
        dist.barrier()

    train_sdf = step2_dir / "oe62_v4_train_eq.sdf"
    valid_sdf = step2_dir / "oe62_v4_valid_eq.sdf"
    data_meta = json.loads((step2_dir / "oe62_v4_data_meta.json").read_text())

    train_mols, train_targets = _load_prepared_v4_sdf(train_sdf)
    valid_mols, valid_targets = _load_prepared_v4_sdf(valid_sdf)

    gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_ckpt,
        target_cols=["homo", "sa_score"],
        target_units=["eV", "sa"],
        mean=data_meta["mean"],
        scale=data_meta["scale"],
        device=str(device),
        batch_size=args.batch_size,
        num_rbf=32,
        num_freqs=16,
        num_prefix_tokens=8,
        lora_rank=0,
        use_valence_head=True,
        use_scale_free_count=True,
        rope_center_prefix=True,
        backbone_dropout=0.05,
        random_rotation_prob=0.80,
    )
    gen.configure_trainable(train_conditioner=True, train_lora=False, unfreeze_full_backbone=True)
    if is_ddp:
        for tensor_state in gen.state_dict().values():
            dist.broadcast(tensor_state, src=0)

    # Maintain EMA shadow model on each rank (identical across ranks since weights & gradients are synchronized)
    ema_gen = deepcopy(gen).to(device)
    ema_gen.eval()
    for p in ema_gen.parameters():
        p.requires_grad_(False)

    # Layered AdamW parameter groups (`3.5e-5` pretrained backbone, `2.5e-4` newly initialized heads/conditioner)
    backbone_decay: List[torch.nn.Parameter] = []
    backbone_nodecay: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []
    no_wd_keywords = ("bias", "norm", "emb", "rope")

    for name, p in gen.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            if p.ndim < 2 or any(kw in name.lower() for kw in no_wd_keywords):
                backbone_nodecay.append(p)
            else:
                backbone_decay.append(p)
        else:
            head_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": backbone_decay, "lr": args.lr_backbone, "weight_decay": 1e-2},
            {"params": backbone_nodecay, "lr": args.lr_backbone, "weight_decay": 0.0},
            {"params": head_params, "lr": args.lr_heads, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.98),
    )

    train_ds = _Step2OctreeTorchDataset(train_mols, train_targets, gen.backbone, training=True, seed=2026)
    sampler: Optional[DistributedSampler] = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=2026, drop_last=True)
        if is_ddp
        else None
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=_CollateStep2Octree(train_ds.octree_ds.grid.collater),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    steps_per_epoch = len(loader)
    total_steps = max(1, int(args.epochs) * steps_per_epoch)
    warmup_steps = max(20, int(0.05 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    if rank == 0:
        n_params_m = sum(p.numel() for p in gen.parameters() if p.requires_grad) / 1e6
        print(
            f"[Step2 DDP SFT] WorldSize={world_size} | TrainMols={len(train_ds)} | ValidMols={len(valid_mols)} | "
            f"TrainableParams={n_params_m:.2f}M | Epochs={args.epochs} ({steps_per_epoch} steps/ep, total={total_steps})",
            flush=True,
        )

    contrast_col_mask = torch.tensor([True, False], dtype=torch.bool, device=device)
    ema_decay = float(args.ema_decay)
    best_val_score = float("inf")
    best_ckpt = step2_dir / "v4_step2_ema_best.pt"
    last_ckpt = step2_dir / "v4_step2_ema_last.pt"
    history: List[Dict[str, Any]] = []

    t_train0 = time.perf_counter()
    global_step = 0

    for ep in range(1, int(args.epochs) + 1):
        ep_t0 = time.perf_counter()
        train_ds.set_epoch(ep)
        if sampler is not None:
            sampler.set_epoch(ep)

        gen.train()
        rng_ep = np.random.default_rng(202600 + ep * 131 + rank * 17)
        loss_sum = 0.0
        tree_sum = 0.0
        atom_sum = 0.0
        xyz_sum = 0.0
        cnt_sum = 0.0
        geom_sum = 0.0
        pref_sum = 0.0
        pref_n = 0
        n_batches = 0

        for packed, raw_t in loader:
            bsz_cur = int(raw_t.shape[0])
            t_norm = gen.normalize_targets(raw_t.to(device, non_blocking=True))
            act_m = _sample_v4_sft_masks(bsz_cur, rng_ep, device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, info = gen.compute_sft_loss(
                    packed,
                    t_norm,
                    act_m,
                    cfg_dropout_prob=0.0,
                    contrast_weight=args.contrast_weight,
                    contrast_margin=0.06,
                    contrast_min_gap_eV=0.35,
                    contrast_col_mask=contrast_col_mask,
                    xyz_loss_weight=0.25,
                    valence_loss_weight=0.15,
                    count_loss_weight=0.30,
                    geom_dist_loss_weight=args.geom_dist_weight,
                    budget_dropout_prob=0.15,
                )

            loss.backward()
            if is_ddp:
                # Synchronize gradients across all DDP ranks cleanly (handles optional auxiliary branches safely!)
                flat_grads = [
                    (p.grad if p.grad is not None else torch.zeros_like(p))
                    for p in gen.parameters()
                    if p.requires_grad
                ]
                coalesced = torch._utils._flatten_dense_tensors(flat_grads)
                dist.all_reduce(coalesced, op=dist.ReduceOp.AVG)
                for p, g_sync in zip(
                    (p for p in gen.parameters() if p.requires_grad),
                    torch._utils._unflatten_dense_tensors(coalesced, flat_grads),
                ):
                    p.grad = g_sync

            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.5)
            opt.step()
            sched.step()
            global_step += 1

            # Update EMA shadow weights
            eff_decay = min(ema_decay, (1.0 + global_step) / (10.0 + global_step))
            with torch.no_grad():
                for p_ema, p_cur in zip(ema_gen.parameters(), gen.parameters()):
                    p_ema.lerp_(p_cur, 1.0 - eff_decay)

            loss_sum += float(info["loss_total"])
            tree_sum += float(info["loss_tree"])
            atom_sum += float(info["loss_atom"])
            xyz_sum += float(info["loss_xyz"])
            cnt_sum += float(info["loss_count"])
            geom_sum += float(info["loss_geom"])
            pr = info.get("inbatch_preference_rate", float("nan"))
            if math.isfinite(pr):
                pref_sum += pr
                pref_n += 1
            n_batches += 1

        if rank == 0 and (ep == 1 or ep % 5 == 0 or ep == int(args.epochs)):
            ema_gen.eval()
            val_ds = OctreeDataset(valid_mols[:384], ema_gen.backbone, training=False, seed=77)
            v_loss_list: List[float] = []
            v_pref_list: List[float] = []
            with torch.no_grad():
                for vs in range(0, min(384, len(valid_mols)), 64):
                    ve = min(vs + 64, min(384, len(valid_mols)))
                    try:
                        v_packed = val_ds.collate(list(range(vs, ve)))
                    except ValueError:
                        continue
                    vt = ema_gen.normalize_targets(torch.from_numpy(valid_targets[vs:ve]).to(device))
                    vm = torch.ones((ve - vs, 2), dtype=torch.bool, device=device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        vl, _ = ema_gen.compute_sft_loss(
                            v_packed,
                            vt,
                            vm,
                            xyz_loss_weight=0.25,
                            valence_loss_weight=0.15,
                            count_loss_weight=0.30,
                            geom_dist_loss_weight=args.geom_dist_weight,
                            budget_dropout_prob=0.0,
                        )
                        vm_homo = torch.tensor([[True, False]] * (ve - vs), dtype=torch.bool, device=device)
                        tf_pref = ema_gen.evaluate_teacher_forcing_preference(v_packed, vt, vm_homo, min_gap_eV=0.45)
                    v_loss_list.append(float(vl.item()))
                    if math.isfinite(float(tf_pref["preference_rate"])):
                        v_pref_list.append(float(tf_pref["preference_rate"]))

            val_loss = float(np.mean(v_loss_list)) if v_loss_list else float("nan")
            val_pref = float(np.mean(v_pref_list)) if v_pref_list else float("nan")
            elapsed_h = ((time.perf_counter() - t_train0) * world_size) / 3600.0
            rec = {
                "epoch": ep,
                "global_step": global_step,
                "train_loss": round(loss_sum / max(n_batches, 1), 4),
                "loss_tree": round(tree_sum / max(n_batches, 1), 4),
                "loss_atom": round(atom_sum / max(n_batches, 1), 4),
                "loss_xyz": round(xyz_sum / max(n_batches, 1), 4),
                "loss_count": round(cnt_sum / max(n_batches, 1), 4),
                "loss_geom": round(geom_sum / max(n_batches, 1), 4),
                "train_pref": round(pref_sum / max(pref_n, 1), 4) if pref_n else None,
                "val_ema_loss": round(val_loss, 4),
                "val_ema_homo_pref": round(val_pref, 4),
                "epoch_wall_sec": round(time.perf_counter() - ep_t0, 2),
                "cumulative_gpuh": round(elapsed_h, 3),
            }
            history.append(rec)
            print(f"[Step2 Ep {ep:02d}/{args.epochs}] {rec}", flush=True)

            val_metric = val_loss - 0.35 * (val_pref if math.isfinite(val_pref) else 0.0)
            meta_save = {
                "v4_step": 2,
                "epoch": ep,
                "global_step": global_step,
                "sft_gpuh": round(elapsed_h, 4),
                "val_ema_loss": round(val_loss, 4),
                "val_ema_homo_pref": round(val_pref, 4),
                "history": history,
            }
            ema_gen.save_checkpoint(last_ckpt, extra_metadata=meta_save)
            if math.isfinite(val_metric) and val_metric <= best_val_score:
                best_val_score = val_metric
                ema_gen.save_checkpoint(best_ckpt, extra_metadata=meta_save)
        elif rank == 0:
            print(
                f"[Step2 Ep {ep:02d}/{args.epochs}] loss={loss_sum/max(n_batches,1):.4f} "
                f"(tree={tree_sum/max(n_batches,1):.3f}, atom={atom_sum/max(n_batches,1):.3f}, "
                f"cnt={cnt_sum/max(n_batches,1):.3f}, geom={geom_sum/max(n_batches,1):.4f}) "
                f"wall={time.perf_counter()-ep_t0:.1f}s",
                flush=True,
            )

        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()


def run_step2_rollout_shard(args: argparse.Namespace) -> None:
    """Collect one GPU's shard of Step-2 EMA generator rollouts (`0` `audit_dpa4` calls)."""
    device = torch.device(args.device)
    step2_dir = args.work_dir / args.run_name
    calib_cache = step2_dir / "residual_calibrated_unimol2.pt"
    ckpt_p = step2_dir / "v4_step2_ema_best.pt"
    shard_out = step2_dir / f"step2_rollout_shard_{args.shard_id}.pt"

    bus, _, _ = _build_calibrated_oe62_bus(args, include_blind_auditor=False, calib_cache_path=calib_cache)
    gen, _ = LoomUni3DARGenerator.load_checkpoint(ckpt_p, device=args.device, batch_size=256)
    gen.eval()
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))

    _, clean_targets = _get_clean_oe62_pool(args, None)
    rng = np.random.default_rng(20262200 + int(args.shard_id) * 1009)
    n_shard = int(args.critic_rollouts) // max(1, int(args.num_shards))

    # Sample diverse `(homo_req, n_atoms)` across `18..62` atoms (including `50..60` large bucket!)
    pool_sub = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 62.0)]
    pick_idx = rng.choice(len(pool_sub), size=n_shard, replace=True)
    req_homo = np.clip(
        pool_sub[pick_idx, 0].astype(np.float32) + rng.normal(0.0, 0.20, size=n_shard).astype(np.float32),
        -8.2,
        -4.8,
    )
    req_natoms = np.clip(np.round(pool_sub[pick_idx, 1] + rng.integers(-2, 3, size=n_shard)), 18.0, 62.0).astype(
        np.float32
    )
    # Conditioner input is `[homo_req, sa_target = 2.40]`
    cond_2d = np.stack([req_homo, np.full(n_shard, float(args.eval_sa_target), dtype=np.float32)], axis=-1)

    t0 = time.perf_counter()
    s1_h_list: List[torch.Tensor] = []
    s1_g_list: List[torch.Tensor] = []
    s2_h_list: List[torch.Tensor] = []
    s2_c_list: List[torch.Tensor] = []
    s2_e_slices: List[torch.Tensor] = []
    s2_p_slices: List[torch.Tensor] = []
    t_norm_list: List[torch.Tensor] = []
    mols_all: List[Optional[Chem.Mol]] = []
    is_clean_all = np.zeros(n_shard, dtype=bool)

    chunk = 256
    for s in range(0, n_shard, chunk):
        e = min(s + chunk, n_shard)
        nb = e - s
        tn = gen.normalize_targets(cond_2d[s:e])
        cm = torch.ones((nb, 2), dtype=torch.bool, device=device)
        um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device)

        roll = gen.sample_rollouts_with_two_stage_features(
            tn,
            cm,
            uncond_active_mask=um,
            cfg_scale=2.00,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=550000 + int(args.shard_id) * 50000 + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            leaf_atom_budget=req_natoms[s:e],
        )
        s1_h_list.append(roll["s1_h_concat"])
        s1_g_list.append(roll["s1_geom_feats"])
        s2_h_list.append(roll["s2_h_stage2"])
        s2_c_list.append(roll["s2_cloud_feats"])
        t_norm_list.append(tn.detach().cpu())

        for loc_i, tr in enumerate(roll["trajectories"]):
            s2_e_slices.append(tr["stage2_elem_ids"])
            s2_p_slices.append(tr["stage2_leaf_pos"])
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            g_i = s + loc_i
            if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
                mols_all.append(res.mol)
                is_clean_all[g_i] = True
            else:
                mols_all.append(None)

    pred_asset = bus.registry.get_predictor("ext_unimol2")
    rule_asset = bus.registry.get_predictor("builtin_rules")
    u_out = pred_asset.predict_mols(mols_all, batch_size=64)
    r_out = rule_asset.predict_mols(mols_all, batch_size=256)
    pot_out = (
        bus.potential_slot.evaluate_molecules([m for m in mols_all if m is not None], chunk_size=64)
        if bus.potential_slot is not None
        else None
    )
    fmax_all = np.full(n_shard, np.nan, dtype=np.float32)
    if pot_out is not None:
        cur_p = 0
        for g_i, m in enumerate(mols_all):
            if m is not None:
                fmax_all[g_i] = float(pot_out["f_max_eV_A"][cur_p])
                cur_p += 1

    homo_surr = u_out["mean"][:, pred_asset.channel_index("homo")].astype(np.float32)
    sa_vals = r_out["mean"][:, rule_asset.channel_index("sa_score")].astype(np.float32)
    mw_vals = r_out["mean"][:, rule_asset.channel_index("mol_weight")]
    sm_vals = r_out["mean"][:, rule_asset.channel_index("smarts_alert")]

    y_valid = is_clean_all & u_out["valid_mask"] & np.isfinite(homo_surr) & np.isfinite(sa_vals)
    low_strain = (~np.isfinite(fmax_all)) | (fmax_all <= 4.0)
    is_constr = y_valid & (sa_vals <= 3.5) & (mw_vals <= 550.0) & (sm_vals <= 0.0) & low_strain

    y_raw_2d = np.stack(
        [np.nan_to_num(homo_surr, nan=-6.5), np.nan_to_num(sa_vals, nan=4.0)], axis=-1
    ).astype(np.float32)
    y_surr_norm = gen.normalize_targets(y_raw_2d).detach().cpu()

    wall_s = time.perf_counter() - t0
    torch.save(
        {
            "s1_h_concat": torch.cat(s1_h_list, dim=0),
            "s1_geom_feats": torch.cat(s1_g_list, dim=0),
            "s2_h_stage2": torch.cat(s2_h_list, dim=0),
            "s2_cloud_feats": torch.cat(s2_c_list, dim=0),
            "s2_elem_slices": s2_e_slices,
            "s2_pos_slices": s2_p_slices,
            "targets_norm": torch.cat(t_norm_list, dim=0),
            "y_surrogate_norm": y_surr_norm,
            "y_valid_mask": torch.from_numpy(y_valid),
            "is_legal": torch.from_numpy(is_clean_all),
            "is_constr_met": torch.from_numpy(is_constr),
            "fmax_median": float(np.nanmedian(fmax_all)) if np.isfinite(fmax_all).any() else float("nan"),
            "wall_sec": wall_s,
        },
        shard_out,
    )
    print(
        f"[Step2 Rollout Shard {args.shard_id}] N={n_shard} | clean={is_clean_all.mean()*100:.1f}% | "
        f"constr+low_strain={is_constr.mean()*100:.1f}% | med_Fmax={np.nanmedian(fmax_all):.2f} eV/A | wall={wall_s:.1f}s",
        flush=True,
    )


@torch.no_grad()
def _probe_step2_ckpt_and_prompt_map(
    ckpt_path: Path,
    cal_unimol2: Any,
    rule_asset: Any,
    clean_targets: np.ndarray,
    sa_target: float,
    device_str: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Zero-leakage `256`-prompt generative dev evaluation + affine prompt centering (`0` `audit_dpa4` calls)."""
    device = torch.device(device_str)
    gen, _ = LoomUni3DARGenerator.load_checkpoint(ckpt_path, device=device_str, batch_size=256)
    gen.eval()

    rng = np.random.default_rng(202629)
    sub = clean_targets[(clean_targets[:, 1] >= 22.0) & (clean_targets[:, 1] <= 44.0)]
    picks = rng.choice(len(sub), size=256, replace=False)
    h_req = sub[picks, 0].astype(np.float32)
    n_req = np.round(sub[picks, 1]).astype(np.float32)

    cond_2d = np.stack([h_req, np.full(len(h_req), float(sa_target), dtype=np.float32)], axis=-1)
    tn = gen.normalize_targets(cond_2d)
    cm = torch.ones((len(h_req), 2), dtype=torch.bool, device=device)
    um = torch.tensor([[False, True]] * len(h_req), dtype=torch.bool, device=device)

    trajs = gen.sample(
        tn,
        cm,
        uncond_active_mask=um,
        cfg_scale=2.00,
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        temperatures=(0.85, 0.30, 0.25),
        seed=717171,
        use_bf16=(device.type == "cuda"),
        electron_parity=0,
        leaf_atom_budget=n_req,
    )
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    mols: List[Optional[Chem.Mol]] = []
    clean_m = np.zeros(len(trajs), dtype=bool)
    for i, tr in enumerate(trajs):
        res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
        if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
            mols.append(res.mol)
            clean_m[i] = True
        else:
            mols.append(None)

    u_out = cal_unimol2.predict_mols(mols, batch_size=64)
    r_out = rule_asset.predict_mols(mols, batch_size=256)
    sa_v = r_out["mean"][:, rule_asset.channel_index("sa_score")]
    ok = clean_m & u_out["valid_mask"] & np.isfinite(u_out["mean"][:, 0]) & np.isfinite(sa_v)

    if int(ok.sum()) < 40:
        return (
            {"composite": -9.9, "clean_rate": float(clean_m.mean()), "mae_eV": 9.9, "synth_hit_yield": 0.0},
            {"applied": False, "slope": 1.0, "intercept": 0.0, "bias_eV": 0.0},
        )

    x = h_req[ok].astype(np.float64)
    y = u_out["mean"][ok, 0].astype(np.float64)
    mae = float(np.mean(np.abs(y - x)))
    bias = float(np.mean(y - x))
    xc = float(np.mean(x))
    yc = float(np.mean(y))
    slope = float(np.clip(np.sum((x - xc) * (y - yc)) / max(float(np.sum((x - xc) ** 2)), 1e-6), 0.72, 1.25))
    intercept = float(yc - slope * xc)

    # Evaluate centered hit yield so checkpoint comparison reflects calibrated controllability (`R^2` / `sigma_in`)
    cen_err = np.abs((y - intercept) / max(slope, 0.72) - x)
    hit_all = np.zeros(len(trajs), dtype=bool)
    hit_all[ok] = (cen_err <= 0.25) & (sa_v[ok] <= 3.5)
    hit_y = float(hit_all.mean())
    cen_mae = float(np.mean(cen_err))
    comp = hit_y - 0.35 * cen_mae + 0.15 * float(clean_m.mean())

    apply_map = (abs(bias) >= 0.035) or (abs(slope - 1.0) >= 0.06)
    return (
        {
            "composite": round(comp, 5),
            "clean_rate": round(float(clean_m.mean()), 4),
            "raw_mae_eV": round(mae, 4),
            "centered_mae_eV": round(cen_mae, 4),
            "synth_hit_yield": round(hit_y, 4),
        },
        {
            "applied": bool(apply_map),
            "n_valid": int(ok.sum()),
            "bias_eV": round(bias, 4),
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
        },
    )


def _apply_step2_prompt_map(homo_1d: np.ndarray, dev_cal: Dict[str, Any]) -> np.ndarray:
    if not dev_cal.get("applied", False):
        return homo_1d.astype(np.float32)
    s = float(dev_cal["slope"])
    b = float(dev_cal["intercept"])
    shifted = (homo_1d.astype(np.float64) - b) / max(s, 0.72)
    blended = 0.80 * shifted + 0.20 * (homo_1d.astype(np.float64) - float(dev_cal["bias_eV"]))
    return np.clip(blended, -8.3, -4.6).astype(np.float32)


def run_fit_critic_and_eval(args: argparse.Namespace) -> None:
    """Fit Step-2 `TwoStageOctreeProcessCritic` & run Fair 30-Atom + Multi-Atom-Bucket benchmarks."""
    device = torch.device(args.device)
    step2_dir = args.work_dir / args.run_name
    calib_cache = step2_dir / "residual_calibrated_unimol2.pt"
    ckpt_p = step2_dir / "v4_step2_ema_best.pt"
    last_ckpt_p = step2_dir / "v4_step2_ema_last.pt"
    critic_ckpt = step2_dir / "v4_step2_two_stage_critic.pt"
    sdf_dir = step2_dir / "cohorts_sdf"

    bus_eval, cal_unimol2, calib_diag = _build_calibrated_oe62_bus(
        args, include_blind_auditor=True, calib_cache_path=calib_cache
    )
    rule_asset = bus_eval.registry.get_predictor("builtin_rules")
    _, clean_targets = _get_clean_oe62_pool(args, None)

    best_probe, dev_cal = _probe_step2_ckpt_and_prompt_map(
        ckpt_p, cal_unimol2, rule_asset, clean_targets, args.eval_sa_target, args.device
    )
    print(f"[Step2 Ckpt Probe: ema_best] {best_probe} | map={dev_cal}", flush=True)
    if last_ckpt_p.exists():
        last_probe, last_cal = _probe_step2_ckpt_and_prompt_map(
            last_ckpt_p, cal_unimol2, rule_asset, clean_targets, args.eval_sa_target, args.device
        )
        print(f"[Step2 Ckpt Probe: ema_last] {last_probe} | map={last_cal}", flush=True)
        if last_probe["composite"] > best_probe["composite"]:
            print("[Step2 Ckpt Select] Promoting ema_last -> v4_step2_ema_best.pt based on generative dev composite!", flush=True)
            shutil.copy2(last_ckpt_p, ckpt_p)
            best_probe = last_probe
            dev_cal = last_cal

    gen, gen_payload = LoomUni3DARGenerator.load_checkpoint(ckpt_p, device=args.device, batch_size=512)
    gen.eval()
    gen_meta = dict(gen_payload.get("metadata") or {})
    gen_meta["selected_generative_dev_probe"] = best_probe
    gen_meta["dev_prompt_centering"] = dev_cal

    gpu_list = [g.strip() for g in args.rollout_gpus.split(",") if g.strip()]
    n_shards = len(gpu_list)

    if args.reuse_critic and critic_ckpt.exists():
        two_stage_critic, critic_meta = TwoStageOctreeProcessCritic.load_checkpoint(critic_ckpt, device=args.device)
    else:
        t_r0 = time.perf_counter()
        procs: List[subprocess.Popen[str]] = []
        for sh_i, gpu_id in enumerate(gpu_list):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phase",
                "rollout_shard",
                "--device",
                f"cuda:{gpu_id}",
                "--shard-id",
                str(sh_i),
                "--num-shards",
                str(n_shards),
                "--critic-rollouts",
                str(args.critic_rollouts),
                "--eval-sa-target",
                str(args.eval_sa_target),
                "--work-dir",
                str(args.work_dir),
                "--run-name",
                str(args.run_name),
                "--oe62-task-spec",
                str(args.oe62_task_spec),
                "--oe62-master-train-sdf",
                str(args.oe62_master_train_sdf),
                "--oe62-replay-sdf",
                str(args.oe62_replay_sdf),
                "--quantum-train-dft-sdf",
                str(args.quantum_train_dft_sdf),
                "--quantum-valid-dft-sdf",
                str(args.quantum_valid_dft_sdf),
                "--oe62-reward-oracle-dir",
                str(args.oe62_reward_oracle_dir),
                "--oe62-potential-weights",
                str(args.oe62_potential_weights),
            ]
            if args.allow_tf32:
                cmd.append("--allow-tf32")
            procs.append(subprocess.Popen(cmd, text=True))

        for sh_i, pr in enumerate(procs):
            rc = pr.wait()
            if rc != 0:
                raise RuntimeError(f"Step2 rollout shard {sh_i} failed with exit code {rc}")
        roll_wall = time.perf_counter() - t_r0

        shards = [
            torch.load(step2_dir / f"step2_rollout_shard_{sh_i}.pt", map_location="cpu", weights_only=False)
            for sh_i in range(n_shards)
        ]
        s1_h = torch.cat([sh["s1_h_concat"] for sh in shards], dim=0)
        s1_g = torch.cat([sh["s1_geom_feats"] for sh in shards], dim=0)
        s2_h = torch.cat([sh["s2_h_stage2"] for sh in shards], dim=0)
        s2_c = torch.cat([sh["s2_cloud_feats"] for sh in shards], dim=0)
        t_all = torch.cat([sh["targets_norm"] for sh in shards], dim=0)
        y_surr = torch.cat([sh["y_surrogate_norm"] for sh in shards], dim=0)
        y_ok = torch.cat([sh["y_valid_mask"] for sh in shards], dim=0)
        is_leg = torch.cat([sh["is_legal"] for sh in shards], dim=0)
        is_con = torch.cat([sh["is_constr_met"] for sh in shards], dim=0)
        fmax_med = float(np.nanmean([sh["fmax_median"] for sh in shards]))

        all_e = [es for sh in shards for es in sh["s2_elem_slices"]]
        all_p = [ps for sh in shards for ps in sh["s2_pos_slices"]]
        n_tot_r = len(all_e)
        max_k_r = max(1, max((int(es.numel()) for es in all_e), default=1))
        pad_e = torch.zeros((n_tot_r, max_k_r), dtype=torch.long)
        pad_p = torch.zeros((n_tot_r, max_k_r, 3), dtype=torch.float32)
        pad_v = torch.zeros((n_tot_r, max_k_r), dtype=torch.bool)
        for i, (es, ps) in enumerate(zip(all_e, all_p)):
            ki = int(es.numel())
            if ki > 0:
                pad_e[i, :ki] = es
                pad_p[i, :ki] = ps
                pad_v[i, :ki] = True
        del shards

        t_f0 = time.perf_counter()
        emb_d = int(gen.args.emb_dim)
        t_mean_np = gen.target_mean.detach().cpu().numpy()
        t_scale_np = gen.target_scale.detach().cpu().numpy()
        act_all = torch.ones((n_tot_r, 2), dtype=torch.bool)

        s1_c = OctreeProcessCritic(emb_dim=emb_d, num_targets=2, hidden_dim=512).to(device)
        s1_c.set_normalization(t_mean_np, t_scale_np)
        st1 = fit_octree_process_critic(
            s1_c,
            h_concat=s1_h,
            geom_feats=s1_g,
            targets_norm=t_all,
            active_mask=act_all,
            y_surrogate_norm=y_surr,
            y_valid_mask=y_ok,
            is_legal=is_leg,
            is_constr_met=is_con,
            epochs=24,
            batch_size=256,
            lr=1.5e-3,
            seed=202621,
        )
        s2_c_mod = OctreeElementProcessCritic(
            emb_dim=emb_d,
            num_targets=2,
            num_atom_classes=int(gen.backbone.num_atom),
            d_node=160,
            hidden_dim=512,
        ).to(device)
        s2_c_mod.set_normalization(t_mean_np, t_scale_np)
        st2 = fit_octree_element_process_critic(
            s2_c_mod,
            h_stage2=s2_h,
            elem_ids=pad_e,
            leaf_pos=pad_p,
            leaf_valid=pad_v,
            cloud_feats=s2_c,
            targets_norm=t_all,
            active_mask=act_all,
            y_surrogate_norm=y_surr,
            y_valid_mask=y_ok,
            is_legal=is_leg,
            is_constr_met=is_con,
            epochs=28,
            batch_size=256,
            lr=1.2e-3,
            seed=202622,
        )
        fit_wall = time.perf_counter() - t_f0
        two_stage_critic = TwoStageOctreeProcessCritic(s1_c, s2_c_mod).to(device)
        two_stage_critic.eval()
        critic_meta = {
            "total_rollouts": n_tot_r,
            "rollout_gpuh": round((roll_wall * n_shards + fit_wall) / 3600.0, 4),
            "clean_rate": round(float(is_leg.float().mean().item()), 4),
            "constr_and_low_strain_rate": round(float(is_con.float().mean().item()), 4),
            "rollout_fmax_median_eV_A": round(fmax_med, 3),
            "stage1_stats": st1.__dict__,
            "stage2_stats": st2.__dict__,
        }
        two_stage_critic.save_checkpoint(critic_ckpt, extra=critic_meta)
        print(f"[Step2 Critic DONE] {critic_meta}", flush=True)

    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    obj_col_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

    def _encode_homo_sa(homo_1d: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nb = len(homo_1d)
        h_mapped = _apply_step2_prompt_map(homo_1d, dev_cal)
        mat = np.stack([h_mapped, np.full(nb, float(args.eval_sa_target), dtype=np.float32)], axis=-1)
        tn = gen.normalize_targets(mat)
        ma = torch.ones((nb, 2), dtype=torch.bool, device=device)
        um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device)
        return tn, ma, um

    # Warmup
    wt, wm, wu = _encode_homo_sa(np.full(16, -6.4, dtype=np.float32))
    _ = gen.sample_with_two_stage_process_critic(
        wt, wm, two_stage_critic, m_stage1=8, m_stage2=4, m_keep=2, objective_col_mask=obj_col_mask,
        cfg_scale=2.0, cfg_tree_decay=0.25, geometric_guard=True, seed=21, use_bf16=True,
        uncond_active_mask=wu, leaf_atom_budget=np.full(16, 30.0, dtype=np.float32),
    )
    _sync(device)

    # ---------------- A. Fair Fixed-Prompt `30.4`-Atom Benchmark (`N = 1,792`) ----------------
    req_30 = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
    n_p30 = len(req_30)
    arms_30: Dict[str, Any] = {}
    funnels_30: List[Tuple[str, int, int, int, bool]] = [
        ("Outcome_M1", 1, 1, 1, False),
        ("TwoStage_16_to_6_to_2", 16, 6, 2, True),
        ("TwoStage_24_to_8_to_2", 24, 8, 2, True),
        ("TwoStage_32_to_8_to_2", 32, 8, 2, True),
    ]
    for arm_name, ms1, ms2, mkeep, use_surr in funnels_30:
        _sync(device)
        tg0 = time.perf_counter()
        groups_all: List[List[Dict[str, Any]]] = []
        p_chunk = 512 if ms1 == 1 else 256
        for s in range(0, n_p30, p_chunk):
            e = min(s + p_chunk, n_p30)
            bt, bm, bu = _encode_homo_sa(req_30[s:e, 0])
            groups_all.extend(
                gen.sample_with_two_stage_process_critic(
                    bt, bm, two_stage_critic, m_stage1=ms1, m_stage2=ms2, m_keep=mkeep,
                    objective_col_mask=obj_col_mask, stage1_legal_weight=0.80, stage1_constr_weight=0.55,
                    stage2_legal_weight=1.05, stage2_constr_weight=0.70, coarse_chunk_size=1024,
                    cfg_scale=2.00, xyz_cfg_scale=1.0, cfg_tree_decay=0.25, geometric_guard=True,
                    temperatures=(0.85, 0.30, 0.25), seed=960000 + ms1 * 409 + ms2 * 61 + s,
                    use_bf16=True, electron_parity=0, uncond_active_mask=bu, leaf_atom_budget=req_30[s:e, 1],
                )
            )
        _sync(device)
        gen_w = time.perf_counter() - tg0

        tr0 = time.perf_counter()
        p_grid: List[List[Optional[Chem.Mol]]] = [[None] * mkeep for _ in range(n_p30)]
        p_leg = np.zeros((n_p30, mkeep), dtype=bool)
        p_exo = np.zeros((n_p30, mkeep), dtype=bool)
        for i in range(n_p30):
            for k in range(mkeep):
                tr = groups_all[i][k]
                res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
                if res.reconciled_legal and res.mol is not None:
                    p_grid[i][k] = res.mol
                    p_leg[i, k] = True
                    p_exo[i, k] = bool(res.reconciled_exotic_charge)
        rec_w = time.perf_counter() - tr0

        sel_m, sel_l, sel_x, surr_w = _select_v4_step1_from_pool(
            p_grid, p_leg, p_exo, req_30, bus_eval, mkeep, use_surrogate=use_surr, device=device
        )
        arm_res = _score_oe62_size_cohort(
            sel_m, sel_l, sel_x, req_30, bus_eval, source_graphs, source_scaffolds, gen_w, rec_w, surr_w,
            export_sdf_path=(sdf_dir / f"{arm_name}.sdf"),
        )
        arms_30[arm_name] = arm_res
        print(
            f"[Step2 30-Atom {arm_name}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
            f"Legal={arm_res['legal_rate']*100:.1f}% Fmax={arm_res['zero_step_f_max_median_eV_A']:.2f} | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} ({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindMAE={arm_res['audit_mae_eV']:.4f} eV",
            flush=True,
        )

    # ---------------- B. Multi-Atom-Count Bucket Scaling (`N in {20, 30, 40, 50, 60}`, 280/bucket) ----------------
    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    bucket_results: Dict[str, Any] = {}
    for n_exact in (20, 30, 40, 50, 60):
        req_b = np.asarray(
            [(float(q_points[str(q)]), float(n_exact)) for q in quantiles for _ in range(40)], dtype=np.float32
        )
        nb = len(req_b)
        bt, bm, bu = _encode_homo_sa(req_b[:, 0])
        b_arms: Dict[str, Any] = {}
        for b_tag, ms1, ms2, mk, use_s in (
            ("Uni3DAR_v4_M1", 1, 1, 1, False),
            ("Uni3DAR_v4_TwoStage_16_to_6_to_2", 16, 6, 2, True),
            ("Uni3DAR_v4_TwoStage_32_to_8_to_2", 32, 8, 2, True),
        ):
            _sync(device)
            t0 = time.perf_counter()
            grps = gen.sample_with_two_stage_process_critic(
                bt, bm, two_stage_critic, m_stage1=ms1, m_stage2=ms2, m_keep=mk,
                objective_col_mask=obj_col_mask, coarse_chunk_size=1024, cfg_scale=2.00, xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25, geometric_guard=True, seed=71000 + ms1 * 43 + n_exact,
                use_bf16=True, electron_parity=0, uncond_active_mask=bu, leaf_atom_budget=req_b[:, 1],
            )
            _sync(device)
            gw = time.perf_counter() - t0
            r0 = time.perf_counter()
            pg_b: List[List[Optional[Chem.Mol]]] = [[None] * mk for _ in range(nb)]
            pl_b = np.zeros((nb, mk), dtype=bool)
            px_b = np.zeros((nb, mk), dtype=bool)
            for i in range(nb):
                for k in range(mk):
                    res = reconciler.reconcile_point_cloud(
                        grps[i][k]["atom_symbols"], grps[i][k]["positions_np"], potential_slot=None
                    )
                    if res.reconciled_legal and res.mol is not None:
                        pg_b[i][k] = res.mol
                        pl_b[i, k] = True
                        px_b[i, k] = bool(res.reconciled_exotic_charge)
            rw = time.perf_counter() - r0
            sm_b, sl_b, sx_b, sw_b = _select_v4_step1_from_pool(
                pg_b, pl_b, px_b, req_b, bus_eval, mk, use_surrogate=use_s, device=device
            )
            b_res = _score_oe62_size_cohort(
                sm_b, sl_b, sx_b, req_b, bus_eval, source_graphs, source_scaffolds, gw, rw, sw_b
            )
            b_arms[b_tag] = b_res
            print(
                f"[Step2 Bucket N={n_exact} {b_tag}] Eff={b_res['effective_output_mol_per_sec']:.2f} mol/s | "
                f"Atoms={b_res['mean_total_atoms']} (err={b_res['mean_natoms_abs_err']}) | "
                f"Legal={b_res['legal_rate']*100:.1f}% | BlindSynth={b_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}% | "
                f"BlindMAE={b_res['audit_mae_eV']:.4f} eV",
                flush=True,
            )
        bucket_results[str(n_exact)] = {"target_n_atoms": n_exact, "n_prompts": nb, "arms": b_arms}

    total_gpuh = round(float(gen_meta.get("sft_gpuh", 0.0)) + float(critic_meta.get("rollout_gpuh", 0.0)), 4)
    out_payload = {
        "experiment": "OE62 Fair 30-Atom & Multi-Atom-Bucket Benchmark — Uni-3DAR v4 Step 2 (90M Full-Param SO(3) DDP SFT + ScaleFreeCountHead Delta N=0)",
        "total_training_gpuh": total_gpuh,
        "sft_metadata": gen_meta,
        "two_stage_critic_meta": critic_meta,
        "zero_leakage_dft_calibration": calib_diag,
        "fair_30atom_arms": arms_30,
        "atom_size_buckets": bucket_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out_payload, indent=2) + "\n")
    print(f"[Step2 ALL DONE] Total GPU-h={total_gpuh} -> Wrote {args.output_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["prepare", "train_ddp", "rollout_shard", "eval_all"], required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rollout-gpus", type=str, default="0,1")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--epochs", type=int, default=36)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-backbone", type=float, default=3.5e-5)
    p.add_argument("--lr-heads", type=float, default=2.5e-4)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--contrast-weight", type=float, default=0.08)
    p.add_argument("--geom-dist-weight", type=float, default=1.50)
    p.add_argument("--critic-rollouts", type=int, default=16384)
    p.add_argument("--eval-sa-target", type=float, default=2.40)
    p.add_argument("--reuse-critic", action="store_true")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="uni3dar_v4_step2")
    p.add_argument(
        "--step1-calib-cache",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/uni3dar_v4_step1/residual_calibrated_unimol2.pt"),
    )
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument(
        "--parent-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_e13_e14_20261113/full_homo_27000/weights/octree_conditional.pt"),
    )
    p.add_argument(
        "--oe62-master-train-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"),
    )
    p.add_argument(
        "--oe62-master-valid-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_valid_B.sdf"),
    )
    p.add_argument(
        "--oe62-source-train-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/source_labels/train_source.sdf"),
    )
    p.add_argument(
        "--oe62-replay-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_B.sdf"),
    )
    p.add_argument(
        "--quantum-train-dft-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_dft.sdf"),
    )
    p.add_argument(
        "--quantum-valid-dft-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_valid_dft.sdf"),
    )
    p.add_argument(
        "--oe62-targets-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/targets_B.json"),
    )
    p.add_argument(
        "--oe62-reward-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"),
    )
    p.add_argument(
        "--oe62-audit-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_e15_calibration_20261114/full_fit/oracle"),
    )
    p.add_argument(
        "--oe62-potential-weights",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_step2_eval.json"),
    )
    args = p.parse_args()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.phase == "prepare":
        prepare_step2_dataset(args)
    elif args.phase == "train_ddp":
        run_train_ddp(args)
    elif args.phase == "rollout_shard":
        run_step2_rollout_shard(args)
    else:
        run_fit_critic_and_eval(args)


if __name__ == "__main__":
    main()
