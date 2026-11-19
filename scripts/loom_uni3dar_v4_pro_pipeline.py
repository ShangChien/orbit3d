"""Uni-3DAR v4-Pro (< 3.0 GPU-h) Ceiling-Shattering Pipeline on OE62 (GPU 0 & GPU 1 ONLY).

Strictly enforces all four v4 Red Lines:
  1. Zero PCA canonicalization (`random_rotation_prob = 0.80`, full SO(3)).
  2. Zero `n_atoms` in `LoomConditioner` (`["homo", "sa_score"]` only; `ScaleFreeBudgetEmbedding`).
  3. Zero inference-time `DPA-4` relaxation (`polish_hydrogens_mlip=False`, `potential_slot=None`, `recycle=1`).
  4. Zero `audit_dpa4` leakage (only `quantum_train_dft` calibration & dev `cal_unimol2`).

Five Stages (~2.3–2.6 GPU-h total vs PropMolFlow's 52.12 GPU-h = ~20x less compute):
  - `relax_pro_shard`: Stratified 22,000-molecule offline `DPA-4` pool across 10 HOMO deciles x 4 size
    bands (`N<25, 25..35, 35..45, 45..65`), relaxing SA cap to `4.20` for `N>=40` so large molecules
    are richly represented (~0.32 GPU-h on 2 GPUs).
  - `sft_pro_ddp`: Full-parameter layer-wise-LR 2-GPU DDP SFT (`npref=0`, 4-way CFG masks incl. 20%
    `[False, True]` for native `HOMO-only CFG`, HOMO-only contrastive margin `0.28`) (~0.62 GPU-h).
  - `align_pro`: Multi-size (`18..60` atoms) 2-round Online Hindsight Equilibrium AWR + HOMO-dominant
    300-step Tree-Vine GRPO (`w_prop=2.60`, `size_anchored_cfg=True` on `sa_score`, `eval_sa_target=2.60`)
    + Detached SO(3) XYZ Distillation + On-Policy 4-Layer Kabsch `PostTreeGraphRefiner` (~0.85 GPU-h).
  - `rollout_pro_shard` + critic fit: Grouped `24,576`-rollout (`3,072` prompts x `8` candidates)
    post-refiner `TwoStageOctreeProcessCritic` (`n_gnn_layers=3`) trained with MSE + Grouped ListNet KL
    + Pairwise Logistic Ranking Loss (~0.48 GPU-h on 2 GPUs).
  - `eval_pro`: Uncontended single-GPU Fair `30.4`-Atom (`N=1,792`, incl. `32->8->3`, `32->8->4`,
    `48->12->4`) + Controlled Multi-Atom Buckets (`N in {20, 30, 40, 50, 60}`) using Damped Affine Prompt
    Inversion + HOMO-only CFG + Stage-2 Trunk-Diverse MMR + Conservative `sd_h / disp_rms` selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state
from loom.generator.octree_backend import OctreeDataset
from loom.utils.dataset_io import write_precise_sdf
from loom.weaving import (
    LoomGRPOTrainer,
    LoomUni3DARGenerator,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    PostTreeGraphRefiner,
    TwoStageOctreeProcessCritic,
    fit_octree_element_process_critic,
    fit_octree_process_critic,
    fit_post_tree_graph_refiner,
)
from loom.weaving.slots import PotentialSlot
from scripts.loom_oe62_size_anchored import (
    _get_clean_oe62_pool,
    _get_source_novelty_sets,
    _safe_collate_indices,
    _score_oe62_size_cohort,
)
from scripts.loom_uni3dar_v4_step1_eval import (
    _build_calibrated_oe62_bus,
    _select_v4_step1_from_pool,
    build_fair_30atom_prompts,
)
from scripts.loom_uni3dar_v4_step2_train import (
    _CollateStep2Octree,
    _Step2OctreeTorchDataset,
    _load_prepared_v4_sdf,
)
from scripts.loom_uni3dar_v4_step3_align_and_eval import (
    _distill_v4_refiner,
    _run_v4_detached_xyz_distillation,
)
from scripts.loom_uni3dar_v4_phaseA_zero_retrain import (
    _apply_affine_prompt_map,
    _fit_damped_affine_prompt_map,
    _select_best_inference_knobs_on_dev,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_4way_cfg_masks(batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    """4-way CFG dropout (`55% [T,T], 15% [T,F], 20% [F,T] SA-only, 10% [F,F]`) enabling native HOMO-only CFG."""
    u = rng.random(size=batch_size)
    mask = np.ones((batch_size, 2), dtype=bool)
    mask[(u >= 0.55) & (u < 0.70), 1] = False  # [T, F] HOMO only
    mask[(u >= 0.70) & (u < 0.90), 0] = False  # [F, T] SA only (uncond branch for HOMO-only CFG!)
    mask[u >= 0.90] = False                    # [F, F] unconditional
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def _select_pro_stratified_candidates(
    train_sdf: Path,
    *,
    max_total: int = 22000,
    small_max_sa: float = 3.55,
    large_max_sa: float = 4.20,
    seed: int = 202650,
) -> Tuple[List[Chem.Mol], np.ndarray]:
    """Stratify OE62 across 10 HOMO deciles x 4 size bands (`<25, 25..35, 35..45, 45..65`), preserving large molecules."""
    all_mols, all_t = _load_prepared_v4_sdf(train_sdf)
    cand_mols: List[Chem.Mol] = []
    cand_t_rows: List[Tuple[float, float]] = []
    cand_n: List[int] = []
    for m, row in zip(all_mols, all_t):
        n_a = int(m.GetNumAtoms())
        sa = float(row[1])
        h = float(row[0])
        if n_a < 12 or n_a > 66 or not (math.isfinite(h) and math.isfinite(sa)):
            continue
        sa_cap = float(large_max_sa) if n_a >= 40 else float(small_max_sa)
        if sa <= sa_cap:
            cand_mols.append(m)
            cand_t_rows.append((h, sa))
            cand_n.append(n_a)

    cand_t = np.asarray(cand_t_rows, dtype=np.float32)
    cand_n_arr = np.asarray(cand_n, dtype=np.int32)
    if len(cand_mols) <= max_total:
        return cand_mols, cand_t

    rng = np.random.default_rng(seed)
    # 4 size bands with explicit large-molecule quotas so N=40..60 never starves!
    bands = [
        ("lt25", cand_n_arr < 25, 0.22),
        ("25_35", (cand_n_arr >= 25) & (cand_n_arr < 35), 0.36),
        ("35_45", (cand_n_arr >= 35) & (cand_n_arr < 45), 0.22),
        ("45_66", cand_n_arr >= 45, 0.20),
    ]
    picked_set: set[int] = set()
    picked_list: List[int] = []
    h_edges = np.quantile(cand_t[:, 0], np.linspace(0.0, 1.0, 11))

    for _, b_mask, frac in bands:
        b_idx = np.nonzero(b_mask)[0]
        if len(b_idx) == 0:
            continue
        target_b = int(round(max_total * frac))
        per_dec = max(1, target_b // 10)
        for d in range(10):
            lo, hi = h_edges[d], h_edges[d + 1]
            in_d = b_idx[
                (cand_t[b_idx, 0] >= lo) & (cand_t[b_idx, 0] <= hi if d == 9 else cand_t[b_idx, 0] < hi)
            ]
            if len(in_d) == 0:
                continue
            take = min(per_dec, len(in_d))
            chosen = rng.choice(in_d, size=take, replace=False)
            for idx_i in chosen.tolist():
                if idx_i not in picked_set:
                    picked_set.add(idx_i)
                    picked_list.append(idx_i)

    if len(picked_list) < max_total:
        rem = [i for i in range(len(cand_mols)) if i not in picked_set]
        if rem:
            extra = rng.choice(rem, size=min(max_total - len(picked_list), len(rem)), replace=False)
            picked_list.extend(extra.tolist())

    picked_arr = np.asarray(picked_list[:max_total], dtype=np.int64)
    rng.shuffle(picked_arr)
    out_mols = [cand_mols[int(i)] for i in picked_arr]
    out_t = cand_t[picked_arr]
    return out_mols, out_t


def run_relax_pro_shard(args: argparse.Namespace) -> None:
    """Offline `DPA-4` gas-phase relaxation shard for stratified 22k multi-size SFT pool (`0` PCA)."""
    pro_dir = args.work_dir / args.run_name
    pro_dir.mkdir(parents=True, exist_ok=True)
    train_sdf = args.step2_dir / "oe62_v4_train_eq.sdf"
    cand_mols, cand_t = _select_pro_stratified_candidates(
        train_sdf,
        max_total=int(args.relax_pool_cap),
        small_max_sa=float(args.small_max_sa),
        large_max_sa=float(args.large_max_sa),
        seed=202650,
    )
    n_shards = max(1, int(args.num_shards))
    sh_id = int(args.shard_id)
    sub_mols = cand_mols[sh_id::n_shards]
    sub_t = cand_t[sh_id::n_shards]

    t0 = time.perf_counter()
    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.oe62_potential_weights), device=args.device)
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
    rel_out = pot_slot.relax_molecules(sub_mols, n_steps=20, chunk_size=48)
    admitted: List[Chem.Mol] = []
    for i, rm in enumerate(rel_out["relaxed_mols"]):
        f_fin = float(rel_out["final_f_max_eV_A"][i])
        if (
            rm is None
            or not bool(rel_out["graph_preserved"][i])
            or not math.isfinite(f_fin)
            or f_fin > 2.45
            or not is_clean_organic_charge_state(rm)
        ):
            continue
        cm = Chem.Mol(rm)
        cm.SetProp("v4_homo_eV", f"{float(sub_t[i, 0]):.6f}")
        cm.SetProp("v4_sa_score", f"{float(sub_t[i, 1]):.4f}")
        admitted.append(cm)

    probe_ds = OctreeDataset(admitted, probe_gen.backbone, training=False, seed=42)
    clean_out: List[Chem.Mol] = []
    for idx, m in enumerate(admitted):
        try:
            item = probe_ds.grid[idx]
            decoded = int(((item["decoder_level"] == 0) & item["decoder_is_second_atom"]).sum().item())
            if not item["cutoff_atom_count"] and decoded == m.GetNumAtoms():
                clean_out.append(m)
        except Exception:
            continue

    shard_sdf = pro_dir / f"pro_relaxed_shard_{sh_id}.sdf"
    write_precise_sdf(shard_sdf, clean_out)
    wall = time.perf_counter() - t0
    n_ge40 = sum(1 for m in clean_out if m.GetNumAtoms() >= 40)
    print(
        f"[Pro Relax Shard {sh_id}] admitted={len(clean_out)}/{len(sub_mols)} (N>=40: {n_ge40}) | wall={wall:.1f}s",
        flush=True,
    )


def ensure_pro_relaxed_pool(args: argparse.Namespace) -> Tuple[Path, float]:
    pro_dir = args.work_dir / args.run_name
    pro_dir.mkdir(parents=True, exist_ok=True)
    pool_sdf = pro_dir / "v4_pro_relaxed_pool.sdf"
    pool_meta = pro_dir / "v4_pro_relaxed_pool.json"
    if args.reuse_relaxed_pool and pool_sdf.exists() and pool_meta.exists():
        meta = json.loads(pool_meta.read_text())
        print(f"[Pro RelaxPool] Reusing {pool_sdf} ({meta})", flush=True)
        return pool_sdf, float(meta.get("relax_pool_gpuh", 0.0))

    gpu_list = [g.strip() for g in args.rollout_gpus.split(",") if g.strip()]
    n_shards = max(1, len(gpu_list))
    t0 = time.perf_counter()
    procs: List[subprocess.Popen[str]] = []
    for sh_i, gpu_id in enumerate(gpu_list):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            "relax_pro_shard",
            "--device",
            f"cuda:{gpu_id}",
            "--shard-id",
            str(sh_i),
            "--num-shards",
            str(n_shards),
            "--relax-pool-cap",
            str(args.relax_pool_cap),
            "--small-max-sa",
            str(args.small_max_sa),
            "--large-max-sa",
            str(args.large_max_sa),
            "--work-dir",
            str(args.work_dir),
            "--run-name",
            str(args.run_name),
            "--step2-dir",
            str(args.step2_dir),
            "--parent-ckpt",
            str(args.parent_ckpt),
            "--oe62-potential-weights",
            str(args.oe62_potential_weights),
        ]
        if args.allow_tf32:
            cmd.append("--allow-tf32")
        procs.append(subprocess.Popen(cmd, text=True))

    for sh_i, pr in enumerate(procs):
        if pr.wait() != 0:
            raise RuntimeError(f"relax_pro_shard {sh_i} failed")
    wall = time.perf_counter() - t0
    gpuh = round((wall * n_shards) / 3600.0, 4)

    merged: List[Chem.Mol] = []
    for sh_i in range(n_shards):
        sh_m, _ = _load_prepared_v4_sdf(pro_dir / f"pro_relaxed_shard_{sh_i}.sdf")
        merged.extend(sh_m)
    write_precise_sdf(pool_sdf, merged)
    n_ge40 = sum(1 for m in merged if m.GetNumAtoms() >= 40)
    meta = {
        "total_admitted": len(merged),
        "n_ge_40_atoms": n_ge40,
        "mean_atoms": round(float(np.mean([m.GetNumAtoms() for m in merged])), 2) if merged else 0.0,
        "wall_sec": round(wall, 2),
        "relax_pool_gpuh": gpuh,
    }
    pool_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[Pro RelaxPool DONE] {meta} -> {pool_sdf}", flush=True)
    return pool_sdf, gpuh


def run_sft_pro_ddp_worker(args: argparse.Namespace) -> None:
    """Full-parameter layer-wise-LR 2-GPU DDP SFT (`npref=0`, 4-way CFG, HOMO-only contrastive margin 0.28)."""
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device)

    pro_dir = args.work_dir / args.run_name
    out_ckpt = pro_dir / "v4_pro_sft.pt"
    pool_sdf = pro_dir / "v4_pro_relaxed_pool.sdf"
    pool_meta_p = pro_dir / "v4_pro_relaxed_pool.json"
    relax_gpuh = float(json.loads(pool_meta_p.read_text()).get("relax_pool_gpuh", 0.0)) if pool_meta_p.exists() else 0.0

    mols, targets = _load_prepared_v4_sdf(pool_sdf)
    if rank == 0:
        n_ge40 = sum(1 for m in mols if m.GetNumAtoms() >= 40)
        print(f"[Pro SFT] Loaded {len(mols)} DPA-4-relaxed conformers (N>=40: {n_ge40})", flush=True)

    data_meta = json.loads((args.step2_dir / "oe62_v4_data_meta.json").read_text())
    gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_ckpt,
        target_cols=["homo", "sa_score"],
        target_units=["eV", "sa"],
        mean=data_meta["mean"],
        scale=data_meta["scale"],
        device=str(device),
        batch_size=64,
        num_rbf=32,
        num_freqs=16,
        num_prefix_tokens=0,
        lora_rank=0,
        lora_include_xyz_heads=False,
        train_xyz_embeddings=False,
        use_valence_head=True,
        use_scale_free_count=True,
        non_trapping_budget=True,
        rope_center_prefix=False,
        random_rotation_prob=0.80,
    )
    gen.configure_trainable(
        train_conditioner=True,
        train_lora=False,
        train_xyz_embeddings=True,
        unfreeze_full_backbone=True,
    )

    ema_gen = deepcopy(gen).eval()
    for p in ema_gen.parameters():
        p.requires_grad_(False)

    ds = _Step2OctreeTorchDataset(mols, targets, gen.backbone, training=True, seed=202651)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=202651) if is_ddp else None
    loader = DataLoader(
        ds,
        batch_size=48,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        collate_fn=_CollateStep2Octree(ds.octree_ds.grid.collater),
        pin_memory=True,
        drop_last=True,
    )

    # Layer-wise discriminative learning rates (preserve base 3D grammar, unlock high HOMO slope!)
    n_layers = len(gen.backbone.decoder.layers)
    top_start = max(0, n_layers - 6)
    head_params: List[torch.nn.Parameter] = []
    top_params: List[torch.nn.Parameter] = []
    bot_params: List[torch.nn.Parameter] = []
    for name, p in gen.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("conditioner.", "valence_head.", "budget_emb.")):
            head_params.append(p)
        elif any(name.startswith(f"backbone.decoder.layers.{i}.") for i in range(top_start, n_layers)) or name.startswith(
            ("backbone.norm.", "backbone.tree_heads.", "backbone.atom_heads.")
        ):
            top_params.append(p)
        else:
            bot_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": bot_params, "lr": 2.2e-5, "weight_decay": 1e-4},
            {"params": top_params, "lr": 6.5e-5, "weight_decay": 1e-4},
            {"params": head_params, "lr": 2.8e-4, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.95),
    )
    n_ep = int(args.sft_epochs)
    total_steps = max(1, len(loader) * n_ep)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=3.0e-6)
    contrast_homo_only = torch.tensor([True, False], dtype=torch.bool, device=device)

    t0 = time.perf_counter()
    g_step = 0
    for ep in range(1, n_ep + 1):
        ep_t0 = time.perf_counter()
        ds.set_epoch(ep)
        if sampler is not None:
            sampler.set_epoch(ep)
        gen.train()
        rng_ep = np.random.default_rng(202660 + ep * 97 + rank * 13)
        l_sum, xyz_sum, pref_sum, n_pref, nb = 0.0, 0.0, 0.0, 0, 0
        for packed, raw_t in loader:
            t_norm = gen.normalize_targets(raw_t.to(device, non_blocking=True))
            act_m = _sample_4way_cfg_masks(int(raw_t.shape[0]), rng_ep, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, info = gen.compute_sft_loss(
                    packed,
                    t_norm,
                    act_m,
                    cfg_dropout_prob=0.0,
                    contrast_weight=0.28,
                    contrast_margin=0.12,
                    contrast_min_gap_eV=0.30,
                    contrast_col_mask=contrast_homo_only,
                    xyz_loss_weight=0.16,
                    valence_loss_weight=0.22,
                    count_loss_weight=0.0,
                    geom_dist_loss_weight=0.0,
                    budget_dropout_prob=0.08,
                )
            loss.backward()
            if is_ddp:
                flat_grads = [
                    (p.grad if p.grad is not None else torch.zeros_like(p))
                    for p in gen.parameters()
                    if p.requires_grad
                ]
                coal = torch._utils._flatten_dense_tensors(flat_grads)
                dist.all_reduce(coal, op=dist.ReduceOp.AVG)
                for p, gs in zip(
                    (p for p in gen.parameters() if p.requires_grad),
                    torch._utils._unflatten_dense_tensors(coal, flat_grads),
                ):
                    p.grad = gs
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
            sched.step()
            g_step += 1
            with torch.no_grad():
                eff_ema = min(0.996, (1.0 + g_step) / (10.0 + g_step))
                for pe, pc in zip(ema_gen.parameters(), gen.parameters()):
                    pe.lerp_(pc, 1.0 - eff_ema)
            l_sum += float(info["loss_total"])
            xyz_sum += float(info["loss_xyz"])
            if math.isfinite(float(info.get("pref_acc", float("nan")))):
                pref_sum += float(info["pref_acc"])
                n_pref += 1
            nb += 1

        if rank == 0:
            pref_avg = (pref_sum / max(n_pref, 1)) if n_pref > 0 else float("nan")
            print(
                f"[Pro SFT Ep {ep:02d}/{n_ep:02d}] loss={l_sum/max(nb,1):.4f} xyz={xyz_sum/max(nb,1):.4f} "
                f"homo_pref={pref_avg:.3f} ({time.perf_counter()-ep_t0:.1f}s)",
                flush=True,
            )

    if is_ddp:
        dist.barrier()
    if rank == 0:
        wall = time.perf_counter() - t0
        sft_gpuh = round((wall * world_size) / 3600.0 + relax_gpuh, 4)
        ema_gen.eval()
        ema_gen.save_checkpoint(
            out_ckpt,
            extra_metadata={
                "version": "4.5-pro-fullparam-so3-sft",
                "pool_size": len(mols),
                "epochs": n_ep,
                "relax_pool_gpuh": round(relax_gpuh, 4),
                "sft_ddp_gpuh": round((wall * world_size) / 3600.0, 4),
                "sft_total_gpuh": sft_gpuh,
                "sft_wall_sec": round(wall, 2),
            },
        )
        print(f"[Pro SFT DONE] Saved {out_ckpt} | sft_total_gpuh={sft_gpuh}", flush=True)
    if is_ddp:
        dist.destroy_process_group()


@torch.no_grad()
def _collect_pro_onpolicy_pairs(
    gen: LoomUni3DARGenerator,
    bus_train: Any,
    cal_unimol2: Any,
    req_h: np.ndarray,
    req_n: np.ndarray,
    *,
    sa_target: float,
    cfg_scale: float,
    max_sa_small: float = 3.55,
    max_sa_large: float = 4.15,
    max_fmax: float = 2.25,
    seed: int = 930000,
    device: torch.device,
) -> Tuple[List[Chem.Mol], np.ndarray, List[Tuple[np.ndarray, Chem.Mol]], Dict[str, Any]]:
    """Sample on-policy molecules with HOMO-only CFG + offline `DPA-4` relax (`0` PCA, `0` `audit_dpa4`)."""
    t0 = time.perf_counter()
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    rule_asset = bus_train.registry.get_predictor("builtin_rules")
    pot_slot = bus_train.potential_slot
    n_proposals = len(req_h)

    gen.eval()
    cand_mols: List[Chem.Mol] = []
    cand_noisy: List[np.ndarray] = []
    gen_bs = 512 if device.type == "cuda" else 64
    for s in range(0, n_proposals, gen_bs):
        e = min(s + gen_bs, n_proposals)
        nb = e - s
        cond_2d = np.stack([req_h[s:e], np.full(nb, float(sa_target), dtype=np.float32)], axis=-1)
        tn = gen.normalize_targets(cond_2d)
        cm = torch.ones((nb, 2), dtype=torch.bool, device=device)
        um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device)
        trajs = gen.sample(
            tn, cm, uncond_active_mask=um, cfg_scale=float(cfg_scale), xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25, geometric_guard=True, temperatures=(0.85, 0.30, 0.25),
            seed=int(seed) + s, use_bf16=(device.type == "cuda"), electron_parity=0,
            leaf_atom_budget=req_n[s:e], budget_guidance_scale=0.35,
        )
        for tr in trajs:
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
                cand_mols.append(res.mol)
                cand_noisy.append(np.asarray(res.mol.GetConformer().GetPositions(), dtype=np.float32))

    if not cand_mols:
        return [], np.zeros((0, 2), dtype=np.float32), [], {"n_proposals": n_proposals, "n_clean": 0, "n_relaxed": 0}

    r_out = rule_asset.predict_mols(cand_mols, batch_size=256)
    sa_c = r_out["mean"][:, rule_asset.channel_index("sa_score")]
    sm_c = r_out["mean"][:, rule_asset.channel_index("smarts_alert")]
    mw_c = r_out["mean"][:, rule_asset.channel_index("mol_weight")]
    keep_i = [
        i for i, m in enumerate(cand_mols)
        if sm_c[i] <= 0.0 and mw_c[i] <= 550.0 and sa_c[i] <= (max_sa_large if m.GetNumAtoms() >= 40 else max_sa_small)
    ]
    if not keep_i:
        return [], np.zeros((0, 2), dtype=np.float32), [], {"n_proposals": n_proposals, "n_clean": len(cand_mols), "n_relaxed": 0}

    sub_m = [cand_mols[i] for i in keep_i]
    sub_pos = [cand_noisy[i] for i in keep_i]
    sub_sa = [float(sa_c[i]) for i in keep_i]

    rel_out = pot_slot.relax_molecules(sub_m, n_steps=20, chunk_size=48)
    rel_mols: List[Chem.Mol] = []
    rel_sa: List[float] = []
    pairs: List[Tuple[np.ndarray, Chem.Mol]] = []
    for i, rm in enumerate(rel_out["relaxed_mols"]):
        ff = float(rel_out["final_f_max_eV_A"][i])
        if (
            rm is not None
            and bool(rel_out["graph_preserved"][i])
            and math.isfinite(ff)
            and ff <= float(max_fmax)
            and rm.GetNumAtoms() == sub_pos[i].shape[0]
            and is_clean_organic_charge_state(rm)
        ):
            cm_r = Chem.Mol(rm)
            rel_mols.append(cm_r)
            rel_sa.append(sub_sa[i])
            pairs.append((sub_pos[i], cm_r))

    if not rel_mols:
        return [], np.zeros((0, 2), dtype=np.float32), [], {"n_proposals": n_proposals, "n_clean": len(cand_mols), "n_relaxed": 0}

    u_out = cal_unimol2.predict_mols(rel_mols, batch_size=64)
    hp = u_out["mean"][:, cal_unimol2.channel_index("homo")]
    ok = u_out["valid_mask"] & np.isfinite(hp)
    fin_m = [rel_mols[i] for i in range(len(rel_mols)) if bool(ok[i])]
    fin_t = np.asarray([[float(hp[i]), rel_sa[i]] for i in range(len(rel_mols)) if bool(ok[i])], dtype=np.float32)
    fin_p = [pairs[i] for i in range(len(pairs)) if bool(ok[i])]
    stats = {
        "n_proposals": n_proposals,
        "n_clean": len(cand_mols),
        "n_sa_pass": len(sub_m),
        "n_relaxed": len(fin_m),
        "wall_sec": round(time.perf_counter() - t0, 2),
    }
    return fin_m, fin_t, fin_p, stats


def _run_pro_hindsight_awr(
    gen: LoomUni3DARGenerator,
    bus_train: Any,
    cal_unimol2: Any,
    anchor_mols: Sequence[Chem.Mol],
    anchor_targets: np.ndarray,
    clean_targets: np.ndarray,
    *,
    rounds: int = 2,
    proposals_per_round: int = 2048,
    steps_per_round: int = 240,
    batch_size: int = 48,
    sa_target: float = 2.60,
    cfg_scale: float = 2.45,
    device: torch.device,
) -> Tuple[List[Chem.Mol], np.ndarray, List[Tuple[np.ndarray, Chem.Mol]], List[Dict[str, Any]]]:
    """Multi-size (`18..58` atoms) Online Hindsight Equilibrium AWR with HOMO-only CFG + real refiner pair harvest."""
    from loom.generator.molecular_state import mol_to_clean_smiles

    n_layers = len(gen.backbone.decoder.layers)
    top_start = max(0, n_layers - 5)
    gen.requires_grad_(False)
    head_p, top_p = [], []
    for name, p in gen.named_parameters():
        if name.startswith(("conditioner.", "valence_head.", "budget_emb.")):
            p.requires_grad_(True)
            head_p.append(p)
        elif any(name.startswith(f"backbone.decoder.layers.{i}.") for i in range(top_start, n_layers)) or name.startswith(
            ("backbone.norm.", "backbone.tree_heads.", "backbone.atom_heads.0.pred_heads.0.")
        ):
            p.requires_grad_(True)
            top_p.append(p)

    pool_mid = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 42.0)]
    pool_big = clean_targets[(clean_targets[:, 1] > 42.0) & (clean_targets[:, 1] <= 60.0)]
    seen_smiles: set[str] = {mol_to_clean_smiles(m) for m in anchor_mols if mol_to_clean_smiles(m)}
    disc_mols: List[Chem.Mol] = []
    disc_t_list: List[np.ndarray] = []
    all_pairs: List[Tuple[np.ndarray, Chem.Mol]] = []
    round_logs: List[Dict[str, Any]] = []
    c_homo_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

    for r_idx in range(rounds):
        tr0 = time.perf_counter()
        rng = np.random.default_rng(202690 + r_idx * 409)
        n_big = int(round(proposals_per_round * 0.22)) if len(pool_big) > 32 else 0
        n_mid = proposals_per_round - n_big
        p_mid = pool_mid[rng.choice(len(pool_mid), size=n_mid, replace=True)]
        if n_big > 0:
            p_big = pool_big[rng.choice(len(pool_big), size=n_big, replace=True)]
            picked = np.concatenate([p_mid, p_big], axis=0)
            rng.shuffle(picked)
        else:
            picked = p_mid

        req_h = np.clip(picked[:, 0] + rng.normal(0.0, 0.25, size=proposals_per_round), -8.2, -4.8).astype(np.float32)
        req_n = np.clip(np.round(picked[:, 1]), 18.0, 60.0).astype(np.float32)

        new_mols, new_t, new_pairs, col_stats = _collect_pro_onpolicy_pairs(
            gen,
            bus_train,
            cal_unimol2,
            req_h,
            req_n,
            sa_target=sa_target,
            cfg_scale=cfg_scale,
            max_sa_small=3.55,
            max_sa_large=4.15,
            max_fmax=2.20,
            seed=930000 + r_idx * 10000,
            device=device,
        )
        all_pairs.extend(new_pairs)
        added = 0
        for m, trow in zip(new_mols, new_t):
            smi = mol_to_clean_smiles(m)
            if smi and smi not in seen_smiles:
                seen_smiles.add(smi)
                disc_mols.append(m)
                disc_t_list.append(trow)
                added += 1

        comb_mols = list(anchor_mols) + list(disc_mols)
        comb_t = np.concatenate([anchor_targets, np.stack(disc_t_list, axis=0)], axis=0) if disc_t_list else anchor_targets
        n_anc = len(anchor_mols)
        n_disc = len(disc_mols)
        ds = OctreeDataset(comb_mols, gen.backbone, training=True, seed=202695 + r_idx, random_rotation_prob=0.80)

        opt = torch.optim.AdamW(
            [
                {"params": top_p, "lr": 3.5e-5, "weight_decay": 1e-4},
                {"params": head_p, "lr": 1.2e-4, "weight_decay": 1e-4},
            ],
            betas=(0.9, 0.95),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps_per_round, 1), eta_min=5e-6)
        losses_r: List[float] = []

        def _col_batch(sd: int) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
            lrng = np.random.default_rng(sd)
            if n_disc >= 16:
                k_d = min(n_disc, int(round(batch_size * 0.45)))
                k_a = batch_size - k_d
                idx_a = lrng.integers(0, n_anc, size=k_a)
                idx_d = n_anc + lrng.integers(0, n_disc, size=k_d)
                b_idx = np.concatenate([idx_a, idx_d])
            else:
                b_idx = lrng.integers(0, len(comb_mols), size=batch_size)
            return _safe_collate_indices(ds, b_idx.tolist(), rng=lrng)

        with ThreadPoolExecutor(max_workers=4) as pool:
            qn = min(6, steps_per_round)
            futs = [pool.submit(_col_batch, 850000 + r_idx * 5000 + i) for i in range(qn)]
            nxt = qn
            for st in range(1, steps_per_round + 1):
                sl = (st - 1) % len(futs)
                pk, kp = futs[sl].result()
                if nxt < steps_per_round:
                    futs[sl] = pool.submit(_col_batch, 850000 + r_idx * 5000 + nxt)
                    nxt += 1
                if pk is None or kp is None:
                    sched.step()
                    continue
                tn_b = gen.normalize_targets(comb_t[kp])
                ma_b = _sample_4way_cfg_masks(len(kp), rng, device)
                gen.train()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss, _ = gen.compute_sft_loss(
                        pk,
                        tn_b,
                        ma_b,
                        cfg_dropout_prob=0.0,
                        xyz_loss_weight=0.14,
                        valence_loss_weight=0.20,
                        count_loss_weight=0.0,
                        geom_dist_loss_weight=0.0,
                        budget_dropout_prob=0.08,
                        contrast_weight=0.25,
                        contrast_margin=0.12,
                        contrast_min_gap_eV=0.30,
                        contrast_col_mask=c_homo_mask,
                    )
                if torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(top_p + head_p, 1.0)
                    opt.step()
                    losses_r.append(float(loss.detach().item()))
                sched.step()

        r_info = {
            "round": r_idx + 1,
            **col_stats,
            "relaxed_added_unique": added,
            "total_discovered_pool": len(disc_mols),
            "total_refiner_pairs": len(all_pairs),
            "mean_awr_loss": round(float(np.mean(losses_r)), 4) if losses_r else None,
            "round_wall_sec": round(time.perf_counter() - tr0, 2),
        }
        round_logs.append(r_info)
        print(f"[Pro Hindsight-AWR Round {r_idx + 1}/{rounds}] {r_info}", flush=True)

    fin_mols = list(anchor_mols) + list(disc_mols)
    fin_t = np.concatenate([anchor_targets, np.stack(disc_t_list, axis=0)], axis=0) if disc_t_list else anchor_targets
    return fin_mols, fin_t, all_pairs, round_logs


@torch.no_grad()
def _pro_dev_score(
    gen: LoomUni3DARGenerator,
    refiner: Optional[PostTreeGraphRefiner],
    bus_train: Any,
    dev_homo: np.ndarray,
    dev_natoms: np.ndarray,
    sa_target: float,
    cfg_scale: float,
    device: torch.device,
) -> Dict[str, float]:
    """Zero-leakage `256`-prompt dev score using HOMO-only CFG (`0` `audit_dpa4` calls)."""
    gen.eval()
    nb = len(dev_homo)
    cond_2d = np.stack([dev_homo.astype(np.float32), np.full(nb, float(sa_target), dtype=np.float32)], axis=-1)
    tn = gen.normalize_targets(cond_2d)
    cm = torch.ones((nb, 2), dtype=torch.bool, device=device)
    um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device)
    trajs = gen.sample(
        tn,
        cm,
        uncond_active_mask=um,
        cfg_scale=float(cfg_scale),
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        temperatures=(0.85, 0.30, 0.25),
        seed=880555,
        use_bf16=(device.type == "cuda"),
        electron_parity=0,
        leaf_atom_budget=dev_natoms,
        budget_guidance_scale=0.35,
    )
    if refiner is not None:
        refiner.refine_trajectories_batch(trajs, batch_size=256)
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    mols: List[Optional[Chem.Mol]] = []
    for tr in trajs:
        res = reconciler.reconcile_point_cloud(
            tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
        )
        mols.append(res.mol if (res.reconciled_legal and not res.reconciled_exotic_charge and res.mol is not None) else None)
    req_b = np.zeros((nb, len(bus_train.spec.objectives)), dtype=np.float64)
    act_b = np.zeros((nb, len(bus_train.spec.objectives)), dtype=bool)
    h_idx = bus_train.spec.objective_names.index("homo")
    req_b[:, h_idx] = dev_homo
    act_b[:, h_idx] = True
    ev = bus_train.evaluate_molecules(mols, requested_targets=req_b, active_mask=act_b, include_audit=False, for_training=False, batch_size=64)
    leg = ev["legal_mask"] & ev["clean_charge_mask"]
    preds = ev["objective_preds"][:, h_idx]
    ok = leg & np.isfinite(preds)
    mae = float(np.mean(np.abs(preds[ok] - dev_homo[ok]))) if ok.any() else 2.5
    slope = float(np.polyfit(dev_homo[ok], preds[ok], deg=1)[0]) if int(ok.sum()) >= 20 else 0.0
    hits = float(ev["constrained_clean_hits"].mean())
    sa_v = ev["constraint_values"].get("synthesizable_sa", np.full(nb, np.nan))
    sa_ok = float(np.mean(sa_v[leg] <= 3.5)) if leg.any() else 0.0
    fm = ev["f_max_eV_A"]
    fm_med = float(np.nanmedian(fm[leg])) if np.isfinite(fm[leg]).any() else 12.0
    comp = hits + 0.35 * float(leg.mean()) + 0.20 * sa_ok - 0.55 * mae + 0.15 * min(slope, 1.0) - 0.04 * min(fm_med, 12.0)
    return {
        "composite": round(comp, 5),
        "dev_synth_hit_yield": round(hits, 4),
        "dev_surr_mae_eV": round(mae, 4),
        "dev_homo_slope": round(slope, 4),
        "dev_legal_rate": round(float(leg.mean()), 4),
        "dev_sa_ok_rate": round(sa_ok, 4),
        "dev_fmax_median": round(fm_med, 3),
    }


def run_align_pro(args: argparse.Namespace) -> None:
    """Stages B1+B2+B3: Stratified 22k Relaxed Pool -> 2-GPU Full-Param SFT -> Hindsight AWR -> 300-Step HOMO-Dominant GRPO -> XYZ Distill -> Refiner."""
    device = torch.device(args.device)
    pro_dir = args.work_dir / args.run_name
    pro_dir.mkdir(parents=True, exist_ok=True)
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    sft_ckpt = pro_dir / "v4_pro_sft.pt"
    final_ckpt = pro_dir / "v4_pro_final.pt"
    ref_ckpt = pro_dir / "v4_pro_post_tree_refiner.pt"

    pool_sdf, _ = ensure_pro_relaxed_pool(args)

    if not (args.reuse_sft and sft_ckpt.exists()):
        gpu_list = [g.strip() for g in args.rollout_gpus.split(",") if g.strip()]
        n_proc = max(1, len(gpu_list))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={n_proc}",
            "--master_port=29661",
            str(Path(__file__).resolve()),
            "--phase",
            "sft_pro_ddp",
            "--sft-epochs",
            str(args.sft_epochs),
            "--work-dir",
            str(args.work_dir),
            "--run-name",
            str(args.run_name),
            "--step2-dir",
            str(args.step2_dir),
            "--parent-ckpt",
            str(args.parent_ckpt),
        ]
        if args.allow_tf32:
            cmd.append("--allow-tf32")
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            raise RuntimeError(f"sft_pro_ddp failed with code {rc}")

    if args.reuse_grpo and final_ckpt.exists() and ref_ckpt.exists():
        print(f"[Pro Align] Reusing existing {final_ckpt} & {ref_ckpt}", flush=True)
        return

    bus_train, cal_unimol2, _ = _build_calibrated_oe62_bus(
        args, include_blind_auditor=False, calib_cache_path=calib_cache
    )
    gen, sft_payload = LoomUni3DARGenerator.load_checkpoint(sft_ckpt, device=args.device, batch_size=128)
    sft_meta = sft_payload.get("metadata") or {}
    anchor_mols, anchor_targets = _load_prepared_v4_sdf(pool_sdf)
    _, clean_targets = _get_clean_oe62_pool(args, None)

    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    d_rng = np.random.default_rng(202655)
    sub_dev_n = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 46.0), 1]
    dev_h_list, dev_n_list = [], []
    for q in quantiles:
        gh = float(q_points[str(q)])
        picked = d_rng.choice(sub_dev_n, size=36, replace=True)
        dev_h_list.extend([gh] * 36)
        dev_n_list.extend(np.round(picked).tolist())
    dev_homo = np.asarray(dev_h_list, dtype=np.float32)
    dev_natoms = np.asarray(dev_n_list, dtype=np.float32)

    sft_dev = _pro_dev_score(gen, None, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device)
    print(f"[Pro SFT Baseline Dev (no refiner)] {sft_dev}", flush=True)

    t_awr0 = time.perf_counter()
    equil_mols, equil_targets, onpolicy_pairs, hindsight_logs = _run_pro_hindsight_awr(
        gen,
        bus_train,
        cal_unimol2,
        anchor_mols,
        anchor_targets,
        clean_targets,
        rounds=int(args.hindsight_rounds),
        proposals_per_round=int(args.hindsight_proposals),
        steps_per_round=int(args.hindsight_steps),
        batch_size=48,
        sa_target=float(args.eval_sa_target),
        cfg_scale=float(args.cfg_scale),
        device=device,
    )
    hindsight_gpuh = (time.perf_counter() - t_awr0) / 3600.0

    prelim_ref, prelim_ref_meta = _distill_v4_refiner(
        equil_mols, onpolicy_pairs, epochs=int(args.refiner_epochs), batch_size=160, device=device, seed=202656
    )
    print(f"[Pro Prelim Refiner] {prelim_ref_meta}", flush=True)

    base_dev = _pro_dev_score(gen, prelim_ref, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device)
    print(f"[Pro Post-Hindsight + PrelimRef] {base_dev}", flush=True)

    # HOMO-dominant 300-step Tree-Vine GRPO (unfreeze top-5 layers + conditioner + budget_emb, freeze XYZ heads)
    gen.configure_trainable(train_conditioner=True, train_lora=False, unfreeze_full_backbone=False)
    n_layers = len(gen.backbone.decoder.layers)
    for l_idx in range(max(0, n_layers - 5), n_layers):
        gen.backbone.decoder.layers[l_idx].requires_grad_(True)
    gen.backbone.norm.requires_grad_(True)
    gen.backbone.tree_heads.requires_grad_(True)
    for p in gen.backbone.atom_heads[0].pred_heads[0].parameters():
        p.requires_grad_(True)
    n_tr = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"[Pro GRPO] Starting {args.grpo_steps}-step HOMO-dominant Tree-Vine GRPO | trainable={n_tr:,}", flush=True)

    r_rng = np.random.default_rng(202657)
    rep_sel = r_rng.choice(len(equil_mols), size=min(6144, len(equil_mols)), replace=False)
    replay_mols = [equil_mols[int(i)] for i in rep_sel]
    replay_targets = equil_targets[rep_sel]

    ema_gen = deepcopy(gen).eval()
    for p in ema_gen.parameters():
        p.requires_grad_(False)
    best_state = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
    best_dev = base_dev

    trainer = LoomGRPOTrainer(
        generator=gen,
        critic_bus=bus_train,
        replay_molecules=replay_mols,
        replay_targets_eV=replay_targets,
        lr=float(args.grpo_lr),
        weight_decay=1e-4,
        kl_beta=float(args.kl_beta),
        clip_eps=0.20,
        ppo_epochs=1,
        replay_weight=0.22,
        replay_batch_size=16,
        w_valid=0.70,
        w_phys=0.30,
        w_prop=2.60,  # HOMO-DOMINANT!
        tolerance_eV=0.25,
        sharp_concentration=True,
        cfg_scale=float(args.cfg_scale),
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        topology_only_policy=True,
        temperatures=(0.85, 0.30, 0.25),
        electron_parity=0,
        size_anchored_cfg=False,
        cfg_anchor_cols=("sa_score",),  # Native HOMO-only CFG decoupled from prompt_atom_budgets!
        size_adherence_weight=0.30,
        size_tolerance_atoms=2.0,
        use_vine_rollouts=True,
        m_trunks=4,
        k_leaves=2,
        post_tree_refiner=prelim_ref,
        polish_hydrogens_mlip=False,
        seed=202658,
    )

    q_vals = np.array([float(q_points[str(q)]) for q in quantiles], dtype=np.float32)
    pool_mid = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 42.0), 1]
    pool_big = clean_targets[(clean_targets[:, 1] > 42.0) & (clean_targets[:, 1] <= 58.0), 1]
    pps = int(args.prompts_per_step)

    def _sample_pro_prompts(step_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Balanced across all 7 HOMO quantiles + small jitter, 75% 18..42 atoms + 25% 42..58 atoms
        q_pick = rng.choice(q_vals, size=pps, replace=True)
        h_req = np.clip(q_pick + rng.normal(0.0, 0.14, size=pps).astype(np.float32), -8.0, -4.9)
        n_b = max(1, pps // 4)
        n_m = pps - n_b
        ns = np.concatenate([rng.choice(pool_mid, size=n_m, replace=True), rng.choice(pool_big, size=n_b, replace=True)])
        rng.shuffle(ns)
        n_req = np.round(ns).astype(np.float32)
        p_t = np.stack([h_req, np.full(pps, float(args.eval_sa_target), dtype=np.float32)], axis=-1)
        p_m = np.ones((pps, 2), dtype=bool)
        return p_t, p_m, n_req

    def _on_ckpt(step_num: int, cur_gen: Any, hist: List[Dict[str, Any]]) -> None:
        nonlocal best_state, best_dev
        with torch.no_grad():
            for p_ema, p_cur in zip(ema_gen.parameters(), cur_gen.parameters()):
                p_ema.data.mul_(0.992).add_(p_cur.data, alpha=0.008)
            for b_ema, b_cur in zip(ema_gen.buffers(), cur_gen.buffers()):
                b_ema.copy_(b_cur)
        if step_num % 50 == 0 or step_num == int(args.grpo_steps):
            cur_dev = _pro_dev_score(ema_gen, prelim_ref, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device)
            print(f"[Pro GRPO Dev@{step_num}] {cur_dev} (best={best_dev['composite']})", flush=True)
            if cur_dev["composite"] >= best_dev["composite"]:
                best_dev = cur_dev
                best_state = {k: v.detach().cpu().clone() for k, v in ema_gen.state_dict().items()}

    t_g0 = time.perf_counter()
    history = trainer.train_loop(
        total_steps=int(args.grpo_steps),
        prompt_sampler_fn=_sample_pro_prompts,
        group_size=8,
        checkpoint_fn=_on_ckpt,
        log_every=15,
    )
    grpo_gpuh = (time.perf_counter() - t_g0) / 3600.0

    raw_dev = _pro_dev_score(gen, prelim_ref, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device)
    print(f"[Pro GRPO Final Raw Dev] {raw_dev}", flush=True)
    if raw_dev["composite"] > best_dev["composite"]:
        best_dev = raw_dev
        best_state = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}

    gen.load_state_dict(best_state, strict=True)
    gen.eval()

    # Detached SO(3) XYZ Distillation
    pre_xyz_p = pro_dir / "v4_pro_pre_xyz_tmp.pt"
    gen.save_checkpoint(pre_xyz_p)
    pre_xyz_dev = _pro_dev_score(gen, None, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device)
    xyz_meta = _run_v4_detached_xyz_distillation(
        gen, equil_mols, equil_targets, steps=int(args.xyz_distill_steps), batch_size=64, lr=2.5e-4, device=device
    )
    post_xyz_dev = _pro_dev_score(gen, None, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device)
    print(f"[Pro XYZ-Distill Check] pre={pre_xyz_dev} -> post={post_xyz_dev}", flush=True)
    if post_xyz_dev["composite"] < pre_xyz_dev["composite"] - 0.01:
        gen, _ = LoomUni3DARGenerator.load_checkpoint(pre_xyz_p, device=args.device, batch_size=128)
        xyz_meta["reverted"] = True
    else:
        xyz_meta["reverted"] = False
    pre_xyz_p.unlink(missing_ok=True)

    # Final On-Policy 4-Layer Kabsch Refiner across N=18..58
    f_rng = np.random.default_rng(202659)
    n_fin_p = 1536
    n_fb = 384
    n_fm = n_fin_p - n_fb
    req_hf = np.clip(f_rng.choice(q_vals, size=n_fin_p, replace=True) + f_rng.normal(0.0, 0.20, size=n_fin_p), -8.2, -4.8).astype(np.float32)
    req_nf = np.concatenate([f_rng.choice(pool_mid, size=n_fm, replace=True), f_rng.choice(pool_big, size=n_fb, replace=True)])
    f_rng.shuffle(req_nf)
    _, _, post_pairs, post_col_stats = _collect_pro_onpolicy_pairs(
        gen, bus_train, cal_unimol2, req_hf, np.round(req_nf).astype(np.float32),
        sa_target=float(args.eval_sa_target), cfg_scale=float(args.cfg_scale),
        max_sa_small=3.80, max_sa_large=4.30, max_fmax=2.40,
        seed=950000, device=device,
    )
    final_ref, final_ref_meta = _distill_v4_refiner(
        equil_mols, list(onpolicy_pairs) + list(post_pairs),
        epochs=int(args.refiner_epochs), batch_size=160, device=device, seed=202660,
    )
    final_ref_meta["post_grpo_collect_stats"] = post_col_stats
    final_ref.save_checkpoint(ref_ckpt, extra=final_ref_meta)

    final_dev = _pro_dev_score(gen, final_ref, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device)
    gen.save_checkpoint(
        final_ckpt,
        extra_metadata={
            "version": "4.5-pro-final",
            "sft_metadata": sft_meta,
            "sft_dev": sft_dev,
            "hindsight_logs": hindsight_logs,
            "hindsight_awr_gpuh": round(hindsight_gpuh, 4),
            "grpo_gpuh": round(grpo_gpuh, 4),
            "grpo_steps": int(args.grpo_steps),
            "grpo_history_tail": history[-6:],
            "xyz_distill_metadata": xyz_meta,
            "final_refiner_metadata": final_ref_meta,
            "final_dev": final_dev,
        },
    )
    print(f"[Pro ALIGN DONE] Saved {final_ckpt} & {ref_ckpt} | final_dev={final_dev}", flush=True)


def run_pro_rollout_shard(args: argparse.Namespace) -> None:
    """Collect GROUPED rollouts (`prompts_per_shard` x `group_size=8`) post-refiner for Ranking-Loss Critic."""
    device = torch.device(args.device)
    pro_dir = args.work_dir / args.run_name
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    ckpt_p = pro_dir / "v4_pro_final.pt"
    ref_ckpt = pro_dir / "v4_pro_post_tree_refiner.pt"
    shard_out = pro_dir / f"pro_rollout_shard_{args.shard_id}.pt"

    gen, _ = LoomUni3DARGenerator.load_checkpoint(ckpt_p, device=args.device, batch_size=512)
    gen.eval()
    refiner, _ = PostTreeGraphRefiner.load_checkpoint(ref_ckpt, device=args.device)
    refiner.eval()

    bus, _, _ = _build_calibrated_oe62_bus(args, include_blind_auditor=False, calib_cache_path=calib_cache)
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    _, clean_targets = _get_clean_oe62_pool(args, None)

    g_sz = int(args.critic_group_size)
    n_prompts_total = int(args.critic_rollouts) // g_sz
    n_shards = max(1, int(args.num_shards))
    p_per_shard = n_prompts_total // n_shards
    n_shard = p_per_shard * g_sz

    rng = np.random.default_rng(2026700 + int(args.shard_id) * 1009)
    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    q_vals = np.array([float(q_points[str(q)]) for q in (10, 25, 35, 50, 65, 75, 90)], dtype=np.float32)
    pool_mid = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 42.0), 1]
    pool_big = clean_targets[(clean_targets[:, 1] > 42.0) & (clean_targets[:, 1] <= 60.0), 1]

    p_big = int(round(p_per_shard * 0.25))
    p_mid = p_per_shard - p_big
    p_homo = np.clip(rng.choice(q_vals, size=p_per_shard, replace=True) + rng.normal(0.0, 0.22, size=p_per_shard), -8.2, -4.8).astype(np.float32)
    p_nat = np.concatenate([rng.choice(pool_mid, size=p_mid, replace=True), rng.choice(pool_big, size=p_big, replace=True)])
    rng.shuffle(p_nat)
    p_nat = np.clip(np.round(p_nat), 18.0, 60.0).astype(np.float32)

    req_homo = np.repeat(p_homo, g_sz)
    req_natoms = np.repeat(p_nat, g_sz)
    group_ids = np.repeat(np.arange(p_per_shard, dtype=np.int64) + int(args.shard_id) * p_per_shard, g_sz)
    cond_2d = np.stack([req_homo, np.full(n_shard, float(args.eval_sa_target), dtype=np.float32)], axis=-1)

    t0 = time.perf_counter()
    s1_h_l, s1_g_l, s2_h_l, s2_c_l, t_norm_l = [], [], [], [], []
    s2_e_sl, s2_p_sl = [], []
    mols_all: List[Optional[Chem.Mol]] = []
    is_clean = np.zeros(n_shard, dtype=bool)

    chunk = 256  # multiple of g_sz=8
    for s in range(0, n_shard, chunk):
        e = min(s + chunk, n_shard)
        nb = e - s
        tn = gen.normalize_targets(cond_2d[s:e])
        cm = torch.ones((nb, 2), dtype=torch.bool, device=device)
        um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device)
        roll = gen.sample_rollouts_with_two_stage_features(
            tn, cm, uncond_active_mask=um, cfg_scale=float(args.cfg_scale), xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25, geometric_guard=True, temperatures=(0.85, 0.30, 0.25),
            seed=710000 + int(args.shard_id) * 60000 + s, use_bf16=(device.type == "cuda"),
            electron_parity=0, leaf_atom_budget=req_natoms[s:e], budget_guidance_scale=0.35,
        )
        s1_h_l.append(roll["s1_h_concat"])
        s1_g_l.append(roll["s1_geom_feats"])
        s2_h_l.append(roll["s2_h_stage2"])
        s2_c_l.append(roll["s2_cloud_feats"])
        t_norm_l.append(tn.detach().cpu())

        trajs_sub = roll["trajectories"]
        refiner.refine_trajectories_batch(trajs_sub)
        for loc_i, tr in enumerate(trajs_sub):
            s2_e_sl.append(tr["stage2_elem_ids"])
            s2_p_sl.append(tr["stage2_leaf_pos"])
            res = reconciler.reconcile_point_cloud(
                tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
            )
            g_i = s + loc_i
            if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
                mols_all.append(res.mol)
                is_clean[g_i] = True
            else:
                mols_all.append(None)

    pred_asset = bus.registry.get_predictor("ext_unimol2")
    rule_asset = bus.registry.get_predictor("builtin_rules")
    u_out = pred_asset.predict_mols(mols_all, batch_size=64)
    r_out = rule_asset.predict_mols(mols_all, batch_size=256)
    pot_out = bus.potential_slot.evaluate_molecules([m for m in mols_all if m is not None], chunk_size=64) if bus.potential_slot else None
    fmax_all = np.full(n_shard, np.nan, dtype=np.float32)
    if pot_out is not None:
        cp = 0
        for g_i, m in enumerate(mols_all):
            if m is not None:
                fmax_all[g_i] = float(pot_out["f_max_eV_A"][cp])
                cp += 1

    homo_surr = u_out["mean"][:, pred_asset.channel_index("homo")].astype(np.float32)
    sa_vals = r_out["mean"][:, rule_asset.channel_index("sa_score")].astype(np.float32)
    mw_vals = r_out["mean"][:, rule_asset.channel_index("mol_weight")]
    sm_vals = r_out["mean"][:, rule_asset.channel_index("smarts_alert")]
    y_valid = is_clean & u_out["valid_mask"] & np.isfinite(homo_surr) & np.isfinite(sa_vals)
    low_strain = (~np.isfinite(fmax_all)) | (fmax_all <= 4.0)
    is_constr = y_valid & (sa_vals <= 3.5) & (mw_vals <= 550.0) & (sm_vals <= 0.0) & low_strain
    y_raw_2d = np.stack([np.nan_to_num(homo_surr, nan=-6.5), np.nan_to_num(sa_vals, nan=4.0)], axis=-1).astype(np.float32)
    y_surr_norm = gen.normalize_targets(y_raw_2d).detach().cpu()

    wall_s = time.perf_counter() - t0
    torch.save(
        {
            "s1_h_concat": torch.cat(s1_h_l, dim=0),
            "s1_geom_feats": torch.cat(s1_g_l, dim=0),
            "s2_h_stage2": torch.cat(s2_h_l, dim=0),
            "s2_cloud_feats": torch.cat(s2_c_l, dim=0),
            "s2_elem_slices": s2_e_sl,
            "s2_pos_slices": s2_p_sl,
            "targets_norm": torch.cat(t_norm_l, dim=0),
            "y_surrogate_norm": y_surr_norm,
            "y_valid_mask": torch.from_numpy(y_valid),
            "is_legal": torch.from_numpy(is_clean),
            "is_constr_met": torch.from_numpy(is_constr),
            "group_ids": torch.from_numpy(group_ids),
            "fmax_median": float(np.nanmedian(fmax_all)) if np.isfinite(fmax_all).any() else float("nan"),
            "wall_sec": wall_s,
        },
        shard_out,
    )
    print(
        f"[Pro Rollout Shard {args.shard_id}] N={n_shard} ({p_per_shard}x{g_sz}) | clean={is_clean.mean()*100:.1f}% | "
        f"constr+low_strain={is_constr.mean()*100:.1f}% | med_Fmax={np.nanmedian(fmax_all):.2f} | wall={wall_s:.1f}s",
        flush=True,
    )


def run_train_pro_critic(args: argparse.Namespace) -> None:
    """Collect grouped 2-GPU rollouts and train Ranking-Loss `TwoStageOctreeProcessCritic` (`n_gnn_layers=3`)."""
    device = torch.device(args.device)
    pro_dir = args.work_dir / args.run_name
    critic_ckpt = pro_dir / "v4_pro_two_stage_critic.pt"
    if args.reuse_critic and critic_ckpt.exists():
        print(f"[Pro Critic] Reusing {critic_ckpt}", flush=True)
        return

    gpu_list = [g.strip() for g in args.rollout_gpus.split(",") if g.strip()]
    n_shards = len(gpu_list)
    t_r0 = time.perf_counter()
    procs = []
    for sh_i, gpu_id in enumerate(gpu_list):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            "rollout_pro_shard",
            "--device",
            f"cuda:{gpu_id}",
            "--shard-id",
            str(sh_i),
            "--num-shards",
            str(n_shards),
            "--critic-rollouts",
            str(args.critic_rollouts),
            "--critic-group-size",
            str(args.critic_group_size),
            "--eval-sa-target",
            str(args.eval_sa_target),
            "--cfg-scale",
            str(args.cfg_scale),
            "--work-dir",
            str(args.work_dir),
            "--run-name",
            str(args.run_name),
            "--step2-dir",
            str(args.step2_dir),
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
            "--oe62-targets-json",
            str(args.oe62_targets_json),
            "--oe62-reward-oracle-dir",
            str(args.oe62_reward_oracle_dir),
            "--oe62-potential-weights",
            str(args.oe62_potential_weights),
        ]
        if args.allow_tf32:
            cmd.append("--allow-tf32")
        procs.append(subprocess.Popen(cmd, text=True))

    for sh_i, pr in enumerate(procs):
        if pr.wait() != 0:
            raise RuntimeError(f"rollout_pro_shard {sh_i} failed")
    roll_wall = time.perf_counter() - t_r0

    shards = [torch.load(pro_dir / f"pro_rollout_shard_{sh_i}.pt", map_location="cpu", weights_only=False) for sh_i in range(n_shards)]
    s1_h = torch.cat([sh["s1_h_concat"] for sh in shards], dim=0)
    s1_g = torch.cat([sh["s1_geom_feats"] for sh in shards], dim=0)
    s2_h = torch.cat([sh["s2_h_stage2"] for sh in shards], dim=0)
    s2_c = torch.cat([sh["s2_cloud_feats"] for sh in shards], dim=0)
    t_all = torch.cat([sh["targets_norm"] for sh in shards], dim=0)
    y_surr = torch.cat([sh["y_surrogate_norm"] for sh in shards], dim=0)
    y_ok = torch.cat([sh["y_valid_mask"] for sh in shards], dim=0)
    is_leg = torch.cat([sh["is_legal"] for sh in shards], dim=0)
    is_con = torch.cat([sh["is_constr_met"] for sh in shards], dim=0)
    gids = torch.cat([sh["group_ids"] for sh in shards], dim=0)
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

    gen, _ = LoomUni3DARGenerator.load_checkpoint(pro_dir / "v4_pro_final.pt", device="cpu", batch_size=16)
    emb_d = int(gen.args.emb_dim)
    num_atom = int(gen.backbone.num_atom)
    t_mean_np = gen.target_mean.detach().cpu().numpy()
    t_scale_np = gen.target_scale.detach().cpu().numpy()
    del gen

    t_f0 = time.perf_counter()
    act_all = torch.ones((n_tot_r, 2), dtype=torch.bool)
    rk_obj_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

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
        group_ids=gids,
        ranking_weight=0.55,
        ranking_obj_mask=rk_obj_mask,
        epochs=26,
        batch_size=256,
        lr=1.4e-3,
        seed=202671,
    )

    s2_c_mod = OctreeElementProcessCritic(
        emb_dim=emb_d,
        num_targets=2,
        num_atom_classes=num_atom,
        d_node=192,
        n_rbf=24,
        n_gnn_layers=3,
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
        group_ids=gids,
        ranking_weight=0.65,
        ranking_obj_mask=rk_obj_mask,
        epochs=30,
        batch_size=256,
        lr=1.1e-3,
        seed=202672,
    )
    fit_wall = time.perf_counter() - t_f0
    ts = TwoStageOctreeProcessCritic(s1_c, s2_c_mod).to(device)
    ts.eval()
    critic_meta = {
        "total_rollouts": n_tot_r,
        "group_size": int(args.critic_group_size),
        "n_gnn_layers": 3,
        "ranking_weights": {"stage1": 0.55, "stage2": 0.65},
        "rollout_gpuh": round((roll_wall * n_shards + fit_wall) / 3600.0, 4),
        "clean_rate": round(float(is_leg.float().mean().item()), 4),
        "constr_and_low_strain_rate": round(float(is_con.float().mean().item()), 4),
        "rollout_fmax_median_eV_A": round(fmax_med, 3),
        "stage1_stats": st1.__dict__,
        "stage2_stats": st2.__dict__,
    }
    ts.save_checkpoint(critic_ckpt, extra=critic_meta)
    print(f"[Pro Critic DONE] {critic_meta}", flush=True)


def run_eval_pro(args: argparse.Namespace) -> None:
    """Single-GPU uncontended Fair 30.4-Atom & Multi-Atom Bucket evaluation for `Uni-3DAR v4-Pro`."""
    device = torch.device(args.device)
    pro_dir = args.work_dir / args.run_name
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    sdf_dir = pro_dir / "cohorts_sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    gen, gen_payload = LoomUni3DARGenerator.load_checkpoint(pro_dir / "v4_pro_final.pt", device=args.device, batch_size=512)
    gen.eval()
    pro_meta = gen_payload.get("metadata") or {}
    refiner, ref_meta = PostTreeGraphRefiner.load_checkpoint(pro_dir / "v4_pro_post_tree_refiner.pt", device=args.device)
    refiner.eval()
    critic, critic_meta = TwoStageOctreeProcessCritic.load_checkpoint(pro_dir / "v4_pro_two_stage_critic.pt", device=args.device)
    critic.eval()

    bus_eval, cal_unimol2, calib_diag = _build_calibrated_oe62_bus(
        args, include_blind_auditor=True, calib_cache_path=calib_cache
    )
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)

    # Fit safe zero-leakage TwoStage (16->6->2) affine prompt map (`alpha in [0.95, 1.08]`, `0` audit calls!)
    q_points_d = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    d_rng = np.random.default_rng(2026901)
    sub_n_d = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 42.0), 1]
    dev_rows = [
        (float(q_points_d[str(q)]), float(round(ni)))
        for q in (10, 25, 35, 50, 65, 75, 90)
        for ni in d_rng.choice(sub_n_d, size=24, replace=True).tolist()
    ]
    dev_req = np.asarray(dev_rows, dtype=np.float32)  # 168 dev prompts
    rule_asset = bus_eval.registry.get_predictor("builtin_rules")
    reconciler_dev = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    obj_col_dev = torch.tensor([True, False], dtype=torch.bool, device=device)
    best_cfg = {
        "eval_sa_target": float(args.eval_sa_target),
        "cfg_scale": float(args.cfg_scale),
        "homo_only_cfg": True,
        "budget_guidance_scale": 0.35,
        "stage2_diversity_weight": 0.10,
    }
    nb_d = len(dev_req)
    d_mat = np.stack([dev_req[:, 0], np.full(nb_d, best_cfg["eval_sa_target"], dtype=np.float32)], axis=-1)
    d_tn = gen.normalize_targets(d_mat)
    d_cm = torch.ones((nb_d, 2), dtype=torch.bool, device=device)
    d_um = torch.tensor([[False, True]] * nb_d, dtype=torch.bool, device=device)
    d_grps = gen.sample_with_two_stage_process_critic(
        d_tn, d_cm, critic, m_stage1=16, m_stage2=6, m_keep=2, objective_col_mask=obj_col_dev,
        stage1_legal_weight=0.80, stage1_constr_weight=0.55, stage2_legal_weight=1.05, stage2_constr_weight=0.70,
        stage2_diversity_weight=best_cfg["stage2_diversity_weight"], coarse_chunk_size=1024,
        cfg_scale=best_cfg["cfg_scale"], xyz_cfg_scale=1.0, cfg_tree_decay=0.25, geometric_guard=True,
        temperatures=(0.85, 0.30, 0.25), seed=884499, use_bf16=True, electron_parity=0,
        uncond_active_mask=d_um, leaf_atom_budget=dev_req[:, 1],
        budget_guidance_scale=best_cfg["budget_guidance_scale"],
    )
    refiner.refine_trajectories_batch([tr for g in d_grps for tr in g], batch_size=256)
    dg_m: List[List[Optional[Chem.Mol]]] = [[None, None] for _ in range(nb_d)]
    dg_l = np.zeros((nb_d, 2), dtype=bool)
    dg_x = np.zeros((nb_d, 2), dtype=bool)
    dg_d = np.full((nb_d, 2), np.nan, dtype=np.float32)
    for i in range(nb_d):
        for k in range(2):
            tr = d_grps[i][k]
            dg_d[i, k] = float(tr.get("refiner_disp_rms_A", np.nan))
            res = reconciler_dev.reconcile_point_cloud(
                tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
            )
            if res.reconciled_legal and res.mol is not None:
                dg_m[i][k] = res.mol
                dg_l[i, k] = True
                dg_x[i, k] = bool(res.reconciled_exotic_charge)
    sel_dm, sel_dl, sel_dx, _ = _select_v4_step1_from_pool(
        dg_m, dg_l, dg_x, dev_req, bus_eval, 2, use_surrogate=True, device=device,
        sd_weight=0.25, disp_rms_grid=dg_d, disp_penalty_weight=0.35,
    )
    du_out = cal_unimol2.predict_mols(sel_dm, batch_size=64)
    dr_out = rule_asset.predict_mols(sel_dm, batch_size=256)
    d_hp = du_out["mean"][:, cal_unimol2.channel_index("homo")]
    d_sa = dr_out["mean"][:, rule_asset.channel_index("sa_score")]
    d_ok = sel_dl & (~sel_dx) & du_out["valid_mask"] & np.isfinite(d_hp) & (d_sa <= 3.6)
    h0 = -6.60
    if int(d_ok.sum()) >= 40:
        xs_d = dev_req[d_ok, 0].astype(np.float64)
        ys_d = d_hp[d_ok].astype(np.float64)
        sl_d, bias_d = np.polyfit(xs_d - h0, ys_d - h0, deg=1)
        sl_cl = float(np.clip(sl_d, 0.78, 1.10))
        alpha_d = float(np.clip(1.0 + 0.50 * ((1.0 / sl_cl) - 1.0), 0.95, 1.08))
        beta_d = float(np.clip(-0.85 * bias_d / sl_cl, -0.35, 0.35))
        dev_ts_mae = float(np.mean(np.abs(ys_d - xs_d)))
    else:
        sl_d, bias_d, alpha_d, beta_d, dev_ts_mae = 1.0, 0.0, 1.0, 0.0, 9.9
    best_pmap = {
        "h_center_eV": h0,
        "fit_mode": "TwoStage_16_to_6_to_2",
        "n_valid": int(d_ok.sum()),
        "dev_twostage_mae_eV": round(dev_ts_mae, 4),
        "dev_twostage_slope": round(float(sl_d), 4),
        "dev_twostage_bias_eV": round(float(bias_d), 4),
        "prompt_alpha": round(alpha_d, 4),
        "prompt_beta_eV": round(beta_d, 4),
    }
    sweep_logs = [best_pmap]
    print(f"[Pro Matched TwoStage Dev Calibration] cfg={best_cfg} | pmap={best_pmap}", flush=True)

    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    obj_col_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

    def _encode(homo_1d: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        nb = len(homo_1d)
        h_m = _apply_affine_prompt_map(homo_1d, best_pmap)
        mat = np.stack([h_m, np.full(nb, float(best_cfg["eval_sa_target"]), dtype=np.float32)], axis=-1)
        tn = gen.normalize_targets(mat)
        ma = torch.ones((nb, 2), dtype=torch.bool, device=device)
        um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device) if best_cfg["homo_only_cfg"] else None
        return tn, ma, um

    wt, wm, wu = _encode(np.full(16, -6.4, dtype=np.float32))
    wg = gen.sample_with_two_stage_process_critic(
        wt, wm, critic, m_stage1=8, m_stage2=4, m_keep=2, objective_col_mask=obj_col_mask,
        stage2_diversity_weight=0.10, cfg_scale=float(best_cfg["cfg_scale"]), cfg_tree_decay=0.25,
        geometric_guard=True, seed=29, use_bf16=True, uncond_active_mask=wu,
        leaf_atom_budget=np.full(16, 30.0, dtype=np.float32),
        budget_guidance_scale=float(best_cfg["budget_guidance_scale"]),
    )
    refiner.refine_trajectories_batch([tr for g in wg for tr in g])
    _sync(device)

    sft_gpuh = float((pro_meta.get("sft_metadata") or {}).get("sft_total_gpuh", 0.0))
    awr_gpuh = float(pro_meta.get("hindsight_awr_gpuh", 0.0))
    grpo_gpuh = float(pro_meta.get("grpo_gpuh", 0.0))
    xyz_gpuh = float((pro_meta.get("xyz_distill_metadata") or {}).get("xyz_distill_gpuh", 0.0))
    ref_gpuh = float(ref_meta.get("refiner_gpuh", 0.0))
    crit_gpuh = float(critic_meta.get("rollout_gpuh", 0.0))
    total_gpuh = round(sft_gpuh + awr_gpuh + grpo_gpuh + xyz_gpuh + ref_gpuh + crit_gpuh, 4)

    if args.eval_part in ("all", "fair30"):
        req_30 = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
        n_p30 = len(req_30)
        funnels_30 = [
            ("Outcome_M1", 1, 1, 1, False),
            ("TwoStage_8_to_4_to_1", 8, 4, 1, False),
            ("TwoStage_8_to_4_to_2", 8, 4, 2, True),
            ("TwoStage_16_to_6_to_2", 16, 6, 2, True),
            ("TwoStage_24_to_8_to_2", 24, 8, 2, True),
            ("TwoStage_32_to_8_to_2", 32, 8, 2, True),
            ("TwoStage_32_to_8_to_3", 32, 8, 3, True),
            ("TwoStage_32_to_8_to_4", 32, 8, 4, True),
            ("TwoStage_48_to_12_to_4", 48, 12, 4, True),
        ]
        arms_30: Dict[str, Any] = {}
        for arm_name, ms1, ms2, mkeep, use_surr in funnels_30:
            _sync(device)
            tg0 = time.perf_counter()
            groups_all: List[List[Dict[str, Any]]] = []
            p_chunk = 512 if ms1 == 1 else (128 if ms1 >= 48 else 256)
            for s in range(0, n_p30, p_chunk):
                e = min(s + p_chunk, n_p30)
                bt, bm, bu = _encode(req_30[s:e, 0])
                groups_all.extend(
                    gen.sample_with_two_stage_process_critic(
                        bt, bm, critic, m_stage1=ms1, m_stage2=ms2, m_keep=mkeep,
                        objective_col_mask=obj_col_mask, stage1_legal_weight=0.80, stage1_constr_weight=0.55,
                        stage2_legal_weight=1.05, stage2_constr_weight=0.70, stage2_diversity_weight=0.10,
                        coarse_chunk_size=1024, cfg_scale=float(best_cfg["cfg_scale"]), xyz_cfg_scale=1.0,
                        cfg_tree_decay=0.25, geometric_guard=True, temperatures=(0.85, 0.30, 0.25),
                        seed=990000 + ms1 * 409 + ms2 * 61 + mkeep * 7 + s, use_bf16=True, electron_parity=0,
                        uncond_active_mask=bu, leaf_atom_budget=req_30[s:e, 1],
                        budget_guidance_scale=float(best_cfg["budget_guidance_scale"]),
                    )
                )
            refiner.refine_trajectories_batch([tr for g in groups_all for tr in g], batch_size=256)
            _sync(device)
            gen_w = time.perf_counter() - tg0

            tr0 = time.perf_counter()
            p_grid: List[List[Optional[Chem.Mol]]] = [[None] * mkeep for _ in range(n_p30)]
            p_leg = np.zeros((n_p30, mkeep), dtype=bool)
            p_exo = np.zeros((n_p30, mkeep), dtype=bool)
            p_disp = np.full((n_p30, mkeep), np.nan, dtype=np.float32)
            for i in range(n_p30):
                for k in range(mkeep):
                    tr = groups_all[i][k]
                    p_disp[i, k] = float(tr.get("refiner_disp_rms_A", np.nan))
                    res = reconciler.reconcile_point_cloud(
                        tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
                    )
                    if res.reconciled_legal and res.mol is not None:
                        p_grid[i][k] = res.mol
                        p_leg[i, k] = True
                        p_exo[i, k] = bool(res.reconciled_exotic_charge)
            rec_w = time.perf_counter() - tr0

            sel_m, sel_l, sel_x, surr_w = _select_v4_step1_from_pool(
                p_grid, p_leg, p_exo, req_30, bus_eval, mkeep, use_surrogate=use_surr, device=device,
                sd_weight=0.30, disp_rms_grid=p_disp, disp_penalty_weight=0.40,
            )
            arm_res = _score_oe62_size_cohort(
                sel_m, sel_l, sel_x, req_30, bus_eval, source_graphs, source_scaffolds, gen_w, rec_w, surr_w,
                export_sdf_path=(sdf_dir / f"{arm_name}.sdf"),
            )
            arm_res["funnel"] = {"m_stage1": ms1, "m_stage2": ms2, "m_keep": mkeep, "use_surrogate": use_surr}
            arms_30[arm_name] = arm_res
            print(
                f"[Pro FINAL 30-Atom {arm_name}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
                f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
                f"Legal={arm_res['legal_rate']*100:.1f}% SA_ok={arm_res['sa_leq_3_5_rate_among_legal']*100:.1f}% Fmax={arm_res['zero_step_f_max_median_eV_A']:.2f} | "
                f"RewSynth={arm_res['synthesizable_clean_hit_yield']*100:.1f}% | "
                f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} ({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
                f"BlindMAE={arm_res['audit_mae_eV']:.4f} eV",
                flush=True,
            )

        fair_payload = {
            "experiment": "OE62 Fair Fixed-Prompt 30.4-Atom Benchmark — Uni-Loom [Uni-3DAR v4-Pro (<3 GPU-h): 22k Multi-Size DPA-4-Equil Full-Param SFT + 300-Step HOMO-Dominant Tree-Vine GRPO + Grouped Ranking 3-Layer GNN TwoStage Critic + Damped Affine HOMO-CFG + MMR]",
            "benchmark_seed": 20261118,
            "n_prompts": n_p30,
            "mean_requested_atoms": round(float(np.mean(req_30[:, 1])), 2),
            "total_training_gpuh": total_gpuh,
            "gpuh_breakdown": {
                "pro_relaxed_sft_gpuh": round(sft_gpuh, 4),
                "pro_hindsight_awr_gpuh": round(awr_gpuh, 4),
                "pro_homo_grpo_gpuh": round(grpo_gpuh, 4),
                "pro_xyz_distill_gpuh": round(xyz_gpuh, 4),
                "pro_refiner_gpuh": round(ref_gpuh, 4),
                "pro_ranking_critic_gpuh": round(crit_gpuh, 4),
            },
            "selected_inference_cfg": best_cfg,
            "damped_affine_prompt_map": best_pmap,
            "dev_sweep_logs": sweep_logs,
            "zero_leakage_dft_calibration": calib_diag,
            "pro_metadata": pro_meta,
            "two_stage_critic_meta": critic_meta,
            "arms": arms_30,
        }
        args.fair_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.fair_output_json.write_text(json.dumps(fair_payload, indent=2) + "\n")
        print(f"[Pro Fair 30-Atom SAVED] {args.fair_output_json} (Total GPU-h={total_gpuh})", flush=True)

    if args.eval_part in ("all", "buckets"):
        q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
        quantiles = (10, 25, 35, 50, 65, 75, 90)
        bucket_results: Dict[str, Any] = {}
        for n_exact in (20, 30, 40, 50, 60):
            req_b = np.asarray([(float(q_points[str(q)]), float(n_exact)) for q in quantiles for _ in range(40)], dtype=np.float32)
            nb = len(req_b)
            bt, bm, bu = _encode(req_b[:, 0])
            b_arms: Dict[str, Any] = {}
            for b_tag, ms1, ms2, mk, use_s in (
                ("Uni3DAR_v4_M1", 1, 1, 1, False),
                ("Uni3DAR_v4_TwoStage_16_to_6_to_2", 16, 6, 2, True),
                ("Uni3DAR_v4_TwoStage_32_to_8_to_3", 32, 8, 3, True),
                ("Uni3DAR_v4_TwoStage_32_to_8_to_4", 32, 8, 4, True),
            ):
                _sync(device)
                t0 = time.perf_counter()
                grps = gen.sample_with_two_stage_process_critic(
                    bt, bm, critic, m_stage1=ms1, m_stage2=ms2, m_keep=mk,
                    objective_col_mask=obj_col_mask, stage2_diversity_weight=0.10, coarse_chunk_size=1024,
                    cfg_scale=float(best_cfg["cfg_scale"]), xyz_cfg_scale=1.0, cfg_tree_decay=0.25,
                    geometric_guard=True, seed=73000 + ms1 * 43 + mk * 11 + n_exact, use_bf16=True,
                    electron_parity=0, uncond_active_mask=bu, leaf_atom_budget=req_b[:, 1],
                    budget_guidance_scale=float(best_cfg["budget_guidance_scale"]),
                )
                refiner.refine_trajectories_batch([tr for g in grps for tr in g], batch_size=256)
                _sync(device)
                gw = time.perf_counter() - t0

                r0 = time.perf_counter()
                pg_b: List[List[Optional[Chem.Mol]]] = [[None] * mk for _ in range(nb)]
                pl_b = np.zeros((nb, mk), dtype=bool)
                px_b = np.zeros((nb, mk), dtype=bool)
                pd_b = np.full((nb, mk), np.nan, dtype=np.float32)
                for i in range(nb):
                    for k in range(mk):
                        tr = grps[i][k]
                        pd_b[i, k] = float(tr.get("refiner_disp_rms_A", np.nan))
                        res = reconciler.reconcile_point_cloud(
                            tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
                        )
                        if res.reconciled_legal and res.mol is not None:
                            pg_b[i][k] = res.mol
                            pl_b[i, k] = True
                            px_b[i, k] = bool(res.reconciled_exotic_charge)
                rw = time.perf_counter() - r0

                sm_b, sl_b, sx_b, sw_b = _select_v4_step1_from_pool(
                    pg_b, pl_b, px_b, req_b, bus_eval, mk, use_surrogate=use_s, device=device,
                    sd_weight=0.30, disp_rms_grid=pd_b, disp_penalty_weight=0.40,
                )
                b_res = _score_oe62_size_cohort(sm_b, sl_b, sx_b, req_b, bus_eval, source_graphs, source_scaffolds, gw, rw, sw_b)
                b_arms[b_tag] = b_res
                print(
                    f"[Pro Bucket N={n_exact} {b_tag}] Eff={b_res['effective_output_mol_per_sec']:.2f} mol/s | "
                    f"Atoms={b_res['mean_total_atoms']} (err={b_res['mean_natoms_abs_err']}) | "
                    f"Legal={b_res['legal_rate']*100:.1f}% Fmax={b_res['zero_step_f_max_median_eV_A']:.2f} | "
                    f"BlindSynth={b_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}% | "
                    f"BlindMAE={b_res['audit_mae_eV']:.4f} eV",
                    flush=True,
                )
            bucket_results[str(n_exact)] = {"target_n_atoms": n_exact, "n_prompts": nb, "arms": b_arms}

        buckets_payload = {
            "experiment": "OE62 Controlled Multi-Atom-Count Scaling (N in {20, 30, 40, 50, 60}) — Uni-Loom [Uni-3DAR v4-Pro]",
            "total_training_gpuh": total_gpuh,
            "buckets": bucket_results,
        }
        args.buckets_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.buckets_output_json.write_text(json.dumps(buckets_payload, indent=2) + "\n")
        print(f"[Pro Buckets ALL DONE] Wrote {args.buckets_output_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["build_relax_pool", "relax_pro_shard", "sft_pro_ddp", "align_pro", "rollout_pro_shard", "train_critic", "train_all", "eval_pro", "all"],
        required=True,
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rollout-gpus", type=str, default="0,1")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--relax-pool-cap", type=int, default=22000)
    p.add_argument("--small-max-sa", type=float, default=3.55)
    p.add_argument("--large-max-sa", type=float, default=4.20)
    p.add_argument("--sft-epochs", type=int, default=10)
    p.add_argument("--hindsight-rounds", type=int, default=2)
    p.add_argument("--hindsight-proposals", type=int, default=2048)
    p.add_argument("--hindsight-steps", type=int, default=240)
    p.add_argument("--grpo-steps", type=int, default=300)
    p.add_argument("--grpo-lr", type=float, default=1.2e-5)
    p.add_argument("--kl-beta", type=float, default=0.025)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--xyz-distill-steps", type=int, default=600)
    p.add_argument("--refiner-epochs", type=int, default=14)
    p.add_argument("--critic-rollouts", type=int, default=24576)
    p.add_argument("--critic-group-size", type=int, default=8)
    p.add_argument("--eval-sa-target", type=float, default=2.25)
    p.add_argument("--cfg-scale", type=float, default=2.35)
    p.add_argument("--eval-part", choices=["all", "fair30", "buckets"], default="all")
    p.add_argument("--reuse-relaxed-pool", action="store_true")
    p.add_argument("--reuse-sft", action="store_true")
    p.add_argument("--reuse-grpo", action="store_true")
    p.add_argument("--reuse-critic", action="store_true")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="uni3dar_v4_pro")
    p.add_argument("--step2-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/uni3dar_v4_step2"))
    p.add_argument("--parent-ckpt", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_e13_e14_20261113/full_homo_27000/weights/octree_conditional.pt"))
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument("--oe62-master-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"))
    p.add_argument("--oe62-master-valid-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_valid_B.sdf"))
    p.add_argument("--oe62-source-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/source_labels/train_source.sdf"))
    p.add_argument("--oe62-replay-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_B.sdf"))
    p.add_argument("--quantum-train-dft-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_dft.sdf"))
    p.add_argument("--quantum-valid-dft-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_valid_dft.sdf"))
    p.add_argument("--oe62-targets-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/targets_B.json"))
    p.add_argument("--oe62-reward-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"))
    p.add_argument("--oe62-audit-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_e15_calibration_20261114/full_fit/oracle"))
    p.add_argument("--oe62-potential-weights", type=Path, default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"))
    p.add_argument("--fair-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_pro_fair_30atom.json"))
    p.add_argument("--buckets-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_pro_atom_buckets.json"))
    args = p.parse_args()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.phase == "build_relax_pool":
        ensure_pro_relaxed_pool(args)
    elif args.phase == "relax_pro_shard":
        run_relax_pro_shard(args)
    elif args.phase == "sft_pro_ddp":
        run_sft_pro_ddp_worker(args)
    elif args.phase == "align_pro":
        run_align_pro(args)
    elif args.phase == "rollout_pro_shard":
        run_pro_rollout_shard(args)
    elif args.phase == "train_critic":
        run_train_pro_critic(args)
    elif args.phase == "train_all":
        run_align_pro(args)
        run_train_pro_critic(args)
    elif args.phase == "eval_pro":
        run_eval_pro(args)
    else:
        run_align_pro(args)
        run_train_pro_critic(args)
        run_eval_pro(args)


if __name__ == "__main__":
    main()
