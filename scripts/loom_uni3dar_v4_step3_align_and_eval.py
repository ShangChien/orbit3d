"""Step 3 of `Uni-3DAR v4 + Uni-Loom` Speed-Preserving Overtake Plan (`~1.5 GPU-h` Step 3, `~7.0 GPU-h` total).

Executes four synergistic sub-stages seamlessly:
  1. `sft_synth` (`~0.28 GPU-h`, 2-GPU DDP via `torchrun` `8` epochs):
     Adapts the universal Step-2 `v4_step2_ema_best.pt` foundation checkpoint onto the synthesizable
     (`SA_Score <= 3.25`, `smarts_alert == 0`) equilibrium OE62 conformer manifold with pure discrete
     `0.01 A` sub-voxel CE (`xyz_loss_weight = 0.35`, `geom_dist_loss_weight = 0.0`) and balanced CFG
     dropout (`65% [T,T], 15% [T,F], 20% [F,F]`), eliminating high-SA crystal-cage / sub-voxel bimodal
     strain while preserving full SO(3) generality and exact `Delta N == 0` `ScaleFreeCountHead`.
  2. `distill_refiner` (`~0.08 GPU-h`):
     Distills `PostTreeGraphRefiner` + `BilinearBondHead` on synthesizable equilibrium conformers perturbed
     by octree Morton-seam noise (`< 0.5 ms` batched SE(3)-equivariant pass, ZERO iterative MLIP loops).
  3. `align_vine_grpo` (`~0.55 GPU-h`):
     Runs `180` steps of online Tree-Vine GRPO (`topology_only_policy=True`, `m_trunks=4, k_leaves=2`,
     `uncond_active_mask=None` so CFG guides BOTH `homo` and `sa_score=1.85`, counterfactual trunk vs leaf
     advantages, EMA shadow weights) under zero-leakage `ResidualCalibratedAsset(ext_unimol2)` + `DPA-4`
     zero-step `F_max` + `SA <= 3.5` (`0` `audit_dpa4` calls!).
  4. `eval_final` (`1` uncontended GPU `cuda:0` for wall-clock timing):
     Fits Step-3 `TwoStageOctreeProcessCritic` (`16,384` rollouts across `GPU 0,1`), measures optional
     zero-leakage mean-bias prompt centering (`0` `audit_dpa4` calls), and runs:
       - Fair Fixed-Per-Prompt `30.4`-Atom OE62 Benchmark (`N = 1,792`, seed `20261118`):
         `Outcome_M1`, `TwoStage_8_to_4_to_1`, `TwoStage_8_to_4_to_2`, `TwoStage_16_to_6_to_2`,
         `TwoStage_24_to_8_to_2`, `TwoStage_32_to_8_to_2`
       - Multi-Atom-Count Bucket Scaling (`N in {20, 30, 40, 50, 60}`, `280` prompts/bucket)
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MOLGEN_COMPAT_DEFAULT = Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat")
for cand in (ROOT / "src" / "compat", ROOT / "src", ROOT, MOLGEN_COMPAT_DEFAULT):
    cstr = str(cand)
    if cand.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)

from loom.chemistry.valence_reconciler import ValenceReconciler  # noqa: E402
from loom.weaving import (  # noqa: E402
    BuiltInRuleAsset,
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
from scripts.loom_uni3dar_v4_step2_train import (  # noqa: E402
    _CollateStep2Octree,
    _Step2OctreeTorchDataset,
    _load_prepared_v4_sdf,
)


def _load_synthesizable_v4_subset(sdf_path: Path, max_sa: float = 3.25) -> Tuple[List[Chem.Mol], np.ndarray]:
    """Filter `oe62_v4_train_eq.sdf` purely by deterministic RDKit `SA_Score <= max_sa` & `smarts_alert == 0`."""
    all_mols, all_t = _load_prepared_v4_sdf(sdf_path)
    rule_asset = BuiltInRuleAsset(asset_id="synth_filter")
    r_out = rule_asset.predict_mols(all_mols, batch_size=512)
    sm_idx = rule_asset.channel_index("smarts_alert")
    sm_alerts = r_out["mean"][:, sm_idx]
    keep = (all_t[:, 1] <= float(max_sa)) & r_out["valid_mask"] & (sm_alerts <= 0.0)
    kept_mols = [all_mols[i] for i in range(len(all_mols)) if bool(keep[i])]
    kept_t = all_t[keep]
    return kept_mols, kept_t


def _sample_synth_sft_masks(batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    """Balanced CFG mask supporting full `[F,F]` unconditional guidance (`65% [T,T], 15% [T,F], 20% [F,F]`)."""
    u = rng.random(size=batch_size)
    mask = np.ones((batch_size, 2), dtype=bool)
    mask[(u >= 0.65) & (u < 0.80), 1] = False
    mask[u >= 0.80] = False
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def run_sft_synth_ddp_worker(args: argparse.Namespace) -> None:
    """2-GPU DDP synthesizable low-strain manifold refinement (`8` epochs, `0.28 GPU-h`)."""
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device)

    step3_dir = args.work_dir / args.run_name
    step3_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = step3_dir / "v4_step3_synth_sft.pt"

    train_sdf = args.step2_dir / "oe62_v4_train_eq.sdf"
    synth_mols, synth_targets = _load_synthesizable_v4_subset(train_sdf, max_sa=float(args.synth_max_sa))
    if rank == 0:
        print(
            f"[Step3 Synth-SFT] Loaded {len(synth_mols)} synthesizable equilibrium conformers (SA <= {args.synth_max_sa})",
            flush=True,
        )

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
        lora_rank=32,
        lora_alpha=64.0,
        lora_dropout=0.05,
        lora_include_ffn=True,
        lora_include_heads=True,
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
        train_lora=True,
        train_xyz_embeddings=False,
        unfreeze_full_backbone=False,
    )
    step2_meta = {"init_source": "parent_ckpt_npref0_lora_r32_so3_non_trapping_budget"}

    ema_gen = deepcopy(gen).eval()
    for p in ema_gen.parameters():
        p.requires_grad_(False)

    ds = _Step2OctreeTorchDataset(synth_mols, synth_targets, gen.backbone, training=True, seed=202630)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=202630) if is_ddp else None
    loader = DataLoader(
        ds,
        batch_size=64,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        collate_fn=_CollateStep2Octree(ds.octree_ds.grid.collater),
        pin_memory=True,
        drop_last=True,
    )

    head_prefixes = ("conditioner.", "valence_head.", "budget_emb.", "scale_free_count_head.")
    backbone_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []
    for name, p in gen.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(head_prefixes):
            head_params.append(p)
        else:
            backbone_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1.8e-4, "weight_decay": 1e-4},
            {"params": head_params, "lr": 3.0e-4, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.95),
    )
    n_ep = int(args.synth_sft_epochs)
    total_steps = max(1, len(loader) * n_ep)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=2.0e-6)
    contrast_col_mask = torch.tensor([True, True], dtype=torch.bool, device=device)

    t0 = time.perf_counter()
    g_step = 0
    for ep in range(1, n_ep + 1):
        ep_t0 = time.perf_counter()
        ds.set_epoch(ep)
        if sampler is not None:
            sampler.set_epoch(ep)
        gen.train()
        rng_ep = np.random.default_rng(202640 + ep * 97 + rank * 13)
        l_sum, xyz_sum, cnt_sum, nb = 0.0, 0.0, 0.0, 0
        for packed, raw_t in loader:
            t_norm = gen.normalize_targets(raw_t.to(device, non_blocking=True))
            act_m = _sample_synth_sft_masks(int(raw_t.shape[0]), rng_ep, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, info = gen.compute_sft_loss(
                    packed,
                    t_norm,
                    act_m,
                    cfg_dropout_prob=0.0,
                    contrast_weight=0.08,
                    contrast_margin=0.06,
                    contrast_min_gap_eV=0.35,
                    contrast_col_mask=contrast_col_mask,
                    xyz_loss_weight=0.15,
                    valence_loss_weight=0.15,
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
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.2)
            opt.step()
            sched.step()
            g_step += 1
            with torch.no_grad():
                eff_ema = min(0.996, (1.0 + g_step) / (10.0 + g_step))
                for pe, pc in zip(ema_gen.parameters(), gen.parameters()):
                    pe.lerp_(pc, 1.0 - eff_ema)
            l_sum += float(info["loss_total"])
            xyz_sum += float(info["loss_xyz"])
            cnt_sum += float(info["loss_count"])
            nb += 1
        if rank == 0:
            print(
                f"[Step3 Synth-SFT Ep {ep:02d}/{n_ep:02d}] loss={l_sum/max(nb,1):.4f} "
                f"(xyz={xyz_sum/max(nb,1):.4f}, cnt={cnt_sum/max(nb,1):.4f}) wall={time.perf_counter()-ep_t0:.1f}s",
                flush=True,
            )

    synth_gpuh = ((time.perf_counter() - t0) * world_size) / 3600.0
    if rank == 0:
        ema_gen.save_checkpoint(
            out_ckpt,
            extra_metadata={
                "step2_metadata": step2_meta,
                "synth_sft_epochs": n_ep,
                "synth_sft_mols": len(synth_mols),
                "synth_sft_gpuh": round(synth_gpuh, 4),
            },
        )
        print(f"[Step3 Synth-SFT DONE] gpuh={synth_gpuh:.4f} -> {out_ckpt}", flush=True)
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


def ensure_sft_synth(args: argparse.Namespace) -> Path:
    """Launch 2-GPU `torchrun` for `sft_synth_ddp` if `v4_step3_synth_sft.pt` is not yet built."""
    step3_dir = args.work_dir / args.run_name
    step3_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = step3_dir / "v4_step3_synth_sft.pt"
    if args.reuse_synth_sft and out_ckpt.exists():
        print(f"[Step3 Synth-SFT] Reusing {out_ckpt}", flush=True)
        return out_ckpt

    gpu_list = [g.strip() for g in args.rollout_gpus.split(",") if g.strip()]
    nproc = max(1, len(gpu_list))
    torchrun_bin = Path(sys.executable).parent / "torchrun"
    if nproc >= 2 and torchrun_bin.exists():
        cmd = [
            str(torchrun_bin),
            f"--nproc_per_node={nproc}",
            "--master_port=29523",
            str(Path(__file__).resolve()),
            "--phase",
            "sft_synth_ddp",
            "--synth-sft-epochs",
            str(args.synth_sft_epochs),
            "--synth-max-sa",
            str(args.synth_max_sa),
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
        subprocess.run(cmd, check=True)
    else:
        run_sft_synth_ddp_worker(args)
    return out_ckpt


def run_distill_refiner(args: argparse.Namespace) -> Path:
    """Distill `PostTreeGraphRefiner` + `BilinearBondHead` on synthesizable equilibrium conformers."""
    device = torch.device(args.device)
    step3_dir = args.work_dir / args.run_name
    step3_dir.mkdir(parents=True, exist_ok=True)
    ref_ckpt = step3_dir / "v4_step3_post_tree_refiner.pt"
    if args.reuse_refiner and ref_ckpt.exists():
        print(f"[Step3 Refiner] Reusing {ref_ckpt}", flush=True)
        return ref_ckpt

    train_sdf = args.step2_dir / "oe62_v4_train_eq.sdf"
    synth_mols, _ = _load_synthesizable_v4_subset(train_sdf, max_sa=3.40)
    t0 = time.perf_counter()
    refiner = PostTreeGraphRefiner(
        num_atom_classes=128,
        d_model=160,
        n_heads=4,
        n_layers=2,
        n_rbf=24,
        d_pair=128,
        max_disp_A=0.08,
        dropout=0.05,
    ).to(device)
    stats = fit_post_tree_graph_refiner(
        refiner,
        synth_mols,
        epochs=int(args.refiner_epochs),
        batch_size=int(args.refiner_batch_size),
        lr=1.5e-3,
        seed=202631,
    )
    wall = time.perf_counter() - t0
    meta = {
        **stats.__dict__,
        "refiner_wall_sec": round(wall, 2),
        "refiner_gpuh": round(wall / 3600.0, 4),
    }
    refiner.save_checkpoint(ref_ckpt, extra=meta)
    print(f"[Step3 Refiner DONE] {meta} -> {ref_ckpt}", flush=True)
    return ref_ckpt


def _configure_v4_grpo_trainable(gen: LoomUni3DARGenerator, top_k_layers: int = 5) -> int:
    """Enable Rank-32 LoRA + conditioner + valence/budget embeddings for Tree-Vine GRPO while keeping 0.01 A XYZ heads frozen."""
    has_lora = int(gen.loom_config.get("lora_rank", 0)) > 0
    if has_lora:
        gen.configure_trainable(
            train_conditioner=True,
            train_lora=True,
            train_xyz_embeddings=False,
            unfreeze_full_backbone=False,
        )
    else:
        gen.configure_trainable(train_conditioner=True, train_lora=False, unfreeze_full_backbone=False)
        n_layers = len(gen.backbone.decoder.layers)
        k = max(0, min(int(top_k_layers), n_layers))
        for l_idx in range(n_layers - k, n_layers):
            gen.backbone.decoder.layers[l_idx].requires_grad_(True)
    gen.backbone.norm.requires_grad_(True)
    return sum(p.numel() for p in gen.parameters() if p.requires_grad)


@torch.no_grad()
def _zero_leakage_dev_score(
    gen: LoomUni3DARGenerator,
    refiner: PostTreeGraphRefiner,
    bus_train: Any,
    dev_homo: np.ndarray,
    dev_natoms: np.ndarray,
    sa_target: float,
    cfg_scale: float,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate `gen` on `256` dev prompts strictly under training `bus_train` (`0` `audit_dpa4` calls)."""
    gen.eval()
    nb = len(dev_homo)
    cond_2d = np.stack([dev_homo.astype(np.float32), np.full(nb, float(sa_target), dtype=np.float32)], axis=-1)
    tn = gen.normalize_targets(cond_2d)
    cm = torch.ones((nb, 2), dtype=torch.bool, device=device)

    trajs = gen.sample(
        tn,
        cm,
        uncond_active_mask=None,
        cfg_scale=float(cfg_scale),
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.20,
        geometric_guard=True,
        temperatures=(0.85, 0.30, 0.25),
        seed=880123,
        use_bf16=(device.type == "cuda"),
        electron_parity=0,
        leaf_atom_budget=dev_natoms,
    )
    refiner.refine_trajectories_batch(trajs)
    admit = bus_train.admit_trajectories(trajs, polish_hydrogens_mlip=False)
    num_obj = len(bus_train.spec.objectives)
    req_mat = np.zeros((nb, num_obj), dtype=np.float32)
    act_mat = np.zeros((nb, num_obj), dtype=bool)
    req_mat[:, 0] = dev_homo
    act_mat[:, 0] = True
    ev = bus_train.evaluate_molecules(
        admit["mols"],
        requested_targets=req_mat,
        active_mask=act_mat,
        include_audit=False,
        for_training=True,
        batch_size=64,
    )
    eval_m = ev["evaluated_mask"]
    mae = float(np.mean(np.abs(ev["objective_preds"][eval_m, 0] - dev_homo[eval_m]))) if eval_m.any() else 9.9
    hit_y = float(ev["constrained_clean_hits"].mean())
    legal_r = float(ev["legal_mask"].mean())
    sa_ok = float(ev["constraints_met"][ev["legal_mask"]].mean()) if ev["legal_mask"].any() else 0.0
    valid_f = ev["f_max_eV_A"][ev["legal_mask"] & np.isfinite(ev["f_max_eV_A"])]
    fmax_med = float(np.median(valid_f)) if len(valid_f) > 0 else 9.9
    composite = hit_y - 0.30 * mae + 0.15 * legal_r + 0.15 * sa_ok - 0.015 * min(fmax_med, 6.0)
    return {
        "composite": round(composite, 5),
        "dev_synth_hit_yield": round(hit_y, 4),
        "dev_surr_mae_eV": round(mae, 4),
        "dev_legal_rate": round(legal_r, 4),
        "dev_sa_ok_rate": round(sa_ok, 4),
        "dev_fmax_median": round(fmax_med, 3),
    }


def run_align_vine_grpo(args: argparse.Namespace) -> Path:
    """Run online Tree-Vine GRPO (`topology_only_policy=True`, `m_trunks=4, k_leaves=2`) on `v4_step3_synth_sft.pt`."""
    device = torch.device(args.device)
    step3_dir = args.work_dir / args.run_name
    step3_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = step3_dir / "v4_step3_final.pt"
    if args.reuse_grpo and final_ckpt.exists():
        print(f"[Step3 GRPO] Reusing {final_ckpt}", flush=True)
        return final_ckpt

    synth_ckpt = ensure_sft_synth(args)
    ref_ckpt = run_distill_refiner(args)
    refiner, ref_meta = PostTreeGraphRefiner.load_checkpoint(ref_ckpt, device=args.device)
    refiner.eval()

    gen, synth_payload = LoomUni3DARGenerator.load_checkpoint(synth_ckpt, device=args.device, batch_size=128)
    synth_meta = synth_payload.get("metadata") or {}
    n_trainable = _configure_v4_grpo_trainable(gen, top_k_layers=int(args.grpo_top_layers))
    print(f"[Step3 GRPO] Loaded {synth_ckpt} | trainable_params={n_trainable:,}", flush=True)

    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    bus_train, _, _ = _build_calibrated_oe62_bus(
        args, include_blind_auditor=False, calib_cache_path=calib_cache
    )

    # Strictly synthesizable (`SA <= 3.15`) equilibrium conformer replay anchor!
    train_sdf = args.step2_dir / "oe62_v4_train_eq.sdf"
    synth_mols_all, synth_targets_all = _load_synthesizable_v4_subset(train_sdf, max_sa=3.15)
    r_rng = np.random.default_rng(202632)
    rep_sel = r_rng.choice(len(synth_mols_all), size=min(4096, len(synth_mols_all)), replace=False)
    replay_mols = [synth_mols_all[int(i)] for i in rep_sel]
    replay_targets = synth_targets_all[rep_sel]
    del synth_mols_all

    _, clean_targets = _get_clean_oe62_pool(args, None)
    pool_sub = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 54.0)]

    dev_idx = r_rng.choice(len(pool_sub), size=256, replace=False)
    dev_homo = pool_sub[dev_idx, 0].astype(np.float32)
    dev_natoms = np.round(pool_sub[dev_idx, 1]).astype(np.float32)

    base_dev = _zero_leakage_dev_score(
        gen, refiner, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
    )
    print(f"[Step3 GRPO] Baseline Synth-SFT dev={base_dev}", flush=True)

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
        replay_weight=float(args.replay_weight),
        replay_batch_size=12,
        w_valid=0.85,
        w_phys=0.35,
        w_prop=1.10,
        tolerance_eV=0.25,
        sharp_concentration=True,
        cfg_scale=float(args.cfg_scale),
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.20,
        geometric_guard=True,
        topology_only_policy=True,
        temperatures=(0.85, 0.30, 0.25),
        electron_parity=0,
        size_anchored_cfg=False,
        size_adherence_weight=0.45,
        size_tolerance_atoms=2.0,
        use_vine_rollouts=True,
        m_trunks=int(args.m_trunks),
        k_leaves=int(args.k_leaves),
        post_tree_refiner=refiner,
        polish_hydrogens_mlip=False,
        seed=202633,
    )

    prompts_per_step = int(args.prompts_per_step)

    def _sample_prompts(step_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        picks = rng.choice(len(pool_sub), size=prompts_per_step, replace=False)
        h_req = np.clip(
            pool_sub[picks, 0].astype(np.float32) + rng.normal(0.0, 0.12, size=prompts_per_step).astype(np.float32),
            -8.0,
            -4.9,
        )
        n_req = np.round(pool_sub[picks, 1]).astype(np.float32)
        p_t = np.stack([h_req, np.full(prompts_per_step, float(args.eval_sa_target), dtype=np.float32)], axis=-1)
        p_m = np.ones((prompts_per_step, 2), dtype=bool)
        return p_t, p_m, n_req

    ema_decay = float(args.grpo_ema_decay)

    def _on_ckpt(step_num: int, cur_gen: Any, hist: List[Dict[str, Any]]) -> None:
        nonlocal best_state, best_dev
        with torch.no_grad():
            for p_ema, p_cur in zip(ema_gen.parameters(), cur_gen.parameters()):
                p_ema.data.mul_(ema_decay).add_(p_cur.data, alpha=1.0 - ema_decay)
            for b_ema, b_cur in zip(ema_gen.buffers(), cur_gen.buffers()):
                b_ema.copy_(b_cur)
        if step_num % 45 == 0 or step_num == int(args.grpo_steps):
            cur_dev = _zero_leakage_dev_score(
                ema_gen, refiner, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
            )
            print(f"[Step3 GRPO Dev@{step_num}] {cur_dev} (best={best_dev['composite']})", flush=True)
            if cur_dev["composite"] >= best_dev["composite"]:
                best_dev = cur_dev
                best_state = {k: v.detach().cpu().clone() for k, v in ema_gen.state_dict().items()}

    t0 = time.perf_counter()
    history = trainer.train_loop(
        total_steps=int(args.grpo_steps),
        prompt_sampler_fn=_sample_prompts,
        group_size=int(args.m_trunks) * int(args.k_leaves),
        checkpoint_fn=_on_ckpt,
        log_every=10,
    )
    grpo_wall = time.perf_counter() - t0
    grpo_gpuh = grpo_wall / 3600.0

    # Also evaluate raw final policy vs EMA policy on the zero-leakage dev slice!
    raw_dev = _zero_leakage_dev_score(
        gen, refiner, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
    )
    print(f"[Step3 GRPO Final Raw Policy Dev] {raw_dev}", flush=True)
    if raw_dev["composite"] > best_dev["composite"]:
        best_dev = raw_dev
        best_state = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}

    ema_gen.load_state_dict(best_state, strict=True)
    ema_gen.save_checkpoint(
        final_ckpt,
        extra_metadata={
            "sft_metadata": synth_meta.get("step2_metadata") or {},
            "synth_sft_metadata": synth_meta,
            "refiner_metadata": ref_meta,
            "grpo_steps": int(args.grpo_steps),
            "m_trunks": int(args.m_trunks),
            "k_leaves": int(args.k_leaves),
            "grpo_wall_sec": round(grpo_wall, 2),
            "grpo_gpuh": round(grpo_gpuh, 4),
            "baseline_synth_sft_dev": base_dev,
            "selected_step3_dev": best_dev,
            "grpo_tail_metrics": history[-1] if history else {},
        },
    )
    (step3_dir / "v4_step3_grpo_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[Step3 GRPO DONE] Saved {final_ckpt} | grpo_gpuh={grpo_gpuh:.3f} | best_dev={best_dev}", flush=True)
    return final_ckpt


@torch.no_grad()
def run_step3_rollout_shard(args: argparse.Namespace) -> None:
    """Collect 1 shard of Step-3 rollouts WITH `uncond_active_mask=None` + `PostTreeGraphRefiner`."""
    device = torch.device(args.device)
    step3_dir = args.work_dir / args.run_name
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    ckpt_p = step3_dir / "v4_step3_final.pt"
    ref_ckpt = step3_dir / "v4_step3_post_tree_refiner.pt"
    shard_out = step3_dir / f"step3_rollout_shard_{args.shard_id}.pt"

    gen, _ = LoomUni3DARGenerator.load_checkpoint(ckpt_p, device=args.device, batch_size=512)
    gen.eval()
    refiner, _ = PostTreeGraphRefiner.load_checkpoint(ref_ckpt, device=args.device)
    refiner.eval()

    bus, _, _ = _build_calibrated_oe62_bus(args, include_blind_auditor=False, calib_cache_path=calib_cache)
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))

    _, clean_targets = _get_clean_oe62_pool(args, None)
    rng = np.random.default_rng(20263300 + int(args.shard_id) * 1009)
    n_shard = int(args.critic_rollouts) // max(1, int(args.num_shards))

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

        roll = gen.sample_rollouts_with_two_stage_features(
            tn,
            cm,
            uncond_active_mask=None,
            cfg_scale=float(args.cfg_scale),
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.20,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=660000 + int(args.shard_id) * 50000 + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            leaf_atom_budget=req_natoms[s:e],
        )
        s1_h_list.append(roll["s1_h_concat"])
        s1_g_list.append(roll["s1_geom_feats"])
        s2_h_list.append(roll["s2_h_stage2"])
        s2_c_list.append(roll["s2_cloud_feats"])
        t_norm_list.append(tn.detach().cpu())

        trajs_sub = roll["trajectories"]
        refiner.refine_trajectories_batch(trajs_sub)
        for loc_i, tr in enumerate(trajs_sub):
            s2_e_slices.append(tr["stage2_elem_ids"])
            s2_p_slices.append(tr["stage2_leaf_pos"])
            res = reconciler.reconcile_point_cloud(
                tr["atom_symbols"],
                tr["positions_np"],
                potential_slot=None,
                bond_order_priors=tr.get("bond_order_probs"),
            )
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
        f"[Step3 Rollout Shard {args.shard_id}] N={n_shard} | clean={is_clean_all.mean()*100:.1f}% | "
        f"constr+low_strain={is_constr.mean()*100:.1f}% | med_Fmax={np.nanmedian(fmax_all):.2f} eV/A | wall={wall_s:.1f}s",
        flush=True,
    )


def _fit_step3_dev_prompt_centering(
    gen: LoomUni3DARGenerator,
    refiner: PostTreeGraphRefiner,
    cal_unimol2: Any,
    clean_targets: np.ndarray,
    sa_target: float,
    cfg_scale: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Optional zero-leakage mean-bias prompt centering on `256` dev rollouts (`0` `audit_dpa4` calls)."""
    rng = np.random.default_rng(202634)
    sub = clean_targets[(clean_targets[:, 1] >= 22.0) & (clean_targets[:, 1] <= 44.0)]
    picks = rng.choice(len(sub), size=256, replace=False)
    h_req = sub[picks, 0].astype(np.float32)
    n_req = np.round(sub[picks, 1]).astype(np.float32)

    cond_2d = np.stack([h_req, np.full(len(h_req), float(sa_target), dtype=np.float32)], axis=-1)
    tn = gen.normalize_targets(cond_2d)
    cm = torch.ones((len(h_req), 2), dtype=torch.bool, device=device)

    trajs = gen.sample(
        tn,
        cm,
        uncond_active_mask=None,
        cfg_scale=float(cfg_scale),
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.20,
        geometric_guard=True,
        temperatures=(0.85, 0.30, 0.25),
        seed=777111,
        use_bf16=(device.type == "cuda"),
        electron_parity=0,
        leaf_atom_budget=n_req,
    )
    refiner.refine_trajectories_batch(trajs)
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    mols: List[Optional[Chem.Mol]] = []
    for tr in trajs:
        res = reconciler.reconcile_point_cloud(
            tr["atom_symbols"],
            tr["positions_np"],
            potential_slot=None,
            bond_order_priors=tr.get("bond_order_probs"),
        )
        mols.append(res.mol if (res.reconciled_legal and not res.reconciled_exotic_charge) else None)

    out = cal_unimol2.predict_mols(mols, batch_size=64)
    ok = out["valid_mask"] & np.isfinite(out["mean"][:, 0])
    if int(ok.sum()) < 48:
        return {"applied": False, "bias_eV": 0.0}

    x = h_req[ok].astype(np.float64)
    y = out["mean"][ok, 0].astype(np.float64)
    bias = float(np.mean(y - x))
    apply_it = abs(bias) >= 0.03
    return {
        "applied": bool(apply_it),
        "n_valid": int(ok.sum()),
        "bias_eV": round(bias, 4),
    }


def _apply_prompt_map(homo_1d: np.ndarray, dev_cal: Dict[str, Any]) -> np.ndarray:
    if not dev_cal.get("applied", False):
        return homo_1d.astype(np.float32)
    shifted = homo_1d.astype(np.float64) - float(dev_cal.get("bias_eV", 0.0))
    return np.clip(shifted, -8.3, -4.6).astype(np.float32)


def run_eval_final(args: argparse.Namespace) -> None:
    """Fit Step-3 `TwoStageOctreeProcessCritic` & run final Fair 30-Atom + Multi-Atom-Bucket benchmarks."""
    device = torch.device(args.device)
    step3_dir = args.work_dir / args.run_name
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    ckpt_p = step3_dir / "v4_step3_final.pt"
    ref_ckpt = step3_dir / "v4_step3_post_tree_refiner.pt"
    critic_ckpt = step3_dir / "v4_step3_two_stage_critic.pt"
    sdf_dir = step3_dir / "cohorts_sdf"

    gen, gen_payload = LoomUni3DARGenerator.load_checkpoint(ckpt_p, device=args.device, batch_size=512)
    gen.eval()
    step3_meta = gen_payload.get("metadata") or {}
    refiner, ref_meta = PostTreeGraphRefiner.load_checkpoint(ref_ckpt, device=args.device)
    refiner.eval()

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
                raise RuntimeError(f"Step3 rollout shard {sh_i} failed with exit code {rc}")
        roll_wall = time.perf_counter() - t_r0

        shards = [
            torch.load(step3_dir / f"step3_rollout_shard_{sh_i}.pt", map_location="cpu", weights_only=False)
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
            seed=202635,
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
            seed=202636,
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
        print(f"[Step3 Critic DONE] {critic_meta}", flush=True)

    bus_eval, cal_unimol2, calib_diag = _build_calibrated_oe62_bus(
        args, include_blind_auditor=True, calib_cache_path=calib_cache
    )
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    obj_col_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

    dev_cal = _fit_step3_dev_prompt_centering(
        gen, refiner, cal_unimol2, clean_targets, args.eval_sa_target, args.cfg_scale, device
    )
    print(f"[Step3 Dev Prompt Centering] {dev_cal}", flush=True)

    def _encode_homo_sa(homo_1d: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        nb = len(homo_1d)
        h_mapped = _apply_prompt_map(homo_1d, dev_cal)
        mat = np.stack([h_mapped, np.full(nb, float(args.eval_sa_target), dtype=np.float32)], axis=-1)
        tn = gen.normalize_targets(mat)
        ma = torch.ones((nb, 2), dtype=torch.bool, device=device)
        return tn, ma

    # Warmup (`PostTreeGraphRefiner` + `TwoStageOctreeProcessCritic`)
    wt, wm = _encode_homo_sa(np.full(16, -6.4, dtype=np.float32))
    w_grps = gen.sample_with_two_stage_process_critic(
        wt, wm, two_stage_critic, m_stage1=8, m_stage2=4, m_keep=2, objective_col_mask=obj_col_mask,
        cfg_scale=float(args.cfg_scale), cfg_tree_decay=0.20, geometric_guard=True, seed=31, use_bf16=True,
        uncond_active_mask=None, leaf_atom_budget=np.full(16, 30.0, dtype=np.float32),
    )
    refiner.refine_trajectories_batch([tr for g in w_grps for tr in g])
    _sync(device)

    # ---------------- A. Fair Fixed-Prompt `30.4`-Atom Benchmark (`N = 1,792`) ----------------
    req_30 = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
    n_p30 = len(req_30)
    arms_30: Dict[str, Any] = {}
    funnels_30: List[Tuple[str, int, int, int, bool]] = [
        ("Outcome_M1", 1, 1, 1, False),
        ("TwoStage_8_to_4_to_1", 8, 4, 1, False),
        ("TwoStage_8_to_4_to_2", 8, 4, 2, True),
        ("TwoStage_16_to_6_to_2", 16, 6, 2, True),
        ("TwoStage_24_to_8_to_2", 24, 8, 2, True),
        ("TwoStage_32_to_8_to_2", 32, 8, 2, True),
        ("TwoStage_32_to_8_to_3", 32, 8, 3, True),
    ]
    for arm_name, ms1, ms2, mkeep, use_surr in funnels_30:
        _sync(device)
        tg0 = time.perf_counter()
        groups_all: List[List[Dict[str, Any]]] = []
        p_chunk = 512 if ms1 == 1 else 256
        for s in range(0, n_p30, p_chunk):
            e = min(s + p_chunk, n_p30)
            bt, bm = _encode_homo_sa(req_30[s:e, 0])
            groups_all.extend(
                gen.sample_with_two_stage_process_critic(
                    bt, bm, two_stage_critic, m_stage1=ms1, m_stage2=ms2, m_keep=mkeep,
                    objective_col_mask=obj_col_mask, stage1_legal_weight=0.80, stage1_constr_weight=0.55,
                    stage2_legal_weight=1.05, stage2_constr_weight=0.70, coarse_chunk_size=1024,
                    cfg_scale=float(args.cfg_scale), xyz_cfg_scale=1.0, cfg_tree_decay=0.20, geometric_guard=True,
                    temperatures=(0.85, 0.30, 0.25), seed=970000 + ms1 * 409 + ms2 * 61 + mkeep * 7 + s,
                    use_bf16=True, electron_parity=0, uncond_active_mask=None, leaf_atom_budget=req_30[s:e, 1],
                )
            )
        flat_arm = [tr for g in groups_all for tr in g]
        refiner.refine_trajectories_batch(flat_arm, batch_size=256)
        _sync(device)
        gen_w = time.perf_counter() - tg0

        tr0 = time.perf_counter()
        p_grid: List[List[Optional[Chem.Mol]]] = [[None] * mkeep for _ in range(n_p30)]
        p_leg = np.zeros((n_p30, mkeep), dtype=bool)
        p_exo = np.zeros((n_p30, mkeep), dtype=bool)
        for i in range(n_p30):
            for k in range(mkeep):
                tr = groups_all[i][k]
                res = reconciler.reconcile_point_cloud(
                    tr["atom_symbols"],
                    tr["positions_np"],
                    potential_slot=None,
                    bond_order_priors=tr.get("bond_order_probs"),
                )
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
        arm_res["funnel"] = {"m_stage1": ms1, "m_stage2": ms2, "m_keep": mkeep, "use_surrogate": use_surr}
        arms_30[arm_name] = arm_res
        print(
            f"[Step3 FINAL 30-Atom {arm_name}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
            f"Legal={arm_res['legal_rate']*100:.1f}% Fmax={arm_res['zero_step_f_max_median_eV_A']:.2f} | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} ({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindMAE={arm_res['audit_mae_eV']:.4f} eV",
            flush=True,
        )

    sft_gpuh = float((step3_meta.get("sft_metadata") or {}).get("sft_gpuh", 0.0))
    synth_sft_gpuh = float((step3_meta.get("synth_sft_metadata") or {}).get("synth_sft_gpuh", 0.0))
    ref_gpuh = float(ref_meta.get("refiner_gpuh", 0.0))
    grpo_gpuh = float(step3_meta.get("grpo_gpuh", 0.0))
    crit_gpuh = float(critic_meta.get("rollout_gpuh", 0.0))
    total_gpuh = round(sft_gpuh + synth_sft_gpuh + ref_gpuh + grpo_gpuh + crit_gpuh, 4)

    fair_payload = {
        "experiment": "OE62 Fair Fixed-Prompt 30.4-Atom Benchmark — Uni-Loom [Uni-3DAR v4: Full-SO(3) Rank-32 LoRA SFT + Non-Trapping ScaleFreeBudgetEmbedding + SE(3) PostTreeGraphRefiner + Tree-Vine GRPO + TwoStage 32->8->2]",
        "benchmark_seed": 20261118,
        "n_prompts": n_p30,
        "mean_requested_atoms": round(float(np.mean(req_30[:, 1])), 2),
        "total_training_gpuh": total_gpuh,
        "gpuh_breakdown": {
            "step2_ddp_sft_gpuh": round(sft_gpuh, 4),
            "step3_synth_manifold_sft_gpuh": round(synth_sft_gpuh, 4),
            "step3_post_tree_refiner_gpuh": round(ref_gpuh, 4),
            "step3_vine_grpo_gpuh": round(grpo_gpuh, 4),
            "step3_two_stage_critic_gpuh": round(crit_gpuh, 4),
        },
        "zero_leakage_dft_calibration": calib_diag,
        "dev_prompt_centering": dev_cal,
        "step3_metadata": step3_meta,
        "two_stage_critic_meta": critic_meta,
        "arms": arms_30,
    }
    args.fair_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.fair_output_json.write_text(json.dumps(fair_payload, indent=2) + "\n")
    print(f"[Step3 Fair 30-Atom SAVED] {args.fair_output_json} (Total GPU-h={total_gpuh})", flush=True)

    # ---------------- B. Multi-Atom-Count Bucket Scaling (`N in {20, 30, 40, 50, 60}`, 280/bucket) ----------------
    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    bucket_results: Dict[str, Any] = {}
    for n_exact in (20, 30, 40, 50, 60):
        req_b = np.asarray(
            [(float(q_points[str(q)]), float(n_exact)) for q in quantiles for _ in range(40)], dtype=np.float32
        )
        nb = len(req_b)
        bt, bm = _encode_homo_sa(req_b[:, 0])
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
                objective_col_mask=obj_col_mask, coarse_chunk_size=1024, cfg_scale=float(args.cfg_scale), xyz_cfg_scale=1.0,
                cfg_tree_decay=0.20, geometric_guard=True, seed=72000 + ms1 * 43 + n_exact,
                use_bf16=True, electron_parity=0, uncond_active_mask=None, leaf_atom_budget=req_b[:, 1],
            )
            refiner.refine_trajectories_batch([tr for g in grps for tr in g], batch_size=256)
            _sync(device)
            gw = time.perf_counter() - t0

            r0 = time.perf_counter()
            pg_b: List[List[Optional[Chem.Mol]]] = [[None] * mk for _ in range(nb)]
            pl_b = np.zeros((nb, mk), dtype=bool)
            px_b = np.zeros((nb, mk), dtype=bool)
            for i in range(nb):
                for k in range(mk):
                    tr = grps[i][k]
                    res = reconciler.reconcile_point_cloud(
                        tr["atom_symbols"],
                        tr["positions_np"],
                        potential_slot=None,
                        bond_order_priors=tr.get("bond_order_probs"),
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
                f"[Step3 Bucket N={n_exact} {b_tag}] Eff={b_res['effective_output_mol_per_sec']:.2f} mol/s | "
                f"Atoms={b_res['mean_total_atoms']} (err={b_res['mean_natoms_abs_err']}) | "
                f"Legal={b_res['legal_rate']*100:.1f}% | BlindSynth={b_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}% | "
                f"BlindMAE={b_res['audit_mae_eV']:.4f} eV",
                flush=True,
            )
        bucket_results[str(n_exact)] = {"target_n_atoms": n_exact, "n_prompts": nb, "arms": b_arms}

    buckets_payload = {
        "experiment": "OE62 Controlled Multi-Atom-Count Scaling (N in {20, 30, 40, 50, 60}) — Uni-Loom [Uni-3DAR v4]",
        "total_training_gpuh": total_gpuh,
        "buckets": bucket_results,
    }
    args.buckets_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.buckets_output_json.write_text(json.dumps(buckets_payload, indent=2) + "\n")
    print(f"[Step3 ALL DONE] Wrote {args.buckets_output_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["sft_synth_ddp", "distill_refiner", "align_vine_grpo", "rollout_shard", "eval_final", "all"],
        required=True,
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rollout-gpus", type=str, default="0,1")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--synth-sft-epochs", type=int, default=10)
    p.add_argument("--synth-max-sa", type=float, default=3.30)
    p.add_argument("--refiner-epochs", type=int, default=14)
    p.add_argument("--refiner-batch-size", type=int, default=192)
    p.add_argument("--grpo-steps", type=int, default=180)
    p.add_argument("--grpo-top-layers", type=int, default=5)
    p.add_argument("--grpo-lr", type=float, default=1.8e-5)
    p.add_argument("--kl-beta", type=float, default=0.030)
    p.add_argument("--replay-weight", type=float, default=0.25)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--m-trunks", type=int, default=4)
    p.add_argument("--k-leaves", type=int, default=2)
    p.add_argument("--grpo-ema-decay", type=float, default=0.992)
    p.add_argument("--critic-rollouts", type=int, default=16384)
    p.add_argument("--eval-sa-target", type=float, default=1.90)
    p.add_argument("--cfg-scale", type=float, default=2.15)
    p.add_argument("--reuse-synth-sft", action="store_true")
    p.add_argument("--reuse-refiner", action="store_true")
    p.add_argument("--reuse-grpo", action="store_true")
    p.add_argument("--reuse-critic", action="store_true")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="uni3dar_v4_step3")
    p.add_argument(
        "--step2-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/uni3dar_v4_step2"),
    )
    p.add_argument(
        "--parent-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_e13_e14_20261113/full_homo_27000/weights/octree_conditional.pt"),
    )
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
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
        "--fair-output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_fair_30atom.json"),
    )
    p.add_argument(
        "--buckets-output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_atom_buckets.json"),
    )
    args = p.parse_args()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.phase == "sft_synth_ddp":
        run_sft_synth_ddp_worker(args)
    elif args.phase == "distill_refiner":
        run_distill_refiner(args)
    elif args.phase == "align_vine_grpo":
        run_align_vine_grpo(args)
    elif args.phase == "rollout_shard":
        run_step3_rollout_shard(args)
    elif args.phase == "eval_final":
        run_eval_final(args)
    else:
        run_align_vine_grpo(args)
        run_eval_final(args)


if __name__ == "__main__":
    main()
