"""Step 3 of `Uni-3DAR v4 + Uni-Loom` Speed-Preserving Overtake Plan (`~1.2-1.8 GPU-h` total, `0` PCA, `0` inference MLIP).

Integrates all three zero-inference-cost architectural & geometric breakthroughs:
  1. **Non-Trapping Global Scale-Free Budget (`non_trapping_budget=True`, `num_prefix_tokens=0`)**:
     Eliminates Step-0 prefix RoPE KV corruption and coarse octant `c_alloc` corner-trapping while
     enforcing Euclidean steric `L2/L1` bounds (`0.48 A` cell <= 2 atoms, `1` preferred).
  2. **Zero-Inference-Cost Geometry & Synthesizability Internalization**:
     - `relax_synth_shard` + `sft_synth_ddp`: 2-GPU offline `DPA-4` equilibrium relaxation (`F_max <= 2.4 eV/A`,
       `0` PCA, `0` `audit_dpa4` calls) of `9,600` synthesizable (`SA <= 3.30`) OE62 conformers (removing
       solid-state X-ray crystal-packing strain) + 12-epoch 2-GPU DDP Rank-32 LoRA SFT.
     - Online Hindsight `DPA-4`-Relaxed AWR (`2` rounds x `2,048` proposals, full SO(3) `random_rotation_prob=0.80`),
       simultaneously harvesting real `(raw_sampled_pos -> DPA-4_relaxed_mol)` pairs.
     - Online Tree-Vine GRPO (`150` steps, `m_trunks=4, k_leaves=2`, `topology_only_policy=True`,
       `size_adherence_weight=0.45`, EMA shadow checkpointing on `256` zero-leakage dev prompts).
     - Post-GRPO Detached `0.01 A` Sub-Voxel XYZ Head Distillation (`hidden.detach()`, full SO(3) rotation)
       so fine bond lengths internalize gas-phase `DPA-4` equilibrium without perturbing aligned topology.
  3. **On-Policy 4-Layer SE(3) `PostTreeGraphRefiner` (`max_disp_A = 0.25 A`)**:
     Distilled directly on real `(Uni-3DAR raw sampled coords -> Kabsch-aligned DPA-4 equilibrium coords)`
     pairs + multi-scale seam perturbations (`< 1 ms` single batched pass at inference, ZERO iterative MLIP).
  4. **Final Evaluation (`eval_final`)**:
     Fits Step-3 `TwoStageOctreeProcessCritic` (`16,384` rollouts across `GPU 0,1`), measures optional
     zero-leakage mean-bias prompt centering (`0` `audit_dpa4` calls), and benchmarks:
       - Fair Fixed-Per-Prompt `30.4`-Atom OE62 Benchmark (`N = 1,792`, seed `20261118`)
       - Multi-Atom-Count Bucket Scaling (`N in {20, 30, 40, 50, 60}`, `280` prompts/bucket)
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
from rdkit import Chem, RDLogger
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MOLGEN_COMPAT_DEFAULT = Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat")
for cand in (ROOT / "src" / "compat", ROOT / "src", ROOT, MOLGEN_COMPAT_DEFAULT):
    cstr = str(cand)
    if cand.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402
from loom.generator.molecular_state import mol_to_clean_smiles  # noqa: E402
from loom.generator.octree_backend import OctreeDataset  # noqa: E402
from loom.utils.dataset_io import write_precise_sdf  # noqa: E402
from loom.vendor.uni3dar.data.atom_data import atom_list  # noqa: E402
from loom.weaving import (  # noqa: E402
    BuiltInRuleAsset,
    LoomGRPOTrainer,
    LoomUni3DARGenerator,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    PotentialSlot,
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
    _safe_collate_indices,
    _sample_conditional_natoms,
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


def _load_synthesizable_v4_subset(sdf_path: Path, max_sa: float = 3.30) -> Tuple[List[Chem.Mol], np.ndarray]:
    """Filter `oe62_v4_train_eq.sdf` purely by deterministic RDKit `SA_Score <= max_sa` & `smarts_alert == 0`."""
    all_mols, all_t = _load_prepared_v4_sdf(sdf_path)
    rule_asset = BuiltInRuleAsset(asset_id="synth_filter")
    r_out = rule_asset.predict_mols(all_mols, batch_size=512)
    sm_idx = rule_asset.channel_index("smarts_alert")
    sm_alerts = r_out["mean"][:, sm_idx]
    keep = (
        (all_t[:, 1] <= float(max_sa))
        & r_out["valid_mask"]
        & (sm_alerts <= 0.0)
        & np.asarray([is_clean_organic_charge_state(m) for m in all_mols], dtype=bool)
    )
    kept_mols = [all_mols[i] for i in range(len(all_mols)) if bool(keep[i])]
    kept_t = all_t[keep]
    return kept_mols, kept_t


def _select_stratified_synth_candidates(
    mols: Sequence[Chem.Mol],
    targets: np.ndarray,
    max_total: int,
    seed: int = 202630,
) -> Tuple[List[Chem.Mol], np.ndarray]:
    """Stratified selection across `[16..66]` atoms oversampling `N >= 28` so 30-60 atom equilibria are dense."""
    rng = np.random.default_rng(seed)
    nat = np.asarray([m.GetNumAtoms() for m in mols], dtype=np.int32)
    elig = np.where((nat >= 16) & (nat <= 66))[0]
    if len(elig) <= max_total:
        return [mols[int(i)] for i in elig], targets[elig]

    n_el = nat[elig].astype(np.float64)
    w = np.where(n_el >= 28.0, 2.6, 1.0)
    w = np.where(n_el >= 42.0, 3.0, w)
    w /= w.sum()
    chosen = rng.choice(elig, size=int(max_total), replace=False, p=w)
    chosen.sort()
    return [mols[int(i)] for i in chosen], targets[chosen]


def run_relax_synth_shard(args: argparse.Namespace) -> None:
    """Relax 1 shard of stratified synthesizable OE62 crystal conformers with gas-phase `DPA-4` (`0` PCA!)."""
    step3_dir = args.work_dir / args.run_name
    step3_dir.mkdir(parents=True, exist_ok=True)
    train_sdf = args.step2_dir / "oe62_v4_train_eq.sdf"
    synth_mols, synth_targets = _load_synthesizable_v4_subset(train_sdf, max_sa=float(args.synth_max_sa))
    cand_mols, cand_targets = _select_stratified_synth_candidates(
        synth_mols, synth_targets, max_total=int(args.relax_pool_cap), seed=202630
    )

    n_shards = max(1, int(args.num_shards))
    sh_id = int(args.shard_id)
    sub_mols = cand_mols[sh_id::n_shards]
    sub_targets = cand_targets[sh_id::n_shards]

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

    rel_out = pot_slot.relax_molecules(sub_mols, n_steps=20, chunk_size=128)
    admitted: List[Chem.Mol] = []
    for i, rm in enumerate(rel_out["relaxed_mols"]):
        f_fin = float(rel_out["final_f_max_eV_A"][i])
        if (
            rm is None
            or not bool(rel_out["graph_preserved"][i])
            or not math.isfinite(f_fin)
            or f_fin > 2.40
            or not is_clean_organic_charge_state(rm)
        ):
            continue
        cm = Chem.Mol(rm)
        h_i, s_i = float(sub_targets[i, 0]), float(sub_targets[i, 1])
        cm.SetProp("v4_homo_eV", f"{h_i:.6f}")
        cm.SetProp("v4_sa_score", f"{s_i:.4f}")
        admitted.append(cm)

    # Verify lossless octree encodability (`cutoff_atom_count == 0`)
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

    shard_sdf = step3_dir / f"relaxed_synth_shard_{sh_id}.sdf"
    shard_meta = step3_dir / f"relaxed_synth_shard_{sh_id}.json"
    write_precise_sdf(shard_sdf, clean_out)
    wall = time.perf_counter() - t0
    shard_meta.write_text(
        json.dumps(
            {
                "shard_id": sh_id,
                "input_count": len(sub_mols),
                "admitted_count": len(clean_out),
                "wall_sec": round(wall, 2),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[Step3 Relax Shard {sh_id}] admitted={len(clean_out)}/{len(sub_mols)} | wall={wall:.1f}s -> {shard_sdf}",
        flush=True,
    )


def ensure_relaxed_synth_pool(args: argparse.Namespace) -> Tuple[Path, float]:
    """Build or reuse `v4_step3_relaxed_synth_pool.sdf` via parallel 2-GPU `DPA-4` relaxation."""
    step3_dir = args.work_dir / args.run_name
    step3_dir.mkdir(parents=True, exist_ok=True)
    pool_sdf = step3_dir / "v4_step3_relaxed_synth_pool.sdf"
    pool_meta = step3_dir / "v4_step3_relaxed_synth_pool.json"
    if args.reuse_relaxed_pool and pool_sdf.exists() and pool_meta.exists():
        meta = json.loads(pool_meta.read_text())
        print(f"[Step3 RelaxPool] Reusing {pool_sdf} ({meta})", flush=True)
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
            "relax_synth_shard",
            "--device",
            f"cuda:{gpu_id}",
            "--shard-id",
            str(sh_i),
            "--num-shards",
            str(n_shards),
            "--relax-pool-cap",
            str(args.relax_pool_cap),
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
            "--oe62-potential-weights",
            str(args.oe62_potential_weights),
        ]
        if args.allow_tf32:
            cmd.append("--allow-tf32")
        procs.append(subprocess.Popen(cmd, text=True))

    for sh_i, pr in enumerate(procs):
        rc = pr.wait()
        if rc != 0:
            raise RuntimeError(f"relax_synth_shard {sh_i} failed with exit code {rc}")
    wall = time.perf_counter() - t0
    gpuh = round((wall * n_shards) / 3600.0, 4)

    merged_mols: List[Chem.Mol] = []
    for sh_i in range(n_shards):
        sh_sdf = step3_dir / f"relaxed_synth_shard_{sh_i}.sdf"
        sh_mols, _ = _load_prepared_v4_sdf(sh_sdf)
        merged_mols.extend(sh_mols)
    write_precise_sdf(pool_sdf, merged_mols)
    meta = {
        "total_admitted": len(merged_mols),
        "mean_atoms": round(float(np.mean([m.GetNumAtoms() for m in merged_mols])), 2) if merged_mols else 0.0,
        "wall_sec": round(wall, 2),
        "relax_pool_gpuh": gpuh,
    }
    pool_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[Step3 RelaxPool DONE] {meta} -> {pool_sdf}", flush=True)
    return pool_sdf, gpuh


def _sample_synth_sft_masks(batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    """Balanced CFG mask supporting full `[F,F]` unconditional guidance (`65% [T,T], 15% [T,F], 20% [F,F]`)."""
    u = rng.random(size=batch_size)
    mask = np.ones((batch_size, 2), dtype=bool)
    mask[(u >= 0.65) & (u < 0.80), 1] = False
    mask[u >= 0.80] = False
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def run_sft_synth_ddp_worker(args: argparse.Namespace) -> None:
    """2-GPU DDP synthesizable `DPA-4`-equilibrium manifold SFT (`12` epochs, `npref=0`, `non_trapping_budget=True`)."""
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
    pool_sdf = step3_dir / "v4_step3_relaxed_synth_pool.sdf"
    pool_meta_p = step3_dir / "v4_step3_relaxed_synth_pool.json"
    relax_gpuh = float(json.loads(pool_meta_p.read_text()).get("relax_pool_gpuh", 0.0)) if pool_meta_p.exists() else 0.0

    synth_mols, synth_targets = _load_prepared_v4_sdf(pool_sdf)
    if rank == 0:
        print(
            f"[Step3 Synth-SFT] Loaded {len(synth_mols)} DPA-4-relaxed synthesizable equilibrium conformers",
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
    step2_meta = {"init_source": "parent_ckpt_npref0_lora_r32_so3_non_trapping_dpa4_eq", "relax_pool_gpuh": relax_gpuh}

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

    head_prefixes = ("conditioner.", "valence_head.", "budget_emb.")
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
            {"params": backbone_params, "lr": 2.2e-4, "weight_decay": 1e-4},
            {"params": head_params, "lr": 3.5e-4, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.95),
    )
    n_ep = int(args.synth_sft_epochs)
    total_steps = max(1, len(loader) * n_ep)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=5.0e-6)
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
        l_sum, xyz_sum, pref_sum, n_pref, nb = 0.0, 0.0, 0.0, 0, 0
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
                    contrast_weight=0.14,
                    contrast_margin=0.08,
                    contrast_min_gap_eV=0.35,
                    contrast_col_mask=contrast_col_mask,
                    xyz_loss_weight=0.15,
                    valence_loss_weight=0.20,
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
                eff_ema = min(0.995, (1.0 + g_step) / (10.0 + g_step))
                for pe, pc in zip(ema_gen.parameters(), gen.parameters()):
                    pe.lerp_(pc, 1.0 - eff_ema)
            l_sum += float(info["loss_total"])
            xyz_sum += float(info["loss_xyz"])
            pr = float(info.get("inbatch_preference_rate", float("nan")))
            if math.isfinite(pr):
                pref_sum += pr
                n_pref += 1
            nb += 1
        if rank == 0:
            print(
                f"[Step3 Synth-SFT Ep {ep:02d}/{n_ep:02d}] loss={l_sum/max(nb,1):.4f} "
                f"(xyz={xyz_sum/max(nb,1):.4f}, pref={pref_sum/max(n_pref,1)*100:.1f}%) wall={time.perf_counter()-ep_t0:.1f}s",
                flush=True,
            )

    synth_gpuh = ((time.perf_counter() - t0) * world_size) / 3600.0 + relax_gpuh
    if rank == 0:
        ema_gen.save_checkpoint(
            out_ckpt,
            extra_metadata={
                "step2_metadata": step2_meta,
                "synth_sft_epochs": n_ep,
                "synth_sft_mols": len(synth_mols),
                "relax_pool_gpuh": round(relax_gpuh, 4),
                "synth_sft_gpuh": round(synth_gpuh, 4),
            },
        )
        print(f"[Step3 Synth-SFT DONE] gpuh={synth_gpuh:.4f} (incl. relax={relax_gpuh:.4f}) -> {out_ckpt}", flush=True)
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


def ensure_sft_synth(args: argparse.Namespace) -> Path:
    """Ensure `DPA-4`-relaxed synthesizable pool + 2-GPU `torchrun` DDP SFT (`v4_step3_synth_sft.pt`)."""
    step3_dir = args.work_dir / args.run_name
    step3_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = step3_dir / "v4_step3_synth_sft.pt"
    ensure_relaxed_synth_pool(args)
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
            "--relax-pool-cap",
            str(args.relax_pool_cap),
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


def _collect_and_relax_onpolicy_batch(
    gen: LoomUni3DARGenerator,
    bus_train: Any,
    cal_unimol2: Any,
    clean_ref_targets: np.ndarray,
    *,
    n_proposals: int,
    sa_target: float,
    cfg_scale: float,
    max_fmax: float,
    seed: int,
    device: torch.device,
) -> Tuple[List[Chem.Mol], np.ndarray, List[Tuple[np.ndarray, Chem.Mol]], Dict[str, Any]]:
    """Sample `n_proposals` on-policy trajectories, filter `SA<=3.45`, and offline `DPA-4`-relax (`0` PCA, `0` audit!).

    Returns:
      - `rel_mols`: `DPA-4`-relaxed equilibrium conformers (`graph_preserved & final_fmax <= max_fmax`)
      - `rel_targets`: `(N_rel, 2)` zero-leakage `[calibrated_homo_eV, sa_score]`
      - `paired_records`: `List[(pre_relax_pos_np, relaxed_mol)]` with exact 1-to-1 atom index correspondence
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    rule_asset = bus_train.registry.get_predictor("builtin_rules")
    pot_slot = bus_train.potential_slot

    h_p04, h_p96 = np.percentile(clean_ref_targets[:, 0], [4.0, 96.0])
    req_h = rng.uniform(h_p04, h_p96, size=n_proposals).astype(np.float32)
    req_n = _sample_conditional_natoms(req_h, clean_ref_targets, rng, window_eV=0.30, clamp_range=(18.0, 60.0))
    large_sel = rng.random(size=n_proposals) < 0.65
    req_n[large_sel] = np.maximum(
        req_n[large_sel], rng.uniform(28.0, 54.0, size=int(large_sel.sum())).astype(np.float32)
    )
    req_n = np.round(req_n).astype(np.float32)

    gen.eval()
    cand_mols: List[Chem.Mol] = []
    cand_noisy_pos: List[np.ndarray] = []
    gen_bs = 512 if device.type == "cuda" else 64
    for s in range(0, n_proposals, gen_bs):
        e = min(s + gen_bs, n_proposals)
        nb = e - s
        cond_2d = np.stack([req_h[s:e], np.full(nb, float(sa_target), dtype=np.float32)], axis=-1)
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
            temperatures=(0.88, 0.32, 0.25),
            seed=seed * 17 + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            leaf_atom_budget=req_n[s:e],
        )
        for tr in trajs:
            raw_syms = tr["atom_symbols"]
            raw_pos = np.asarray(tr["positions_np"], dtype=np.float32)
            res = reconciler.reconcile_point_cloud(raw_syms, raw_pos, potential_slot=None)
            if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
                m_c = res.mol
                conf_p = np.asarray(m_c.GetConformer().GetPositions(), dtype=np.float32)
                # Prefer raw Uni-3DAR coordinates whenever atom ordering matches 1-to-1 (`0` H added/dropped)!
                if (
                    m_c.GetNumAtoms() == len(raw_syms)
                    and [a.GetSymbol() for a in m_c.GetAtoms()] == list(raw_syms)
                    and float(np.max(np.linalg.norm(conf_p - raw_pos, axis=-1))) < 0.35
                ):
                    noisy_use = raw_pos.copy()
                else:
                    noisy_use = conf_p.copy()
                cand_mols.append(m_c)
                cand_noisy_pos.append(noisy_use)

    if cand_mols:
        r_out = rule_asset.predict_mols(cand_mols, batch_size=256)
        sa_v = r_out["mean"][:, rule_asset.channel_index("sa_score")]
        mw_v = r_out["mean"][:, rule_asset.channel_index("mol_weight")]
        sm_v = r_out["mean"][:, rule_asset.channel_index("smarts_alert")]
        keep_idx = [
            k for k in range(len(cand_mols))
            if (sa_v[k] <= 3.55) and (mw_v[k] <= 550.0) and (sm_v[k] <= 0.0)
        ]
    else:
        keep_idx = []

    sub_mols = [cand_mols[k] for k in keep_idx]
    sub_noisy = [cand_noisy_pos[k] for k in keep_idx]

    rel_mols: List[Chem.Mol] = []
    paired_records: List[Tuple[np.ndarray, Chem.Mol]] = []
    if sub_mols and pot_slot is not None:
        rel_out = pot_slot.relax_molecules(sub_mols, n_steps=24, chunk_size=96)
        for loc_i, rm in enumerate(rel_out["relaxed_mols"]):
            f_fin = float(rel_out["final_f_max_eV_A"][loc_i])
            if (
                rm is not None
                and bool(rel_out["graph_preserved"][loc_i])
                and math.isfinite(f_fin)
                and f_fin <= float(max_fmax)
                and is_clean_organic_charge_state(rm)
            ):
                cm = Chem.Mol(rm)
                rel_mols.append(cm)
                if cm.GetNumAtoms() == sub_noisy[loc_i].shape[0]:
                    paired_records.append((sub_noisy[loc_i], cm))

    if rel_mols:
        u_out = cal_unimol2.predict_mols(rel_mols, batch_size=64)
        r2_out = rule_asset.predict_mols(rel_mols, batch_size=256)
        h_vals = u_out["mean"][:, cal_unimol2.channel_index("homo")].astype(np.float32)
        s_vals = r2_out["mean"][:, rule_asset.channel_index("sa_score")].astype(np.float32)
        ok_m = u_out["valid_mask"] & np.isfinite(h_vals) & np.isfinite(s_vals)
        kept_rel = [rel_mols[i] for i in range(len(rel_mols)) if bool(ok_m[i])]
        rel_targets = np.stack([h_vals[ok_m], s_vals[ok_m]], axis=-1) if kept_rel else np.zeros((0, 2), dtype=np.float32)
    else:
        kept_rel = []
        rel_targets = np.zeros((0, 2), dtype=np.float32)

    stats = {
        "proposals": n_proposals,
        "clean_legal_count": len(cand_mols),
        "pre_sa_count": len(sub_mols),
        "relaxed_admitted": len(kept_rel),
        "paired_refiner_count": len(paired_records),
        "wall_sec": round(time.perf_counter() - t0, 2),
    }
    return kept_rel, rel_targets, paired_records, stats


def _run_v4_online_hindsight_awr(
    gen: LoomUni3DARGenerator,
    bus_train: Any,
    cal_unimol2: Any,
    anchor_mols: List[Chem.Mol],
    anchor_targets: np.ndarray,
    clean_ref_targets: np.ndarray,
    *,
    rounds: int,
    proposals_per_round: int,
    steps_per_round: int,
    batch_size: int,
    lr: float,
    sa_target: float,
    cfg_scale: float,
    device: torch.device,
) -> Tuple[List[Chem.Mol], np.ndarray, List[Tuple[np.ndarray, Chem.Mol]], List[Dict[str, Any]]]:
    """Full-SO(3) (`random_rotation_prob=0.80`) Online Hindsight `DPA-4`-Relaxed AWR + Refiner Pair Harvest."""
    seen_smis = {mol_to_clean_smiles(m) for m in anchor_mols}
    disc_mols: List[Chem.Mol] = []
    disc_targets_list: List[np.ndarray] = []
    disc_weights_list: List[float] = []
    all_pairs: List[Tuple[np.ndarray, Chem.Mol]] = []
    round_logs: List[Dict[str, Any]] = []
    rng = np.random.default_rng(202633)
    c_col_mask = torch.tensor([True, True], dtype=torch.bool, device=device)

    for r_idx in range(rounds):
        tr0 = time.perf_counter()
        rel_mols, rel_targets, pairs_r, col_stats = _collect_and_relax_onpolicy_batch(
            gen,
            bus_train,
            cal_unimol2,
            clean_ref_targets,
            n_proposals=proposals_per_round,
            sa_target=sa_target,
            cfg_scale=cfg_scale,
            max_fmax=2.15,
            seed=8200 + r_idx * 131,
            device=device,
        )
        all_pairs.extend(pairs_r)

        added_r = 0
        for m_rel, t_rel in zip(rel_mols, rel_targets):
            if float(t_rel[1]) > 3.45:
                continue
            smi = mol_to_clean_smiles(m_rel)
            if not smi or smi in seen_smis:
                continue
            seen_smis.add(smi)
            disc_mols.append(m_rel)
            disc_targets_list.append(t_rel)
            w_m = 2.6 if m_rel.GetNumAtoms() >= 28 else 1.3
            disc_weights_list.append(w_m)
            added_r += 1

        comb_mols = list(anchor_mols) + list(disc_mols)
        anc_nat = np.asarray([m.GetNumAtoms() for m in anchor_mols], dtype=np.float64)
        w_anc = np.where(anc_nat >= 28.0, 1.5, 1.0)
        if disc_targets_list:
            comb_targets = np.concatenate([anchor_targets, np.stack(disc_targets_list, axis=0)], axis=0)
            w_dis = np.asarray(disc_weights_list, dtype=np.float64) * (len(anchor_mols) / max(len(disc_mols), 1)) * 0.9
            w_all = np.concatenate([w_anc, np.clip(w_dis, 1.0, 7.0)], axis=0)
        else:
            comb_targets = anchor_targets
            w_all = w_anc
        p_all = w_all / w_all.sum()

        awr_ds = OctreeDataset(
            comb_mols, gen.backbone, training=True, seed=3030 + r_idx, random_rotation_prob=0.80
        )
        trainable = gen.configure_trainable(
            train_conditioner=True, train_lora=True, train_xyz_embeddings=False, unfreeze_full_backbone=False
        )
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps_per_round, 1), eta_min=lr * 0.2)

        def _col_awr(sd: int) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
            lrng = np.random.default_rng(sd)
            c_idx = lrng.choice(len(awr_ds), size=batch_size, replace=True, p=p_all)
            return _safe_collate_indices(awr_ds, c_idx.tolist(), rng=lrng)

        losses_r: List[float] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            q_n = min(6, steps_per_round)
            futs = [pool.submit(_col_awr, 510000 + r_idx * 10000 + i) for i in range(q_n)]
            nxt = q_n
            for st_i in range(steps_per_round):
                if st_i > 0 and st_i % 80 == 0:
                    awr_ds.set_grid_epoch(r_idx * 10 + st_i // 80)
                sl = st_i % len(futs)
                pk, kp = futs[sl].result()
                if nxt < steps_per_round:
                    futs[sl] = pool.submit(_col_awr, 510000 + r_idx * 10000 + nxt)
                    nxt += 1
                if pk is None or kp is None:
                    sched.step()
                    continue
                tn_b = gen.normalize_targets(comb_targets[kp])
                ma_b = _sample_synth_sft_masks(len(kp), rng, device)
                gen.train()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss, _ = gen.compute_sft_loss(
                        pk,
                        tn_b,
                        ma_b,
                        cfg_dropout_prob=0.0,
                        xyz_loss_weight=0.15,
                        valence_loss_weight=0.22,
                        count_loss_weight=0.0,
                        geom_dist_loss_weight=0.0,
                        budget_dropout_prob=0.08,
                        contrast_weight=0.14,
                        contrast_margin=0.08,
                        contrast_min_gap_eV=0.35,
                        contrast_col_mask=c_col_mask,
                    )
                if torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    losses_r.append(float(loss.detach().item()))
                sched.step()

        r_info = {
            "round": r_idx + 1,
            **col_stats,
            "relaxed_added_unique": added_r,
            "total_discovered_pool": len(disc_mols),
            "total_refiner_pairs": len(all_pairs),
            "mean_awr_loss": round(float(np.mean(losses_r)), 4) if losses_r else None,
            "round_wall_sec": round(time.perf_counter() - tr0, 2),
        }
        round_logs.append(r_info)
        print(f"[Step3 Hindsight-AWR Round {r_idx + 1}/{rounds}] {r_info}", flush=True)

    final_mols = list(anchor_mols) + list(disc_mols)
    final_targets = (
        np.concatenate([anchor_targets, np.stack(disc_targets_list, axis=0)], axis=0)
        if disc_targets_list
        else anchor_targets
    )
    return final_mols, final_targets, all_pairs, round_logs


def _distill_v4_refiner(
    equil_mols: Sequence[Chem.Mol],
    paired_trajs: Sequence[Tuple[np.ndarray, Chem.Mol]],
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int = 202631,
) -> Tuple[PostTreeGraphRefiner, Dict[str, Any]]:
    """Distill 4-layer SE(3)-equivariant `PostTreeGraphRefiner` (`max_disp_A=0.25`) on real + multi-scale seam pairs."""
    t0 = time.perf_counter()
    refiner = PostTreeGraphRefiner(
        num_atom_classes=128,
        d_model=192,
        n_heads=6,
        n_layers=4,
        n_rbf=32,
        d_pair=128,
        max_disp_A=0.25,
        dropout=0.05,
    ).to(device)
    stats = fit_post_tree_graph_refiner(
        refiner,
        equil_mols,
        paired_trajectories=paired_trajs,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=1.4e-3,
        seed=seed,
    )
    wall = time.perf_counter() - t0
    meta = {
        **stats.__dict__,
        "num_equil_mols": len(equil_mols),
        "num_real_onpolicy_pairs": len(paired_trajs),
        "max_disp_A": 0.25,
        "n_layers": 4,
        "refiner_wall_sec": round(wall, 2),
        "refiner_gpuh": round(wall / 3600.0, 4),
    }
    return refiner, meta


def _run_v4_detached_xyz_distillation(
    gen: LoomUni3DARGenerator,
    equil_mols: Sequence[Chem.Mol],
    equil_targets: np.ndarray,
    *,
    steps: int = 600,
    batch_size: int = 64,
    lr: float = 3.0e-4,
    device: torch.device,
) -> Dict[str, Any]:
    """Freeze 100% of Transformer + conditioner + budget_emb + tree/element heads (`hidden.detach()`);
    distill ONLY sub-voxel `0.01 A` `X, Y, Z` heads + intra-voxel embeddings on `DPA-4`-relaxed conformers
    under full SO(3) rotation (`random_rotation_prob = 0.80`).
    """
    t0 = time.perf_counter()
    if not bool(gen.loom_config.get("lora_include_xyz_heads", False)):
        gen.enable_v11_xyz_heads(rank=None, alpha=None, train_xyz_embeddings=True)
    gen.requires_grad_(False)
    xyz_params: List[torch.nn.Parameter] = []
    ahead = gen.backbone.atom_heads[0]
    for ax_idx in (1, 2, 3):
        for p in ahead.pred_heads[ax_idx].parameters():
            p.requires_grad_(True)
            xyz_params.append(p)
    for emb_idx in (0, 1, 2):
        for p in ahead.emb_layers[emb_idx].parameters():
            p.requires_grad_(True)
            xyz_params.append(p)

    n_xyz = sum(p.numel() for p in xyz_params)
    print(
        f"[Step3 XYZ-Distill] Detached SO(3) sub-voxel equilibrium training: {n_xyz:,} params on {len(equil_mols)} mols",
        flush=True,
    )

    ds = OctreeDataset(
        list(equil_mols), gen.backbone, training=True, seed=202699, random_rotation_prob=0.80
    )
    opt = torch.optim.AdamW(xyz_params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
    rng = np.random.default_rng(202699)

    def _col_xyz(sd: int) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        lrng = np.random.default_rng(sd)
        c_idx = lrng.integers(0, len(ds), size=batch_size)
        return _safe_collate_indices(ds, c_idx.tolist(), rng=lrng)

    init_loss: Optional[float] = None
    losses: List[float] = []
    gen.eval()

    with ThreadPoolExecutor(max_workers=4) as pool:
        qn = min(6, steps)
        futs = [pool.submit(_col_xyz, 910000 + i) for i in range(qn)]
        nxt = qn
        for st in range(1, steps + 1):
            if st % 100 == 0:
                ds.set_grid_epoch(st // 100)
            sl = (st - 1) % len(futs)
            pk, kp = futs[sl].result()
            if nxt < steps:
                futs[sl] = pool.submit(_col_xyz, 910000 + nxt)
                nxt += 1
            if pk is None or kp is None:
                sched.step()
                continue

            bsz_cur = len(kp)
            tn = gen.normalize_targets(equil_targets[kp])
            ma = _sample_synth_sft_masks(bsz_cur, rng, device)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in pk.items()}
            bd["decoder_phy_pos"] = bd["decoder_phy_pos"].float()

            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                dec_in, tgt, _ = gen.backbone.get_feat_from_dataloader(bd)
                cu = bd["decoder_input_len"].to(device, dtype=torch.int32)
                max_l = int(bd["decoder_input_max_len"].item())
                if gen.budget_emb is not None:
                    root_idx = cu[:-1].long()
                    root_tot = tgt.count[root_idx].to(dec_in.remaining_atoms.dtype)
                    owners_tok = torch.repeat_interleave(
                        torch.arange(bsz_cur, device=device), torch.diff(cu).long()
                    )
                    if gen.non_trapping_budget:
                        non_cls = dec_in.level >= gen.backbone.atom_level_index
                        dec_in.remaining_atoms = torch.where(
                            non_cls, root_tot[owners_tok], torch.zeros_like(dec_in.remaining_atoms)
                        )
                    else:
                        rem_a_m = dec_in.remaining_atoms.clone()
                        rem_a_m[root_idx] = root_tot
                        dec_in.remaining_atoms = rem_a_m
                hid_atom = gen.forward_sequence(dec_in, cu, max_l, tn, ma)[tgt.is_gen_atom].detach()

            labels_a = [v.long() for v in tgt.to_list(tgt.is_gen_atom)]
            real_atom = (labels_a[0] >= 0) & (labels_a[0] < len(atom_list))
            null_atom = ~real_atom
            if not bool(real_atom.any().item()):
                sched.step()
                continue

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits_a = ahead(hid_atom, labels_a)
                loss_real = (
                    F.cross_entropy(logits_a[1].float()[real_atom], labels_a[1][real_atom])
                    + F.cross_entropy(logits_a[2].float()[real_atom], labels_a[2][real_atom])
                    + F.cross_entropy(logits_a[3].float()[real_atom], labels_a[3][real_atom])
                ) / 3.0
                if bool(null_atom.any().item()):
                    loss_null = (
                        F.cross_entropy(logits_a[1].float()[null_atom], labels_a[1][null_atom])
                        + F.cross_entropy(logits_a[2].float()[null_atom], labels_a[2][null_atom])
                        + F.cross_entropy(logits_a[3].float()[null_atom], labels_a[3][null_atom])
                    ) / 3.0
                    loss_xyz = loss_real + 0.25 * loss_null
                else:
                    loss_xyz = loss_real

            if torch.isfinite(loss_xyz):
                opt.zero_grad(set_to_none=True)
                loss_xyz.backward()
                torch.nn.utils.clip_grad_norm_(xyz_params, 1.0)
                opt.step()
                lv = float(loss_real.detach().item())
                if init_loss is None:
                    init_loss = lv
                losses.append(lv)
            sched.step()

            if (st % 200 == 0) or (st == steps):
                avg_l = float(np.mean(losses[-200:])) if losses else float("nan")
                print(
                    f"[Step3 XYZ-Distill {st:04d}/{steps:04d}] xyz_ce={avg_l:.4f} (init={init_loss}) ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )

    gen.configure_trainable()
    gen.eval()
    wall = time.perf_counter() - t0
    return {
        "steps": steps,
        "pool_size": len(equil_mols),
        "init_xyz_ce": round(float(init_loss), 4) if init_loss is not None else None,
        "final_xyz_ce": round(float(np.mean(losses[-100:])), 4) if losses else None,
        "wall_sec": round(wall, 2),
        "xyz_distill_gpuh": round(wall / 3600.0, 4),
    }


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
    refiner: Optional[PostTreeGraphRefiner],
    bus_train: Any,
    dev_homo: np.ndarray,
    dev_natoms: np.ndarray,
    sa_target: float,
    cfg_scale: float,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate `gen` (+ optional `refiner`) on `256` dev prompts strictly under `bus_train` (`0` `audit_dpa4` calls)."""
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
    if refiner is not None:
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
    nat_ok = [m.GetNumAtoms() for m, ok in zip(admit["mols"], ev["legal_mask"]) if m is not None and bool(ok)]
    composite = hit_y - 0.30 * mae + 0.15 * legal_r + 0.15 * sa_ok - 0.018 * min(fmax_med, 6.0)
    return {
        "composite": round(composite, 5),
        "dev_synth_hit_yield": round(hit_y, 4),
        "dev_surr_mae_eV": round(mae, 4),
        "dev_legal_rate": round(legal_r, 4),
        "dev_sa_ok_rate": round(sa_ok, 4),
        "dev_fmax_median": round(fmax_med, 3),
        "dev_mean_atoms": round(float(np.mean(nat_ok)), 2) if nat_ok else 0.0,
    }


def run_distill_refiner(args: argparse.Namespace) -> Path:
    """Standalone distillation entrypoint for 4-layer `PostTreeGraphRefiner` (`max_disp_A=0.25`)."""
    device = torch.device(args.device)
    step3_dir = args.work_dir / args.run_name
    step3_dir.mkdir(parents=True, exist_ok=True)
    ref_ckpt = step3_dir / "v4_step3_post_tree_refiner.pt"
    if args.reuse_refiner and ref_ckpt.exists():
        print(f"[Step3 Refiner] Reusing {ref_ckpt}", flush=True)
        return ref_ckpt

    pool_sdf, _ = ensure_relaxed_synth_pool(args)
    eq_mols, _ = _load_prepared_v4_sdf(pool_sdf)
    refiner, meta = _distill_v4_refiner(
        eq_mols, [], epochs=int(args.refiner_epochs), batch_size=int(args.refiner_batch_size), device=device
    )
    refiner.save_checkpoint(ref_ckpt, extra=meta)
    print(f"[Step3 Refiner DONE] {meta} -> {ref_ckpt}", flush=True)
    return ref_ckpt


def run_align_vine_grpo(args: argparse.Namespace) -> Path:
    """Run full Step 3 alignment: Relaxed SFT -> Online Hindsight AWR -> Prelim Refiner -> Tree-Vine GRPO -> Detached XYZ Distill -> Final On-Policy Refiner."""
    device = torch.device(args.device)
    step3_dir = args.work_dir / args.run_name
    step3_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = step3_dir / "v4_step3_final.pt"
    ref_ckpt = step3_dir / "v4_step3_post_tree_refiner.pt"
    if args.reuse_grpo and final_ckpt.exists() and ref_ckpt.exists():
        print(f"[Step3 Align] Reusing {final_ckpt} and {ref_ckpt}", flush=True)
        return final_ckpt

    synth_ckpt = ensure_sft_synth(args)
    pool_sdf = step3_dir / "v4_step3_relaxed_synth_pool.sdf"
    anchor_mols, anchor_targets = _load_prepared_v4_sdf(pool_sdf)

    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    bus_train, cal_unimol2, _ = _build_calibrated_oe62_bus(
        args, include_blind_auditor=False, calib_cache_path=calib_cache
    )
    _, clean_targets = _get_clean_oe62_pool(args, None)
    pool_sub = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 54.0)]

    r_rng = np.random.default_rng(202632)
    dev_idx = r_rng.choice(len(pool_sub), size=256, replace=False)
    dev_homo = pool_sub[dev_idx, 0].astype(np.float32)
    dev_natoms = np.round(pool_sub[dev_idx, 1]).astype(np.float32)

    gen, synth_payload = LoomUni3DARGenerator.load_checkpoint(synth_ckpt, device=args.device, batch_size=128)
    synth_meta = synth_payload.get("metadata") or {}
    sft_raw_dev = _zero_leakage_dev_score(
        gen, None, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
    )
    print(f"[Step3 Baseline Relaxed-SFT (no refiner)] dev={sft_raw_dev}", flush=True)

    # ---------------- Stage 3B: Online Hindsight DPA-4-Relaxed AWR + Real Refiner Pair Harvest ----------------
    t_awr0 = time.perf_counter()
    equil_mols, equil_targets, onpolicy_pairs, hindsight_logs = _run_v4_online_hindsight_awr(
        gen,
        bus_train,
        cal_unimol2,
        anchor_mols,
        anchor_targets,
        clean_targets,
        rounds=int(args.hindsight_rounds),
        proposals_per_round=int(args.hindsight_proposals),
        steps_per_round=int(args.hindsight_steps),
        batch_size=64,
        lr=1.4e-4,
        sa_target=float(args.eval_sa_target),
        cfg_scale=float(args.cfg_scale),
        device=device,
    )
    hindsight_gpuh = (time.perf_counter() - t_awr0) / 3600.0

    # Distill preliminary 4-layer SE(3) PostTreeGraphRefiner (`max_disp_A=0.25`) on equilibrium + Stage-3B real pairs!
    prelim_refiner, prelim_ref_meta = _distill_v4_refiner(
        equil_mols,
        onpolicy_pairs,
        epochs=int(args.refiner_epochs),
        batch_size=int(args.refiner_batch_size),
        device=device,
        seed=202631,
    )
    print(f"[Step3 Prelim Refiner] {prelim_ref_meta}", flush=True)

    base_dev = _zero_leakage_dev_score(
        gen, prelim_refiner, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
    )
    print(f"[Step3 Post-Hindsight + PrelimRefiner] dev={base_dev}", flush=True)

    # ---------------- Stage 3C: Online Tree-Vine GRPO ----------------
    n_trainable = _configure_v4_grpo_trainable(gen, top_k_layers=int(args.grpo_top_layers))
    print(f"[Step3 GRPO] Starting Tree-Vine GRPO | trainable_params={n_trainable:,}", flush=True)

    rep_sel = r_rng.choice(len(equil_mols), size=min(4096, len(equil_mols)), replace=False)
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
        replay_weight=float(args.replay_weight),
        replay_batch_size=16,
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
        post_tree_refiner=prelim_refiner,
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
        if step_num % 40 == 0 or step_num == int(args.grpo_steps):
            cur_dev = _zero_leakage_dev_score(
                ema_gen, prelim_refiner, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
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

    raw_dev = _zero_leakage_dev_score(
        gen, prelim_refiner, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
    )
    print(f"[Step3 GRPO Final Raw Policy Dev] {raw_dev}", flush=True)
    if raw_dev["composite"] > best_dev["composite"]:
        best_dev = raw_dev
        best_state = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}

    gen.load_state_dict(best_state, strict=True)
    gen.eval()

    # ---------------- Stage 3D: Detached Sub-Voxel `0.01 A` XYZ Head Distillation ----------------
    pre_xyz_ckpt = step3_dir / "v4_step3_pre_xyz_tmp.pt"
    gen.save_checkpoint(pre_xyz_ckpt)
    pre_xyz_no_ref_dev = _zero_leakage_dev_score(
        gen, None, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
    )
    xyz_distill_meta = _run_v4_detached_xyz_distillation(
        gen,
        equil_mols,
        equil_targets,
        steps=int(args.xyz_distill_steps),
        batch_size=64,
        lr=3.0e-4,
        device=device,
    )
    post_xyz_no_ref_dev = _zero_leakage_dev_score(
        gen, None, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
    )
    print(
        f"[Step3 XYZ-Distill Dev Check (no refiner)] pre={pre_xyz_no_ref_dev} -> post={post_xyz_no_ref_dev}",
        flush=True,
    )
    if post_xyz_no_ref_dev["composite"] < pre_xyz_no_ref_dev["composite"] - 0.01:
        print("[Step3 XYZ-Distill] Reverting to pre-XYZ checkpoint based on zero-leakage dev composite", flush=True)
        gen, _ = LoomUni3DARGenerator.load_checkpoint(pre_xyz_ckpt, device=args.device, batch_size=128)
        xyz_distill_meta["reverted"] = True
    else:
        xyz_distill_meta["reverted"] = False
    pre_xyz_ckpt.unlink(missing_ok=True)

    # ---------------- Final On-Policy `PostTreeGraphRefiner` Refresh (`max_disp_A=0.25`) ----------------
    _, _, post_pairs, post_col_stats = _collect_and_relax_onpolicy_batch(
        gen,
        bus_train,
        cal_unimol2,
        clean_targets,
        n_proposals=1536,
        sa_target=float(args.eval_sa_target),
        cfg_scale=float(args.cfg_scale),
        max_fmax=2.05,
        seed=9901,
        device=device,
    )
    all_refiner_pairs = list(onpolicy_pairs) + list(post_pairs)
    final_refiner, final_ref_meta = _distill_v4_refiner(
        equil_mols,
        all_refiner_pairs,
        epochs=int(args.refiner_epochs) + 2,
        batch_size=int(args.refiner_batch_size),
        device=device,
        seed=202639,
    )
    final_ref_meta["prelim_refiner_gpuh"] = prelim_ref_meta["refiner_gpuh"]
    final_ref_meta["post_pair_collection_gpuh"] = round(float(post_col_stats["wall_sec"]) / 3600.0, 4)
    final_ref_meta["refiner_gpuh"] = round(
        float(prelim_ref_meta["refiner_gpuh"])
        + float(final_ref_meta["refiner_gpuh"])
        + float(final_ref_meta["post_pair_collection_gpuh"]),
        4,
    )
    final_refiner.save_checkpoint(ref_ckpt, extra=final_ref_meta)

    final_dev = _zero_leakage_dev_score(
        gen, final_refiner, bus_train, dev_homo, dev_natoms, args.eval_sa_target, args.cfg_scale, device
    )
    print(f"[Step3 Final Dev (with On-Policy 4-Layer Refiner)] {final_dev}", flush=True)

    gen.save_checkpoint(
        final_ckpt,
        extra_metadata={
            "sft_metadata": synth_meta.get("step2_metadata") or {},
            "synth_sft_metadata": synth_meta,
            "hindsight_awr_gpuh": round(hindsight_gpuh, 4),
            "hindsight_logs": hindsight_logs,
            "xyz_distill_metadata": xyz_distill_meta,
            "refiner_metadata": final_ref_meta,
            "grpo_steps": int(args.grpo_steps),
            "m_trunks": int(args.m_trunks),
            "k_leaves": int(args.k_leaves),
            "grpo_wall_sec": round(grpo_wall, 2),
            "grpo_gpuh": round(grpo_gpuh, 4),
            "sft_raw_dev": sft_raw_dev,
            "baseline_post_hindsight_dev": base_dev,
            "selected_grpo_dev": best_dev,
            "final_step3_dev": final_dev,
            "grpo_tail_metrics": history[-1] if history else {},
        },
    )
    (step3_dir / "v4_step3_grpo_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[Step3 ALIGN DONE] Saved {final_ckpt} & {ref_ckpt} | final_dev={final_dev}", flush=True)
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

    synth_sft_gpuh = float((step3_meta.get("synth_sft_metadata") or {}).get("synth_sft_gpuh", 0.0))
    hindsight_gpuh = float(step3_meta.get("hindsight_awr_gpuh", 0.0))
    xyz_gpuh = float((step3_meta.get("xyz_distill_metadata") or {}).get("xyz_distill_gpuh", 0.0))
    ref_gpuh = float(ref_meta.get("refiner_gpuh", 0.0))
    grpo_gpuh = float(step3_meta.get("grpo_gpuh", 0.0))
    crit_gpuh = float(critic_meta.get("rollout_gpuh", 0.0))
    total_gpuh = round(synth_sft_gpuh + hindsight_gpuh + xyz_gpuh + ref_gpuh + grpo_gpuh + crit_gpuh, 4)

    fair_payload = {
        "experiment": "OE62 Fair Fixed-Prompt 30.4-Atom Benchmark — Uni-Loom [Uni-3DAR v4: Full-SO(3) DPA-4-Equil SFT + Hindsight AWR + Tree-Vine GRPO + Detached XYZ Distill + 4-Layer SE(3) PostTreeGraphRefiner + TwoStage 32->8->2]",
        "benchmark_seed": 20261118,
        "n_prompts": n_p30,
        "mean_requested_atoms": round(float(np.mean(req_30[:, 1])), 2),
        "total_training_gpuh": total_gpuh,
        "gpuh_breakdown": {
            "step3_relaxed_synth_sft_gpuh": round(synth_sft_gpuh, 4),
            "step3_online_hindsight_awr_gpuh": round(hindsight_gpuh, 4),
            "step3_vine_grpo_gpuh": round(grpo_gpuh, 4),
            "step3_detached_xyz_distill_gpuh": round(xyz_gpuh, 4),
            "step3_onpolicy_refiner_gpuh": round(ref_gpuh, 4),
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
                f"Legal={b_res['legal_rate']*100:.1f}% Fmax={b_res['zero_step_f_max_median_eV_A']:.2f} | "
                f"BlindSynth={b_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}% | "
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
        choices=[
            "relax_synth_shard",
            "sft_synth_ddp",
            "distill_refiner",
            "align_vine_grpo",
            "rollout_shard",
            "eval_final",
            "all",
        ],
        required=True,
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rollout-gpus", type=str, default="0,1")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--relax-pool-cap", type=int, default=9600)
    p.add_argument("--synth-sft-epochs", type=int, default=12)
    p.add_argument("--synth-max-sa", type=float, default=3.30)
    p.add_argument("--hindsight-rounds", type=int, default=2)
    p.add_argument("--hindsight-proposals", type=int, default=2048)
    p.add_argument("--hindsight-steps", type=int, default=240)
    p.add_argument("--xyz-distill-steps", type=int, default=600)
    p.add_argument("--refiner-epochs", type=int, default=14)
    p.add_argument("--refiner-batch-size", type=int, default=160)
    p.add_argument("--grpo-steps", type=int, default=150)
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
    p.add_argument("--reuse-relaxed-pool", action="store_true")
    p.add_argument("--reuse-synth-sft", action="store_true")
    p.add_argument("--reuse-refiner", action="store_true")
    p.add_argument("--reuse-grpo", action="store_true")
    p.add_argument("--reuse-critic", action="store_true")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="uni3dar_v4_step3_clean")
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

    if args.phase == "relax_synth_shard":
        run_relax_synth_shard(args)
    elif args.phase == "sft_synth_ddp":
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
