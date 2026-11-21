"""Multi-node 16xA100 DDP pretraining & BF16-vs-TF32 precision audit for general FlowMol."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import RDLogger
import torch
import torch.distributed as dist

RDLogger.DisableLog("rdApp.*")
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from loom.flowmol import (
    DynamicEdgeBudgetBatchSampler,
    FlatMoleculeCorpus,
    FlowMolConfig,
    FlowMolModel,
    collate_flowmol_batch,
    evaluate_sampled_molecules,
)


def _setup_dist() -> Tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group(backend="nccl")
        return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, dev


def _cleanup_dist() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def _reduce_dict(metrics: Dict[str, float], device: torch.device, world_size: int) -> Dict[str, float]:
    if world_size <= 1 or not dist.is_initialized():
        return metrics
    keys = sorted(metrics.keys())
    vec = torch.tensor([float(metrics[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(vec, op=dist.ReduceOp.AVG)
    return {k: float(vec[i].item()) for i, k in enumerate(keys)}


@torch.no_grad()
def _update_ema(ema_model: FlowMolModel, model: FlowMolModel, step: int, max_decay: float) -> None:
    decay = min(max_decay, (1.0 + step) / (10.0 + step))
    one_minus = 1.0 - decay
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data.float(), alpha=one_minus)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.data.copy_(b.data)


def _lr_at_step(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr_ratio: float = 0.05) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(warmup_steps, 1))
    progress = min(1.0, float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _sample_eval_sizes(meta: Dict[str, Any], n_mols: int, seed: int, domain: Optional[str] = None) -> List[int]:
    return FlowMolModel.sample_empirical_sizes(meta, n_mols=n_mols, domain=domain, seed=seed, default_size=30)


@torch.no_grad()
def evaluate_validation(
    model: FlowMolModel,
    val_loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
    *,
    source_names: Sequence[str] = (),
    use_bf16: bool,
    max_batches: int = 40,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    totals: Dict[str, float] = {"loss_total": 0.0, "loss_x": 0.0, "loss_a": 0.0, "loss_c": 0.0, "loss_e": 0.0}
    n_src = len(source_names)
    src_loss_sum = np.zeros(max(n_src, 1), dtype=np.float64)
    src_mol_cnt = np.zeros(max(n_src, 1), dtype=np.float64)
    n_batches = 0
    amp_enabled = bool(use_bf16 and device.type == "cuda")
    fork_devs = [device.index] if (device.type == "cuda" and device.index is not None) else []
    for batch in val_loader:
        batch = batch.to(device)
        do_sc = bool(model.config.self_conditioning and ((n_batches % 2) == 1))
        with torch.random.fork_rng(devices=fork_devs, enabled=True):
            torch.manual_seed(100_000 + rank * 1009 + n_batches)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                per_mol, m = model.compute_loss(
                    batch,
                    use_self_conditioning=do_sc,
                    apply_distortion=False,
                    return_per_molecule=True,
                )
        for k in totals:
            totals[k] += float(m[k].item())
        if n_src > 0 and batch.source_id is not None:
            sid_np = batch.source_id.detach().cpu().numpy()
            pm_np = per_mol.detach().float().cpu().numpy()
            for sid in range(n_src):
                sm = sid_np == sid
                if sm.any():
                    src_loss_sum[sid] += float(pm_np[sm].sum())
                    src_mol_cnt[sid] += float(sm.sum())
        n_batches += 1
        if n_batches >= max_batches:
            break
    model.train(was_training)
    denom = max(n_batches, 1)
    avg = _reduce_dict({f"val_{k}": v / denom for k, v in totals.items()}, device, world_size)
    if n_src > 0:
        src_vec = torch.tensor(np.concatenate([src_loss_sum, src_mol_cnt]), dtype=torch.float64, device=device)
        if world_size > 1 and dist.is_initialized():
            dist.all_reduce(src_vec, op=dist.ReduceOp.SUM)
        src_vec_np = src_vec.cpu().numpy()
        sums_all, cnts_all = src_vec_np[:n_src], src_vec_np[n_src:]
        for sid, sname in enumerate(source_names):
            if cnts_all[sid] > 0:
                avg[f"val_loss_src_{sname}"] = float(sums_all[sid] / cnts_all[sid])
    return avg


@torch.no_grad()
def evaluate_generation(
    ema_model: FlowMolModel,
    meta: Dict[str, Any],
    device: torch.device,
    *,
    allowed_elements: Optional[Tuple[str, ...]] = None,
    n_mols: int = 128,
    use_bf16: bool = True,
    seed: int = 2026,
) -> Dict[str, Dict[str, float]]:
    eval_specs: List[Tuple[str, str, int, str, Optional[Sequence[str]], Optional[str]]] = [
        ("open118_heun_25", "heun", 25, "quadratic_end", None, None),
        ("active_heun_25", "heun", 25, "quadratic_end", allowed_elements, None),
        ("active_euler_100", "euler", 100, "uniform", allowed_elements, None),
    ]
    per_src_hist = meta.get("per_source_n_atoms_hist") or {}
    per_src_elems = meta.get("per_source_active_elements") or {}
    if "tmqm" in per_src_hist and per_src_hist["tmqm"]:
        eval_specs.append(
            ("tmqm_heun_25", "heun", 25, "quadratic_end", per_src_elems.get("tmqm") or allowed_elements, "tmqm")
        )

    report: Dict[str, Dict[str, float]] = {}
    for spec_idx, (label, solver, n_steps, grid, elem_allow, domain) in enumerate(eval_specs):
        sizes = _sample_eval_sizes(meta, n_mols=n_mols, seed=seed + spec_idx * 97, domain=domain)
        t0 = time.perf_counter()
        all_pairs = []
        for i in range(0, len(sizes), 32):
            sub_sizes = sizes[i : i + 32]
            samples = ema_model.sample(
                sub_sizes,
                device=device,
                n_steps=n_steps,
                solver=solver,
                grid_type=grid,
                allowed_elements=elem_allow,
                use_bf16=use_bf16,
                seed=seed + spec_idx * 1000 + i,
                build_rdkit=True,
            )
            all_pairs.extend([(s["rdkit_mol"], s["diag"]) for s in samples])
        dt = max(time.perf_counter() - t0, 1e-6)
        stats = evaluate_sampled_molecules(all_pairs)
        stats["mol_per_sec_1gpu"] = round(len(all_pairs) / dt, 2)
        report[label] = stats
    return report


def _save_trisol_checkpoint(
    out_dir: Path,
    step: int,
    epoch: int,
    step_in_epoch: int,
    samples_seen: int,
    best_val: float,
    raw_model: FlowMolModel,
    ema_model: FlowMolModel,
    optimizer: torch.optim.Optimizer,
    cfg: FlowMolConfig,
    meta: Dict[str, Any],
) -> Path:
    ckpt_dir = out_dir / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_dir / "checkpoint.pt.tmp"
    final_path = ckpt_dir / "checkpoint.pt"
    payload = {
        "format": "flowmol_foundation_ckpt_v1",
        "step": step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "samples_seen": samples_seen,
        "best_val_loss": best_val,
        "config": cfg.to_dict(),
        "corpus_meta": {k: v for k, v in meta.items() if k not in ("n_atoms_hist", "per_source_n_atoms_hist")},
        "n_atoms_hist": meta.get("n_atoms_hist", {}),
        "per_source_n_atoms_hist": meta.get("per_source_n_atoms_hist", {}),
        "per_source_active_elements": meta.get("per_source_active_elements", {}),
        "model_state_dict": raw_model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    # Touch Trisol completion marker strictly AFTER checkpoint is atomically visible
    (ckpt_dir / "COMMITTED").write_text(f"step={step}\nepoch={epoch}\nsamples_seen={samples_seen}\n")
    return ckpt_dir


def _save_inference_weights(
    path: Path,
    step: int,
    samples_seen: int,
    ema_model: FlowMolModel,
    cfg: FlowMolConfig,
    meta: Dict[str, Any],
) -> None:
    del cfg
    ema_model.save_pretrained(path, step=step, samples_seen=samples_seen, corpus_meta=meta)


class _FlowMolLossModule(torch.nn.Module):
    """DDP-safe wrapper so forward() invokes compute_loss() and fires all DDP reduction hooks."""

    def __init__(self, model: FlowMolModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch, *, use_self_conditioning: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.model.compute_loss(batch, use_self_conditioning=use_self_conditioning)


def _run_training_loop(
    args: argparse.Namespace,
    cfg: FlowMolConfig,
    train_ds: FlatMoleculeCorpus,
    val_ds: FlatMoleculeCorpus,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    *,
    use_bf16: bool,
    max_steps: int,
    target_samples: int,
    run_tag: str,
    out_dir: Path,
    enable_checkpoints: bool,
) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    raw_model = FlowMolModel(cfg).to(device)
    ema_model = copy.deepcopy(raw_model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    if _is_main(rank):
        pcount = raw_model.count_parameters()
        print(
            f"[{run_tag}] preset={args.preset} params={pcount['total']:,} world_size={world_size} "
            f"bf16={use_bf16} max_edges={args.max_edges} target_samples={target_samples:,} max_steps={max_steps:,}",
            flush=True,
        )

    fused_ok = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=fused_ok,
    )

    global_step = 0
    start_epoch = 0
    resume_skip_batches = 0
    samples_seen = 0
    best_val_loss = float("inf")

    # Handle Trisol resume or explicit --resume
    resume_path: Optional[Path] = None
    if args.resume:
        resume_path = Path(args.resume)
    elif os.environ.get("TRISOL_RESUME", "").lower() == "true" and os.environ.get("TRISOL_RESUME_CHECKPOINT"):
        cand = Path(os.environ["TRISOL_RESUME_CHECKPOINT"]) / "checkpoint.pt"
        if cand.exists():
            resume_path = cand

    if resume_path is not None and resume_path.exists():
        if _is_main(rank):
            print(f"[{run_tag}] resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = int(ckpt["step"])
        start_epoch = int(ckpt["epoch"])
        resume_skip_batches = int(ckpt.get("step_in_epoch", 0))
        samples_seen = int(ckpt["samples_seen"])
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    loss_module = _FlowMolLossModule(raw_model)
    ddp_model: torch.nn.Module = (
        DDP(loss_module, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        if world_size > 1
        else loss_module
    )

    train_sampler = DynamicEdgeBudgetBatchSampler(
        train_ds.n_atoms_view,
        max_edges_per_batch=args.max_edges,
        max_mols_per_batch=args.max_mols_per_batch,
        min_mols_per_batch=2,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        shuffle=True,
    )
    val_sampler = DynamicEdgeBudgetBatchSampler(
        val_ds.n_atoms_view,
        max_edges_per_batch=args.max_edges,
        max_mols_per_batch=args.max_mols_per_batch,
        min_mols_per_batch=1,
        rank=rank,
        world_size=world_size,
        seed=args.seed + 999,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_flowmol_batch,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=max(1, args.num_workers // 2) if args.num_workers > 0 else 0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_flowmol_batch,
    )

    # Estimate total steps if driven by target_samples
    if max_steps <= 0:
        total_epoch_steps = max(train_sampler.total_steps_in_epoch(), 1)
        avg_mols_per_step = max(1.0, (len(train_ds) / total_epoch_steps))
        est_steps = int(math.ceil(target_samples / avg_mols_per_step))
        effective_total_steps = max(est_steps, 1000)
    else:
        effective_total_steps = max_steps

    source_names = tuple(train_ds.meta.get("source_names", ()))

    metrics_file = out_dir / f"metrics_{run_tag}.jsonl"
    if _is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")

    amp_enabled = bool(use_bf16 and device.type == "cuda")
    t_run_start = time.perf_counter()
    t_log_start = t_run_start
    mols_since_log = 0
    nan_skips = 0
    last_log_metrics: Dict[str, float] = {}
    recent_grad_norms: List[float] = []

    epoch = start_epoch
    step_in_epoch = resume_skip_batches
    done = False
    while not done:
        skip_now = resume_skip_batches if epoch == start_epoch else 0
        train_sampler.set_epoch(epoch, skip_batches=skip_now)
        step_in_epoch = skip_now
        ddp_model.train()
        for batch in train_loader:
            if (max_steps > 0 and global_step >= max_steps) or (target_samples > 0 and samples_seen >= target_samples):
                done = True
                break
            step_in_epoch += 1

            lr = _lr_at_step(global_step, args.warmup_steps, effective_total_steps, args.lr, args.min_lr_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Synchronize self-conditioning pass across all DDP ranks to eliminate 1-pass vs 2-pass stragglers
            do_sc = bool(cfg.self_conditioning and ((global_step % 2) == 1))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled, cache_enabled=False):
                loss, step_m = ddp_model(batch, use_self_conditioning=do_sc)

            # Synchronize finiteness across all ranks; if non-finite, run zeroed backward to flush DDP reducer hooks!
            finite_flag = torch.tensor([1 if torch.isfinite(loss).item() else 0], dtype=torch.int32, device=device)
            if world_size > 1:
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if finite_flag.item() == 0:
                nan_skips += 1
                (torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0).backward()
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip).item())
            gn_ok = torch.tensor([1 if math.isfinite(grad_norm) else 0], dtype=torch.int32, device=device)
            if world_size > 1:
                dist.all_reduce(gn_ok, op=dist.ReduceOp.MIN)
            if gn_ok.item() == 0:
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            global_step += 1
            _update_ema(ema_model, raw_model, global_step, args.ema_decay)

            # Exact global molecule count across all ranks
            local_mols = torch.tensor([batch.batch_size], dtype=torch.int64, device=device)
            if world_size > 1:
                dist.all_reduce(local_mols, op=dist.ReduceOp.SUM)
            step_mols = int(local_mols.item())
            samples_seen += step_mols
            mols_since_log += step_mols
            recent_grad_norms.append(grad_norm)
            if len(recent_grad_norms) > 200:
                recent_grad_norms.pop(0)

            if global_step % args.log_every == 0 or global_step == 1:
                reduced = _reduce_dict(
                    {k: float(v.item()) for k, v in step_m.items()} | {"grad_norm": grad_norm},
                    device,
                    world_size,
                )
                dt = max(time.perf_counter() - t_log_start, 1e-6)
                mol_per_sec = mols_since_log / dt
                t_log_start = time.perf_counter()
                mols_since_log = 0
                row = {
                    "run": run_tag,
                    "step": global_step,
                    "epoch": epoch,
                    "samples_seen": samples_seen,
                    "lr": lr,
                    "mol_per_sec": round(mol_per_sec, 1),
                    "nan_skips": nan_skips,
                    "elapsed_sec": round(time.perf_counter() - t_run_start, 1),
                    **{k: round(v, 5) for k, v in reduced.items()},
                }
                last_log_metrics = row
                if _is_main(rank):
                    print(
                        f"[{run_tag}] step={global_step:6d} samples={samples_seen/1e6:6.2f}M "
                        f"tot={reduced['loss_total']:.4f} x={reduced['loss_x']:.4f} "
                        f"a={reduced['loss_a']:.4f} c={reduced['loss_c']:.4f} e={reduced['loss_e']:.4f} "
                        f"gn={reduced['grad_norm']:.3f} lr={lr:.2e} thr={mol_per_sec:.0f} mol/s",
                        flush=True,
                    )
                    with metrics_file.open("a") as mf:
                        mf.write(json.dumps(row) + "\n")

            if args.val_every > 0 and global_step % args.val_every == 0:
                val_m = evaluate_validation(
                    ema_model, val_loader, device, rank, world_size,
                    source_names=source_names, use_bf16=use_bf16,
                )
                gen_m: Dict[str, Any] = {}
                if _is_main(rank) and args.sample_eval_mols > 0:
                    gen_m = evaluate_generation(
                        ema_model,
                        train_ds.meta,
                        device,
                        allowed_elements=train_ds.active_elements,
                        n_mols=args.sample_eval_mols,
                        use_bf16=use_bf16,
                        seed=args.seed + global_step,
                    )
                if dist.is_initialized():
                    dist.barrier()
                if _is_main(rank):
                    vtot = val_m["val_loss_total"]
                    print(f"[{run_tag}][VAL] step={global_step} {json.dumps(val_m)} gen={json.dumps(gen_m)}", flush=True)
                    with metrics_file.open("a") as mf:
                        mf.write(json.dumps({"event": "val", "step": global_step, "samples_seen": samples_seen, **val_m, "gen": gen_m}) + "\n")
                    if enable_checkpoints and vtot < best_val_loss:
                        best_val_loss = vtot
                        _save_inference_weights(out_dir / "best_ema.pt", global_step, samples_seen, ema_model, cfg, train_ds.meta)

            if enable_checkpoints and args.save_every > 0 and global_step % args.save_every == 0:
                if _is_main(rank):
                    cdir = _save_trisol_checkpoint(
                        out_dir, global_step, epoch, step_in_epoch, samples_seen, best_val_loss,
                        raw_model, ema_model, optimizer, cfg, train_ds.meta,
                    )
                    print(f"[{run_tag}][CKPT] saved & committed {cdir}", flush=True)
                if dist.is_initialized():
                    dist.barrier()

        epoch += 1
        step_in_epoch = 0

    # Final validation, sampling, and artifact export
    val_final = evaluate_validation(
        ema_model, val_loader, device, rank, world_size,
        source_names=source_names, use_bf16=use_bf16, max_batches=60,
    )
    gen_final: Dict[str, Any] = {}
    if _is_main(rank) and args.sample_eval_mols > 0:
        gen_final = evaluate_generation(
            ema_model,
            train_ds.meta,
            device,
            allowed_elements=train_ds.active_elements,
            n_mols=max(args.sample_eval_mols, 64),
            use_bf16=use_bf16,
            seed=args.seed + 77777,
        )
    if dist.is_initialized():
        dist.barrier()

    total_elapsed = time.perf_counter() - t_run_start
    summary = {
        "run_tag": run_tag,
        "use_bf16": use_bf16,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "elapsed_sec": round(total_elapsed, 2),
        "avg_mol_per_sec": round(samples_seen / max(total_elapsed, 1e-6), 1),
        "nan_skips": nan_skips,
        "median_grad_norm_tail": round(float(np.median(recent_grad_norms)) if recent_grad_norms else 0.0, 4),
        "last_train": last_log_metrics,
        "final_val": val_final,
        "final_gen": gen_final,
    }

    if _is_main(rank) and enable_checkpoints:
        _save_trisol_checkpoint(
            out_dir, global_step, epoch, step_in_epoch, samples_seen, best_val_loss,
            raw_model, ema_model, optimizer, cfg, train_ds.meta,
        )
        _save_inference_weights(out_dir / "final_ema.pt", global_step, samples_seen, ema_model, cfg, train_ds.meta)
        (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if dist.is_initialized():
        dist.barrier()
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="16xA100 DDP pretraining & precision audit for general FlowMol.")
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path(os.environ.get("TRISOL_OUTPUT_DIR", "/trisol/output")))
    p.add_argument("--preset", type=str, default="standard", choices=["tiny", "small", "standard", "flagship"])
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--max-edges", type=int, default=110_000)
    p.add_argument("--max-mols-per-batch", type=int, default=192)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--target-samples", type=int, default=80_000_000)
    p.add_argument("--max-steps", type=int, default=0, help="If >0, stops at max_steps (overrides target_samples)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--sample-eval-mols", type=int, default=64)
    p.add_argument("--source-weights", type=str, default="", help="JSON dict e.g. '{\"oe62\": 4.0, \"qm9\": 2.0}'")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision-audit-steps", type=int, default=0, help="Run Roadmap 3.1 TF32 vs BF16+FP32 audit for N steps")
    args = p.parse_args()

    rank, local_rank, world_size, device = _setup_dist()
    try:
        src_weights = json.loads(args.source_weights) if args.source_weights.strip() else None
        train_ds = FlatMoleculeCorpus(
            args.corpus_dir,
            split="train",
            max_atoms=args.max_atoms,
            source_weights=src_weights,
            load_in_memory=True,
            seed=args.seed,
        )
        val_ds = FlatMoleculeCorpus(
            args.corpus_dir,
            split="val",
            max_atoms=args.max_atoms,
            load_in_memory=True,
            seed=args.seed + 1,
        )
        if _is_main(rank):
            print(f"[data] loaded train={len(train_ds):,} val={len(val_ds):,} sources={train_ds.meta.get('source_counts')}", flush=True)

        cfg = FlowMolConfig.preset(args.preset, grad_checkpointing=args.grad_checkpointing)

        if args.precision_audit_steps > 0:
            # Roadmap Section 3.1 mandatory 2,000-step pure TF32 vs BF16+local FP32 admission gate
            steps = args.precision_audit_steps
            tf32_res = _run_training_loop(
                args, cfg, train_ds, val_ds, rank, local_rank, world_size, device,
                use_bf16=False, max_steps=steps, target_samples=0,
                run_tag="audit_tf32", out_dir=args.output_dir, enable_checkpoints=False,
            )
            bf16_res = _run_training_loop(
                args, cfg, train_ds, val_ds, rank, local_rank, world_size, device,
                use_bf16=True, max_steps=steps, target_samples=0,
                run_tag="audit_bf16", out_dir=args.output_dir, enable_checkpoints=False,
            )
            if _is_main(rank):
                dx = abs(bf16_res["final_val"]["val_loss_x"] - tf32_res["final_val"]["val_loss_x"])
                de = abs(bf16_res["final_val"]["val_loss_e"] - tf32_res["final_val"]["val_loss_e"])
                speedup = bf16_res["avg_mol_per_sec"] / max(tf32_res["avg_mol_per_sec"], 1e-6)
                audit = {
                    "steps": steps,
                    "preset": args.preset,
                    "world_size": world_size,
                    "tf32": tf32_res,
                    "bf16_fp32_whitelist": bf16_res,
                    "abs_diff_val_loss_x": round(dx, 5),
                    "abs_diff_val_loss_e": round(de, 5),
                    "bf16_speedup": round(speedup, 3),
                    "passed": bool(dx < 0.05 and de < 0.05 and bf16_res["nan_skips"] == 0),
                }
                (args.output_dir / "precision_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
                print("PRECISION_AUDIT:", json.dumps(audit, indent=2), flush=True)
        else:
            _run_training_loop(
                args, cfg, train_ds, val_ds, rank, local_rank, world_size, device,
                use_bf16=True, max_steps=args.max_steps, target_samples=args.target_samples,
                run_tag=f"flowmol_{args.preset}", out_dir=args.output_dir, enable_checkpoints=True,
            )
    finally:
        _cleanup_dist()


if __name__ == "__main__":
    main()
