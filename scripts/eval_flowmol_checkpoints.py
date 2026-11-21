"""Parallel multi-GPU large-sample (N>=1024) validation & generation benchmark for FlowMol-118 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import RDLogger
import torch
from torch.utils.data import DataLoader

from loom.flowmol import (
    DynamicEdgeBudgetBatchSampler,
    FlatMoleculeCorpus,
    FlowMolModel,
    collate_flowmol_batch,
    evaluate_sampled_molecules,
)

RDLogger.DisableLog("rdApp.*")


def _setup_worker() -> Tuple[int, int, int, torch.device]:
    """Use torchrun env vars WITHOUT initializing NCCL so unequal/idle ranks never hit NCCL barrier timeouts."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")
        return rank, local_rank, world_size, torch.device("cpu")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, dev


@torch.no_grad()
def _eval_single_gpu_val(
    model: FlowMolModel,
    val_loader: DataLoader,
    device: torch.device,
    source_names: Sequence[str],
    *,
    use_bf16: bool = True,
    max_batches: int = 300,
    seed: int = 202611,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {"loss_total": 0.0, "loss_x": 0.0, "loss_a": 0.0, "loss_c": 0.0, "loss_e": 0.0}
    n_src = len(source_names)
    src_sum = np.zeros(max(n_src, 1), dtype=np.float64)
    src_cnt = np.zeros(max(n_src, 1), dtype=np.float64)
    amp_enabled = bool(use_bf16 and device.type == "cuda")
    fork_devs = [device.index] if (device.type == "cuda" and device.index is not None) else []
    n_batches = 0
    total_mols = 0

    for batch in val_loader:
        batch = batch.to(device)
        do_sc = bool(model.config.self_conditioning and ((n_batches % 2) == 1))
        with torch.random.fork_rng(devices=fork_devs, enabled=True):
            torch.manual_seed(seed + n_batches * 13)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                per_mol, m = model.compute_loss(
                    batch,
                    use_self_conditioning=do_sc,
                    apply_distortion=False,
                    return_per_molecule=True,
                )
        for k in totals:
            totals[k] += float(m[k].item())
        total_mols += int(batch.batch_size)
        if n_src > 0 and batch.source_id is not None:
            sid_np = batch.source_id.detach().cpu().numpy()
            pm_np = per_mol.detach().float().cpu().numpy()
            for sid in range(n_src):
                sm = sid_np == sid
                if sm.any():
                    src_sum[sid] += float(pm_np[sm].sum())
                    src_cnt[sid] += float(sm.sum())
        n_batches += 1
        if n_batches >= max_batches:
            break

    denom = max(n_batches, 1)
    out = {f"val_{k}": round(v / denom, 6) for k, v in totals.items()}
    out["val_mols_evaluated"] = float(total_mols)
    for sid, sname in enumerate(source_names):
        if src_cnt[sid] > 0:
            out[f"val_loss_src_{sname}"] = round(float(src_sum[sid] / src_cnt[sid]), 6)
    return out


@torch.no_grad()
def _eval_single_gpu_generation(
    model: FlowMolModel,
    corpus_meta: Dict[str, Any],
    active_elements: Tuple[str, ...],
    device: torch.device,
    *,
    n_mols: int = 1024,
    batch_chunk: int = 64,
    use_bf16: bool = True,
    seed: int = 7777,
) -> Dict[str, Dict[str, float]]:
    per_src_hist = corpus_meta.get("per_source_n_atoms_hist") or {}
    per_src_elems = corpus_meta.get("per_source_active_elements") or {}

    suites: List[Tuple[str, str, int, str, Optional[Sequence[str]], Optional[str]]] = [
        ("open118_heun_25", "heun", 25, "quadratic_end", None, None),
        ("active_heun_25", "heun", 25, "quadratic_end", active_elements, None),
        ("active_euler_100", "euler", 100, "uniform", active_elements, None),
    ]
    for dom in ("qm9", "geom_drugs", "oe62", "tmqm"):
        if dom in per_src_hist and per_src_hist[dom]:
            suites.append(
                (f"{dom}_heun_25", "heun", 25, "quadratic_end", per_src_elems.get(dom) or active_elements, dom)
            )

    report: Dict[str, Dict[str, float]] = {}
    for s_idx, (label, solver, n_steps, grid, elem_allow, domain) in enumerate(suites):
        sizes = FlowMolModel.sample_empirical_sizes(
            corpus_meta, n_mols=n_mols, domain=domain, seed=seed + s_idx * 101
        )
        # Sort sizes for near-uniform edge load per sub-batch (eliminates padding/tail skew, identically distributed!)
        sizes_sorted = sorted(sizes)
        t0 = time.perf_counter()
        all_pairs = []
        for i in range(0, len(sizes_sorted), batch_chunk):
            sub_sizes = sizes_sorted[i : i + batch_chunk]
            samples = model.sample(
                sub_sizes,
                device=device,
                n_steps=n_steps,
                solver=solver,
                grid_type=grid,
                allowed_elements=elem_allow,
                use_bf16=use_bf16,
                seed=seed + s_idx * 10_000 + i,
                build_rdkit=True,
            )
            all_pairs.extend([(s["rdkit_mol"], s["diag"]) for s in samples])
        dt = max(time.perf_counter() - t0, 1e-6)
        stats = evaluate_sampled_molecules(all_pairs)
        stats["mol_per_sec_1gpu"] = round(len(all_pairs) / dt, 2)
        stats["elapsed_sec"] = round(dt, 2)
        report[label] = stats
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Parallel multi-GPU N>=1024 benchmark for FlowMol checkpoints.")
    p.add_argument("--candidates-dir", type=Path, required=True)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-mols", type=int, default=1024)
    p.add_argument("--batch-chunk", type=int, default=64)
    p.add_argument("--val-batches", type=int, default=300)
    p.add_argument("--max-edges", type=int, default=95_000)
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--seed", type=int, default=202611)
    args = p.parse_args()

    rank, _, world_size, device = _setup_worker()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_files = sorted(args.candidates_dir.glob("*.pt"))
    if not ckpt_files:
        raise RuntimeError(f"No .pt candidate checkpoints found in {args.candidates_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assigned = [ck for idx, ck in enumerate(ckpt_files) if (idx % world_size) == rank]
    if not assigned and rank != 0:
        print(f"[rank {rank}/{world_size}] idle (0 assigned checkpoints), exiting cleanly.", flush=True)
        return

    val_ds = FlatMoleculeCorpus(
        args.corpus_dir,
        split="val",
        max_atoms=args.max_atoms,
        load_in_memory=True,
        seed=args.seed,
    )
    val_sampler = DynamicEdgeBudgetBatchSampler(
        val_ds.n_atoms_view,
        max_edges_per_batch=args.max_edges,
        max_mols_per_batch=192,
        min_mols_per_batch=1,
        rank=0,
        world_size=1,
        seed=args.seed + 999,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_flowmol_batch,
    )
    source_names = tuple(val_ds.meta.get("source_names", ()))
    print(
        f"[rank {rank}/{world_size}] evaluating {len(assigned)}/{len(ckpt_files)} checkpoints: {[c.name for c in assigned]}",
        flush=True,
    )

    for ck_path in assigned:
        t0 = time.perf_counter()
        model, ck_meta = FlowMolModel.from_pretrained(ck_path, device=device, use_ema=True, strict=True)
        step = int(ck_meta.get("step", 0))
        samples_seen = int(ck_meta.get("samples_seen", 0))
        print(f"[rank {rank}] START {ck_path.name} (step={step:,}, samples={samples_seen:,})", flush=True)

        val_metrics = _eval_single_gpu_val(
            model,
            val_loader,
            device,
            source_names,
            use_bf16=True,
            max_batches=args.val_batches,
            seed=args.seed,
        )
        gen_metrics = _eval_single_gpu_generation(
            model,
            val_ds.meta,
            val_ds.active_elements,
            device,
            n_mols=args.n_mols,
            batch_chunk=args.batch_chunk,
            use_bf16=True,
            seed=args.seed,
        )
        rec = {
            "checkpoint": ck_path.name,
            "step": step,
            "samples_seen": samples_seen,
            "eval_elapsed_sec": round(time.perf_counter() - t0, 2),
            "n_mols_per_suite": args.n_mols,
            "val": val_metrics,
            "generation": gen_metrics,
        }
        tmp_single = args.output_dir / f".eval_{ck_path.stem}.json.tmp"
        out_single = args.output_dir / f"eval_{ck_path.stem}.json"
        tmp_single.write_text(json.dumps(rec, indent=2) + "\n")
        tmp_single.replace(out_single)
        print(
            f"[rank {rank}] DONE {ck_path.name} in {rec['eval_elapsed_sec']:.1f}s: "
            f"val_tot={val_metrics['val_loss_total']:.4f} "
            f"euler100_vc={gen_metrics['active_euler_100']['valid_connected_rate']:.4f} "
            f"heun25_vc={gen_metrics['active_heun_25']['valid_connected_rate']:.4f} "
            f"tmqm_vc={gen_metrics.get('tmqm_heun_25', {}).get('valid_connected_rate', 0.0):.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rank == 0:
        # Wait for all atomic per-checkpoint JSON files on shared output filesystem (zero NCCL timeout risk)
        deadline = time.perf_counter() + 7200.0
        while time.perf_counter() < deadline:
            if all((args.output_dir / f"eval_{c.stem}.json").exists() for c in ckpt_files):
                break
            time.sleep(2.0)

        records = []
        for ck_path in ckpt_files:
            jf = args.output_dir / f"eval_{ck_path.stem}.json"
            if jf.exists():
                records.append(json.loads(jf.read_text()))
        records.sort(key=lambda r: int(r["step"]))

        best_by_val = min(records, key=lambda r: r["val"]["val_loss_total"])["checkpoint"] if records else ""
        for r in records:
            g = r["generation"]
            comp = (
                0.30 * g["active_euler_100"]["valid_connected_rate"]
                + 0.25 * g["active_heun_25"]["valid_connected_rate"]
                + 0.15 * g["open118_heun_25"]["valid_connected_rate"]
                + 0.15 * g.get("geom_drugs_heun_25", g["active_heun_25"])["valid_connected_rate"]
                + 0.15 * g.get("tmqm_heun_25", g["active_heun_25"])["valid_connected_rate"]
            )
            r["composite_valid_connected_score"] = round(float(comp), 6)
        best_by_gen = max(records, key=lambda r: r["composite_valid_connected_score"])["checkpoint"] if records else ""

        summary = {
            "n_candidates": len(records),
            "n_mols_per_suite": args.n_mols,
            "val_batches": args.val_batches,
            "best_by_val_loss": best_by_val,
            "best_by_composite_generation": best_by_gen,
            "candidates": records,
        }
        (args.output_dir / "checkpoint_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("=== CHECKPOINT BENCHMARK SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
