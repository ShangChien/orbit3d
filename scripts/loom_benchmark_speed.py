"""GPU0 throughput benchmark comparing Legacy `ConditionalUni3DAR` vs `LoomUni3DARGenerator` (`loom.weaving`).

Benchmarks both:
  1. Conditional 3D octree generation throughput (`mol/s`, `ms/mol`) across batch sizes & Selective Logit-CFG.
  2. Sequence SFT forward+backward training throughput (`steps/s`, `mol/s`) comparing legacy `flash_attn_varlen_func`
     Python loop vs right-padded fused `F.scaled_dot_product_attention`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from rdkit import Chem

from loom.generator.octree_backend import OctreeDataset, load_octree_checkpoint, trajectory_to_instance
from loom.generator.random_streams import candidate_seed
from loom.weaving import LoomUni3DARGenerator


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--sample-sdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-molecules", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(4)
    print(f"[Benchmark] Loading legacy checkpoint from {args.legacy_checkpoint} on {device}...", flush=True)
    legacy_model, legacy_state = load_octree_checkpoint(args.legacy_checkpoint, device=str(device), batch_size=32)
    legacy_model.float().eval()

    loom_gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.legacy_checkpoint,
        target_cols=["homo", "lumo"],
        target_units=["eV", "eV"],
        mean=[-5.75, -1.85],
        scale=[0.85, 0.95],
        device=str(device),
        batch_size=32,
        num_rbf=32,
        num_freqs=16,
        num_prefix_tokens=4,
        lora_rank=16,
        lora_alpha=32.0,
    )
    loom_gen.eval()

    results: dict = {
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "total_molecules_per_arm": args.total_molecules,
        "sampling": [],
        "training_step": [],
    }

    # Warmup legacy & loom
    print("[Benchmark] Warming up CUDA kernels...", flush=True)
    n_prop_leg = len(legacy_state["target_cols"])
    w_low = torch.zeros((4, n_prop_leg), device=device)
    w_act = torch.ones((4, n_prop_leg), dtype=torch.bool, device=device)
    if legacy_model.condition_version in (2, 3):
        w_act = w_act.unsqueeze(-1).expand(-1, -1, 2)
    with torch.no_grad():
        _ = legacy_model.generate_conditional_trajectories(
            w_low, w_low, w_act, cfg_scale=1.0, evaluate_self_consistency=False,
            row_seeds=[candidate_seed(1, i) for i in range(4)], electron_parity=0
        )
    w_t = torch.zeros((8, 2), device=device)
    w_m = torch.tensor([[True, False]] * 8, device=device)
    _ = loom_gen.sample(w_t, w_m, cfg_scale=1.0, seed=1, use_bf16=True)
    _ = loom_gen.sample(w_t, w_m, cfg_scale=1.75, xyz_cfg_scale=1.0, seed=2, use_bf16=True)
    _sync(device)

    # 1. Legacy Sampling Baseline (batch=32, FP32, RowRandomStreams)
    total_n = args.total_molecules
    b_leg = 32
    _sync(device)
    t0 = time.perf_counter()
    leg_valid = 0
    allowed_leg = tuple(legacy_model.args.allow_atoms.split(","))
    for start in range(0, total_n, b_leg):
        cur_b = min(b_leg, total_n - start)
        low_t = torch.full((cur_b, n_prop_leg), -0.5, device=device)
        act_t = torch.zeros((cur_b, n_prop_leg), dtype=torch.bool, device=device)
        act_t[:, 0] = True
        if legacy_model.condition_version in (2, 3):
            act_t = act_t.unsqueeze(-1).expand(-1, -1, 2)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            trajs = legacy_model.generate_conditional_trajectories(
                low_t, low_t, act_t, cfg_scale=1.0, tree_temp=0.85, atom_temp=0.30, xyz_temp=0.25,
                evaluate_self_consistency=False,
                row_seeds=[candidate_seed(24411, start + i) for i in range(cur_b)],
                electron_parity=0,
            )
        for tr in trajs:
            try:
                trajectory_to_instance(tr, element_table=allowed_leg)
                leg_valid += 1
            except ValueError:
                pass
    _sync(device)
    leg_sec = time.perf_counter() - t0
    leg_mol_s = total_n / leg_sec
    results["sampling"].append({
        "arm": "legacy_fp32_rowstreams_b32_cfg1.0",
        "batch_size": b_leg,
        "precision": "fp32",
        "cfg_scale": 1.0,
        "wall_seconds": leg_sec,
        "mol_per_sec": leg_mol_s,
        "speedup_vs_legacy": 1.0,
        "valid_rate": leg_valid / total_n,
    })
    print(f"  Legacy b=32 FP32 CFG=1.0: {leg_mol_s:.2f} mol/s (valid={leg_valid}/{total_n})", flush=True)

    # 2. Loom Sampling across batch sizes & Selective Logit-CFG
    allowed_loom = tuple(loom_gen.args.allow_atoms.split(","))
    for cfg_val, batch_sizes in [(1.0, [32, 64, 128, 256]), (1.75, [64, 128])]:
        for b_sz in batch_sizes:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            t0 = time.perf_counter()
            n_val = 0
            for start in range(0, total_n, b_sz):
                cur_b = min(b_sz, total_n - start)
                t_norm = torch.full((cur_b, 2), -0.5, device=device)
                m_act = torch.zeros((cur_b, 2), dtype=torch.bool, device=device)
                m_act[:, 0] = True
                trajs = loom_gen.sample(
                    t_norm, m_act, cfg_scale=cfg_val, xyz_cfg_scale=1.0,
                    temperatures=(0.85, 0.30, 0.25), seed=24411 + start, use_bf16=True, electron_parity=0
                )
                for tr in trajs:
                    try:
                        trajectory_to_instance(tr, element_table=allowed_loom)
                        n_val += 1
                    except ValueError:
                        pass
            _sync(device)
            sec = time.perf_counter() - t0
            mps = total_n / sec
            peak_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
            row = {
                "arm": f"loom_bf16_b{b_sz}_cfg{cfg_val}",
                "batch_size": b_sz,
                "precision": "bf16",
                "cfg_scale": cfg_val,
                "xyz_cfg_scale": 1.0,
                "wall_seconds": sec,
                "mol_per_sec": mps,
                "speedup_vs_legacy": mps / leg_mol_s,
                "valid_rate": n_val / total_n,
                "peak_vram_gb": round(peak_gb, 3),
            }
            results["sampling"].append(row)
            print(
                f"  Loom b={b_sz} BF16 CFG={cfg_val}: {mps:.2f} mol/s ({mps/leg_mol_s:.2f}x speedup, "
                f"valid={n_val}/{total_n}, peak={peak_gb:.2f} GB)",
                flush=True,
            )

    # 3. Training Forward+Backward Step Benchmark (batch=32, 15 steps after 3 warmup steps)
    sup = Chem.SDMolSupplier(str(args.sample_sdf), removeHs=False)
    sample_mols = []
    for m in sup:
        if m is not None:
            sample_mols.append(m)
            if len(sample_mols) >= 128:
                break

    ds = OctreeDataset(sample_mols, loom_gen.backbone, training=True, seed=42)
    packed_batches = [ds.collate(np.arange(i * 32, (i + 1) * 32) % len(sample_mols)) for i in range(6)]

    # 3a. Legacy SFT step (conditioner-only, FP32, flash_attn_varlen_func loop)
    legacy_model.requires_grad_(False)
    legacy_model.interval_conditioner.requires_grad_(True)
    legacy_model.train()
    opt_leg = torch.optim.AdamW([p for p in legacy_model.parameters() if p.requires_grad], lr=1e-4)
    for step in range(13):
        if step == 3:
            _sync(device)
            t_train0 = time.perf_counter()
        pb = packed_batches[step % len(packed_batches)]
        y_leg = torch.randn((32, n_prop_leg), device=device)
        m_leg = torch.ones((32, n_prop_leg), dtype=torch.bool, device=device)
        if legacy_model.condition_version in (2, 3):
            m_leg = m_leg.unsqueeze(-1).expand(-1, -1, 2)
        loss_l, _ = legacy_model.compute_sft_and_understanding_loss(
            pb, y_leg, y_leg, y_leg, m_leg, understanding_weight=0.0
        )
        opt_leg.zero_grad(set_to_none=True)
        loss_l.backward()
        opt_leg.step()
    _sync(device)
    leg_train_sec = (time.perf_counter() - t_train0) / 10.0

    # 3b. Loom SFT step (LoomConditioner + LoRA r=16 across all 12 layers, BF16, fused right-padded SDPA)
    params_loom = loom_gen.configure_trainable(train_conditioner=True, train_lora=True)
    loom_gen.train()
    opt_loom = torch.optim.AdamW(params_loom, lr=2e-4)
    for step in range(13):
        if step == 3:
            _sync(device)
            t_loom0 = time.perf_counter()
        pb = packed_batches[step % len(packed_batches)]
        y_lm = torch.randn((32, 2), device=device)
        m_lm = torch.ones((32, 2), dtype=torch.bool, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss_m, _ = loom_gen.compute_sft_loss(pb, y_lm, m_lm, cfg_dropout_prob=0.15, contrast_weight=0.25)
        opt_loom.zero_grad(set_to_none=True)
        loss_m.backward()
        opt_loom.step()
    _sync(device)
    loom_train_sec = (time.perf_counter() - t_loom0) / 10.0

    results["training_step"] = [
        {
            "arm": "legacy_sft_no_contrast_fp32_varlen_loop",
            "batch_size": 32,
            "sec_per_step": leg_train_sec,
            "mol_per_sec": 32.0 / leg_train_sec,
            "trainable_params": sum(p.numel() for p in legacy_model.parameters() if p.requires_grad),
        },
        {
            "arm": "loom_sft_plus_contrast_lora16_bf16_fused_sdpa",
            "batch_size": 32,
            "sec_per_step": loom_train_sec,
            "mol_per_sec": 32.0 / loom_train_sec,
            "speedup_vs_legacy_even_with_contrast_2x_pass": leg_train_sec / loom_train_sec,
            "trainable_params": sum(p.numel() for p in params_loom),
        },
    ]
    print(
        f"  Training step (b=32): Legacy(1-pass)={leg_train_sec*1000:.1f} ms/step vs "
        f"Loom(LoRA+Contrast 2-pass)={loom_train_sec*1000:.1f} ms/step ({leg_train_sec/loom_train_sec:.2f}x)",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[Benchmark] Saved summary to {args.output}", flush=True)


if __name__ == "__main__":
    main()
