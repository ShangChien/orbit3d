#!/usr/bin/env python3
"""
Benchmark Suite: 3D Equivariant Flow Matching + DeePMD-kit DPA-3 MLIP
Orthogonal Manifold-Projected Interval Guidance on OLED Frontier Orbital Properties
(Bandgap E_g / gap, HOMO, LUMO, and Joint Multi-Objective Frontier Orbital Alignment).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure compat (dgl, torch_scatter), deepmd, PropMolFlow, and mol_gen are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "orbit3d" / "compat"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from rdkit import Chem

from benchmarks.flow_matching.guided_flow_sampler import GuidanceConfig, GuidedFlowSampler
from orbit3d.utils.evaluation import HARTREE_TO_EV, evaluate_generated_molecules


def save_molecules_to_sdf(molecules, out_sdf_path: Path):
    """Save valid RDKit molecules from SampledMolecule list to an SDF file."""
    out_sdf_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_sdf_path))
    n_written = 0
    for smol in molecules:
        mol = smol.rdkit_mol
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                writer.write(mol)
                n_written += 1
            except Exception:
                pass
    writer.close()
    return n_written


def run_property_benchmark(
    prop_name: str,
    flow_ckpt_path: Path,
    propmolflow_root: Path,
    dpa_model_path: Path,
    target_interval_raw: Tuple[float, float],
    extra_intervals_raw: Dict[str, Tuple[float, float]],
    n_molecules: int = 64,
    n_timesteps: int = 100,
    device: str = "cuda:0",
    out_dir: Path = Path("results"),
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Run paired comparison of 4 methods on a target frontier orbital interval task:
      1. Baseline (PropMolFlow conditional flow matching)
      2. Naive Interval Guidance (lambda * g_prop without DPA-3 projection)
      3. Ours: DPA-3 Orthogonal Manifold-Projected Guidance
      4. Ours + Terminal DPA-3 Local Relaxation
    """
    all_props = [prop_name] + [p for p in extra_intervals_raw.keys() if p != prop_name]
    print(f"\n{'='*80}")
    print(f"[Task: {prop_name.upper()}] Loading FlowMol ({flow_ckpt_path.name}), GVPRegressor({all_props}), and DPA-3.1-3M (SPICE2)...")
    sampler = GuidedFlowSampler(
        flow_ckpt_path=flow_ckpt_path,
        propmolflow_root=propmolflow_root,
        property_names=all_props,
        dpa_model_path=dpa_model_path,
        dpa_head="SPICE2",
        device=device,
    )

    # Sample realistic atom counts (between 14 and 23 atoms, typical organic conjugation scaffold size in rQM9)
    torch.manual_seed(seed)
    atom_counts_pool = torch.tensor([15, 16, 17, 18, 19, 20, 21, 22], device=device)
    n_atoms = atom_counts_pool[torch.randint(0, len(atom_counts_pool), (n_molecules,), device=device)]

    # Build normalized interval tensors
    target_intervals_norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    all_raw = {prop_name: target_interval_raw, **extra_intervals_raw}
    for p, (raw_min, raw_max) in all_raw.items():
        norm_min = float(sampler.prop_guidance.normalize_value(p, raw_min))
        norm_max = float(sampler.prop_guidance.normalize_value(p, raw_max))
        if norm_min > norm_max:
            norm_min, norm_max = norm_max, norm_min
        t_min = torch.full((n_molecules,), norm_min, dtype=torch.float32, device=device)
        t_max = torch.full((n_molecules,), norm_max, dtype=torch.float32, device=device)
        target_intervals_norm[p] = (t_min, t_max)
        print(
            f"  Target {p}: [{raw_min:.4f}, {raw_max:.4f}] Ha "
            f"([{raw_min*HARTREE_TO_EV:.2f}, {raw_max*HARTREE_TO_EV:.2f}] eV) "
            f"-> norm [{norm_min:.3f}, {norm_max:.3f}]"
        )

    methods = {
        "1_baseline": GuidanceConfig(
            mode="baseline",
            lambda_prop=0.0,
            eta_phys=0.0,
            terminal_relax_steps=0,
        ),
        "2_naive_guidance": GuidanceConfig(
            mode="naive_guidance",
            lambda_prop=3.5,
            eta_phys=0.0,
            t_start=0.45,
            t_end=1.00,
            max_guidance_disp=0.12,
            terminal_relax_steps=0,
        ),
        "3_dpa_orthogonal": GuidanceConfig(
            mode="dpa_orthogonal",
            lambda_prop=3.5,
            eta_phys=0.03,
            t_start=0.45,
            t_end=1.00,
            max_guidance_disp=0.08,
            atom_level_proj=True,
            mol_level_proj=True,
            terminal_relax_steps=0,
        ),
        "4_dpa_iso_relaxed": GuidanceConfig(
            mode="dpa_orthogonal_relaxed",
            lambda_prop=3.5,
            eta_phys=0.03,
            t_start=0.45,
            t_end=1.00,
            max_guidance_disp=0.08,
            atom_level_proj=True,
            mol_level_proj=True,
            iso_barrier_relax=True,
            terminal_relax_steps=10,
            terminal_relax_lr=0.007,
            terminal_max_disp=0.022,
        ),
        "5_dpa_co_guided_relaxed": GuidanceConfig(
            mode="dpa_co_guided_relaxed",
            lambda_prop=3.5,
            eta_phys=0.03,
            t_start=0.45,
            t_end=1.00,
            max_guidance_disp=0.08,
            atom_level_proj=True,
            mol_level_proj=True,
            discrete_guidance_scale=0.85,
            discrete_t_start=0.20,
            discrete_t_end=0.85,
            iso_barrier_relax=True,
            terminal_relax_steps=10,
            terminal_relax_lr=0.007,
            terminal_max_disp=0.022,
        ),
    }

    task_results: Dict[str, Dict[str, float]] = {}
    for method_name, cfg in methods.items():
        t0 = time.time()
        mols, diag = sampler.sample_Within_Interval(
            n_atoms=n_atoms,
            target_intervals_norm=target_intervals_norm,
            primary_cond_prop=prop_name,
            guidance_cfg=cfg,
            n_timesteps=n_timesteps,
            seed=seed,
        )
        elapsed = time.time() - t0

        metrics = evaluate_generated_molecules(
            molecules=mols,
            target_intervals_norm=target_intervals_norm,
            prop_guidance=sampler.prop_guidance,
            dpa_guidance=sampler.dpa_guidance,
            device=device,
        )
        metrics.update(diag)
        metrics["sampling_time_sec"] = elapsed

        sdf_path = out_dir / "sdf" / f"{prop_name}_{method_name}.sdf"
        n_saved = save_molecules_to_sdf(mols, sdf_path)
        metrics["n_saved_sdf"] = n_saved

        task_results[method_name] = metrics
        print(
            f"  [{method_name:24s}] "
            f"HitRate={metrics['joint_interval_hit_rate']*100:5.1f}% | "
            f"IntErr={metrics.get(f'{prop_name}_interval_err_eV', 0.0):5.3f} eV | "
            f"MolStab={metrics.get('frac_mol_stable', 0.0)*100:5.1f}% | "
            f"Valid={metrics.get('frac_valid_mols', 0.0)*100:5.1f}% | "
            f"3DGeomValid={metrics.get('geom_3d_valid_frac', 0.0)*100:5.1f}% | "
            f"DPA_RMS_F={metrics.get('dpa_rms_force_eV_A', 0.0):5.2f} eV/A | "
            f"LowStrain={metrics.get('dpa_low_strain_frac', 0.0)*100:5.1f}% "
            f"({elapsed:.1f}s)"
        )

    return task_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--propmolflow_root",
        type=str,
        default="/sjtu-combustion/wjw/chensq/PropMolFlow",
    )
    parser.add_argument(
        "--dpa_model_path",
        type=str,
        default="/sjtu-combustion/wjw/chensq/pretrained/DPA-3.1-3M.pt",
    )
    parser.add_argument("--out_dir", type=str, default="/sjtu-combustion/wjw/chensq/results/oled_benchmark")
    parser.add_argument("--n_molecules", type=int, default=64)
    parser.add_argument("--n_timesteps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--part_name", type=str, default="")
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated: gap,homo,lumo,joint or all")
    parser.add_argument("--dual_gpu", action="store_true", help="Dispatch tasks in parallel across cuda:0 and cuda:1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dual_gpu and torch.cuda.device_count() >= 2:
        import subprocess
        print(f"Launching Dual-A800 Parallel Benchmark across cuda:0 and cuda:1 (N={args.n_molecules}, steps={args.n_timesteps})...", flush=True)
        cmd_gpu0 = [
            sys.executable, __file__,
            "--propmolflow_root", args.propmolflow_root,
            "--dpa_model_path", args.dpa_model_path,
            "--out_dir", args.out_dir,
            "--n_molecules", str(args.n_molecules),
            "--n_timesteps", str(args.n_timesteps),
            "--device", "cuda:0",
            "--part_name", "cuda_0",
            "--tasks", "gap,lumo",
        ]
        cmd_gpu1 = [
            sys.executable, __file__,
            "--propmolflow_root", args.propmolflow_root,
            "--dpa_model_path", args.dpa_model_path,
            "--out_dir", args.out_dir,
            "--n_molecules", str(args.n_molecules),
            "--n_timesteps", str(args.n_timesteps),
            "--device", "cuda:0",
            "--part_name", "cuda_1",
            "--tasks", "homo,joint",
        ]
        t_start = time.time()
        env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}
        p0 = subprocess.Popen(cmd_gpu0, env=env0)
        p1 = subprocess.Popen(cmd_gpu1, env=env1)
        rc0 = p0.wait()
        rc1 = p1.wait()
        if rc0 != 0 or rc1 != 0:
            raise RuntimeError(f"Dual-GPU worker failed: rc0={rc0}, rc1={rc1}")
        merged = {}
        for part_file in [out_dir / "benchmark_cuda_0.json", out_dir / "benchmark_cuda_1.json"]:
            if part_file.exists():
                with open(part_file, "r") as f:
                    merged.update(json.load(f))
        out_json = out_dir / "benchmark_summary.json"
        with open(out_json, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nDual-A800 Parallel Benchmark finished in {time.time() - t_start:.1f}s! Saved to {out_json}", flush=True)
        return

    sys.path.insert(0, args.propmolflow_root)

    propmolflow_root = Path(args.propmolflow_root)
    dpa_model_path = Path(args.dpa_model_path)
    selected = set(args.tasks.split(",")) if args.tasks != "all" else {"gap", "homo", "lumo", "joint"}

    all_benchmarks = {}

    # 1. Bandgap (E_g / gap) OOD High-Gap Blue-Emitter / Host Interval: [0.320, 0.345] Ha (~[8.71, 9.39] eV)
    gap_ckpt = propmolflow_root / "checkpoints/out-of-distribution/gap/epoch=1637-step=640223.ckpt"
    if "gap" in selected and gap_ckpt.exists():
        all_benchmarks["gap_ood_interval"] = run_property_benchmark(
            prop_name="gap",
            flow_ckpt_path=gap_ckpt,
            propmolflow_root=propmolflow_root,
            dpa_model_path=dpa_model_path,
            target_interval_raw=(0.320, 0.345),
            extra_intervals_raw={},
            n_molecules=args.n_molecules,
            n_timesteps=args.n_timesteps,
            device=args.device,
            out_dir=out_dir,
            seed=101,
        )

    # 2. HOMO Energy Level Target Interval: [-0.205, -0.190] Ha (~[-5.58, -5.17] eV, OLED HTL/Host alignment)
    homo_ckpt = propmolflow_root / "checkpoints/out-of-distribution/homo/epoch=1468-step=574066.ckpt"
    if "homo" in selected and homo_ckpt.exists():
        all_benchmarks["homo_interval"] = run_property_benchmark(
            prop_name="homo",
            flow_ckpt_path=homo_ckpt,
            propmolflow_root=propmolflow_root,
            dpa_model_path=dpa_model_path,
            target_interval_raw=(-0.205, -0.190),
            extra_intervals_raw={},
            n_molecules=args.n_molecules,
            n_timesteps=args.n_timesteps,
            device=args.device,
            out_dir=out_dir,
            seed=102,
        )

    # 3. LUMO Energy Level Target Interval: [0.060, 0.080] Ha (~[1.63, 2.18] eV, OLED ETL alignment)
    lumo_ckpt = propmolflow_root / "checkpoints/out-of-distribution/lumo/epoch=1865-step=729605.ckpt"
    if "lumo" in selected and lumo_ckpt.exists():
        all_benchmarks["lumo_interval"] = run_property_benchmark(
            prop_name="lumo",
            flow_ckpt_path=lumo_ckpt,
            propmolflow_root=propmolflow_root,
            dpa_model_path=dpa_model_path,
            target_interval_raw=(0.060, 0.080),
            extra_intervals_raw={},
            n_molecules=args.n_molecules,
            n_timesteps=args.n_timesteps,
            device=args.device,
            out_dir=out_dir,
            seed=103,
        )

    # 4. Joint Multi-Objective OLED Frontier Orbital Alignment (Simultaneous E_g + HOMO + LUMO Intervals)
    if "joint" in selected and gap_ckpt.exists() and homo_ckpt.exists() and lumo_ckpt.exists():
        all_benchmarks["joint_gap_homo_lumo"] = run_property_benchmark(
            prop_name="gap",
            flow_ckpt_path=gap_ckpt,
            propmolflow_root=propmolflow_root,
            dpa_model_path=dpa_model_path,
            target_interval_raw=(0.285, 0.320),
            extra_intervals_raw={
                "homo": (-0.245, -0.220),
                "lumo": (0.055, 0.085),
            },
            n_molecules=args.n_molecules,
            n_timesteps=args.n_timesteps,
            device=args.device,
            out_dir=out_dir,
            seed=104,
        )

    dev_tag = args.part_name if args.part_name else args.device.replace(":", "_")
    part_json = out_dir / f"benchmark_{dev_tag}.json"
    with open(part_json, "w") as f:
        json.dump(all_benchmarks, f, indent=2)
    if args.tasks == "all":
        out_json = out_dir / "benchmark_summary.json"
        with open(out_json, "w") as f:
            json.dump(all_benchmarks, f, indent=2)
        print(f"\nSaved full benchmark results to {out_json}", flush=True)


if __name__ == "__main__":
    main()
