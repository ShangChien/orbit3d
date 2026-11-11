#!/usr/bin/env python3
"""
Full DPA4 Scaling Family Benchmark (Nano, Mini, Neo, Air, Plus) vs. DPA-3.1-3M & DPA4-OMat24.
Measures:
  1. Saturated GPU Inference Throughput (atoms/ms & ms/mol on NVIDIA A800) across all scales.
  2. Zero-Shot DFT Equilibrium Force Calibration on 32 Heavy Ir(III) OLED Complexes (3D-SAT-OLED).
  3. De Novo 3D OLED Frontier Orbital Flow Matching (OOD E_g, HOMO Co-Guided, Joint 3-Property),
     evaluated under unified DPA4-Plus-OMol25-100M (8.8M, wB97M-V) reference forces + covalent BondMAE.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "orbit3d" / "compat"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from benchmarks.flow_matching.dpa_potential import DPAPotentialGuidance
from orbit3d.utils.evaluation import evaluate_generated_molecules
from benchmarks.flow_matching.guided_flow_sampler import GuidanceConfig, GuidedFlowSampler
from benchmarks.runners.run_3d_sat_oled_benchmark import OLED_ATOM_TYPES, load_oled_split


def get_scaling_catalog(pretrained_dir: Path) -> Dict[str, Dict[str, str]]:
    return {
        "dpa31_3m_spice2": {
            "label": "DPA-3.1-3M (SPICE2)",
            "family": "DPA-3.1",
            "path": str(pretrained_dir / "DPA-3.1-3M.pt"),
            "head": "SPICE2",
        },
        "dpa4_air_omat24": {
            "label": "DPA4-Air (OMat24 PBE)",
            "family": "DPA4-OMat24",
            "path": str(pretrained_dir / "DPA4-Air-OMat24-v20260805.pt"),
            "head": "Default",
        },
        "dpa4_nano_omol25": {
            "label": "DPA4-Nano (OMol25)",
            "family": "DPA4-OMol25",
            "path": str(pretrained_dir / "DPA4-Nano-OMol25-100M-v20260820.pt"),
            "head": "Default",
        },
        "dpa4_mini_omol25": {
            "label": "DPA4-Mini (OMol25)",
            "family": "DPA4-OMol25",
            "path": str(pretrained_dir / "DPA4-Mini-OMol25-100M-v20260820.pt"),
            "head": "Default",
        },
        "dpa4_neo_omol25": {
            "label": "DPA4-Neo (OMol25)",
            "family": "DPA4-OMol25",
            "path": str(pretrained_dir / "DPA4-Neo-OMol25-100M-v20260820.pt"),
            "head": "Default",
        },
        "dpa4_air_omol25": {
            "label": "DPA4-Air (OMol25)",
            "family": "DPA4-OMol25",
            "path": str(pretrained_dir / "DPA4-Air-OMol25-100M-v20260820.pt"),
            "head": "Default",
        },
        "dpa4_plus_omol25": {
            "label": "DPA4-Plus (OMol25)",
            "family": "DPA4-OMol25",
            "path": str(pretrained_dir / "DPA4-Plus-OMol25-100M-v20260820.pt"),
            "head": "Default",
        },
    }


def run_throughput_and_ir_calibration(pretrained_dir: Path, oled_dir: Path, device: str = "cuda:0") -> Dict:
    print("\n================================================================================")
    print("[Part 1] Evaluating All 7 MLIPs: Saturated Throughput (atoms/ms) & Ir(III) DFT Calibration...", flush=True)
    catalog = get_scaling_catalog(pretrained_dir)

    # 1) 48-mol organic batch (912 atoms) & saturated 128-mol batch (2432 atoms) for atoms/ms throughput
    torch.manual_seed(42)
    coords_48 = torch.randn(48 * 19, 3, device=device) * 1.6
    atypes_48 = torch.randint(0, 5, (48 * 19,), device=device)
    batch_48 = torch.arange(48, device=device).repeat_interleave(19)

    n_sat_mols = 128
    n_sat_atoms = n_sat_mols * 19  # 2,432 atoms
    coords_sat = torch.randn(n_sat_atoms, 3, device=device) * 1.6
    atypes_sat = torch.randint(0, 5, (n_sat_atoms,), device=device)
    batch_sat = torch.arange(n_sat_mols, device=device).repeat_interleave(19)

    # 2) 32 Ground-Truth DFT Ir(III) OLED Complexes from 3D-SAT-OLED
    test_mols = load_oled_split(oled_dir / "test.npz")[:32]
    coords_ir = torch.cat([m["coords"] for m in test_mols], dim=0).to(device)
    atypes_ir = torch.cat([m["a_idx"] for m in test_mols], dim=0).to(device)
    batch_ir = torch.cat(
        [torch.full((m["coords"].shape[0],), i, dtype=torch.long) for i, m in enumerate(test_mols)],
        dim=0,
    ).to(device)
    ir_idx = OLED_ATOM_TYPES.index("Ir")

    results = {}
    for key, meta in catalog.items():
        ckpt_mb = os.path.getsize(meta["path"]) / (1024.0 * 1024.0)
        m = DPAPotentialGuidance(
            model_path=meta["path"],
            head=meta["head"],
            atom_type_map=["C", "H", "N", "O", "F"],
            device=device,
            max_force_norm=None,
        )
        n_params = sum(p.numel() for p in m.wrapper.parameters())

        for _ in range(3):
            m.compute_energy_and_forces(coords_48, atypes_48, batch_48, clip_forces=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(12):
            m.compute_energy_and_forces(coords_48, atypes_48, batch_48, clip_forces=False)
        torch.cuda.synchronize()
        batch48_ms = (time.perf_counter() - t0) * 1000.0 / 12.0

        for _ in range(2):
            m.compute_energy_and_forces(coords_sat, atypes_sat, batch_sat, clip_forces=False)
        torch.cuda.synchronize()
        t_sat = time.perf_counter()
        for _ in range(8):
            m.compute_energy_and_forces(coords_sat, atypes_sat, batch_sat, clip_forces=False)
        torch.cuda.synchronize()
        sat_ms = (time.perf_counter() - t_sat) * 1000.0 / 8.0
        atoms_per_ms = n_sat_atoms / sat_ms

        # Ir(III) DFT equilibrium residual force calibration
        m.set_atom_type_map(OLED_ATOM_TYPES)
        _, F_ir = m.compute_energy_and_forces(coords_ir, atypes_ir, batch_ir, clip_forces=False, zero_com_forces=True)
        fnorm = F_ir.norm(dim=-1)
        ir_f = fnorm[atypes_ir == ir_idx]
        lig_f = fnorm[atypes_ir != ir_idx]

        results[key] = {
            "label": meta["label"],
            "family": meta["family"],
            "ckpt_MB": round(ckpt_mb, 2),
            "params_M": round(n_params / 1e6, 3),
            "batch48_ms": round(batch48_ms, 2),
            "ms_per_mol": round(batch48_ms / 48.0, 3),
            "saturated_atoms_per_ms": round(atoms_per_ms, 2),
            "ir_complex_all_mean_F": round(float(fnorm.mean().item()), 4),
            "ir_complex_all_med_F": round(float(fnorm.median().item()), 4),
            "ir_metal_mean_F": round(float(ir_f.mean().item()), 4),
            "ir_metal_med_F": round(float(ir_f.median().item()), 4),
            "ir_ligand_med_F": round(float(lig_f.median().item()), 4),
        }
        print(
            f"  [{meta['label']:22s}] {n_params/1e6:5.2f}M ({ckpt_mb:5.1f}MB) | "
            f"Thru={atoms_per_ms:5.1f} atoms/ms ({batch48_ms/48:.2f} ms/mol) | "
            f"Ir-DFT Mean|F|={fnorm.mean().item():.3f} (Ir={ir_f.mean().item():.3f}, LigMed={lig_f.median().item():.3f})",
            flush=True,
        )
        del m
        torch.cuda.empty_cache()

    return results


def run_scaling_flow_task(
    task_key: str,
    prop_name: str,
    flow_ckpt_path: Path,
    propmolflow_root: Path,
    pretrained_dir: Path,
    target_interval_raw: Tuple[float, float],
    extra_intervals_raw: Dict[str, Tuple[float, float]],
    n_molecules: int,
    n_timesteps: int,
    use_co_guidance: bool,
    seed: int,
    device: str = "cuda:0",
) -> Dict[str, Dict[str, float]]:
    print(f"\n================================================================================")
    print(f"[Part 2: {task_key}] Evaluating Full DPA4 Scaling Family (Nano->Mini->Neo->Air->Plus)...", flush=True)

    catalog = get_scaling_catalog(pretrained_dir)
    all_props = [prop_name] + [p for p in extra_intervals_raw.keys() if p != prop_name]

    # Use flagship DPA4-Plus-OMol25-100M (8.8M, 6.3 meV/A force MAE) as unified gold-standard oracle!
    ref_oracle = DPAPotentialGuidance(
        model_path=catalog["dpa4_plus_omol25"]["path"],
        head="Default",
        device=device,
    )
    sampler = GuidedFlowSampler(
        flow_ckpt_path=flow_ckpt_path,
        propmolflow_root=propmolflow_root,
        property_names=all_props,
        dpa_model_path=None,
        device=device,
    )
    ref_oracle.set_atom_type_map(sampler.flow_model.atom_type_map)

    torch.manual_seed(seed)
    n_atoms = sampler.flow_model.sample_n_atoms(n_molecules).to(device)

    target_intervals_norm = {}
    raw_dict = {prop_name: target_interval_raw, **extra_intervals_raw}
    for p, (raw_min, raw_max) in raw_dict.items():
        norm_min = float(sampler.prop_guidance.normalize_value(p, raw_min))
        norm_max = float(sampler.prop_guidance.normalize_value(p, raw_max))
        if norm_min > norm_max:
            norm_min, norm_max = norm_max, norm_min
        target_intervals_norm[p] = (
            torch.full((n_molecules,), norm_min, dtype=torch.float32, device=device),
            torch.full((n_molecules,), norm_max, dtype=torch.float32, device=device),
        )

    guided_mode = "dpa_co_guided_relaxed" if use_co_guidance else "dpa_orthogonal_relaxed"
    guided_cfg = GuidanceConfig(
        mode=guided_mode,
        lambda_prop=3.5,
        eta_phys=0.03,
        t_start=0.45,
        t_end=1.00,
        max_guidance_disp=0.08,
        atom_level_proj=True,
        mol_level_proj=True,
        discrete_guidance_scale=0.85 if use_co_guidance else 0.0,
        discrete_t_start=0.20,
        discrete_t_end=0.85,
        iso_barrier_relax=True,
        terminal_relax_steps=10,
        terminal_relax_lr=0.007,
        terminal_max_disp=0.022,
    )

    runs = [
        ("0_naive_no_mlip", None, GuidanceConfig(
            mode="naive_guidance", lambda_prop=3.5, eta_phys=0.0,
            t_start=0.45, t_end=1.00, max_guidance_disp=0.12, terminal_relax_steps=0
        )),
        ("1_dpa31_3m_spice2", "dpa31_3m_spice2", guided_cfg),
        ("2_dpa4_air_omat24", "dpa4_air_omat24", guided_cfg),
        ("3_dpa4_nano_omol25", "dpa4_nano_omol25", guided_cfg),
        ("4_dpa4_mini_omol25", "dpa4_mini_omol25", guided_cfg),
        ("5_dpa4_neo_omol25", "dpa4_neo_omol25", guided_cfg),
        ("6_dpa4_air_omol25", "dpa4_air_omol25", guided_cfg),
        ("7_dpa4_plus_omol25", "dpa4_plus_omol25", guided_cfg),
    ]

    results = {}
    for run_name, mlip_key, cfg in runs:
        if mlip_key is not None:
            meta = catalog[mlip_key]
            sampler.dpa_guidance = DPAPotentialGuidance(
                model_path=meta["path"],
                head=meta["head"],
                atom_type_map=sampler.flow_model.atom_type_map,
                device=device,
            )
        else:
            sampler.dpa_guidance = None

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mols, diag = sampler.sample_Within_Interval(
            n_atoms=n_atoms,
            target_intervals_norm=target_intervals_norm,
            primary_cond_prop=prop_name,
            guidance_cfg=cfg,
            n_timesteps=n_timesteps,
            seed=seed,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        metrics = evaluate_generated_molecules(
            molecules=mols,
            target_intervals_norm=target_intervals_norm,
            prop_guidance=sampler.prop_guidance,
            dpa_guidance=ref_oracle,
            device=device,
        )
        metrics.update(diag)
        metrics["sampling_time_sec"] = elapsed
        results[run_name] = metrics

        print(
            f"  [{run_name:20s}] Hit={metrics['joint_interval_hit_rate']*100:5.1f}% | "
            f"IntErr={metrics.get(f'{prop_name}_interval_err_eV', 0.0)*1000:6.1f} meV | "
            f"BondMAE={metrics.get('bond_length_mae_A', 0.0):6.4f} A | "
            f"PlusRef_F={metrics.get('dpa_rms_force_eV_A', 0.0):5.2f} eV/A (Med={metrics.get('dpa_median_rms_force_eV_A', 0.0):4.2f}) | "
            f"LowStrain={metrics.get('dpa_low_strain_frac', 0.0)*100:5.1f}% ({elapsed:.1f}s)",
            flush=True,
        )
        if sampler.dpa_guidance is not None:
            del sampler.dpa_guidance
            sampler.dpa_guidance = None
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--propmolflow_root", type=str, default="/sjtu-combustion/wjw/chensq/PropMolFlow")
    parser.add_argument("--pretrained_dir", type=str, default="/sjtu-combustion/wjw/chensq/pretrained")
    parser.add_argument("--oled_dir", type=str, default="/sjtu-combustion/wjw/chensq/3D-SAT-OLED/SAT_OLED_Model/oled")
    parser.add_argument("--out_dir", type=str, default="/sjtu-combustion/wjw/chensq/results/dpa4_scaling")
    parser.add_argument("--n_molecules", type=int, default=48)
    parser.add_argument("--n_timesteps", type=int, default=80)
    parser.add_argument("--worker", type=str, default="all", choices=["all", "gpu0", "gpu1"])
    parser.add_argument("--dual_gpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dual_gpu and torch.cuda.device_count() >= 2:
        print("Launching Dual-A800 Full DPA4 Scaling Family Benchmark (Nano, Mini, Neo, Air, Plus)...", flush=True)
        cmd0 = [
            sys.executable, __file__,
            "--propmolflow_root", args.propmolflow_root,
            "--pretrained_dir", args.pretrained_dir,
            "--oled_dir", args.oled_dir,
            "--out_dir", args.out_dir,
            "--n_molecules", str(args.n_molecules),
            "--n_timesteps", str(args.n_timesteps),
            "--worker", "gpu0",
        ]
        cmd1 = [
            sys.executable, __file__,
            "--propmolflow_root", args.propmolflow_root,
            "--pretrained_dir", args.pretrained_dir,
            "--oled_dir", args.oled_dir,
            "--out_dir", args.out_dir,
            "--n_molecules", str(args.n_molecules),
            "--n_timesteps", str(args.n_timesteps),
            "--worker", "gpu1",
        ]
        t0 = time.time()
        p0 = subprocess.Popen(cmd0, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        p1 = subprocess.Popen(cmd1, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})
        rc0 = p0.wait()
        rc1 = p1.wait()
        if rc0 != 0 or rc1 != 0:
            raise RuntimeError(f"Dual-GPU scaling benchmark failed: rc0={rc0}, rc1={rc1}")

        merged = {}
        for pf in [out_dir / "scaling_gpu0.json", out_dir / "scaling_gpu1.json"]:
            if pf.exists():
                with open(pf, "r") as f:
                    merged.update(json.load(f))
        out_json = out_dir / "dpa4_scaling_summary.json"
        with open(out_json, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nFull DPA4 Scaling Benchmark finished in {time.time() - t0:.1f}s -> {out_json}", flush=True)
        return

    sys.path.insert(0, args.propmolflow_root)
    pmf = Path(args.propmolflow_root)
    pdir = Path(args.pretrained_dir)
    odir = Path(args.oled_dir)

    out_data = {}
    if args.worker in ("all", "gpu0"):
        out_data["hardware_and_ir_calibration"] = run_throughput_and_ir_calibration(pdir, odir, device="cuda:0")
        out_data["qm9_gap_ood"] = run_scaling_flow_task(
            task_key="qm9_gap_ood",
            prop_name="gap",
            flow_ckpt_path=pmf / "checkpoints/out-of-distribution/gap/epoch=1637-step=640223.ckpt",
            propmolflow_root=pmf,
            pretrained_dir=pdir,
            target_interval_raw=(0.320, 0.345),
            extra_intervals_raw={},
            n_molecules=args.n_molecules,
            n_timesteps=args.n_timesteps,
            use_co_guidance=False,
            seed=101,
            device="cuda:0",
        )
        with open(out_dir / "scaling_gpu0.json", "w") as f:
            json.dump(out_data, f, indent=2)

    if args.worker in ("all", "gpu1"):
        out_data["qm9_homo"] = run_scaling_flow_task(
            task_key="qm9_homo",
            prop_name="homo",
            flow_ckpt_path=pmf / "checkpoints/out-of-distribution/homo/epoch=1468-step=574066.ckpt",
            propmolflow_root=pmf,
            pretrained_dir=pdir,
            target_interval_raw=(-0.205, -0.190),
            extra_intervals_raw={},
            n_molecules=args.n_molecules,
            n_timesteps=args.n_timesteps,
            use_co_guidance=True,
            seed=102,
            device="cuda:0",
        )
        out_data["qm9_joint_gap_homo_lumo"] = run_scaling_flow_task(
            task_key="qm9_joint_gap_homo_lumo",
            prop_name="gap",
            flow_ckpt_path=pmf / "checkpoints/out-of-distribution/gap/epoch=1637-step=640223.ckpt",
            propmolflow_root=pmf,
            pretrained_dir=pdir,
            target_interval_raw=(0.285, 0.320),
            extra_intervals_raw={"homo": (-0.245, -0.220), "lumo": (0.055, 0.085)},
            n_molecules=args.n_molecules,
            n_timesteps=args.n_timesteps,
            use_co_guidance=False,
            seed=104,
            device="cuda:0",
        )
        with open(out_dir / "scaling_gpu1.json", "w") as f:
            json.dump(out_data, f, indent=2)


if __name__ == "__main__":
    main()
