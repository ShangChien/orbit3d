#!/usr/bin/env python3
"""
Comprehensive Head-to-Head Empirical Benchmark: Tencent HyFF (v0.2.0) vs. DPA-4 Series in ORBIT-3D
==================================================================================================
Executes 4 rigorous, cross-evaluated experiments across dual NVIDIA A800-80GB GPUs:
  Exp 1: Hardware Throughput, VRAM, Weight Footprint, Ir(III) DFT Calibration & Short-Range Clash Stress Test
         (All 5 DPA-4 OMol25 scales + DPA-3.1 + DPA4-OMat24 vs. All 5 HyFF OMol25 scales: presto..largo).
  Exp 2: De Novo 3D Molecular Generation (PropMolFlow QM9 OOD Gap, HOMO, Joint 3-Property OLED)
         with Dual Independent Cross-MLIP Referees (`DPA4-Plus` & `HyFF-andante`) to eliminate self-bias.
  Exp 3: Heavy Organometallic Ir(III) 3D-SAT-OLED Generation (All-Atom Native Ir vs Legacy Dummy-Hack)
         + Singlet (M=1) vs Triplet (M=3) Spin Conditioning & Zero-Shot Delta E(T1-S0) vs DFT E_ad.
  Exp 4: End-to-End ORBIT-3D 5-Stage Core Pipeline Replacement Test across Mode B (Uni-Mol Oracle)
         and Mode C (Shared-Backbone Native Equivariant Descriptor Hook: `fitting_net` vs `readout_norms[-1]`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [
    str(PROJECT_ROOT / "src" / "orbit3d" / "compat"),
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT),
    "/sjtu-combustion/wjw/chensq/PropMolFlow",
    "/sjtu-combustion/wjw/chensq/repos/Hyra-results/AI4Science/hyff/src",
    "/sjtu-combustion/wjw/chensq/hyff_site_packages",
    "/sjtu-combustion/wjw/chensq/deepmd_v32",
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr

from orbit3d.physics import DPAPotentialGuidance, HyFFPotentialGuidance
from orbit3d.pipeline import run_from_config
from orbit3d.utils.evaluation import evaluate_generated_molecules
from benchmarks.flow_matching.guided_flow_sampler import GuidanceConfig, GuidedFlowSampler
from benchmarks.flow_matching.interval_guidance import orthogonal_manifold_projection
from benchmarks.runners.run_3d_sat_oled_benchmark import (
    EquivariantOLEDFlowField,
    EquivariantOLEDPropertyRegressor,
    OLED_ATOM_TYPES,
    compute_oled_hybrid_dpa_forces,
    load_oled_split,
)


HYFF_META = {
    "hyff_presto": {"tier": "presto", "label": "HyFF-presto (OMol25)", "active_M": 0.05},
    "hyff_vivace": {"tier": "vivace", "label": "HyFF-vivace (OMol25)", "active_M": 0.16},
    "hyff_moderato": {"tier": "moderato", "label": "HyFF-moderato (OMol25)", "active_M": 0.83},
    "hyff_andante": {"tier": "andante", "label": "HyFF-andante (OMol25)", "active_M": 5.00},
    "hyff_largo": {"tier": "largo", "label": "HyFF-largo (OMol25)", "active_M": 34.00},
}


def get_dpa_catalog(pretrained_dir: Path) -> Dict[str, Dict[str, str]]:
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


def _hyff_ckpt_size_mb(tier: str) -> float:
    from hyff.registry import resolve

    sel = resolve(tier)
    w_path = Path(sel.weights)
    ckpt_dir = w_path.parent
    total_bytes = sum(f.stat().st_size for f in ckpt_dir.glob("*") if f.is_file())
    return total_bytes / (1024.0 * 1024.0)


def _make_realistic_organic_batch(n_mols: int, device: str, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds non-overlapping 19-atom organic conformations (min d_ij >= 1.08 A) for clean throughput timing."""
    rng = np.random.default_rng(seed)
    # Base 19-atom cage/chain template with realistic >= 1.25 A spacing
    grid = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                if len(grid) < 19:
                    grid.append([ix * 1.42, iy * 1.42, iz * 1.42])
    base = np.asarray(grid, dtype=np.float32)
    base -= base.mean(axis=0, keepdims=True)

    coords_list = []
    for _ in range(n_mols):
        jitter = rng.normal(0.0, 0.08, size=base.shape).astype(np.float32)
        c = base + jitter
        coords_list.append(c)
    coords = torch.tensor(np.concatenate(coords_list, axis=0), dtype=torch.float32, device=device)
    atypes = torch.tensor(rng.integers(0, 5, size=(n_mols * 19,)), dtype=torch.long, device=device)
    batch_idx = torch.arange(n_mols, dtype=torch.long, device=device).repeat_interleave(19)
    return coords, atypes, batch_idx


# ==============================================================================
# EXP 1: Hardware Speed, VRAM, Disk Size, Ir(III) Calibration & Clash Stress Test
# ==============================================================================
def run_exp1_hardware_ir_and_clash(pretrained_dir: Path, oled_dir: Path, device: str = "cuda:0") -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print("[Exp 1A] Full-Series Speed, VRAM, Footprint & Heavy Ir(III) Equilibrium Calibration", flush=True)
    print("=" * 88)

    coords_48, atypes_48, batch_48 = _make_realistic_organic_batch(48, device=device, seed=42)
    coords_128, atypes_128, batch_128 = _make_realistic_organic_batch(128, device=device, seed=43)
    n_sat_atoms = 128 * 19  # 2,432 atoms

    test_mols = load_oled_split(oled_dir / "test.npz")[:32]
    coords_ir = torch.cat([m["coords"] for m in test_mols], dim=0).to(device)
    atypes_ir = torch.cat([m["a_idx"] for m in test_mols], dim=0).to(device)
    batch_ir = torch.cat(
        [torch.full((m["coords"].shape[0],), i, dtype=torch.long) for i, m in enumerate(test_mols)],
        dim=0,
    ).to(device)
    ir_idx = OLED_ATOM_TYPES.index("Ir")

    calib_results: Dict[str, Any] = {}

    # 1) Evaluate all DPA models
    dpa_cat = get_dpa_catalog(pretrained_dir)
    for key, meta in dpa_cat.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ckpt_mb = os.path.getsize(meta["path"]) / (1024.0 * 1024.0)
        m = DPAPotentialGuidance(
            model_path=meta["path"],
            head=meta["head"],
            atom_type_map=["C", "H", "N", "O", "F"],
            device=device,
            max_force_norm=None,
        )
        n_params = sum(p.numel() for p in m.wrapper.parameters())
        try:
            d_desc = m.probe_native_descriptor_dim()
        except Exception:
            d_desc = -1

        for _ in range(3):
            m.compute_energy_and_forces(coords_48, atypes_48, batch_48, clip_forces=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            m.compute_energy_and_forces(coords_48, atypes_48, batch_48, clip_forces=False)
        torch.cuda.synchronize()
        batch48_ms = (time.perf_counter() - t0) * 1000.0 / 10.0

        for _ in range(2):
            m.compute_energy_and_forces(coords_128, atypes_128, batch_128, clip_forces=False)
        torch.cuda.synchronize()
        t_sat = time.perf_counter()
        for _ in range(6):
            m.compute_energy_and_forces(coords_128, atypes_128, batch_128, clip_forces=False)
        torch.cuda.synchronize()
        sat_ms = (time.perf_counter() - t_sat) * 1000.0 / 6.0
        atoms_per_ms = n_sat_atoms / sat_ms
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        m.set_atom_type_map(OLED_ATOM_TYPES)
        _, F_ir = m.compute_energy_and_forces(coords_ir, atypes_ir, batch_ir, clip_forces=False, zero_com_forces=True)
        fnorm = F_ir.norm(dim=-1)
        ir_f = fnorm[atypes_ir == ir_idx]
        lig_f = fnorm[atypes_ir != ir_idx]

        calib_results[key] = {
            "label": meta["label"],
            "family": meta["family"],
            "ckpt_MB": round(ckpt_mb, 2),
            "params_M": round(n_params / 1e6, 3),
            "active_params_M": round(n_params / 1e6, 3),
            "desc_dim": d_desc,
            "peak_vram_MB": round(peak_vram_mb, 1),
            "batch48_ms": round(batch48_ms, 2),
            "ms_per_mol": round(batch48_ms / 48.0, 3),
            "saturated_atoms_per_ms": round(atoms_per_ms, 2),
            "ir_S0_all_mean_F": round(float(fnorm.mean().item()), 4),
            "ir_S0_all_med_F": round(float(fnorm.median().item()), 4),
            "ir_S0_metal_mean_F": round(float(ir_f.mean().item()), 4),
            "ir_S0_ligand_med_F": round(float(lig_f.median().item()), 4),
        }
        print(
            f"  [{meta['label']:24s}] {n_params/1e6:6.2f}M ({ckpt_mb:6.1f}MB) | VRAM={peak_vram_mb:6.1f}MB | "
            f"B48={batch48_ms:6.2f}ms ({atoms_per_ms:5.1f} atoms/ms) | Ir-DFT |F|={fnorm.mean().item():.3f} (Ir={ir_f.mean().item():.3f})",
            flush=True,
        )
        del m
        torch.cuda.empty_cache()

    # 2) Evaluate all 5 HyFF models (presto, vivace, moderato, andante, largo)
    for key, hmeta in HYFF_META.items():
        tier = hmeta["tier"]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ckpt_mb = _hyff_ckpt_size_mb(tier)
        # Test native route for presto/vivace/moderato/andante, and native (or fallback) for largo
        route = "native" if tier != "largo" else "reference"
        m = HyFFPotentialGuidance(
            tier=tier,
            atom_type_map=["C", "H", "N", "O", "F"],
            device=device,
            max_force_norm=None,
            route=route,
        )
        n_params = sum(p.numel() for p in m.ref_model.parameters())
        d_desc = m.probe_native_descriptor_dim()

        for _ in range(2):
            m.compute_energy_and_forces(coords_48, atypes_48, batch_48, clip_forces=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_rep48 = 4 if tier == "largo" else 10
        for _ in range(n_rep48):
            m.compute_energy_and_forces(coords_48 + 0.002, atypes_48, batch_48, clip_forces=False)
        torch.cuda.synchronize()
        batch48_ms = (time.perf_counter() - t0) * 1000.0 / n_rep48

        if tier == "largo":
            # Evaluate 48-mol throughput for largo to stay fast
            atoms_per_ms = (48 * 19) / batch48_ms
        else:
            for _ in range(2):
                m.compute_energy_and_forces(coords_128, atypes_128, batch_128, clip_forces=False)
            torch.cuda.synchronize()
            t_sat = time.perf_counter()
            for _ in range(6):
                m.compute_energy_and_forces(coords_128 + 0.002, atypes_128, batch_128, clip_forces=False)
            torch.cuda.synchronize()
            sat_ms = (time.perf_counter() - t_sat) * 1000.0 / 6.0
            atoms_per_ms = n_sat_atoms / sat_ms

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        m.set_atom_type_map(OLED_ATOM_TYPES)
        _, F_ir_s0 = m.compute_energy_and_forces(
            coords_ir, atypes_ir, batch_ir, clip_forces=False, zero_com_forces=True, multiplicities=1
        )
        _, F_ir_t1 = m.compute_energy_and_forces(
            coords_ir, atypes_ir, batch_ir, clip_forces=False, zero_com_forces=True, multiplicities=3
        )
        fnorm_s0 = F_ir_s0.norm(dim=-1)
        fnorm_t1 = F_ir_t1.norm(dim=-1)
        ir_f_s0 = fnorm_s0[atypes_ir == ir_idx]
        lig_f_s0 = fnorm_s0[atypes_ir != ir_idx]
        ir_f_t1 = fnorm_t1[atypes_ir == ir_idx]

        calib_results[key] = {
            "label": hmeta["label"],
            "family": "HyFF-OMol25",
            "ckpt_MB": round(ckpt_mb, 2),
            "params_M": round(n_params / 1e6, 3),
            "active_params_M": hmeta["active_M"],
            "desc_dim": d_desc,
            "peak_vram_MB": round(peak_vram_mb, 1),
            "batch48_ms": round(batch48_ms, 2),
            "ms_per_mol": round(batch48_ms / 48.0, 3),
            "saturated_atoms_per_ms": round(atoms_per_ms, 2),
            "ir_S0_all_mean_F": round(float(fnorm_s0.mean().item()), 4),
            "ir_S0_all_med_F": round(float(fnorm_s0.median().item()), 4),
            "ir_S0_metal_mean_F": round(float(ir_f_s0.mean().item()), 4),
            "ir_S0_ligand_med_F": round(float(lig_f_s0.median().item()), 4),
            "ir_T1_all_mean_F": round(float(fnorm_t1.mean().item()), 4),
            "ir_T1_metal_mean_F": round(float(ir_f_t1.mean().item()), 4),
        }
        print(
            f"  [{hmeta['label']:24s}] {n_params/1e6:6.2f}M ({ckpt_mb:6.1f}MB) | VRAM={peak_vram_mb:6.1f}MB | "
            f"B48={batch48_ms:6.2f}ms ({atoms_per_ms:5.1f} atoms/ms) | Ir-DFT S0|F|={fnorm_s0.mean().item():.3f} (Ir={ir_f_s0.mean().item():.3f}) T1|F|={fnorm_t1.mean().item():.3f}",
            flush=True,
        )
        m.close()
        del m
        torch.cuda.empty_cache()

    # ==========================================================================
    # [Exp 1B] Short-Range Steric Clash Stress Test (ZBL vs Learned Pair RBF)
    # ==========================================================================
    print("\n" + "=" * 88)
    print("[Exp 1B] Short-Range Steric Clash Robustness Test (d_clash = 0.25, 0.45, 0.65, 0.85 A)", flush=True)
    print("=" * 88)

    clash_dists = [0.25, 0.45, 0.65, 0.85]
    clash_mols = load_oled_split(oled_dir / "test.npz")[:16]

    clash_models = [
        ("dpa4_mini_omol25", "dpa", dpa_cat["dpa4_mini_omol25"]),
        ("dpa4_plus_omol25", "dpa", dpa_cat["dpa4_plus_omol25"]),
        ("hyff_presto", "hyff", "presto"),
        ("hyff_vivace", "hyff", "vivace"),
        ("hyff_moderato", "hyff", "moderato"),
        ("hyff_andante", "hyff", "andante"),
        ("hyff_largo_native", "hyff_native_raw", "largo"),
        ("hyff_largo_safe", "hyff", "largo"),
    ]

    clash_results: Dict[str, Any] = {}
    for m_key, m_kind, m_spec in clash_models:
        torch.cuda.empty_cache()
        if m_kind == "dpa":
            ff = DPAPotentialGuidance(
                model_path=m_spec["path"],
                head=m_spec["head"],
                atom_type_map=OLED_ATOM_TYPES,
                device=device,
                max_force_norm=15.0,
            )
        elif m_kind == "hyff_native_raw":
            ff = None
        else:
            ff = HyFFPotentialGuidance(
                tier=m_spec,
                atom_type_map=OLED_ATOM_TYPES,
                device=device,
                max_force_norm=15.0,
                route="reference" if m_spec == "largo" else "auto",
            )

        per_dist = {}
        for d_c in clash_dists:
            n_finite = 0
            n_repulsive = 0
            n_recovered = 0
            rep_forces = []

            for mol_d in clash_mols:
                c = mol_d["coords"].clone().to(device)
                a = mol_d["a_idx"].clone().to(device)
                i0 = int(mol_d["bond_src"][0].item())
                j0 = int(mol_d["bond_dst"][0].item())
                u_ij = c[i0] - c[j0]
                u_ij = u_ij / (torch.norm(u_ij) + 1e-8)
                mid = 0.5 * (c[i0] + c[j0])
                c[i0] = mid + 0.5 * d_c * u_ij
                c[j0] = mid - 0.5 * d_c * u_ij

                if m_kind == "hyff_native_raw":
                    # Test raw hyff.predict_batch(route="native") on largo without fallback
                    import hyff

                    pt = Chem.GetPeriodicTable()
                    z_np = np.array([pt.GetAtomicNumber(OLED_ATOM_TYPES[int(x)]) for x in a.cpu().tolist()], dtype=np.int64)
                    try:
                        out = hyff.predict_batch(
                            "largo",
                            c.detach().cpu().numpy(),
                            z_np,
                            np.array([0, len(z_np)], dtype=np.int64),
                            route="native",
                            device="cuda",
                        )
                        f_raw = torch.tensor(out["forces"], device=device, dtype=torch.float32)
                        is_fin = bool(torch.isfinite(f_raw).all())
                    except Exception:
                        is_fin = False
                        f_raw = torch.zeros_like(c)
                    if is_fin:
                        n_finite += 1
                        proj = float(((f_raw[i0] - f_raw[j0]) * u_ij).sum().item())
                        if proj > 0.0:
                            n_repulsive += 1
                        rep_forces.append(proj)
                    continue

                # Raw unclipped check
                _, f_raw = ff.compute_energy_and_forces(c, a, clip_forces=False, zero_com_forces=False)
                is_fin = bool(torch.isfinite(f_raw).all() and (f_raw.abs().sum() > 0))
                if is_fin:
                    n_finite += 1
                    proj = float(((f_raw[i0] - f_raw[j0]) * u_ij).sum().item())
                    if proj > 0.0:
                        n_repulsive += 1
                    rep_forces.append(proj)

                # 15-step relaxation recovery check
                c_rel, _, _ = ff.relax_conformers(c, a, n_steps=15, step_size=0.010, max_atom_disp=0.05)
                dmat = torch.cdist(c_rel, c_rel) + torch.eye(len(c_rel), device=device) * 10.0
                if float(dmat.min().item()) >= 0.95:
                    n_recovered += 1

            n_tot = len(clash_mols)
            per_dist[f"d_{d_c:.2f}A"] = {
                "finite_rate": round(n_finite / n_tot, 4),
                "repulsive_outward_rate": round(n_repulsive / n_tot, 4),
                "mean_repulsive_proj_eV_A": round(float(np.mean(rep_forces)) if rep_forces else 0.0, 2),
                "recovery_rate_15step": round(n_recovered / n_tot, 4),
            }

        clash_results[m_key] = per_dist
        d025 = per_dist["d_0.25A"]
        d065 = per_dist["d_0.65A"]
        print(
            f"  [{m_key:20s}] d=0.25A: Finite={d025['finite_rate']*100:5.1f}% Repulse={d025['repulsive_outward_rate']*100:5.1f}% "
            f"(F_rep={d025['mean_repulsive_proj_eV_A']:8.1f}) Recov={d025['recovery_rate_15step']*100:5.1f}% | "
            f"d=0.65A: Recov={d065['recovery_rate_15step']*100:5.1f}%",
            flush=True,
        )
        if hasattr(ff, "close") and ff is not None:
            ff.close()
        del ff
        torch.cuda.empty_cache()

    return {
        "calibration_and_throughput": calib_results,
        "short_range_clash_stress": clash_results,
    }


# ==============================================================================
# EXP 2: De Novo QM9 Flow Matching with Dual Cross-MLIP Referees
# ==============================================================================
@torch.no_grad()
def _compute_referee_force_stats(molecules: List[Any], referee_ff: Any, device: str) -> Dict[str, float]:
    graphs = [m.g.to(device) for m in molecules]
    batched_g = dgl.batch(graphs)
    node_batch_idx = torch.repeat_interleave(
        torch.arange(batched_g.batch_size, device=device, dtype=torch.long),
        batched_g.batch_num_nodes(),
    )
    coords = batched_g.ndata["x_1"]
    a_idx = batched_g.ndata["a_1"][:, :5].argmax(dim=-1)
    mol_E, forces = referee_ff.compute_energy_and_forces(
        coords=coords,
        dataset_atom_indices=a_idx,
        node_batch_idx=node_batch_idx,
        clip_forces=False,
        zero_com_forces=True,
    )
    f_sq = (forces * forces).sum(dim=-1)
    mol_f_sq = torch.zeros(batched_g.batch_size, dtype=coords.dtype, device=device)
    mol_f_sq.index_add_(0, node_batch_idx, f_sq)
    mol_rms = torch.sqrt(mol_f_sq / batched_g.batch_num_nodes().clamp(min=1).to(coords.dtype))
    return {
        "rms_force_eV_A": round(float(mol_rms.mean().item()), 4),
        "median_rms_force_eV_A": round(float(mol_rms.median().item()), 4),
        "low_strain_frac": round(float((mol_rms < 1.5).float().mean().item()), 4),
    }


def run_exp2_qm9_flow_task(
    task_key: str,
    prop_name: str,
    flow_ckpt_path: Path,
    propmolflow_root: Path,
    pretrained_dir: Path,
    target_interval_raw: Tuple[float, float],
    extra_intervals_raw: Dict[str, Tuple[float, float]],
    n_molecules: int = 48,
    n_timesteps: int = 80,
    use_co_guidance: bool = False,
    seed: int = 101,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print(f"[Exp 2: {task_key}] DPA-4 vs HyFF Flow Guidance + Dual Cross-MLIP Referees", flush=True)
    print("=" * 88)

    dpa_cat = get_dpa_catalog(pretrained_dir)
    all_props = [prop_name] + [p for p in extra_intervals_raw.keys() if p != prop_name]

    sampler = GuidedFlowSampler(
        flow_ckpt_path=flow_ckpt_path,
        propmolflow_root=propmolflow_root,
        property_names=all_props,
        dpa_model_path=None,
        device=device,
    )
    atom_map = sampler.flow_model.atom_type_map

    # Dual independent flagship referees: DPA4-Plus AND HyFF-andante!
    ref_dpa_plus = DPAPotentialGuidance(
        model_path=dpa_cat["dpa4_plus_omol25"]["path"],
        head="Default",
        atom_type_map=atom_map,
        device=device,
    )
    ref_hyff_andante = HyFFPotentialGuidance(
        tier="andante",
        atom_type_map=atom_map,
        device=device,
        route="reference",
    )

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
        ("0_naive_no_mlip", None, None, GuidanceConfig(
            mode="naive_guidance", lambda_prop=3.5, eta_phys=0.0,
            t_start=0.45, t_end=1.00, max_guidance_disp=0.12, terminal_relax_steps=0
        )),
        ("1_dpa4_nano", "dpa", "dpa4_nano_omol25", guided_cfg),
        ("2_dpa4_mini", "dpa", "dpa4_mini_omol25", guided_cfg),
        ("3_dpa4_air", "dpa", "dpa4_air_omol25", guided_cfg),
        ("4_dpa4_plus", "dpa", "dpa4_plus_omol25", guided_cfg),
        ("5_hyff_presto", "hyff", "presto", guided_cfg),
        ("6_hyff_vivace", "hyff", "vivace", guided_cfg),
        ("7_hyff_moderato", "hyff", "moderato", guided_cfg),
        ("8_hyff_andante", "hyff", "andante", guided_cfg),
    ]

    results = {}
    for run_name, backend_kind, spec, cfg in runs:
        if backend_kind == "dpa":
            meta = dpa_cat[spec]
            sampler.dpa_guidance = DPAPotentialGuidance(
                model_path=meta["path"],
                head=meta["head"],
                atom_type_map=atom_map,
                device=device,
            )
        elif backend_kind == "hyff":
            sampler.dpa_guidance = HyFFPotentialGuidance(
                tier=spec,
                atom_type_map=atom_map,
                device=device,
                route="auto",
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
            dpa_guidance=ref_dpa_plus,
            device=device,
        )
        hyff_ref_stats = _compute_referee_force_stats(mols, ref_hyff_andante, device=device)
        metrics["hyff_andante_rms_force_eV_A"] = hyff_ref_stats["rms_force_eV_A"]
        metrics["hyff_andante_median_rms_force_eV_A"] = hyff_ref_stats["median_rms_force_eV_A"]
        metrics["hyff_andante_low_strain_frac"] = hyff_ref_stats["low_strain_frac"]
        metrics.update(diag)
        metrics["sampling_time_sec"] = round(elapsed, 3)
        results[run_name] = metrics

        print(
            f"  [{run_name:18s}] Hit={metrics['joint_interval_hit_rate']*100:5.1f}% | "
            f"BondMAE={metrics.get('bond_length_mae_A', 0.0):6.4f}A | "
            f"DPA4+Ref_MedF={metrics.get('dpa_median_rms_force_eV_A', 0.0):5.3f} | "
            f"HyFFRef_MedF={metrics['hyff_andante_median_rms_force_eV_A']:5.3f} eV/A | "
            f"Time={elapsed:5.2f}s",
            flush=True,
        )

        if sampler.dpa_guidance is not None:
            if hasattr(sampler.dpa_guidance, "close"):
                sampler.dpa_guidance.close()
            del sampler.dpa_guidance
            sampler.dpa_guidance = None
        torch.cuda.empty_cache()

    ref_hyff_andante.close()
    del ref_dpa_plus, ref_hyff_andante
    torch.cuda.empty_cache()
    return results


# ==============================================================================
# EXP 3: Heavy Organometallic Ir(III) 3D-SAT-OLED + Spin Multiplicity S0 vs T1
# ==============================================================================
def _eval_oled_all_atom(
    coords: torch.Tensor,
    mol_ref: Dict[str, torch.Tensor],
    regressor: EquivariantOLEDPropertyRegressor,
    prop_mean: torch.Tensor,
    prop_std: torch.Tensor,
    target_min_eV: float,
    target_max_eV: float,
    ref_dpa_plus: DPAPotentialGuidance,
    ref_hyff_andante: HyFFPotentialGuidance,
    prop_idx: int,
) -> Dict[str, float]:
    device = coords.device
    a_idx = mol_ref["a_idx"].to(device)
    bond_src = mol_ref["bond_src"].to(device)
    bond_dst = mol_ref["bond_dst"].to(device)
    bond_ref_len = mol_ref["bond_ref_len"].to(device)
    ir_idx = OLED_ATOM_TYPES.index("Ir")

    d_bonds = torch.norm(coords[bond_src] - coords[bond_dst], dim=-1)
    abs_err = torch.abs(d_bonds - bond_ref_len)
    bond_strain_mae = float(abs_err.mean().item())
    max_bond_stretch = float(abs_err.max().item())
    bond_intact = float(max_bond_stretch < 0.22)

    is_ir_bond = (a_idx[bond_src] == ir_idx) | (a_idx[bond_dst] == ir_idx)
    ir_coord_mae = float(abs_err[is_ir_bond].mean().item()) if bool(is_ir_bond.any()) else bond_strain_mae

    dist_mat = torch.cdist(coords, coords)
    eye = torch.eye(coords.shape[0], dtype=torch.bool, device=device)
    bonded = eye.clone()
    bonded[bond_src, bond_dst] = True
    bonded[bond_dst, bond_src] = True
    min_nonbond = float(dist_mat[~bonded].min().item()) if (~bonded).any() else 2.0
    clash_free = float(min_nonbond > 1.05)

    with torch.no_grad():
        pred_norm = regressor(coords, a_idx)
        pred_eV = pred_norm * prop_std + prop_mean
        y_hat = float(pred_eV[prop_idx].item())

    in_interval = float((y_hat >= target_min_eV) and (y_hat <= target_max_eV))
    interval_err = max(0.0, target_min_eV - y_hat, y_hat - target_max_eV)

    _, f_dpa = ref_dpa_plus.compute_energy_and_forces(coords, a_idx, clip_forces=False, zero_com_forces=True)
    _, f_hyff = ref_hyff_andante.compute_energy_and_forces(coords, a_idx, clip_forces=False, zero_com_forces=True)
    rms_dpa = float(torch.sqrt((f_dpa * f_dpa).sum(dim=-1).mean()).item())
    rms_hyff = float(torch.sqrt((f_hyff * f_hyff).sum(dim=-1).mean()).item())

    return {
        "in_interval": in_interval,
        "interval_err_eV": interval_err,
        "bond_strain_mae_A": bond_strain_mae,
        "ir_coord_bond_mae_A": ir_coord_mae,
        "max_bond_stretch_A": max_bond_stretch,
        "geom_valid": float(bond_intact and clash_free),
        "dpa4_plus_rms_force_eV_A": rms_dpa,
        "hyff_andante_rms_force_eV_A": rms_hyff,
    }


def run_exp3_oled_3dsat_and_spin(pretrained_dir: Path, oled_dir: Path, device: str = "cuda:0") -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print("[Exp 3A] Heavy Organometallic Ir(III) 3D-SAT-OLED Generation (All-Atom DPA-4 vs HyFF S0/T1)", flush=True)
    print("=" * 88)

    train_mols = load_oled_split(oled_dir / "train.npz")
    test_mols = load_oled_split(oled_dir / "test.npz")
    dpa_cat = get_dpa_catalog(pretrained_dir)

    all_train_props = torch.stack([m["props"][:3] for m in train_mols], dim=0).to(device)
    prop_mean = all_train_props.mean(dim=0)
    prop_std = all_train_props.std(dim=0).clamp(min=1e-4)

    regressor = EquivariantOLEDPropertyRegressor().to(device)
    opt_reg = torch.optim.AdamW(regressor.parameters(), lr=1.5e-3, weight_decay=1e-4)
    torch.manual_seed(42)
    for _ in range(1, 26):
        regressor.train()
        perm = torch.randperm(len(train_mols))
        for idx in perm[:300]:
            m = train_mols[int(idx)]
            coords = m["coords"].to(device)
            coords_noisy = coords + 0.04 * torch.randn_like(coords)
            a_idx = m["a_idx"].to(device)
            target_norm = (m["props"][:3].to(device) - prop_mean) / prop_std
            loss = F.smooth_l1_loss(regressor(coords_noisy, a_idx), target_norm)
            opt_reg.zero_grad()
            loss.backward()
            opt_reg.step()
    regressor.eval()
    for p in regressor.parameters():
        p.requires_grad_(False)

    flow_net = EquivariantOLEDFlowField().to(device)
    opt_flow = torch.optim.AdamW(flow_net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    noise_sigma = 0.35
    for _ in range(1, 16):
        flow_net.train()
        perm = torch.randperm(len(train_mols))
        for idx in perm[:250]:
            m = train_mols[int(idx)]
            x_1 = m["coords"].to(device)
            a_idx = m["a_idx"].to(device)
            bond_src = m["bond_src"].to(device)
            bond_dst = m["bond_dst"].to(device)
            y_true_norm = ((m["props"][0].to(device) - prop_mean[0]) / prop_std[0]).view(1)
            y_min_n, y_max_n = flow_net.interval_cond.sample_training_intervals(y_true_norm)
            t_val = float(torch.rand(1).item())
            noise = noise_sigma * torch.randn_like(x_1)
            noise = noise - noise.mean(dim=0, keepdim=True)
            x_0 = x_1 + noise
            x_t = (1.0 - t_val) * x_0 + t_val * x_1
            x_1_pred = flow_net(x_t, a_idx, t_val, y_min_n, y_max_n, bond_src, bond_dst)
            loss = F.mse_loss(x_1_pred, x_1)
            opt_flow.zero_grad()
            loss.backward()
            opt_flow.step()
    flow_net.eval()

    # Load guidance force fields + dual referees
    dpa31_legacy = DPAPotentialGuidance(
        model_path=dpa_cat["dpa31_3m_spice2"]["path"], head="SPICE2", atom_type_map=OLED_ATOM_TYPES, device=device
    )
    dpa4_mini = DPAPotentialGuidance(
        model_path=dpa_cat["dpa4_mini_omol25"]["path"], head="Default", atom_type_map=OLED_ATOM_TYPES, device=device
    )
    dpa4_plus = DPAPotentialGuidance(
        model_path=dpa_cat["dpa4_plus_omol25"]["path"], head="Default", atom_type_map=OLED_ATOM_TYPES, device=device
    )
    hyff_presto = HyFFPotentialGuidance(tier="presto", atom_type_map=OLED_ATOM_TYPES, device=device, route="reference")
    hyff_vivace = HyFFPotentialGuidance(tier="vivace", atom_type_map=OLED_ATOM_TYPES, device=device, route="reference")
    hyff_moderato_s0 = HyFFPotentialGuidance(
        tier="moderato", atom_type_map=OLED_ATOM_TYPES, device=device, default_multiplicity=1, route="reference"
    )
    hyff_moderato_t1 = HyFFPotentialGuidance(
        tier="moderato", atom_type_map=OLED_ATOM_TYPES, device=device, default_multiplicity=3, route="reference"
    )
    hyff_andante_s0 = HyFFPotentialGuidance(
        tier="andante", atom_type_map=OLED_ATOM_TYPES, device=device, default_multiplicity=1, route="reference"
    )
    hyff_andante_t1 = HyFFPotentialGuidance(
        tier="andante", atom_type_map=OLED_ATOM_TYPES, device=device, default_multiplicity=3, route="reference"
    )

    modes = [
        ("1_baseline_flow", 0.0, None, 0),
        ("2_naive_guidance", 1.25, None, 0),
        ("3_legacy_dpa31_hack", 1.25, ("legacy_hack", dpa31_legacy), 8),
        ("4_dpa4_mini_all_atom", 1.25, ("all_atom", dpa4_mini), 8),
        ("5_dpa4_plus_all_atom", 1.25, ("all_atom", dpa4_plus), 8),
        ("6_hyff_presto_S0", 1.25, ("all_atom", hyff_presto), 8),
        ("7_hyff_vivace_S0", 1.25, ("all_atom", hyff_vivace), 8),
        ("8_hyff_moderato_S0", 1.25, ("all_atom", hyff_moderato_s0), 8),
        ("8b_hyff_moderato_T1", 1.25, ("all_atom", hyff_moderato_t1), 8),
        ("9_hyff_andante_S0", 1.25, ("all_atom", hyff_andante_s0), 8),
        ("9b_hyff_andante_T1", 1.25, ("all_atom", hyff_andante_t1), 8),
    ]

    tasks = [
        ("OLED_Eg_Bandgap", 0, 0.35, 1.10),
        ("OLED_HOMO_Level", 1, 0.30, 1.05),
        ("OLED_LUMO_Level", 2, 0.30, 1.05),
    ]

    n_eval = 20
    n_steps = 25
    dt = 1.0 / n_steps
    oled_gen_summary: Dict[str, Any] = {}

    for task_name, prop_idx, std_low, std_high in tasks:
        target_min_eV = float((prop_mean[prop_idx] + std_low * prop_std[prop_idx]).item())
        target_max_eV = float((prop_mean[prop_idx] + std_high * prop_std[prop_idx]).item())
        y_min_norm = ((torch.tensor(target_min_eV, device=device) - prop_mean[prop_idx]) / prop_std[prop_idx]).view(1)
        y_max_norm = ((torch.tensor(target_max_eV, device=device) - prop_mean[prop_idx]) / prop_std[prop_idx]).view(1)
        print(f"\n  --- Task: {task_name} [{target_min_eV:.3f}, {target_max_eV:.3f}] eV ---", flush=True)

        task_res = {}
        for mode_name, lam_prop, ff_pair, relax_steps in modes:
            t_start_mode = time.perf_counter()
            metrics_list = []
            for idx in range(n_eval):
                m = test_mols[idx]
                x_ref = m["coords"].to(device)
                a_idx = m["a_idx"].to(device)
                bond_src = m["bond_src"].to(device)
                bond_dst = m["bond_dst"].to(device)
                node_batch_idx = torch.zeros(x_ref.shape[0], dtype=torch.long, device=device)

                torch.manual_seed(1000 + idx)
                noise = noise_sigma * torch.randn_like(x_ref)
                noise = noise - noise.mean(dim=0, keepdim=True)
                x_t = x_ref + noise

                for s_idx in range(n_steps):
                    t_val = s_idx * dt
                    with torch.no_grad():
                        x_1_hat = flow_net(x_t, a_idx, t_val, y_min_norm, y_max_norm, bond_src, bond_dst)
                        v_flow = (x_1_hat - x_t) / max(1.0 - t_val, 0.05)
                        x_next = x_t + dt * v_flow

                    if lam_prop > 0.0 and t_val >= 0.35:
                        with torch.enable_grad():
                            x_in = x_next.detach().clone().requires_grad_(True)
                            pred_n = regressor(x_in, a_idx)[prop_idx]
                            hinge = torch.relu(y_min_norm[0] - pred_n).pow(2) + torch.relu(pred_n - y_max_norm[0]).pow(2)
                            if hinge.item() > 0:
                                grad_x = torch.autograd.grad(hinge, x_in)[0]
                                g_prop = -grad_x.detach()
                                g_prop = g_prop - g_prop.mean(dim=0, keepdim=True)
                                rms_g = torch.sqrt((g_prop * g_prop).sum(dim=-1).mean()) + 1e-8
                                gain = min(1.0, math.sqrt(float(hinge.item())) / 0.12)
                                g_prop = g_prop * (gain / rms_g)
                            else:
                                g_prop = torch.zeros_like(x_next)

                        if ff_pair is not None:
                            ff_mode, ff_obj = ff_pair
                            if ff_mode == "legacy_hack":
                                _, f_phys = compute_oled_hybrid_dpa_forces(x_next, m, ff_obj, clip_forces=True)
                            else:
                                _, f_phys = ff_obj.compute_energy_and_forces(
                                    x_next, a_idx, node_batch_idx, clip_forces=True, zero_com_forces=True
                                )
                            g_tilde, _ = orthogonal_manifold_projection(
                                g_prop=g_prop,
                                f_phys=f_phys,
                                node_batch_idx=node_batch_idx,
                                batch_size=1,
                                atom_level_projection=True,
                                mol_level_projection=True,
                            )
                            step_shift = dt * (lam_prop * g_tilde + 0.018 * f_phys)
                        else:
                            step_shift = dt * (lam_prop * g_prop)

                        s_norm = torch.norm(step_shift, dim=-1, keepdim=True) + 1e-8
                        step_shift = step_shift * torch.clamp(0.08 / s_norm, max=1.0)
                        x_next = x_next + step_shift - step_shift.mean(dim=0, keepdim=True)

                    x_t = x_next

                if relax_steps > 0 and ff_pair is not None:
                    ff_mode, ff_obj = ff_pair
                    for _ in range(relax_steps):
                        if ff_mode == "legacy_hack":
                            _, f_rel = compute_oled_hybrid_dpa_forces(x_t, m, ff_obj, clip_forces=True)
                        else:
                            _, f_rel = ff_obj.compute_energy_and_forces(
                                x_t, a_idx, node_batch_idx, clip_forces=True, zero_com_forces=True
                            )
                        disp = 0.005 * f_rel
                        d_norm = torch.norm(disp, dim=-1, keepdim=True) + 1e-8
                        disp = disp * torch.clamp(0.018 / d_norm, max=1.0)
                        x_cand = x_t + disp - disp.mean(dim=0, keepdim=True)
                        with torch.no_grad():
                            p_old = float((regressor(x_t, a_idx)[prop_idx] * prop_std[prop_idx] + prop_mean[prop_idx]).item())
                            p_new = float((regressor(x_cand, a_idx)[prop_idx] * prop_std[prop_idx] + prop_mean[prop_idx]).item())
                        err_old = max(0.0, target_min_eV - p_old, p_old - target_max_eV)
                        err_new = max(0.0, target_min_eV - p_new, p_new - target_max_eV)
                        if err_new <= err_old + 0.008:
                            x_t = x_cand

                res = _eval_oled_all_atom(
                    coords=x_t,
                    mol_ref=m,
                    regressor=regressor,
                    prop_mean=prop_mean,
                    prop_std=prop_std,
                    target_min_eV=target_min_eV,
                    target_max_eV=target_max_eV,
                    ref_dpa_plus=dpa4_plus,
                    ref_hyff_andante=hyff_andante_s0,
                    prop_idx=prop_idx,
                )
                metrics_list.append(res)

            agg = {k: round(float(np.mean([x[k] for x in metrics_list])), 4) for k in metrics_list[0].keys()}
            agg["dpa4_plus_median_rms_force_eV_A"] = round(
                float(np.median([x["dpa4_plus_rms_force_eV_A"] for x in metrics_list])), 4
            )
            agg["hyff_andante_median_rms_force_eV_A"] = round(
                float(np.median([x["hyff_andante_rms_force_eV_A"] for x in metrics_list])), 4
            )
            agg["valid_and_in_interval"] = round(
                float(np.mean([x["geom_valid"] * x["in_interval"] for x in metrics_list])), 4
            )
            agg["wall_time_sec"] = round(time.perf_counter() - t_start_mode, 2)
            task_res[mode_name] = agg
            print(
                f"    [{mode_name:22s}] Hit={agg['in_interval']*100:5.1f}% | GeomValid={agg['geom_valid']*100:5.1f}% | "
                f"Valid&Hit={agg['valid_and_in_interval']*100:5.1f}% | BondMAE={agg['bond_strain_mae_A']:.4f}A "
                f"(IrCoord={agg['ir_coord_bond_mae_A']:.4f}A) | DPA+MedF={agg['dpa4_plus_median_rms_force_eV_A']:5.2f} | "
                f"HyFFMedF={agg['hyff_andante_median_rms_force_eV_A']:5.2f}",
                flush=True,
            )
        oled_gen_summary[task_name] = task_res

    for obj in [hyff_presto, hyff_vivace, hyff_moderato_s0, hyff_moderato_t1, hyff_andante_s0, hyff_andante_t1]:
        obj.close()
    del dpa31_legacy, dpa4_mini, dpa4_plus
    torch.cuda.empty_cache()

    # ==========================================================================
    # [Exp 3B] Zero-Shot Spin Splitting Delta E(T1 - S0) on 64 Ir(III) Complexes
    # ==========================================================================
    print("\n" + "=" * 88)
    print("[Exp 3B] Zero-Shot Triplet-Singlet Splitting E(M=3) - E(M=1) vs DFT E_ad on 64 Ir(III) Complexes", flush=True)
    print("=" * 88)

    eval64 = test_mols[:64]
    coords_64 = torch.cat([m["coords"] for m in eval64], dim=0).to(device)
    atypes_64 = torch.cat([m["a_idx"] for m in eval64], dim=0).to(device)
    batch_64 = torch.cat(
        [torch.full((m["coords"].shape[0],), i, dtype=torch.long) for i, m in enumerate(eval64)], dim=0
    ).to(device)
    dft_ead = np.array([float(m["props"][3].item()) for m in eval64], dtype=np.float64)
    # Convert Hartree to eV if needed
    if np.median(np.abs(dft_ead)) < 0.5:
        dft_ead = dft_ead * 27.211386245988

    spin_results: Dict[str, Any] = {
        "dpa4_all_scales": {
            "supports_spin_multiplicity": False,
            "delta_E_T1_S0_mean_eV": 0.0,
            "note": "DPA-4 takes only (Z, R) without spin multiplicity input; E(M=3) - E(M=1) == 0 identically.",
        }
    }

    for tier in ["presto", "vivace", "moderato", "andante", "largo"]:
        ff = HyFFPotentialGuidance(tier=tier, atom_type_map=OLED_ATOM_TYPES, device=device, route="reference")
        e_s0, f_s0 = ff.compute_energy_and_forces(coords_64, atypes_64, batch_64, clip_forces=False, multiplicities=1)
        e_t1, f_t1 = ff.compute_energy_and_forces(coords_64, atypes_64, batch_64, clip_forces=False, multiplicities=3)
        delta_st = (e_t1 - e_s0).detach().cpu().numpy().astype(np.float64)
        force_diff_rms = float(torch.norm(f_t1 - f_s0, dim=-1).mean().item())

        # Also relax 16 complexes on T1 surface vs S0 surface to compute Adiabatic Delta E!
        c_t1_rel, _, e_t1_adiab = ff.relax_conformers(
            coords_64[: sum(m["coords"].shape[0] for m in eval64[:16])],
            atypes_64[: sum(m["coords"].shape[0] for m in eval64[:16])],
            batch_64[: sum(m["coords"].shape[0] for m in eval64[:16])],
            n_steps=12,
            step_size=0.006,
            multiplicities=3,
        )
        adiab_relax_gain_eV = float((e_t1[:16] - e_t1_adiab).mean().item())

        r_val = float(pearsonr(delta_st, dft_ead)[0]) if np.std(delta_st) > 1e-6 else 0.0
        rho_val = float(spearmanr(delta_st, dft_ead)[0]) if np.std(delta_st) > 1e-6 else 0.0
        mae_eV = float(np.mean(np.abs(delta_st - dft_ead)))

        spin_results[f"hyff_{tier}"] = {
            "supports_spin_multiplicity": True,
            "dft_ead_mean_eV": round(float(np.mean(dft_ead)), 4),
            "vertical_delta_E_T1_S0_mean_eV": round(float(np.mean(delta_st)), 4),
            "vertical_delta_E_T1_S0_std_eV": round(float(np.std(delta_st)), 4),
            "t1_relaxation_stabilization_eV": round(adiab_relax_gain_eV, 4),
            "mean_spin_force_shift_eV_A": round(force_diff_rms, 4),
            "zero_shot_mae_vs_dft_ead_eV": round(mae_eV, 4),
            "pearson_r_vs_dft_ead": round(r_val, 4),
            "spearman_rho_vs_dft_ead": round(rho_val, 4),
        }
        print(
            f"  [hyff_{tier:8s}] Vertical dE(T1-S0)={np.mean(delta_st):.3f}+/-{np.std(delta_st):.3f} eV "
            f"(DFT E_ad={np.mean(dft_ead):.3f} eV) | T1-RelaxGain={adiab_relax_gain_eV:.3f} eV | "
            f"|F_T1 - F_S0|={force_diff_rms:.3f} eV/A | MAE={mae_eV:.3f} eV | r={r_val:.3f}",
            flush=True,
        )
        ff.close()
        del ff
        torch.cuda.empty_cache()

    return {
        "oled_3dsat_generation": oled_gen_summary,
        "spin_splitting_zero_shot": spin_results,
    }


# ==============================================================================
# EXP 4: End-to-End ORBIT-3D Core 5-Stage Pipeline Replacement (Mode B & Mode C)
# ==============================================================================
def _build_conjugated_dataset(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # 48 diverse pi-conjugated donor-acceptor & heteroaromatic OLED/organic molecules with frontier gap (eV)
    rows = [
        ("c1ccccc1", 6.55), ("c1ccc2ccccc2c1", 4.75), ("c1ccc2cc3ccccc3cc2c1", 3.58),
        ("c1ccc(-c2ccccc2)cc1", 5.12), ("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1", 4.28),
        ("c1ccc2c(c1)[nH]c1ccccc12", 4.45), ("c1ccc2c(c1)oc1ccccc12", 4.82),
        ("c1ccc2c(c1)sc1ccccc12", 4.56), ("N#Cc1ccc(-c2ccccc2)cc1", 4.48),
        ("N#Cc1ccc(-c2ccc(N(c3ccccc3)c3ccccc3)cc2)cc1", 3.18),
        ("N#Cc1cc(C#N)c(-n2c3ccccc3c3ccccc32)cc1-n1c2ccccc2c2ccccc21", 2.86),
        ("COc1ccc(-c2nc3ccccc3s2)cc1", 3.92), ("Cc1ccc(-c2nc3ccccc3o2)cc1", 4.15),
        ("O=C1c2ccccc2C(=O)c2ccccc21", 3.42), ("O=S1(=O)c2ccccc2-c2ccccc21", 4.35),
        ("c1ccc(N(c2ccccc2)c2ccccc2)cc1", 4.05), ("Fc1ccc(-c2ccc(F)cc2)cc1", 5.18),
        ("Clc1ccc(-c2nc3ccccc3[nH]2)cc1", 4.08), ("c1cnc(-c2ccccn2)nc1", 4.95),
        ("c1ccc(-c2nc(-c3ccccc3)nc(-c3ccccc3)n2)cc1", 4.38),
        ("N#Cc1ccc(-n2c3ccccc3c3ccccc32)cc1", 3.64),
        ("O=C(c1ccccc1)c1ccc(-n2c3ccccc3c3ccccc32)cc1", 3.31),
        ("Cc1cc(-n2c3ccccc3c3ccccc32)ccc1C#N", 3.59),
        ("COc1ccc(N(c2ccccc2)c2ccc(C#N)cc2)cc1", 3.26),
        ("Fc1ccc(C#N)c(-n2c3ccccc3c3ccccc32)c1", 3.52),
        ("c1ccc2c(c1)c1ccccc1n2-c1ccc(-c2nc3ccccc3s2)cc1", 3.48),
        ("N#Cc1cccc(-n2c3ccccc3c3ccccc32)c1", 3.72),
        ("O=C1c2ccccc2-c2ccccc21", 3.68),
        ("c1ccc(-c2ccc3nsnc3c2)cc1", 2.94),
        ("c1ccc(N(c2ccccc2)c2ccc(-c3ccc4nsnc4c3)cc2)cc1", 2.48),
        ("COc1ccc(-c2ccc3nsnc3c2-c2ccc(OC)cc2)cc1", 2.66),
        ("Fc1ccc(-c2ccc3c(c2)C(=O)c2ccccc2-3)cc1", 3.55),
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("SMILES,TARGET\n")
        for smi, val in rows:
            f.write(f"{smi},{val}\n")


def run_exp4_orbit3d_e2e(
    pretrained_dir: Path,
    unimol_dir: Path,
    work_dir: Path,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print("[Exp 4] End-to-End ORBIT-3D 5-Stage Core Pipeline Replacement (Mode B & Mode C)", flush=True)
    print("=" * 88)

    dpa_cat = get_dpa_catalog(pretrained_dir)
    ref_dpa_plus = DPAPotentialGuidance(
        model_path=dpa_cat["dpa4_plus_omol25"]["path"],
        head="Default",
        atom_type_map=["C", "H", "N", "O", "F", "S", "Cl", "Br", "P", "I"],
        device=device,
    )
    ref_hyff_andante = HyFFPotentialGuidance(
        tier="andante",
        atom_type_map=["C", "H", "N", "O", "F", "S", "Cl", "Br", "P", "I"],
        device=device,
        route="reference",
    )

    from orbit3d.generator import ATOM_TYPES
    from rdkit.Chem import PandasTools

    configs_to_test = [
        # Mode C: Shared-Backbone Native Equivariant Descriptor Hook (`fitting_net` vs `readout_norms[-1]`)
        ("modeC_dpa4_mini", {"oracle_mode": "internal_dpa4", "dpa4_scale": "Mini"}, {"mlip_backend": "dpa4", "dpa4_scale": "Mini"}),
        ("modeC_dpa4_air", {"oracle_mode": "internal_dpa4", "dpa4_scale": "Air"}, {"mlip_backend": "dpa4", "dpa4_scale": "Air"}),
        ("modeC_hyff_presto", {"oracle_mode": "internal_hyff", "mlip_backend": "hyff", "hyff_tier": "presto"}, {"mlip_backend": "hyff", "hyff_tier": "presto"}),
        ("modeC_hyff_vivace", {"oracle_mode": "internal_hyff", "mlip_backend": "hyff", "hyff_tier": "vivace"}, {"mlip_backend": "hyff", "hyff_tier": "vivace"}),
        ("modeC_hyff_moderato", {"oracle_mode": "internal_hyff", "mlip_backend": "hyff", "hyff_tier": "moderato"}, {"mlip_backend": "hyff", "hyff_tier": "moderato"}),
        # Mode B: Internal 3D Uni-Mol Oracle + Pluggable Stage 4 Force Gate & Stage 5 Conformer Relaxer
        ("modeB_unimol_dpa4_mini", {"oracle_mode": "internal_unimol", "n_folds": 3, "oracle_epochs": 12}, {"mlip_backend": "dpa4", "dpa4_scale": "Mini"}),
        ("modeB_unimol_hyff_vivace", {"oracle_mode": "internal_unimol", "n_folds": 3, "oracle_epochs": 12}, {"mlip_backend": "hyff", "hyff_tier": "vivace"}),
        ("modeB_unimol_hyff_moderato", {"oracle_mode": "internal_unimol", "n_folds": 3, "oracle_epochs": 12}, {"mlip_backend": "hyff", "hyff_tier": "moderato"}),
    ]

    e2e_results: Dict[str, Any] = {}
    for run_tag, extra_oracle_cfg, extra_relax_cfg in configs_to_test:
        run_dir = work_dir / run_tag
        if run_dir.exists():
            shutil.rmtree(run_dir)
        data_csv = run_dir / "data" / "train.csv"
        _build_conjugated_dataset(data_csv)

        oracle_cfg = {
            "smiles_col": "SMILES",
            "target_cols": "TARGET",
            "train_oracle": True,
            "n_ensemble_heads": 5,
            "oracle_epochs": 25,
            "oracle_lr": 2e-4,
            "batch_size": 8,
            **extra_oracle_cfg,
        }
        stage5_cfg = {
            "enabled": True,
            "relax_steps": 20,
            "step_size": 0.008,
            "prop_grad_weight": 0.25,
            **extra_relax_cfg,
        }

        # Prepare symlinked builtin_pretrained_dir containing both dpa4 and unimol weights
        builtin_link = run_dir / "pretrained_bundle"
        builtin_link.mkdir(parents=True, exist_ok=True)
        for f in pretrained_dir.glob("DPA4-*.pt"):
            dst = builtin_link / f.name
            if not dst.exists():
                os.symlink(f, dst)
        if unimol_dir.exists():
            um_sub = builtin_link / "unimol"
            if not um_sub.exists():
                os.symlink(unimol_dir, um_sub)

        cfg = {
            "mode": "train_and_predict",
            "seed": 20260319,
            "device": device,
            "mount_paths": {
                "train_data_path": str(data_csv),
                "checkpoint_save_dir": str(run_dir / "ckpt"),
                "checkpoint_load_dir": str(run_dir / "ckpt"),
                "output_dir": str(run_dir / "outputs"),
                "builtin_pretrained_dir": str(builtin_link),
            },
            "oracle_config": oracle_cfg,
            "generator_config": {
                "d_model": 160,
                "n_heads": 4,
                "n_layers": 6,
                "sft_epochs": 12,
                "sft_lr": 3e-4,
                "grpo_mode": "multi_window_general",
                "grpo_steps": 10,
                "group_size": 6,
                "lr": 3e-5,
                "temperature": 0.90,
                "kl_beta": 0.04,
                "sft_anchor_weight": 0.15,
                "uncertainty_weight": 0.35,
            },
            "inference_config": {
                "target_interval": [3.30, 4.10],
                "run_specialist_grpo_before_gen": True,
                "specialist_grpo_steps": 8,
                "num_molecules": 20,
                "base_seed": 20260319,
                "temperature": 0.85,
                "top_p": 0.92,
            },
            "stage5_relax_config": stage5_cfg,
        }
        cfg_path = run_dir / "config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        t0 = time.perf_counter()
        summary = run_from_config(str(cfg_path))
        wall_s = time.perf_counter() - t0

        # Read oracle_meta.json
        oracle_val_mae = None
        desc_dim = None
        df_tr = pd.read_csv(data_csv)
        tr_std = float(df_tr["TARGET"].std(ddof=0))
        for sub in ["internal_dpa4_oracle", "internal_hyff_oracle", "internal_unimol_oracle"]:
            meta_p = run_dir / "ckpt" / sub / "oracle_meta.json"
            if meta_p.exists():
                with open(meta_p, "r") as f:
                    om = json.load(f)
                if "val_mae_physical" in om:
                    oracle_val_mae = float(om["val_mae_physical"])
                elif "cv_l1_norm" in om:
                    oracle_val_mae = float(om["cv_l1_norm"]) * tr_std
                desc_dim = om.get("desc_dim", 512)

        # Cross-evaluate generated SDF molecules using BOTH DPA4-Plus and HyFF-andante!
        sdf_path = summary.get("output_sdf", str(run_dir / "outputs" / "generated_molecules.sdf"))
        dpa_f_list, hyff_f_list = [], []
        if os.path.exists(sdf_path):
            df_sdf = PandasTools.LoadSDF(sdf_path, removeHs=False)
            for rmol in df_sdf["ROMol"].tolist():
                if rmol is None or rmol.GetNumConformers() == 0:
                    continue
                syms = [a.GetSymbol() for a in rmol.GetAtoms()]
                a_t = torch.tensor(
                    [ATOM_TYPES.index(s) if s in ATOM_TYPES else 0 for s in syms],
                    dtype=torch.long,
                    device=device,
                )
                pos_t = torch.tensor(rmol.GetConformer().GetPositions(), dtype=torch.float32, device=device)
                _, fd = ref_dpa_plus.compute_energy_and_forces(pos_t, a_t, clip_forces=False, zero_com_forces=True)
                _, fh = ref_hyff_andante.compute_energy_and_forces(pos_t, a_t, clip_forces=False, zero_com_forces=True)
                dpa_f_list.append(float(torch.norm(fd, dim=-1).mean().item()))
                hyff_f_list.append(float(torch.norm(fh, dim=-1).mean().item()))

        df_csv = pd.read_csv(summary["output_csv"])
        hit_col = "is_hit" if "is_hit" in df_csv.columns else "in_interval"
        hit_rate = float(df_csv[hit_col].mean()) if hit_col in df_csv.columns else float(summary.get("hit_rate", 0.0))

        # Reload determinism check
        cfg["mode"] = "predict"
        cfg["inference_config"]["run_specialist_grpo_before_gen"] = False
        cfg["mount_paths"]["output_dir"] = str(run_dir / "outputs_reload")
        cfg_rel = run_dir / "config_reload.json"
        with open(cfg_rel, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        sum_rel = run_from_config(str(cfg_rel))
        df_rel = pd.read_csv(sum_rel["output_csv"])
        max_reload_diff = float(np.max(np.abs(df_csv["y_pred"].values - df_rel["y_pred"].values)))

        res_item = {
            "desc_dim": desc_dim,
            "oracle_val_mae_eV": round(float(oracle_val_mae), 4) if oracle_val_mae is not None else None,
            "interval_hit_rate": round(hit_rate, 4),
            "self_init_force_eV_A": round(float(df_csv["init_force_eva"].mean()), 4),
            "self_final_force_eV_A": round(float(df_csv["final_force_eva"].mean()), 4),
            "cross_dpa4_plus_mean_force_eV_A": round(float(np.mean(dpa_f_list)), 4) if dpa_f_list else None,
            "cross_hyff_andante_mean_force_eV_A": round(float(np.mean(hyff_f_list)), 4) if hyff_f_list else None,
            "reload_max_diff": round(max_reload_diff, 8),
            "end_to_end_wall_sec": round(wall_s, 2),
        }
        e2e_results[run_tag] = res_item
        print(
            f"  [{run_tag:26s}] D={str(desc_dim):>3s} | ValMAE={res_item['oracle_val_mae_eV']} eV | "
            f"Hit={hit_rate*100:5.1f}% | SelfF: {res_item['self_init_force_eV_A']:.2f}->{res_item['self_final_force_eV_A']:.2f} | "
            f"Cross(DPA+={res_item['cross_dpa4_plus_mean_force_eV_A']}, HyFF={res_item['cross_hyff_andante_mean_force_eV_A']}) | "
            f"{wall_s:.1f}s",
            flush=True,
        )
        torch.cuda.empty_cache()

    ref_hyff_andante.close()
    del ref_dpa_plus, ref_hyff_andante
    return e2e_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--propmolflow_root", type=str, default="/sjtu-combustion/wjw/chensq/PropMolFlow")
    parser.add_argument("--pretrained_dir", type=str, default="/sjtu-combustion/wjw/chensq/pretrained")
    parser.add_argument("--unimol_dir", type=str, default="/sjtu-combustion/wjw/chensq/molgen3d/pretrained/unimol")
    parser.add_argument("--oled_dir", type=str, default="/sjtu-combustion/wjw/chensq/3D-SAT-OLED/SAT_OLED_Model/oled")
    parser.add_argument("--out_dir", type=str, default="/sjtu-combustion/wjw/chensq/results/hyff_vs_dpa4")
    parser.add_argument("--n_molecules", type=int, default=48)
    parser.add_argument("--n_timesteps", type=int, default=80)
    parser.add_argument("--worker", type=str, default="all", choices=["all", "gpu0", "gpu1", "exp3", "exp4", "merge"])
    parser.add_argument("--dual_gpu", action="store_true")
    parser.add_argument("--resume_exp3_exp4", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_exp3_exp4 and torch.cuda.device_count() >= 2:
        print("Resuming Exp 3 (GPU 0) and Exp 4 (GPU 1) in parallel across Dual A800 GPUs...", flush=True)
        base_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--propmolflow_root", args.propmolflow_root,
            "--pretrained_dir", args.pretrained_dir,
            "--unimol_dir", args.unimol_dir,
            "--oled_dir", args.oled_dir,
            "--out_dir", args.out_dir,
            "--n_molecules", str(args.n_molecules),
            "--n_timesteps", str(args.n_timesteps),
        ]
        t0 = time.time()
        p0 = subprocess.Popen(base_cmd + ["--worker", "exp3"], env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        p1 = subprocess.Popen(base_cmd + ["--worker", "exp4"], env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})
        rc0 = p0.wait()
        rc1 = p1.wait()
        if rc0 != 0 or rc1 != 0:
            raise RuntimeError(f"Resume Exp3/Exp4 failed: rc0={rc0}, rc1={rc1}")
        args.worker = "merge"

    if args.dual_gpu and torch.cuda.device_count() >= 2:
        print("Launching Dual-A800 Comprehensive HyFF vs. DPA-4 Benchmark Suite...", flush=True)
        base_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--propmolflow_root", args.propmolflow_root,
            "--pretrained_dir", args.pretrained_dir,
            "--unimol_dir", args.unimol_dir,
            "--oled_dir", args.oled_dir,
            "--out_dir", args.out_dir,
            "--n_molecules", str(args.n_molecules),
            "--n_timesteps", str(args.n_timesteps),
        ]
        t0 = time.time()
        p0 = subprocess.Popen(base_cmd + ["--worker", "gpu0"], env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        p1 = subprocess.Popen(base_cmd + ["--worker", "gpu1"], env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})
        rc0 = p0.wait()
        rc1 = p1.wait()
        if rc0 != 0 or rc1 != 0:
            raise RuntimeError(f"Dual-GPU benchmark failed: rc0={rc0}, rc1={rc1}")

        merged: Dict[str, Any] = {}
        for part_file in [out_dir / "part_gpu0.json", out_dir / "part_gpu1.json"]:
            if part_file.exists():
                with open(part_file, "r") as f:
                    part = json.load(f)
                for k, v in part.items():
                    if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                        merged[k].update(v)
                    else:
                        merged[k] = v

        summary_path = out_dir / "hyff_vs_dpa4_summary.json"
        with open(summary_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nALL 4 EXPERIMENTS COMPLETED IN {time.time() - t0:.1f}s -> {summary_path}", flush=True)
        return

    sys.path.insert(0, args.propmolflow_root)
    pmf = Path(args.propmolflow_root)
    pdir = Path(args.pretrained_dir)
    umdir = Path(args.unimol_dir)
    odir = Path(args.oled_dir)

    part_out: Dict[str, Any] = {}
    if args.worker in ("all", "gpu0"):
        part_out["exp1_hardware_ir_and_clash"] = run_exp1_hardware_ir_and_clash(pdir, odir, device="cuda:0")
        with open(out_dir / "part_gpu0.json", "w") as f:
            json.dump(part_out, f, indent=2)

        exp2_dict = {}
        exp2_dict["qm9_gap_ood"] = run_exp2_qm9_flow_task(
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
        exp2_dict["qm9_joint_gap_homo_lumo"] = run_exp2_qm9_flow_task(
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
        part_out["exp2_qm9_flow_cross_referee"] = exp2_dict
        with open(out_dir / "part_gpu0.json", "w") as f:
            json.dump(part_out, f, indent=2)

    if args.worker in ("all", "gpu1"):
        part_out.setdefault("exp2_qm9_flow_cross_referee", {})["qm9_homo"] = run_exp2_qm9_flow_task(
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
        with open(out_dir / "part_gpu1.json", "w") as f:
            json.dump(part_out, f, indent=2)

        part_out["exp3_oled_3dsat_ir_and_spin"] = run_exp3_oled_3dsat_and_spin(pdir, odir, device="cuda:0")
        with open(out_dir / "part_gpu1.json", "w") as f:
            json.dump(part_out, f, indent=2)

        part_out["exp4_orbit3d_e2e_pipeline"] = run_exp4_orbit3d_e2e(
            pretrained_dir=pdir,
            unimol_dir=umdir,
            work_dir=out_dir / "e2e_runs",
            device="cuda:0",
        )
        with open(out_dir / "part_gpu1.json", "w") as f:
            json.dump(part_out, f, indent=2)

    if args.worker == "exp3":
        res3 = {"exp3_oled_3dsat_ir_and_spin": run_exp3_oled_3dsat_and_spin(pdir, odir, device="cuda:0")}
        with open(out_dir / "part_exp3.json", "w") as f:
            json.dump(res3, f, indent=2)
        return

    if args.worker == "exp4":
        res4 = {
            "exp4_orbit3d_e2e_pipeline": run_exp4_orbit3d_e2e(
                pretrained_dir=pdir,
                unimol_dir=umdir,
                work_dir=out_dir / "e2e_runs",
                device="cuda:0",
            )
        }
        with open(out_dir / "part_exp4.json", "w") as f:
            json.dump(res4, f, indent=2)
        return

    if args.worker == "merge":
        merged_all: Dict[str, Any] = {}
        for pf in [
            out_dir / "part_gpu0.json",
            out_dir / "part_gpu1.json",
            out_dir / "part_exp3.json",
            out_dir / "part_exp4.json",
        ]:
            if pf.exists():
                with open(pf, "r") as f:
                    pdata = json.load(f)
                for k, v in pdata.items():
                    if k in merged_all and isinstance(merged_all[k], dict) and isinstance(v, dict):
                        merged_all[k].update(v)
                    else:
                        merged_all[k] = v
        summary_path = out_dir / "hyff_vs_dpa4_summary.json"
        with open(summary_path, "w") as f:
            json.dump(merged_all, f, indent=2)
        print(f"\nMERGED ALL 4 EXPERIMENTS -> {summary_path}", flush=True)
        return

    if args.worker == "all":
        summary_path = out_dir / "hyff_vs_dpa4_summary.json"
        with open(summary_path, "w") as f:
            json.dump(part_out, f, indent=2)


if __name__ == "__main__":
    main()
