#!/usr/bin/env python3
"""
Comprehensive Head-to-Head Comparison:
DPA-3.1-3M (SPICE2) vs. DPA4-Neo-OMol25-100M vs. DPA4-Air-OMol25-100M vs. DPA4-Air-OMat24 (PBE)
for Orthogonal Manifold-Projected 3D Molecular Flow Matching.

Evaluates:
  Part A: GPU Batched Throughput, Memory, Precision & Parameter Footprint on NVIDIA A800.
  Part B: De Novo 3D OLED Frontier Orbital Generation (PropMolFlow QM9 OOD: E_g, HOMO, LUMO, Joint).
  Part C: Pure Unmasked Organometallic Ir(III) OLED Emitter Generation (3D-SAT-OLED).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "orbit3d" / "compat"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.flow_matching.dpa_potential import DPAPotentialGuidance
from orbit3d.utils.evaluation import evaluate_generated_molecules
from benchmarks.flow_matching.guided_flow_sampler import GuidanceConfig, GuidedFlowSampler
from benchmarks.flow_matching.interval_guidance import orthogonal_manifold_projection
from benchmarks.runners.run_3d_sat_oled_benchmark import (
    OLED_ATOM_TYPES,
    EquivariantOLEDFlowField,
    EquivariantOLEDPropertyRegressor,
    load_oled_split,
)

HARTREE_TO_EV = 27.211386245988


def get_mlip_catalog(pretrained_dir: Path) -> Dict[str, Dict[str, str]]:
    return {
        "dpa31_3m_spice2": {
            "label": "DPA-3.1-3M (SPICE2, FP64)",
            "path": str(pretrained_dir / "DPA-3.1-3M.pt"),
            "head": "SPICE2",
        },
        "dpa4_neo_omol25": {
            "label": "DPA4-Neo-OMol25-100M (FP32)",
            "path": str(pretrained_dir / "DPA4-Neo-OMol25-100M-v20260820.pt"),
            "head": "Default",
        },
        "dpa4_air_omol25": {
            "label": "DPA4-Air-OMol25-100M (FP32)",
            "path": str(pretrained_dir / "DPA4-Air-OMol25-100M-v20260820.pt"),
            "head": "Default",
        },
        "dpa4_air_omat24": {
            "label": "DPA4-Air-OMat24 (Inorganic PBE, FP32)",
            "path": str(pretrained_dir / "DPA4-Air-OMat24-v20260805.pt"),
            "head": "Default",
        },
    }


def run_speed_microbenchmark(pretrained_dir: Path, oled_dir: Path, device: str = "cuda:0") -> Dict[str, Dict[str, float]]:
    print("\n================================================================================")
    print("[Part A] Running MLIP Throughput & Footprint Microbenchmark on A800...", flush=True)
    catalog = get_mlip_catalog(pretrained_dir)

    # Synthetic 48-molecule organic batch (~19 atoms/mol = 912 atoms)
    torch.manual_seed(42)
    coords_org = torch.randn(48 * 19, 3, device=device) * 1.6
    atypes_org = torch.randint(0, 5, (48 * 19,), device=device)
    batch_org = torch.arange(48, device=device).repeat_interleave(19)

    # Real 24-complex Ir(III) organometallic batch from 3D-SAT-OLED (~2000 atoms)
    test_mols = load_oled_split(oled_dir / "test.npz")[:24]
    coords_ir = torch.cat([m["coords"] for m in test_mols], dim=0).to(device)
    atypes_ir = torch.cat([m["a_idx"] for m in test_mols], dim=0).to(device)
    batch_ir = torch.cat(
        [torch.full((m["coords"].shape[0],), b_i, dtype=torch.long) for b_i, m in enumerate(test_mols)],
        dim=0,
    ).to(device)

    speed_results = {}
    for key, meta in catalog.items():
        ckpt_mb = os.path.getsize(meta["path"]) / (1024.0 * 1024.0)
        m_org = DPAPotentialGuidance(
            model_path=meta["path"],
            head=meta["head"],
            atom_type_map=["C", "H", "N", "O", "F"],
            device=device,
        )
        n_params = sum(p.numel() for p in m_org.wrapper.parameters())

        for _ in range(3):
            m_org.compute_energy_and_forces(coords_org, atypes_org, batch_org)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_rep = 15
        for _ in range(n_rep):
            m_org.compute_energy_and_forces(coords_org, atypes_org, batch_org)
        torch.cuda.synchronize()
        org_batch_ms = (time.perf_counter() - t0) * 1000.0 / n_rep

        m_org.set_atom_type_map(OLED_ATOM_TYPES)
        for _ in range(3):
            m_org.compute_energy_and_forces(coords_ir, atypes_ir, batch_ir)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(10):
            m_org.compute_energy_and_forces(coords_ir, atypes_ir, batch_ir)
        torch.cuda.synchronize()
        ir_batch_ms = (time.perf_counter() - t1) * 1000.0 / 10.0

        speed_results[key] = {
            "label": meta["label"],
            "ckpt_size_MB": round(ckpt_mb, 2),
            "active_params_M": round(n_params / 1e6, 3),
            "coord_dtype": str(m_org.coord_dtype).replace("torch.", ""),
            "qm9_batch48_ms": round(org_batch_ms, 2),
            "qm9_ms_per_mol": round(org_batch_ms / 48.0, 3),
            "oled_ir_batch24_ms": round(ir_batch_ms, 2),
            "oled_ir_ms_per_complex": round(ir_batch_ms / 24.0, 3),
        }
        print(
            f"  [{key:20s}] size={ckpt_mb:5.1f} MB | params={n_params/1e6:5.2f}M | {str(m_org.coord_dtype):13s} | "
            f"QM9(48)={org_batch_ms:6.1f} ms ({org_batch_ms/48:.2f} ms/mol) | "
            f"Ir-OLED(24)={ir_batch_ms:6.1f} ms ({ir_batch_ms/24:.2f} ms/mol)",
            flush=True,
        )
        del m_org
        torch.cuda.empty_cache()

    return speed_results


def run_propmolflow_mlip_comparison(
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
    print(f"[Part B: {task_key}] Comparing DPA-3.1 vs DPA4 on PropMolFlow ({prop_name})...", flush=True)

    catalog = get_mlip_catalog(pretrained_dir)
    all_props = [prop_name] + [p for p in extra_intervals_raw.keys() if p != prop_name]

    # Load base sampler once (initially with DPA4-Air-OMol25 as the reference oracle)
    ref_oracle = DPAPotentialGuidance(
        model_path=catalog["dpa4_air_omol25"]["path"],
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

    target_intervals_norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
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
        ("2_dpa4_neo_omol25", "dpa4_neo_omol25", guided_cfg),
        ("3_dpa4_air_omol25", "dpa4_air_omol25", guided_cfg),
        ("4_dpa4_air_omat24", "dpa4_air_omat24", guided_cfg),
    ]

    results: Dict[str, Dict[str, float]] = {}
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

        # Evaluate all generated conformers using the unified DPA4-Air-OMol25 reference oracle
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
            f"OMol25_RMS_F={metrics.get('dpa_rms_force_eV_A', 0.0):5.2f} eV/A | "
            f"LowStrain={metrics.get('dpa_low_strain_frac', 0.0)*100:5.1f}% ({elapsed:.1f}s)",
            flush=True,
        )
        if sampler.dpa_guidance is not None:
            del sampler.dpa_guidance
            sampler.dpa_guidance = None
        torch.cuda.empty_cache()

    return results


def run_pure_organometallic_oled_comparison(
    oled_dir: Path,
    pretrained_dir: Path,
    device: str = "cuda:0",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Part C: Evaluate PURE UNMASKED MLIP forces (no artificial Ir masking or harmonic springs!)
    on 3D-SAT-OLED heavy transition-metal Ir(III) complexes across DPA-3.1-3M vs DPA4-OMol25 vs DPA4-OMat24.
    """
    print("\n================================================================================")
    print("[Part C] Pure Unmasked Ir(III) Organometallic OLED Comparison (3D-SAT-OLED)...", flush=True)

    train_mols = load_oled_split(oled_dir / "train.npz")
    test_mols = load_oled_split(oled_dir / "test.npz")
    dev = torch.device(device)

    all_train_props = torch.stack([m["props"][:3] for m in train_mols], dim=0).to(dev)
    prop_mean = all_train_props.mean(dim=0)
    prop_std = all_train_props.std(dim=0).clamp(min=1e-4)

    # Train differentiable 3D OLED regressor & flow field with fixed seed
    torch.manual_seed(42)
    regressor = EquivariantOLEDPropertyRegressor().to(dev)
    opt_reg = torch.optim.AdamW(regressor.parameters(), lr=1.5e-3, weight_decay=1e-4)
    for _ in range(25):
        regressor.train()
        for idx in torch.randperm(len(train_mols))[:300]:
            m = train_mols[int(idx)]
            c = m["coords"].to(dev) + 0.04 * torch.randn_like(m["coords"].to(dev))
            pred = regressor(c, m["a_idx"].to(dev))
            t_norm = (m["props"][:3].to(dev) - prop_mean) / prop_std
            loss = F.smooth_l1_loss(pred, t_norm)
            opt_reg.zero_grad()
            loss.backward()
            opt_reg.step()
    regressor.eval()
    for p in regressor.parameters():
        p.requires_grad_(False)

    flow_net = EquivariantOLEDFlowField().to(dev)
    opt_flow = torch.optim.AdamW(flow_net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    noise_sigma = 0.35
    for _ in range(15):
        flow_net.train()
        for idx in torch.randperm(len(train_mols))[:250]:
            m = train_mols[int(idx)]
            x_1 = m["coords"].to(dev)
            a_idx = m["a_idx"].to(dev)
            y_true_n = ((m["props"][0].to(dev) - prop_mean[0]) / prop_std[0]).view(1)
            y_min_n, y_max_n = flow_net.interval_cond.sample_training_intervals(y_true_n)
            t_val = float(torch.rand(1).item())
            noise = noise_sigma * torch.randn_like(x_1)
            noise = noise - noise.mean(dim=0, keepdim=True)
            x_t = (1.0 - t_val) * (x_1 + noise) + t_val * x_1
            x_1_pred = flow_net(x_t, a_idx, t_val, y_min_n, y_max_n, m["bond_src"].to(dev), m["bond_dst"].to(dev))
            loss = F.mse_loss(x_1_pred, x_1)
            opt_flow.zero_grad()
            loss.backward()
            opt_flow.step()
    flow_net.eval()

    catalog = get_mlip_catalog(pretrained_dir)
    mlip_models: Dict[str, DPAPotentialGuidance] = {}
    for k, meta in catalog.items():
        mlip_models[k] = DPAPotentialGuidance(
            model_path=meta["path"],
            head=meta["head"],
            atom_type_map=OLED_ATOM_TYPES,
            device=device,
            max_force_norm=15.0,
        )
    ref_omol25 = mlip_models["dpa4_air_omol25"]

    tasks = [
        ("OLED_Eg_Bandgap", 0, 0.35, 1.10),
        ("OLED_HOMO_Level", 1, 0.30, 1.05),
        ("OLED_LUMO_Level", 2, 0.30, 1.05),
    ]
    modes = [
        ("0_naive_no_mlip", None),
        ("1_dpa31_3m_spice2_pure", "dpa31_3m_spice2"),
        ("2_dpa4_air_omat24_pure", "dpa4_air_omat24"),
        ("3_dpa4_neo_omol25_pure", "dpa4_neo_omol25"),
        ("4_dpa4_air_omol25_pure", "dpa4_air_omol25"),
    ]

    n_eval = 32
    n_steps = 25
    dt = 1.0 / n_steps
    ir_idx = OLED_ATOM_TYPES.index("Ir")

    sat_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task_name, prop_idx, std_low, std_high in tasks:
        target_min_eV = float((prop_mean[prop_idx] + std_low * prop_std[prop_idx]).item())
        target_max_eV = float((prop_mean[prop_idx] + std_high * prop_std[prop_idx]).item())
        y_min_norm = ((torch.tensor(target_min_eV, device=dev) - prop_mean[prop_idx]) / prop_std[prop_idx]).view(1)
        y_max_norm = ((torch.tensor(target_max_eV, device=dev) - prop_mean[prop_idx]) / prop_std[prop_idx]).view(1)
        print(f"\n  --- {task_name} [{target_min_eV:.3f}, {target_max_eV:.3f}] eV ---", flush=True)

        task_res: Dict[str, Dict[str, float]] = {}
        for mode_name, mlip_key in modes:
            active_mlip = mlip_models[mlip_key] if mlip_key is not None else None

            # Batched state over all n_eval complexes!
            x_list = []
            a_list = []
            b_idx_list = []
            for idx in range(n_eval):
                m = test_mols[idx]
                x_ref = m["coords"].to(dev)
                torch.manual_seed(1000 + idx)
                noise = noise_sigma * torch.randn_like(x_ref)
                noise = noise - noise.mean(dim=0, keepdim=True)
                x_list.append(x_ref + noise)
                a_list.append(m["a_idx"].to(dev))
                b_idx_list.append(torch.full((x_ref.shape[0],), idx, dtype=torch.long, device=dev))

            a_all = torch.cat(a_list, dim=0)
            b_all = torch.cat(b_idx_list, dim=0)

            t_start_mode = time.perf_counter()
            for s_idx in range(n_steps):
                t_val = s_idx * dt
                next_x_list = []
                g_prop_list = []
                for idx in range(n_eval):
                    m = test_mols[idx]
                    x_t = x_list[idx]
                    a_i = a_list[idx]
                    with torch.no_grad():
                        x_1_hat = flow_net(x_t, a_i, t_val, y_min_norm, y_max_norm, m["bond_src"].to(dev), m["bond_dst"].to(dev))
                        v_flow = (x_1_hat - x_t) / max(1.0 - t_val, 0.05)
                        x_next = x_t + dt * v_flow
                    next_x_list.append(x_next)

                    if t_val >= 0.35:
                        with torch.enable_grad():
                            x_in = x_next.detach().clone().requires_grad_(True)
                            pred_n = regressor(x_in, a_i)[prop_idx]
                            hinge = torch.relu(y_min_norm[0] - pred_n).pow(2) + torch.relu(pred_n - y_max_norm[0]).pow(2)
                            if hinge.item() > 0:
                                grad_x = torch.autograd.grad(hinge, x_in)[0]
                                gp = -grad_x.detach()
                                gp = gp - gp.mean(dim=0, keepdim=True)
                                rms_g = torch.sqrt((gp * gp).sum(dim=-1).mean()) + 1e-8
                                gp = gp * (min(1.0, math.sqrt(float(hinge.item())) / 0.12) / rms_g)
                            else:
                                gp = torch.zeros_like(x_next)
                        g_prop_list.append(gp)

                if t_val >= 0.35:
                    x_packed = torch.cat(next_x_list, dim=0)
                    gp_packed = torch.cat(g_prop_list, dim=0)
                    if active_mlip is not None:
                        # PURE UNMASKED MLIP FORCE across all 32 Ir(III) complexes in 1 super-frame!
                        _, f_phys = active_mlip.compute_energy_and_forces(
                            coords=x_packed,
                            dataset_atom_indices=a_all,
                            node_batch_idx=b_all,
                            clip_forces=True,
                            zero_com_forces=True,
                        )
                        g_tilde, _ = orthogonal_manifold_projection(
                            g_prop=gp_packed,
                            f_phys=f_phys,
                            node_batch_idx=b_all,
                            batch_size=n_eval,
                            atom_level_projection=True,
                            mol_level_projection=True,
                        )
                        shift_all = dt * (1.35 * g_tilde + 0.018 * f_phys)
                    else:
                        shift_all = dt * (1.35 * gp_packed)

                    snorm = torch.norm(shift_all, dim=-1, keepdim=True) + 1e-8
                    shift_all = shift_all * torch.clamp(0.08 / snorm, max=1.0)
                    splits = [x.shape[0] for x in next_x_list]
                    for idx, sh in enumerate(torch.split(shift_all, splits, dim=0)):
                        next_x_list[idx] = next_x_list[idx] + sh - sh.mean(dim=0, keepdim=True)

                x_list = next_x_list

            # Terminal 10-step pure unmasked MLIP relaxation with interval barrier
            if active_mlip is not None:
                for _ in range(10):
                    x_packed = torch.cat(x_list, dim=0)
                    _, f_rel = active_mlip.compute_energy_and_forces(
                        coords=x_packed,
                        dataset_atom_indices=a_all,
                        node_batch_idx=b_all,
                        clip_forces=True,
                        zero_com_forces=True,
                    )
                    splits = [x.shape[0] for x in x_list]
                    for idx, fr in enumerate(torch.split(f_rel, splits, dim=0)):
                        disp = 0.005 * fr
                        dn = torch.norm(disp, dim=-1, keepdim=True) + 1e-8
                        disp = disp * torch.clamp(0.018 / dn, max=1.0)
                        x_cand = x_list[idx] + disp - disp.mean(dim=0, keepdim=True)
                        with torch.no_grad():
                            p_old = float((regressor(x_list[idx], a_list[idx])[prop_idx] * prop_std[prop_idx] + prop_mean[prop_idx]).item())
                            p_new = float((regressor(x_cand, a_list[idx])[prop_idx] * prop_std[prop_idx] + prop_mean[prop_idx]).item())
                        e_old = max(0.0, target_min_eV - p_old, p_old - target_max_eV)
                        e_new = max(0.0, target_min_eV - p_new, p_new - target_max_eV)
                        if e_new <= e_old + 1e-4:
                            x_list[idx] = x_cand

            elapsed_mode = time.perf_counter() - t_start_mode

            # Evaluate generated Ir(III) complexes on ground-truth bond geometry + DPA4-Air-OMol25 reference forces
            x_final_packed = torch.cat(x_list, dim=0)
            _, ref_forces_all = ref_omol25.compute_energy_and_forces(
                coords=x_final_packed,
                dataset_atom_indices=a_all,
                node_batch_idx=b_all,
                clip_forces=False,
                zero_com_forces=True,
            )
            ref_splits = torch.split(ref_forces_all, [x.shape[0] for x in x_list], dim=0)

            m_metrics = []
            for idx in range(n_eval):
                m = test_mols[idx]
                coords = x_list[idx]
                bs = m["bond_src"].to(dev)
                bd = m["bond_dst"].to(dev)
                bref = m["bond_ref_len"].to(dev)
                a_i = a_list[idx]

                d_bonds = torch.norm(coords[bs] - coords[bd], dim=-1)
                strain_all = float(torch.abs(d_bonds - bref).mean().item())
                max_stretch = float(torch.abs(d_bonds - bref).max().item())

                # Specifically measure Ir-ligand coordination bond accuracy!
                ir_bond_mask = (a_i[bs] == ir_idx) | (a_i[bd] == ir_idx)
                ir_coord_mae = float(torch.abs(d_bonds[ir_bond_mask] - bref[ir_bond_mask]).mean().item()) if ir_bond_mask.any() else strain_all

                dist_mat = torch.cdist(coords, coords)
                bonded = torch.eye(coords.shape[0], dtype=torch.bool, device=dev)
                bonded[bs, bd] = True
                bonded[bd, bs] = True
                clash_free = float(dist_mat[~bonded].min().item() > 1.05)
                geom_valid = float((max_stretch < 0.22) and (clash_free > 0.5))

                with torch.no_grad():
                    y_hat = float((regressor(coords, a_i)[prop_idx] * prop_std[prop_idx] + prop_mean[prop_idx]).item())
                in_int = float((y_hat >= target_min_eV) and (y_hat <= target_max_eV))
                int_err = max(0.0, target_min_eV - y_hat, y_hat - target_max_eV)

                rf = ref_splits[idx]
                rms_f = float(torch.sqrt((rf * rf).sum(dim=-1).mean()).item())
                ir_atom_f = float(rf[a_i == ir_idx].norm(dim=-1).mean().item()) if (a_i == ir_idx).any() else rms_f

                m_metrics.append({
                    "in_interval": in_int,
                    "interval_err_eV": int_err,
                    "bond_strain_mae_A": strain_all,
                    "ir_coord_bond_mae_A": ir_coord_mae,
                    "max_bond_stretch_A": max_stretch,
                    "geom_valid": geom_valid,
                    "valid_and_in_interval": geom_valid * in_int,
                    "omol25_rms_force_eV_A": rms_f,
                    "ir_metal_force_eV_A": ir_atom_f,
                })

            agg = {k: float(np.mean([x[k] for x in m_metrics])) for k in m_metrics[0].keys()}
            agg["time_sec"] = elapsed_mode
            task_res[mode_name] = agg
            print(
                f"    [{mode_name:24s}] Hit={agg['in_interval']*100:5.1f}% | "
                f"GeomValid={agg['geom_valid']*100:5.1f}% | Valid&Hit={agg['valid_and_in_interval']*100:5.1f}% | "
                f"BondMAE={agg['bond_strain_mae_A']:5.3f} A | Ir-BondMAE={agg['ir_coord_bond_mae_A']:5.3f} A | "
                f"OMol25_F={agg['omol25_rms_force_eV_A']:5.2f} eV/A (Ir_F={agg['ir_metal_force_eV_A']:5.2f}) ({elapsed_mode:.1f}s)",
                flush=True,
            )
        sat_summary[task_name] = task_res

    return sat_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--propmolflow_root", type=str, default="/sjtu-combustion/wjw/chensq/PropMolFlow")
    parser.add_argument("--pretrained_dir", type=str, default="/sjtu-combustion/wjw/chensq/pretrained")
    parser.add_argument("--oled_dir", type=str, default="/sjtu-combustion/wjw/chensq/3D-SAT-OLED/SAT_OLED_Model/oled")
    parser.add_argument("--out_dir", type=str, default="/sjtu-combustion/wjw/chensq/results/dpa3_vs_dpa4")
    parser.add_argument("--n_molecules", type=int, default=48)
    parser.add_argument("--n_timesteps", type=int, default=80)
    parser.add_argument("--worker", type=str, default="all", choices=["all", "gpu0", "gpu1"])
    parser.add_argument("--dual_gpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dual_gpu and torch.cuda.device_count() >= 2:
        print("Launching Dual-A800 DPA-3.1 vs DPA-4 Head-to-Head Comparison...", flush=True)
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
            raise RuntimeError(f"Dual-GPU comparison failed: rc0={rc0}, rc1={rc1}")

        merged = {}
        for pf in [out_dir / "part_gpu0.json", out_dir / "part_gpu1.json"]:
            if pf.exists():
                with open(pf, "r") as f:
                    merged.update(json.load(f))
        out_json = out_dir / "dpa3_vs_dpa4_summary.json"
        with open(out_json, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nDual-A800 DPA-3.1 vs DPA-4 Comparison completed in {time.time() - t0:.1f}s -> {out_json}", flush=True)
        return

    sys.path.insert(0, args.propmolflow_root)
    pmf = Path(args.propmolflow_root)
    pdir = Path(args.pretrained_dir)
    odir = Path(args.oled_dir)

    out_data = {}
    if args.worker in ("all", "gpu0"):
        out_data["speed_profile"] = run_speed_microbenchmark(pdir, odir, device="cuda:0")
        out_data["qm9_gap_ood"] = run_propmolflow_mlip_comparison(
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
        out_data["qm9_homo"] = run_propmolflow_mlip_comparison(
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
        with open(out_dir / "part_gpu0.json", "w") as f:
            json.dump(out_data, f, indent=2)

    if args.worker in ("all", "gpu1"):
        out_data["qm9_joint_gap_homo_lumo"] = run_propmolflow_mlip_comparison(
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
        out_data["oled_3dsat_pure_unmasked"] = run_pure_organometallic_oled_comparison(
            oled_dir=odir,
            pretrained_dir=pdir,
            device="cuda:0",
        )
        with open(out_dir / "part_gpu1.json", "w") as f:
            json.dump(out_data, f, indent=2)

    if args.worker == "all":
        with open(out_dir / "dpa3_vs_dpa4_summary.json", "w") as f:
            json.dump(out_data, f, indent=2)


if __name__ == "__main__":
    main()
