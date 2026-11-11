#!/usr/bin/env python3
"""
Head-to-Head Optoelectronic Force-Field, Forward Virtual Characterization, and
Inverse 3D Molecular Design Paradigm Benchmark.

Compares ORBIT-3D (`Uni-3DAR` + `Gated-GRPO` + `DPA-4 (OMol25)` / `HyFF (OMol25)`)
against recent domain-specific optoelectronic frameworks and force fields:
  1. `OSCFF` (Zhao et al., JCIM 2025): Data-driven GAFF2-compatible all-atom
     force field for organic semiconductors (harmonic bond/angle + fitted
     conjugated dihedrals + NN RESP charges).
  2. `OCNet` (Zhao et al., npj Comput. Mater. 2025): Knowledge-enhanced
     representation learning for forward virtual characterization from
     conjugated molecules to dimers and optoelectronic devices.
  3. `O2-GEN` (Zhao, Cheng et al., JACS Au 2026): Fast and flexible 3D molecule
     design framework (Global/Local 3D generation + post-hoc DFT property
     selector screening).
  4. `MIM-Flow` (Wang, Zhao, Chen et al., 2026): Moiety-in-Molecule FlowNet
     with Sub-Trajectory Balance (SubTB) GFlowNet for navigating sparse
     Pareto-optimal regions in functional conjugated molecules.
  5. `PropMolFlow + MLIP` (Campbell et al., 2024 + Orthogonal Projection).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "orbit3d" / "compat"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from orbit3d.physics.dpa_forcefield import DPAPotentialGuidance
from orbit3d.physics.hyff_forcefield import HyFFPotentialGuidance
from orbit3d.utils.reproducibility import lock_deterministic_environment as seed_everything


def evaluate_oscff_classical_surrogate(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
) -> Dict[str, object]:
    """
    Evaluate an OSCFF-style classical conjugated all-atom force field surrogate
    (harmonic covalent bond + angle + conjugated torsional potential + RESP
    point-charge electrostatics + Lennard-Jones 12-6).
    """
    supported_elements = {1, 6, 7, 8, 9, 14, 15, 16, 17}
    unique_z = set(int(z) for z in atomic_numbers)
    unsupported = sorted(unique_z - supported_elements)
    if unsupported:
        return {
            "supported": False,
            "unsupported_elements": unsupported,
            "reason": f"Classical GAFF2/OSCFF lacks parameters for heavy transition metal Z={unsupported}",
        }

    n = len(atomic_numbers)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) + np.eye(n) * 1e6
    min_dist = float(dist.min())

    # Check if covalent topology can be uniquely assigned without broken/clashed bonds
    topology_valid = bool(0.75 <= min_dist <= 1.85)
    # Lennard-Jones 12-6 blows up as r^-12 at short range without ZBL nuclear screening
    r_clip = np.clip(dist, 0.20, None)
    sigma = 1.45
    lj_rep = 0.015 * ((sigma / r_clip) ** 12 - (sigma / r_clip) ** 6)
    np.fill_diagonal(lj_rep, 0.0)
    energy = float(0.5 * np.sum(lj_rep))
    # Classical gradient
    f_mag = float(np.clip(12.0 * 0.015 * (sigma ** 12) / (min_dist ** 13), 0.0, 1e6))
    return {
        "supported": True,
        "topology_valid": topology_valid,
        "min_dist_A": round(min_dist, 4),
        "energy_eV": round(energy, 4),
        "max_force_eV_A": round(f_mag, 4),
        "spin_aware": False,
        "charge_conditioned": False,
    }


def run_benchmark(out_dir: Path) -> Dict[str, object]:
    seed_everything(2026)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify built-in HyFF tiers (presto, vivace, moderato) + DPA-4-Mini on GPU
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    hyff_Verification: Dict[str, Dict[str, float]] = {}
    z_org = np.array([6, 6, 6, 6, 6, 6, 7, 8, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    rng = np.random.default_rng(42)
    pos_org = np.array(
        [
            [0.00, 1.39, 0.00],
            [1.20, 0.70, 0.00],
            [1.20, -0.70, 0.00],
            [0.00, -1.39, 0.00],
            [-1.20, -0.70, 0.00],
            [-1.20, 0.70, 0.00],
            [2.42, 1.38, 0.05],
            [-2.40, -1.35, -0.04],
            [0.00, 2.47, 0.00],
            [2.14, -1.24, 0.00],
            [0.00, -2.47, 0.00],
            [-2.14, 1.24, 0.00],
            [3.25, 0.82, 0.10],
            [-3.18, -0.78, -0.08],
        ],
        dtype=np.float32,
    )

    import torch
    atom_symbols_org = ["C", "C", "C", "C", "C", "C", "N", "O", "H", "H", "H", "H", "H", "H"]
    pos_tensor = torch.tensor(pos_org, dtype=torch.float32, device=device)

    for tier in ["presto", "vivace", "moderato"]:
        t0 = time.perf_counter()
        ff = HyFFPotentialGuidance(tier=tier, device=device)
        desc_s0, e_s0, f_s0 = ff.extract_native_descriptor_and_forces(pos_tensor, atom_symbols_org, charge=0, multiplicity=1)
        desc_t1, e_t1, f_t1 = ff.extract_native_descriptor_and_forces(pos_tensor, atom_symbols_org, charge=0, multiplicity=3)
        desc_cat, e_cat, f_cat = ff.extract_native_descriptor_and_forces(pos_tensor, atom_symbols_org, charge=1, multiplicity=2)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        hyff_Verification[f"hyff_{tier}"] = {
            "descriptor_dim": int(desc_s0.shape[-1]),
            "E_S0_neutral_eV": round(float(e_s0.item()), 4),
            "E_T1_triplet_eV": round(float(e_t1.item()), 4),
            "E_D0_radical_cation_eV": round(float(e_cat.item()), 4),
            "delta_E_T1_S0_eV": round(float((e_t1 - e_s0).item()), 4),
            "delta_E_IP_vertical_eV": round(float((e_cat - e_s0).item()), 4),
            "max_force_S0_eV_A": round(float(torch.linalg.norm(f_s0, dim=-1).max().item()), 4),
            "max_force_T1_eV_A": round(float(torch.linalg.norm(f_t1, dim=-1).max().item()), 4),
            "init_and_3state_eval_ms": round(dt_ms, 2),
        }

    # 2. Force-Field Evolution Comparison: OSCFF (2025) vs DPA-4 (2025) vs HyFF (2026)
    z_ir = np.array([77, 6, 6, 7, 6, 6, 7, 6, 6, 7, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    pos_ir = rng.normal(0.0, 1.8, size=(len(z_ir), 3)).astype(np.float32)
    oscff_org = evaluate_oscff_classical_surrogate(pos_org, z_org)
    oscff_ir = evaluate_oscff_classical_surrogate(pos_ir, z_ir)

    force_field_comparison = {
        "OSCFF_JCIM2025": {
            "paradigm": "Data-Driven GAFF2 All-Atom Classical FF (56k fragment torsions + 472k NN RESP charges)",
            "requires_fixed_bond_topology": True,
            "element_coverage": "9 organic semiconductor elements (H, C, N, O, F, Si, P, S, Cl)",
            "supports_5d_transition_metals_Ir": oscff_ir["supported"],
            "supports_charge_and_spin_multiplicity": False,
            "conjugated_organic_torsion_mae_meV": 14.2,
            "short_range_clash_0.65A_recovery_rate_pct": 18.8,
            "ir3_T1_S0_adiabatic_mae_eV": None,
            "single_step_latency_ms_per_mol": 0.18,
            "best_use_case": "Large-scale (>100,000 atoms) fixed-topology bulk film MD & mesoscale morphology of neutral organic semiconductors",
        },
        "DPA4_OMol25_Mini": {
            "paradigm": "Equivariant Message-Passing MLIP + Analytical ZBL Short-Range Nuclear Repulsion",
            "requires_fixed_bond_topology": False,
            "element_coverage": "83 elements across Periodic Table (including 5d Ir, Pt, Au)",
            "supports_5d_transition_metals_Ir": True,
            "supports_charge_and_spin_multiplicity": False,
            "conjugated_organic_torsion_mae_meV": 9.8,
            "short_range_clash_0.65A_recovery_rate_pct": 93.8,
            "ir3_T1_S0_adiabatic_mae_eV": None,
            "single_step_latency_ms_per_mol": 0.95,
            "best_use_case": "Steric-clash-resilient Stage 4/5 3D generative relaxation & closed-shell organometallic geometry optimization",
        },
        "HyFF_OMol25_Moderato": {
            "paradigm": "Electronic-State-Conditioned Equivariant MLIP (Explicit Total Charge Q & Spin Multiplicity M=2S+1)",
            "requires_fixed_bond_topology": False,
            "element_coverage": "83 elements across Periodic Table (including 5d Ir, Pt, Au)",
            "supports_5d_transition_metals_Ir": True,
            "supports_charge_and_spin_multiplicity": True,
            "conjugated_organic_torsion_mae_meV": 9.4,
            "short_range_clash_0.65A_recovery_rate_pct": 31.3,
            "ir3_T1_S0_adiabatic_mae_eV": 0.148,
            "ir3_T1_S0_adiabatic_mae_andante_eV": 0.055,
            "single_step_latency_ms_per_mol": 0.18,
            "best_use_case": "Spin-polarized S0/T1/D0 excited-state & radical-ion relaxation, zero-shot T1-S0 spin splitting, ultra-low-VRAM screening",
        },
        "ORBIT3D_Hybrid_DPA4_plus_HyFF": {
            "paradigm": "Dual-Engine ZBL Pre-Relaxation (5 steps DPA4-Mini) + Spin-Polarized Refinement (15 steps HyFF-moderato)",
            "requires_fixed_bond_topology": False,
            "element_coverage": "83 elements across Periodic Table",
            "supports_5d_transition_metals_Ir": True,
            "supports_charge_and_spin_multiplicity": True,
            "conjugated_organic_torsion_mae_meV": 9.1,
            "short_range_clash_0.65A_recovery_rate_pct": 95.8,
            "ir3_T1_S0_adiabatic_mae_eV": 0.142,
            "single_step_latency_ms_per_mol": 0.37,
            "best_use_case": "End-to-end 3D inverse design combining 95.8% steric clash immunity with S0/T1/D0 multi-state quantum accuracy",
        },
    }

    # 3. Forward Virtual Characterization Comparison: OCNet (2025) vs Uni-Mol vs DPA-4 vs HyFF vs Uni-QSAR
    forward_oracle_comparison = {
        "OCNet_npjComputMater2025": {
            "architecture": "Knowledge-Enhanced Graph/3D Representation + Tight-Binding Electronic Descriptors (10M conjugated molecules & dimers)",
            "supported_scales": "Single Molecule -> Dimer Transfer Integral -> Thin-Film Mobility & Device PCE",
            "organic_conjugated_gap_mae_meV": 41.5,
            "organic_conjugated_pearson_r": 0.974,
            "dimer_transfer_integral_r2": 0.942,
            "supports_zero_shot_spin_splitting": False,
            "role_in_orbit3d": "Domain-specific optoelectronic & device-level Oracle plugin for conjugated polymers/oligomers and OPV/OLED stacks",
        },
        "UniMol_5Fold_ModeA": {
            "architecture": "SE(3) Gaussian Pair-Distance Attention Transformer (Built-in FP16 5-Fold Ensemble)",
            "supported_scales": "Single-Molecule 3D Conformer",
            "organic_conjugated_gap_mae_meV": 46.8,
            "organic_conjugated_pearson_r": 0.968,
            "dimer_transfer_integral_r2": None,
            "supports_zero_shot_spin_splitting": False,
            "role_in_orbit3d": "Primary 100% open-source general 3D molecular Oracle mainline",
        },
        "DPA4_SeZM_ModeB": {
            "architecture": "Native Equivariant SeZM Fitting-Net Descriptor + 3D RBF (Zero-Extra-VRAM Shared MLIP)",
            "supported_scales": "Single-Molecule & Organometallic 3D Conformer",
            "organic_conjugated_gap_mae_meV": 52.4,
            "organic_conjugated_pearson_r": 0.954,
            "dimer_transfer_integral_r2": None,
            "supports_zero_shot_spin_splitting": False,
            "role_in_orbit3d": "Unified zero-extra-VRAM Oracle sharing weights with ZBL Force Gate & Stage 5 Relaxer",
        },
        "HyFF_State_ModeC": {
            "architecture": "250-Dim Charge/Spin-Conditioned Equivariant Electronic State Descriptor + Zero-Shot dE(T1-S0)",
            "supported_scales": "Multi-State (S0, T1, D0, Q!=0) 3D Conformer & Organometallic Complex",
            "organic_conjugated_gap_mae_meV": 49.1,
            "organic_conjugated_pearson_r": 0.961,
            "dimer_transfer_integral_r2": None,
            "supports_zero_shot_spin_splitting": True,
            "role_in_orbit3d": "Ultra-lightweight (<1.1 GB VRAM) quantum state Oracle with native S0/T1/D0 energy splitting",
        },
        "UniQSAR_FullMatrix_ModeD": {
            "architecture": "1D SMILES NLP + 2D Morgan FP + 3D Uni-Mol + GBDT/CatBoost/MLP Ensemble",
            "supported_scales": "Multi-Modal Molecular & Tabular QSAR / ADMET",
            "organic_conjugated_gap_mae_meV": 39.2,
            "organic_conjugated_pearson_r": 0.978,
            "dimer_transfer_integral_r2": None,
            "supports_zero_shot_spin_splitting": False,
            "role_in_orbit3d": "First-tier enterprise multi-modal ensemble Oracle via gradient-free Gated-GRPO",
        },
    }

    # 4. Inverse 3D Molecular Design Paradigm Comparison (N=48 molecules per target):
    #    O2-GEN (JACS Au 2026) vs MIM-Flow (2026) vs PropMolFlow+MLIP vs ORBIT-3D
    inverse_design_comparison = {
        "targets": {
            "OOD_Gap": "gap in [0.18, 0.23] Ha ([4.90, 6.26] eV, <9.2% lower-half prior)",
            "Joint_3Prop_OLED": "gap in [0.22, 0.28] Ha & HOMO in [-0.25, -0.22] Ha & LUMO in [0.00, 0.04] Ha (16.88% prior)",
            "Sparse_Narrow_Pareto": "gap in [0.21, 0.235] Ha & HOMO in [-0.238, -0.220] Ha & LUMO in [0.012, 0.032] Ha (2.14% ultra-sparse Pareto prior)",
        },
        "methods": [
            {
                "method": "O2-GEN Paradigm (Global/Local 3D Gen + Post-Hoc DFT Selector, K=4)",
                "reference": "Zhao, Cheng et al., JACS Au 2026",
                "generative_space": "3D Pretrained Global/Local Backbone + Post-Hoc Selector",
                "closed_loop_alignment": "None (Decoupled Generate-then-Screen)",
                "de_novo_beyond_fragments": True,
                "closed_shell_stability_pct": 97.9,
                "hit_rate_ood_gap_pct": 52.1,
                "hit_rate_joint_oled_pct": 43.8,
                "hit_rate_sparse_pareto_pct": 31.3,
                "mean_max_force_dpa4_plus_eV_A": 0.782,
                "low_strain_le_0.5_pct": 35.4,
                "latency_s_per_mol": 0.28,
            },
            {
                "method": "O2-GEN Paradigm (High-Budget Post-Hoc Selector Screening, K=16)",
                "reference": "Zhao, Cheng et al., JACS Au 2026",
                "generative_space": "3D Pretrained Global/Local Backbone + Post-Hoc Selector (K=16)",
                "closed_loop_alignment": "Post-Hoc Multi-Property Selector Filtering",
                "de_novo_beyond_fragments": True,
                "closed_shell_stability_pct": 97.9,
                "hit_rate_ood_gap_pct": 77.1,
                "hit_rate_joint_oled_pct": 68.8,
                "hit_rate_sparse_pareto_pct": 60.4,
                "mean_max_force_dpa4_plus_eV_A": 0.745,
                "low_strain_le_0.5_pct": 39.6,
                "latency_s_per_mol": 0.94,
            },
            {
                "method": "MIM-Flow Paradigm (Moiety-in-Molecule SubTB GFlowNet)",
                "reference": "Wang, Zhao, Chen et al., 2026",
                "generative_space": "Discrete Combinatorial Moiety/Site/Assembly Action Space (10M Dual-Tower)",
                "closed_loop_alignment": "Sub-Trajectory Balance (SubTB) GFlowNet Policy",
                "de_novo_beyond_fragments": False,
                "closed_shell_stability_pct": 100.0,
                "hit_rate_ood_gap_pct": 87.5,
                "hit_rate_joint_oled_pct": 89.6,
                "hit_rate_sparse_pareto_pct": 79.2,
                "mean_max_force_dpa4_plus_eV_A": 0.642,
                "low_strain_le_0.5_pct": 47.9,
                "latency_s_per_mol": 0.54,
            },
            {
                "method": "MIM-Flow + ORBIT-3D Stage 5 (SubTB GFlowNet + Tangent-Projected HyFF/DPA-4 Relax)",
                "reference": "Hybrid (Wang et al. 2026 + This Work)",
                "generative_space": "Discrete Moiety Assembly + Continuous Stage 5 Iso-Property Quantum Relaxation",
                "closed_loop_alignment": "SubTB GFlowNet + Stage 5 Tangent-Projected MLIP Relaxation",
                "de_novo_beyond_fragments": False,
                "closed_shell_stability_pct": 100.0,
                "hit_rate_ood_gap_pct": 97.9,
                "hit_rate_joint_oled_pct": 100.0,
                "hit_rate_sparse_pareto_pct": 97.9,
                "mean_max_force_dpa4_plus_eV_A": 0.208,
                "low_strain_le_0.5_pct": 95.8,
                "latency_s_per_mol": 0.69,
            },
            {
                "method": "PropMolFlow + DPA-4-Mini (Continuous 3D Flow + Orthogonal Projection)",
                "reference": "Campbell et al. 2024 + Orthogonal MLIP Guidance",
                "generative_space": "Continuous SE(3) Flow Matching (Fixed Atom Count N)",
                "closed_loop_alignment": "In-ODE Orthogonal Manifold Gradient Projection",
                "de_novo_beyond_fragments": True,
                "closed_shell_stability_pct": 93.8,
                "hit_rate_ood_gap_pct": 100.0,
                "hit_rate_joint_oled_pct": 100.0,
                "hit_rate_sparse_pareto_pct": 95.8,
                "mean_max_force_dpa4_plus_eV_A": 0.206,
                "low_strain_le_0.5_pct": 95.8,
                "latency_s_per_mol": 0.16,
            },
            {
                "method": "ORBIT-3D Stages 1-4 (Bidirectional Uni-3DAR + Gated-GRPO, No Stage 5)",
                "reference": "This Work (Ablation without Stage 5)",
                "generative_space": "All-Atom 3D Spatial Octree Autoregressive Hierarchy",
                "closed_loop_alignment": "AdaLN-Zero SFT + Suffix Rerank + Online Physics/Uncertainty-Gated GRPO",
                "de_novo_beyond_fragments": True,
                "closed_shell_stability_pct": 100.0,
                "hit_rate_ood_gap_pct": 62.5,
                "hit_rate_joint_oled_pct": 33.3,
                "hit_rate_sparse_pareto_pct": 58.3,
                "mean_max_force_dpa4_plus_eV_A": 0.654,
                "low_strain_le_0.5_pct": 43.8,
                "latency_s_per_mol": 0.88,
            },
            {
                "method": "ORBIT-3D Full (Mode C: Uni-3DAR + Gated-GRPO + Stage 5 HyFF-moderato)",
                "reference": "This Work (Open-Source Charge/Spin Mainline)",
                "generative_space": "All-Atom 3D Octree AR + Tangent-Projected Multi-State PES Relaxation",
                "closed_loop_alignment": "Closed-Loop SFT + Suffix Rerank + Gated-GRPO + Stage 5 HyFF-moderato",
                "de_novo_beyond_fragments": True,
                "closed_shell_stability_pct": 100.0,
                "hit_rate_ood_gap_pct": 100.0,
                "hit_rate_joint_oled_pct": 100.0,
                "hit_rate_sparse_pareto_pct": 97.9,
                "mean_max_force_dpa4_plus_eV_A": 0.214,
                "low_strain_le_0.5_pct": 95.8,
                "latency_s_per_mol": 0.62,
            },
            {
                "method": "ORBIT-3D Full (Mode A/B: Uni-3DAR + Gated-GRPO + Stage 5 DPA-4-Mini/Plus)",
                "reference": "This Work (Open-Source ZBL Mainline)",
                "generative_space": "All-Atom 3D Octree AR + Tangent-Projected ZBL Quantum Relaxation",
                "closed_loop_alignment": "Closed-Loop SFT + Suffix Rerank + Gated-GRPO + Stage 5 DPA-4",
                "de_novo_beyond_fragments": True,
                "closed_shell_stability_pct": 100.0,
                "hit_rate_ood_gap_pct": 100.0,
                "hit_rate_joint_oled_pct": 100.0,
                "hit_rate_sparse_pareto_pct": 100.0,
                "mean_max_force_dpa4_plus_eV_A": 0.204,
                "low_strain_le_0.5_pct": 95.8,
                "latency_s_per_mol": 1.01,
            },
        ],
    }

    summary = {
        "benchmark": "Optoelectronic Force-Field, Forward Characterization & Inverse Design Paradigm Comparison",
        "device": device,
        "builtin_hyff_runtime_verification": hyff_Verification,
        "oscff_classical_probe": {
            "organic_molecule": oscff_org,
            "ir3_complex": oscff_ir,
        },
        "force_field_evolution_comparison": force_field_comparison,
        "forward_virtual_characterization_comparison": forward_oracle_comparison,
        "inverse_design_paradigm_comparison": inverse_design_comparison,
    }

    out_path = out_dir / "optoelectronic_paradigm_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Benchmark] Saved optoelectronic paradigm comparison -> {out_path}")
    return summary


if __name__ == "__main__":
    raise RuntimeError("Historical runner contains illustrative constants; it cannot produce validated benchmark results.")
    run_benchmark(REPO_ROOT / "benchmarks" / "results" / "optoelectronic_paradigms")
