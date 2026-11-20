"""Generate Publication Figures for Paper 2 (`Uni-3DAR-Mol`): `GEOM-Drugs` 10k Benchmark, `N=30..100` Scaling & Speculative Octree Ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "docs" / "experiments"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((EXP_DIR / name).read_text())


def render_geom_drugs_figure(out_paths: List[Path]) -> None:
    d_10k = _load("geom_drugs_uni3dar_mol_10k_eval.json")
    d_sc = _load("geom_drugs_large_molecule_scaling.json")
    arms = d_10k["arms"]
    pub = d_10k["published_reference_baselines"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#B8B2A6",
            "axes.linewidth": 0.95,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(19.4, 6.1), dpi=220)
    fig.patch.set_facecolor("#FAFCFB")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.tick_params(colors="#475569")
        ax.grid(True, linestyle="--", linewidth=0.65, color="#E4ECE9", alpha=0.85)

    # =========================================================================
    # Panel A: GEOM-Drugs 10k Validity/Stability Comparison
    # =========================================================================
    ax0 = axes[0]
    bar_specs = [
        ("MiDi\n(500 NFE)", pub["MiDi-Adaptive (ECML 2023)"]["valid_connected_pct"], pub["MiDi-Adaptive (ECML 2023)"]["mol_stability_pct"], "#94A3B8"),
        ("EQGAT-diff\n(500 NFE)", pub["EQGAT-diff (NeurIPS 2023)"]["valid_connected_pct"], pub["EQGAT-diff (NeurIPS 2023)"]["mol_stability_pct"], "#64748B"),
        ("FlowMol\n(100 NFE)", pub["FlowMol (2024)"]["valid_connected_pct"], pub["FlowMol (2024)"]["mol_stability_pct"], "#475569"),
        ("SemlaFlow\n(100 NFE)", pub["SemlaFlow (ICLR 2025)"]["valid_connected_pct"], pub["SemlaFlow (ICLR 2025)"]["mol_stability_pct"], "#334155"),
        ("Vanilla\nUni-3DAR", arms["Vanilla_Uni3DAR_Pretrained_M1"]["valid_and_connected_rate"] * 100.0, arms["Vanilla_Uni3DAR_Pretrained_M1"]["explicit_mol_stability"] * 100.0, "#F59E0B"),
        ("Uni-3DAR-\nMol\n(1\u21921\u21921)", arms["Uni3DAR_Mol_Base_M1"]["valid_and_connected_rate"] * 100.0, arms["Uni3DAR_Mol_Base_M1"]["explicit_mol_stability"] * 100.0, "#14B8A6"),
        ("Uni-3DAR-\nMol\n(16\u21924\u21921)", arms["Uni3DAR_Mol_SpecTree_M2"]["valid_and_connected_rate"] * 100.0, arms["Uni3DAR_Mol_SpecTree_M2"]["explicit_mol_stability"] * 100.0, "#0F766E"),
        ("Uni-3DAR-\nMol\n(32\u21928\u21921)", arms["Uni3DAR_Mol_SpecTree_M4"]["valid_and_connected_rate"] * 100.0, arms["Uni3DAR_Mol_SpecTree_M4"]["explicit_mol_stability"] * 100.0, "#065F46"),
    ]
    xs = np.arange(len(bar_specs))
    vc_vals = [b[1] for b in bar_specs]
    ms_vals = [b[2] for b in bar_specs]
    b_cols = [b[3] for b in bar_specs]

    b1 = ax0.bar(xs - 0.17, vc_vals, width=0.32, color=b_cols, edgecolor="#1E293B", lw=0.8, label="Valid & Connected (%)")
    b2 = ax0.bar(xs + 0.17, ms_vals, width=0.32, color="#CCFBF1", edgecolor="#0F766E", lw=0.85, hatch="//", label="Explicit Mol Stability (%)")
    for bar, v in zip(b1, vc_vals):
        ax0.text(bar.get_x() + bar.get_width() / 2.0, v + 0.9, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#0F172A")
    ax0.set_xticks(xs)
    ax0.set_xticklabels([b[0] for b in bar_specs], fontsize=6.8, fontweight="bold", color="#1E293B")
    ax0.set_ylim(0.0, 111.0)
    ax0.set_ylabel("Percentage on 10,000 Molecules (%)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax0.set_title("A. GEOM-Drugs 10k Unconditional Benchmark", fontsize=10.5, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax0.legend(loc="upper center", bbox_to_anchor=(0.56, 0.99), fontsize=7.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # =========================================================================
    # Panel B: Asymptotic Wall-Clock Scaling (N in {30, 45, 60, 80, 100})
    # =========================================================================
    ax1 = axes[1]
    b_list = [int(k) for k in d_sc["buckets_atoms"]]
    u_m1_lat = [float(d_sc["buckets"][str(n)]["arms"]["Uni3DAR_Mol_Base_M1"]["gpu_sec_per_1k_molecules"]) for n in b_list]
    u_m2_lat = [float(d_sc["buckets"][str(n)]["arms"]["Uni3DAR_Mol_SpecTree_M2"]["gpu_sec_per_1k_molecules"]) for n in b_list]

    ax1.plot(b_list, u_m1_lat, "-o", color="#0D9488", lw=2.7, ms=7.2, label="Uni-3DAR-Mol (1\u21921\u21921, Sparse Octree)")
    ax1.plot(b_list, u_m2_lat, "--s", color="#0F766E", lw=2.3, ms=6.6, label="Uni-3DAR-Mol (SpecTree 16\u21924\u21921)")

    has_flow = "FlowMol_Dense_ODE100_M1" in d_sc["buckets"][str(b_list[0])]["arms"]
    if has_flow:
        fl_lat = [float(d_sc["buckets"][str(n)]["arms"]["FlowMol_Dense_ODE100_M1"]["gpu_sec_per_1k_molecules"]) for n in b_list]
        ax1.plot(b_list, fl_lat, "-D", color="#E11D48", lw=2.5, ms=6.8, label="Dense O(N^2) CTMC Flow [100 Steps, M=1]")
        sp_100 = fl_lat[-1] / max(u_m1_lat[-1], 1e-6)
        ax1.annotate(
            f"{sp_100:.1f}x faster at N=100\n({u_m1_lat[-1]:.0f}s vs {fl_lat[-1]:.0f}s / 1k)",
            xy=(b_list[-1], u_m1_lat[-1]),
            xytext=(52, max(fl_lat) * 0.62),
            fontsize=7.6,
            fontweight="bold",
            color="#0D9488",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDFA", edgecolor="#0D9488", lw=0.9),
            arrowprops=dict(arrowstyle="->", color="#0D9488", lw=1.1),
        )

    ax1.set_xlabel("Target Molecule Size N (Total Explicit Atoms)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Measured Latency per 1,000 Molecules (A800 GPU-sec)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax1.set_xticks(b_list)
    ax1.set_title("B. Large-Molecule Wall-Clock Scaling (N = 30..100)", fontsize=10.5, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="upper left", fontsize=7.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # =========================================================================
    # Panel C: 3D Geometric Distributional Fidelity (10k JSD & Strained Rings)
    # =========================================================================
    ax2 = axes[2]
    arm_order = [
        ("Vanilla Uni-3DAR", arms["Vanilla_Uni3DAR_Pretrained_M1"], "#F59E0B"),
        ("Uni-3DAR-Mol (1\u21921\u21921)", arms["Uni3DAR_Mol_Base_M1"], "#14B8A6"),
        ("Uni-3DAR-Mol (16\u21924\u21921)", arms["Uni3DAR_Mol_SpecTree_M2"], "#0F766E"),
        ("Uni-3DAR-Mol (32\u21928\u21921)", arms["Uni3DAR_Mol_SpecTree_M4"], "#065F46"),
    ]
    metric_defs = [
        ("Mean Bond\nJSD", lambda r: float(r["bond_length_jsd_mean"])),
        ("Arom C=C\nJSD", lambda r: float(r["bond_length_jsd_per_type"]["C-C:AROMATIC"])),
        ("Bond Angle\nJSD", lambda r: float(r["bond_angle_jsd"])),
        ("Ring Size\nJSD", lambda r: float(r["ring_size_jsd"])),
        ("Strained 3/4\nRing Frac", lambda r: float(r["strained_3_4_ring_fraction"])),
    ]
    xm = np.arange(len(metric_defs))
    w_bar = 0.19
    for idx_a, (a_lbl, a_rec, a_col) in enumerate(arm_order):
        offs = (idx_a - 1.5) * w_bar
        vals_a = [fn(a_rec) for _, fn in metric_defs]
        bars_a = ax2.bar(xm + offs, vals_a, width=w_bar * 0.94, color=a_col, edgecolor="#1E293B", lw=0.7, label=a_lbl)
        if idx_a in (0, 3):
            for br, vv in zip(bars_a, vals_a):
                ax2.text(
                    br.get_x() + br.get_width() / 2.0,
                    vv + 0.0022,
                    f"{vv:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                    fontweight="bold",
                    color="#0F172A",
                    rotation=45,
                )

    ax2.set_xticks(xm)
    ax2.set_xticklabels([m[0] for m in metric_defs], fontsize=7.6, fontweight="bold", color="#1E293B")
    ax2.set_ylim(0.0, 0.155)
    ax2.set_ylabel("Divergence (JSD) / Fraction (lower is better)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax2.set_title("C. 3D Bond, Angle & Ring Fidelity (10k Test Ref)", fontsize=10.5, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax2.legend(loc="upper right", fontsize=7.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    plt.tight_layout(w_pad=1.6)
    for dst in out_paths:
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, bbox_inches="tight")
        if dst.suffix.lower() == ".png":
            fig.savefig(dst.with_suffix(".pdf"), bbox_inches="tight")
        print(f"[Plot SAVED] {dst}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        nargs="*",
        default=[
            ROOT / "docs" / "figures" / "geom_drugs_pareto_and_scaling.png",
            ROOT / "paper" / "uni3dar_mol" / "figures" / "geom_drugs_pareto_and_scaling.png",
        ],
    )
    args = p.parse_args()
    render_geom_drugs_figure(list(args.out))


if __name__ == "__main__":
    main()
