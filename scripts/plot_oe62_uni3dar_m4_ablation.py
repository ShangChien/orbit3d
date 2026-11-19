"""Plot Internal Architecture Ablation (`M=4`: `4->4->4` vs `32->8->4` x `PostTreeGraphRefiner`) & SDL Multi-Objective RSI.

Reads:
  - `docs/experiments/oe62_uni3dar_v4_pro_m4_ablation.json`
  - `docs/experiments/oe62_sdl_multi_objective_epistemic_rsi.json` (optional Panel C)
and renders a clean 3-panel architecture & epistemic breakdown without cluttering the cross-model main figure.
"""

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
AB_JSON = ROOT / "docs" / "experiments" / "oe62_uni3dar_v4_pro_m4_ablation.json"
SDL_JSON = ROOT / "docs" / "experiments" / "oe62_sdl_multi_objective_epistemic_rsi.json"


def render_ablation_and_sdl_figure(out_paths: List[Path]) -> None:
    ab = json.loads(AB_JSON.read_text())["arms"]
    sdl = json.loads(SDL_JSON.read_text()) if SDL_JSON.exists() else None

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#B8B2A6",
            "axes.linewidth": 0.95,
        }
    )
    bg_fig = "#FAFCFB"
    bg_ax = "#FFFFFF"
    n_cols = 3 if sdl is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6.3 * n_cols, 5.4), dpi=220)
    fig.patch.set_facecolor(bg_fig)
    for ax in axes:
        ax.set_facecolor(bg_ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.tick_params(colors="#475569")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.65, color="#E4ECE9", alpha=0.85)

    order = [
        ("Naive_Outer_4to4to4_NoRefiner", "Outer M=4\n(No Tree, No Refiner)"),
        ("Naive_Outer_4to4to4_WithRefiner", "Outer M=4\n+ Seam Refiner"),
        ("SpeculativeTree_32to8to4_NoRefiner", "Speculative Tree\n(32→8→4, No Refiner)"),
        ("Full_UniLoom_32to8to4_WithRefiner", "Full Uni-Loom M=4\n(32→8→4 + Refiner)"),
    ]
    colors = ["#94A3B8", "#38BDF8", "#34D399", "#0D9488"]

    x = np.arange(len(order))
    short_names = [lbl for _, lbl in order]
    hits = [float(ab[k]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0 for k, _ in order]
    fmaxs = [float(ab[k]["zero_step_f_max_median_eV_A"]) for k, _ in order]
    effs = [float(ab[k]["effective_output_mol_per_sec"]) for k, _ in order]

    # Panel A: Blind Synth-Clean Hit Yield (%) at Fixed Medium Effort (M=4)
    ax0 = axes[0]
    bars0 = ax0.bar(x, hits, width=0.56, color=colors, edgecolor="#1E293B", linewidth=0.8)
    for b, hv, ev in zip(bars0, hits, effs):
        ax0.text(
            b.get_x() + b.get_width() / 2.0,
            hv + 1.1,
            f"{hv:.1f}%\n({ev:.1f} mol/s)",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#0F172A",
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels(short_names, fontsize=8.0, fontweight="bold")
    ax0.set_ylim(0, max(hits) * 1.24)
    ax0.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Fixed Medium Tier (M=4) Hit-Yield Decomposition", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)

    # Panel B: Zero-Step Atomic Force Strain (Median F_max, eV/A)
    ax1 = axes[1]
    bars1 = ax1.bar(x, fmaxs, width=0.56, color=colors, edgecolor="#1E293B", linewidth=0.8)
    for b, fv in zip(bars1, fmaxs):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            fv + 0.08,
            f"{fv:.2f} eV/Å",
            ha="center",
            va="bottom",
            fontsize=8.1,
            fontweight="bold",
            color="#0F172A",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=8.0, fontweight="bold")
    ax1.set_ylim(0, max(fmaxs) * 1.22)
    ax1.set_ylabel("Zero-Step Median Atomic Force F_max (eV/Å, lower is better)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax1.set_title("B. Sub-Millisecond SE(3) Seam Healing vs Strain", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)

    # Panel C: Multi-Objective (HOMO, LUMO) Partial-Assay Blindspot Calibration
    if sdl is not None:
        ax2 = axes[2]
        st = sdl["partial_assay_calibration_stats"]
        pre_mae = st["pre_calibration_blindspot_mae"]
        post_mae = st["post_calibration_blindspot_mae"]
        ch_x = np.arange(2)
        w_b = 0.32
        b_pre = ax2.bar(ch_x - w_b / 2, pre_mae, width=w_b, color="#F43F5E", edgecolor="#1E293B", lw=0.8, label="Pre-Assay OOD Blindspot")
        b_post = ax2.bar(ch_x + w_b / 2, post_mae, width=w_b, color="#0D9488", edgecolor="#1E293B", lw=0.8, label="Post Partial-Assay Calibration")
        for b in list(b_pre) + list(b_post):
            v = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
        ax2.set_xticks(ch_x)
        ax2.set_xticklabels(["HOMO Blindspot\n(Partial Readouts)", "LUMO Blindspot\n(Partial Readouts)"], fontsize=8.4, fontweight="bold")
        ax2.set_ylim(0, max(max(pre_mae), max(post_mae)) * 1.25)
        ax2.set_ylabel("Held-Out DFT Blindspot MAE (eV)", fontsize=9.6, fontweight="bold", color="#1E293B")
        ax2.set_title("C. Self-Driving Lab 2D Partial-Assay Calibration", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
        ax2.legend(loc="upper right", fontsize=7.6, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    plt.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, facecolor=bg_fig)
        print(f"Wrote {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extra-out", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    targets = [
        ROOT / "docs" / "figures" / "oe62_uni3dar_m4_ablation_and_sdl.png",
        ROOT / "paper" / "figures" / "fig10_uni3dar_m4_ablation_and_sdl.png",
        ROOT / "paper" / "figures" / "fig10_uni3dar_m4_ablation_and_sdl.pdf",
        ROOT / "paper" / "ncs" / "figures" / "ed_fig5_uni3dar_m4_ablation_and_sdl.png",
        ROOT / "paper" / "ncs" / "figures" / "ed_fig5_uni3dar_m4_ablation_and_sdl.pdf",
        *args.extra_out,
    ]
    render_ablation_and_sdl_figure(targets)


if __name__ == "__main__":
    main()
