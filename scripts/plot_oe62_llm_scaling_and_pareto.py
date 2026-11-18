"""Generate LLM-style Training/Test-Time Compute Scaling & Campaign-Cost Pareto Figures on OE62.

Reads archived JSON benchmarks from `docs/experiments/` (`PropMolFlow` 3h/6h/12h/24h/52h,
`52h + Loom-AWR`, `Uni-Loom v2.0` Outcome/ProcessCritic, `Size-Anchored Uni-Loom`,
plus `PBE0/def2-TZVP` DFT anchors) and renders publication-ready
3-panel scaling figures in PNG and PDF formats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "docs" / "experiments"

CAMPAIGN_MOLS = 50_000


def _load_json(name: str) -> Optional[Dict[str, Any]]:
    p = EXP_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _extract_curve(
    arms_dict: Dict[str, Any],
    keys: List[str],
    metric: str,
) -> Tuple[List[float], List[float], List[str]]:
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    for k in keys:
        if k not in arms_dict:
            continue
        r = arms_dict[k]
        eff = float(r["effective_output_mol_per_sec"])
        gpu_sec_per_1k = 1000.0 / max(eff, 1e-6)
        if metric == "blind_synth":
            val = float(r["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        elif metric == "rew_synth":
            val = float(r["synthesizable_clean_hit_yield"]) * 100.0
        else:
            raise ValueError(metric)
        xs.append(gpu_sec_per_1k)
        ys.append(val)
        short = (
            k.replace("Outcome_", "")
            .replace("Process_", "")
            .replace("_nosurr", "*")
            .replace("_to_", "→")
        )
        labels.append(short)
    return xs, ys, labels


def _campaign_gpuh(train_gpuh: float, gpu_sec_per_1k_list: List[float]) -> List[float]:
    out: List[float] = []
    for s1k in gpu_sec_per_1k_list:
        infer_gpuh = (s1k * (CAMPAIGN_MOLS / 1000.0)) / 3600.0
        out.append(train_gpuh + infer_gpuh)
    return out


def render_oe62_scaling_figure(out_paths: List[Path]) -> None:
    uniloom = _load_json("oe62_octree_process_critic.json")
    sz_clean = _load_json("oe62_size_anchored_clean_gpu0_eval.json")
    pmf_map = {
        3.0: _load_json("oe62_propmolflow_3gpuh.json"),
        6.0: _load_json("oe62_propmolflow_6gpuh.json"),
        12.0: _load_json("oe62_propmolflow_12gpuh.json"),
        24.0: _load_json("oe62_propmolflow_24gpuh.json"),
        52.0: _load_json("oe62_propmolflow_52gpuh.json"),
    }
    pmf_awr = _load_json("oe62_propmolflow_52gpuh_loom_awr.json")

    assert uniloom is not None and sz_clean is not None and pmf_map[52.0] is not None and pmf_awr is not None
    sz_h = 1.28

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#B8B2A6",
            "axes.linewidth": 0.9,
        }
    )

    bg_fig = "#FAFCFB"
    bg_ax = "#FFFFFF"
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 6.0), dpi=220)
    fig.patch.set_facecolor(bg_fig)
    for ax in axes:
        ax.set_facecolor(bg_ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.tick_params(colors="#475569")
        ax.grid(True, which="major", linestyle="--", linewidth=0.65, color="#E4ECE9", alpha=0.85)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.45, color="#EEF3F1", alpha=0.70)

    # Fresh Natural Saturated Palette
    c_ul_proc = "#0D9488"      # emerald teal: Uni-Loom v2.0 ProcessCritic
    c_ul_out = "#EA580C"       # tangerine: Uni-Loom v2.0 Outcome
    c_sz = "#7C3AED"           # violet: Size-Anchored Uni-Loom ([homo, N])
    c_pmf_52 = "#334155"       # deep slate: PMF 52h
    c_pmf_24 = "#64748B"       # mid slate: PMF 24h
    c_pmf_12 = "#94A3B8"       # light slate: PMF 12h
    c_pmf_awr = "#E11D48"      # saturated coral: PMF 52h + Loom-AWR

    # =========================================================================
    # Panel A: Training / Post-Training Compute Scaling (1.28 -> 52.12 GPU-h)
    # =========================================================================
    ax0 = axes[0]
    hrs = [h for h in (3.0, 6.0, 12.0, 24.0, 52.0) if pmf_map[h] is not None]
    pmf_m8_blind = [
        float(pmf_map[h]["tracks"]["coord_reconciled"]["M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        for h in hrs
    ]
    pmf_m8_rew = [
        float(pmf_map[h]["tracks"]["native_graph"]["M8"]["synthesizable_clean_hit_yield"]) * 100.0
        for h in hrs
    ]
    pmf_m1_blind = [
        float(pmf_map[h]["tracks"]["coord_reconciled"]["M1"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        for h in hrs
    ]

    ax0.plot(hrs, pmf_m8_rew, "-s", color="#7F8C8D", lw=1.9, ms=6.2, alpha=0.88, label="PMF M=8 Surrogate SynthHit")
    ax0.plot(hrs, pmf_m8_blind, "-o", color=c_pmf_52, lw=2.4, ms=7.0, label="PMF M=8 Blind DPA-4 SynthHit")
    ax0.plot(hrs, pmf_m1_blind, "--^", color="#8E8B82", lw=1.6, ms=5.8, label="PMF M=1 Blind DPA-4 (Plateau)")

    # Online Loom-AWR (+0.12 GPU-h, 56 steps on 52h)
    awr_h = 52.12
    awr_rew = float(pmf_awr["tracks"]["native_graph"]["M8"]["synthesizable_clean_hit_yield"]) * 100.0
    awr_blind = float(pmf_awr["tracks"]["native_graph"]["M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    ax0.plot([52.0, awr_h], [pmf_m8_rew[-1], awr_rew], "-", color=c_pmf_awr, lw=2.2)
    ax0.plot([52.0, awr_h], [pmf_m8_blind[-1], awr_blind], "-", color=c_pmf_awr, lw=2.2)
    ax0.scatter([awr_h], [awr_rew], marker="*", s=165, color=c_pmf_awr, zorder=6, label="PMF 52h + Loom-AWR (+0.12h)")
    ax0.scatter([awr_h], [awr_blind], marker="D", s=64, color=c_pmf_awr, zorder=6)

    # Uni-Loom reference stars at ~3.0 GPU-h post-training budget
    ul_16_4_blind = float(uniloom["arms"]["Process_16_to_4"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    ul_16_4_rew = float(uniloom["arms"]["Process_16_to_4"]["synthesizable_clean_hit_yield"]) * 100.0
    ax0.scatter([3.0], [ul_16_4_blind], marker="o", s=95, color=c_ul_proc, edgecolor="white", lw=0.9, zorder=7)
    ax0.scatter([3.0], [ul_16_4_rew], marker="s", s=82, color=c_ul_proc, alpha=0.75, edgecolor="white", lw=0.9, zorder=7)
    ax0.axhline(ul_16_4_blind, color=c_ul_proc, linestyle=":", lw=1.2, alpha=0.65)

    sz_b = float(sz_clean["arms"]["Process_16_to_4"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    ax0.scatter([sz_h], [sz_b], marker="P", s=110, color=c_sz, edgecolor="white", lw=0.7, zorder=7,
                label="Size-Anchored Uni-Loom (33.5 atoms, 1.28h)")

    ax0.annotate("Uni-Loom 16→4 (3.0h)\n66.2% Rew / 45.1% Blind", xy=(3.0, ul_16_4_rew), xytext=(3.6, 71.5),
                 fontsize=8.0, fontweight="bold", color=c_ul_proc,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_ul_proc, alpha=0.92, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_ul_proc, lw=0.95))
    ax0.annotate(f"Size-Anchored 16→4 (1.28h)\n{sz_b:.1f}% Blind | 1,525 Scaf",
                 xy=(sz_h, sz_b), xytext=(1.12, 16.0),
                 fontsize=7.7, fontweight="bold", color=c_sz,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_sz, alpha=0.92, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_sz, lw=0.95))
    ax0.annotate(f"+0.12h Loom-AWR (56 steps)\n{awr_rew:.1f}% Rew / {awr_blind:.1f}% Blind", xy=(awr_h, awr_rew), xytext=(12.0, 73.5),
                 fontsize=8.0, fontweight="bold", color=c_pmf_awr,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.92, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.95))
    ax0.annotate(f"52h Base Blind: {pmf_m8_blind[-1]:.1f}%\n(24h: {pmf_m8_blind[-2]:.1f}%)", xy=(52.0, pmf_m8_blind[-1]),
                 xytext=(15.5, 34.5), fontsize=7.8, color=c_pmf_52, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_pmf_52, alpha=0.92, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_pmf_52, lw=0.85))

    ax0.set_xscale("log")
    ax0.set_xticks([1.3, 3, 6, 12, 24, 52])
    ax0.set_xticklabels(["1.3", "3", "6", "12", "24", "52"])
    ax0.get_xaxis().set_minor_formatter(NullFormatter())
    ax0.set_xlim(1.05, 68.0)
    ax0.set_ylim(3.0, 82.0)
    ax0.set_xlabel("Training / Post-Training Compute (A800 GPU-hours, log)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax0.set_ylabel("Synthesizable Clean Hit Yield (%)", fontsize=10.2, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Training Compute Scaling & Online Loom-AWR", fontsize=10.9, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax0.legend(loc="lower right", fontsize=7.3, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel B: Test-Time Inference Compute Scaling (Blind DPA-4 SynthHit %)
    # =========================================================================
    ax1 = axes[1]
    out_keys = ["Outcome_M1", "Outcome_M2", "Outcome_M4", "Outcome_M8"]
    proc_keys = ["Outcome_M1", "Process_8_to_1_nosurr", "Process_8_to_2", "Process_16_to_2", "Process_16_to_4"]
    pmf_keys = ["M1", "M2", "M4", "M8"]

    ux_out, uy_out, ul_out_lbl = _extract_curve(uniloom["arms"], out_keys, "blind_synth")
    ux_pr, uy_pr, ul_pr_lbl = _extract_curve(uniloom["arms"], proc_keys, "blind_synth")
    sx_pr, sy_pr, s_lbl = _extract_curve(sz_clean["arms"], proc_keys, "blind_synth")
    px_12, py_12, _ = _extract_curve(pmf_map[12.0]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_24, py_24, _ = _extract_curve(pmf_map[24.0]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_52, py_52, pmf_lbl = _extract_curve(pmf_map[52.0]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_awr, py_awr, awr_lbl = _extract_curve(pmf_awr["tracks"]["native_graph"], pmf_keys, "blind_synth")

    ax1.plot(px_12, py_12, "-o", color=c_pmf_12, lw=1.5, ms=5.2, alpha=0.8, label="PMF @ 12h (+Loom-Rec, 30.2 atoms)")
    ax1.plot(px_24, py_24, "-o", color=c_pmf_24, lw=1.7, ms=5.6, alpha=0.88, label="PMF @ 24h (+Loom-Rec, 30.6 atoms)")
    ax1.plot(px_52, py_52, "-o", color=c_pmf_52, lw=2.3, ms=6.6, label="PMF @ 52h (+Loom-Rec, 30.0 atoms)")
    ax1.plot(px_awr, py_awr, "--D", color=c_pmf_awr, lw=2.2, ms=6.0, label="PMF @ 52h + Loom-AWR (30.3 atoms)")
    ax1.plot(ux_out, uy_out, "-o", color=c_ul_out, lw=2.0, ms=6.2, label="Uni-Loom v2.0 Best-of-M (unanchored)")
    ax1.plot(ux_pr, uy_pr, "-o", color=c_ul_proc, lw=2.5, ms=7.0, label="Uni-Loom v2.0 + ProcessCritic (15.6 atoms)")
    ax1.plot(sx_pr, sy_pr, "-P", color=c_sz, lw=2.2, ms=7.0, label="Size-Anchored Uni-Loom (33.5 atoms)")

    for x, y, lb in zip(sx_pr, sy_pr, s_lbl):
        off = (-8, 7) if lb == "16→4" else (4, -10)
        ax1.annotate(lb, (x, y), textcoords="offset points", xytext=off, ha="left", fontsize=7.3, color=c_sz, fontweight="bold")
    for x, y, lb in zip(ux_pr, uy_pr, ul_pr_lbl):
        off = (-14, 7) if lb == "16→2" else ((-3, 7) if lb != "8→1*" else (-12, 6))
        ax1.annotate(lb, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=7.6, color=c_ul_proc, fontweight="bold")
    for x, y, lb in zip(px_awr, py_awr, awr_lbl):
        ax1.annotate(lb, (x, y), textcoords="offset points", xytext=(-4, 6), ha="right", fontsize=7.5, color=c_pmf_awr, fontweight="bold")

    ax1.set_xscale("log")
    ax1.set_xticks([35, 50, 100, 200, 400, 800])
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.get_xaxis().set_minor_formatter(NullFormatter())
    ax1.set_xlim(28, 1080)
    ax1.set_ylim(3.0, 58.0)
    ax1.set_xlabel("Inference Compute per 1,000 Proposals (A800 GPU-sec, log)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=10.2, fontweight="bold", color="#1E293B")
    ax1.set_title("B. Test-Time Compute Scaling (Uncontended Single A800)", fontsize=10.9, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="upper left", fontsize=7.0, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel C: End-to-End 50k-Molecule Campaign Compute (A800 GPU-hours)
    # =========================================================================
    ax2 = axes[2]
    ax2.plot(_campaign_gpuh(12.0, px_12), py_12, "-o", color=c_pmf_12, lw=1.6, ms=5.2, alpha=0.85, label="From-Scratch PMF @ 12h")
    ax2.plot(_campaign_gpuh(24.0, px_24), py_24, "-o", color=c_pmf_24, lw=1.8, ms=5.6, alpha=0.90, label="From-Scratch PMF @ 24h")
    ax2.plot(_campaign_gpuh(52.0, px_52), py_52, "-o", color=c_pmf_52, lw=2.3, ms=6.6, label="From-Scratch PMF @ 52h")
    ax2.plot(_campaign_gpuh(52.12, px_awr), py_awr, "--D", color=c_pmf_awr, lw=2.3, ms=6.0, label="PMF @ 52h + Loom-AWR")
    ax2.plot(_campaign_gpuh(2.5, ux_out), uy_out, "-o", color=c_ul_out, lw=2.1, ms=6.2, label="Uni-Loom v2.0 Best-of-M (~2.5h)")
    ax2.plot(_campaign_gpuh(3.0, ux_pr), uy_pr, "-o", color=c_ul_proc, lw=2.6, ms=7.0, label="Uni-Loom v2.0 + ProcessCritic (~3.0h)")
    ax2.plot(_campaign_gpuh(sz_h, sx_pr), sy_pr, "-P", color=c_sz, lw=2.2, ms=7.0, label="Size-Anchored Uni-Loom (1.28h)")

    # Annotate PBE0 DFT Unbiased Yields on top endpoints (pure GPU-h, zero monetary prices)
    u_cost_top = _campaign_gpuh(3.0, ux_pr)[-1]
    p52_cost_top = _campaign_gpuh(52.0, px_52)[-1]
    pawr_cost_top = _campaign_gpuh(52.12, px_awr)[-1]
    sz_cost_top = _campaign_gpuh(sz_h, sx_pr)[-1]
    ax2.annotate(f"16→4 ({u_cost_top:.1f} GPU-h)\nPBE0 Unbiased: 50.0%\n(Fmax 1.28, MAE 0.347)",
                 xy=(u_cost_top, uy_pr[-1]), xytext=(2.15, 48.2),
                 fontsize=7.4, fontweight="bold", color=c_ul_proc,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_ul_proc, alpha=0.92, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_ul_proc, lw=0.95))
    ax2.annotate(f"Size-Anchored 16→4 ({sz_cost_top:.1f} GPU-h)\n{sz_b:.1f}% Blind | 1,525 Scaf",
                 xy=(sz_cost_top, sy_pr[-1]), xytext=(10.2, 35.0), ha="center",
                 fontsize=7.3, fontweight="bold", color=c_sz,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_sz, alpha=0.92, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_sz, lw=0.95))
    ax2.annotate(f"PMF+Loom-AWR M8 ({pawr_cost_top:.1f} GPU-h)\nPBE0 Unbiased: 50.0%–54.8%\n(MAE 0.270–0.288 eV)",
                 xy=(pawr_cost_top, py_awr[-1]), xytext=(13.5, 49.5),
                 fontsize=7.4, fontweight="bold", color=c_pmf_awr,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.92, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.95))
    ax2.annotate(f"PMF 52h M8 ({p52_cost_top:.1f} GPU-h)\nPBE0 Unbiased: 47.6% raw\n(MAE 0.345 eV)",
                 xy=(p52_cost_top, py_52[-1]), xytext=(18.5, 34.0),
                 fontsize=7.3, fontweight="bold", color=c_pmf_52,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_pmf_52, alpha=0.92, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_pmf_52, lw=0.95))

    ax2.set_xscale("log")
    ax2.set_xticks([2, 3, 5, 10, 20, 40, 70])
    ax2.set_xticklabels(["2", "3", "5", "10", "20", "40", "70"])
    ax2.get_xaxis().set_minor_formatter(NullFormatter())
    ax2.set_xlim(1.8, 88.0)
    ax2.set_ylim(3.0, 59.0)
    ax2.set_xlabel("Total 50k-Molecule Campaign Compute (Train/Adapt + Infer, A800 GPU-h, log)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=10.2, fontweight="bold", color="#1E293B")
    ax2.set_title("C. End-to-End 50k-Campaign Compute Frontier & PBE0 Anchor", fontsize=10.9, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax2.legend(loc="lower right", fontsize=6.9, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    fig.text(
        0.5,
        0.01,
        "OE62 HOMO (±0.25 eV, SA_Score ≤ 3.5, Clean Closed-Shell Charge, N = 1,792 proposals/arm, Uncontended Single NVIDIA A800-80GB). "
        "*8→1: zero external surrogate calls. Panel C sums task training/adaptation + 50k-proposal inference compute in A800 GPU-hours (Uni-3DAR upstream pretrain excluded).",
        ha="center",
        fontsize=8.2,
        color="#475569",
    )
    plt.tight_layout(rect=[0.0, 0.035, 1.0, 1.0])

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
        ROOT / "docs" / "figures" / "oe62_llm_style_accuracy_vs_cost.png",
        ROOT / "paper" / "figures" / "fig8_oe62_cross_paradigm_scaling.png",
        ROOT / "paper" / "figures" / "fig8_oe62_cross_paradigm_scaling.pdf",
        ROOT / "paper" / "ncs" / "figures" / "ed_fig3_oe62_cross_paradigm_scaling.png",
        ROOT / "paper" / "ncs" / "figures" / "ed_fig3_oe62_cross_paradigm_scaling.pdf",
        *args.extra_out,
    ]
    render_oe62_scaling_figure(targets)


if __name__ == "__main__":
    main()
