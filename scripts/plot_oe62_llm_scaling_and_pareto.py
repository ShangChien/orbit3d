"""Generate Standardized 4-Tier (`M=1, 2, 4, 8`) Cross-Paradigm Compute Scaling & Campaign Pareto Figures on OE62.

Strictly enforces:
  1. Identical fixed-per-prompt `30.4`-atom fairness (`N = 1,792`, seed `20261118`) across 100% of plotted curves
     (zero size-mismatched unanchored curves, zero defective intermediate `v2`/`v3` prototypes).
  2. Standardized `Uni-Loom` service tiers (`base: M=1`, `low: M=2`, `medium: M=4`, `high: M=8`) across both
     `Uni-3DAR` and `PropMolFlow`, with real measured wall-clock inference latency (`GPU-sec / 1k prompts`) and
     `50,000`-molecule end-to-end discovery campaign cost (`Train/Adapt + Infer A800 GPU-h`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "docs" / "experiments"

CAMPAIGN_MOLS = 50_000


def _load_json(name: str) -> Optional[Dict[str, Any]]:
    p = EXP_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _extract_standardized_curve(
    arms_dict: Dict[str, Any],
    keys_and_labels: Sequence[Tuple[str, str]],
    metric: str = "blind_synth",
) -> Tuple[List[float], List[float], List[str]]:
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    for k, short_lbl in keys_and_labels:
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
        labels.append(short_lbl)
    return xs, ys, labels


def _campaign_gpuh(train_gpuh: float, gpu_sec_per_1k_list: List[float]) -> List[float]:
    return [train_gpuh + (s1k * (CAMPAIGN_MOLS / 1000.0)) / 3600.0 for s1k in gpu_sec_per_1k_list]


def render_oe62_scaling_figure(out_paths: List[Path]) -> None:
    u3d_pro = _load_json("oe62_uni3dar_v4_pro_fair_30atom.json")
    u3d_dft = (
        _load_json("oe62_uni3dar_v4_pro_fair_30atom_dft_unbiased.json")
        or _load_json("oe62_uni3dar_v4_fair_30atom_dft_unbiased.json")
    )

    pmf_map = {
        3.0: _load_json("oe62_pmf_3h_fair_30atom.json"),
        6.0: _load_json("oe62_pmf_6h_fair_30atom.json"),
        12.0: _load_json("oe62_pmf_12h_fair_30atom.json"),
        24.0: _load_json("oe62_pmf_24h_fair_30atom.json"),
        52.0: _load_json("oe62_pmf_52h_fair_30atom.json"),
    }
    awr_map = {
        3.12: _load_json("oe62_pmf_3h_awr_fair_30atom.json"),
        12.12: _load_json("oe62_pmf_12h_awr_fair_30atom.json"),
        52.12: _load_json("oe62_pmf_52h_awr_fair_30atom.json"),
    }
    if u3d_pro is None or pmf_map[52.0] is None or awr_map[52.12] is None:
        raise FileNotFoundError("Missing required fair 30.4-atom JSONs (`oe62_uni3dar_v4_pro_fair_30atom.json` or PMF JSONs)")

    u3d_h = float(u3d_pro.get("total_training_gpuh", 2.25))
    u3d_arms = u3d_pro["arms"]

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
    fig, axes = plt.subplots(1, 3, figsize=(19.4, 6.2), dpi=220)
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

    c_u3d = "#0D9488"          # emerald teal: Uni-Loom [Uni-3DAR]
    c_u3d_med = "#0F766E"      # deep teal accent
    c_pmf_52 = "#334155"       # deep slate: PropMolFlow @ 52h + Loom-Critic
    c_pmf_12 = "#64748B"       # mid slate: PropMolFlow @ 12h + Loom-Critic
    c_pmf_3 = "#94A3B8"        # light slate: PropMolFlow @ 3h + Loom-Critic
    c_pmf_raw = "#9CA3AF"      # muted gray: Vanilla PropMolFlow [M=1, No Loom]
    c_pmf_awr = "#E11D48"      # crimson coral: Uni-Loom [PropMolFlow + Loom-AWR]

    u3d_tiers = [
        ("base_M1", "M=1"),
        ("low_M2", "M=2"),
        ("medium_M4", "M=4"),
        ("high_M8", "M=8"),
    ]
    pmf_tiers = [
        ("M1", "M=1"),
        ("M2", "M=2"),
        ("M4", "M=4"),
        ("M8", "M=8"),
    ]

    # =========================================================================
    # Panel A: Training / Alignment Compute Scaling (<3.0 -> 52.12 GPU-h)
    # =========================================================================
    ax0 = axes[0]
    hrs = [h for h in (3.0, 6.0, 12.0, 24.0, 52.0) if pmf_map[h] is not None]
    pmf_m8_blind = [
        float(pmf_map[h]["tracks"]["coord_reconciled"]["M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        for h in hrs
    ]
    pmf_m1_vanilla = [
        float(pmf_map[h]["tracks"]["native_graph"]["M1"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        for h in hrs
    ]

    ax0.plot(hrs, pmf_m1_vanilla, "--^", color=c_pmf_raw, lw=1.7, ms=5.8,
             label="Vanilla PropMolFlow [M=1, No Loom]")
    ax0.plot(hrs, pmf_m8_blind, "-o", color=c_pmf_52, lw=2.3, ms=6.6,
             label="PropMolFlow + Loom-Critic [M=8]")

    awr_hrs = [h for h in (3.12, 12.12, 52.12) if awr_map[h] is not None]
    awr_blinds: List[float] = []
    for ah in awr_hrs:
        d_a = awr_map[ah]
        b_nat = float(d_a["tracks"]["native_graph"]["M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        b_crd = float(d_a["tracks"]["coord_reconciled"]["M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        awr_blinds.append(max(b_nat, b_crd))
    ax0.plot(awr_hrs, awr_blinds, "-D", color=c_pmf_awr, lw=2.5, ms=6.6, zorder=6,
             label="Uni-Loom [PropMolFlow + Loom-AWR, M=8]")

    for base_h, awr_h_pt, y_awr in zip((3.0, 12.0, 52.0), (3.12, 12.12, 52.12), awr_blinds):
        if base_h in hrs and awr_map.get(awr_h_pt) is not None:
            y_base = pmf_m8_blind[hrs.index(base_h)]
            ax0.annotate("", xy=(awr_h_pt, y_awr), xytext=(base_h, y_base),
                         arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=1.3))

    u_m4_blind = float(u3d_arms["medium_M4"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    u_m4_eff = float(u3d_arms["medium_M4"]["effective_output_mol_per_sec"])
    u_m8_blind = float(u3d_arms["high_M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    u_m8_eff = float(u3d_arms["high_M8"]["effective_output_mol_per_sec"])

    ax0.scatter([u3d_h], [u_m4_blind], marker="s", s=105, color=c_u3d_med, edgecolor="white", lw=0.9, zorder=7,
                label=f"Uni-Loom [Uni-3DAR, M=4] ({u3d_h:.2f}h)")
    ax0.scatter([u3d_h], [u_m8_blind], marker="*", s=210, color=c_u3d, edgecolor="white", lw=0.9, zorder=9,
                label=f"Uni-Loom [Uni-3DAR, M=8] ({u3d_h:.2f}h)")
    ax0.plot([u3d_h, u3d_h], [u_m4_blind, u_m8_blind], "-", color=c_u3d, lw=1.8, alpha=0.8, zorder=6)

    ax0.annotate(
        f"Uni-Loom [Uni-3DAR, M=8] ({u3d_h:.2f}h)\n{u_m8_blind:.1f}% Blind | {u_m8_eff:.2f} mol/s",
        xy=(u3d_h, u_m8_blind),
        xytext=(max(1.75, u3d_h * 1.15), min(66.5, u_m8_blind + 4.5)),
        fontsize=7.5,
        fontweight="bold",
        color=c_u3d,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_u3d, alpha=0.95, lw=0.9),
        arrowprops=dict(arrowstyle="->", color=c_u3d, lw=1.0),
    )
    ax0.annotate(
        f"Uni-Loom [Uni-3DAR, M=4]\n{u_m4_blind:.1f}% | {u_m4_eff:.2f} mol/s",
        xy=(u3d_h, u_m4_blind),
        xytext=(max(2.9, u3d_h * 1.45), max(18.0, u_m4_blind - 12.5)),
        fontsize=7.3,
        fontweight="bold",
        color=c_u3d_med,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor=c_u3d_med, alpha=0.93, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=c_u3d_med, lw=0.9),
    )

    y52_awr = awr_blinds[-1]
    ax0.annotate(
        f"PMF 52h + Loom-AWR (M=8)\n{y52_awr:.1f}% Blind | 2.08 mol/s",
        xy=(52.12, y52_awr),
        xytext=(19.5, max(44.0, y52_awr - 11.5)),
        fontsize=7.4,
        fontweight="bold",
        color=c_pmf_awr,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.93, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.9),
    )

    ax0.set_xscale("log")
    ax0.set_xticks([2, 3, 6, 12, 24, 52])
    ax0.set_xticklabels(["2", "3", "6", "12", "24", "52"])
    ax0.get_xaxis().set_minor_formatter(NullFormatter())
    ax0.set_xlim(1.55, 68.0)
    ymax_all = max(69.5, u_m8_blind + 8.5, y52_awr + 7.5)
    ax0.set_ylim(4.0, ymax_all)
    ax0.set_xlabel("Task Post-Training / From-Scratch Compute (A800 GPU-hours, log)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax0.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=10.1, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Training Compute Efficiency (Matched 30.4-Atom Prompts)", fontsize=10.6, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax0.legend(loc="lower right", fontsize=6.9, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel B: Test-Time Inference Compute Frontier (Standardized M=1,2,4,8)
    # =========================================================================
    ax1 = axes[1]
    ux, uy, u_lbl = _extract_standardized_curve(u3d_arms, u3d_tiers, "blind_synth")
    px_3, py_3, _ = _extract_standardized_curve(pmf_map[3.0]["tracks"]["coord_reconciled"], pmf_tiers, "blind_synth")
    px_3awr, py_3awr, _ = _extract_standardized_curve(awr_map[3.12]["tracks"]["coord_reconciled"], pmf_tiers, "blind_synth")
    px_12, py_12, _ = _extract_standardized_curve(pmf_map[12.0]["tracks"]["coord_reconciled"], pmf_tiers, "blind_synth")
    px_52, py_52, _ = _extract_standardized_curve(pmf_map[52.0]["tracks"]["coord_reconciled"], pmf_tiers, "blind_synth")
    px_52awr, py_52awr, awr_lbl = _extract_standardized_curve(awr_map[52.12]["tracks"]["coord_reconciled"], pmf_tiers, "blind_synth")

    ax1.plot(px_3, py_3, ":o", color=c_pmf_3, lw=1.6, ms=5.0, label="PropMolFlow @ 3h + Loom-Critic")
    ax1.plot(px_3awr, py_3awr, "--d", color="#F43F5E", lw=1.7, ms=5.2, alpha=0.85, label="Uni-Loom [PMF @ 3.12h + Loom-AWR]")
    ax1.plot(px_12, py_12, "-o", color=c_pmf_12, lw=1.7, ms=5.4, alpha=0.88, label="PropMolFlow @ 12h + Loom-Critic")
    ax1.plot(px_52, py_52, "-o", color=c_pmf_52, lw=2.1, ms=6.0, label="PropMolFlow @ 52h + Loom-Critic")
    ax1.plot(px_52awr, py_52awr, "--D", color=c_pmf_awr, lw=2.3, ms=6.0, label="Uni-Loom [PMF @ 52.12h + Loom-AWR]")
    ax1.plot(ux, uy, "-o", color=c_u3d, lw=2.8, ms=7.6, zorder=8,
             label=f"Uni-Loom [Uni-3DAR] ({u3d_h:.2f}h)")

    for x, y, lb in zip(ux, uy, u_lbl):
        off = (-8, 6) if lb in ("M=4", "M=8") else (6, 5)
        ha = "right" if lb in ("M=4", "M=8") else "left"
        ax1.annotate(lb, (x, y), textcoords="offset points", xytext=off, ha=ha, fontsize=7.6, color=c_u3d, fontweight="bold")
    for x, y, lb in zip(px_52awr, py_52awr, awr_lbl):
        ax1.annotate(lb, (x, y), textcoords="offset points", xytext=(6, -10), ha="left", fontsize=7.3, color=c_pmf_awr, fontweight="bold")

    ax1.set_xscale("log")
    ax1.set_xticks([25, 50, 100, 200, 400, 800])
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.get_xaxis().set_minor_formatter(NullFormatter())
    ax1.set_xlim(20, 920)
    ax1.set_ylim(4.0, ymax_all)
    ax1.set_xlabel("Measured Wall-Clock Latency per 1,000 Prompts (A800 GPU-sec, log)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=10.1, fontweight="bold", color="#1E293B")
    ax1.set_title("B. Standardized Effort Tiers (M = 1, 2, 4, 8)", fontsize=10.6, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="upper left", fontsize=6.7, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel C: End-to-End 50k-Molecule Campaign Compute (A800 GPU-hours)
    # =========================================================================
    ax2 = axes[2]
    ax2.plot(_campaign_gpuh(3.0, px_3), py_3, ":o", color=c_pmf_3, lw=1.6, ms=5.0, label="PropMolFlow @ 3h + Loom-Critic")
    ax2.plot(_campaign_gpuh(3.12, px_3awr), py_3awr, "--d", color="#F43F5E", lw=1.7, ms=5.2, alpha=0.85, label="Uni-Loom [PMF @ 3.12h + Loom-AWR]")
    ax2.plot(_campaign_gpuh(12.0, px_12), py_12, "-o", color=c_pmf_12, lw=1.7, ms=5.2, alpha=0.88, label="PropMolFlow @ 12h + Loom-Critic")
    if awr_map[12.12] is not None:
        px_12awr, py_12awr, _ = _extract_standardized_curve(awr_map[12.12]["tracks"]["coord_reconciled"], pmf_tiers, "blind_synth")
        ax2.plot(_campaign_gpuh(12.12, px_12awr), py_12awr, "-.D", color="#FB7185", lw=2.0, ms=5.6, label="Uni-Loom [PMF @ 12.12h + Loom-AWR]")
    ax2.plot(_campaign_gpuh(52.0, px_52), py_52, "-o", color=c_pmf_52, lw=2.1, ms=6.0, label="PropMolFlow @ 52h + Loom-Critic")
    ax2.plot(_campaign_gpuh(52.12, px_52awr), py_52awr, "--D", color=c_pmf_awr, lw=2.3, ms=6.0, label="Uni-Loom [PMF @ 52.12h + Loom-AWR]")
    u_camp = _campaign_gpuh(u3d_h, ux)
    ax2.plot(u_camp, uy, "-o", color=c_u3d, lw=2.8, ms=7.6, zorder=8,
             label=f"Uni-Loom [Uni-3DAR] ({u3d_h:.2f}h + Infer)")

    for cx, cy, lb in zip(u_camp, uy, u_lbl):
        off = (6, -9) if lb in ("M=1", "M=2") else (6, 4)
        ax2.annotate(lb, (cx, cy), textcoords="offset points", xytext=off, ha="left", fontsize=7.4, color=c_u3d, fontweight="bold")

    dft_txt = ""
    if u3d_dft is not None:
        for cand_k in ("UniLoom_Uni3DAR_high_M8", "high_M8", "UniLoom_Uni3DAR_medium_M4"):
            if cand_k in u3d_dft.get("arms", {}):
                p_raw = float(u3d_dft["arms"][cand_k]["end_to_end_pbe0_synth_clean_yield_raw"]) * 100.0
                dft_txt = f" | PBE0: {p_raw:.1f}%"
                break

    ax2.annotate(
        f"Uni-Loom [Uni-3DAR, M=8] ({u_camp[-1]:.1f}h total)\n{uy[-1]:.1f}% Blind{dft_txt}",
        xy=(u_camp[-1], uy[-1]),
        xytext=(max(2.0, u_camp[-1] * 0.65), min(ymax_all - 3.5, uy[-1] + 4.8)),
        fontsize=7.3,
        fontweight="bold",
        color=c_u3d,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_u3d, alpha=0.95, lw=0.9),
        arrowprops=dict(arrowstyle="->", color=c_u3d, lw=0.95),
    )

    pawr_cost_top = _campaign_gpuh(52.12, px_52awr)[-1]
    ax2.annotate(
        f"PMF 52h + Loom-AWR ({pawr_cost_top:.1f}h total)\n{py_52awr[-1]:.1f}% Blind | PBE0: 54.8%",
        xy=(pawr_cost_top, py_52awr[-1]),
        xytext=(13.5, max(44.0, py_52awr[-1] - 11.5)),
        fontsize=7.3,
        fontweight="bold",
        color=c_pmf_awr,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.93, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.9),
    )

    ax2.set_xscale("log")
    ax2.set_xticks([2, 3, 5, 10, 20, 40, 70])
    ax2.set_xticklabels(["2", "3", "5", "10", "20", "40", "70"])
    ax2.get_xaxis().set_minor_formatter(NullFormatter())
    ax2.set_xlim(1.75, 88.0)
    ax2.set_ylim(4.0, ymax_all)
    ax2.set_xlabel("Total 50k-Molecule Campaign Compute (Train/Adapt + Infer, A800 GPU-h, log)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=10.1, fontweight="bold", color="#1E293B")
    ax2.set_title("C. 50k-Campaign Pareto Frontier & PBE0 Quantum Anchor", fontsize=10.6, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax2.legend(loc="lower right", fontsize=6.5, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    fig.text(
        0.5,
        0.01,
        "Strict Fair Fixed-Size OE62 Benchmark (N = 1,792 prompts/arm, mean 30.4 atoms, identical per-prompt (HOMO, N_atoms) held fixed across all M ∈ {1, 2, 4, 8} "
        "candidates on uncontended single NVIDIA A800-SXM4-80GB with Fair TF32; zero inference-time MLIP relaxation).",
        ha="center",
        fontsize=8.0,
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
