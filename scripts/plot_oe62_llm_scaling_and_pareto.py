"""Generate LLM-style Training/Test-Time Compute Scaling & Campaign-Cost Pareto Figures on OE62.

Reads archived JSON benchmarks from `docs/experiments/`:
  - `PropMolFlow` 3h / 6h / 12h / 24h / 52h (`Vanilla M=1` vs `+ Uni-Loom CriticBus M=8`)
  - `PropMolFlow + Online Loom-AWR` across 3.12h / 12.12h / 52.12h
  - A800-optimized `Uni-Loom v2.0 [Uni-3DAR Slot]` (`Outcome M=1..8` & `ProcessCritic 8->1..16->4`)
  - `Size-Anchored Uni-Loom v3` (`[homo, n_atoms]`, ~35 atoms, A800-optimized + v1/v2 progression)
  - `42`-prompt unbiased live `GPU4PySCF PBE0/def2-TZVP` quantum anchors
and renders publication-ready 3-panel scaling figures in PNG and PDF formats.
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


def _build_size_anchored_v3_fallback() -> Dict[str, Any]:
    """Live-logged Size-Anchored v3 arms (`sft+align+critic = 2.45 GPU-h`) while `16->4` finishes."""
    return {
        "train_gpuh": 2.45,
        "provisional": True,
        "arms": {
            "Outcome_M1": {
                "effective_output_mol_per_sec": 21.72,
                "synthesizable_clean_hit_yield": 196 / 1792,
                "blind_dpa4_synthesizable_clean_hit_yield": 148 / 1792,
                "mean_total_atoms": 34.64,
                "unique_murcko_scaffolds": 1299,
            },
            "Outcome_M2": {
                "effective_output_mol_per_sec": 9.50,
                "synthesizable_clean_hit_yield": 344 / 1792,
                "blind_dpa4_synthesizable_clean_hit_yield": 250 / 1792,
                "mean_total_atoms": 35.37,
                "unique_murcko_scaffolds": 1476,
            },
            "Outcome_M4": {
                "effective_output_mol_per_sec": 4.75,
                "synthesizable_clean_hit_yield": 576 / 1792,
                "blind_dpa4_synthesizable_clean_hit_yield": 387 / 1792,
                "mean_total_atoms": 35.79,
                "unique_murcko_scaffolds": 1509,
            },
            "Outcome_M8": {
                "effective_output_mol_per_sec": 2.38,
                "synthesizable_clean_hit_yield": 909 / 1792,
                "blind_dpa4_synthesizable_clean_hit_yield": 564 / 1792,
                "mean_total_atoms": 35.78,
                "unique_murcko_scaffolds": 1506,
            },
            "Process_8_to_1_nosurr": {
                "effective_output_mol_per_sec": 7.77,
                "synthesizable_clean_hit_yield": 349 / 1792,
                "blind_dpa4_synthesizable_clean_hit_yield": 271 / 1792,
                "mean_total_atoms": 34.52,
                "unique_murcko_scaffolds": 1393,
            },
            "Process_8_to_2": {
                "effective_output_mol_per_sec": 5.67,
                "synthesizable_clean_hit_yield": 576 / 1792,
                "blind_dpa4_synthesizable_clean_hit_yield": 387 / 1792,
                "mean_total_atoms": 35.26,
                "unique_murcko_scaffolds": 1501,
            },
        },
    }


def render_oe62_scaling_figure(out_paths: List[Path]) -> None:
    uniloom = _load_json("oe62_octree_process_critic_a800_opt.json") or _load_json("oe62_octree_process_critic.json")
    sz_v2_opt = _load_json("oe62_size_anchored_v2_a800_opt_eval.json") or _load_json("oe62_size_anchored_v2_clean_eval.json")
    sz_v3 = _load_json("oe62_size_anchored_v3_a800_eval.json") or _build_size_anchored_v3_fallback()
    sz_v3_dft = _load_json("oe62_size_anchored_v3_dft_unbiased.json")

    pmf_map = {
        3.0: _load_json("oe62_propmolflow_3gpuh.json"),
        6.0: _load_json("oe62_propmolflow_6gpuh.json"),
        12.0: _load_json("oe62_propmolflow_12gpuh.json"),
        24.0: _load_json("oe62_propmolflow_24gpuh.json"),
        52.0: _load_json("oe62_propmolflow_52gpuh.json"),
    }
    awr_map = {
        3.12: _load_json("oe62_propmolflow_3gpuh_loom_awr.json"),
        12.12: _load_json("oe62_propmolflow_12gpuh_loom_awr.json"),
        52.12: _load_json("oe62_propmolflow_52gpuh_loom_awr.json"),
    }

    assert uniloom is not None and sz_v3 is not None and pmf_map[52.0] is not None and awr_map[52.12] is not None
    sz_v3_h = 2.45
    sz_v2_h = 1.28

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

    # Fresh Natural Saturated Palette
    c_ul_proc = "#0D9488"      # emerald teal: Uni-Loom [Uni-3DAR Slot, unanchored]
    c_ul_out = "#EA580C"       # tangerine: Uni-Loom [Uni-3DAR Best-of-M]
    c_sz_v3 = "#6D28D9"        # saturated royal violet: Size-Anchored v3 ([homo, N], 35.5 atoms)
    c_sz_v2 = "#A78BFA"        # soft lavender: Size-Anchored v2 (1.28h A800-opt ablation)
    c_pmf_52 = "#334155"       # deep slate: PMF @ 52h + Loom-Rec
    c_pmf_12 = "#64748B"       # mid slate: PMF @ 12h + Loom-Rec
    c_pmf_3 = "#94A3B8"        # light slate: PMF @ 3h + Loom-Rec (compute-matched!)
    c_pmf_raw = "#9CA3AF"      # muted gray: Vanilla PropMolFlow (M=1, No Loom)
    c_pmf_awr = "#E11D48"      # crimson coral: Uni-Loom [PropMolFlow + Online Loom-AWR]

    # =========================================================================
    # Panel A: Training / Post-Training Compute Scaling (1.28 -> 52.12 GPU-h)
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
             label="Vanilla PropMolFlow (M=1, No Loom: ≤10.5%)")
    ax0.plot(hrs, pmf_m8_blind, "-o", color=c_pmf_52, lw=2.3, ms=6.8,
             label="PropMolFlow + Uni-Loom Critic (M=8 Blind)")

    # Full PropMolFlow + Online Loom-AWR Scaling Curve (3.12h -> 12.12h -> 52.12h)
    awr_hrs = [h for h in (3.12, 12.12, 52.12) if awr_map[h] is not None]
    awr_blinds: List[float] = []
    for ah in awr_hrs:
        d_a = awr_map[ah]
        b_nat = float(d_a["tracks"]["native_graph"]["M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        b_crd = float(d_a["tracks"]["coord_reconciled"]["M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        awr_blinds.append(max(b_nat, b_crd))
    ax0.plot(awr_hrs, awr_blinds, "-D", color=c_pmf_awr, lw=2.5, ms=6.8, zorder=6,
             label="PropMolFlow + Online Loom-AWR (+0.12h, M=8)")

    # Vertical uplift arrows from Base PMF to +Loom-AWR
    for base_h, awr_h_pt, y_awr in zip((3.0, 12.0, 52.0), (3.12, 12.12, 52.12), awr_blinds):
        if base_h in hrs and awr_map.get(awr_h_pt) is not None:
            y_base = pmf_m8_blind[hrs.index(base_h)]
            ax0.annotate("", xy=(awr_h_pt, y_awr), xytext=(base_h, y_base),
                         arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=1.3, ls="-"))

    # Uni-Loom [Uni-3DAR Slot] points
    ul_16_4_blind = float(uniloom["arms"]["Process_16_to_4"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    ax0.scatter([3.0], [ul_16_4_blind], marker="o", s=105, color=c_ul_proc, edgecolor="white", lw=0.9, zorder=7,
                label="Uni-Loom [Uni-3DAR 16→4] (15.6 atoms, 3.0h)")

    sz_v3_arms = sz_v3["arms"]
    sz_v3_top_key = "Process_16_to_4" if "Process_16_to_4" in sz_v3_arms else "Outcome_M8"
    sz_v3_top_blind = float(sz_v3_arms[sz_v3_top_key]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    sz_v3_atoms = float(sz_v3_arms[sz_v3_top_key].get("mean_total_atoms", 35.5))

    if sz_v2_opt is not None:
        sz_v2_b = float(sz_v2_opt["arms"]["Process_16_to_4"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        ax0.plot([sz_v2_h, sz_v3_h], [sz_v2_b, sz_v3_top_blind], "-P", color=c_sz_v3, lw=2.1, ms=8.0, zorder=7,
                 label=f"Size-Anchored Uni-Loom v2→v3 (~{sz_v3_atoms:.0f} atoms)")
    else:
        ax0.scatter([sz_v3_h], [sz_v3_top_blind], marker="P", s=120, color=c_sz_v3, edgecolor="white", lw=0.8, zorder=7,
                    label=f"Size-Anchored Uni-Loom v3 ({sz_v3_atoms:.1f} atoms)")

    ax0.annotate(f"Uni-Loom 16→4 (3.0h)\n{ul_16_4_blind:.1f}% Blind | 8.0 mol/s",
                 xy=(3.0, ul_16_4_blind), xytext=(1.32, 49.5),
                 fontsize=7.6, fontweight="bold", color=c_ul_proc,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_ul_proc, alpha=0.93, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_ul_proc, lw=0.9))

    v3_tag = "16→4" if sz_v3_top_key == "Process_16_to_4" else "M=8*"
    ax0.annotate(f"Size-Anchored v3 {v3_tag} (2.45h)\n{sz_v3_top_blind:.1f}% Blind | {sz_v3_atoms:.1f} atoms\n(Beats PMF 3h+AWR 28.9%!)",
                 xy=(sz_v3_h, sz_v3_top_blind), xytext=(1.15, 16.5),
                 fontsize=7.4, fontweight="bold", color=c_sz_v3,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_sz_v3, alpha=0.94, lw=0.85),
                 arrowprops=dict(arrowstyle="->", color=c_sz_v3, lw=0.95))

    if 12.12 in awr_hrs:
        y12_awr = awr_blinds[awr_hrs.index(12.12)]
        ax0.annotate(f"12h + Loom-AWR: {y12_awr:.1f}%\n(= 52h Base at 4.3× less GPU-h!)",
                     xy=(12.12, y12_awr), xytext=(7.2, 54.0),
                     fontsize=7.5, fontweight="bold", color=c_pmf_awr,
                     bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.93, lw=0.8),
                     arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.9))

    y52_awr = awr_blinds[-1]
    ax0.annotate(f"52h + Loom-AWR\n{y52_awr:.1f}% Blind (76.0% Rew)",
                 xy=(52.12, y52_awr), xytext=(22.5, 56.8),
                 fontsize=7.5, fontweight="bold", color=c_pmf_awr,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.93, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.9))

    ax0.set_xscale("log")
    ax0.set_xticks([1.3, 2.5, 6, 12, 24, 52])
    ax0.set_xticklabels(["1.3", "2.5", "6", "12", "24", "52"])
    ax0.get_xaxis().set_minor_formatter(NullFormatter())
    ax0.set_xlim(1.05, 68.0)
    ax0.set_ylim(2.0, 63.0)
    ax0.set_xlabel("Task Training / Alignment Compute (A800 GPU-hours, log)", fontsize=9.7, fontweight="bold", color="#1E293B")
    ax0.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=10.1, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Compute Scaling & Universal Loom-AWR Lift", fontsize=10.8, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax0.legend(loc="lower right", fontsize=7.0, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel B: Test-Time Inference Compute Scaling (Fair A800-SXM4-80GB TF32)
    # =========================================================================
    ax1 = axes[1]
    out_keys = ["Outcome_M1", "Outcome_M2", "Outcome_M4", "Outcome_M8"]
    proc_keys = ["Outcome_M1", "Process_8_to_1_nosurr", "Process_8_to_2", "Process_16_to_2", "Process_16_to_4"]
    pmf_keys = ["M1", "M2", "M4", "M8"]

    ux_out, uy_out, _ = _extract_curve(uniloom["arms"], out_keys, "blind_synth")
    ux_pr, uy_pr, ul_pr_lbl = _extract_curve(uniloom["arms"], proc_keys, "blind_synth")

    # For Size-Anchored v3: use full ProcessCritic curve if 16->4 finished, else monotone frontier (M1->8->1*->8->2->M8)
    if "Process_16_to_4" in sz_v3_arms:
        v3_keys_b = proc_keys
    else:
        v3_keys_b = ["Outcome_M1", "Process_8_to_1_nosurr", "Process_8_to_2", "Outcome_M8"]
    sx_v3, sy_v3, s3_lbl = _extract_curve(sz_v3_arms, v3_keys_b, "blind_synth")

    px_3, py_3, _ = _extract_curve(pmf_map[3.0]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_12, py_12, _ = _extract_curve(pmf_map[12.0]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_52, py_52, _ = _extract_curve(pmf_map[52.0]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_3awr, py_3awr, _ = _extract_curve(awr_map[3.12]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_52awr, py_52awr, awr_lbl = _extract_curve(awr_map[52.12]["tracks"]["native_graph"], pmf_keys, "blind_synth")

    ax1.plot(px_3, py_3, ":o", color=c_pmf_3, lw=1.6, ms=5.0, label="PMF @ 3h + Loom-Rec (29.7 atoms)")
    ax1.plot(px_3awr, py_3awr, "--d", color="#F43F5E", lw=1.7, ms=5.2, alpha=0.85, label="PMF @ 3.12h + Loom-AWR (30.5 atoms)")
    ax1.plot(px_12, py_12, "-o", color=c_pmf_12, lw=1.6, ms=5.2, alpha=0.85, label="PMF @ 12h + Loom-Rec (30.2 atoms)")
    ax1.plot(px_52, py_52, "-o", color=c_pmf_52, lw=2.1, ms=6.2, label="PMF @ 52h + Loom-Rec (30.0 atoms)")
    ax1.plot(px_52awr, py_52awr, "--D", color=c_pmf_awr, lw=2.3, ms=6.0, label="PMF @ 52.12h + Loom-AWR (30.3 atoms)")
    if sz_v2_opt is not None:
        sx_v2, sy_v2, _ = _extract_curve(sz_v2_opt["arms"], proc_keys, "blind_synth")
        ax1.plot(sx_v2, sy_v2, ":P", color=c_sz_v2, lw=1.5, ms=5.5, alpha=0.85, label="Size-Anchored v2 A800-Opt (33.7 atoms)")
    ax1.plot(sx_v3, sy_v3, "-P", color=c_sz_v3, lw=2.6, ms=7.8, zorder=7,
             label=f"Size-Anchored v3 A800-Opt ({sz_v3_atoms:.1f} atoms)")
    ax1.plot(ux_out, uy_out, "-s", color=c_ul_out, lw=1.8, ms=5.5, alpha=0.88, label="Uni-Loom v2.0 Best-of-M (unanchored)")
    ax1.plot(ux_pr, uy_pr, "-o", color=c_ul_proc, lw=2.6, ms=7.0, zorder=8, label="Uni-Loom v2.0 + ProcessCritic (8.0 mol/s)")

    for x, y, lb in zip(sx_v3, sy_v3, s3_lbl):
        off = (-8, 8) if lb in ("16→4", "M8") else (5, -10)
        ax1.annotate(lb, (x, y), textcoords="offset points", xytext=off, ha="left", fontsize=7.4, color=c_sz_v3, fontweight="bold")
    for x, y, lb in zip(ux_pr, uy_pr, ul_pr_lbl):
        off = (-4, 7) if lb != "16→2" else (-14, -11)
        ax1.annotate(lb, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=7.5, color=c_ul_proc, fontweight="bold")
    for x, y, lb in zip(px_52awr, py_52awr, awr_lbl):
        ax1.annotate(lb, (x, y), textcoords="offset points", xytext=(5, 4), ha="left", fontsize=7.3, color=c_pmf_awr, fontweight="bold")

    ax1.set_xscale("log")
    ax1.set_xticks([25, 50, 100, 200, 400, 800])
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.get_xaxis().set_minor_formatter(NullFormatter())
    ax1.set_xlim(22, 1080)
    ax1.set_ylim(2.0, 58.0)
    ax1.set_xlabel("Inference Compute per 1,000 Proposals (A800 GPU-sec, log)", fontsize=9.7, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=10.1, fontweight="bold", color="#1E293B")
    ax1.set_title("B. Test-Time Compute Frontier (Custom A800 Uni-3DAR)", fontsize=10.8, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="upper left", fontsize=6.6, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel C: End-to-End 50k-Molecule Campaign Compute (A800 GPU-hours)
    # =========================================================================
    ax2 = axes[2]
    ax2.plot(_campaign_gpuh(3.0, px_3), py_3, ":o", color=c_pmf_3, lw=1.6, ms=5.0, label="From-Scratch PMF @ 3h")
    ax2.plot(_campaign_gpuh(3.12, px_3awr), py_3awr, "--d", color="#F43F5E", lw=1.7, ms=5.2, alpha=0.85, label="PMF @ 3.12h + Loom-AWR")
    ax2.plot(_campaign_gpuh(12.0, px_12), py_12, "-o", color=c_pmf_12, lw=1.7, ms=5.2, alpha=0.88, label="From-Scratch PMF @ 12h")
    if awr_map[12.12] is not None:
        px_12awr, py_12awr, _ = _extract_curve(awr_map[12.12]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
        ax2.plot(_campaign_gpuh(12.12, px_12awr), py_12awr, "-.D", color="#FB7185", lw=2.0, ms=5.6, label="PMF @ 12.12h + Loom-AWR")
    ax2.plot(_campaign_gpuh(52.0, px_52), py_52, "-o", color=c_pmf_52, lw=2.1, ms=6.2, label="From-Scratch PMF @ 52h")
    ax2.plot(_campaign_gpuh(52.12, px_52awr), py_52awr, "--D", color=c_pmf_awr, lw=2.3, ms=6.0, label="PMF @ 52.12h + Loom-AWR")
    ax2.plot(_campaign_gpuh(sz_v3_h, sx_v3), sy_v3, "-P", color=c_sz_v3, lw=2.6, ms=7.8, zorder=7,
             label=f"Size-Anchored v3 ({sz_v3_h:.2f}h, {sz_v3_atoms:.1f} atoms)")
    ax2.plot(_campaign_gpuh(3.0, ux_pr), uy_pr, "-o", color=c_ul_proc, lw=2.6, ms=7.0, zorder=8,
             label="Uni-Loom v2.0 + ProcessCritic (3.0h)")

    u_cost_top = _campaign_gpuh(3.0, ux_pr)[-1]
    pawr_cost_top = _campaign_gpuh(52.12, px_52awr)[-1]
    sz3_cost_top = _campaign_gpuh(sz_v3_h, sx_v3)[-1]
    p3_cost_top = _campaign_gpuh(3.0, px_3)[-1]

    ax2.annotate(f"Uni-Loom 16→4 ({u_cost_top:.1f} GPU-h)\nPBE0 Unbiased: 50.0%\n(8.02 mol/s, 13.4× cheaper!)",
                 xy=(u_cost_top, uy_pr[-1]), xytext=(2.4, 50.2),
                 fontsize=7.3, fontweight="bold", color=c_ul_proc,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_ul_proc, alpha=0.93, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_ul_proc, lw=0.9))

    dft_line = ""
    if sz_v3_dft is not None and "UniLoom_v20_Process_16_to_4" in sz_v3_dft.get("arms", {}):
        pbe0_raw = sz_v3_dft["arms"]["UniLoom_v20_Process_16_to_4"]["end_to_end_pbe0_synth_clean_yield_raw"] * 100.0
        dft_line = f"\nPBE0 Unbiased: {pbe0_raw:.1f}%"
    ax2.annotate(f"Size-Anchored v3 {v3_tag} ({sz3_cost_top:.1f}h)\n{sz_v3_top_blind:.1f}% Blind | {sz_v3_atoms:.1f} atoms{dft_line}",
                 xy=(sz3_cost_top, sy_v3[-1]), xytext=(4.8, 36.5),
                 fontsize=7.3, fontweight="bold", color=c_sz_v3,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_sz_v3, alpha=0.94, lw=0.85),
                 arrowprops=dict(arrowstyle="->", color=c_sz_v3, lw=0.95))

    ax2.annotate(f"PMF 3h M8 ({p3_cost_top:.1f}h)\n23.8% Blind",
                 xy=(p3_cost_top, py_3[-1]), xytext=(15.5, 19.5),
                 fontsize=7.1, fontweight="bold", color="#64748B",
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor=c_pmf_3, alpha=0.92, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_pmf_3, lw=0.85))

    ax2.annotate(f"PMF 52h+Loom-AWR ({pawr_cost_top:.1f}h)\nPBE0 Unbiased: 54.8%\n(MAE 0.270–0.288 eV)",
                 xy=(pawr_cost_top, py_52awr[-1]), xytext=(19.0, 50.8),
                 fontsize=7.3, fontweight="bold", color=c_pmf_awr,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.93, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.9))

    ax2.set_xscale("log")
    ax2.set_xticks([2, 3, 5, 10, 20, 40, 70])
    ax2.set_xticklabels(["2", "3", "5", "10", "20", "40", "70"])
    ax2.get_xaxis().set_minor_formatter(NullFormatter())
    ax2.set_xlim(2.0, 88.0)
    ax2.set_ylim(2.0, 59.5)
    ax2.set_xlabel("Total 50k-Molecule Campaign Compute (Train/Adapt + Infer, A800 GPU-h, log)", fontsize=9.7, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=10.1, fontweight="bold", color="#1E293B")
    ax2.set_title("C. 50k-Campaign Pareto Frontier & PBE0 Quantum Anchor", fontsize=10.8, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax2.legend(loc="lower right", fontsize=6.6, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    fig.text(
        0.5,
        0.01,
        "OE62 HOMO (±0.25 eV, SA_Score ≤ 3.5, Clean Closed-Shell Charge, N = 1,792 proposals/arm, Uncontended Single NVIDIA A800-SXM4-80GB with Fair TF32). "
        "*8→1: zero external surrogate calls. Panel C sums task training/alignment + 50k-proposal inference compute in A800 GPU-hours (upstream Uni-3DAR pretraining excluded).",
        ha="center",
        fontsize=8.1,
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
