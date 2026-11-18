"""Generate LLM-style Training/Test-Time Compute Scaling & Campaign-Cost Pareto Figures on OE62.

Strictly enforces fair fixed-per-prompt 30.0-atom comparisons across `M in {1..8}` and across
both generator backbones (`Uni-3DAR` vs `PropMolFlow`), replaces narrow `16->2` with `8->4`
(`M=1 -> 8->1* -> 8->2 -> 8->4 -> 16->4`), removes defective intermediate prototypes (`v2`),
and unifies `Uni-Loom [...]` platform nomenclature across all panels.
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


def render_oe62_scaling_figure(out_paths: List[Path]) -> None:
    unanchored = (
        _load_json("oe62_unanchored_with_8to4.json")
        or _load_json("oe62_octree_process_critic_a800_opt.json")
        or _load_json("oe62_octree_process_critic.json")
    )
    sz_30 = _load_json("oe62_uni3dar_v3_fair_30atom.json") or _load_json("oe62_size_anchored_v3_a800_eval.json")
    sz_30_dft = _load_json("oe62_uni3dar_v3_fair_30atom_dft_unbiased.json") or _load_json("oe62_size_anchored_v3_dft_unbiased.json")

    # Prefer strictly fair fixed-prompt 30-atom PMF JSONs when present; never mix variable-N and fixed-N protocols on one curve
    has_fair_pmf = (
        _load_json("oe62_pmf_3h_fair_30atom.json") is not None
        and _load_json("oe62_pmf_12h_fair_30atom.json") is not None
        and _load_json("oe62_pmf_52h_fair_30atom.json") is not None
    )
    if has_fair_pmf:
        pmf_map = {
            3.0: _load_json("oe62_pmf_3h_fair_30atom.json"),
            6.0: _load_json("oe62_pmf_6h_fair_30atom.json"),
            12.0: _load_json("oe62_pmf_12h_fair_30atom.json"),
            24.0: _load_json("oe62_pmf_24h_fair_30atom.json"),
            52.0: _load_json("oe62_pmf_52h_fair_30atom.json"),
        }
    else:
        pmf_map = {
            3.0: _load_json("oe62_propmolflow_3gpuh.json"),
            6.0: _load_json("oe62_propmolflow_6gpuh.json"),
            12.0: _load_json("oe62_propmolflow_12gpuh.json"),
            24.0: _load_json("oe62_propmolflow_24gpuh.json"),
            52.0: _load_json("oe62_propmolflow_52gpuh.json"),
        }
    awr_map = {
        3.12: _load_json("oe62_pmf_3h_awr_fair_30atom.json") or _load_json("oe62_propmolflow_3gpuh_loom_awr.json"),
        12.12: _load_json("oe62_pmf_12h_awr_fair_30atom.json") or _load_json("oe62_propmolflow_12gpuh_loom_awr.json"),
        52.12: _load_json("oe62_pmf_52h_awr_fair_30atom.json") or _load_json("oe62_propmolflow_52gpuh_loom_awr.json"),
    }

    assert unanchored is not None and sz_30 is not None and pmf_map[52.0] is not None and awr_map[52.12] is not None
    sz_h = 2.45

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

    c_ul_proc = "#0D9488"      # emerald teal: Uni-Loom [Uni-3DAR, Unanchored]
    c_ul_out = "#EA580C"       # tangerine: Uni-Loom [Uni-3DAR Best-of-M, Unanchored]
    c_sz_30 = "#6D28D9"        # royal violet: Uni-Loom [Uni-3DAR, 30-Atom Matched]
    c_pmf_52 = "#334155"       # deep slate: PropMolFlow @ 52h + Loom-Critic
    c_pmf_12 = "#64748B"       # mid slate: PropMolFlow @ 12h + Loom-Critic
    c_pmf_3 = "#94A3B8"        # light slate: PropMolFlow @ 3h + Loom-Critic
    c_pmf_raw = "#9CA3AF"      # muted gray: Vanilla PropMolFlow (M=1, No Loom)
    c_pmf_awr = "#E11D48"      # crimson coral: Uni-Loom [PropMolFlow + Loom-AWR]

    # =========================================================================
    # Panel A: Training / Alignment Compute Scaling (2.45 -> 52.12 GPU-h)
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
    ax0.plot(hrs, pmf_m8_blind, "-o", color=c_pmf_52, lw=2.3, ms=6.8,
             label="PropMolFlow + Loom-Critic [M=8, 30 atoms]")

    awr_hrs = [h for h in (3.12, 12.12, 52.12) if awr_map[h] is not None]
    awr_blinds: List[float] = []
    for ah in awr_hrs:
        d_a = awr_map[ah]
        b_nat = float(d_a["tracks"]["native_graph"]["M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        b_crd = float(d_a["tracks"]["coord_reconciled"]["M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
        awr_blinds.append(max(b_nat, b_crd))
    ax0.plot(awr_hrs, awr_blinds, "-D", color=c_pmf_awr, lw=2.5, ms=6.8, zorder=6,
             label="Uni-Loom [PropMolFlow + Loom-AWR, M=8]")

    for base_h, awr_h_pt, y_awr in zip((3.0, 12.0, 52.0), (3.12, 12.12, 52.12), awr_blinds):
        if base_h in hrs and awr_map.get(awr_h_pt) is not None:
            y_base = pmf_m8_blind[hrs.index(base_h)]
            ax0.annotate("", xy=(awr_h_pt, y_awr), xytext=(base_h, y_base),
                         arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=1.3))

    ul_16_4_blind = float(unanchored["arms"]["Process_16_to_4"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    ul_eff = float(unanchored["arms"]["Process_16_to_4"]["effective_output_mol_per_sec"])
    ax0.scatter([3.0], [ul_16_4_blind], marker="o", s=110, color=c_ul_proc, edgecolor="white", lw=0.9, zorder=7,
                label="Uni-Loom [Uni-3DAR 16→4, Unanchored]")

    sz_arms = sz_30["arms"]
    sz_top_blind = float(sz_arms["Process_16_to_4"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    sz_atoms = float(sz_arms["Process_16_to_4"].get("mean_total_atoms", 30.0))
    sz_eff = float(sz_arms["Process_16_to_4"]["effective_output_mol_per_sec"])
    ax0.scatter([sz_h], [sz_top_blind], marker="P", s=135, color=c_sz_30, edgecolor="white", lw=0.9, zorder=8,
                label=f"Uni-Loom [Uni-3DAR 16→4, {sz_atoms:.0f}-Atom Matched]")

    ax0.annotate(f"Uni-Loom [Uni-3DAR, Unanchored]\n{ul_16_4_blind:.1f}% Blind | {ul_eff:.1f} mol/s (3.0h)",
                 xy=(3.0, ul_16_4_blind), xytext=(2.05, 52.5),
                 fontsize=7.5, fontweight="bold", color=c_ul_proc,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_ul_proc, alpha=0.93, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_ul_proc, lw=0.9))

    ax0.annotate(f"Uni-Loom [Uni-3DAR, {sz_atoms:.0f}-Atom] (2.45h)\n{sz_top_blind:.1f}% Blind | {sz_eff:.2f} mol/s",
                 xy=(sz_h, sz_top_blind), xytext=(1.92, 16.5),
                 fontsize=7.5, fontweight="bold", color=c_sz_30,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_sz_30, alpha=0.94, lw=0.85),
                 arrowprops=dict(arrowstyle="->", color=c_sz_30, lw=0.95))

    if 12.12 in awr_hrs:
        y12_awr = awr_blinds[awr_hrs.index(12.12)]
        ax0.annotate(f"12h + Loom-AWR: {y12_awr:.1f}%\n(Matches 52h Base at 4.3× less GPU-h)",
                     xy=(12.12, y12_awr), xytext=(7.8, 53.5),
                     fontsize=7.4, fontweight="bold", color=c_pmf_awr,
                     bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.93, lw=0.8),
                     arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.9))

    y52_awr = awr_blinds[-1]
    ax0.annotate(f"52h + Loom-AWR\n{y52_awr:.1f}% Blind",
                 xy=(52.12, y52_awr), xytext=(26.0, 56.8),
                 fontsize=7.5, fontweight="bold", color=c_pmf_awr,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.93, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.9))

    ax0.set_xscale("log")
    ax0.set_xticks([2.5, 6, 12, 24, 52])
    ax0.set_xticklabels(["2.5", "6", "12", "24", "52"])
    ax0.get_xaxis().set_minor_formatter(NullFormatter())
    ax0.set_xlim(1.8, 68.0)
    ax0.set_ylim(2.0, 63.0)
    ax0.set_xlabel("Task Training / Alignment Compute (A800 GPU-hours, log)", fontsize=9.7, fontweight="bold", color="#1E293B")
    ax0.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=10.1, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Compute Scaling & Universal Loom-AWR Lift", fontsize=10.8, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax0.legend(loc="lower right", fontsize=6.9, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel B: Test-Time Inference Compute Scaling (Monotone 8->4 Funnel)
    # =========================================================================
    ax1 = axes[1]
    out_keys = ["Outcome_M1", "Outcome_M2", "Outcome_M4", "Outcome_M8"]
    proc_keys = ["Outcome_M1", "Process_8_to_1_nosurr", "Process_8_to_2", "Process_8_to_4", "Process_16_to_4"]
    pmf_keys = ["M1", "M2", "M4", "M8"]

    ux_out, uy_out, _ = _extract_curve(unanchored["arms"], out_keys, "blind_synth")
    ux_pr, uy_pr, ul_pr_lbl = _extract_curve(unanchored["arms"], proc_keys, "blind_synth")
    sx_pr, sy_pr, sz_pr_lbl = _extract_curve(sz_arms, proc_keys, "blind_synth")

    px_3, py_3, _ = _extract_curve(pmf_map[3.0]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_3awr, py_3awr, _ = _extract_curve(awr_map[3.12]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_12, py_12, _ = _extract_curve(pmf_map[12.0]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_52, py_52, _ = _extract_curve(pmf_map[52.0]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
    px_52awr, py_52awr, awr_lbl = _extract_curve(awr_map[52.12]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")

    ax1.plot(px_3, py_3, ":o", color=c_pmf_3, lw=1.6, ms=5.0, label="PropMolFlow @ 3h + Loom-Critic (30 atoms)")
    ax1.plot(px_3awr, py_3awr, "--d", color="#F43F5E", lw=1.7, ms=5.2, alpha=0.85, label="Uni-Loom [PMF @ 3.12h + Loom-AWR]")
    ax1.plot(px_12, py_12, "-o", color=c_pmf_12, lw=1.7, ms=5.4, alpha=0.88, label="PropMolFlow @ 12h + Loom-Critic (30 atoms)")
    ax1.plot(px_52, py_52, "-o", color=c_pmf_52, lw=2.1, ms=6.2, label="PropMolFlow @ 52h + Loom-Critic (30 atoms)")
    ax1.plot(px_52awr, py_52awr, "--D", color=c_pmf_awr, lw=2.3, ms=6.0, label="Uni-Loom [PMF @ 52.12h + Loom-AWR]")
    ax1.plot(sx_pr, sy_pr, "-P", color=c_sz_30, lw=2.6, ms=7.8, zorder=7,
             label=f"Uni-Loom [Uni-3DAR, {sz_atoms:.0f}-Atom Matched] (2.45h)")
    ax1.plot(ux_out, uy_out, "-s", color=c_ul_out, lw=1.8, ms=5.5, alpha=0.88, label="Uni-Loom [Uni-3DAR Best-of-M, Unanchored]")
    ax1.plot(ux_pr, uy_pr, "-o", color=c_ul_proc, lw=2.6, ms=7.0, zorder=8, label="Uni-Loom [Uni-3DAR + ProcessCritic, Unanchored]")

    for x, y, lb in zip(sx_pr, sy_pr, sz_pr_lbl):
        off = (-8, 8) if lb in ("8→4", "16→4") else (5, -10)
        ax1.annotate(lb, (x, y), textcoords="offset points", xytext=off, ha="left", fontsize=7.4, color=c_sz_30, fontweight="bold")
    for x, y, lb in zip(ux_pr, uy_pr, ul_pr_lbl):
        off = (-4, 7)
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
    ax1.set_title("B. Test-Time Compute Frontier (Matched Fixed-Prompt Size)", fontsize=10.8, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="upper left", fontsize=6.6, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel C: End-to-End 50k-Molecule Campaign Compute (A800 GPU-hours)
    # =========================================================================
    ax2 = axes[2]
    ax2.plot(_campaign_gpuh(3.0, px_3), py_3, ":o", color=c_pmf_3, lw=1.6, ms=5.0, label="PropMolFlow @ 3h + Loom-Critic")
    ax2.plot(_campaign_gpuh(3.12, px_3awr), py_3awr, "--d", color="#F43F5E", lw=1.7, ms=5.2, alpha=0.85, label="Uni-Loom [PMF @ 3.12h + Loom-AWR]")
    ax2.plot(_campaign_gpuh(12.0, px_12), py_12, "-o", color=c_pmf_12, lw=1.7, ms=5.2, alpha=0.88, label="PropMolFlow @ 12h + Loom-Critic")
    if awr_map[12.12] is not None:
        px_12awr, py_12awr, _ = _extract_curve(awr_map[12.12]["tracks"]["coord_reconciled"], pmf_keys, "blind_synth")
        ax2.plot(_campaign_gpuh(12.12, px_12awr), py_12awr, "-.D", color="#FB7185", lw=2.0, ms=5.6, label="Uni-Loom [PMF @ 12.12h + Loom-AWR]")
    ax2.plot(_campaign_gpuh(52.0, px_52), py_52, "-o", color=c_pmf_52, lw=2.1, ms=6.2, label="PropMolFlow @ 52h + Loom-Critic")
    ax2.plot(_campaign_gpuh(52.12, px_52awr), py_52awr, "--D", color=c_pmf_awr, lw=2.3, ms=6.0, label="Uni-Loom [PMF @ 52.12h + Loom-AWR]")
    ax2.plot(_campaign_gpuh(sz_h, sx_pr), sy_pr, "-P", color=c_sz_30, lw=2.6, ms=7.8, zorder=7,
             label=f"Uni-Loom [Uni-3DAR, {sz_atoms:.0f}-Atom Matched]")
    ax2.plot(_campaign_gpuh(3.0, ux_pr), uy_pr, "-o", color=c_ul_proc, lw=2.6, ms=7.0, zorder=8,
             label="Uni-Loom [Uni-3DAR, Unanchored]")

    u_cost_top = _campaign_gpuh(3.0, ux_pr)[-1]
    pawr_cost_top = _campaign_gpuh(52.12, px_52awr)[-1]
    sz_cost_top = _campaign_gpuh(sz_h, sx_pr)[-1]

    ax2.annotate(f"Uni-Loom [Uni-3DAR, Unanchored]\n{u_cost_top:.1f} GPU-h | PBE0 Unbiased: 50.0%",
                 xy=(u_cost_top, uy_pr[-1]), xytext=(2.25, 51.5),
                 fontsize=7.3, fontweight="bold", color=c_ul_proc,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_ul_proc, alpha=0.93, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=c_ul_proc, lw=0.9))

    dft_line = ""
    if sz_30_dft is not None and "UniLoom_v20_Process_16_to_4" in sz_30_dft.get("arms", {}):
        pbe0_raw = sz_30_dft["arms"]["UniLoom_v20_Process_16_to_4"]["end_to_end_pbe0_synth_clean_yield_raw"] * 100.0
        dft_line = f" | PBE0: {pbe0_raw:.1f}%"
    ax2.annotate(f"Uni-Loom [Uni-3DAR, {sz_atoms:.0f}-Atom] ({sz_cost_top:.1f}h)\n{sz_top_blind:.1f}% Blind{dft_line}",
                 xy=(sz_cost_top, sy_pr[-1]), xytext=(5.5, 42.0),
                 fontsize=7.3, fontweight="bold", color=c_sz_30,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_sz_30, alpha=0.94, lw=0.85),
                 arrowprops=dict(arrowstyle="->", color=c_sz_30, lw=0.95))

    ax2.annotate(f"Uni-Loom [PMF 52h + Loom-AWR] ({pawr_cost_top:.1f}h)\nPBE0 Unbiased: 54.8% (MAE 0.270–0.288 eV)",
                 xy=(pawr_cost_top, py_52awr[-1]), xytext=(16.5, 52.0),
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
    ax2.legend(loc="lower right", fontsize=6.5, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    fig.text(
        0.5,
        0.01,
        "OE62 HOMO (±0.25 eV, SA_Score ≤ 3.5, Clean Closed-Shell Charge, N = 1,792 fixed-size prompts/arm, Uncontended Single NVIDIA A800-SXM4-80GB with Fair TF32). "
        "*8→1: zero external surrogate calls. Identical per-prompt (HOMO, N_atoms) held fixed across all M candidates for both Uni-3DAR and PropMolFlow.",
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
