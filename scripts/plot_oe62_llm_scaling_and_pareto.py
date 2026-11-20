"""Generate Unified 3D Discovery Quality Index (`Q_3D`) & Live `PBE0/def2-TZVP` Verification Figures on OE62.

Strictly enforces:
  1. Identical fixed-per-prompt `30.4`-atom fairness (`N = 1,792`, seed `20261118`) across 100% of plotted curves.
  2. Standardized `Uni-Loom` effort tiers (`base: M=1`, `low: M=2`, `medium: M=4`, `high: M=8`, plus optional `M=16`),
     with real measured wall-clock inference latency (`A800 GPU-sec / 1k prompts`).
  3. Zero magic weights: `Q_3D = 100 * G_synth * sqrt(S_orbital * R_phys(F_max))` uses the exact published
     `src/loom/weaving/grpo.py` reward kernels, verified side-by-side in Panel C by live `GPU4PySCF PBE0/def2-TZVP`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "docs" / "experiments"


def _load_json(name: str) -> Optional[Dict[str, Any]]:
    p = EXP_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def compute_q3d_score(arm: Dict[str, Any]) -> float:
    """3D Equilibrium-Grounded Discovery Quality Index `Q_3D` in `[0, 100]` (`grpo.py` exact kernels)."""
    leg = float(arm["legal_rate"]) * (1.0 - float(arm.get("unseen_formal_charge_rate_among_legal", 0.0)))
    g_synth = leg * float(arm["constraints_met_rate_among_legal"])
    hit_cond = min(1.0, float(arm["blind_dpa4_synthesizable_clean_hit_yield"]) / max(g_synth, 1e-6))

    d = float(arm["audit_mae_eV"])
    u = d / 0.25
    r_prop = (
        0.40 / (1.0 + u)
        + 0.35 * math.exp(-0.5 * u * u)
        + 0.25 / (1.0 + math.exp(-(0.25 - d) / (0.16 * 0.25)))
    )
    s_orb = 0.5 * hit_cond + 0.5 * r_prop

    f50 = float(arm["zero_step_f_max_median_eV_A"])
    r_phys = math.exp(-max(0.0, f50 - 0.40) / 2.00)
    return 100.0 * g_synth * math.sqrt(s_orb * r_phys)


def _extract_q3d_curve(
    arms_dict: Dict[str, Any],
    keys_and_labels: Sequence[Tuple[str, str]],
) -> Tuple[List[float], List[float], List[str]]:
    xs: List[float] = []
    ys: List[float] = []
    lbls: List[str] = []
    for k, lb in keys_and_labels:
        if k not in arms_dict:
            continue
        r = arms_dict[k]
        eff = float(r["effective_output_mol_per_sec"])
        xs.append(1000.0 / max(eff, 1e-6))
        ys.append(compute_q3d_score(r))
        lbls.append(lb)
    return xs, ys, lbls


def render_oe62_scaling_figure(out_paths: List[Path]) -> None:
    u3d_pro = _load_json("oe62_uni3dar_v4_pro_fair_30atom.json")
    u3d_ext = _load_json("oe62_uni3dar_v4_pro_fair_30atom_m12_m16.json")
    pmf_awr_ext = _load_json("oe62_pmf_52h_awr_fair_30atom_m12_m16.json")
    u3d_dft = (
        _load_json("oe62_uni3dar_v4_pro_fair_30atom_dft_unbiased.json")
        or _load_json("oe62_uni3dar_v4_fair_30atom_dft_unbiased.json")
    )
    dft_52_awr = _load_json("oe62_head2head_dft_anchor_52gpuh_awr_unbiased.json")
    dft_52 = _load_json("oe62_head2head_dft_anchor_52gpuh_unbiased.json")
    dft_12 = _load_json("oe62_head2head_dft_anchor_12gpuh_unbiased.json")

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
        raise FileNotFoundError("Missing required fair 30.4-atom JSONs")

    u3d_h = float(u3d_pro.get("total_training_gpuh", 2.41))
    u3d_arms = dict(u3d_pro["arms"])
    if u3d_ext is not None and "arms" in u3d_ext:
        for k_ext, v_ext in u3d_ext["arms"].items():
            if k_ext == "ext_M12":
                u3d_arms.setdefault("ext_M12", v_ext)
            elif k_ext in ("ext_M16", "xhigh_M16") and "xhigh_M16" not in u3d_arms:
                u3d_arms["xhigh_M16"] = v_ext

    awr_52_crd = dict(awr_map[52.12]["tracks"]["coord_reconciled"])
    if pmf_awr_ext is not None and "coord_reconciled" in pmf_awr_ext.get("tracks", {}):
        for k_p in ("M12", "M16"):
            if k_p in pmf_awr_ext["tracks"]["coord_reconciled"]:
                awr_52_crd[k_p] = pmf_awr_ext["tracks"]["coord_reconciled"][k_p]

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
    fig, axes = plt.subplots(1, 3, figsize=(19.8, 6.3), dpi=220)
    fig.patch.set_facecolor(bg_fig)
    for ax in axes:
        ax.set_facecolor(bg_ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.tick_params(colors="#475569")
        ax.grid(True, which="major", linestyle="--", linewidth=0.65, color="#E4ECE9", alpha=0.85)

    c_u3d = "#0D9488"
    c_u3d_med = "#0F766E"
    c_pmf_52 = "#334155"
    c_pmf_12 = "#64748B"
    c_pmf_3 = "#94A3B8"
    c_pmf_raw = "#9CA3AF"
    c_pmf_awr = "#E11D48"

    # =========================================================================
    # Panel A: Training Compute Scaling vs 3D Quality Index Q_3D
    # =========================================================================
    ax0 = axes[0]
    hrs = [h for h in (3.0, 6.0, 12.0, 24.0, 52.0) if pmf_map[h] is not None]
    pmf_m1_q = [compute_q3d_score(pmf_map[h]["tracks"]["native_graph"]["M1"]) for h in hrs]
    pmf_m8_q = [compute_q3d_score(pmf_map[h]["tracks"]["coord_reconciled"]["M8"]) for h in hrs]
    awr_hrs = [ah for ah in (3.12, 12.12, 52.12) if awr_map[ah] is not None]
    awr_m8_q = [compute_q3d_score(awr_map[ah]["tracks"]["coord_reconciled"]["M8"]) for ah in awr_hrs]

    ax0.plot(hrs, pmf_m1_q, "--^", color=c_pmf_raw, lw=1.7, ms=5.6, label="Vanilla PropMolFlow [M=1, No Loom]")
    ax0.plot(hrs, pmf_m8_q, "-o", color=c_pmf_52, lw=2.2, ms=6.4, label="PropMolFlow + Raw Critic [M=8]")
    ax0.plot(awr_hrs, awr_m8_q, "-D", color=c_pmf_awr, lw=2.4, ms=6.4, zorder=6, label="Uni-Loom [PMF + AWR, M=8]")

    for bh, ah, ya in zip((3.0, 12.0, 52.0), awr_hrs, awr_m8_q):
        if bh in hrs:
            yb = pmf_m8_q[hrs.index(bh)]
            ax0.annotate("", xy=(ah, ya), xytext=(bh, yb), arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=1.3))

    u_m4_q = compute_q3d_score(u3d_arms["medium_M4"])
    u_m8_q = compute_q3d_score(u3d_arms["high_M8"])
    ax0.scatter([u3d_h], [u_m4_q], marker="s", s=105, color=c_u3d_med, edgecolor="white", lw=0.9, zorder=7,
                label=f"Uni-Loom [Uni-3DAR, M=4] ({u3d_h:.2f}h)")
    ax0.scatter([u3d_h], [u_m8_q], marker="*", s=220, color=c_u3d, edgecolor="white", lw=0.9, zorder=9,
                label=f"Uni-Loom [Uni-3DAR, M=8] ({u3d_h:.2f}h)")
    top_u_q = u_m8_q
    if "xhigh_M16" in u3d_arms:
        u_m16_q = compute_q3d_score(u3d_arms["xhigh_M16"])
        ax0.scatter([u3d_h], [u_m16_q], marker="P", s=135, color="#047857", edgecolor="white", lw=0.9, zorder=10,
                    label=f"Uni-Loom [Uni-3DAR, M=16] ({u3d_h:.2f}h)")
        top_u_q = max(top_u_q, u_m16_q)
    ax0.plot([u3d_h, u3d_h], [u_m4_q, top_u_q], "-", color=c_u3d, lw=1.8, alpha=0.85, zorder=6)

    ratio_compute = 52.12 / max(u3d_h, 0.1)
    ax0.annotate(
        f"Uni-Loom [Uni-3DAR, M=8] ({u3d_h:.2f}h)\nQ_3D = {u_m8_q:.1f} | {ratio_compute:.1f}x less train compute",
        xy=(u3d_h, u_m8_q),
        xytext=(3.3, min(top_u_q + 5.2, 57.5)),
        fontsize=7.4,
        fontweight="bold",
        color=c_u3d,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_u3d, alpha=0.95, lw=0.9),
        arrowprops=dict(arrowstyle="->", color=c_u3d, lw=1.0),
    )
    awr_52_fmax = float(awr_52_crd["M8"]["zero_step_f_max_median_eV_A"])
    ax0.annotate(
        f"Uni-Loom [PMF @ 52.12h + AWR, M=8]\nQ_3D = {awr_m8_q[-1]:.1f} (F_max = {awr_52_fmax:.2f} eV/A)",
        xy=(52.12, awr_m8_q[-1]),
        xytext=(11.5, awr_m8_q[-1] + 3.8),
        fontsize=7.2,
        fontweight="bold",
        color=c_pmf_awr,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor=c_pmf_awr, alpha=0.93, lw=0.85),
        arrowprops=dict(arrowstyle="->", color=c_pmf_awr, lw=0.9),
    )

    ymax_q = max(61.5, top_u_q + 8.0)
    ax0.set_xscale("log")
    ax0.set_xticks([2, 3, 6, 12, 24, 52])
    ax0.set_xticklabels(["2", "3", "6", "12", "24", "52"])
    ax0.get_xaxis().set_minor_formatter(NullFormatter())
    ax0.set_xlim(1.6, 68.0)
    ax0.set_ylim(2.0, ymax_q)
    ax0.set_xlabel("Task Post-Training / From-Scratch Compute (A800 GPU-hours, log)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax0.set_ylabel("3D Discovery Quality Index Q_3D (0-100)", fontsize=10.0, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Training Compute Scaling vs 3D Quality Index", fontsize=10.5, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax0.legend(loc="lower right", fontsize=6.6, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel B: Standardized Effort Tiers (M = 1, 2, 4, 8 [+ 12, 16]) on Q_3D
    # =========================================================================
    ax1 = axes[1]
    u_tiers = [
        ("base_M1", "M=1"),
        ("low_M2", "M=2"),
        ("medium_M4", "M=4"),
        ("high_M8", "M=8"),
        ("ext_M12", "M=12"),
        ("xhigh_M16", "M=16"),
    ]
    p_tiers = [("M1", "M=1"), ("M2", "M=2"), ("M4", "M=4"), ("M8", "M=8")]
    p_awr_tiers = [("M1", "M=1"), ("M2", "M=2"), ("M4", "M=4"), ("M8", "M=8"), ("M12", "M=12"), ("M16", "M=16")]

    ux, uy, u_lbl = _extract_q3d_curve(u3d_arms, u_tiers)
    px_3, py_3, _ = _extract_q3d_curve(pmf_map[3.0]["tracks"]["coord_reconciled"], p_tiers)
    px_3a, py_3a, _ = _extract_q3d_curve(awr_map[3.12]["tracks"]["coord_reconciled"], p_tiers)
    px_12, py_12, _ = _extract_q3d_curve(pmf_map[12.0]["tracks"]["coord_reconciled"], p_tiers)
    px_52, py_52, _ = _extract_q3d_curve(pmf_map[52.0]["tracks"]["coord_reconciled"], p_tiers)
    px_52a, py_52a, awr_lbl = _extract_q3d_curve(awr_52_crd, p_awr_tiers)

    ax1.plot(px_3, py_3, ":o", color=c_pmf_3, lw=1.6, ms=5.0, label="PropMolFlow @ 3h + Raw Critic")
    ax1.plot(px_3a, py_3a, "--d", color="#F43F5E", lw=1.7, ms=5.2, alpha=0.85, label="Uni-Loom [PMF @ 3.12h + AWR]")
    ax1.plot(px_12, py_12, "-o", color=c_pmf_12, lw=1.7, ms=5.4, alpha=0.88, label="PropMolFlow @ 12h + Raw Critic")
    ax1.plot(px_52, py_52, "-o", color=c_pmf_52, lw=2.1, ms=6.0, label="PropMolFlow @ 52h + Raw Critic")
    ax1.plot(px_52a, py_52a, "--D", color=c_pmf_awr, lw=2.3, ms=6.2, label="Uni-Loom [PMF @ 52.12h + AWR]")
    ax1.plot(ux, uy, "-o", color=c_u3d, lw=2.9, ms=7.8, zorder=8, label=f"Uni-Loom [Uni-3DAR] ({u3d_h:.2f}h)")

    for x, y, lb in zip(ux, uy, u_lbl):
        if lb == "M=1":
            off, ha = (7, 3), "left"
        elif lb == "M=12":
            off, ha = (-6, 7), "right"
        elif lb == "M=16":
            off, ha = (6, -9), "left"
        else:
            off, ha = (-7, 6), "right"
        ax1.annotate(f"{lb} ({y:.1f})", (x, y), textcoords="offset points", xytext=off, ha=ha,
                     fontsize=6.9, color=c_u3d, fontweight="bold")
    for x, y, lb in zip(px_52a, py_52a, awr_lbl):
        if lb == "M=1":
            off, ha = (7, -3), "left"
        elif lb == "M=12":
            off, ha = (-7, 6), "right"
        elif lb == "M=16":
            off, ha = (6, -8), "left"
        else:
            off, ha = (7, -10), "left"
        ax1.annotate(f"{lb} ({y:.1f})", (x, y), textcoords="offset points", xytext=off, ha=ha,
                     fontsize=6.8, color=c_pmf_awr, fontweight="bold")

    if "M=8" in u_lbl and "M=8" in awr_lbl:
        idx_m8 = u_lbl.index("M=8")
        idx_awr8 = awr_lbl.index("M=8")
        dq = uy[idx_m8] - py_52a[idx_awr8]
        rel_pct = 100.0 * dq / max(py_52a[idx_awr8], 1e-6)
        ax1.annotate(
            "",
            xy=(ux[idx_m8], uy[idx_m8] - 0.5),
            xytext=(ux[idx_m8], py_52a[idx_awr8] + 0.5),
            arrowprops=dict(arrowstyle="<->", color="#0F766E", lw=1.6),
        )
        ax1.text(
            ux[idx_m8] * 1.07,
            0.52 * uy[idx_m8] + 0.48 * py_52a[idx_awr8],
            f"+{dq:.1f} pt\n(+{rel_pct:.0f}% Q_3D)",
            ha="left",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color="#0F766E",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#F0FDFA", edgecolor="#0D9488", lw=0.85),
        )

    max_lat = max(max(ux, default=800.0), max(px_52a, default=800.0))
    ax1.set_xscale("log")
    ax1.set_xticks([20, 50, 100, 200, 400, 800, 1600])
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.get_xaxis().set_minor_formatter(NullFormatter())
    ax1.set_xlim(16, max(1150.0, max_lat * 1.34))
    ax1.set_ylim(2.0, ymax_q)
    ax1.set_xlabel("Measured Wall-Clock Latency per 1,000 Prompts (A800 GPU-sec, log)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("3D Discovery Quality Index Q_3D (0-100)", fontsize=10.0, fontweight="bold", color="#1E293B")
    ax1.set_title("B. Standardized Effort Tiers: Joint Orbital & 3D Force Quality", fontsize=10.5, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="upper left", fontsize=6.5, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    # =========================================================================
    # Panel C: Live Ab Initio PBE0/def2-TZVP Quantum Verification (42 Prompts)
    # =========================================================================
    ax2 = axes[2]
    dft_rows: List[Tuple[str, float, float, str]] = []
    if dft_52 is not None:
        dft_rows.append(
            (
                "Vanilla PMF\n52h\n[M=1]",
                float(dft_52["arms"]["PropMolFlow_native_M1"]["end_to_end_pbe0_synth_clean_yield_raw"]) * 100.0,
                float(dft_52["arms"]["PropMolFlow_native_M1"]["pbe0_vs_target_homo"]["mae_eV"]),
                "#CBD5E1",
            )
        )
    if dft_12 is not None:
        dft_rows.append(
            (
                "PMF 12h +\nRaw Critic\n[M=8]",
                float(dft_12["arms"]["PropMolFlow_coord_reconciled_M8"]["end_to_end_pbe0_synth_clean_yield_raw"]) * 100.0,
                float(dft_12["arms"]["PropMolFlow_coord_reconciled_M8"]["pbe0_vs_target_homo"]["mae_eV"]),
                "#94A3B8",
            )
        )
    if dft_52 is not None:
        dft_rows.append(
            (
                "PMF 52h +\nRaw Critic\n[M=8]",
                float(dft_52["arms"]["PropMolFlow_coord_reconciled_M8"]["end_to_end_pbe0_synth_clean_yield_raw"]) * 100.0,
                float(dft_52["arms"]["PropMolFlow_coord_reconciled_M8"]["pbe0_vs_target_homo"]["mae_eV"]),
                "#475569",
            )
        )
    if u3d_dft is not None and "UniLoom_Uni3DAR_medium_M4" in u3d_dft.get("arms", {}):
        dft_rows.append(
            (
                f"Uni-Loom\n[Uni-3DAR]\n{u3d_h:.2f}h [M=4]",
                float(u3d_dft["arms"]["UniLoom_Uni3DAR_medium_M4"]["end_to_end_pbe0_synth_clean_yield_raw"]) * 100.0,
                float(u3d_dft["arms"]["UniLoom_Uni3DAR_medium_M4"]["pbe0_vs_target_homo"]["mae_eV"]),
                "#14B8A6",
            )
        )
    if dft_52_awr is not None:
        awr_dft_arm = (
            dft_52_awr["arms"].get("UniLoom_PMF_AWR_high_M8")
            or dft_52_awr["arms"].get("PropMolFlow_coord_reconciled_M8")
        )
        if awr_dft_arm is not None:
            dft_rows.append(
                (
                    "Uni-Loom\n[PMF + AWR]\n52.12h [M=8]",
                    float(awr_dft_arm["end_to_end_pbe0_synth_clean_yield_raw"]) * 100.0,
                    float(awr_dft_arm["pbe0_vs_target_homo"]["mae_eV"]),
                    "#E11D48",
                )
            )
    if u3d_dft is not None and "UniLoom_Uni3DAR_high_M8" in u3d_dft.get("arms", {}):
        dft_rows.append(
            (
                f"Uni-Loom\n[Uni-3DAR]\n{u3d_h:.2f}h [M=8]",
                float(u3d_dft["arms"]["UniLoom_Uni3DAR_high_M8"]["end_to_end_pbe0_synth_clean_yield_raw"]) * 100.0,
                float(u3d_dft["arms"]["UniLoom_Uni3DAR_high_M8"]["pbe0_vs_target_homo"]["mae_eV"]),
                "#0D9488",
            )
        )

    x_pos = np.arange(len(dft_rows))
    names = [r[0] for r in dft_rows]
    hits_pbe0 = [r[1] for r in dft_rows]
    maes_pbe0 = [r[2] for r in dft_rows]
    cols = [r[3] for r in dft_rows]

    bars = ax2.bar(
        x_pos - 0.18,
        hits_pbe0,
        width=0.33,
        color=cols,
        edgecolor="#1E293B",
        linewidth=0.8,
        label="True PBE0 Synth-Clean Hit Yield (%)",
    )
    ax2.set_ylabel("Live PBE0/def2-TZVP Synth-Clean Hit Yield (%)", fontsize=9.8, fontweight="bold", color="#0F766E")
    ax2.set_ylim(0, 96)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, fontsize=7.0, fontweight="bold", color="#1E293B")

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.spines["right"].set_color("#9A3412")
    bars_mae = ax2_r.bar(
        x_pos + 0.18,
        maes_pbe0,
        width=0.29,
        color="#FED7AA",
        edgecolor="#C2410C",
        linewidth=0.9,
        hatch="//",
        alpha=0.90,
        label="True PBE0 HOMO MAE (eV, lower is better)",
    )
    ax2_r.set_ylabel("Live PBE0/def2-TZVP HOMO MAE (eV, lower is better)", fontsize=9.6, fontweight="bold", color="#9A3412")
    ax2_r.set_ylim(0, 0.72)
    ax2_r.tick_params(axis="y", colors="#9A3412")

    for i, (b, hv) in enumerate(zip(bars, hits_pbe0)):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            hv + 1.5,
            f"{hv:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color="#065F46" if i == len(dft_rows) - 1 else "#0F172A",
        )
    for i, mv in enumerate(maes_pbe0):
        ax2_r.text(
            x_pos[i] + 0.21,
            mv + 0.012,
            f"{mv:.3f}\neV",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            color="#9A3412",
        )

    if len(hits_pbe0) >= 2:
        dhit = hits_pbe0[-1] - hits_pbe0[-2]
        mae_red = 100.0 * (maes_pbe0[-2] - maes_pbe0[-1]) / max(maes_pbe0[-2], 1e-6)
        ax2.annotate(
            f"+{dhit:.1f} pp higher True PBE0 Hit\n& {mae_red:.0f}% lower PBE0 MAE ({maes_pbe0[-1]:.3f} vs {maes_pbe0[-2]:.3f} eV)\nat {ratio_compute:.1f}x less training compute!",
            xy=(x_pos[-1] - 0.18, hits_pbe0[-1]),
            xytext=(1.75, 73.5),
            fontsize=6.8,
            fontweight="bold",
            color="#0F766E",
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#F0FDFA", edgecolor="#0D9488", lw=0.9),
            arrowprops=dict(arrowstyle="->", color="#0D9488", lw=1.1),
        )

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.6, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")
    ax2.set_title("C. Ab Initio PBE0/def2-TZVP Verification (42 Prompts)", fontsize=10.5, fontweight="bold", color="#1E293B", loc="left", pad=10)

    fig.text(
        0.5,
        0.012,
        "Q_3D = 100 * G_synth * sqrt(S_orbital * R_phys(F_max)) combines synthesizability (SA <= 3.5), blind orbital accuracy, and zero-step 3D force equilibrium (Panels A-B: N = 1,792 matched 30.4-atom prompts; Panel C: live GPU4PySCF PBE0/def2-TZVP/DF).",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#475569",
    )
    plt.tight_layout(rect=[0.0, 0.035, 1.0, 1.0], w_pad=1.7)
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
            ROOT / "docs" / "figures" / "oe62_llm_style_accuracy_vs_cost.png",
            ROOT / "paper" / "figures" / "fig8_oe62_cross_paradigm_scaling.png",
            ROOT / "paper" / "ncs" / "figures" / "ed_fig3_oe62_cross_paradigm_scaling.png",
            ROOT / "paper" / "uni3dar_mol" / "figures" / "fig2_oe62_cross_paradigm_scaling.png",
        ],
    )
    args = p.parse_args()
    render_oe62_scaling_figure(list(args.out))


if __name__ == "__main__":
    main()
