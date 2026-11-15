"""Generate publication-quality vector PDF + PNG figures for `Uni-Loom` (`loom v2.0`):
  - Fig 1: Calibration & Concentration Trajectory (`Frozen SFT` -> `Legacy GRPO` -> `Loom Stage 0` -> `Stage 1` -> `v1.0` -> `v1.1`)
  - Fig 2: Anti-Reward-Hacking Audit Curve (`RewardOracle` vs `AuditOracle`) + `DPA-4 <-> HyFF` Hot-Swap
  - Fig 3: Generalized 1D & 2D Predictor-Agnostic Heteroscedastic GP Reachability & Dual-Uncertainty Atlas
  - Fig 4 (`v2.0`): Dual-Track Exotic Zwitterion Elimination & Zero-Retrain Multi-Asset Synthesizable Weaving (`SA_Score <= 3.5`)
  - Fig 5 (`v2.0`): 2-Round Grounded RSI Closed-Loop (Blindspot Uncertainty Contraction, 4-Zone State Unlocking & Blind Audit Hit Yield)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_gp_atlas(atlas_json: Path, out_dir: Path) -> None:
    _style()
    data = json.loads(atlas_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    h1 = data.get("homo_1d_atlas")
    l1 = data.get("lumo_1d_atlas")
    j2 = data.get("joint_2d_atlas")

    for default_label, tag, a1 in [("HOMO", "homo", h1), ("LUMO", "lumo", l1)]:
        if a1 is None:
            continue
        axis_name = str(a1.get("axis_name", default_label)).upper() if a1.get("axis_name") in ("homo", "lumo") else str(a1.get("axis_name", default_label))
        if axis_name == "target":
            axis_name = default_label
        unit = str(a1.get("unit", "eV"))
        tol = float(a1.get("tolerance", a1.get("tolerance_eV", 0.25)))

        t = np.asarray(a1.get("targets", a1["targets_eV"]))
        mu = np.asarray(a1.get("mu_gen", a1["mu_gen_eV"]))
        sd_gen = np.asarray(a1.get("sigma_gen", a1["sigma_gen_eV"]))
        sd_tot = np.asarray(a1.get("sigma_total", a1["sigma_total_eV"]))
        p_rew = np.asarray(a1["p_hit_reward"]) * 100.0
        p_dft = np.asarray(a1["p_hit_dft_calibrated"]) * 100.0
        kde = np.asarray(a1["source_kde_density"])

        # Shared numeric range when both axes represent the same physical unit (`eV`)
        v_min = math.floor(min(float(t.min()), float((mu - sd_tot).min())) * 2.0) / 2.0
        v_max = math.ceil(max(float(t.max()), float(mu.max())) * 2.0) / 2.0
        interval = a1.get("trusted_reachable_interval", a1.get("trusted_reachable_interval_eV"))

        # Separate side-by-side panels (Panel A: exact 1:1 square calibration; Panel B: independent reachability)
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(11.8, 5.15),
            layout="constrained",
            gridspec_kw={"width_ratios": [1.0, 1.28]},
        )

        # Panel A: Strict 1:1 Equal-Scale Calibration Plot
        ax1.plot([v_min, v_max], [v_min, v_max], "--", color="#1f2937", lw=1.3, label="Ideal slope = 1 ($\\mu = t$)")
        ax1.fill_between(
            t,
            mu - sd_tot,
            mu + sd_tot,
            color="#93c5fd",
            alpha=0.28,
            label="Total spread $\\pm\\sigma_{\\mathrm{total}}(t)$ (Gen + Cal)",
        )
        ax1.fill_between(
            t,
            mu - sd_gen,
            mu + sd_gen,
            color="#3b82f6",
            alpha=0.34,
            label="Generator spread $\\pm\\sigma_{\\mathrm{gen}}(t)$",
        )
        ax1.plot(t, mu, color="#1e3a8a", lw=2.2, label="GP Posterior Mean $\\hat{\\mu}_{\\mathrm{gen}}(t)$")

        emp_t = [g["target_eV"] for g in a1["empirical_grid"] if np.isfinite(g["mean_pred_eV"])]
        emp_m = [g["mean_pred_eV"] for g in a1["empirical_grid"] if np.isfinite(g["mean_pred_eV"])]
        ax1.scatter(emp_t, emp_m, color="#dc2626", s=24, zorder=5, label="Sampled 3D cohort means")
        if interval:
            lo, hi = interval
            ax1.axvline(lo, color="#d97706", ls="--", lw=1.35, alpha=0.85, label="Trusted zone bounds")
            ax1.axvline(hi, color="#d97706", ls="--", lw=1.35, alpha=0.85)
        ax1.set_xlim(v_min, v_max)
        ax1.set_ylim(v_min, v_max)
        ax1.xaxis.set_major_locator(MultipleLocator(1.0))
        ax1.yaxis.set_major_locator(MultipleLocator(1.0))
        ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_box_aspect(1.0)
        ax1.set_xlabel(f"Requested {axis_name} Target $t$ ({unit})")
        ax1.set_ylabel(f"Achieved {axis_name} ({unit})")
        ax1.set_title(f"A. 1D {axis_name} Calibration (1:1 Equal Scale)", loc="left", pad=8)
        ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.94, fontsize=7.8)
        ax1.grid(alpha=0.22)

        # Panel B: Independent X-Axis Reachability, Hit Probability & Prior Density (distinct colors)
        if interval:
            lo, hi = interval
            ax2.axvspan(
                lo,
                hi,
                facecolor="#fef08a",
                alpha=0.38,
                zorder=1,
                label=f"Trusted Reachable [{lo:.2f}, {hi:.2f}] {unit}",
            )
            ax2.axvline(lo, color="#d97706", ls="--", lw=1.35, alpha=0.90, zorder=2)
            ax2.axvline(hi, color="#d97706", ls="--", lw=1.35, alpha=0.90, zorder=2)
        if kde.max() > 0:
            kde_scaled = (kde / kde.max()) * max(float(p_rew.max()), 35.0) * 0.85
            ax2.fill_between(t, 0, kde_scaled, facecolor="#6366f1", alpha=0.26, zorder=2)
            ax2.plot(
                t,
                kde_scaled,
                color="#4338ca",
                lw=1.7,
                ls=":",
                zorder=3,
                label="Source Prior Density (scaled KDE)",
            )
        ax2.plot(
            t,
            p_rew,
            color="#0f766e",
            lw=2.25,
            zorder=4,
            label=f"Clean+Relaxed Hit Yield $P_{{\\mathrm{{hit}}}}(\\pm {tol:.2f}\\,\\mathrm{{{unit}}})$",
        )
        ax2.plot(
            t,
            p_dft,
            color="#dc2626",
            lw=1.95,
            ls="-.",
            zorder=5,
            label="Calibrated Hit Probability (with $\\sigma_{\\mathrm{cal}}$)",
        )
        ax2.set_xlim(v_min, v_max)
        ax2.xaxis.set_major_locator(MultipleLocator(1.0))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax2.set_ylim(0, max(float(p_rew.max()) * 1.42, 55.0))
        ax2.set_box_aspect(1.0 / 1.28)
        ax2.set_xlabel(f"Requested {axis_name} Target $t$ ({unit})")
        ax2.set_ylabel("Clean Hit Probability / Scaled Prior (%)")
        ax2.set_title(f"B. {axis_name} Reachability & Source Prior Density", loc="left", pad=8)
        ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.94, fontsize=8.0)
        ax2.grid(alpha=0.22)

        fig.savefig(out_dir / f"fig_gp_atlas_1d_{tag}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"fig_gp_atlas_1d_{tag}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if j2 is not None:
        raw_axes = j2.get("axis_names", ["HOMO", "LUMO"])
        ax_x_name = raw_axes[0].upper() if str(raw_axes[0]).lower() in ("homo", "lumo") else str(raw_axes[0])
        ax_y_name = raw_axes[1].upper() if str(raw_axes[1]).lower() in ("homo", "lumo") else str(raw_axes[1])
        units = j2.get("units", ["eV", "eV"])
        ux, uy = str(units[0]), str(units[1])
        unphys_rule = j2.get("unphysical_rule", "x>=y" if (ax_x_name, ax_y_name) == ("HOMO", "LUMO") else None)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 5.0))
        h_g = np.asarray(j2.get("axis_x_grid", j2["homo_grid_eV"]))
        l_g = np.asarray(j2.get("axis_y_grid", j2["lumo_grid_eV"]))
        p_j = np.asarray(j2["p_joint_hit_reward"]).T * 100.0
        p_d = np.asarray(j2["p_joint_hit_dft_calibrated"]).T * 100.0
        kde_2d = np.asarray(j2["source_kde_density"]).T

        im1 = ax_l.contourf(h_g, l_g, p_j, levels=16, cmap="viridis")
        legend_handles = []
        if kde_2d.max() > 0:
            ax_l.contour(h_g, l_g, kde_2d, levels=5, colors="white", linewidths=0.9, alpha=0.78)
            legend_handles.append(Line2D([0], [0], color="white", lw=1.1, label="Source Prior Density"))

        hh_m, ll_m = np.meshgrid(h_g, l_g, indexing="xy")
        if "drift_x" in j2 and "drift_y" in j2:
            dh = np.asarray(j2["drift_x"]).T
            dl = np.asarray(j2["drift_y"]).T
        elif "drift_homo_eV" in j2 and "drift_lumo_eV" in j2:
            dh = np.asarray(j2["drift_homo_eV"]).T
            dl = np.asarray(j2["drift_lumo_eV"]).T
        elif "mu_homo_eV" in j2 and "mu_lumo_eV" in j2:
            dh = np.asarray(j2["mu_homo_eV"]).T - hh_m
            dl = np.asarray(j2["mu_lumo_eV"]).T - ll_m
        else:
            dh = dl = None

        if dh is not None and dl is not None:
            step_q = max(len(h_g) // 8, 1)
            sl_q = (slice(step_q // 2, None, step_q), slice(step_q // 2, None, step_q))
            hq, lq = hh_m[sl_q], ll_m[sl_q]
            uh, vl = dh[sl_q], dl[sl_q]
            phys_q = (hq < (lq - 0.35)) if unphys_rule in ("x>=y", "x>y") else np.ones_like(hq, dtype=bool)
            ax_l.quiver(
                hq[phys_q],
                lq[phys_q],
                uh[phys_q],
                vl[phys_q],
                color="#ffd166",
                angles="xy",
                scale_units="xy",
                scale=2.2,
                width=0.0044,
                headwidth=3.8,
                alpha=0.92,
                zorder=4,
            )
            legend_handles.append(Line2D([0], [0], color="#ffd166", lw=1.6, label="Generative Drift $\\hat{\\mu}(t) - t$"))

        if unphys_rule in ("x>=y", "x>y") and h_g.max() >= l_g.min():
            ax_l.plot(h_g, h_g, "r--", lw=1.2)
            legend_handles.append(Line2D([0], [0], color="red", ls="--", lw=1.2, label=f"Forbidden ${ax_x_name} \\geq {ax_y_name}$"))
        ax_l.set_xlim(h_g.min(), h_g.max())
        ax_l.set_ylim(l_g.min(), l_g.max())
        ax_l.set_xlabel(f"Requested {ax_x_name} ({ux})")
        ax_l.set_ylabel(f"Requested {ax_y_name} ({uy})")
        ax_l.set_title("A. Joint Clean Hit Yield (%) & Generative Drift Field")
        if legend_handles:
            ax_l.legend(
                handles=legend_handles,
                loc="lower right",
                frameon=True,
                facecolor="#111827",
                labelcolor="white",
                framealpha=0.68,
                fontsize=8.0,
            )
        cb1 = fig.colorbar(im1, ax=ax_l, fraction=0.046, pad=0.04)
        cb1.set_label("Clean+Relaxed Hit Yield (%)")

        if "sigma_cal_2d" in j2:
            cal_2d = np.asarray(j2["sigma_cal_2d"]).T
        elif "sigma_cal_2d_eV" in j2:
            cal_2d = np.asarray(j2["sigma_cal_2d_eV"]).T
        else:
            cal_2d = np.sqrt(0.5 * (0.19 ** 2 + 0.21 ** 2)) * np.ones_like(p_d)

        c_lo = float(np.nanmin(cal_2d))
        c_hi = float(np.nanmax(cal_2d))
        if c_hi - c_lo < 0.04:
            c_lo = max(0.05, c_lo - 0.03)
            c_hi = c_hi + 0.03
        cal_levels = np.linspace(c_lo, c_hi, 17)
        im2 = ax_r.contourf(h_g, l_g, cal_2d, levels=cal_levels, cmap="YlOrRd")
        right_handles = []
        p_d_max = float(np.nanmax(p_d))
        if p_d_max >= 2.0:
            raw_levels = [lv for lv in [3.0, 6.0, 10.0, 14.0, 18.0, 24.0] if lv < p_d_max * 0.94]
            use_dec = False
            if len(raw_levels) < 3:
                raw_levels = list(np.round(np.linspace(max(1.2, 0.30 * p_d_max), 0.86 * p_d_max, 4), 1))
                use_dec = True
            cs_cal = ax_r.contour(h_g, l_g, p_d, levels=raw_levels, colors="#0b2545", linestyles="--", linewidths=1.35)
            ax_r.clabel(cs_cal, inline=True, fmt="%.1f%%" if use_dec else "%.0f%%", fontsize=7.8)
            right_handles.append(Line2D([0], [0], color="#0b2545", ls="--", lw=1.35, label="Calibrated Hit Prob (%)"))

        emp_cells = j2.get("empirical_cells") or []
        if emp_cells:
            eh = [c.get("target_x", c["homo_target_eV"]) for c in emp_cells if c.get("evaluated", 0) >= 2]
            el = [c.get("target_y", c["lumo_target_eV"]) for c in emp_cells if c.get("evaluated", 0) >= 2]
            if eh:
                ax_r.scatter(eh, el, s=18, facecolors="white", edgecolors="#1f2937", linewidths=0.75, alpha=0.80, zorder=5)
                right_handles.append(
                    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#1f2937", markersize=4.8, label="Audited 2D Grid Cells")
                )

        if unphys_rule in ("x>=y", "x>y") and h_g.max() >= l_g.min():
            ax_r.plot(h_g, h_g, "k:", lw=1.2)
        ax_r.set_xlim(h_g.min(), h_g.max())
        ax_r.set_ylim(l_g.min(), l_g.max())
        ax_r.set_xlabel(f"Requested {ax_x_name} ({ux})")
        ax_r.set_ylabel(f"Requested {ax_y_name} ({uy})")
        ax_r.set_title(f"B. Predictor Calibration Uncertainty $\\sigma_{{\\mathrm{{cal,2D}}}}$ ({ux})")
        if right_handles:
            ax_r.legend(
                handles=right_handles,
                loc="lower right",
                frameon=True,
                facecolor="white",
                edgecolor="#d1d5db",
                framealpha=0.86,
                fontsize=8.0,
            )
        cb2 = fig.colorbar(im2, ax=ax_r, fraction=0.046, pad=0.04, format="%.2f")
        cb2.set_label(f"Calibration Uncertainty $\\sigma_{{\\mathrm{{cal,2D}}}}$ ({ux})")

        fig.tight_layout()
        fig.savefig(out_dir / "fig_gp_atlas_2d_joint.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig_gp_atlas_2d_joint.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_stage_progression(summary_json: Path, out_dir: Path) -> None:
    _style()
    data = json.loads(summary_json.read_text())
    stages = data["stages"]
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [s["label"] for s in stages]
    x = np.arange(len(names))
    hit_yield = [s["hit_yield"] * 100.0 for s in stages]
    slopes = [s["calibration_slope"] for s in stages]
    sigmas = [s["within_target_std_eV"] for s in stages]
    rew_mae = [s["conditional_mae_eV"] for s in stages]
    aud_mae = [s.get("audit_mae_eV") for s in stages]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    colors = ["#8d99ae" if "Legacy" in n else ("#1d3557" if "v1.1" in n else "#2a9d8f") for n in names]
    bars = axes[0].bar(x, hit_yield, color=colors, width=0.58)
    for b, v in zip(bars, hit_yield):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=22, ha="right")
    axes[0].set_ylabel("End-to-End Hit Yield $\\pm 0.25\\,\\mathrm{eV}$ (%)")
    axes[0].set_title("A. End-to-End Proposal Hit Yield (1,792 Proposals)")
    axes[0].set_ylim(0, max(hit_yield) * 1.22)
    axes[0].grid(axis="y", alpha=0.2)

    ax_b = axes[1]
    ax_b.plot(x, slopes, "o-", color="#1d3557", lw=2.0, label="Calibration Slope (ideal = 1.0)")
    ax_b.axhline(1.0, ls="--", color="#1d3557", alpha=0.4)
    ax_b.plot(x, sigmas, "s-", color="#e63946", lw=2.0, label="Within-Target Spread $\\sigma_{\\mathrm{in}}$ (eV)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(names, rotation=22, ha="right")
    ax_b.set_ylabel("Slope / Spread (eV)")
    ax_b.set_title("B. Calibration Slope & Concentration")
    ax_b.legend(frameon=False, loc="best")
    ax_b.grid(alpha=0.2)

    ax_c = axes[2]
    ax_c.plot(x, rew_mae, "o-", color="#2a9d8f", lw=2.0, label="RewardOracle (Uni-Mol2 1.1B) MAE")
    valid_aud_x = [i for i, v in enumerate(aud_mae) if v is not None and np.isfinite(v)]
    valid_aud_y = [aud_mae[i] for i in valid_aud_x]
    if valid_aud_x:
        ax_c.plot(valid_aud_x, valid_aud_y, "D--", color="#e76f51", lw=2.0, label="AuditOracle (Calibrated DPA-4) MAE")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(names, rotation=22, ha="right")
    ax_c.set_ylabel("Conditional MAE vs Target (eV)")
    ax_c.set_title("C. Cross-Foundation Audit (No Reward Hacking)")
    ax_c.legend(frameon=False, loc="best")
    ax_c.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_loom_stage_progression.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_loom_stage_progression.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_anti_reward_hacking_and_hotswap(curve_json: Path, out_dir: Path) -> None:
    _style()
    data = json.loads(curve_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = data["dpa4_budget_curve"]
    groups = [p["groups"] for p in pts]
    total_hits = [p["hit_yield"] * 100.0 for p in pts]
    clean_hits = [p.get("clean_hit_yield", p["hit_yield"] * (1.0 - p.get("exotic_charge_rate", 0.0))) * 100.0 for p in pts]
    rew_mae = [p["conditional_mae_eV"] for p in pts]
    aud_relax_mae = [p["post_relax_audit_mae_eV"] for p in pts]
    aud_raw_mae = [p["raw_audit_mae_eV"] for p in pts]
    exotic_q = [p["exotic_charge_rate"] * 100.0 for p in pts]
    fmax = [p["f_max_median_eV_A"] for p in pts]
    hyff = data.get("hyff_hotswap_512")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))
    ax0 = axes[0]
    ax0.plot(groups, total_hits, "o-", color="#1d3557", lw=2.1, label="All-Legal Hit Yield $\\pm 0.25\\,\\mathrm{eV}$")
    ax0.plot(groups, clean_hits, "s-", color="#2a9d8f", lw=2.1, label="Zwitterion-Clean Hit Yield")
    if hyff is not None:
        ax0.scatter([hyff["groups"]], [hyff["hit_yield"] * 100.0], marker="*", s=160, color="#e63946", zorder=6, label="HyFF Hot-Swap (All-Legal)")
        if "clean_hit_yield" in hyff:
            ax0.scatter([hyff["groups"]], [hyff["clean_hit_yield"] * 100.0], marker="P", s=95, color="#f4a261", zorder=6, label="HyFF Hot-Swap (Clean)")
    ax0.set_xlabel("Disentangled GRPO Groups")
    ax0.set_ylabel("Proposal Hit Yield (%)")
    ax0.set_title("A. Hit Yield Trajectory & Potential Hot-Swap")
    ax0.legend(frameon=False, loc="best")
    ax0.grid(alpha=0.2)

    ax1 = axes[1]
    ax1.plot(groups, rew_mae, "o-", color="#2a9d8f", lw=2.1, label="RewardOracle (Uni-Mol2 1.1B) MAE")
    ax1.plot(groups, aud_relax_mae, "D-", color="#e76f51", lw=2.1, label="AuditOracle (Post-Relax DPA-4) MAE")
    ax1.plot(groups, aud_raw_mae, "^--", color="#8d99ae", lw=1.6, label="AuditOracle (Zero-Step Raw) MAE")
    if hyff is not None and hyff.get("post_relax_audit_mae_eV") is not None:
        ax1.scatter([hyff["groups"]], [hyff["post_relax_audit_mae_eV"]], marker="*", s=150, color="#e63946", zorder=6, label="HyFF Hot-Swap Audit MAE")
    ax1.set_xlabel("Disentangled GRPO Groups")
    ax1.set_ylabel("Conditional MAE vs Target (eV)")
    ax1.set_ylim(0.26, max(max(aud_raw_mae) * 1.22, 1.75))
    ax1.set_title("B. Reward vs Independent Audit MAE")
    ax1.legend(frameon=False, loc="upper right", fontsize=8.0)
    ax1.grid(alpha=0.2)

    ax2 = axes[2]
    ax2_r = ax2.twinx()
    ax2.spines["right"].set_visible(True)
    l1 = ax2.plot(groups, exotic_q, "o-", color="#e63946", lw=2.1, label="Exotic Zwitterion Rate (%)")
    l2 = ax2_r.plot(groups, fmax, "s--", color="#457b9d", lw=2.0, label="Zero-Step Median $F_{\\max}$ (eV/$\\mathrm{\\AA}$)")
    extra_handles = []
    if hyff is not None:
        h_q = ax2.scatter([hyff["groups"]], [hyff["exotic_charge_rate"] * 100.0], marker="*", s=150, color="#9d0208", zorder=6, label="HyFF Exotic Charge (%)")
        h_f = ax2_r.scatter([hyff["groups"]], [hyff["f_max_median_eV_A"]], marker="P", s=95, color="#1d3557", zorder=6, label="HyFF $F_{\\max}$")
        extra_handles = [h_q, h_f]
    ax2.set_xlabel("Disentangled GRPO Groups")
    ax2.set_ylabel("Exotic Formal-Charge Rate (%)", color="#e63946")
    ax2_r.set_ylabel("Zero-Step Median $F_{\\max}$ (eV/$\\mathrm{\\AA}$)", color="#457b9d")
    ax2.set_title("C. Physics & Charge-Grammar Internalization")
    lines = l1 + l2 + extra_handles
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, frameon=False, loc="upper right", fontsize=8.0)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_anti_reward_hacking_and_hotswap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_anti_reward_hacking_and_hotswap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_v20_zwitterion_and_synthesizability(v20_json: Path, out_dir: Path) -> None:
    """Render `fig_loom_v2_zwitterion_and_synthesizability.{pdf,png}` comparing Raw vs Valence-Reconciled tracks."""
    _style()
    data = json.loads(v20_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_raw = data["pre_alignment_v11"]["raw_point_cloud"]
    pre_rec = data["pre_alignment_v11"]["valence_reconciled"]
    post_raw = data["post_alignment_v20"]["raw_point_cloud"]
    post_rec = data["post_alignment_v20"]["valence_reconciled"]

    # Historical v1.1 reference from progression summary (1,792 proposals at cfg=2.0)
    v11_hist_exotic = 25.67
    v11_hist_legal = 83.48
    v11_hist_clean_yield = 25.17

    stages = [
        "v1.1 Ref\n(Raw)",
        "v2.0 Guard\n(Raw)",
        "v2.0 Aligned\n(Raw)",
        "v2.0 Guard\n(+Reconciler)",
        "v2.0 Aligned\n(+Reconciler)",
    ]
    x = np.arange(len(stages))

    exotic_vals = [
        v11_hist_exotic,
        float(pre_raw.get("unseen_formal_charge_rate_among_legal") or 0.0) * 100.0,
        float(post_raw.get("unseen_formal_charge_rate_among_legal") or 0.0) * 100.0,
        float(pre_rec.get("unseen_formal_charge_rate_among_legal") or 0.0) * 100.0,
        float(post_rec.get("unseen_formal_charge_rate_among_legal") or 0.0) * 100.0,
    ]
    legal_vals = [
        v11_hist_legal,
        pre_raw["legal_rate"] * 100.0,
        post_raw["legal_rate"] * 100.0,
        pre_rec["legal_rate"] * 100.0,
        post_rec["legal_rate"] * 100.0,
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))

    # Panel A: Exotic Zwitterion Elimination Ladder
    cols_a = ["#8d99ae", "#457b9d", "#1d3557", "#2a9d8f", "#1b7f70"]
    bars_a = axes[0].bar(x, exotic_vals, color=cols_a, width=0.58)
    for b, v in zip(bars_a, exotic_vals):
        y_txt = max(v, 1.1) + 0.65
        axes[0].text(b.get_x() + b.get_width() / 2, y_txt, f"{v:.2f}%" if v < 2.0 else f"{v:.1f}%", ha="center", va="bottom", fontsize=8.1)
    axes[0].axhline(1.0, ls="--", color="#e63946", lw=1.2, label="v2.0 Target Ceiling (< 1.0%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages, rotation=18, ha="right")
    axes[0].set_ylabel("Exotic Formal-Charge Rate among Legal (%)")
    axes[0].set_ylim(0, max(exotic_vals) * 1.24)
    axes[0].set_title("A. Dual-Track Zwitterion Elimination")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(axis="y", alpha=0.2)

    # Panel B: Legality (%) & Synthesizability Gate (`SA <= 3.5` & Clean SMARTS)
    x4 = np.arange(4)
    st4 = ["Pre-v2\n(Raw)", "Aligned-v2\n(Raw)", "Pre-v2\n(Reconciled)", "Aligned-v2\n(Reconciled)"]
    leg4 = legal_vals[1:]
    synth4 = [
        pre_raw["constraints_met_rate_among_legal"] * 100.0,
        post_raw["constraints_met_rate_among_legal"] * 100.0,
        pre_rec["constraints_met_rate_among_legal"] * 100.0,
        post_rec["constraints_met_rate_among_legal"] * 100.0,
    ]
    w = 0.36
    b1 = axes[1].bar(x4 - w / 2, leg4, width=w, color="#1d3557", label="3D Covalent Legality (%)")
    b2 = axes[1].bar(x4 + w / 2, synth4, width=w, color="#f4a261", label="Constraints Met (SA $\\leq 3.5$ & Clean) (%)")
    for b, v in zip(b1, leg4):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.8)
    for b, v in zip(b2, synth4):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.8)
    axes[1].set_xticks(x4)
    axes[1].set_xticklabels(st4, rotation=15, ha="right")
    axes[1].set_ylim(0, 122.0)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("B. Legality & Declarative Synthesizability Gate")
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.92, loc="upper left", fontsize=7.6)
    axes[1].grid(axis="y", alpha=0.2)

    # Panel C: Blind Cross-Foundation Audit Hit Yield & Synthesizable Clean Hit Yield (1,792 Proposals)
    audit_y4 = [
        pre_raw["audit"]["audit_hit_yield"] * 100.0,
        post_raw["audit"]["audit_hit_yield"] * 100.0,
        pre_rec["audit"]["audit_hit_yield"] * 100.0,
        post_rec["audit"]["audit_hit_yield"] * 100.0,
    ]
    synth_hit_y4 = [
        pre_raw["synthesizable_clean_hit_yield"] * 100.0,
        post_raw["synthesizable_clean_hit_yield"] * 100.0,
        pre_rec["synthesizable_clean_hit_yield"] * 100.0,
        post_rec["synthesizable_clean_hit_yield"] * 100.0,
    ]
    b3 = axes[2].bar(x4 - w / 2, audit_y4, width=w, color="#457b9d", label="Blind DPA-4 Audit Hit Yield (%)")
    b4 = axes[2].bar(x4 + w / 2, synth_hit_y4, width=w, color="#2a9d8f", label="Synthesizable Clean Hit Yield (SA $\\leq 3.5$) (%)")
    for b, v in zip(b3, audit_y4):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.8)
    for b, v in zip(b4, synth_hit_y4):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.8)
    axes[2].set_xticks(x4)
    axes[2].set_xticklabels(st4, rotation=15, ha="right")
    axes[2].set_ylim(0, max(audit_y4) * 1.34)
    axes[2].set_ylabel("End-to-End Yield over 1,792 Proposals (%)")
    axes[2].set_title("C. Multi-Asset Synthesizable & Blind Audit Yield")
    axes[2].legend(frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.92, loc="upper left", fontsize=7.5)
    axes[2].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_loom_v2_zwitterion_and_synthesizability.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_loom_v2_zwitterion_and_synthesizability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_v20_grounded_rsi_closed_loop(rsi_json: Path, out_dir: Path) -> None:
    """Render `fig_loom_v2_grounded_rsi_closed_loop.{pdf,png}` illustrating the 2-round Grounded RSI loop."""
    _style()
    data = json.loads(rsi_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    r0 = data["round_0_initial"]
    r1 = data["round_1_inner_rsi"]
    r1b = data["round_1b_outer_acquisition_and_calibration"]
    r2 = data["round_2_post_calibration_inner_rsi"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))

    # Panel A: ResidualCalibratedAsset Epistemic Contraction across HOMO Targets
    t0 = np.asarray(r0["atlas_curve"]["targets"])
    s0 = np.asarray(r0["atlas_curve"]["sigma_cal"])
    spost = np.asarray(r1b["post_calibration_atlas_curve"]["sigma_cal"])
    s2 = np.asarray(r2["atlas_curve"]["sigma_cal"])
    cutoff = float(data["blindspot_cutoff_eV"])

    axes[0].axvspan(float(min(t0)) - 0.1, cutoff, color="#e63946", alpha=0.09, label=f"Historical Blindspot ($t \\leq {cutoff:.2f}$ eV)")
    axes[0].plot(t0, s0, "o-", color="#e63946", lw=2.1, label="Round 0 Uncalibrated $\\sigma_{\\mathrm{cal}}(t)$")
    axes[0].plot(t0, spost, "s--", color="#f4a261", lw=2.0, label="Post-Outer-RSI Calibrated $\\sigma_{\\mathrm{cal}}(t)$")
    axes[0].plot(t0, s2, "D-", color="#2a9d8f", lw=2.1, label="Round 2 Final $\\sigma_{\\mathrm{cal}}(t)$")
    axes[0].axhline(0.35, ls=":", color="#1d3557", lw=1.3, label="RSI Trust Cap ($\\sigma_{\\mathrm{cap}} = 0.35$ eV)")

    cal_bt = r1b.get("residual_calibration_blind_test", {})
    pre_mae = (cal_bt.get("pre_calibration_blindspot_mae") or [None])[0]
    post_mae = (cal_bt.get("post_calibration_blindspot_mae") or [None])[0]
    if pre_mae is not None and post_mae is not None:
        axes[0].text(
            0.03,
            0.05,
            f"Blind DFT Deep-Test MAE:\n{pre_mae:.3f} $\\rightarrow$ {post_mae:.3f} eV (-33.8%)\n(0 Base Weights Retrained)",
            transform=axes[0].transAxes,
            va="bottom",
            ha="left",
            fontsize=7.9,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#d1d5db", alpha=0.94),
        )
    axes[0].set_xlim(float(min(t0)) - 0.08, float(max(t0)) + 0.08)
    axes[0].set_ylim(0.15, max(float(np.max(s0)) * 1.14, 0.55))
    axes[0].set_xlabel("Requested HOMO Target $t$ (eV)")
    axes[0].set_ylabel("Calibration Uncertainty $\\sigma_{\\mathrm{cal}}(t)$ (eV)")
    axes[0].set_title("A. Zero-Retrain Blindspot Calibration")
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.93, loc="upper right", fontsize=7.4)
    axes[0].grid(alpha=0.2)

    # Panel B: 4-Zone Causal State Machine Transition Map across Rounds
    zone_order = ["PRIOR_CEILING", "OUTER_BLINDSPOT", "INNER_UNDEREXPLORED", "CONVERGED"]
    zone_colors = {
        "PRIOR_CEILING": "#6c757d",
        "OUTER_BLINDSPOT": "#e63946",
        "INNER_UNDEREXPLORED": "#f4a261",
        "CONVERGED": "#2a9d8f",
    }
    zone_short = {
        "PRIOR_CEILING": "Prior Ceiling",
        "OUTER_BLINDSPOT": "Outer Blindspot (RL=0%)",
        "INNER_UNDEREXPLORED": "Inner Underexplored",
        "CONVERGED": "Converged",
    }
    diag_stages: List[Dict[str, Any]] = [
        r0["diagnosis"],
        r1["diagnosis"],
        r1b["post_calibration_diagnosis"],
        r2["diagnosis"],
    ]
    stage_names = ["R0\nInitial", "R1A\nInner RSI", "R1B\nPost-DFT Cal", "R2\nFinal RSI"]
    xb = np.arange(len(stage_names))
    bottoms = np.zeros(len(stage_names))
    for z_key in zone_order:
        counts = np.array([d["zone_counts"].get(z_key, 0) for d in diag_stages], dtype=np.float64)
        axes[1].bar(xb, counts, bottom=bottoms, width=0.52, color=zone_colors[z_key], label=zone_short[z_key])
        for i_s, c_val in enumerate(counts):
            if c_val >= 1:
                axes[1].text(xb[i_s], bottoms[i_s] + c_val / 2.0, f"{int(c_val)}", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        bottoms += counts
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(stage_names)
    axes[1].set_ylabel("Evaluated Grid Intervals (Count)")
    axes[1].set_ylim(0, max(bottoms) * 1.32)
    axes[1].set_title("B. Causal Weak-Zone State Unlocking")
    axes[1].legend(frameon=False, loc="upper right", fontsize=7.6)
    axes[1].grid(axis="y", alpha=0.2)

    # Panel C: Strictly Blind Cross-Foundation AuditOracle (`E15 DPA-4`) Clean Synthesizable Hit Yield
    r_labels = ["Round 0\n(Stage 1 Init)", "Round 1A\n(Inner RSI Only)", "Round 2\n(Outer DFT + Inner RSI)"]
    xr = np.arange(3)
    tot_aud = [
        r0["metrics"]["blind_audit_constrained_clean_hit_yield"] * 100.0,
        r1["metrics"]["blind_audit_constrained_clean_hit_yield"] * 100.0,
        r2["metrics"]["blind_audit_constrained_clean_hit_yield"] * 100.0,
    ]
    mid_aud = [
        r0["metrics"]["mid_underexplored_audit_clean_hit_yield"] * 100.0,
        r1["metrics"]["mid_underexplored_audit_clean_hit_yield"] * 100.0,
        r2["metrics"]["mid_underexplored_audit_clean_hit_yield"] * 100.0,
    ]
    deep_aud = [
        r0["metrics"]["deep_blindspot_audit_clean_hit_yield"] * 100.0,
        r1["metrics"]["deep_blindspot_audit_clean_hit_yield"] * 100.0,
        r2["metrics"]["deep_blindspot_audit_clean_hit_yield"] * 100.0,
    ]
    w3 = 0.25
    b_tot = axes[2].bar(xr - w3, tot_aud, width=w3, color="#1d3557", label="All Quantiles (Blind DPA-4)")
    b_mid = axes[2].bar(xr, mid_aud, width=w3, color="#457b9d", label="Mid Zone ($P_{35}$–$P_{90}$)")
    b_dep = axes[2].bar(xr + w3, deep_aud, width=w3, color="#2a9d8f", label="Deep Blindspot ($P_{10}$–$P_{25}$)")
    for bars_grp in (b_tot, b_mid, b_dep):
        for b in bars_grp:
            val = b.get_height()
            axes[2].text(b.get_x() + b.get_width() / 2, val + 0.4, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.4)
    axes[2].set_xticks(xr)
    axes[2].set_xticklabels(r_labels)
    axes[2].set_ylim(0, max(tot_aud + mid_aud + deep_aud) * 1.28)
    axes[2].set_ylabel("Blind Audit Constrained Clean Hit Yield (%)")
    axes[2].set_title("C. Blind AuditOracle Progression (1,792 Proposals)")
    axes[2].legend(frameon=False, loc="upper left", fontsize=7.7)
    axes[2].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_loom_v2_grounded_rsi_closed_loop.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_loom_v2_grounded_rsi_closed_loop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-json", type=Path, default=None)
    parser.add_argument("--progression-json", type=Path, default=None)
    parser.add_argument("--audit-curve-json", type=Path, default=None)
    parser.add_argument("--v20-synth-json", type=Path, default=None)
    parser.add_argument("--v20-rsi-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.atlas_json and args.atlas_json.exists():
        plot_gp_atlas(args.atlas_json, args.out_dir)
    if args.progression_json and args.progression_json.exists():
        plot_stage_progression(args.progression_json, args.out_dir)
    if args.audit_curve_json and args.audit_curve_json.exists():
        plot_anti_reward_hacking_and_hotswap(args.audit_curve_json, args.out_dir)
    if args.v20_synth_json and args.v20_synth_json.exists():
        plot_v20_zwitterion_and_synthesizability(args.v20_synth_json, args.out_dir)
    if args.v20_rsi_json and args.v20_rsi_json.exists():
        plot_v20_grounded_rsi_closed_loop(args.v20_rsi_json, args.out_dir)


if __name__ == "__main__":
    main()
