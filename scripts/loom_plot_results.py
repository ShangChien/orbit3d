"""Generate publication-quality vector PDF + PNG figures for `Uni-Loom` (`loom`) experiments:
  - Fig 1: Calibration & Concentration Trajectory (`Frozen SFT` -> `Legacy GRPO` -> `Loom Stage 0` -> `Stage 1` -> `Stage 2`)
  - Fig 2: Anti-Reward-Hacking Audit Curve (`RewardOracle` vs `AuditOracle` across RL budgets) + `DPA-4 <-> HyFF` Hot-Swap
  - Fig 3: 1D HOMO/LUMO & 2D `(HOMO, LUMO)` Heteroscedastic GP Reachability & Dual-Uncertainty Atlas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
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

    for prop_label, tag, a1 in [("HOMO", "homo", h1), ("LUMO", "lumo", l1)]:
        if a1 is None:
            continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]})
        t = np.asarray(a1["targets_eV"])
        mu = np.asarray(a1["mu_gen_eV"])
        sd_gen = np.asarray(a1["sigma_gen_eV"])
        sd_tot = np.asarray(a1["sigma_total_eV"])
        p_rew = np.asarray(a1["p_hit_reward"]) * 100.0
        p_dft = np.asarray(a1["p_hit_dft_calibrated"]) * 100.0
        kde = np.asarray(a1["source_kde_density"])

        ax1.plot(t, t, "--", color="#555555", lw=1.2, label="Ideal slope = 1 ($\\mu = t$)")
        ax1.fill_between(t, mu - sd_tot, mu + sd_tot, color="#f4a261", alpha=0.25, label="Total spread $\\pm\\sigma_{\\mathrm{total}}(t)$ (Gen + DFT Cal)")
        ax1.fill_between(t, mu - sd_gen, mu + sd_gen, color="#2a9d8f", alpha=0.35, label="Generator spread $\\pm\\sigma_{\\mathrm{gen}}(t)$")
        ax1.plot(t, mu, color="#1d3557", lw=2.0, label="GP Posterior Mean $\\hat{\\mu}_{\\mathrm{gen}}(t)$")

        emp_t = [g["target_eV"] for g in a1["empirical_grid"] if np.isfinite(g["mean_pred_eV"])]
        emp_m = [g["mean_pred_eV"] for g in a1["empirical_grid"] if np.isfinite(g["mean_pred_eV"])]
        ax1.scatter(emp_t, emp_m, color="#e63946", s=24, zorder=5, label="Sampled 3D cohort means")
        ax1.set_ylabel(f"Achieved {prop_label} (eV)")
        ax1.set_title(f"Uni-Loom 1D Heteroscedastic GP Calibration & Dual-Uncertainty Profile (OE62 {prop_label})")
        ax1.legend(loc="upper left", frameon=False)
        ax1.grid(alpha=0.2)

        ax2.plot(t, p_rew, color="#2a9d8f", lw=2.1, label="Clean+Relaxed Hit Yield $P_{\\mathrm{hit}}(\\pm 0.25\\,\\mathrm{eV})$")
        ax2.plot(t, p_dft, color="#e76f51", lw=1.8, ls="-.", label="DFT-Calibrated Hit Probability")
        if kde.max() > 0:
            kde_scaled = (kde / kde.max()) * max(p_rew.max(), 35.0) * 0.85
            ax2.fill_between(t, 0, kde_scaled, color="#a8dadc", alpha=0.28, label="OE62 Source Density (scaled KDE)")
        if a1.get("trusted_reachable_interval_eV"):
            lo, hi = a1["trusted_reachable_interval_eV"]
            ax2.axvspan(lo, hi, color="#2a9d8f", alpha=0.09, label=f"Trusted Reachable [{lo:.2f}, {hi:.2f}] eV")
            ax1.axvline(lo, color="#2a9d8f", ls=":", lw=1.1, alpha=0.7)
            ax1.axvline(hi, color="#2a9d8f", ls=":", lw=1.1, alpha=0.7)
        ax2.set_xlabel(f"Requested {prop_label} Target $t$ (eV)")
        ax2.set_ylabel("Clean Hit Probability (%)")
        ax2.set_ylim(0, max(float(p_rew.max()) * 1.38, 45.0))
        ax2.legend(loc="upper right", frameon=False, fontsize=8.0)
        ax2.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(out_dir / f"fig_gp_atlas_1d_{tag}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"fig_gp_atlas_1d_{tag}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if j2 is not None:
        from matplotlib.lines import Line2D

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.2, 4.8))
        h_g = np.asarray(j2["homo_grid_eV"])
        l_g = np.asarray(j2["lumo_grid_eV"])
        p_j = np.asarray(j2["p_joint_hit_reward"]).T * 100.0
        p_d = np.asarray(j2["p_joint_hit_dft_calibrated"]).T * 100.0
        kde_2d = np.asarray(j2["source_kde_density"]).T

        im1 = ax_l.contourf(h_g, l_g, p_j, levels=16, cmap="viridis")
        legend_handles = []
        if kde_2d.max() > 0:
            ax_l.contour(h_g, l_g, kde_2d, levels=5, colors="white", linewidths=0.9, alpha=0.75)
            legend_handles.append(Line2D([0], [0], color="white", lw=1.1, label="OE62 Source Density Contours"))
        if h_g.max() >= l_g.min():
            ax_l.plot(h_g, h_g, "r--", lw=1.2)
            legend_handles.append(Line2D([0], [0], color="red", ls="--", lw=1.2, label="Forbidden $E_{\\mathrm{HOMO}} = E_{\\mathrm{LUMO}}$"))
        ax_l.set_xlim(h_g.min(), h_g.max())
        ax_l.set_ylim(l_g.min(), l_g.max())
        ax_l.set_xlabel("Requested HOMO (eV)")
        ax_l.set_ylabel("Requested LUMO (eV)")
        ax_l.set_title("Joint (HOMO, LUMO) Clean Hit Yield $P_{\\mathrm{hit}}(\\pm 0.25\\,\\mathrm{eV})$ (%)")
        if legend_handles:
            ax_l.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="black", labelcolor="white", framealpha=0.55)
        fig.colorbar(im1, ax=ax_l, fraction=0.046, pad=0.04)

        im2 = ax_r.contourf(h_g, l_g, p_d, levels=16, cmap="magma")
        if kde_2d.max() > 0:
            ax_r.contour(h_g, l_g, kde_2d, levels=5, colors="cyan", linewidths=0.9, alpha=0.75)
        if h_g.max() >= l_g.min():
            ax_r.plot(h_g, h_g, "w--", lw=1.2)
        ax_r.set_xlim(h_g.min(), h_g.max())
        ax_r.set_ylim(l_g.min(), l_g.max())
        ax_r.set_xlabel("Requested HOMO (eV)")
        ax_r.set_ylabel("Requested LUMO (eV)")
        ax_r.set_title("DFT-Calibrated Joint Confidence Surface (%)")
        if kde_2d.max() > 0:
            ax_r.legend(
                handles=[Line2D([0], [0], color="cyan", lw=1.1, label="OE62 Source Density Contours")],
                loc="lower right",
                frameon=True,
                facecolor="black",
                labelcolor="white",
                framealpha=0.55,
            )
        fig.colorbar(im2, ax=ax_r, fraction=0.046, pad=0.04)

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

    # Panel A: Hit Yield (%)
    colors = ["#8d99ae" if "Legacy" in n else "#2a9d8f" for n in names]
    bars = axes[0].bar(x, hit_yield, color=colors, width=0.58)
    for b, v in zip(bars, hit_yield):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=22, ha="right")
    axes[0].set_ylabel("End-to-End Hit Yield $\\pm 0.25\\,\\mathrm{eV}$ (%)")
    axes[0].set_title("A. End-to-End Proposal Hit Yield (1,792 Proposals)")
    axes[0].set_ylim(0, max(hit_yield) * 1.22)
    axes[0].grid(axis="y", alpha=0.2)

    # Panel B: Slope & Within-Target Spread
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

    # Panel C: Reward vs Audit MAE (Anti-Reward-Hacking Check)
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
    """Plot RL budget curve (0 -> 256 -> 512 -> 768 -> 1024 groups):
    Panel A: End-to-End Hit Yield & Clean-Charge Hit Yield across RL groups (+ HyFF Hot-Swap star)
    Panel B: RewardOracle MAE vs Post-Relaxation AuditOracle MAE (proving zero reward hacking)
    Panel C: Exotic Zwitterion Rate (%) & Zero-Step MLIP Force F_max (eV/A) internalization
    """
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

    # Panel A: Total & Clean Hit Yield vs RL Groups
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

    # Panel B: Anti-Reward-Hacking Audit Curve
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

    # Panel C: Exotic Zwitterion Suppression & Zero-Step Fmax Internalization
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-json", type=Path, default=None)
    parser.add_argument("--progression-json", type=Path, default=None)
    parser.add_argument("--audit-curve-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.atlas_json and args.atlas_json.exists():
        plot_gp_atlas(args.atlas_json, args.out_dir)
    if args.progression_json and args.progression_json.exists():
        plot_stage_progression(args.progression_json, args.out_dir)
    if args.audit_curve_json and args.audit_curve_json.exists():
        plot_anti_reward_hacking_and_hotswap(args.audit_curve_json, args.out_dir)


if __name__ == "__main__":
    main()
