"""Generate 100% Standalone Zero-Loom Publication Figures 2, 3, and 4 for `Uni-3DAR-Mol`.

Reads ONLY standalone single-output (`M_out = 1`, `0` external predictors) experiment records:
  - `docs/experiments/qm9_uni3dar_mol_standalone.json`
  - `docs/experiments/oe62_uni3dar_mol_standalone.json`
  - `docs/experiments/oe62_pmf_{3h,12h,52h}_fair_30atom.json` (`tracks["native_graph"]["M1"]`)
  - `docs/experiments/oe62_atom_size_buckets_scaling.json` (`PMF_52h_M1`)
  - `docs/experiments/oe62_head2head_dft_anchor_52gpuh_awr_unbiased.json` (`PropMolFlow_native_M1`)
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
EXP_DIR = ROOT / "docs" / "experiments"
FIG_UNI_DIR = ROOT / "paper" / "uni3dar_mol" / "figures"
FIG_DOC_DIR = ROOT / "docs" / "figures"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((EXP_DIR / name).read_text())


def _style_axes(axes) -> None:
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.tick_params(colors="#475569")
        ax.grid(True, linestyle="--", linewidth=0.65, color="#E4ECE9", alpha=0.85)


def _save_both(fig: plt.Figure, stem: str, extra_png: Path | None = None) -> None:
    for base in (FIG_UNI_DIR, FIG_DOC_DIR):
        base.mkdir(parents=True, exist_ok=True)
        fig.savefig(base / f"{stem}.png", bbox_inches="tight")
        fig.savefig(base / f"{stem}.pdf", bbox_inches="tight")
        print(f"[Plot SAVED] {base / stem}.pdf/.png")
    if extra_png is not None:
        extra_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra_png, bbox_inches="tight")
        print(f"[Plot SAVED] {extra_png}")


def render_fig2_qm9_and_oe62_scaling(artifact_dir: Path | None = None) -> None:
    qm9 = _load("qm9_uni3dar_mol_standalone.json")
    oe62 = _load("oe62_uni3dar_mol_standalone.json")
    pmf_3h = _load("oe62_pmf_3h_fair_30atom.json")["tracks"]["native_graph"]["M1"]
    pmf_12h = _load("oe62_pmf_12h_fair_30atom.json")["tracks"]["native_graph"]["M1"]
    pmf_52h = _load("oe62_pmf_52h_fair_30atom.json")["tracks"]["native_graph"]["M1"]
    pmf_dft = _load("oe62_head2head_dft_anchor_52gpuh_awr_unbiased.json")["arms"]["PropMolFlow_native_M1"]

    fig, axes = plt.subplots(1, 3, figsize=(19.8, 6.2), dpi=220)
    fig.patch.set_facecolor("#FAFCFB")
    _style_axes(axes)

    # =========================================================================
    # Panel A: Disjoint 50k/50k QM9 6-Property Conditional Scaling (Normalized)
    # =========================================================================
    ax0 = axes[0]
    props = ["alpha", "gap", "homo", "lumo", "mu", "cv"]
    prop_lbls = ["\u03b1\n(Bohr\u00b3)", "\u0394\u03b5\n(meV)", "\u03b5_HOMO\n(meV)", "\u03b5_LUMO\n(meV)", "\u03bc\n(D)", "C_v\n(cal/mol\u00b7K)"]
    pub = qm9["published_baselines"]
    pmf_ref = np.array([float(pub["PropMolFlow (6 models)"][p]) for p in props], dtype=float)
    geo_ref = np.array([float(pub["GeoLDM (6 models)"][p]) for p in props], dtype=float) / pmf_ref

    tier_specs = [
        ("GeoLDM (6\u00d7 M=1)", geo_ref, "#94A3B8"),
        ("Vanilla Uni-3DAR (1\u21921\u21921)", np.array([float(qm9["per_property"][p]["tiers"]["Vanilla_M1_Raw"]["gvp"]["mae"]) for p in props]) / pmf_ref, "#F59E0B"),
        ("Uni-3DAR-Mol (1\u21921\u21921)", np.array([float(qm9["per_property"][p]["tiers"]["K1"]["gvp"]["mae"]) for p in props]) / pmf_ref, "#5EEAD4"),
        ("Uni-3DAR-Mol (20\u21926\u21921)", np.array([float(qm9["per_property"][p]["tiers"]["K4"]["gvp"]["mae"]) for p in props]) / pmf_ref, "#14B8A6"),
        ("Uni-3DAR-Mol (38\u219212\u21921)", np.array([float(qm9["per_property"][p]["tiers"]["K8"]["gvp"]["mae"]) for p in props]) / pmf_ref, "#0F766E"),
        ("Uni-3DAR-Mol (74\u219222\u21921)", np.array([float(qm9["per_property"][p]["tiers"]["K16"]["gvp"]["mae"]) for p in props]) / pmf_ref, "#064E3B"),
    ]
    xp = np.arange(len(props))
    wb = 0.132
    ax0.axhline(1.0, color="#E11D48", ls="--", lw=1.35, zorder=2, label="6\u00d7 PropMolFlow [t,N] (156 GPU-h, M=1 = 1.0\u00d7)")
    for idx_t, (tlbl, tvals, tcol) in enumerate(tier_specs):
        offs = (idx_t - 2.5) * wb
        ax0.bar(xp + offs, np.clip(tvals, 0.0, 5.4), width=wb * 0.93, color=tcol, edgecolor="#1E293B", lw=0.65, label=tlbl, zorder=3)
    ax0.set_xticks(xp)
    ax0.set_xticklabels(prop_lbls, fontsize=8.0, fontweight="bold", color="#1E293B")
    ax0.set_ylim(0.0, 5.65)
    ax0.set_ylabel("GVP Test MAE Normalized to PropMolFlow (\u2193)", fontsize=9.4, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Disjoint 50k/50k QM9 6-Property Scaling (M_out=1)", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax0.legend(loc="upper left", fontsize=6.5, ncol=2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # =========================================================================
    # Panel B: OE62 30.4-Atom Pure HOMO Pareto (Latency vs Clean Hit Rate)
    # =========================================================================
    ax1 = axes[1]
    fair = oe62["fair_30atom_1792"]
    u_keys = ["K1", "K2", "K4", "K8", "K12"]
    u_lat = [float(fair[k]["gpu_sec_per_1k_prompts"]) for k in u_keys]
    u_hit = [float(fair[k]["blind_dpa4_clean_hit_pct"]) for k in u_keys]
    u_mae = [float(fair[k]["blind_dpa4_homo_mae_eV"]) for k in u_keys]
    u_fun = [f"{fair[k]['internal_funnel']['m_stage1']}\u2192{fair[k]['internal_funnel']['m_stage2']}\u21921" for k in u_keys]
    tot_gpuh = float(oe62["uni3dar_mol_training_metadata"]["total_training_gpuh"])

    van = fair["Vanilla_Uni3DAR_M1_Raw"]
    ax1.scatter(
        [float(van["gpu_sec_per_1k_prompts"])],
        [float(van["blind_dpa4_clean_hit_pct"])],
        s=115, marker="^", color="#F59E0B", edgecolor="#1E293B", lw=1.0, zorder=5,
        label=f"Vanilla Uni-3DAR (1\u21921\u21921, {van['blind_dpa4_clean_hit_pct']:.1f}%)",
    )

    pmf_pts = [
        ("PMF 3h", 1000.0 * float(pmf_3h["gen_wall_seconds"]) / 1792.0, 100.0 * float(pmf_3h["blind_dpa4_clean_hits"]) / 1792.0, float(pmf_3h["audit_mae_eV"])),
        ("PMF 12h", 1000.0 * float(pmf_12h["gen_wall_seconds"]) / 1792.0, 100.0 * float(pmf_12h["blind_dpa4_clean_hits"]) / 1792.0, float(pmf_12h["audit_mae_eV"])),
        ("PMF 52h", 1000.0 * float(pmf_52h["gen_wall_seconds"]) / 1792.0, 100.0 * float(pmf_52h["blind_dpa4_clean_hits"]) / 1792.0, float(pmf_52h["audit_mae_eV"])),
    ]
    ax1.plot([p[1] for p in pmf_pts], [p[2] for p in pmf_pts], ":D", color="#E11D48", lw=1.8, ms=7.2, zorder=4, label="PropMolFlow (3h / 12h / 52h, M=1)")
    for plbl, px, py, pmae in pmf_pts:
        if plbl == "PMF 12h":
            tx, ty = px * 0.34, py + 1.2
        elif plbl == "PMF 52h":
            tx, ty = px * 0.56, py + 1.9
        else:
            tx, ty = px + 11.0, py - 3.2
        ax1.annotate(
            f"{plbl}: {py:.1f}% ({pmae:.2f}eV)",
            xy=(px, py), xytext=(tx, ty),
            fontsize=6.8, fontweight="bold", color="#BE123C",
        )

    ax1.plot(u_lat, u_hit, "-o", color="#0F766E", lw=2.7, ms=7.8, zorder=6, label=f"Uni-3DAR-Mol ({tot_gpuh:.2f} GPU-h, K\u2081\u2192K\u2082\u21921)")
    offsets_u = [(5, -3.4), (24, -3.6), (48, -4.0), (-260, 2.3), (-210, -5.2)]
    for (flbl, lx, hy, hm), (dx, dy) in zip(zip(u_fun, u_lat, u_hit, u_mae), offsets_u):
        ax1.annotate(
            f"{flbl}\n{hy:.1f}% ({hm:.2f}eV)",
            xy=(lx, hy),
            xytext=(max(5.0, lx + dx), hy + dy),
            fontsize=6.7,
            fontweight="bold",
            color="#064E3B",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#F0FDFA", edgecolor="#0D9488", lw=0.65, alpha=0.92),
        )

    ax1.set_xscale("log")
    ax1.set_xlim(min(min(u_lat), float(van["gpu_sec_per_1k_prompts"])) * 0.68, max(u_lat) * 1.68)
    y_top = max(max(u_hit), pmf_pts[-1][2]) + 11.0
    ax1.set_ylim(0.0, y_top)
    ax1.set_xlabel("Single-GPU Generation Latency per 1,000 Prompts (A800 GPU-s, log)", fontsize=9.2, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Blind DPA-4 Clean HOMO \u00b10.25 eV Hit Rate (%)", fontsize=9.4, fontweight="bold", color="#1E293B")
    ax1.set_title("B. OE62 30.4-Atom Pure HOMO Pareto (N=1,792, M_out=1)", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="lower right", fontsize=6.9, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # =========================================================================
    # Panel C: Unbiased 42-Prompt Live GPU4PySCF PBE0/def2-TZVP Verification
    # =========================================================================
    ax2 = axes[2]
    dft_arms = oe62["unbiased_42_pbe0"]
    pmf_raw_hit_pct = round(100.0 * float(pmf_dft["pbe0_vs_target_homo"]["n"]) * float(pmf_dft["pbe0_vs_target_homo"]["hit_rate_025eV_raw"]) / 42.0, 2)
    pmf_cen_hit_pct = round(100.0 * float(pmf_dft["pbe0_vs_target_homo"]["n"]) * float(pmf_dft["pbe0_vs_target_homo"]["hit_rate_025eV_centered"]) / 42.0, 2)
    pmf_scf_pct = round(100.0 * float(pmf_dft["n_scf_converged"]) / 42.0, 2)
    pmf_mae = float(pmf_dft["pbe0_vs_target_homo"]["mae_eV"])

    entries_c = [
        ("PropMolFlow\n52h (M=1)", pmf_scf_pct, pmf_raw_hit_pct, pmf_cen_hit_pct, pmf_mae, "#E11D48"),
    ]
    palette_u = ["#14B8A6", "#065F46", "#0F766E"]
    for idx_d, (ak, av) in enumerate(dft_arms.items()):
        short_lbl = "Uni-3DAR-Mol\n(" + ak.split("_")[-1].replace("to", "\u2192") + ")"
        entries_c.append(
            (
                short_lbl,
                round(100.0 * float(av["n_scf_converged"]) / 42.0, 2),
                float(av["end_to_end_pbe0_clean_hit_pct_raw"]),
                float(av["end_to_end_pbe0_clean_hit_pct_centered"]),
                float(av["pbe0_vs_target_homo"]["mae_eV"]),
                palette_u[idx_d % len(palette_u)],
            )
        )

    xc = np.arange(len(entries_c))
    wc = 0.24
    b_scf = ax2.bar(xc - wc, [e[1] for e in entries_c], width=wc * 0.94, color="#CBD5E1", edgecolor="#334155", lw=0.8, label="Valid Closed-Shell SCF Yield (%)")
    b_raw = ax2.bar(xc, [e[2] for e in entries_c], width=wc * 0.94, color=[e[5] for e in entries_c], edgecolor="#1E293B", lw=0.85, label="PBE0 \u00b10.25 eV Hit Rate (Raw %)")
    b_cen = ax2.bar(xc + wc, [e[3] for e in entries_c], width=wc * 0.94, color="#CCFBF1", edgecolor="#0F766E", lw=0.85, hatch="//", label="PBE0 \u00b10.25 eV Hit Rate (Centered %)")

    for br, e_row in zip(b_raw, entries_c):
        ax2.text(
            br.get_x() + br.get_width() / 2.0,
            e_row[2] + 1.4,
            f"{e_row[2]:.1f}%\n({e_row[4]:.2f}eV)",
            ha="center", va="bottom", fontsize=6.7, fontweight="bold", color="#0F172A",
        )
    for bc, e_row in zip(b_cen, entries_c):
        ax2.text(
            bc.get_x() + bc.get_width() / 2.0,
            e_row[3] + 1.2,
            f"{e_row[3]:.1f}%",
            ha="center", va="bottom", fontsize=6.7, fontweight="bold", color="#0F766E",
        )

    ax2.set_xticks(xc)
    ax2.set_xticklabels([e[0] for e in entries_c], fontsize=8.0, fontweight="bold", color="#1E293B")
    ax2.set_ylim(0.0, 114.0)
    ax2.set_ylabel("End-to-End Yield over 42 Prompts (%)", fontsize=9.4, fontweight="bold", color="#1E293B")
    ax2.set_title("C. Live GPU4PySCF PBE0/def2-TZVP Verification (42 Prompts)", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax2.legend(loc="upper left", fontsize=6.9, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    plt.tight_layout(w_pad=1.6)
    extra = (artifact_dir / "fig2_qm9_and_oe62_standalone_scaling.png") if artifact_dir else None
    _save_both(fig, "fig2_qm9_and_oe62_standalone_scaling", extra_png=extra)
    plt.close(fig)


def render_fig3_oe62_atom_scaling(artifact_dir: Path | None = None) -> None:
    oe62 = _load("oe62_uni3dar_mol_standalone.json")
    pmf_b = _load("oe62_atom_size_buckets_scaling.json")["buckets"]

    buckets = [20, 30, 40, 50, 60]
    u_b = oe62["atom_buckets_280"]

    pmf_lat = [1000.0 * float(pmf_b[str(n)]["arms"]["PMF_52h_M1"]["gen_wall_seconds"]) / 280.0 for n in buckets]
    pmf_mae = [float(pmf_b[str(n)]["arms"]["PMF_52h_M1"]["audit_mae_eV"]) for n in buckets]
    pmf_hit = [100.0 * float(pmf_b[str(n)]["arms"]["PMF_52h_M1"]["blind_dpa4_clean_hits"]) / 280.0 for n in buckets]

    k1_lat = [float(u_b[str(n)]["Uni3DAR_Mol_K1"]["gpu_sec_per_1k_prompts"]) for n in buckets]
    k4_lat = [float(u_b[str(n)]["Uni3DAR_Mol_K4"]["gpu_sec_per_1k_prompts"]) for n in buckets]
    k8_lat = [float(u_b[str(n)]["Uni3DAR_Mol_K8"]["gpu_sec_per_1k_prompts"]) for n in buckets]

    k1_mae = [float(u_b[str(n)]["Uni3DAR_Mol_K1"]["blind_dpa4_homo_mae_eV"]) for n in buckets]
    k4_mae = [float(u_b[str(n)]["Uni3DAR_Mol_K4"]["blind_dpa4_homo_mae_eV"]) for n in buckets]
    k8_mae = [float(u_b[str(n)]["Uni3DAR_Mol_K8"]["blind_dpa4_homo_mae_eV"]) for n in buckets]

    k1_hit = [float(u_b[str(n)]["Uni3DAR_Mol_K1"]["blind_dpa4_clean_hit_pct"]) for n in buckets]
    k4_hit = [float(u_b[str(n)]["Uni3DAR_Mol_K4"]["blind_dpa4_clean_hit_pct"]) for n in buckets]
    k8_hit = [float(u_b[str(n)]["Uni3DAR_Mol_K8"]["blind_dpa4_clean_hit_pct"]) for n in buckets]

    fig, axes = plt.subplots(1, 3, figsize=(19.4, 5.8), dpi=220)
    fig.patch.set_facecolor("#FAFCFB")
    _style_axes(axes)

    series = [
        ("PropMolFlow 52h (M=1, 100-Step Flow)", pmf_lat, pmf_mae, pmf_hit, "-D", "#E11D48", 2.4),
        ("Uni-3DAR-Mol (1\u21921\u21921)", k1_lat, k1_mae, k1_hit, "-^", "#14B8A6", 2.3),
        ("Uni-3DAR-Mol (SpecTree K=4: 32\u21928\u21921)", k4_lat, k4_mae, k4_hit, "--s", "#0F766E", 2.4),
        ("Uni-3DAR-Mol (SpecTree K=8: 64\u219216\u21921)", k8_lat, k8_mae, k8_hit, "-o", "#064E3B", 2.7),
    ]

    for lbl, lat_v, mae_v, hit_v, mk, col, lw in series:
        axes[0].plot(buckets, lat_v, mk, color=col, lw=lw, ms=6.8, label=lbl)
        axes[1].plot(buckets, mae_v, mk, color=col, lw=lw, ms=6.8, label=lbl)
        axes[2].plot(buckets, hit_v, mk, color=col, lw=lw, ms=6.8, label=lbl)

    sp_60 = pmf_lat[-1] / max(k1_lat[-1], 1e-6)
    axes[0].annotate(
        f"1\u21921\u21921: {sp_60:.1f}\u00d7 faster\nat N=60 ({k1_lat[-1]:.0f}s vs {pmf_lat[-1]:.0f}s)",
        xy=(60, k1_lat[-1]),
        xytext=(38.5, 13.5),
        fontsize=7.3, fontweight="bold", color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#F0FDFA", edgecolor="#0D9488", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#0D9488", lw=1.0),
    )

    for ax in axes:
        ax.set_xticks(buckets)
        ax.set_xlabel("Target Molecule Size N (Total Explicit Atoms)", fontsize=9.4, fontweight="bold", color="#1E293B")

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Latency per 1,000 Prompts (A800 GPU-s, log)", fontsize=9.4, fontweight="bold", color="#1E293B")
    axes[0].set_title("A. Generation Latency vs. Molecule Size (N=20..60)", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    axes[0].legend(loc="upper left", fontsize=6.9, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    axes[1].set_ylabel("Blind DPA-4 HOMO MAE (eV, lower is better)", fontsize=9.4, fontweight="bold", color="#1E293B")
    axes[1].set_title("B. Blind Orbital Accuracy Across Molecule Sizes", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    axes[1].legend(loc="upper left", fontsize=6.9, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    axes[2].set_ylabel("Blind DPA-4 Clean \u00b10.25 eV Hit Rate (%)", fontsize=9.4, fontweight="bold", color="#1E293B")
    axes[2].set_title("C. Clean Orbital Hit Rate Across Molecule Sizes", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    axes[2].legend(loc="upper right", fontsize=6.9, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    plt.tight_layout(w_pad=1.6)
    extra = (artifact_dir / "fig3_oe62_standalone_atom_scaling.png") if artifact_dir else None
    _save_both(fig, "fig3_oe62_standalone_atom_scaling", extra_png=extra)
    plt.close(fig)


def render_fig4_internal_ablation(artifact_dir: Path | None = None) -> None:
    qm9 = _load("qm9_uni3dar_mol_standalone.json")
    oe62 = _load("oe62_uni3dar_mol_standalone.json")
    abl = oe62["ablation_k4_1792"]
    van = oe62["fair_30atom_1792"]["Vanilla_Uni3DAR_M1_Raw"]

    fig, axes = plt.subplots(1, 3, figsize=(19.6, 5.9), dpi=220)
    fig.patch.set_facecolor("#FAFCFB")
    _style_axes(axes)

    # =========================================================================
    # Panel A: Speculative Octree vs Naive Leaf Decoding (Hit Rate & MAE)
    # =========================================================================
    ax0 = axes[0]
    abl_rows = [
        ("Vanilla\n(1\u21921\u21921 Raw)", van, "#F59E0B"),
        ("NoTree\n1\u21921\u21921 (Off)", abl["NoTree_1to1to1_NoRefiner"], "#94A3B8"),
        ("NoTree\n1\u21921\u21921 (+Ref)", abl["NoTree_1to1to1_WithRefiner"], "#64748B"),
        ("Naive Leaf\n4\u21924\u21921 (Off)", abl["NaiveLeaf_4to4to1_NoRefiner"], "#38BDF8"),
        ("Naive Leaf\n4\u21924\u21921 (+Ref)", abl["NaiveLeaf_4to4to1_WithRefiner"], "#0284C7"),
        ("SpecTree\n32\u21928\u21921 (Off)", abl["SpecTree_32to8to1_NoRefiner"], "#14B8A6"),
        ("SpecTree\n32\u21928\u21921 (+Ref)", abl["Full_Uni3DAR_Mol_32to8to1_WithRefiner"], "#065F46"),
    ]
    xa = np.arange(len(abl_rows))
    hits_a = [float(r[1]["blind_dpa4_clean_hit_pct"]) for r in abl_rows]
    maes_a = [float(r[1]["blind_dpa4_homo_mae_eV"]) for r in abl_rows]
    cols_a = [r[2] for r in abl_rows]

    bars_h = ax0.bar(xa, hits_a, width=0.58, color=cols_a, edgecolor="#1E293B", lw=0.85)
    for br, hv, mv in zip(bars_h, hits_a, maes_a):
        ax0.text(
            br.get_x() + br.get_width() / 2.0,
            hv + 1.1,
            f"{hv:.1f}%\n({mv:.2f}eV)",
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#0F172A",
        )
    ax0.set_xticks(xa)
    ax0.set_xticklabels([r[0] for r in abl_rows], fontsize=6.9, fontweight="bold", color="#1E293B")
    ax0.set_ylim(0.0, max(hits_a) + 13.5)
    ax0.set_ylabel("Blind DPA-4 Clean \u00b10.25 eV Hit Rate (%)", fontsize=9.4, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Speculative Tree vs. Naive Leaf Decoding (N=1,792)", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)

    # =========================================================================
    # Panel B: <0.5 ms PostTreeGraphRefiner Seam Healing (Zero-Step F_max)
    # =========================================================================
    ax1 = axes[1]
    f_med = [float(r[1]["zero_step_f_max_median_eV_A"] or 0.0) for r in abl_rows]
    f_p90 = [float(r[1]["zero_step_f_max_p90_eV_A"] or 0.0) for r in abl_rows]
    wb = 0.34
    b_m = ax1.bar(xa - wb / 2.0, f_med, width=wb * 0.94, color=cols_a, edgecolor="#1E293B", lw=0.8, label="Zero-Step Median F_max (eV/\u00c5)")
    b_p = ax1.bar(xa + wb / 2.0, np.clip(f_p90, 0.0, 24.0), width=wb * 0.94, color="#FFE4E6", edgecolor="#BE123C", lw=0.8, hatch="//", label="Zero-Step 90th-Pct F_max (capped 24)")
    for br, fv in zip(b_m, f_med):
        ax1.text(br.get_x() + br.get_width() / 2.0, fv + 0.35, f"{fv:.2f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#0F172A")
    ax1.set_xticks(xa)
    ax1.set_xticklabels([r[0] for r in abl_rows], fontsize=6.9, fontweight="bold", color="#1E293B")
    ax1.set_ylim(0.0, 26.8)
    ax1.set_ylabel("Zero-Step DPA-4 Atomic Force F_max (eV/\u00c5, \u2193)", fontsize=9.4, fontweight="bold", color="#1E293B")
    ax1.set_title("B. Sub-Millisecond PostTreeGraphRefiner Seam Healing", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="upper right", fontsize=7.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # =========================================================================
    # Panel C: Ground-Truth ProcessCritic Anti-Shortcut Blind Dev Pearson r
    # =========================================================================
    ax2 = axes[2]
    dev_r = qm9["uni3dar_mol_training_metadata"]["dev_blind_mismatched_target_pearson_r"]
    props = ["alpha", "gap", "homo", "lumo", "mu", "cv"]
    plbls = ["\u03b1", "\u0394\u03b5", "\u03b5_HOMO", "\u03b5_LUMO", "\u03bc", "C_v"]
    r_gt = [float(dev_r[p]) for p in props]
    r_test_k16 = [float(qm9["per_property"][p]["tiers"]["K16"]["gvp"]["pearson_r"]) for p in props]

    xr = np.arange(len(props))
    wr = 0.34
    br1 = ax2.bar(xr - wr / 2.0, r_gt, width=wr * 0.94, color="#0D9488", edgecolor="#1E293B", lw=0.8, label="Internal GT ProcessCritic Blind Dev r (t' \u2260 y)")
    br2 = ax2.bar(xr + wr / 2.0, r_test_k16, width=wr * 0.94, color="#064E3B", edgecolor="#1E293B", lw=0.8, label="Generated 74\u219222\u21921 GVP Test Pearson r")
    for br, rv in zip(br1, r_gt):
        ax2.text(br.get_x() + br.get_width() / 2.0, rv + 0.015, f"{rv:.3f}", ha="center", va="bottom", fontsize=6.1, fontweight="bold", color="#0F766E")
    for br, rv in zip(br2, r_test_k16):
        ax2.text(br.get_x() + br.get_width() / 2.0, rv + 0.015, f"{rv:.3f}", ha="center", va="bottom", fontsize=6.1, fontweight="bold", color="#064E3B")

    ax2.set_xticks(xr)
    ax2.set_xticklabels(plbls, fontsize=8.4, fontweight="bold", color="#1E293B")
    ax2.set_ylim(0.0, 1.12)
    ax2.set_ylabel("Pearson Correlation r (0 External Predictors)", fontsize=9.4, fontweight="bold", color="#1E293B")
    ax2.set_title("C. Pure Ground-Truth ProcessCritic & Test Fidelity", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax2.legend(loc="lower right", fontsize=6.9, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    plt.tight_layout(w_pad=1.6)
    extra = (artifact_dir / "fig4_uni3dar_mol_internal_ablation.png") if artifact_dir else None
    _save_both(fig, "fig4_uni3dar_mol_internal_ablation", extra_png=extra)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=Path, default=None)
    args = p.parse_args()
    render_fig2_qm9_and_oe62_scaling(args.artifact_dir)
    render_fig3_oe62_atom_scaling(args.artifact_dir)
    render_fig4_internal_ablation(args.artifact_dir)


if __name__ == "__main__":
    main()
