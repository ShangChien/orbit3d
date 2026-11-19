"""Plot Controlled Multi-Atom-Count Scaling (N=20..60): Standardized `M=1, 4, 8` Tiers (`Uni-3DAR` vs `PropMolFlow`).

Merges `docs/experiments/oe62_uni3dar_v4_pro_atom_buckets.json` (`Uni-Loom [Uni-3DAR]` standardized
`base: M=1`, `medium: M=4`, `high: M=8` service tiers) with `docs/experiments/oe62_atom_size_buckets_scaling.json`
(`PropMolFlow @ 52h` `M=1`, `M=8`, and `Uni-Loom [PMF @ 52.12h + Loom-AWR, M=8]`) across exact requested
atom counts `N in {20, 30, 40, 50, 60}` (`280` prompts/bucket).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PMF_BUCKETS_JSON = ROOT / "docs" / "experiments" / "oe62_atom_size_buckets_scaling.json"
U3D_BUCKETS_JSON = ROOT / "docs" / "experiments" / "oe62_uni3dar_v4_pro_atom_buckets.json"


def render_atom_scaling_figure(out_paths: List[Path]) -> None:
    pmf_d = json.loads(PMF_BUCKETS_JSON.read_text())
    u3d_d = json.loads(U3D_BUCKETS_JSON.read_text())
    u3d_h = float(u3d_d.get("total_training_gpuh", 2.25))

    b_map: Dict[str, Dict[str, Any]] = {}
    for n_str, blk in pmf_d["buckets"].items():
        b_map[n_str] = {"arms": dict(blk["arms"])}
    for n_str, blk in u3d_d["buckets"].items():
        b_map.setdefault(n_str, {"arms": {}})["arms"].update(blk["arms"])

    ns = sorted(int(k) for k in b_map.keys())

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
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 5.8), dpi=220)
    fig.patch.set_facecolor(bg_fig)
    for ax in axes:
        ax.set_facecolor(bg_ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.tick_params(colors="#475569")
        ax.grid(True, which="major", linestyle="--", linewidth=0.65, color="#E4ECE9", alpha=0.85)

    series = [
        ("Uni3DAR_base_M1", f"Uni-Loom [Uni-3DAR, M=1] ({u3d_h:.2f}h)", "#14B8A6", "--s", 1.9, 6.0),
        ("Uni3DAR_medium_M4", f"Uni-Loom [Uni-3DAR, M=4] ({u3d_h:.2f}h)", "#0F766E", "-^", 2.3, 6.8),
        ("Uni3DAR_high_M8", f"Uni-Loom [Uni-3DAR, M=8] ({u3d_h:.2f}h)", "#065F46", "-o", 2.7, 7.4),
        ("PMF_52h_M1", "Vanilla PropMolFlow @ 52h [M=1, No Loom]", "#94A3B8", "--o", 1.8, 5.6),
        ("PMF_52h_M8", "PropMolFlow @ 52h + Loom-Critic [M=8]", "#334155", "-o", 2.2, 6.4),
        ("PMF_52h_LoomAWR_M8", "Uni-Loom [PMF @ 52.12h + Loom-AWR, M=8]", "#E11D48", "-D", 2.4, 6.4),
    ]

    ax0, ax1, ax2 = axes
    for key, label, color, fmt, lw, ms in series:
        xs_valid: List[int] = []
        lat_1k: List[float] = []
        mae_list: List[float] = []
        hit_list: List[float] = []
        for n in ns:
            arms = b_map[str(n)]["arms"]
            if key not in arms:
                continue
            r = arms[key]
            eff = float(r["effective_output_mol_per_sec"])
            bmae = r.get("audit_mae_eV")
            if bmae is None:
                continue
            xs_valid.append(n)
            lat_1k.append(1000.0 / max(eff, 1e-6))
            mae_list.append(float(bmae))
            hit_list.append(float(r["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0)

        if xs_valid:
            ax0.plot(xs_valid, lat_1k, fmt, color=color, lw=lw, ms=ms, label=label)
            ax1.plot(xs_valid, mae_list, fmt, color=color, lw=lw, ms=ms, label=label)
            ax2.plot(xs_valid, hit_list, fmt, color=color, lw=lw, ms=ms, label=label)

    ax0.set_xticks(ns)
    ax0.set_ylim(-40, 1680)
    ax0.set_xlabel("Exact Requested Atom Count N (Total Heavy + H Atoms)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax0.set_ylabel("Measured Latency per 1,000 Prompts (A800 GPU-sec)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Inference Latency Scaling: Sparse Octree vs Dense O(N²) ODE", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax0.legend(loc="upper left", fontsize=6.9, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    lat60_pmf = 1000.0 / float(b_map["60"]["arms"]["PMF_52h_M8"]["effective_output_mol_per_sec"])
    lat60_u4 = 1000.0 / float(b_map["60"]["arms"]["Uni3DAR_medium_M4"]["effective_output_mol_per_sec"])
    lat60_u8 = 1000.0 / float(b_map["60"]["arms"]["Uni3DAR_high_M8"]["effective_output_mol_per_sec"])
    ax0.annotate(
        f"N=60 Speedup vs PMF M=8:\n{lat60_pmf/lat60_u4:.1f}× faster (M=4)\n{lat60_pmf/lat60_u8:.1f}× faster (M=8)",
        xy=(60, lat60_u4),
        xytext=(23.0, 960),
        fontsize=7.4,
        fontweight="bold",
        color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#FFFFFF", edgecolor="#0F766E", alpha=0.94, lw=0.85),
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.0),
    )

    ax1.set_xticks(ns)
    ax1.set_ylim(0.15, 1.55)
    ax1.set_xlabel("Exact Requested Atom Count N (Total Heavy + H Atoms)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Blind DPA-4 HOMO MAE (eV, lower is better)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax1.set_title("B. Blind Orbital Accuracy vs Molecular Size", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="upper left", fontsize=6.9, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    ax2.set_xticks(ns)
    ax2.set_ylim(-3, 82)
    ax2.set_xlabel("Exact Requested Atom Count N (Total Heavy + H Atoms)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax2.set_title("C. Synthesizable Clean Hit Yield (SA ≤ 3.5) vs Molecular Size", fontsize=10.3, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax2.legend(loc="upper right", fontsize=6.9, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    h50_awr = float(b_map["50"]["arms"]["PMF_52h_LoomAWR_M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    h50_u8 = float(b_map["50"]["arms"]["Uni3DAR_high_M8"]["blind_dpa4_synthesizable_clean_hit_yield"]) * 100.0
    ax2.annotate(
        f"N=50 Large-Molecule Lead:\nUni-3DAR M=8: {h50_u8:.1f}%\nvs PMF 52h+AWR M=8: {h50_awr:.1f}%",
        xy=(50, h50_u8),
        xytext=(38.0, 50.0),
        fontsize=7.4,
        fontweight="bold",
        color="#065F46",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#FFFFFF", edgecolor="#065F46", alpha=0.94, lw=0.85),
        arrowprops=dict(arrowstyle="->", color="#065F46", lw=1.0),
    )

    fig.text(
        0.5,
        0.01,
        "Controlled Fixed-Size OE62 Benchmark (280 prompts = 7 HOMO quantiles × 40 per exact atom count N ∈ {20, 30, 40, 50, 60}; "
        "identical N held fixed across all M ∈ {1, 4, 8} candidates on single uncontended NVIDIA A800-SXM4-80GB with Fair TF32).",
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
        ROOT / "docs" / "figures" / "oe62_atom_scaling_speed_and_mae.png",
        ROOT / "paper" / "figures" / "fig9_oe62_atom_size_scaling.png",
        ROOT / "paper" / "figures" / "fig9_oe62_atom_size_scaling.pdf",
        ROOT / "paper" / "ncs" / "figures" / "ed_fig4_oe62_atom_size_scaling.png",
        ROOT / "paper" / "ncs" / "figures" / "ed_fig4_oe62_atom_size_scaling.pdf",
        *args.extra_out,
    ]
    render_atom_scaling_figure(targets)


if __name__ == "__main__":
    main()
