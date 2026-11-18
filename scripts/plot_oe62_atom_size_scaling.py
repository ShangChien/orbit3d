"""Plot Controlled Multi-Atom-Count Scaling (N=20..60): Uni-3DAR vs PropMolFlow Latency, MAE & Yield.

Reads `docs/experiments/oe62_atom_size_buckets_scaling.json` (`280` prompts/bucket across exact
atom counts `N in {20, 30, 40, 50, 60}`) and renders a 3-panel comparison:
  - Panel A: End-to-End Inference Latency (`GPU-sec per 1,000 proposals`) vs Exact Atom Count `N`
             (highlighting `O(N)` sparse octree scaling in `Uni-3DAR` vs `O(N^2)` dense complete-graph
              ODE scaling in `PropMolFlow`).
  - Panel B: Blind `DPA-4` HOMO MAE (`eV`) vs Exact Atom Count `N`.
  - Panel C: Blind `DPA-4` Synthesizable Clean Hit Yield (`%`, `|y-t|<=0.25 eV` & `SA<=3.5`) vs `N`.
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
EXP_JSON = ROOT / "docs" / "experiments" / "oe62_atom_size_buckets_scaling.json"


def render_atom_scaling_figure(out_paths: List[Path]) -> None:
    d = json.loads(EXP_JSON.read_text())
    b_map: Dict[str, Any] = d["buckets"]
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
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.8), dpi=220)
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
        ("Uni3DAR_M1", "Uni-Loom [Uni-3DAR M=1] (2.45h)", "#0D9488", "--s", 1.9, 6.2),
        ("Uni3DAR_Process_8_to_4", "Uni-Loom [Uni-3DAR 8→4] (2.45h)", "#7C3AED", "-^", 2.2, 6.8),
        ("Uni3DAR_Process_16_to_4", "Uni-Loom [Uni-3DAR 16→4] (2.45h)", "#5B21B6", "-P", 2.6, 7.8),
        ("PMF_52h_M1", "Vanilla PropMolFlow @ 52h [M=1, No Loom]", "#94A3B8", "--o", 1.8, 5.8),
        ("PMF_52h_M8", "PropMolFlow @ 52h + Loom-Critic [M=8]", "#334155", "-o", 2.3, 6.6),
        ("PMF_52h_LoomAWR_M8", "Uni-Loom [PMF @ 52.12h + Loom-AWR, M=8]", "#E11D48", "-D", 2.4, 6.4),
    ]

    # Panel A: Latency (GPU-sec per 1,000 proposals)
    ax0 = axes[0]
    # Panel B: Blind DPA-4 HOMO MAE (eV)
    ax1 = axes[1]
    # Panel C: Blind DPA-4 Synthesizable Clean Hit Yield (%)
    ax2 = axes[2]

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

        ax0.plot(xs_valid, lat_1k, fmt, color=color, lw=lw, ms=ms, label=label)
        ax1.plot(xs_valid, mae_list, fmt, color=color, lw=lw, ms=ms, label=label)
        ax2.plot(xs_valid, hit_list, fmt, color=color, lw=lw, ms=ms, label=label)

    ax0.set_xticks(ns)
    ax0.set_xlabel("Exact Requested Atom Count N (Total Heavy + H Atoms)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax0.set_ylabel("End-to-End Latency per 1,000 Proposals (A800 GPU-sec)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax0.set_title("A. Inference Latency Scaling: Sparse Octree vs Dense O(N²) ODE", fontsize=10.4, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax0.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    ax1.set_xticks(ns)
    ax1.set_xlabel("Exact Requested Atom Count N (Total Heavy + H Atoms)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Blind DPA-4 HOMO MAE (eV, lower is better)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax1.set_title("B. Blind Orbital Accuracy vs Molecular Size", fontsize=10.4, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax1.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    ax2.set_xticks(ns)
    ax2.set_xlabel("Exact Requested Atom Count N (Total Heavy + H Atoms)", fontsize=9.6, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Blind DPA-4 Synth-Clean Hit Yield (%)", fontsize=9.8, fontweight="bold", color="#1E293B")
    ax2.set_title("C. Synthesizable Clean Hit Yield (SA ≤ 3.5) vs Molecular Size", fontsize=10.4, fontweight="bold", color="#1E293B", loc="left", pad=10)
    ax2.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor=bg_ax, edgecolor="#CBD5E1")

    fig.text(
        0.5,
        0.01,
        "Controlled Fixed-Size OE62 Benchmark (280 prompts = 7 HOMO quantiles × 40 per exact atom count N ∈ {20, 30, 40, 50, 60}; "
        "identical N held fixed across all M candidates on single uncontended NVIDIA A800-SXM4-80GB with Fair TF32).",
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
        *args.extra_out,
    ]
    render_atom_scaling_figure(targets)


if __name__ == "__main__":
    main()
