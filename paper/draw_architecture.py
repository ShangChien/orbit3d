#!/usr/bin/env python3
"""Generate Figure 1 (Uni-Loom 4-Layer Weaving & Grounded RSI Architecture) as vector PDF and PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    bullets: list[str],
    header_color: str,
    face_color: str = "#F8FAFC",
    border_color: str = "#CBD5E1",
    title_size: float = 9.5,
    body_size: float = 8.4,
    header_h: float = 0.40,
):
    body = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        facecolor=face_color,
        edgecolor=border_color,
        linewidth=1.3,
        zorder=2,
    )
    ax.add_patch(body)

    header = FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        facecolor=header_color,
        edgecolor=header_color,
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(header)

    ax.text(
        x + 0.14,
        y + h - header_h / 2.0,
        title,
        color="white",
        fontsize=title_size,
        fontweight="bold",
        va="center",
        ha="left",
        zorder=4,
    )

    top_y = y + h - header_h - 0.16
    usable_h = h - header_h - 0.24
    step = usable_h / max(len(bullets), 1)
    for idx, line in enumerate(bullets):
        ly = top_y - idx * step
        ax.text(
            x + 0.14,
            ly,
            line,
            color="#1E293B",
            fontsize=body_size,
            va="top",
            ha="left",
            linespacing=1.20,
            zorder=4,
        )


def add_poly_arrow(
    ax,
    pts: list[tuple[float, float]],
    color: str = "#334155",
    label: str | None = None,
    label_pos: tuple[float, float] | None = None,
    linestyle: str = "-",
    lw: float = 1.65,
):
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        # Draw all segments up to the final segment start
        ax.plot(
            [p[0] for p in pts[:-1]] + [pts[-2][0]],
            [p[1] for p in pts[:-1]] + [pts[-2][1]],
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )
    arr = FancyArrowPatch(
        pts[-2],
        pts[-1],
        arrowstyle="-|>,head_length=6.2,head_width=3.6",
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        zorder=5,
    )
    ax.add_patch(arr)
    if label and label_pos:
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            fontsize=7.8,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=color, alpha=0.96, lw=0.8),
            zorder=6,
        )


def build_architecture_figure(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.6, 9.6), dpi=300)
    ax.set_xlim(0.0, 15.6)
    ax.set_ylim(0.0, 9.6)
    ax.axis("off")

    # Layer bands (leaving clean outer right gutter x in [14.95, 15.45] and inter-band gutters)
    bands = [
        (0.20, 6.85, 14.70, 2.52, "#F1F5F9", "LAYERS 1 & 2: PREDICTOR-AGNOSTIC ASSET HUB & DECLARATIVE CRITIC BUS"),
        (0.20, 3.60, 14.70, 2.62, "#F0FDF4", "LAYER 3: SCALE-DECOUPLED 3D WEAVING & DUAL-TRACK VALENCE PURITY ENGINE"),
        (0.20, 0.42, 14.70, 2.55, "#FFFBEB", "LAYER 4: HETEROSCEDASTIC GP-ATLAS & GROUNDED RECURSIVE SELF-IMPROVEMENT (RSI)"),
    ]
    for bx, by, bw, bh, bcol, blabel in bands:
        band = FancyBboxPatch(
            (bx, by),
            bw,
            bh,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=bcol,
            edgecolor="#CBD5E1",
            linewidth=1.0,
            zorder=1,
        )
        ax.add_patch(band)
        ax.text(
            bx + 0.16,
            by + bh - 0.14,
            blabel,
            fontsize=8.2,
            fontweight="bold",
            color="#475569",
            ha="left",
            va="top",
            zorder=2,
        )

    # --- ROW 1 (y=7.00 .. 9.00) ---
    add_card(
        ax,
        x=0.38,
        y=7.00,
        w=4.55,
        h=1.98,
        title="1A. Plug-and-Play Predictor Assets (Zero Retrain)",
        bullets=[
            "• 3D Differentiable Oracles: Uni-Mol (scalar y + grad ∇_R y)",
            "• 3D / 2D Black-Box QSAR: Uni-Mol2 1.1B, Morgan K-Fold, ONNX",
            "• Analytical Chemistry Rules: SA_Score, QED, MW, SMARTS alerts",
            "• Capability Metadata: modality, units, uncertainty & grad flags",
        ],
        header_color="#1E3A8A",
    )

    add_card(
        ax,
        x=5.22,
        y=7.00,
        w=4.45,
        h=1.98,
        title="1B. ResidualCalibratedAsset & PotentialSlot",
        bullets=[
            "• Bayesian Wrapper: y_cal(M) = y_base(M) + f_ridge(φ) + δ_GP(φ)",
            "• Noise-Subtracted Epistemic σ_cal Contracts on New Data",
            "• Hot-Swappable MLIP Slot: DPA-4  <->  HyFF vivace",
            "• Zero-Step Force F_max, FIRE Relaxation & Kabsch RMSD",
        ],
        header_color="#0F766E",
    )

    add_card(
        ax,
        x=9.96,
        y=7.00,
        w=4.74,
        h=1.98,
        title="2. TaskWeavingSpec (YAML) & CompositeCriticBus",
        bullets=[
            "• Objectives: Point (v ± τ), Interval or Max/Min (auto K-dim)",
            "• Inequality Constraints: SA <= 3.5, SMARTS == 0 (0 cond dims)",
            "• Cross-Oracle Auditor |y_1 - y_2| or Single-Oracle AD Tanimoto",
            "• Monotonic Advantage Hierarchy Prevents Reward Hacking",
        ],
        header_color="#334155",
    )

    # --- ROW 2 (y=3.75 .. 5.86) ---
    add_card(
        ax,
        x=0.38,
        y=3.75,
        w=4.55,
        h=2.10,
        title="3A. GeneratorSlot (Uni-3DAR + Track A Guard)",
        bullets=[
            "• Unbounded Fourier Target Encoder + 24-Layer LoRA (r=16)",
            "• Tree-Depth Annealed CFG (32.5 mol/s BF16 Batched Decoding)",
            "• Track A1: Bond-Order Neutral Valence Coordination Guard",
            "• Track A2: HeavyAtomValenceHead Auxiliary Loss (num_H, q)",
        ],
        header_color="#1D4ED8",
    )

    add_card(
        ax,
        x=5.22,
        y=3.75,
        w=4.45,
        h=2.10,
        title="3B. Track B ValenceReconciler & Coord Refiner",
        bullets=[
            "• Whitelists Benign Dipoles (-NO2, N-Oxide, Sulfoxide, Azide)",
            "• Freezes Heavy Atoms (RMSD = 0.00 Å) -> Neutral Lewis Graph",
            "• Rebuilds Standard H + Batched MLIP H-Only Local Polish",
            "• Optional Differentiable Refiner: min_R [E_MLIP + λ L_obj(R)]",
        ],
        header_color="#15803D",
    )

    add_card(
        ax,
        x=9.96,
        y=3.75,
        w=4.74,
        h=2.10,
        title="3C. Scale-Decoupled Policy & XYZ Distillation",
        bullets=[
            "• Stage 2.5A (TopoGRPO): Policy Gradient ONLY on Coarse",
            "  Octree & Element Tokens (Eliminates 3x XYZ Variance)",
            "• Stage 2.5B (Sub-Voxel XYZ): Detached 0.01 Å LoRA Heads",
            "  Distilled on 40-Step MLIP Equilibrium Geometries",
        ],
        header_color="#0F766E",
    )

    # --- ROW 3 (y=0.58 .. 2.62) ---
    add_card(
        ax,
        x=0.38,
        y=0.58,
        w=4.55,
        h=2.02,
        title="4A. Generalized 1D/2D GPAtlas & 4-Zone Diagnoser",
        bullets=[
            "• Dual Variance: σ_total²(t) = σ_gen²(t) + σ_cal²(t)",
            "• Pre-Generation Gatekeeper Reroutes Untrusted Targets",
            "• CONVERGED (20% Anchor)  |  INNER_UNDEREXPLORED (80% RL)",
            "• OUTER_BLINDSPOT (RL = 0%)  |  PRIOR_CEILING (RL = 0%)",
        ],
        header_color="#B45309",
    )

    add_card(
        ax,
        x=5.22,
        y=0.58,
        w=4.45,
        h=2.02,
        title="4B. Inner-Loop RSI (Zero External Data Cost)",
        bullets=[
            "• Triggered on Trusted Weak Zones (σ_cal <= τ_cal)",
            "• Deficit-Weighted Curriculum + Anchor Replay Sampling",
            "• Automated Hindsight Relabeling + Stage 2.5A/B Update",
            "• Non-Regression Acceptance Gate Before Promotion",
        ],
        header_color="#D97706",
    )

    add_card(
        ax,
        x=9.96,
        y=0.58,
        w=4.74,
        h=2.02,
        title="4C. Outer-Loop Active Acquisition & Calibration",
        bullets=[
            "• Triggered on Frozen Blindspots (σ_cal > τ_cal)",
            "• Murcko-Diverse Utility u(M) = σ_cal · I[Clean & Constr] · d_scaf",
            "• Exports .sdf Batch -> External DFT / Wet-Lab Measurement",
            "• IncrementalCalibrator Updates Δ_GP -> Unlocks Blindspot!",
        ],
        header_color="#BE123C",
    )

    # --- Clean Gutter-Routed Orthogonal Arrows (ZERO Card Crossings) ---
    # Row 1 horizontal
    add_poly_arrow(ax, [(4.93, 7.99), (5.22, 7.99)], color="#1E3A8A")
    add_poly_arrow(ax, [(9.67, 7.99), (9.96, 7.99)], color="#0F766E")

    # Gutter 1 (y in [6.22, 6.85]): Row 1 -> Row 2
    add_poly_arrow(
        ax,
        [(10.90, 7.00), (10.90, 6.56), (2.65, 6.56), (2.65, 5.85)],
        color="#1D4ED8",
        linestyle="--",
        lw=1.45,
        label="Target Specs (v ± τ)",
        label_pos=(4.65, 6.56),
    )
    add_poly_arrow(
        ax,
        [(7.45, 7.00), (7.45, 5.85)],
        color="#0F766E",
        label="MLIP Forces & ∇_R y",
        label_pos=(7.45, 6.40),
    )
    add_poly_arrow(
        ax,
        [(12.60, 7.00), (12.60, 5.85)],
        color="#334155",
        label="Hierarchical Advantages",
        label_pos=(12.60, 6.40),
    )

    # Row 2 horizontal
    add_poly_arrow(ax, [(4.93, 4.80), (5.22, 4.80)], color="#1D4ED8")
    add_poly_arrow(ax, [(9.67, 4.80), (9.96, 4.80)], color="#15803D")

    # Gutter 2 (y in [2.97, 3.60]): Row 2 feedback & Row 2 -> Row 3
    add_poly_arrow(
        ax,
        [(12.30, 3.75), (12.30, 3.45), (3.85, 3.45), (3.85, 3.75)],
        color="#0F766E",
        label="Policy & Equilibrium XYZ Update",
        label_pos=(8.85, 3.45),
    )
    add_poly_arrow(
        ax,
        [(6.35, 3.75), (6.35, 3.24), (2.95, 3.24), (2.95, 2.60)],
        color="#B45309",
        label="Clean 3D Proposals + Audit",
        label_pos=(4.75, 3.24),
    )
    add_poly_arrow(
        ax,
        [(7.45, 2.60), (7.45, 3.04), (1.35, 3.04), (1.35, 3.75)],
        color="#D97706",
        label="Curriculum Targets",
        label_pos=(1.35, 3.38),
    )

    # Row 3 horizontal (4A -> 4B)
    add_poly_arrow(ax, [(4.93, 1.59), (5.22, 1.59)], color="#D97706")

    # Bottom Gutter (y = 0.18): 4A -> 4C for Blindspot
    add_poly_arrow(
        ax,
        [(3.20, 0.58), (3.20, 0.18), (11.80, 0.18), (11.80, 0.58)],
        color="#BE123C",
        label="σ_cal > τ_cal (Blindspot Frozen -> Export Active Learning Batch)",
        label_pos=(7.50, 0.18),
    )

    # Outer Right Margin Gutter (x = 15.22, y = 1.59 -> 9.48 -> 7.45 -> 8.98): 4C -> 1B
    add_poly_arrow(
        ax,
        [(14.70, 1.59), (15.25, 1.59), (15.25, 9.47), (7.45, 9.47), (7.45, 8.98)],
        color="#BE123C",
        lw=1.80,
        label="Zero-Retrain DFT Feedback Updates Δ_GP(φ) & Unlocks Blindspot",
        label_pos=(11.35, 9.47),
    )

    fig.tight_layout(pad=0.18)
    pdf_path = out_dir / "fig1_uniloom_architecture.pdf"
    png_path = out_dir / "fig1_uniloom_architecture.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")


if __name__ == "__main__":
    build_architecture_figure(Path(__file__).resolve().parent / "figures")
