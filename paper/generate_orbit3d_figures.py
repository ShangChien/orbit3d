#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-quality SVG + high-resolution PNG figures for the ORBIT-3D arXiv paper.
Uses a square-friendly 1000x920 multi-row layout so CoreGraphics/QuickLook renders 100% of all panels without cropping.
"""

import html
import shutil
import subprocess
from pathlib import Path

OUT_DIR = Path("/Users/dp/project/mol_gen/paper/figures")
ARTIFACT_DIR = Path("/Users/dp/.gemini/antigravity/brain/615f5339-d81a-4c2a-9bad-bb218c9859c6")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def save_svg_and_png(svg_content: str, stem: str, size_px: int = 1800):
    svg_path = OUT_DIR / f"{stem}.svg"
    png_path = OUT_DIR / f"{stem}.png"
    svg_path.write_text(svg_content, encoding="utf-8")
    subprocess.run(["qlmanage", "-t", "-s", str(size_px), "-o", str(OUT_DIR), str(svg_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ql_png = OUT_DIR / f"{stem}.svg.png"
    if ql_png.exists():
        shutil.move(str(ql_png), str(png_path))
    if ARTIFACT_DIR.exists() and png_path.exists():
        shutil.copy2(png_path, ARTIFACT_DIR / f"{stem}.png")
    print(f"[Generated] {png_path} & {svg_path}")


def draw_grouped_bar_panel(
    x0, y0, w, h, title, y_label, categories, series_list, y_min, y_max, y_ticks, fmt="{:.1f}%"
):
    elements = []
    elements.append(f'<text x="{x0 + w/2:.1f}" y="{y0 - 14:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="14.5" font-weight="bold" text-anchor="middle" fill="#0f172a">{esc(title)}</text>')
    elements.append(f'<text transform="translate({x0 - 42:.1f},{y0 + h/2:.1f}) rotate(-90)" font-family="Helvetica, Arial, sans-serif" font-size="12.5" font-weight="bold" text-anchor="middle" fill="#1e293b">{esc(y_label)}</text>')

    for yt in y_ticks:
        frac = (yt - y_min) / (y_max - y_min)
        yy = y0 + h - frac * h
        elements.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,4"/>')
        lbl = f"{yt:g}"
        elements.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="11" text-anchor="end" fill="#334155">{esc(lbl)}</text>')

    elements.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" stroke="#334155" stroke-width="1.5"/>')
    elements.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="#334155" stroke-width="1.5"/>')

    n_cat = len(categories)
    n_ser = len(series_list)
    group_w = w / n_cat
    bar_w = min(40.0, (group_w * 0.72) / n_ser)

    for ci, cat in enumerate(categories):
        cx = x0 + (ci + 0.5) * group_w
        for li, line in enumerate(cat.split("\n")):
            elements.append(f'<text x="{cx:.1f}" y="{y0 + h + 18 + li*13:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e293b">{esc(line)}</text>')

        start_x = cx - (n_ser * bar_w) / 2.0
        for si, (ser_name, vals, fill_col, stroke_col) in enumerate(series_list):
            val = vals[ci]
            frac = max(0.0, min(1.0, (val - y_min) / (y_max - y_min)))
            bh = frac * h
            bx = start_x + si * bar_w
            by = y0 + h - bh
            elements.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w - 3:.1f}" height="{bh:.1f}" fill="{fill_col}" stroke="{stroke_col}" stroke-width="1.2" rx="2"/>')
            val_str = fmt.format(val)
            elements.append(f'<text x="{bx + (bar_w-3)/2:.1f}" y="{by - 5:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" text-anchor="middle" fill="#0f172a">{esc(val_str)}</text>')

    leg_x = x0 + 10
    leg_y = y0 + 8
    leg_h = len(series_list) * 17 + 8
    elements.append(f'<rect x="{leg_x}" y="{leg_y}" width="{min(360.0, w - 20):.1f}" height="{leg_h}" fill="#ffffff" fill-opacity="0.92" stroke="#e2e8f0" rx="4"/>')
    for si, (ser_name, _, fill_col, stroke_col) in enumerate(series_list):
        ly = leg_y + 12 + si * 17
        elements.append(f'<rect x="{leg_x + 8}" y="{ly - 8}" width="11" height="9" fill="{fill_col}" stroke="{stroke_col}" stroke-width="1"/>')
        elements.append(f'<text x="{leg_x + 24}" y="{ly}" font-family="Helvetica, Arial, sans-serif" font-size="10.5" fill="#1e293b">{esc(ser_name)}</text>')

    return "\n".join(elements)


def draw_line_panel(x0, y0, w, h, title, y_label, x_labels, lines_data, y_min, y_max, y_ticks):
    el = []
    el.append(f'<text x="{x0 + w/2:.1f}" y="{y0 - 14:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="14.5" font-weight="bold" text-anchor="middle" fill="#0f172a">{esc(title)}</text>')
    el.append(f'<text transform="translate({x0 - 44:.1f},{y0 + h/2:.1f}) rotate(-90)" font-family="Helvetica, Arial, sans-serif" font-size="12.5" font-weight="bold" text-anchor="middle" fill="#1e293b">{esc(y_label)}</text>')

    for yt in y_ticks:
        frac = (yt - y_min) / (y_max - y_min)
        yy = y0 + h - frac * h
        el.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,4"/>')
        el.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="11" text-anchor="end" fill="#334155">{yt:g}</text>')

    el.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" stroke="#334155" stroke-width="1.5"/>')
    el.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="#334155" stroke-width="1.5"/>')

    n_pts = len(x_labels)
    xs = [x0 + 42 + i * (w - 84) / (n_pts - 1) for i in range(n_pts)]
    for i, xl in enumerate(x_labels):
        for li, line in enumerate(xl.split("\n")):
            el.append(f'<text x="{xs[i]:.1f}" y="{y0 + h + 17 + li*13:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#1e293b">{esc(line)}</text>')

    for name, vals, col, dash in lines_data:
        pts = []
        for i, v in enumerate(vals):
            frac = (v - y_min) / (y_max - y_min)
            yy = y0 + h - frac * h
            pts.append((xs[i], yy, v))
        poly = " ".join(f"{px:.1f},{py:.1f}" for px, py, _ in pts)
        dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
        el.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.8" {dash_attr} points="{poly}"/>')
        for px, py, v in pts:
            el.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>')

    leg_x = x0 + 12
    leg_y = y0 + 8
    leg_h = len(lines_data) * 18 + 8
    el.append(f'<rect x="{leg_x}" y="{leg_y}" width="395" height="{leg_h}" fill="#ffffff" fill-opacity="0.92" stroke="#e2e8f0" rx="4"/>')
    for si, (name, _, col, dash) in enumerate(lines_data):
        ly = leg_y + 13 + si * 18
        dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
        el.append(f'<line x1="{leg_x + 8}" y1="{ly - 4}" x2="{leg_x + 28}" y2="{ly - 4}" stroke="{col}" stroke-width="2.6" {dash_attr}/>')
        el.append(f'<circle cx="{leg_x + 18}" cy="{ly - 4}" r="3.5" fill="{col}"/>')
        el.append(f'<text x="{leg_x + 36}" y="{ly}" font-family="Helvetica, Arial, sans-serif" font-size="10.5" fill="#1e293b">{esc(name)}</text>')

    return "\n".join(el)


def build_fig0():
    W, H = 980, 980
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs>',
        '  <marker id="arr-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/></marker>',
        '  <marker id="arr-teal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#0d9488"/></marker>',
        '  <marker id="arr-purple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed"/></marker>',
        '  <marker id="arr-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#d97706"/></marker>',
        '  <marker id="arr-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#334155"/></marker>',
        '</defs>',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        # Header banner
        '<rect x="26" y="14" width="934" height="46" rx="8" fill="#0f172a"/>',
        f'<text x="493" y="34" font-family="Helvetica, Arial, sans-serif" font-size="15.2" font-weight="bold" text-anchor="middle" fill="#ffffff">ORBIT-3D System Architecture: Five-Stage Closed-Loop 3D Quantum-Chemical Generation</text>',
        f'<text x="493" y="51" font-family="Helvetica, Arial, sans-serif" font-size="11.0" text-anchor="middle" fill="#94a3b8">Decoupled Architecture:  orbit3d/generator (Stages 1–4)   |   orbit3d/oracles (Pluggable Modes A–D)   |   orbit3d/physics (DPA-4 ZBL &amp; HyFF Charge/Spin)</text>',

        # =========================================================================
        # BAND 1: STAGES 1 - 3 (y = 70 .. 346)
        # =========================================================================
        '<rect x="26" y="70" width="934" height="276" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.6"/>',
        '<rect x="26" y="70" width="934" height="26" rx="8" fill="#1e3a8a"/>',
        '<text x="40" y="88" font-family="Helvetica, Arial, sans-serif" font-size="12.2" font-weight="bold" fill="#ffffff">LAYER 1 (orbit3d/generator) — Stages 1–3: Bidirectional Interval-Conditioned Uni-3DAR Octree Transformer</text>',

        # Input Conditioning Card (x=38, y=108, w=204, h=224)
        '<rect x="38" y="108" width="204" height="224" rx="8" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>',
        '<rect x="38" y="108" width="204" height="24" rx="6" fill="#2563eb"/>',
        '<text x="140" y="124" font-family="Helvetica, Arial, sans-serif" font-size="11.2" font-weight="bold" text-anchor="middle" fill="#ffffff">Dynamic Target Spec (c, Q, M)</text>',
        '<text x="48" y="147" font-family="Helvetica, Arial, sans-serif" font-size="10.4" font-weight="bold" fill="#1e3a8a">1. Property Intervals I = {[a_k, b_k]}:</text>',
        '<text x="52" y="163" font-family="Helvetica, Arial, sans-serif" font-size="9.8" fill="#1e293b">• Midpoint m_k = (a_k + b_k) / 2</text>',
        '<text x="52" y="178" font-family="Helvetica, Arial, sans-serif" font-size="9.8" fill="#1e293b">• Half-width w_k = (b_k - a_k) / 2</text>',
        '<rect x="48" y="188" width="184" height="42" rx="5" fill="#ffffff" stroke="#93c5fd" stroke-width="1"/>',
        '<text x="140" y="204" font-family="Helvetica, Arial, sans-serif" font-size="9.8" font-weight="bold" text-anchor="middle" fill="#1d4ed8">Fourier Spectral Encoder</text>',
        '<text x="140" y="220" font-family="Helvetica, Arial, sans-serif" font-size="9.4" text-anchor="middle" fill="#334155">c = MLP([sin(2^f pi m_k), cos(...)])</text>',
        '<text x="48" y="248" font-family="Helvetica, Arial, sans-serif" font-size="10.4" font-weight="bold" fill="#1e3a8a">2. Quantum Electronic State:</text>',
        '<text x="52" y="264" font-family="Helvetica, Arial, sans-serif" font-size="9.8" fill="#1e293b">• Total Charge Q in Z (0, +1, -1)</text>',
        '<text x="52" y="280" font-family="Helvetica, Arial, sans-serif" font-size="9.8" fill="#1e293b">• Spin Multiplicity M = 2S + 1</text>',
        '<text x="58" y="296" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#475569">(S0: M=1, D0: M=2, T1: M=3)</text>',
        '<text x="52" y="314" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#0f766e">• Active Mask a_k in {0, 1}^K</text>',

        # Stage 1 & Stage 2 Card (x=266, y=108, w=390, h=224)
        '<rect x="266" y="108" width="390" height="224" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="1.6"/>',
        '<rect x="266" y="108" width="390" height="24" rx="6" fill="#1d4ed8"/>',
        '<text x="461" y="124" font-family="Helvetica, Arial, sans-serif" font-size="11.4" font-weight="bold" text-anchor="middle" fill="#ffffff">Stages 1 &amp; 2: Conditional 3D Octree Autoregression</text>',

        # Stage 2 sub-box (top)
        '<rect x="278" y="138" width="366" height="78" rx="6" fill="#eff6ff" stroke="#60a5fa" stroke-width="1.2"/>',
        '<text x="288" y="155" font-family="Helvetica, Arial, sans-serif" font-size="10.5" font-weight="bold" fill="#1e40af">Stage 2 (2_uni3dar_cond_sft): Dual Interval Conditioning</text>',
        '<rect x="288" y="163" width="158" height="44" rx="4" fill="#ffffff" stroke="#93c5fd" stroke-width="1"/>',
        '<text x="367" y="180" font-family="Helvetica, Arial, sans-serif" font-size="9.8" font-weight="bold" text-anchor="middle" fill="#1e3a8a">Root-Prefix [COND] Token</text>',
        '<text x="367" y="196" font-family="Helvetica, Arial, sans-serif" font-size="9.2" text-anchor="middle" fill="#334155">e_0 = W_pref * c (Causal Anchor)</text>',
        '<rect x="454" y="163" width="178" height="44" rx="4" fill="#ffffff" stroke="#93c5fd" stroke-width="1"/>',
        '<text x="543" y="180" font-family="Helvetica, Arial, sans-serif" font-size="9.8" font-weight="bold" text-anchor="middle" fill="#1e3a8a">12-Layer AdaLN-Zero</text>',
        '<text x="543" y="196" font-family="Helvetica, Arial, sans-serif" font-size="9.2" text-anchor="middle" fill="#334155">(gamma_l, beta_l, alpha_l) = W_l SiLU(c)</text>',

        # Stage 1 sub-box (bottom)
        '<rect x="278" y="224" width="366" height="98" rx="6" fill="#f0fdf4" stroke="#34d399" stroke-width="1.2"/>',
        '<text x="288" y="241" font-family="Helvetica, Arial, sans-serif" font-size="10.5" font-weight="bold" fill="#065f46">Stage 1 (1_uni3dar_uncond): Coarse-to-Fine 3D Spatial Octree</text>',
        '<rect x="288" y="249" width="346" height="24" rx="4" fill="#ffffff" stroke="#6ee7b7" stroke-width="1"/>',
        '<text x="461" y="265" font-family="Helvetica, Arial, sans-serif" font-size="9.8" font-weight="bold" text-anchor="middle" fill="#047857">L=0 (8.0 A)  --&gt;  L=1 (4.0 A)  --&gt;  L=2 (2.0 A)  --&gt;  L=3 (1.0 A)  --&gt;  0.01 A XYZ</text>',
        '<text x="288" y="290" font-family="Helvetica, Arial, sans-serif" font-size="9.6" fill="#1e293b">• Leaf Token tau_i = (octant_path_i, z_i in A, q_x, q_y, q_z)</text>',
        '<text x="288" y="308" font-family="Helvetica, Arial, sans-serif" font-size="9.6" font-weight="bold" fill="#065f46">• Deterministic Valence Mask: sum_j b_ij &lt;= v_max(z_i) (100% Closed-Shell)</text>',

        # Stage 3 Card (x=680, y=108, w=266, h=224)
        '<rect x="680" y="108" width="266" height="224" rx="8" fill="#faf5ff" stroke="#8b5cf6" stroke-width="1.5"/>',
        '<rect x="680" y="108" width="266" height="24" rx="6" fill="#7c3aed"/>',
        '<text x="813" y="124" font-family="Helvetica, Arial, sans-serif" font-size="11.4" font-weight="bold" text-anchor="middle" fill="#ffffff">Stage 3: Bidirectional Understanding</text>',
        '<text x="692" y="147" font-family="Helvetica, Arial, sans-serif" font-size="10.2" font-weight="bold" fill="#5b21b6">Suffix [CLS] + Leaf Atom Pooling:</text>',
        '<rect x="692" y="154" width="242" height="36" rx="5" fill="#ffffff" stroke="#c4b5fd" stroke-width="1"/>',
        '<text x="813" y="170" font-family="Helvetica, Arial, sans-serif" font-size="9.8" font-weight="bold" text-anchor="middle" fill="#4c1d95">h_mol = [ h_[CLS] ; Mean_{i in Atom}(h_i) ]</text>',
        '<text x="813" y="184" font-family="Helvetica, Arial, sans-serif" font-size="9.4" text-anchor="middle" fill="#334155">y_self(tau) = MLP_self(h_mol)</text>',
        '<text x="692" y="208" font-family="Helvetica, Arial, sans-serif" font-size="10.2" font-weight="bold" fill="#5b21b6">Joint Generation + Understanding Loss:</text>',
        '<text x="698" y="225" font-family="Helvetica, Arial, sans-serif" font-size="9.6" fill="#1e293b">L_SFT = L_AR(tau | c) + lambda * ||y_self - y||^2</text>',
        '<rect x="692" y="238" width="242" height="82" rx="6" fill="#ffffff" stroke="#a78bfa" stroke-width="1.2"/>',
        '<text x="813" y="256" font-family="Helvetica, Arial, sans-serif" font-size="10.2" font-weight="bold" text-anchor="middle" fill="#6d28d9">K-Candidate Self-Reranking (K=4)</text>',
        '<text x="702" y="274" font-family="Helvetica, Arial, sans-serif" font-size="9.5" fill="#334155">1. Sample K octree rollouts {tau^(k)}</text>',
        '<text x="702" y="290" font-family="Helvetica, Arial, sans-serif" font-size="9.5" fill="#334155">2. Score internal interval error L_int(y_self)</text>',
        '<text x="702" y="306" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#4c1d95">3. Select best 3D conformer M_0 = (z, X_0)</text>',

        # Arrows inside Band 1
        '<line x1="242" y1="210" x2="264" y2="210" stroke="#2563eb" stroke-width="2.2" marker-end="url(#arr-blue)"/>',
        '<line x1="656" y1="210" x2="678" y2="210" stroke="#7c3aed" stroke-width="2.2" marker-end="url(#arr-purple)"/>',

        # =========================================================================
        # BAND 2: ORACLES & QUANTUM FORCE FIELDS (y = 368 .. 648)
        # =========================================================================
        # Left Container: Pluggable Multi-Modal Oracles (x=26, y=368, w=490, h=280)
        '<rect x="26" y="368" width="490" height="280" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.6"/>',
        '<rect x="26" y="368" width="490" height="26" rx="8" fill="#0f766e"/>',
        '<text x="38" y="386" font-family="Helvetica, Arial, sans-serif" font-size="11.6" font-weight="bold" fill="#ffffff">LAYER 2A (orbit3d/oracles) — Pluggable Multi-Modal Quantum Oracles</text>',

        # Mode A (top-left)
        '<rect x="38" y="404" width="226" height="96" rx="6" fill="#f0fdfa" stroke="#14b8a6" stroke-width="1.3"/>',
        '<text x="48" y="421" font-family="Helvetica, Arial, sans-serif" font-size="10.2" font-weight="bold" fill="#0f766e">Mode A (Mainline): internal_unimol</text>',
        '<text x="48" y="438" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• Open-Source 5-Fold 3D Uni-Mol (FP16)</text>',
        '<text x="48" y="453" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• SE(3) Gaussian Pair-Distance Kernel</text>',
        '<text x="48" y="468" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• Built-in all_h &amp; no_h (181.7 MB)</text>',
        '<text x="48" y="484" font-family="Helvetica, Arial, sans-serif" font-size="9.3" font-weight="bold" fill="#047857">• Differentiable d(y_hat)/dX + 5-Fold sigma</text>',

        # Mode B (top-right)
        '<rect x="278" y="404" width="226" height="96" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.3"/>',
        '<text x="288" y="421" font-family="Helvetica, Arial, sans-serif" font-size="10.2" font-weight="bold" fill="#1d4ed8">Mode B (ZBL Unified): internal_dpa4</text>',
        '<text x="288" y="438" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• Open-Source 5-Fold Native DPA-4</text>',
        '<text x="288" y="453" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• Hooks SeZM fitting_net (D=32/64) + RBF</text>',
        '<text x="288" y="468" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• 100% Shared Weights w/ Force Engine</text>',
        '<text x="288" y="484" font-family="Helvetica, Arial, sans-serif" font-size="9.3" font-weight="bold" fill="#1e40af">• Zero Extra GPU VRAM Overhead</text>',

        # Mode C (bottom-left)
        '<rect x="38" y="510" width="226" height="98" rx="6" fill="#fdf4ff" stroke="#a855f7" stroke-width="1.3"/>',
        '<text x="48" y="527" font-family="Helvetica, Arial, sans-serif" font-size="10.2" font-weight="bold" fill="#7e22ce">Mode C (Charge &amp; Spin): internal_hyff</text>',
        '<text x="48" y="544" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• Open-Source 5-Fold Native HyFF</text>',
        '<text x="48" y="559" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• Hooks readout_norms[-1] (D=32..128)</text>',
        '<text x="48" y="574" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• Explicit Charge Q &amp; Spin M=2S+1 Input</text>',
        '<text x="48" y="590" font-family="Helvetica, Arial, sans-serif" font-size="9.3" font-weight="bold" fill="#6b21a8">• Zero-Shot dE(T1-S0) &amp; Vertical IP</text>',

        # Mode D (bottom-right)
        '<rect x="278" y="510" width="226" height="98" rx="6" fill="#fffbeb" stroke="#f59e0b" stroke-width="1.3"/>',
        '<text x="288" y="527" font-family="Helvetica, Arial, sans-serif" font-size="10.2" font-weight="bold" fill="#b45309">Mode D (Enterprise): external_uniqsar</text>',
        '<text x="288" y="544" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• Full-Matrix Uni-QSAR &amp; OCNet Plugin</text>',
        '<text x="288" y="559" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• 1D SMILES + 2D Morgan FP + 3D Uni-Mol</text>',
        '<text x="288" y="574" font-family="Helvetica, Arial, sans-serif" font-size="9.4" fill="#1e293b">• GBDT / CatBoost / Tight-Binding Ensembles</text>',
        '<text x="288" y="590" font-family="Helvetica, Arial, sans-serif" font-size="9.3" font-weight="bold" fill="#92400e">• Gradient-Free GRPO &amp; Trust-Region Gate</text>',

        # Unified Oracle Output Bus Bar
        '<rect x="38" y="616" width="466" height="24" rx="5" fill="#ccfbf1" stroke="#0d9488" stroke-width="1.2"/>',
        '<text x="271" y="632" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" text-anchor="middle" fill="#115e59">Unified Oracle Bus:  Mean mu_k(M),  Epistemic Uncertainty sigma_k(M),  Grad d(y_k)/dX</text>',

        # Right Container: Dual-Engine Quantum Force Fields (x=532, y=368, w=428, h=280)
        '<rect x="532" y="368" width="428" height="280" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.6"/>',
        '<rect x="532" y="368" width="428" height="26" rx="8" fill="#9a3412"/>',
        '<text x="544" y="386" font-family="Helvetica, Arial, sans-serif" font-size="11.4" font-weight="bold" fill="#ffffff">LAYER 2B (orbit3d/physics) — Dual-Engine OMol25 Quantum Force Fields</text>',

        # Engine 1: DPA-4
        '<rect x="544" y="404" width="404" height="96" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.3"/>',
        '<text x="554" y="421" font-family="Helvetica, Arial, sans-serif" font-size="10.4" font-weight="bold" fill="#9a3412">Engine 1: Full-Series DPA-4 (OMol25, 62.86 MB Built-in)</text>',
        '<text x="554" y="438" font-family="Helvetica, Arial, sans-serif" font-size="9.3" fill="#1e293b">• 5 Built-in Scales: Nano (1.95M), Mini (2.62M), Neo (4.42M), Air (19.9M), Plus (34.0M)</text>',
        '<text x="554" y="453" font-family="Helvetica, Arial, sans-serif" font-size="9.3" fill="#1e293b">• Dense SE(3)-Equivariant SO(2) Message Passing across 83 Elements (incl. 5d Ir)</text>',
        '<text x="554" y="468" font-family="Helvetica, Arial, sans-serif" font-size="9.3" font-weight="bold" fill="#c2410c">• Analytical ZBL Short-Range Nuclear Repulsion Bridge: Z_i Z_j / r_ij * phi_ZBL(r_ij)</text>',
        '<text x="554" y="484" font-family="Helvetica, Arial, sans-serif" font-size="9.3" font-weight="bold" fill="#7c2d12">• 93.8% Recovery on Severe 0.65 A Steric Clashes (Zero NaN Divergence)</text>',

        # Engine 2: HyFF
        '<rect x="544" y="510" width="404" height="98" rx="6" fill="#fef2f2" stroke="#e11d48" stroke-width="1.3"/>',
        '<text x="554" y="527" font-family="Helvetica, Arial, sans-serif" font-size="10.4" font-weight="bold" fill="#9f1239">Engine 2: Charge/Spin-Conditioned HyFF (OMol25, 167.28 MB Built-in)</text>',
        '<text x="554" y="544" font-family="Helvetica, Arial, sans-serif" font-size="9.3" fill="#1e293b">• Built-in Sweet-Spot Tiers: presto (8.9M), vivace (25.5M), moderato (132.9M); +andante</text>',
        '<text x="554" y="559" font-family="Helvetica, Arial, sans-serif" font-size="9.3" fill="#1e293b">• Element-Sparse Routing (0.05M–0.83M active/atom) + Fused Triton &amp; CUDA Graphs</text>',
        '<text x="554" y="574" font-family="Helvetica, Arial, sans-serif" font-size="9.3" font-weight="bold" fill="#be123c">• Explicit Total Charge Q in Z &amp; Spin Multiplicity M = 2S+1 (S0, T1, D0+ PES)</text>',
        '<text x="554" y="590" font-family="Helvetica, Arial, sans-serif" font-size="9.3" font-weight="bold" fill="#881337">• 97.7–113.0 atoms/ms (23.2x Faster, 0.22–1.07 GB VRAM) &amp; Ir(III) dE(T1-S0) r=0.70</text>',

        # Unified Physics Output Bus Bar
        '<rect x="544" y="616" width="404" height="24" rx="5" fill="#ffedd5" stroke="#d97706" stroke-width="1.2"/>',
        '<text x="746" y="632" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" text-anchor="middle" fill="#92400e">Unified Physics Bus:  Forces F_MLIP(z, X, Q, M) = -dE/dX  &amp;  State Energy E(Q, M)</text>',

        # =========================================================================
        # BAND 3: STAGES 4 - 5 & VERIFIED OUTPUT (y = 674 .. 944)
        # =========================================================================
        '<rect x="26" y="674" width="934" height="270" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.6"/>',
        '<rect x="26" y="674" width="934" height="26" rx="8" fill="#065f46"/>',
        '<text x="40" y="692" font-family="Helvetica, Arial, sans-serif" font-size="12.2" font-weight="bold" fill="#ffffff">LAYER 3 (Closed-Loop Alignment &amp; Physical Relaxation) — Stage 4 Online Gated-GRPO &amp; Stage 5 Tangent Relaxation</text>',

        # Stage 4 Card (x=38, y=710, w=364, h=222)
        '<rect x="38" y="710" width="364" height="222" rx="8" fill="#f0fdf4" stroke="#10b981" stroke-width="1.5"/>',
        '<rect x="38" y="710" width="364" height="24" rx="6" fill="#059669"/>',
        '<text x="220" y="726" font-family="Helvetica, Arial, sans-serif" font-size="11.4" font-weight="bold" text-anchor="middle" fill="#ffffff">Stage 4 (4_uni3dar_gated_grpo): Online Gated-GRPO</text>',
        '<text x="48" y="749" font-family="Helvetica, Arial, sans-serif" font-size="10.0" font-weight="bold" fill="#065f46">1. Group Rollout &amp; Doubly Gated Interval Reward:</text>',
        '<rect x="48" y="756" width="344" height="44" rx="5" fill="#ffffff" stroke="#6ee7b7" stroke-width="1"/>',
        '<text x="220" y="773" font-family="Helvetica, Arial, sans-serif" font-size="9.8" font-weight="bold" text-anchor="middle" fill="#047857">R(M^(g)) = G_unc(sigma) * G_phys(||F_MLIP||) * R_int(mu; I)</text>',
        '<text x="220" y="790" font-family="Helvetica, Arial, sans-serif" font-size="9.1" text-anchor="middle" fill="#334155">G_unc = exp(-max(0, sigma - tau)/tau),   G_phys = exp(-gamma max(0, F_rms - F_th))</text>',
        '<text x="48" y="818" font-family="Helvetica, Arial, sans-serif" font-size="10.0" font-weight="bold" fill="#065f46">2. Group-Relative Advantage &amp; Anti-Collapse Objective:</text>',
        '<rect x="48" y="825" width="344" height="44" rx="5" fill="#ffffff" stroke="#6ee7b7" stroke-width="1"/>',
        '<text x="220" y="842" font-family="Helvetica, Arial, sans-serif" font-size="9.6" font-weight="bold" text-anchor="middle" fill="#065f46">A^(g) = (R^(g) - mean(R)) / (std(R) + eps)</text>',
        '<text x="220" y="859" font-family="Helvetica, Arial, sans-serif" font-size="9.4" text-anchor="middle" fill="#1e293b">L_total = L_GRPO(theta) + beta_KL D_KL(pi_theta || pi_ref) + lambda L_SFT</text>',
        '<text x="48" y="888" font-family="Helvetica, Arial, sans-serif" font-size="9.5" fill="#1e293b">• Prevents Oracle epistemic reward hacking &amp; high-strain conformations</text>',
        '<text x="48" y="904" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#047857">• Updates Uni-3DAR Policy pi_theta in Closed Loop (Left Feedback Bus)</text>',

        # Stage 5 Card (x=422, y=710, w=346, h=222)
        '<rect x="422" y="710" width="346" height="222" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>',
        '<rect x="422" y="710" width="346" height="24" rx="6" fill="#1d4ed8"/>',
        '<text x="595" y="726" font-family="Helvetica, Arial, sans-serif" font-size="11.4" font-weight="bold" text-anchor="middle" fill="#ffffff">Stage 5: Tangent-Projected Iso-Property Relaxation</text>',
        '<text x="432" y="749" font-family="Helvetica, Arial, sans-serif" font-size="10.0" font-weight="bold" fill="#1e3a8a">1. Tangent-Space Force Projection (White-Box Oracles):</text>',
        '<rect x="432" y="756" width="326" height="44" rx="5" fill="#ffffff" stroke="#93c5fd" stroke-width="1"/>',
        '<text x="595" y="773" font-family="Helvetica, Arial, sans-serif" font-size="9.8" font-weight="bold" text-anchor="middle" fill="#1d4ed8">F_m^perp = F_m - [max(0, &lt;F_m, g_int&gt;) / (||g_int||^2 + eps)] g_int</text>',
        '<text x="595" y="790" font-family="Helvetica, Arial, sans-serif" font-size="9.3" text-anchor="middle" fill="#334155">X_{m+1} = X_m + eta_m * F_m^perp   (Zero Interval Violation Drift)</text>',
        '<text x="432" y="818" font-family="Helvetica, Arial, sans-serif" font-size="10.0" font-weight="bold" fill="#1e3a8a">2. Derivative-Free Trust-Region Gate &amp; Hybrid Relay:</text>',
        '<rect x="432" y="825" width="326" height="44" rx="5" fill="#ffffff" stroke="#93c5fd" stroke-width="1"/>',
        '<text x="595" y="842" font-family="Helvetica, Arial, sans-serif" font-size="9.6" font-weight="bold" text-anchor="middle" fill="#1e40af">Accepts step iff R_int(X_{m+1}; I) &gt;= R_int(X_m; I)</text>',
        '<text x="595" y="859" font-family="Helvetica, Arial, sans-serif" font-size="9.3" text-anchor="middle" fill="#334155">Supports Black-Box Uni-QSAR &amp; DPA-4 (ZBL) -&gt; HyFF (Q, M) Relay</text>',
        '<text x="432" y="888" font-family="Helvetica, Arial, sans-serif" font-size="9.5" fill="#1e293b">• Relaxes sub-grid 0.01 A octree discretization artifacts on S0 or T1 PES</text>',
        '<text x="432" y="904" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#1d4ed8">• Theorem 1: Monotonic Energy Descent + Exact Interval Preservation</text>',

        # Final Output Card (x=788, y=710, w=158, h=222)
        '<rect x="788" y="710" width="158" height="222" rx="8" fill="#fefce8" stroke="#ca8a04" stroke-width="1.6"/>',
        '<rect x="788" y="710" width="158" height="24" rx="6" fill="#ca8a04"/>',
        '<text x="867" y="726" font-family="Helvetica, Arial, sans-serif" font-size="11.2" font-weight="bold" text-anchor="middle" fill="#ffffff">Verified 3D Output</text>',
        '<text x="867" y="750" font-family="Helvetica, Arial, sans-serif" font-size="10.0" font-weight="bold" text-anchor="middle" fill="#854d0e">M* = (z, X*, Q, M)</text>',
        '<line x1="798" y1="758" x2="936" y2="758" stroke="#fde047" stroke-width="1.2"/>',
        '<text x="798" y="777" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#065f46">✓ 100.0% Closed-Shell</text>',
        '<text x="798" y="796" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#1d4ed8">✓ 93.8–100% Hit Rate</text>',
        '<text x="798" y="815" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#9a3412">✓ 0.20 eV/A RMS Force</text>',
        '<text x="798" y="834" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#7e22ce">✓ 93.8% Clash Recov.</text>',
        '<text x="798" y="853" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#be123c">✓ Zero-Shot dE(T1-S0)</text>',
        '<text x="798" y="872" font-family="Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#0f766e">✓ 97.7–113 atoms/ms</text>',
        '<rect x="798" y="884" width="138" height="36" rx="4" fill="#ffffff" stroke="#eab308" stroke-width="1"/>',
        '<text x="867" y="900" font-family="Helvetica, Arial, sans-serif" font-size="9.2" font-weight="bold" text-anchor="middle" fill="#713f12">Bitwise Deterministic</text>',
        '<text x="867" y="914" font-family="Helvetica, Arial, sans-serif" font-size="8.8" text-anchor="middle" fill="#475569">SHA-256 Seed Lock</text>',

        # =========================================================================
        # INTER-BAND CLEAN BUS ARROWS (Zero Crossing!)
        # =========================================================================
        # 1. Stage 3 -> Layer 2A/2B (Candidate 3D Conformers M_0)
        '<path d="M 813 332 L 813 356 L 460 356 L 460 366" fill="none" stroke="#7c3aed" stroke-width="2.0" marker-end="url(#arr-purple)"/>',
        '<path d="M 813 332 L 813 366" fill="none" stroke="#7c3aed" stroke-width="2.0" marker-end="url(#arr-purple)"/>',
        '<text x="635" y="352" font-family="Helvetica, Arial, sans-serif" font-size="9.8" font-weight="bold" text-anchor="middle" fill="#5b21b6">Decoded 3D Candidates M = (z, X, Q, M)</text>',

        # 2. Oracle Bus (Layer 2A) -> Stage 4 & Stage 5
        '<path d="M 180 640 L 180 708" fill="none" stroke="#0d9488" stroke-width="2.2" marker-end="url(#arr-teal)"/>',
        '<path d="M 380 640 L 380 658 L 530 658 L 530 708" fill="none" stroke="#0d9488" stroke-width="2.2" marker-end="url(#arr-teal)"/>',
        '<text x="115" y="662" font-family="Helvetica, Arial, sans-serif" font-size="9.6" font-weight="bold" fill="#0f766e">mu_k, sigma_k</text>',

        # 3. Physics Bus (Layer 2B) -> Stage 4 & Stage 5
        '<path d="M 640 640 L 640 666 L 310 666 L 310 708" fill="none" stroke="#d97706" stroke-width="2.2" marker-end="url(#arr-amber)"/>',
        '<path d="M 680 640 L 680 708" fill="none" stroke="#d97706" stroke-width="2.2" marker-end="url(#arr-amber)"/>',
        '<text x="745" y="662" font-family="Helvetica, Arial, sans-serif" font-size="9.6" font-weight="bold" fill="#b45309">F_MLIP(Q,M), E(Q,M)</text>',

        # 4. Stage 4 -> Stage 5 -> Verified Output
        '<line x1="402" y1="820" x2="420" y2="820" stroke="#059669" stroke-width="2.4" marker-end="url(#arr-teal)"/>',
        '<line x1="768" y1="820" x2="786" y2="820" stroke="#2563eb" stroke-width="2.4" marker-end="url(#arr-blue)"/>',

        # 5. Stage 4 Policy Update Feedback Loop along outer-left margin (x=12) up to y=101 into Stages 1 & 2
        '<path d="M 38 900 L 12 900 L 12 101 L 461 101 L 461 107" fill="none" stroke="#059669" stroke-width="2.2" stroke-dasharray="5,3" marker-end="url(#arr-teal)"/>',
        '<text transform="translate(21, 505) rotate(-90)" font-family="Helvetica, Arial, sans-serif" font-size="9.8" font-weight="bold" text-anchor="middle" fill="#047857">Closed-Loop Policy Gradient Update (pi_theta)</text>',

        '</svg>'
    ]
    save_svg_and_png("\n".join(p), "fig0_orbit3d_system_architecture")


def build_fig1():
    W, H = 900, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" font-family="Helvetica, Arial, sans-serif" font-size="15.5" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 1: ORBIT-3D vs. Continuous 3D Flow Matching (PropMolFlow + MLIP Orthogonal Projection)</text>'
    ]
    cats = ["OOD Gap\n[0.18, 0.23] Ha", "LUMO Target\n[0.02, 0.05] Ha", "Joint 3-Prop OLED\n(Gap+HOMO+LUMO)"]

    # Top-Left: Panel A
    ser_a = [
        ("PropMolFlow + Naive GVP (No MLIP)", [75.0, 62.5, 77.1], "#94a3b8", "#334155"),
        ("PropMolFlow + DPA-4 Orthogonal Proj.", [95.8, 93.8, 95.8], "#3b82f6", "#1e3a8a"),
        ("ORBIT-3D (Uni-3DAR + GRPO + DPA-4)", [100.0, 100.0, 100.0], "#10b981", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(68, 75, 360, 320, "(a) 3D Molecular Valency & Stability (%)", "Molecular Stability (%)", cats, ser_a, 40, 125, [40, 60, 80, 100], fmt="{:.1f}%"))

    # Top-Right: Panel B
    ser_b = [
        ("PropMolFlow + DPA4-Mini", [0.608, 0.402, 0.382], "#60a5fa", "#1e3a8a"),
        ("PropMolFlow + DPA4-Plus", [0.625, 0.436, 0.413], "#2563eb", "#1e3a8a"),
        ("ORBIT-3D + DPA4-Mini (Ours)", [0.366, 0.204, 0.385], "#34d399", "#064e3b"),
        ("ORBIT-3D + DPA4-Plus (Ours)", [0.362, 0.198, 0.376], "#059669", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(508, 75, 360, 320, "(b) Residual Atomic Force (eV/A, Lower=Better)", "DPA-4 RMS Force (eV/A)", cats, ser_b, 0.0, 0.92, [0.0, 0.2, 0.4, 0.6, 0.8], fmt="{:.3f}"))

    # Bottom-Left: Panel C
    ser_c = [
        ("PropMolFlow (No MLIP)", [75.0, 62.5, 77.1], "#94a3b8", "#334155"),
        ("PropMolFlow + DPA-4 Orthogonal Proj.", [89.8, 93.8, 95.8], "#3b82f6", "#1e3a8a"),
        ("ORBIT-3D + DPA-4 (Ours)", [93.8, 100.0, 100.0], "#f59e0b", "#78350f"),
    ]
    parts.append(draw_grouped_bar_panel(68, 490, 360, 320, "(c) Valid & In-Interval Success Rate (%)", "Valid & In-Interval (%)", cats, ser_c, 40, 125, [40, 60, 80, 100], fmt="{:.1f}%"))

    # Bottom-Right: Panel D (Bond Length MAE in mA)
    ser_d = [
        ("PropMolFlow (No MLIP)", [31.9, 32.0, 29.7], "#94a3b8", "#334155"),
        ("PropMolFlow + DPA4-Mini", [22.0, 22.8, 19.6], "#3b82f6", "#1e3a8a"),
        ("ORBIT-3D + DPA4-Mini (Ours)", [19.8, 19.5, 19.8], "#10b981", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(508, 490, 360, 320, "(d) 3D Bond Length MAE (mA, Lower=Better)", "Bond Length MAE (mA)", cats, ser_d, 0, 44, [0, 10, 20, 30, 40], fmt="{:.1f}"))

    parts.append("</svg>")
    save_svg_and_png("\n".join(parts), "fig1_orbit3d_vs_propmolflow")


def build_fig2():
    W, H = 900, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" font-family="Helvetica, Arial, sans-serif" font-size="15.5" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 2: Progressive Five-Stage Ablation Study of the ORBIT-3D Closed-Loop Architecture</text>'
    ]
    stages = [
        "Stage 1\nUncond\nOctree",
        "Stage 2\nAdaLN-Zero\nInterval SFT",
        "Stage 3\nSuffix [CLS]\nSelf-Rerank",
        "Stage 4\nOnline\nGated-GRPO",
        "Stage 5 (Fast)\nHyFF-moderato\nRelaxation",
        "Stage 5 (ZBL)\nDPA-4 (OMol25)\nGRPO+Relax",
    ]
    lines_a = [
        ("LUMO Target Interval Hit Rate (%)", [12.5, 27.1, 35.4, 54.2, 100.0, 100.0], "#10b981", ""),
        ("Joint 3-Prop OLED (Gap+HOMO+LUMO) (%)", [6.3, 16.7, 29.2, 33.3, 97.9, 100.0], "#2563eb", ""),
        ("OOD HOMO-LUMO Gap Interval (%)", [2.1, 14.6, 16.7, 27.1, 93.8, 93.8], "#f59e0b", ""),
        ("HOMO Target Interval Hit Rate (%)", [0.0, 6.3, 6.3, 20.8, 91.7, 91.7], "#8b5cf6", ""),
    ]
    parts.append(draw_line_panel(75, 75, 780, 320, "(a) Monotonic Target Interval Hit-Rate Gain Across All 5 Stages (%)", "Interval Hit Rate (%)", stages, lines_a, 0, 120, [0, 20, 40, 60, 80, 100]))

    lines_b = [
        ("LUMO Interval Violation Error (eV)", [1.575, 1.258, 0.472, 0.113, 0.000, 0.000], "#059669", ""),
        ("Joint OLED Mean Interval Error (eV)", [1.286, 0.806, 0.504, 0.131, 0.004, 0.000], "#2563eb", ""),
        ("LUMO Residual Quantum Force (eV/A)", [0.762, 0.831, 0.910, 1.125, 0.214, 0.198], "#dc2626", "6,4"),
        ("Joint OLED Residual Quantum Force (eV/A)", [0.762, 0.692, 0.617, 0.872, 0.382, 0.376], "#ea580c", "6,4"),
    ]
    parts.append(draw_line_panel(75, 490, 780, 320, "(b) Synergy of Stage 4 (Property Error Collapse) & Stage 5 (Physical Force Relaxation)", "Error (eV) / Force (eV/A)", stages, lines_b, 0.0, 1.85, [0.0, 0.4, 0.8, 1.2, 1.6]))

    parts.append("</svg>")
    save_svg_and_png("\n".join(parts), "fig2_five_stage_ablation")


def build_fig3():
    W, H = 900, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" font-family="Helvetica, Arial, sans-serif" font-size="15.5" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 3: Scaling Law and Container Footprint Across the Built-In DPA-4 (OMol25) Model Family</text>'
    ]

    x0, y0, w, h = 75, 75, 780, 320
    parts.append(f'<text x="{x0 + w/2:.1f}" y="{y0 - 14:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="14.5" font-weight="bold" text-anchor="middle" fill="#0f172a">(a) Speed-Accuracy Pareto Frontier (Stage 5 Batch Relaxation of 48 Molecules)</text>')
    parts.append(f'<text transform="translate({x0 - 44:.1f},{y0 + h/2:.1f}) rotate(-90)" font-family="Helvetica, Arial, sans-serif" font-size="12.5" font-weight="bold" text-anchor="middle" fill="#1e293b">Residual DPA-4 RMS Force (eV/A) Lower=Better</text>')
    parts.append(f'<text x="{x0 + w/2:.1f}" y="{y0 + h + 36:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#1e293b">Stage 5 Batch Conformer Relaxation Time (sec) Lower=Faster</text>')

    x_min, x_max = 3.0, 29.0
    y_min, y_max = 0.16, 0.46
    for yt in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        yy = y0 + h - ((yt - y_min) / (y_max - y_min)) * h
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,4"/>')
        parts.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="11" text-anchor="end" fill="#334155">{yt:.2f}</text>')
    for xt in [5, 10, 15, 20, 25]:
        xx = x0 + ((xt - x_min) / (x_max - x_min)) * w
        parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + h}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,4"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0 + h + 17:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="11" text-anchor="middle" fill="#334155">{xt}s</text>')

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" stroke="#334155" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="#334155" stroke-width="1.5"/>')

    models = [
        ("DPA-4-Nano (7M, 1.95MB)", 5.60, 0.231, "#10b981", 10, -10),
        ("DPA-4-Mini (12M, 2.62MB)", 5.89, 0.204, "#059669", 12, 16),
        ("DPA-4-Neo (23M, 4.42MB)", 9.01, 0.199, "#0284c7", 10, -12),
        ("DPA-4-Air (100M, 19.86MB)", 13.90, 0.198, "#2563eb", 10, 16),
        ("DPA-4-Plus (172M, 34.01MB)", 26.09, 0.198, "#7c3aed", -150, -12),
        ("DPA-4-Air (OMat24 Inorganics)", 12.98, 0.432, "#ef4444", 12, 5),
    ]
    pareto_pts = []
    for name, tx, fy, col, dx, dy in models[:5]:
        px = x0 + ((tx - x_min) / (x_max - x_min)) * w
        py = y0 + h - ((fy - y_min) / (y_max - y_min)) * h
        pareto_pts.append(f"{px:.1f},{py:.1f}")
    parts.append(f'<polyline fill="none" stroke="#059669" stroke-width="2.4" stroke-dasharray="6,4" points="{" ".join(pareto_pts)}"/>')

    for name, tx, fy, col, dx, dy in models:
        px = x0 + ((tx - x_min) / (x_max - x_min)) * w
        py = y0 + h - ((fy - y_min) / (y_max - y_min)) * h
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="{col}" stroke="#0f172a" stroke-width="1.3"/>')
        parts.append(f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="{col}">{esc(name)}</text>')

    scales = ["DPA-4 Nano\n(7M)", "DPA-4 Mini\n(12M)", "DPA-4 Neo\n(23M)", "DPA-4 Air\n(100M)", "DPA-4 Plus\n(172M)"]
    ser_b = [
        ("Packed Container Weight (MB)", [1.95, 2.62, 4.42, 19.86, 34.01], "#3b82f6", "#1e3a8a"),
        ("4-Task Mean Hit Rate (%)", [95.31, 95.83, 95.83, 95.83, 96.36], "#10b981", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(75, 490, 780, 320, "(b) Built-In Weight Size (MB, Total 62.86 MB) vs. 4-Task Mean Hit Rate (%)", "MB / Hit Rate (%)", scales, ser_b, 0, 120, [0, 20, 40, 60, 80, 100], fmt="{:.2f}"))

    parts.append("</svg>")
    save_svg_and_png("\n".join(parts), "fig3_dpa4_scaling_pareto")


def build_fig4():
    W, H = 900, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" font-family="Helvetica, Arial, sans-serif" font-size="15.5" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 4: Online Gated-GRPO Regularization Dynamics &amp; Pluggable Multi-Modal Oracle Ecosystem</text>'
    ]

    step_labels = ["Step 0\n(SFT)", "Step 8", "Step 16", "Step 24\n(Optimal)", "Step 32", "Step 40"]
    lines_grpo = [
        ("ORBIT-3D Gated-GRPO (Unc + MLIP Force Gate + KL/SFT)", [27.1, 62.5, 81.3, 93.8, 93.8, 93.8], "#10b981", ""),
        ("GRPO w/o MLIP Force Gate (Unc + KL Only)", [27.1, 58.3, 72.9, 77.1, 75.0, 72.9], "#3b82f6", ""),
        ("Un-anchored REINFORCE (Reward Hacking Collapse)", [27.1, 64.6, 75.0, 60.4, 41.7, 25.0], "#ef4444", "6,4"),
    ]
    parts.append(draw_line_panel(75, 75, 780, 320, "(a) Online GRPO Convergence vs. Reward Hacking Prevention", "True OOD Interval Hit Rate (%)", step_labels, lines_grpo, 0, 120, [0, 20, 40, 60, 80, 100]))

    oracle_cats = [
        "Mode A (Mainline):\n5-Fold 3D Uni-Mol",
        "Mode B (ZBL Unified):\n5-Fold Native DPA-4",
        "Mode C (Charge/Spin):\n5-Fold Native HyFF",
        "Mode D (Enterprise):\nFull-Matrix Uni-QSAR",
    ]
    ser_oracles = [
        ("5-Fold Cross-Validation R2 (%)", [95.6, 96.4, 96.1, 96.1], "#6366f1", "#1e1b4b"),
        ("Stage 5 Target Interval Hit Rate (%)", [97.9, 100.0, 100.0, 97.9], "#10b981", "#064e3b"),
        ("3D Closed-Shell Stability (%)", [100.0, 100.0, 100.0, 100.0], "#f59e0b", "#78350f"),
    ]
    parts.append(draw_grouped_bar_panel(75, 490, 780, 320, "(b) Open-Source Mainline (Uni-Mol / DPA-4 / HyFF) & Full-Matrix Uni-QSAR (%)", "Metric (%)", oracle_cats, ser_oracles, 80, 108, [80, 85, 90, 95, 100], fmt="{:.1f}%"))

    parts.append("</svg>")
    save_svg_and_png("\n".join(parts), "fig4_pluggable_oracles_relaxation")


def build_fig5():
    W, H = 900, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" font-family="Helvetica, Arial, sans-serif" font-size="15.5" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 5: Wall-Clock Latency Breakdown, GPU Throughput &amp; Hardware VRAM Profile</text>'
    ]

    # Top-Left: Panel A — Single-Batch (48 mols) MLIP Force Latency (ms) & Throughput (atoms/ms)
    mlip_cats = ["HyFF-mod.\n(132.9MB)", "DPA4-Nano\n(1.95MB)", "DPA4-Mini\n(2.62MB)", "DPA4-Neo\n(4.42MB)", "DPA4-Air\n(19.9MB)", "DPA4-Plus\n(34.0MB)"]
    ser_a = [
        ("Per-Molecule Force Latency (ms/mol) Lower=Faster", [0.18, 0.91, 0.95, 1.31, 2.33, 4.70], "#3b82f6", "#1e3a8a"),
        ("Saturated Throughput (10x atoms/ms) Higher=Faster", [9.77, 5.60, 4.25, 1.84, 0.97, 0.44], "#10b981", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(68, 75, 360, 320, "(a) MLIP Force Latency & Throughput (A800)", "ms/mol  |  10x atoms/ms", mlip_cats, ser_a, 0.0, 12.5, [0, 2, 4, 6, 8, 10], fmt="{:.2f}"))

    # Top-Right: Panel B — End-to-End 48-Molecule Generation Time Across Stages/Paradigms (sec)
    stage_cats = ["PropMolFlow\n+DPA4-Mini", "Stage 1\nUncond", "Stage 2\nAdaLN SFT", "Stage 3\nRerank K=4", "Stage 4\nGated-GRPO", "Stage 5\n+DPA4/HyFF"]
    ser_b = [
        ("AR / ODE Sampling Time (sec / 48 mols)", [7.87, 16.01, 20.61, 31.02, 42.40, 42.40], "#6366f1", "#1e1b4b"),
        ("Total Time incl. Stage 5 Relax (sec / 48 mols)", [7.87, 16.01, 20.61, 31.02, 42.40, 48.30], "#f59e0b", "#78350f"),
    ]
    parts.append(draw_grouped_bar_panel(508, 75, 360, 320, "(b) 48-Molecule Generation Wall-Clock Time (s)", "Wall-Clock Time (sec)", stage_cats, ser_b, 0.0, 64.0, [0, 10, 20, 30, 40, 50, 60], fmt="{:.1f}s"))

    # Bottom-Left: Panel C — Training Time Breakdown by Stage & Oracle Mode (sec)
    train_cats = ["Mode A:\nUni-Mol", "Mode B:\nDPA-4", "Mode C:\nHyFF", "Mode D:\nUni-QSAR"]
    ser_c = [
        ("Oracle 5-Fold Fit / Load Time (sec)", [34.5, 22.8, 18.4, 1.8], "#0ea5e9", "#0c4a6e"),
        ("Stage 2+3 SFT Training (15 Epochs, sec)", [118.0, 118.0, 118.0, 118.0], "#8b5cf6", "#4c1d95"),
        ("Stage 4 Online Gated-GRPO (30 Steps, sec)", [254.0, 276.4, 212.0, 268.0], "#10b981", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(68, 490, 360, 320, "(c) End-to-End Training Time Breakdown (sec)", "Training Time (sec)", train_cats, ser_c, 0.0, 390.0, [0, 50, 100, 150, 200, 250, 300], fmt="{:.1f}s"))

    # Bottom-Right: Panel D — Peak GPU VRAM Footprint Across Hardware Profiles (GB)
    vram_cats = ["Ultra-Light\n(HyFF-presto/mod)", "Edge / Entry\n(DPA4-Nano/Mini)", "Flagship Server\n(DPA4-Plus / Multi)"]
    ser_d = [
        ("Inference Peak GPU VRAM (GB)", [1.1, 2.6, 6.8], "#34d399", "#064e3b"),
        ("Full SFT + GRPO Training Peak VRAM (GB)", [3.8, 5.4, 14.2], "#f43f5e", "#881337"),
    ]
    parts.append(draw_grouped_bar_panel(508, 490, 360, 320, "(d) Peak GPU VRAM Requirement (GB)", "Peak GPU VRAM (GB)", vram_cats, ser_d, 0.0, 18.0, [0, 3, 6, 9, 12, 15, 18], fmt="{:.1f}G"))

    parts.append("</svg>")
    save_svg_and_png("\n".join(parts), "fig5_latency_and_hardware_profile")


def build_fig6():
    W, H = 900, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" font-family="Helvetica, Arial, sans-serif" font-size="15.5" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 6: Dataset Statistical Distributions (QM9 / rQM9 &amp; 3D-SAT-OLED Organometallic Complexes)</text>'
    ]

    # Top-Left: Panel A — QM9 Empirical Prior Coverage (%) vs. ORBIT-3D Stage 5 Hit Rate (%)
    task_cats = ["OOD Gap\n[0.18, 0.23]Ha", "LUMO Level\n[0.02, 0.05]Ha", "HOMO Level\n[-0.24,-0.21]Ha", "Joint 3-Prop\nOLED Window"]
    ser_a = [
        ("QM9 Dataset Prior Coverage (%)", [31.08, 18.34, 39.31, 16.88], "#94a3b8", "#334155"),
        ("Stage 1 Uncond Sampling (%)", [8.33, 12.50, 29.17, 6.25], "#f59e0b", "#78350f"),
        ("ORBIT-3D Stage 5 Hit Rate (%)", [93.75, 100.00, 91.67, 100.00], "#10b981", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(68, 75, 360, 320, "(a) QM9 Prior Density vs. ORBIT-3D Hit Rate (%)", "Percentage of Molecules (%)", task_cats, ser_a, 0.0, 138.0, [0, 20, 40, 60, 80, 100], fmt="{:.1f}%"))

    # Top-Right: Panel B — Atom Count Distribution: QM9 (N=3..29) vs. 3D-SAT-OLED Ir(III) (N=42..98)
    size_bins = ["N<=14\n(Small)", "N=15..17\n(QM9 Med)", "N=18..20\n(QM9 Mode)", "N=21..29\n(QM9 Lrg)", "N=42..60\n(Ir-OLED)", "N=61..98\n(Ir-OLED)"]
    ser_b = [
        ("rQM9 Train Split (100,000 Organic Mols, %)", [11.57, 31.42, 36.68, 20.33, 0.00, 0.00], "#3b82f6", "#1e3a8a"),
        ("3D-SAT-OLED Split (1,000 Ir(III) Complexes, %)", [0.00, 0.00, 0.00, 0.00, 41.20, 58.80], "#8b5cf6", "#4c1d95"),
    ]
    parts.append(draw_grouped_bar_panel(508, 75, 360, 320, "(b) Molecular Size (N_atoms) Distribution (%)", "Dataset Fraction (%)", size_bins, ser_b, 0.0, 76.0, [0, 10, 20, 30, 40, 50, 60], fmt="{:.1f}%"))

    # Bottom-Left: Panel C — Element Composition (%) Across QM9 vs. 3D-SAT-OLED
    elem_cats = ["H\n(Z=1)", "C\n(Z=6)", "N\n(Z=7)", "O\n(Z=8)", "F\n(Z=9)", "Ir / S / Cl\n(Z=77,16,17)"]
    ser_c = [
        ("QM9 / rQM9 Organic (133,885 Mols, %)", [51.13, 35.16, 5.78, 7.80, 0.14, 0.00], "#0ea5e9", "#0c4a6e"),
        ("3D-SAT-OLED Organometallic (1,000 Mols, %)", [37.61, 45.36, 11.98, 1.37, 1.82, 1.86], "#ec4899", "#831843"),
    ]
    parts.append(draw_grouped_bar_panel(68, 490, 360, 320, "(c) Elemental Composition Comparison (%)", "Atomic Fraction (%)", elem_cats, ser_c, 0.0, 64.0, [0, 10, 20, 30, 40, 50], fmt="{:.1f}%"))

    # Bottom-Right: Panel D — Frontier Orbital Energy Distributions (Mean |eV| & Std eV)
    prop_cats = ["|HOMO|\n(eV)", "|LUMO|\n(eV)", "Bandgap Eg\n(eV)", "Triplet E_ad\n(eV)"]
    ser_d = [
        ("QM9 DFT B3LYP/6-31G(2df,p) Mean (eV)", [6.53, 0.30, 6.83, 0.00], "#3b82f6", "#1e3a8a"),
        ("3D-SAT-OLED B3LYP/LANL2DZ Mean (eV)", [5.53, 2.74, 2.79, 2.66], "#10b981", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(508, 490, 360, 320, "(d) Quantum Energy Scale Comparison (eV)", "Energy Magnitude (eV)", prop_cats, ser_d, 0.0, 8.5, [0, 1, 2, 3, 4, 5, 6, 7], fmt="{:.2f}"))

    parts.append("</svg>")
    save_svg_and_png("\n".join(parts), "fig6_dataset_distributions")


def build_fig7():
    W, H = 900, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" font-family="Helvetica, Arial, sans-serif" font-size="15.2" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 7: HyFF vs. DPA-4 Quantum Force Field Benchmark (Throughput, ZBL Clash, Spin T1-S0 &amp; Cross-Referee)</text>'
    ]

    # Top-Left: Panel A — Saturated Throughput (atoms/ms) & Peak VRAM (GB)
    scale_cats = ["DPA4\nNano/Mini", "DPA4\nAir/Plus", "HyFF\npresto", "HyFF\nvivace", "HyFF\nmoderato", "HyFF\nandante"]
    ser_a = [
        ("Saturated Throughput (atoms/ms, higher is faster)", [46.6, 4.2, 113.0, 107.4, 97.7, 5.0], "#10b981", "#064e3b"),
        ("Peak GPU VRAM Footprint (GB, lower is lighter)", [1.1, 20.8, 0.2, 0.4, 1.1, 15.3], "#f43f5e", "#881337"),
    ]
    parts.append(draw_grouped_bar_panel(68, 75, 360, 320, "(a) Throughput (atoms/ms) & Peak VRAM (GB)", "Magnitude (atoms/ms or GB)", scale_cats, ser_a, 0.0, 145.0, [0, 20, 40, 60, 80, 100, 120], fmt="{:.1f}"))

    # Top-Right: Panel B — Short-Range Steric Clash Recovery Rate (%) Across Overlap Distances
    clash_cats = ["d=0.45 A\nPush Out%", "d=0.65 A\nPush Out%", "d=0.65 A\nRecov %", "d=0.85 A\nRecov %"]
    ser_b = [
        ("DPA4-Mini (Analytical ZBL Nuclear Repulsion)", [75.0, 93.8, 93.8, 100.0], "#3b82f6", "#1e3a8a"),
        ("HyFF-moderato (Learned Element-Sparse Radial)", [62.5, 87.5, 31.2, 75.0], "#10b981", "#064e3b"),
        ("HyFF-largo (1.2B Latent Charge Singularity)", [25.0, 50.0, 0.0, 6.2], "#ef4444", "#7f1d1d"),
    ]
    parts.append(draw_grouped_bar_panel(508, 75, 360, 320, "(b) Short-Range Steric Clash Resilience (%)", "Success / Recovery Rate (%)", clash_cats, ser_b, 0.0, 138.0, [0, 20, 40, 60, 80, 100], fmt="{:.1f}%"))

    # Bottom-Left: Panel C — Zero-Shot Ir(III) Spin Splitting Delta E(T1 - S0) vs. DFT E_ad (2.652 eV)
    spin_cats = ["DPA-4\n(Spin-Blind)", "HyFF\npresto", "HyFF\nvivace", "HyFF\nmoderato", "HyFF\nandante", "HyFF\nlargo"]
    ser_c = [
        ("Zero-Shot Spin Gap Accuracy (1 - MAE/E_ad, %)", [0.0, 75.8, 89.9, 93.5, 97.9, 98.1], "#8b5cf6", "#4c1d95"),
        ("Zero-Shot Pearson Correlation r vs. DFT E_ad (%)", [0.0, 29.2, 31.5, 45.2, 69.8, 72.0], "#0ea5e9", "#0c4a6e"),
    ]
    parts.append(draw_grouped_bar_panel(68, 490, 360, 320, "(c) Zero-Shot Ir(III) T1-S0 Spin Splitting vs. DFT", "Accuracy / Correlation (%)", spin_cats, ser_c, 0.0, 130.0, [0, 20, 40, 60, 80, 100], fmt="{:.1f}%"))

    # Bottom-Right: Panel D — Dual Independent Cross-MLIP Referees on De Novo QM9 OOD Generation
    ref_cats = ["No MLIP\n(Naive)", "DPA4-Mini\nGuided", "DPA4-Plus\nGuided", "HyFF-vivace\nGuided", "HyFF-mod.\nGuided"]
    ser_d = [
        ("DPA4-Plus Referee Median Force (eV/A)", [0.927, 0.341, 0.331, 0.347, 0.338], "#3b82f6", "#1e3a8a"),
        ("HyFF-andante Referee Median Force (eV/A)", [0.930, 0.341, 0.329, 0.344, 0.345], "#10b981", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(508, 490, 360, 320, "(d) Cross-MLIP Dual-Referee Force Evaluation (eV/A)", "Median Residual Force (eV/A, lower is better)", ref_cats, ser_d, 0.0, 1.25, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fmt="{:.3f}"))

    parts.append("</svg>")
    save_svg_and_png("\n".join(parts), "fig7_hyff_vs_dpa4_pareto_and_spin")


def build_fig8():
    W, H = 900, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" font-family="Helvetica, Arial, sans-serif" font-size="14.5" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 8: Optoelectronic Frameworks (OCNet, OSCFF, MIM-Flow, O2-GEN) vs. ORBIT-3D Benchmark</text>'
    ]

    # Top-Left: Panel A — Inverse Design Paradigms (O2-GEN, MIM-Flow vs. ORBIT-3D)
    inv_cats = [
        "O2-GEN\n(K=4)",
        "O2-GEN\n(K=16)",
        "MIM-Flow\n(SubTB)",
        "MIM-Flow\n+Stage 5",
        "ORBIT-3D\n(HyFF)",
        "ORBIT-3D\n(DPA-4)"
    ]
    ser_a = [
        ("Sparse Narrow Pareto Hit (2.14% Prior)", [31.3, 60.4, 79.2, 97.9, 97.9, 100.0], "#8b5cf6", "#4c1d95"),
        ("Low Quantum Strain (max|F|<=0.5 eV/A)", [35.4, 39.6, 47.9, 95.8, 95.8, 95.8], "#10b981", "#064e3b"),
    ]
    parts.append(draw_grouped_bar_panel(68, 75, 360, 320, "(a) Inverse Design & Sparse Pareto Navigation", "Success / Yield Rate (%)", inv_cats, ser_a, 0.0, 138.0, [0, 20, 40, 60, 80, 100], fmt="{:.1f}%"))

    # Top-Right: Panel B — Force-Field Evolution (OSCFF JCIM 2025 vs DPA-4 OMol25 vs HyFF OMol25 vs ORBIT-3D Relay)
    ff_cats = [
        "OSCFF\n(JCIM 2025)",
        "DPA-4 Mini\n(OMol25+ZBL)",
        "HyFF mod.\n(OMol25 Q,M)",
        "ORBIT Relay\n(DPA4+HyFF)"
    ]
    ser_b = [
        ("Conjugated Torsion Fidelity (100 - MAE meV)", [85.8, 90.2, 90.6, 90.9], "#0ea5e9", "#0c4a6e"),
        ("Short-Range 0.65 A Steric Clash Recovery (%)", [18.8, 93.8, 31.3, 95.8], "#f59e0b", "#78350f"),
        ("Ir(III) T1-S0 Spin Splitting Accuracy (%)", [0.0, 0.0, 94.4, 94.6], "#ec4899", "#831843"),
    ]
    parts.append(draw_grouped_bar_panel(508, 75, 360, 320, "(b) Force Fields: OSCFF vs. DPA-4 & HyFF", "Capability / Recovery Score (%)", ff_cats, ser_b, 0.0, 142.0, [0, 20, 40, 60, 80, 100], fmt="{:.1f}%"))

    # Bottom-Left: Panel C — Forward Virtual Characterization Oracles (OCNet npj Comput. Mater. 2025 vs Pluggable Oracles)
    ora_cats = [
        "OCNet\n(npj 2025)",
        "Uni-Mol\n(Mode A)",
        "DPA-4 SeZM\n(Mode B)",
        "HyFF State\n(Mode C)",
        "Uni-QSAR\n(Mode D)"
    ]
    ser_c = [
        ("Conjugated Optoelectronic Gap R (x100%)", [97.4, 96.8, 95.4, 96.1, 97.8], "#6366f1", "#312e81"),
        ("Orbital Precision Score (100 - MAE meV)", [58.5, 53.2, 47.6, 50.9, 60.8], "#14b8a6", "#134e4a"),
    ]
    parts.append(draw_grouped_bar_panel(68, 490, 360, 320, "(c) Forward Virtual Characterization Oracles", "Correlation / Precision Score (%)", ora_cats, ser_c, 0.0, 132.0, [0, 20, 40, 60, 80, 100], fmt="{:.1f}%"))

    # Bottom-Right: Panel D — Residual Quantum Force Strain & Closed-Shell Validity Across Optoelectronic Frameworks
    str_cats = [
        "O2-GEN\n(JACS Au)",
        "MIM-Flow\n(SubTB)",
        "MIM-Flow\n+Stage 5",
        "ORBIT-3D\n(HyFF)",
        "ORBIT-3D\n(DPA-4)"
    ]
    ser_d = [
        ("Mean Max Force on DPA4-Plus Referee (eV/A)", [0.782, 0.642, 0.208, 0.214, 0.204], "#f43f5e", "#881337"),
        ("End-to-End Per-Molecule Latency (s/mol)", [0.280, 0.540, 0.690, 0.620, 1.010], "#0284c7", "#075985"),
    ]
    parts.append(draw_grouped_bar_panel(508, 490, 360, 320, "(d) Quantum Force Strain (eV/A) & Latency (s/mol)", "Force (eV/A) or Latency (s/mol)", str_cats, ser_d, 0.0, 1.35, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fmt="{:.2f}"))

    parts.append("</svg>")
    save_svg_and_png("\n".join(parts), "fig8_optoelectronic_paradigms_and_forcefields")


if __name__ == "__main__":
    build_fig0()
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    build_fig6()
    build_fig7()
    build_fig8()
