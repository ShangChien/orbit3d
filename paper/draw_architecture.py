#!/usr/bin/env python3
"""Draw Figure 1 (Uni-Loom architecture) as editable SVG, vector PDF, and PNG preview.

Modelled after the clean minimalist editorial style of the legacy ORBIT-3D system diagram:
pure white canvas, airy whitespace, hairline rounded stage cards, custom geometric vector
icons, and clean orthogonal dashed/solid feedback paths.
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import fitz  # PyMuPDF
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

WIDTH, HEIGHT = 1280, 640
INK = "#172A3A"
MUTED = "#526575"
RULE = "#D7E0E7"
SLATE = "#7C8D9B"
BLUE = "#3F6C9E"
TEAL = "#247C75"
AMBER = "#B46B35"
STEM = "fig1_uniloom_architecture"

ARIAL_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
ARIAL_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def _register_fonts() -> tuple[str, str]:
    if ARIAL_REGULAR.exists() and ARIAL_BOLD.exists():
        try:
            pdfmetrics.registerFont(TTFont("SansRegular", str(ARIAL_REGULAR)))
            pdfmetrics.registerFont(TTFont("SansBold", str(ARIAL_BOLD)))
            return "SansRegular", "SansBold"
        except Exception:
            pass
    return "Helvetica", "Helvetica-Bold"


class Figure:
    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.font_regular, self.font_bold = _register_fonts()
        self.pdf = canvas.Canvas(str(output_dir / f"{STEM}.pdf"), pagesize=(WIDTH, HEIGHT))
        self.pdf.setTitle("Uni-Loom predictor-agnostic weaving and grounded RSI architecture")
        self.pdf.setAuthor("Uni-Loom authors")
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
            '<title id="title">Uni-Loom architecture</title>',
            '<desc id="desc">External property models, differentiable 3D predictors, and MLIPs '
            'plug into a declarative Critic Bus without retraining. The five-stage forward pipeline '
            '(Weaving spec, Octree policy, Neutral graph, Critic bus, 4-zone atlas) feeds trusted '
            'intervals to scale-decoupled inner RSI and frozen blindspots to outer-loop DFT acquisition '
            'and zero-retrain Bayesian residual calibration.</desc>',
        ]
        self.rect(0, 0, WIDTH, HEIGHT, fill="#FFFFFF", stroke=None, radius=0, line_width=0)

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke=RULE, radius=12, line_width=1.5):
        self.svg.append(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{radius:g}" '
            f'fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{line_width:g}"/>'
        )
        self.pdf.setLineWidth(line_width)
        if fill:
            self.pdf.setFillColor(HexColor(fill))
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
        if radius > 0:
            self.pdf.roundRect(x, HEIGHT - y - h, w, h, radius, stroke=bool(stroke), fill=bool(fill))
        else:
            self.pdf.rect(x, HEIGHT - y - h, w, h, stroke=bool(stroke), fill=bool(fill))

    def polygon(self, points, fill, stroke=None, line_width=1.0, opacity=1.0):
        pts = [(float(x), float(y)) for x, y in points]
        coords = " ".join(f"{x:g},{y:g}" for x, y in pts)
        op_attr = f' fill-opacity="{opacity:g}"' if opacity < 1.0 else ""
        stroke_attr = f' stroke="{stroke}" stroke-width="{line_width:g}"' if stroke else ""
        self.svg.append(f'<polygon points="{coords}" fill="{fill}"{op_attr}{stroke_attr}/>')
        self.pdf.saveState()
        self.pdf.setFillColor(HexColor(fill))
        if opacity < 1.0 and hasattr(self.pdf, "setFillAlpha"):
            self.pdf.setFillAlpha(opacity)
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
            self.pdf.setLineWidth(line_width)
        p = self.pdf.beginPath()
        p.moveTo(pts[0][0], HEIGHT - pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, HEIGHT - y)
        p.close()
        self.pdf.drawPath(p, stroke=int(bool(stroke)), fill=1)
        self.pdf.restoreState()

    def line(self, points, color=RULE, width=2.0, dash=False, arrow=False):
        pts = [(float(x), float(y)) for x, y in points]
        coords = " ".join(f"{x:g},{y:g}" for x, y in pts)
        extra = ' stroke-dasharray="6 5"' if dash else ""
        self.svg.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width:g}" '
            f'stroke-linejoin="round" stroke-linecap="round"{extra}/>'
        )
        self.pdf.setStrokeColor(HexColor(color))
        self.pdf.setLineWidth(width)
        self.pdf.setLineCap(1)
        self.pdf.setLineJoin(1)
        self.pdf.setDash([6, 5] if dash else [])
        p = self.pdf.beginPath()
        p.moveTo(pts[0][0], HEIGHT - pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, HEIGHT - y)
        self.pdf.drawPath(p)
        self.pdf.setDash([])
        if arrow:
            x, y = pts[-1]
            a, b = pts[-2]
            dx, dy = x - a, y - b
            length = math.hypot(dx, dy)
            dx, dy = dx / length, dy / length
            self.polygon(
                [
                    (x, y),
                    (x - 9 * dx + 4 * dy, y - 9 * dy - 4 * dx),
                    (x - 9 * dx - 4 * dy, y - 9 * dy + 4 * dx),
                ],
                color,
            )

    def circle(self, x, y, r, fill, stroke=None, line_width=1.5):
        self.svg.append(
            f'<circle cx="{x:g}" cy="{y:g}" r="{r:g}" fill="{fill or "none"}" '
            f'stroke="{stroke or "none"}" stroke-width="{line_width:g}"/>'
        )
        self.pdf.setLineWidth(line_width)
        if fill:
            self.pdf.setFillColor(HexColor(fill))
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.circle(x, HEIGHT - y, r, fill=int(bool(fill)), stroke=int(bool(stroke)))

    def text(self, x, y, text, size=18.0, color=INK, bold=False, anchor="start", max_width=None):
        font = self.font_bold if bold else self.font_regular
        measured = stringWidth(text, font, size)
        if max_width is not None and measured > max_width:
            raise ValueError(f"Text does not fit: {text!r} ({measured:.1f} > {max_width})")
        svg_weight = 700 if bold else 400
        self.svg.append(
            f'<text x="{x:g}" y="{y:g}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size:g}" font-weight="{svg_weight}" fill="{color}" '
            f'text-anchor="{anchor}">{html.escape(text)}</text>'
        )
        self.pdf.setFillColor(HexColor(color))
        self.pdf.setFont(font, size)
        method = (
            self.pdf.drawCentredString
            if anchor == "middle"
            else self.pdf.drawRightString
            if anchor == "end"
            else self.pdf.drawString
        )
        method(x, HEIGHT - y, text)

    def save(self, render_png=True):
        self.svg.append("</svg>")
        svg_path = self.output_dir / f"{STEM}.svg"
        pdf_path = self.output_dir / f"{STEM}.pdf"
        png_path = self.output_dir / f"{STEM}.png"
        svg_path.write_text("\n".join(self.svg) + "\n", encoding="utf-8")
        self.pdf.showPage()
        self.pdf.save()
        if render_png:
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            pix = page.get_pixmap(dpi=240, alpha=False)
            pix.save(str(png_path))
            doc.close()
        return pdf_path, png_path, svg_path


def spec_sliders_icon(f: Figure, cx: float, cy: float):
    """3 horizontal target sliders with constraint check dot."""
    for j, (offset, col) in enumerate([(-34, BLUE), (10, BLUE), (42, TEAL)]):
        y = cy - 22 + 22 * j
        f.line([(cx - 56, y), (cx + 56, y)], RULE, 2.2)
        f.circle(cx + offset, y, 5.5, col)
    # Subtle constraint tick near the third slider dot
    f.circle(cx + 42, cy + 22, 9.5, None, stroke=TEAL, line_width=1.4)


def octree_guard_icon(f: Figure, cx: float, cy: float):
    """Subdivided 3D octree cube with neutral coordination-guard ring."""
    x, y, side = cx - 28, cy - 19, 44
    for dx, dy in [(0, 0), (20, -14)]:
        f.line(
            [(x + dx, y + dy), (x + side + dx, y + dy), (x + side + dx, y + side + dy), (x + dx, y + side + dy), (x + dx, y + dy)],
            BLUE,
            1.6,
        )
    for dx, dy in [(0, 0), (side, 0), (side, side), (0, side)]:
        f.line([(x + dx, y + dy), (x + dx + 20, y + dy - 14)], BLUE, 1.6)
    f.line([(x + 22, y), (x + 22, y + 44)], BLUE, 1.1)
    f.line([(x, y + 22), (x + 44, y + 22)], BLUE, 1.1)
    f.circle(x + 11, y + 11, 4.6, BLUE)
    f.circle(x + 33, y + 33, 4.6, BLUE)
    f.circle(x + 36, y - 5, 4.8, TEAL)
    f.circle(x + 36, y - 5, 9.2, None, stroke=TEAL, line_width=1.4)


def neutral_valence_icon(f: Figure, cx: float, cy: float):
    """Locked 6-membered heavy-atom ring with explicit neutral peripheral H stubs."""
    pts = [
        (cx - 24, cy - 11),
        (cx - 2, cy - 24),
        (cx + 22, cy - 10),
        (cx + 22, cy + 15),
        (cx - 2, cy + 27),
        (cx - 24, cy + 14),
    ]
    f.line(pts + [pts[0]], "#96A3AD", 2.4)
    # Explicit peripheral neutral H stubs on two heavy atoms
    h1 = (cx + 42, cy - 20)
    h2 = (cx + 42, cy + 25)
    f.line([pts[2], h1], TEAL, 2.0)
    f.line([pts[3], h2], TEAL, 2.0)
    for i, (px, py) in enumerate(pts):
        col = AMBER if i == 2 else "#5D6C76"
        f.circle(px, py, 5.4, col)
    f.circle(h1[0], h1[1], 3.8, TEAL)
    f.circle(h2[0], h2[1], 3.8, TEAL)


def critic_bus_icon(f: Figure, cx: float, cy: float):
    """Left: 6-ring under inward MLIP forces; Right: monotonic reward hierarchy bars."""
    mx = cx - 32
    pts = [
        (mx - 19, cy - 9),
        (mx, cy - 20),
        (mx + 19, cy - 9),
        (mx + 19, cy + 12),
        (mx, cy + 23),
        (mx - 19, cy + 12),
    ]
    f.line(pts + [pts[0]], "#96A3AD", 2.1)
    for i, (px, py) in enumerate(pts):
        f.circle(px, py, 4.5, AMBER if i == 2 else "#5D6C76")
    # Inward MLIP forces
    f.line([(mx + 35, cy - 22), (mx + 23, cy - 13)], AMBER, 1.9, arrow=True)
    f.line([(mx - 35, cy + 24), (mx - 23, cy + 15)], AMBER, 1.9, arrow=True)

    # Monotonic rank bars
    rx = cx + 24
    for i, (length, col) in enumerate([(44, BLUE), (31, "#9CB2C6"), (18, "#C9D5DF")]):
        by = cy - 18 + i * 18
        f.circle(rx - 4, by, 4.2, col)
        f.line([(rx + 8, by), (rx + 8 + length, by)], col, 5.5)


def atlas_uncertainty_icon(f: Figure, cx: float, cy: float):
    """Minimalist 4-zone calibration curve with σ_gen (blue) and σ_cal (amber) uncertainty bands."""
    x0, x1 = cx - 56, cx + 56
    y0, y1 = cy + 24, cy - 26
    # Axes
    f.line([(x0, y0), (x1, y0)], SLATE, 1.5)
    f.line([(x0, y0), (x0, y1)], SLATE, 1.5)
    # Ideal slope=1 dashed diagonal
    f.line([(x0 + 6, y0 - 4), (x1 - 4, y1 + 4)], SLATE, 1.2, dash=True)
    # Epistemic σ_cal widening fan into OOD right side
    xs = [x0 + 10, x0 + 34, x0 + 58, x0 + 82, x1 - 4]
    mean_ys = [y0 - 8, y0 - 19, y0 - 30, y0 - 39, y0 - 43]
    gen_half = [5, 5, 5, 6, 7]
    cal_half = [7, 8, 9, 14, 21]
    amber_poly = [(x, y - c) for x, y, c in zip(xs, mean_ys, cal_half)] + list(
        reversed([(x, y + c) for x, y, c in zip(xs, mean_ys, cal_half)])
    )
    blue_poly = [(x, y - g) for x, y, g in zip(xs, mean_ys, gen_half)] + list(
        reversed([(x, y + g) for x, y, g in zip(xs, mean_ys, gen_half)])
    )
    f.polygon(amber_poly, "#F5DFC9", opacity=0.75)
    f.polygon(blue_poly, "#DCE8F5", opacity=0.88)
    f.line(list(zip(xs, mean_ys)), TEAL, 2.2)
    for x, y in zip(xs, mean_ys):
        f.circle(x, y, 2.7, TEAL)


def draw(output_dir: Path, render_png: bool = True):
    f = Figure(output_dir)

    # Top Header Band (y = 18 .. 102)
    f.text(24, 46, "UNI-LOOM", size=23, bold=True, color=INK)
    f.text(24, 73, "loom v2.0", size=15, color=MUTED)

    f.text(
        184,
        46,
        "Weave external predictors, valence-pure 3D, and RSI",
        size=21.0,
        bold=True,
        color=INK,
        max_width=552,
    )
    f.text(
        184,
        76,
        "Zero-retrain asset mounting, scale-decoupled RL, and blindspot-safe calibration",
        size=15.5,
        color=MUTED,
        max_width=552,
    )

    # Top Input Pill: External Asset Hub
    f.rect(756, 18, 204, 80, fill="#F2F6FA", stroke=None, radius=10)
    f.text(858, 48, "External asset hub", size=18.5, bold=True, color=INK, anchor="middle", max_width=188)
    f.text(858, 75, "QSAR / 3D grad / MLIP", size=15.5, color=MUTED, anchor="middle", max_width=188)
    f.line([(858, 102), (858, 144)], BLUE, 1.8, arrow=True)

    # Middle 5-Stage Forward Pipeline (y = 148 .. 336, h = 188, w = 204, gap = 40)
    stages = [
        (24, BLUE, "#F7FAFD", "TASK SPEC", "Weaving spec", "Targets + SA / SMARTS", spec_sliders_icon),
        (268, BLUE, "#F7FAFD", "GENERATOR", "Octree policy", "Uni-3DAR + Track A guard", octree_guard_icon),
        (512, TEAL, "#F2F8F6", "VALENCE GATE", "Neutral graph", "Track B (0.00 \u00c5 heavy)", neutral_valence_icon),
        (756, AMBER, "#FCF7F2", "CRITIC & MLIP", "Critic bus", "DPA-4 / HyFF + AD audit", critic_bus_icon),
        (1000, TEAL, "#F2F8F6", "GP-ATLAS", "4-zone atlas", "Separate \u03c3_gen and \u03c3_cal", atlas_uncertainty_icon),
    ]
    for x, color, fill, kicker, title, subtitle, icon in stages:
        f.rect(x, 148, 204, 188, fill=fill, stroke=RULE, radius=12, line_width=1.5)
        f.text(x + 16, 176, kicker, size=14.0, bold=True, color=color, max_width=172)
        f.text(x + 102, 207, title, size=21.0, bold=True, color=INK, anchor="middle", max_width=184)
        icon(f, x + 102, 260)
        f.text(x + 102, 318, subtitle, size=14.5, color=MUTED, anchor="middle", max_width=188)

    # Horizontal forward arrows between the 5 stages
    for x in [228, 472, 716, 960]:
        f.line([(x + 5, 242), (x + 35, 242)], SLATE, 2.0, arrow=True)

    # Dashed TEAL Collector Bus (y = 340 .. 408) feeding Inner-Loop RSI Card
    f.line([(614, 340), (614, 408)], TEAL, 1.7, dash=True, arrow=True)
    f.line([(858, 340), (858, 370), (614, 370)], TEAL, 1.7, dash=True)
    f.line([(1032, 340), (1032, 370), (858, 370)], TEAL, 1.7, dash=True)
    f.circle(614, 370, 3.0, TEAL)
    f.circle(858, 370, 3.0, TEAL)
    f.text(
        628,
        396,
        "Clean 3D + monotonic rewards (trusted \u03c3_cal \u2264 \u03c4_cal)",
        size=15.0,
        color=TEAL,
        max_width=392,
    )

    # Bottom Inner-Loop Alignment Card (x=268, y=412, w=692, h=120)
    f.rect(268, 412, 692, 120, fill="#F5F9F8", stroke=RULE, radius=10, line_width=1.5)
    f.text(292, 444, "Scale-decoupled inner RSI", size=21.0, bold=True, color=INK, max_width=644)
    f.text(
        292,
        474,
        "Stage 2.5A TopoGRPO on coarse tokens + detached 0.01 \u00c5 XYZ distill",
        size=17.5,
        color=MUTED,
        max_width=644,
    )
    f.text(
        292,
        502,
        "80% deficit curriculum + 20% anchor replay; blindspots frozen at 0%",
        size=16.5,
        color=MUTED,
        max_width=644,
    )

    # Dashed BLUE return arrow from Inner RSI -> Octree Policy (Stage 2)
    f.line([(370, 408), (370, 340)], BLUE, 1.8, dash=True, arrow=True)
    f.text(384, 384, "TopoGRPO + XYZ", size=16.0, color=BLUE, max_width=190)

    # Downward AMBER trigger arrow from 4-zone Atlas (Stage 5) -> Outer-Loop RSI Card
    f.line([(1148, 340), (1148, 408)], AMBER, 1.8, arrow=True)
    f.text(1136, 396, "\u03c3_cal > \u03c4_cal", size=15.5, color=AMBER, anchor="end", max_width=94)

    # Bottom Outer-Loop Calibration & Audit Card (x=1000, y=412, w=204, h=120)
    f.rect(1000, 412, 204, 120, fill="#FFFFFF", stroke="#DCC6B4", radius=10, line_width=1.5)
    f.text(1102, 443, "Outer-loop RSI", size=18.5, bold=True, color=INK, anchor="middle", max_width=184)
    f.text(1102, 474, "Murcko SDF + DFT", size=16.5, color=MUTED, anchor="middle", max_width=184)
    f.text(1102, 502, "Blind DPA-4 audit", size=16.5, color=MUTED, anchor="middle", max_width=184)

    # Clean Outer-Right Orthogonal AMBER Return Path -> External Asset Hub Pill
    f.line([(1204, 472), (1244, 472), (1244, 58), (966, 58)], AMBER, 1.8, arrow=True)
    f.text(1105, 44, "Zero-retrain \u0394_GP calibration", size=16.0, color=AMBER, anchor="middle", max_width=255)

    # Two-Line Editorial Footer (y = 580 .. 608)
    f.text(
        24,
        580,
        "Raw point-cloud and valence-reconciled metrics are reported side by side; blind DPA-4 audit stays strictly isolated.",
        size=18.0,
        color=MUTED,
        max_width=1232,
    )
    f.text(
        24,
        608,
        "External predictors mount without retraining; Bayesian residual calibration contracts blindspot \u03c3_cal to unlock inner RL.",
        size=16.5,
        color=MUTED,
        max_width=1232,
    )

    return f.save(render_png=render_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()
    pdf_p, png_p, svg_p = draw(args.out_dir, render_png=not args.no_png)
    print(f"Saved {pdf_p}, {png_p}, and {svg_p}")
