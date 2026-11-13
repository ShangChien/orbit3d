#!/usr/bin/env python3
"""Draw Figure 0 as editable SVG and vector PDF; rasterize the PDF for preview.

Requires reportlab; PNG export uses pdftoppm (Poppler). Paths are relative to this
file, and no external artwork, model weights, or experiment results are needed.
"""
from __future__ import annotations

import argparse
import html
import math
import shutil
import subprocess
from pathlib import Path

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

WIDTH, HEIGHT = 1200, 605
INK = "#172A3A"
MUTED = "#526575"
RULE = "#D7E0E7"
BLUE = "#3F6C9E"
TEAL = "#247C75"
AMBER = "#B46B35"
STEM = "fig0_orbit3d_system_architecture"


class Figure:
    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.pdf = canvas.Canvas(str(output_dir / f"{STEM}.pdf"), pagesize=(WIDTH, HEIGHT))
        self.pdf.setTitle("ORBIT-3D terminal-feedback architecture")
        self.pdf.setAuthor("ORBIT-3D authors")
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
            '<title id="title">ORBIT-3D terminal-feedback architecture</title>',
            '<desc id="desc">Native-coordinate SFT initializes an autoregressive 3D generator. Point requests produce complete molecular proposals, followed by structure checks, fixed-budget energy relaxation and terminal property scoring. Optional policy feedback uses the original trajectories. Frozen independent models and DFT validate fresh pools.</desc>',
        ]
        self.rect(0, 0, WIDTH, HEIGHT, fill="#FFFFFF", stroke=None, radius=0)

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke=RULE, radius=12, line_width=1.5):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{line_width}"/>')
        self.pdf.setLineWidth(line_width)
        if fill:
            self.pdf.setFillColor(HexColor(fill))
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.roundRect(x, HEIGHT-y-h, w, h, radius, stroke=bool(stroke), fill=bool(fill))

    def line(self, points, color=RULE, width=2, dash=False, arrow=False):
        pts = [(float(x), float(y)) for x, y in points]
        coords = " ".join(f"{x:g},{y:g}" for x, y in pts)
        extra = ' stroke-dasharray="6 5"' if dash else ""
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{extra}/>')
        self.pdf.setStrokeColor(HexColor(color))
        self.pdf.setLineWidth(width)
        self.pdf.setLineCap(1)
        self.pdf.setLineJoin(1)
        self.pdf.setDash([6, 5] if dash else [])
        p = self.pdf.beginPath()
        p.moveTo(pts[0][0], HEIGHT-pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, HEIGHT-y)
        self.pdf.drawPath(p)
        self.pdf.setDash([])
        if arrow:
            x, y = pts[-1]
            a, b = pts[-2]
            dx, dy = x-a, y-b
            length = math.hypot(dx, dy)
            dx, dy = dx/length, dy/length
            self.polygon([(x, y), (x-9*dx+4*dy, y-9*dy-4*dx), (x-9*dx-4*dy, y-9*dy+4*dx)], color)

    def polygon(self, points, fill):
        coords = " ".join(f"{x:g},{y:g}" for x, y in points)
        self.svg.append(f'<polygon points="{coords}" fill="{fill}"/>')
        self.pdf.setFillColor(HexColor(fill))
        p = self.pdf.beginPath()
        p.moveTo(points[0][0], HEIGHT-points[0][1])
        for x, y in points[1:]:
            p.lineTo(x, HEIGHT-y)
        p.close()
        self.pdf.drawPath(p, stroke=0, fill=1)

    def circle(self, x, y, r, fill, stroke=None):
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="1.5"/>')
        self.pdf.setFillColor(HexColor(fill))
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.circle(x, HEIGHT-y, r, fill=1, stroke=bool(stroke))

    def text(self, x, y, text, size=20, color=INK, bold=False, anchor="start", max_width=None):
        font = "Helvetica-Bold" if bold else "Helvetica"
        measured = stringWidth(text, font, size)
        if max_width is not None and measured > max_width:
            raise ValueError(f"Text does not fit: {text!r} ({measured:.1f} > {max_width})")
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{html.escape(text)}</text>')
        self.pdf.setFillColor(HexColor(color))
        self.pdf.setFont(font, size)
        method = self.pdf.drawCentredString if anchor == "middle" else self.pdf.drawRightString if anchor == "end" else self.pdf.drawString
        method(x, HEIGHT-y, text)

    def save(self, render_png=True):
        self.svg.append("</svg>")
        (self.output_dir / f"{STEM}.svg").write_text("\n".join(self.svg)+"\n", encoding="utf-8")
        self.pdf.showPage()
        self.pdf.save()
        if render_png:
            poppler = shutil.which("pdftoppm")
            if not poppler:
                raise RuntimeError("SVG and PDF saved; pdftoppm is required for the PNG preview.")
            subprocess.run([poppler, "-png", "-singlefile", "-scale-to", "3000", str(self.output_dir/f"{STEM}.pdf"), str(self.output_dir/STEM)], check=True)
        return self.output_dir / f"{STEM}.svg"


def octree_icon(f, cx, cy):
    x, y, side = cx-27, cy-20, 44
    for dx, dy in [(0, 0), (20, -14)]:
        f.line([(x+dx, y+dy), (x+side+dx, y+dy), (x+side+dx, y+side+dy), (x+dx, y+side+dy), (x+dx, y+dy)], BLUE, 1.6)
    for dx, dy in [(0, 0), (side, 0), (side, side), (0, side)]:
        f.line([(x+dx, y+dy), (x+dx+20, y+dy-14)], BLUE, 1.6)
    f.line([(x+22, y), (x+22, y+44)], BLUE, 1.1)
    f.line([(x, y+22), (x+44, y+22)], BLUE, 1.1)
    for dx, dy in [(11, 11), (33, 33), (36, -5)]:
        f.circle(x+dx, y+dy, 4.6, BLUE)


def rank_icon(f, cx, cy):
    for i, (length, color) in enumerate([(84, BLUE), (55, "#9CB2C6"), (30, "#C9D5DF")]):
        y = cy-27+i*26
        f.circle(cx-47, y, 6, color)
        f.line([(cx-28, y), (cx-28+length, y)], color, 7)
    f.line([(cx-56, cy-30), (cx-51, cy-24), (cx-39, cy-37)], "#FFFFFF", 2)


def molecule(f, cx, cy, distorted=False):
    pts = [(cx-25, cy-11), (cx-2, cy-25), (cx+23, cy-10), (cx+23, cy+16), (cx, cy+29), (cx-25, cy+15)]
    if distorted:
        pts[1] = (cx-9, cy-25)
        pts[3] = (cx+16, cy+9)
    f.line(pts+[pts[0]], "#96A3AD", 2.5)
    for i, (x, y) in enumerate(pts):
        f.circle(x, y, 5.7, AMBER if i == 2 else "#5D6C76")


def draw(output_dir: Path, render_png=True):
    f = Figure(output_dir)
    f.text(24, 42, "ORBIT-3D", size=23, bold=True, color=INK)
    f.rect(254, 20, 192, 78, fill="#F2F6FA", stroke=None, radius=10)
    f.text(350, 49, "Native 3D + labels", size=19, bold=True, anchor="middle", max_width=176)
    f.text(350, 77, "Pretrained prior / SFT", size=17, color=MUTED, anchor="middle", max_width=176)
    f.line([(350, 102), (350, 146)], BLUE, 1.7, arrow=True)
    f.text(484, 50, "Generate, relax, then evaluate", size=25, bold=True, max_width=652)
    f.text(484, 82, "Point conditions are separate from acceptance tolerances", size=19, color=MUTED, max_width=652)

    stages = [
        (24, BLUE, "#F7FAFD", "REQUEST", "Target point", "HOMO or LUMO", None),
        (254, BLUE, "#F7FAFD", "POLICY", "3D generator", "tree / atoms / XYZ", octree_icon),
        (484, TEAL, "#F2F8F6", "STRUCTURE", "Chemical gate", "graph / charge / stereo", None),
        (714, AMBER, "#FCF7F2", "GEOMETRY", "Energy relax", "DPA, fixed budget", None),
        (944, TEAL, "#F2F8F6", "PROPERTY", "Point score", "Terminal geometry", rank_icon),
    ]
    for x, color, fill, label, title, subtitle, icon in stages:
        f.rect(x, 150, 192, 185, fill=fill, stroke=RULE, radius=12)
        f.text(x+16, 178, label, size=15, bold=True, color=color)
        f.text(x+96, 209, title, size=22, bold=True, anchor="middle", max_width=176)
        if icon:
            icon(f, x+96, 260)
        elif x == 24:
            for j, offset in enumerate([-32, 7, 36]):
                y = 238+23*j
                f.line([(x+40, y), (x+152, y)], RULE, 2)
                f.circle(x+96+offset, y, 5.5, BLUE)
        else:
            molecule(f, x+96, 254)
            if x == 714:
                f.line([(x+149, 235), (x+132, 244)], AMBER, 2, arrow=True)
                f.line([(x+44, 277), (x+62, 270)], AMBER, 2, arrow=True)
        f.text(x+96, 317, subtitle, size=16.5, color=MUTED, anchor="middle", max_width=176)
    for x in [216, 446, 676, 906]:
        f.line([(x+4, 243), (x+32, 243)], "#7C8D9B", 2, arrow=True)

    # One collector combines full-molecule outcomes, including rejected starts.
    f.line([(580, 340), (580, 406)], TEAL, 1.7, dash=True, arrow=True)
    f.line([(810, 340), (810, 370), (580, 370)], TEAL, 1.7, dash=True)
    f.line([(1040, 340), (1040, 370), (810, 370)], TEAL, 1.7, dash=True)
    f.circle(580, 370, 3, TEAL); f.circle(810, 370, 3, TEAL)
    f.text(620, 395, "Structure / physics / target feedback", size=17, color=TEAL, max_width=430)
    f.rect(254, 410, 652, 110, fill="#F5F9F8", stroke=RULE, radius=10)
    f.text(278, 443, "Optional policy alignment", size=22, bold=True)
    f.text(278, 473, "Update the original sampled trajectory", size=20, color=MUTED)
    f.text(278, 500, "Keep every rejected proposal in the denominator", size=18, color=MUTED, max_width=606)
    f.line([(350, 406), (350, 340)], BLUE, 1.8, dash=True, arrow=True)
    f.text(370, 383, "GRPO", size=17, color=BLUE)

    f.line([(1136, 243), (1171, 243), (1171, 465), (1141, 465)], AMBER, 1.8, arrow=True)
    f.rect(944, 410, 192, 110, fill="#FFFFFF", stroke="#DCC6B4", radius=10)
    f.text(1040, 441, "Independent check", size=18, bold=True, anchor="middle", max_width=176)
    f.text(1040, 473, "Frozen new pools", size=18, color=MUTED, anchor="middle", max_width=176)
    f.text(1040, 502, "Uni-Mol2 + DFT", size=18, color=MUTED, anchor="middle", max_width=176)
    f.text(24, 563, "Property adapters can use disjoint terminal-DFT labels for calibration.", size=20, color=MUTED, max_width=1112)
    f.text(24, 590, "Training, selection and independent validation retain separate data and budgets.", size=18, color=MUTED, max_width=1112)
    return f.save(render_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()
    print(draw(args.out_dir, render_png=not args.no_png))
