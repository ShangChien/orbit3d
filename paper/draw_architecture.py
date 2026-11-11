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

WIDTH, HEIGHT = 1200, 550
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
        self.pdf.setTitle("ORBIT-3D five-stage architecture")
        self.pdf.setAuthor("ORBIT-3D authors")
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
            '<title id="title">ORBIT-3D five-stage architecture</title>',
            '<desc id="desc">Stages 1 to 3 learn a 3D molecular policy; optional Stage 4 aligns it. Stage 5 refines sampled candidates. A dashed arrow shows training feedback. Shared property and force evaluators support alignment and refinement. Outputs must pass explicit validation.</desc>',
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


def interval_icon(f, cx, cy):
    for i, (a, b, v) in enumerate([(-27, 19, -7), (-4, 43, 22), (-37, 1, -18)]):
        y = cy-25+i*25
        f.line([(cx-55, y), (cx+55, y)], RULE, 2)
        f.line([(cx+a, y), (cx+b, y)], "#ACC3DC", 11)
        f.circle(cx+v, y, 5.6, BLUE)


def rank_icon(f, cx, cy):
    for i, (length, color) in enumerate([(84, BLUE), (55, "#9CB2C6"), (30, "#C9D5DF")]):
        y = cy-27+i*26
        f.circle(cx-47, y, 6, color)
        f.line([(cx-28, y), (cx-28+length, y)], color, 7)
    f.line([(cx-56, cy-30), (cx-51, cy-24), (cx-39, cy-37)], "#FFFFFF", 2)


def reward_icon(f, cx, cy):
    f.line([(cx-54, cy+27), (cx+56, cy+27)], "#B6D5D1", 1.5)
    for x, height, fill in [(cx-46, 22, "#BED9D4"), (cx-10, 41, "#76AFA6"), (cx+26, 62, TEAL)]:
        f.rect(x, cy+26-height, 23, height, fill=fill, stroke=None, radius=3)


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
    f.rect(24, 18, 1112, 44, fill="#F3F6F8", stroke=None, radius=9)
    f.text(43, 47, "INPUTS", size=17, bold=True, color=MUTED)
    f.text(130, 47, "3D structures  ·  target property intervals  ·  supported electronic states", size=21, max_width=980)

    f.text(24, 96, "POLICY LEARNING", size=17, bold=True, color=BLUE)
    f.line([(191, 90), (906, 90)], RULE, 1.5)
    f.text(944, 96, "REFINEMENT", size=17, bold=True, color=AMBER)

    # The only feedback edge in the diagram is a training-time policy update.
    f.line([(810, 147), (810, 120), (350, 120), (350, 147)], BLUE, 1.7, dash=True, arrow=True)
    f.rect(470, 105, 220, 27, fill="#FFFFFF", stroke=None, radius=0)
    f.text(580, 125, "conditioner update", size=19, color=BLUE, anchor="middle")

    stages = [
        (24, BLUE, "#F7FAFD", "01", "3D prior", "Pretrained octree", octree_icon),
        (254, BLUE, "#F7FAFD", "02", "Interval SFT", "Target conditioning", interval_icon),
        (484, BLUE, "#F7FAFD", "03", "Self-ranking", "Select candidates", rank_icon),
        (714, TEAL, "#F2F8F6", "04", "Gated GRPO", "Optional alignment", reward_icon),
        (944, AMBER, "#FCF7F2", "05", "Relaxation", "Accept or backtrack", None),
    ]
    for x, color, fill, number, title, subtitle, icon in stages:
        f.rect(x, 150, 192, 185, fill=fill, stroke=RULE, radius=12)
        f.text(x+16, 178, number, size=17, bold=True, color=color)
        f.text(x+96, 209, title, size=24, bold=True, anchor="middle", max_width=176)
        if icon:
            icon(f, x+96, 260)
        else:
            molecule(f, x+96, 254)
            f.line([(x+149, 235), (x+132, 244)], AMBER, 2, arrow=True)
            f.line([(x+44, 277), (x+62, 270)], AMBER, 2, arrow=True)
        f.text(x+96, 317, subtitle, size=19, color=MUTED, anchor="middle", max_width=176)
    for x in [216, 446, 676, 906]:
        f.line([(x+4, 243), (x+32, 243)], "#7C8D9B", 2, arrow=True)

    # Shared evaluators have one short branch to each consumer, with no crossings.
    f.line([(465, 420), (465, 380), (1040, 380), (1040, 340)], MUTED, 1.8, arrow=True)
    f.line([(810, 380), (810, 340)], MUTED, 1.8, arrow=True)
    f.circle(810, 380, 3.3, MUTED)
    f.rect(540, 366, 185, 28, fill="#FFFFFF", stroke=None, radius=0)
    f.text(632, 386, "shared evaluation", size=18, color=MUTED, anchor="middle")

    f.rect(24, 420, 882, 112, fill="#FAFBFC", stroke=RULE, radius=10)
    f.line([(465, 438), (465, 515)], RULE, 1.3)
    f.text(46, 452, "Property oracle", size=22, bold=True)
    f.text(46, 481, "Predictions + uncertainty", size=20, color=MUTED)
    f.text(46, 511, "3D adapters / fingerprint models", size=18, color=MUTED, max_width=399)
    f.text(488, 452, "Force field", size=22, bold=True)
    f.text(488, 481, "Energy + interatomic forces", size=20, color=MUTED)
    f.text(488, 511, "DPA-4 / HyFF", size=18, color=MUTED)

    f.line([(1136, 260), (1167, 260), (1167, 476), (1141, 476)], AMBER, 1.8, arrow=True)
    f.rect(944, 420, 192, 112, fill="#FFFFFF", stroke="#DCC6B4", radius=10)
    f.text(1040, 452, "Validate output", size=21, bold=True, anchor="middle", max_width=174)
    f.text(1040, 483, "Chemistry · targets", size=19, color=MUTED, anchor="middle", max_width=174)
    f.text(1040, 512, "Residual forces", size=19, color=MUTED, anchor="middle", max_width=174)
    return f.save(render_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()
    print(draw(args.out_dir, render_png=not args.no_png))
