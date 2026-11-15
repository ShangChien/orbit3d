# Uni-Loom Manuscripts & Submission Packages

**arXiv Title:** *Uni-Loom: Predictor-Agnostic 3D Molecular Weaving with Dual-Track Valence Purity and Grounded Recursive Self-Improvement*  
**Nature Computational Science Title:** *Predictor-agnostic 3D molecular weaving with dual-track valence purity and grounded self-improvement*

This directory contains the self-contained English arXiv LaTeX/Markdown manuscripts, editorial vector architecture generator (`paper/draw_architecture.py`), bibliography tooling, and a complete *Nature Computational Science* (`Article`) submission package under [`paper/ncs/`](ncs/README.md) for **Uni-Loom (`loom v2.0.0`)**.

## Directory Contents

| File / Directory | Description |
|---|---|
| `main.tex` | Primary single-column arXiv LaTeX manuscript |
| `main.bbl` | Synchronized `thebibliography` file included directly by `main.tex` (arXiv AutoTeX compatible) |
| `references.bib` | BibTeX bibliography database (`26` entries; `22` cited) |
| `UniLoom_arXiv_Paper.md` | Complete English Markdown edition of the arXiv preprint embedding `figures/fig1..fig7*.png` |
| `draw_architecture.py` | Vector PDF / 300-DPI PNG / SVG generator for `figures/fig1_uniloom_architecture.{pdf,png,svg}` in a minimalist horizontal Codex architectural style |
| `sync_bbl.py` | Regenerates `main.bbl` in exact citation order from `main.tex` and `references.bib` |
| `build_arxiv_submission.py` | Packages `main.tex`, `main.bbl`, `references.bib`, and `figures/fig1..fig7*.pdf` into `UniLoom_arXiv_Submission.tar.gz` |
| `figures/` | All 7 main-paper figures as vector `.pdf` and raster `.png` (plus `.svg` for Figure 1) |
| `main.pdf` | Compiled arXiv PDF manuscript |
| `UniLoom_arXiv_Submission.tar.gz` | Upload-ready arXiv source archive |
| `ncs/` | Complete *Nature Computational Science* (`Article`) submission directory (`main_ncs.tex`, `main_ncs.bbl`, `sync_ncs_bbl.py`, `build_ncs_submission.py`, `main_ncs.pdf`, `UniLoom_NCS_Submission.tar.gz`, `UniLoom_NCS_Manuscript.md`, `figures/*.pdf`, `README.md`) |

## Rebuilding Figures, Bibliography, PDF, and arXiv Archive

From the repository root:

```bash
# 1. Regenerate experimental plots (Figures 2-7) and Figure 1 architecture diagram
.venv/bin/python scripts/loom_plot_results.py \
  --v20-synth-json docs/experiments/v20_synthesizable_homo_eval.json \
  --v20-rsi-json docs/experiments/v20_grounded_rsi_2round_eval.json \
  --atlas-json docs/experiments/gp_atlas.json \
  --out-dir docs/figures
uv run --offline --no-project --with reportlab --with pymupdf python paper/draw_architecture.py

# 2. Synchronize main.bbl in citation order
.venv/bin/python paper/sync_bbl.py

# 3. Compile main.pdf (via tectonic or pdflatex)
(cd paper && tectonic -X compile main.tex)

# 4. Package upload-ready arXiv tarball
.venv/bin/python paper/build_arxiv_submission.py
```

For rebuilding the *Nature Computational Science* package, see [`paper/ncs/README.md`](ncs/README.md).
