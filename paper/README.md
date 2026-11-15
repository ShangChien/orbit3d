# Uni-Loom arXiv Preprint Manuscript

**Title:** *Uni-Loom: Predictor-Agnostic 3D Molecular Weaving with Dual-Track Valence Purity and Grounded Recursive Self-Improvement*

This directory contains the self-contained English arXiv LaTeX manuscript, bibliography, figure generation script, and arXiv submission packager for **Uni-Loom (`loom v2.0`)**.

## Directory Contents

| File | Description |
|---|---|
| `main.tex` | Primary single-column arXiv LaTeX manuscript |
| `references.bib` | BibTeX bibliography database |
| `main.bbl` | Synchronized `thebibliography` file included directly by `main.tex` (for arXiv AutoTeX compatibility without running BibTeX) |
| `draw_architecture.py` | Vector PDF/PNG generator for `figures/fig1_uniloom_architecture.pdf` |
| `sync_bbl.py` | Regenerates `main.bbl` in exact citation order from `main.tex` and `references.bib` |
| `build_arxiv_submission.py` | Packages `main.tex`, `main.bbl`, `references.bib`, and `figures/fig1..fig7*.pdf` into `UniLoom_arXiv_Submission.tar.gz` |
| `main.pdf` | Compiled PDF manuscript for local review |
| `UniLoom_arXiv_Submission.tar.gz` | Upload-ready arXiv source archive |

## Rebuilding Figures, Bibliography, PDF, and arXiv Archive

From the repository root:

```bash
# 1. Regenerate experimental plots (Figures 2-7) and architecture diagram (Figure 1)
.venv/bin/python scripts/loom_plot_results.py \
  --v20-synth-json docs/experiments/v20_synthesizable_homo_eval.json \
  --v20-rsi-json docs/experiments/v20_grounded_rsi_2round_eval.json \
  --atlas-json docs/experiments/gp_atlas.json \
  --out-dir docs/figures
.venv/bin/python paper/draw_architecture.py

# 2. Synchronize main.bbl in citation order
.venv/bin/python paper/sync_bbl.py

# 3. Compile main.pdf (via tectonic or pdflatex)
cd paper && tectonic main.tex && cd ..

# 4. Package upload-ready arXiv tarball
.venv/bin/python paper/build_arxiv_submission.py
```
