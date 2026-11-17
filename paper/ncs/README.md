# Uni-Loom *Nature Computational Science* Submission Package

**Title:** *Predictor-agnostic 3D molecular weaving with dual-track valence purity and grounded self-improvement*

This directory holds the submission-ready *Nature Computational Science* (`Article`) LaTeX manuscript, Nature-formatted bibliography, source tarball, and Markdown edition for **Uni-Loom (`loom v2.0.0`)**.

## Journal Format & Budget Compliance

| Section / Item | Limit (*Nature Computational Science* `Article`) | Verified Count |
|---|---|---|
| Title | $\le 15$ words | **11 words** |
| Abstract | Unstructured single paragraph ($\sim$150–185 words, 0 citations, 0 math) | **179 words** (0 citations, 0 math) |
| Main Text (`Introduction` + `Results` + `Discussion`) | $\le 3{,}500$ words (excl. Abstract, Methods, floats) | **2,041 words** |
| Methods | $\le 3{,}000$ words | **1,275 words** |
| Main Display Items | $\le 6$ total | **6** (Figures 1–5 + two-part Table 1a/b) |
| Extended Data Items | $\le 10$ total | **7** (Extended Data Figs. 1–2 + Extended Data Tables 1–5) |

## Directory Contents

| File | Description |
|---|---|
| `main_ncs.tex` | *Nature Computational Science* `Article` LaTeX manuscript (`super,sort&compress,comma` Nature citations, 19 pages compiled) |
| `main_ncs.bbl` | Synchronized 22-entry Nature `thebibliography` included directly by `main_ncs.tex` |
| `references.bib` | BibTeX bibliography synchronized with the main preprint |
| `sync_ncs_bbl.py` | Regenerates Nature-formatted `main_ncs.bbl` in exact citation order from `main_ncs.tex` and `references.bib` |
| `build_ncs_submission.py` | Packages the 10 source files (`main_ncs.tex`, `main_ncs.bbl`, `references.bib`, and `figures/*.pdf`) into `UniLoom_NCS_Submission.tar.gz` |
| `figures/` | Staged 7 vector PDFs (`fig1..fig5` main + `ed_fig1..ed_fig2` Extended Data) |
| `main_ncs.pdf` | Compiled 19-page PDF manuscript (zero LaTeX Overfull/Undefined warnings) |
| `UniLoom_NCS_Submission.tar.gz` | Upload-ready source tarball containing the 10 LaTeX/figure files |
| `UniLoom_NCS_Manuscript.md` | Complete Markdown mirror of the NCS manuscript with relative `../figures/*.png` embeds |

## Rebuilding the Bibliography, PDF, and Submission Tarball

From the repository root:

```bash
# 1. Regenerate Figure 1 and stage vector PDFs if updated
uv run --offline --no-project --with reportlab --with pymupdf python paper/draw_architecture.py
cp paper/figures/fig1_uniloom_architecture.pdf paper/ncs/figures/fig1_uniloom_architecture.pdf
cp paper/figures/fig2_stage_progression.pdf paper/ncs/figures/ed_fig1_stage_progression.pdf
cp paper/figures/fig3_audit_and_hotswap.pdf paper/ncs/figures/fig2_audit_and_hotswap.pdf
cp paper/figures/fig4_gp_atlas_1d_homo.pdf paper/ncs/figures/ed_fig2_gp_atlas_1d_homo.pdf
cp paper/figures/fig5_gp_atlas_2d_joint.pdf paper/ncs/figures/fig3_gp_atlas_2d_joint.pdf
cp paper/figures/fig6_zwitterion_and_synthesizability.pdf paper/ncs/figures/fig4_zwitterion_and_synthesizability.pdf
cp paper/figures/fig7_grounded_rsi_closed_loop.pdf paper/ncs/figures/fig5_grounded_rsi_closed_loop.pdf

# 2. Synchronize Nature-formatted main_ncs.bbl (22 citations)
.venv/bin/python paper/ncs/sync_ncs_bbl.py

# 3. Compile main_ncs.pdf
(cd paper/ncs && tectonic -X compile main_ncs.tex)

# 4. Package upload-ready UniLoom_NCS_Submission.tar.gz (10 files)
.venv/bin/python paper/ncs/build_ncs_submission.py
```
