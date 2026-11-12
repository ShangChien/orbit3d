# ORBIT-3D manuscript

`ORBIT3D_arXiv_Paper.md` is the editable manuscript source. `build_manuscript.py`
generates the synchronized LaTeX, numeric bibliography, and pilot-results table.
Edit the Markdown and `references.bib`; do not edit the generated TeX or BBL.

The main results are organized by property prediction, proposal quality, controlled
ablations, quantum diagnostics and replay. Exploratory adaptation and the archived
octree/GVP pilot are separated into appendices. Short tables remain unbroken. Current tables include property training, stage/reward controls,
five GRPO objectives across three training seeds, four-target generation, matched
DPA/HyFF property prediction, fragment recombination and matched refinement. Independent-surrogate and
quantum disagreements remain explicit. These experiments establish the stated
implementation behavior and limited empirical scope, not general superiority.

## Figure 0

`draw_architecture.py` draws editable SVG and vector PDF from the same primitives.
PNG is rendered from the PDF with Poppler. The manuscript embeds the vector PDF.

```bash
uv run --no-project --with reportlab python paper/draw_architecture.py
```

Install Poppler to provide `pdftoppm`, or pass `--no-png` to generate SVG/PDF only.
The compatibility entrypoint `generate_orbit3d_figures.py` now generates Figure 0
by default. Its `--legacy-figures` option reproduces the historical illustrative
plots; those plots are not used by the revised manuscript and their fixed arrays
must not be treated as experiment outputs.

## Training-seed figure

`draw_grpo_seeds.py` reads the complete 45-run public matrix and draws the two
property-evaluator panels. Points represent training-seed aggregates, with means
shown separately; no uncertainty bars are implied.

```bash
uv run --no-project --with matplotlib python paper/draw_grpo_seeds.py
```

## Build the manuscript

```bash
python paper/sync_current_evidence.py
uv run --no-project --with pypandoc-binary python paper/build_manuscript.py
```

The pilot JSON is checked against `data/provenance.json` before table generation.
The source driver is identified by filename and SHA-256. The archive contains
aggregate data, not all checkpoints or sample-level records needed for a full
experiment rerun.

Compile `ORBIT3D_arXiv_Paper.tex` in a LaTeX editor, or use a TeX installation:

```bash
cd paper
pdflatex ORBIT3D_arXiv_Paper.tex
pdflatex ORBIT3D_arXiv_Paper.tex
```

A synchronized BBL is included directly, so BibTeX is not required. Regenerate
it from `references.bib` with `build_manuscript.py`; this preserves the numbered
citation order and primary-source links in the PDF.

## Build the source archive

```bash
python paper/build_arxiv_submission.py
```

This rebuilds the local source archive; it does not submit or publish the paper.
The archive includes the main TeX, bibliography, vector Figures 0 and 1, the
original ten-molecule and qualified-design five-molecule quantum input SDFs,
all three raw gradient arrays, and pilot
data with provenance, plus the checksum-verified current evidence archive.
Research data is packaged in the root `anc/` directory, following the
[arXiv ancillary-file format](https://info.arxiv.org/help/ancillary_files.html).
The manuscript itself does not depend on those files for typesetting. Superseded
figures are excluded.
