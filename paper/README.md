# ORBIT-3D manuscript

This directory contains the manuscript for the implemented and evaluated system.
The proposed next architecture and experiments are maintained separately in
[design changes and implementation status](../docs/design_changes.md) and
[experiment designs and execution status](../docs/experiments.md).
The implemented algorithms are described in [architecture](../docs/architecture.md).
Unfinished experiments are not reported as completed results. The built draft now
centers native-coordinate OE62 point control, complete-pool terminal quantum
validation, full-data SFT and generated-geometry predictor calibration. Earlier
OCELOT studies and initial geometry controls are preserved in appendices. GRPO
remains an optional method whose contribution must be measured. E16 currently
appears as a frozen protocol; its unfinished outcomes are not in the conclusions.

## Files

| Path | Purpose |
| --- | --- |
| `ORBIT3D_arXiv_Paper.md` | Editable manuscript; result tables are regenerated from evidence |
| `references.bib`, `manuscript_template.tex` | Bibliography and typesetting template |
| `sync_current_evidence.py` | Select and checksum the evidence used by the manuscript |
| `build_manuscript.py` | Generate LaTeX, BBL and result tables |
| `draw_architecture.py`, `draw_grpo_seeds.py`, `draw_oe62_seeds.py` | Editable architecture and two measured GRPO seed comparisons |
| `figures/` | Current SVG/PDF/PNG versions of these three figures |
| `ancillary_readme.txt` | Source description of the bundled research evidence |
| `build/data/` | Generated, Git-ignored evidence staging; rebuilt from `docs/experiments/` and `data/` |
| `build_arxiv_submission.py` | Package the typesetting sources and ancillary data |
| `ORBIT3D_arXiv_Paper.pdf`, `ORBIT3D_arXiv_Submission.tar.gz` | Built manuscript and source archive |

The main evidence includes three-seed policy and reward-attribution controls,
846 successful DFT single points from 848 frozen terminal candidates, a shared
condition-off pool, uniformly processed BRICS proposals, and 2026 successful
new single points for the full-source SFT comparison. Terminal-DFT predictor
adaptation uses separate fitting/validation/test scaffolds and is tested on the
new pool. Selection improvements are reported separately from generation yield.
The twelve known-hit BRICS optimizations measure conditional hit persistence,
not whole-proposal optimized yield. All manuscript tables are generated from
checksum-bound research records.

## Build

```bash
python paper/sync_current_evidence.py
uv run --no-project --with pypandoc-binary python paper/build_manuscript.py
cd paper
pdflatex ORBIT3D_arXiv_Paper.tex
pdflatex ORBIT3D_arXiv_Paper.tex
cd ..
python paper/build_arxiv_submission.py
```

A synchronized BBL is included directly; BibTeX is not required. Edit the Markdown
and BibTeX sources rather than the generated TeX/BBL. Table generation fails if a
required result or checksum is missing. `sync_current_evidence.py` creates
`build/data/` from a clean checkout; generated data copies are not tracked separately. Tectonic can also compile the generated
TeX. Rebuild the figures only when their source or experiment data changes:

```bash
uv run --no-project --with reportlab python paper/draw_architecture.py
uv run --no-project --with matplotlib python paper/draw_grpo_seeds.py
uv run --no-project --with matplotlib python paper/draw_oe62_seeds.py --homo docs/experiments/oe62_formal_homo_hyff_repeats_20261113.json --lumo docs/experiments/oe62_formal_lumo_hyff_repeats_20261113.json
```

Figure 0 shows the current point-generation and terminal-feedback path, using SVG/vector PDF; PNG rendering requires Poppler (`pdftoppm`). Figure 1
shows actual training-seed aggregates and means. Figure 2 reports the OE62
paired HyFF effects, with three generator seeds and no significance claim.

The source archive includes TeX, bibliography, the three vector figures and research
data under `anc/`. It supports inspection of the reported evidence, not a complete
training rerun: datasets and model weights remain separate. Building the archive
does not submit or publish the paper.
