# ORBIT-3D manuscript

This directory contains the manuscript for the implemented and evaluated system.
The proposed next architecture and experiments are maintained separately in
[design changes and implementation status](../docs/design_changes.md) and
[experiment designs and execution status](../docs/experiments.md).
The implemented algorithms are described in [architecture](../docs/architecture.md).
Unfinished experiments are not reported as completed results. The current research direction selects methods by measured target-control gains; GRPO is a comparison method, not a required paper centerpiece. The mean/width SFT, reciprocal condition contrast and hindsight pilot is tracked as E10 in the experiment plan; it is not yet a manuscript performance claim.

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

The main results include matched property predictors, BRICS comparisons, 36 SFT
continuation evaluations, 45 GRPO evaluations, fixed-start refinement controls,
preselected quantum cohorts and full replay. The native-coordinate OE62 extension
adds frozen-model evaluation on 4,762 held-out structures, 48,384 proposals across three independent generator training seeds,
independent HyFF assessment, 48-start geometry comparisons and a 256-proposal
wheel replay. GRPO improves property-hit yield but not time-matched target MAE.
An independent PBE0/def2-TZVP study adds 19 source references and 24 frozen policy
candidates: 20 converge with graph preservation and 12 retain the target.
Same-coordinate force checks distinguish MLIP and DFT convergence. Supplementary experiments retain
negative adaptation results, spatial/scaffold checks and excited-state diagnostics.

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

Figure 0 uses SVG/vector PDF; PNG rendering requires Poppler (`pdftoppm`). Figure 1
shows actual training-seed aggregates and means. Figure 2 reports the OE62
paired HyFF effects, with three generator seeds and no significance claim.

The source archive includes TeX, bibliography, all three vector figures and research
data under `anc/`. It supports inspection of the reported evidence, not a complete
training rerun: datasets and model weights remain separate. Building the archive
does not submit or publish the paper.
