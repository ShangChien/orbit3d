# OCELOT vertical excitation targets

The publisher's README defines `s0s1` and `s0t1` as vertical first-singlet and
first-triplet excitation energies, in eV. Their per-molecule difference is a
vertical S1−T1 separation at the source S0 geometry. It is not an adiabatic gap
between separately optimized excited-state geometries or an emission energy.
The [OCELOT study](https://pubs.rsc.org/en/content/articlelanding/2023/sc/d2sc04676h)
describes molecule-specific IP-tuned LC-ωHPBE/Def2SVP DFT/TDDFT calculations.

The expanded supervised experiment predicts `homo`, `lumo`, `s0s1`, `s0t1`.
It reuses the exact audited ETKDG/UFF conformers and identifiers of the earlier
8192/512/512 3D experiment; no new conformer selection or relabeling is performed.
The original DFT geometries remain unavailable. A separate fingerprint ensemble
uses the existing 20356/2282/2322 scaffold split and the same four target names.
Both training scripts accept explicit ordered target lists and retain source
hashes and all prior geometry exclusions.

Each evaluation reports separate target MAEs plus the error of the predicted
S1−T1 difference for each molecule, then averages absolute errors. A difference
between ensemble means is separately identified and cannot be substituted for
MAE. Regression coverage includes opposite errors that cancel in the mean while
having nonzero paired MAE, mismatched sample counts and nonfinite pairs.

These are supervised predictions against the official CSV. They do not resolve
the independent PySCF/Gaussian protocol discrepancy or establish excited-state
minimum accuracy. The separate quantum calculations retain their own geometry,
state, functional, basis, omega and failure records.


`scripts/migrate_octree_targets.py` explicitly expands a trained generator's
ordered target interface. Existing target normalizations, conditioner blocks and
self-head rows are retained. New conditioner input columns and new readout rows
start at zero; new readouts therefore begin at their training-set means. The
existing conditioner behavior is mathematically preserved when added targets
are inactive, with floating-point comparisons checked separately. The migrated
checkpoint blocks conditional inference until labelled adaptation is completed.
It records both source checkpoint/data hashes and the new-target training status.

## Predeclared generator pilot

The four-target adaptation starts from the unweighted 5000-step control model,
without selecting between the element-weighting experiment's outcomes. Its
structural backbone remains frozen; only the expanded conditioner and readout
are trained for 2000 steps, batch 8, learning rate 2e-4 and seed 42, using the
same 8188 encodable training conformers. No GRPO or element reweighting is used.
Calibration uses 512 unused validation molecules, separate from fitted and
model-selection records.

Three windows are evaluated with inference seeds 2026–2028 and 64 proposals per
seed: orbitals only (HOMO −8.2 to −7.2, LUMO −0.6 to 0.8), vertical excitations
only (S1 2.5–3.5, T1 1.5–2.5), and all four jointly, in eV. Unspecified targets
remain inactive. Ranking/refinement budgets and default sampling temperatures
are unchanged. Both the optimizing 3D model and the independent four-target
fingerprint model evaluate the retained molecules; all proposals remain in the
success denominator. The vertical S1−T1 gap is derived from paired predictions,
not introduced as an independently inconsistent fifth target.

Pure `mode: train` with SDF labels and zero GRPO steps now loads only the generator
and labelled training data. It does not need an Oracle, forcefield or domain file.
Pseudo-labelled training, alignment and generation still load their evaluators.
The first optical launch exposed this unnecessary dependency while the property
model was still training and stopped before any generator optimizer step. The
retry retains that failed run and reuses its exact prepared SDF and migrated
initial checkpoint in a fresh output directory.


## Held-out prediction result

On the identical 512 held-out molecules, the four-target DPA adapter has MAEs
0.2895/0.2326/0.2314/0.1381 eV for HOMO/LUMO/vertical S1/vertical T1. The
independent fingerprint model has 0.3593/0.3354/0.2968/0.2616 eV. The paired
vertical S1−T1 gap MAEs are 0.2226 and 0.2546 eV; absolute differences of means
are only 0.0537 and 0.0274 eV and are not substituted for MAE. Training budgets
are 8192 and 20356, respectively, so this is a shared-test comparison rather
than a controlled architectural ablation. New ordinary/native predictions agree
within the existing 1e-5 eV runtime tolerance; checkpoint parameters and predictions
from identical descriptors reload exactly. See `experiments/optical_matched512_20261112.json`
and the two source training summaries.

## Running the expanded interface

`configs/ocelot_optical_octree.json` is the portable four-target example. Set its
mounted paths to actual artifacts. To construct its `optical_initial.pt`, use the
explicit migration command; a two-target checkpoint cannot be renamed into a
four-target model:

```bash
python scripts/migrate_octree_targets.py \
  --checkpoint /path/to/labelled_wide_generator.pt \
  --training-sdf /path/to/ocelot_optical_generator8192/train.sdf \
  --targets homo lumo s0s1 s0t1 --units eV eV eV eV \
  --output /path/to/optical_initial.pt
orbit3d --config configs/ocelot_optical_octree.json
```

The saved final checkpoint can be evaluated over the frozen three-window protocol
without further training:

```bash
python scripts/evaluate_target_windows.py \
  --config configs/ocelot_optical_octree.json \
  --checkpoint /path/to/optical_generator/octree_conditional.pt \
  --windows data/optical_target_windows.json --output /path/to/fresh_evaluation \
  --independent-oracle /path/to/fingerprint_optical/oracle
```

The property model must have the same ordered targets and units, and its
applicability domain must be built from its own training structures. Supply the
original source CSV labels when preparing the generator SDF; QM9s spectra are
not substitutes for these four OCELOT quantities.
