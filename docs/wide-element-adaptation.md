# OCELOT element coverage and reconstruction protocol

The initial data preparation used the legacy ten-element set. The source OCELOT
CSV also contains B/Si/Se/As/Te, so absence from an earlier training set does not
establish absence from the source data. The raw inventory has 135 B, 195 Si,
177 Se, 21 As and 25 Te records (overlapping element counts, before quarantine).

`prepare_ocelot.py --elements ...` now declares the accepted elements explicitly.
With all 15 source elements and an 80-atom bound, the scaffold-separated split has
20356/2282/2322 train/valid/test records, 104 conflicting-label rows quarantined,
one radical excluded, six duplicate aliases and 180 deferred larger structures.
[The split manifest](../data/ocelot_wide80_split_manifest.json) fixes source and
output checksums; previous split files are unchanged.

The geometry method is also explicit. Among neutral connected records within the
80-atom limit, MMFF has parameters for 190 Si structures and none of the B/Se/As/Te
records. UFF covers 193 Si, 133 B, 14 Se, 18 As and 25 Te records at the parameter
check stage. This is not an embedding or optimization success count. Unsupported
UFF Se types remain exclusions; no element replacement or per-record change of
forcefield is used. [The inventory](experiments/ocelot_wide_inventory_20261112.json)
retains the exact definitions.

The expanded property experiment explicitly selects ETKDGv3+UFF for **every**
training, validation and test input. OCELOT property labels remain unchanged;
UFF energies do not become electronic-property labels. The chosen reconstruction
method is recorded in the property checkpoint and checked during calibration.
This is reconstructed input geometry, not the unavailable original DFT geometry.
The 8192-record training run prioritizes rare elements within each pre-existing
split, then fills the remaining budget in deterministic hash order. The run completed with 8192 training, 512 validation and 512 test structures,
three bootstrap adapters and 40 epochs. Test HOMO/LUMO MAEs are 0.2985/0.2382 eV;
the corresponding train-mean baseline errors are recorded in the linked evidence.
The 233 geometry-preparation failures remain in the exclusion ledger.

| Test subgroup | Records | HOMO MAE (eV) | LUMO MAE (eV) |
| --- | ---: | ---: | ---: |
| B | 13 | 0.3057 | 0.1960 |
| Si | 7 | 0.3215 | 0.2346 |
| Se | 1 | 0.4319 | 0.5112 |
| As | 1 | 0.2235 | 0.1663 |
| Te | 4 | 0.4623 | 0.2634 |

Subgroups overlap. The single Se/As observations are individual errors, not
reliable generalization estimates. The one validation Te structure has much larger
errors of 1.1884/0.8300 eV. Rare-element training counts are B 107, Si 171, Se 12,
As 16 and Te 20; vocabulary coverage does not establish equal predictive quality.

A disjoint 512-record calibration set gives 91.41% joint test coverage at the
nominal 90% level, with mean interval widths 2.13/2.03 eV. Spread/error Spearman
correlations remain weak (0.216/0.122); the default uncertainty reward stays zero.
Exact parameter/scaler replay passes; repeated native forward predictions differ
by at most 3.82e-6 eV on validation and are checked separately.
[Training, exclusions, subgroup errors, calibration, hashes and graph audit](experiments/ocelot_wide8192_20261112.json)
retain the complete evidence.

Generator adaptation accepts its own explicit element set and the same geometry
method. New elements still require checkpoint migration, observed training coverage
and backbone adaptation. Explicit scaffold graph operations use molecule-local
element identities, preserving B without altering the legacy dense-grid vocabulary.


The generator reuses these exact labeled training conformers. Atom-conserving
encoding and rotation-safe span checks admit 8188 of 8192 structures at depth 7;
four exclusions retain their reasons. Full-backbone supervised adaptation is a
separate experiment and does not inherit a generation-accuracy claim from this
property model.

Coordinate-only reconstruction was audited on all 321 rare-element training
structures. Exact and 0.01 Å-rounded coordinates give the same classification:
295 exact identities, 18 additions of previously unspecified stereo, seven
changed charge/bond assignments and one invalid result. The 18 stereo additions
preserve the source constitution and every specified stereo constraint; the
seven graph changes remain failures. This audit does not test the full octree
transform. Known input graphs are preserved directly in property training; the
coordinate-only generator still faces bond-assignment ambiguity.


The independent fingerprint ensemble now also covers all 15 training elements.
It uses all 20356 training graphs, three bootstrap fits and 128 trees per fit;
2322 scaffold-held-out test graphs give HOMO/LUMO MAEs of 0.3542/0.3398 eV.
On the identical 512 test identities used by the 3D model, its MAEs are
0.3502/0.3338 eV, versus 0.2985/0.2382 eV for the 3D adapters. Training budgets
and representations differ, so this is a paired prediction check, not a controlled
architecture ranking. Reloaded predictions are exactly equal. Rare-element counts
and errors are retained in [the independent evaluator report](experiments/fingerprint_wide20356_20261112.json).

The first 4000-step generator run completes 64 proposals: 40 valid, three
in-domain retained and zero joint hits. A separate reload reproduces all 64 raw
proposals and all three exports, including acceptance trajectories. This initial
run is followed by three predeclared target windows and three inference seeds;
its zero-hit result is retained in that assessment.


## Free generation and rare-element diagnosis

The completed fixed-budget campaign uses three inference seeds per target window.
The [windows were specified from training-label quantiles](../data/wide_target_windows_protocol.json)
before this evaluation; their widths and difficulty differ.

| Window | Attempts | Valid | In domain | Optimizing joint hits | Independent joint hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 192 | 110 | 9 | 2 | 1 |
| Central | 192 | 95 | 9 | 0 | 0 |
| Lower orbital gap | 192 | 89 | 12 | 0 | 0 |

All 576 attempts remain in the denominator. None of the 294 valid structures
contains B/Si/Se/As/Te, although S/Cl/Br occur. Only 94 valid structures have no
formal atomic charges; net neutrality does not imply absence of internal charge
separation. [Per-candidate results and independent scores](experiments/wide_target_campaign_20261112.json)
retain these deficiencies. Expanded training coverage has not established useful
wide-element generation.

A teacher-forced inspection of the full 512-structure test set gives zero top-1
accuracy for B (16 atom decisions), Si (13), Se (one), As (one) and Te (four),
versus C 99.64%. At the inherited atom temperature 0.3, mean true-label
probabilities are B 0.000356 and Si 0.001911; at temperature 1 they are 0.02699
and 0.05825. Thus raising temperature alone does not repair the learned ranking.
[Full class probabilities](experiments/wide_element_likelihood_20261112.json)
include all classes and their counts. These diagnostic test inspections guide
model development and are not untouched final tests for later interventions.

Two follow-up interventions are kept separate: a fixed-checkpoint temperature
comparison, and a matched training comparison with inverse-square-root atom
frequency weighting. No default recipe or broad-element capability claim is
changed before observing their full chemical-validity and independent results.

## Saved-cohort spatial coverage

A separate audit of all 960 attempts from the three-window 4000-step campaign and
the two-window 5000-step control distinguishes atom count from actual spatial
coverage. Of 515 chemically valid outputs, ten exceed 40 atoms, six exceed the old
14.80 Å rotation-safe training span, and five have an axis span greater than the
old 15.36 Å box in their saved orientation. The largest valid diameter is 22.30 Å.
The six extended structures pass the unchanged connectivity, valence, stereo and
bond-geometry validator again; their precise SDF coordinates and inferred graphs
are saved separately. All six contain internally charged atoms, so this is not
evidence of high-quality long chromophore design. Passing a broad bond-geometry
check also does not imply low strain or energetic stability.

The full attempt ledger, coordinate/input hashes, atom counts, axis spans and
bond-length/covalent-radius diagnostics are in
`experiments/wide_spatial_audit_20261112.json` and
`experiments/wide_spatial_inputs_20261112.json`; the six structures are in
`experiments/wide_spatial_valid_extended_20261112.sdf`. This cohort was already
generated; no extra samples were drawn to obtain favorable long examples.
