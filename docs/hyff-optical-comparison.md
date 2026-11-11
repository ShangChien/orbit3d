# Matched-input HyFF optical predictor comparison

The existing HyFF-presto diagnostic used only 512 training molecules, whereas the
wide DPA adapter used 8192. Their previous errors are not a fair supervised-budget
comparison. This experiment reuses the exact 8192/512/512 train/validation/test
ETKDG/UFF geometries and four target labels from the completed DPA optical run.

Use HyFF-presto's frozen descriptors, three independent bootstrap adapters, seed
42, 40 epochs and batch 16, retaining the same property-head training procedure
and validation-based checkpoint selection. The supplied pretrained checkpoint
is already available and has passed real force/descriptor execution checks.
Calibration then uses 512 unused validation molecules with the same preparation
method. No model or evaluator is changed in an already completed generation run.

Report paired test errors for HOMO, LUMO, vertical S1, vertical T1 and their paired
vertical difference. The descriptor widths, foundation architectures and pretraining
histories differ, so matched downstream data do not isolate architectural causality.
A separately trained 3D model can test geometry-dependent surrogate disagreement;
it is still not quantum ground truth, and sharing OCELOT labels can correlate errors.
Any rescoring of generated outputs keeps every original attempt/failure denominator.

The portable `scripts/evaluate_generated_3d.py` rescores existing exports without
changing coordinates, candidate selection or the original attempt denominator.
It maps target names rather than assuming the two evaluators have the same output
order or dimension. For a four-target independent model and a two-target source,
only common targets enter the reported prediction disagreement.

The output distinguishes unrestricted property hits, independent applicability,
property-plus-source-force joint counts, and property-plus-independent-force joint
counts. This prevents an evaluator change from silently replacing the physical
criterion. Raw independent forces are retained separately. Sharing the same
training labels limits statistical independence; these comparisons remain
surrogate checks rather than independent quantum accuracy measurements.

## Completed matched test

All train/validation/test SourceIDs match exactly. On the common 512 test molecules:

| Property (MAE, eV) | DPA-4, 8192 train | HyFF-presto, 8192 train | Fingerprint, 20356 train |
| --- | ---: | ---: | ---: |
| HOMO | 0.2895 | 0.3077 | 0.3593 |
| LUMO | 0.2326 | 0.2502 | 0.3354 |
| Vertical S1 | 0.2314 | 0.2475 | 0.2968 |
| Vertical T1 | 0.1381 | 0.1661 | 0.2616 |
| Paired vertical S1−T1 | 0.2226 | 0.2176 | 0.2546 |

HyFF has higher errors on the four individual targets and a slightly lower paired
gap error in this fit. This is one downstream training seed, not evidence of a
stable architecture ranking. Saved/reloaded HyFF parameters, descriptor predictions
and repeated native predictions were identical in this environment. The 512 unused
validation calibration molecules gave joint test coverage 0.9043, but mean interval
widths were 2.27/2.07/2.25/1.90 eV and std/error Spearman correlations only
0.155/0.043/0.035/0.221. Uncertainty gating remains disabled by default.

HyFF also rescored all 25 retained molecules from the completed optical generation
runs without changing their geometry. The HOMO/LUMO, S1/T1 and four-target windows
gave 1/192, 1/192 and 0/192 joint hits, respectively, under either the original DPA
force threshold or HyFF's own force threshold. These totals agree with the previous
surrogate totals; they do not prove agreement for every candidate or validate DFT.

Machine-readable evidence: `experiments/optical_matched512_20261112.json`,
`experiments/hyff_optical8192_20261112.json`,
`experiments/hyff_optical_calibration_20261112.json`,
`experiments/optical_hyff_independent_20261112.json` and their input manifests.
