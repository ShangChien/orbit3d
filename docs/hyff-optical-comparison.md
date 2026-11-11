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
