# HyFF property training and octree refinement

The octree workflow now selects `internal_hyff` explicitly instead of always
constructing DPA. HyFF exposes one batched PyTorch reference graph for native
atom descriptors, property coordinate gradients, energy and raw forces. Molecule
boundaries and Q/M inputs remain separate. The native CUDA energy/force route is
still independently selectable; it does not supply a substitute autograd graph.
HyFF's float64 reduced energies remain float64 through both adapter routes.

A real Presto checkpoint passed serial/batch descriptor, force and energy checks
(maximum energy difference 4.17e-7 eV). A property-gradient diagnostic agreed with
finite differences and had zero gradient into another molecule. This diagnostic
head was explicitly untrained and was not used as a property predictor.

Three bootstrap adapters were then trained on 512 OCELOT structures for 15 epochs.
The separate 64-record test gave HOMO/LUMO MAE 0.3853/0.3208 eV, versus the training-
mean baseline's 0.4990/0.5086 eV. Saved adapter/scaler replay and repeated backbone
predictions matched exactly in this run. Calibration used 128 unused validation
records, yielding 90.625% joint test coverage but broad 2.32/2.39 eV intervals.
Uncertainty penalties remain disabled.

A 32-attempt end-to-end run produced 26 valid proposals, three domain-eligible
structures and two property/force joint successes. The independent fingerprint
model also found two joint successes. This is a small workflow demonstration;
it is not a matched comparison against the larger DPA training cohort or DFT.
[Metrics and checkpoint metadata](experiments/hyff_property_workflow_20261112.json)
retain the training, calibration, generation and independent evaluation results.

Train with `scripts/train_ocelot_3d.py --backend hyff --hyff-tier presto`, passing
explicit data/output/checkpoint paths and the desired sample/epoch budgets.
For the octree configuration use `oracle_mode: internal_hyff`, `hyff_tier: presto`,
the trained property checkpoint directory and the exact same forcefield weights.
The property checkpoint's forcefield checksum prevents swapping in a different
tier without retraining. Use `hyff_route: reference` for the tested descriptor
workflow. The same backend/tier arguments are accepted by calibration and saved-
checkpoint evaluation scripts.
