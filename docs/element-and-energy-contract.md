# Element identity and physical energy precision

`Mol3DInstance` now carries an explicit `element_table` with its integer atom
indices and exposes `symbols` and `atomic_numbers`. The default remains the
original ten-element table. A caller can supply a different table without
changing old dense-grid token offsets; dense-grid encoding rejects nonlegacy
tables rather than reinterpreting them. The force-field interface remaps symbols
into that force field's declared table before evaluation. Tests cover B, Si, Se,
As and Te graph round-trips and mismatched instance/force-field index orders.

Octree checkpoints may explicitly declare `allowed_elements` alongside tree depth.
The upstream periodic-table dictionary already contains these symbols, so token
IDs and parameter shapes remain stable. `scripts/migrate_octree_elements.py`
records the source hash, expands sampling support, and requires backbone
adaptation before inference. Existing embedding rows are retained; this alone
does not demonstrate trained broad-element generation. Charge/spin-conditioned
generation remains a separate task.

A 32-proposal GPU comparison before the subsequent energy-precision correction
finds identical tokens, decisions, coordinates, properties and raw forces after
the element-table refactor. See
[the compatibility result](experiments/element_table_compatibility_20261112.json).

## Energy precision correction

DeePMD retains reduced total energies at its energy precision but casts
atom-resolved outputs to the coordinate-input dtype. The old wrapper supplied
float32 coordinates, cast atom energies again to float32, and summed them in
float32. Heavy-element reference energies then obscured small geometry-dependent
energy changes; a Si example gave a 0.0867 eV/A finite-difference discrepancy even
when the outer sum was changed to float64, because atomic values had already
been rounded by the model's output conversion.

Coordinate I/O and atomic-energy aggregation now use float64, and shared/native
Oracle energy outputs and batch acceptance buffers preserve float64. Network
layers retain their checkpoint precision. The force tensor keeps the coordinate
caller's dtype. An analytic regression with 1e8 eV atomic reference offsets
fails before the change and resolves the small physical energy difference after
it. Real B/Si/Se/As/Te force checks now have maximum analytic/finite-difference
error 0.000355 eV/A at a 0.002 A step; changing the external type-index order
leaves energies and forces unchanged. These checks validate execution and
identity, not broad-element property accuracy.
[Raw results](experiments/extended_elements_energy64_20261112.json) are retained.

## Acceptance precision and replay scope

The correction reveals one serial/batch acceptance divergence among eight
30-atom structures with strict zero acceptance tolerances. The largest resulting
property difference is 1.72e-5 eV and raw-force difference 2.17e-4 eV/A; interval-hit
decisions remain identical. This test is recorded as
[failed](experiments/energy64_strict_serial_batch_20261112.json), not retroactively
passed. A separate diagnostic with explicit 1e-5 eV property and energy tolerances
matches every decision and remains below the original output-comparison bounds
(maximum property error 3.73e-6 eV, raw-force error 3.70e-5 eV/A).
[That result](experiments/energy64_tolerant_serial_batch_20261112.json) does not
establish arbitrary-input bitwise equality. Both tolerances are now configurable
through `stage5_relax_config`; defaults remain zero. Nonzero budgets are checked
against both the previous and the initial state, so they cannot accumulate
without bound. The final target-hit criterion remains the requested interval.

The replay checker also supports an empty retained SDF: all raw tokens and
candidate statuses must still match and no malformed nonempty SDF is ignored.
The 32-attempt, zero-retained depth-7 experiment
[replays successfully](experiments/empty_candidate_reload_20261112.json); replay
success is not generation success.
