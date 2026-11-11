# Main implementation: pretrained Uni-3DAR octree

Decision updated on 2026-11-11 after remote experiments. The primary backend is
`uni3dar_octree`, with checkpoint identity `uni3dar_octree_v1`. The earlier
`dense_grid_v2` freeze served the initial correctness repair; that implementation
now remains a comparison baseline. A 256-record dense-grid run produced no novel
structures relative to its adaptation set, and a larger scratch-training run
became nonfinite at epoch 46. Continuing to tune that representation was not the
right route to a useful octree generator.

The CLI dispatches to `pipeline/octree_runner.py`. `generator/octree_model.py`
contains the conditional octree and structure-understanding head;
`generator/octree_backend.py` supplies direct SDF-to-octree training and checkpoint
loading. This is the same backend used by the new experimental scripts. The
historical benchmark runners remain archived implementations, not alternative
entrypoints for reproducing the new results.

The pretrained model has 12 layers, width 768, 12 attention heads, 0.24 Å leaf
cells, 0.01 Å coordinate bins, and merge level 6. This mainline supports neutral
closed-shell H/C/N/O/F molecules. The force field's broader element vocabulary
does not expand the generator's capabilities. SFT trains the zero-initialized
interval conditioner and condition-masked property head while freezing the
pretrained backbone. GRPO includes tree, element, X, Y and Z decisions, but is
opt-in: the first matched-budget comparison did not demonstrate an improvement.

QM9s supplies structures and its original 240-channel spectra are retained
separately. OCELOT supplies HOMO/LUMO training labels in eV. The generator's
HOMO/LUMO labels on QM9s are explicitly OCELOT-trained oracle predictions, not
QM9s reference measurements. The adapter no longer sends octree training data
through the unrelated dense-grid quantizer.

Deployment uses a fixed proposal budget, structure-only self-ranking, an explicit
external-oracle shortlist budget, and finite-step constrained relaxation. Every
attempt remains in the result denominator. Checkpoints contain the entire model
and target scaling; prediction does not require the upstream pretrained file.
The training structure list is retained in new checkpoints for adaptation-set
novelty, which is distinct from novelty against the original pretraining corpus.

## Dependency boundary

Required Uni-3DAR source is vendored in the package, pinned to commit
`bba82f5ffdcd85963e1dff68a52415cbf046e6c7`. The original MIT license and source hashes
are retained. Private `sys.path` injection is unnecessary. Local changes are
listed in `third_party/uni3dar_vendor_manifest.json`: private imports, idempotent
atom dictionary, explicit module loading, PyTorch SDPA/UniCore compatibility,
and an ORBIT real-valued rotary implementation replacing the upstream file with
its separate license header. No model weights or datasets are redistributed.

Property coordinate derivatives use DeepMD's differentiable embedding path.
The force-inference path detaches edge vectors and cannot supply the full
property gradient; finite differences exposed and verified this repair.
Independent bootstrap adapters share frozen descriptors but have separate fitted
weights and resampled training records. Their disagreement is not yet calibrated.

See [remote experiment report](qm9s-ocelot-experiments-2026-11-11.md) for actual
scope and results. Broad-element generation, scaffold growth, uncertainty
calibration and independent quantum calculations remain separate research work.

## Interval and sampling update

New adaptation checkpoints use `uni3dar_octree_v2`. They explicitly encode
lower/upper presence masks and train on bounded, lower-only, upper-only and
inactive target examples. Bounded-only v1 checkpoints retain their original
model construction and reject one-sided conditions; no checkpoint is silently
relabelled as a trained v2 model.

The recommended configuration uses `sampling_mode: independent` and
`precision: float32`. A counter stream keyed by run seed and candidate ID supplies
each candidate's uniforms; batch compaction does not consume another candidate's
stream. FP32 replay of four tracked candidates matched across batch1/8/48 in the
real-model check. BF16 replay did not match across those batch sizes. BF16 is an
explicit faster numerical mode, not a promise of batch-independent token replay.
