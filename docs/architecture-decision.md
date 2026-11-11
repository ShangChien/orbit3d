# Main implementation: pretrained Uni-3DAR octree

Decision updated on 2026-11-12 after remote experiments. The primary backend is
`uni3dar_octree`, with explicit v1/v2/v3 checkpoint contracts; current conditioner
experiments use v3 and retain matched v2 controls. The earlier
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
cells, 0.01 Å coordinate bins, and merge level 6. The initial property-conditioned mainline uses neutral closed-shell H/C/N/O/F
molecules. Explicit depth-seven and 15-element labelled OCELOT adaptation now have
separate fixed-budget experiments; their rare-element quality and target success
remain weak. A structural-only MIM boron model has its own quantum cohort. The force field's broader element vocabulary
does not expand the generator's capabilities. SFT trains the zero-initialized
interval conditioner and condition-masked property head while freezing the
pretrained backbone. GRPO includes tree, element, X, Y and Z decisions, but is
opt-in: the corrected FP32 pilot has encouraging optimizing-surrogate results,
the complete five-objective matrix includes three training seeds per objective
and three inference seeds per trained policy. Aggregate gains do not establish a
uniform component ranking, and independent evaluators disagree.
See [the reward ablation](reward-ablation.md) for both positive and negative results.

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
weights and resampled training records. Held-out residual intervals have been calibrated, but ensemble spread ranks
property error weakly; the uncertainty reward remains disabled.

See [remote experiment report](qm9s-ocelot-experiments-2026-11-11.md) for actual
scope and results. Expanded elements, explicit scaffold growth, uncertainty calibration and quantum
calculations have separately scoped results; none establishes general effectiveness.

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

## Smooth interval conditioning (experimental v3)

A held-out 64-molecule diagnostic found that the v2 condition vector changed by
4.15% on average for a normalized interval shift of only1e-5, and by10.24% for
1e-4. Its exponential Fourier frequencies reach32768π. Correct conditions gave
a generation loss1.0961 versus1.1050 with shuffled targets and1.1180 without
conditions, showing weak conditional separation and excessive local sensitivity.
The raw diagnostic is `experiments/condition_diagnostic_8000_20261111.json`.

The opt-in `uni3dar_octree_v3` uses bounded low frequencies and trains on intervals
that contain the reference value at random positions and widths, instead of
always placing the reference at the interval midpoint. The migration tool keeps
the learned structural prior and property head, explicitly resets the conditioner,
and requires adaptation before conditional inference. Existing v1/v2 checkpoints
retain their original frequencies and are never relabelled. A frequency-buffer
mismatch is rejected on load. Generation benefit remains an experimental question;
the completed matched comparison did not establish that v3 improves joint success.
See `experiments/conditioning_factorial_20261112.json`.


## Shared evaluators and scaffold proposal boundary

DPA and HyFF now use an explicit factory in the octree workflow. Each trained
property adapter is tied to the exact forcefield checkpoint digest. HyFF's
reference graph supports batched property gradients; its native force engine is
a separate declared execution route. Both preserve float64 physical energies.
[The HyFF workflow](hyff-property-workflow.md) documents actual training and replay.

[Scaffold graph growth](scaffold-generation.md) is an explicit optional proposal
path with attachment endpoints, bond orders and fixed world coordinates. Its local
XYZ support and whole-fragment proposal budget do not imply that the pretrained
octree has learned scaffold context. That neural extension remains separate.
