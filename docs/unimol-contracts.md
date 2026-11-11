# Optional Uni-Mol and Uni-QSAR paths

The main experiments use DPA and HyFF. Optional Uni-Mol code now requires every
backbone parameter to match before loading; one matching tensor no longer counts
as a pretrained model. The new task head is initialized separately. Atom-count
limits and unsupported elements reject the complete record instead of truncating
atoms or substituting an unknown token.

New internal Uni-Mol training fits each target scaler on that fold's training
rows only, saves the split indices, and verifies scaler/model hashes on reload.
Each fold is converted to physical units before mean and spread are computed.
Legacy shared-scaler checkpoints retain their old point-prediction definition
and are explicitly marked `legacy_shared_scaler`; loading does not retroactively
repair the way an old model was trained. Validation loss is weighted by the
number of target values, so a short final batch does not receive a full batch's
weight. Zero-epoch training cannot produce an apparently trained head.

The Uni-QSAR bridge no longer suppresses all process warnings, substitutes methane
for an empty graph, strips disconnected hydrogen atoms, replaces nonfinite
descriptors with arbitrary constants, or silently truncates the SMILES sequence.
The tokenizer preserves single-backslash stereo tokens. Any external fitted
imputation/preprocessing contract still requires explicit implementation and
verification against the corresponding trained system.

Regression tests use small fitted models to check parameter coverage, fold-only
scaling, nonlinear physical-unit aggregation, coordinate derivatives and exact
reload. A real Uni-Mol pretrained checkpoint is not available in the current
remote experiment set, so these tests are not a claim of GPU model parity or
measured Uni-Mol property accuracy. The bridge's native-model comparison remains
separate from the verified DPA/HyFF workflow.
