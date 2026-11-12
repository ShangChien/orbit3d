# Relocatable inference artifacts

`scripts/export_inference_bundle.py` packages a completed, explicitly named octree
run into a fresh directory. It includes the adapted generator, property adapter
and scaler, force-field weights, applicability domain, project wheel, dependency
lock and original whole-cohort replay artifacts. It rewrites active model paths
relative to the bundle and removes training/pretraining path requirements.

```bash
python scripts/export_inference_bundle.py \
  --run /path/to/completed_run \
  --output /persistent/artifacts/inference_snapshot \
  --wheel dist/orbit3d-1.0.0-py3-none-any.whl \
  --requirements requirements.lock.txt \
  --asset-root /path/to/original_working_directory \
  --archive /persistent/artifacts/inference_snapshot.tar
```

The optional uncompressed tar belongs on persistent storage. It is a model and
code artifact, not a Docker image or a copy of the training dataset. Every bundled
file has a SHA-256 entry. The included README gives dependency installation and
source-build switches; compatible GPU dependencies are still required.

From the bundle directory, install its wheel and run:

```bash
python verify_bundle.py
# Or generate a new cohort:
orbit3d --config config.json
```

The completed remote check copied all assets away from their original locations
and installed the included wheel into a separate import directory. The 23 bundled
files passed hash checks. Whole-cohort replay reproduced all 64 raw candidates and
six exports, using the original 1e-5 coordinate/property tolerances and exact raw
token checks. The uncompressed archive is 460,984,320 bytes. The wheel SHA-256 is
`a0c097653212e47f45c0cf44fdbcf43d55f12015add272d32dcb97d582d633d3`.

This used the validated remote Python dependencies; it does not claim a newly
built operating-system image or an independently reinstalled combined GPU stack.
The original runs and active training environment were not changed. Evidence is
in `experiments/relocated_bundle_manifest_20261112.json`,
`experiments/relocated_bundle_reload_20261112.json` and
`experiments/relocated_artifact_verification_20261112.json`. Private server paths
and the archive location remain in the Git-ignored `local_dev/` notes.

## Fixed 20000-step orbital model

A second bundle contains the predeclared seed-42 total-20000 HOMO/LUMO model,
the same model used for the qualified-design quantum selection. The reference
is its entire original-window/seed-2026 run, not a chosen successful molecule.
The current wheel is installed into another isolated import directory, and all
64 raw candidates plus nine exports replay at the same 1e-5 coordinate/property
tolerances with exact token checks. All 23 input files pass their hashes.

The uncompressed EFS archive is 451,563,520 bytes with SHA-256
`9799b68df945049290009d493f8e10d487bf95f76aa4dbd68a5085709351f3c8`.
Its wheel SHA-256 is
`aea55fc0bf97aba072975ffcaf2abb0bc9187d35db9e59b6edec71e4b7fb3e18`.
Neither this model choice nor the quantum subset is selected using the best
score among the three continuation seeds. The original four-target bundle remains
available separately. [Artifact verification](experiments/orbital20k_artifact_verification_20261112.json),
[file manifest](experiments/orbital20k_bundle_manifest_20261112.json) and
[whole-cohort replay](experiments/orbital20k_bundle_reload_20261112.json) identify
the new two-target artifact.
