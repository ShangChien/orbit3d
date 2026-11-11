> This is the initial correctness-repair record. The mainline has since moved to
> pretrained Uni-3DAR octree; see the [subsequent real-data experiments](qm9s-ocelot-experiments-2026-11-11.md).

# Architecture repair validation 2026-11-11

This change establishes `dense_grid_v2` as an explicit, versioned correctness
baseline. It does not migrate the external hierarchical octree trainer or reuse
its archived performance numbers. Existing unversioned generator checkpoints are
intentionally rejected because condition and termination semantics have changed.

## Implemented behavior

- Molecular objects retain the supplied RDKit graph, formal charge, multiplicity,
  stereochemistry and coordinates. Coordinate-only outputs use explicit-H bond
  inference and strict validation. Invalid reconstructions, disconnected outputs,
  unsupported elements, empty tokens, out-of-range coordinates and oversized
  inputs fail explicitly. No atom substitution, truncation, arbitrary fallback
  molecules or all-single-bond reconstruction remains in the CLI path.
- EOS is sampled at valid atom boundaries; model capacity and maximum atoms are
  explicit. Named target intervals carry independent lower/upper masks and saved
  normalization. Training, GRPO, refinement and export use all active targets.
- GRPO uses a clipped rollout ratio and differentiable current-policy KL under a
  shared grammar. Rollouts use top-p=1 for this objective. Rewards gate nonnegative
  utility; invalid structures receive an independent penalty. SFT replay remains.
  Property-head regression uses a condition-masked structure pass.
- Stage 5 uses whole-movable-vector projection, true-oracle property evaluation,
  finite-step energy/per-target acceptance, bounded backtracking and a fixed
  initial tolerance budget. Frozen coordinates are supported without recentering.
  Black-box nonlinear stacking uses its actual predictor for acceptance rather
  than a mean replacement. Raw force-field failures raise errors, not zero forces.
- Uni-QSAR average uncertainty now retains within-model fold disagreement and
  computes physical-unit disagreement after inverse scaling. The DPA shared-head
  training loss no longer penalizes head disagreement; its scaler fits training
  indices only. This is not yet a calibrated independent-fold ensemble.
- All attempted samples have a status. Failed outputs remain in the denominator
  and are not written as valid SDF molecules. Run configuration and generator
  checkpoint hash are recorded. Multi-target export and disabled-refinement paths
  work without scalar coercion or fake zero-force measurements.
- Incremental inference caches each layer's K/V and conditional modulation in
  bounded buffers. Full-prefix inference remains available for comparison.
- HyFF checkpoint manifests are packaged. Hidden DPA-3 fallback and private host
  paths in runtime resolvers were removed. Missing Uni-Mol weights fail explicitly.
  Python support is constrained to 3.11–3.12; PyTorch is pinned to the remotely
  tested 2.7.1; the lock file now resolves. CI builds the wheel and runs CPU tests.
- Optional research sources have exact commits and retained MIT license texts.
  Dataset files remain ignored; the provenance inventory and acquisition tool
  distinguish matching local hashes from fully verified source/license provenance.
  The historical optoelectronic runner containing fixed comparison values is
  blocked from presenting a new benchmark run.

## Evidence

| Check | Observed result | Scope |
| --- | --- | --- |
| Initial chemical/termination regressions | 7 failed before repair, 7 passed after | Includes benzene, carbonyl, triple bond, stereochemistry, unsupported element, truncation, disconnected atoms and EOS |
| Full local suite | 20 passed | Includes masked intervals, KL gradient, reward monotonicity, projection, backtracking/frozen atoms, fold variance, nonlinear stacking, missing checkpoint, SDF indices, training/reload/export, radical rejection, graph freezing and incremental logits |
| Isolated remote suite | 20 passed | Existing scientific environment, CPU execution; pytest installed only under the isolated validation directory |
| Real DPA4-Mini experiment | RMS force 0.854856 → 0.730803 eV/Å | Ethane, 5 steps, bond-length interval [1.494129, 1.504129] Å; final 1.504126 Å; accepted-step energies nonincreasing |
| Synthetic generator integration | 24/24 valid and requested atom count matched after 46 fitting steps | Six tiny training molecules, deliberately overfit; no held-out or quantum-property accuracy claim |
| Wheel | HyFF registry loaded from isolated wheel installation; all five tiers listed | No source-tree resource fallback; GPU HyFF kernels not exercised |
| Dependency and static checks | Lock resolution and critical Ruff checks passed; diff whitespace check passed | Docker image and remote GitHub CI execution not performed |
| Local dataset inventory | 9 hashes verified; 9 records still have incomplete source/license fields | Verification exits 2 until exact release provenance is supplied |

The initial local environment used Python 3.11.15, PyTorch 2.10.0, RDKit 2023.09.6 and NumPy 1.26.4.
The remote suite and real DPA run used the pre-existing PyTorch 2.7.1 scientific
environment. A separate fresh isolated environment was installed with `uv sync --frozen`
(PyTorch 2.7.1, RDKit 2026.9.1, NumPy 1.26.4) and also passed the tests. Two A800 devices were occupied by other workloads, so these validations
used CPU and did not interrupt existing jobs or the MIM-Flow download.

The DPA experiment uses a deterministic geometric property (C–C bond length) to
exercise the physical acceptance contract. It does not validate HOMO/LUMO accuracy,
DFT agreement, thermodynamic stability or learned generation quality. The weight
SHA-256 is `ae8c2574df58b527f84b5eac4631bd2f5722d99dc62c2b826bea365b02df033c`.
Detailed local evidence is in ignored `local_dev/review_evidence/` and
`local_dev/synthetic_generator_validation/`.

## Reproduce

```bash
uv sync --frozen
uv run --frozen pytest tests -q
uv build --wheel
python scripts/check_wheel.py
uv run --frozen python scripts/smoke_generator.py --output local_dev/synthetic_generator_validation
python scripts/manage_data.py
```

The optional real force-field experiment additionally requires a compatible
DeePMD runtime and a DPA checkpoint:

```bash
python scripts/validate_mlip.py --weight /path/to/DPA4-Mini.pt --device cpu --output local_dev/mlip_result.json
```

Docker uses the lock file, takes `ORBIT3D_EXTRA=deepmd` by default (or `hyff`), and
expects separately mounted weights/data. No GPU Docker compatibility claim is
made until the chosen backend is executed in that built image.

## Remaining research and engineering work

This section records the initial baseline. Current evidence is maintained in
[qm9s-ocelot-experiments-2026-11-11.md](qm9s-ocelot-experiments-2026-11-11.md)
and the dated updates in [TODO.md](../TODO.md). The pretrained octree migration,
bootstrap training, calibration, ranking, independent sampling streams and native
batched evaluation have since been implemented and tested. Spatial representation
migration, scaffold generation, broad-element/open-shell training and the complete
paper experiment matrix remain open.

## Runtime and vendoring update (2026-11-12)

The private `orbit3d.compat.hyff` package no longer inserts a generic `hyff`
package into `sys.path`. All imports, including dynamically loaded factories and
native kernels, resolve within the package; checkpoint constructor IDs retain
the upstream values. The Apache-2.0 license and third-party notices are included
in the wheel, with the exact upstream commit and original hashes recorded in
`third_party/hyff_vendor_manifest.json`. An unrelated installed `hyff` module
does not affect the bundled implementation. All five checkpoint tiers pass real
GPU reference/native energy and force comparisons in the clean remote runtime;
see [the results](experiments/hyff_private_namespace_20261112.json).

The Dockerfile uses a multi-stage build, the locked dependencies and a source
build of DeepMD compatible with pinned PyTorch. PyPI and Debian mirrors are
configurable build arguments. The remote platform's mounted Docker daemon ran
out of its system-layer quota while exporting the image after dependency
installation. **A completed runtime image and GPU execution inside that image
have not been verified.** The base/tool images were archived to EFS and unused
build layers removed. Docker storage settings are left unchanged. Real GPU
checks run in the already validated clean Python environment on the remote
A800 host; macOS is used for CPU tests and packaging only.
