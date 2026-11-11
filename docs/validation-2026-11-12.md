# Current source and installed-wheel validation

169 CPU regression tests passed on macOS with the frozen dependency set, including
complete octree decisions and KL gradients, electronic-state and graph checks,
paired target errors, target/element migration, fixed-scaffold world coordinates,
full replay comparison and real serialized Uni-QSAR loading. Two sklearn warnings
identify the legacy calibrated SVC API used intentionally in its checkpoint test;
they are not suppressed. CUDA computation is performed only on the remote Linux
runtime.

Seven added cases cover explicit/implicit hydrogen identity, imine E/Z addition
versus inversion or loss, isotopes, changed hydrogen counts and the actual quantum
graph-check rejection. The [bond-perception study](bond-perception-diagnostic.md)
identified this defect on known validation geometries.

A newly built wheel containing the identity correction passes all 24 tests in
`test_molecular_identity.py`, `test_quantum.py` and `test_quantum_ecp.py` in the
remote quantum runtime, with zero skips. Its import path points to a fresh
installation outside the source checkout. The [wheel hash and result](experiments/quantum_identity_wheel_20261112.json)
identify this artifact separately from the earlier ECP-only wheel.

Three additional real-PySCF regressions skip on that macOS environment and pass
in the remote quantum environment, including a fresh wheel installed into an
isolated import directory. They check matching def2 ECP electron counts,
unchanged light-element integrals and preservation of legacy Hamiltonian records.
The [installed-wheel evidence](experiments/quantum_ecp_wheel_20261112.json) and
[numerical ECP diagnostics](quantum-ecp.md) distinguish these three passing
regressions from the initial failed CPU/GPU force gate. CI now includes the CPU
quantum extra; the hosted workflow has not run for these local commits.

A fresh Python 3.11 environment was populated offline from the locked dependency
cache with `uv sync --frozen --extra octree --no-install-project --offline`, then
installed the built wheel without an editable checkout. From outside the repository,
`python -I -m orbit3d.cli --help`, octree model import and the packaged HyFF manifest
check passed. CI now repeats this installed-wheel check after the source tests.
CI itself has not been observed for these local commits. The current CLI no longer
injects private host or container source paths, and argument help does not import
the training stack.

Pure SDF-labelled octree SFT no longer loads an unused Oracle or forcefield.
A two-step real-model CPU regression completes with neither evaluator path present
and records only its actual generator/data inputs. A 2000-step four-target remote
adaptation uses the same source change with the real pretrained model. Evaluators
remain required for pseudo-labels, alignment and generation.

The earlier full mainline replay under the shared evaluator factory compares all
64 raw proposals and all retained outputs, not just the first valid molecule.
Its public summary is `experiments/latest_mainline_reload_20261112.json`. The
installed mainline DeepMD wheel, the immutable experiment source snapshots and the
separate quantum runtime are recorded independently; this CPU wheel check is not
a fresh-install validation of every optional CUDA combination.
