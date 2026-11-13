ORBIT-3D ancillary research evidence

unified_evidence.json maps source filenames to their original JSON text (the
"json" field), SHA-256 and repository location. Verify the text hash before
decoding it. It includes current results as well as labelled diagnostic failures.

The three NPZ files contain the full quantum, DPA and HyFF excitation-gradient
arrays. Units are eV/Angstrom. Quantum keys are <index>_<target>_quantum_eV_A;
surrogate keys are <index>_<target>_eV_A. Targets s0s1 and s0t1 denote vertical
first-singlet and first-triplet excitations. Indices 0, 2, 5, 7 and 9 refer to
the five completed cases in the original ten-molecule quantum SDF. Surrogate
files also store the exact coordinates used for both model evaluations.

Predictions, quantum references, norms, cosines, original failure indices and
calculation settings are in the corresponding JSON entries. The entry named
excitation_gradient_comparison_20261112.json binds all three NPZ files by SHA-256
and gives the pre-existing target window for the explicitly post-hoc interval
gradient check. Files can be read with numpy.load(..., allow_pickle=False).

qualified20k_quantum_input_20261112.sdf is a separate five-structure selection
from eight surrogate-qualified outputs in 384 attempts. Its selection map,
full quantum coordinates/forces, per-target outcomes and unchanged-geometry
surrogate comparisons are in the qualified20k JSON entries. The three unselected
qualified outputs have unknown quantum outcomes. The label-overlap entry is a
post-hoc exact-graph check, not a reproduction of the original OCELOT quantum
geometry or tuning parameters. No fitted offset is applied.

The bond-perception entries retain all three methods on 511 known validation
geometries and 384 saved generated coordinates. Their identity comparison
normalizes explicit hydrogens while retaining specified stereo and isotopes.

sft_duration_20261112.json contains the complete 36-run matrix: three stochastic
continuations from a common trained 5000-step checkpoint, two predeclared
milestones, two target windows and three inference seeds. Per-attempt diagnostics,
held-out element predictions, input/checkpoint hashes and descriptive seed
variation remain attached to each model. The original 5000-step training is not
replicated; no best seed or checkpoint is selected by this report.

This archive does not redistribute upstream training datasets or model weights.
It supports inspection of reported evidence and the gradient counterexample,
not a complete model-training rerun. No arXiv submission has been performed by
the local archive builder.

OE62 extension: source-coordinate split and model evidence, a frozen 4762-molecule
holdout, separate pilot and formal three-generator-seed point-control pools,
independent HyFF reassessment, paired statistics, final wheel replay, matched
geometry optimization and unconditional training-fragment controls. Source
coordinates and weights remain separate; source DFT labels do not become
quantum labels for generated or optimized geometries.

OE62 quantum extension: two source-reference SDFs contain 13 reused and six
additional fixed source geometries. The point-quantum input SDF contains the
24 preselected formal GRPO200/time-SFT candidates. Corresponding JSON entries
retain source labels, input and optimized coordinates, gradient traces where
available, all failures, and fixed-model comparisons at identical geometries.
The primary protocol is PBE0/def2-TZVP with PBE-D3BJ/def2-TZVP optimization;
source FHI-aims Tier2/PBE-TS settings differ. The two-case QZVP/TZVPD diagnostic
is post-hoc and does not replace primary values. The original worker's geometry
"steps" field counts gradient callback evaluations, including the initial one.
