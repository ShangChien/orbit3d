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

The pilot JSON and provenance describe a separate historical octree/GVP protocol.
They must not be merged with current unified-package success denominators.

This archive does not redistribute upstream training datasets or model weights.
It supports inspection of reported evidence and the gradient counterexample,
not a complete model-training rerun. No arXiv submission has been performed by
the local archive builder.
