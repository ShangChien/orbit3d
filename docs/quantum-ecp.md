# Basis and effective-core-potential consistency

PySCF requires an ECP to be specified separately from the orbital basis. Its
[official molecular-input documentation](https://pyscf.org/user/gto.html#ecp)
lists the matching scalar def2 ECPs. The previous quantum constructor supplied
`basis='def2-svp'` without an ECP. This has no effect on the light elements used in
the published boron cohort, but is wrong for the valence basis on Te and I.

A real PySCF check reproduces the error: neutral HI has 54 explicitly treated
electrons with the old input, instead of 26 with I's 28-electron core potential.
TeH2 likewise requires removal of 28 Te core electrons. This is a Hamiltonian
error, not a geometry or batch-size problem.

`make_quantum_molecule` now constructs the orbital basis and matching scalar def2
ECP together. All ground, excited-state, gradient and Hessian entrypoints use
this constructor. New runs record `ecp_policy` and the actual core-electron counts.
`theory_from_record` assigns the old no-ECP convention to records predating this
field, preserving what was actually calculated instead of relabeling old results.
Recovery from an SCF sidecar also preserves that recorded policy.

Three real PySCF regressions on the remote environment verify HI/HI+ and TeH2
electron counts, identical HBr overlap/nuclear-attraction integrals for the
unchanged light-element Hamiltonian, and explicit preservation of legacy records.
The macOS core suite passes 162 tests; these three optional-PySCF tests skip there
and pass in the remote quantum environment.

The first fixed-omega HI CPU/GPU test reproduces energies within its 1e-7 Ha
criterion but **fails** the predeclared 1e-5 eV/Angstrom force-parity threshold
(maximum difference 1.95e-5). A separate probe identifies an additional numerical
issue: omitting atom-grid response changes the HI derivative by about 4.8e-4
eV/Angstrom. Full grid response agrees with a 0.001 Angstrom energy difference
within 1.65e-5 eV/Angstrom at the original SCF tolerance. These measurements are
retained; the ECP fix alone is not a claim that all gradient-accuracy gates pass.

`scripts/validate_quantum_ecp.py` reports energy, force-parity and finite-difference
checks separately, with unchanged thresholds and explicit SCF/grid-response
settings. A tighter 1e-11 SCF and full-grid-response diagnostic is separate from
the initial failed run. The standard cohort protocol is not changed midway through
an experiment, and no GPU-to-CPU fallback is hidden in the evaluator.
