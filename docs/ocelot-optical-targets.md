# OCELOT vertical excitation targets

The publisher's README defines `s0s1` and `s0t1` as vertical first-singlet and
first-triplet excitation energies, in eV. Their per-molecule difference is a
vertical S1−T1 separation at the source S0 geometry. It is not an adiabatic gap
between separately optimized excited-state geometries or an emission energy.
The [OCELOT study](https://pubs.rsc.org/en/content/articlelanding/2023/sc/d2sc04676h)
describes molecule-specific IP-tuned LC-ωHPBE/Def2SVP DFT/TDDFT calculations.

The expanded supervised experiment predicts `homo`, `lumo`, `s0s1`, `s0t1`.
It reuses the exact audited ETKDG/UFF conformers and identifiers of the earlier
8192/512/512 3D experiment; no new conformer selection or relabeling is performed.
The original DFT geometries remain unavailable. A separate fingerprint ensemble
uses the existing 20356/2282/2322 scaffold split and the same four target names.
Both training scripts accept explicit ordered target lists and retain source
hashes and all prior geometry exclusions.

Each evaluation reports separate target MAEs plus the error of the predicted
S1−T1 difference for each molecule, then averages absolute errors. A difference
between ensemble means is separately identified and cannot be substituted for
MAE. Regression coverage includes opposite errors that cancel in the mean while
having nonzero paired MAE, mismatched sample counts and nonfinite pairs.

These are supervised predictions against the official CSV. They do not resolve
the independent PySCF/Gaussian protocol discrepancy or establish excited-state
minimum accuracy. The separate quantum calculations retain their own geometry,
state, functional, basis, omega and failure records.


`scripts/migrate_octree_targets.py` explicitly expands a trained generator's
ordered target interface. Existing target normalizations, conditioner blocks and
self-head rows are retained. New conditioner input columns and new readout rows
start at zero; new readouts therefore begin at their training-set means. The
existing conditioner behavior is mathematically preserved when added targets
are inactive, with floating-point comparisons checked separately. The migrated
checkpoint blocks conditional inference until labelled adaptation is completed.
It records both source checkpoint/data hashes and the new-target training status.
