# Fixed-geometry excitation-gradient comparison

The surrogate coordinate derivative has passed finite-difference checks of the
surrogate itself. That verifies differentiation, not agreement with the physical
excitation response. This experiment compares both DPA and HyFF adapter derivatives
with full-TDDFT vertical S1/T1 derivatives on all five completed structures of the
predeclared ten-molecule boron cohort. All five are outside the OCELOT applicability
domain; no member is selected using a favorable gradient or prediction.

Keep the exact saved optimized coordinates, original per-molecule omega, functional,
basis, grid, SCF and response tolerances. Do not optimize, retune, use TDA, or change
density-fitting conventions. The excitation derivative is the total excited-state
gradient minus the S0 energy gradient, converted from Hartree/Bohr to eV/Angstrom.
The ground reference must reproduce within 1e-6 Ha and the excitation root within
1e-4 eV before comparing gradients; otherwise retain a failed record.

Report per-molecule and per-state gradient norms, cosine similarity, Cartesian
absolute disagreement and the quantum directional slope along the unit negative
surrogate gradient. Positive slope means that locally decreasing the surrogate
excitation would increase the corresponding quantum excitation. This is a local
fixed-omega derivative diagnostic, not a state optimization or proof of in-domain
model accuracy. Original quantum failures remain separate; new gradient failures
are also retained rather than replaced with zeros or simpler calculations.

Run `scripts/export_quantum_geometry_gradients.py` once per frozen property model
in the molecular-model environment, then `scripts/compare_excitation_gradients.py`
in the existing validated quantum environment. This keeps the two dependency
stacks separate and passes only coordinate- and cohort-bound numeric artifacts.
