# Complete generated-boron quantum cohort

All ten chemically valid B-containing molecules from a fixed 64-proposal structural
generation run were selected before quantum calculation. No property or force
preselection removed a member. All ten completed the earlier 50-step DPA energy
relaxation; nine then had RMS force below 0.5 eV/Å.

The original quantum protocol completes S0 optimization, graph checks and full
spin-adapted TDDFT for five of ten molecules. Its budgets are 100 SCF cycles and
80 geometry steps per optimization. The calculation level is the same internally
defined IP-tuned LC-wPBE/def2-SVP protocol across the cohort, with the recorded grid
and convergence tolerances, no density fitting and no TDA substitution.

| Cohort index | Original candidate ID | Initial DPA-relaxed RMS (eV/Å) | Quantum outcome |
| --- | --- | ---: | --- |
| 0 | 6 | 0.2372 | Completed ground state and TDDFT |
| 1 | 7 | 2.0589 | SCF did not converge at omega=0.0862, charge=1 |
| 2 | 8 | 0.1508 | Completed ground state and TDDFT |
| 3 | 23 | 0.2570 | Quantum optimization changed the molecular graph or stereochemistry |
| 4 | 26 | 0.2661 | stereo_geometry: coordinates disagree with a specified double-bond configuration |
| 5 | 34 | 0.1913 | Completed ground state and TDDFT |
| 6 | 36 | 0.2718 | Geometry optimization did not converge |
| 7 | 51 | 0.1535 | Completed ground state and TDDFT |
| 8 | 58 | 0.2873 | bond_geometry: 35-7 (H-C) length 1.816837 A |
| 9 | 62 | 0.2245 | Completed ground state and TDDFT |

| Completed index | HOMO (eV) | LUMO (eV) | Vertical S1 (eV) | Vertical T1 (eV) | S1–T1 (eV) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | -6.7160 | 0.1891 | 3.6260 | 2.3549 | 1.2712 |
| 2 | -8.0341 | -0.6488 | 2.2769 | 1.7825 | 0.4944 |
| 5 | -7.3192 | -2.4475 | 1.7557 | 1.1977 | 0.5580 |
| 7 | -7.6506 | -0.4032 | 3.8086 | 2.7404 | 1.0683 |
| 9 | -6.3905 | -0.7195 | 2.6159 | 1.5986 | 1.0172 |

SCF/geometry failures stay in the ten-molecule denominator. The five completed
rows alone cannot estimate accuracy or stability across all attempted designs.
No vibrational minimum claim is made for this cohort. The generator received
structural B adaptation only; these outputs do not demonstrate B-conditioned
property design. The values are vertical excitations at the optimized S0 geometry,
not adiabatic or emission energies.

Longer-budget retries preserve the original failures. Raising molecule 1’s SCF
budget to 200 resolves the SCF iteration-limit issue but subsequently fails graph
preservation. Molecule 6’s separate 160-step retry completes geometry/omega convergence but fails the specified double-bond stereo check; its RMS force is 0.00557 eV/Å. The original numerical failure is retained alongside this later chemical failure.
Recovered final SCF states for molecules 3, 4 and 8 have small residual forces
despite graph/stereo failures, including new bonding and hydrogen transfer.

The official OCELOT release lacks per-molecule omega and original DFT geometries,
and its naphthalene labels have not been reproduced by this implementation. These
are internally consistent protocol diagnostics, not a reproduced OCELOT accuracy
benchmark. [Full paired records and original manifest](experiments/boron_quantum_cohort_20261112.json)
preserve the exact source hash, settings, coordinates, forces and all failures.

## Paired surrogate diagnostic at the quantum geometries

All five completed structures are outside the declared wide-OCELOT applicability
domain; the generation workflow would exclude them from qualified designs. For
transparency, the frozen four-target models were nevertheless evaluated on the
same saved optimized geometries as an extrapolation diagnostic:

| Model | HOMO | LUMO | Vertical S1 | Vertical T1 | Paired S1–T1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPA property adapter | 0.4355 | 0.3303 | 0.5212 | 0.2120 | 0.4104 |
| HyFF property adapter | 0.4171 | 0.2614 | 0.4009 | 0.1879 | 0.2693 |
| Fingerprint ensemble | 0.2889 | 0.5443 | 0.9038 | 0.6094 | 0.3451 |

Values are mean absolute disagreements in eV on the five completed cases, against
the internally defined quantum protocol. The five failures remain in the original
cohort; they do not receive fabricated property values. These results are neither
OCELOT holdout MAEs nor evidence of accuracy in the model's applicability domain.
DPA forces at these quantum geometries are also retained as cross-potential
diagnostics; they do not compare identical-theory training forces. [Every paired
value, domain decision and input hash](experiments/boron_quantum_oracles_20261112.json)
is available in the public record.

The [original ten generated input geometries](experiments/boron_quantum_input_20261112.sdf)
are included with the exact manifest checksum. The original worker version saved
precise optimized SDFs but omitted full coordinate/force arrays from its JSON.
Those arrays have now been recovered from the saved final SCF states without
repeating geometry optimization. Coordinates agree with the optimized SDFs within
5.1e-13 Å. Each recovery retains its own scalar values and differences from the
original result; it does not overwrite the original SCF/TDDFT measurements.


The later HyFF row uses the same five optimized SDFs and exactly the same quantum
reference scalars; its fingerprint predictions reproduce the earlier comparison
exactly. No geometry optimization, omega retuning or calculation-level change was
introduced. HyFF's RMS forces at those geometries range from 0.507 to 0.853 eV/Å,
so the independent potential also disagrees with the quantum stationary forces.
These models were not trained on this exact quantum level; that range is not an
identical-theory force MAE. All five remain outside the HyFF property model's domain.

The comparison now has a portable entrypoint rather than requiring private
working-directory assumptions:

```bash
python scripts/compare_quantum_oracles.py   --quantum-run /path/to/completed_quantum_run   --source-sdf /path/to/original_cohort.sdf   --config /path/to/four_target_dpa_or_hyff_config.json   --fingerprint-model /path/to/fingerprint_oracle   --output /path/to/fresh_comparison
```

It verifies the original cohort checksum, atom-indexed graph identity and target
units, uses every completed quantum record and retains the original failure
indices. Raw independent forces and input hashes accompany the report. The new
paired results and manifest are `experiments/boron_quantum_hyff_20261112.json`
and `experiments/boron_quantum_hyff_inputs_20261112.json`.
