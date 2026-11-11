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
preservation. Molecule 6’s 160-step geometry retry is separate and still running.
Recovered final SCF states for molecules 3, 4 and 8 have small residual forces
despite graph/stereo failures, including new bonding and hydrogen transfer.

The official OCELOT release lacks per-molecule omega and original DFT geometries,
and its naphthalene labels have not been reproduced by this implementation. These
are internally consistent protocol diagnostics, not a reproduced OCELOT accuracy
benchmark. [Full paired records and original manifest](experiments/boron_quantum_cohort_20261112.json)
preserve the exact source hash, settings, coordinates, forces and all failures.
