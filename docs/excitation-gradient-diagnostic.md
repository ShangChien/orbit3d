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

## Completed five-case result

All five cases reproduce their stored restricted ground energies and vertical
roots within the predeclared thresholds and complete both excitation gradients.
Native and differentiable surrogate predictions agree exactly on these inputs.
The gradient computations took 606.8 seconds in total on a shared A800.

| Model | Mean S1 cosine | S1 range | Mean T1 cosine | T1 range |
| --- | ---: | ---: | ---: | ---: |
| DPA adapter | 0.630 | 0.345–0.819 | 0.658 | 0.500–0.771 |
| HyFF adapter | 0.422 | 0.326–0.555 | 0.573 | 0.326–0.782 |

Every individual-property direction has positive alignment. Gradient magnitudes
remain different: DPA norms are 0.31–0.73 times the corresponding quantum norms;
HyFF norms span 0.32–1.17. This supports informative local directions on these five
cases, not calibrated derivative magnitudes or accuracy on other structures.

A **post-hoc** check uses the already defined vertical target window, S1 in
[2.5, 3.5] eV and T1 in [1.5, 2.5] eV. In cohort case 2, DPA predicts S1 =
3.5425 eV while the recomputed quantum S1 is 2.2770 eV. The property gradients
have cosine +0.819, but the two predictions violate opposite bounds, so the
squared-interval-loss gradients have cosine **−0.819**. Descending the surrogate
interval loss would locally increase the quantum interval loss. DPA also assigns
zero interval loss to two quantum-infeasible cases; HyFF does so for two cases.

This is a scalar-prediction error changing the desired direction, not a broken
autograd implementation. All five structures are out of domain and are already
excluded from qualified pipeline outputs. The example does not justify replacing
the model gradient with an arbitrary sign or relaxing domain checks. It explains
why the acceptance rule is conditional on evaluator accuracy and cannot certify
quantum target preservation.

JSON records, input hashes and all three raw gradient arrays are available as
`experiments/boron_excitation_gradients_20261112.*` and
`experiments/boron_quantum_{dpa,hyff}_gradients_20261112.*`. Recompute the descriptive
summary and interval example with:

```bash
python scripts/summarize_excitation_gradients.py \
  --output docs/experiments/excitation_gradient_comparison_20261112.json
```
