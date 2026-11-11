# Refinement rules on identical starting structures

The three full-GRPO source runs contribute 192 proposals and 34 selected molecules.
All four rules start from each selected molecule's saved **pre-refinement**
coordinates and inferred graph. Selection is fixed. Each rule has at most 30
steps with initial step size 0.008 and whole-molecule displacement cap 0.025 Å.
The production source configuration uses 0.004; this table is the explicitly
recorded 0.008 step-size diagnostic. A second control uses the source step size. This
is a step-budget comparison; acceptance/backtracking changes evaluator cost.

| Rule | Optimizing property hits | Optimizing joint hits / 192 | Independent joint hits / 192 | Mean final RMS force (eV/Å) |
| --- | ---: | ---: | ---: | ---: |
| No refinement | 6 | 0 | 0 | 2.5858 |
| Energy only | 6 | 6 | 17 | 0.1243 |
| Projection without acceptance | 6 | 5 | 7 | 1.4452 |
| Full per-target/energy acceptance | 8 | 8 | 8 | 1.3572 |

All 34 structures completed each rule. Pure energy minimization gives the lowest
mean force; full acceptance gives the most joint successes in this cohort. Full
acceptance can stagnate at large residual force, so the average across retained
structures must not be confused with the qualified subset's force threshold.
Both energy-only and projection-only refinement lose one of six initially
property-feasible molecules; full acceptance preserves all six. The independent
fingerprint evaluator instead favors energy-only refinement (17 versus 8 joint
hits): it assigns the same 17 property hits to every rule and cannot detect
geometric property changes. This disagreement directly limits an effectiveness
claim based on the optimizing Oracle. No rule has established superiority under
quantum reference evaluation or across training seeds and target intervals.

[Full per-molecule traces and independent checks](experiments/refinement_ablation_20261112.json)
retain the complete 192-attempt denominator for every rule. This is a matched
re-evaluation using serial float64 physical energy reporting, not a merge with
older energy32 experiments.
