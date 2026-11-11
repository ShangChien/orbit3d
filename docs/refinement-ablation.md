# Refinement rules on identical starting structures

The three full-GRPO source runs contribute 192 proposals and 34 selected molecules.
All four rules start from each selected molecule's saved **pre-refinement**
coordinates and inferred graph. Selection is fixed. Each rule has at most 30
steps with the production initial step size 0.004 and whole-molecule displacement
cap 0.025 Å. This is a step-budget comparison; acceptance/backtracking changes evaluator cost.

| Rule | Optimizing property hits | Optimizing joint hits / 192 | Independent joint hits / 192 | Mean final RMS force (eV/Å) |
| --- | ---: | ---: | ---: | ---: |
| No refinement | 6 | 0 | 0 | 2.5858 |
| Energy only | 7 | 7 | 17 | 0.1709 |
| Projection without acceptance | 7 | 7 | 8 | 1.0572 |
| Full per-target/energy acceptance | 8 | 8 | 8 | 1.3537 |

All 34 structures completed each rule. Pure energy minimization gives the lowest
mean force. Under the optimizing 3D Oracle, full acceptance gives 8 joint hits
versus 7 for energy-only; the independent evaluator reverses that ranking. Full
acceptance can stagnate at large residual force, so the average across retained
structures must not be confused with the qualified subset's force threshold.
Both energy-only and projection-only refinement lose one of six initially
property-feasible molecules; full acceptance preserves all six. The independent
fingerprint evaluator instead favors energy-only refinement (17 versus 8 joint
hits): it assigns the same 17 property hits to every rule and cannot detect
geometric property changes. This disagreement directly limits an effectiveness
claim based on the optimizing Oracle. No rule has established superiority under
quantum reference evaluation or across training seeds and target intervals.

[Full per-molecule traces and independent checks](experiments/refinement_matched_20261112.json)
retain the complete 192-attempt denominator for every rule. This is a matched
re-evaluation using serial float64 physical energy reporting, not a merge with
older energy32 experiments.

An earlier explicitly recorded [0.008 step-size diagnostic](experiments/refinement_ablation_20261112.json)
gives optimizing joint counts 0/6/5/8 and independent counts 0/17/7/8 in the same
rule order. It is kept separately rather than presented as the production protocol.
