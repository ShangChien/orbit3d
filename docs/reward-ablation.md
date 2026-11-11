# GRPO reward ablations

The completed matrix contains five objectives, training seeds 42–44 and inference
seeds 2026–2028: 45 runs, 64 proposals per run. Every policy starts from one shared
v3 SFT checkpoint and receives 50 rollout iterations, two optimizer passes per
group and 16 proposals per rollout. The forcefield, Oracle, applicability domain,
intervals, external-ranking cap and strict 30-step refinement protocol are fixed.
The uncertainty treatment uses weight 1; the Full baseline uses weight 0.

| Objective | Valid / 576 | Optimizing joint / 576 | Independent joint / 576 |
| --- | ---: | ---: | ---: |
| full | 482 | 20 | 20 |
| no kl | 483 | 14 | 17 |
| no physics | 482 | 14 | 17 |
| no replay | 484 | 16 | 19 |
| uncertainty on | 489 | 17 | 20 |

| Objective | Optimizing counts, train seeds 42/43/44 (each /192) | Independent counts (each /192) |
| --- | --- | --- |
| full | 8/7/5 | 8/5/7 |
| no kl | 2/3/9 | 3/5/9 |
| no physics | 3/6/5 | 5/5/7 |
| no replay | 7/4/5 | 8/7/4 |
| uncertainty on | 6/7/4 | 8/5/7 |

Full has more aggregate hits than the removal controls, but the effect is not
uniform across seeds: no-KL seed 44 outperforms Full under both evaluators. Adding
the uncertainty term leaves the independent total unchanged and lowers the
optimizing total. These small results do not establish a robust component ranking.
All models share one SFT initialization; SFT variability is not measured.
The independent predictor is a separately trained fingerprint model, not DFT truth.

The report gives per-training-seed rates, sample standard deviations, ranges and
all 27 training-cluster bootstrap resamples through their percentile interval.
With only three clusters these are descriptive sensitivity intervals, not reliable
95% coverage or a significance test. Inference samples are not substituted for
independent training replications.

[Complete 45-run record](experiments/grpo_factorial_20261112.json) contains source
hashes and every cell. The assembly script rejects incomplete, duplicate or
wrong-budget matrices. The earlier [seed-42 pilot](experiments/reward_ablation_20261112.json)
remains archived; it is no longer the only evidence for the reward comparisons.

A stage-bundle comparison gives unconditional-prior 1/192 optimizing and 2/192
independent hits versus SFT 4/192 for both. The prior uses verified EMA weights
and constant shortlist scores. This changes both conditioning and self-shortlisting,
so it is not a single-factor conditioning intervention.
[Stage records](experiments/stage_and_training_seeds_20261112.json) retain all counts.
