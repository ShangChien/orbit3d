# GRPO reward ablations

The four models start from the same v3 OCELOT conditioner and use training seed 42,
50 updates, 16 candidates per group, identical intervals, and the same forcefield,
Oracle and applicability domain. All physical energies use the float64 reporting
path. Evaluation uses seeds 2026–2028, 64 attempts each, even-electron sampling,
external ranking and 30 refinement steps with strict zero acceptance tolerances.

| Training objective | Valid / 192 | Optimization-Oracle joint hits / 192 | Independent-surrogate joint hits / 192 |
| --- | ---: | ---: | ---: |
| Full | 161 | 8 | 8 |
| KL weight 0 | 159 | 2 | 3 |
| Physics weight 0 | 158 | 3 | 5 |
| SFT replay weight 0 | 169 | 7 | 8 |

The full objective performs better than the no-KL/no-physics controls in this
pilot. Replay does not improve the independent joint count. These are one
training-seed comparisons; the three inference seeds do not establish robustness
across training runs. The independent model is an OCELOT fingerprint ensemble,
not DFT ground truth. No uncertainty penalty was used because held-out ensemble
spread has weak error-ranking ability.

[Raw per-run metrics, independent results and artifact hashes](experiments/reward_ablation_20261112.json)
record all denominators and retained pools. The old energy32 experiment is kept
separately and is not a causal control for these results.


Additional full-objective training seeds 43 and 44 are complete. Each is evaluated
with the same three inference seeds and 192-attempt denominator:

| Training seed | Optimizing joint hits / 192 | Independent joint hits / 192 |
| --- | ---: | ---: |
| 42 | 8 | 8 |
| 43 | 7 | 5 |
| 44 | 5 | 7 |

Both evaluators total 20/576 (3.47%); the across-training-seed standard deviation
is 0.80 percentage points for each. All runs share one SFT checkpoint, so SFT
training variability is not measured. The remaining reward controls are being
replicated at seeds 43/44, along with an explicit nonzero-uncertainty treatment.

A matched stage-bundle control gives unconditional-prior 1/192 optimizing and
2/192 independent joint hits, versus SFT 4/192 on both. The prior uses verified
upstream EMA weights and zero identity conditioning/constant shortlist scores.
SFT changes both interval conditioning and the self-shortlist, so this is not an
isolated conditioning intervention. [All stage and training-seed records](experiments/stage_and_training_seeds_20261112.json)
retain each denominator, checkpoint hash and independent evaluation.
