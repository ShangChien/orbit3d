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
