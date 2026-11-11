# Training-fragment recombination baseline

This comparison uses RDKit's [BRICS graph fragmentation and construction](https://rdkit.org/docs/source/rdkit.Chem.BRICS.html).
Every fragment is obtained by cutting exactly one BRICS bond of an existing
training molecule. Each piece therefore has one labelled attachment point. The
library retains its training SourceIDs and exact source hash; no validation/test
structures or property labels enter fragment choice.

For each candidate seed, sample a compatible attachment-label pair with probability
proportional to the number of fragment pairs, then sample one fragment from each
group. RDKit joins that single seed with that single partner at depth zero; there
is no recursive search or replacement of a failed conformer with another product.
If more than one complete product is enumerated, choose uniformly from sorted
unique canonical SMILES. The attempt ledger retains both fragments and the number
of products considered. Candidates over 80 explicit atoms or the depth-seven
rotation-safe span remain failed attempts. Geometry uses the same ETKDGv3/UFF
100-step preparation as the wide OCELOT predictor, with per-candidate seeds.

The pilot uses the same 8188 training conformers as the wide generator, the
original and central intervals, three inference seeds and 64 proposals each.
The two windows reuse each seed's unconditional graph pool. Domain filtering
precedes the same 32-call external-ranking cap and 16-retained cap, followed by
30 constrained-refinement steps of size 0.004 with zero energy/property slack.
The reference generator's saved target scales determine ranking normalization.
The frozen wide two-target DPA adapter and independent fingerprint evaluator are
unchanged. Invalid graphs, unsupported geometry, duplicates and OOD molecules
remain in the original denominator.

The baseline has no learned self-shortlist: eligible molecules enter the external
budget in candidate order. The octree uses its learned head, although its observed
domain-eligible counts are below the external budget in the matched wide runs.
Candidate/iteration limits are matched; actual evaluator calls and total compute
are measured separately, not asserted equal. Library construction and conformer
costs are retained, and model-loading reuse is not a cross-method speed claim.
Exact training-set novelty and graph/charge diagnostics distinguish recombination
from simply retrieving existing training molecules. No synthetic-route guarantee
is inferred from a BRICS-compatible bond.

Run with the portable script:

```bash
python scripts/evaluate_brics_baseline.py \
  --training-sdf /path/to/ocelot_wide_generator8192/train.sdf \
  --reference-checkpoint /path/to/unweighted_5000_step.pt \
  --config /path/to/wide_two_target_config.json \
  --windows /path/to/original_and_central_windows.json \
  --independent-oracle /path/to/fingerprint_wide/oracle \
  --output /path/to/fresh_brics_run
```

## Completed three-seed comparison

Every row has 192 attempted candidates. The octree comparison is the unweighted
5000-step control, using the same 8188 training conformers and target scaling.

| Method / interval | Valid 3D | In domain | Retained | DPA joint | Fingerprint joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Octree / original | 113 | 9 | 9 | 4 | 4 |
| BRICS / original | 173 | 173 | 48 | 10 | 4 |
| Octree / central | 108 | 13 | 13 | 3 | 2 |
| BRICS / central | 173 | 173 | 48 | 45 | 45 |

BRICS's independently trained HyFF evaluator gives 13 and 44 joint hits for the
original and central windows. Using either the original DPA force or HyFF force
criterion gives the same counts. All 48 retained products per window are novel
relative to the adaptation structures. Their mean SA scores are 2.70 and 2.75;
this is a graph heuristic, not a demonstrated synthesis route.

The baseline substantially outperforms this octree control on the central window
under both original evaluators. In the original window, the fingerprint joint
count ties the octree, despite the DPA count being higher. Thus there is no uniform
ranking across targets and evaluators. Low domain coverage of octree proposals is
a concrete bottleneck; successful fragment recombination does not establish that
3D autoregression cannot improve with further training.

Actual external ranking calls are 96 per BRICS window versus 9/13 for octree.
BRICS refinement evaluation/gradient calls are 1572/1071 (original) and 1548/65
(central); octree uses 235/90 and 462/148. These differ despite matched maximum
candidate and iteration budgets. BRICS library construction took 17.16 seconds
once. The two windows reuse the same 192 unconditional proposals, so the 384
condition-specific attempts are not independent graph samples. Timings were taken
on a shared GPU and are not a throughput comparison.

Rebuild the table from all six per-method records, preserving failures and seeds:

```bash
python scripts/summarize_brics_comparison.py \
  --output docs/experiments/brics_comparison_20261112.json
```

The original per-attempt graph actions, input hashes, independent evaluations and
aggregate are preserved in `docs/experiments/brics_*_20261112.json`.
