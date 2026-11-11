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
