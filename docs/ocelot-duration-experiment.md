# Longer labelled OCELOT adaptation

The matched 1000-step continuation of the wide 4000-step generator gives seven
optimizing-model and six independent-model joint hits in 384 proposals. The
paired-condition and element-weighting pilots do not improve both measures.
This follow-up tests ordinary labelled structural adaptation for longer, without
adding another reward or changing the sampling distribution.

The initial checkpoint is the unweighted 5000-step control from the completed
element-balance experiment. Continue for 15000 steps with all backbone parameters
trainable, the same 8188 training conformers, batch 8, backbone learning rate 1e-5,
conditioner/readout rate 2e-4, seed 42 and the existing cosine schedule. The
optimizer starts fresh; this is a continuation from weights, not an exact resumed
optimizer trajectory. Element cross entropy remains unweighted and GRPO is off.

Predeclare evaluation at total 10000 and 20000 SFT steps (5000 and 15000 steps
of this continuation). A stored 15000-total checkpoint is not used to choose the
reported model. Compare the same 511 encodable validation structures, original
and central target windows, inference seeds 2026–2028, 64 proposals per run and
the frozen wide two-target DPA/fingerprint evaluators. Default temperature 0.3,
ranking/refinement budgets and strict acceptance rules are unchanged. The existing
5000-step control supplies the baseline; no new baseline samples are selected.

Report validity, graph/charge diagnostics, rare-element occurrence, domain
eligibility, self/teacher-forcing diagnostics and joint success over all 384
proposals per checkpoint. A training-loss decrease or increased rare-element
frequency alone does not establish improved molecular design. The initial protocol specified one continuation seed; the predeclared extension
below adds two more before the milestone evaluations finish.

## Replicate continuation seeds

Before the seed-42 milestone evaluations finish, add seeds 43 and 44 with the
same 5000-step checkpoint, 15000 further SFT steps and both fixed milestones.
All training conformers, optimizer settings, sampling temperatures, target windows
and downstream evaluators remain fixed. Report every seed and both checkpoints;
no model is selected using the test labels or the best generation score.

These are three independent stochastic **SFT continuations from a common trained
starting point**. They do not measure variation in the original 5000-step training
or upstream pretraining. Inference seeds remain 2026–2028 for each trained model.
The portable `scripts/run_sft_duration_campaign.py` runs the continuation,
validation diagnostics and the same six evaluations per milestone. Isolated
source snapshots keep ongoing jobs independent of later repository edits.

## Completed three-seed matrix

All 36 evaluations are complete: three continuation seeds × two fixed milestones
× two windows × three inference seeds. The common starting checkpoint, training
structures and validation structures match by SHA-256. Every cell contains the
original 64 proposals, including failed and filtered candidates.

| Total SFT steps | Window | Attempts | Valid | In domain | DPA joint | Fingerprint joint |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5000 control | Original | 192 | 113 | 9 | 4 | 4 |
| 10000 | Original | 576 | 393 | 51 | 9 | 15 |
| 20000 | Original | 576 | 370 | 54 | 12 | 11 |
| 5000 control | Central | 192 | 108 | 13 | 3 | 2 |
| 10000 | Central | 576 | 376 | 41 | 9 | 8 |
| 20000 | Central | 576 | 349 | 41 | 10 | 10 |

The 5000-step control is one trained model; its 192 proposals per window are not
triplicated to manufacture three baseline training seeds. For the continuations,
per-seed joint counts below are ordered 42 / 43 / 44, each out of 192:

| Total steps / window | DPA joint by training seed | Fingerprint joint by training seed |
| --- | --- | --- |
| 10000 / original | 3 / 4 / 2 | 2 / 7 / 6 |
| 10000 / central | 2 / 4 / 3 | 2 / 4 / 2 |
| 20000 / original | 3 / 4 / 5 | 3 / 3 / 5 |
| 20000 / central | 5 / 4 / 1 | 5 / 3 / 2 |

At 20000 steps the mean DPA joint rates are 2.08% / 1.74% for original/central,
with across-training-seed SDs of 0.52 / 1.08 percentage points. Fingerprint rates
are 1.91% / 1.74%, with SDs of 0.60 / 0.80 points. These are descriptive variation
estimates from three continuations, not significance tests.

Pooling both windows, longer training changes validity from 769/1152 to 719/1152,
domain eligibility from 92 to 95, optimizing-model joint hits from 18 to 22, and
independent-model joint hits from 23 to 21. The direction varies by seed and
window. This budget of additional SFT does not resolve the proposal-quality
bottleneck or establish a consistent design improvement.

Element learning does improve. At 20000 steps, all three models recover 8/12 B
and 27/28 Si teacher-forced validation decisions, versus zero for both in the
5000-step control. C accuracy stays 99.64–99.70%. Valid B-containing generated
outputs increase from 16 to 67 and Si-containing outputs from 67 to 74, each
counted across 1152 attempts. One valid Te-containing output occurs; Se/As remain
absent, and all three one-decision Se/As/Te validation categories still have zero
top-1 accuracy. These small counts do not establish reliable rare-element design.

The [complete machine-readable matrix](experiments/sft_duration_20261112.json)
retains all runs, rare-element/charge diagnostics, evaluator calls, model/input
hashes and descriptive seed statistics. The earlier
[seed-42-only snapshot](experiments/sft_duration_seed42_20261112.json) remains
labelled incomplete as it was when published; it is superseded by the full matrix.
No default objective, sampling temperature or best-seed checkpoint is selected
from these outcomes. The separately packaged seed-42 total-20000 checkpoint and
its quantum selection were fixed before the other seeds completed.
