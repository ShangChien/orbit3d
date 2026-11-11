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
frequency alone does not establish improved molecular design. This remains an
exploratory duration study with one SFT seed.

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

## First completed continuation (seed 42; other seeds pending)

| Total SFT steps | Window | Attempts | Valid | In domain | DPA joint | Fingerprint joint |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5000 control | Original | 192 | 113 | 9 | 4 | 4 |
| 10000 | Original | 192 | 122 | 9 | 3 | 2 |
| 20000 | Original | 192 | 126 | 15 | 3 | 3 |
| 5000 control | Central | 192 | 108 | 13 | 3 | 2 |
| 10000 | Central | 192 | 112 | 9 | 2 | 2 |
| 20000 | Central | 192 | 119 | 18 | 5 | 5 |

At 10000 steps, valid B/Si-containing outputs total 6/26; at 20000, 23/27.
The 10000-step teacher-forced Si accuracy is 25/28 versus 0/28 for the 5000-step
control. These are improved element observations, while end-to-end gains remain
small and target-dependent. This is one stochastic continuation, not a
three-seed conclusion. The public `experiments/sft_duration_seed42_20261112.json`
explicitly marks the three-seed matrix incomplete and leaves the training-seed
standard deviation undefined. Seeds 43 and 44 remain in progress.
