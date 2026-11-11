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
