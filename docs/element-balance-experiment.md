# Element-balanced supervision experiment

The expanded generator recognizes common elements in held-out teacher-forced
contexts but gives zero top-1 accuracy to the sampled B/Si/Se/As/Te decisions.
Rare elements have far fewer atom labels than C/H, so this experiment changes
their contribution to supervised element prediction rather than only changing
the sampling temperature.

From the supplied training SDF only, compute `w_element = sqrt(max_count/count)`.
Element cross entropy is the sum of weighted decision losses divided by the sum
of weights on those labels. Special tokens keep weight 1; tree and XYZ losses
are unchanged. No validation/test counts enter these weights. The checkpoint
records counts, weights and the exact reduction. The optional configuration is
`generator_config.element_loss_weighting: inverse_sqrt_frequency`; default `none`
preserves the existing objective.

The matched pilot starts both models from the same expanded-element 4000-step
checkpoint and uses 1000 further SFT steps, batch 8, seed 42, the same 8188 training
conformers, backbone learning rate 1e-5 and conditioner/head learning rate 2e-4.
Both use the existing v3 interval sampler and train all backbone parameters.
One uses the original objective, the other the weighted element term. No GRPO
is applied. Validation uses the same 511 encodable structures from a 512-record pool and per-element probabilities; one spatially incompatible structure is retained in the preflight exclusion ledger. No coordinates or labels are clipped.
free generation uses the original and central windows, three inference seeds
and 64 proposals each, at the unchanged atom temperature 0.3. The new independent
fingerprint model evaluates every retained candidate.

Weighted training losses are not directly comparable to unweighted losses.
Judgment uses per-element held-out errors, loss of common-element quality,
chemical validity, raw element occurrence, graph/charge diagnostics and joint
success over every proposal. Increased rare-element frequency alone is not a
successful molecular design result. The earlier test-set diagnosis is retained
as exploratory feedback, not used to advertise a final generalization score.
