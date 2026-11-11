# Paired interval-training experiment

The expanded-element checkpoint has held-out weighted generation losses of
1.26236 under correct intervals and 1.26560 under shuffled intervals. This small
separation motivates an explicit experiment; it does not by itself prove that
conditioning is unused, since the diagnostic averages many token decisions.

Two continuations start from the identical 4000-step checkpoint. Both update only
the interval conditioner for 600 steps, batch 8, seed 42, learning rate 5e-5 with
cosine decay and weight decay 1e-4. Structural and property-head weights are frozen.
Both use the same 8188 labeled OCELOT conformers, rotations and batches, and full
intervals with normalized halfwidth 0.25. The control uses positive-condition
likelihood only. The treatment adds weight 0.5 times a per-molecule hinge:

`max(0, 0.02 + generation_loss(correct) - generation_loss(incompatible))`.

An incompatible condition is centered on the nearest other batch label whose
interval excludes the molecule's actual label in at least one target. Rows with
no incompatible label do not contribute to the paired term; their positive
likelihood remains trained. Within each molecule, tree and atom decisions are
averaged separately and XYZ losses carry the existing 0.1 weight. Negative
intervals use the same molecular trajectory as positive intervals. Every pair
must actually exclude the true label; identical/overlapping conditions cannot
become false negatives. The ordinary SFT loss and default model behavior are
unchanged when this experimental term is absent.

Both variants will be checked on the same 64 validation structures, then on the
original and central target windows with inference seeds 2026–2028, 64 proposals
per run and the same ranking/refinement budgets. The independent evaluator is the
new 15-element fingerprint ensemble. The previous test-set teacher-forcing
inspection is exploratory model-design feedback; it is not an untouched final
test for this intervention. Improved teacher-forcing separation alone will not
establish improved conditional generation. No default objective change is made
before the matched experiment.
