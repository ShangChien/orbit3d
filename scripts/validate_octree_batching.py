"""Measure token replay across batch sizes, order and precision on real weights."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from orbit3d.generator.octree_backend import load_octree_checkpoint
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.utils.reproducibility import lock_deterministic_environment

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()
lock_deterministic_environment(42)
torch.set_num_threads(4)
model, state = load_octree_checkpoint(a.checkpoint, a.device)
lo = torch.tensor(
    (np.array([-8.2, -0.6]) - state["mean"]) / state["scale"], dtype=torch.float32, device=a.device
)
hi = torch.tensor(
    (np.array([-7.2, 0.8]) - state["mean"]) / state["scale"], dtype=torch.float32, device=a.device
)
reference = {}
report = {}
for precision in ["float32", "bfloat16"]:
    reports = []
    for size, reverse in [(1, False), (8, False), (48, False), (8, True)]:
        result = {}
        groups = [[i] for i in range(4)] if size == 1 else [list(range(size))]
        if reverse:
            groups = [list(reversed(ids)) for ids in groups]
        fixed_trajectories = []
        for ids in groups:
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bfloat16"),
            ):
                rows = model.generate_conditional_trajectories(
                    lo.repeat(len(ids), 1),
                    hi.repeat(len(ids), 1),
                    torch.ones(len(ids), 2, dtype=torch.bool, device=a.device),
                    evaluate_self_consistency=False,
                    row_seeds=[candidate_seed(2026, i) for i in ids],
                )
            for i, row in zip(ids, rows):
                if i < 4:
                    result[i] = row["decoder_feat"].cpu().numpy()
                    if size == 1:
                        fixed_trajectories.append(row)
        if size == 1:
            reference[precision] = result
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bfloat16"):
                together = model.trajectory_distributions(fixed_trajectories, lo.repeat(4, 1), hi.repeat(4, 1), torch.ones(4, 2, dtype=torch.bool, device=a.device))
                max_error = 0.0
                for i, row in enumerate(fixed_trajectories):
                    alone = model.trajectory_distributions([row], lo[None], hi[None], torch.ones(1, 2, dtype=torch.bool, device=a.device))
                    for (batched, labels, owner), (single, slabels, _) in zip(together, alone):
                        selected = batched[owner == i]
                        assert torch.equal(labels[owner == i], slabels)
                        finite = torch.isfinite(selected) & torch.isfinite(single)
                        assert torch.equal(torch.isfinite(selected), torch.isfinite(single))
                        if finite.any():
                            max_error = max(max_error, float((selected[finite] - single[finite]).abs().max()))
            report[precision + "_max_fixed_trajectory_logprob_error"] = max_error
        divergent = [i for i in range(4) if not np.array_equal(result[i], reference[precision][i])]
        reports.append({"batch_size": size, "reversed_order": reverse, "tracked_candidates": 4, "token_divergent_ids": divergent})
        print(precision, reports[-1], flush=True)
    report[precision] = reports
# Exact row streams are tested separately; this assertion covers numerical logits' effect on tokens.
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + "\n")
assert not any(row["token_divergent_ids"] for row in report["float32"]), "FP32 batch replay diverged"
