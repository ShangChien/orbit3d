"""Record sampling latency, raw trajectories and CPU operator counts on real weights."""

import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from orbit3d.generator.octree_backend import load_octree_checkpoint
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--batch-size", type=int, default=8)
a = p.parse_args()
freeze_run(a.output, __file__)
record_input_hashes(a.output, {"checkpoint": a.checkpoint})
lock_deterministic_environment(42)
torch.set_num_threads(4)
model, state = load_octree_checkpoint(a.checkpoint, "cuda:0", a.batch_size)
lo = torch.tensor(
    (np.array([-8.2, -0.6]) - state["mean"]) / state["scale"], dtype=torch.float32, device="cuda:0"
).repeat(a.batch_size, 1)
hi = torch.tensor(
    (np.array([-7.2, 0.8]) - state["mean"]) / state["scale"], dtype=torch.float32, device="cuda:0"
).repeat(a.batch_size, 1)
mask = torch.ones_like(lo, dtype=torch.bool)
seeds = [candidate_seed(2026, i) for i in range(a.batch_size)]


def sample():
    with torch.no_grad():
        return model.generate_conditional_trajectories(
            lo, hi, mask, evaluate_self_consistency=False, row_seeds=seeds, electron_parity=0
        )


sample()  # excluded warm-up
seconds = []
reference = None
for repeat in range(3):
    torch.cuda.synchronize()
    start = time.perf_counter()
    rows = sample()
    torch.cuda.synchronize()
    seconds.append(time.perf_counter() - start)
    if reference is None:
        reference = rows
    assert all(torch.equal(x["decoder_feat"], y["decoder_feat"]) for x, y in zip(reference, rows))
arrays = {}
for i, row in enumerate(reference):
    arrays[f"tokens_{i}"] = row["decoder_feat"].cpu().numpy()
    arrays[f"positions_{i}"] = row["positions_np"]
    arrays[f"scores_{i}"] = row["scores"].cpu().numpy()
np.savez_compressed(a.output / "raw.npz", **arrays)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
    sample()
    torch.cuda.synchronize()
ops = {e.key: {"calls": e.count, "self_cpu_us": e.self_cpu_time_total} for e in profiler.key_averages()}
report = {
    "batch_size": a.batch_size,
    "seconds": seconds,
    "median_seconds": statistics.median(seconds),
    "operator_counts": {k: ops[k] for k in ["aten::index", "aten::nonzero", "aten::cat"] if k in ops},
    "top_cpu_operators": dict(
        sorted(ops.items(), key=lambda item: item[1]["self_cpu_us"], reverse=True)[:15]
    ),
    "precision": "float32",
    "gpu": torch.cuda.get_device_name(),
    "timing_scope": "Shared GPU; uninstrumented sampling/CPU decoding after warm-up. Operator counts measured in a separate CPU profiling pass.",
}
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
