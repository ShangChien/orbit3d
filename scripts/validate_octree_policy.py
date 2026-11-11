"""Compare cached rollout probabilities with full-prefix GRPO probabilities."""

import argparse
import json
from pathlib import Path
import torch
from orbit3d.constraints import TargetIntervals
from orbit3d.generator.octree_backend import load_octree_checkpoint, interval_arrays
from orbit3d.vendor.uni3dar.models import uni3dar_sampler as sampler

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
cfg = json.loads(a.config.read_text())
torch.set_num_threads(4)
torch.manual_seed(42)
device = cfg.get("device", "cuda:0")
model, state = load_octree_checkpoint(cfg["mount_paths"]["checkpoint_load_path"], device)
model.args.rank_by = "joint"
intervals = TargetIntervals.parse(cfg["inference_config"]["target_interval"], state["target_cols"])
lower, upper, active = interval_arrays(intervals, state["mean"], state["scale"], model.condition_version)
lo = torch.tensor(lower, device=device).repeat(4, 1)
hi = torch.tensor(upper, device=device).repeat(4, 1)
mask = torch.tensor(active, device=device).unsqueeze(0).expand(4, *active.shape)
original = sampler.softmax_sampling


def sampling_with_behavior_score(
    logits, top_p=1.0, temperature=1.0, top_k=None, prob_proc_func=None, uniforms=None
):
    assert top_p == 1.0 and top_k is None
    token, _ = original(logits, top_p, temperature, top_k, prob_proc_func, uniforms)
    probs = (logits.float() / temperature).softmax(-1)
    if prob_proc_func is not None:
        probs = prob_proc_func(probs)
    probs = probs / probs.sum(-1, keepdim=True)
    return token, probs.gather(1, token.long()[:, None]).squeeze(1).log()


sampler.softmax_sampling = sampling_with_behavior_score
try:
    with torch.no_grad():
        trajectories = model.generate_conditional_trajectories(lo, hi, mask, evaluate_self_consistency=False)
        packets = model.trajectory_distributions(trajectories, lo, hi, mask)
        teacher = torch.zeros(4, device=device)
        count = torch.zeros_like(teacher)
        for logp, labels, owner in packets:
            teacher.index_add_(0, owner, logp.gather(1, labels[:, None]).squeeze(1))
            count.index_add_(0, owner, torch.ones_like(owner, dtype=count.dtype))
        rollout = torch.stack([t["scores"].sum() for t in trajectories])
        error = (teacher - rollout).abs()
        report = {
            "trajectories": 4,
            "decision_counts": count.tolist(),
            "rollout_log_probability": rollout.tolist(),
            "teacher_log_probability": teacher.tolist(),
            "max_mean_token_logprob_error": float((error / count).max()),
            "precision": "float32",
        }
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        assert float((error / count).max()) < 1e-4, "Rollout and GRPO distributions differ"
finally:
    sampler.softmax_sampling = original
