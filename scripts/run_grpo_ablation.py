"""Matched 50-step GRPO controls: full reward, no KL, no physics, no replay."""

import argparse
import copy
import gc
import json
from pathlib import Path
import torch
from orbit3d.pipeline.octree_runner import run_octree_from_config
from orbit3d.utils.run_evidence import freeze_run

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
freeze_run(a.output, __file__)
base = json.loads(a.config.read_text())
base["mode"] = "predict"
base["generator_config"].update(
    grpo_steps=50,
    group_size=16,
    grpo_lr=2e-5,
    kl_beta=0.02,
    physics_weight=0.2,
    uncertainty_weight=0.0,
    replay_weight=0.1,
)
base["inference_config"].update(
    electron_parity=0,
    base_seed=2026,
    num_candidates=64,
    num_molecules=16,
    oracle_candidates=32,
    precision="float32",
    sampling_mode="independent",
)
base["stage5_relax_config"].update(batch_refinement=True, enabled=True, relax_steps=30)
models = {}
reports = {}
for name, setting in [
    ("full", {}),
    ("no_kl", {"kl_beta": 0.0}),
    ("no_physics", {"physics_weight": 0.0}),
    ("no_replay", {"replay_weight": 0.0}),
]:
    cfg = copy.deepcopy(base)
    cfg["generator_config"].update(setting)
    cfg["mount_paths"].update(
        output_dir=str(a.output / name / "training"), checkpoint_save_dir=str(a.output / name / "checkpoints")
    )
    summary = run_octree_from_config(cfg)
    models[name] = summary["checkpoint"]
    reports[name] = summary
    (a.output / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    (a.output / "training_summaries.json").write_text(json.dumps(reports, indent=2) + "\n")
    gc.collect()
    torch.cuda.empty_cache()
