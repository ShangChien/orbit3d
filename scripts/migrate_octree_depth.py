"""Grow tree depth at fixed 0.24 A cells / 0.01 A bins; adaptation is mandatory."""

import argparse
import hashlib
from pathlib import Path
import torch
from loom.generator.octree_backend import load_octree_checkpoint, build_octree, checkpoint_contract, required_condition_elements

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--merge-level", type=int, choices=[7, 8], default=7)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
old, state = load_octree_checkpoint(a.checkpoint, device="cpu")
if a.merge_level <= old.args.merge_level:
    raise ValueError("Depth migration must add at least one level")
config = {"merge_level": a.merge_level, "allowed_elements": old.args.allow_atoms.split(",")}
new = build_octree(
    "cpu", len(state["target_cols"]), condition_version=old.condition_version, representation_config=config
)
weights = old.state_dict()
levels = new.cur_level.weight.detach().clone()
levels[: len(old.cur_level.weight)] = old.cur_level.weight.detach()
levels[len(old.cur_level.weight) :] = old.cur_level.weight[-1].detach()
weights["cur_level.weight"] = levels
new.load_state_dict(weights, strict=True)
pending = required_condition_elements(state)
if pending:
    state['condition_migration'] = {**state.get('condition_migration', {}),
        'requires_condition_adaptation': True, 'required_elements': sorted(pending)}
state.update(
    model_state_dict=new.state_dict(),
    representation_config=config,
    contract=checkpoint_contract(
        new, state["target_cols"], state["target_units"], state["mean"], state["scale"]
    ),
    representation_migration={
        "source_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "old_merge_level": old.args.merge_level,
        "new_merge_level": a.merge_level,
        "grid_len_A": old.args.grid_len,
        "xyz_resolution_A": old.args.xyz_resolution,
        "initialization": "Existing level rows unchanged; new rows copy the former root row",
        "requires_adaptation": True,
    },
)
state.pop("history", None)
state.pop("grpo_history", None)
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(state, a.output)
print(state["representation_migration"])
