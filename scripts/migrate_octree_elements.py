"""Declare an expanded element support set; retain token IDs and require training."""

import argparse
import hashlib
from pathlib import Path
import torch
from orbit3d.generator.octree_backend import load_octree_checkpoint, build_octree, checkpoint_contract, required_condition_elements

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--elements", nargs="+", required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
old, state = load_octree_checkpoint(a.checkpoint, device="cpu")
previous = old.args.allow_atoms.split(",")
if not set(previous) < set(a.elements):
    raise ValueError("Expansion must retain all old elements and add new elements")
config = {"merge_level": old.args.merge_level, "allowed_elements": a.elements}
new = build_octree(
    "cpu", len(state["target_cols"]), condition_version=old.condition_version, representation_config=config
)
assert old.dictionary.atom_type == new.dictionary.atom_type
new.load_state_dict(old.state_dict(), strict=True)
pending = required_condition_elements(state) | (set(a.elements) - set(previous))
state.update(
    model_state_dict=new.state_dict(),
    representation_config=config,
    contract=checkpoint_contract(
        new, state["target_cols"], state["target_units"], state["mean"], state["scale"]
    ),
    condition_migration={'requires_condition_adaptation': True, 'required_elements': sorted(pending)},
    representation_migration={
        "source_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "old_elements": previous,
        "new_elements": a.elements,
        "token_ids_unchanged": True,
        "initialization": "Existing periodic-table embedding rows retained; newly enabled categories require adaptation",
        "requires_adaptation": True,
    },
)
state.pop("history", None)
state.pop("grpo_history", None)
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(state, a.output)
print(state["representation_migration"])
