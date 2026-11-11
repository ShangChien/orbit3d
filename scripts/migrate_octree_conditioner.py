"""Explicitly reset interval conditioning while retaining a trained structural prior."""

import argparse
import hashlib
from pathlib import Path
import torch
from orbit3d.generator.octree_backend import load_octree_checkpoint, build_octree, checkpoint_contract

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
old, state = load_octree_checkpoint(a.checkpoint, device='cpu')
if old.condition_version >= 3:
    raise ValueError('Input already uses smooth conditioning')
torch.manual_seed(a.seed)
new = build_octree('cpu', len(state['target_cols']), condition_version=3)
parameters = {k: v for k, v in old.state_dict().items() if not k.startswith('interval_conditioner.')}
missing = new.load_state_dict(parameters, strict=False)
assert not missing.unexpected_keys and all(k.startswith('interval_conditioner.') for k in missing.missing_keys)
state.update(model_state_dict=new.state_dict(), backend='uni3dar_octree_v3',
             contract=checkpoint_contract(new, state['target_cols'], state['target_units'], state['mean'], state['scale']),
             condition_migration={'source_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                                  'seed': a.seed, 'conditioner_reset': True,
                                  'structural_prior_and_property_head_retained': True,
                                  'requires_condition_adaptation': True})
state.pop('history', None)
state.pop('grpo_history', None)
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(state, a.output)
print(state['condition_migration'])
