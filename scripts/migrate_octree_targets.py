"""Explicitly add labelled property targets; adaptation is required before inference."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from loom.generator.octree_backend import load_octree_checkpoint, build_octree, checkpoint_contract, required_condition_elements
from loom.generator.target_migration import expand_interval_conditioner, expand_property_readout

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--training-sdf', type=Path, required=True)
p.add_argument('--targets', nargs='+', required=True)
p.add_argument('--units', nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
old, state = load_octree_checkpoint(a.checkpoint, 'cpu')
old_names = state['target_cols']
if len(set(a.targets)) != len(a.targets) or not set(old_names) < set(a.targets) or len(a.units) != len(a.targets):
    raise ValueError('Target expansion must add targets and retain existing unique names with matching units')
for index, name in enumerate(old_names):
    if a.units[a.targets.index(name)] != state['target_units'][index]:
        raise ValueError('Existing target units cannot change during expansion')
molecules = list(Chem.SDMolSupplier(str(a.training_sdf), removeHs=False))
assert molecules and all(m is not None for m in molecules)
labels = np.array([[float(m.GetProp(name)) for name in a.targets] for m in molecules], dtype=np.float32)
if not np.isfinite(labels).all():
    raise ValueError('New target normalization requires finite training labels')
mean, scale = labels.mean(0), labels.std(0).clip(1e-6)
for before, name in enumerate(old_names):
    after = a.targets.index(name)
    mean[after], scale[after] = state['mean'][before], state['scale'][before]
torch.manual_seed(a.seed)
new = build_octree('cpu', len(a.targets), condition_version=old.condition_version,
                  representation_config=state.get('representation_config'))
backbone = {key: value for key, value in old.state_dict().items()
            if not key.startswith(('interval_conditioner.', 'prop_understanding_head.'))}
missing = new.load_state_dict(backbone, strict=False)
assert not missing.unexpected_keys and all(key.startswith(('interval_conditioner.', 'prop_understanding_head.')) for key in missing.missing_keys)
expand_interval_conditioner(old.interval_conditioner, new.interval_conditioner, old_names, a.targets)
expand_property_readout(old.prop_understanding_head, new.prop_understanding_head, old_names, a.targets)
required = required_condition_elements(state) | set(old.args.allow_atoms.split(','))
state.update(model_state_dict=new.state_dict(), target_cols=a.targets, target_units=a.units, mean=mean, scale=scale,
             contract=checkpoint_contract(new, a.targets, a.units, mean, scale),
             condition_migration={'requires_condition_adaptation': True, 'required_elements': sorted(required),
                                  'required_targets': sorted(set(a.targets)-set(old_names))},
             target_migration={'source_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                               'training_sdf_sha256': hashlib.sha256(a.training_sdf.read_bytes()).hexdigest(),
                               'old_targets': old_names, 'new_targets': a.targets, 'seed': a.seed,
                               'normalization': 'Existing target scales retained; new scales fitted only on the supplied training labels',
                               'initialization': 'Old conditioner blocks/readout rows retained; new conditioner input columns and readout rows zero-initialized',
                               'new_target_predictions_trained': False})
state.pop('history', None);state.pop('grpo_history', None)
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(state, a.output)
print(state['target_migration'])
