"""Matched conditioner-only continuation with optional incompatible-interval pairs.

Experimental objective, not a new default. The structural backbone and property
head remain frozen; both treatment and zero-weight control use identical data,
positive intervals, batches and optimizer settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.octree_backend import load_octree_checkpoint, OctreeDataset
from orbit3d.generator.condition_contrast import incompatible_condition_indices, condition_margin_loss
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--sdf', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--weight', type=float, required=True)
p.add_argument('--margin', type=float, default=.02)
p.add_argument('--halfwidth', type=float, default=.25)
p.add_argument('--steps', type=int, default=600)
p.add_argument('--batch-size', type=int, default=8)
p.add_argument('--lr', type=float, default=5e-5)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
if a.steps <= 0 or a.weight < 0 or a.margin <= 0 or a.halfwidth <= 0:
    p.error('Require positive steps, margin, halfwidth and nonnegative contrast weight')
freeze_run(a.output, __file__)
record_input_hashes(a.output, {'initial_checkpoint': a.checkpoint, 'labeled_structures': a.sdf})
config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
(a.output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
lock_deterministic_environment(a.seed)
torch.set_num_threads(2)
model, state = load_octree_checkpoint(a.checkpoint, a.device, a.batch_size)
if state.get('representation_migration', {}).get('requires_adaptation') or state.get('condition_migration', {}).get('requires_condition_adaptation'):
    raise ValueError('This experiment requires completed structural and property adaptation')
molecules = list(Chem.SDMolSupplier(str(a.sdf), removeHs=False))
assert molecules and all(m is not None for m in molecules)
names, mean, scale = (state[key] for key in ['target_cols', 'mean', 'scale'])
labels = np.array([[float(m.GetProp(n)) for n in names] for m in molecules], dtype=np.float32)
assert np.isfinite(labels).all()
labels = (labels-mean)/scale
dataset = OctreeDataset(molecules, model, seed=a.seed)
model.requires_grad_(False)
model.interval_conditioner.requires_grad_(True)
parameters = list(model.interval_conditioner.parameters())
optimizer = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
rng = np.random.default_rng(a.seed)
history = []
for step in range(a.steps):
    model.train()
    dataset.grid.set_epoch(step+1)
    ids = rng.integers(0, len(molecules), size=a.batch_size)
    packed = dataset.collate(ids)
    target = torch.as_tensor(labels[ids], device=a.device)
    mask = torch.ones((*target.shape, 2), device=a.device, dtype=torch.bool)
    negative_ids, eligible = incompatible_condition_indices(target, a.halfwidth)
    with torch.autocast(device_type=a.device.split(':')[0], dtype=torch.bfloat16, enabled=a.device.startswith('cuda')):
        loss, stats = model.compute_sft_and_understanding_loss(
            packed, target, target-a.halfwidth, target+a.halfwidth, mask,
            understanding_weight=0, return_per_molecule=bool(a.weight))
        contrast = torch.zeros((), device=a.device)
        separation = None
        if a.weight and eligible.any():
            wrong = target[negative_ids]
            _, wrong_stats = model.compute_sft_and_understanding_loss(
                packed, target, wrong-a.halfwidth, wrong+a.halfwidth, mask,
                understanding_weight=0, return_per_molecule=True)
            positive = stats.pop('per_molecule_generation_loss')
            negative = wrong_stats['per_molecule_generation_loss']
            contrast = condition_margin_loss(positive, negative, eligible, a.margin)
            separation = float((negative[eligible]-positive[eligible]).mean().detach())
            loss = loss + a.weight*contrast
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite training objective at step {step+1}')
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    row = {'step': step+1, 'positive_generation_loss': stats['loss_gen'],
           'total_loss': float(loss.detach()), 'contrast_loss': float(contrast.detach()),
           'eligible_pairs': int(eligible.sum()), 'mean_negative_minus_positive': separation,
           'gradient_norm': float(norm), 'source_indices': ids.tolist()}
    history.append(row)
    if step % 10 == 0 or step == a.steps-1:
        print({k: v for k, v in row.items() if k != 'source_indices'}, flush=True)
state.update(model_state_dict=model.state_dict(), history=history, initial_weights_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
             training_objective={'type': 'conditioner_only_interval_margin_experiment', **config},
             optimizer_resumed=False, train_backbone=False)
state['training_smiles'] = sorted(set(state.get('training_smiles', [])) | {Chem.MolToSmiles(Chem.RemoveHs(m)) for m in molecules})
torch.save(state, a.output/'octree_conditional.pt')
(a.output/'history.json').write_text(json.dumps(history, indent=2)+'\n')
