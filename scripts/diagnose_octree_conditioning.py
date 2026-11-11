"""Held-out teacher-forcing checks of conditional information and local smoothness."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.octree_backend import load_octree_checkpoint, OctreeDataset
from orbit3d.utils.run_evidence import freeze_run
from orbit3d.utils.reproducibility import lock_deterministic_environment

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--sdf', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--limit', type=int, default=64)
p.add_argument('--device', default='cuda:0')
p.add_argument('--max-atoms', type=int, default=80)
p.add_argument('--batch-size', type=int, default=8)
a = p.parse_args()
freeze_run(a.output, __file__)
lock_deterministic_environment(42)
torch.set_num_threads(2)
model, state = load_octree_checkpoint(a.checkpoint, a.device)
allowed = set(model.args.allow_atoms.split(','))
span = model.args.grid_len * 2**model.args.merge_level - .56
excluded = []
mols = []
for source_index, m in enumerate(Chem.SDMolSupplier(a.sdf, removeHs=False)):
    if m is None or m.GetNumAtoms() > a.max_atoms or any(x.GetSymbol() not in allowed for x in m.GetAtoms()):
        excluded.append({'source_index': source_index, 'reason': 'unreadable or outside requested atom/element coverage'})
        continue
    xyz = m.GetConformer().GetPositions()
    if np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1).max() > span:
        excluded.append({'source_index': source_index, 'reason': 'outside rotation-safe spatial span'})
        continue
    identity = Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True)
    if identity in state['training_smiles']:
        raise ValueError('Held-out diagnostics must not reuse adaptation molecules')
    mols.append(m)
    if len(mols) == a.limit:
        break
if len(mols) < 8:
    raise ValueError('Insufficient compatible held-out molecules')
y = np.array([[float(m.GetProp(k)) for k in state['target_cols']] for m in mols], dtype=np.float32)
y = torch.tensor((y-state['mean'])/state['scale'], device=a.device)
dataset = OctreeDataset(mols, model, training=False)
permutation = np.random.default_rng(43).permutation(len(mols))
records = []
with torch.no_grad():
    for begin in range(0, len(mols), a.batch_size):
        ids = np.arange(begin, min(begin+a.batch_size, len(mols)))
        packed = dataset.collate(ids)
        target = y[ids]
        mask = torch.ones((*target.shape, 2), device=a.device, dtype=torch.bool)
        row = {'records': len(ids)}
        for kind, center, active in [('correct', target, mask), ('shuffled', y[permutation[ids]], mask),
                                     ('unconditional', target, torch.zeros_like(mask))]:
            _, stats = model.compute_sft_and_understanding_loss(packed, target, center-.25, center+.25, active, understanding_weight=0)
            row[kind+'_generation_loss'] = stats['loss_gen']
        base = model.interval_conditioner(target-.25, target+.25, mask)
        for delta in [1e-5, 1e-4, 1e-3, .01, .1]:
            changed = model.interval_conditioner(target-.25+delta, target+.25+delta, mask)
            row[f'condition_relative_change_{delta}'] = float(((base-changed).norm(dim=-1)/base.norm(dim=-1)).mean())
        records.append(row)
report = {'scope': 'Held-out conditional teacher-forcing diagnostic, not generation success',
          'records': len(mols), 'max_atoms': a.max_atoms, 'merge_level': model.args.merge_level, 'excluded': excluded, 'source_sdf_sha256': hashlib.sha256(Path(a.sdf).read_bytes()).hexdigest(),
          'checkpoint_sha256': hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
          'condition_version': model.condition_version, 'batch_metrics': records,
          'mean': {key: float(np.average([r[key] for r in records], weights=[r['records'] for r in records]))
                   for key in records[0] if key != 'records'}}
(a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
