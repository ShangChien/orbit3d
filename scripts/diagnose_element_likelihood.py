"""Teacher-forced element probabilities at explicit sampling temperatures."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.octree_backend import load_octree_checkpoint, OctreeDataset
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--sdf', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--limit', type=int, default=512)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
freeze_run(a.output, __file__)
record_input_hashes(a.output, {'model': a.checkpoint, 'held_out_structures': a.sdf})
lock_deterministic_environment(42)
torch.set_num_threads(2)
model, state = load_octree_checkpoint(a.checkpoint, a.device)
molecules = list(Chem.SDMolSupplier(str(a.sdf), removeHs=False))[:a.limit]
assert molecules and all(m is not None for m in molecules)
training = set(state['training_smiles'])
assert all(Chem.MolToSmiles(Chem.RemoveHs(m)) not in training for m in molecules)
dataset = OctreeDataset(molecules, model, training=False)
labels = np.array([[float(m.GetProp(n)) for n in state['target_cols']] for m in molecules], dtype=np.float32)
labels = torch.tensor((labels-state['mean'])/state['scale'], device=a.device)
observations = defaultdict(list)
with torch.no_grad():
    for start in range(0, len(molecules), a.batch_size):
        indices = np.arange(start, min(start+a.batch_size, len(molecules)))
        batch = {key: value.to(a.device) if isinstance(value, torch.Tensor) else value
                 for key, value in dataset.collate(indices).items()}
        batch['decoder_phy_pos'] = batch['decoder_phy_pos'].float()
        decoder, target, _ = model.get_feat_from_dataloader(batch)
        y = labels[indices]
        hidden, _ = model.forward_conditional_sequence(decoder, batch['decoder_input_len'].int(),
            int(batch['decoder_input_max_len']), y-.25, y+.25,
            torch.ones((*y.shape, 2), dtype=torch.bool, device=a.device), compute_understanding=False)
        selected = target.is_gen_atom
        truth = [value.long() for value in target.to_list(selected)]
        logits = model.atom_heads[0](hidden[selected], truth)[0].float()
        logits = logits.masked_fill(~model.allowed_atom_mask(a.device), -torch.inf)
        correct = logits.argmax(-1) == truth[0]
        probability = {temperature: (logits/temperature).softmax(-1).gather(1, truth[0][:, None]).squeeze(1)
                       for temperature in [.3, .6, 1.]}
        for i, kind in enumerate(truth[0].tolist()):
            observations[kind].append({'top1_correct': bool(correct[i]),
                                      **{f'p_true_T{t}': float(values[i]) for t, values in probability.items()}})
        print('teacher-forced structures', start+len(indices), flush=True)
# Atom-head labels use the model's atom output categories, including the null token.
report = {'scope': 'Held-out teacher-forced element probabilities, not free-generation frequency',
          'molecules': len(molecules), 'element_head_categories': {},
          'dictionary_atom_types': model.dictionary.atom_type}
category_names = {value: name for name, value in model.dictionary.atom_type.items()}
for category, rows in observations.items():
    report['element_head_categories'][str(category)] = {
        'element_or_special_token': category_names[category],
        'decisions': len(rows), 'top1_accuracy': float(np.mean([row['top1_correct'] for row in rows])),
        **{key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'top1_correct'}}
(a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report['element_head_categories'], indent=2))
