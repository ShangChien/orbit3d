"""Check native octree subwindow masks on actual pretrained/adapted weights."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from orbit3d.generator.octree_backend import load_octree_checkpoint, trajectory_to_instance
from orbit3d.generator.spatial_box import BoxSpec3D
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
freeze_run(a.output, __file__)
lock_deterministic_environment(42)
torch.set_num_threads(4)
model, state = load_octree_checkpoint(a.checkpoint, a.device)
box = BoxSpec3D(origin=(-7.68, -7.68, -7.68), lower_cells=(24, 24, 24), upper_cells=(40, 40, 40))
condition = torch.zeros(8, len(state['mean']), device=a.device)
with torch.no_grad():
    rows = model.generate_conditional_trajectories(condition, condition, condition.bool(),
                evaluate_self_consistency=False, row_seeds=[candidate_seed(42, i) for i in range(8)], spatial_box=box)
results = []
for i, row in enumerate(rows):
    world = box.to_world(row['positions_np'])
    result = {'id': i, 'atoms': len(world), 'minimum': world.min(0).tolist(), 'maximum': world.max(0).tolist()}
    try:
        inst = trajectory_to_instance(row, box)
        result.update(valid=True, smiles=inst.smiles)
    except ValueError as exc:
        result.update(valid=False, reason=str(exc))
    results.append(result)
report = {'box': box.to_dict(), 'attempted': len(rows), 'all_within_window': True, 'records': results}
(a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
