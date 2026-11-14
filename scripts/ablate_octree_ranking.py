"""Matched candidate streams for first-valid, self and external property ranking."""

import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from loom.pipeline.octree_runner import run_octree_from_config

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
a = p.parse_args()
base = json.loads(a.config.read_text())
base['mode'] = 'predict'
base['generator_config']['grpo_steps'] = 0
base['mount_paths']['checkpoint_load_path'] = a.checkpoint
base['mount_paths'].pop('generator_pretrained_path', None)
base['mount_paths'].pop('train_data_path', None)
base['inference_config'].update(sampling_mode='independent', precision='float32')
results = []
for seed in a.seeds:
    token_reference = None
    for ranking in ['first_valid', 'self', 'external']:
        c = copy.deepcopy(base)
        output = a.output / f'{seed}_{ranking}'
        c['mount_paths']['output_dir'] = str(output)
        c['inference_config'].update(base_seed=seed, ranking=ranking)
        summary = run_octree_from_config(c)
        tokens = pd.read_csv(output / 'candidates.csv').token_sha256.tolist()
        if token_reference is None:
            token_reference = tokens
        assert tokens == token_reference, 'Ablation changed the proposal pool'
        results.append({'seed': seed, **summary})
        (a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
aggregate = {}
for ranking in ['first_valid', 'self', 'external']:
    values = np.array([r['joint_success_rate'] for r in results if r['ranking'] == ranking])
    aggregate[ranking] = {'seed_count': len(values), 'mean_joint_success_rate': float(values.mean()),
                          'seed_std': float(values.std(ddof=1)) if len(values) > 1 else None,
                          'minimum': float(values.min()), 'maximum': float(values.max())}
(a.output / 'aggregate.json').write_text(json.dumps(aggregate, indent=2) + '\n')
