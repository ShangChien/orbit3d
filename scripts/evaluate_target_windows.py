"""Evaluate a saved policy on predeclared intervals, preserving every proposal."""
import argparse
import copy
import gc
import json
import subprocess
import sys
from pathlib import Path
import torch
from orbit3d.pipeline.octree_runner import run_octree_from_config
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--windows', type=Path, required=True, help='JSON mapping names to target interval dictionaries')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
p.add_argument('--independent-oracle', type=Path)
p.add_argument('--reference-first-run', type=Path, help='Verify the first window/seed against an existing post-training run')
a = p.parse_args()
freeze_run(a.output, __file__)
record_input_hashes(a.output, {'configuration': a.config, 'checkpoint': a.checkpoint,
                              'target_windows': a.windows, 'independent_oracle': a.independent_oracle})
base = json.loads(a.config.read_text())
windows = json.loads(a.windows.read_text())
base['mode'] = 'predict'
base['generator_config']['grpo_steps'] = 0
base['mount_paths'].update(checkpoint_load_path=str(a.checkpoint))
base['mount_paths'].pop('train_data_path', None)
records = []
for name, bounds in windows.items():
    for seed in a.seeds:
        cfg = copy.deepcopy(base)
        run = a.output / f'{name}_seed{seed}'
        cfg['inference_config'].update(target_interval=bounds, base_seed=seed)
        cfg['mount_paths']['output_dir'] = str(run)
        summary = run_octree_from_config(cfg)
        if not records and a.reference_first_run:
            from orbit3d.utils.replay_validation import compare_run_outputs
            replay = compare_run_outputs(a.reference_first_run, run)
            (a.output / 'reload_summary.json').write_text(json.dumps(replay, indent=2) + '\n')
        subprocess.run([sys.executable, str(Path(__file__).with_name('audit_generation_failures.py')),
                        '--run', str(run), '--output', str(run / 'raw_audit')], check=True)
        row = {'window': name, 'inference_seed': seed, 'intervals': bounds, **summary,
               'raw_audit': json.loads((run / 'raw_audit/summary.json').read_text())}
        if a.independent_oracle:
            subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_generated.py')),
                            '--oracle', str(a.independent_oracle), '--runs', str(run)], check=True)
            row['independent'] = json.loads((run / 'independent_evaluation.json').read_text())
        records.append(row)
        (a.output / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
