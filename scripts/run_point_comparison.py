"""Matched single-property SFT/continued-SFT/GRPO pilot on frozen source data."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--dpa', type=Path, required=True)
    p.add_argument('--fingerprint', type=Path, required=True)
    p.add_argument('--forcefield', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target', choices=['homo', 'lumo'], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stage', choices=['sft', 'arms', 'evaluate', 'time-control', 'all'], default='all')
    a = p.parse_args()
    if not (a.output / 'run_manifest.json').exists():
        freeze_run(a.output, __file__)
        record_input_hashes(a.output, {'data_manifest': a.data / 'manifest.json', 'prior': a.prior})
    points = json.loads((a.data / 'manifest.json').read_text())['single_target_points'][a.target]
    configs = a.output / 'configs'
    configs.mkdir(exist_ok=True)
    costs_path = a.output / 'costs.json'
    costs = json.loads(costs_path.read_text()) if costs_path.exists() else []
    scripts = Path(__file__).resolve().parent

    def execute(name, command):
        # Fresh child output directories enforce that an existing run is not overwritten.
        start = time.perf_counter()
        with (a.output / f'{name}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        costs.append({'name': name, 'command': command, 'wall_seconds': time.perf_counter()-start,
                      'returncode': result.returncode, 'device': a.device})
        costs_path.write_text(json.dumps(costs, indent=2)+'\n')
        result.check_returncode()
        print('completed', name, costs[-1]['wall_seconds'], flush=True)

    base = {
        'device': a.device, 'seed': a.seed, 'threads': 4,
        'mount_paths': {'train_data_path': str(a.data/'train.sdf'),
                        'forcefield_path': str(a.forcefield),
                        'property_checkpoint_dir': str(a.dpa/'oracle')},
        'generator_config': {'backend': 'uni3dar_octree', 'condition_version': 4,
                             'active_targets': [a.target], 'label_source': 'sdf',
                             'batch_size': 8, 'grpo_steps': 0},
        'oracle_config': {'oracle_mode': 'internal_dpa4', 'target_cols': ['homo', 'lumo'],
                          'target_units': ['eV', 'eV'],
                          'applicability_domain_path': str(a.dpa/'applicability.json')},
        'inference_config': {'target_point': {a.target: points['50']},
                             'target_tolerance': {a.target: .25}, 'num_candidates': 64,
                             'num_molecules': 16, 'oracle_candidates': 64, 'batch_size': 8,
                             'sampling_mode': 'independent', 'precision': 'float32',
                             'tree_temperature': .85, 'atom_temperature': .3, 'xyz_temperature': .25,
                             'cfg_scale': 1., 'electron_parity': 0, 'max_rms_force_eva': .1},
        'stage5_relax_config': {'enabled': False, 'batch_refinement': True},
    }

    def launch(name, mode, checkpoint, settings=None, inference=None):
        cfg = copy.deepcopy(base)
        cfg['mode'] = mode
        cfg['mount_paths'].update(checkpoint_load_path=str(checkpoint),
                                 checkpoint_save_dir=str(a.output/name/'weights'),
                                 output_dir=str(a.output/name/'run'))
        cfg['generator_config'].update(settings or {})
        cfg['inference_config'].update(inference or {})
        path = configs/f'{name}.json'
        path.write_text(json.dumps(cfg, indent=2)+'\n')
        execute(name, [sys.executable, '-u', '-m', 'orbit3d.cli', '--config', str(path)])

    frozen = a.output/'sft/weights/octree_conditional.pt'
    if a.stage in ['sft', 'all']:
        launch('sft', 'train', a.prior, {'sft_steps': 5000, 'sft_lr': 2e-4,
               'backbone_lr': 1e-5, 'train_backbone': True, 'checkpoint_every': 2500})
    if a.stage in ['arms', 'all']:
        launch('continue', 'train', frozen, {'sft_steps': 400, 'sft_lr': 2e-5,
               'train_backbone': False, 'train_property_head': False, 'lr_schedule': 'constant',
               'checkpoint_every': 100})
        launch('grpo', 'align', frozen, {'grpo_steps': 200, 'group_size': 16, 'grpo_lr': 2e-5,
               'kl_beta': .02, 'physics_weight': 0., 'uncertainty_weight': 0., 'replay_weight': 0.,
               'alignment_checkpoints': [50, 100],
               'point_target_pool': [{a.target: points[q]} for q in ['25', '50', '75']]})
    if a.stage == 'time-control':
        budget=next(row['wall_seconds'] for row in costs if row['name']=='grpo' and row['returncode']==0)
        launch('continue_wall', 'train', frozen, {'sft_steps': 1000000, 'sft_wall_seconds': budget,
               'sft_lr': 2e-5, 'train_backbone': False, 'train_property_head': False,
               'lr_schedule': 'constant'})
    if a.stage in ['evaluate', 'time-control', 'all']:
        arms = {'frozen': frozen, 'continue400': a.output/'continue/weights/octree_conditional.pt',
                'grpo200': a.output/'grpo/weights/octree_aligned.pt',
                'continue100': a.output/'continue/weights/octree_step_100.pt',
                'continue200': a.output/'continue/weights/octree_step_200.pt',
                'grpo50': a.output/'grpo/weights/octree_group_50.pt',
                'grpo100': a.output/'grpo/weights/octree_group_100.pt'}
        if a.stage == 'time-control':
            arms={'continue_wall':a.output/'continue_wall/weights/octree_conditional.pt'}
        for arm, checkpoint in arms.items():
            quantiles = ['25','50','75'] if arm in ['continue100','continue200','grpo50','grpo100'] else ['10','25','50','75','90']
            for q in quantiles:
                for seed in [2026, 2027, 2028]:
                    name = f'eval_{arm}_p{q}_s{seed}'
                    launch(name, 'predict', checkpoint,
                           inference={'target_point': {a.target: points[q]}, 'base_seed': seed})
                    execute(name+'_raw', [sys.executable, '-u', str(scripts/'evaluate_point_pool.py'),
                            '--run', str(a.output/name/'run'), '--output', str(a.output/name/'raw'),
                            '--independent', str(a.fingerprint/'oracle'), '--device', a.device])
        # Close the condition on the same frozen policy, with the original task retained.
        for seed in ([2026, 2027, 2028] if a.stage != 'time-control' else []):
            name = f'eval_frozen_off_s{seed}'
            launch(name, 'predict', frozen, inference={'condition_control': 'off', 'base_seed': seed})
            execute(name+'_raw', [sys.executable, '-u', str(scripts/'evaluate_point_pool.py'),
                    '--run', str(a.output/name/'run'), '--output', str(a.output/name/'raw'),
                    '--independent', str(a.fingerprint/'oracle'), '--device', a.device])


if __name__ == '__main__':
    main()
