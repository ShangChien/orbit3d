"""Train matched single-property SFT and GRPO arms; evaluation uses evaluate_point_campaign.py."""
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
    p.add_argument('--forcefield', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frozen-sft', type=Path)
    p.add_argument('--target', choices=['homo', 'lumo'], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--point-reward-type',choices=['inverse_point_distance','eligibility_only'],default='inverse_point_distance')
    p.add_argument('--point-advantage-mode',choices=['joint','conditional_property_v1'],default='joint')
    p.add_argument('--stage', choices=['sft', 'arms', 'grpo'], required=True)
    a = p.parse_args()
    if not (a.output / 'run_manifest.json').exists():
        freeze_run(a.output, __file__)
        record_input_hashes(a.output, {'data_manifest': a.data / 'manifest.json', 'prior': a.prior,
                                      'frozen_sft': a.frozen_sft})
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

    def launch(name, mode, checkpoint, settings=None, inference=None, resume=None):
        cfg = copy.deepcopy(base)
        cfg['mode'] = mode
        cfg['mount_paths'].update(checkpoint_load_path=str(checkpoint),
                                 checkpoint_save_dir=str(a.output/name/'weights'),
                                 output_dir=str(a.output/name/'run'))
        if resume is not None:
            cfg['mount_paths']['alignment_resume_path']=str(resume)
        cfg['generator_config'].update(settings or {})
        cfg['inference_config'].update(inference or {})
        path = configs/f'{name}.json'
        path.write_text(json.dumps(cfg, indent=2)+'\n')
        execute(name, [sys.executable, '-u', '-m', 'orbit3d.cli', '--config', str(path)])

    frozen = a.frozen_sft or a.output/'sft/weights/octree_conditional.pt'
    if a.stage == 'sft':
        launch('sft', 'train', a.prior, {'sft_steps': 5000, 'sft_lr': 2e-4,
               'backbone_lr': 1e-5, 'train_backbone': True, 'checkpoint_every': 2500})
    if a.stage == 'arms':
        launch('continue', 'train', frozen, {'sft_steps': 400, 'sft_lr': 2e-5,
               'train_backbone': False, 'train_property_head': False, 'lr_schedule': 'constant',
               'checkpoint_every': 100})
    if a.stage in ['arms', 'grpo']:
        alignment = {'point_reward_type':a.point_reward_type,'point_advantage_mode':a.point_advantage_mode,
               'grpo_steps': 200, 'group_size': 16, 'grpo_lr': 2e-5,
               'kl_beta': .02, 'physics_weight': 0., 'uncertainty_weight': 0., 'replay_weight': 0.,
               'alignment_checkpoints': [50, 100],
               'point_target_pool': [{a.target: points[q]} for q in ['25', '50', '75']]}
        launch('grpo_part100','align',frozen,{**alignment,'alignment_stop_after':100})
        launch('grpo','align',frozen,alignment,
               resume=a.output/'grpo_part100/weights/octree_resume_100.pt')


if __name__ == '__main__':
    main()
