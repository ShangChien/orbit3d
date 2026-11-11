"""Replicate the predeclared 5k-to-20k SFT continuation and fixed evaluations."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def main():
    parser = argparse.ArgumentParser()
    for name in ['config', 'checkpoint', 'validation-sdf', 'windows', 'independent-oracle', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    freeze_run(args.output, __file__)
    cfg = json.loads(args.config.read_text())
    cfg.update(mode='train', seed=args.seed)
    cfg['mount_paths'].update(checkpoint_load_path=str(args.checkpoint.resolve()),
                             checkpoint_save_dir=str(args.output.resolve() / 'checkpoints'),
                             output_dir=str(args.output.resolve() / 'training'))
    cfg['generator_config'].update(sft_steps=15000, checkpoint_every=5000,
                                   train_backbone=True, backbone_lr=1e-5, sft_lr=2e-4,
                                   batch_size=8, grpo_steps=0, element_loss_weighting='none')
    config = args.output.resolve() / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    record_input_hashes(args.output, {
        'initial_checkpoint': args.checkpoint, 'baseline_configuration': args.config,
        'training_sdf': cfg['mount_paths']['train_data_path'], 'validation_sdf': args.validation_sdf,
        'target_windows': args.windows, 'independent_model': args.independent_oracle,
    })
    scripts = Path(__file__).resolve().parent

    def run(arguments, name):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), name, flush=True)
        with (args.output / f'{name}.log').open('w') as stream:
            subprocess.run([sys.executable, *map(str, arguments)], stdout=stream, stderr=subprocess.STDOUT, check=True)

    run(['-m', 'orbit3d.cli', '--config', config], 'train')
    for label, filename in [('total10000', 'octree_step_5000.pt'), ('total20000', 'octree_conditional.pt')]:
        checkpoint = args.output / 'checkpoints' / filename
        run([scripts / 'diagnose_element_likelihood.py', '--checkpoint', checkpoint,
             '--sdf', args.validation_sdf, '--output', args.output / f'{label}_validation'], f'{label}_validation')
        run([scripts / 'evaluate_target_windows.py', '--config', config, '--checkpoint', checkpoint,
             '--windows', args.windows, '--output', args.output / f'{label}_evaluation',
             '--independent-oracle', args.independent_oracle], f'{label}_evaluation')
    print('SFT continuation and both milestone evaluations complete', flush=True)


if __name__ == '__main__':
    main()
