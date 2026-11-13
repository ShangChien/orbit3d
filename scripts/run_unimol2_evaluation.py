"""After both size comparisons finish, freeze selection and evaluate unchanged pools."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--training-pids', type=int, nargs=2, required=True,
                   help='310M and 1.1B training PIDs, respectively')
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2', '3']
    names = ['unimol2_310m', 'unimol2_11b']
    while True:
        pending = []
        for name, pid in zip(names, args.training_pids):
            if (args.root/'training'/name/'completed.json').exists():
                continue
            os.kill(pid, 0)  # A failed trainer stops the queue instead of selecting a partial run.
            pending.append(name)
        if not pending:
            break
        print('WAITING', ', '.join(pending), flush=True)
        time.sleep(30)
    scripts = Path(__file__).parent
    def run(name, *extra):
        print('RUNNING', name, flush=True)
        subprocess.run([sys.executable, '-u', str(scripts/name),
                        '--root', str(args.root), *map(str, extra)], check=True)
    run('select_unimol2_development.py', '--output', args.root/'selection_unimol2')
    run('profile_dual_oracle.py', '--output', args.root/'profile_unimol2',
        '--selection', args.root/'selection_unimol2/selection.json')
    run('reevaluate_dual_oracle.py', '--output', args.root/'reevaluation_unimol2',
        '--profile', 'profile_unimol2')
    run('evaluate_optimized_quantum_cohort.py', '--profile', 'profile_unimol2',
        '--output', args.root/'quantum_cohort/optimized_model_predictions_unimol2.json')
    print('UNIMOL2 EVALUATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
