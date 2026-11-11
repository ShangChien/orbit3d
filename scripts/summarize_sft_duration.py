"""Summarize complete fixed SFT milestone matrices, retaining continuation seeds."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--expected-seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--initial-control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources, models, rows, validation = {}, [], [], []

    def read(path, label):
        sources[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    for root in args.runs:
        cfg = read(root / 'config.json', root.name + '/config.json')
        seed = cfg['seed']
        inputs = read(root / 'input_manifest.json', f'seed{seed}/input_manifest.json')
        models.append({'training_seed': seed, 'initial_checkpoint': inputs['initial_checkpoint'],
                       'training_sdf': inputs['training_sdf']})
        for steps in [10000, 20000]:
            label = f'seed{seed}/total{steps}'
            results = read(root / f'total{steps}_evaluation/runs.json', label + '/runs.json')
            expected = set(itertools.product(['original', 'central'], [2026, 2027, 2028]))
            keys = [(r['window'], r['inference_seed']) for r in results]
            if len(keys) != 6 or set(keys) != expected:
                raise ValueError(f'Incomplete or duplicated inference matrix: {label}')
            if any(r['attempted'] != 64 or r['independent']['attempted'] != 64 for r in results):
                raise ValueError(f'Changed attempt budget: {label}')
            for row in results:
                row.pop('checkpoint', None)
                rows.append({'training_seed': seed, 'total_steps': steps, **row})
            diagnostic = read(root / f'total{steps}_validation/summary.json', label + '/validation.json')
            diagnostic_inputs = read(root / f'total{steps}_validation/input_manifest.json', label + '/validation_inputs.json')
            validation.append({'training_seed': seed, 'total_steps': steps, 'result': diagnostic,
                               'inputs': diagnostic_inputs})
    seeds = [r['training_seed'] for r in models]
    if len(seeds) != len(set(seeds)) or sorted(seeds) != sorted(args.expected_seeds):
        raise ValueError('Missing, duplicated or unexpected continuation training seeds')
    for key in ['initial_checkpoint', 'training_sdf']:
        if len({json.dumps(r[key], sort_keys=True) for r in models}) != 1:
            raise ValueError(f'Training replicas differ in {key}')
    if len({json.dumps(r['inputs']['held_out_structures'], sort_keys=True) for r in validation}) != 1:
        raise ValueError('Validation structures differ between checkpoints')
    aggregate = {}
    for steps in [10000, 20000]:
        aggregate[str(steps)] = {}
        for window in ['original', 'central']:
            per_seed = []
            for seed in sorted(seeds):
                group = [r for r in rows if (r['total_steps'], r['window'], r['training_seed']) == (steps, window, seed)]
                raw = [r for run in group for r in run['raw_audit']['records']]
                entry = {'training_seed': seed, **{key: sum(r[key] for r in group) for key in
                         ['attempted', 'raw_valid', 'domain_eligible', 'retained', 'qualified', 'external_ranking_calls',
                          'refinement_evaluation_calls', 'refinement_gradient_calls']}}
                entry['fingerprint_joint'] = sum(r['independent']['independent_joint_successes'] for r in group)
                entry['valid_without_formal_atomic_charges'] = sum(r['valid'] and r['charged_atoms'] == 0 for r in raw)
                entry['valid_containing_rare_element'] = {element: sum(r['valid'] and r['element_counts'].get(element, 0) > 0 for r in raw)
                                                        for element in ['B', 'Si', 'Se', 'As', 'Te']}
                per_seed.append(entry)
            totals = {key: sum(r[key] for r in per_seed) for key in
                      ['attempted', 'raw_valid', 'domain_eligible', 'retained', 'qualified', 'fingerprint_joint']}
            stats = {}
            for key in ['qualified', 'fingerprint_joint']:
                rates = np.array([r[key] / r['attempted'] for r in per_seed])
                stats[key] = {'mean_rate': float(rates.mean()),
                              'training_seed_sd': float(rates.std(ddof=1)) if len(seeds) > 1 else None,
                              'training_seed_range': [float(rates.min()), float(rates.max())]}
            aggregate[str(steps)][window] = {**totals, 'per_training_seed': per_seed, 'rate_statistics': stats}
    initial = read(args.initial_control, 'initial_control_results.json')['evaluation']['control']
    initial_summary = {}
    for window in ['original', 'central']:
        group = [r for r in initial if r['window'] == window]
        if len(group) != 3 or sum(r['attempted'] for r in group) != 192:
            raise ValueError('Incomplete 5000-step reference matrix')
        initial_summary[window] = {key: sum(r[key] for r in group) for key in
                                   ['attempted', 'raw_valid', 'domain_eligible', 'qualified']}
        initial_summary[window]['fingerprint_joint'] = sum(r['independent']['independent_joint_successes'] for r in group)
    report = {'protocol': 'Common 5000-step trained initialization; 15000 further full-backbone SFT steps, seed-specific stochastic training, fixed evaluations at total 10000 and 20000 steps.',
              'complete_predeclared_three_seed_matrix': sorted(seeds) == [42, 43, 44],
              'training_seeds': sorted(seeds), 'inference_seeds': [2026, 2027, 2028],
              'initial_5000_step_control': initial_summary, 'aggregate': aggregate,
              'models': models, 'validation': validation, 'runs': rows, 'source_sha256': sources,
              'limitations': ['The shared initial 5000-step training is not replicated.',
                              'The two milestones are fixed in advance; do not select the best seed or checkpoint as an unbiased final result.',
                              'Across-seed standard deviations and ranges are descriptive, not statistical significance tests.',
                              'The fingerprint evaluator is a separately trained surrogate sharing OCELOT labels, not quantum reference truth.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
