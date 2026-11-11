"""Summarize fixed-omega gradients and a post-hoc pre-existing interval check."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=Path('docs/experiments'))
    parser.add_argument('--windows', type=Path, default=Path('data/optical_target_windows.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {'quantum': args.evidence / 'boron_excitation_gradients_20261112.json',
             'quantum_gradients': args.evidence / 'boron_excitation_gradients_20261112.npz',
             'target_windows': args.windows}
    cohort = json.loads(paths['quantum'].read_text())
    if cohort['completed_gradients'] != 5 or len(cohort['records']) != 5:
        raise ValueError('The predeclared five-case gradient cohort is incomplete')
    quantum = np.load(paths['quantum_gradients'])
    bounds = json.loads(args.windows.read_text())['vertical']
    names = ['s0s1', 's0t1']
    lower, upper = np.array([bounds[n] for n in names]).T
    aggregate, interval_rows = {}, []
    for model in ['dpa', 'hyff']:
        paths[model + '_predictions'] = args.evidence / f'boron_quantum_{model}_gradients_20261112.json'
        paths[model + '_gradients'] = args.evidence / f'boron_quantum_{model}_gradients_20261112.npz'
        metadata = json.loads(paths[model + '_predictions'].read_text())
        gradients = np.load(paths[model + '_gradients'])
        backend = metadata['backend']
        aggregate[model] = {}
        for name in names:
            entries = [r['comparison'][name][backend] for r in cohort['records']]
            cosine = np.array([r['cosine'] for r in entries])
            ratios = np.array([r['surrogate_norm_eV_A'] / r['quantum_norm_eV_A'] for r in entries])
            aggregate[model][name] = {'n': len(entries), 'mean_cosine': float(cosine.mean()),
                                     'cosine_range': [float(cosine.min()), float(cosine.max())],
                                     'surrogate_to_quantum_norm_ratio_range': [float(ratios.min()), float(ratios.max())]}
        for reference, prediction in zip(cohort['records'], metadata['records']):
            index = reference['quantum_index']
            if index != prediction['quantum_index']:
                raise ValueError('Gradient and prediction atom/cohort order differs')
            true = np.array([reference['vertical_excitations_eV'][n] for n in names])
            pred = np.array([prediction['prediction_eV'][n] for n in names])
            true_residual, pred_residual = true - np.clip(true, lower, upper), pred - np.clip(pred, lower, upper)
            true_gradient = sum(2 * r * quantum[f'{index}_{n}_quantum_eV_A'] for r, n in zip(true_residual, names))
            pred_gradient = sum(2 * r * gradients[f'{index}_{n}_eV_A'] for r, n in zip(pred_residual, names))
            norm_q, norm_s = float(np.linalg.norm(true_gradient)), float(np.linalg.norm(pred_gradient))
            dot = float(np.sum(true_gradient * pred_gradient))
            interval_rows.append({'model': model, 'quantum_index': index,
                                  'quantum_eV': dict(zip(names, true.tolist())), 'surrogate_eV': dict(zip(names, pred.tolist())),
                                  'quantum_interval_loss_eV2': float(true_residual @ true_residual),
                                  'surrogate_interval_loss_eV2': float(pred_residual @ pred_residual),
                                  'interval_gradient_cosine': dot / (norm_q * norm_s) if norm_q and norm_s else None,
                                  'quantum_interval_loss_slope_along_unit_surrogate_descent_eV2_A': -dot / norm_s if norm_s else None,
                                  'domain': prediction['domain']})
    report = {'attempted_quantum_cohort': cohort['attempted_quantum_cohort'],
              'completed_quantum_gradient_cases': 5, 'per_property': aggregate,
              'post_hoc_interval_check': {'window': bounds, 'loss': 'sum of squared violations in eV, as in Stage 5', 'records': interval_rows},
              'source_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
              'limitations': ['All five molecules are out of the training applicability domain and are excluded from qualified pipeline designs.',
                              'The individual-property gradient comparison was predeclared; the interval-loss analysis is post hoc, using an existing target window.',
                              'Positive property-gradient alignment does not guarantee aligned interval-loss descent when scalar predictions lie on the wrong side of a bound.',
                              'These are local fixed-omega derivatives, not finite-step optimization or a reproduced OCELOT reference benchmark.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
