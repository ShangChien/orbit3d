"""Paired, fixed-checkpoint whole-proposal comparisons from frozen DFT reports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_terminal_baselines import hit_array


def compare(report, reference, repeats=4000):
    if report['control'] != 'correct':
        raise ValueError('Arm comparisons require distinct correct-condition proposal pools')
    goals = dict(sorted({str(r['quantile']): r['target_eV'] for r in report['records']}.items(), key=lambda x: int(x[0])))
    seeds = sorted({r['inference_seed'] for r in report['records']})
    base = report['arms'][reference]
    slots, remainder = divmod(base['proposals'], len(goals) * len(seeds))
    if remainder or not slots:
        raise ValueError('Unequal request budgets require a different comparison protocol')
    reference_hits = hit_array(report, reference, goals, seeds, slots)
    output = {}
    for arm, metrics in report['arms'].items():
        if arm == reference:
            continue
        if metrics['proposals'] != base['proposals']:
            raise ValueError('Paired arms must use equal proposal budgets')
        difference = hit_array(report, arm, goals, seeds, slots) - reference_hits
        rng = np.random.default_rng(1614)
        samples = []
        for _ in range(repeats):
            chosen = rng.integers(len(seeds), size=len(seeds))
            ids = rng.integers(slots, size=(len(seeds), slots))
            samples.append(np.concatenate([difference[:, s, idx] for s, idx in zip(chosen, ids)], axis=1).mean())
        output[arm] = {
            'yield_delta': float(difference.mean()),
            'sampling_delta95': np.quantile(samples, [.025, .975]).tolist(),
            'terminal_yield_delta': (metrics['terminal_selected']-base['terminal_selected'])/base['proposals'],
            'mae_delta_eV_successful_endpoints': metrics['mae_equal_target_eV']-base['mae_equal_target_eV'],
            'response_slope_delta': metrics['response_slope']-base['response_slope'],
        }
    return {'scope': 'Fixed checkpoints, paired inference-seed blocks and candidate slots shared across target requests. 4000 bootstrap draws, seed1614. Failures remain zeros in full-proposal hit vectors. Intervals describe sampling, not training variance. Successful-endpoint MAE is conditional on different accepted populations.',
            'reference': reference, 'target': report['target'], 'goals_eV': goals, 'comparisons': output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.report.read_text()), args.reference)
    result['source_sha256'] = {str(args.report): hashlib.sha256(args.report.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['comparisons'], indent=2))


if __name__ == '__main__':
    main()
