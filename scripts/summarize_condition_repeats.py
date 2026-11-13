"""Describe paired conditioner comparisons, with training seeds as repeat units."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ['point_frozen', 'point_sft', 'point_contrast', 'point_time']
EVALUATORS = ['dpa', 'fingerprint', 'unimol2']
METRICS = ['target_mae_eV', 'mean_absolute_response_bias_eV', 'response_slope',
           'mean_prediction_std_eV', 'coverage', 'hit_rate_fixed_025']


def describe(values):
    return {'values': values, 'mean': float(np.mean(values)),
            'sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'min': min(values), 'max': max(values)}


def aggregate(runs):
    seeds = [r['training_seed'] for r in runs]
    assert len(set(seeds)) == len(seeds), 'Duplicate training seeds'
    assert len({r['target'] for r in runs}) == 1, 'Different target properties'
    assert len({r['actual_proposals'] for r in runs}) == 1, 'Unequal proposal budgets'
    assert all(r['independent_inputs'] == runs[0]['independent_inputs'] for r in runs), 'Different independent evaluators or source split'
    per_seed, cell_signature = [], None
    for run in runs:
        correct = [c for c in run['cells'] if c['control'] == 'correct']
        signature = sorted((c['arm'], c['quantile'], c['goal_eV'], c['attempted']) for c in correct)
        assert all(c['preference_sigma_eV'] is None for c in correct), 'Point repeats cannot mix width conditions'
        assert {c['arm'] for c in correct} == set(ARMS)
        assert len(correct) == 20 and {c['quantile'] for c in correct} == {'10', '25', '50', '75', '90'}
        if cell_signature is None: cell_signature = signature
        assert signature == cell_signature, 'Target grids or candidate counts differ'
        row = {'training_seed': run['training_seed'], 'arms': {}}
        for arm in ARMS:
            cells = [c for c in correct if c['arm'] == arm]
            attempted = sum(c['attempted'] for c in cells)
            valid = sum(c['valid'] for c in cells)
            result = {'attempted': attempted, 'valid_rate': valid / attempted,
                      'novel_valid_fraction': sum(c['novel_valid'] for c in cells) / valid,
                      'mean_internal_diversity': float(np.mean([c['internal_diversity'] for c in cells])),
                      'evaluators': {}}
            for evaluator in EVALUATORS:
                def score(control):
                    return next(r for r in run['aggregates'] if r['arm'] == arm and r['evaluator'] == evaluator and r['control'] == control)
                metrics = {key: score('correct')[key] for key in METRICS}
                assert all(value is not None for value in metrics.values()), 'Missing target-cell evaluation'
                metrics['condition_specificity_eV'] = score('shuffled')['target_mae_eV'] - metrics['target_mae_eV']
                metrics['off_minus_correct_mae_eV'] = score('off')['target_mae_eV'] - metrics['target_mae_eV']
                result['evaluators'][evaluator] = metrics
            row['arms'][arm] = result
        per_seed.append(row)
    arms, paired = {}, {}
    for arm in ARMS:
        arms[arm] = {key: describe([r['arms'][arm][key] for r in per_seed]) for key in ['valid_rate', 'novel_valid_fraction', 'mean_internal_diversity']}
        arms[arm]['evaluators'] = {ev: {key: describe([r['arms'][arm]['evaluators'][ev][key] for r in per_seed])
                for key in per_seed[0]['arms'][arm]['evaluators'][ev]} for ev in EVALUATORS}
    for control in ['point_frozen', 'point_sft', 'point_time']:
        paired[control] = {ev: {key: describe([r['arms']['point_contrast']['evaluators'][ev][key] - r['arms'][control]['evaluators'][ev][key] for r in per_seed])
                for key in per_seed[0]['arms'][control]['evaluators'][ev]} for ev in EVALUATORS}
    return {'target': runs[0]['target'], 'training_seeds': seeds,
            'scope': 'Descriptive paired training-seed repeats. Inference seeds and targets are not independent training repeats. B informed pilot development; quantum cohort is separate.',
            'mae_definition': 'Five target quantiles equally weighted; errors and response computed only on domain-evaluable structures. Coverage and hit yield retain all proposals.',
            'response_bias_definition': 'Mean over five targets of absolute(mean prediction minus requested target); pilot-derived metric frozen before additional training repeats.',
            'actual_pools': sum(r['actual_pools'] for r in runs), 'actual_proposals': sum(r['actual_proposals'] for r in runs),
            'arms': arms, 'contrast_minus_control': paired, 'per_training_seed': per_seed,
            'independent_inputs': runs[0]['independent_inputs']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summaries', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = aggregate([json.loads(path.read_text()) for path in a.summaries])
    result['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in a.summaries}
    result['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__': main()
