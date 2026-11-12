"""Summarize the fixed 24-candidate cohort without replacing failed candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def error_stats(errors):
    errors = np.asarray(errors)
    if not len(errors):
        return {'n': 0}
    return {'n': len(errors), 'mae_eV': float(abs(errors).mean()),
            'rmse_eV': float(np.sqrt(np.mean(errors**2))),
            'bias_eV': float(errors.mean()), 'std_eV': float(errors.std()),
            'absolute_error_p90_eV': float(np.quantile(abs(errors), .9)),
            'hits025': int(np.sum(abs(errors) <= .25))}


def summarize(rows):
    original = [r for r in rows if 'quantum_input' in r]
    preserved = [r for r in rows if r['graph_preserved'] and 'quantum_optimized' in r]
    report = {'planned': len(rows), 'singlepoints': len(original),
              'optimized_singlepoints': sum('quantum_optimized' in r for r in rows),
              'graph_preserved': len(preserved), 'models': {},
              'failed_ids': [r['id'] for r in rows if r['status'] != 'completed'],
              'changed_identity_ids': [r['id'] for r in rows if 'quantum_optimized' in r and not r['graph_preserved']]}
    def target_errors(records, key):
        return [r[key][r['property_index']]-r['target'] for r in records]
    report['quantum_input'] = error_stats(target_errors(original, 'quantum_input'))
    report['quantum_input_preserved_subset'] = error_stats(target_errors(preserved, 'quantum_input'))
    report['quantum_optimized_preserved'] = error_stats(target_errors(preserved, 'quantum_optimized'))
    report['quantum_optimized_preserved']['hit_fraction_planned'] = (
        report['quantum_optimized_preserved'].get('hits025', 0)/len(rows) if rows else None)
    report['quantum_geometry_mean_absolute_change_eV'] = float(np.mean([
        abs(r['quantum_optimized'][r['property_index']]-r['quantum_input'][r['property_index']])
        for r in preserved])) if preserved else None
    for model in ['dpa_historical', 'dpa_common', 'unimol']:
        per_model = {}
        for stage, records, qkey in [('input', original, 'quantum_input'),
                                      ('input_preserved_subset', preserved, 'quantum_input'),
                                      ('optimized_preserved', preserved, 'quantum_optimized')]:
            pkey = 'optimized_models' if stage == 'optimized_preserved' else 'input_models'
            per_model[stage] = {
                'target_error': error_stats([r[pkey][model][r['property_index']]-r['target'] for r in records]),
                'error_against_quantum': error_stats([
                    r[pkey][model][r['property_index']]-r[qkey][r['property_index']] for r in records]),
            }
        report['models'][model] = per_model
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--input-predictions', type=Path, required=True)
    p.add_argument('--optimized-predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evaluator', required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    before = json.loads(a.input_predictions.read_text())
    after = json.loads(a.optimized_predictions.read_text())
    assert manifest['selected'] == manifest['expected'] == 24
    assert manifest['sdf_sha256'] == after['cohort_sha256']
    rows = []
    for i, record in enumerate(manifest['records']):
        result = json.loads((a.results/f'job{i:02d}/result.json').read_text())
        assert result['input_sha256'] == manifest['sdf_sha256']
        row = dict(record, property_index=['homo', 'lumo'].index(record['property']),
                   status=result['status'], reason=result.get('reason'),
                   graph_preserved=result.get('geometry', {}).get('graph_preserved', False),
                   input_models={name: values[i] for name, values in before['models'].items()},
                   elapsed_seconds=result['elapsed_seconds'])
        if 'singlepoint' in result:
            row['quantum_input'] = result['singlepoint']['prediction_eV']
        if 'optimized_singlepoint' in result:
            row['quantum_optimized'] = result['optimized_singlepoint']['prediction_eV']
            j = after['indices'].index(i)
            row['optimized_models'] = {name: values[j] for name, values in after['models'].items()}
            row['geometry_steps'] = result['geometry']['steps']
            row['identity'] = result['geometry'].get('identity')
            row['identity_error'] = result['geometry'].get('identity_error')
        rows.append(row)
    groups = {}
    for prop in ['homo', 'lumo']:
        for method in sorted({r['method'] for r in rows}):
            chosen = [r for r in rows if r['property'] == prop and r['method'] == method]
            groups[f'{prop}/{method}'] = summarize(chosen)
            for rule in ['A_top1', 'fixed_random']:
                subset = [r for r in chosen if r['rule'] == rule]
                if subset:
                    groups[f'{prop}/{method}/{rule}'] = summarize(subset)
    report = {
        'evaluator': a.evaluator,
        'scope': 'Fixed exploratory cohort, not population success estimation. Geometry failures remain in all 24 planned denominators. No candidate replacement or evaluator-based selection.',
        'protocol': 'GPU4PySCF PBE0/def2-TZVP/DF single points; PBE-D3BJ/def2-TZVP/DF geometry optimization. Approximate OE62 comparison, not identical Tier2/PBE-TS protocol.',
        'selection': {k: v for k, v in manifest.items() if k != 'records'},
        'all': summarize(rows), 'groups': groups, 'records': rows,
        'evidence_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
            a.manifest, a.input_predictions, a.optimized_predictions]},
    }
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'all': report['all'], 'groups': {k: {
        q: v[q] for q in ['planned', 'graph_preserved', 'quantum_input', 'quantum_optimized_preserved']}
        for k, v in groups.items()}}, indent=2))


if __name__ == '__main__':
    main()
