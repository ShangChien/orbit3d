"""Recompute the fixed-budget BRICS versus octree comparison from public records."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(evidence):
    filenames = {
        'brics': 'brics_wide_baseline_20261112.json',
        'octree': 'element_balance_20261112.json',
        'hyff': 'brics_hyff_independent_20261112.json',
        'octree_hyff': 'octree5000_hyff_independent_20261112.json',
    }
    data = {key: json.loads((evidence / name).read_text()) for key, name in filenames.items()}
    sources = {'brics': data['brics'], 'octree_5000': data['octree']['evaluation']['control']}
    results = {}
    for model, rows in sources.items():
        expected = {(window, seed) for window in ['original', 'central'] for seed in [2026, 2027, 2028]}
        keys = [(row['window'], row['inference_seed']) for row in rows]
        if len(keys) != 6 or set(keys) != expected or any(row['attempted'] != 64 for row in rows):
            raise ValueError('Incomplete or duplicated model/window/seed matrix')
        results[model] = {}
        for window in ['original', 'central']:
            selected = sorted((row for row in rows if row['window'] == window), key=lambda row: row['inference_seed'])
            aggregate = {key: sum(row[key] for row in selected) for key in [
                'attempted', 'raw_valid', 'domain_eligible', 'retained', 'qualified',
                'external_ranking_calls', 'refinement_evaluation_calls', 'refinement_gradient_calls',
                'export_recheck_calls',
            ]}
            aggregate['fingerprint_joint'] = sum(row['independent']['independent_joint_successes'] for row in selected)
            aggregate['dpa_joint_by_seed'] = [row['qualified'] for row in selected]
            aggregate['fingerprint_joint_by_seed'] = [row['independent']['independent_joint_successes'] for row in selected]
            if model == 'brics':
                aggregate['raw_graph_valid'] = sum(row['raw_graph_valid'] for row in selected)
                aggregate['retained_novel_vs_training'] = sum(row['retained_novel_vs_training'] for row in selected)
                aggregate['retained_mean_sa'] = sum(row['independent']['synthetic_accessibility']['mean'] * row['retained'] for row in selected) / aggregate['retained']
            rescored = {row['source_run']: row for row in data['hyff' if model == 'brics' else 'octree_hyff']}
            independent = [rescored[f'{window}_seed{row["inference_seed"]}'] for row in selected]
            if any(row['attempted'] != 64 for row in independent):
                raise ValueError('Independent denominator differs from source attempts')
            for force in ['source', 'own']:
                key = f'independent_joint_with_{force}_force'
                aggregate[f'hyff_joint_with_{force}_force'] = sum(row[key] for row in independent)
                aggregate[f'hyff_joint_with_{force}_force_by_seed'] = [row[key] for row in independent]
            results[model][window] = aggregate
    return {
        'protocol': '64 proposals per seed/window, external-ranking cap 32, retained cap 16, 30 refinement steps; 3 inference seeds. Caps match, actual calls and total compute do not.',
        'limitations': [
            'BRICS reuses the same 192 unconditional proposals across two target windows; 384 condition-specific attempts are not 384 independent graph draws.',
            'Both methods use the same 8188 training conformers, DPA evaluator, target scaling and interval definitions. Octree also uses upstream pretrained weights.',
            'Fingerprint and HyFF are separately trained surrogates sharing OCELOT labels, not quantum ground truth.',
            'Training novelty is relative to the 8188 adaptation structures, not the upstream pretraining corpus; SA is a heuristic, not a synthesis route.',
        ],
        'source_sha256': {name: hashlib.sha256((evidence / name).read_bytes()).hexdigest() for name in filenames.values()},
        'models': results,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=Path('docs/experiments'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.evidence), indent=2) + '\n')
