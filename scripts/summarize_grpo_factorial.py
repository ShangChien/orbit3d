"""Join the predeclared GRPO matrix only when every training/inference cell exists."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

VARIANTS = ['full', 'no_kl', 'no_physics', 'no_replay', 'uncertainty_on']
TRAINING_SEEDS = [42, 43, 44]
INFERENCE_SEEDS = [2026, 2027, 2028]


def summarize(runs):
    expected = set(itertools.product(VARIANTS, TRAINING_SEEDS, INFERENCE_SEEDS))
    keys = [(r['variant'], r['training_seed'], r['inference_seed']) for r in runs]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('The 5 x 3 x 3 matrix has missing, duplicate or unexpected cells')
    if any(r['attempted'] != 64 or r['independent']['attempted'] != 64 for r in runs):
        raise ValueError('Every matrix cell requires the fixed 64-proposal denominator')
    aggregate = {}
    for variant in VARIANTS:
        selected = [r for r in runs if r['variant'] == variant]
        per_seed = []
        for seed in TRAINING_SEEDS:
            group = [r for r in selected if r['training_seed'] == seed]
            per_seed.append({'training_seed': seed, 'attempted': sum(r['attempted'] for r in group),
                             'valid': sum(r['raw_valid'] for r in group),
                             'optimizing_joint': sum(r['qualified'] for r in group),
                             'independent_joint': sum(r['independent']['independent_joint_successes'] for r in group)})
        stats = {}
        for label in ['optimizing_joint', 'independent_joint']:
            rates = np.array([r[label]/r['attempted'] for r in per_seed])
            # Resample training-seed clusters, retaining all three inference runs.
            resampled = [rates[list(indices)].mean() for indices in itertools.product(range(3), repeat=3)]
            stats[label] = {'total': sum(r[label] for r in per_seed), 'mean_rate': float(rates.mean()),
                            'training_seed_sd': float(rates.std(ddof=1)),
                            'training_seed_range': [float(rates.min()), float(rates.max())],
                            'exploratory_cluster_bootstrap_percentiles_95': np.quantile(resampled,[.025,.975]).tolist()}
        aggregate[variant] = {'attempted': sum(r['attempted'] for r in selected),
                              'raw_valid': sum(r['raw_valid'] for r in selected),
                              'domain_eligible': sum(r['domain_eligible'] for r in selected),
                              'per_training_seed': per_seed, **stats}
    return aggregate


def main():
    p=argparse.ArgumentParser();p.add_argument('--stages',type=Path,required=True)
    p.add_argument('--seed42-ablation',type=Path,required=True)
    p.add_argument('--remaining',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    stages=json.loads(a.stages.read_text());original=json.loads(a.seed42_ablation.read_text())
    # This source collection explicitly contains only full-objective replicas;
    # its seed-43/44 records predate the per-row variant field.
    assert all(r.get('variant', 'full') == 'full' for r in stages['grpo_replicas'])
    runs=[{**r, 'variant':'full'} for r in stages['grpo_replicas']] + [
        {**r,'training_seed':42} for r in original['runs'] if r['variant']!='full'
    ] + json.loads(a.remaining.read_text())
    aggregate=summarize(runs)
    for row in runs: row.pop('checkpoint',None)
    report={'protocol':'5 objectives x 3 GRPO training seeds x 3 inference seeds; 50 rollouts x 16 proposals and 2 optimizer passes, common SFT initialization',
            'training_seeds':TRAINING_SEEDS,'inference_seeds':INFERENCE_SEEDS,
            'source_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
                             [('stage_and_full_runs',a.stages),('seed42_removals',a.seed42_ablation),('remaining_runs',a.remaining)]},
            'aggregate':aggregate,'runs':sorted(runs,key=lambda r:(VARIANTS.index(r['variant']),r['training_seed'],r['inference_seed'])),
            'limitations':['One common SFT initialization; these are not independent SFT replications.',
                           'Only three training-seed clusters. The 27 enumerated bootstrap resamples are descriptive sensitivity intervals, not reliable 95% coverage or a significance test.',
                           'The independent evaluator is a trained fingerprint surrogate, not DFT reference truth.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
