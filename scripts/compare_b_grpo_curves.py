"""Compare two fixed-budget GRPO curves and their gains over their own parents."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from orbit3d.oracles.unimol2_task import file_sha256, graph_set, summarize_task
from summarize_b_task_experiment import aggregate, paired_hit_bootstrap


def interaction_interval(margin, margin_parent, direct, direct_parent, repeats=4000):
    """Paired difference of hit-rate gains; preserve cross-target stream reuse."""
    indexed = [{(p['quantile'], p['seed']): p for p in pools}
               for pools in [margin, margin_parent, direct, direct_parent]]
    assert all(set(x) == set(indexed[0]) for x in indexed)
    by_seed = {}
    for key in sorted(indexed[0]):
        pools = [x[key] for x in indexed]
        assert len({p['goal_eV'] for p in pools}) == 1
        assert len({p['target'] for p in pools}) == 1
        assert all([r['candidate_id'] for r in p['records']] ==
                   [r['candidate_id'] for r in pools[0]['records']] for p in pools)
        column = ['homo', 'lumo'].index(pools[0]['target'])
        hits = [np.array([int(r['evaluated'] and
                 abs(r['prediction_eV'][column] - p['goal_eV']) <= .25)
                 for r in p['records']]) for p in pools]
        by_seed.setdefault(key[1], []).append(hits[0]-hits[1]-hits[2]+hits[3])
    rng = np.random.default_rng(20261114)
    samples, points = [], []
    for rows in by_seed.values():
        stream = np.mean(rows, axis=0)
        indices = rng.integers(len(stream), size=(repeats, len(stream)))
        samples.append(stream[indices].mean(axis=1)); points.append(stream.mean())
    return dict(difference_of_gains_pp=float(100*np.mean(points)),
                paired_sampling_95_pp=(100*np.quantile(np.mean(samples, axis=0), [.025,.975])).tolist(),
                scope='(margin+GRPO - margin) - (SFT+GRPO - SFT); fixed training seed, pointwise sampling interval only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direct', type=Path, required=True)
    parser.add_argument('--margin', type=Path, required=True)
    parser.add_argument('--source-train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = graph_set(args.source_train)
    assets = set(source)
    inputs = {str(args.source_train): file_sha256(args.source_train)}
    for root in [args.direct, args.margin]:
        path = root/'training/rollouts.jsonl'
        inputs[str(path)] = file_sha256(path)
        for line in path.read_text().splitlines():
            assets.update(r['smiles'] for r in json.loads(line)['rollout_records'] if r.get('smiles'))
    def read(root, phase, group):
        path = root/phase/f'group{group}'/'summary.json'
        inputs[str(path)] = file_sha256(path)
        return json.loads(path.read_text())['pools']
    def common_novelty(pools):
        pools = copy.deepcopy(pools)
        for p in pools:
            p['metrics'] = summarize_task(p['records'], p['target'], p['goal_eV'], source, assets)
        return aggregate(pools)
    results = {}
    for phase, groups in [('development', [0,64,128,256,512,1024]), ('validation', [0,1024])]:
        direct_parent = read(args.direct, phase, 0)
        margin_parent = read(args.margin, phase, 0)
        results[phase] = {}
        for group in groups:
            direct, margin = read(args.direct, phase, group), read(args.margin, phase, group)
            results[phase][group] = dict(
                direct=common_novelty(direct), margin=common_novelty(margin),
                margin_minus_direct=paired_hit_bootstrap(margin, direct),
                direct_gain=paired_hit_bootstrap(direct, direct_parent),
                margin_gain=paired_hit_bootstrap(margin, margin_parent),
                interaction=interaction_interval(margin, margin_parent, direct, direct_parent))
    costs = {name:json.loads((root/'training/costs.json').read_text())
             for name,root in [('direct',args.direct),('margin',args.margin)]}
    protocol = json.loads((args.margin/'protocol.json').read_text())
    result = dict(primary_endpoint='Fresh validation, fixed group1024; development curve secondary',
        scope='Equal GRPO proposals and updates, not equal total time. Both use frozen B, not independent DFT truth. '
              'Common novelty reference is source plus the study-wide union of both full training rollout graph sets; '
              'original per-checkpoint novelty and checkpoint selections remain unchanged.',
        results=results, training_costs=costs, margin_pretraining=protocol['margin_pretraining'],
        common_novelty_graphs=len(assets), inputs=inputs, analysis_script_sha256=file_sha256(Path(__file__)))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
