"""Compare conditional pools to one shared off pool without multiplying its sample size."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def hit_array(report, arm, goals, seeds, slots):
    values = np.zeros((len(goals), len(seeds), slots))
    off = report['control'] == 'off'
    for row in report['records']:
        if row['arm'] != arm or row['status'] != 'completed':
            continue
        value = row['singlepoint']['prediction_eV'][['homo', 'lumo'].index(report['target'])]
        for i, (q, goal) in enumerate(goals.items()):
            if off or row['quantile'] == int(q):
                values[i, seeds.index(row['inference_seed']), row['candidate_id']] = abs(value-goal) <= .25
    return values


def compare(correct, off, repeats=4000):
    assert correct['control'] == 'correct' and off['control'] == 'off'
    assert correct['target'] == off['target']
    baseline, = off['arms']
    goals = {str(r['quantile']): r['target_eV'] for r in correct['records']}
    goals = dict(sorted(goals.items(), key=lambda item: int(item[0])))
    seeds = sorted({r['inference_seed'] for r in off['records']})
    slots, remainder = divmod(off['arms'][baseline]['proposals'], len(seeds))
    assert remainder == 0
    reference = hit_array(off, baseline, goals, seeds, slots)
    output = {}
    for arm, metrics in correct['arms'].items():
        assert metrics['proposals'] == len(goals)*len(seeds)*slots
        difference = hit_array(correct, arm, goals, seeds, slots)-reference
        rng = np.random.default_rng(1521)
        samples = []
        for _ in range(repeats):
            selected = rng.integers(0, len(seeds), size=len(seeds))
            ids = rng.integers(0, slots, size=(len(seeds), slots))
            samples.append(float(np.concatenate([difference[:, seed, idx] for seed, idx in zip(selected, ids)], axis=1).mean()))
        output[arm] = {
            'correct_yield': metrics['yield_equal_target'],
            'shared_off_yield': off['arms'][baseline]['yield_equal_target'],
            'correct_minus_off_yield': float(difference.mean()),
            'sampling_delta95': np.quantile(samples, [.025, .975]).tolist(),
            'correct_mae_eV': metrics['mae_equal_target_eV'],
            'off_mae_eV': off['arms'][baseline]['mae_equal_target_eV'],
        }
    return {'scope': 'Fixed training seed; paired inference-seed block and candidate-slot bootstrap (4000, seed1521). The same off proposals are shared across targets and arms in every replicate. Does not estimate training variance. MAE conditions on successful quantum endpoints.', 'goals_eV': goals, 'shared_off_unique_proposals': off['arms'][baseline]['proposals'], 'arms': output}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--correct', type=Path, required=True)
    p.add_argument('--off', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = compare(json.loads(a.correct.read_text()), json.loads(a.off.read_text()))
    report['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.correct, a.off]}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['arms'], indent=2))


if __name__ == '__main__':
    main()
