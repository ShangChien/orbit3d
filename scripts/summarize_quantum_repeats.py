"""Pool disjoint per-training-seed quantum cohorts without losing failed slots."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_point_quantum import summarize


def aggregate(reports):
    first = reports[0]
    optimize = first['optimization_requested']
    arms = first['selection']['arms']
    seeds, rows, per_seed = [], [], []
    for report in reports:
        assert report['protocol'] == first['protocol'], 'Quantum protocols differ'
        assert report['optimization_requested'] == optimize, 'Do not combine different quantum stages'
        assert report['selection']['arms'] == arms, 'Policy comparisons differ'
        assigned = {int(cell.split(':')[1]) for cell in report['selection']['seed_assignment']}
        assert len(assigned) == 1, 'Each cohort must belong to one training seed'
        seed = assigned.pop()
        assert seed not in seeds, 'Duplicate training cohort; CPU/GPU repeats are not extra samples'
        seeds.append(seed)
        per_seed.append({'training_seed': seed, 'selection': report['selection'],
                         'all': report['all'], 'groups': report['groups'],
                         'by_selection': report['by_selection']})
        for record in report['records']:
            assert record['training_seed'] == seed
            rows.append({**record, 'cohort_index': record['index'], 'index': len(rows)})
    planned = sum(report['all']['planned'] for report in reports)
    return {
        'scope': 'Frozen policy cohorts pooled across training seeds. Selected-candidate diagnostic, not a population success estimate. CPU and GPU calculations of the same cohort must be summarized separately.',
        'protocol': first['protocol'], 'optimization_requested': optimize,
        'training_seeds': seeds, 'arms': arms, 'per_training_seed': per_seed,
        'all': summarize(rows, planned, optimize),
        'groups': {key: summarize([row for row in rows if f"{row['target']}/{row['arm']}" == key],
                    sum(report['groups'][key]['planned'] for report in reports), optimize)
                   for key in first['groups']},
        'by_selection': {key: summarize([row for row in rows if row['selection'] == key],
                    sum(report['by_selection'][key]['planned'] for report in reports), optimize)
                         for key in first['by_selection']},
        'records': rows,
        'index_definition': 'index is global within this aggregate; (training_seed, cohort_index, input_sha256) retains the original worker identity.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = aggregate([json.loads(path.read_text()) for path in args.summaries])
    result['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.summaries}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['groups'], indent=2))


if __name__ == '__main__':
    main()
