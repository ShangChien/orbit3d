import copy
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_point_quantum import summarize
from summarize_quantum_repeats import aggregate


def cohort(seed, successful=True):
    row = {'index': 0, 'training_seed': seed, 'input_sha256': str(seed), 'target': 'homo',
           'target_eV': -6., 'arm': 'conditional200', 'selection': 'top1',
           'dpa_prediction_eV': [-6., -2.], 'singlepoint': {'prediction_eV': [-6.1, -2.]},
           'geometry': {'converged': successful, 'graph_preserved': successful}}
    if successful: row['optimized_singlepoint'] = {'prediction_eV': [-6.2, -2.]}
    summary = summarize([row], 2, True)
    return {'protocol': 'fixed', 'optimization_requested': True,
            'selection': {'arms': ['conditional200'], 'seed_assignment': [f'50:{seed}:2052']},
            'records': [row], 'all': summary, 'groups': {'homo/conditional200': summary},
            'by_selection': {'top1': summarize([row], 1), 'fixed_hash': summarize([], 1)}}


def test_repeated_local_indices_and_missing_slots_preserved():
    result = aggregate([cohort(42), cohort(43, successful=False)])
    assert result['all']['planned'] == 4
    assert result['all']['selected'] == 2
    assert result['all']['accepted_optimized'] == 1
    assert result['all']['optimized_target']['hit_rate_planned']['0.25'] == .25
    assert result['all']['failed_or_unaccepted_indices'] == [1]
    assert [(row['training_seed'], row['cohort_index']) for row in result['records']] == [(42, 0), (43, 0)]
    assert result['all']['input_target_same_accepted_subset']['evaluated'] == 1


def test_engine_repeat_cannot_inflate_sample_count():
    a = cohort(42)
    b = copy.deepcopy(a)
    b['records'][0]['engine'] = 'cpu'
    with pytest.raises(AssertionError, match='Duplicate training cohort'):
        aggregate([a, b])


def test_unrequested_geometry_not_combined_with_optimized_cohorts():
    a, b = cohort(42), cohort(43)
    b['optimization_requested'] = False
    with pytest.raises(AssertionError, match='different quantum stages'):
        aggregate([a, b])
