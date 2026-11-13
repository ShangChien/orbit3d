import copy
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_condition_repeats import aggregate


def fixture():
    path = Path(__file__).resolve().parents[1] / 'docs/experiments/oe62_point_contrast_homo_seed42_20261113.json'
    return json.loads(path.read_text())


def test_real_pilot_retains_raw_denominator_and_paired_tradeoff():
    run = fixture()
    report = aggregate([run])
    assert report['actual_proposals'] == 3072
    assert report['per_training_seed'][0]['arms']['point_contrast']['attempted'] == 640
    diff = report['contrast_minus_control']['point_time']['unimol2']
    assert diff['target_mae_eV']['mean'] > 0
    assert diff['mean_absolute_response_bias_eV']['mean'] < 0
    assert diff['target_mae_eV']['sample_sd'] is None


def test_same_seed_rescoring_is_not_a_training_repeat():
    run = fixture()
    with pytest.raises(AssertionError, match='Duplicate'):
        aggregate([run, copy.deepcopy(run)])
    other = copy.deepcopy(run); other['training_seed'] = 43
    other['cells'][0]['goal_eV'] += .1
    with pytest.raises(AssertionError, match='grids'):
        aggregate([run, other])
