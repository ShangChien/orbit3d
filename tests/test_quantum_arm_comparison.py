import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location('arm_compare', Path(__file__).parents[1] / 'scripts/compare_quantum_arms.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_failed_and_missing_endpoints_stay_in_full_proposal_denominator():
    metric = dict(proposals=4, terminal_selected=2, mae_equal_target_eV=.1, response_slope=.5)
    rows = [dict(arm=arm, quantile=q, target_eV=-6., inference_seed=1, candidate_id=slot,
                 status=status, singlepoint={'prediction_eV': [-6., -1.]})
            for arm, q, slot, status in [('base', 25, 0, 'completed'), ('new', 25, 0, 'completed'),
                                         ('new', 75, 0, 'completed'), ('base', 75, 1, 'scf_failed')]]
    report = dict(control='correct', target='homo', arms={'base': metric, 'new': metric}, records=rows)
    result = module.compare(report, 'base', repeats=20)['comparisons']['new']
    assert result['yield_delta'] == .25
    report['arms']['new'] = {**metric, 'proposals': 6}
    with pytest.raises(ValueError, match='equal proposal'):
        module.compare(report, 'base')
