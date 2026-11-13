import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('terminal_baselines', Path(__file__).parents[1]/'scripts/compare_terminal_baselines.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_identical_pools_have_zero_delta_and_do_not_multiply_off_proposals():
    off_rows = [dict(arm='off',status='completed',quantile=50,target_eV=-6.,inference_seed=s,candidate_id=i,singlepoint={'prediction_eV':[-6.+i,0.]}) for s in [1,2] for i in range(2)]
    off = dict(control='off',target='homo',records=off_rows,arms={'off':dict(proposals=4,yield_equal_target=.25,mae_equal_target_eV=.75)})
    correct_rows = [{**r,'arm':'a','quantile':q,'target_eV':target} for q,target in [(25,-6.),(75,-4.)] for r in off_rows]
    correct = dict(control='correct',target='homo',records=correct_rows,arms={'a':dict(proposals=8,yield_equal_target=.25,mae_equal_target_eV=.75)})
    result = module.compare(correct,off,repeats=100)
    assert result['shared_off_unique_proposals'] == 4
    assert result['arms']['a']['correct_minus_off_yield'] == 0
    assert result['arms']['a']['sampling_delta95'] == [0,0]
