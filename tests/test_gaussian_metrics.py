import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_gaussian_pilot import statistics


def test_width_acceptance_does_not_change_fixed_error_or_failure_denominator():
    narrow=statistics([0.,.2,.4],0.,6,.1)
    broad=statistics([0.,.2,.4],0.,6,.5)
    assert narrow['property_hits_fixed_025']==broad['property_hits_fixed_025']==2
    assert narrow['coverage']==broad['coverage']==.5
    assert narrow['target_mae_eV']==broad['target_mae_eV']
    assert narrow['hit_rate_all_attempts']['0.25']==broad['hit_rate_all_attempts']['0.25']==2/6
    assert narrow['hit_rate_all_attempts']['0.1']==1/6
    assert broad['hit_rate_all_attempts']['0.5']==3/6


def test_no_evaluable_structure_has_undefined_mae_and_zero_hit_yield():
    result=statistics([],0.,8,.25)
    assert result['target_mae_eV'] is None and result['prediction_std_eV'] is None
    assert result['coverage']==0 and result['hit_rate_all_attempts']['0.25']==0
