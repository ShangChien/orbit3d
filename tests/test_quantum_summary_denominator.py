import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('point_quantum',Path(__file__).parents[1]/'scripts/summarize_point_quantum.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_missing_candidate_slot_stays_in_quantum_hit_denominator():
    row={'index':0,'target':'homo','target_eV':-6.,'dpa_prediction_eV':[-6.,-2.],
         'singlepoint':{'prediction_eV':[-6.,-2.]}}
    result=module.summarize([row],planned=2)
    assert result['planned']==2 and result['selected']==1
    assert result['input_target']['hit_rate_planned']['0.25']==.5
    assert result['input_target']['coverage']==.5
    assert result['optimized_target']['hit_rate_planned']['0.25']==0


def test_singlepoint_only_stage_does_not_report_unrequested_optimization_as_failure():
    row={'index':0,'target':'homo','target_eV':-6.,'dpa_prediction_eV':[-6.,-2.],
         'singlepoint':{'prediction_eV':[-6.,-2.]}}
    result=module.summarize([row],planned=2,optimization_requested=False)
    assert result['input_target']['coverage']==.5
    assert result['optimized_target'] is None and result['accepted_optimized'] is None
    assert result['failed_or_unaccepted_indices'] is None
    assert result['failed_singlepoint_indices']==[]
