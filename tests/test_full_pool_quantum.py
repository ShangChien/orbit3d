import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('full_quantum',Path(__file__).parents[1]/'scripts/summarize_full_pool_quantum.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_failures_and_rejected_proposals_remain_in_yield_denominator():
    rows=[{'status':'completed','singlepoint':{'prediction_eV':[-6.,-1.]}},
          {'status':'failed'}, {'status':'worker_failed'}]
    result=module.metric(rows,10,-6.)
    assert result['hits']==1 and result['yield_all_proposals']==.1
    assert result['terminal_selected']==3 and result['dft_completed']==1
    assert result['mae_eV']==0


def test_empty_quantum_cell_is_missing_error_not_zero():
    result=module.metric([],64,-6.)
    assert result['hits']==0 and result['mae_eV'] is None
    assert result['prediction_std_eV'] is None


def test_lumo_uses_lumo_prediction_not_homo_column():
    rows=[{'status':'completed','singlepoint':{'prediction_eV':[-6.,-1.2]}}]
    assert module.metric(rows,64,-1.2,property_index=1)['hits']==1
    assert module.metric(rows,64,-1.2,property_index=0)['hits']==0
