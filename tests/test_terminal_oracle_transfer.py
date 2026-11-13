import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('transfer',Path(__file__).parents[1]/'scripts/evaluate_terminal_oracle_transfer.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_property_calibration_distinguishes_false_hits_and_missing_quantum():
    row=dict(target='homo',target_eV=-6.,truth=[-5.,-1.],baseline=[-6.,-1.],affine=[-5.,-1.],adapted=[-5.1,-1.])
    missing={**row,'truth':None}
    result=module.metrics([row,missing])
    assert result['baseline']['paired']==1 and result['baseline']['false_positive']==1
    assert result['adapted']['false_positive']==0 and result['adapted']['mae_eV'][0]<.11
    assert result['affine']['mae_eV']==[0.,0.]
