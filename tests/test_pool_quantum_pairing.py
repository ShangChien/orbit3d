import importlib.util
from pathlib import Path
import pytest
import numpy as np
spec=importlib.util.spec_from_file_location('pool_quantum',Path(__file__).parents[1]/'scripts/compare_pool_quantum.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_prediction_reuse_requires_same_ordered_geometry():
    symbols=['C','H','H','H','H'];elements=('C','H')
    centered=np.array([[0.,0.,0.],[.625,.625,.625],[.625,-.625,-.625],[-.625,.625,-.625],[-.625,-.625,.625]])
    record={'symbols':symbols,'coordinates_A':centered.tolist()}
    raw=centered+16.
    assert module.verify_geometry(record,symbols,raw,elements)==0
    assert module.verify_geometry(record,symbols,raw+8.,elements)==0
    altered=raw.copy();altered[-1,0]+=.001
    with pytest.raises(ValueError,match='frozen raw candidate'):
        module.verify_geometry(record,symbols,altered,elements)
    with pytest.raises(ValueError,match='frozen raw candidate'):
        module.verify_geometry(record,['H','C','H','H','H'],raw,elements)


def test_proxy_hit_agreement_keeps_both_false_positive_and_false_negative():
    rows=[{'target_eV':0.,'quantum_eV':actual,'predictions_eV':{'dpa':pred}}
          for pred,actual in [(0.,0.),(0.,1.),(1.,0.),(1.,1.)]]
    result=module.summarize(rows,'dpa')
    assert result['target_hit_agreement_025']=={'true_positive':1,'false_positive':1,'false_negative':1,'true_negative':1}
