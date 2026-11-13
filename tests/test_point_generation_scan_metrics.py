"""Point-scan denominators and active-property semantics."""
import importlib
from pathlib import Path


def test_invalid_and_ood_rows_stay_in_denominator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    metrics=importlib.import_module('experiment_point_generation_scan').target_metrics
    rows=[{'prediction':[-6.,99.],'domain':{'in_domain':True},'rms_force':.05},
          {'prediction':[-5.,0.],'domain':{'in_domain':False},'rms_force':.05},
          {'valid':False}]
    m=metrics(rows,0,-6.)
    assert m['attempted']==3 and m['evaluated']==1 and m['coverage']==1/3
    assert m['mae_eV']==0 and m['hits']['0.25']==1 and m['hits_rms01']==1
    assert metrics(rows,0,-6.,in_domain=False)['mae_eV']==.5
    assert metrics([],0,-6.)['mae_eV'] is None
