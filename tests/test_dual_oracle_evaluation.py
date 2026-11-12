import importlib
from pathlib import Path
import numpy as np


def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    return importlib.import_module('reevaluate_dual_oracle'),importlib.import_module('summarize_dual_cross_evaluation')


def test_cross_model_metrics_keep_attempts_and_do_not_replace_missing_B(monkeypatch):
    m,_=modules(monkeypatch)
    rows=[{'prediction':[-6.,-1.],'dpa_common_prediction':[-6.1,-1.],'unimol_prediction':[-6.2,-1.], 'domain':{'in_domain':True}},
          {'prediction':[-6.,-1.],'domain':{'in_domain':True}},
          {'prediction':[-6.,-1.],'dpa_common_prediction':[-6.,-1.],'unimol_prediction':[-6.,-1.], 'domain':{'in_domain':False}},
          {'valid':False}]
    stats=m.target_stats(rows,0,-6.,64)
    assert stats['attempted']==64 and stats['matched_domain_evaluated']==1
    assert stats['models']['A_historical']['hits025']==1
    assert stats['models']['B_unimol']['evaluated']==1
    assert np.isclose(stats['models']['B_unimol']['mae'],.2)
    assert m.target_stats(rows,0,-6.,64,matched=False)['models']['A_historical']['evaluated']==2


def test_target_macro_is_not_biased_by_number_of_seeds(monkeypatch):
    _,m=modules(monkeypatch)
    pools=[]
    for target,value in [(0.,1.),(1.,9.),(1.,9.)]:
        metric={'mae':value,'bias':-value,'std':value,'p90_absolute_error':value,'hits025':0}
        pools.append({'target':target,'attempted':64,'raw':{'models':{name:metric for name in m.MODELS},'matched_domain_evaluated':1,'A_B_mean_absolute_disagreement':.1},
                      'diversity':{'valid':1,'unique_fraction_valid':1.,'scaffolds':1,'morgan_distance':.5}})
    out=m.macro(pools)
    assert out['attempted']==192
    assert out['models']['B_unimol']['mae']==5
    assert out['models']['B_unimol']['mean_abs_bias']==5
