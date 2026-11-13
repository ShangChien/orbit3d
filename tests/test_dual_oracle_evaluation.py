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
    assert out['raw_mean_unique_fraction']==1.
    assert 'mean_unique_fraction' not in out


def test_quantum_failures_keep_planned_denominator_and_missing_targets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    m=importlib.import_module('summarize_dual_quantum')
    models={name:[.1,0.] for name in ['dpa_historical','dpa_common','unimol']}
    common={'property':'homo','property_index':0,'target':0.,'quantum_input':[.1,0.],
            'input_models':models,'optimized_models':models}
    rows=[{**common,'id':'q0','status':'completed','graph_preserved':True,'quantum_optimized':[.1,0.]},
          {**common,'id':'q1','status':'failed','graph_preserved':False},
          {**common,'id':'q2','status':'completed','target':2.,'graph_preserved':False,'quantum_optimized':[2.,0.]}]
    out=m.summarize(rows)
    assert out['planned']==3 and out['singlepoints']==3 and out['optimized_singlepoints']==2
    assert out['graph_preserved']==1
    assert out['quantum_optimized_preserved']['hit_fraction_planned']==1/3
    assert out['failed_ids']==['q1'] and out['changed_identity_ids']==['q2']
    assert out['quantum_optimized_equal_target']['missing_target_cells']==[{'property':'homo','target':2.}]
    assert out['joint_proxy_diagnostic']['optimized']['both_proxy_hits']==1
    assert out['joint_proxy_diagnostic']['optimized']['quantum_hits_among_both']==1
    assert out['joint_proxy_diagnostic']['optimized']['planned']==3
