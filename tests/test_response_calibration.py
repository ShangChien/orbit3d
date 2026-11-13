import importlib
from pathlib import Path
import numpy as np


def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    return importlib.import_module('prepare_response_calibration'),importlib.import_module('summarize_response_calibration')


def test_linear_response_recovery_and_supported_gp_grid(monkeypatch):
    prep,_=modules(monkeypatch)
    x=np.linspace(-3,1,5);fit=prep.fit_response(x,.4*x-1,np.full(5,.01))
    assert np.isclose(fit['linear_slope'],.4) and np.isclose(fit['linear_intercept'],-1)
    assert min(fit['grid'])==min(x) and max(fit['grid'])==max(x)
    assert np.isfinite(fit['gp_mean']).all()


def test_weighted_selection_has_no_replacement_or_underflow_fallback(monkeypatch):
    _,s=modules(monkeypatch)
    rows=[dict(candidate_id=i,smiles=str(i),prediction=[100+i],independent_prediction=[0],domain={'in_domain':True}) for i in range(6)]
    # All direct exp weights would underflow, but log-domain sampling still prefers lower errors.
    selected=s.choose(rows,0,0,'gaussian025',42,k=4)
    assert [r['candidate_id'] for r in selected]==[0,1,2,3]
    assert selected==s.choose(rows,0,0,'gaussian025',42,k=4)
    assert len(s.choose(rows[:2],0,0,'uniform',42))==2


def test_common_graph_pool_and_fixed_gaussian_ranking(monkeypatch):
    _,s=modules(monkeypatch)
    rows=[dict(candidate_id=i,smiles=smiles,prediction=[y],independent_prediction=[0],domain={'in_domain':ad})
          for i,(smiles,y,ad) in enumerate([('C',.3,True),('C',.0,True),('CC',.2,True),('CCC',.1,False)])]
    assert [r['candidate_id'] for r in s.unique_eligible(rows)]==[0,2]
    assert [r['candidate_id'] for r in s.choose(rows,0,0,'top4',1)]==[2,0]
    e=np.array([.5,-.1,.3]);assert np.array_equal(np.argsort(abs(e)),np.argsort(-np.exp(-.5*(e/.25)**2)))


def test_same_pool_topk_minimizes_dpa_absolute_error(monkeypatch):
    _,s=modules(monkeypatch)
    rows=[dict(candidate_id=i,smiles=str(i),prediction=[v],independent_prediction=[v],domain={'in_domain':True}) for i,v in enumerate([-.7,.4,-.1,.2,1.,.9,.6])]
    best=np.mean([abs(r['prediction'][0]) for r in s.choose(rows,0,0,'top4',0)])
    for method in ['gaussian025','gaussian050','uniform']:
        for seed in range(20):
            chosen=s.choose(rows,0,0,method,seed)
            assert len(chosen)==len({r['smiles'] for r in chosen})==4
            assert np.mean([abs(r['prediction'][0]) for r in chosen])>=best-1e-12
