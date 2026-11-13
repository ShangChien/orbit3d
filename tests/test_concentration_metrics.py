"""Accuracy and spread differ, and repeated graphs cannot fill a selection quota."""
import importlib
from pathlib import Path
import numpy as np


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    return importlib.import_module('summarize_concentration_scan')


def test_exact_mean_does_not_hide_molecular_error(monkeypatch):
    m=module(monkeypatch)
    rows=[{'prediction':[v],'domain':{'in_domain':True}} for v in [-2.,2.]]
    stats=m.error_metrics(rows,0,0.,'prediction')
    assert stats['bias']==0 and stats['mae']==2 and stats['std']==2
    assert stats['hits025']==0
    assert np.isclose(stats['rmse']**2,stats['bias']**2+stats['std']**2)
    assert m.error_metrics([{'valid':False}],0,0.,'prediction')['evaluated']==0


def test_selection_deduplicates_and_excludes_ood(monkeypatch):
    m=module(monkeypatch)
    rows=[dict(candidate_id=i,smiles=s,prediction=[v],domain={'in_domain':d})
          for i,(s,v,d) in enumerate([('C',0.,True),('C',.1,True),('CC',0.,False),('CCC',.2,True)])]
    assert [r['candidate_id'] for r in m.select(rows,0,0.)]==[0,3]
