import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('ranking',Path(__file__).parents[1]/'scripts/summarize_calibrated_ranking.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_quantum_failure_does_not_cause_ranking_replacement():
    rows=[dict(index=0,candidate_id=0,baseline=[-6.,0.],truth=None),dict(index=1,candidate_id=1,baseline=[-6.1,0.],truth=[-6.,0.])]
    r=module.select(rows,'baseline',0,-6.,1)
    assert r['selected_indices']==[0] and r['hits']==0 and r['quantum_failed_selected']==[0]


def test_short_pool_preserves_requested_slots_and_raw_denominator():
    r=module.select([dict(index=0,candidate_id=0,baseline=[-6.,0.],truth=[-6.,0.])],'baseline',0,-6.,4)
    total=module.aggregate([{**r,'quantile':50,'inference_seed':2091}])
    assert total['delivery_coverage']==.25
    assert total['hit_rate_requested_slots']==.25
    assert total['retained_hit_yield_all_proposals_equal_goal']==1/64


def test_point_distance_rank_fidelity_uses_all_eligible_endpoints():
    rows=[dict(index=i,candidate_id=i,baseline=[-6.+p,0.],truth=[-6.+t,0.]) for i,(p,t) in enumerate([(0.,.4),(.2,.2),(.4,0.)])]
    r=module.select(rows,'baseline',0,-6.,1)
    assert r['preselection_rank_pairs']==3
    assert r['preselection_distance_spearman']==-1
    assert r['hits']==0
