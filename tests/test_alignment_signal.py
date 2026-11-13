import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('alignment_signal',Path(__file__).parents[1]/'scripts/analyze_alignment_signal.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def history(rewards,eligible):
    return [{'step':1,'gradient_norm':0.,'parameter_update_l2':0.,'policy_kl_before_last_update':0.,
             'rollout_records':[{'reward':r,'evaluation_status':'completed' if ok else 'failed'} for r,ok in zip(rewards,eligible)]}]


def test_binary_rewards_have_no_within_category_signal():
    r=module.analyze(history([1,1,-1,-1],[True,True,False,False]))
    assert r['within_category_advantage_fraction']==0 and r['two_plus_eligible_groups']==1


def test_single_eligible_candidate_has_no_property_comparison():
    r=module.analyze(history([.2,-1,-1,-1],[True,False,False,False]))
    assert r['within_category_advantage_fraction']==pytest.approx(0) and r['one_eligible_groups']==1


def test_all_eligible_varied_rewards_are_pure_property_signal():
    r=module.analyze(history([.1,.5,.9],[True,True,True]))
    assert r['within_category_advantage_fraction']==pytest.approx(1)
    assert r['eligible_negative_advantages']==1
