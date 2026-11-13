import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('reward_attribution',Path(__file__).parents[1]/'scripts/summarize_reward_attribution.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def score(eligible,hits):return {'attempted':100,'evaluated':eligible,'hits':hits,'mae':.5 if eligible else None}


def test_more_supply_without_better_conditional_hits():
    r=module.decompose(score(20,5),score(40,10))
    assert r['supply_component']==pytest.approx(.05)
    assert r['conditional_hit_component']==0


def test_better_conditional_hits_without_more_supply():
    r=module.decompose(score(20,5),score(20,10))
    assert r['conditional_hit_component']==pytest.approx(.05)
    assert r['supply_component']==0


def test_mixed_change_has_exact_symmetric_decomposition():
    r=module.decompose(score(20,5),score(40,16))
    assert r['supply_component']+r['conditional_hit_component']==pytest.approx(r['yield_delta'])


def test_zero_coverage_does_not_impute_a_conditional_success_rate():
    r=module.decompose(score(0,0),score(20,5))
    assert r['yield_delta']==.05 and r['supply_component'] is None
