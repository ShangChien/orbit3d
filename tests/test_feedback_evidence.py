import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('feedback_export',Path(__file__).parents[1]/'scripts/export_feedback_evidence.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_supply_and_conditional_hit_accounting_reconstruct_total_yield_difference():
    control={'cells':{'50':{'proposals':100,'terminal_selected':50,'hits':10}}}
    treatment={'cells':{'50':{'proposals':100,'terminal_selected':60,'hits':24}}}
    result=module.decompose_yield(control,treatment)['mean']
    assert result['total_delta']==pytest.approx(.14)
    assert result['supply_contribution']==pytest.approx(.03)
    assert result['conditional_hit_contribution']==pytest.approx(.11)
    assert result['supply_contribution']+result['conditional_hit_contribution']==pytest.approx(result['total_delta'])
