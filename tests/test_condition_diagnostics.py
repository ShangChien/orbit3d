import importlib.util
from pathlib import Path
import torch
import pytest

spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/diagnose_octree_conditioning.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_point_audit_keeps_exact_value_and_single_active_target():
    value=torch.tensor([[-1.,2.],[.5,3.]])
    lower,upper,mask=module.condition_inputs(4,value,['homo','lumo'],['homo'])
    torch.testing.assert_close(lower,value);torch.testing.assert_close(upper,value)
    assert mask[:,0].all() and not mask[:,1].any()
    assert not module.condition_inputs(4,value,['homo','lumo'],['homo'],False)[2].any()


def test_interval_audit_preserves_width_and_rejects_unconfigured_gaussian():
    value=torch.zeros(2,2)
    low,high,mask=module.condition_inputs(3,value,['homo','lumo'],['homo','lumo'])
    assert torch.all(low==-.25) and torch.all(high==.25) and mask.all()
    with pytest.raises(ValueError,match='explicit width'):
        module.condition_inputs(5,value,['homo','lumo'],['homo'])
