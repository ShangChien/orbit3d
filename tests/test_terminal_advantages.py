import torch
from orbit3d.generator.terminal_feedback import terminal_advantages


def test_one_accepted_endpoint_cannot_supply_property_comparison():
    structural=torch.tensor([True,True,True,False]);physical=torch.tensor([True,False,False,False])
    a,c=terminal_advantages(structural,physical,physical,torch.tensor([.8,0.,0.,0.]),'property')
    assert not c['property'].any()
    assert c['physical'][0]>0 and c['physical'][1]<0 and c['physical'][3]==0
    assert a.std(unbiased=False)<=1.00001


def test_bad_eligible_property_can_receive_negative_property_signal():
    mask=torch.ones(4,dtype=torch.bool)
    a,c=terminal_advantages(mask,mask,mask,torch.tensor([.1,.2,.8,.9]),'property')
    assert not c['structure'].any() and not c['physical'].any()
    assert a[0]<0 and a[-1]>0 and abs(float(a.mean()))<1e-6


def test_structure_only_is_independent_of_target_quality():
    s=torch.tensor([True,False,True,False]);p=torch.tensor([True,False,False,False])
    a,_=terminal_advantages(s,p,p,torch.zeros(4),'structure')
    b,_=terminal_advantages(s,p,p,torch.ones(4),'structure')
    torch.testing.assert_close(a,b)
