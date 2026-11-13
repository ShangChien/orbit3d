import sys
from pathlib import Path
import torch
from orbit3d.generator.octree_model import PointConditioner, GaussianConditioner
from orbit3d.generator.gaussian_conditions import sample_gaussian_conditions, reciprocal_pairs
from orbit3d.generator.condition_contrast import condition_margin_loss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from migrate_gaussian_conditioner import expand_point_weights


def test_migration_preserves_values_and_learns_width():
    torch.manual_seed(42)
    old=PointConditioner(16,2,2);new=GaussianConditioner(16,2,2)
    prefix='interval_conditioner.'
    weights=expand_point_weights({prefix+k:v for k,v in old.state_dict().items()})
    new.load_state_dict({k[len(prefix):]:v for k,v in weights.items()},strict=True)
    mu=torch.tensor([[.2,99.],[.4,-99.]])
    mask=torch.tensor([[True,False],[True,False]])
    narrow=torch.full_like(mu,.1);wide=torch.full_like(mu,.5)
    torch.testing.assert_close(old(mu,mu,mask),new(mu,narrow,mask))
    torch.testing.assert_close(new(mu,narrow,mask),new(mu,wide,mask))
    new(mu,narrow,mask).square().sum().backward()
    assert new.prop_encoders[0][0].weight.grad[:,1].abs().sum()>0
    with torch.no_grad():new.prop_encoders[0][0].weight[:,1].add_(torch.randn(16)*.1)
    assert not torch.allclose(new(mu,narrow,mask),new(mu,wide,mask))
    assert torch.count_nonzero(new(mu,narrow,torch.zeros_like(mask)))==0


def test_sampling_noise_tracks_requested_width_not_output_variance():
    torch.manual_seed(42)
    target=torch.zeros(10000,2)
    mu,sigma,mask=sample_gaussian_conditions(target,torch.tensor([2.,1.],dtype=torch.float64),torch.tensor([True,False]),(.25,.25),dropout=0)
    torch.testing.assert_close(sigma[0],torch.tensor([.125,.25]))
    assert mu.dtype == sigma.dtype == target.dtype
    z=(mu-target)/sigma
    assert z.mean().abs()<.03 and abs(float(z.std())-1)<.03
    assert mask[:,0].all() and not mask[:,1].any()


def test_reciprocal_preferences_reject_overlapping_or_reversed_requests():
    target=torch.tensor([[-1.],[1.]])
    sigma=torch.full_like(target,.25);mask=torch.ones(2,1,2,dtype=torch.bool)
    partner,eligible=reciprocal_pairs(target,target,sigma,mask)
    assert partner.tolist()==[1,0] and eligible.all()
    _,wrong=reciprocal_pairs(target,target.flip(0),sigma,mask)
    assert not wrong.any()
    _,overlap=reciprocal_pairs(target,torch.zeros_like(target),sigma,mask)
    assert not overlap.any()
    positive=torch.tensor([.5,.6],requires_grad=True);negative=torch.tensor([.4,.5],requires_grad=True)
    condition_margin_loss(positive,negative,eligible).backward()
    assert (positive.grad>0).all() and (negative.grad<0).all()
