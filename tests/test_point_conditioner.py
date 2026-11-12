import torch
from orbit3d.generator.octree_model import PointConditioner


def test_only_active_values_affect_condition_and_gradients():
    torch.manual_seed(42)
    model=PointConditioner(16,2,2)
    y=torch.tensor([[.2,.7]],requires_grad=True)
    active=torch.tensor([[[True,True],[False,False]]])
    c=model(y,y,active)
    changed=y.detach().clone();changed[0,1]=1000
    torch.testing.assert_close(c,model(changed,changed,active))
    moved=y.detach().clone();moved[0,0]=1.2
    assert not torch.allclose(c,model(moved,moved,active))
    c.square().sum().backward()
    assert y.grad[0,0].abs()>0
    assert y.grad[0,1]==0
    assert torch.count_nonzero(model(y,y,torch.zeros_like(active)))==0


def test_point_distance_changes_inside_acceptance_window_and_ignores_inactive():
    from orbit3d.constraints import TargetPoints
    points=TargetPoints.parse({'homo':-7.},{'homo':.25},['homo','lumo'])
    assert points.intervals().hit([-7.1,99.])
    assert points.distance([-7.1,99.])>points.distance([-7.,1000.])
    lo,hi,mask=points.arrays([-8.,0.],[2.,1.])
    assert lo.tolist()==[.5,0.]
    assert (lo==hi).all() and mask.tolist()==[[True,True],[False,False]]
