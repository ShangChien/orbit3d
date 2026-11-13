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


def test_terminal_alignment_updates_original_rollout_not_relaxed_coordinates(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from test_alignment_resume import Policy
    from orbit3d.constraints import TargetPoints
    from orbit3d.generator.octree_alignment import align_octree
    class CheckedPolicy(Policy):
        def trajectory_distributions(self,trajectories,*args,**kwargs):
            assert all(np.count_nonzero(t['positions_np'])==0 for t in trajectories)
            return super().trajectory_distributions(trajectories,*args,**kwargs)
    def instance(t,**kwargs):
        return SimpleNamespace(graph=t['value'],coords=t['positions_np'],symbols=['C'],smiles=str(t['value']))
    monkeypatch.setattr('orbit3d.generator.octree_alignment.trajectory_to_instance',instance)
    def terminal(inst,*args):
        return SimpleNamespace(coords=np.ones((1,3))*9),{'physical_accepted':True,'coordinates_A':[[9.,9.,9.]],'final':{'prediction_eV':[inst.graph]}}
    monkeypatch.setattr('orbit3d.physics.terminal_evaluation.evaluate_terminal',terminal)
    point=TargetPoints.parse({'y':0.},{'y':1.},['y'])
    history=align_octree(CheckedPolicy(),[],None,np.zeros(1),np.ones(1),point.intervals(),None,None,
        steps=1,group_size=4,physics_weight=0,replay_weight=0,points=point,terminal_feedback_level='property')
    rows=history[0]['rollout_records']
    assert all(r['terminal']['coordinates_A']==[[9.,9.,9.]] for r in rows)
    assert all(r['coordinates_A']==[[0.,0.,0.]] for r in rows)
    assert any(r['terminal_advantages']['property']<0 for r in rows)
