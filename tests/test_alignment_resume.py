import copy
import pytest
from types import SimpleNamespace
import numpy as np
import torch
from orbit3d.constraints import TargetPoints
from orbit3d.generator.octree_alignment import align_octree


class Policy(torch.nn.Module):
    condition_version=4
    args=SimpleNamespace(allow_atoms='C')
    def __init__(self):
        super().__init__();self.interval_conditioner=torch.nn.Linear(1,2)
    def generate_conditional_trajectories(self,low,high,mask,**kwargs):
        values=torch.rand(len(low))+torch.arange(len(low))
        return [{'decoder_feat':torch.tensor([[i]]),'atom_symbols':['C'],
                 'positions_np':np.zeros((1,3)),'value':float(value)} for i,value in enumerate(values)]
    def trajectory_distributions(self,trajectories,low,high,mask,**kwargs):
        features=torch.tensor([[t['value']] for t in trajectories])+low
        logits=self.interval_conditioner(features)
        return [(logits.log_softmax(-1),torch.arange(len(low))%2,torch.arange(len(low)))]


@pytest.mark.parametrize("advantage_mode",["joint","conditional_property_v1"])
@pytest.mark.parametrize("reward_type",["inverse_point_distance","eligibility_only"])
def test_resume_preserves_optimizer_reference_rng_and_group_sequence(monkeypatch,tmp_path,reward_type,advantage_mode):
    def instance(t,**kwargs):
        return SimpleNamespace(graph=t['value'],coords=t['positions_np'],symbols=['C'],smiles=str(t['value']))
    monkeypatch.setattr('orbit3d.generator.octree_alignment.trajectory_to_instance',instance)
    oracle=SimpleNamespace(predict_3d_mols=lambda mols:(np.array(mols)[:,None],np.zeros((len(mols),1))))
    ff=SimpleNamespace(atom_type_map=['C'],compute_energy_and_forces=lambda *a,**k:(0.,torch.zeros(1,3)))
    pool=[TargetPoints.parse({'y':value},{'y':1.},['y']) for value in [0.,.5]]
    torch.manual_seed(21);base=Policy();rng=torch.get_rng_state()
    def run(model,reward_override=reward_type,advantage_override=advantage_mode,**kwargs):
        return align_octree(model,[],None,np.zeros(1),np.ones(1),pool[0].intervals(),oracle,ff,
                            steps=4,group_size=4,lr=.03,kl_beta=.2,physics_weight=0,replay_weight=0,
                            points=pool[0],point_pool=pool,point_reward_type=reward_override,point_advantage_mode=advantage_override,**kwargs)
    full=copy.deepcopy(base);full_history=run(full)
    torch.set_rng_state(rng);partial=copy.deepcopy(base);saved={}
    def snapshot(step,model,history,state):
        saved.update(copy.deepcopy({'model_state_dict':model.state_dict(),'grpo_history':history,
                                    'alignment_training_state':state}))
    run(partial,stop_after=2,checkpoint_callback=snapshot)
    assert saved['alignment_training_state']['completed_groups']==2
    torch.save(saved,tmp_path/'resume.pt')
    saved=torch.load(tmp_path/'resume.pt',weights_only=False)
    torch.rand(100)  # A fresh process/model may have consumed unrelated randomness.
    resumed=copy.deepcopy(base)  # The reference remains the original SFT policy.
    resumed_history=run(resumed,resume_checkpoint=saved)
    for key,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[key],atol=0,rtol=0)
    assert full_history==resumed_history
    other='eligibility_only' if reward_type=='inverse_point_distance' else 'inverse_point_distance'
    with pytest.raises(ValueError,match='same GRPO targets, optimizer and reward settings'):
        run(copy.deepcopy(base),reward_override=other,resume_checkpoint=saved)
    other_mode='joint' if advantage_mode=='conditional_property_v1' else 'conditional_property_v1'
    with pytest.raises(ValueError,match='same GRPO targets, optimizer and reward settings'):
        run(copy.deepcopy(base),advantage_override=other_mode,resume_checkpoint=saved)


def test_eligibility_reward_ignores_property_value_but_keeps_evaluation(monkeypatch):
    def instance(t,**kwargs):
        if int(t['decoder_feat'][0,0])==3:raise ValueError('invalid graph')
        return SimpleNamespace(graph=t['value'],coords=t['positions_np'],symbols=['C'],smiles=str(t['value']))
    monkeypatch.setattr('orbit3d.generator.octree_alignment.trajectory_to_instance',instance)
    calls=[]
    def predict(mols):
        calls.extend(mols);return np.array(mols)[:,None]*100,np.zeros((len(mols),1))
    oracle=SimpleNamespace(predict_3d_mols=predict)
    ff=SimpleNamespace(atom_type_map=['C'],compute_energy_and_forces=lambda *a,**k:(0.,torch.zeros(1,3)))
    point=TargetPoints.parse({'y':0.},{'y':1.},['y'])
    history=align_octree(Policy(),[],None,np.zeros(1),np.ones(1),point.intervals(),oracle,ff,
        steps=1,group_size=4,physics_weight=0,replay_weight=0,points=point,point_reward_type='eligibility_only')
    rows=history[0]['rollout_records']
    assert len(calls)==3 and [r['reward'] for r in rows]==[1.,1.,1.,-1.]
    assert len({r['point_distance'] for r in rows[:3]})==3
    assert rows[0]['advantage']==rows[1]['advantage']==rows[2]['advantage']>rows[3]['advantage']
