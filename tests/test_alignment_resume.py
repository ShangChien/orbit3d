import copy
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


def test_resume_preserves_optimizer_reference_rng_and_group_sequence(monkeypatch):
    def instance(t,**kwargs):
        return SimpleNamespace(graph=t['value'],coords=t['positions_np'],symbols=['C'],smiles=str(t['value']))
    monkeypatch.setattr('orbit3d.generator.octree_alignment.trajectory_to_instance',instance)
    oracle=SimpleNamespace(predict_3d_mols=lambda mols:(np.array(mols)[:,None],np.zeros((len(mols),1))))
    ff=SimpleNamespace(atom_type_map=['C'],compute_energy_and_forces=lambda *a,**k:(0.,torch.zeros(1,3)))
    pool=[TargetPoints.parse({'y':value},{'y':1.},['y']) for value in [0.,.5]]
    torch.manual_seed(21);base=Policy();rng=torch.get_rng_state()
    def run(model,**kwargs):
        return align_octree(model,[],None,np.zeros(1),np.ones(1),pool[0].intervals(),oracle,ff,
                            steps=4,group_size=4,lr=.03,kl_beta=.2,physics_weight=0,replay_weight=0,
                            points=pool[0],point_pool=pool,**kwargs)
    full=copy.deepcopy(base);full_history=run(full)
    torch.set_rng_state(rng);partial=copy.deepcopy(base);saved={}
    def snapshot(step,model,history,state):
        saved.update(copy.deepcopy({'model_state_dict':model.state_dict(),'grpo_history':history,
                                    'alignment_training_state':state}))
    run(partial,stop_after=2,checkpoint_callback=snapshot)
    assert saved['alignment_training_state']['completed_groups']==2
    torch.rand(100)  # A fresh process/model may have consumed unrelated randomness.
    resumed=copy.deepcopy(base)  # The reference remains the original SFT policy.
    resumed_history=run(resumed,resume_checkpoint=saved)
    for key,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[key],atol=0,rtol=0)
    assert full_history==resumed_history
