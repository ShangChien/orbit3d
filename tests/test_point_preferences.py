import torch
import pytest
from orbit3d.generator.point_preferences import nearby_size_matched_pairs, bidirectional_preference_loss, nearest_condition_candidates, multi_condition_loss


def test_pairing_requires_size_and_nearby_labels_and_reverses():
    partner,eligible=nearby_size_matched_pairs(torch.tensor([-7.,-6.5,-6.6,-5.]),torch.tensor([20,23,60,21]),torch.ones(4,dtype=torch.bool))
    assert partner.tolist()==[1,0,2,3] and eligible.tolist()==[True,True,False,False]


def test_molecule_and_condition_preferences_are_both_necessary():
    partner=torch.tensor([1,0]);enabled=torch.tensor([True,True]);reference=torch.zeros(2)
    good,_=bidirectional_preference_loss(torch.tensor([-.1,-.1]),torch.tensor([.1,.1]),reference,reference,partner,enabled)
    # Both molecules prefer their own condition, but a molecule-wide score shift
    # can still make one molecule preferred for both requested values.
    shifted,stats=bidirectional_preference_loss(torch.tensor([-1.1,.9]),torch.tensor([-.9,1.1]),reference,reference,partner,enabled)
    assert good==0 and shifted>0 and stats['condition_loss']==0 and stats['sample_loss']>0


def test_soft_targets_do_not_treat_equal_properties_as_hard_negatives():
    values=torch.tensor([-7.,-7.,-6.,-5.]);ids=nearest_condition_candidates(values)
    assert ids.shape==(4,4) and torch.equal(ids[:,0],torch.arange(4))
    loss=torch.zeros((1,4),requires_grad=True);distance=torch.tensor([[0.,0.,.5,1.]])
    value,_=multi_condition_loss(loss,torch.zeros_like(loss),distance,torch.ones(1,dtype=torch.bool))
    value.backward()
    assert torch.isclose(loss.grad[0,0],loss.grad[0,1])
    assert loss.grad[0,0]>0 and loss.grad[0,3]<0


@pytest.mark.parametrize('method', ['bidirectional', 'bidirectional_softmax', 'multi_condition'])
def test_objective_freezes_reference_but_backpropagates_to_conditioner(method):
    from orbit3d.generator.point_preferences import PointPreferenceObjective
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.interval_conditioner = torch.nn.Linear(1, 1, bias=False)
            self.dropout = torch.nn.Dropout(.9)

        def compute_sft_and_understanding_loss(self, packed, target, lower, upper, mask, **kwargs):
            assert not self.training
            score = (self.interval_conditioner(self.dropout(lower))*packed).sum(-1)
            return score.mean(), {'per_molecule_generation_loss': score}

    model = Model().train()
    objective = PointPreferenceObjective(model, method, 0, [1.])
    y = torch.tensor([[-7.], [-6.5], [-6.], [-5.5]])
    loss, stats = objective(model, y, y, y, y, torch.ones(4, 1, 2, dtype=torch.bool), torch.tensor([20]*4))
    loss.backward()
    assert model.training and not objective.reference.training
    assert model.interval_conditioner.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in objective.reference.parameters())
    assert stats['preference_eligible'] == 4


def test_two_way_softmax_keeps_distinct_channels_inside_the_old_linear_hinge_region():
    from orbit3d.generator.point_preferences import bidirectional_softmax_loss
    positive=torch.tensor([-.01,.01],requires_grad=True)
    negative=torch.tensor([-.005,.005],requires_grad=True)
    zero=torch.zeros(2);partner=torch.tensor([1,0]);enabled=torch.ones(2,dtype=torch.bool)
    hinge,stats=bidirectional_preference_loss(positive,negative,zero,zero,partner,enabled)
    assert abs(stats['condition_loss']-stats['sample_loss'])<1e-8
    smooth,stats=bidirectional_softmax_loss(positive,negative,zero,zero,partner,enabled)
    assert stats['sample_loss']>stats['condition_loss']
    condition_only=.02*torch.nn.functional.softplus((positive-negative)/.02).mean()
    combined_grad=torch.autograd.grad(smooth,positive,retain_graph=True)[0]
    condition_grad=torch.autograd.grad(condition_only,positive)[0]
    assert not torch.allclose(combined_grad,condition_grad)
