import torch
from loom.generator.octree_alignment import categorical_policy_loss
from loom.generator.alignment_diagnostics import categorical_kl_loss


def test_kl_probe_matches_training_normalization_for_unequal_trajectory_lengths():
    torch.manual_seed(42)
    raw=torch.randn(3,4,requires_grad=True)
    labels=torch.tensor([1,2,3]);owners=torch.tensor([0,0,1]);adv=torch.tensor([1.,-1.])
    current=[(raw.log_softmax(-1),labels,owners)]
    old=[(current[0][0].detach(),labels,owners)]
    reference=[(torch.randn(3,4).log_softmax(-1),labels,owners)]
    joint=categorical_policy_loss(current,old,reference,adv,beta=.2)
    policy=categorical_policy_loss(current,old,None,adv,beta=0)
    kl=.2*categorical_kl_loss(current,reference,adv)
    torch.testing.assert_close(joint,policy+kl)
    first=torch.autograd.grad(joint,raw,retain_graph=True)[0]
    second=torch.autograd.grad(policy+kl,raw)[0]
    torch.testing.assert_close(first,second,atol=1e-7,rtol=1e-6)
