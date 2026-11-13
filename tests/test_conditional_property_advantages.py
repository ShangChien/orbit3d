import torch
from orbit3d.generator.octree_alignment import group_relative_advantages


def test_joint_mode_preserves_existing_normalization_exactly():
    reward=torch.tensor([.9,.1,-1.,-1.])
    actual,parts=group_relative_advantages(reward,reward>=0)
    assert torch.equal(actual,(reward-reward.mean())/reward.std(unbiased=False).clamp_min(1e-6))
    assert parts is None


def test_poor_eligible_candidate_can_receive_negative_advantage_without_increasing_scale():
    reward=torch.tensor([.9,.1]+[-1.]*14)
    adv,(eligibility,prop)=group_relative_advantages(reward,reward>=0,'conditional_property_v1')
    assert adv[0]>0 and adv[1]<0
    assert torch.all(prop[2:]==0) and torch.all(eligibility[2:]<0)
    torch.testing.assert_close(adv,eligibility+prop)
    torch.testing.assert_close(adv.mean(),torch.tensor(0.),atol=1e-7,rtol=0)
    assert adv.std(unbiased=False)<=1.000001


def test_zero_one_or_tied_eligible_candidates_reduce_to_binary_eligibility():
    for values in [[-1.]*4,[.4,-1.,-1.,-1.],[.4,.4,-1.,-1.]]:
        reward=torch.tensor(values);mask=reward>=0
        actual,parts=group_relative_advantages(reward,mask,'conditional_property_v1')
        binary=torch.where(mask,1.,-1.)
        expected,_=group_relative_advantages(binary,mask)
        torch.testing.assert_close(actual,expected)
        assert torch.count_nonzero(parts[1])==0


def test_tiny_all_eligible_property_differences_are_not_amplified_to_unit_variance():
    reward=torch.tensor([.50001,.49999,.5,.5])
    adv,parts=group_relative_advantages(reward,torch.ones(4,dtype=torch.bool),'conditional_property_v1')
    assert 0<float(adv.std(unbiased=False))<.001
    assert torch.count_nonzero(parts[0])==0


def test_permutation_does_not_change_candidate_advantages():
    reward=torch.tensor([.6,-1.,.1,.9,-1.]);mask=reward>=0
    expected,_=group_relative_advantages(reward,mask,'conditional_property_v1')
    permutation=torch.tensor([3,0,4,1,2])
    actual,_=group_relative_advantages(reward[permutation],mask[permutation],'conditional_property_v1')
    torch.testing.assert_close(actual,expected[permutation])
