import pytest
import torch
from loom.generator.random_streams import RowRandomStreams, candidate_seed, inverse_cdf


def test_row_streams_survive_reordering_removal_and_neighbor_retries():
    seeds = [candidate_seed(42, i) for i in range(4)]
    together = RowRandomStreams(seeds)
    expected = torch.stack([together.draw([0, 1, 2, 3], "cpu") for _ in range(3)])
    changed = RowRandomStreams(seeds)
    first = changed.draw([3, 1, 0, 2], "cpu")
    torch.testing.assert_close(first, expected[0, [3, 1, 0, 2]], rtol=0, atol=0)
    changed.draw([0, 2], "cpu")
    torch.testing.assert_close(changed.draw([1, 3], "cpu"), expected[1, [1, 3]], rtol=0, atol=0)
    retried = RowRandomStreams([seeds[0], candidate_seed(42, 1, retry=1), seeds[2], seeds[3]])
    actual = retried.draw([0, 1, 2, 3], "cpu")
    torch.testing.assert_close(actual[[0, 2, 3]], expected[0, [0, 2, 3]], rtol=0, atol=0)
    assert actual[1] != expected[0, 1]


def test_inverse_cdf_respects_zero_support_and_tail():
    probs = torch.tensor([[0.0, 0.3, 0.7, 0.0]]).repeat(3, 1)
    assert inverse_cdf(probs, torch.tensor([0.0, 0.2, 1.0 - 1e-12], dtype=torch.float64)).tolist() == [
        1,
        1,
        2,
    ]
    with pytest.raises(ValueError, match="support"):
        inverse_cdf(torch.zeros(1, 4), torch.tensor([0.5]))


@pytest.mark.parametrize('terminal', [False, True])
def test_atom_sampling_masks_forbidden_logits_before_softmax(terminal):
    from types import SimpleNamespace
    from loom.vendor.uni3dar.models.uni3dar_sampler import Uni3DARSampler
    from loom.generator.electronic_constraints import mask_terminal_element_logits
    class Head:
        def inference(self, logits, samplers, **kwargs):
            return samplers[0](logits)
    model=SimpleNamespace(num_atom=4,num_tree=1,atom_null_id=3,dictionary={'H':0,'C':1},
                          args=SimpleNamespace(allow_atoms='H,C',rank_by='all',data_type='molecule'),
                          atom_heads=[Head()],atom_temperature=[.3],xyz_temperature=[.25])
    funcs,_=Uni3DARSampler.get_atom_predict_funcs(model,2,torch.device('cpu'),None,{},
                    sample_uniforms=lambda rows:torch.tensor([.1,.8],dtype=torch.float64),
                    atom_logit_mask=(lambda logits, rows:mask_terminal_element_logits(logits,torch.zeros(2,dtype=torch.long),
                                      torch.ones(2,dtype=torch.bool),0,torch.tensor([1,0,0,0]))) if terminal else None)
    # A forbidden class can dominate the unmasked softmax; permitted classes
    # must retain their relative distribution instead of underflowing to zero.
    logits=torch.tensor([[20.,-40.,0.,-40.]] if terminal else [[0.,0.,1000.,0.]]).repeat(2,1)
    chosen,scores=funcs[0](logits,torch.arange(2))
    assert chosen.tolist()==([1,3] if terminal else [0,3])
    assert torch.isfinite(scores).all()


def test_top_p_one_preserves_full_categorical_distribution():
    from loom.vendor.uni3dar.models.uni3dar_sampler import softmax_sampling
    generator=torch.Generator().manual_seed(7)
    logits=torch.randn(16,602,generator=generator)*5
    observed=[]
    def capture(probs):
        observed.append(probs.clone())
        return probs
    softmax_sampling(logits,prob_proc_func=capture,
                     uniforms=torch.full((16,),.5,dtype=torch.float64))
    torch.testing.assert_close(observed[0],logits.softmax(-1),atol=0,rtol=0)
