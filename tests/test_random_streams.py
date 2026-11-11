import pytest
import torch
from orbit3d.generator.random_streams import RowRandomStreams, candidate_seed, inverse_cdf


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
