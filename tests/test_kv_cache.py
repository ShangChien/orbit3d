import pytest
import torch
from orbit3d.vendor.uni3dar.models.attention import Attention
from orbit3d.vendor.uni3dar.models.kv_cache import compact_kv
from orbit3d.vendor.flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func


@pytest.mark.parametrize('mode', ['concat', 'buffer'])
def test_cached_causal_attention_matches_full_prefix_with_growth_and_compaction(mode):
    torch.manual_seed(42)
    layer = Attention(32, 4, causal=True).eval()
    torch.nn.init.normal_(layer.out_proj.weight, std=.1)
    x = torch.randn(3, 153, 32)
    cache = {'step': 1, 'mode': mode}
    with torch.no_grad():
        expected = layer(x.reshape(-1, 32), torch.tensor([0, 153, 306, 459]), 153).reshape(3, 153, 32)
        # The long chunk exercises the cached FlashAttention/SDPA branch.
        first = layer(x[:, :9], None, None, cache)
        compact_kv(cache, torch.tensor([True, False, True]))
        second = layer(x[[0, 2], 9:138], None, None, cache)
        pointer = cache['k'].data_ptr()
        third = layer(x[[0, 2], 138:], None, None, cache)
    torch.testing.assert_close(first, expected[:, :9], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(torch.cat([second, third], 1), expected[[0, 2], 9:], atol=2e-6, rtol=1e-5)
    assert cache['k'].shape == (2, 4, 153, 8)
    if mode == 'buffer':
        # The third append fits the doubled capacity and reuses storage.
        assert cache['k'].data_ptr() == pointer


def test_unequal_packed_attention_uses_suffix_causality_without_cross_sequence_leakage():
    torch.manual_seed(7)
    q, k, v = torch.randn(5, 2, 8), torch.randn(11, 2, 8), torch.randn(11, 2, 8)
    actual = flash_attn_varlen_func(q, k, v, torch.tensor([0, 2, 5]), torch.tensor([0, 4, 11]), 3, 7, causal=True)
    expected = torch.cat([flash_attn_func(q[:2][None], k[:4][None], v[:4][None], causal=True)[0],
                          flash_attn_func(q[2:][None], k[4:][None], v[4:][None], causal=True)[0]])
    torch.testing.assert_close(actual, expected)
    # A one-query suffix must see all preceding keys, not just the first key.
    one = flash_attn_func(torch.zeros(1, 1, 1, 1), torch.zeros(1, 4, 1, 1),
                          torch.arange(4.).reshape(1, 4, 1, 1), causal=True)
    assert one.item() == pytest.approx(1.5)
