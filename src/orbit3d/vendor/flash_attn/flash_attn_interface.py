import torch
import torch.nn.functional as F

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False, **kwargs):
    # q: [B, Sq, H, D], k: [B, Sk, H, D], v: [B, Sk, H, D]
    q_t = q.transpose(1, 2)  # [B, H, Sq, D]
    k_t = k.transpose(1, 2)  # [B, H, Sk, D]
    v_t = v.transpose(1, 2)  # [B, H, Sk, D]
    # FlashAttention's cached causal queries align with the suffix of K/V.
    # PyTorch's bare is_causal uses the upper-left corner for unequal lengths.
    mask = None
    if causal and q.shape[1] != k.shape[1]:
        query_index = torch.arange(q.shape[1], device=q.device)[:, None] + k.shape[1] - q.shape[1]
        mask = torch.arange(k.shape[1], device=q.device)[None, :] <= query_index
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=causal and mask is None,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous()

def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    deterministic=False,
    **kwargs
):
    # Packed variable-length sequences have independent Q and K boundaries.
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
    if len(q_lens) != len(k_lens):
        raise ValueError('Packed Q and K must contain the same number of sequences')
    outs = []
    for qi, ki, vi in zip(torch.split(q, q_lens), torch.split(k, k_lens), torch.split(v, k_lens)):
        out = flash_attn_func(qi[None], ki[None], vi[None], dropout_p=dropout_p,
                              softmax_scale=softmax_scale, causal=causal, deterministic=deterministic)
        outs.append(out[0])
    return torch.cat(outs).contiguous()
