import torch
import torch.nn.functional as F

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False, **kwargs):
    # q: [B, Sq, H, D], k: [B, Sk, H, D], v: [B, Sk, H, D]
    q_t = q.transpose(1, 2)  # [B, H, Sq, D]
    k_t = k.transpose(1, 2)  # [B, H, Sk, D]
    v_t = v.transpose(1, 2)  # [B, H, Sk, D]
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        dropout_p=dropout_p,
        is_causal=causal,
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
    # q, k, v: [Total_T, H, D], cu_seqlens_q: [B + 1]
    bsz = cu_seqlens_q.shape[0] - 1
    if bsz == 1:
        q_b = q.unsqueeze(0).transpose(1, 2)
        k_b = k.unsqueeze(0).transpose(1, 2)
        v_b = v.unsqueeze(0).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_b, k_b, v_b,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        return out.transpose(1, 2).squeeze(0).contiguous()

    lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    q_splits = torch.split(q, lens, dim=0)
    k_splits = torch.split(k, lens, dim=0)
    v_splits = torch.split(v, lens, dim=0)

    outs = []
    for qi, ki, vi in zip(q_splits, k_splits, v_splits):
        # qi: [L, H, D] -> [1, H, L, D]
        out_i = F.scaled_dot_product_attention(
            qi.unsqueeze(0).transpose(1, 2),
            ki.unsqueeze(0).transpose(1, 2),
            vi.unsqueeze(0).transpose(1, 2),
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        outs.append(out_i.transpose(1, 2).squeeze(0))
    return torch.cat(outs, dim=0).contiguous()
