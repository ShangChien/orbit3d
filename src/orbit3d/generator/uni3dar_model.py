"""
Conditional 3D Autoregressive Generator (`ConditionalUni3DAR`) & Deterministic Sampler.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from .octree_tokenizer import (
    ATOM_OFFSET,
    BOS_TOKEN,
    COORD_OFFSET,
    EOS_TOKEN,
    Mol3DInstance,
    NUM_ATOM_TYPES,
    NUM_COORD_BINS,
    PAD_TOKEN,
    VOCAB_SIZE,
    decode_tokens_to_mol3d,
    mol3d_to_rdkit_mol,
    mol_to_clean_smiles,
)


class _AdaLNZeroTransformerBlock(nn.Module):
    """Transformer decoder block with AdaLN-Zero dynamic interval conditioning."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 6),
        )
        nn.init.zeros_(self.ada_mod[-1].weight)
        nn.init.zeros_(self.ada_mod[-1].bias)

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        mod = self.ada_mod(c_emb).unsqueeze(1)
        g1, b1, a1, g2, b2, a2 = mod.chunk(6, dim=-1)

        h1 = self.norm1(x) * (1.0 + g1) + b1
        attn_out, _ = self.attn(h1, h1, h1, attn_mask=attn_mask, need_weights=False)
        x = x + (1.0 + torch.tanh(a1)) * attn_out

        h2 = self.norm2(x) * (1.0 + g2) + b2
        x = x + (1.0 + torch.tanh(a2)) * self.mlp(h2)
        return x


class ConditionalUni3DAR(nn.Module):
    """
    Dense-grid research baseline (not upstream Uni-3DAR) with Dynamic Interval Conditioning (`[low, high, mask]`)
    and Suffix 3D Property Understanding Head.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 12,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 256,
        cond_dim: int = 7,
        output_dim: int = 3,
        max_atoms: int = 40,
    ) -> None:
        super().__init__()
        self.backend_id = "dense_grid_v2"
        self.max_atoms = max_atoms
        self.cond_dim = cond_dim
        if 4 * max_atoms + 2 > max_seq_len:
            raise ValueError("Sequence capacity too small for max_atoms")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([_AdaLNZeroTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.suffix_prop_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, tokens: torch.Tensor, cond_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = tokens.shape
        device = tokens.device
        if L > self.max_seq_len:
            raise ValueError("Sequence too long")
        pos = torch.arange(L, device=device).unsqueeze(0)
        c_emb = self.cond_proj(cond_vec)

        x = self.tok_emb(tokens) + self.pos_emb(pos)
        x[:, 0, :] = x[:, 0, :] + c_emb

        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, c_emb, causal_mask)

        h = self.final_norm(x)
        logits = self.lm_head(h)

        non_pad = (tokens != PAD_TOKEN).float().unsqueeze(-1)
        pooled = (h * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1.0)
        pred_props = self.suffix_prop_head(pooled)
        return logits, pred_props

    @torch.no_grad()
    def forward_step(self, token, cond_vec, cache=None):
        """Inference-only KV cache; positions and conditional modulation match forward."""
        if self.training:
            raise ValueError("Incremental decoding requires eval mode")
        B = token.shape[0]
        if cache is None:
            c = self.cond_proj(cond_vec)
            cache = {"position": 0, "condition": c, "blocks": []}
            for block in self.blocks:
                heads = block.attn.num_heads
                shape = (B, heads, self.max_seq_len, self.d_model // heads)
                cache["blocks"].append(
                    {
                        "mod": block.ada_mod(c).unsqueeze(1).chunk(6, dim=-1),
                        "k": torch.empty(shape, device=token.device, dtype=c.dtype),
                        "v": torch.empty(shape, device=token.device, dtype=c.dtype),
                    }
                )
        position = cache["position"]
        if position >= self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        x = self.tok_emb(token.reshape(B, 1)) + self.pos_emb.weight[position][None, None]
        if position == 0:
            x = x + cache["condition"][:, None]
        for block, state in zip(self.blocks, cache["blocks"]):
            g1, b1, a1, g2, b2, a2 = state["mod"]
            h = block.norm1(x) * (1 + g1) + b1
            q, k, v = F.linear(h, block.attn.in_proj_weight, block.attn.in_proj_bias).chunk(3, -1)
            heads = block.attn.num_heads
            dim = self.d_model // heads
            q = q.view(B, 1, heads, dim).transpose(1, 2)
            state["k"][:, :, position : position + 1] = k.view(B, 1, heads, dim).transpose(1, 2)
            state["v"][:, :, position : position + 1] = v.view(B, 1, heads, dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(
                q,
                state["k"][:, :, : position + 1],
                state["v"][:, :, : position + 1],
                dropout_p=0.0,
                is_causal=False,
            )
            attn = block.attn.out_proj(attn.transpose(1, 2).reshape(B, 1, self.d_model))
            x = x + (1 + torch.tanh(a1)) * attn
            x = x + (1 + torch.tanh(a2)) * block.mlp(block.norm2(x) * (1 + g2) + b2)
        cache["position"] += 1
        return self.lm_head(self.final_norm(x)), cache

    def constrained_logits(self, logits, temperature=1.0, start_step=0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        out = torch.full_like(logits, -torch.inf)
        for index in range(logits.shape[1]):
            step = index + start_step
            if step == self.max_atoms * 4:
                out[:, index, EOS_TOKEN] = logits[:, index, EOS_TOKEN]
            elif step % 4 == 0:
                out[:, index, ATOM_OFFSET:COORD_OFFSET] = logits[:, index, ATOM_OFFSET:COORD_OFFSET]
                if step >= 4:
                    out[:, index, EOS_TOKEN] = logits[:, index, EOS_TOKEN]
            else:
                out[:, index, COORD_OFFSET:] = logits[:, index, COORD_OFFSET:]
        return out / temperature

    @torch.no_grad()
    def sample_autoregressive(self, cond_vec, max_atoms=None, temperature=0.85, top_p=0.92, use_cache=True):
        if max_atoms is not None and max_atoms != self.max_atoms:
            raise ValueError("Sampling length must match saved model specification")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0,1]")
        B = cond_vec.shape[0]
        device = cond_vec.device
        seq = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
        ended = torch.zeros(B, dtype=torch.bool, device=device)
        cache = None
        use_cache = use_cache and not self.training
        for step in range(self.max_atoms * 4 + 1):
            if use_cache:
                logits, cache = self.forward_step(seq[:, -1], cond_vec, cache)
                probs = self.constrained_logits(logits, temperature, start_step=step)[:, -1].softmax(-1)
            else:
                logits, _ = self(seq, cond_vec)
                probs = self.constrained_logits(logits, temperature)[:, -1].softmax(-1)
            sorted_probs, idx = probs.sort(descending=True, stable=True)
            remove = sorted_probs.cumsum(-1) - sorted_probs >= top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0)
            sampled = torch.multinomial(sorted_probs, 1)
            token = idx.gather(-1, sampled).squeeze(-1)
            token = torch.where(ended, torch.full_like(token, PAD_TOKEN), token)
            seq = torch.cat([seq, token[:, None]], 1)
            ended |= token == EOS_TOKEN
            if ended.all():
                break
        logits, _ = self(seq[:, :-1], cond_vec)
        logp = self.constrained_logits(logits, temperature).log_softmax(-1)
        selected = logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
        selected = selected.masked_fill(seq[:, 1:] == PAD_TOKEN, 0)
        _, props = self(seq, torch.zeros_like(cond_vec))
        return seq, selected, props


@torch.no_grad()
def sample_molecule_deterministic_by_id(
    model: ConditionalUni3DAR,
    cond_raw: torch.Tensor,
    mol_id: int,
    base_seed: int = 2026,
    temperature: float = 0.85,
    top_p: float = 0.92,
    device: str = "cuda:0",
) -> Tuple[Mol3DInstance, torch.Tensor]:
    """
    Generates molecule `#mol_id` using an isolated RNG stream seeded strictly by
    `base_seed * 100000 + mol_id`, isolating RNG state. Numerical reproducibility depends on backend and batching.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mol_seed = int(base_seed) * 100000 + int(mol_id)
    torch.manual_seed(mol_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mol_seed)

    try:
        cond_1 = cond_raw.unsqueeze(0).to(device)
        seqs, _, pred_props = model.sample_autoregressive(cond_1, temperature=temperature, top_p=top_p)
        inst = decode_tokens_to_mol3d(seqs[0].cpu().tolist())
        inst.props = pred_props[0].cpu().numpy()
        mol_rd = mol3d_to_rdkit_mol(inst)
        if mol_rd is not None:
            inst.smiles = mol_to_clean_smiles(mol_rd)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    return inst, pred_props[0]
