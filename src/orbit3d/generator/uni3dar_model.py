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
    Pure-PyTorch `Uni-3DAR` with Dynamic Interval Conditioning (`[low, high, mask]`)
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
    ) -> None:
        super().__init__()
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

        self.blocks = nn.ModuleList(
            [_AdaLNZeroTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.suffix_prop_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
        )

    def forward(
        self, tokens: torch.Tensor, cond_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = tokens.shape
        device = tokens.device
        pos = torch.arange(L, device=device).unsqueeze(0).clamp(max=self.max_seq_len - 1)
        c_emb = self.cond_proj(cond_vec)

        x = self.tok_emb(tokens) + self.pos_emb(pos)
        x[:, 0, :] = x[:, 0, :] + c_emb

        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1
        )
        for blk in self.blocks:
            x = blk(x, c_emb, causal_mask)

        h = self.final_norm(x)
        logits = self.lm_head(h)

        non_pad = (tokens != PAD_TOKEN).float().unsqueeze(-1)
        pooled = (h * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1.0)
        pred_props = self.suffix_prop_head(pooled)
        return logits, pred_props

    @torch.no_grad()
    def sample_autoregressive(
        self,
        cond_vec: torch.Tensor,
        max_atoms: int = 12,
        temperature: float = 0.85,
        top_p: float = 0.92,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = cond_vec.size(0)
        device = cond_vec.device
        seq = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
        total_steps = max_atoms * 4

        for step in range(total_steps):
            logits, _ = self.forward(seq, cond_vec)
            next_logits = logits[:, -1, :] / max(temperature, 1e-4)

            mask = torch.full_like(next_logits, float("-inf"))
            if step % 4 == 0:
                mask[:, ATOM_OFFSET : ATOM_OFFSET + NUM_ATOM_TYPES] = 0.0
            else:
                mask[:, COORD_OFFSET : COORD_OFFSET + NUM_COORD_BINS] = 0.0

            masked_logits = next_logits + mask
            probs = F.softmax(masked_logits, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            sampled_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_tok = sorted_idx.gather(-1, sampled_in_sorted)
            seq = torch.cat([seq, next_tok], dim=1)

        eos_col = torch.full((B, 1), EOS_TOKEN, dtype=torch.long, device=device)
        seq = torch.cat([seq, eos_col], dim=1)
        logits_full, pred_props = self.forward(seq[:, :-1], cond_vec)
        logp = F.log_softmax(logits_full, dim=-1)
        tok_logp = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
        return seq, tok_logp, pred_props


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
    `base_seed * 100000 + mol_id`, guaranteeing 100% reproducibility regardless of batch size!
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
    mol_seed = int(base_seed) * 100000 + int(mol_id)
    torch.manual_seed(mol_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mol_seed)

    try:
        cond_1 = cond_raw.unsqueeze(0).to(device)
        seqs, _, pred_props = model.sample_autoregressive(
            cond_1, temperature=temperature, top_p=top_p
        )
        inst = decode_tokens_to_mol3d(seqs[0].cpu().tolist())
        inst.props = pred_props[0].cpu().numpy()
        mol_rd = mol3d_to_rdkit_mol(inst)
        if mol_rd is not None:
            inst.smiles = mol_to_clean_smiles(mol_rd)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)

    return inst, pred_props[0]
