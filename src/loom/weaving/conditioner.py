"""High-capacity property conditioner combining Gaussian RBF + Fourier expansion,
multi-token attention prefixes, and unclamped AdaLN-Zero modulation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class LoomConditioner(nn.Module):
    """High-bandwidth continuous scalar / multi-property conditioner for 3D generators.

    Addresses the three bottlenecks identified in historical point-conditioning:
      1. Replaces bare `Linear(1, D)` with multi-scale Gaussian RBF + log-spaced Fourier features.
      2. Injects `num_prefix_tokens` dedicated condition KV tokens at the start of the causal sequence
         (plus zero-initialized root residual when `num_prefix_tokens == 0`).
      3. Uses DiT-style zero-initialized AdaLN-Zero without the `0.08 * tanh` ceiling.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        num_props: int = 2,
        num_layers: int = 12,
        num_rbf: int = 32,
        rbf_min: float = -4.5,
        rbf_max: float = 4.5,
        num_freqs: int = 16,
        num_prefix_tokens: int = 0,
        modulation_cap: float = 4.0,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_props = int(num_props)
        self.num_layers = int(num_layers)
        self.num_rbf = int(num_rbf)
        self.num_freqs = int(num_freqs)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.modulation_cap = float(modulation_cap)

        centers = torch.linspace(rbf_min, rbf_max, num_rbf, dtype=torch.float32)
        spacing = (rbf_max - rbf_min) / max(num_rbf - 1, 1)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = 1.0 / (2.0 * (spacing ** 2))

        freqs = (2.0 ** torch.linspace(-1.0, 4.0, num_freqs, dtype=torch.float32)) * math.pi
        self.register_buffer("fourier_freqs", freqs)

        # Per-property feature vector: [val, active_flag] + RBF(num_rbf) + sin/cos(2 * num_freqs)
        in_dim = 2 + num_rbf + 2 * num_freqs
        self.prop_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, emb_dim),
                    nn.LayerNorm(emb_dim),
                    nn.SiLU(),
                    nn.Linear(emb_dim, emb_dim),
                    nn.LayerNorm(emb_dim),
                    nn.SiLU(),
                )
                for _ in range(num_props)
            ]
        )
        self.prop_id_emb = nn.Embedding(num_props, emb_dim)
        self.inactive_prop_emb = nn.Parameter(torch.zeros(num_props, emb_dim))

        self.global_mlp = nn.Sequential(
            nn.Linear(num_props * emb_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
        )

        # Multi-token causal prefix embeddings: [B, P, emb_dim] (zero-initialized when enabled)
        if self.num_prefix_tokens > 0:
            self.prefix_queries = nn.Parameter(torch.randn(self.num_prefix_tokens, emb_dim) * 0.02)
            proj_out = nn.Linear(emb_dim, emb_dim)
            nn.init.zeros_(proj_out.weight)
            nn.init.zeros_(proj_out.bias)
            self.prefix_proj = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.SiLU(),
                proj_out,
            )
            self.null_prefix = nn.Parameter(torch.zeros(self.num_prefix_tokens, emb_dim))
        else:
            self.prefix_queries = None
            self.prefix_proj = None
            self.null_prefix = None

        # Root residual injection (zero-initialized gate)
        self.root_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        nn.init.normal_(self.root_proj.weight, mean=0.0, std=0.02)
        self.root_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

        # Unclamped AdaLN-Zero per-layer & final-norm modulations (exact zero init)
        self.layer_mods = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * emb_dim, bias=True)) for _ in range(num_layers)]
        )
        for mod in self.layer_mods:
            nn.init.zeros_(mod[1].weight)
            nn.init.zeros_(mod[1].bias)

        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * emb_dim, bias=True))
        nn.init.zeros_(self.out_mod[1].weight)
        nn.init.zeros_(self.out_mod[1].bias)

    def _expand_scalar(self, val: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Expand normalized scalar `val` `(B,)` and boolean `active` `(B,)` to `(B, in_dim)`."""
        val_clean = torch.where(active, val.float(), torch.zeros_like(val, dtype=torch.float32))
        diff = val_clean.unsqueeze(-1) - self.rbf_centers.unsqueeze(0)
        rbf = torch.exp(-self.rbf_gamma * diff.square())
        angles = val_clean.unsqueeze(-1) * self.fourier_freqs.unsqueeze(0)
        sin_cos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        act_f = active.float().unsqueeze(-1)
        raw = torch.cat([val_clean.unsqueeze(-1), rbf, sin_cos], dim=-1) * act_f
        return torch.cat([raw, act_f], dim=-1)

    def encode_condition(self, values: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(global_c, has_any)` where `global_c` is `(B, emb_dim)` and `has_any` is `(B, 1)` bool."""
        active = active_mask.any(dim=-1) if active_mask.ndim == 3 else active_mask.bool()
        bsz = values.shape[0]
        device = values.device
        per_prop = []
        for k in range(self.num_props):
            act_k = active[:, k]
            feat_k = self._expand_scalar(values[:, k], act_k)
            enc_k = self.prop_encoders[k](feat_k) + self.prop_id_emb(
                torch.full((bsz,), k, dtype=torch.long, device=device)
            )
            emb_k = torch.where(act_k.unsqueeze(-1), enc_k, self.inactive_prop_emb[k].unsqueeze(0))
            per_prop.append(emb_k)
        fused = self.global_mlp(torch.cat(per_prop, dim=-1))
        has_any = active.any(dim=-1, keepdim=True)
        global_c = fused * has_any.to(fused.dtype)
        return global_c, has_any

    def forward(self, values: torch.Tensor, unused_upper: torch.Tensor | None, active_mask: torch.Tensor) -> torch.Tensor:
        global_c, _ = self.encode_condition(values, active_mask)
        return global_c

    def build_prefix_tokens(self, global_c: torch.Tensor, has_any: torch.Tensor) -> torch.Tensor | None:
        """Build `(B, P, emb_dim)` condition prefix tokens; uses learned `null_prefix` when unconditional."""
        if self.num_prefix_tokens <= 0:
            return None
        bsz = global_c.shape[0]
        queries = self.prefix_queries.unsqueeze(0).expand(bsz, -1, -1)
        c_rep = global_c.float().unsqueeze(1).expand(-1, self.num_prefix_tokens, -1)
        cond_prefix = queries + self.prefix_proj(torch.cat([queries, c_rep], dim=-1))
        null_prefix = self.null_prefix.unsqueeze(0).expand(bsz, -1, -1)
        mask_3d = has_any.view(bsz, 1, 1)
        return torch.where(mask_3d, cond_prefix, null_prefix)

    def forward_root_prefix(self, global_c: torch.Tensor) -> torch.Tensor:
        return self.root_gate * self.root_proj(global_c.float())

    def _apply_mod(self, mod_net: nn.Module, h: torch.Tensor, global_c: torch.Tensor) -> torch.Tensor:
        raw = mod_net(global_c.float())
        if self.modulation_cap > 0:
            raw = self.modulation_cap * torch.tanh(raw / self.modulation_cap)
        scale, shift = raw.to(h.dtype).chunk(2, dim=-1)
        return h * (1.0 + scale) + shift

    def modulate_layer(self, layer_idx: int, h: torch.Tensor, global_c: torch.Tensor) -> torch.Tensor:
        return self._apply_mod(self.layer_mods[layer_idx], h, global_c)

    def modulate_out(self, h: torch.Tensor, global_c: torch.Tensor) -> torch.Tensor:
        return self._apply_mod(self.out_mod, h, global_c)
