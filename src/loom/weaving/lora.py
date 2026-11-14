"""Lightweight zero-perturbation LoRA adapters for 3D generative backbones."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper around any `nn.Linear` (including `loom.vendor.uni3dar` `Linear`).

    `lora_B` is zero-initialized so at initialization `LoRALinear(x) == base(x)` bit-for-bit.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty((self.rank, self.in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.rank), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for p in self.base.parameters():
            p.requires_grad_(False)
        self._eval_weight: torch.Tensor | None = None
        self._eval_weight_bf16: torch.Tensor | None = None

    def _clear_eval_cache(self) -> None:
        self._eval_weight = None
        self._eval_weight_bf16 = None

    def train(self, mode: bool = True) -> "LoRALinear":
        self._clear_eval_cache()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        self._clear_eval_cache()
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        need_lora_grad = torch.is_grad_enabled() and (self.lora_A.requires_grad or self.lora_B.requires_grad)
        if not self.training and not need_lora_grad:
            if self._eval_weight is None or self._eval_weight.device != self.base.weight.device:
                with torch.no_grad():
                    merged = (self.base.weight + (self.lora_B @ self.lora_A) * self.scaling).detach()
                    self._eval_weight = merged
                    self._eval_weight_bf16 = merged.to(torch.bfloat16) if merged.is_cuda else None
            if x.dtype == torch.bfloat16 and self._eval_weight_bf16 is not None:
                w = self._eval_weight_bf16
            else:
                w = self._eval_weight.to(x.dtype)
            b = self.base.bias.to(x.dtype) if self.base.bias is not None else None
            return F.linear(x, w, b)

        self._clear_eval_cache()
        base_out = self.base(x)
        x_drop = self.dropout(x)
        delta = F.linear(F.linear(x_drop, self.lora_A.to(x_drop.dtype)), self.lora_B.to(x_drop.dtype)) * self.scaling
        return base_out + delta.to(base_out.dtype)


def inject_uni3dar_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    include_ffn: bool = True,
    include_heads: bool = False,
) -> List[str]:
    """Attach `LoRALinear` adapters across all Transformer attention (and optional FFN/head) projections."""
    injected: List[str] = []
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        for idx, layer in enumerate(model.decoder.layers):
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                sub = getattr(layer.attn, proj_name, None)
                if isinstance(sub, nn.Linear) and not isinstance(sub, LoRALinear):
                    setattr(layer.attn, proj_name, LoRALinear(sub, rank=rank, alpha=alpha, dropout=dropout))
                    injected.append(f"decoder.layers.{idx}.attn.{proj_name}")
            if include_ffn and hasattr(layer, "ffn"):
                for w_name in ("w1", "w2", "w3"):
                    sub = getattr(layer.ffn, w_name, None)
                    if isinstance(sub, nn.Linear) and not isinstance(sub, LoRALinear):
                        setattr(layer.ffn, w_name, LoRALinear(sub, rank=rank, alpha=alpha, dropout=dropout))
                        injected.append(f"decoder.layers.{idx}.ffn.{w_name}")

    if include_heads:
        for t_idx, head in enumerate(getattr(model, "tree_heads", [])):
            for h_name in ("dense", "out"):
                sub = getattr(head, h_name, None)
                if isinstance(sub, nn.Linear) and not isinstance(sub, LoRALinear):
                    setattr(head, h_name, LoRALinear(sub, rank=rank, alpha=alpha, dropout=dropout))
                    injected.append(f"tree_heads.{t_idx}.{h_name}")
        for t_idx, ahead in enumerate(getattr(model, "atom_heads", [])):
            for p_idx, phead in enumerate(getattr(ahead, "pred_heads", [])):
                if p_idx == 0:  # Element classification head only; leave fine XYZ heads pristine
                    for h_name in ("dense", "out"):
                        sub = getattr(phead, h_name, None)
                        if isinstance(sub, nn.Linear) and not isinstance(sub, LoRALinear):
                            setattr(phead, h_name, LoRALinear(sub, rank=rank, alpha=alpha, dropout=dropout))
                            injected.append(f"atom_heads.{t_idx}.pred_heads.0.{h_name}")
    return injected


def iter_lora_parameters(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    for name, param in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            yield name, param
