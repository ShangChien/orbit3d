"""ORBIT implementation of multi-axis rotary position encoding.

This file replaces the upstream RoPE implementation. It uses real-valued pair
rotations and has no learned parameters, preserving the pretrained tensor layout.
"""

import torch


class RoPEnD(torch.nn.Module):
    def __init__(self, head_dim, theta=10000.0, n=3):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.n = n
        self.pairs_per_axis = (head_dim // 2) // n
        self.cosines = None
        self.sines = None
        self._cos_bf16 = None
        self._sin_bf16 = None
        self._rates_cache: dict[torch.device, torch.Tensor] = {}

    def _get_rates(self, device: torch.device) -> torch.Tensor:
        rates = self._rates_cache.get(device)
        if rates is None:
            width = self.pairs_per_axis
            exponents = torch.arange(width, device=device, dtype=torch.float32) / width
            rates = self.theta ** (-exponents)
            self._rates_cache[device] = rates
        return rates

    def calc_and_cache(self, grid_pos):
        rates = self._get_rates(grid_pos.device)
        angles = (grid_pos.float().unsqueeze(-1) * rates).flatten(-2)
        remainder = self.head_dim // 2 - angles.shape[-1]
        if remainder:
            angles = torch.nn.functional.pad(angles, (0, remainder))
        self.cosines = angles.cos().unsqueeze(-2)
        self.sines = angles.sin().unsqueeze(-2)
        self._cos_bf16 = self.cosines.to(torch.bfloat16)
        self._sin_bf16 = self.sines.to(torch.bfloat16)

    def apply_qk(self, q, k):
        if q.shape == k.shape and q.dtype == k.dtype:
            qk = torch.stack((q, k), dim=0)
            if q.dtype == torch.bfloat16 and self._cos_bf16 is not None:
                c = self._cos_bf16.unsqueeze(0)
                s = self._sin_bf16.unsqueeze(0)
                pairs = qk.reshape(*qk.shape[:-1], -1, 2)
                first = pairs[..., 0]
                second = pairs[..., 1]
                out = torch.stack((first * c - second * s, first * s + second * c), dim=-1).flatten(-2)
                return out[0], out[1]
            c = self.cosines.unsqueeze(0)
            s = self.sines.unsqueeze(0)
            pairs = qk.float().reshape(*qk.shape[:-1], -1, 2)
            first = pairs[..., 0]
            second = pairs[..., 1]
            out = torch.stack((first * c - second * s, first * s + second * c), dim=-1).flatten(-2).to(q.dtype)
            return out[0], out[1]

        def rotate(value):
            pairs = value.float().reshape(*value.shape[:-1], -1, 2)
            first, second = pairs.unbind(-1)
            result = torch.stack(
                (first * self.cosines - second * self.sines, first * self.sines + second * self.cosines),
                dim=-1,
            )
            return result.flatten(-2).to(value.dtype)

        return rotate(q), rotate(k)
