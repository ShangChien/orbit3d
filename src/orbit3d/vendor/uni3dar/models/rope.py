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

    def calc_and_cache(self, grid_pos):
        width = self.pairs_per_axis
        exponents = torch.arange(width, device=grid_pos.device, dtype=torch.float32) / width
        rates = self.theta ** (-exponents)
        angles = (grid_pos.float().unsqueeze(-1) * rates).flatten(-2)
        remainder = self.head_dim // 2 - angles.shape[-1]
        if remainder:
            angles = torch.nn.functional.pad(angles, (0, remainder))
        self.cosines = angles.cos().unsqueeze(-2)
        self.sines = angles.sin().unsqueeze(-2)

    def apply_qk(self, q, k):
        def rotate(value):
            pairs = value.float().reshape(*value.shape[:-1], -1, 2)
            first, second = pairs.unbind(-1)
            result = torch.stack(
                (first * self.cosines - second * self.sines, first * self.sines + second * self.cosines),
                dim=-1,
            )
            return result.flatten(-2).to(value.dtype)

        return rotate(q), rotate(k)
