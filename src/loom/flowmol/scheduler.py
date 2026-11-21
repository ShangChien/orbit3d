"""FP32 interpolant schedules, distorted time sampling, and inference time grids for FlowMol."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple
import torch
import torch.nn as nn


CANONICAL_FEATS: Tuple[str, ...] = ("x", "a", "c", "e")


class FlowInterpolantScheduler(nn.Module):
    """Per-modality interpolant schedule alpha_t in [0, 1] and time derivative alpha_t_prime.

    All schedule arithmetic is strictly forced into FP32 per Ampere precision rules.
    Default cosine exponents (nu_x=1.0, nu_a=2.0, nu_c=2.0, nu_e=1.5) follow FlowMol2/3:
    coordinates settle first, followed by bonds, followed by discrete atom/charge identities.
    """

    def __init__(
        self,
        schedule_type: str | Mapping[str, str] = "cosine",
        nu_params: Mapping[str, float] | None = None,
        feats: Sequence[str] = CANONICAL_FEATS,
    ) -> None:
        super().__init__()
        self.feats = tuple(feats)
        if isinstance(schedule_type, str):
            self.schedule_dict: Dict[str, str] = {f: schedule_type for f in self.feats}
        else:
            self.schedule_dict = {f: str(schedule_type[f]) for f in self.feats}

        default_nu = {"x": 1.0, "a": 2.0, "c": 2.0, "e": 1.5}
        merged_nu = {**default_nu, **(dict(nu_params) if nu_params is not None else {})}
        for f, stype in self.schedule_dict.items():
            if stype not in ("cosine", "linear", "power"):
                raise ValueError(f"Unsupported schedule type '{stype}' for feature '{f}'")
        self.register_buffer(
            "nu_vec",
            torch.tensor([float(merged_nu[f]) for f in self.feats], dtype=torch.float32),
            persistent=True,
        )

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Return `(B, len(feats))` in float32."""
        with torch.autocast(device_type=t.device.type, enabled=False):
            tf = t.float().clamp(1e-6, 1.0 - 1e-6).unsqueeze(-1)
            cols = []
            for idx, feat in enumerate(self.feats):
                stype = self.schedule_dict[feat]
                nu = self.nu_vec[idx]
                if stype == "linear":
                    a = tf
                elif stype == "power":
                    a = torch.pow(tf, nu)
                else:  # cosine: 1 - cos^2(pi/2 * t^nu) = sin^2(pi/2 * t^nu)
                    a = torch.sin(0.5 * math.pi * torch.pow(tf, nu)).square()
                cols.append(a)
            return torch.cat(cols, dim=-1)

    def alpha_t_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Return exact analytical d(alpha_t)/dt of shape `(B, len(feats))` in float32."""
        with torch.autocast(device_type=t.device.type, enabled=False):
            tf = t.float().clamp(1e-6, 1.0 - 1e-6).unsqueeze(-1)
            cols = []
            for idx, feat in enumerate(self.feats):
                stype = self.schedule_dict[feat]
                nu = self.nu_vec[idx]
                if stype == "linear":
                    ap = torch.ones_like(tf)
                elif stype == "power":
                    ap = nu * torch.pow(tf, nu - 1.0)
                else:
                    # d/dt sin^2(pi/2 * t^nu) = pi/2 * sin(pi * t^nu) * nu * t^(nu - 1)
                    ap = 0.5 * math.pi * torch.sin(math.pi * torch.pow(tf, nu)) * nu * torch.pow(tf, nu - 1.0)
                cols.append(ap)
            return torch.cat(cols, dim=-1)

    def loss_weights(self, t: torch.Tensor, min_w: float = 0.05, max_w: float = 1.5) -> torch.Tensor:
        """Time-dependent endpoint loss weights clamped to `[min_w, max_w]` in float32."""
        with torch.autocast(device_type=t.device.type, enabled=False):
            a = self.alpha_t(t)
            w = a / (1.0 - a).clamp_min(1e-5)
            return w.clamp(min=min_w, max=max_w)


def sample_train_timesteps(
    batch_size: int,
    device: torch.device,
    *,
    mode: str = "poly_mix",
    t_min: float = 0.002,
    t_max: float = 0.998,
    uniform_frac: float = 0.10,
) -> torch.Tensor:
    """Sample training timepoints in `(t_min, t_max)` strictly in FP32.

    `poly_mix` allocates 90% density to a high-t tilted distribution (`1 - (1 - u)^1.6`,
    putting extra emphasis on bond/geometry crystallization) and 10% uniform coverage.
    """
    with torch.autocast(device_type=device.type, enabled=False):
        u = torch.rand((batch_size,), dtype=torch.float32, device=device)
        if mode == "uniform":
            t = u
        elif mode == "poly_mix":
            skewed = 1.0 - torch.pow(1.0 - u, 1.6)
            use_uniform = torch.rand((batch_size,), dtype=torch.float32, device=device) < uniform_frac
            t = torch.where(use_uniform, u, skewed)
        elif mode == "logit_normal":
            t = torch.sigmoid(torch.randn((batch_size,), dtype=torch.float32, device=device))
        else:
            raise ValueError(f"Unsupported timestep sampling mode: {mode}")
        return t * (t_max - t_min) + t_min


def build_inference_tspan(
    n_steps: int,
    device: torch.device,
    *,
    grid_type: str = "quadratic_end",
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """Build monotonic integration time grid of length `n_steps + 1` in FP32.

    `quadratic_end` concentrates steps near t=1 where bond orders & fine bond lengths sharpen,
    enabling high validity in 20-25 steps.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    with torch.autocast(device_type=device.type, enabled=False):
        s = torch.linspace(0.0, 1.0, n_steps + 1, dtype=torch.float32, device=device)
        if grid_type == "uniform":
            grid = s
        elif grid_type == "quadratic_end":
            # Mildly denser near t=1: mix of uniform and 1 - (1 - s)^1.5
            grid = 0.35 * s + 0.65 * (1.0 - torch.pow(1.0 - s, 1.5))
        elif grid_type == "log_end":
            grid = 1.0 - torch.pow(1.0 - s, 2.0)
        else:
            raise ValueError(f"Unsupported grid_type: {grid_type}")
        grid[0] = 0.0
        grid[-1] = 1.0
        return t_start + (t_end - t_start) * grid
