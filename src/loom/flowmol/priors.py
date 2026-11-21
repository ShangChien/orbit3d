"""FP32 geometric priors, size-adaptive Rg scaling, proper SO(3) Kabsch OT alignment, and distortion."""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def coord_prior_std_for_size(
    n_atoms: int | torch.Tensor,
    base_std: float = 1.15,
    size_scale: bool = True,
    ref_atoms: float = 25.0,
) -> float | torch.Tensor:
    """Physical 3D radius-of-gyration scaling `sigma(N) = base_std * (N / ref_atoms)^(1/3)`."""
    if not size_scale:
        if isinstance(n_atoms, torch.Tensor):
            return torch.full_like(n_atoms, float(base_std), dtype=torch.float32)
        return float(base_std)
    if isinstance(n_atoms, torch.Tensor):
        return float(base_std) * torch.pow(n_atoms.float().clamp_min(2.0) / float(ref_atoms), 1.0 / 3.0)
    return float(base_std) * ((max(float(n_atoms), 2.0) / float(ref_atoms)) ** (1.0 / 3.0))


def remove_com_fp32(
    pos: torch.Tensor,
    node_batch_idx: torch.Tensor,
    batch_size: int,
    n_atoms_per_mol: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Subtract per-molecule center of mass strictly in FP32."""
    with torch.autocast(device_type=pos.device.type, enabled=False):
        pos_f = pos.float()
        sums = torch.zeros((batch_size, 3), dtype=torch.float32, device=pos.device)
        sums.index_add_(0, node_batch_idx, pos_f)
        if n_atoms_per_mol is None:
            counts = torch.zeros((batch_size,), dtype=torch.float32, device=pos.device)
            counts.index_add_(0, node_batch_idx, torch.ones((pos.shape[0],), dtype=torch.float32, device=pos.device))
        else:
            counts = n_atoms_per_mol.to(device=pos.device, dtype=torch.float32)
        means = sums / counts.clamp_min(1.0).unsqueeze(-1)
        return pos_f - means[node_batch_idx]


@torch.no_grad()
def proper_kabsch_align_fp32(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """Align zero-COM source cloud `x_0` `(N, 3)` to target `x_1` `(N, 3)` via proper SO(3) rotation.

    Enforces det(R) = +1 so chirality is strictly preserved. Always runs in FP32.
    """
    x0 = x_0.float()
    x1 = x_1.float()
    x0_mean = x0.mean(dim=0, keepdim=True)
    x1_mean = x1.mean(dim=0, keepdim=True)
    x0_c = x0 - x0_mean
    x1_c = x1 - x1_mean

    H = x0_c.T @ x1_c
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.T
    det = torch.linalg.det(V @ U.T)
    if det < 0:
        V = V.clone()
        V[:, -1] = -V[:, -1]
    R = V @ U.T
    aligned = (x0_c @ R.T) + x1_mean
    return aligned - aligned.mean(dim=0, keepdim=True) + x1_mean


@torch.no_grad()
def sample_aligned_coord_prior_np(
    x_1: np.ndarray,
    rng: np.random.Generator,
    *,
    prior_std: float = 1.15,
    size_scale_prior: bool = True,
    use_permutation: bool = True,
    use_rigid: bool = True,
) -> np.ndarray:
    """Fast CPU-worker NumPy/SciPy implementation of size-scaled zero-COM Gaussian prior + OT + SO(3) Kabsch."""
    n_atoms = int(x_1.shape[0])
    x1 = np.asarray(x_1, dtype=np.float32)
    x1 = x1 - x1.mean(axis=0, keepdims=True)

    eff_std = float(coord_prior_std_for_size(n_atoms, base_std=prior_std, size_scale=size_scale_prior))
    x0 = rng.standard_normal((n_atoms, 3)).astype(np.float32) * eff_std
    x0 = x0 - x0.mean(axis=0, keepdims=True)

    if use_permutation and n_atoms > 1:
        diff = x1[:, None, :] - x0[None, :, :]
        cost = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
        _, col_ind = linear_sum_assignment(cost)
        x0 = x0[col_ind]
        x0 = x0 - x0.mean(axis=0, keepdims=True)

    if use_rigid and n_atoms >= 3:
        H = x0.T @ x1
        U, _, Vt = np.linalg.svd(H.astype(np.float64))
        V = Vt.T
        if np.linalg.det(V @ U.T) < 0.0:
            V[:, -1] *= -1.0
        R = (V @ U.T).astype(np.float32)
        x0 = x0 @ R.T
        x0 = x0 - x0.mean(axis=0, keepdims=True)

    return x0.astype(np.float32, copy=False)


def apply_train_geometry_distortion_fp32(
    x_t: torch.Tensor,
    t_mol: torch.Tensor,
    node_batch_idx: torch.Tensor,
    batch_size: int,
    *,
    prob: float = 0.25,
    max_distortion: float = 0.20,
    min_t: float = 0.30,
) -> torch.Tensor:
    """FlowMol3-style train-time geometry distortion on intermediate `x_t` (keeping `x_1` pristine)."""
    if prob <= 0.0 or max_distortion <= 0.0:
        return x_t
    with torch.autocast(device_type=x_t.device.type, enabled=False):
        x_f = x_t.float()
        active_mol = (torch.rand((batch_size,), device=x_t.device) < prob) & (t_mol.float() >= min_t)
        if not active_mol.any():
            return x_f
        amp_mol = torch.rand((batch_size,), device=x_t.device) * float(max_distortion) * active_mol.float()
        amp_node = amp_mol[node_batch_idx].unsqueeze(-1)
        noise = (torch.rand_like(x_f) * 2.0 - 1.0) * amp_node
        noise = remove_com_fp32(noise, node_batch_idx, batch_size)
        return x_f + noise
