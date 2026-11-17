"""Unit tests for Scheme 2 capabilities:
1. SE(3)-invariant inertial + polarity canonical frame (`canonicalize_coords_pca`).
2. Size-Anchored CFG (`uncond_active_mask`) in `LoomUni3DARGenerator.sample` and `trajectory_distributions`.
3. Column-restricted contrastive negative selection (`contrast_col_mask`) in `_select_incompatible_negatives`.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from loom.weaving.uni3dar_slot import LoomUni3DARGenerator, _select_incompatible_negatives
from scripts.loom_qm9_scheme2_natoms_pca import canonicalize_coords_pca


def test_canonicalize_coords_pca_se3_invariance_and_chirality() -> None:
    symbols = ["C", "C", "O", "N", "F", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.00, 0.00, 0.00],
            [1.48, 0.14, -0.08],
            [2.12, 1.24, 0.22],
            [-1.24, 0.68, -0.31],
            [1.92, -0.95, 0.58],
            [-1.18, 1.66, -0.11],
            [-2.04, 0.27, 0.24],
            [0.06, -0.49, 0.98],
            [0.11, -0.78, -0.79],
        ],
        dtype=np.float32,
    )
    ref = canonicalize_coords_pca(symbols, coords)
    assert np.allclose(ref.mean(axis=0), 0.0, atol=1e-5)

    d_orig = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    d_ref = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
    assert np.allclose(d_orig, d_ref, atol=1e-5)

    # Signed tetrahedral volume at chiral C0 bonded to (C1, N3, H7, H8) must retain exact sign (proper rotation det=+1)
    def _signed_vol(p: np.ndarray) -> float:
        v1 = p[1] - p[0]
        v2 = p[3] - p[0]
        v3 = p[7] - p[0]
        return float(np.dot(v1, np.cross(v2, v3)))

    vol_orig = _signed_vol(coords)
    vol_ref = _signed_vol(ref)
    assert np.isclose(vol_orig, vol_ref, atol=1e-4)

    for seed in range(8):
        rot = R.random(random_state=100 + seed).as_matrix()
        shift = np.array([-4.5 + seed, 2.3, 8.1 - 0.5 * seed])
        transformed = (coords @ rot.T + shift).astype(np.float32)
        aligned = canonicalize_coords_pca(symbols, transformed)
        assert np.allclose(ref, aligned, atol=1e-4)


def test_fixed_natoms_contrastive_negative_selection() -> None:
    # Column 0 = target property, Column 1 = n_atoms
    targets = torch.tensor(
        [
            [0.5, 10.0],
            [0.6, 22.0],  # huge n_atoms gap (12), tiny prop gap (0.1) -> MUST NOT be chosen as negative for row 0!
            [3.2, 10.0],  # large prop gap (2.7), zero n_atoms gap -> MUST be chosen!
            [3.4, 18.0],
        ],
        dtype=torch.float32,
    )
    train_mask = torch.ones((4, 2), dtype=torch.bool)
    col_mask = torch.tensor([[True, False]])
    select_mask = train_mask & col_mask

    neg_idx, eligible = _select_incompatible_negatives(targets, select_mask, min_gap_eV=1.5, deterministic=True)
    assert bool(eligible.all())
    # Rows 0 and 1 can only pair with rows 2 or 3
    assert int(neg_idx[0].item()) in (2, 3)
    assert int(neg_idx[1].item()) in (2, 3)

    # Verify swapping only col 0 preserves row's exact n_atoms (col 1)
    swapped = torch.where(col_mask, targets[neg_idx], targets)
    assert torch.allclose(swapped[:, 1], targets[:, 1])
    assert torch.all((swapped[:, 0] - targets[:, 0]).abs() > 1.5)


def test_size_anchored_cfg_sample_and_trajectory_distributions() -> None:
    backbone = LoomUni3DARGenerator._build_backbone(
        merge_level=4,
        allowed_elements=("H", "C", "N", "O", "F"),
        batch_size=2,
        emb_dim=64,
        num_head=4,
        layer=2,
        ffn_multiple=2,
    )
    gen = LoomUni3DARGenerator(
        backbone=backbone,
        target_cols=["mu", "n_atoms"],
        target_units=["D", "atoms"],
        mean=[2.5, 18.0],
        scale=[1.5, 3.0],
        num_rbf=16,
        num_freqs=8,
        num_prefix_tokens=2,
        lora_rank=0,
        use_valence_head=True,
    )
    gen.configure_trainable(train_conditioner=True, train_lora=False, unfreeze_full_backbone=True)

    t_norm = gen.normalize_targets(np.array([[1.2, 15.0], [4.1, 20.0]], dtype=np.float32))
    cond_mask = torch.ones((2, 2), dtype=torch.bool)
    uncond_mask = torch.tensor([[False, True], [False, True]], dtype=torch.bool)

    trajs = gen.sample(
        t_norm,
        cond_mask,
        cfg_scale=2.0,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        seed=42,
        use_bf16=False,
        max_decode_steps=24,
        uncond_active_mask=uncond_mask,
    )
    assert len(trajs) == 2
    packets = gen.trajectory_distributions(
        trajs,
        t_norm,
        cond_mask,
        cfg_scale=2.0,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        topology_only_policy=True,
        uncond_active_mask=uncond_mask,
    )
    assert len(packets) >= 1
    for logp, labels, owner in packets:
        assert torch.isfinite(logp.gather(1, labels[:, None])).all()
        assert int(owner.max().item()) < 2
