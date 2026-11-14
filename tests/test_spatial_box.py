from types import SimpleNamespace
import numpy as np
import pytest
import torch
from loom.generator.spatial_box import BoxSpec3D


def test_window_masks_occupancy_bits_before_sampling():
    box = BoxSpec3D(upper_cells=(32, 32, 32))
    logits = box.mask_tree_logits(torch.zeros(1, 255), torch.zeros(1, 3), torch.tensor([6]))
    assert torch.isfinite(logits).sum() == 1
    assert torch.isfinite(logits[0, 0])  # occupancy bit 1: lower x/y/z child
    with pytest.raises(ValueError, match='No occupied'):
        box.mask_tree_logits(torch.zeros(1, 255), torch.tensor([[40, 40, 40]]), torch.tensor([1]))


def test_box_preserves_native_grid_and_explicit_world_origin():
    box = BoxSpec3D(origin=(-7.68, -7.68, -7.68), lower_cells=(16, 16, 16), upper_cells=(48, 48, 48))
    box.validate_model(SimpleNamespace(grid_len=.24, xyz_resolution=.01, merge_level=6))
    np.testing.assert_allclose(box.to_world([[7.68, 7.68, 7.68]]), [[0, 0, 0]])
    with pytest.raises(ValueError, match='violate'):
        box.to_world([[0, 0, 0]])
    with pytest.raises(ValueError, match='migration'):
        box.validate_model(SimpleNamespace(grid_len=.48, xyz_resolution=.01, merge_level=6))
    with pytest.raises(ValueError, match='integer'):
        BoxSpec3D(lower_cells=(0.5, 0, 0))
