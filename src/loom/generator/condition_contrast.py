"""Paired interval supervision using real labels as incompatible conditions."""
import torch


def incompatible_condition_indices(targets, halfwidth=.25):
    """Nearest other label whose interval excludes the target in at least one axis."""
    distances = (targets[:, None] - targets[None]).abs().amax(-1)
    distances = distances.masked_fill(distances <= halfwidth, torch.inf)
    nearest, indices = distances.min(-1)
    return indices, torch.isfinite(nearest)


def condition_margin_loss(positive, negative, eligible, margin=.02):
    """Require lower per-molecule generation loss under the compatible condition."""
    if not eligible.any():
        return positive.sum() * 0
    return torch.relu(margin + positive[eligible] - negative[eligible]).mean()
