"""Subwindows of the trained octree grid, without changing its physical mapping."""

from dataclasses import asdict, dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class BoxSpec3D:
    origin: tuple = (0.0, 0.0, 0.0)
    lower_cells: tuple = (0, 0, 0)
    upper_cells: tuple = (64, 64, 64)
    grid_len: float = 0.24
    resolution: float = 0.01
    merge_level: int = 6
    representation: str = 'uni3dar_grid_v1'

    def __post_init__(self):
        lo, hi, origin = map(np.asarray, [self.lower_cells, self.upper_cells, self.origin])
        if any(x.shape != (3,) for x in [lo, hi, origin]) or not np.isfinite(origin).all():
            raise ValueError('Box origin and bounds must have three finite components')
        if not (np.equal(lo, lo.astype(int)).all() and np.equal(hi, hi.astype(int)).all()):
            raise ValueError('Subwindows use integer native grid cells')
        if (lo < 0).any() or (hi > 2**self.merge_level).any() or (hi <= lo).any():
            raise ValueError('Empty or out-of-range octree subwindow')

    def validate_model(self, args):
        if (self.grid_len, self.resolution, self.merge_level, self.representation) != (
            args.grid_len, args.xyz_resolution, args.merge_level, 'uni3dar_grid_v1'
        ):
            raise ValueError('Physical grid changes require a trained representation migration')

    @property
    def world_bounds(self):
        return (np.asarray(self.origin) + np.asarray(self.lower_cells) * self.grid_len,
                np.asarray(self.origin) + np.asarray(self.upper_cells) * self.grid_len)

    def to_world(self, positions):
        coords = np.asarray(positions) + np.asarray(self.origin)
        lo, hi = self.world_bounds
        if not np.isfinite(coords).all() or ((coords < lo - 1e-6) | (coords >= hi + 1e-6)).any():
            raise ValueError('Decoded coordinates violate the octree subwindow')
        return coords

    def mask_tree_logits(self, logits, cell_origins, levels):
        # Logit index j encodes the nonempty occupancy bitset j + 1.
        device = logits.device
        child_bits = torch.arange(8, device=device)
        shifts = torch.stack([child_bits & 1, (child_bits >> 1) & 1, (child_bits >> 2) & 1], -1)
        edge = (2 ** (levels - 1)).reshape(-1, 1, 1)
        child_lo = cell_origins[:, None, :] + shifts[None] * edge
        child_hi = child_lo + edge
        lo = torch.tensor(self.lower_cells, device=device)
        hi = torch.tensor(self.upper_cells, device=device)
        allowed = ((child_lo < hi) & (child_hi > lo)).all(-1)
        forbidden = ((~allowed).long() * (1 << child_bits)).sum(-1)
        patterns = torch.arange(1, 256, device=device)
        supported = (patterns[None] & forbidden[:, None]) == 0
        if not supported.any(-1).all():
            raise ValueError('No occupied child cell is feasible in the requested window')
        return logits.masked_fill(~supported, -torch.inf)

    def to_dict(self):
        return asdict(self)
