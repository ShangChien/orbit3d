# ORBIT-3D modification: private package imports; see NOTICE.txt.
"""Explicit checkpoint family for the experimental symmetric radial potential."""
import torch

from orbit3d.compat.hyff.models.pair_radials import PairRadialConfig, PairRadialPotential


ARCHITECTURE = 'smooth_l2_pair_conditioning'


def load_checkpoint(path, device='cpu'):
    raise ValueError('Checkpoint loading is not available here; use orbit3d.compat.hyff.weights.load_model')
