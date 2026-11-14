"""
Abstract Base Class for 3D Molecular Property Oracles (`base.py`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem


class PropertyOracleBase(nn.Module):
    """
    Unified contract for all 3 property oracle modes (`external_uniqsar`,
    `internal_unimol`, and `internal_dpa4`).
    """

    output_dim: int = 1
    target_cols: List[str] = ["TARGET"]
    has_3d_unimol: bool = True

    def predict_3d_mols(self, mols: List[Chem.Mol]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def forward_differentiable_3d(
        self,
        coords: torch.Tensor,
        atom_symbols: List[str],
        smiles: str = "C",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def compute_multi_target_interval_loss(
        self,
        y_pred: torch.Tensor,
        target_intervals: Union[Tuple[float, float], Dict[str, Tuple[float, float]], List[Tuple[float, float]]],
        weights: Optional[Dict[Union[str, int], float]] = None,
    ) -> torch.Tensor:
        from ..constraints import TargetIntervals
        intervals=TargetIntervals.parse(target_intervals,self.target_cols)
        violations=intervals.violations(y_pred)
        if weights:
            w=torch.tensor([weights.get(name,weights.get(i,1.)) for i,name in enumerate(self.target_cols)],
                           device=y_pred.device,dtype=y_pred.dtype)
            if not torch.isfinite(w).all() or (w<0).any(): raise ValueError("Invalid target weights")
            violations=violations*w
        return violations.sum()
