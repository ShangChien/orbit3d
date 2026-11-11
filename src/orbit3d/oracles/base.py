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
        y_vec = y_pred.view(self.output_dim)
        if (
            isinstance(target_intervals, (tuple, list))
            and len(target_intervals) == 2
            and isinstance(target_intervals[0], (int, float))
        ):
            return F.relu(float(target_intervals[0]) - y_vec[0]) + F.relu(y_vec[0] - float(target_intervals[1]))
        total = 0.0 * y_vec.sum()
        if isinstance(target_intervals, dict):
            for k, (lo, hi) in target_intervals.items():
                idx = self.target_cols.index(k) if isinstance(k, str) else int(k)
                w = float(weights.get(k, 1.0)) if weights else 1.0
                total = total + w * (F.relu(float(lo) - y_vec[idx]) + F.relu(y_vec[idx] - float(hi)))
        else:
            for idx, iv in enumerate(target_intervals):
                if iv is not None:
                    w = float(weights.get(idx, 1.0)) if weights else 1.0
                    total = total + w * (F.relu(float(iv[0]) - y_vec[idx]) + F.relu(y_vec[idx] - float(iv[1])))
        return total
