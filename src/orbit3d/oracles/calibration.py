"""Held-out normalized-residual calibration, separate from ensemble standard deviation."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np


def finite_sample_quantile(scores, coverage):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n < 2 or not np.isfinite(scores).all() or not 0 < coverage < 1:
        raise ValueError("Need finite calibration scores and coverage in (0,1)")
    rank = int(np.ceil((n + 1) * coverage))
    if rank > n:
        raise ValueError("Calibration set is too small for this coverage")
    return np.sort(scores, axis=0)[rank - 1]


@dataclass
class ResidualCalibration:
    target_cols: list
    coverage: float
    std_floor: float
    marginal_q: list
    joint_q: float
    n_calibration: int

    @classmethod
    def fit(cls, names, truth, mean, std, coverage=0.9, std_floor=0.05):
        truth, mean, std = map(lambda x: np.asarray(x, dtype=float), (truth, mean, std))
        if truth.ndim != 2 or truth.shape[-1] != len(names) or std_floor <= 0:
            raise ValueError("Target dimensions and standard deviation floor must match the calibration contract")
        if truth.shape != mean.shape or mean.shape != std.shape or (std < 0).any():
            raise ValueError("Calibration prediction/target shapes or standard deviations are invalid")
        scores = np.abs(truth - mean) / np.maximum(std, std_floor)
        return cls(
            list(names),
            coverage,
            std_floor,
            finite_sample_quantile(scores, coverage).tolist(),
            float(finite_sample_quantile(scores.max(-1), coverage)),
            len(truth),
        )

    def intervals(self, mean, std, simultaneous=True):
        mean, std = np.asarray(mean), np.asarray(std)
        q = self.joint_q if simultaneous else np.array(self.marginal_q)
        radius = np.maximum(std, self.std_floor) * q
        return mean - radius, mean + radius

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
