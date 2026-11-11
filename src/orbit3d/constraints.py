"""Shared named interval semantics in physical units."""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class TargetIntervals:
    names: tuple
    bounds: tuple

    @classmethod
    def parse(cls, raw, names):
        names = tuple(names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Target names must be nonempty and unique")
        if isinstance(raw, dict):
            if set(raw) - set(names):
                raise ValueError("Unknown target name")
            values = [raw.get(n) for n in names]
        elif (
            isinstance(raw, (list, tuple)) and len(raw) == 2 and all(v is None or np.isscalar(v) for v in raw)
        ):
            if len(names) != 1:
                raise ValueError("Use named intervals for multiple targets")
            values = [raw]
        else:
            values = list(raw)
            if len(values) != len(names):
                raise ValueError("Target count mismatch")
        bounds = []
        for iv in values:
            if iv is None:
                bounds.append(None)
                continue
            if len(iv) != 2:
                raise ValueError("Expected lower/upper pair")
            lo, hi = (-np.inf if iv[0] is None else float(iv[0])), (np.inf if iv[1] is None else float(iv[1]))
            if np.isnan(lo) or np.isnan(hi) or lo > hi or (not np.isfinite(lo) and not np.isfinite(hi)):
                raise ValueError("Invalid or unconstrained interval")
            bounds.append((lo, hi))
        if all(iv is None for iv in bounds):
            raise ValueError("No active targets")
        return cls(names, tuple(bounds))

    def violations(self, predictions):
        y = torch.as_tensor(predictions).reshape(-1)
        if y.numel() != len(self.names):
            raise ValueError("Prediction dimension mismatch")
        return torch.stack(
            [
                torch.zeros_like(y[i]) if iv is None else torch.relu(iv[0] - y[i]) + torch.relu(y[i] - iv[1])
                for i, iv in enumerate(self.bounds)
            ]
        )

    def hit(self, predictions):
        y = torch.as_tensor(predictions)
        return bool(torch.isfinite(y).all() and (self.violations(y) == 0).all())

    def condition(self, mean, scale):
        mean = np.broadcast_to(np.asarray(mean), (len(self.names),))
        scale = np.broadcast_to(np.asarray(scale), (len(self.names),))
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid target normalization")
        # Each target: normalized lower/upper plus separate bound masks.
        out = []
        for i, iv in enumerate(self.bounds):
            out.extend(
                [
                    0 if iv is None or not np.isfinite(iv[0]) else (iv[0] - mean[i]) / scale[i],
                    0 if iv is None or not np.isfinite(iv[1]) else (iv[1] - mean[i]) / scale[i],
                    float(iv is not None and np.isfinite(iv[0])),
                    float(iv is not None and np.isfinite(iv[1])),
                ]
            )
        return np.array(out, dtype=np.float32)
