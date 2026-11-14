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


@dataclass(frozen=True)
class TargetPoints:
    """Point conditioning is separate from the acceptance radius."""
    names: tuple
    values: tuple
    radii: tuple

    @classmethod
    def parse(cls, raw, radii, names):
        names = tuple(names)
        if set(raw) - set(names) or set(radii) - set(names):
            raise ValueError('Unknown point target name')
        values = tuple(raw.get(n) for n in names)
        tolerance = tuple(radii.get(n) for n in names)
        if not any(v is not None for v in values):
            raise ValueError('No active point targets')
        for value, radius in zip(values, tolerance):
            if value is not None and (not np.isfinite(value) or radius is None or not np.isfinite(radius) or radius <= 0):
                raise ValueError('Active point targets require finite values and positive radii')
        return cls(names, values, tolerance)

    def intervals(self):
        return TargetIntervals.parse({n: [v-r, v+r] for n,v,r in zip(self.names,self.values,self.radii) if v is not None},self.names)

    def arrays(self, mean, scale):
        mean, scale = np.asarray(mean), np.asarray(scale)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid point normalization')
        active = np.array([v is not None for v in self.values])
        values = np.array([(v-mean[i])/scale[i] if v is not None else 0. for i,v in enumerate(self.values)],dtype=np.float32)
        return values, values.copy(), np.repeat(active[:,None],2,axis=1)

    def distance(self, predictions):
        y = np.asarray(predictions).reshape(-1)
        if len(y) != len(self.names):
            raise ValueError('Point prediction dimension mismatch')
        return float(np.mean([((y[i]-v)/self.radii[i])**2 for i,v in enumerate(self.values) if v is not None]))
