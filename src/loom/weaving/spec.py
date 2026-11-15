"""Layer 2A: Declarative Predictor-Agnostic Task & Constraint Weaving Specification (`spec.py`).

Decouples `Uni-Loom` completely from hardcoded `HOMO / LUMO` assumptions so any combination of:
  1. `objectives`  — Point (`v ± tol`), interval (`[lo, hi]`), or directional (`minimize` / `maximize`) targets
                     drawn from one or multiple independent external/internal `PredictorAsset`s;
  2. `constraints` — Inequality feasibility gates (`SA_Score <= 3.5`, `QED >= 0.55`, `MW <= 550`,
                     `smarts_alert <= 0`, or external toxicity/solubility QSAR thresholds) that shape
                     Stage 1 relabeling and Stage 2 GRPO advantages without consuming conditioner axes;
  3. `auditors`    — Anti-collapse guardrails supporting BOTH cross-foundation model disagreement
                     (`mode="cross_oracle"`) AND single-predictor enterprise setups (`mode="single_oracle_ad"`
                     combining Morgan applicability domain + ensemble spread + MLIP atomic strain).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import yaml


@dataclass(frozen=True)
class ObjectiveChannelSpec:
    name: str
    asset_id: str
    asset_channel: str
    mode: str = "point"  # "point" | "interval" | "minimize" | "maximize"
    tolerance: float = 0.25
    unit: str = "unspecified"
    weight: float = 1.0
    interval_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None

    def __post_init__(self) -> None:
        if self.mode not in {"point", "interval", "minimize", "maximize"}:
            raise ValueError(f"Unsupported ObjectiveChannelSpec mode: {self.mode}")
        if self.tolerance <= 0 or not np.isfinite(self.tolerance):
            raise ValueError(f"Objective '{self.name}' requires positive finite tolerance")
        if self.weight <= 0 or not np.isfinite(self.weight):
            raise ValueError(f"Objective '{self.name}' requires positive finite weight")

    def distance_to_goal(self, preds: np.ndarray, requested_goal: np.ndarray | float) -> np.ndarray:
        """Non-negative error distance (`0` inside interval or at point)."""
        y = np.asarray(preds, dtype=np.float64)
        g = np.asarray(requested_goal, dtype=np.float64)
        if self.mode == "point":
            return np.abs(y - g)
        if self.mode == "interval":
            if self.interval_bounds is not None:
                lo_raw, hi_raw = self.interval_bounds
                lo = -np.inf if lo_raw is None else float(lo_raw)
                hi = np.inf if hi_raw is None else float(hi_raw)
            else:
                lo = g - self.tolerance
                hi = g + self.tolerance
            return np.maximum(0.0, lo - y) + np.maximum(0.0, y - hi)
        if self.mode == "minimize":
            return np.maximum(0.0, y - g)
        # maximize
        return np.maximum(0.0, g - y)

    def is_hit(self, preds: np.ndarray, requested_goal: np.ndarray | float) -> np.ndarray:
        y = np.asarray(preds, dtype=np.float64)
        dist = self.distance_to_goal(y, requested_goal)
        if self.mode == "interval":
            return np.isfinite(y) & (dist <= 1e-8)
        return np.isfinite(y) & (dist <= self.tolerance)


@dataclass(frozen=True)
class ConstraintChannelSpec:
    name: str
    asset_id: str
    asset_channel: str
    op: str  # "leq" | "geq" | "between"
    threshold: float | Tuple[float, float]
    soft_margin: float = 0.50
    weight: float = 0.60
    hard_gate: bool = True

    def __post_init__(self) -> None:
        if self.op not in {"leq", "geq", "between"}:
            raise ValueError(f"Unsupported ConstraintChannelSpec op: {self.op}")
        if self.soft_margin <= 0:
            raise ValueError("Constraint soft_margin must be positive")

    def violation(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        out = np.full_like(v, np.inf, dtype=np.float64)
        fin = np.isfinite(v)
        if not fin.any():
            return out
        vf = v[fin]
        if self.op == "leq":
            thr = float(self.threshold)  # type: ignore[arg-type]
            out[fin] = np.maximum(0.0, vf - thr)
        elif self.op == "geq":
            thr = float(self.threshold)  # type: ignore[arg-type]
            out[fin] = np.maximum(0.0, thr - vf)
        else:
            lo, hi = self.threshold  # type: ignore[misc]
            out[fin] = np.maximum(0.0, float(lo) - vf) + np.maximum(0.0, vf - float(hi))
        return out

    def satisfied_mask(self, values: np.ndarray) -> np.ndarray:
        viol = self.violation(values)
        return np.isfinite(viol) & (viol <= 1e-8)

    def smooth_satisfaction_score(self, values: np.ndarray) -> np.ndarray:
        """Continuous shaping score in `[0, 1]` (`1.0` when satisfied, decaying smoothly on violation)."""
        viol = self.violation(values)
        sc = np.zeros_like(viol, dtype=np.float64)
        fin = np.isfinite(viol)
        sc[fin] = np.exp(-viol[fin] / self.soft_margin)
        return sc


@dataclass(frozen=True)
class AuditorChannelSpec:
    name: str
    objective_name: str
    mode: str = "cross_oracle"  # "cross_oracle" | "single_oracle_ad"
    auditor_asset_id: Optional[str] = None
    auditor_channel: Optional[str] = None
    disagreement_cap: float = 0.35
    tanimoto_cutoff: float = 0.25

    def __post_init__(self) -> None:
        if self.mode not in {"cross_oracle", "single_oracle_ad"}:
            raise ValueError(f"Unsupported AuditorChannelSpec mode: {self.mode}")
        if self.mode == "cross_oracle" and not self.auditor_asset_id:
            raise ValueError(f"Auditor '{self.name}' with mode='cross_oracle' requires auditor_asset_id")


@dataclass(frozen=True)
class TaskWeavingSpec:
    task_name: str
    objectives: Tuple[ObjectiveChannelSpec, ...]
    description: str = ""
    assets: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    potential_asset_id: Optional[str] = None
    constraints: Tuple[ConstraintChannelSpec, ...] = field(default_factory=tuple)
    auditors: Tuple[AuditorChannelSpec, ...] = field(default_factory=tuple)
    unphysical_pair_rule: Optional[str] = None  # "x>=y" | "x>y" | None
    reconcile_valence: bool = True

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("TaskWeavingSpec requires at least one ObjectiveChannelSpec")
        names = [o.name for o in self.objectives]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate objective channel names: {names}")

    @property
    def objective_names(self) -> List[str]:
        return [o.name for o in self.objectives]

    @property
    def objective_units(self) -> List[str]:
        return [o.unit for o in self.objectives]

    def evaluate_unphysical_2d(self, pts_2d: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_2d, dtype=np.float64)
        if self.unphysical_pair_rule is None or pts.shape[-1] < 2:
            return np.zeros(len(pts), dtype=bool)
        rule = self.unphysical_pair_rule.replace(" ", "").lower()
        if rule in ("x>=y", "axis0>=axis1", "homo>=lumo"):
            return pts[:, 0] >= pts[:, 1]
        if rule in ("x>y", "axis0>axis1"):
            return pts[:, 0] > pts[:, 1]
        if rule in ("x<=y", "axis0<=axis1"):
            return pts[:, 0] <= pts[:, 1]
        raise ValueError(f"Unsupported unphysical_pair_rule: '{self.unphysical_pair_rule}'")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskWeavingSpec":
        objs = tuple(
            ObjectiveChannelSpec(
                name=str(o["name"]),
                asset_id=str(o["asset_id"]),
                asset_channel=str(o.get("asset_channel", o["name"])),
                mode=str(o.get("mode", "point")),
                tolerance=float(o.get("tolerance", 0.25)),
                unit=str(o.get("unit", "unspecified")),
                weight=float(o.get("weight", 1.0)),
                interval_bounds=(
                    tuple(o["interval_bounds"]) if o.get("interval_bounds") is not None else None  # type: ignore[arg-type]
                ),
            )
            for o in raw["objectives"]
        )
        cons = tuple(
            ConstraintChannelSpec(
                name=str(c["name"]),
                asset_id=str(c.get("asset_id", "builtin_rules")),
                asset_channel=str(c["asset_channel"]),
                op=str(c["op"]),
                threshold=(
                    (float(c["threshold"][0]), float(c["threshold"][1]))
                    if isinstance(c["threshold"], (list, tuple))
                    else float(c["threshold"])
                ),
                soft_margin=float(c.get("soft_margin", 0.50)),
                weight=float(c.get("weight", 0.60)),
                hard_gate=bool(c.get("hard_gate", True)),
            )
            for c in raw.get("constraints", [])
        )
        auds = tuple(
            AuditorChannelSpec(
                name=str(a["name"]),
                objective_name=str(a["objective_name"]),
                mode=str(a.get("mode", "cross_oracle")),
                auditor_asset_id=a.get("auditor_asset_id"),
                auditor_channel=a.get("auditor_channel"),
                disagreement_cap=float(a.get("disagreement_cap", 0.35)),
                tanimoto_cutoff=float(a.get("tanimoto_cutoff", 0.25)),
            )
            for a in raw.get("auditors", [])
        )
        return cls(
            task_name=str(raw["task_name"]),
            description=str(raw.get("description", "")),
            assets=tuple(raw.get("assets", [])),
            potential_asset_id=raw.get("potential_asset_id"),
            objectives=objs,
            constraints=cons,
            auditors=auds,
            unphysical_pair_rule=raw.get("unphysical_pair_rule"),
            reconcile_valence=bool(raw.get("reconcile_valence", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "description": self.description,
            "potential_asset_id": self.potential_asset_id,
            "reconcile_valence": self.reconcile_valence,
            "unphysical_pair_rule": self.unphysical_pair_rule,
            "assets": list(self.assets),
            "objectives": [
                {
                    "name": o.name,
                    "asset_id": o.asset_id,
                    "asset_channel": o.asset_channel,
                    "mode": o.mode,
                    "tolerance": o.tolerance,
                    "unit": o.unit,
                    "weight": o.weight,
                    "interval_bounds": list(o.interval_bounds) if o.interval_bounds else None,
                }
                for o in self.objectives
            ],
            "constraints": [
                {
                    "name": c.name,
                    "asset_id": c.asset_id,
                    "asset_channel": c.asset_channel,
                    "op": c.op,
                    "threshold": list(c.threshold) if isinstance(c.threshold, tuple) else c.threshold,
                    "soft_margin": c.soft_margin,
                    "weight": c.weight,
                    "hard_gate": c.hard_gate,
                }
                for c in self.constraints
            ],
            "auditors": [
                {
                    "name": a.name,
                    "objective_name": a.objective_name,
                    "mode": a.mode,
                    "auditor_asset_id": a.auditor_asset_id,
                    "auditor_channel": a.auditor_channel,
                    "disagreement_cap": a.disagreement_cap,
                    "tanimoto_cutoff": a.tanimoto_cutoff,
                }
                for a in self.auditors
            ],
        }

    @classmethod
    def from_file(cls, path: Path | str) -> "TaskWeavingSpec":
        p = Path(path)
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        return cls.from_dict(raw)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if p.suffix.lower() in (".yaml", ".yml"):
            p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        else:
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
