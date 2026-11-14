"""Unified five-dimensional calibration, concentration, physical internalization, and audit metrics."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import numpy as np


def compute_loom_metrics(
    requested_targets_eV: Sequence[float],
    reward_predictions_eV: Sequence[float],
    legal_mask: Sequence[bool],
    evaluated_mask: Sequence[bool],
    *,
    audit_predictions_eV: Optional[Sequence[float]] = None,
    audit_evaluated_mask: Optional[Sequence[bool]] = None,
    f_max_eV_A: Optional[Sequence[float]] = None,
    unseen_charge_mask: Optional[Sequence[bool]] = None,
    tolerance_eV: float = 0.25,
) -> Dict[str, Any]:
    """Compute the unified 5-dimensional Uni-Loom metrics over ALL attempted proposals."""
    req = np.asarray(requested_targets_eV, dtype=np.float64)
    rew = np.asarray(reward_predictions_eV, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=bool)
    ev = np.asarray(evaluated_mask, dtype=bool) & np.isfinite(rew)

    n_total = len(req)
    if n_total == 0:
        raise ValueError("Cannot compute metrics on empty proposal list")

    n_legal = int(legal.sum())
    n_eval = int(ev.sum())

    # 1. Calibration Slope & Endpoint Bias + 2. Concentration (Within-Target Spread)
    slope = float("nan")
    intercept = float("nan")
    r2 = float("nan")
    within_target_std = float("nan")
    conditional_mae = float("nan")
    per_target_summary: Dict[str, Dict[str, float]] = {}

    if n_eval >= 2:
        req_ev = req[ev]
        rew_ev = rew[ev]
        conditional_mae = float(np.mean(np.abs(rew_ev - req_ev)))
        if np.std(req_ev) > 1e-8:
            slope_val, intercept_val = np.polyfit(req_ev, rew_ev, 1)
            slope = float(slope_val)
            intercept = float(intercept_val)
            corr = np.corrcoef(req_ev, rew_ev)[0, 1]
            r2 = float(corr ** 2) if np.isfinite(corr) else float("nan")

        unique_targets = sorted(set(np.round(req, 6).tolist()))
        pooled_vars = []
        pooled_weights = []
        for t_val in unique_targets:
            t_mask = np.isclose(req, t_val, atol=1e-5)
            t_ev = t_mask & ev
            t_n_total = int(t_mask.sum())
            t_n_ev = int(t_ev.sum())
            if t_n_ev > 0:
                vals = rew[t_ev]
                t_mean = float(np.mean(vals))
                t_std = float(np.std(vals, ddof=0))
                t_mae = float(np.mean(np.abs(vals - t_val)))
                t_hits = int(np.sum(np.abs(vals - t_val) <= tolerance_eV))
                if t_n_ev >= 2:
                    pooled_vars.append(t_std ** 2)
                    pooled_weights.append(t_n_ev)
            else:
                t_mean = t_std = t_mae = float("nan")
                t_hits = 0
            per_target_summary[f"{t_val:.4f}"] = {
                "target_eV": float(t_val),
                "attempted": t_n_total,
                "evaluated": t_n_ev,
                "mean_pred_eV": t_mean,
                "bias_eV": float(t_mean - t_val) if np.isfinite(t_mean) else float("nan"),
                "std_eV": t_std,
                "mae_eV": t_mae,
                "hits": t_hits,
                "hit_yield": t_hits / max(t_n_total, 1),
            }
        if pooled_vars:
            within_target_std = float(np.sqrt(np.average(pooled_vars, weights=pooled_weights)))

    # 3. Physical Validity & Internalization
    legal_rate = n_legal / n_total
    evaluated_rate = n_eval / n_total
    unseen_charge_rate = (
        float(np.mean(np.asarray(unseen_charge_mask, dtype=bool)[legal]))
        if (unseen_charge_mask is not None and n_legal > 0)
        else None
    )
    f_max_median = None
    f_max_p90 = None
    if f_max_eV_A is not None:
        f_arr = np.asarray(f_max_eV_A, dtype=np.float64)
        valid_f = f_arr[legal & np.isfinite(f_arr)]
        if len(valid_f) > 0:
            f_max_median = float(np.median(valid_f))
            f_max_p90 = float(np.percentile(valid_f, 90))

    # 4. End-to-End Hit Yield (Denominator = ALL attempted proposals!)
    hits_mask = ev & (np.abs(rew - req) <= tolerance_eV)
    n_hits = int(hits_mask.sum())
    hit_yield = n_hits / n_total
    conditional_hit_rate = n_hits / max(n_eval, 1)

    # 5. Cross-Foundation Audit Gap (Anti-Reward-Hacking Check)
    audit_report = None
    if audit_predictions_eV is not None:
        aud = np.asarray(audit_predictions_eV, dtype=np.float64)
        aud_ev = (
            np.asarray(audit_evaluated_mask, dtype=bool)
            if audit_evaluated_mask is not None
            else np.isfinite(aud)
        ) & np.isfinite(aud)
        both = ev & aud_ev
        n_both = int(both.sum())
        if n_both > 0:
            aud_mae = float(np.mean(np.abs(aud[both] - req[both])))
            rew_both_mae = float(np.mean(np.abs(rew[both] - req[both])))
            aud_hits = int(np.sum(np.abs(aud[aud_ev] - req[aud_ev]) <= tolerance_eV))
            aud_slope = (
                float(np.polyfit(req[both], aud[both], 1)[0])
                if n_both >= 2 and np.std(req[both]) > 1e-8
                else float("nan")
            )
            audit_report = {
                "audit_evaluated": int(aud_ev.sum()),
                "paired_evaluated": n_both,
                "audit_mae_eV": aud_mae,
                "reward_paired_mae_eV": rew_both_mae,
                "audit_minus_reward_mae_gap_eV": float(aud_mae - rew_both_mae),
                "mean_reward_minus_audit_pred_eV": float(np.mean(rew[both] - aud[both])),
                "reward_audit_disagreement_mae_eV": float(np.mean(np.abs(rew[both] - aud[both]))),
                "audit_slope": aud_slope,
                "audit_hits": aud_hits,
                "audit_hit_yield": aud_hits / n_total,
            }

    return {
        "attempted": n_total,
        "legal": n_legal,
        "evaluated": n_eval,
        "legal_rate": legal_rate,
        "evaluated_rate": evaluated_rate,
        "calibration_slope": slope,
        "calibration_intercept_eV": intercept,
        "calibration_r2": r2,
        "within_target_std_eV": within_target_std,
        "conditional_mae_eV": conditional_mae,
        "tolerance_eV": tolerance_eV,
        "hits": n_hits,
        "hit_yield": hit_yield,
        "conditional_hit_rate": conditional_hit_rate,
        "unseen_formal_charge_rate_among_legal": unseen_charge_rate,
        "zero_step_f_max_median_eV_A": f_max_median,
        "zero_step_f_max_p90_eV_A": f_max_p90,
        "per_target": per_target_summary,
        "audit": audit_report,
    }
