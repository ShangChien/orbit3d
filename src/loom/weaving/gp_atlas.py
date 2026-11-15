"""Generalized Predictor-Agnostic Heteroscedastic GP Reachability, Dual-Uncertainty & Gatekeeper Atlas.

Separates two distinct uncertainty sources for end-users, enterprise asset alignment, and Grounded RSI:
  1. Generator Dispersion (`sigma_gen(t)`): intrinsic conditional spread of generated molecules around `mu_gen(t)`.
  2. Predictor Calibration Uncertainty (`sigma_cal(t)`): cross-foundation disagreement (`|y_rew - y_aud|`),
     residual GP posterior standard deviation, or single-oracle applicability-domain OOD uncertainty.

Supports arbitrary 1D and 2D property axes, optional physical-feasibility masks (e.g. `x >= y` only when
configured for orbital pairs like `HOMO / LUMO`, or `None` for unconstrained property pairs), and a
Pre-Generation Gatekeeper (`recommend_target`) while preserving full backward-compatible dict keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.special import expit, logit
from scipy.stats import gaussian_kde, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def _gaussian_box_hit_prob(mean: np.ndarray, target: np.ndarray, sigma: np.ndarray, tol: float) -> np.ndarray:
    s = np.maximum(sigma, 1e-4)
    upper = (target + tol - mean) / s
    lower = (target - tol - mean) / s
    return np.clip(norm.cdf(upper) - norm.cdf(lower), 0.0, 1.0)


def _eval_unphysical_rule(pts_2d: np.ndarray, rule: Optional[str]) -> np.ndarray:
    if rule is None:
        return np.zeros(len(pts_2d), dtype=bool)
    r = rule.replace(" ", "").lower()
    if r in ("x>=y", "axis0>=axis1", "homo>=lumo"):
        return pts_2d[:, 0] >= pts_2d[:, 1]
    if r in ("x>y", "axis0>axis1"):
        return pts_2d[:, 0] > pts_2d[:, 1]
    if r in ("x<=y", "axis0<=axis1"):
        return pts_2d[:, 0] <= pts_2d[:, 1]
    return np.zeros(len(pts_2d), dtype=bool)


@dataclass
class GridPointSummary1D:
    target_eV: float
    attempted: int
    evaluated: int
    eval_rate: float
    mean_pred_eV: float
    std_pred_eV: float
    cal_rmse_eV: float


class HeteroscedasticGPAtlas1D:
    """Predictor-agnostic 1D Heteroscedastic GP Reachability, Dual-Uncertainty & Gatekeeper Map."""

    def __init__(
        self,
        length_scale: float = 0.60,
        constant_cal_sigma_eV: float = 0.19,
        *,
        axis_name: str = "target",
        unit: str = "eV",
    ) -> None:
        self.length_scale = float(length_scale)
        self.constant_cal_sigma_eV = float(constant_cal_sigma_eV)
        self.axis_name = str(axis_name)
        self.unit = str(unit)
        self.gp_mean: Optional[GaussianProcessRegressor] = None
        self.gp_log_std: Optional[GaussianProcessRegressor] = None
        self.gp_logit_eval: Optional[GaussianProcessRegressor] = None
        self.gp_log_cal: Optional[GaussianProcessRegressor] = None
        self.grid_summaries: List[GridPointSummary1D] = []
        self.source_kde: Optional[gaussian_kde] = None
        self.target_min: float = -10.0
        self.target_max: float = 0.0

    @classmethod
    def fit_from_proposals(
        cls,
        requested_targets_eV: Sequence[float],
        reward_preds_eV: Sequence[float],
        evaluated_mask: Sequence[bool],
        *,
        audit_or_dft_preds_eV: Optional[Sequence[float]] = None,
        sigma_cal_samples: Optional[Sequence[float]] = None,
        source_property_values_eV: Optional[Sequence[float]] = None,
        constant_cal_sigma_eV: float = 0.19,
        axis_name: str = "target",
        unit: str = "eV",
    ) -> "HeteroscedasticGPAtlas1D":
        atlas = cls(constant_cal_sigma_eV=constant_cal_sigma_eV, axis_name=axis_name, unit=unit)
        req = np.asarray(requested_targets_eV, dtype=np.float64)
        rew = np.asarray(reward_preds_eV, dtype=np.float64)
        ev = np.asarray(evaluated_mask, dtype=bool) & np.isfinite(rew)
        aud = np.asarray(audit_or_dft_preds_eV, dtype=np.float64) if audit_or_dft_preds_eV is not None else None
        cal_direct = np.asarray(sigma_cal_samples, dtype=np.float64) if sigma_cal_samples is not None else None

        if source_property_values_eV is not None and len(source_property_values_eV) >= 10:
            src_arr = np.asarray(source_property_values_eV, dtype=np.float64)
            src_arr = src_arr[np.isfinite(src_arr)]
            if len(src_arr) >= 10:
                atlas.source_kde = gaussian_kde(src_arr)

        unique_t = np.array(sorted(set(np.round(req, 5).tolist())), dtype=np.float64)
        atlas.target_min = float(unique_t.min())
        atlas.target_max = float(unique_t.max())

        x_valid: List[float] = []
        y_mean: List[float] = []
        y_sem_var: List[float] = []
        y_log_std: List[float] = []
        x_all: List[float] = []
        y_logit_ev: List[float] = []
        x_cal: List[float] = []
        y_log_cal: List[float] = []

        for t_val in unique_t:
            m = np.isclose(req, t_val, atol=1e-4)
            n_att = int(m.sum())
            m_ev = m & ev
            n_ev = int(m_ev.sum())
            p_ev = np.clip((n_ev + 0.5) / (n_att + 1.0), 1e-3, 1.0 - 1e-3)
            x_all.append(float(t_val))
            y_logit_ev.append(float(logit(p_ev)))

            if n_ev >= 2:
                vals = rew[m_ev]
                mu = float(np.mean(vals))
                sd = float(max(np.std(vals, ddof=1), 0.03))
                x_valid.append(float(t_val))
                y_mean.append(mu)
                y_sem_var.append((sd ** 2) / n_ev)
                y_log_std.append(float(np.log(sd)))
            else:
                mu = float(np.mean(rew[m_ev])) if n_ev == 1 else float("nan")
                sd = float("nan")

            cal_rmse = constant_cal_sigma_eV
            if aud is not None:
                m_both = m_ev & np.isfinite(aud)
                if int(m_both.sum()) >= 2:
                    cal_rmse = float(max(np.sqrt(np.mean((rew[m_both] - aud[m_both]) ** 2)), 0.03))
                    x_cal.append(float(t_val))
                    y_log_cal.append(float(np.log(cal_rmse)))
            elif cal_direct is not None:
                m_cd = m_ev & np.isfinite(cal_direct)
                if int(m_cd.sum()) >= 2:
                    cal_rmse = float(max(np.median(cal_direct[m_cd]), 0.03))
                    x_cal.append(float(t_val))
                    y_log_cal.append(float(np.log(cal_rmse)))

            atlas.grid_summaries.append(
                GridPointSummary1D(
                    target_eV=float(t_val),
                    attempted=n_att,
                    evaluated=n_ev,
                    eval_rate=n_ev / max(n_att, 1),
                    mean_pred_eV=mu,
                    std_pred_eV=sd,
                    cal_rmse_eV=cal_rmse,
                )
            )

        span = max(atlas.target_max - atlas.target_min, 0.5)
        ls_bounds = (0.10 * span, 3.0 * span)

        if len(x_valid) < 2:
            raise ValueError("Need at least 2 grid targets with >=2 evaluated molecules to fit 1D GP Atlas")

        X_v = np.asarray(x_valid, dtype=np.float64)[:, None]
        k_mean = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=0.35 * span, length_scale_bounds=ls_bounds, nu=2.5
        )
        atlas.gp_mean = GaussianProcessRegressor(
            kernel=k_mean,
            alpha=np.asarray(y_sem_var, dtype=np.float64),
            normalize_y=True,
            n_restarts_optimizer=4,
            random_state=0,
        ).fit(X_v, np.asarray(y_mean, dtype=np.float64))

        het_ls_bounds = (0.12 * span, 0.85 * span)
        k_std = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=0.35 * span, length_scale_bounds=het_ls_bounds, nu=1.5
        ) + WhiteKernel(noise_level=0.02, noise_level_bounds=(1e-4, 0.15))
        atlas.gp_log_std = GaussianProcessRegressor(
            kernel=k_std, normalize_y=True, n_restarts_optimizer=3, random_state=0
        ).fit(X_v, np.asarray(y_log_std, dtype=np.float64))

        X_a = np.asarray(x_all, dtype=np.float64)[:, None]
        k_ev = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=0.40 * span, length_scale_bounds=ls_bounds, nu=1.5
        ) + WhiteKernel(noise_level=0.03, noise_level_bounds=(1e-4, 0.25))
        atlas.gp_logit_eval = GaussianProcessRegressor(
            kernel=k_ev, normalize_y=True, n_restarts_optimizer=3, random_state=0
        ).fit(X_a, np.asarray(y_logit_ev, dtype=np.float64))

        if len(x_cal) >= 2:
            X_c = np.asarray(x_cal, dtype=np.float64)[:, None]
            k_cal = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
                length_scale=0.35 * span, length_scale_bounds=het_ls_bounds, nu=1.5
            ) + WhiteKernel(noise_level=0.02, noise_level_bounds=(1e-4, 0.15))
            atlas.gp_log_cal = GaussianProcessRegressor(
                kernel=k_cal, normalize_y=True, n_restarts_optimizer=3, random_state=0
            ).fit(X_c, np.asarray(y_log_cal, dtype=np.float64))

        return atlas

    def query(self, targets_eV: Sequence[float] | np.ndarray, tolerance_eV: float = 0.25) -> Dict[str, Any]:
        assert self.gp_mean is not None and self.gp_log_std is not None and self.gp_logit_eval is not None
        t_arr = np.asarray(targets_eV, dtype=np.float64).reshape(-1)
        X = t_arr[:, None]

        mu_gen, sigma_epistemic = self.gp_mean.predict(X, return_std=True)
        log_sd = self.gp_log_std.predict(X)
        sigma_gen = np.exp(np.clip(log_sd, np.log(0.02), np.log(5.0)))
        p_eval = expit(self.gp_logit_eval.predict(X))

        if self.gp_log_cal is not None:
            sigma_cal = np.exp(np.clip(self.gp_log_cal.predict(X), np.log(0.02), np.log(5.0)))
        else:
            sigma_cal = np.full_like(t_arr, self.constant_cal_sigma_eV)

        sigma_total = np.sqrt(sigma_gen ** 2 + sigma_cal ** 2 + sigma_epistemic ** 2)
        p_hit_reward = p_eval * _gaussian_box_hit_prob(
            mu_gen, t_arr, np.sqrt(sigma_gen ** 2 + sigma_epistemic ** 2), tolerance_eV
        )
        p_hit_dft = p_eval * _gaussian_box_hit_prob(mu_gen, t_arr, sigma_total, tolerance_eV)
        bias = mu_gen - t_arr
        src_density = self.source_kde(t_arr) if self.source_kde is not None else np.zeros_like(t_arr)

        zones: List[str] = []
        for idx in range(len(t_arr)):
            if p_hit_reward[idx] >= 0.16 and abs(bias[idx]) <= 1.40 * tolerance_eV:
                zones.append("calibrated_reachable")
            elif p_hit_reward[idx] >= 0.08 and abs(bias[idx]) <= 2.20 * tolerance_eV:
                zones.append("boundary_degrading")
            else:
                zones.append("unreachable_ood")

        reachable_mask = np.array([z == "calibrated_reachable" for z in zones], dtype=bool)
        trusted_interval = (
            [float(t_arr[reachable_mask].min()), float(t_arr[reachable_mask].max())]
            if reachable_mask.any()
            else None
        )

        return {
            "axis_name": self.axis_name,
            "unit": self.unit,
            "targets": t_arr.tolist(),
            "mu_gen": mu_gen.tolist(),
            "bias": bias.tolist(),
            "sigma_gen": sigma_gen.tolist(),
            "sigma_cal": sigma_cal.tolist(),
            "sigma_total": sigma_total.tolist(),
            # Legacy keys preserved for seamless v1.x compatibility
            "targets_eV": t_arr.tolist(),
            "mu_gen_eV": mu_gen.tolist(),
            "bias_eV": bias.tolist(),
            "sigma_gen_eV": sigma_gen.tolist(),
            "sigma_epistemic_eV": sigma_epistemic.tolist(),
            "sigma_cal_eV": sigma_cal.tolist(),
            "sigma_total_eV": sigma_total.tolist(),
            "p_eval": p_eval.tolist(),
            "p_hit_reward": p_hit_reward.tolist(),
            "p_hit_dft_calibrated": p_hit_dft.tolist(),
            "source_kde_density": src_density.tolist(),
            "zones": zones,
            "trusted_reachable_interval": trusted_interval,
            "trusted_reachable_interval_eV": trusted_interval,
            "tolerance": tolerance_eV,
            "tolerance_eV": tolerance_eV,
            "empirical_grid": [vars(g) for g in self.grid_summaries],
        }

    def recommend_target(
        self,
        requested_target: float,
        *,
        tolerance: float = 0.25,
        cal_sigma_cap: float = 0.45,
        n_scan: int = 161,
    ) -> Dict[str, Any]:
        """Pre-Generation Gatekeeper: verify target feasibility and suggest nearest trusted Pareto target if OOD."""
        t_req = float(requested_target)
        q_one = self.query([t_req], tolerance_eV=tolerance)
        zone = q_one["zones"][0]
        sig_cal = float(q_one["sigma_cal"][0])
        p_hit = float(q_one["p_hit_dft_calibrated"][0])

        if zone == "calibrated_reachable" and sig_cal <= cal_sigma_cap:
            return {
                "status": "trusted",
                "requested_target": t_req,
                "recommended_target": t_req,
                "expected_calibrated_hit_prob": p_hit,
                "sigma_cal": sig_cal,
                "warning": None,
            }

        grid = np.linspace(self.target_min, self.target_max, n_scan)
        q_grid = self.query(grid, tolerance_eV=tolerance)
        trusted_idx = [
            i
            for i in range(n_scan)
            if q_grid["zones"][i] == "calibrated_reachable" and q_grid["sigma_cal"][i] <= cal_sigma_cap
        ]
        if not trusted_idx:
            # Fallback to maximum calibrated hit probability point
            best_i = int(np.argmax(q_grid["p_hit_dft_calibrated"]))
        else:
            best_i = min(trusted_idx, key=lambda i: abs(float(grid[i]) - t_req))

        rec_t = float(grid[best_i])
        reason = "predictor_blindspot" if sig_cal > cal_sigma_cap else zone
        return {
            "status": "rerouted_warning",
            "reason": reason,
            "requested_target": t_req,
            "recommended_target": rec_t,
            "requested_calibrated_hit_prob": p_hit,
            "recommended_calibrated_hit_prob": float(q_grid["p_hit_dft_calibrated"][best_i]),
            "sigma_cal": sig_cal,
            "warning": (
                f"Requested {self.axis_name}={t_req:.3f} {self.unit} falls in '{reason}' "
                f"(sigma_cal={sig_cal:.3f}, P_hit={p_hit*100:.1f}%); "
                f"nearest trusted reachable target is {rec_t:.3f} {self.unit}."
            ),
        }


class HeteroscedasticGPAtlas2D:
    """Generalized 2D Joint Heteroscedastic GP Reachability, Dual-Uncertainty & Gatekeeper Atlas."""

    def __init__(
        self,
        constant_cal_sigma_eV: Tuple[float, float] = (0.19, 0.21),
        *,
        axis_names: Tuple[str, str] = ("homo", "lumo"),
        units: Tuple[str, str] = ("eV", "eV"),
        unphysical_rule: Optional[str] = "DEFAULT",
    ) -> None:
        self.constant_cal_sigma_eV = constant_cal_sigma_eV
        self.axis_names = (str(axis_names[0]), str(axis_names[1]))
        self.units = (str(units[0]), str(units[1]))
        if unphysical_rule == "DEFAULT":
            # Only apply `x >= y` automatically for canonical `(homo, lumo)` orbital pairs
            self.unphysical_rule: Optional[str] = (
                "x>=y" if tuple(a.lower() for a in self.axis_names) == ("homo", "lumo") else None
            )
        else:
            self.unphysical_rule = unphysical_rule

        self.gp_mu_h: Optional[GaussianProcessRegressor] = None
        self.gp_mu_l: Optional[GaussianProcessRegressor] = None
        self.gp_log_sd_h: Optional[GaussianProcessRegressor] = None
        self.gp_log_sd_l: Optional[GaussianProcessRegressor] = None
        self.gp_logit_eval: Optional[GaussianProcessRegressor] = None
        self.gp_log_cal_h: Optional[GaussianProcessRegressor] = None
        self.gp_log_cal_l: Optional[GaussianProcessRegressor] = None
        self.source_kde: Optional[gaussian_kde] = None
        self.empirical_cells: List[Dict[str, Any]] = []
        self.bounds_x: Tuple[float, float] = (-10.0, 0.0)
        self.bounds_y: Tuple[float, float] = (-5.0, 5.0)

    @classmethod
    def fit_from_proposals(
        cls,
        requested_targets_2d_eV: np.ndarray,
        reward_preds_2d_eV: np.ndarray,
        evaluated_mask: Sequence[bool],
        *,
        audit_or_dft_preds_2d_eV: Optional[np.ndarray] = None,
        sigma_cal_samples_2d: Optional[np.ndarray] = None,
        source_2d_eV: Optional[np.ndarray] = None,
        constant_cal_sigma_eV: Tuple[float, float] = (0.19, 0.21),
        axis_names: Tuple[str, str] = ("homo", "lumo"),
        units: Tuple[str, str] = ("eV", "eV"),
        unphysical_rule: Optional[str] = "DEFAULT",
        tolerances: Tuple[float, float] = (0.25, 0.25),
    ) -> "HeteroscedasticGPAtlas2D":
        atlas = cls(
            constant_cal_sigma_eV=constant_cal_sigma_eV,
            axis_names=axis_names,
            units=units,
            unphysical_rule=unphysical_rule,
        )
        req = np.asarray(requested_targets_2d_eV, dtype=np.float64)
        rew = np.asarray(reward_preds_2d_eV, dtype=np.float64)
        ev = np.asarray(evaluated_mask, dtype=bool) & np.isfinite(rew).all(axis=-1)
        aud = np.asarray(audit_or_dft_preds_2d_eV, dtype=np.float64) if audit_or_dft_preds_2d_eV is not None else None
        cal_dir = np.asarray(sigma_cal_samples_2d, dtype=np.float64) if sigma_cal_samples_2d is not None else None

        if source_2d_eV is not None and len(source_2d_eV) >= 20:
            s2 = np.asarray(source_2d_eV, dtype=np.float64)
            s2 = s2[np.isfinite(s2).all(axis=-1)]
            if len(s2) >= 20:
                if len(s2) > 8000:
                    rng = np.random.default_rng(0)
                    s2 = s2[rng.choice(len(s2), size=8000, replace=False)]
                atlas.source_kde = gaussian_kde(s2.T)

        rounded = np.round(req, 4)
        unique_cells = np.unique(rounded, axis=0)
        atlas.bounds_x = (float(unique_cells[:, 0].min()), float(unique_cells[:, 0].max()))
        atlas.bounds_y = (float(unique_cells[:, 1].min()), float(unique_cells[:, 1].max()))

        X_v: List[np.ndarray] = []
        mu_h_list, mu_l_list = [], []
        sem_h_list, sem_l_list = [], []
        lsd_h_list, lsd_l_list = [], []
        X_all: List[np.ndarray] = []
        logit_ev_list: List[float] = []
        X_cal: List[np.ndarray] = []
        lcal_h_list, lcal_l_list = [], []

        tol_x, tol_y = float(tolerances[0]), float(tolerances[1])

        for cell in unique_cells:
            m = np.isclose(req[:, 0], cell[0], atol=1e-3) & np.isclose(req[:, 1], cell[1], atol=1e-3)
            n_att = int(m.sum())
            m_ev = m & ev
            n_ev = int(m_ev.sum())
            p_ev = np.clip((n_ev + 0.5) / (n_att + 1.0), 1e-3, 1.0 - 1e-3)
            X_all.append(cell)
            logit_ev_list.append(float(logit(p_ev)))

            if n_ev >= 2:
                vals = rew[m_ev]
                mh, ml = float(np.mean(vals[:, 0])), float(np.mean(vals[:, 1]))
                sh = float(max(np.std(vals[:, 0], ddof=1), 0.03))
                sl = float(max(np.std(vals[:, 1], ddof=1), 0.03))
                hits_2d = int(
                    np.sum((np.abs(vals[:, 0] - cell[0]) <= tol_x) & (np.abs(vals[:, 1] - cell[1]) <= tol_y))
                )
                X_v.append(cell)
                mu_h_list.append(mh)
                mu_l_list.append(ml)
                sem_h_list.append((sh ** 2) / n_ev)
                sem_l_list.append((sl ** 2) / n_ev)
                lsd_h_list.append(float(np.log(sh)))
                lsd_l_list.append(float(np.log(sl)))
            else:
                mh = ml = sh = sl = float("nan")
                hits_2d = 0

            cal_h = float(constant_cal_sigma_eV[0])
            cal_l = float(constant_cal_sigma_eV[1])
            if aud is not None:
                m_both = m_ev & np.isfinite(aud).all(axis=-1)
                if int(m_both.sum()) >= 2:
                    cal_h = float(max(np.sqrt(np.mean((rew[m_both, 0] - aud[m_both, 0]) ** 2)), 0.03))
                    cal_l = float(max(np.sqrt(np.mean((rew[m_both, 1] - aud[m_both, 1]) ** 2)), 0.03))
                    X_cal.append(cell)
                    lcal_h_list.append(float(np.log(cal_h)))
                    lcal_l_list.append(float(np.log(cal_l)))
            elif cal_dir is not None:
                m_cd = m_ev & np.isfinite(cal_dir).all(axis=-1)
                if int(m_cd.sum()) >= 2:
                    cal_h = float(max(np.sqrt(np.mean(cal_dir[m_cd, 0] ** 2)), 0.03))
                    cal_l = float(max(np.sqrt(np.mean(cal_dir[m_cd, 1] ** 2)), 0.03))
                    X_cal.append(cell)
                    lcal_h_list.append(float(np.log(cal_h)))
                    lcal_l_list.append(float(np.log(cal_l)))

            atlas.empirical_cells.append(
                {
                    "target_x": float(cell[0]),
                    "target_y": float(cell[1]),
                    "homo_target_eV": float(cell[0]),
                    "lumo_target_eV": float(cell[1]),
                    "attempted": n_att,
                    "evaluated": n_ev,
                    "eval_rate": n_ev / max(n_att, 1),
                    "mean_x": mh,
                    "mean_y": ml,
                    "mean_homo_eV": mh,
                    "mean_lumo_eV": ml,
                    "std_x": sh,
                    "std_y": sl,
                    "std_homo_eV": sh,
                    "std_lumo_eV": sl,
                    "cal_rmse_x": cal_h,
                    "cal_rmse_y": cal_l,
                    "cal_rmse_homo_eV": cal_h,
                    "cal_rmse_lumo_eV": cal_l,
                    "cal_rmse_2d_eV": float(np.sqrt(0.5 * (cal_h ** 2 + cal_l ** 2))),
                    "joint_hits_025eV": hits_2d,
                    "joint_hit_yield": hits_2d / max(n_att, 1),
                }
            )

        if len(X_v) < 4:
            raise ValueError("Need at least 4 populated 2D grid cells to fit 2D GP Atlas")

        Xv_arr = np.asarray(X_v, dtype=np.float64)
        Xa_arr = np.asarray(X_all, dtype=np.float64)

        def _make_2d_kernel(noise: bool = False) -> Any:
            base = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
                length_scale=[0.6, 0.6], length_scale_bounds=(0.15, 4.0), nu=2.5
            )
            if noise:
                base = base + WhiteKernel(noise_level=0.03, noise_level_bounds=(1e-4, 1.5))
            return base

        atlas.gp_mu_h = GaussianProcessRegressor(
            kernel=_make_2d_kernel(False), alpha=np.asarray(sem_h_list), normalize_y=True, n_restarts_optimizer=3, random_state=0
        ).fit(Xv_arr, np.asarray(mu_h_list))
        atlas.gp_mu_l = GaussianProcessRegressor(
            kernel=_make_2d_kernel(False), alpha=np.asarray(sem_l_list), normalize_y=True, n_restarts_optimizer=3, random_state=0
        ).fit(Xv_arr, np.asarray(mu_l_list))
        atlas.gp_log_sd_h = GaussianProcessRegressor(
            kernel=_make_2d_kernel(True), normalize_y=True, n_restarts_optimizer=2, random_state=0
        ).fit(Xv_arr, np.asarray(lsd_h_list))
        atlas.gp_log_sd_l = GaussianProcessRegressor(
            kernel=_make_2d_kernel(True), normalize_y=True, n_restarts_optimizer=2, random_state=0
        ).fit(Xv_arr, np.asarray(lsd_l_list))
        atlas.gp_logit_eval = GaussianProcessRegressor(
            kernel=_make_2d_kernel(True), normalize_y=True, n_restarts_optimizer=2, random_state=0
        ).fit(Xa_arr, np.asarray(logit_ev_list))

        if len(X_cal) >= 4:
            Xc_arr = np.asarray(X_cal, dtype=np.float64)
            atlas.gp_log_cal_h = GaussianProcessRegressor(
                kernel=_make_2d_kernel(True), normalize_y=True, n_restarts_optimizer=2, random_state=0
            ).fit(Xc_arr, np.asarray(lcal_h_list))
            atlas.gp_log_cal_l = GaussianProcessRegressor(
                kernel=_make_2d_kernel(True), normalize_y=True, n_restarts_optimizer=2, random_state=0
            ).fit(Xc_arr, np.asarray(lcal_l_list))

        return atlas

    def query_grid(
        self,
        homo_grid_eV: Sequence[float] | np.ndarray,
        lumo_grid_eV: Sequence[float] | np.ndarray,
        tolerance_eV: float | Tuple[float, float] = 0.25,
    ) -> Dict[str, Any]:
        assert (
            self.gp_mu_h is not None
            and self.gp_mu_l is not None
            and self.gp_log_sd_h is not None
            and self.gp_log_sd_l is not None
            and self.gp_logit_eval is not None
        )
        if isinstance(tolerance_eV, (tuple, list)):
            tol_x, tol_y = float(tolerance_eV[0]), float(tolerance_eV[1])
        else:
            tol_x = tol_y = float(tolerance_eV)

        h_vec = np.asarray(homo_grid_eV, dtype=np.float64)
        l_vec = np.asarray(lumo_grid_eV, dtype=np.float64)
        hh, ll = np.meshgrid(h_vec, l_vec, indexing="ij")
        pts = np.stack([hh.ravel(), ll.ravel()], axis=-1)

        mu_h, ep_h = self.gp_mu_h.predict(pts, return_std=True)
        mu_l, ep_l = self.gp_mu_l.predict(pts, return_std=True)
        sd_h = np.exp(np.clip(self.gp_log_sd_h.predict(pts), np.log(0.02), np.log(5.0)))
        sd_l = np.exp(np.clip(self.gp_log_sd_l.predict(pts), np.log(0.02), np.log(5.0)))
        p_eval = expit(self.gp_logit_eval.predict(pts))

        if self.gp_log_cal_h is not None and self.gp_log_cal_l is not None:
            cal_h = np.exp(np.clip(self.gp_log_cal_h.predict(pts), np.log(0.03), np.log(3.0)))
            cal_l = np.exp(np.clip(self.gp_log_cal_l.predict(pts), np.log(0.03), np.log(3.0)))
        else:
            cal_h = np.full(len(pts), self.constant_cal_sigma_eV[0], dtype=np.float64)
            cal_l = np.full(len(pts), self.constant_cal_sigma_eV[1], dtype=np.float64)
        cal_2d = np.sqrt(0.5 * (cal_h ** 2 + cal_l ** 2))

        drift_h = mu_h - pts[:, 0]
        drift_l = mu_l - pts[:, 1]

        unphysical = _eval_unphysical_rule(pts, self.unphysical_rule)
        p_h_rew = _gaussian_box_hit_prob(mu_h, pts[:, 0], np.sqrt(sd_h ** 2 + ep_h ** 2), tol_x)
        p_l_rew = _gaussian_box_hit_prob(mu_l, pts[:, 1], np.sqrt(sd_l ** 2 + ep_l ** 2), tol_y)
        p_joint_reward = np.where(unphysical, 0.0, p_eval * p_h_rew * p_l_rew)

        tot_h = np.sqrt(sd_h ** 2 + ep_h ** 2 + cal_h ** 2)
        tot_l = np.sqrt(sd_l ** 2 + ep_l ** 2 + cal_l ** 2)
        p_joint_dft = np.where(
            unphysical,
            0.0,
            p_eval * _gaussian_box_hit_prob(mu_h, pts[:, 0], tot_h, tol_x) * _gaussian_box_hit_prob(mu_l, pts[:, 1], tot_l, tol_y),
        )

        kde_vals = self.source_kde(pts.T) if self.source_kde is not None else np.zeros(len(pts))
        shape = (len(h_vec), len(l_vec))
        return {
            "axis_names": list(self.axis_names),
            "units": list(self.units),
            "unphysical_rule": self.unphysical_rule,
            "axis_x_grid": h_vec.tolist(),
            "axis_y_grid": l_vec.tolist(),
            "mu_x": mu_h.reshape(shape).tolist(),
            "mu_y": mu_l.reshape(shape).tolist(),
            "drift_x": drift_h.reshape(shape).tolist(),
            "drift_y": drift_l.reshape(shape).tolist(),
            "sigma_gen_x": sd_h.reshape(shape).tolist(),
            "sigma_gen_y": sd_l.reshape(shape).tolist(),
            "sigma_cal_x": cal_h.reshape(shape).tolist(),
            "sigma_cal_y": cal_l.reshape(shape).tolist(),
            "sigma_cal_2d": cal_2d.reshape(shape).tolist(),
            # Legacy v1.x keys preserved
            "homo_grid_eV": h_vec.tolist(),
            "lumo_grid_eV": l_vec.tolist(),
            "mu_homo_eV": mu_h.reshape(shape).tolist(),
            "mu_lumo_eV": mu_l.reshape(shape).tolist(),
            "drift_homo_eV": drift_h.reshape(shape).tolist(),
            "drift_lumo_eV": drift_l.reshape(shape).tolist(),
            "sigma_gen_homo_eV": sd_h.reshape(shape).tolist(),
            "sigma_gen_lumo_eV": sd_l.reshape(shape).tolist(),
            "sigma_cal_homo_eV": cal_h.reshape(shape).tolist(),
            "sigma_cal_lumo_eV": cal_l.reshape(shape).tolist(),
            "sigma_cal_2d_eV": cal_2d.reshape(shape).tolist(),
            "has_heteroscedastic_cal_gp": bool(self.gp_log_cal_h is not None),
            "p_eval": p_eval.reshape(shape).tolist(),
            "p_joint_hit_reward": p_joint_reward.reshape(shape).tolist(),
            "p_joint_hit_dft_calibrated": p_joint_dft.reshape(shape).tolist(),
            "unphysical_mask": unphysical.reshape(shape).tolist(),
            "source_kde_density": kde_vals.reshape(shape).tolist(),
            "tolerance_eV": tol_x,
            "tolerances": [tol_x, tol_y],
            "empirical_cells": self.empirical_cells,
        }

    def recommend_target_2d(
        self,
        requested_xy: Tuple[float, float],
        *,
        tolerances: Tuple[float, float] = (0.25, 0.25),
        min_calibrated_hit_prob: float = 0.05,
        cal_sigma_cap: float = 0.48,
        n_grid: int = 41,
    ) -> Dict[str, Any]:
        """2D Pre-Generation Gatekeeper: detects unphysical/blindspot/OOD 2D requests and suggests nearest Pareto target."""
        tx, ty = float(requested_xy[0]), float(requested_xy[1])
        q1 = self.query_grid([tx], [ty], tolerance_eV=tolerances)
        unphys = bool(q1["unphysical_mask"][0][0])
        p_cal = float(q1["p_joint_hit_dft_calibrated"][0][0])
        s_cal = float(q1["sigma_cal_2d"][0][0])

        if (not unphys) and p_cal >= min_calibrated_hit_prob and s_cal <= cal_sigma_cap:
            return {
                "status": "trusted",
                "requested_target": [tx, ty],
                "recommended_target": [tx, ty],
                "expected_calibrated_hit_prob": p_cal,
                "sigma_cal_2d": s_cal,
                "warning": None,
            }

        gx = np.linspace(self.bounds_x[0], self.bounds_x[1], n_grid)
        gy = np.linspace(self.bounds_y[0], self.bounds_y[1], n_grid)
        qg = self.query_grid(gx, gy, tolerance_eV=tolerances)
        p_mat = np.asarray(qg["p_joint_hit_dft_calibrated"])
        s_mat = np.asarray(qg["sigma_cal_2d"])
        u_mat = np.asarray(qg["unphysical_mask"], dtype=bool)

        feasible = (~u_mat) & (p_mat >= min_calibrated_hit_prob) & (s_mat <= cal_sigma_cap)
        xx, yy = np.meshgrid(gx, gy, indexing="ij")
        if feasible.any():
            dists = ((xx - tx) / max(tolerances[0], 1e-3)) ** 2 + ((yy - ty) / max(tolerances[1], 1e-3)) ** 2
            dists = np.where(feasible, dists, np.inf)
            bi, bj = np.unravel_index(int(np.argmin(dists)), dists.shape)
        else:
            bi, bj = np.unravel_index(int(np.argmax(p_mat)), p_mat.shape)

        rec_xy = [float(gx[bi]), float(gy[bj])]
        reason = "unphysical_target" if unphys else ("predictor_blindspot" if s_cal > cal_sigma_cap else "low_reachability_ood")
        return {
            "status": "rerouted_warning",
            "reason": reason,
            "requested_target": [tx, ty],
            "recommended_target": rec_xy,
            "requested_calibrated_hit_prob": p_cal,
            "recommended_calibrated_hit_prob": float(p_mat[bi, bj]),
            "sigma_cal_2d": s_cal,
            "warning": (
                f"Requested ({self.axis_names[0]}={tx:.2f}, {self.axis_names[1]}={ty:.2f}) flagged as '{reason}' "
                f"(P_cal={p_cal*100:.1f}%, sigma_cal={s_cal:.2f}); nearest trusted target is "
                f"({rec_xy[0]:.2f}, {rec_xy[1]:.2f})."
            ),
        }
