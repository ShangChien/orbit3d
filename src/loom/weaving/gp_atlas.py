"""Hierarchical Heteroscedastic Gaussian Process Reachability & Dual-Uncertainty Atlas.

Separates two distinct uncertainty sources for end-users and scientific auditing:
  1. Generator Dispersion (`sigma_gen(t)`): intrinsic conditional spread of generated molecules around `mu_gen(t)`.
  2. Predictor Calibration Uncertainty (`sigma_cal(t)`): discrepancy between `RewardOracle` and `AuditOracle` / `DFT`.

Supports both 1D single-property profiles and 2D `(HOMO, LUMO)` joint feasibility heatmaps
with source-dataset KDE support contours and automatic OOD boundary warnings.
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
    """1D Heteroscedastic GP Reachability & Dual-Uncertainty Map."""

    def __init__(self, length_scale: float = 0.60, constant_cal_sigma_eV: float = 0.19) -> None:
        self.length_scale = float(length_scale)
        self.constant_cal_sigma_eV = float(constant_cal_sigma_eV)
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
        source_property_values_eV: Optional[Sequence[float]] = None,
        constant_cal_sigma_eV: float = 0.19,
    ) -> "HeteroscedasticGPAtlas1D":
        atlas = cls(constant_cal_sigma_eV=constant_cal_sigma_eV)
        req = np.asarray(requested_targets_eV, dtype=np.float64)
        rew = np.asarray(reward_preds_eV, dtype=np.float64)
        ev = np.asarray(evaluated_mask, dtype=bool) & np.isfinite(rew)
        aud = np.asarray(audit_or_dft_preds_eV, dtype=np.float64) if audit_or_dft_preds_eV is not None else None

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
        k_mean = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=0.35 * span, length_scale_bounds=ls_bounds, nu=2.5)
        atlas.gp_mean = GaussianProcessRegressor(
            kernel=k_mean, alpha=np.asarray(y_sem_var, dtype=np.float64), normalize_y=True, n_restarts_optimizer=4, random_state=0
        ).fit(X_v, np.asarray(y_mean, dtype=np.float64))

        k_std = ConstantKernel(1.0, (1e-3, 1e2)) * Matern(length_scale=0.45 * span, length_scale_bounds=ls_bounds, nu=1.5) + WhiteKernel(
            noise_level=0.02, noise_level_bounds=(1e-4, 1.0)
        )
        atlas.gp_log_std = GaussianProcessRegressor(
            kernel=k_std, normalize_y=True, n_restarts_optimizer=3, random_state=0
        ).fit(X_v, np.asarray(y_log_std, dtype=np.float64))

        X_a = np.asarray(x_all, dtype=np.float64)[:, None]
        k_ev = ConstantKernel(1.0, (1e-3, 1e2)) * Matern(length_scale=0.45 * span, length_scale_bounds=ls_bounds, nu=1.5) + WhiteKernel(
            noise_level=0.05, noise_level_bounds=(1e-4, 2.0)
        )
        atlas.gp_logit_eval = GaussianProcessRegressor(
            kernel=k_ev, normalize_y=True, n_restarts_optimizer=3, random_state=0
        ).fit(X_a, np.asarray(y_logit_ev, dtype=np.float64))

        if len(x_cal) >= 2:
            X_c = np.asarray(x_cal, dtype=np.float64)[:, None]
            k_cal = ConstantKernel(1.0, (1e-3, 1e2)) * Matern(length_scale=0.50 * span, length_scale_bounds=ls_bounds, nu=1.5) + WhiteKernel(
                noise_level=0.03, noise_level_bounds=(1e-4, 1.0)
            )
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
        p_hit_reward = p_eval * _gaussian_box_hit_prob(mu_gen, t_arr, np.sqrt(sigma_gen ** 2 + sigma_epistemic ** 2), tolerance_eV)
        p_hit_dft = p_eval * _gaussian_box_hit_prob(mu_gen, t_arr, sigma_total, tolerance_eV)
        bias = mu_gen - t_arr
        src_density = self.source_kde(t_arr) if self.source_kde is not None else np.zeros_like(t_arr)

        zones: List[str] = []
        for idx in range(len(t_arr)):
            if p_hit_reward[idx] >= 0.16 and abs(bias[idx]) <= 0.35:
                zones.append("calibrated_reachable")
            elif p_hit_reward[idx] >= 0.08 and abs(bias[idx]) <= 0.55:
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
            "trusted_reachable_interval_eV": trusted_interval,
            "tolerance_eV": tolerance_eV,
            "empirical_grid": [vars(g) for g in self.grid_summaries],
        }


class HeteroscedasticGPAtlas2D:
    """2D Joint `(HOMO, LUMO)` Heteroscedastic GP Reachability & Dual-Uncertainty Atlas."""

    def __init__(self, constant_cal_sigma_eV: Tuple[float, float] = (0.19, 0.21)) -> None:
        self.constant_cal_sigma_eV = constant_cal_sigma_eV
        self.gp_mu_h: Optional[GaussianProcessRegressor] = None
        self.gp_mu_l: Optional[GaussianProcessRegressor] = None
        self.gp_log_sd_h: Optional[GaussianProcessRegressor] = None
        self.gp_log_sd_l: Optional[GaussianProcessRegressor] = None
        self.gp_logit_eval: Optional[GaussianProcessRegressor] = None
        self.source_kde: Optional[gaussian_kde] = None
        self.empirical_cells: List[Dict[str, Any]] = []

    @classmethod
    def fit_from_proposals(
        cls,
        requested_targets_2d_eV: np.ndarray,
        reward_preds_2d_eV: np.ndarray,
        evaluated_mask: Sequence[bool],
        *,
        source_2d_eV: Optional[np.ndarray] = None,
        constant_cal_sigma_eV: Tuple[float, float] = (0.19, 0.21),
    ) -> "HeteroscedasticGPAtlas2D":
        atlas = cls(constant_cal_sigma_eV=constant_cal_sigma_eV)
        req = np.asarray(requested_targets_2d_eV, dtype=np.float64)
        rew = np.asarray(reward_preds_2d_eV, dtype=np.float64)
        ev = np.asarray(evaluated_mask, dtype=bool) & np.isfinite(rew).all(axis=-1)

        if source_2d_eV is not None and len(source_2d_eV) >= 20:
            s2 = np.asarray(source_2d_eV, dtype=np.float64)
            s2 = s2[np.isfinite(s2).all(axis=-1)]
            if len(s2) >= 20:
                # Subsample up to 8000 points for fast KDE queries
                if len(s2) > 8000:
                    rng = np.random.default_rng(0)
                    s2 = s2[rng.choice(len(s2), size=8000, replace=False)]
                atlas.source_kde = gaussian_kde(s2.T)

        rounded = np.round(req, 4)
        unique_cells = np.unique(rounded, axis=0)

        X_v: List[np.ndarray] = []
        mu_h_list, mu_l_list = [], []
        sem_h_list, sem_l_list = [], []
        lsd_h_list, lsd_l_list = [], []
        X_all: List[np.ndarray] = []
        logit_ev_list: List[float] = []

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
                hits_2d = int(np.sum((np.abs(vals[:, 0] - cell[0]) <= 0.25) & (np.abs(vals[:, 1] - cell[1]) <= 0.25)))
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

            atlas.empirical_cells.append(
                {
                    "homo_target_eV": float(cell[0]),
                    "lumo_target_eV": float(cell[1]),
                    "attempted": n_att,
                    "evaluated": n_ev,
                    "eval_rate": n_ev / max(n_att, 1),
                    "mean_homo_eV": mh,
                    "mean_lumo_eV": ml,
                    "std_homo_eV": sh,
                    "std_lumo_eV": sl,
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

        return atlas

    def query_grid(
        self,
        homo_grid_eV: Sequence[float] | np.ndarray,
        lumo_grid_eV: Sequence[float] | np.ndarray,
        tolerance_eV: float = 0.25,
    ) -> Dict[str, Any]:
        assert (
            self.gp_mu_h is not None
            and self.gp_mu_l is not None
            and self.gp_log_sd_h is not None
            and self.gp_log_sd_l is not None
            and self.gp_logit_eval is not None
        )
        h_vec = np.asarray(homo_grid_eV, dtype=np.float64)
        l_vec = np.asarray(lumo_grid_eV, dtype=np.float64)
        hh, ll = np.meshgrid(h_vec, l_vec, indexing="ij")
        pts = np.stack([hh.ravel(), ll.ravel()], axis=-1)

        mu_h, ep_h = self.gp_mu_h.predict(pts, return_std=True)
        mu_l, ep_l = self.gp_mu_l.predict(pts, return_std=True)
        sd_h = np.exp(np.clip(self.gp_log_sd_h.predict(pts), np.log(0.02), np.log(5.0)))
        sd_l = np.exp(np.clip(self.gp_log_sd_l.predict(pts), np.log(0.02), np.log(5.0)))
        p_eval = expit(self.gp_logit_eval.predict(pts))

        unphysical = pts[:, 0] >= pts[:, 1]  # HOMO >= LUMO is physically forbidden for ground-state closed-shell molecules
        p_h_rew = _gaussian_box_hit_prob(mu_h, pts[:, 0], np.sqrt(sd_h ** 2 + ep_h ** 2), tolerance_eV)
        p_l_rew = _gaussian_box_hit_prob(mu_l, pts[:, 1], np.sqrt(sd_l ** 2 + ep_l ** 2), tolerance_eV)
        p_joint_reward = np.where(unphysical, 0.0, p_eval * p_h_rew * p_l_rew)

        tot_h = np.sqrt(sd_h ** 2 + ep_h ** 2 + self.constant_cal_sigma_eV[0] ** 2)
        tot_l = np.sqrt(sd_l ** 2 + ep_l ** 2 + self.constant_cal_sigma_eV[1] ** 2)
        p_joint_dft = np.where(
            unphysical,
            0.0,
            p_eval * _gaussian_box_hit_prob(mu_h, pts[:, 0], tot_h, tolerance_eV) * _gaussian_box_hit_prob(mu_l, pts[:, 1], tot_l, tolerance_eV),
        )

        kde_vals = self.source_kde(pts.T) if self.source_kde is not None else np.zeros(len(pts))
        shape = (len(h_vec), len(l_vec))
        return {
            "homo_grid_eV": h_vec.tolist(),
            "lumo_grid_eV": l_vec.tolist(),
            "mu_homo_eV": mu_h.reshape(shape).tolist(),
            "mu_lumo_eV": mu_l.reshape(shape).tolist(),
            "sigma_gen_homo_eV": sd_h.reshape(shape).tolist(),
            "sigma_gen_lumo_eV": sd_l.reshape(shape).tolist(),
            "p_eval": p_eval.reshape(shape).tolist(),
            "p_joint_hit_reward": p_joint_reward.reshape(shape).tolist(),
            "p_joint_hit_dft_calibrated": p_joint_dft.reshape(shape).tolist(),
            "unphysical_mask": unphysical.reshape(shape).tolist(),
            "source_kde_density": kde_vals.reshape(shape).tolist(),
            "tolerance_eV": tolerance_eV,
            "empirical_cells": self.empirical_cells,
        }
