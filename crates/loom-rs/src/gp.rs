//! Exact Cholesky Linear Algebra, Ridge Regression, Matérn-3/2 & Matérn-5/2 Gaussian Processes,
//! Zero-Retrain `ResidualCalibratedAsset` (`29 + K` features), and `HeteroscedasticGPAtlas` (1D & 2D).

use crate::chem_rules::{compute_residual_calibration_features_29d, HeavyMolGraph};
use serde::{Deserialize, Serialize};

/// High-accuracy `erf(x)` in `f64` (max error `< 1.5e-7`).
#[inline]
pub fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

#[inline]
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_f64(z * std::f64::consts::FRAC_1_SQRT_2))
}

#[inline]
pub fn gaussian_box_hit_prob(mu: f64, target: f64, sigma: f64, tol: f64) -> f64 {
    let s = sigma.max(1e-4);
    let hi = norm_cdf((target + tol - mu) / s);
    let lo = norm_cdf((target - tol - mu) / s);
    (hi - lo).clamp(0.0, 1.0)
}

/// Lower-triangular Cholesky factorization `A = L * L^T` for an `n x n` SPD row-major matrix.
pub fn cholesky_lower(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Forward substitution `L * y = b` (`L` is `n x n` lower triangular).
pub fn solve_lower(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    y
}

/// Full Cholesky solve `(L * L^T) * x = b`.
pub fn solve_cholesky(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let y = solve_lower(l, n, b);
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

// ============================================================================
// Ridge Linear Trend (`alpha = 6.0`, centered with intercept)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeTrend {
    pub weights: Vec<f64>,
    pub intercept: f64,
}

impl RidgeTrend {
    pub fn fit(x_rows: &[Vec<f64>], y: &[f64], alpha: f64) -> Option<Self> {
        let n = x_rows.len();
        if n < 3 || y.len() != n {
            return None;
        }
        let d = x_rows[0].len();
        let mut mu_x = vec![0.0f64; d];
        let mut mu_y = 0.0f64;
        for i in 0..n {
            mu_y += y[i];
            for j in 0..d {
                mu_x[j] += x_rows[i][j];
            }
        }
        mu_y /= n as f64;
        for j in 0..d {
            mu_x[j] /= n as f64;
        }

        let mut xtx = vec![0.0f64; d * d];
        let mut xty = vec![0.0f64; d];
        for i in 0..n {
            let dy = y[i] - mu_y;
            for j in 0..d {
                let dx_j = x_rows[i][j] - mu_x[j];
                xty[j] += dx_j * dy;
                for k in 0..=j {
                    xtx[j * d + k] += dx_j * (x_rows[i][k] - mu_x[k]);
                }
            }
        }
        for j in 0..d {
            xtx[j * d + j] += alpha.max(1e-8);
            for k in (j + 1)..d {
                xtx[j * d + k] = xtx[k * d + j];
            }
        }
        let l = cholesky_lower(&xtx, d)?;
        let weights = solve_cholesky(&l, d, &xty);
        let mut dot = 0.0f64;
        for j in 0..d {
            dot += mu_x[j] * weights[j];
        }
        Some(Self {
            weights,
            intercept: mu_y - dot,
        })
    }

    #[inline]
    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut acc = self.intercept;
        for (w, &xi) in self.weights.iter().zip(x.iter()) {
            acc += w * xi;
        }
        acc
    }
}

// ============================================================================
// Isotropic & ARD Matérn-3/2 / Matérn-5/2 Gaussian Processes
// ============================================================================

#[inline]
fn matern32_cov(dist_scaled: f64, sigma_p_sq: f64) -> f64 {
    let s3 = 3.0f64.sqrt() * dist_scaled;
    sigma_p_sq * (1.0 + s3) * (-s3).exp()
}

#[inline]
fn matern52_cov(dist_scaled: f64, sigma_p_sq: f64) -> f64 {
    let s5 = 5.0f64.sqrt() * dist_scaled;
    sigma_p_sq * (1.0 + s5 + (5.0 / 3.0) * dist_scaled * dist_scaled) * (-s5).exp()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaternNu {
    Nu15,
    Nu25,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedMaternGp {
    pub nu: MaternNu,
    pub x_train: Vec<Vec<f64>>,
    pub length_scales: Vec<f64>,
    pub sigma_p_sq: f64,
    pub noise_var: f64,
    pub y_mean: f64,
    pub y_std: f64,
    pub normalize_y: bool,
    pub chol_l: Vec<f64>,
    pub alpha_vec: Vec<f64>,
}

impl FittedMaternGp {
    #[inline]
    fn scaled_dist(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut sq = 0.0f64;
        if self.length_scales.len() == 1 {
            let inv_l = 1.0 / self.length_scales[0].max(1e-6);
            for k in 0..a.len() {
                let d = (a[k] - b[k]) * inv_l;
                sq += d * d;
            }
        } else {
            for k in 0..a.len() {
                let inv_l = 1.0 / self.length_scales[k].max(1e-6);
                let d = (a[k] - b[k]) * inv_l;
                sq += d * d;
            }
        }
        sq.sqrt()
    }

    #[inline]
    fn kernel_val(&self, a: &[f64], b: &[f64]) -> f64 {
        let d = self.scaled_dist(a, b);
        match self.nu {
            MaternNu::Nu15 => matern32_cov(d, self.sigma_p_sq),
            MaternNu::Nu25 => matern52_cov(d, self.sigma_p_sq),
        }
    }

    pub fn fit_fixed(
        nu: MaternNu,
        x_train: Vec<Vec<f64>>,
        y_raw: &[f64],
        length_scales: Vec<f64>,
        sigma_p_sq: f64,
        noise_var: f64,
        per_point_alpha: Option<&[f64]>,
        normalize_y: bool,
    ) -> Option<(Self, f64)> {
        let n = x_train.len();
        if n == 0 || y_raw.len() != n {
            return None;
        }
        let (y_mean, y_std, y_proc) = if normalize_y {
            let m = y_raw.iter().sum::<f64>() / (n as f64);
            let v = y_raw.iter().map(|&y| (y - m) * (y - m)).sum::<f64>() / (n as f64);
            let s = v.sqrt();
            let s_safe = if s > 1e-8 { s } else { 1.0 };
            let yp: Vec<f64> = y_raw.iter().map(|&y| (y - m) / s_safe).collect();
            (m, s_safe, yp)
        } else {
            (0.0, 1.0, y_raw.to_vec())
        };

        let mut gp = Self {
            nu,
            x_train,
            length_scales,
            sigma_p_sq,
            noise_var,
            y_mean,
            y_std,
            normalize_y,
            chol_l: Vec::new(),
            alpha_vec: Vec::new(),
        };

        let mut k_mat = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut kij = gp.kernel_val(&gp.x_train[i], &gp.x_train[j]);
                if i == j {
                    kij += noise_var + 1e-8;
                    if let Some(pa) = per_point_alpha {
                        kij += pa[i].max(0.0);
                    }
                }
                k_mat[i * n + j] = kij;
                k_mat[j * n + i] = kij;
            }
        }

        let chol_l = cholesky_lower(&k_mat, n)?;
        let alpha_vec = solve_cholesky(&chol_l, n, &y_proc);

        // Log marginal likelihood: `-0.5 * y^T alpha - sum_i ln(L_ii) - 0.5 * n * ln(2*pi)`
        let mut quad = 0.0f64;
        let mut log_det_half = 0.0f64;
        for i in 0..n {
            quad += y_proc[i] * alpha_vec[i];
            log_det_half += chol_l[i * n + i].ln();
        }
        let lml = -0.5 * quad - log_det_half - 0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln();
        gp.chol_l = chol_l;
        gp.alpha_vec = alpha_vec;
        Some((gp, lml))
    }

    /// Grid-search bounded hyperparameter optimization on exact Log Marginal Likelihood (`LML`).
    pub fn fit_grid_lml(
        nu: MaternNu,
        x_train: Vec<Vec<f64>>,
        y_raw: &[f64],
        ls_candidates: &[Vec<f64>],
        sp_candidates: &[f64],
        nv_candidates: &[f64],
        per_point_alpha: Option<&[f64]>,
        normalize_y: bool,
    ) -> Option<Self> {
        let mut best_gp: Option<Self> = None;
        let mut best_lml = f64::NEG_INFINITY;
        for ls in ls_candidates {
            for &sp in sp_candidates {
                for &nv in nv_candidates {
                    if let Some((gp, lml)) = Self::fit_fixed(
                        nu,
                        x_train.clone(),
                        y_raw,
                        ls.clone(),
                        sp,
                        nv,
                        per_point_alpha,
                        normalize_y,
                    ) {
                        if lml > best_lml {
                            best_lml = lml;
                            best_gp = Some(gp);
                        }
                    }
                }
            }
        }
        best_gp
    }

    /// Returns `(mean, total_std_including_white_noise, pure_epistemic_std)`.
    pub fn predict(&self, x: &[f64]) -> (f64, f64, f64) {
        let n = self.x_train.len();
        let mut k_star = vec![0.0f64; n];
        let mut dot_mean = 0.0f64;
        for i in 0..n {
            let kv = self.kernel_val(&self.x_train[i], x);
            k_star[i] = kv;
            dot_mean += kv * self.alpha_vec[i];
        }
        let v = solve_lower(&self.chol_l, n, &k_star);
        let v_norm_sq: f64 = v.iter().map(|&vi| vi * vi).sum();

        let mean = self.y_mean + self.y_std * dot_mean;
        let var_ep_unscaled = (self.sigma_p_sq - v_norm_sq).max(0.0);
        let var_tot_unscaled = (self.sigma_p_sq + self.noise_var - v_norm_sq).max(0.0);
        let ep_std = self.y_std * var_ep_unscaled.sqrt();
        let tot_std = self.y_std * var_tot_unscaled.sqrt();
        (mean, tot_std, ep_std)
    }
}

// ============================================================================
// Zero-Retrain `ResidualCalibratedAsset` (`src/loom/weaving/registry.py:504-872`)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedChannelState {
    pub x_anchors: Vec<Vec<f64>>,
    pub residuals: Vec<f64>,
    pub is_feedback: Vec<bool>,
    pub trend: Option<RidgeTrend>,
    pub gp: Option<FittedMaternGp>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualCalibratedAsset {
    pub num_channels: usize,
    pub channel_names: Vec<String>,
    pub prior_blindspot_std: f64,
    pub pred_mean: Vec<f64>,
    pub pred_scale: Vec<f64>,
    pub stats_frozen: bool,
    pub channels: Vec<CalibratedChannelState>,
}

impl ResidualCalibratedAsset {
    pub fn new(channel_names: Vec<String>, prior_blindspot_std: f64) -> Self {
        let k = channel_names.len().max(1);
        let channels = (0..k)
            .map(|_| CalibratedChannelState {
                x_anchors: Vec::new(),
                residuals: Vec::new(),
                is_feedback: Vec::new(),
                trend: None,
                gp: None,
            })
            .collect();
        Self {
            num_channels: k,
            channel_names,
            prior_blindspot_std: prior_blindspot_std.max(0.05),
            pred_mean: vec![0.0; k],
            pred_scale: vec![1.0; k],
            stats_frozen: false,
            channels,
        }
    }

    pub fn build_feature_vec(&self, g: &HeavyMolGraph, y_base: &[f64]) -> Vec<f64> {
        let f29 = compute_residual_calibration_features_29d(g);
        let mut out = Vec::with_capacity(29 + self.num_channels);
        out.extend_from_slice(&f29);
        for c in 0..self.num_channels {
            let yb = y_base.get(c).copied().unwrap_or(0.0);
            let denom = self.pred_scale[c].max(0.20);
            out.push(0.85 * ((yb - self.pred_mean[c]) / denom));
        }
        out
    }

    /// Seed in-domain anchor points (`residuals = y_true - y_base` or `0.0` trusted source anchors)
    /// or ingest external high-fidelity feedback (`is_feedback = true`, e.g. DFT labels in Outer-RSI).
    pub fn ingest_samples(
        &mut self,
        graphs: &[HeavyMolGraph],
        y_base_rows: &[Vec<f64>],
        y_true_rows: &[Vec<f64>],
        is_feedback: bool,
    ) {
        let m = graphs.len().min(y_base_rows.len()).min(y_true_rows.len());
        if m == 0 {
            return;
        }
        if !self.stats_frozen {
            for c in 0..self.num_channels {
                let vals: Vec<f64> = y_base_rows
                    .iter()
                    .filter_map(|r| r.get(c).copied().filter(|v| v.is_finite()))
                    .collect();
                if !vals.is_empty() {
                    let mean = vals.iter().sum::<f64>() / (vals.len() as f64);
                    let var = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (vals.len() as f64);
                    self.pred_mean[c] = mean;
                    self.pred_scale[c] = var.sqrt().max(0.25);
                }
            }
            self.stats_frozen = true;
        }

        for i in 0..m {
            let x = self.build_feature_vec(&graphs[i], &y_base_rows[i]);
            for c in 0..self.num_channels {
                let yb = y_base_rows[i].get(c).copied().unwrap_or(f64::NAN);
                let yt = y_true_rows[i].get(c).copied().unwrap_or(f64::NAN);
                if yb.is_finite() && yt.is_finite() {
                    self.channels[c].x_anchors.push(x.clone());
                    self.channels[c].residuals.push(yt - yb);
                    self.channels[c].is_feedback.push(is_feedback);
                }
            }
        }
        self.refit_gps();
    }

    pub fn refit_gps(&mut self) {
        let prior_var = (self.prior_blindspot_std * self.prior_blindspot_std).max(0.04);
        let sp_cands = [0.88 * prior_var, prior_var, 1.15 * prior_var];
        let ls_cands = [vec![0.48], vec![0.65], vec![0.85]];
        let nv_cands = [0.005, 0.03, 0.10];

        for ch in &mut self.channels {
            if ch.x_anchors.len() < 4 {
                ch.trend = None;
                ch.gp = None;
                continue;
            }
            let n_fb = ch.is_feedback.iter().filter(|&&b| b).count();
            ch.trend = if n_fb >= 3 {
                RidgeTrend::fit(&ch.x_anchors, &ch.residuals, 6.0)
            } else {
                None
            };
            let r_detrended: Vec<f64> = ch
                .x_anchors
                .iter()
                .zip(ch.residuals.iter())
                .map(|(x, &r)| {
                    if let Some(ref tr) = ch.trend {
                        r - tr.predict(x)
                    } else {
                        r
                    }
                })
                .collect();

            ch.gp = FittedMaternGp::fit_grid_lml(
                MaternNu::Nu15,
                ch.x_anchors.clone(),
                &r_detrended,
                &ls_cands,
                &sp_cands,
                &nv_cands,
                None,
                false,
            );
        }
    }

    /// Predict calibrated `(y_cal, sigma_out)` combining base uncertainty + epistemic GP residual uncertainty.
    pub fn predict(&self, g: &HeavyMolGraph, y_base: &[f64], std_base: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let x = self.build_feature_vec(g, y_base);
        let mut y_cal = Vec::with_capacity(self.num_channels);
        let mut s_out = Vec::with_capacity(self.num_channels);
        for c in 0..self.num_channels {
            let yb = y_base.get(c).copied().unwrap_or(0.0);
            let sb = std_base.get(c).copied().unwrap_or(0.0);
            let ch = &self.channels[c];
            if let Some(ref gp) = ch.gp {
                let tr = ch.trend.as_ref().map_or(0.0, |t| t.predict(&x));
                let (delta_gp, _tot_sd, ep_sd) = gp.predict(&x);
                y_cal.push(yb + tr + delta_gp);
                s_out.push((sb * sb + ep_sd * ep_sd).sqrt());
            } else {
                y_cal.push(yb);
                s_out.push((sb * sb + self.prior_blindspot_std * self.prior_blindspot_std).sqrt());
            }
        }
        (y_cal, s_out)
    }
}

// ============================================================================
// 1D & 2D Heteroscedastic GP Reachability & Calibration Atlas (`gp_atlas.py`)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalAtlasSample {
    pub requested_targets: Vec<f64>,
    pub evaluated: bool,
    pub reward_pred: Vec<f64>,
    pub audit_pred: Option<Vec<f64>>,
    pub sigma_cal_est: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atlas1DPoint {
    pub target: f64,
    pub mu_gen: f64,
    pub bias: f64,
    pub sigma_in: f64,
    pub sigma_ep: f64,
    pub sigma_cal: f64,
    pub p_eval: f64,
    pub p_hit_rew: f64,
    pub p_hit_cal: f64,
    pub zone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gatekeeper1DRecommendation {
    pub status: String,
    pub reason: String,
    pub requested_target: f64,
    pub recommended_target: f64,
    pub sigma_cal: f64,
    pub p_hit_cal: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroscedasticGPAtlas1D {
    pub axis_name: String,
    pub tolerance: f64,
    pub target_min: f64,
    pub target_max: f64,
    pub gp_mean: FittedMaternGp,
    pub gp_log_std: FittedMaternGp,
    pub gp_logit_eval: FittedMaternGp,
    pub gp_log_cal: Option<FittedMaternGp>,
    pub fallback_cal_std: f64,
}

impl HeteroscedasticGPAtlas1D {
    pub fn fit(
        axis_name: &str,
        tolerance: f64,
        samples: &[ProposalAtlasSample],
        axis_idx: usize,
    ) -> Result<Self, String> {
        let mut buckets: std::collections::BTreeMap<i64, Vec<&ProposalAtlasSample>> =
            std::collections::BTreeMap::new();
        for s in samples {
            if let Some(&t) = s.requested_targets.get(axis_idx) {
                if t.is_finite() {
                    let key = (t * 10000.0).round() as i64;
                    buckets.entry(key).or_default().push(s);
                }
            }
        }
        if buckets.len() < 2 {
            return Err("HeteroscedasticGPAtlas1D requires >= 2 distinct target grid cells".into());
        }

        let mut x_all = Vec::new();
        let mut y_logit_ev = Vec::new();

        let mut x_pop = Vec::new();
        let mut y_mu = Vec::new();
        let mut v_sem = Vec::new();
        let mut y_lsd = Vec::new();

        let mut x_cal = Vec::new();
        let mut y_lcal = Vec::new();

        let mut t_min = f64::INFINITY;
        let mut t_max = f64::NEG_INFINITY;

        for (&key, group) in &buckets {
            let t = (key as f64) / 10000.0;
            t_min = t_min.min(t);
            t_max = t_max.max(t);
            let n_att = group.len() as f64;
            let ev_vals: Vec<f64> = group
                .iter()
                .filter(|g| g.evaluated)
                .filter_map(|g| g.reward_pred.get(axis_idx).copied().filter(|v| v.is_finite()))
                .collect();
            let n_ev = ev_vals.len() as f64;
            let p_ev = ((n_ev + 0.5) / (n_att + 1.0)).clamp(1e-3, 1.0 - 1e-3);
            x_all.push(vec![t]);
            y_logit_ev.push((p_ev / (1.0 - p_ev)).ln());

            if ev_vals.len() >= 2 {
                let mu = ev_vals.iter().sum::<f64>() / n_ev;
                let var = ev_vals.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / (n_ev - 1.0);
                let sd = var.sqrt().max(0.03);
                x_pop.push(vec![t]);
                y_mu.push(mu);
                v_sem.push((sd * sd) / n_ev);
                y_lsd.push(sd.ln());
            }

            // Cross-oracle audit discrepancy or single-oracle AD sigma_cal samples
            let mut diff_sq = Vec::new();
            let mut direct_cal = Vec::new();
            for g in group.iter().filter(|g| g.evaluated) {
                if let (Some(&yr), Some(ref aud)) = (g.reward_pred.get(axis_idx), &g.audit_pred) {
                    if let Some(&ya) = aud.get(axis_idx) {
                        if yr.is_finite() && ya.is_finite() {
                            diff_sq.push((yr - ya) * (yr - ya));
                        }
                    }
                }
                if let Some(ref sc) = g.sigma_cal_est {
                    if let Some(&s_val) = sc.get(axis_idx) {
                        if s_val.is_finite() {
                            direct_cal.push(s_val);
                        }
                    }
                }
            }
            if diff_sq.len() >= 2 {
                let rmse = (diff_sq.iter().sum::<f64>() / (diff_sq.len() as f64)).sqrt().max(0.03);
                x_cal.push(vec![t]);
                y_lcal.push(rmse.ln());
            } else if direct_cal.len() >= 2 {
                direct_cal.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = direct_cal[direct_cal.len() / 2].max(0.03);
                x_cal.push(vec![t]);
                y_lcal.push(med.ln());
            }
        }

        if x_pop.len() < 2 {
            return Err("HeteroscedasticGPAtlas1D requires >= 2 populated cells with >= 2 valid molecules".into());
        }

        let span = (t_max - t_min).max(0.5);
        let ls_mean = [vec![0.18 * span], vec![0.35 * span], vec![0.80 * span]];
        let ls_aux = [vec![0.18 * span], vec![0.35 * span], vec![0.65 * span]];
        let sp_cands = [0.25, 1.0, 4.0];
        let nv_cands = [0.005, 0.02, 0.08];

        let gp_mean = FittedMaternGp::fit_grid_lml(
            MaternNu::Nu25,
            x_pop.clone(),
            &y_mu,
            &ls_mean,
            &sp_cands,
            &[1e-6],
            Some(&v_sem),
            true,
        )
        .ok_or_else(|| "Failed to fit gp_mean".to_string())?;

        let gp_log_std = FittedMaternGp::fit_grid_lml(
            MaternNu::Nu15,
            x_pop,
            &y_lsd,
            &ls_aux,
            &sp_cands,
            &nv_cands,
            None,
            true,
        )
        .ok_or_else(|| "Failed to fit gp_log_std".to_string())?;

        let gp_logit_eval = FittedMaternGp::fit_grid_lml(
            MaternNu::Nu15,
            x_all,
            &y_logit_ev,
            &ls_mean,
            &sp_cands,
            &nv_cands,
            None,
            true,
        )
        .ok_or_else(|| "Failed to fit gp_logit_eval".to_string())?;

        let gp_log_cal = if x_cal.len() >= 2 {
            FittedMaternGp::fit_grid_lml(
                MaternNu::Nu15,
                x_cal,
                &y_lcal,
                &ls_aux,
                &sp_cands,
                &nv_cands,
                None,
                true,
            )
        } else {
            None
        };

        Ok(Self {
            axis_name: axis_name.to_string(),
            tolerance: tolerance.max(1e-4),
            target_min: t_min,
            target_max: t_max,
            gp_mean,
            gp_log_std,
            gp_logit_eval,
            gp_log_cal,
            fallback_cal_std: 0.19,
        })
    }

    pub fn query(&self, t: f64) -> Atlas1DPoint {
        let pt = [t];
        let (mu_gen, _tot_sd, sigma_ep) = self.gp_mean.predict(&pt);
        let (lsd, _, _) = self.gp_log_std.predict(&pt);
        let sigma_in = lsd.clamp(0.02f64.ln(), 5.0f64.ln()).exp();
        let (logit_ev, _, _) = self.gp_logit_eval.predict(&pt);
        let p_eval = 1.0 / (1.0 + (-logit_ev).exp());
        let sigma_cal = if let Some(ref gcal) = self.gp_log_cal {
            let (lcal, _, _) = gcal.predict(&pt);
            lcal.clamp(0.02f64.ln(), 5.0f64.ln()).exp()
        } else {
            self.fallback_cal_std
        };

        let sigma_in_eff = (sigma_in * sigma_in + sigma_ep * sigma_ep).sqrt();
        let sigma_total = (sigma_in_eff * sigma_in_eff + sigma_cal * sigma_cal).sqrt();
        let bias = mu_gen - t;
        let p_hit_rew = p_eval * gaussian_box_hit_prob(mu_gen, t, sigma_in_eff, self.tolerance);
        let p_hit_cal = p_eval * gaussian_box_hit_prob(mu_gen, t, sigma_total, self.tolerance);

        let zone = if p_hit_rew >= 0.16 && bias.abs() <= 1.40 * self.tolerance {
            "calibrated_reachable"
        } else if p_hit_rew >= 0.08 && bias.abs() <= 2.20 * self.tolerance {
            "boundary_degrading"
        } else {
            "unreachable_ood"
        };

        Atlas1DPoint {
            target: t,
            mu_gen,
            bias,
            sigma_in,
            sigma_ep,
            sigma_cal,
            p_eval,
            p_hit_rew,
            p_hit_cal,
            zone: zone.to_string(),
        }
    }

    pub fn query_curve(&self, n_points: usize) -> Vec<Atlas1DPoint> {
        let n = n_points.max(2);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let frac = (i as f64) / ((n - 1) as f64);
            let t = self.target_min + frac * (self.target_max - self.target_min);
            out.push(self.query(t));
        }
        out
    }

    pub fn recommend_target(&self, t_req: f64, cal_sigma_cap: f64) -> Gatekeeper1DRecommendation {
        let q = self.query(t_req);
        if q.zone == "calibrated_reachable" && q.sigma_cal <= cal_sigma_cap {
            return Gatekeeper1DRecommendation {
                status: "trusted".to_string(),
                reason: "in_calibrated_reachable_zone".to_string(),
                requested_target: t_req,
                recommended_target: t_req,
                sigma_cal: q.sigma_cal,
                p_hit_cal: q.p_hit_cal,
            };
        }

        let reason = if q.sigma_cal > cal_sigma_cap {
            "predictor_blindspot"
        } else {
            "prior_or_boundary_drift"
        };

        let grid = self.query_curve(161);
        let mut best_trusted: Option<&Atlas1DPoint> = None;
        for pt in &grid {
            if pt.zone == "calibrated_reachable" && pt.sigma_cal <= cal_sigma_cap {
                if best_trusted.map_or(true, |b| (pt.target - t_req).abs() < (b.target - t_req).abs()) {
                    best_trusted = Some(pt);
                }
            }
        }
        let chosen = best_trusted.unwrap_or_else(|| {
            grid.iter()
                .max_by(|a, b| a.p_hit_cal.partial_cmp(&b.p_hit_cal).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
        });

        Gatekeeper1DRecommendation {
            status: "rerouted_warning".to_string(),
            reason: reason.to_string(),
            requested_target: t_req,
            recommended_target: chosen.target,
            sigma_cal: q.sigma_cal,
            p_hit_cal: q.p_hit_cal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atlas2DPoint {
    pub tx: f64,
    pub ty: f64,
    pub mu_x: f64,
    pub mu_y: f64,
    pub sigma_cal_2d: f64,
    pub p_eval: f64,
    pub p_joint_rew: f64,
    pub p_joint_cal: f64,
    pub unphysical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroscedasticGPAtlas2D {
    pub axis_names: [String; 2],
    pub tolerances: [f64; 2],
    pub bounds_x: [f64; 2],
    pub bounds_y: [f64; 2],
    pub unphysical_x_geq_y: bool,
    pub gp_mu_x: FittedMaternGp,
    pub gp_mu_y: FittedMaternGp,
    pub gp_log_sd_x: FittedMaternGp,
    pub gp_log_sd_y: FittedMaternGp,
    pub gp_logit_eval: FittedMaternGp,
    pub gp_log_cal_x: Option<FittedMaternGp>,
    pub gp_log_cal_y: Option<FittedMaternGp>,
}

impl HeteroscedasticGPAtlas2D {
    pub fn fit(
        axis_names: [String; 2],
        tolerances: [f64; 2],
        samples: &[ProposalAtlasSample],
    ) -> Result<Self, String> {
        let mut buckets: std::collections::BTreeMap<(i64, i64), Vec<&ProposalAtlasSample>> =
            std::collections::BTreeMap::new();
        for s in samples {
            if s.requested_targets.len() >= 2
                && s.requested_targets[0].is_finite()
                && s.requested_targets[1].is_finite()
            {
                let kx = (s.requested_targets[0] * 1000.0).round() as i64;
                let ky = (s.requested_targets[1] * 1000.0).round() as i64;
                buckets.entry((kx, ky)).or_default().push(s);
            }
        }
        if buckets.len() < 4 {
            return Err("HeteroscedasticGPAtlas2D requires >= 4 distinct 2D grid cells".into());
        }

        let mut x_all = Vec::new();
        let mut y_logit_ev = Vec::new();
        let mut x_pop = Vec::new();
        let mut mu_x_v = Vec::new();
        let mut mu_y_v = Vec::new();
        let mut sem_x = Vec::new();
        let mut sem_y = Vec::new();
        let mut lsd_x = Vec::new();
        let mut lsd_y = Vec::new();
        let mut x_cal = Vec::new();
        let mut lcal_x = Vec::new();
        let mut lcal_y = Vec::new();

        let mut bx = [f64::INFINITY, f64::NEG_INFINITY];
        let mut by = [f64::INFINITY, f64::NEG_INFINITY];

        for (&(kx, ky), group) in &buckets {
            let tx = (kx as f64) / 1000.0;
            let ty = (ky as f64) / 1000.0;
            bx = [bx[0].min(tx), bx[1].max(tx)];
            by = [by[0].min(ty), by[1].max(ty)];

            let n_att = group.len() as f64;
            let ev_pairs: Vec<(f64, f64)> = group
                .iter()
                .filter(|g| g.evaluated && g.reward_pred.len() >= 2)
                .map(|g| (g.reward_pred[0], g.reward_pred[1]))
                .filter(|(a, b)| a.is_finite() && b.is_finite())
                .collect();
            let n_ev = ev_pairs.len() as f64;
            let p_ev = ((n_ev + 0.5) / (n_att + 1.0)).clamp(1e-3, 1.0 - 1e-3);
            x_all.push(vec![tx, ty]);
            y_logit_ev.push((p_ev / (1.0 - p_ev)).ln());

            if ev_pairs.len() >= 2 {
                let mx = ev_pairs.iter().map(|p| p.0).sum::<f64>() / n_ev;
                let my = ev_pairs.iter().map(|p| p.1).sum::<f64>() / n_ev;
                let vx = ev_pairs.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>() / (n_ev - 1.0);
                let vy = ev_pairs.iter().map(|p| (p.1 - my).powi(2)).sum::<f64>() / (n_ev - 1.0);
                let sx = vx.sqrt().max(0.03);
                let sy = vy.sqrt().max(0.03);
                x_pop.push(vec![tx, ty]);
                mu_x_v.push(mx);
                mu_y_v.push(my);
                sem_x.push(sx * sx / n_ev);
                sem_y.push(sy * sy / n_ev);
                lsd_x.push(sx.ln());
                lsd_y.push(sy.ln());
            }

            let cal_pairs: Vec<(f64, f64)> = group
                .iter()
                .filter(|g| g.evaluated)
                .filter_map(|g| {
                    if let Some(ref aud) = g.audit_pred {
                        if g.reward_pred.len() >= 2 && aud.len() >= 2 {
                            return Some(((g.reward_pred[0] - aud[0]).powi(2), (g.reward_pred[1] - aud[1]).powi(2)));
                        }
                    }
                    if let Some(ref sc) = g.sigma_cal_est {
                        if sc.len() >= 2 {
                            return Some((sc[0] * sc[0], sc[1] * sc[1]));
                        }
                    }
                    None
                })
                .collect();
            if cal_pairs.len() >= 2 {
                let cx = (cal_pairs.iter().map(|p| p.0).sum::<f64>() / (cal_pairs.len() as f64))
                    .sqrt()
                    .max(0.03);
                let cy = (cal_pairs.iter().map(|p| p.1).sum::<f64>() / (cal_pairs.len() as f64))
                    .sqrt()
                    .max(0.03);
                x_cal.push(vec![tx, ty]);
                lcal_x.push(cx.ln());
                lcal_y.push(cy.ln());
            }
        }

        if x_pop.len() < 4 {
            return Err("HeteroscedasticGPAtlas2D requires >= 4 populated 2D cells".into());
        }

        let ls_cands = [vec![0.35, 0.35], vec![0.65, 0.65], vec![1.25, 1.25]];
        let sp_cands = [0.25, 1.0, 4.0];
        let nv_cands = [0.01, 0.04, 0.12];

        let gp_mu_x = FittedMaternGp::fit_grid_lml(MaternNu::Nu25, x_pop.clone(), &mu_x_v, &ls_cands, &sp_cands, &[1e-6], Some(&sem_x), true)
            .ok_or("Failed gp_mu_x")?;
        let gp_mu_y = FittedMaternGp::fit_grid_lml(MaternNu::Nu25, x_pop.clone(), &mu_y_v, &ls_cands, &sp_cands, &[1e-6], Some(&sem_y), true)
            .ok_or("Failed gp_mu_y")?;
        let gp_log_sd_x = FittedMaternGp::fit_grid_lml(MaternNu::Nu25, x_pop.clone(), &lsd_x, &ls_cands, &sp_cands, &nv_cands, None, true)
            .ok_or("Failed gp_log_sd_x")?;
        let gp_log_sd_y = FittedMaternGp::fit_grid_lml(MaternNu::Nu25, x_pop, &lsd_y, &ls_cands, &sp_cands, &nv_cands, None, true)
            .ok_or("Failed gp_log_sd_y")?;
        let gp_logit_eval = FittedMaternGp::fit_grid_lml(MaternNu::Nu25, x_all, &y_logit_ev, &ls_cands, &sp_cands, &nv_cands, None, true)
            .ok_or("Failed gp_logit_eval")?;

        let (gp_log_cal_x, gp_log_cal_y) = if x_cal.len() >= 4 {
            (
                FittedMaternGp::fit_grid_lml(MaternNu::Nu25, x_cal.clone(), &lcal_x, &ls_cands, &sp_cands, &nv_cands, None, true),
                FittedMaternGp::fit_grid_lml(MaternNu::Nu25, x_cal, &lcal_y, &ls_cands, &sp_cands, &nv_cands, None, true),
            )
        } else {
            (None, None)
        };

        let unphys = axis_names[0].eq_ignore_ascii_case("homo") && axis_names[1].eq_ignore_ascii_case("lumo");

        Ok(Self {
            axis_names,
            tolerances,
            bounds_x: bx,
            bounds_y: by,
            unphysical_x_geq_y: unphys,
            gp_mu_x,
            gp_mu_y,
            gp_log_sd_x,
            gp_log_sd_y,
            gp_logit_eval,
            gp_log_cal_x,
            gp_log_cal_y,
        })
    }

    pub fn query(&self, tx: f64, ty: f64) -> Atlas2DPoint {
        let pt = [tx, ty];
        let unphysical = self.unphysical_x_geq_y && (tx >= ty);
        let (mu_x, _, ep_x) = self.gp_mu_x.predict(&pt);
        let (mu_y, _, ep_y) = self.gp_mu_y.predict(&pt);
        let sx = self.gp_log_sd_x.predict(&pt).0.clamp(0.03f64.ln(), 3.0f64.ln()).exp();
        let sy = self.gp_log_sd_y.predict(&pt).0.clamp(0.03f64.ln(), 3.0f64.ln()).exp();
        let p_eval = 1.0 / (1.0 + (-self.gp_logit_eval.predict(&pt).0).exp());
        let cx = self.gp_log_cal_x.as_ref().map_or(0.19, |g| g.predict(&pt).0.clamp(0.03f64.ln(), 3.0f64.ln()).exp());
        let cy = self.gp_log_cal_y.as_ref().map_or(0.21, |g| g.predict(&pt).0.clamp(0.03f64.ln(), 3.0f64.ln()).exp());
        let sigma_cal_2d = (0.5 * (cx * cx + cy * cy)).sqrt();

        let eff_x = (sx * sx + ep_x * ep_x).sqrt();
        let eff_y = (sy * sy + ep_y * ep_y).sqrt();
        let tot_x = (eff_x * eff_x + cx * cx).sqrt();
        let tot_y = (eff_y * eff_y + cy * cy).sqrt();

        let p_joint_rew = if unphysical {
            0.0
        } else {
            p_eval
                * gaussian_box_hit_prob(mu_x, tx, eff_x, self.tolerances[0])
                * gaussian_box_hit_prob(mu_y, ty, eff_y, self.tolerances[1])
        };
        let p_joint_cal = if unphysical {
            0.0
        } else {
            p_eval
                * gaussian_box_hit_prob(mu_x, tx, tot_x, self.tolerances[0])
                * gaussian_box_hit_prob(mu_y, ty, tot_y, self.tolerances[1])
        };

        Atlas2DPoint {
            tx,
            ty,
            mu_x,
            mu_y,
            sigma_cal_2d,
            p_eval,
            p_joint_rew,
            p_joint_cal,
            unphysical,
        }
    }
}
