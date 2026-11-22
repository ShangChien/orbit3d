//! Grounded Recursive Self-Improvement (`RSI`) State Machine (`src/loom/weaving/rsi.py`):
//! - `RSIDiagnoser`: 4-Quadrant Causal Classification (`PRIOR_CEILING`, `OUTER_BLINDSPOT`,
//!   `CONVERGED`, `INNER_UNDEREXPLORED`) & Safe Curriculum Weight Allocation
//! - `OuterRSIAcquisition`: Uncertainty-Weighted Farthest-Point Tanimoto + Murcko Scaffold Diversity
//! - `IncrementalCalibrator`: Zero-Retrain High-Fidelity Feedback Ingestion & Validation Audit

use crate::chem_rules::{
    canonical_graph_signature, murcko_scaffold_signature, HeavyMolGraph, MolRecord, MorganBits2048,
};
use crate::critic_bus::MoleculeCriticEvaluation;
use crate::gp::{HeteroscedasticGPAtlas1D, ResidualCalibratedAsset};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QuadrantState {
    PriorCeiling,
    OuterBlindspot,
    Converged,
    InnerUnderexplored,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSIDiagnosticThresholds {
    pub hit_floor: f64,
    pub bias_cap: f64,
    pub cal_sigma_cap: f64,
    pub spread_cap: f64,
    pub eval_floor: f64,
    pub ceiling_bias_cap: f64,
    pub source_density_floor_ratio: f64,
}

impl Default for RSIDiagnosticThresholds {
    fn default() -> Self {
        Self {
            hit_floor: 0.24,
            bias_cap: 0.30,
            cal_sigma_cap: 0.32,
            spread_cap: 0.48,
            eval_floor: 0.35,
            ceiling_bias_cap: 0.55,
            source_density_floor_ratio: 0.06,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridCellDiagnosis1D {
    pub target: f64,
    pub state: QuadrantState,
    pub p_hit_rew: f64,
    pub p_hit_cal: f64,
    pub bias: f64,
    pub sigma_in: f64,
    pub sigma_cal: f64,
    pub p_eval: f64,
    pub source_kde: f64,
    pub inner_deficit: f64,
    pub acq_priority: f64,
    pub curriculum_prob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSIDiagnosticReport1D {
    pub cells: Vec<GridCellDiagnosis1D>,
    pub n_converged: usize,
    pub n_inner_underexplored: usize,
    pub n_outer_blindspot: usize,
    pub n_prior_ceiling: usize,
    pub blindspot_targets: Vec<f64>,
}

/// Compute 1D Gaussian KDE over `source_values` at `grid_targets`.
pub fn gaussian_kde_1d(source_values: &[f64], grid_targets: &[f64]) -> Vec<f64> {
    let finite_src: Vec<f64> = source_values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_src.len() < 2 {
        return vec![1.0; grid_targets.len()];
    }
    let n = finite_src.len() as f64;
    let mean = finite_src.iter().sum::<f64>() / n;
    let var = finite_src.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    // Silverman's rule of thumb bandwidth `h = 1.06 * std * n^(-1/5)`
    let bw = (1.06 * var.sqrt() * n.powf(-0.20)).max(0.10);
    let norm = 1.0 / (n * bw * (2.0 * std::f64::consts::PI).sqrt());

    grid_targets
        .iter()
        .map(|&t| {
            let s: f64 = finite_src
                .iter()
                .map(|&x| {
                    let z = (t - x) / bw;
                    (-0.5 * z * z).exp()
                })
                .sum();
            norm * s
        })
        .collect()
}

pub struct RSIDiagnoser {
    pub thresholds: RSIDiagnosticThresholds,
}

impl Default for RSIDiagnoser {
    fn default() -> Self {
        Self {
            thresholds: RSIDiagnosticThresholds::default(),
        }
    }
}

impl RSIDiagnoser {
    pub fn diagnose_1d(
        &self,
        atlas: &HeteroscedasticGPAtlas1D,
        grid_targets: &[f64],
        source_values: Option<&[f64]>,
    ) -> RSIDiagnosticReport1D {
        let thr = &self.thresholds;
        let kdes = if let Some(src) = source_values {
            gaussian_kde_1d(src, grid_targets)
        } else {
            vec![0.0; grid_targets.len()]
        };
        let rho_max = kdes.iter().copied().fold(0.0f64, f64::max);

        let mut cells = Vec::with_capacity(grid_targets.len());
        let mut n_conv = 0usize;
        let mut n_inner = 0usize;
        let mut n_blind = 0usize;
        let mut n_ceil = 0usize;
        let mut blindspot_targets = Vec::new();

        for (i, &t) in grid_targets.iter().enumerate() {
            let q = atlas.query(t);
            let kde_i = kdes[i];

            let (state, deficit, acq_pri) = if rho_max > 0.0
                && kde_i <= thr.source_density_floor_ratio * rho_max
                && q.bias.abs() > thr.ceiling_bias_cap
            {
                n_ceil += 1;
                (QuadrantState::PriorCeiling, 0.0, 0.0)
            } else if q.p_eval >= thr.eval_floor && q.sigma_cal > thr.cal_sigma_cap {
                n_blind += 1;
                blindspot_targets.push(t);
                (
                    QuadrantState::OuterBlindspot,
                    0.0,
                    (q.sigma_cal - thr.cal_sigma_cap) * q.p_eval,
                )
            } else if q.p_hit_rew >= thr.hit_floor
                && q.bias.abs() <= thr.bias_cap
                && q.sigma_cal <= thr.cal_sigma_cap
                && q.sigma_in <= thr.spread_cap
            {
                n_conv += 1;
                (QuadrantState::Converged, 0.0, 0.0)
            } else {
                n_inner += 1;
                let def = (thr.hit_floor - q.p_hit_rew).max(0.0)
                    + 0.40 * (q.sigma_in - thr.spread_cap).max(0.0)
                    + 0.30 * (q.bias.abs() - 0.15).max(0.0)
                    + 0.05;
                (QuadrantState::InnerUnderexplored, def, 0.0)
            };

            cells.push(GridCellDiagnosis1D {
                target: t,
                state,
                p_hit_rew: q.p_hit_rew,
                p_hit_cal: q.p_hit_cal,
                bias: q.bias,
                sigma_in: q.sigma_in,
                sigma_cal: q.sigma_cal,
                p_eval: q.p_eval,
                source_kde: kde_i,
                inner_deficit: deficit,
                acq_priority: acq_pri,
                curriculum_prob: 0.0,
            });
        }

        // Safe curriculum weight allocation (`OUTER_BLINDSPOT` & `PRIOR_CEILING` locked to `0.0%`!)
        let def_sum: f64 = cells
            .iter()
            .filter(|c| c.state == QuadrantState::InnerUnderexplored)
            .map(|c| c.inner_deficit)
            .sum();

        if n_inner > 0 && n_conv > 0 && def_sum > 0.0 {
            for c in &mut cells {
                match c.state {
                    QuadrantState::InnerUnderexplored => {
                        c.curriculum_prob = 0.80 * (c.inner_deficit / def_sum);
                    }
                    QuadrantState::Converged => {
                        c.curriculum_prob = 0.20 / (n_conv as f64);
                    }
                    _ => c.curriculum_prob = 0.0,
                }
            }
        } else if n_inner > 0 && def_sum > 0.0 {
            for c in &mut cells {
                if c.state == QuadrantState::InnerUnderexplored {
                    c.curriculum_prob = c.inner_deficit / def_sum;
                }
            }
        } else if n_conv > 0 {
            for c in &mut cells {
                if c.state == QuadrantState::Converged {
                    c.curriculum_prob = 1.0 / (n_conv as f64);
                }
            }
        } else if !cells.is_empty() {
            let non_ceil: Vec<usize> = (0..cells.len())
                .filter(|&i| cells[i].state != QuadrantState::PriorCeiling)
                .collect();
            if !non_ceil.is_empty() {
                let w = 1.0 / (non_ceil.len() as f64);
                for idx in non_ceil {
                    cells[idx].curriculum_prob = w;
                }
            } else {
                let w = 1.0 / (cells.len() as f64);
                for c in &mut cells {
                    c.curriculum_prob = w;
                }
            }
        }

        RSIDiagnosticReport1D {
            cells,
            n_converged: n_conv,
            n_inner_underexplored: n_inner,
            n_outer_blindspot: n_blind,
            n_prior_ceiling: n_ceil,
            blindspot_targets,
        }
    }

    /// Anti-regression acceptance gate for Inner-RSI self-improvement checkpoints.
    #[inline]
    pub fn acceptance_gate(
        pre_weak_hit: f64,
        post_weak_hit: f64,
        pre_global_hit: f64,
        post_global_hit: f64,
        pre_legal: f64,
        post_legal: f64,
    ) -> bool {
        let hit_improved =
            (post_weak_hit > pre_weak_hit + 1e-4) || (post_global_hit > pre_global_hit + 1e-4);
        let legal_ok = (post_legal - pre_legal) >= -0.02;
        hit_improved && legal_ok
    }
}

// ============================================================================
// `OuterRSIAcquisition` (`rsi.py:314-452`)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquisitionCandidate {
    pub proposal_index: usize,
    pub requested_target: f64,
    pub reward_pred: f64,
    pub sigma_cal: f64,
    pub f_max_ev_a: f64,
    pub murcko_scaffold: String,
    pub min_tanimoto_dist: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquisitionBatchManifest {
    pub selected: Vec<AcquisitionCandidate>,
    pub unique_murcko_scaffolds: usize,
    pub mean_sigma_cal: f64,
    pub sdf_export: String,
}

/// Select `budget_k` diverse, low-strain, clean-charge, high-uncertainty molecules for external
/// DFT / experimental labeling via greedy uncertainty-weighted farthest-point Tanimoto sampling!
pub fn select_outer_rsi_acquisition_batch(
    records: &[MolRecord],
    evals: &[MoleculeCriticEvaluation],
    requested_targets: &[f64],
    blindspot_targets: &[f64],
    budget_k: usize,
    max_fmax_ev_a: f64,
    min_sigma_cal: f64,
) -> AcquisitionBatchManifest {
    let n = records.len().min(evals.len()).min(requested_targets.len());
    let mut pool_idx = Vec::new();
    let mut pool_fps = Vec::new();
    let mut pool_scaffolds = Vec::new();
    let mut pool_u = Vec::new();
    let mut seen_graphs = HashSet::new();

    for i in 0..n {
        let ev = &evals[i];
        if !ev.legal || !ev.clean_charge || !ev.constraints_met || !ev.evaluated {
            continue;
        }
        if ev.f_max_ev_a.is_finite() && ev.f_max_ev_a > max_fmax_ev_a {
            continue;
        }
        let scal = ev.sigma_cal_est.first().copied().unwrap_or(0.0);
        if !scal.is_finite() {
            continue;
        }
        let req_t = requested_targets[i];
        let in_blind_req = blindspot_targets.iter().any(|&b| (b - req_t).abs() <= 0.35);
        if scal < min_sigma_cal && !in_blind_req {
            continue;
        }

        let g = HeavyMolGraph::from_record(&records[i]);
        let gsig = canonical_graph_signature(&g);
        if !seen_graphs.insert(gsig) {
            continue;
        }
        let fp = MorganBits2048::from_graph(&g);
        let scaf = murcko_scaffold_signature(&g);
        let u_base = scal.max(0.05) * if in_blind_req { 1.25 } else { 1.0 };

        pool_idx.push(i);
        pool_fps.push(fp);
        pool_scaffolds.push(scaf);
        pool_u.push(u_base);
    }

    let m = pool_idx.len();
    let k_take = budget_k.min(m);
    let mut chosen_mask = vec![false; m];
    let mut min_dist = vec![1.0f64; m];
    let mut selected = Vec::with_capacity(k_take);
    let mut unique_scafs = HashSet::new();
    let mut sdf_export = String::new();

    for _ in 0..k_take {
        let mut best_p = None;
        let mut best_util = f64::NEG_INFINITY;
        for p in 0..m {
            if chosen_mask[p] {
                continue;
            }
            let util = pool_u[p] * (0.15 + min_dist[p]);
            if util > best_util {
                best_util = util;
                best_p = Some(p);
            }
        }
        let Some(p_star) = best_p else { break };
        chosen_mask[p_star] = true;
        let orig_i = pool_idx[p_star];
        let d_recorded = min_dist[p_star];

        for p in 0..m {
            if !chosen_mask[p] {
                let d = 1.0 - pool_fps[p_star].tanimoto(&pool_fps[p]);
                if d < min_dist[p] {
                    min_dist[p] = d;
                }
            }
        }

        unique_scafs.insert(pool_scaffolds[p_star].clone());
        let mut rec_out = records[orig_i].clone();
        rec_out
            .properties
            .insert("requested_target".into(), requested_targets[orig_i]);
        rec_out.properties.insert(
            "sigma_cal".into(),
            evals[orig_i].sigma_cal_est.first().copied().unwrap_or(0.0),
        );
        sdf_export.push_str(&rec_out.to_sdf());

        selected.push(AcquisitionCandidate {
            proposal_index: orig_i,
            requested_target: requested_targets[orig_i],
            reward_pred: evals[orig_i].objective_preds.first().copied().unwrap_or(f64::NAN),
            sigma_cal: evals[orig_i].sigma_cal_est.first().copied().unwrap_or(0.0),
            f_max_ev_a: evals[orig_i].f_max_ev_a,
            murcko_scaffold: pool_scaffolds[p_star].clone(),
            min_tanimoto_dist: d_recorded,
        });
    }

    let mean_sigma_cal = if selected.is_empty() {
        0.0
    } else {
        selected.iter().map(|s| s.sigma_cal).sum::<f64>() / (selected.len() as f64)
    };

    AcquisitionBatchManifest {
        selected,
        unique_murcko_scaffolds: unique_scafs.len(),
        mean_sigma_cal,
        sdf_export,
    }
}

// ============================================================================
// `IncrementalCalibrator` (`rsi.py:454-525`)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalCalibrationReport {
    pub ingested_count: usize,
    pub pre_val_mae: Vec<f64>,
    pub post_val_mae: Vec<f64>,
    pub pre_mean_std: Vec<f64>,
    pub post_mean_std: Vec<f64>,
}

/// Zero-retrain incremental residual calibration update + blind validation audit.
pub fn run_incremental_calibration(
    calibrator: &mut ResidualCalibratedAsset,
    feedback_graphs: &[HeavyMolGraph],
    feedback_y_base: &[Vec<f64>],
    feedback_y_true: &[Vec<f64>],
    val_graphs: &[HeavyMolGraph],
    val_y_base: &[Vec<f64>],
    val_s_base: &[Vec<f64>],
    val_y_true: &[Vec<f64>],
) -> IncrementalCalibrationReport {
    let k = calibrator.num_channels;
    let eval_val = |cal: &ResidualCalibratedAsset| -> (Vec<f64>, Vec<f64>) {
        if val_graphs.is_empty() {
            return (vec![f64::NAN; k], vec![f64::NAN; k]);
        }
        let mut mae = vec![0.0f64; k];
        let mut mstd = vec![0.0f64; k];
        let mut cnt = vec![0usize; k];
        for i in 0..val_graphs.len() {
            let (yc, sc) = cal.predict(&val_graphs[i], &val_y_base[i], &val_s_base[i]);
            for c in 0..k {
                let yt = val_y_true[i].get(c).copied().unwrap_or(f64::NAN);
                if yc[c].is_finite() && yt.is_finite() {
                    mae[c] += (yc[c] - yt).abs();
                    mstd[c] += sc[c];
                    cnt[c] += 1;
                }
            }
        }
        for c in 0..k {
            if cnt[c] > 0 {
                mae[c] /= cnt[c] as f64;
                mstd[c] /= cnt[c] as f64;
            }
        }
        (mae, mstd)
    };

    let (pre_val_mae, pre_mean_std) = eval_val(calibrator);
    calibrator.ingest_samples(feedback_graphs, feedback_y_base, feedback_y_true, true);
    let (post_val_mae, post_mean_std) = eval_val(calibrator);

    IncrementalCalibrationReport {
        ingested_count: feedback_graphs.len(),
        pre_val_mae,
        post_val_mae,
        pre_mean_std,
        post_mean_std,
    }
}
