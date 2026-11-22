//! `CompositeCriticBus`, `MorganDomain` Applicability Domain, Disentangled `GRPO` Group Advantages,
//! Unified `MlipSlot` (`HyffEngine` | `Dpa4Engine`), and `PhysicsConstrainedCoordRefiner`.

use crate::chem_rules::{
    canonical_graph_signature, formal_charge_signature, is_clean_organic_charge_state,
    BuiltInRuleMetrics, HeavyMolGraph, MolRecord, MorganBits2048,
};
use crate::gp::ResidualCalibratedAsset;
use crate::oracle_train::TrainableBootstrapOracle;
use crate::spec::{zero_step_physical_reward, AuditorMode, TaskWeavingSpec};
use dpa4_rs::Dpa4Engine;
use hyff_rs::HyffEngine;
use mlip_core::{AsyncPotentialBackend, MolecularSystem, PotentialEval};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Unified hot-swappable Quantum MLIP slot (`HyFF-L2` <-> `DPA-4`).
pub enum MlipSlot {
    Hyff(HyffEngine),
    Dpa4(Dpa4Engine),
}

impl MlipSlot {
    pub fn family(&self) -> &'static str {
        match self {
            Self::Hyff(_) => "hyff",
            Self::Dpa4(_) => "dpa4",
        }
    }

    pub fn model_name(&self) -> &str {
        match self {
            Self::Hyff(e) => e.model_name(),
            Self::Dpa4(e) => e.model_name(),
        }
    }

    pub fn desc_dim(&self) -> usize {
        match self {
            Self::Hyff(e) => e.weights.cfg.hidden,
            Self::Dpa4(e) => e.weights.cfg.channels,
        }
    }

    pub async fn eval_async(
        &mut self,
        sys: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<PotentialEval, String> {
        match self {
            Self::Hyff(e) => e.eval_async(sys, need_forces, need_descriptor).await,
            Self::Dpa4(e) => e.eval_async(sys, need_forces, need_descriptor).await,
        }
    }

    pub async fn vjp_descriptor_coords_async(
        &mut self,
        sys: &MolecularSystem,
        d_desc: &[f32],
    ) -> Result<Vec<[f64; 3]>, String> {
        match self {
            Self::Hyff(e) => e.vjp_descriptor_coords_async(sys, d_desc).await,
            Self::Dpa4(e) => e.vjp_descriptor_coords_async(sys, d_desc).await,
        }
    }
}

/// Morgan 2048-bit Applicability Domain (`src/loom/oracles/applicability.py`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorganDomain {
    pub ref_fps: Vec<MorganBits2048>,
    pub training_elements: HashSet<u8>,
    pub cutoff: f64,
    pub calibrated_target_bounds: Vec<[f64; 2]>,
}

impl MorganDomain {
    pub fn from_records(records: &[MolRecord], cutoff: f64, target_cols: &[String]) -> Self {
        let mut ref_fps = Vec::with_capacity(records.len());
        let mut training_elements = HashSet::new();
        let mut bounds = vec![[f64::INFINITY, f64::NEG_INFINITY]; target_cols.len()];

        for rec in records {
            if !rec.valence_valid {
                continue;
            }
            for &z in &rec.atomic_numbers {
                training_elements.insert(z);
            }
            let g = HeavyMolGraph::from_record(rec);
            ref_fps.push(MorganBits2048::from_graph(&g));
            for (k, col) in target_cols.iter().enumerate() {
                if let Some(&v) = rec.properties.get(col) {
                    if v.is_finite() {
                        bounds[k][0] = bounds[k][0].min(v);
                        bounds[k][1] = bounds[k][1].max(v);
                    }
                }
            }
        }
        Self {
            ref_fps,
            training_elements,
            cutoff,
            calibrated_target_bounds: bounds,
        }
    }

    /// Returns `(in_domain, max_tanimoto, has_unseen_elements)`.
    pub fn check(&self, rec: &MolRecord, g: &HeavyMolGraph) -> (bool, f64, bool) {
        let unseen = rec
            .atomic_numbers
            .iter()
            .any(|z| !self.training_elements.contains(z));
        if self.ref_fps.is_empty() {
            return (!unseen, 1.0, unseen);
        }
        let fp = MorganBits2048::from_graph(g);
        let mut max_sim = 0.0f64;
        for rfp in &self.ref_fps {
            let s = fp.tanimoto(rfp);
            if s > max_sim {
                max_sim = s;
            }
        }
        (!unseen && max_sim >= self.cutoff, max_sim, unseen)
    }
}

/// Named predictor entry mounted on the `CriticAssetHub`.
pub struct MountedOracle {
    pub oracle: TrainableBootstrapOracle,
    pub calibrator: Option<ResidualCalibratedAsset>,
    pub domain: Option<MorganDomain>,
    pub is_audit_only: bool,
}

impl MountedOracle {
    pub fn predict_from_desc(
        &self,
        g: &HeavyMolGraph,
        atom_repr: &[f32],
        coords_a: &[[f64; 3]],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let preds = self.oracle.predict(atom_repr, coords_a)?;
        let y_base: Vec<f64> = preds.mean.iter().map(|&v| v as f64).collect();
        let s_base: Vec<f64> = preds.std.iter().map(|&v| v as f64).collect();
        if let Some(ref cal) = self.calibrator {
            Ok(cal.predict(g, &y_base, &s_base))
        } else {
            Ok((y_base, s_base))
        }
    }

    pub fn channel_index(&self, channel: &str) -> Option<usize> {
        self.oracle
            .inner
            .target_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(channel))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoleculeCriticEvaluation {
    pub legal: bool,
    pub clean_charge: bool,
    pub charge_signature: String,
    pub constraints_met: bool,
    pub hard_constraints_met: bool,
    pub constraint_score: f64,
    pub f_max_ev_a: f64,
    pub f_rms_ev_a: f64,
    pub phys_score: f64,
    pub evaluated: bool,
    pub objective_preds: Vec<f64>,
    pub objective_stds: Vec<f64>,
    pub audit_preds: Option<Vec<f64>>,
    pub sigma_cal_est: Vec<f64>,
    pub is_hit: bool,
    pub is_clean_hit: bool,
    pub valid_score: f64,
    pub prop_score: f64,
    pub rule_metrics: BuiltInRuleMetrics,
}

/// Evaluate a batch of `MolRecord`s through `TaskWeavingSpec`, `MlipSlot`, and mounted `MountedOracle`s.
pub async fn evaluate_molecules_on_bus(
    spec: &TaskWeavingSpec,
    records: &[MolRecord],
    requested_goals: &[Vec<f64>],
    active_masks: &[Vec<bool>],
    mlip: Option<&mut MlipSlot>,
    oracles: &HashMap<String, MountedOracle>,
    seen_charge_signatures: Option<&HashSet<String>>,
    for_training: bool,
    sharp_concentration: bool,
) -> Result<Vec<MoleculeCriticEvaluation>, String> {
    let n_mols = records.len();
    let k_obj = spec.objectives.len();
    let mut out = Vec::with_capacity(n_mols);
    let mut mlip_opt = mlip;

    for i in 0..n_mols {
        let rec = &records[i];
        let goals = requested_goals.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
        let act = active_masks.get(i).map(|v| v.as_slice()).unwrap_or(&[]);

        let g = HeavyMolGraph::from_record(rec);
        let rules = BuiltInRuleMetrics::evaluate(&g);
        let legal = rec.valence_valid && !rec.atomic_numbers.is_empty();
        let clean_charge = legal && is_clean_organic_charge_state(rec);
        let charge_sig = formal_charge_signature(rec);

        // 1. Zero-step MLIP forces & frozen atomic descriptor (computed once and shared across all 3D oracles!)
        let mut f_max = f64::NAN;
        let mut f_rms = f64::NAN;
        let mut phys_score = 0.0;
        let mut cached_desc: Option<(String, Vec<f32>)> = None;

        if legal {
            if let Some(ref mut m_slot) = mlip_opt {
                if let Ok(mut sys) =
                    MolecularSystem::new(rec.atomic_numbers.clone(), rec.positions.clone(), 0.0, 1)
                {
                    if let Ok(mut ev) = m_slot.eval_async(&sys, true, true).await {
                        let _ = &mut sys;
                        ev.zero_com_forces(None);
                        f_max = ev.max_force_norm(None);
                        f_rms = ev.rms_force(None);
                        phys_score = zero_step_physical_reward(f_rms, f_max);
                        if let Some(d) = ev.descriptor {
                            cached_desc = Some((m_slot.family().to_string(), d));
                        }
                    }
                }
            }
        }

        // Helper closure to query any channel `(asset_id, channel)` -> `(val, std, ok)`
        let can_eval_oracle = if for_training { clean_charge } else { legal };
        let mut oracle_cache: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();

        let mut query_asset = |asset_id: &str, channel: &str, allow_audit: bool| -> (f64, f64, bool) {
            if asset_id == "builtin_rules" {
                if !legal {
                    return (f64::NAN, 0.0, false);
                }
                if let Some(v) = rules.channel(channel) {
                    return (v, 0.0, v.is_finite());
                }
                return (f64::NAN, 0.0, false);
            }
            if !can_eval_oracle {
                return (f64::NAN, f64::NAN, false);
            }
            let Some(mo) = oracles.get(asset_id) else {
                // Fallback if `rec.properties` already carries precomputed value (e.g. replay/offline test)
                if let Some(&v) = rec.properties.get(channel) {
                    return (v, 0.10, v.is_finite());
                }
                return (f64::NAN, f64::NAN, false);
            };
            if for_training && mo.is_audit_only && !allow_audit {
                return (f64::NAN, f64::NAN, false);
            }
            let Some(c_idx) = mo.channel_index(channel) else {
                return (f64::NAN, f64::NAN, false);
            };
            if !oracle_cache.contains_key(asset_id) {
                if let Some((ref fam, ref desc)) = cached_desc {
                    if fam == &mo.oracle.backbone_family {
                        if let Ok(pair) = mo.predict_from_desc(&g, desc, &rec.positions) {
                            oracle_cache.insert(asset_id.to_string(), pair);
                        }
                    }
                }
            }
            if let Some((ys, ss)) = oracle_cache.get(asset_id) {
                let y = ys.get(c_idx).copied().unwrap_or(f64::NAN);
                let s = ss.get(c_idx).copied().unwrap_or(0.12);
                (y, s, y.is_finite())
            } else if let Some(&v) = rec.properties.get(channel) {
                (v, 0.12, v.is_finite())
            } else {
                (f64::NAN, f64::NAN, false)
            }
        };

        // 2. Evaluate inequality constraints (`ConstraintChannelSpec`)
        let mut constraints_met = legal;
        let mut hard_constraints_met = legal;
        let mut c_w_sum = 0.0f64;
        let mut c_s_sum = 0.0f64;
        for cspec in &spec.constraints {
            let (val, _, ok) = query_asset(&cspec.asset_id, &cspec.asset_channel, false);
            let sat = ok && cspec.is_satisfied(val);
            constraints_met &= sat;
            if cspec.hard_gate {
                hard_constraints_met &= sat;
            }
            c_w_sum += cspec.weight;
            c_s_sum += cspec.weight * if ok { cspec.smooth_satisfaction_score(val) } else { 0.0 };
        }
        let constraint_score = if spec.constraints.is_empty() {
            if legal { 1.0 } else { 0.0 }
        } else if c_w_sum > 0.0 {
            c_s_sum / c_w_sum
        } else {
            1.0
        };

        // 3. Evaluate directional objectives
        let mut obj_preds = vec![f64::NAN; k_obj];
        let mut obj_stds = vec![f64::NAN; k_obj];
        let mut evaluated = can_eval_oracle;
        let mut rew_w_sum = 0.0f64;
        let mut rew_acc = 0.0f64;
        let mut all_active_hit = true;
        let mut any_active = false;

        for (j, ospec) in spec.objectives.iter().enumerate() {
            let is_act = act.get(j).copied().unwrap_or(true);
            let (y, s, ok) = query_asset(&ospec.asset_id, ospec.resolved_channel(), false);
            obj_preds[j] = y;
            obj_stds[j] = s;
            if is_act {
                any_active = true;
                evaluated &= ok;
                let goal_j = goals.get(j).copied().unwrap_or(0.0);
                if ok && ospec.is_hit(y, goal_j) {
                    // hit on channel j
                } else {
                    all_active_hit = false;
                }
                rew_w_sum += ospec.weight;
                if ok {
                    rew_acc += ospec.weight * ospec.reward(y, goal_j, sharp_concentration);
                }
            }
        }
        if !any_active {
            all_active_hit = false;
        }
        let r_raw = if evaluated && rew_w_sum > 1e-8 {
            rew_acc / rew_w_sum
        } else {
            0.0
        };

        let prop_score = if !evaluated {
            0.0
        } else if hard_constraints_met {
            0.20 * constraint_score + 0.80 * r_raw
        } else {
            (0.18 * constraint_score).min(0.18)
        };

        let valid_score = if !legal {
            -1.0
        } else if !clean_charge {
            -0.35
        } else if !hard_constraints_met {
            0.35 * constraint_score
        } else if seen_charge_signatures.map_or(true, |set| set.contains(&charge_sig)) {
            1.0
        } else {
            0.80
        };

        // 4. Dual-mode Auditors (`cross_oracle` & `single_oracle_ad`) -> `sigma_cal_est`
        let mut audit_preds: Option<Vec<f64>> = None;
        let mut sigma_cal_est = vec![0.19f64; k_obj];

        for (j, ospec) in spec.objectives.iter().enumerate() {
            let aud_spec_opt = spec
                .auditors
                .iter()
                .find(|a| a.objective_name == ospec.name);
            let Some(aud) = aud_spec_opt else {
                sigma_cal_est[j] = if obj_stds[j].is_finite() { obj_stds[j] } else { 0.19 };
                continue;
            };

            match aud.mode {
                AuditorMode::CrossOracle => {
                    if !for_training {
                        if let Some(ref aud_id) = aud.auditor_asset_id {
                            let aud_ch = aud
                                .auditor_channel
                                .as_deref()
                                .unwrap_or_else(|| ospec.resolved_channel());
                            let (ya, _, ok_a) = query_asset(aud_id, aud_ch, true);
                            if ok_a {
                                let ap = audit_preds.get_or_insert_with(|| vec![f64::NAN; k_obj]);
                                ap[j] = ya;
                                if obj_preds[j].is_finite() {
                                    sigma_cal_est[j] = (obj_preds[j] - ya).abs();
                                }
                            }
                        }
                    } else {
                        sigma_cal_est[j] = if obj_stds[j].is_finite() { obj_stds[j] } else { 0.19 };
                    }
                }
                AuditorMode::SingleOracleAd => {
                    let s_ens = if obj_stds[j].is_finite() { obj_stds[j] } else { 0.12 };
                    let mut p_ad = 0.0f64;
                    let mut p_extrap = 0.0f64;
                    if let Some(mo) = oracles.get(&ospec.asset_id) {
                        if let Some(ref dom) = mo.domain {
                            if legal {
                                let (_, s_max, unseen) = dom.check(rec, &g);
                                p_ad = 0.55 * (aud.tanimoto_cutoff + 0.08 - s_max).max(0.0)
                                    + if unseen { 0.50 } else { 0.0 };
                            } else {
                                p_ad = 0.30;
                            }
                            if let (Some(c_idx), Some(&g_req)) =
                                (mo.channel_index(ospec.resolved_channel()), goals.get(j))
                            {
                                if let Some(&[b_lo, b_hi]) = dom.calibrated_target_bounds.get(c_idx) {
                                    if b_lo.is_finite() && b_hi.is_finite() {
                                        let d_out = (b_lo - g_req).max(g_req - b_hi).max(0.0);
                                        if d_out > 0.02 {
                                            p_extrap = (0.14 + 0.80 * d_out).min(0.45);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let p_strain = if f_max.is_finite() && f_max > 4.0 {
                        (0.05 * (f_max - 4.0)).min(0.40)
                    } else {
                        0.0
                    };
                    sigma_cal_est[j] =
                        (s_ens * s_ens + p_ad * p_ad + p_extrap * p_extrap + p_strain * p_strain).sqrt();
                }
            }
        }

        let is_hit = evaluated && all_active_hit;
        let is_clean_hit = is_hit && clean_charge && constraints_met;

        out.push(MoleculeCriticEvaluation {
            legal,
            clean_charge,
            charge_signature: charge_sig,
            constraints_met,
            hard_constraints_met,
            constraint_score,
            f_max_ev_a: f_max,
            f_rms_ev_a: f_rms,
            phys_score,
            evaluated,
            objective_preds: obj_preds,
            objective_stds: obj_stds,
            audit_preds,
            sigma_cal_est,
            is_hit,
            is_clean_hit,
            valid_score,
            prop_score,
            rule_metrics: rules,
        });
    }

    Ok(out)
}

// ============================================================================
// Disentangled Group Advantages (`grpo.py:31-130`)
// ============================================================================

#[inline]
fn mean_and_std_unbiased_false(vals: &[f32]) -> (f32, f32) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let m = vals.iter().sum::<f32>() / (vals.len() as f32);
    let v = vals.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / (vals.len() as f32);
    (m, v.sqrt())
}

/// Strictly monotonic per-group advantage normalization:
/// `Clean Low-Strain Hit > Clean Miss > Exotic Zwitterion > Invalid Graph`.
pub fn compute_disentangled_group_advantages(
    evals: &[MoleculeCriticEvaluation],
    w_valid: f32,
    w_phys: f32,
    w_prop: f32,
) -> (Vec<f32>, BTreeMap<String, Vec<f32>>) {
    let g = evals.len();
    if g == 0 {
        return (Vec::new(), BTreeMap::new());
    }

    // 1. Channel 1: adv_valid
    let v_scores: Vec<f32> = evals.iter().map(|e| e.valid_score as f32).collect();
    let (v_mean, v_std) = mean_and_std_unbiased_false(&v_scores);
    let adv_valid: Vec<f32> = if v_std > 1e-6 {
        v_scores.iter().map(|&v| (v - v_mean) / v_std.max(1e-6)).collect()
    } else {
        vec![0.0; g]
    };

    // 2. Channel 2: adv_phys
    let mut adv_phys = vec![0.0f32; g];
    let val_idx: Vec<usize> = (0..g).filter(|&i| evals[i].legal).collect();
    if w_phys > 0.0 && !val_idx.is_empty() {
        if val_idx.len() >= 2 {
            let p_vals: Vec<f32> = val_idx.iter().map(|&i| evals[i].phys_score as f32).collect();
            let (pm, ps) = mean_and_std_unbiased_false(&p_vals);
            if ps > 1e-6 {
                for (&idx, &pv) in val_idx.iter().zip(p_vals.iter()) {
                    adv_phys[idx] = ((pv - pm) / ps.max(0.05)).clamp(-1.5, 2.0);
                }
            }
        } else {
            let idx = val_idx[0];
            adv_phys[idx] = (evals[idx].phys_score as f32).clamp(0.0, 1.0);
        }
        if val_idx.len() < g {
            for i in 0..g {
                if !evals[i].legal {
                    adv_phys[i] = -0.65;
                }
            }
            let pm_all = adv_phys.iter().sum::<f32>() / (g as f32);
            for v in &mut adv_phys {
                *v -= pm_all;
            }
        }
    }

    // 3. Channel 3: adv_prop
    let mut adv_prop = vec![0.0f32; g];
    let ev_idx: Vec<usize> = (0..g).filter(|&i| evals[i].evaluated).collect();
    if w_prop > 0.0 && !ev_idx.is_empty() {
        if ev_idx.len() >= 2 {
            let r_vals: Vec<f32> = ev_idx.iter().map(|&i| evals[i].prop_score as f32).collect();
            let (rm, rs) = mean_and_std_unbiased_false(&r_vals);
            if rs > 1e-6 {
                for (&idx, &rv) in ev_idx.iter().zip(r_vals.iter()) {
                    adv_prop[idx] = ((rv - rm) / rs.max(0.05)).clamp(-1.35, 2.0);
                }
            }
        } else {
            let idx = ev_idx[0];
            adv_prop[idx] = (evals[idx].prop_score as f32).clamp(0.0, 1.0);
        }
        if ev_idx.len() < g {
            for i in 0..g {
                if !evals[i].evaluated {
                    adv_prop[i] = -0.85;
                }
            }
            let rm_all = adv_prop.iter().sum::<f32>() / (g as f32);
            for v in &mut adv_prop {
                *v -= rm_all;
            }
        }
    }

    let mut comb: Vec<f32> = (0..g)
        .map(|i| w_valid * adv_valid[i] + w_phys * adv_phys[i] + w_prop * adv_prop[i])
        .collect();
    let (cm, cs) = mean_and_std_unbiased_false(&comb);
    let denom = cs.max(0.20);
    for v in &mut comb {
        *v = (*v - cm) / denom;
    }

    let mut ch = BTreeMap::new();
    ch.insert("adv_valid".to_string(), adv_valid);
    ch.insert("adv_phys".to_string(), adv_phys);
    ch.insert("adv_prop".to_string(), adv_prop);
    (comb, ch)
}

// ============================================================================
// `PhysicsConstrainedCoordRefiner` (`src/loom/weaving/registry.py:874-990`)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordRefinerOptions {
    pub prop_weight: f64,
    pub step_size: f64,
    pub max_atom_disp: f64,
    pub force_gate_ev_a: f64,
    pub tolerance: f64,
    pub n_steps: usize,
}

impl Default for CoordRefinerOptions {
    fn default() -> Self {
        Self {
            prop_weight: 0.35,
            step_size: 0.010,
            max_atom_disp: 0.025,
            force_gate_ev_a: 2.50,
            tolerance: 0.25,
            n_steps: 12,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordRefineResult {
    pub positions: Vec<[f64; 3]>,
    pub initial_pred: f64,
    pub final_pred: f64,
    pub final_fmax_ev_a: f64,
    pub steps_taken: usize,
}

/// Jointly refine 3D coordinates `R` along MLIP forces `F_mlip` + strain-gated analytical
/// property gradient `-grad_coords(y)` while strictly preserving the initial 3D covalent graph!
pub async fn refine_coords_physics_constrained(
    rec: &MolRecord,
    mlip: &mut MlipSlot,
    oracle: &TrainableBootstrapOracle,
    channel_idx: usize,
    goal: f64,
    opts: &CoordRefinerOptions,
) -> Result<CoordRefineResult, String> {
    let n = rec.atomic_numbers.len();
    if n == 0 {
        return Err("Empty molecule in PhysicsConstrainedCoordRefiner".into());
    }
    let init_g = HeavyMolGraph::from_record(rec);
    let init_sig = canonical_graph_signature(&init_g);

    let mut cur_pos = rec.positions.clone();
    let mut best_pos = cur_pos.clone();
    let mut init_pred = f64::NAN;
    let mut best_pred = f64::NAN;
    let mut best_err = f64::INFINITY;
    let mut last_fmax = 0.0f64;
    let mut steps_taken = 0usize;

    for step in 0..opts.n_steps {
        let sys = MolecularSystem::new(rec.atomic_numbers.clone(), cur_pos.clone(), 0.0, 1)?;
        let mut ev = mlip.eval_async(&sys, true, true).await?;
        let desc = ev.descriptor.clone().ok_or("MLIP did not return descriptor")?;
        ev.zero_com_forces(None);
        ev.clip_forces(15.0);
        let f_max = ev.max_force_norm(None);
        last_fmax = f_max;

        let (y_pred, _std, d_repr, d_coords_rbf) =
            oracle.predict_and_input_vjp(&desc, &cur_pos, channel_idx)?;
        if step == 0 {
            init_pred = y_pred;
        }
        let abs_err = (y_pred - goal).abs();
        if abs_err < best_err {
            best_err = abs_err;
            best_pred = y_pred;
            best_pos.copy_from_slice(&cur_pos);
        }

        let err = y_pred - goal;
        let mut f_prop = vec![[0.0f64; 3]; n];
        if err.abs() > 0.45 * opts.tolerance {
            let d_coords_bb = mlip.vjp_descriptor_coords_async(&sys, &d_repr).await?;
            let mut grad_y = vec![[0.0f64; 3]; n];
            let mut g_mean = [0.0f64; 3];
            for i in 0..n {
                for a in 0..3 {
                    grad_y[i][a] = d_coords_bb[i][a] + d_coords_rbf[i][a];
                    g_mean[a] += grad_y[i][a] / (n as f64);
                }
            }
            let coef = -(err / opts.tolerance.max(1e-3)).clamp(-2.0, 2.0);
            for i in 0..n {
                for a in 0..3 {
                    f_prop[i][a] = coef * (grad_y[i][a] - g_mean[a]);
                }
            }
        }

        // Strain-gated projection (`f_max > force_gate_ev_a` and `F_mlip · F_prop < 0`)
        if f_max > opts.force_gate_ev_a {
            let mut f_dot_f = 0.0f64;
            let mut f_dot_p = 0.0f64;
            for i in 0..n {
                for a in 0..3 {
                    let fv = ev.forces_ev_a[i][a];
                    let pv = f_prop[i][a];
                    f_dot_f += fv * fv;
                    f_dot_p += fv * pv;
                }
            }
            if f_dot_f > 1e-8 && f_dot_p < 0.0 {
                let proj = f_dot_p / f_dot_f;
                for i in 0..n {
                    for a in 0..3 {
                        f_prop[i][a] -= proj * ev.forces_ev_a[i][a];
                    }
                }
            }
        }

        // Box-clipped step update & covalent graph identity check
        let mut next_pos = cur_pos.clone();
        for i in 0..n {
            for a in 0..3 {
                let dr = (opts.step_size * (ev.forces_ev_a[i][a] + opts.prop_weight * f_prop[i][a]))
                    .clamp(-opts.max_atom_disp, opts.max_atom_disp);
                next_pos[i][a] += dr;
            }
        }

        let cand = MolRecord::from_point_cloud(&rec.title, &rec.atomic_numbers, &next_pos, false);
        if !cand.valence_valid || cand.atomic_numbers.len() != n {
            break;
        }
        let cand_sig = canonical_graph_signature(&HeavyMolGraph::from_record(&cand));
        if cand_sig != init_sig {
            break;
        }

        cur_pos = next_pos;
        steps_taken = step + 1;
    }

    // Final evaluation on `cur_pos` if `steps_taken > 0`
    if steps_taken > 0 {
        let sys = MolecularSystem::new(rec.atomic_numbers.clone(), cur_pos.clone(), 0.0, 1)?;
        if let Ok(mut ev) = mlip.eval_async(&sys, true, true).await {
            ev.zero_com_forces(None);
            last_fmax = ev.max_force_norm(None);
            if let Some(desc) = ev.descriptor {
                if let Ok(preds) = oracle.predict(&desc, &cur_pos) {
                    let yp = preds.mean[channel_idx] as f64;
                    if (yp - goal).abs() < best_err {
                        best_pred = yp;
                        best_pos = cur_pos;
                    }
                }
            }
        }
    }

    Ok(CoordRefineResult {
        positions: best_pos,
        initial_pred: init_pred,
        final_pred: best_pred,
        final_fmax_ev_a: last_fmax,
        steps_taken,
    })
}
