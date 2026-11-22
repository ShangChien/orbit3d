//! Declarative Task Weaving Specification (`TaskWeavingSpec`, `ObjectiveChannelSpec`,
//! `ConstraintChannelSpec`, `AuditorChannelSpec`) matching `src/loom/weaving/spec.py` & `grpo.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveMode {
    Point,
    Interval,
    Minimize,
    Maximize,
}

impl Default for ObjectiveMode {
    fn default() -> Self {
        Self::Point
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveChannelSpec {
    pub name: String,
    pub asset_id: String,
    #[serde(default)]
    pub asset_channel: String,
    #[serde(default)]
    pub mode: ObjectiveMode,
    #[serde(default = "default_tol")]
    pub tolerance: f64,
    #[serde(default = "default_unit")]
    pub unit: String,
    #[serde(default = "default_one")]
    pub weight: f64,
    #[serde(default)]
    pub interval_bounds: Option<(Option<f64>, Option<f64>)>,
}

fn default_tol() -> f64 {
    0.25
}
fn default_unit() -> String {
    "eV".to_string()
}
fn default_one() -> f64 {
    1.0
}

impl ObjectiveChannelSpec {
    pub fn resolved_channel(&self) -> &str {
        if self.asset_channel.is_empty() {
            &self.name
        } else {
            &self.asset_channel
        }
    }

    #[inline]
    pub fn distance_to_goal(&self, y: f64, goal: f64) -> f64 {
        if !y.is_finite() || !goal.is_finite() {
            return f64::INFINITY;
        }
        match self.mode {
            ObjectiveMode::Point => (y - goal).abs(),
            ObjectiveMode::Interval => {
                let (lo, hi) = if let Some((lo_opt, hi_opt)) = self.interval_bounds {
                    (
                        lo_opt.unwrap_or(f64::NEG_INFINITY),
                        hi_opt.unwrap_or(f64::INFINITY),
                    )
                } else {
                    (goal - self.tolerance, goal + self.tolerance)
                };
                (lo - y).max(0.0) + (y - hi).max(0.0)
            }
            ObjectiveMode::Minimize => (y - goal).max(0.0),
            ObjectiveMode::Maximize => (goal - y).max(0.0),
        }
    }

    #[inline]
    pub fn is_hit(&self, y: f64, goal: f64) -> bool {
        if !y.is_finite() {
            return false;
        }
        let d = self.distance_to_goal(y, goal);
        match self.mode {
            ObjectiveMode::Interval => d <= 1e-8,
            _ => d <= self.tolerance,
        }
    }

    #[inline]
    pub fn reward(&self, y: f64, goal: f64, sharp_concentration: bool) -> f64 {
        let d = self.distance_to_goal(y, goal);
        property_distance_reward(d, self.tolerance, sharp_concentration)
    }
}

#[inline]
fn sigmoid_f64(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Hybrid Cauchy (long-range pull) + Gaussian (concentration) + smooth hit-window bonus in `(0, 1]`.
#[inline]
pub fn property_distance_reward(distance: f64, tolerance: f64, sharp_concentration: bool) -> f64 {
    if !distance.is_finite() {
        return 0.0;
    }
    let tol = tolerance.max(1e-4);
    let d = distance.max(0.0);
    let u = d / tol;
    if sharp_concentration {
        let g = d / (0.65 * tol);
        let base = 0.30 / (1.0 + u * u) + 0.40 * (-0.5 * g * g).exp();
        let hit_bonus = 0.30 * sigmoid_f64((tol - d) / (0.12 * tol));
        base + hit_bonus
    } else {
        let base = 0.40 / (1.0 + u) + 0.35 * (-0.5 * u * u).exp();
        let hit_bonus = 0.25 * sigmoid_f64((tol - d) / (0.16 * tol));
        base + hit_bonus
    }
}

/// Smooth bounded zero-step MLIP force/strain reward in `(0, 1]`.
#[inline]
pub fn zero_step_physical_reward(f_rms_ev_a: f64, f_max_ev_a: f64) -> f64 {
    if !f_rms_ev_a.is_finite() || !f_max_ev_a.is_finite() {
        return 0.0;
    }
    let r_rms = (-(f_rms_ev_a - 0.15).max(0.0) / 0.80).exp();
    let r_max = (-(f_max_ev_a - 0.40).max(0.0) / 2.00).exp();
    r_rms * r_max
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintOp {
    Leq,
    Geq,
    Between,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConstraintThreshold {
    Scalar(f64),
    Range([f64; 2]),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintChannelSpec {
    pub name: String,
    #[serde(default = "default_builtin_rules")]
    pub asset_id: String,
    pub asset_channel: String,
    pub op: ConstraintOp,
    pub threshold: ConstraintThreshold,
    #[serde(default = "default_soft_margin")]
    pub soft_margin: f64,
    #[serde(default = "default_constraint_weight")]
    pub weight: f64,
    #[serde(default = "default_true")]
    pub hard_gate: bool,
}

fn default_builtin_rules() -> String {
    "builtin_rules".to_string()
}
fn default_soft_margin() -> f64 {
    0.50
}
fn default_constraint_weight() -> f64 {
    0.60
}
fn default_true() -> bool {
    true
}

impl ConstraintChannelSpec {
    #[inline]
    pub fn violation(&self, v: f64) -> f64 {
        if !v.is_finite() {
            return f64::INFINITY;
        }
        match (self.op, self.threshold) {
            (ConstraintOp::Leq, ConstraintThreshold::Scalar(thr)) => (v - thr).max(0.0),
            (ConstraintOp::Geq, ConstraintThreshold::Scalar(thr)) => (thr - v).max(0.0),
            (ConstraintOp::Between, ConstraintThreshold::Range([lo, hi])) => {
                (lo - v).max(0.0) + (v - hi).max(0.0)
            }
            (ConstraintOp::Between, ConstraintThreshold::Scalar(thr)) => (v - thr).abs(),
            (ConstraintOp::Leq, ConstraintThreshold::Range([_, hi])) => (v - hi).max(0.0),
            (ConstraintOp::Geq, ConstraintThreshold::Range([lo, _])) => (lo - v).max(0.0),
        }
    }

    #[inline]
    pub fn is_satisfied(&self, v: f64) -> bool {
        let viol = self.violation(v);
        viol.is_finite() && viol <= 1e-8
    }

    #[inline]
    pub fn smooth_satisfaction_score(&self, v: f64) -> f64 {
        let viol = self.violation(v);
        if !viol.is_finite() {
            0.0
        } else {
            (-viol / self.soft_margin.max(1e-6)).exp()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditorMode {
    CrossOracle,
    SingleOracleAd,
}

impl Default for AuditorMode {
    fn default() -> Self {
        Self::SingleOracleAd
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditorChannelSpec {
    pub name: String,
    pub objective_name: String,
    #[serde(default)]
    pub mode: AuditorMode,
    #[serde(default)]
    pub auditor_asset_id: Option<String>,
    #[serde(default)]
    pub auditor_channel: Option<String>,
    #[serde(default = "default_disagreement_cap")]
    pub disagreement_cap: f64,
    #[serde(default = "default_tanimoto_cutoff")]
    pub tanimoto_cutoff: f64,
}

fn default_disagreement_cap() -> f64 {
    0.35
}
fn default_tanimoto_cutoff() -> f64 {
    0.25
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskWeavingSpec {
    pub task_name: String,
    #[serde(default)]
    pub description: String,
    pub objectives: Vec<ObjectiveChannelSpec>,
    #[serde(default)]
    pub constraints: Vec<ConstraintChannelSpec>,
    #[serde(default)]
    pub auditors: Vec<AuditorChannelSpec>,
    #[serde(default)]
    pub unphysical_pair_rule: Option<String>,
    #[serde(default = "default_true")]
    pub reconcile_valence: bool,
}

impl TaskWeavingSpec {
    /// Default synthesizable orbital discovery task matching `configs/tasks/oe62_orbital_synthesizable.yaml`.
    pub fn default_synthesizable_orbital(prop_name: &str, reward_asset_id: &str) -> Self {
        Self {
            task_name: format!("synthesizable_{prop_name}"),
            description: format!("Synthesizable & reactive-clean {prop_name} discovery (SA <= 3.5, Alerts == 0, MW <= 550)"),
            objectives: vec![ObjectiveChannelSpec {
                name: prop_name.to_string(),
                asset_id: reward_asset_id.to_string(),
                asset_channel: prop_name.to_string(),
                mode: ObjectiveMode::Point,
                tolerance: 0.25,
                unit: "eV".to_string(),
                weight: 1.0,
                interval_bounds: None,
            }],
            constraints: vec![
                ConstraintChannelSpec {
                    name: "synthesizable_sa".to_string(),
                    asset_id: "builtin_rules".to_string(),
                    asset_channel: "sa_score".to_string(),
                    op: ConstraintOp::Leq,
                    threshold: ConstraintThreshold::Scalar(3.5),
                    soft_margin: 0.60,
                    weight: 0.70,
                    hard_gate: true,
                },
                ConstraintChannelSpec {
                    name: "reactive_clean".to_string(),
                    asset_id: "builtin_rules".to_string(),
                    asset_channel: "smarts_alert".to_string(),
                    op: ConstraintOp::Leq,
                    threshold: ConstraintThreshold::Scalar(0.0),
                    soft_margin: 0.50,
                    weight: 0.80,
                    hard_gate: true,
                },
                ConstraintChannelSpec {
                    name: "mol_weight_cap".to_string(),
                    asset_id: "builtin_rules".to_string(),
                    asset_channel: "mol_weight".to_string(),
                    op: ConstraintOp::Leq,
                    threshold: ConstraintThreshold::Scalar(550.0),
                    soft_margin: 60.0,
                    weight: 0.35,
                    hard_gate: false,
                },
            ],
            auditors: vec![AuditorChannelSpec {
                name: format!("{prop_name}_ad_auditor"),
                objective_name: prop_name.to_string(),
                mode: AuditorMode::SingleOracleAd,
                auditor_asset_id: None,
                auditor_channel: None,
                disagreement_cap: 0.35,
                tanimoto_cutoff: 0.25,
            }],
            unphysical_pair_rule: None,
            reconcile_valence: true,
        }
    }

    #[inline]
    pub fn evaluate_unphysical_2d(&self, x0: f64, x1: f64) -> bool {
        let Some(ref rule) = self.unphysical_pair_rule else {
            return false;
        };
        let norm: String = rule
            .chars()
            .filter(|c| !c.is_whitespace())
            .flat_map(|c| c.to_lowercase())
            .collect();
        match norm.as_str() {
            "x>=y" | "axis0>=axis1" | "homo>=lumo" => x0 >= x1,
            "x>y" | "axis0>axis1" => x0 > x1,
            "x<=y" | "axis0<=axis1" => x0 <= x1,
            _ => false,
        }
    }
}
