use crate::config::PrecomputedCond;
use crate::cpu_backend::{
    add_atom_head_step_emb, embed_octree_token, eval_atom_head_axis, eval_tree_head, forward_decoder_step_cpu,
    precompute_condition, CpuKvSlot,
};
use crate::format::ModelBundle;
use crate::gpu_backend::{StepHeadMode, WgpuDecoderBackend, KV_TOTAL_SLOT_TOKENS, MAX_MOD_BRANCHES, MAX_STEP_TOKENS};
use crate::refiner_critic::{
    compute_coarse_voxel_geometry, compute_element_cloud_features, eval_stage1_critic, eval_stage2_critic,
    run_post_tree_refiner, CriticPrediction,
};
use crate::valence::{
    build_raw_point_cloud_graph, elem_id_to_symbol, finalize_sampled_molecule, max_coord_element, rcov_angstrom,
    reconcile_point_cloud, symbol_to_elem_id, SampledUni3darMolecule,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const SUBCELL_ORDERS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// Deterministic SplitMix64 PRNG (identical across x86_64, aarch64, and wasm32).
#[derive(Clone, Debug)]
pub struct DetRng {
    state: u64,
}

impl DetRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Categorical sampling at `temperature`, returning `(chosen_idx, log_softmax(orig_logits)[chosen_idx])`.
    pub fn sample_logits_with_score(&mut self, logits: &[f32], temperature: f32) -> (usize, f32) {
        let inv_t = 1.0 / temperature.max(1e-4);
        let mut max_orig = f32::NEG_INFINITY;
        let mut max_scaled = f32::NEG_INFINITY;
        let mut first_finite = 0usize;
        for (i, &l) in logits.iter().enumerate() {
            if l.is_finite() {
                if max_orig == f32::NEG_INFINITY {
                    first_finite = i;
                }
                if l > max_orig {
                    max_orig = l;
                }
                let sc = l * inv_t;
                if sc > max_scaled {
                    max_scaled = sc;
                }
            }
        }
        if !max_orig.is_finite() {
            return (0, -20.0);
        }

        let mut sum_p = 0.0f32;
        let mut sum_orig = 0.0f32;
        let mut probs = vec![0.0f32; logits.len()];
        for (i, &l) in logits.iter().enumerate() {
            if l.is_finite() {
                let p = (l * inv_t - max_scaled).exp();
                probs[i] = p;
                sum_p += p;
                sum_orig += (l - max_orig).exp();
            }
        }
        let inv_p = 1.0 / sum_p.max(1e-20);
        let u = self.next_f32();
        let mut cdf = 0.0f32;
        let mut chosen = first_finite;
        for (i, &p) in probs.iter().enumerate() {
            if p > 0.0 {
                chosen = i;
                cdf += p * inv_p;
                if u <= cdf {
                    break;
                }
            }
        }
        let log_prob_orig = (logits[chosen] - max_orig) - sum_orig.max(1e-20).ln();
        (chosen, log_prob_orig)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uni3darSampleOptions {
    #[serde(default)]
    pub budget_atoms: Option<usize>,
    #[serde(default = "default_spec_scale")]
    pub speculative_scale: usize,
    #[serde(default = "default_cfg_scale")]
    pub cfg_scale: f32,
    #[serde(default = "default_one")]
    pub xyz_cfg_scale: f32,
    #[serde(default = "default_cfg_decay")]
    pub cfg_tree_decay: f32,
    #[serde(default = "default_budget_g")]
    pub budget_guidance_scale: f32,
    #[serde(default = "default_budget_tol")]
    pub leaf_budget_tol: f32,
    #[serde(default = "default_tree_temp")]
    pub tree_temp: f32,
    #[serde(default = "default_atom_temp")]
    pub atom_temp: f32,
    #[serde(default = "default_xyz_temp")]
    pub xyz_temp: f32,
    #[serde(default)]
    pub allowed_elements: Option<Vec<String>>,
    #[serde(default = "default_true")]
    pub geometric_guard: bool,
    #[serde(default = "default_true")]
    pub even_electron_parity: bool,
    #[serde(default = "default_true")]
    pub use_refiner: bool,
    #[serde(default = "default_true")]
    pub reconcile_valence: bool,
    #[serde(default = "default_single_mol_mode")]
    pub single_molecule_mode: String,
    #[serde(default)]
    pub cond_targets: Option<Vec<f32>>,
    #[serde(default)]
    pub cond_active: Option<Vec<bool>>,
    #[serde(default)]
    pub max_decode_steps: Option<usize>,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub force_cpu: bool,
}

fn default_spec_scale() -> usize {
    2
}
fn default_cfg_scale() -> f32 {
    2.0
}
fn default_one() -> f32 {
    1.0
}
fn default_cfg_decay() -> f32 {
    0.25
}
fn default_budget_g() -> f32 {
    0.45
}
fn default_budget_tol() -> f32 {
    2.0
}
fn default_tree_temp() -> f32 {
    0.85
}
fn default_atom_temp() -> f32 {
    0.30
}
fn default_xyz_temp() -> f32 {
    0.25
}
fn default_true() -> bool {
    true
}
fn default_single_mol_mode() -> String {
    "largest_fragment".to_string()
}
fn default_seed() -> u64 {
    42
}

impl Default for Uni3darSampleOptions {
    fn default() -> Self {
        Self {
            budget_atoms: Some(26),
            speculative_scale: 2,
            cfg_scale: 2.0,
            xyz_cfg_scale: 1.0,
            cfg_tree_decay: 0.25,
            budget_guidance_scale: 0.45,
            leaf_budget_tol: 2.0,
            tree_temp: 0.85,
            atom_temp: 0.30,
            xyz_temp: 0.25,
            allowed_elements: None,
            geometric_guard: true,
            even_electron_parity: true,
            use_refiner: true,
            reconcile_valence: true,
            single_molecule_mode: "largest_fragment".to_string(),
            cond_targets: None,
            cond_active: None,
            max_decode_steps: None,
            seed: 42,
            force_cpu: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct QueueItem {
    pos: [i32; 3],
    typ: i32,
    level: i32,
}

#[derive(Clone, Copy, Debug)]
struct DecToken {
    typ: usize,
    xyz_bins: [usize; 3],
    level: i32,
    rem_a: i32,
    phy_pos: [f32; 3],
    score: f32,
    is_root: bool,
}

#[derive(Clone, Debug)]
struct BranchState {
    rng: DetRng,
    q_buf: Vec<QueueItem>,
    q_head: usize,
    tokens: Vec<DecToken>,
    rem_lvl_cells: Vec<f32>,
    init_rem_a: i32,
    budget_f: f32,
    step_idx: usize,
    kv_len: usize,
    paused_at_leaf: bool,
    finished: bool,
    empty_coarse: bool,
    // Placed leaf atoms for fast O(N) geometric guard & parity
    placed_elems: Vec<usize>,
    placed_pos: Vec<[f32; 3]>,
    acc_parity: usize,
    // Critic hidden accumulators
    cs_h_root: Vec<f32>,
    cs_h_coarse_sum: Vec<f32>,
    cs_h_coarse_cnt: f32,
    cs_h_l1_sum: Vec<f32>,
    cs_h_l1_cnt: f32,
    cs_h_l1_last: Vec<f32>,
    s2_h_atom_sum: Vec<f32>,
    s2_h_atom_cnt: f32,
    s2_h_hvy_sum: Vec<f32>,
    s2_h_hvy_cnt: f32,
    s2_h_atom_last: Vec<f32>,
    coarse_geom_14: [f32; 14],
    stage1_util: Option<f32>,
    stage2_pred: Option<CriticPrediction>,
    refined_pos: Option<Vec<[f32; 3]>>,
    bond_order_probs: Option<Vec<f32>>,
}

impl BranchState {
    fn new(bundle: &ModelBundle, seed: u64, budget_atoms: Option<usize>) -> Self {
        let bb = &bundle.header.backbone;
        let d = bb.emb_dim;
        let ml = bb.merge_level as i32;
        let cls_p = bb.cls_phy_pos();
        let budget_f = budget_atoms.unwrap_or(0) as f32;
        let init_rem_a = if bb.non_trapping_budget && bb.has_budget_emb && budget_f > 0.0 {
            budget_atoms.unwrap_or(0).max(1) as i32
        } else {
            0
        };

        let mut rem_lvl_cells = vec![0.0f32; (bb.merge_level + 2).max(8)];
        rem_lvl_cells[bb.merge_level] = 1.0;

        let root_tok = DecToken {
            typ: bb.atom_mask_id,
            xyz_bins: [bb.xyz_null_id; 3],
            level: ml,
            rem_a: init_rem_a,
            phy_pos: [cls_p, cls_p, cls_p],
            score: 0.0,
            is_root: true,
        };
        let q_init = QueueItem {
            pos: [0, 0, 0],
            typ: -1,
            level: ml,
        };

        Self {
            rng: DetRng::new(seed),
            q_buf: vec![q_init],
            q_head: 0,
            tokens: vec![root_tok],
            rem_lvl_cells,
            init_rem_a,
            budget_f,
            step_idx: 0,
            kv_len: 0,
            paused_at_leaf: false,
            finished: false,
            empty_coarse: false,
            placed_elems: Vec::new(),
            placed_pos: Vec::new(),
            acc_parity: 0,
            cs_h_root: vec![0.0; d],
            cs_h_coarse_sum: vec![0.0; d],
            cs_h_coarse_cnt: 0.0,
            cs_h_l1_sum: vec![0.0; d],
            cs_h_l1_cnt: 0.0,
            cs_h_l1_last: vec![0.0; d],
            s2_h_atom_sum: vec![0.0; d],
            s2_h_atom_cnt: 0.0,
            s2_h_hvy_sum: vec![0.0; d],
            s2_h_hvy_cnt: 0.0,
            s2_h_atom_last: vec![0.0; d],
            coarse_geom_14: [0.0; 14],
            stage1_util: None,
            stage2_pred: None,
            refined_pos: None,
            bond_order_probs: None,
        }
    }

    fn build_stage1_h_concat(&self, d: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(4 * d);
        out.extend_from_slice(&self.cs_h_root);
        let inv_c = 1.0 / self.cs_h_coarse_cnt.max(1.0);
        let c_mean: Vec<f32> = self.cs_h_coarse_sum.iter().map(|&v| v * inv_c).collect();
        let l1_mean: Vec<f32> = if self.cs_h_l1_cnt > 0.0 {
            let inv_l1 = 1.0 / self.cs_h_l1_cnt;
            self.cs_h_l1_sum.iter().map(|&v| v * inv_l1).collect()
        } else {
            c_mean.clone()
        };
        let l1_last: &[f32] = if self.cs_h_l1_cnt > 0.0 {
            &self.cs_h_l1_last
        } else {
            &c_mean
        };
        out.extend_from_slice(&c_mean);
        out.extend_from_slice(&l1_mean);
        out.extend_from_slice(l1_last);
        out
    }

    fn build_stage2_h_concat(&self, d: usize) -> Vec<f32> {
        let s1 = self.build_stage1_h_concat(d);
        let l1_mean = &s1[2 * d..3 * d];
        let l1_last = &s1[3 * d..4 * d];
        let a_mean: Vec<f32> = if self.s2_h_atom_cnt > 0.0 {
            let inv_a = 1.0 / self.s2_h_atom_cnt;
            self.s2_h_atom_sum.iter().map(|&v| v * inv_a).collect()
        } else {
            l1_mean.to_vec()
        };
        let hv_mean: Vec<f32> = if self.s2_h_hvy_cnt > 0.0 {
            let inv_hv = 1.0 / self.s2_h_hvy_cnt;
            self.s2_h_hvy_sum.iter().map(|&v| v * inv_hv).collect()
        } else {
            a_mean.clone()
        };
        let a_last: &[f32] = if self.s2_h_atom_cnt > 0.0 {
            &self.s2_h_atom_last
        } else {
            l1_last
        };

        let mut out = Vec::with_capacity(6 * d);
        out.extend_from_slice(&s1[..3 * d]); // h_root, c_mean, l1_mean
        out.extend_from_slice(&a_mean);
        out.extend_from_slice(&hv_mean);
        out.extend_from_slice(a_last);
        out
    }

    fn compute_coarse_geom(&mut self, atom_mask_id: usize, grid_len: f32) {
        let mut levels: Vec<i32> = self.tokens.iter().map(|t| t.level).collect();
        let mut types: Vec<usize> = self.tokens.iter().map(|t| t.typ).collect();
        let mut phy_pos: Vec<[f32; 3]> = self.tokens.iter().map(|t| t.phy_pos).collect();
        let mut scores: Vec<f32> = self.tokens.iter().map(|t| t.score).collect();
        // Include peeked `level == 0` mask token if paused at leaf boundary (matching `uni3dar_slot.py` line 4025)
        if !self.empty_coarse && self.q_head < self.q_buf.len() {
            let pk = self.q_buf[self.q_head];
            if pk.level == 0 && pk.typ == (atom_mask_id as i32) {
                levels.push(0);
                types.push(atom_mask_id);
                phy_pos.push([
                    (pk.pos[0] as f32 + 0.5) * grid_len,
                    (pk.pos[1] as f32 + 0.5) * grid_len,
                    (pk.pos[2] as f32 + 0.5) * grid_len,
                ]);
                scores.push(0.0);
            }
        }
        self.coarse_geom_14 = compute_coarse_voxel_geometry(&levels, &types, &phy_pos, &scores, atom_mask_id);
    }
}

fn resolve_funnel(spec_scale: usize, has_critics: bool) -> (usize, usize) {
    if !has_critics || spec_scale <= 1 {
        return (1, 1);
    }
    let k = spec_scale.clamp(2, 12);
    let m_s1 = match k {
        2 => 8,
        3 => 12,
        4 => 16,
        _ => 4 * k,
    };
    let m_s2 = match k {
        2 => 2,
        3 => 3,
        _ => k.min(8),
    };
    (m_s1, m_s2)
}

fn resolve_max_decode_steps(bundle: &ModelBundle, opts: &Uni3darSampleOptions) -> usize {
    let ml = bundle.header.backbone.merge_level;
    let lvl_extra = 8 * ml.saturating_sub(6);
    if let Some(ms) = opts.max_decode_steps {
        return ms.max(32);
    }
    if let Some(b) = opts.budget_atoms {
        if b > 0 {
            return (3 * b + 48 + lvl_extra).max(96);
        }
    }
    if ml >= 7 {
        320
    } else {
        256
    }
}

fn prefix_phy_positions(bundle: &ModelBundle) -> Vec<[f32; 3]> {
    let Some(cc) = &bundle.header.conditioner else {
        return Vec::new();
    };
    let p = cc.num_prefix_tokens;
    if p == 0 {
        return Vec::new();
    }
    let bb = &bundle.header.backbone;
    let cls_p = bb.cls_phy_pos();
    if cc.rope_center_prefix {
        vec![[cls_p, cls_p, cls_p]; p]
    } else {
        (0..p)
            .map(|i| {
                let off = ((i as f32) - (p as f32)) * 16.0 * bb.xyz_resolution;
                let c = cls_p + off;
                [c, c, c]
            })
            .collect()
    }
}

/// Apply hierarchical octree volume packing priors (`L5 -> L4 -> L3 -> L2 -> L1`).
fn apply_hierarchical_budget_guidance(
    tree_logits: &mut [f32],
    rel_lvl: i32,
    budget_f: f32,
    rem_lvl_cells: &[f32],
    leaf_budget_tol: f32,
    guidance_scale: f32,
) {
    if budget_f <= 0.0 || rel_lvl < 1 || rel_lvl > 5 || (rel_lvl as usize) >= rem_lvl_cells.len() {
        return;
    }
    let g = guidance_scale;
    let (div, k_hi, reach_hi, pop_cap, w_q, inf_p, slack_f): (f32, f32, f32, f32, f32, f32, f32) = match rel_lvl {
        5 => (3.00, 6.0, 6.5, 8.5, 0.25 * g, 0.0, 0.25),
        4 => (1.36, 5.0, 5.5, 7.5, 0.30 * g, 0.0, 0.18),
        3 => (0.99, 3.4, 3.6, 4.5, 0.35 * g, 0.0, 0.12),
        2 => (0.99, 2.0, 2.0, 2.5, 0.55, -10.0, 0.0),
        1 => (0.99, 1.8, 2.0, 2.5, 0.70, -12.0, 0.0),
        _ => return,
    };
    if w_q <= 0.0 && inf_p == 0.0 {
        return;
    }

    let r_idx = rel_lvl as usize;
    let rem_after = (rem_lvl_cells[r_idx] - 1.0).max(0.0);
    let cur_child = rem_lvl_cells[r_idx - 1];
    let goal_c = (budget_f / div).max(if rel_lvl >= 4 { 2.0 } else { 1.0 });
    let tol_c = if slack_f > 0.0 {
        (slack_f * goal_c).max(leaf_budget_tol)
    } else {
        leaf_budget_tol
    };
    let ideal_k = ((goal_c - cur_child) / (rem_after + 1.0)).clamp(1.0, k_hi);

    let mut feas = [false; 255];
    let mut any_feas = false;
    if inf_p != 0.0 {
        for t in 0..255usize {
            let pop = ((t + 1) as u32).count_ones() as f32;
            let min_reach = cur_child + pop + rem_after;
            let max_reach = cur_child + pop + reach_hi * rem_after;
            let ok = min_reach <= goal_c + tol_c && max_reach >= goal_c - tol_c && pop <= pop_cap;
            feas[t] = ok;
            if ok {
                any_feas = true;
            }
        }
    }

    for t in 0..255usize {
        let pop = ((t + 1) as u32).count_ones() as f32;
        let diff = pop - ideal_k;
        let mut delta = -w_q * diff * diff;
        if pop_cap < 8.0 && pop > pop_cap {
            delta -= 8.0;
        }
        if inf_p != 0.0 && any_feas && !feas[t] {
            delta += inf_p;
        }
        tree_logits[t] += delta;
    }
}

fn update_rem_lvl_cells(rem_lvl_cells: &mut [f32], rel_lvl: i32, chosen_pop: f32) {
    let idx = rel_lvl as usize;
    if rel_lvl >= 1 && idx < rem_lvl_cells.len() {
        rem_lvl_cells[idx] = (rem_lvl_cells[idx] - 1.0).max(0.0);
        rem_lvl_cells[idx - 1] += chosen_pop;
    }
}

/// Prepare placed-atom `(rcov, saturated)` flags matching `_prepare_placed_geometry`.
fn prepare_placed_geometry(placed_elems: &[usize], placed_pos: &[[f32; 3]]) -> (Vec<f32>, Vec<bool>) {
    let k = placed_elems.len();
    let mut rcov = Vec::with_capacity(k);
    let mut maxc = Vec::with_capacity(k);
    for &eid in placed_elems {
        let sym = elem_id_to_symbol(eid);
        rcov.push(rcov_angstrom(sym));
        maxc.push(max_coord_element(sym));
    }
    let mut sat = vec![false; k];
    if k <= 1 {
        return (rcov, sat);
    }
    for i in 0..k {
        let mut val_used = 0i32;
        let mut has_o_nb = false;
        for j in 0..k {
            if i == j {
                continue;
            }
            let dx = placed_pos[i][0] - placed_pos[j][0];
            let dy = placed_pos[i][1] - placed_pos[j][1];
            let dz = placed_pos[i][2] - placed_pos[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let r_pair = rcov[i] + rcov[j];
            if d <= 1.26 * r_pair {
                val_used += 1;
                if placed_elems[j] == 7 {
                    has_o_nb = true;
                }
            }
            if d <= 0.91 * r_pair {
                val_used += 1;
            }
            if d <= 0.83 * r_pair {
                val_used += 1;
            }
        }
        let eff_max = if placed_elems[i] == 6 && has_o_nb { 4 } else { maxc[i] };
        sat[i] = val_used >= eff_max;
    }
    (rcov, sat)
}

fn apply_element_guard_bias(
    logits: &mut [f32],
    vox_origin: [f32; 3],
    grid_len: f32,
    allowed_elem_ids: &[usize],
    placed_pos: &[[f32; 3]],
    p_rcov: &[f32],
    p_sat: &[bool],
) {
    if placed_pos.is_empty() {
        return;
    }
    let k = placed_pos.len();
    let mut d_max = Vec::with_capacity(k);
    let mut d_cen = Vec::with_capacity(k);
    let cen = [
        vox_origin[0] + 0.5 * grid_len,
        vox_origin[1] + 0.5 * grid_len,
        vox_origin[2] + 0.5 * grid_len,
    ];
    for j in 0..k {
        let pj = placed_pos[j];
        let mx = (vox_origin[0] - pj[0]).abs().max((vox_origin[0] + grid_len - pj[0]).abs());
        let my = (vox_origin[1] - pj[1]).abs().max((vox_origin[1] + grid_len - pj[1]).abs());
        let mz = (vox_origin[2] - pj[2]).abs().max((vox_origin[2] + grid_len - pj[2]).abs());
        d_max.push((mx * mx + my * my + mz * mz).sqrt());
        let cx = cen[0] - pj[0];
        let cy = cen[1] - pj[1];
        let cz = cen[2] - pj[2];
        d_cen.push((cx * cx + cy * cy + cz * cz).sqrt());
    }

    for &eid in allowed_elem_ids {
        let sym = elem_id_to_symbol(eid);
        let ra = rcov_angstrom(sym);
        let va = max_coord_element(sym);
        let mut clash = false;
        let mut sat_in = false;
        let mut deg = 0i32;
        for j in 0..k {
            let r_sum = p_rcov[j] + ra;
            if d_max[j] < 0.61 * r_sum {
                clash = true;
            }
            if p_sat[j] && d_cen[j] < 1.22 * r_sum {
                sat_in = true;
            }
            if d_cen[j] < 1.24 * r_sum {
                deg += 1;
            }
        }
        let mut b = 0.0f32;
        if clash {
            b -= 16.0;
        }
        if sat_in {
            b -= 12.0;
        }
        if deg > va {
            b -= 14.0;
        }
        if eid < logits.len() {
            logits[eid] += b;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_zbin_guard_bias(
    logits: &mut [f32],
    vox_origin: [f32; 3],
    chosen_elem: usize,
    chosen_bx: usize,
    chosen_by: usize,
    xyz_res: f32,
    placed_pos: &[[f32; 3]],
    p_rcov: &[f32],
    p_sat: &[bool],
) {
    if placed_pos.is_empty() {
        return;
    }
    let r_new = if chosen_elem < 118 {
        rcov_angstrom(elem_id_to_symbol(chosen_elem))
    } else {
        0.80
    };
    let xa = vox_origin[0] + (chosen_bx as f32) * xyz_res;
    let ya = vox_origin[1] + (chosen_by as f32) * xyz_res;
    let k = placed_pos.len();

    for (bz, l_ref) in logits.iter_mut().enumerate() {
        let za = vox_origin[2] + (bz as f32) * xyz_res;
        let mut pen = 0.0f32;
        for j in 0..k {
            let dx = xa - placed_pos[j][0];
            let dy = ya - placed_pos[j][1];
            let dz = za - placed_pos[j][2];
            let d3 = (dx * dx + dy * dy + dz * dz).max(1e-8).sqrt();
            let r_sum = p_rcov[j] + r_new;
            if d3 < 0.63 * r_sum {
                pen += -25.0 * (0.63 * r_sum - d3) - 8.0;
            }
            if p_sat[j] && d3 < 1.24 * r_sum {
                pen += -12.0 * (1.24 * r_sum - d3) - 4.0;
            }
        }
        *l_ref += pen.max(-40.0);
    }
}

/// Apply post-Transformer prediction (`TreeHead` or 4-axis `AtomHead` + BFS queue expansion) to a single branch.
#[allow(clippy::too_many_arguments)]
fn advance_branch_from_hidden(
    bundle: &ModelBundle,
    br: &mut BranchState,
    dec_y: &[f32],
    dec_u: Option<&[f32]>,
    pre_tree_c: Option<&[f32]>,
    pre_tree_u: Option<&[f32]>,
    opts: &Uni3darSampleOptions,
    allowed_mask: &[bool],
    allowed_elem_ids: &[usize],
    max_decode_steps: usize,
    stop_at_leaf_boundary: bool,
) -> Result<(), String> {
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    br.step_idx += 1;
    let last_tok = *br.tokens.last().unwrap();
    let next_item = br.q_buf[br.q_head];
    br.q_head += 1;

    let need_pred = next_item.typ < 0;
    let is_pred_tree = need_pred && next_item.level > 0;
    let is_pred_atom = need_pred && last_tok.level == 0 && last_tok.typ == bb.atom_mask_id;

    if br.step_idx == 1 {
        br.cs_h_root.copy_from_slice(dec_y);
    }
    if is_pred_tree {
        for c in 0..d {
            br.cs_h_coarse_sum[c] += dec_y[c];
        }
        br.cs_h_coarse_cnt += 1.0;
        if next_item.level == 1 {
            for c in 0..d {
                br.cs_h_l1_sum[c] += dec_y[c];
            }
            br.cs_h_l1_cnt += 1.0;
            br.cs_h_l1_last.copy_from_slice(dec_y);
        }
    }

    let mut pred_typ = next_item.typ.max(0) as usize;
    let mut pred_xyz = [bb.xyz_null_id; 3];
    let mut pred_score = 0.0f32;

    if is_pred_tree {
        let tc_owned;
        let tree_c: &[f32] = if let Some(tc) = pre_tree_c {
            tc
        } else {
            tc_owned = eval_tree_head(bundle, dec_y, 1)?;
            &tc_owned
        };
        let mut tree_logits = if let (Some(du), true) = (dec_u, opts.cfg_scale != 1.0) {
            let tu_owned;
            let tree_u: &[f32] = if let Some(tu) = pre_tree_u {
                tu
            } else {
                tu_owned = eval_tree_head(bundle, du, 1)?;
                &tu_owned
            };
            let decay_mult = if opts.cfg_tree_decay != 0.0 {
                1.0 - opts.cfg_tree_decay * ((3.0 - (next_item.level as f32)).clamp(0.0, 2.0) / 2.0)
            } else {
                1.0
            };
            let eff_cfg = 1.0 + (opts.cfg_scale - 1.0) * decay_mult;
            (0..255).map(|i| tree_u[i] + eff_cfg * (tree_c[i] - tree_u[i])).collect()
        } else {
            tree_c.to_vec()
        };

        apply_hierarchical_budget_guidance(
            &mut tree_logits,
            next_item.level,
            br.budget_f,
            &br.rem_lvl_cells,
            opts.leaf_budget_tol,
            opts.budget_guidance_scale,
        );

        let (t_pred, t_sc) = br.rng.sample_logits_with_score(&tree_logits, opts.tree_temp);
        let pop = ((t_pred + 1) as u32).count_ones() as f32;
        update_rem_lvl_cells(&mut br.rem_lvl_cells, next_item.level, pop);

        pred_typ = bb.tree0_split_id + 1 + t_pred;
        pred_score = t_sc;

        // Expand non-empty octants `b in 0..7` in Morton order
        let bitmask = t_pred + 1;
        let child_lvl = next_item.level - 1;
        let child_step = 1i32 << child_lvl.max(0);
        for b in 0..8usize {
            if (bitmask & (1 << b)) != 0 {
                let off = SUBCELL_ORDERS[b];
                let cpos = [
                    next_item.pos[0] + off[0] * child_step,
                    next_item.pos[1] + off[1] * child_step,
                    next_item.pos[2] + off[2] * child_step,
                ];
                br.q_buf.push(QueueItem {
                    pos: cpos,
                    typ: bb.atom_mask_id as i32,
                    level: child_lvl,
                });
                br.q_buf.push(QueueItem {
                    pos: cpos,
                    typ: -1,
                    level: child_lvl,
                });
            }
        }
    } else if is_pred_atom {
        let mut feat_c = dec_y.to_vec();
        let mut feat_u = dec_u.map(|u| u.to_vec());
        let vox_origin = [
            (next_item.pos[0] as f32) * bb.grid_len,
            (next_item.pos[1] as f32) * bb.grid_len,
            (next_item.pos[2] as f32) * bb.grid_len,
        ];
        let (p_rcov, p_sat) = if opts.geometric_guard && !br.placed_elems.is_empty() {
            prepare_placed_geometry(&br.placed_elems, &br.placed_pos)
        } else {
            (Vec::new(), Vec::new())
        };
        let is_terminal = br.q_head >= br.q_buf.len() || br.step_idx >= max_decode_steps;

        let mut chosen_4 = [0usize; 4];
        let mut sub_sc = 0.0f32;
        for axis in 0..4usize {
            let lc = eval_atom_head_axis(bundle, axis, &feat_c, 1)?;
            let scale_ax = if axis == 0 { opts.cfg_scale } else { opts.xyz_cfg_scale };
            let mut cur_logits = if let (Some(ref fu), true) = (&feat_u, scale_ax != 1.0) {
                let lu = eval_atom_head_axis(bundle, axis, fu, 1)?;
                (0..lc.len()).map(|i| lu[i] + scale_ax * (lc[i] - lu[i])).collect()
            } else {
                lc
            };

            if axis == 0 {
                if opts.geometric_guard && !br.placed_pos.is_empty() {
                    apply_element_guard_bias(
                        &mut cur_logits,
                        vox_origin,
                        bb.grid_len,
                        allowed_elem_ids,
                        &br.placed_pos,
                        &p_rcov,
                        &p_sat,
                    );
                }
                for (id, &ok) in allowed_mask.iter().enumerate() {
                    if !ok {
                        cur_logits[id] = f32::NEG_INFINITY;
                    }
                }
                if opts.even_electron_parity && is_terminal {
                    // Terminal parity guard: require `(acc_parity + elem_parity) % 2 == 0`
                    let mut saved = cur_logits.clone();
                    let mut any_ok = false;
                    for id in 0..cur_logits.len() {
                        if !cur_logits[id].is_finite() {
                            continue;
                        }
                        let ep = if id < 118 { (id + 1) % 2 } else { 0 };
                        if (br.acc_parity + ep) % 2 == 0 {
                            any_ok = true;
                        } else {
                            saved[id] = f32::NEG_INFINITY;
                        }
                    }
                    if any_ok {
                        cur_logits = saved;
                    }
                }
            } else if axis == 3 && opts.geometric_guard && !br.placed_pos.is_empty() {
                apply_zbin_guard_bias(
                    &mut cur_logits,
                    vox_origin,
                    chosen_4[0],
                    chosen_4[1],
                    chosen_4[2],
                    bb.xyz_resolution,
                    &br.placed_pos,
                    &p_rcov,
                    &p_sat,
                );
            }

            let temp = if axis == 0 { opts.atom_temp } else { opts.xyz_temp };
            let (sampled_id, sc) = br.rng.sample_logits_with_score(&cur_logits, temp);
            chosen_4[axis] = sampled_id;
            if axis == 0 {
                sub_sc += sc; // rank_by == "atom" -> score_ratio = [1, 0, 0, 0]
            }
            if axis < 3 {
                add_atom_head_step_emb(bundle, axis, sampled_id, &mut feat_c)?;
                if let Some(ref mut fu) = feat_u {
                    add_atom_head_step_emb(bundle, axis, sampled_id, fu)?;
                }
            }
        }

        pred_typ = chosen_4[0];
        pred_xyz = [chosen_4[1], chosen_4[2], chosen_4[3]];
        pred_score = sub_sc;

        if pred_typ < 118 {
            let apos = [
                vox_origin[0] + (pred_xyz[0] as f32) * bb.xyz_resolution,
                vox_origin[1] + (pred_xyz[1] as f32) * bb.xyz_resolution,
                vox_origin[2] + (pred_xyz[2] as f32) * bb.xyz_resolution,
            ];
            br.placed_elems.push(pred_typ);
            br.placed_pos.push(apos);
            br.acc_parity = (br.acc_parity + ((pred_typ + 1) % 2)) % 2;

            for c in 0..d {
                br.s2_h_atom_sum[c] += dec_y[c];
            }
            br.s2_h_atom_cnt += 1.0;
            br.s2_h_atom_last.copy_from_slice(dec_y);
            if pred_typ != 0 {
                for c in 0..d {
                    br.s2_h_hvy_sum[c] += dec_y[c];
                }
                br.s2_h_hvy_cnt += 1.0;
            }
        }
    }

    // Write predicted token into `br.tokens`
    let edge_len = (1i32 << next_item.level.max(0)) as f32;
    let pred_phy = if next_item.level == 0 {
        [
            (next_item.pos[0] as f32) * bb.grid_len + (pred_xyz[0] as f32) * bb.xyz_resolution,
            (next_item.pos[1] as f32) * bb.grid_len + (pred_xyz[1] as f32) * bb.xyz_resolution,
            (next_item.pos[2] as f32) * bb.grid_len + (pred_xyz[2] as f32) * bb.xyz_resolution,
        ]
    } else {
        [
            ((next_item.pos[0] as f32) + 0.5 * edge_len) * bb.grid_len,
            ((next_item.pos[1] as f32) + 0.5 * edge_len) * bb.grid_len,
            ((next_item.pos[2] as f32) + 0.5 * edge_len) * bb.grid_len,
        ]
    };
    br.tokens.push(DecToken {
        typ: pred_typ,
        xyz_bins: pred_xyz,
        level: next_item.level,
        rem_a: br.init_rem_a,
        phy_pos: pred_phy,
        score: pred_score,
        is_root: false,
    });

    let q_empty = br.q_head >= br.q_buf.len();
    let force_stop = br.step_idx >= max_decode_steps;
    if q_empty || force_stop {
        br.finished = true;
        if stop_at_leaf_boundary {
            br.empty_coarse = true;
        }
        return Ok(());
    }
    if stop_at_leaf_boundary && br.q_buf[br.q_head].level <= 0 {
        br.paused_at_leaf = true;
        return Ok(());
    }

    // Pop next `[MASK]` token to form the upcoming 2-token step `[prev_pred, next_MASK]`
    pop_and_append_next_mask(bundle, br);
    Ok(())
}

fn pop_and_append_next_mask(bundle: &ModelBundle, br: &mut BranchState) {
    let bb = &bundle.header.backbone;
    let mask_item = br.q_buf[br.q_head];
    br.q_head += 1;
    let edge_len = (1i32 << mask_item.level.max(0)) as f32;
    let mask_phy = [
        ((mask_item.pos[0] as f32) + 0.5 * edge_len) * bb.grid_len,
        ((mask_item.pos[1] as f32) + 0.5 * edge_len) * bb.grid_len,
        ((mask_item.pos[2] as f32) + 0.5 * edge_len) * bb.grid_len,
    ];
    br.tokens.push(DecToken {
        typ: mask_item.typ.max(0) as usize,
        xyz_bins: [bb.xyz_null_id; 3],
        level: mask_item.level,
        rem_a: if mask_item.level >= 0 { br.init_rem_a } else { 0 },
        phy_pos: mask_phy,
        score: 0.0,
        is_root: false,
    });
}

fn center_preview_positions(pos: &[[f32; 3]]) -> Vec<[f32; 3]> {
    if pos.is_empty() {
        return Vec::new();
    }
    let n = pos.len() as f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut cz = 0.0f32;
    for p in pos {
        cx += p[0];
        cy += p[1];
        cz += p[2];
    }
    cx /= n;
    cy /= n;
    cz /= n;
    pos.iter().map(|p| [p[0] - cx, p[1] - cy, p[2] - cz]).collect()
}

#[allow(clippy::too_many_arguments)]
async fn run_branch_chunk<F>(
    bundle: &ModelBundle,
    mut gpu: Option<&mut WgpuDecoderBackend>,
    branches: &mut [BranchState],
    cond: &PrecomputedCond,
    uncond: Option<&PrecomputedCond>,
    pref_pos: &[[f32; 3]],
    opts: &Uni3darSampleOptions,
    allowed_mask: &[bool],
    allowed_elem_ids: &[usize],
    max_decode_steps: usize,
    stop_at_leaf_boundary: bool,
    prefill_before_resume: bool,
    max_slots: usize,
    max_kv_tokens: usize,
    emit_progress_on_zero: bool,
    on_step: &mut F,
) -> Result<(), String>
where
    F: FnMut(usize, usize, &[[f32; 3]], &[usize]),
{
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let use_cfg = uncond.is_some();
    let cfg_mult = if use_cfg { 2 } else { 1 };
    let n_br = branches.len();
    let p_len = cond.num_prefix;

    let mut cpu_kvs: Vec<(CpuKvSlot, Option<CpuKvSlot>)> = if gpu.is_none() {
        (0..n_br)
            .map(|_| {
                let kc = CpuKvSlot::new(bb.num_layers, max_kv_tokens, d);
                let ku = if use_cfg {
                    Some(CpuKvSlot::new(bb.num_layers, max_kv_tokens, d))
                } else {
                    None
                };
                (kc, ku)
            })
            .collect()
    } else {
        Vec::new()
    };

    // If resuming Stage-2 leaf decoding from paused Stage-1 branches, prefill each branch's coarse sequence in chunks of <= 16 tokens!
    if prefill_before_resume {
        for br in branches.iter_mut() {
            br.kv_len = 0;
            br.paused_at_leaf = false;
        }
        let target_lens: Vec<usize> = branches.iter().map(|b| p_len + b.tokens.len()).collect();
        let mut cursor = 0usize;
        while branches
            .iter()
            .enumerate()
            .any(|(i, b)| !b.finished && cursor < target_lens[i])
        {
            let active_ids: Vec<usize> = (0..n_br)
                .filter(|&i| !branches[i].finished && cursor < target_lens[i])
                .collect();
            let seq_q = if cursor == 0 {
                p_len + 1
            } else {
                let min_rem = active_ids.iter().map(|&i| target_lens[i] - cursor).min().unwrap_or(1);
                let cap_q = (MAX_STEP_TOKENS / (active_ids.len() * cfg_mult).max(1)).max(1);
                min_rem.min(16).min(cap_q)
            };

            if let Some(ref mut g) = gpu {
                let mut x_batch = Vec::with_capacity(active_ids.len() * cfg_mult * seq_q * d);
                let mut pos_batch = Vec::with_capacity(active_ids.len() * cfg_mult * seq_q);
                let mut meta = Vec::with_capacity(active_ids.len() * cfg_mult);

                for &bi in &active_ids {
                    let br = &branches[bi];
                    let (xc, xu_opt, p_seq) = build_token_span_embeddings(bundle, br, cond, uncond, pref_pos, cursor, seq_q)?;
                    x_batch.extend_from_slice(&xc);
                    pos_batch.extend_from_slice(&p_seq);
                    meta.push(((bi * cfg_mult) as u32, br.kv_len as u32));
                    if let Some(xu) = xu_opt {
                        x_batch.extend_from_slice(&xu);
                        pos_batch.extend_from_slice(&p_seq);
                        meta.push(((bi * cfg_mult + 1) as u32, br.kv_len as u32));
                    }
                }
                let _ = g
                    .forward_step_batch_async(
                        &x_batch,
                        &pos_batch,
                        &meta,
                        seq_q,
                        max_slots,
                        max_kv_tokens,
                        StepHeadMode::HiddenOnly,
                    )
                    .await?;
            } else {
                for &bi in &active_ids {
                    let br = &branches[bi];
                    let (xc, xu_opt, p_seq) = build_token_span_embeddings(bundle, br, cond, uncond, pref_pos, cursor, seq_q)?;
                    let _ = forward_decoder_step_cpu(bundle, &xc, &p_seq, cond, &mut cpu_kvs[bi].0)?;
                    if let (Some(xu), Some(uc), Some(ref mut ku)) = (xu_opt, uncond, &mut cpu_kvs[bi].1) {
                        let _ = forward_decoder_step_cpu(bundle, &xu, &p_seq, uc, ku)?;
                    }
                }
            }
            for &bi in &active_ids {
                branches[bi].kv_len += seq_q;
            }
            cursor += seq_q;
        }

        // Now pop the first `level == 0` `[MASK]` token on each non-finished branch to form `[last_tree_pred, first_leaf_MASK]`
        for br in branches.iter_mut() {
            if !br.finished {
                if br.q_head < br.q_buf.len() {
                    pop_and_append_next_mask(bundle, br);
                } else {
                    br.finished = true;
                }
            }
        }
    }

    // Main autoregressive step loop across active branches in this chunk
    let est_tot_atoms = opts.budget_atoms.unwrap_or(28).max(4);
    let mut last_emitted_atoms = 0usize;

    loop {
        let active_ids: Vec<usize> = (0..n_br)
            .filter(|&i| !branches[i].finished && !branches[i].paused_at_leaf)
            .collect();
        if active_ids.is_empty() {
            break;
        }

        let is_step0 = branches[active_ids[0]].step_idx == 0;
        let seq_q = if is_step0 { p_len + 1 } else { 2 };
        let head_mode = if stop_at_leaf_boundary {
            StepHeadMode::HiddenAndTree
        } else {
            StepHeadMode::HiddenOnly
        };

        if let Some(ref mut g) = gpu {
            let b_run = active_ids.len() * cfg_mult;
            let mut x_batch = Vec::with_capacity(b_run * seq_q * d);
            let mut pos_batch = Vec::with_capacity(b_run * seq_q);
            let mut meta = Vec::with_capacity(b_run);

            for &bi in &active_ids {
                let br = &branches[bi];
                let start_cur = if is_step0 {
                    0
                } else {
                    p_len + br.tokens.len() - 2
                };
                let (xc, xu_opt, p_seq) = build_token_span_embeddings(bundle, br, cond, uncond, pref_pos, start_cur, seq_q)?;
                x_batch.extend_from_slice(&xc);
                pos_batch.extend_from_slice(&p_seq);
                meta.push(((bi * cfg_mult) as u32, br.kv_len as u32));
                if let Some(xu) = xu_opt {
                    x_batch.extend_from_slice(&xu);
                    pos_batch.extend_from_slice(&p_seq);
                    meta.push(((bi * cfg_mult + 1) as u32, br.kv_len as u32));
                }
            }

            let step_out = g
                .forward_step_batch_async(&x_batch, &pos_batch, &meta, seq_q, max_slots, max_kv_tokens, head_mode)
                .await?;

            for (loc_j, &bi) in active_ids.iter().enumerate() {
                let c_row = loc_j * cfg_mult;
                let dec_y = &step_out.last_hidden[c_row * d..(c_row + 1) * d];
                let dec_u = if use_cfg {
                    Some(&step_out.last_hidden[(c_row + 1) * d..(c_row + 2) * d])
                } else {
                    None
                };
                let pre_tc = step_out
                    .tree_logits
                    .as_ref()
                    .map(|tl| &tl[c_row * 255..(c_row + 1) * 255]);
                let pre_tu = if use_cfg {
                    step_out
                        .tree_logits
                        .as_ref()
                        .map(|tl| &tl[(c_row + 1) * 255..(c_row + 2) * 255])
                } else {
                    None
                };

                branches[bi].kv_len += seq_q;
                advance_branch_from_hidden(
                    bundle,
                    &mut branches[bi],
                    dec_y,
                    dec_u,
                    pre_tc,
                    pre_tu,
                    opts,
                    allowed_mask,
                    allowed_elem_ids,
                    max_decode_steps,
                    stop_at_leaf_boundary,
                )?;
            }
        } else {
            for &bi in &active_ids {
                let start_cur = if is_step0 {
                    0
                } else {
                    p_len + branches[bi].tokens.len() - 2
                };
                let (xc, xu_opt, p_seq) =
                    build_token_span_embeddings(bundle, &branches[bi], cond, uncond, pref_pos, start_cur, seq_q)?;
                let dec_y = forward_decoder_step_cpu(bundle, &xc, &p_seq, cond, &mut cpu_kvs[bi].0)?;
                let dec_u = if let (Some(xu), Some(uc), Some(ref mut ku)) = (xu_opt, uncond, &mut cpu_kvs[bi].1) {
                    Some(forward_decoder_step_cpu(bundle, &xu, &p_seq, uc, ku)?)
                } else {
                    None
                };
                branches[bi].kv_len += seq_q;
                advance_branch_from_hidden(
                    bundle,
                    &mut branches[bi],
                    &dec_y,
                    dec_u.as_deref(),
                    None,
                    None,
                    opts,
                    allowed_mask,
                    allowed_elem_ids,
                    max_decode_steps,
                    stop_at_leaf_boundary,
                )?;
            }
        }

        // Stream live 3D atom placement frames from primary branch `0` every 2 new atoms
        if emit_progress_on_zero && !branches.is_empty() {
            let cur_a = branches[0].placed_elems.len();
            if cur_a >= last_emitted_atoms + 2 || (branches[0].finished && cur_a > last_emitted_atoms) {
                last_emitted_atoms = cur_a;
                let centered = center_preview_positions(&branches[0].placed_pos);
                on_step(
                    cur_a.min(est_tot_atoms),
                    est_tot_atoms.max(cur_a),
                    &centered,
                    &branches[0].placed_elems,
                );
            }
        }
    }

    Ok(())
}

fn build_token_span_embeddings(
    bundle: &ModelBundle,
    br: &BranchState,
    cond: &PrecomputedCond,
    uncond: Option<&PrecomputedCond>,
    pref_pos: &[[f32; 3]],
    start_global_idx: usize,
    seq_q: usize,
) -> Result<(Vec<f32>, Option<Vec<f32>>, Vec<[f32; 3]>), String> {
    let d = bundle.header.backbone.emb_dim;
    let p_len = cond.num_prefix;
    let mut xc = Vec::with_capacity(seq_q * d);
    let mut xu = uncond.map(|_| Vec::with_capacity(seq_q * d));
    let mut pos = Vec::with_capacity(seq_q);

    for q in 0..seq_q {
        let g = start_global_idx + q;
        if g < p_len {
            xc.extend_from_slice(&cond.prefix_tokens[g * d..(g + 1) * d]);
            if let (Some(ref mut xu_v), Some(uc)) = (&mut xu, uncond) {
                xu_v.extend_from_slice(&uc.prefix_tokens[g * d..(g + 1) * d]);
            }
            pos.push(pref_pos[g]);
        } else {
            let t_idx = g - p_len;
            let tok = br.tokens[t_idx];
            if tok.is_root {
                let ec = embed_octree_token(bundle, tok.typ, tok.xyz_bins, tok.level, tok.rem_a, Some(&cond.root_delta))?;
                xc.extend_from_slice(&ec);
                if let (Some(ref mut xu_v), Some(uc)) = (&mut xu, uncond) {
                    let eu =
                        embed_octree_token(bundle, tok.typ, tok.xyz_bins, tok.level, tok.rem_a, Some(&uc.root_delta))?;
                    xu_v.extend_from_slice(&eu);
                }
            } else {
                let e = embed_octree_token(bundle, tok.typ, tok.xyz_bins, tok.level, tok.rem_a, None)?;
                xc.extend_from_slice(&e);
                if let Some(ref mut xu_v) = xu {
                    xu_v.extend_from_slice(&e);
                }
            }
            pos.push(tok.phy_pos);
        }
    }
    Ok((xc, xu, pos))
}

/// Full-stack `Uni-3DAR-Mol` sampling pipeline:
/// `LoomConditioner` -> `Two-Stage Speculative Octree Funnel (m_s1 -> m_s2 -> 1)` ->
/// `PostTreeGraphRefiner` -> `Stage-2 Element Critic` -> `ValenceReconciler (RMSD_heavy == 0.0 Å)`.
pub async fn sample_molecule_async<F>(
    bundle: &ModelBundle,
    mut gpu: Option<&mut WgpuDecoderBackend>,
    opts: &Uni3darSampleOptions,
    mut on_step: F,
) -> Result<SampledUni3darMolecule, String>
where
    F: FnMut(usize, usize, &[[f32; 3]], &[usize]),
{
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;

    // 1. Precompute conditional & unconditional (for CFG) `LoomConditioner` states
    let cond = precompute_condition(bundle, opts.cond_targets.as_deref(), opts.cond_active.as_deref())?;
    let use_cfg = cond.has_any && (opts.cfg_scale != 1.0 || opts.xyz_cfg_scale != 1.0);
    let uncond = if use_cfg {
        let null_act = vec![false; bundle.header.target_cols.len()];
        Some(precompute_condition(bundle, opts.cond_targets.as_deref(), Some(&null_act))?)
    } else {
        None
    };
    let cfg_mult = if use_cfg { 2 } else { 1 };
    let pref_pos = prefix_phy_positions(bundle);

    // 2. Build allowed element mask (`allowed_elements` + `atom_null_id = 345`)
    let mut allowed_mask = vec![false; bb.num_atom_vocab];
    let mut allowed_elem_ids = Vec::new();
    let elem_syms: Vec<String> = opts
        .allowed_elements
        .clone()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            if bb.allowed_elements.is_empty() {
                ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            } else {
                bb.allowed_elements.clone()
            }
        });
    for sym in &elem_syms {
        if let Some(eid) = symbol_to_elem_id(sym.trim()) {
            if eid < bb.num_atom_vocab && !allowed_mask[eid] {
                allowed_mask[eid] = true;
                allowed_elem_ids.push(eid);
            }
        }
    }
    if allowed_elem_ids.is_empty() {
        for eid in [0usize, 5, 6, 7, 8] {
            allowed_mask[eid] = true;
            allowed_elem_ids.push(eid);
        }
    }
    allowed_mask[bb.atom_null_id] = true;

    // 3. Upload interleaved `[cond, uncond, cond, uncond, ...]` modulations ONCE to GPU!
    if let Some(ref mut g) = gpu {
        let mut mod_refs: Vec<&PrecomputedCond> = Vec::with_capacity(MAX_MOD_BRANCHES);
        if let Some(ref uc) = uncond {
            for _ in 0..(MAX_MOD_BRANCHES / 2) {
                mod_refs.push(&cond);
                mod_refs.push(uc);
            }
        } else {
            for _ in 0..MAX_MOD_BRANCHES {
                mod_refs.push(&cond);
            }
        }
        g.upload_branch_modulations(&mod_refs)?;
    }

    let has_critics = bundle.header.stage1_critic.is_some() && bundle.header.stage2_critic.is_some();
    let (m_s1, m_s2) = resolve_funnel(opts.speculative_scale, has_critics);
    let max_decode_steps = resolve_max_decode_steps(bundle, opts);
    let need_kv_full = (2 * max_decode_steps + cond.num_prefix + 16).max(208);

    // 4. Execute either single-shot (`m_s1 == 1`) or Two-Stage Speculative Funnel (`m_s1 -> m_s2 -> 1`)
    let mut final_candidates: Vec<BranchState> = if m_s1 == 1 {
        let slots = cfg_mult; // 1 or 2 slots
        let kv_cap = KV_TOTAL_SLOT_TOKENS / slots;
        let safe_max_steps = max_decode_steps.min((kv_cap.saturating_sub(cond.num_prefix + 8)) / 2);
        let mut brs = vec![BranchState::new(bundle, opts.seed, opts.budget_atoms)];
        run_branch_chunk(
            bundle,
            gpu,
            &mut brs,
            &cond,
            uncond.as_ref(),
            &pref_pos,
            opts,
            &allowed_mask,
            &allowed_elem_ids,
            safe_max_steps,
            false,
            false,
            slots,
            kv_cap,
            true,
            &mut on_step,
        )
        .await?;
        brs[0].compute_coarse_geom(bb.atom_mask_id, bb.grid_len);
        brs
    } else {
        // ---------------- Stage 1: Coarse `L7 -> L1` Tree Exploration (`m_s1` branches) ----------------
        // Always grant >= 416 KV tokens (`>= 204` coarse steps) per slot so large OE62 / GEOM-Drugs trees never truncate early!
        let s1_max_slots = 4usize;
        let s1_max_kv = KV_TOTAL_SLOT_TOKENS / s1_max_slots;
        let s1_b_chunk = (s1_max_slots / cfg_mult).max(1);
        let s1_max_steps = max_decode_steps.min((s1_max_kv.saturating_sub(cond.num_prefix + 8)) / 2);

        let mut s1_branches: Vec<BranchState> = (0..m_s1)
            .map(|i| {
                let br_seed = opts.seed.wrapping_mul(6364136223846793005).wrapping_add((i as u64 + 1) * 1442695040888963407);
                BranchState::new(bundle, br_seed, opts.budget_atoms)
            })
            .collect();

        for chunk in s1_branches.chunks_mut(s1_b_chunk) {
            run_branch_chunk(
                bundle,
                gpu.as_deref_mut(),
                chunk,
                &cond,
                uncond.as_ref(),
                &pref_pos,
                opts,
                &allowed_mask,
                &allowed_elem_ids,
                s1_max_steps,
                true,
                false,
                s1_max_slots,
                s1_max_kv,
                false,
                &mut on_step,
            )
            .await?;
        }

        // Score all `m_s1` coarse prefixes with Stage-1 `OctreeProcessCritic`
        for br in &mut s1_branches {
            br.compute_coarse_geom(bb.atom_mask_id, bb.grid_len);
            if br.empty_coarse {
                br.stage1_util = Some(-1e9);
                continue;
            }
            let h_s1 = br.build_stage1_h_concat(d);
            let mut u1 = eval_stage1_critic(
                bundle,
                &h_s1,
                &br.coarse_geom_14,
                &cond.targets_norm,
                &cond.active_mask,
                0.80,
                0.55,
            )?
            .map_or(0.0, |p| p.utility);
            if br.budget_f > 0.0 {
                u1 -= 0.35 * (br.rem_lvl_cells[0] - br.budget_f).abs();
            }
            br.stage1_util = Some(u1);
        }

        s1_branches.sort_by(|a, b| {
            b.stage1_util
                .unwrap_or(-1e9)
                .partial_cmp(&a.stage1_util.unwrap_or(-1e9))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        s1_branches.truncate(m_s2);

        // Emit coarse L1 voxel preview on winner #0 so UI immediately visualizes the chosen 3D scaffold!
        if !s1_branches.is_empty() {
            let l1_pts: Vec<[f32; 3]> = s1_branches[0]
                .tokens
                .iter()
                .filter(|t| t.typ == bb.atom_mask_id && t.level == 1)
                .map(|t| t.phy_pos)
                .collect();
            if !l1_pts.is_empty() {
                let c_pts = center_preview_positions(&l1_pts);
                let c_ids = vec![5usize; c_pts.len()]; // carbon grey scaffold preview
                let est = opts.budget_atoms.unwrap_or(c_pts.len()).max(c_pts.len());
                on_step(1, est, &c_pts, &c_ids);
            }
        }

        // ---------------- Stage 2: Fine `L0` Leaf Decoding (`m_s2` survivors) ----------------
        let s2_max_slots = if need_kv_full <= 416 && m_s2 * cfg_mult >= 4 {
            4
        } else {
            cfg_mult.max(2) // 2 slots -> 832 KV tokens per slot!
        };
        let s2_max_kv = KV_TOTAL_SLOT_TOKENS / s2_max_slots;
        let s2_b_chunk = (s2_max_slots / cfg_mult).max(1);
        let s2_max_steps = max_decode_steps.min((s2_max_kv.saturating_sub(cond.num_prefix + 8)) / 2);

        for (ci, chunk) in s1_branches.chunks_mut(s2_b_chunk).enumerate() {
            run_branch_chunk(
                bundle,
                gpu.as_deref_mut(),
                chunk,
                &cond,
                uncond.as_ref(),
                &pref_pos,
                opts,
                &allowed_mask,
                &allowed_elem_ids,
                s2_max_steps,
                false,
                true,
                s2_max_slots,
                s2_max_kv,
                ci == 0,
                &mut on_step,
            )
            .await?;
        }
        s1_branches
    };

    // 5. Run `PostTreeGraphRefiner` + `Stage-2 OctreeElementProcessCritic` across all leaf candidates
    for br in &mut final_candidates {
        if opts.use_refiner && !br.placed_elems.is_empty() {
            if let Some(ref_res) = run_post_tree_refiner(bundle, &br.placed_elems, &br.placed_pos)? {
                br.refined_pos = Some(ref_res.refined_pos);
                br.bond_order_probs = Some(ref_res.bond_order_probs);
            }
        }
        let eval_pos = br.refined_pos.as_deref().unwrap_or(&br.placed_pos);
        if bundle.header.stage2_critic.is_some() {
            let cloud_42 = compute_element_cloud_features(&br.placed_elems, eval_pos, &br.coarse_geom_14);
            let h_s2 = br.build_stage2_h_concat(d);
            if let Some(mut p2) = eval_stage2_critic(
                bundle,
                &h_s2,
                &br.placed_elems,
                eval_pos,
                &cloud_42,
                &cond.targets_norm,
                &cond.active_mask,
                1.05,
                0.70,
            )? {
                if br.budget_f > 0.0 {
                    p2.utility -= 0.35 * ((br.placed_elems.len() as f32) - br.budget_f).abs();
                }
                br.stage2_pred = Some(p2);
            }
        }
    }

    if final_candidates.len() > 1 {
        final_candidates.sort_by(|a, b| {
            let ua = a.stage2_pred.as_ref().map_or(-1e9, |p| p.utility);
            let ub = b.stage2_pred.as_ref().map_or(-1e9, |p| p.utility);
            ub.partial_cmp(&ua).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // 6. Reconcile valence (`RMSD_heavy == 0.0 Å`) on top-ranked candidates:
    //    Pass 1: strict 1-fragment connected heavy skeleton across all candidates.
    //    Pass 2: if `single_molecule_mode != "off"`, allow largest heavy component (`>= 70%` heavy atoms).
    //    Pass 3: raw covalent point-cloud graph fallback on candidate #0.
    let mut chosen_idx = 0usize;
    let mut chosen_rg = None;

    if opts.reconcile_valence {
        for (idx, br) in final_candidates.iter().enumerate() {
            let pos_ref = br.refined_pos.as_deref().unwrap_or(&br.placed_pos);
            if let Some(rg) = reconcile_point_cloud(
                &br.placed_elems,
                pos_ref,
                br.bond_order_probs.as_deref(),
                4500,
                false,
            ) {
                chosen_idx = idx;
                chosen_rg = Some(rg);
                break;
            }
        }
        if chosen_rg.is_none() && opts.single_molecule_mode != "off" {
            for (idx, br) in final_candidates.iter().enumerate() {
                let pos_ref = br.refined_pos.as_deref().unwrap_or(&br.placed_pos);
                if let Some(rg) = reconcile_point_cloud(
                    &br.placed_elems,
                    pos_ref,
                    br.bond_order_probs.as_deref(),
                    4500,
                    true,
                ) {
                    chosen_idx = idx;
                    chosen_rg = Some(rg);
                    break;
                }
            }
        }
    }

    let rg = chosen_rg.unwrap_or_else(|| {
        let br0 = &final_candidates[0];
        let pos0 = br0.refined_pos.as_deref().unwrap_or(&br0.placed_pos);
        build_raw_point_cloud_graph(&br0.placed_elems, pos0)
    });

    let win_br = &final_candidates[chosen_idx];
    let mut predicted_props = BTreeMap::new();
    if let (Some(ref p2), Some(ref c2_cfg)) = (&win_br.stage2_pred, &bundle.header.stage2_critic) {
        for (k, col) in bundle.header.target_cols.iter().enumerate() {
            if let Some(&mn) = p2.mu_norm.get(k) {
                let mean = c2_cfg
                    .y_mean
                    .get(k)
                    .or_else(|| bundle.header.target_mean.get(k))
                    .copied()
                    .unwrap_or(0.0);
                let scale = c2_cfg
                    .y_scale
                    .get(k)
                    .or_else(|| bundle.header.target_scale.get(k))
                    .copied()
                    .unwrap_or(1.0);
                predicted_props.insert(col.clone(), ((mn * scale + mean) * 10000.0).round() / 10000.0);
            }
        }
    }

    Ok(finalize_sampled_molecule(
        rg,
        win_br.stage1_util,
        win_br.stage2_pred.as_ref().map(|p| p.utility),
        predicted_props,
        [m_s1, m_s2, 1],
        win_br.step_idx,
    ))
}
