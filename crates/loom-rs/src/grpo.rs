//! Stage 2 / 2.5A Disentangled Multi-Channel Online `TopoGRPO` (`grpo.rs`) &
//! Grounded `Inner-RSI` Self-Improvement Runner!
//!
//! Combines:
//! 1. KV-cached conditional octree rollout (`prompts_per_step * group_size`) with 3D steric/valence guard.
//! 2. `CompositeCriticBus` (`BuiltInRuleMetrics` + Zero-Step MLIP Strain + Shared-Backbone 3D Oracle).
//! 3. Scale-decoupled `topology_only_policy` (`Tree` + `Element` tokens ONLY; sub-voxel `XYZ` excluded).
//! 4. Exact PPO clipped surrogate (`eps = 0.20`) + NaN-free dual-finite categorical `KL` + Replay SFT anchor.

use crate::chem_rules::MolRecord;
use crate::critic_bus::{
    compute_disentangled_group_advantages, evaluate_molecules_on_bus, MlipSlot, MountedOracle,
};
use crate::lora_model::{
    apply_element_geometric_guard, backward_sequence_batch, build_allowed_atom_vocab_mask,
    eval_task_head_with_lora_bwd, eval_task_head_with_lora_fwd, forward_sequence_batch,
    sample_rollout_trajectory_kv, LoomAdamW, LoomAdapterGrads, LoomGpuTrainContext,
    LoomLoraAdapter, MergedRolloutWeights, RolloutOptions, SeqForwardTape,
};
use crate::octree_encode::{encode_molecule_to_octree, EncodedOctreeSeq};
use crate::sft::{compute_sft_batch_loss_and_grads, SftOptions};
use crate::spec::TaskWeavingSpec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uni3dar_rs::format::ModelBundle;
use uni3dar_rs::sampler::DetRng;
use uni3dar_rs::valence::{elem_id_to_symbol, rcov_angstrom, SampledUni3darMolecule};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopoGrpoOptions {
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default = "default_prompts")]
    pub prompts_per_step: usize,
    #[serde(default = "default_group_size")]
    pub group_size: usize,
    #[serde(default = "default_ppo_epochs")]
    pub ppo_epochs: usize,
    #[serde(default = "default_grpo_lr")]
    pub lr: f32,
    #[serde(default = "default_wd")]
    pub weight_decay: f32,
    #[serde(default = "default_clip_eps")]
    pub clip_eps: f32,
    #[serde(default = "default_kl_beta")]
    pub kl_beta: f32,
    #[serde(default = "default_replay_w")]
    pub replay_weight: f32,
    #[serde(default = "default_replay_bs")]
    pub replay_batch_size: usize,
    #[serde(default = "default_cfg_scale")]
    pub cfg_scale: f32,
    #[serde(default = "default_cfg_decay")]
    pub cfg_tree_decay: f32,
    #[serde(default = "default_true")]
    pub topology_only_policy: bool,
    #[serde(default = "default_true")]
    pub geometric_guard: bool,
    #[serde(default = "default_true")]
    pub reconcile_valence: bool,
    #[serde(default)]
    pub sharp_concentration: bool,
    /// Target grid (or `[min, max]` endpoints) for sampling prompts during online RL.
    pub target_grid: Vec<Vec<f64>>,
    /// Optional curriculum sampling probabilities over `target_grid` (from `RSIDiagnoser`!).
    #[serde(default)]
    pub curriculum_probs: Option<Vec<f64>>,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_steps() -> usize {
    8
}
fn default_prompts() -> usize {
    2
}
fn default_group_size() -> usize {
    4
}
fn default_ppo_epochs() -> usize {
    2
}
fn default_grpo_lr() -> f32 {
    3.0e-5
}
fn default_wd() -> f32 {
    1.0e-4
}
fn default_clip_eps() -> f32 {
    0.20
}
fn default_kl_beta() -> f32 {
    0.025
}
fn default_replay_w() -> f32 {
    0.15
}
fn default_replay_bs() -> usize {
    4
}
fn default_cfg_scale() -> f32 {
    1.75
}
fn default_cfg_decay() -> f32 {
    0.25
}
fn default_true() -> bool {
    true
}
fn default_seed() -> u64 {
    42
}

impl Default for TopoGrpoOptions {
    fn default() -> Self {
        Self {
            steps: 8,
            prompts_per_step: 2,
            group_size: 4,
            ppo_epochs: 2,
            lr: 3.0e-5,
            weight_decay: 1.0e-4,
            clip_eps: 0.20,
            kl_beta: 0.025,
            replay_weight: 0.15,
            replay_batch_size: 4,
            cfg_scale: 1.75,
            cfg_tree_decay: 0.25,
            topology_only_policy: true,
            geometric_guard: true,
            reconcile_valence: true,
            sharp_concentration: false,
            target_grid: vec![vec![-6.50], vec![-6.00], vec![-5.50]],
            curriculum_probs: None,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopoGrpoStepMetrics {
    pub step: usize,
    pub total_steps: usize,
    pub proposals: usize,
    pub valid_rate: f64,
    pub clean_charge_rate: f64,
    pub constraints_met_rate: f64,
    pub hit_rate: f64,
    pub clean_hit_rate: f64,
    pub median_fmax_ev_a: f64,
    pub mean_prop_reward: f64,
    pub loss_policy: f32,
    pub kl_div: f32,
    pub loss_replay: f32,
    pub grad_norm: f32,
    pub best_molecule: Option<SampledUni3darMolecule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopoGrpoReport {
    pub steps_completed: usize,
    pub total_proposals: usize,
    pub history: Vec<TopoGrpoStepMetrics>,
    pub elapsed_ms: f64,
}

// ============================================================================
// Teacher-Forced Policy Distribution Packet Evaluation (`trajectory_distributions`)
// ============================================================================

struct PolicyEvalCache {
    tape_c: SeqForwardTape,
    tape_u: Option<SeqForwardTape>,
    do_cfg: bool,

    // Packet 0: Tree splits (`[m_tree, 255]`)
    tree_owners: Vec<usize>,
    tree_labels: Vec<usize>,
    tree_s_eff: Vec<f32>,
    tree_tok_pos: Vec<(usize, usize)>,
    h_tree_c: Vec<f32>,
    tree_mp_c: Vec<f32>,
    tree_ma_c: Vec<f32>,
    h_tree_u: Vec<f32>,
    tree_mp_u: Vec<f32>,
    tree_ma_u: Vec<f32>,
    tree_log_probs: Vec<f32>, // [m_tree * 255]

    // Packet 1: Leaf elements (`[m_atom, V_atom]`)
    atom_owners: Vec<usize>,
    atom_labels: Vec<usize>,
    atom_tok_pos: Vec<(usize, usize)>,
    h_atom_c: Vec<f32>,
    atom_mp_c: Vec<f32>,
    atom_ma_c: Vec<f32>,
    h_atom_u: Vec<f32>,
    atom_mp_u: Vec<f32>,
    atom_ma_u: Vec<f32>,
    atom_log_probs: Vec<f32>, // [m_atom * V_atom]
}

fn log_softmax_in_place(row: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &x in row.iter() {
        if x.is_finite() && x > max_v {
            max_v = x;
        }
    }
    if !max_v.is_finite() {
        return;
    }
    let mut sum_e = 0.0f32;
    for &x in row.iter() {
        if x.is_finite() {
            sum_e += (x - max_v).exp();
        }
    }
    let lse = max_v + sum_e.max(1e-20).ln();
    for x in row.iter_mut() {
        if x.is_finite() {
            *x -= lse;
        }
    }
}

async fn evaluate_topology_policy_packets(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    seqs: &[EncodedOctreeSeq],
    raw_targets: &[Vec<f32>],
    active_masks: &[Vec<bool>],
    cfg_scale: f32,
    cfg_tree_decay: f32,
    temperatures: [f32; 3],
    geometric_guard: bool,
    even_electron_parity: bool,
    save_checkpoints: bool,
) -> Result<PolicyEvalCache, String> {
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let v_atom = bb.num_atom_vocab;
    let b_sz = seqs.len();

    let any_active = active_masks.iter().any(|m| m.iter().any(|&x| x));
    let do_cfg = any_active && (cfg_scale - 1.0).abs() > 1e-4;

    let tape_c = forward_sequence_batch(bundle, gpu, adapter, seqs, raw_targets, active_masks, 1.0, save_checkpoints).await?;
    let tape_u = if do_cfg {
        let null_m = vec![vec![false; active_masks[0].len()]; b_sz];
        Some(forward_sequence_batch(bundle, gpu, adapter, seqs, raw_targets, &null_m, 1.0, save_checkpoints).await?)
    } else {
        None
    };

    let max_seq = tape_c.max_seq;
    let mut tree_owners = Vec::new();
    let mut tree_labels = Vec::new();
    let mut tree_s_eff = Vec::new();
    let mut tree_tok_pos = Vec::new();
    let mut h_tree_c = Vec::new();
    let mut h_tree_u = Vec::new();

    let mut atom_owners = Vec::new();
    let mut atom_labels = Vec::new();
    let mut atom_tok_pos = Vec::new();
    let mut h_atom_c = Vec::new();
    let mut h_atom_u = Vec::new();
    let mut atom_meta = Vec::new();

    // Also reconstruct trajectory-local prior placed atoms for exact `geometric_guard` replay!
    let mut prior_traces: Vec<Vec<(usize, [f32; 3], i32)>> = Vec::new();

    for (b, seq) in seqs.iter().enumerate() {
        for tt in &seq.tree_targets {
            let off = (b * max_seq + tt.token_pos) * d;
            h_tree_c.extend_from_slice(&tape_c.hidden_out[off..off + d]);
            if let Some(ref tu) = tape_u {
                h_tree_u.extend_from_slice(&tu.hidden_out[off..off + d]);
            }
            let decay_f = ((3 - tt.level).clamp(0, 2) as f32) * 0.5;
            let s_eff = if do_cfg {
                1.0 + (cfg_scale - 1.0) * (1.0 - cfg_tree_decay * decay_f)
            } else {
                1.0
            };
            tree_owners.push(b);
            tree_labels.push(tt.label_255);
            tree_s_eff.push(s_eff);
            tree_tok_pos.push((b, tt.token_pos));
        }

        let mut cur_placed: Vec<(usize, [f32; 3], i32)> = Vec::new();
        for at in &seq.atom_targets {
            let off = (b * max_seq + at.token_pos) * d;
            h_atom_c.extend_from_slice(&tape_c.hidden_out[off..off + d]);
            if let Some(ref tu) = tape_u {
                h_atom_u.extend_from_slice(&tu.hidden_out[off..off + d]);
            }
            atom_owners.push(b);
            atom_labels.push(at.elem_id);
            atom_tok_pos.push((b, at.token_pos));
            atom_meta.push(*at);
            prior_traces.push(cur_placed.clone());

            if at.elem_id < 342 {
                // Odd token `at.token_pos + 1` holds the fine `phy_pos`!
                let fine_p = seq.tokens[at.token_pos + 1].phy_pos;
                let rc_me = rcov_angstrom(elem_id_to_symbol(at.elem_id));
                let mut my_deg = 0i32;
                for item in &mut cur_placed {
                    let dx = fine_p[0] - item.1[0];
                    let dy = fine_p[1] - item.1[1];
                    let dz = fine_p[2] - item.1[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    let rc_sum = rc_me + rcov_angstrom(elem_id_to_symbol(item.0));
                    if dist <= 1.24 * rc_sum {
                        my_deg += 1;
                        item.2 += 1;
                    }
                }
                cur_placed.push((at.elem_id, fine_p, my_deg));
            }
        }
    }

    // Packet 0: Tree log_probs
    let m_tree = tree_owners.len();
    let (lt_c, tree_mp_c, tree_ma_c) = eval_task_head_with_lora_fwd(bundle, gpu, adapter, "bb.tree_head", &h_tree_c, m_tree, 255, 1.0).await?;
    let (lt_u, tree_mp_u, tree_ma_u) = if do_cfg {
        eval_task_head_with_lora_fwd(bundle, gpu, adapter, "bb.tree_head", &h_tree_u, m_tree, 255, 1.0).await?
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };

    let inv_t_tree = 1.0 / temperatures[0].max(1e-4);
    let mut tree_log_probs = vec![0.0f32; m_tree * 255];
    for r in 0..m_tree {
        let s = tree_s_eff[r];
        let row = &mut tree_log_probs[r * 255..(r + 1) * 255];
        for v in 0..255 {
            let z = if do_cfg {
                lt_u[r * 255 + v] + s * (lt_c[r * 255 + v] - lt_u[r * 255 + v])
            } else {
                lt_c[r * 255 + v]
            };
            row[v] = z * inv_t_tree;
        }
        log_softmax_in_place(row);
    }

    // Packet 1: Leaf element log_probs
    let m_atom = atom_owners.len();
    let allowed_mask = build_allowed_atom_vocab_mask(bundle);
    let (la_c, atom_mp_c, atom_ma_c) = eval_task_head_with_lora_fwd(bundle, gpu, adapter, "bb.atom_head.pred.0", &h_atom_c, m_atom, v_atom, 1.0).await?;
    let (la_u, atom_mp_u, atom_ma_u) = if do_cfg {
        eval_task_head_with_lora_fwd(bundle, gpu, adapter, "bb.atom_head.pred.0", &h_atom_u, m_atom, v_atom, 1.0).await?
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };

    let inv_t_atom = 1.0 / temperatures[1].max(1e-4);
    let mut atom_log_probs = vec![0.0f32; m_atom * v_atom];
    for r in 0..m_atom {
        let row = &mut atom_log_probs[r * v_atom..(r + 1) * v_atom];
        for v in 0..v_atom {
            if !allowed_mask[v] {
                row[v] = f32::NEG_INFINITY;
            } else {
                let z = if do_cfg {
                    la_u[r * v_atom + v] + cfg_scale * (la_c[r * v_atom + v] - la_u[r * v_atom + v])
                } else {
                    la_c[r * v_atom + v]
                };
                row[v] = z;
            }
        }
        let meta = &atom_meta[r];
        if geometric_guard && !prior_traces[r].is_empty() {
            let p_elems: Vec<usize> = prior_traces[r].iter().map(|t| t.0).collect();
            let p_pos: Vec<[f32; 3]> = prior_traces[r].iter().map(|t| t.1).collect();
            let p_deg: Vec<i32> = prior_traces[r].iter().map(|t| t.2).collect();
            apply_element_geometric_guard(row, meta.voxel_center, &p_elems, &p_pos, &p_deg, bb.atom_null_id);
        }
        if even_electron_parity && meta.is_terminal_leaf {
            for eid in 0..118.min(v_atom) {
                if ((meta.prior_z_parity + eid + 1) & 1) != 0 {
                    row[eid] = f32::NEG_INFINITY;
                }
            }
            if (meta.prior_z_parity & 1) != 0 && bb.atom_null_id < v_atom {
                row[bb.atom_null_id] = f32::NEG_INFINITY;
            }
        }
        for x in row.iter_mut() {
            if x.is_finite() {
                *x *= inv_t_atom;
            }
        }
        log_softmax_in_place(row);
    }

    Ok(PolicyEvalCache {
        tape_c,
        tape_u,
        do_cfg,
        tree_owners,
        tree_labels,
        tree_s_eff,
        tree_tok_pos,
        h_tree_c,
        tree_mp_c,
        tree_ma_c,
        h_tree_u,
        tree_mp_u,
        tree_ma_u,
        tree_log_probs,
        atom_owners,
        atom_labels,
        atom_tok_pos,
        h_atom_c,
        atom_mp_c,
        atom_ma_c,
        h_atom_u,
        atom_mp_u,
        atom_ma_u,
        atom_log_probs,
    })
}

/// Compute exact PPO Clipped Surrogate + NaN-free Dual-Finite Categorical KL & pre-`log_softmax`
/// logit gradients for a single decision packet (`Tree` or `Element`)!
fn compute_packet_ppo_kl_loss_and_dlogits(
    cur_lp: &[f32],
    old_lp: &[f32],
    ref_lp: &[f32],
    labels: &[usize],
    owners: &[usize],
    mol_decision_counts: &[usize],
    advantages: &[f32],
    b_sz: usize,
    vocab: usize,
    inv_temp: f32,
    clip_eps: f32,
    kl_beta: f32,
) -> (f32, f32, Vec<f32>) {
    let m = owners.len();
    let mut d_logits_guided = vec![0.0f32; m * vocab];
    let mut sum_policy_loss = 0.0f32;
    let mut sum_kl = 0.0f32;

    for r in 0..m {
        let b = owners[r];
        let g_weight = 1.0 / ((b_sz as f32) * (mol_decision_counts[b].max(1) as f32));
        let y = labels[r].min(vocab - 1);
        let a_m = advantages[b];

        let cur_row = &cur_lp[r * vocab..(r + 1) * vocab];
        let old_row = &old_lp[r * vocab..(r + 1) * vocab];
        let ref_row = &ref_lp[r * vocab..(r + 1) * vocab];
        let dl_row = &mut d_logits_guided[r * vocab..(r + 1) * vocab];

        // 1. PPO Clipped Surrogate
        let lc = cur_row[y];
        let lo = old_row[y];
        let mut d_lc = 0.0f32;
        if lc.is_finite() && lo.is_finite() {
            let ratio = (lc - lo).clamp(-20.0, 20.0).exp();
            let surr1 = ratio * a_m;
            let ratio_clip = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
            let surr2 = ratio_clip * a_m;
            let obj_ppo = -surr1.min(surr2);
            sum_policy_loss += g_weight * obj_ppo;

            // Subgradient active iff `surr1 <= surr2` (or unclipped region)
            let unclipped_active = if a_m >= 0.0 {
                ratio <= 1.0 + clip_eps
            } else {
                ratio >= 1.0 - clip_eps
            };
            if unclipped_active {
                d_lc = -g_weight * ratio * a_m;
            }
        }

        // 2. Exact Dual-Finite Categorical KL (`F_{r,v} = isfinite(cur) && isfinite(ref)`)
        let mut kl_m = 0.0f32;
        let mut u_bar = 0.0f32;
        let mut u_vec = vec![0.0f32; vocab];
        let mut p_vec = vec![0.0f32; vocab];

        for v in 0..vocab {
            let l_theta = cur_row[v];
            let l_ref = ref_row[v];
            if l_theta.is_finite() {
                p_vec[v] = l_theta.exp();
            }
            if l_theta.is_finite() && l_ref.is_finite() {
                let p = p_vec[v];
                let raw_diff = l_theta - l_ref;
                let delta = raw_diff.clamp(-30.0, 30.0);
                kl_m += p * delta;
                if kl_beta > 0.0 {
                    let in_clamp = if raw_diff.abs() < 30.0 { 1.0 } else { 0.0 };
                    let u_v = kl_beta * (delta + in_clamp);
                    u_vec[v] = u_v;
                    u_bar += p * u_v;
                }
            }
        }
        sum_policy_loss += g_weight * kl_beta * kl_m;
        sum_kl += g_weight * kl_m;

        // Combine `d_lc * (I[v==y] - P_theta[v])` and `g_weight * P_theta[v] * (u_v - u_bar)` and chain `inv_temp`!
        for v in 0..vocab {
            if cur_row[v].is_finite() {
                let p = p_vec[v];
                let ind = if v == y { 1.0 } else { 0.0 };
                let grad_ppo = d_lc * (ind - p);
                let grad_kl = if ref_row[v].is_finite() && kl_beta > 0.0 {
                    g_weight * p * (u_vec[v] - u_bar)
                } else {
                    0.0
                };
                dl_row[v] = (grad_ppo + grad_kl) * inv_temp;
            }
        }
    }

    (sum_policy_loss, sum_kl, d_logits_guided)
}

/// Sample a prompt index from either `curriculum_probs` or uniform over `target_grid`.
fn sample_prompt_index(n_grid: usize, probs: Option<&[f64]>, rng: &mut DetRng) -> usize {
    if n_grid <= 1 {
        return 0;
    }
    if let Some(pr) = probs {
        if pr.len() == n_grid {
            let sum_p: f64 = pr.iter().copied().filter(|x| *x > 0.0 && x.is_finite()).sum();
            if sum_p > 1e-9 {
                let u = (rng.next_f32() as f64) * sum_p;
                let mut acc = 0.0f64;
                for (i, &w) in pr.iter().enumerate() {
                    if w > 0.0 && w.is_finite() {
                        acc += w;
                        if u <= acc {
                            return i;
                        }
                    }
                }
            }
        }
    }
    (rng.next_u64() as usize) % n_grid
}

/// Run Online `TopoGRPO` (`Stage 2 / 2.5A`) closed-loop training!
pub async fn run_online_topo_grpo<F>(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &mut LoomLoraAdapter,
    spec: &TaskWeavingSpec,
    mlip: Option<&mut MlipSlot>,
    oracles: &HashMap<String, MountedOracle>,
    replay_records: &[MolRecord],
    opts: &TopoGrpoOptions,
    mut on_step: F,
) -> Result<TopoGrpoReport, String>
where
    F: FnMut(&TopoGrpoStepMetrics),
{
    if opts.target_grid.is_empty() {
        return Err("TopoGrpoOptions.target_grid must be non-empty".into());
    }
    adapter.set_topology_policy_mode(true);

    // Frozen reference policy `pi_ref` snapshot at step 0 (`deepcopy` equivalent, ~18 MB)
    let ref_adapter = adapter.clone();

    // Collect seen formal-charge signatures from replay set (if provided)
    let seen_charges: Option<HashSet<String>> = if replay_records.is_empty() {
        None
    } else {
        let mut set = HashSet::new();
        set.insert("neutral".to_string());
        for r in replay_records {
            set.insert(crate::chem_rules::formal_charge_signature(r));
        }
        Some(set)
    };

    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let v_atom = bb.num_atom_vocab;
    let temps = [0.85f32, 0.30, 0.25];
    let p_cnt = adapter.conditioner.cfg.num_props;

    let mut rng = DetRng::new(opts.seed ^ 0x6AA0_2026);
    let mut opt = LoomAdamW::new();
    let mut history = Vec::with_capacity(opts.steps);
    let mut mlip_opt = mlip;
    let mut total_proposals = 0usize;

    let sft_replay_opts = SftOptions {
        cfg_dropout_prob: 0.05,
        contrast_weight: 0.0,
        ..SftOptions::default()
    };

    for step_idx in 1..=opts.steps.max(1) {
        let p_per_step = opts.prompts_per_step.max(1);
        let g_sz = opts.group_size.max(2);
        let b_total = p_per_step * g_sz;

        let mut step_seqs = Vec::with_capacity(b_total);
        let mut step_recs = Vec::with_capacity(b_total);
        let mut step_mols = Vec::with_capacity(b_total);
        let mut req_goals_f64 = Vec::with_capacity(b_total);
        let mut req_goals_f32 = Vec::with_capacity(b_total);
        let mut act_masks = Vec::with_capacity(b_total);

        // 1. Batched Conditional Octree Rollouts (`P * G`) using pre-merged SIMD `MergedRolloutWeights`!
        let merged_rollout = MergedRolloutWeights::build(bundle, adapter)?;
        for _p in 0..p_per_step {
            let g_idx = sample_prompt_index(
                opts.target_grid.len(),
                opts.curriculum_probs.as_deref(),
                &mut rng,
            );
            let base_goal = &opts.target_grid[g_idx];
            let mut goal_f64 = vec![0.0f64; p_cnt];
            let mut goal_f32 = vec![0.0f32; p_cnt];
            let mut act_m = vec![false; p_cnt];
            for k in 0..p_cnt.min(base_goal.len()) {
                goal_f64[k] = base_goal[k];
                goal_f32[k] = base_goal[k] as f32;
                act_m[k] = true;
            }

            for _g in 0..g_sz {
                let r_opts = RolloutOptions {
                    cfg_scale: opts.cfg_scale,
                    xyz_cfg_scale: 1.0,
                    cfg_tree_decay: opts.cfg_tree_decay,
                    temperatures: temps,
                    geometric_guard: opts.geometric_guard,
                    even_electron_parity: true,
                    reconcile_valence: opts.reconcile_valence,
                    max_decode_steps: 220,
                    seed: rng.next_u64(),
                };
                let (enc, rec, mol) = sample_rollout_trajectory_kv(
                    bundle,
                    adapter,
                    &merged_rollout,
                    &goal_f32,
                    &act_m,
                    &r_opts,
                )?;
                step_seqs.push(enc);
                step_recs.push(rec);
                step_mols.push(mol);
                req_goals_f64.push(goal_f64.clone());
                req_goals_f32.push(goal_f32.clone());
                act_masks.push(act_m.clone());
            }
        }
        total_proposals += b_total;

        // 2. Multi-Channel Critic Evaluation (`CompositeCriticBus`)
        let evals = evaluate_molecules_on_bus(
            spec,
            &step_recs,
            &req_goals_f64,
            &act_masks,
            mlip_opt.as_deref_mut(),
            oracles,
            seen_charges.as_ref(),
            true,
            opts.sharp_concentration,
        )
        .await?;

        // 3. Per-Group Disentangled Monotonic Advantages (`grpo.py:31-92`)
        let mut all_adv = Vec::with_capacity(b_total);
        for p in 0..p_per_step {
            let grp_evals = &evals[p * g_sz..(p + 1) * g_sz];
            let (grp_adv, _) = compute_disentangled_group_advantages(grp_evals, 0.75, 0.30, 1.00);
            all_adv.extend(grp_adv);
        }

        // 4. Evaluate `old_packets` (`pi_old`) and `ref_packets` (`pi_ref`) once under `no_grad` (`save_checkpoints = false`)
        let old_cache = evaluate_topology_policy_packets(
            bundle,
            gpu,
            adapter,
            &step_seqs,
            &req_goals_f32,
            &act_masks,
            opts.cfg_scale,
            opts.cfg_tree_decay,
            temps,
            opts.geometric_guard,
            true,
            false,
        )
        .await?;
        let ref_cache = evaluate_topology_policy_packets(
            bundle,
            gpu,
            &ref_adapter,
            &step_seqs,
            &req_goals_f32,
            &act_masks,
            opts.cfg_scale,
            opts.cfg_tree_decay,
            temps,
            opts.geometric_guard,
            true,
            false,
        )
        .await?;

        // Total supervised topology decisions per molecule (`Tree + Element` when `topology_only_policy == true`)
        let mut mol_dec_counts = vec![0usize; b_total];
        for &b in &old_cache.tree_owners {
            mol_dec_counts[b] += 1;
        }
        for &b in &old_cache.atom_owners {
            mol_dec_counts[b] += 1;
        }

        // 5. PPO Epochs (`ppo_epochs = 2`) + Supervised Replay Anchor
        let mut last_loss_pol = 0.0f32;
        let mut last_kl = 0.0f32;
        let mut last_replay = 0.0f32;
        let mut last_gnorm = 0.0f32;

        for _ppo_ep in 0..opts.ppo_epochs.max(1) {
            let cur_cache = evaluate_topology_policy_packets(
                bundle,
                gpu,
                adapter,
                &step_seqs,
                &req_goals_f32,
                &act_masks,
                opts.cfg_scale,
                opts.cfg_tree_decay,
                temps,
                opts.geometric_guard,
                true,
                true,
            )
            .await?;

            let (loss_t, kl_t, dl_tree_guided) = compute_packet_ppo_kl_loss_and_dlogits(
                &cur_cache.tree_log_probs,
                &old_cache.tree_log_probs,
                &ref_cache.tree_log_probs,
                &cur_cache.tree_labels,
                &cur_cache.tree_owners,
                &mol_dec_counts,
                &all_adv,
                b_total,
                255,
                1.0 / temps[0],
                opts.clip_eps,
                opts.kl_beta,
            );
            let (loss_a, kl_a, dl_atom_guided) = compute_packet_ppo_kl_loss_and_dlogits(
                &cur_cache.atom_log_probs,
                &old_cache.atom_log_probs,
                &ref_cache.atom_log_probs,
                &cur_cache.atom_labels,
                &cur_cache.atom_owners,
                &mol_dec_counts,
                &all_adv,
                b_total,
                v_atom,
                1.0 / temps[1],
                opts.clip_eps,
                opts.kl_beta,
            );
            last_loss_pol = loss_t + loss_a;
            last_kl = kl_t + kl_a;

            let mut grads: LoomAdapterGrads = adapter.zero_grads();
            let max_seq = cur_cache.tape_c.max_seq;
            let mut dh_c = vec![0.0f32; b_total * max_seq * d];
            let mut dh_u = if cur_cache.do_cfg {
                vec![0.0f32; b_total * max_seq * d]
            } else {
                Vec::new()
            };

            // Backprop Packet 0 (`Tree`) into conditional & optional unconditional branches
            let m_tree = cur_cache.tree_owners.len();
            if m_tree > 0 {
                let mut dl_tc = vec![0.0f32; m_tree * 255];
                let mut dl_tu = if cur_cache.do_cfg { vec![0.0f32; m_tree * 255] } else { Vec::new() };
                for r in 0..m_tree {
                    let s = cur_cache.tree_s_eff[r];
                    for v in 0..255 {
                        let g = dl_tree_guided[r * 255 + v];
                        if cur_cache.do_cfg {
                            dl_tc[r * 255 + v] = s * g;
                            dl_tu[r * 255 + v] = (1.0 - s) * g;
                        } else {
                            dl_tc[r * 255 + v] = g;
                        }
                    }
                }
                let dht_c = eval_task_head_with_lora_bwd(
                    bundle,
                    gpu,
                    adapter,
                    "bb.tree_head",
                    &cur_cache.h_tree_c,
                    &cur_cache.tree_mp_c,
                    &cur_cache.tree_ma_c,
                    &dl_tc,
                    m_tree,
                    1.0,
                    Some(&mut grads),
                )
                .await?;
                for (r, &(b, tp)) in cur_cache.tree_tok_pos.iter().enumerate() {
                    let dst = &mut dh_c[(b * max_seq + tp) * d..(b * max_seq + tp + 1) * d];
                    for c in 0..d {
                        dst[c] += dht_c[r * d + c];
                    }
                }
                if cur_cache.do_cfg {
                    let dht_u = eval_task_head_with_lora_bwd(
                        bundle,
                        gpu,
                        adapter,
                        "bb.tree_head",
                        &cur_cache.h_tree_u,
                        &cur_cache.tree_mp_u,
                        &cur_cache.tree_ma_u,
                        &dl_tu,
                        m_tree,
                        1.0,
                        Some(&mut grads),
                    )
                    .await?;
                    for (r, &(b, tp)) in cur_cache.tree_tok_pos.iter().enumerate() {
                        let dst = &mut dh_u[(b * max_seq + tp) * d..(b * max_seq + tp + 1) * d];
                        for c in 0..d {
                            dst[c] += dht_u[r * d + c];
                        }
                    }
                }
            }

            // Backprop Packet 1 (`Element`) into conditional & optional unconditional branches
            let m_atom = cur_cache.atom_owners.len();
            if m_atom > 0 {
                let mut dl_ac = vec![0.0f32; m_atom * v_atom];
                let mut dl_au = if cur_cache.do_cfg { vec![0.0f32; m_atom * v_atom] } else { Vec::new() };
                for i in 0..(m_atom * v_atom) {
                    let g = dl_atom_guided[i];
                    if cur_cache.do_cfg {
                        dl_ac[i] = opts.cfg_scale * g;
                        dl_au[i] = (1.0 - opts.cfg_scale) * g;
                    } else {
                        dl_ac[i] = g;
                    }
                }
                let dha_c = eval_task_head_with_lora_bwd(
                    bundle,
                    gpu,
                    adapter,
                    "bb.atom_head.pred.0",
                    &cur_cache.h_atom_c,
                    &cur_cache.atom_mp_c,
                    &cur_cache.atom_ma_c,
                    &dl_ac,
                    m_atom,
                    1.0,
                    Some(&mut grads),
                )
                .await?;
                for (r, &(b, tp)) in cur_cache.atom_tok_pos.iter().enumerate() {
                    let dst = &mut dh_c[(b * max_seq + tp) * d..(b * max_seq + tp + 1) * d];
                    for c in 0..d {
                        dst[c] += dha_c[r * d + c];
                    }
                }
                if cur_cache.do_cfg {
                    let dha_u = eval_task_head_with_lora_bwd(
                        bundle,
                        gpu,
                        adapter,
                        "bb.atom_head.pred.0",
                        &cur_cache.h_atom_u,
                        &cur_cache.atom_mp_u,
                        &cur_cache.atom_ma_u,
                        &dl_au,
                        m_atom,
                        1.0,
                        Some(&mut grads),
                    )
                    .await?;
                    for (r, &(b, tp)) in cur_cache.atom_tok_pos.iter().enumerate() {
                        let dst = &mut dh_u[(b * max_seq + tp) * d..(b * max_seq + tp + 1) * d];
                        for c in 0..d {
                            dst[c] += dha_u[r * d + c];
                        }
                    }
                }
            }

            backward_sequence_batch(bundle, gpu, adapter, &cur_cache.tape_c, &dh_c, 1.0, &mut grads).await?;
            if let Some(ref tu) = cur_cache.tape_u {
                backward_sequence_batch(bundle, gpu, adapter, tu, &dh_u, 1.0, &mut grads).await?;
            }

            // Optional Supervised Replay Anchor (`0.15 * L_replay`)
            if opts.replay_weight > 0.0 && !replay_records.is_empty() {
                let t_cols = adapter.conditioner.target_cols.clone();
                let mut r_seqs = Vec::new();
                for _ in 0..opts.replay_batch_size.max(1) {
                    let r_idx = (rng.next_u64() as usize) % replay_records.len();
                    if let Ok(enc) = encode_molecule_to_octree(&replay_records[r_idx], bb, &t_cols, true, 0.50, &mut rng) {
                        r_seqs.push(enc);
                    }
                }
                if !r_seqs.is_empty() {
                    let (lr_tot, _, _, _, _, _) = compute_sft_batch_loss_and_grads(
                        bundle,
                        gpu,
                        adapter,
                        &r_seqs,
                        &sft_replay_opts,
                        opts.replay_weight,
                        &mut rng,
                        &mut grads,
                    )
                    .await?;
                    last_replay = lr_tot;
                }
            }

            let (gnorm, _) = opt.step(adapter, &grads, opts.lr, opts.weight_decay, 1.0);
            last_gnorm = gnorm;
        }

        // Summarize step telemetry & select best generated molecule for live 3D UI visualization!
        let n_f = b_total as f64;
        let n_valid = evals.iter().filter(|e| e.legal).count();
        let n_clean = evals.iter().filter(|e| e.clean_charge).count();
        let n_cmet = evals.iter().filter(|e| e.constraints_met).count();
        let n_hit = evals.iter().filter(|e| e.is_hit).count();
        let n_chit = evals.iter().filter(|e| e.is_clean_hit).count();
        let mean_prop_rew = evals.iter().map(|e| e.prop_score).sum::<f64>() / n_f;

        let mut fmax_vals: Vec<f64> = evals
            .iter()
            .filter_map(|e| if e.f_max_ev_a.is_finite() { Some(e.f_max_ev_a) } else { None })
            .collect();
        fmax_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_fmax = if fmax_vals.is_empty() {
            0.0
        } else {
            fmax_vals[fmax_vals.len() / 2]
        };

        let best_idx = (0..b_total)
            .filter(|&i| evals[i].legal)
            .max_by(|&a, &b| {
                let sa = (evals[a].is_clean_hit as i32 as f64) * 10.0 + evals[a].prop_score + 0.2 * evals[a].phys_score;
                let sb = (evals[b].is_clean_hit as i32 as f64) * 10.0 + evals[b].prop_score + 0.2 * evals[b].phys_score;
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            });
        let best_mol = best_idx.map(|idx| {
            let mut m = step_mols[idx].clone();
            for (k, ospec) in spec.objectives.iter().enumerate() {
                if let Some(&yp) = evals[idx].objective_preds.get(k) {
                    if yp.is_finite() {
                        m.predicted_props.insert(ospec.name.clone(), yp as f32);
                    }
                }
            }
            m
        });

        let sm = TopoGrpoStepMetrics {
            step: step_idx,
            total_steps: opts.steps.max(1),
            proposals: b_total,
            valid_rate: (n_valid as f64) / n_f,
            clean_charge_rate: if n_valid > 0 { (n_clean as f64) / (n_valid as f64) } else { 0.0 },
            constraints_met_rate: (n_cmet as f64) / n_f,
            hit_rate: (n_hit as f64) / n_f,
            clean_hit_rate: (n_chit as f64) / n_f,
            median_fmax_ev_a: med_fmax,
            mean_prop_reward: mean_prop_rew,
            loss_policy: last_loss_pol,
            kl_div: last_kl,
            loss_replay: last_replay,
            grad_norm: last_gnorm,
            best_molecule: best_mol,
        };
        on_step(&sm);
        history.push(sm);
    }

    Ok(TopoGrpoReport {
        steps_completed: history.len(),
        total_proposals,
        history,
        elapsed_ms: 0.0,
    })
}
