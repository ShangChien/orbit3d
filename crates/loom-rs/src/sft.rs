//! Stage 0/1 `LoRA SFT` (with Dual-Pass Contrastive Margin Loss & Hindsight Relabeling)
//! and Stage 2.5B Detached Sub-Voxel `0.01 Å XYZ` Equilibrium Head Distillation (`sft.rs`).

use crate::chem_rules::MolRecord;
use crate::lora_model::{
    backward_sequence_batch, build_allowed_atom_vocab_mask, cross_entropy_row_grad,
    eval_task_head_with_lora_bwd, eval_task_head_with_lora_fwd, forward_sequence_batch,
    LoomAdamW, LoomAdapterGrads, LoomGpuTrainContext, LoomLoraAdapter, SeqForwardTape,
};
use crate::octree_encode::{encode_molecule_to_octree, EncodedOctreeSeq};
use serde::{Deserialize, Serialize};
use uni3dar_rs::format::ModelBundle;
use uni3dar_rs::sampler::DetRng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftOptions {
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_min_lr")]
    pub min_lr: f32,
    #[serde(default = "default_wd")]
    pub weight_decay: f32,
    #[serde(default = "default_grad_clip")]
    pub max_grad_norm: f32,
    #[serde(default = "default_cfg_drop")]
    pub cfg_dropout_prob: f32,
    #[serde(default = "default_xyz_w")]
    pub xyz_loss_weight: f32,
    #[serde(default = "default_val_w")]
    pub valence_loss_weight: f32,
    #[serde(default = "default_contrast_w")]
    pub contrast_weight: f32,
    #[serde(default = "default_contrast_m")]
    pub contrast_margin: f32,
    #[serde(default = "default_contrast_gap")]
    pub contrast_min_gap: f32,
    #[serde(default = "default_rot_prob")]
    pub random_rotation_prob: f32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_batch_size() -> usize {
    8
}
fn default_epochs() -> usize {
    3
}
fn default_lr() -> f32 {
    2.5e-4
}
fn default_min_lr() -> f32 {
    2.5e-5
}
fn default_wd() -> f32 {
    1.0e-4
}
fn default_grad_clip() -> f32 {
    1.0
}
fn default_cfg_drop() -> f32 {
    0.15
}
fn default_xyz_w() -> f32 {
    0.10
}
fn default_val_w() -> f32 {
    0.15
}
fn default_contrast_w() -> f32 {
    0.35
}
fn default_contrast_m() -> f32 {
    0.06
}
fn default_contrast_gap() -> f32 {
    0.40
}
fn default_rot_prob() -> f32 {
    0.50
}
fn default_seed() -> u64 {
    42
}

impl Default for SftOptions {
    fn default() -> Self {
        Self {
            batch_size: 8,
            epochs: 3,
            lr: 2.5e-4,
            min_lr: 2.5e-5,
            weight_decay: 1.0e-4,
            max_grad_norm: 1.0,
            cfg_dropout_prob: 0.15,
            xyz_loss_weight: 0.10,
            valence_loss_weight: 0.15,
            contrast_weight: 0.35,
            contrast_margin: 0.06,
            contrast_min_gap: 0.40,
            random_rotation_prob: 0.50,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftStepMetrics {
    pub step: usize,
    pub total_steps: usize,
    pub loss_total: f32,
    pub loss_tree: f32,
    pub loss_atom: f32,
    pub loss_xyz: f32,
    pub loss_valence: f32,
    pub loss_contrast: f32,
    pub grad_norm: f32,
    pub lr: f32,
    pub step_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftTrainReport {
    pub steps_completed: usize,
    pub molecules_seen: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub history: Vec<SftStepMetrics>,
    pub elapsed_ms: f64,
}

struct PassHeadEval {
    loss_tree: f32,
    loss_atom: f32,
    loss_xyz: f32,
    loss_valence: f32,
    per_mol_loss: Vec<f32>,
    // Saved tensors to compute exact head + sequence VJP with optional per-molecule contrast weights!
    h_tree: Vec<f32>,
    tree_owners: Vec<usize>,
    tree_labels: Vec<usize>,
    tree_logits: Vec<f32>,
    tree_mid_pre: Vec<f32>,
    tree_mid_act: Vec<f32>,
    tree_tok_pos: Vec<(usize, usize)>,

    h_atom: Vec<f32>,
    atom_owners: Vec<usize>,
    atom_elem: Vec<usize>,
    atom_xyz: Vec<[usize; 3]>,
    atom_z: [Vec<f32>; 4],
    atom_logits: [Vec<f32>; 4],
    atom_mid_pre: [Vec<f32>; 4],
    atom_mid_act: [Vec<f32>; 4],
    atom_tok_pos: Vec<(usize, usize)>,

    heavy_rows: Vec<usize>, // indices into `0..M_atom`
    heavy_targets: Vec<(usize, usize)>,
}

async fn eval_heads_for_sft_pass(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    seqs: &[EncodedOctreeSeq],
    tape: &SeqForwardTape,
    allowed_mask: &[bool],
    xyz_loss_weight: f32,
) -> Result<PassHeadEval, String> {
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let b_sz = seqs.len();
    let max_seq = tape.max_seq;

    let mut h_tree = Vec::new();
    let mut tree_owners = Vec::new();
    let mut tree_labels = Vec::new();
    let mut tree_tok_pos = Vec::new();

    let mut h_atom = Vec::new();
    let mut atom_owners = Vec::new();
    let mut atom_elem = Vec::new();
    let mut atom_xyz = Vec::new();
    let mut atom_tok_pos = Vec::new();
    let mut heavy_rows = Vec::new();
    let mut heavy_targets = Vec::new();

    for (b, seq) in seqs.iter().enumerate() {
        for tt in &seq.tree_targets {
            let off = (b * max_seq + tt.token_pos) * d;
            h_tree.extend_from_slice(&tape.hidden_out[off..off + d]);
            tree_owners.push(b);
            tree_labels.push(tt.label_255);
            tree_tok_pos.push((b, tt.token_pos));
        }
        for at in &seq.atom_targets {
            let a_row = atom_owners.len();
            let off = (b * max_seq + at.token_pos) * d;
            h_atom.extend_from_slice(&tape.hidden_out[off..off + d]);
            atom_owners.push(b);
            atom_elem.push(at.elem_id);
            atom_xyz.push(at.xyz_bins);
            atom_tok_pos.push((b, at.token_pos));
            if let Some(hv) = at.heavy_valence {
                heavy_rows.push(a_row);
                heavy_targets.push(hv);
            }
        }
    }

    let m_tree = tree_owners.len();
    let (tree_logits, tree_mid_pre, tree_mid_act) = eval_task_head_with_lora_fwd(
        bundle,
        gpu,
        adapter,
        "bb.tree_head",
        &h_tree,
        m_tree,
        255,
        1.0,
    )
    .await?;

    let m_atom = atom_owners.len();
    let z0 = h_atom.clone();
    let mut z1 = z0.clone();
    let emb0 = bundle.f32_slice("bb.atom_head.emb.0")?;
    for r in 0..m_atom {
        let id = atom_elem[r].min((emb0.len() / d).saturating_sub(1));
        for c in 0..d {
            z1[r * d + c] += emb0[id * d + c];
        }
    }
    let mut z2 = z1.clone();
    let emb1 = bundle.f32_slice("bb.atom_head.emb.1")?;
    for r in 0..m_atom {
        let id = atom_xyz[r][0].min((emb1.len() / d).saturating_sub(1));
        for c in 0..d {
            z2[r * d + c] += emb1[id * d + c];
        }
    }
    let mut z3 = z2.clone();
    let emb2 = bundle.f32_slice("bb.atom_head.emb.2")?;
    for r in 0..m_atom {
        let id = atom_xyz[r][1].min((emb2.len() / d).saturating_sub(1));
        for c in 0..d {
            z3[r * d + c] += emb2[id * d + c];
        }
    }

    let (mut l_a0, mp0, ma0) = eval_task_head_with_lora_fwd(bundle, gpu, adapter, "bb.atom_head.pred.0", &z0, m_atom, bb.num_atom_vocab, 1.0).await?;
    for r in 0..m_atom {
        let row = &mut l_a0[r * bb.num_atom_vocab..(r + 1) * bb.num_atom_vocab];
        for v in 0..bb.num_atom_vocab {
            if !allowed_mask[v] {
                row[v] = f32::NEG_INFINITY;
            }
        }
    }
    let (l_a1, mp1, ma1) = eval_task_head_with_lora_fwd(bundle, gpu, adapter, "bb.atom_head.pred.1", &z1, m_atom, bb.num_xyz, 1.0).await?;
    let (l_a2, mp2, ma2) = eval_task_head_with_lora_fwd(bundle, gpu, adapter, "bb.atom_head.pred.2", &z2, m_atom, bb.num_xyz, 1.0).await?;
    let (l_a3, mp3, ma3) = eval_task_head_with_lora_fwd(bundle, gpu, adapter, "bb.atom_head.pred.3", &z3, m_atom, bb.num_xyz, 1.0).await?;

    // Compute token-mean losses AND per-molecule losses
    let mut dummy = vec![0.0f32; bb.num_atom_vocab.max(255)];
    let mut sum_tree = 0.0f32;
    let mut mol_tree_sum = vec![0.0f32; b_sz];
    let mut mol_tree_cnt = vec![0usize; b_sz];
    for r in 0..m_tree {
        let ce = cross_entropy_row_grad(&tree_logits[r * 255..(r + 1) * 255], tree_labels[r], 0.0, &mut dummy[..255]);
        sum_tree += ce;
        mol_tree_sum[tree_owners[r]] += ce;
        mol_tree_cnt[tree_owners[r]] += 1;
    }

    let mut sum_atom = 0.0f32;
    let mut sum_xyz = 0.0f32;
    let mut mol_atom_sum = vec![0.0f32; b_sz];
    let mut mol_xyz_sum = vec![0.0f32; b_sz];
    let mut mol_atom_cnt = vec![0usize; b_sz];
    for r in 0..m_atom {
        let ce_a = cross_entropy_row_grad(
            &l_a0[r * bb.num_atom_vocab..(r + 1) * bb.num_atom_vocab],
            atom_elem[r],
            0.0,
            &mut dummy[..bb.num_atom_vocab],
        );
        let ce_x = cross_entropy_row_grad(&l_a1[r * bb.num_xyz..(r + 1) * bb.num_xyz], atom_xyz[r][0], 0.0, &mut dummy[..bb.num_xyz]);
        let ce_y = cross_entropy_row_grad(&l_a2[r * bb.num_xyz..(r + 1) * bb.num_xyz], atom_xyz[r][1], 0.0, &mut dummy[..bb.num_xyz]);
        let ce_z = cross_entropy_row_grad(&l_a3[r * bb.num_xyz..(r + 1) * bb.num_xyz], atom_xyz[r][2], 0.0, &mut dummy[..bb.num_xyz]);
        let ce_xyz = (ce_x + ce_y + ce_z) / 3.0;
        sum_atom += ce_a;
        sum_xyz += ce_xyz;
        mol_atom_sum[atom_owners[r]] += ce_a;
        mol_xyz_sum[atom_owners[r]] += ce_xyz;
        mol_atom_cnt[atom_owners[r]] += 1;
    }

    let mut per_mol_loss = vec![0.0f32; b_sz];
    for b in 0..b_sz {
        let lt = if mol_tree_cnt[b] > 0 { mol_tree_sum[b] / (mol_tree_cnt[b] as f32) } else { 0.0 };
        let la = if mol_atom_cnt[b] > 0 { mol_atom_sum[b] / (mol_atom_cnt[b] as f32) } else { 0.0 };
        let lx = if mol_atom_cnt[b] > 0 { mol_xyz_sum[b] / (mol_atom_cnt[b] as f32) } else { 0.0 };
        per_mol_loss[b] = lt + la + xyz_loss_weight * lx;
    }

    Ok(PassHeadEval {
        loss_tree: if m_tree > 0 { sum_tree / (m_tree as f32) } else { 0.0 },
        loss_atom: if m_atom > 0 { sum_atom / (m_atom as f32) } else { 0.0 },
        loss_xyz: if m_atom > 0 { sum_xyz / (m_atom as f32) } else { 0.0 },
        loss_valence: 0.0,
        per_mol_loss,
        h_tree,
        tree_owners,
        tree_labels,
        tree_logits,
        tree_mid_pre,
        tree_mid_act,
        tree_tok_pos,
        h_atom,
        atom_owners,
        atom_elem,
        atom_xyz,
        atom_z: [z0, z1, z2, z3],
        atom_logits: [l_a0, l_a1, l_a2, l_a3],
        atom_mid_pre: [mp0, mp1, mp2, mp3],
        atom_mid_act: [ma0, ma1, ma2, ma3],
        atom_tok_pos,
        heavy_rows,
        heavy_targets,
    })
}

async fn backprop_sft_pass(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    tape: &SeqForwardTape,
    eval: &mut PassHeadEval,
    base_gen_scale: f32,
    per_mol_extra_scale: &[f32],
    xyz_loss_weight: f32,
    valence_loss_weight: f32,
    grads: &mut LoomAdapterGrads,
) -> Result<(), String> {
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let b_sz = tape.b_sz;
    let max_seq = tape.max_seq;

    let m_tree = eval.tree_owners.len();
    let m_atom = eval.atom_owners.len();

    let mut mol_tree_cnt = vec![0usize; b_sz];
    for &b in &eval.tree_owners {
        mol_tree_cnt[b] += 1;
    }
    let mut mol_atom_cnt = vec![0usize; b_sz];
    for &b in &eval.atom_owners {
        mol_atom_cnt[b] += 1;
    }

    let mut d_hidden = vec![0.0f32; b_sz * max_seq * d];

    // 1. Tree head backward
    if m_tree > 0 {
        let inv_mt = base_gen_scale / (m_tree as f32);
        let mut dl_tree = vec![0.0f32; m_tree * 255];
        for r in 0..m_tree {
            let b = eval.tree_owners[r];
            let c_scale = per_mol_extra_scale[b] / (mol_tree_cnt[b].max(1) as f32);
            let w_r = inv_mt + c_scale;
            cross_entropy_row_grad(
                &eval.tree_logits[r * 255..(r + 1) * 255],
                eval.tree_labels[r],
                w_r,
                &mut dl_tree[r * 255..(r + 1) * 255],
            );
        }
        let dh_tree = eval_task_head_with_lora_bwd(
            bundle,
            gpu,
            adapter,
            "bb.tree_head",
            &eval.h_tree,
            &eval.tree_mid_pre,
            &eval.tree_mid_act,
            &dl_tree,
            m_tree,
            1.0,
            Some(grads),
        )
        .await?;
        for (r, &(b, tp)) in eval.tree_tok_pos.iter().enumerate() {
            let dst = &mut d_hidden[(b * max_seq + tp) * d..(b * max_seq + tp + 1) * d];
            let src = &dh_tree[r * d..(r + 1) * d];
            for c in 0..d {
                dst[c] += src[c];
            }
        }
    }

    // 2. Atom & XYZ heads backward (`dz3 -> dz2 -> dz1 -> dz0`)
    if m_atom > 0 {
        let inv_ma = base_gen_scale / (m_atom as f32);
        let mut dl_a0 = vec![0.0f32; m_atom * bb.num_atom_vocab];
        let mut dl_xyz = [
            vec![0.0f32; m_atom * bb.num_xyz],
            vec![0.0f32; m_atom * bb.num_xyz],
            vec![0.0f32; m_atom * bb.num_xyz],
        ];
        for r in 0..m_atom {
            let b = eval.atom_owners[r];
            let c_scale = per_mol_extra_scale[b] / (mol_atom_cnt[b].max(1) as f32);
            let w_elem = inv_ma + c_scale;
            let w_xyz = (xyz_loss_weight / 3.0) * (inv_ma + c_scale);
            cross_entropy_row_grad(
                &eval.atom_logits[0][r * bb.num_atom_vocab..(r + 1) * bb.num_atom_vocab],
                eval.atom_elem[r],
                w_elem,
                &mut dl_a0[r * bb.num_atom_vocab..(r + 1) * bb.num_atom_vocab],
            );
            for ax in 0..3 {
                cross_entropy_row_grad(
                    &eval.atom_logits[ax + 1][r * bb.num_xyz..(r + 1) * bb.num_xyz],
                    eval.atom_xyz[r][ax],
                    w_xyz,
                    &mut dl_xyz[ax][r * bb.num_xyz..(r + 1) * bb.num_xyz],
                );
            }
        }

        let dz3 = eval_task_head_with_lora_bwd(
            bundle,
            gpu,
            adapter,
            "bb.atom_head.pred.3",
            &eval.atom_z[3],
            &eval.atom_mid_pre[3],
            &eval.atom_mid_act[3],
            &dl_xyz[2],
            m_atom,
            1.0,
            Some(grads),
        )
        .await?;
        let mut dz2 = eval_task_head_with_lora_bwd(
            bundle,
            gpu,
            adapter,
            "bb.atom_head.pred.2",
            &eval.atom_z[2],
            &eval.atom_mid_pre[2],
            &eval.atom_mid_act[2],
            &dl_xyz[1],
            m_atom,
            1.0,
            Some(grads),
        )
        .await?;
        for i in 0..dz2.len() {
            dz2[i] += dz3[i];
        }
        let mut dz1 = eval_task_head_with_lora_bwd(
            bundle,
            gpu,
            adapter,
            "bb.atom_head.pred.1",
            &eval.atom_z[1],
            &eval.atom_mid_pre[1],
            &eval.atom_mid_act[1],
            &dl_xyz[0],
            m_atom,
            1.0,
            Some(grads),
        )
        .await?;
        for i in 0..dz1.len() {
            dz1[i] += dz2[i];
        }
        let mut dz0 = eval_task_head_with_lora_bwd(
            bundle,
            gpu,
            adapter,
            "bb.atom_head.pred.0",
            &eval.atom_z[0],
            &eval.atom_mid_pre[0],
            &eval.atom_mid_act[0],
            &dl_a0,
            m_atom,
            1.0,
            Some(grads),
        )
        .await?;
        for i in 0..dz0.len() {
            dz0[i] += dz1[i];
        }

        // 3. Optional `HeavyAtomValenceHead` auxiliary loss (only on primary positive pass `base_gen_scale > 0`)
        if base_gen_scale > 0.0 && valence_loss_weight > 0.0 && !eval.heavy_rows.is_empty() {
            if let Some(ref vh) = adapter.valence_head {
                let mut h_hvy = Vec::with_capacity(eval.heavy_rows.len() * d);
                for &ar in &eval.heavy_rows {
                    h_hvy.extend_from_slice(&eval.h_atom[ar * d..(ar + 1) * d]);
                }
                let (v_loss, dh_hvy) = vh.forward_and_bwd(
                    &h_hvy,
                    &eval.heavy_targets,
                    base_gen_scale * valence_loss_weight,
                    grads.valence.as_mut(),
                );
                eval.loss_valence = v_loss;
                for (k, &ar) in eval.heavy_rows.iter().enumerate() {
                    for c in 0..d {
                        dz0[ar * d + c] += dh_hvy[k * d + c];
                    }
                }
            }
        }

        for (r, &(b, tp)) in eval.atom_tok_pos.iter().enumerate() {
            let dst = &mut d_hidden[(b * max_seq + tp) * d..(b * max_seq + tp + 1) * d];
            let src = &dz0[r * d..(r + 1) * d];
            for c in 0..d {
                dst[c] += src[c];
            }
        }
    }

    backward_sequence_batch(bundle, gpu, adapter, tape, &d_hidden, 1.0, grads).await
}

/// Run a single SFT batch (`compute_sft_loss` + optional contrastive second pass) and accumulate into `grads`!
/// (`grad_weight` lets `TopoGRPO` reuse this directly for `0.15 * L_replay`!)
pub async fn compute_sft_batch_loss_and_grads(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    seqs: &[EncodedOctreeSeq],
    opts: &SftOptions,
    grad_weight: f32,
    rng: &mut DetRng,
    grads: &mut LoomAdapterGrads,
) -> Result<(f32, f32, f32, f32, f32, f32), String> {
    let b_sz = seqs.len();
    if b_sz == 0 {
        return Ok((0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }

    let raw_targets: Vec<Vec<f32>> = seqs.iter().map(|s| s.targets_raw.clone()).collect();
    let mut train_masks: Vec<Vec<bool>> = Vec::with_capacity(b_sz);
    for s in seqs {
        let drop = opts.cfg_dropout_prob > 0.0 && rng.next_f32() < opts.cfg_dropout_prob;
        if drop {
            train_masks.push(vec![false; s.active_mask.len()]);
        } else {
            train_masks.push(s.active_mask.clone());
        }
    }

    let allowed_mask = build_allowed_atom_vocab_mask(bundle);
    let tape_pos = forward_sequence_batch(
        bundle,
        gpu,
        adapter,
        seqs,
        &raw_targets,
        &train_masks,
        1.0,
        true,
    )
    .await?;
    let mut eval_pos = eval_heads_for_sft_pass(
        bundle,
        gpu,
        adapter,
        seqs,
        &tape_pos,
        &allowed_mask,
        opts.xyz_loss_weight,
    )
    .await?;

    // Check incompatible negative pairing for `condition_margin_loss`
    let mut pos_extra_scale = vec![0.0f32; b_sz];
    let mut loss_contrast = 0.0f32;

    if opts.contrast_weight > 0.0 && b_sz >= 2 {
        let mut neg_partner: Vec<Option<usize>> = vec![None; b_sz];
        let mut n_elig = 0usize;
        for i in 0..b_sz {
            if !train_masks[i].iter().any(|&b| b) {
                continue;
            }
            let mut best_j = None;
            let mut best_u = -1.0f32;
            for j in 0..b_sz {
                if i == j {
                    continue;
                }
                let mut max_gap = 0.0f32;
                for p in 0..train_masks[i].len() {
                    if train_masks[i][p] {
                        let gap = (raw_targets[i][p] - raw_targets[j][p]).abs();
                        if gap > max_gap {
                            max_gap = gap;
                        }
                    }
                }
                if max_gap > opts.contrast_min_gap {
                    let u = rng.next_f32();
                    if u > best_u {
                        best_u = u;
                        best_j = Some(j);
                    }
                }
            }
            if best_j.is_some() {
                neg_partner[i] = best_j;
                n_elig += 1;
            }
        }

        if n_elig > 0 {
            let neg_targets: Vec<Vec<f32>> = (0..b_sz)
                .map(|i| {
                    if let Some(j) = neg_partner[i] {
                        raw_targets[j].clone()
                    } else {
                        raw_targets[i].clone()
                    }
                })
                .collect();
            let tape_neg = forward_sequence_batch(
                bundle,
                gpu,
                adapter,
                seqs,
                &neg_targets,
                &train_masks,
                1.0,
                true,
            )
            .await?;
            let mut eval_neg = eval_heads_for_sft_pass(
                bundle,
                gpu,
                adapter,
                seqs,
                &tape_neg,
                &allowed_mask,
                opts.xyz_loss_weight,
            )
            .await?;

            let mut neg_extra_scale = vec![0.0f32; b_sz];
            let mut c_sum = 0.0f32;
            let unit_c = grad_weight * opts.contrast_weight / (n_elig as f32);
            for i in 0..b_sz {
                if neg_partner[i].is_some() {
                    let viol = opts.contrast_margin + eval_pos.per_mol_loss[i] - eval_neg.per_mol_loss[i];
                    if viol > 0.0 {
                        c_sum += viol;
                        pos_extra_scale[i] = unit_c;
                        neg_extra_scale[i] = -unit_c;
                    }
                }
            }
            loss_contrast = c_sum / (n_elig as f32);

            if neg_extra_scale.iter().any(|&v| v != 0.0) {
                backprop_sft_pass(
                    bundle,
                    gpu,
                    adapter,
                    &tape_neg,
                    &mut eval_neg,
                    0.0,
                    &neg_extra_scale,
                    opts.xyz_loss_weight,
                    0.0,
                    grads,
                )
                .await?;
            }
        }
    }

    backprop_sft_pass(
        bundle,
        gpu,
        adapter,
        &tape_pos,
        &mut eval_pos,
        grad_weight,
        &pos_extra_scale,
        opts.xyz_loss_weight,
        opts.valence_loss_weight,
        grads,
    )
    .await?;

    let loss_gen = eval_pos.loss_tree
        + eval_pos.loss_atom
        + opts.xyz_loss_weight * eval_pos.loss_xyz
        + opts.valence_loss_weight * eval_pos.loss_valence;
    let loss_total = loss_gen + opts.contrast_weight * loss_contrast;

    Ok((
        loss_total,
        eval_pos.loss_tree,
        eval_pos.loss_atom,
        eval_pos.loss_xyz,
        eval_pos.loss_valence,
        loss_contrast,
    ))
}

/// Full Stage 0/1 `LoRA SFT` Training Loop over `MolRecord`s!
pub async fn run_lora_sft<F>(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &mut LoomLoraAdapter,
    records: &[MolRecord],
    opts: &SftOptions,
    mut on_step: F,
) -> Result<SftTrainReport, String>
where
    F: FnMut(&SftStepMetrics),
{
    if records.is_empty() {
        return Err("No molecules provided for LoRA SFT".into());
    }
    adapter.set_topology_policy_mode(true);
    let mut rng = DetRng::new(opts.seed);
    let b_sz = opts.batch_size.max(1);
    let epochs = opts.epochs.max(1);

    // Filter valid encodable records upfront
    let bb = &bundle.header.backbone;
    let t_cols = adapter.conditioner.target_cols.clone();
    let mut valid_recs: Vec<&MolRecord> = Vec::new();
    let mut test_rng = DetRng::new(opts.seed ^ 0x1234);
    for r in records {
        if r.valence_valid
            && encode_molecule_to_octree(r, bb, &t_cols, false, 0.0, &mut test_rng).is_ok()
        {
            valid_recs.push(r);
        }
    }
    if valid_recs.is_empty() {
        return Err("No valid molecules passed Uni-3DAR octree encoding".into());
    }

    let steps_per_ep = (valid_recs.len() + b_sz - 1) / b_sz;
    let total_steps = steps_per_ep * epochs;
    let warmup = (total_steps / 10).min(250);

    let mut opt = LoomAdamW::new();
    let mut history = Vec::with_capacity(total_steps);
    let mut step_idx = 0usize;
    let mut molecules_seen = 0usize;

    for _ep in 0..epochs {
        let mut perm: Vec<usize> = (0..valid_recs.len()).collect();
        for i in (1..perm.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }

        for chunk in perm.chunks(b_sz) {
            let mut seqs = Vec::with_capacity(chunk.len());
            for &idx in chunk {
                if let Ok(enc) = encode_molecule_to_octree(
                    valid_recs[idx],
                    bb,
                    &t_cols,
                    true,
                    opts.random_rotation_prob,
                    &mut rng,
                ) {
                    seqs.push(enc);
                }
            }
            if seqs.is_empty() {
                continue;
            }

            step_idx += 1;
            molecules_seen += seqs.len();

            // Cosine LR schedule with linear warmup
            let lr = if warmup > 0 && step_idx <= warmup {
                opts.lr * ((step_idx as f32) / (warmup as f32))
            } else if total_steps > warmup {
                let prog = ((step_idx - warmup) as f32) / ((total_steps - warmup) as f32);
                let cos_v = 0.5 * (1.0 + (std::f32::consts::PI * prog.clamp(0.0, 1.0)).cos());
                opts.min_lr + (opts.lr - opts.min_lr) * cos_v
            } else {
                opts.lr
            };

            let mut grads = adapter.zero_grads();
            let (l_tot, l_tree, l_atom, l_xyz, l_val, l_con) = compute_sft_batch_loss_and_grads(
                bundle,
                gpu,
                adapter,
                &seqs,
                opts,
                1.0,
                &mut rng,
                &mut grads,
            )
            .await?;

            let (gnorm, _) = opt.step(adapter, &grads, lr, opts.weight_decay, opts.max_grad_norm);
            let m = SftStepMetrics {
                step: step_idx,
                total_steps,
                loss_total: l_tot,
                loss_tree: l_tree,
                loss_atom: l_atom,
                loss_xyz: l_xyz,
                loss_valence: l_val,
                loss_contrast: l_con,
                grad_norm: gnorm,
                lr,
                step_ms: 0.0,
            };
            on_step(&m);
            history.push(m);
        }
    }

    let initial_loss = history.first().map_or(0.0, |h| h.loss_total);
    let final_loss = history.last().map_or(0.0, |h| h.loss_total);

    Ok(SftTrainReport {
        steps_completed: step_idx,
        molecules_seen,
        initial_loss,
        final_loss,
        history,
        elapsed_ms: 0.0,
    })
}

// ============================================================================
// Stage 2.5B: Detached Sub-Voxel `0.01 Å XYZ` Head Equilibrium Distillation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XyzDistillOptions {
    #[serde(default = "default_xyz_bs")]
    pub batch_size: usize,
    #[serde(default = "default_xyz_epochs")]
    pub epochs: usize,
    #[serde(default = "default_xyz_lr")]
    pub lr: f32,
    #[serde(default = "default_xyz_wd")]
    pub weight_decay: f32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_xyz_bs() -> usize {
    16
}
fn default_xyz_epochs() -> usize {
    4
}
fn default_xyz_lr() -> f32 {
    1.0e-4
}
fn default_xyz_wd() -> f32 {
    1.0e-3
}

impl Default for XyzDistillOptions {
    fn default() -> Self {
        Self {
            batch_size: 16,
            epochs: 4,
            lr: 1.0e-4,
            weight_decay: 1.0e-3,
            seed: 42,
        }
    }
}

/// Freeze 100% of backbone + conditioner + topology/element LoRA, and distill MLIP-relaxed
/// `0.01 Å` equilibrium coordinates into `bb.atom_head.pred.{1,2,3}` (`111,744` scalars at `r=16`)
/// with ZERO Transformer backward pass!
pub async fn run_xyz_head_distillation<F>(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &mut LoomLoraAdapter,
    equilibrium_records: &[MolRecord],
    opts: &XyzDistillOptions,
    mut on_step: F,
) -> Result<SftTrainReport, String>
where
    F: FnMut(usize, usize, f32),
{
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    adapter.set_xyz_distill_only_mode(d, bb.num_xyz, opts.seed);

    let t_cols = adapter.conditioner.target_cols.clone();
    let mut rng = DetRng::new(opts.seed ^ 0x25B0);
    let mut valid_recs: Vec<&MolRecord> = Vec::new();
    for r in equilibrium_records {
        if r.valence_valid
            && encode_molecule_to_octree(r, bb, &t_cols, false, 0.0, &mut rng).is_ok()
        {
            valid_recs.push(r);
        }
    }
    if valid_recs.is_empty() {
        return Err("No valid equilibrium molecules for XYZ distillation".into());
    }

    let b_sz = opts.batch_size.max(1);
    let epochs = opts.epochs.max(1);
    let total_steps = ((valid_recs.len() + b_sz - 1) / b_sz) * epochs;
    let mut opt = LoomAdamW::new();
    let mut history = Vec::with_capacity(total_steps);
    let mut step_idx = 0usize;
    let mut molecules_seen = 0usize;

    let emb0 = bundle.f32_slice("bb.atom_head.emb.0")?;
    let emb1 = bundle.f32_slice("bb.atom_head.emb.1")?;
    let emb2 = bundle.f32_slice("bb.atom_head.emb.2")?;

    for _ep in 0..epochs {
        let mut perm: Vec<usize> = (0..valid_recs.len()).collect();
        for i in (1..perm.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        for chunk in perm.chunks(b_sz) {
            let mut seqs = Vec::with_capacity(chunk.len());
            for &idx in chunk {
                if let Ok(enc) = encode_molecule_to_octree(valid_recs[idx], bb, &t_cols, true, 0.80, &mut rng) {
                    seqs.push(enc);
                }
            }
            if seqs.is_empty() {
                continue;
            }
            step_idx += 1;
            molecules_seen += seqs.len();

            let raw_t: Vec<Vec<f32>> = seqs.iter().map(|s| s.targets_raw.clone()).collect();
            let act_m: Vec<Vec<bool>> = seqs.iter().map(|s| s.active_mask.clone()).collect();

            // Detached `no_grad` forward (`save_checkpoints = false`)!
            let tape = forward_sequence_batch(bundle, gpu, adapter, &seqs, &raw_t, &act_m, 1.0, false).await?;
            let max_seq = tape.max_seq;

            let mut z0 = Vec::new();
            let mut elems = Vec::new();
            let mut u_xyz = Vec::new();
            for (b, s) in seqs.iter().enumerate() {
                for at in &s.atom_targets {
                    if at.elem_id < 342 {
                        let off = (b * max_seq + at.token_pos) * d;
                        z0.extend_from_slice(&tape.hidden_out[off..off + d]);
                        elems.push(at.elem_id);
                        u_xyz.push(at.xyz_bins);
                    }
                }
            }
            let m_real = elems.len();
            if m_real == 0 {
                continue;
            }

            let mut z1 = z0.clone();
            for r in 0..m_real {
                let id = elems[r].min((emb0.len() / d).saturating_sub(1));
                for c in 0..d {
                    z1[r * d + c] += emb0[id * d + c];
                }
            }
            let mut z2 = z1.clone();
            for r in 0..m_real {
                let id = u_xyz[r][0].min((emb1.len() / d).saturating_sub(1));
                for c in 0..d {
                    z2[r * d + c] += emb1[id * d + c];
                }
            }
            let mut z3 = z2.clone();
            for r in 0..m_real {
                let id = u_xyz[r][1].min((emb2.len() / d).saturating_sub(1));
                for c in 0..d {
                    z3[r * d + c] += emb2[id * d + c];
                }
            }

            let zs = [&z1, &z2, &z3];
            let mut grads = adapter.zero_grads();
            let mut loss_xyz = 0.0f32;
            let inv_3m = 1.0 / (3.0 * (m_real as f32));

            for ax in 0..3 {
                let pfx = format!("bb.atom_head.pred.{}", ax + 1);
                let (logits, mp, ma) = eval_task_head_with_lora_fwd(bundle, gpu, adapter, &pfx, zs[ax], m_real, bb.num_xyz, 1.0).await?;
                let mut dl = vec![0.0f32; m_real * bb.num_xyz];
                for r in 0..m_real {
                    loss_xyz += cross_entropy_row_grad(
                        &logits[r * bb.num_xyz..(r + 1) * bb.num_xyz],
                        u_xyz[r][ax],
                        inv_3m,
                        &mut dl[r * bb.num_xyz..(r + 1) * bb.num_xyz],
                    ) / (3.0 * (m_real as f32));
                }
                let _ = eval_task_head_with_lora_bwd(
                    bundle,
                    gpu,
                    adapter,
                    &pfx,
                    zs[ax],
                    &mp,
                    &ma,
                    &dl,
                    m_real,
                    1.0,
                    Some(&mut grads),
                )
                .await?;
            }

            let (gnorm, _) = opt.step(adapter, &grads, opts.lr, opts.weight_decay, 1.0);
            on_step(step_idx, total_steps, loss_xyz);
            history.push(SftStepMetrics {
                step: step_idx,
                total_steps,
                loss_total: loss_xyz,
                loss_tree: 0.0,
                loss_atom: 0.0,
                loss_xyz,
                loss_valence: 0.0,
                loss_contrast: 0.0,
                grad_norm: gnorm,
                lr: opts.lr,
                step_ms: 0.0,
            });
        }
    }

    let initial_loss = history.first().map_or(0.0, |h| h.loss_total);
    let final_loss = history.last().map_or(0.0, |h| h.loss_total);
    Ok(SftTrainReport {
        steps_completed: step_idx,
        molecules_seen,
        initial_loss,
        final_loss,
        history,
        elapsed_ms: 0.0,
    })
}
