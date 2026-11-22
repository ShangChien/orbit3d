//! Differentiable Shared-Backbone 3D Distance-Biased Attention Property Oracle Trainer
//! (`_DPA4EquivariantPropertyAdapter` + `_BootstrapAdapters` from `src/loom/oracles/dpa4_oracle.py`).
//!
//! Supports:
//! 1. Exact analytical reverse-mode VJP wrt all adapter parameters (`AdamW` + Huber `beta=0.1` + grad clip `5.0`).
//! 2. Exact analytical reverse-mode VJP wrt frozen MLIP descriptors `dL/dD_a` AND 3D pairwise distances `dL/dR_i`
//!    so `PhysicsConstrainedCoordRefiner` gets exact analytical `grad_coords` when combined with `hyff-rs` / `dpa4-rs`!
//! 3. 5-member Bootstrap resampling with scaffold-grouped validation split & best-checkpoint selection.
//! 4. Compact JSON/binary serialization compatible with `mlip_core::BootstrapPropertyOracle`.

use crate::chem_rules::{murcko_scaffold_signature, HeavyMolGraph};
use mlip_core::{
    AdapterHeadWeights, BootstrapPropertyOracle, EquivariantAdapterMember, LayerNormWeights,
    LinearWeights, PropertyPrediction, TargetScaler,
};
use serde::{Deserialize, Serialize};
use uni3dar_rs::sampler::DetRng;

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleTrainOptions {
    #[serde(default = "default_members")]
    pub n_members: usize,
    #[serde(default = "default_hidden")]
    pub hidden_dim: usize,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_wd")]
    pub weight_decay: f32,
    #[serde(default = "default_grad_clip")]
    pub max_grad_norm: f32,
    #[serde(default = "default_val_frac")]
    pub val_fraction: f32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_members() -> usize {
    5
}
fn default_hidden() -> usize {
    128
}
fn default_epochs() -> usize {
    35
}
fn default_lr() -> f32 {
    2.0e-4
}
fn default_wd() -> f32 {
    1.0e-4
}
fn default_grad_clip() -> f32 {
    5.0
}
fn default_val_frac() -> f32 {
    0.15
}
fn default_seed() -> u64 {
    42
}

impl Default for OracleTrainOptions {
    fn default() -> Self {
        Self {
            n_members: 5,
            hidden_dim: 128,
            epochs: 35,
            lr: 2.0e-4,
            weight_decay: 1.0e-4,
            max_grad_norm: 5.0,
            val_fraction: 0.15,
            seed: 42,
        }
    }
}

/// Pre-extracted MLIP descriptor + 3D geometry + target vector for fast multi-epoch Bootstrap training.
#[derive(Debug, Clone)]
pub struct CachedDescriptorSample {
    pub atom_repr: Vec<f32>,
    pub coords_a: Vec<[f64; 3]>,
    pub targets: Vec<f32>,
    pub scaffold_sig: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleTrainReport {
    pub property_names: Vec<String>,
    pub property_units: Vec<String>,
    pub desc_dim: usize,
    pub hidden_dim: usize,
    pub n_members: usize,
    pub n_train: usize,
    pub n_val: usize,
    pub member_val_norm_mae: Vec<f32>,
    pub ensemble_val_mae_per_prop: Vec<f32>,
    pub elapsed_ms: f64,
}

/// Serializable `BootstrapPropertyOracle` bundle (`.loom_oracle` JSON/MsgPack-friendly struct).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedOracleBundle {
    pub backbone_family: String,
    pub property_names: Vec<String>,
    pub property_units: Vec<String>,
    pub scalers: Vec<TargetScaler>,
    pub desc_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub rbf_centers: [f32; 16],
    pub members: Vec<SerializedMemberWeights>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedMemberWeights {
    pub params: Vec<f32>,
}

// ============================================================================
// Flat-Parameter Differentiable Adapter Member + AdamW State
// ============================================================================

#[derive(Clone, Debug)]
struct ParamOffsets {
    in_proj_w: (usize, usize),
    in_proj_b: (usize, usize),
    norm1_w: (usize, usize),
    norm1_b: (usize, usize),
    dist_rbf_w: (usize, usize),
    dist_rbf_b: (usize, usize),
    qkv_w: (usize, usize),
    qkv_b: (usize, usize),
    out_proj_w: (usize, usize),
    out_proj_b: (usize, usize),
    norm2_w: (usize, usize),
    norm2_b: (usize, usize),
    ffn0_w: (usize, usize),
    ffn0_b: (usize, usize),
    ffn1_w: (usize, usize),
    ffn1_b: (usize, usize),
    gate_w: (usize, usize),
    gate_b: (usize, usize),
    hnorm_w: (usize, usize),
    hnorm_b: (usize, usize),
    hfc0_w: (usize, usize),
    hfc0_b: (usize, usize),
    hfc1_w: (usize, usize),
    hfc1_b: (usize, usize),
    hfc2_w: (usize, usize),
    hfc2_b: (usize, usize),
    total: usize,
    /// Boolean flag per parameter indicating whether `AdamW` weight decay applies (`true` on Linear weights, `false` on biases/LayerNorm).
    decay_mask: Vec<bool>,
}

impl ParamOffsets {
    fn new(desc_dim: usize, h: usize, out_dim: usize) -> Self {
        let mut cur = 0usize;
        let mut decay_mask = Vec::new();
        let mut alloc = |len: usize, decay: bool| -> (usize, usize) {
            let s = cur;
            cur += len;
            decay_mask.resize(cur, decay);
            (s, cur)
        };
        let in_proj_w = alloc(h * desc_dim, true);
        let in_proj_b = alloc(h, false);
        let norm1_w = alloc(h, false);
        let norm1_b = alloc(h, false);
        let dist_rbf_w = alloc(4 * 16, true);
        let dist_rbf_b = alloc(4, false);
        let qkv_w = alloc(3 * h * h, true);
        let qkv_b = alloc(3 * h, false);
        let out_proj_w = alloc(h * h, true);
        let out_proj_b = alloc(h, false);
        let norm2_w = alloc(h, false);
        let norm2_b = alloc(h, false);
        let ffn0_w = alloc(2 * h * h, true);
        let ffn0_b = alloc(2 * h, false);
        let ffn1_w = alloc(h * 2 * h, true);
        let ffn1_b = alloc(h, false);
        let gate_w = alloc(h, true);
        let gate_b = alloc(1, false);
        let hnorm_w = alloc(2 * h, false);
        let hnorm_b = alloc(2 * h, false);
        let hfc0_w = alloc(h * 2 * h, true);
        let hfc0_b = alloc(h, false);
        let hfc1_w = alloc((h / 2) * h, true);
        let hfc1_b = alloc(h / 2, false);
        let hfc2_w = alloc(out_dim * (h / 2), true);
        let hfc2_b = alloc(out_dim, false);

        Self {
            in_proj_w,
            in_proj_b,
            norm1_w,
            norm1_b,
            dist_rbf_w,
            dist_rbf_b,
            qkv_w,
            qkv_b,
            out_proj_w,
            out_proj_b,
            norm2_w,
            norm2_b,
            ffn0_w,
            ffn0_b,
            ffn1_w,
            ffn1_b,
            gate_w,
            gate_b,
            hnorm_w,
            hnorm_b,
            hfc0_w,
            hfc0_b,
            hfc1_w,
            hfc1_b,
            hfc2_w,
            hfc2_b,
            total: cur,
            decay_mask,
        }
    }
}

fn default_rbf_centers() -> [f32; 16] {
    let mut c = [0.0f32; 16];
    for i in 0..16 {
        c[i] = 0.5 + (5.5 * (i as f32)) / 15.0;
    }
    c
}

fn fill_kaiming(p: &mut [f32], range: (usize, usize), fan_in: usize, rng: &mut DetRng) {
    let bound = (1.0 / (fan_in.max(1) as f32)).sqrt();
    for idx in range.0..range.1 {
        p[idx] = (rng.next_f32() * 2.0 - 1.0) * bound;
    }
}

fn init_member_params(off: &ParamOffsets, desc_dim: usize, h: usize, out_dim: usize, rng: &mut DetRng) -> Vec<f32> {
    let mut p = vec![0.0f32; off.total];
    fill_kaiming(&mut p, off.in_proj_w, desc_dim, rng);
    p[off.norm1_w.0..off.norm1_w.1].fill(1.0);
    fill_kaiming(&mut p, off.dist_rbf_w, 16, rng);
    fill_kaiming(&mut p, off.qkv_w, h, rng);
    fill_kaiming(&mut p, off.out_proj_w, h, rng);
    p[off.norm2_w.0..off.norm2_w.1].fill(1.0);
    fill_kaiming(&mut p, off.ffn0_w, h, rng);
    fill_kaiming(&mut p, off.ffn1_w, 2 * h, rng);
    fill_kaiming(&mut p, off.gate_w, h, rng);
    p[off.hnorm_w.0..off.hnorm_w.1].fill(1.0);
    fill_kaiming(&mut p, off.hfc0_w, 2 * h, rng);
    fill_kaiming(&mut p, off.hfc1_w, h, rng);
    let _ = out_dim;
    fill_kaiming(&mut p, off.hfc2_w, h / 2, rng);
    p
}

fn pack_member_to_mlip(
    p: &[f32],
    off: &ParamOffsets,
    desc_dim: usize,
    h: usize,
    out_dim: usize,
    rbf_centers: [f32; 16],
) -> EquivariantAdapterMember {
    let lw = |wr: (usize, usize), br: (usize, usize), in_d: usize, out_d: usize| LinearWeights {
        weight: p[wr.0..wr.1].to_vec(),
        bias: p[br.0..br.1].to_vec(),
        in_dim: in_d,
        out_dim: out_d,
    };
    let lnw = |wr: (usize, usize), br: (usize, usize)| LayerNormWeights {
        weight: p[wr.0..wr.1].to_vec(),
        bias: p[br.0..br.1].to_vec(),
        eps: 1e-5,
    };
    EquivariantAdapterMember {
        desc_dim,
        hidden_dim: h,
        output_dim: out_dim,
        rbf_centers,
        in_proj: lw(off.in_proj_w, off.in_proj_b, desc_dim, h),
        dist_rbf: lw(off.dist_rbf_w, off.dist_rbf_b, 16, 4),
        qkv: lw(off.qkv_w, off.qkv_b, h, 3 * h),
        out_proj: lw(off.out_proj_w, off.out_proj_b, h, h),
        norm1: lnw(off.norm1_w, off.norm1_b),
        norm2: lnw(off.norm2_w, off.norm2_b),
        ffn0: lw(off.ffn0_w, off.ffn0_b, h, 2 * h),
        ffn1: lw(off.ffn1_w, off.ffn1_b, 2 * h, h),
        heads: vec![AdapterHeadWeights {
            gate: lw(off.gate_w, off.gate_b, h, 1),
            head_norm: lnw(off.hnorm_w, off.hnorm_b),
            head_fc0: lw(off.hfc0_w, off.hfc0_b, 2 * h, h),
            head_fc1: lw(off.hfc1_w, off.hfc1_b, h, h / 2),
            head_fc2: lw(off.hfc2_w, off.hfc2_b, h / 2, out_dim),
        }],
    }
}

fn unpack_member_from_mlip(m: &EquivariantAdapterMember, off: &ParamOffsets) -> Result<Vec<f32>, String> {
    if m.heads.len() != 1 {
        return Err(format!("Expected 1 head per bootstrap member, got {}", m.heads.len()));
    }
    let mut p = vec![0.0f32; off.total];
    let mut copy_r = |r: (usize, usize), src: &[f32]| {
        p[r.0..r.1].copy_from_slice(src);
    };
    copy_r(off.in_proj_w, &m.in_proj.weight);
    copy_r(off.in_proj_b, &m.in_proj.bias);
    copy_r(off.norm1_w, &m.norm1.weight);
    copy_r(off.norm1_b, &m.norm1.bias);
    copy_r(off.dist_rbf_w, &m.dist_rbf.weight);
    copy_r(off.dist_rbf_b, &m.dist_rbf.bias);
    copy_r(off.qkv_w, &m.qkv.weight);
    copy_r(off.qkv_b, &m.qkv.bias);
    copy_r(off.out_proj_w, &m.out_proj.weight);
    copy_r(off.out_proj_b, &m.out_proj.bias);
    copy_r(off.norm2_w, &m.norm2.weight);
    copy_r(off.norm2_b, &m.norm2.bias);
    copy_r(off.ffn0_w, &m.ffn0.weight);
    copy_r(off.ffn0_b, &m.ffn0.bias);
    copy_r(off.ffn1_w, &m.ffn1.weight);
    copy_r(off.ffn1_b, &m.ffn1.bias);
    let hd = &m.heads[0];
    copy_r(off.gate_w, &hd.gate.weight);
    copy_r(off.gate_b, &hd.gate.bias);
    copy_r(off.hnorm_w, &hd.head_norm.weight);
    copy_r(off.hnorm_b, &hd.head_norm.bias);
    copy_r(off.hfc0_w, &hd.head_fc0.weight);
    copy_r(off.hfc0_b, &hd.head_fc0.bias);
    copy_r(off.hfc1_w, &hd.head_fc1.weight);
    copy_r(off.hfc1_b, &hd.head_fc1.bias);
    copy_r(off.hfc2_w, &hd.head_fc2.weight);
    copy_r(off.hfc2_b, &hd.head_fc2.bias);
    Ok(p)
}

#[inline]
fn linear_fwd(x: &[f32], w: &[f32], b: &[f32], rows: usize, in_d: usize, out_d: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out_d];
    for r in 0..rows {
        let xr = &x[r * in_d..(r + 1) * in_d];
        let yr = &mut y[r * out_d..(r + 1) * out_d];
        for o in 0..out_d {
            let wr = &w[o * in_d..(o + 1) * in_d];
            let mut acc = b[o];
            for i in 0..in_d {
                acc += xr[i] * wr[i];
            }
            yr[o] = acc;
        }
    }
    y
}

#[inline]
fn linear_bwd(
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    rows: usize,
    in_d: usize,
    out_d: usize,
    dw_acc: Option<&mut [f32]>,
    db_acc: Option<&mut [f32]>,
    need_dx: bool,
) -> Vec<f32> {
    let mut dx = if need_dx { vec![0.0f32; rows * in_d] } else { Vec::new() };
    let mut dw_opt = dw_acc;
    let mut db_opt = db_acc;

    for r in 0..rows {
        let xr = &x[r * in_d..(r + 1) * in_d];
        let dyr = &dy[r * out_d..(r + 1) * out_d];
        if let Some(ref mut db) = db_opt {
            for o in 0..out_d {
                db[o] += dyr[o];
            }
        }
        if let Some(ref mut dw) = dw_opt {
            for o in 0..out_d {
                let g = dyr[o];
                if g == 0.0 {
                    continue;
                }
                let dwr = &mut dw[o * in_d..(o + 1) * in_d];
                for i in 0..in_d {
                    dwr[i] += g * xr[i];
                }
            }
        }
        if need_dx {
            let dxr = &mut dx[r * in_d..(r + 1) * in_d];
            for o in 0..out_d {
                let g = dyr[o];
                if g == 0.0 {
                    continue;
                }
                let wr = &w[o * in_d..(o + 1) * in_d];
                for i in 0..in_d {
                    dxr[i] += g * wr[i];
                }
            }
        }
    }
    dx
}

struct LnTape {
    x_hat: Vec<f32>,
    inv_std: Vec<f32>,
}

fn layernorm_fwd(x: &[f32], w: &[f32], b: &[f32], rows: usize, d: usize, eps: f32) -> (Vec<f32>, LnTape) {
    let mut y = vec![0.0f32; rows * d];
    let mut x_hat = vec![0.0f32; rows * d];
    let mut inv_std = vec![0.0f32; rows];
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let mut mean = 0.0f64;
        for &v in xr {
            mean += v as f64;
        }
        mean /= d as f64;
        let mut var = 0.0f64;
        for &v in xr {
            let diff = (v as f64) - mean;
            var += diff * diff;
        }
        var /= d as f64;
        let inv = (1.0 / (var + (eps as f64)).sqrt()) as f32;
        inv_std[r] = inv;
        for c in 0..d {
            let xh = (((xr[c] as f64) - mean) as f32) * inv;
            x_hat[r * d + c] = xh;
            y[r * d + c] = xh * w[c] + b[c];
        }
    }
    (y, LnTape { x_hat, inv_std })
}

fn layernorm_bwd(
    dy: &[f32],
    w: &[f32],
    tape: &LnTape,
    rows: usize,
    d: usize,
    dw_acc: Option<&mut [f32]>,
    db_acc: Option<&mut [f32]>,
) -> Vec<f32> {
    let mut dx = vec![0.0f32; rows * d];
    let mut dw_opt = dw_acc;
    let mut db_opt = db_acc;
    let df = d as f32;

    for r in 0..rows {
        let dyr = &dy[r * d..(r + 1) * d];
        let xhr = &tape.x_hat[r * d..(r + 1) * d];
        if let Some(ref mut dw) = dw_opt {
            for c in 0..d {
                dw[c] += dyr[c] * xhr[c];
            }
        }
        if let Some(ref mut db) = db_opt {
            for c in 0..d {
                db[c] += dyr[c];
            }
        }
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        for c in 0..d {
            let dxhat = dyr[c] * w[c];
            sum1 += dxhat;
            sum2 += dxhat * xhr[c];
        }
        let inv = tape.inv_std[r] / df;
        let dxr = &mut dx[r * d..(r + 1) * d];
        for c in 0..d {
            let dxhat = dyr[c] * w[c];
            dxr[c] = inv * (df * dxhat - sum1 - xhr[c] * sum2);
        }
    }
    dx
}

/// Core reverse-mode pass for 1 molecule through a single adapter member:
/// - Computes normalized prediction `z_pred (out_dim,)`.
/// - Given upstream `dz_pred (out_dim,)`, optionally accumulates parameter gradients into `d_params`
///   AND/OR returns `(d_atom_repr (N * desc_dim), d_coords_rbf (N, 3))`!
fn member_forward_and_vjp(
    p: &[f32],
    off: &ParamOffsets,
    desc_dim: usize,
    h: usize,
    out_dim: usize,
    rbf_centers: &[f32; 16],
    atom_repr: &[f32],
    coords_a: &[[f64; 3]],
    dz_upstream_fn: impl FnOnce(&[f32]) -> (f32, Vec<f32>),
    mut d_params: Option<&mut [f32]>,
    need_input_grads: bool,
) -> (f32, Vec<f32>, Vec<f32>, Vec<[f64; 3]>) {
    let n = coords_a.len();
    let head_dim = h / 4;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut cz = 0.0f64;
    for c in coords_a {
        cx += c[0];
        cy += c[1];
        cz += c[2];
    }
    cx /= n as f64;
    cy /= n as f64;
    cz /= n as f64;
    let coords_f32: Vec<[f32; 3]> = coords_a
        .iter()
        .map(|c| [(c[0] - cx) as f32, (c[1] - cy) as f32, (c[2] - cz) as f32])
        .collect();

    // 1. h0 = norm1(in_proj(atom_repr))
    let h0_pre = linear_fwd(
        atom_repr,
        &p[off.in_proj_w.0..off.in_proj_w.1],
        &p[off.in_proj_b.0..off.in_proj_b.1],
        n,
        desc_dim,
        h,
    );
    let (h0, ln1_tape) = layernorm_fwd(
        &h0_pre,
        &p[off.norm1_w.0..off.norm1_w.1],
        &p[off.norm1_b.0..off.norm1_b.1],
        n,
        h,
        1e-5,
    );

    // 2. Pairwise RBF & 4-head attention bias
    let mut dist_ij = vec![0.0f32; n * n];
    let mut rbf_flat = vec![0.0f32; n * n * 16];
    for i in 0..n {
        for j in 0..n {
            let dx = coords_f32[i][0] - coords_f32[j][0];
            let dy = coords_f32[i][1] - coords_f32[j][1];
            let dz = coords_f32[i][2] - coords_f32[j][2];
            let d_val = (dx * dx + dy * dy + dz * dz).sqrt();
            dist_ij[i * n + j] = d_val;
            let base = (i * n + j) * 16;
            for k in 0..16 {
                let z = (d_val - rbf_centers[k]) / 0.4f32;
                rbf_flat[base + k] = (-0.5f32 * z * z).exp();
            }
        }
    }
    let attn_bias = linear_fwd(
        &rbf_flat,
        &p[off.dist_rbf_w.0..off.dist_rbf_w.1],
        &p[off.dist_rbf_b.0..off.dist_rbf_b.1],
        n * n,
        16,
        4,
    );

    // 3. 4-head distance-biased attention
    let qkv = linear_fwd(
        &h0,
        &p[off.qkv_w.0..off.qkv_w.1],
        &p[off.qkv_b.0..off.qkv_b.1],
        n,
        h,
        3 * h,
    );
    let mut probs_all = vec![0.0f32; 4 * n * n];
    let mut ctx = vec![0.0f32; n * h];

    for hd in 0..4 {
        let hd_off = hd * head_dim;
        for i in 0..n {
            let q_sl = &qkv[i * 3 * h + hd_off..i * 3 * h + hd_off + head_dim];
            let p_row = &mut probs_all[(hd * n + i) * n..(hd * n + i + 1) * n];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..n {
                let k_sl = &qkv[j * 3 * h + h + hd_off..j * 3 * h + h + hd_off + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_sl[d] * k_sl[d];
                }
                let s = dot * scale + attn_bias[(i * n + j) * 4 + hd];
                p_row[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut denom = 0.0f32;
            for j in 0..n {
                p_row[j] = (p_row[j] - max_s).exp();
                denom += p_row[j];
            }
            let inv_d = 1.0 / denom.max(1e-20);
            let ctx_sl = &mut ctx[i * h + hd_off..i * h + hd_off + head_dim];
            for j in 0..n {
                let pr = p_row[j] * inv_d;
                p_row[j] = pr;
                let v_sl = &qkv[j * 3 * h + 2 * h + hd_off..j * 3 * h + 2 * h + hd_off + head_dim];
                for d in 0..head_dim {
                    ctx_sl[d] += pr * v_sl[d];
                }
            }
        }
    }

    let attn_out = linear_fwd(
        &ctx,
        &p[off.out_proj_w.0..off.out_proj_w.1],
        &p[off.out_proj_b.0..off.out_proj_b.1],
        n,
        h,
        h,
    );
    let mut h1 = vec![0.0f32; n * h];
    for idx in 0..(n * h) {
        h1[idx] = h0[idx] + attn_out[idx];
    }

    // 4. Residual FFN: h2 = h1 + ffn1(silu(ffn0(norm2(h1))))
    let (h1_norm, ln2_tape) = layernorm_fwd(
        &h1,
        &p[off.norm2_w.0..off.norm2_w.1],
        &p[off.norm2_b.0..off.norm2_b.1],
        n,
        h,
        1e-5,
    );
    let ffn0_pre = linear_fwd(
        &h1_norm,
        &p[off.ffn0_w.0..off.ffn0_w.1],
        &p[off.ffn0_b.0..off.ffn0_b.1],
        n,
        h,
        2 * h,
    );
    let ffn0_act: Vec<f32> = ffn0_pre.iter().map(|&v| silu(v)).collect();
    let ffn1_out = linear_fwd(
        &ffn0_act,
        &p[off.ffn1_w.0..off.ffn1_w.1],
        &p[off.ffn1_b.0..off.ffn1_b.1],
        n,
        2 * h,
        h,
    );
    let mut h2 = vec![0.0f32; n * h];
    for idx in 0..(n * h) {
        h2[idx] = h1[idx] + ffn1_out[idx];
    }

    // 5. Mean + gated pooling readout
    let inv_n = 1.0f32 / (n as f32);
    let mut g_mean = vec![0.0f32; h];
    for i in 0..n {
        for c in 0..h {
            g_mean[c] += h2[i * h + c] * inv_n;
        }
    }
    let gate_logits = linear_fwd(
        &h2,
        &p[off.gate_w.0..off.gate_w.1],
        &p[off.gate_b.0..off.gate_b.1],
        n,
        h,
        1,
    );
    let max_gl = gate_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut gate_w = vec![0.0f32; n];
    let mut g_den = 0.0f32;
    for i in 0..n {
        gate_w[i] = (gate_logits[i] - max_gl).exp();
        g_den += gate_w[i];
    }
    for i in 0..n {
        gate_w[i] /= g_den.max(1e-20);
    }
    let mut u_cat = vec![0.0f32; 2 * h];
    u_cat[..h].copy_from_slice(&g_mean);
    for i in 0..n {
        let wi = gate_w[i];
        for c in 0..h {
            u_cat[h + c] += wi * h2[i * h + c];
        }
    }

    let (u_norm, hnorm_tape) = layernorm_fwd(
        &u_cat,
        &p[off.hnorm_w.0..off.hnorm_w.1],
        &p[off.hnorm_b.0..off.hnorm_b.1],
        1,
        2 * h,
        1e-5,
    );
    let fc0_pre = linear_fwd(
        &u_norm,
        &p[off.hfc0_w.0..off.hfc0_w.1],
        &p[off.hfc0_b.0..off.hfc0_b.1],
        1,
        2 * h,
        h,
    );
    let fc0_act: Vec<f32> = fc0_pre.iter().map(|&v| silu(v)).collect();
    let fc1_pre = linear_fwd(
        &fc0_act,
        &p[off.hfc1_w.0..off.hfc1_w.1],
        &p[off.hfc1_b.0..off.hfc1_b.1],
        1,
        h,
        h / 2,
    );
    let fc1_act: Vec<f32> = fc1_pre.iter().map(|&v| silu(v)).collect();
    let z_pred = linear_fwd(
        &fc1_act,
        &p[off.hfc2_w.0..off.hfc2_w.1],
        &p[off.hfc2_b.0..off.hfc2_b.1],
        1,
        h / 2,
        out_dim,
    );

    let (loss_val, dz) = dz_upstream_fn(&z_pred);
    if d_params.is_none() && !need_input_grads {
        return (loss_val, z_pred, Vec::new(), Vec::new());
    }

    // ================= Reverse-Mode VJP =================
    let dfc1_act = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.hfc2_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.hfc2_w.1 - off.hfc2_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.hfc2_b.1 - off.hfc2_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(&fc1_act, &p[off.hfc2_w.0..off.hfc2_w.1], &dz, 1, h / 2, out_dim, dw, db, true)
    };
    let dfc1_pre: Vec<f32> = dfc1_act
        .iter()
        .zip(fc1_pre.iter())
        .map(|(&g, &x)| g * silu_grad(x))
        .collect();

    let dfc0_act = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.hfc1_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.hfc1_w.1 - off.hfc1_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.hfc1_b.1 - off.hfc1_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(&fc0_act, &p[off.hfc1_w.0..off.hfc1_w.1], &dfc1_pre, 1, h, h / 2, dw, db, true)
    };
    let dfc0_pre: Vec<f32> = dfc0_act
        .iter()
        .zip(fc0_pre.iter())
        .map(|(&g, &x)| g * silu_grad(x))
        .collect();

    let du_norm = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.hfc0_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.hfc0_w.1 - off.hfc0_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.hfc0_b.1 - off.hfc0_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(&u_norm, &p[off.hfc0_w.0..off.hfc0_w.1], &dfc0_pre, 1, 2 * h, h, dw, db, true)
    };

    let du_cat = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.hnorm_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.hnorm_w.1 - off.hnorm_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.hnorm_b.1 - off.hnorm_b.0]))
            }
            None => (None, None),
        };
        layernorm_bwd(&du_norm, &p[off.hnorm_w.0..off.hnorm_w.1], &hnorm_tape, 1, 2 * h, dw, db)
    };

    let dg_mean = &du_cat[..h];
    let dg_attn = &du_cat[h..];

    let mut d_gate_w = vec![0.0f32; n];
    let mut dh2 = vec![0.0f32; n * h];
    for i in 0..n {
        let mut dot = 0.0f32;
        for c in 0..h {
            dot += dg_attn[c] * h2[i * h + c];
            dh2[i * h + c] += dg_mean[c] * inv_n + gate_w[i] * dg_attn[c];
        }
        d_gate_w[i] = dot;
    }
    let gw_dot_dgw: f32 = gate_w.iter().zip(d_gate_w.iter()).map(|(&w, &dw)| w * dw).sum();
    let d_gate_logits: Vec<f32> = (0..n).map(|i| gate_w[i] * (d_gate_w[i] - gw_dot_dgw)).collect();

    let dh2_from_gate = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.gate_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.gate_w.1 - off.gate_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.gate_b.1 - off.gate_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(&h2, &p[off.gate_w.0..off.gate_w.1], &d_gate_logits, n, h, 1, dw, db, true)
    };
    for idx in 0..(n * h) {
        dh2[idx] += dh2_from_gate[idx];
    }

    // 4-bwd. Residual FFN VJP
    let mut dh1 = dh2.clone();
    let dffn0_act = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.ffn1_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.ffn1_w.1 - off.ffn1_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.ffn1_b.1 - off.ffn1_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(&ffn0_act, &p[off.ffn1_w.0..off.ffn1_w.1], &dh2, n, 2 * h, h, dw, db, true)
    };
    let dffn0_pre: Vec<f32> = dffn0_act
        .iter()
        .zip(ffn0_pre.iter())
        .map(|(&g, &x)| g * silu_grad(x))
        .collect();
    let dh1_norm = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.ffn0_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.ffn0_w.1 - off.ffn0_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.ffn0_b.1 - off.ffn0_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(&h1_norm, &p[off.ffn0_w.0..off.ffn0_w.1], &dffn0_pre, n, h, 2 * h, dw, db, true)
    };
    let dh1_from_ln2 = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.norm2_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.norm2_w.1 - off.norm2_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.norm2_b.1 - off.norm2_b.0]))
            }
            None => (None, None),
        };
        layernorm_bwd(&dh1_norm, &p[off.norm2_w.0..off.norm2_w.1], &ln2_tape, n, h, dw, db)
    };
    for idx in 0..(n * h) {
        dh1[idx] += dh1_from_ln2[idx];
    }

    // 3-bwd. 4-head distance-biased attention VJP
    let mut dh0 = dh1.clone();
    let dctx = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.out_proj_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.out_proj_w.1 - off.out_proj_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.out_proj_b.1 - off.out_proj_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(&ctx, &p[off.out_proj_w.0..off.out_proj_w.1], &dh1, n, h, h, dw, db, true)
    };

    let mut dqkv = vec![0.0f32; n * 3 * h];
    let mut d_attn_bias = vec![0.0f32; n * n * 4];

    for hd in 0..4 {
        let hd_off = hd * head_dim;
        for i in 0..n {
            let dctx_sl = &dctx[i * h + hd_off..i * h + hd_off + head_dim];
            let p_row = &probs_all[(hd * n + i) * n..(hd * n + i + 1) * n];
            let mut dp_row = vec![0.0f32; n];
            let mut p_dot_dp = 0.0f32;
            for j in 0..n {
                let pr = p_row[j];
                let v_sl = &qkv[j * 3 * h + 2 * h + hd_off..j * 3 * h + 2 * h + hd_off + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += dctx_sl[d] * v_sl[d];
                    dqkv[j * 3 * h + 2 * h + hd_off + d] += pr * dctx_sl[d];
                }
                dp_row[j] = dot;
                p_dot_dp += pr * dot;
            }
            for j in 0..n {
                let ds = p_row[j] * (dp_row[j] - p_dot_dp);
                d_attn_bias[(i * n + j) * 4 + hd] += ds;
                let ds_scaled = ds * scale;
                // copy q_sl / k_sl values before mutating dqkv
                for d in 0..head_dim {
                    let q_val = qkv[i * 3 * h + hd_off + d];
                    let k_val = qkv[j * 3 * h + h + hd_off + d];
                    dqkv[i * 3 * h + hd_off + d] += ds_scaled * k_val;
                    dqkv[j * 3 * h + h + hd_off + d] += ds_scaled * q_val;
                }
            }
        }
    }

    let dh0_from_qkv = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.qkv_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.qkv_w.1 - off.qkv_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.qkv_b.1 - off.qkv_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(&h0, &p[off.qkv_w.0..off.qkv_w.1], &dqkv, n, h, 3 * h, dw, db, true)
    };
    for idx in 0..(n * h) {
        dh0[idx] += dh0_from_qkv[idx];
    }

    // 2-bwd. Distance RBF attention bias VJP -> optional 3D coordinate gradient!
    let d_rbf = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.dist_rbf_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.dist_rbf_w.1 - off.dist_rbf_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.dist_rbf_b.1 - off.dist_rbf_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(
            &rbf_flat,
            &p[off.dist_rbf_w.0..off.dist_rbf_w.1],
            &d_attn_bias,
            n * n,
            16,
            4,
            dw,
            db,
            need_input_grads,
        )
    };

    let mut d_coords = if need_input_grads {
        vec![[0.0f64; 3]; n]
    } else {
        Vec::new()
    };
    if need_input_grads {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d_val = dist_ij[i * n + j];
                if d_val <= 1e-8 {
                    continue;
                }
                let base = (i * n + j) * 16;
                let mut d_dist = 0.0f32;
                for k in 0..16 {
                    // d phi / d dist = -((dist - mu_k) / 0.16) * phi
                    d_dist += d_rbf[base + k] * (-((d_val - rbf_centers[k]) / 0.16f32) * rbf_flat[base + k]);
                }
                let coef = (d_dist / d_val) as f64;
                let rx = (coords_f32[i][0] - coords_f32[j][0]) as f64;
                let ry = (coords_f32[i][1] - coords_f32[j][1]) as f64;
                let rz = (coords_f32[i][2] - coords_f32[j][2]) as f64;
                d_coords[i][0] += coef * rx;
                d_coords[i][1] += coef * ry;
                d_coords[i][2] += coef * rz;
                d_coords[j][0] -= coef * rx;
                d_coords[j][1] -= coef * ry;
                d_coords[j][2] -= coef * rz;
            }
        }
    }

    // 1-bwd. norm1 + in_proj VJP -> optional `d_atom_repr`!
    let dh0_pre = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.norm1_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.norm1_w.1 - off.norm1_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.norm1_b.1 - off.norm1_b.0]))
            }
            None => (None, None),
        };
        layernorm_bwd(&dh0, &p[off.norm1_w.0..off.norm1_w.1], &ln1_tape, n, h, dw, db)
    };

    let d_atom_repr = {
        let (dw, db) = match d_params.as_deref_mut() {
            Some(dp) => {
                let (_, rest) = dp.split_at_mut(off.in_proj_w.0);
                let (w_sl, b_sl) = rest.split_at_mut(off.in_proj_w.1 - off.in_proj_w.0);
                (Some(w_sl), Some(&mut b_sl[..off.in_proj_b.1 - off.in_proj_b.0]))
            }
            None => (None, None),
        };
        linear_bwd(
            atom_repr,
            &p[off.in_proj_w.0..off.in_proj_w.1],
            &dh0_pre,
            n,
            desc_dim,
            h,
            dw,
            db,
            need_input_grads,
        )
    };

    (loss_val, z_pred, d_atom_repr, d_coords)
}

/// Smooth-L1 / Huber loss with `beta = 0.1` (`dpa4_oracle.py:397`).
#[inline]
fn huber_loss_and_grad(z_pred: &[f32], z_true: &[f32]) -> (f32, Vec<f32>) {
    let k = z_pred.len().max(1);
    let inv_k = 1.0f32 / (k as f32);
    let mut loss = 0.0f32;
    let mut dz = vec![0.0f32; z_pred.len()];
    for i in 0..z_pred.len() {
        let delta = z_pred[i] - z_true[i];
        let ad = delta.abs();
        if ad < 0.1 {
            loss += (delta * delta / 0.2) * inv_k;
            dz[i] = (delta / 0.1) * inv_k;
        } else {
            loss += (ad - 0.05) * inv_k;
            dz[i] = delta.signum() * inv_k;
        }
    }
    (loss, dz)
}

// ============================================================================
// Trainable 5-Member Bootstrap Property Oracle
// ============================================================================

#[derive(Debug, Clone)]
pub struct TrainableBootstrapOracle {
    pub backbone_family: String,
    pub inner: BootstrapPropertyOracle,
    offsets: ParamOffsets,
    member_params: Vec<Vec<f32>>,
}

impl TrainableBootstrapOracle {
    pub fn from_mlip_oracle(backbone_family: &str, inner: BootstrapPropertyOracle) -> Result<Self, String> {
        let m0 = inner.members.first().ok_or("Empty BootstrapPropertyOracle")?;
        let offsets = ParamOffsets::new(m0.desc_dim, m0.hidden_dim, m0.output_dim);
        let mut member_params = Vec::with_capacity(inner.members.len());
        for m in &inner.members {
            member_params.push(unpack_member_from_mlip(m, &offsets)?);
        }
        Ok(Self {
            backbone_family: backbone_family.to_string(),
            inner,
            offsets,
            member_params,
        })
    }

    pub fn to_serialized_bundle(&self) -> SerializedOracleBundle {
        let m0 = &self.inner.members[0];
        SerializedOracleBundle {
            backbone_family: self.backbone_family.clone(),
            property_names: self.inner.target_names.clone(),
            property_units: self.inner.target_units.clone(),
            scalers: self.inner.scalers.clone(),
            desc_dim: m0.desc_dim,
            hidden_dim: m0.hidden_dim,
            output_dim: m0.output_dim,
            rbf_centers: m0.rbf_centers,
            members: self
                .member_params
                .iter()
                .map(|p| SerializedMemberWeights { params: p.clone() })
                .collect(),
        }
    }

    pub fn from_serialized_bundle(b: SerializedOracleBundle) -> Result<Self, String> {
        let offsets = ParamOffsets::new(b.desc_dim, b.hidden_dim, b.output_dim);
        let mut members = Vec::with_capacity(b.members.len());
        let mut member_params = Vec::with_capacity(b.members.len());
        for mw in b.members {
            if mw.params.len() != offsets.total {
                return Err(format!(
                    "Serialized oracle param length mismatch: expected {}, got {}",
                    offsets.total,
                    mw.params.len()
                ));
            }
            members.push(pack_member_to_mlip(
                &mw.params,
                &offsets,
                b.desc_dim,
                b.hidden_dim,
                b.output_dim,
                b.rbf_centers,
            ));
            member_params.push(mw.params);
        }
        Ok(Self {
            backbone_family: b.backbone_family,
            inner: BootstrapPropertyOracle {
                target_names: b.property_names,
                target_units: b.property_units,
                scalers: b.scalers,
                members,
            },
            offsets,
            member_params,
        })
    }

    pub fn predict(&self, atom_repr: &[f32], coords_a: &[[f64; 3]]) -> Result<PropertyPrediction, String> {
        self.inner.predict(atom_repr, coords_a)
    }

    /// Compute ensemble prediction `(mean, std)` on `channel_idx` AND exact analytical upstream
    /// `(d_mean / d_atom_repr, d_mean / d_coords_rbf)` so `hyff-rs` / `dpa4-rs` can complete `grad_coords`!
    pub fn predict_and_input_vjp(
        &self,
        atom_repr: &[f32],
        coords_a: &[[f64; 3]],
        channel_idx: usize,
    ) -> Result<(f64, f64, Vec<f32>, Vec<[f64; 3]>), String> {
        let m0 = &self.inner.members[0];
        let k_out = m0.output_dim;
        if channel_idx >= k_out {
            return Err(format!("Oracle channel_idx {channel_idx} >= {k_out}"));
        }
        let n = coords_a.len();
        let m_cnt = self.member_params.len();
        let inv_m = 1.0f32 / (m_cnt as f32);

        let scale_c = match self.inner.scalers[channel_idx] {
            TargetScaler::Identity => 1.0f32,
            TargetScaler::Affine { scale, .. } => scale,
            TargetScaler::YeoJohnson { scale, .. } => scale,
        };

        let mut d_repr_total = vec![0.0f32; n * m0.desc_dim];
        let mut d_coords_total = vec![[0.0f64; 3]; n];
        let mut preds_c = Vec::with_capacity(m_cnt);

        for p in &self.member_params {
            let (_, z_pred, d_repr, d_c) = member_forward_and_vjp(
                p,
                &self.offsets,
                m0.desc_dim,
                m0.hidden_dim,
                k_out,
                &m0.rbf_centers,
                atom_repr,
                coords_a,
                |_z| {
                    let mut dz = vec![0.0f32; k_out];
                    dz[channel_idx] = inv_m * scale_c;
                    (0.0, dz)
                },
                None,
                true,
            );
            let y_val = self.inner.scalers[channel_idx].inverse(z_pred[channel_idx]) as f64;
            preds_c.push(y_val);
            for idx in 0..d_repr_total.len() {
                d_repr_total[idx] += d_repr[idx];
            }
            for i in 0..n {
                d_coords_total[i][0] += d_c[i][0];
                d_coords_total[i][1] += d_c[i][1];
                d_coords_total[i][2] += d_c[i][2];
            }
        }

        let mean = preds_c.iter().sum::<f64>() / (m_cnt as f64);
        let var = preds_c.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (m_cnt as f64);
        Ok((mean, var.sqrt(), d_repr_total, d_coords_total))
    }

    /// Train a 5-member Bootstrap 3D Property Oracle from pre-cached frozen MLIP descriptors!
    pub fn fit_from_cached_descriptors<F>(
        backbone_family: &str,
        property_names: Vec<String>,
        property_units: Vec<String>,
        desc_dim: usize,
        samples: &[CachedDescriptorSample],
        opts: &OracleTrainOptions,
        mut on_progress: F,
    ) -> Result<(Self, OracleTrainReport), String>
    where
        F: FnMut(usize, usize, usize, f32),
    {
        if samples.len() < 4 {
            return Err(format!("Oracle training requires >= 4 valid samples, got {}", samples.len()));
        }
        let k_out = property_names.len();
        if k_out == 0 {
            return Err("property_names must be non-empty".into());
        }
        let h = (opts.hidden_dim / 4).max(4) * 4;

        // 1. Scaffold-grouped train/validation split (zero scaffold-identity leakage!)
        let mut scaffold_groups: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (idx, s) in samples.iter().enumerate() {
            scaffold_groups.entry(s.scaffold_sig.clone()).or_default().push(idx);
        }
        let mut group_list: Vec<Vec<usize>> = scaffold_groups.into_values().collect();
        let mut split_rng = DetRng::new(opts.seed ^ 0xA11C_E555);
        for i in (1..group_list.len()).rev() {
            let j = (split_rng.next_u64() as usize) % (i + 1);
            group_list.swap(i, j);
        }

        let target_val_n = ((samples.len() as f32) * opts.val_fraction.clamp(0.05, 0.40)).round() as usize;
        let target_val_n = target_val_n.clamp(1, samples.len() - 2);
        let mut val_indices = Vec::new();
        let mut train_indices = Vec::new();
        for grp in group_list {
            if val_indices.len() < target_val_n && (samples.len() - (val_indices.len() + grp.len())) >= 2 {
                val_indices.extend(grp);
            } else {
                train_indices.extend(grp);
            }
        }
        if val_indices.is_empty() {
            val_indices.push(train_indices.pop().unwrap());
        }

        // 2. Fit affine target standardization on `train_indices` ONLY
        let mut means = vec![0.0f32; k_out];
        let mut scales = vec![1.0f32; k_out];
        let mut scalers = Vec::with_capacity(k_out);
        for c in 0..k_out {
            let vals: Vec<f64> = train_indices.iter().map(|&idx| samples[idx].targets[c] as f64).collect();
            let mean = vals.iter().sum::<f64>() / (vals.len() as f64);
            let var = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (vals.len() as f64);
            let sc = var.sqrt();
            let sc_safe = if sc > 1e-6 { sc as f32 } else { 1.0f32 };
            means[c] = mean as f32;
            scales[c] = sc_safe;
            scalers.push(TargetScaler::Affine {
                mean: means[c],
                scale: scales[c],
            });
        }

        let norm_targets: Vec<Vec<f32>> = samples
            .iter()
            .map(|s| {
                (0..k_out)
                    .map(|c| (s.targets[c] - means[c]) / scales[c])
                    .collect()
            })
            .collect();

        // 3. Train `n_members` independent Bootstrap adapters via `AdamW`
        let offsets = ParamOffsets::new(desc_dim, h, k_out);
        let rbf_centers = default_rbf_centers();
        let n_members = opts.n_members.max(1);
        let mut member_params = Vec::with_capacity(n_members);
        let mut members_mlip = Vec::with_capacity(n_members);
        let mut member_val_norm_mae = Vec::with_capacity(n_members);

        for m_idx in 0..n_members {
            let mut rng = DetRng::new(opts.seed.wrapping_add(m_idx as u64));
            let mut params = init_member_params(&offsets, desc_dim, h, k_out, &mut rng);
            let mut best_params = params.clone();
            let mut best_val_mae = f32::INFINITY;

            // Bootstrap resample with replacement over `train_indices`
            let n_tr = train_indices.len();
            let boot_base: Vec<usize> = (0..n_tr)
                .map(|_| train_indices[(rng.next_u64() as usize) % n_tr])
                .collect();

            let mut m_adam = vec![0.0f32; offsets.total];
            let mut v_adam = vec![0.0f32; offsets.total];
            let mut step_t = 0usize;
            let mut grad = vec![0.0f32; offsets.total];

            for ep in 0..opts.epochs.max(1) {
                let mut order = boot_base.clone();
                for i in (1..order.len()).rev() {
                    let j = (rng.next_u64() as usize) % (i + 1);
                    order.swap(i, j);
                }

                for &s_idx in &order {
                    grad.fill(0.0);
                    let samp = &samples[s_idx];
                    let z_t = &norm_targets[s_idx];
                    let (_loss, _zp, _, _) = member_forward_and_vjp(
                        &params,
                        &offsets,
                        desc_dim,
                        h,
                        k_out,
                        &rbf_centers,
                        &samp.atom_repr,
                        &samp.coords_a,
                        |zp| huber_loss_and_grad(zp, z_t),
                        Some(&mut grad),
                        false,
                    );

                    // Global L2 gradient clipping (`max_grad_norm = 5.0`)
                    let gnorm_sq: f32 = grad.iter().map(|&g| g * g).sum();
                    let gnorm = gnorm_sq.sqrt();
                    let clip_coef = if gnorm > opts.max_grad_norm {
                        opts.max_grad_norm / (gnorm + 1e-8)
                    } else {
                        1.0
                    };

                    // AdamW update (`betas = (0.9, 0.999), eps = 1e-8`)
                    step_t += 1;
                    let bc1 = 1.0 - 0.9f32.powi(step_t as i32);
                    let bc2 = 1.0 - 0.999f32.powi(step_t as i32);
                    for i in 0..offsets.total {
                        let g = grad[i] * clip_coef;
                        if !g.is_finite() {
                            continue;
                        }
                        if offsets.decay_mask[i] {
                            params[i] *= 1.0 - opts.lr * opts.weight_decay;
                        }
                        m_adam[i] = 0.9 * m_adam[i] + 0.1 * g;
                        v_adam[i] = 0.999 * v_adam[i] + 0.001 * g * g;
                        let m_hat = m_adam[i] / bc1;
                        let v_hat = v_adam[i] / bc2;
                        params[i] -= opts.lr * m_hat / (v_hat.sqrt() + 1e-8);
                    }
                }

                // Validation normalized MAE check
                let mut val_err = 0.0f32;
                for &v_idx in &val_indices {
                    let samp = &samples[v_idx];
                    let z_t = &norm_targets[v_idx];
                    let (_, zp, _, _) = member_forward_and_vjp(
                        &params,
                        &offsets,
                        desc_dim,
                        h,
                        k_out,
                        &rbf_centers,
                        &samp.atom_repr,
                        &samp.coords_a,
                        |_| (0.0, Vec::new()),
                        None,
                        false,
                    );
                    let mut row_mae = 0.0f32;
                    for c in 0..k_out {
                        row_mae += (zp[c] - z_t[c]).abs();
                    }
                    val_err += row_mae / (k_out as f32);
                }
                val_err /= val_indices.len() as f32;
                if val_err < best_val_mae {
                    best_val_mae = val_err;
                    best_params.copy_from_slice(&params);
                }
                on_progress(m_idx + 1, n_members, ep + 1, best_val_mae);
            }

            members_mlip.push(pack_member_to_mlip(
                &best_params,
                &offsets,
                desc_dim,
                h,
                k_out,
                rbf_centers,
            ));
            member_params.push(best_params);
            member_val_norm_mae.push(best_val_mae);
        }

        let oracle = Self {
            backbone_family: backbone_family.to_string(),
            inner: BootstrapPropertyOracle {
                target_names: property_names.clone(),
                target_units: property_units.clone(),
                scalers,
                members: members_mlip,
            },
            offsets,
            member_params,
        };

        // Compute ensemble validation MAE in physical units
        let mut ensemble_val_mae = vec![0.0f32; k_out];
        for &v_idx in &val_indices {
            let samp = &samples[v_idx];
            let preds = oracle.predict(&samp.atom_repr, &samp.coords_a)?;
            for c in 0..k_out {
                ensemble_val_mae[c] += (preds.mean[c] - samp.targets[c]).abs();
            }
        }
        for c in 0..k_out {
            ensemble_val_mae[c] /= val_indices.len() as f32;
        }

        let report = OracleTrainReport {
            property_names,
            property_units,
            desc_dim,
            hidden_dim: h,
            n_members,
            n_train: train_indices.len(),
            n_val: val_indices.len(),
            member_val_norm_mae,
            ensemble_val_mae_per_prop: ensemble_val_mae,
            elapsed_ms: 0.0,
        };
        Ok((oracle, report))
    }
}

/// Build a `CachedDescriptorSample` scaffold key from a `MolRecord` / `HeavyMolGraph`.
pub fn scaffold_key_for_graph(g: &HeavyMolGraph) -> String {
    murcko_scaffold_signature(g)
}
