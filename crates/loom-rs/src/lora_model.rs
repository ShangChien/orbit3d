//! Differentiable `Uni-3DAR` `QLoRA / FP16` + `LoomConditioner` + `HeavyAtomValenceHead` + `XYZ` Head
//! Training & Rollout Engine (`lora_model.rs`), supporting both `wgpu` Compute Shaders and SIMD CPU!

use crate::chem_rules::MolRecord;
use crate::octree_encode::{AtomDecisionTarget, EncodedOctreeSeq, SeqToken, TreeDecisionTarget, SUBCELL_ORDERS};
use bytemuck::{Pod, Zeroable};
use half::f16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uni3dar_rs::config::ConditionerConfig;
use uni3dar_rs::cpu_backend::{apply_rope_in_place, compute_rope_3d, embed_octree_token};
use uni3dar_rs::format::{erf_approx, gelu_exact, silu, ModelBundle};
use uni3dar_rs::sampler::DetRng;
use uni3dar_rs::valence::{
    build_raw_point_cloud_graph, elem_id_to_symbol, finalize_sampled_molecule, max_coord_element,
    rcov_angstrom, reconcile_point_cloud, symbol_to_elem_id, SampledUni3darMolecule,
};
use wgpu::util::DeviceExt;

pub const LOOM_LORA_MAGIC: &[u8; 8] = b"LOOMLORA";

#[inline]
pub fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

#[inline]
pub fn gelu_grad(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2));
    let pdf = (1.0 / (2.0 * std::f32::consts::PI).sqrt()) * (-0.5 * x * x).exp();
    cdf + x * pdf
}

#[inline]
pub fn apply_rope_bwd_in_place(dvec: &mut [f32], num_heads: usize, head_dim: usize, cos_v: &[f32], sin_v: &[f32]) {
    let half = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for p in 0..half {
            let dy0 = dvec[base + 2 * p];
            let dy1 = dvec[base + 2 * p + 1];
            let c = cos_v[p];
            let s = sin_v[p];
            dvec[base + 2 * p] = dy0 * c + dy1 * s;
            dvec[base + 2 * p + 1] = -dy0 * s + dy1 * c;
        }
    }
}

// ============================================================================
// LoRA Module Pair & Gradient Accumulator
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraPair {
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub scaling: f32,
    pub trainable: bool,
    /// Row-major `(rank, in_dim)`
    pub a: Vec<f32>,
    /// Row-major `(out_dim, rank)` (zero-initialized so `LoRA(x) == 0` at step 0!)
    pub b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LoraGrad {
    pub da: Vec<f32>,
    pub db: Vec<f32>,
}

impl LoraPair {
    pub fn new_zero_init(rank: usize, alpha: f32, in_dim: usize, out_dim: usize, rng: &mut DetRng) -> Self {
        let r = rank.max(1);
        let bound = (1.0 / (in_dim.max(1) as f32)).sqrt();
        let mut a = vec![0.0f32; r * in_dim];
        for v in &mut a {
            *v = (rng.next_f32() * 2.0 - 1.0) * bound;
        }
        let b = vec![0.0f32; out_dim * r];
        Self {
            rank: r,
            in_dim,
            out_dim,
            scaling: alpha / (r as f32),
            trainable: true,
            a,
            b,
        }
    }

    pub fn zero_grad(&self) -> LoraGrad {
        LoraGrad {
            da: vec![0.0f32; self.a.len()],
            db: vec![0.0f32; self.b.len()],
        }
    }

    /// Compute `delta = scale * ((X @ A^T) @ B^T)` of shape `(rows, out_dim)`.
    pub fn forward_delta(&self, x: &[f32], rows: usize, scale_mult: f32) -> Vec<f32> {
        let sc = self.scaling * scale_mult;
        if sc == 0.0 || rows == 0 {
            return vec![0.0f32; rows * self.out_dim];
        }
        let r = self.rank;
        let in_d = self.in_dim;
        let out_d = self.out_dim;
        let mut xa = vec![0.0f32; rows * r];
        for m in 0..rows {
            let xm = &x[m * in_d..(m + 1) * in_d];
            for k in 0..r {
                let ak = &self.a[k * in_d..(k + 1) * in_d];
                let mut dot = 0.0f32;
                for i in 0..in_d {
                    dot += xm[i] * ak[i];
                }
                xa[m * r + k] = dot;
            }
        }
        let mut out = vec![0.0f32; rows * out_d];
        for m in 0..rows {
            let xam = &xa[m * r..(m + 1) * r];
            let om = &mut out[m * out_d..(m + 1) * out_d];
            for o in 0..out_d {
                let bo = &self.b[o * r..(o + 1) * r];
                let mut dot = 0.0f32;
                for k in 0..r {
                    dot += xam[k] * bo[k];
                }
                om[o] = dot * sc;
            }
        }
        out
    }

    /// Accumulate `dx += scale * ((dY @ B) @ A)` AND (if `grad.is_some() && self.trainable`) accumulate `da, db`!
    pub fn backward_accum(
        &self,
        x: &[f32],
        dy: &[f32],
        rows: usize,
        scale_mult: f32,
        dx_accum: &mut [f32],
        grad: Option<&mut LoraGrad>,
    ) {
        let sc = self.scaling * scale_mult;
        if rows == 0 {
            return;
        }
        let r = self.rank;
        let in_d = self.in_dim;
        let out_d = self.out_dim;

        // `dy_b = dY @ B` of shape `(rows, r)`
        let mut dy_b = vec![0.0f32; rows * r];
        for m in 0..rows {
            let dym = &dy[m * out_d..(m + 1) * out_d];
            let dybm = &mut dy_b[m * r..(m + 1) * r];
            for o in 0..out_d {
                let g = dym[o];
                if g == 0.0 {
                    continue;
                }
                let bo = &self.b[o * r..(o + 1) * r];
                for k in 0..r {
                    dybm[k] += g * bo[k];
                }
            }
        }

        if sc != 0.0 {
            for m in 0..rows {
                let dybm = &dy_b[m * r..(m + 1) * r];
                let dxm = &mut dx_accum[m * in_d..(m + 1) * in_d];
                for k in 0..r {
                    let g = dybm[k] * sc;
                    if g == 0.0 {
                        continue;
                    }
                    let ak = &self.a[k * in_d..(k + 1) * in_d];
                    for i in 0..in_d {
                        dxm[i] += g * ak[i];
                    }
                }
            }
        }

        if let Some(gr) = grad {
            if self.trainable && sc != 0.0 {
                // Recompute `xa = X @ A^T` `(rows, r)` for `db += sc * dY^T @ xa`
                let mut xa = vec![0.0f32; rows * r];
                for m in 0..rows {
                    let xm = &x[m * in_d..(m + 1) * in_d];
                    for k in 0..r {
                        let ak = &self.a[k * in_d..(k + 1) * in_d];
                        let mut dot = 0.0f32;
                        for i in 0..in_d {
                            dot += xm[i] * ak[i];
                        }
                        xa[m * r + k] = dot;
                    }
                }
                for m in 0..rows {
                    let dym = &dy[m * out_d..(m + 1) * out_d];
                    let xam = &xa[m * r..(m + 1) * r];
                    for o in 0..out_d {
                        let g = dym[o] * sc;
                        if g == 0.0 {
                            continue;
                        }
                        let dbo = &mut gr.db[o * r..(o + 1) * r];
                        for k in 0..r {
                            dbo[k] += g * xam[k];
                        }
                    }
                    let xm = &x[m * in_d..(m + 1) * in_d];
                    let dybm = &dy_b[m * r..(m + 1) * r];
                    for k in 0..r {
                        let g = dybm[k] * sc;
                        if g == 0.0 {
                            continue;
                        }
                        let dak = &mut gr.da[k * in_d..(k + 1) * in_d];
                        for i in 0..in_d {
                            dak[i] += g * xm[i];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Trainable `LoomConditioner` + `HeavyAtomValenceHead`
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLinear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct DenseLinearGrad {
    pub dw: Vec<f32>,
    pub db: Option<Vec<f32>>,
}

impl DenseLinear {
    pub fn zero_init(in_dim: usize, out_dim: usize, with_bias: bool) -> Self {
        Self {
            in_dim,
            out_dim,
            w: vec![0.0; out_dim * in_dim],
            b: if with_bias { Some(vec![0.0; out_dim]) } else { None },
        }
    }

    pub fn kaiming_init(in_dim: usize, out_dim: usize, with_bias: bool, rng: &mut DetRng) -> Self {
        let bound = (1.0 / (in_dim.max(1) as f32)).sqrt();
        let mut w = vec![0.0f32; out_dim * in_dim];
        for v in &mut w {
            *v = (rng.next_f32() * 2.0 - 1.0) * bound;
        }
        Self {
            in_dim,
            out_dim,
            w,
            b: if with_bias { Some(vec![0.0; out_dim]) } else { None },
        }
    }

    pub fn from_bundle(bundle: &ModelBundle, prefix: &str) -> Result<Self, String> {
        let (out_d, in_d, packed, bias) = bundle.dequantize_linear_f16_u32(prefix)?;
        let pairs = (in_d + 1) / 2;
        let mut w = vec![0.0f32; out_d * in_d];
        for r in 0..out_d {
            for c in 0..in_d {
                let word = packed[r * pairs + (c >> 1)];
                let bits = if (c & 1) == 0 {
                    (word & 0xFFFF) as u16
                } else {
                    (word >> 16) as u16
                };
                w[r * in_d + c] = f16::from_bits(bits).to_f32();
            }
        }
        Ok(Self {
            in_dim: in_d,
            out_dim: out_d,
            w,
            b: bias,
        })
    }

    pub fn zero_grad(&self) -> DenseLinearGrad {
        DenseLinearGrad {
            dw: vec![0.0; self.w.len()],
            db: self.b.as_ref().map(|b| vec![0.0; b.len()]),
        }
    }

    pub fn fwd(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * self.out_dim];
        let chunks4 = self.in_dim / 4;
        for r in 0..rows {
            let xr = &x[r * self.in_dim..(r + 1) * self.in_dim];
            let yr = &mut y[r * self.out_dim..(r + 1) * self.out_dim];
            for o in 0..self.out_dim {
                let wr = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
                let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for k in 0..chunks4 {
                    let b = 4 * k;
                    a0 += xr[b] * wr[b];
                    a1 += xr[b + 1] * wr[b + 1];
                    a2 += xr[b + 2] * wr[b + 2];
                    a3 += xr[b + 3] * wr[b + 3];
                }
                let mut acc = self.b.as_ref().map_or(0.0, |bv| bv[o]) + ((a0 + a1) + (a2 + a3));
                for i in (4 * chunks4)..self.in_dim {
                    acc += xr[i] * wr[i];
                }
                yr[o] = acc;
            }
        }
        y
    }

    pub fn bwd(&self, x: &[f32], dy: &[f32], rows: usize, grad: Option<&mut DenseLinearGrad>) -> Vec<f32> {
        let mut dx = vec![0.0f32; rows * self.in_dim];
        let mut g_opt = grad;
        for r in 0..rows {
            let xr = &x[r * self.in_dim..(r + 1) * self.in_dim];
            let dyr = &dy[r * self.out_dim..(r + 1) * self.out_dim];
            let dxr = &mut dx[r * self.in_dim..(r + 1) * self.in_dim];
            for o in 0..self.out_dim {
                let g = dyr[o];
                if g == 0.0 {
                    continue;
                }
                let wr = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
                for i in 0..self.in_dim {
                    dxr[i] += g * wr[i];
                }
                if let Some(ref mut gr) = g_opt {
                    if let Some(ref mut db) = gr.db {
                        db[o] += g;
                    }
                    let dwr = &mut gr.dw[o * self.in_dim..(o + 1) * self.in_dim];
                    for i in 0..self.in_dim {
                        dwr[i] += g * xr[i];
                    }
                }
            }
        }
        dx
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayerNorm {
    pub dim: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct DenseLayerNormGrad {
    pub dw: Vec<f32>,
    pub db: Vec<f32>,
}

impl DenseLayerNorm {
    pub fn ones(dim: usize) -> Self {
        Self {
            dim,
            w: vec![1.0; dim],
            b: vec![0.0; dim],
        }
    }

    pub fn from_bundle(bundle: &ModelBundle, prefix: &str) -> Result<Self, String> {
        let w = bundle.f32_slice(&format!("{prefix}.weight"))?.to_vec();
        let b = bundle.f32_slice(&format!("{prefix}.bias"))?.to_vec();
        Ok(Self {
            dim: w.len(),
            w,
            b,
        })
    }

    pub fn zero_grad(&self) -> DenseLayerNormGrad {
        DenseLayerNormGrad {
            dw: vec![0.0; self.dim],
            db: vec![0.0; self.dim],
        }
    }

    pub fn fwd(&self, x: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.dim;
        let mut y = vec![0.0f32; rows * d];
        let mut x_hat = vec![0.0f32; rows * d];
        let mut inv_std = vec![0.0f32; rows];
        for r in 0..rows {
            let xr = &x[r * d..(r + 1) * d];
            let mean = xr.iter().sum::<f32>() / (d as f32);
            let var = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (d as f32);
            let inv = 1.0 / (var + 1e-5).sqrt();
            inv_std[r] = inv;
            for c in 0..d {
                let xh = (xr[c] - mean) * inv;
                x_hat[r * d + c] = xh;
                y[r * d + c] = xh * self.w[c] + self.b[c];
            }
        }
        (y, x_hat, inv_std)
    }

    pub fn bwd(
        &self,
        dy: &[f32],
        x_hat: &[f32],
        inv_std: &[f32],
        rows: usize,
        mut grad: Option<&mut DenseLayerNormGrad>,
    ) -> Vec<f32> {
        let d = self.dim;
        let df = d as f32;
        let mut dx = vec![0.0f32; rows * d];
        for r in 0..rows {
            let dyr = &dy[r * d..(r + 1) * d];
            let xhr = &x_hat[r * d..(r + 1) * d];
            if let Some(ref mut gr) = grad {
                for c in 0..d {
                    gr.dw[c] += dyr[c] * xhr[c];
                    gr.db[c] += dyr[c];
                }
            }
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            for c in 0..d {
                let dxh = dyr[c] * self.w[c];
                s1 += dxh;
                s2 += dxh * xhr[c];
            }
            let inv = inv_std[r] / df;
            for c in 0..d {
                let dxh = dyr[c] * self.w[c];
                dx[r * d + c] = inv * (df * dxh - s1 - xhr[c] * s2);
            }
        }
        dx
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropEncoderMlp {
    pub fc0: DenseLinear,
    pub ln0: DenseLayerNorm,
    pub fc1: DenseLinear,
    pub ln1: DenseLayerNorm,
}

#[derive(Debug, Clone)]
pub struct PropEncoderGrad {
    pub fc0: DenseLinearGrad,
    pub ln0: DenseLayerNormGrad,
    pub fc1: DenseLinearGrad,
    pub ln1: DenseLayerNormGrad,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableConditioner {
    pub cfg: ConditionerConfig,
    pub target_cols: Vec<String>,
    pub target_units: Vec<String>,
    pub target_mean: Vec<f32>,
    pub target_scale: Vec<f32>,
    pub rbf_centers: Vec<f32>,
    pub fourier_freqs: Vec<f32>,
    pub prop_id_emb: Vec<f32>,
    pub inactive_prop_emb: Vec<f32>,
    pub prop_encoders: Vec<PropEncoderMlp>,
    pub global_fc0: DenseLinear,
    pub global_ln0: DenseLayerNorm,
    pub global_fc1: DenseLinear,
    pub global_ln1: DenseLayerNorm,
    pub root_proj: DenseLinear,
    pub root_gate: f32,
    pub layer_mod: Vec<DenseLinear>,
    pub out_mod: DenseLinear,
}

#[derive(Debug, Clone)]
pub struct ConditionerGrad {
    pub d_prop_id_emb: Vec<f32>,
    pub d_inactive_prop_emb: Vec<f32>,
    pub prop_encoders: Vec<PropEncoderGrad>,
    pub global_fc0: DenseLinearGrad,
    pub global_ln0: DenseLayerNormGrad,
    pub global_fc1: DenseLinearGrad,
    pub global_ln1: DenseLayerNormGrad,
    pub root_proj: DenseLinearGrad,
    pub d_root_gate: f32,
    pub layer_mod: Vec<DenseLinearGrad>,
    pub out_mod: DenseLinearGrad,
}

impl TrainableConditioner {
    pub fn from_bundle_or_fresh(
        bundle: &ModelBundle,
        custom_targets: Option<(Vec<String>, Vec<String>, Vec<f32>, Vec<f32>)>,
        rng: &mut DetRng,
    ) -> Result<Self, String> {
        let d = bundle.header.backbone.emb_dim;
        let n_layers = bundle.header.backbone.num_layers;

        // Reuse existing bundle conditioner iff `custom_targets` is None (or matches `target_cols`) and `conditioner` exists
        let can_reuse = match (&bundle.header.conditioner, &custom_targets) {
            (Some(cc), None) => cc.num_props > 0,
            (Some(_), Some((cols, _, _, _))) => cols == &bundle.header.target_cols,
            _ => false,
        };

        if can_reuse {
            let cc = bundle.header.conditioner.clone().unwrap();
            let mut prop_encoders = Vec::with_capacity(cc.num_props);
            for k in 0..cc.num_props {
                prop_encoders.push(PropEncoderMlp {
                    fc0: DenseLinear::from_bundle(bundle, &format!("cond.prop_enc.{k}.0"))?,
                    ln0: DenseLayerNorm::from_bundle(bundle, &format!("cond.prop_enc.{k}.1"))?,
                    fc1: DenseLinear::from_bundle(bundle, &format!("cond.prop_enc.{k}.3"))?,
                    ln1: DenseLayerNorm::from_bundle(bundle, &format!("cond.prop_enc.{k}.4"))?,
                });
            }
            let mut layer_mod = Vec::with_capacity(n_layers);
            for l in 0..n_layers {
                layer_mod.push(DenseLinear::from_bundle(bundle, &format!("cond.layer_mod.{l}"))?);
            }
            let (t_cols, t_units, t_mean, t_scale) = if let Some(ct) = custom_targets {
                ct
            } else {
                (
                    bundle.header.target_cols.clone(),
                    bundle.header.target_units.clone(),
                    bundle.header.target_mean.clone(),
                    bundle.header.target_scale.clone(),
                )
            };
            return Ok(Self {
                rbf_centers: bundle.f32_slice("cond.rbf_centers")?.to_vec(),
                fourier_freqs: bundle.f32_slice("cond.fourier_freqs")?.to_vec(),
                prop_id_emb: bundle.f32_slice("cond.prop_id_emb")?.to_vec(),
                inactive_prop_emb: bundle.f32_slice("cond.inactive_prop_emb")?.to_vec(),
                prop_encoders,
                global_fc0: DenseLinear::from_bundle(bundle, "cond.global_mlp.0")?,
                global_ln0: DenseLayerNorm::from_bundle(bundle, "cond.global_mlp.1")?,
                global_fc1: DenseLinear::from_bundle(bundle, "cond.global_mlp.3")?,
                global_ln1: DenseLayerNorm::from_bundle(bundle, "cond.global_mlp.4")?,
                root_proj: DenseLinear::from_bundle(bundle, "cond.root_proj")?,
                root_gate: cc.root_gate,
                layer_mod,
                out_mod: DenseLinear::from_bundle(bundle, "cond.out_mod")?,
                cfg: cc,
                target_cols: t_cols,
                target_units: t_units,
                target_mean: t_mean,
                target_scale: t_scale,
            });
        }

        // Fresh zero-perturbation `LoomConditioner` (`modulation_cap = 4.0`, `num_rbf = 32`, `num_freqs = 16`)
        let (t_cols, t_units, t_mean, t_scale) = custom_targets.unwrap_or_else(|| {
            (
                vec!["homo".to_string()],
                vec!["eV".to_string()],
                vec![-5.75],
                vec![0.85],
            )
        });
        let p_cnt = t_cols.len().max(1);
        let num_rbf = 32usize;
        let num_freqs = 16usize;
        let delta = 9.0f32 / 31.0;
        let rbf_gamma = 1.0 / (2.0 * delta * delta);
        let rbf_centers: Vec<f32> = (0..num_rbf).map(|r| -4.5 + delta * (r as f32)).collect();
        let fourier_freqs: Vec<f32> = (0..num_freqs)
            .map(|f| 2.0f32.powf(-1.0 + (5.0 * (f as f32)) / 15.0) * std::f32::consts::PI)
            .collect();
        let in_dim = 2 + num_rbf + 2 * num_freqs;

        let mut prop_id_emb = vec![0.0f32; p_cnt * d];
        for v in &mut prop_id_emb {
            *v = (rng.next_f32() * 2.0 - 1.0) * 0.02;
        }
        let inactive_prop_emb = vec![0.0f32; p_cnt * d];
        let mut prop_encoders = Vec::with_capacity(p_cnt);
        for _ in 0..p_cnt {
            prop_encoders.push(PropEncoderMlp {
                fc0: DenseLinear::kaiming_init(in_dim, d, true, rng),
                ln0: DenseLayerNorm::ones(d),
                fc1: DenseLinear::kaiming_init(d, d, true, rng),
                ln1: DenseLayerNorm::ones(d),
            });
        }
        // Zero-init `layer_mod`, `out_mod`, and `root_gate = 0.0` so initial conditioned output == base prior!
        let layer_mod = (0..n_layers)
            .map(|_| DenseLinear::zero_init(d, 2 * d, true))
            .collect();
        let out_mod = DenseLinear::zero_init(d, 2 * d, true);

        Ok(Self {
            cfg: ConditionerConfig {
                num_props: p_cnt,
                num_rbf,
                num_freqs,
                num_prefix_tokens: 0,
                modulation_cap: 4.0,
                rbf_gamma,
                rope_center_prefix: false,
                root_gate: 0.0,
            },
            target_cols: t_cols,
            target_units: t_units,
            target_mean: t_mean,
            target_scale: t_scale,
            rbf_centers,
            fourier_freqs,
            prop_id_emb,
            inactive_prop_emb,
            prop_encoders,
            global_fc0: DenseLinear::kaiming_init(p_cnt * d, 2 * d, true, rng),
            global_ln0: DenseLayerNorm::ones(2 * d),
            global_fc1: DenseLinear::kaiming_init(2 * d, d, true, rng),
            global_ln1: DenseLayerNorm::ones(d),
            root_proj: DenseLinear::kaiming_init(d, d, false, rng),
            root_gate: 0.0,
            layer_mod,
            out_mod,
        })
    }

    pub fn zero_grad(&self) -> ConditionerGrad {
        ConditionerGrad {
            d_prop_id_emb: vec![0.0; self.prop_id_emb.len()],
            d_inactive_prop_emb: vec![0.0; self.inactive_prop_emb.len()],
            prop_encoders: self
                .prop_encoders
                .iter()
                .map(|pe| PropEncoderGrad {
                    fc0: pe.fc0.zero_grad(),
                    ln0: pe.ln0.zero_grad(),
                    fc1: pe.fc1.zero_grad(),
                    ln1: pe.ln1.zero_grad(),
                })
                .collect(),
            global_fc0: self.global_fc0.zero_grad(),
            global_ln0: self.global_ln0.zero_grad(),
            global_fc1: self.global_fc1.zero_grad(),
            global_ln1: self.global_ln1.zero_grad(),
            root_proj: self.root_proj.zero_grad(),
            d_root_gate: 0.0,
            layer_mod: self.layer_mod.iter().map(|lm| lm.zero_grad()).collect(),
            out_mod: self.out_mod.zero_grad(),
        }
    }
}

pub struct CondBatchTape {
    pub b_sz: usize,
    pub has_any: Vec<bool>,
    pub active_masks: Vec<Vec<bool>>,
    pub prop_feats: Vec<Vec<f32>>, // [k][b_sz * in_dim]
    pub pe_h0_ln: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub pe_h0_act: Vec<Vec<f32>>,
    pub pe_h1_ln: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub cat_props: Vec<f32>, // [b_sz * P * D]
    pub gm0_ln: (Vec<f32>, Vec<f32>, Vec<f32>),
    pub gm0_act: Vec<f32>,
    pub gm1_ln: (Vec<f32>, Vec<f32>, Vec<f32>),
    pub global_c: Vec<f32>,     // [b_sz * D]
    pub c_silu: Vec<f32>,       // [b_sz * D]
    pub root_proj_raw: Vec<f32>,// [b_sz * D]
    pub root_delta: Vec<f32>,   // [b_sz * D]
    pub layer_gamma_beta: Vec<Vec<f32>>, // [layer][b_sz * 2*D] (post-cap tanh)
    pub out_gamma_beta: Vec<f32>,        // [b_sz * 2*D]
}

impl TrainableConditioner {
    pub fn forward_batch(&self, raw_targets: &[Vec<f32>], active_masks: &[Vec<bool>]) -> CondBatchTape {
        let b_sz = raw_targets.len();
        let p_cnt = self.cfg.num_props;
        let d = self.out_mod.in_dim;
        let in_dim = 2 + self.cfg.num_rbf + 2 * self.cfg.num_freqs;

        let mut prop_feats = vec![vec![0.0f32; b_sz * in_dim]; p_cnt];
        let mut has_any = vec![false; b_sz];

        for b in 0..b_sz {
            for k in 0..p_cnt {
                let is_act = active_masks
                    .get(b)
                    .and_then(|m| m.get(k))
                    .copied()
                    .unwrap_or(false);
                if is_act {
                    has_any[b] = true;
                    let raw_v = raw_targets[b].get(k).copied().unwrap_or(0.0);
                    let m = self.target_mean.get(k).copied().unwrap_or(0.0);
                    let sc = self.target_scale.get(k).copied().unwrap_or(1.0).max(1e-6);
                    let v = (raw_v - m) / sc;
                    let dst = &mut prop_feats[k][b * in_dim..(b + 1) * in_dim];
                    let mut pos = 0usize;
                    dst[pos] = v;
                    pos += 1;
                    for &c in &self.rbf_centers {
                        let diff = v - c;
                        dst[pos] = (-self.cfg.rbf_gamma * diff * diff).exp();
                        pos += 1;
                    }
                    for &f in &self.fourier_freqs {
                        dst[pos] = (v * f).sin();
                        pos += 1;
                    }
                    for &f in &self.fourier_freqs {
                        dst[pos] = (v * f).cos();
                        pos += 1;
                    }
                    dst[pos] = 1.0;
                }
            }
        }

        let mut pe_h0_ln = Vec::with_capacity(p_cnt);
        let mut pe_h0_act = Vec::with_capacity(p_cnt);
        let mut pe_h1_ln = Vec::with_capacity(p_cnt);
        let mut cat_props = vec![0.0f32; b_sz * p_cnt * d];

        for k in 0..p_cnt {
            let pe = &self.prop_encoders[k];
            let z0 = pe.fc0.fwd(&prop_feats[k], b_sz);
            let ln0 = pe.ln0.fwd(&z0, b_sz);
            let a0: Vec<f32> = ln0.0.iter().map(|&v| silu(v)).collect();
            let z1 = pe.fc1.fwd(&a0, b_sz);
            let ln1 = pe.ln1.fwd(&z1, b_sz);

            for b in 0..b_sz {
                let is_act = active_masks
                    .get(b)
                    .and_then(|m| m.get(k))
                    .copied()
                    .unwrap_or(false);
                let dst = &mut cat_props[(b * p_cnt + k) * d..(b * p_cnt + k + 1) * d];
                if is_act {
                    for c in 0..d {
                        dst[c] = silu(ln1.0[b * d + c]) + self.prop_id_emb[k * d + c];
                    }
                } else {
                    dst.copy_from_slice(&self.inactive_prop_emb[k * d..(k + 1) * d]);
                }
            }
            pe_h0_ln.push(ln0);
            pe_h0_act.push(a0);
            pe_h1_ln.push(ln1);
        }

        let gm0_pre = self.global_fc0.fwd(&cat_props, b_sz);
        let gm0_ln = self.global_ln0.fwd(&gm0_pre, b_sz);
        let gm0_act: Vec<f32> = gm0_ln.0.iter().map(|&v| silu(v)).collect();
        let gm1_pre = self.global_fc1.fwd(&gm0_act, b_sz);
        let gm1_ln = self.global_ln1.fwd(&gm1_pre, b_sz);

        let mut global_c = vec![0.0f32; b_sz * d];
        for b in 0..b_sz {
            if has_any[b] {
                for c in 0..d {
                    global_c[b * d + c] = silu(gm1_ln.0[b * d + c]);
                }
            }
        }
        let c_silu: Vec<f32> = global_c.iter().map(|&v| silu(v)).collect();

        let root_proj_raw = self.root_proj.fwd(&global_c, b_sz);
        let root_delta: Vec<f32> = root_proj_raw.iter().map(|&v| v * self.root_gate).collect();

        let cap = self.cfg.modulation_cap;
        let apply_cap = |mut raw: Vec<f32>| -> Vec<f32> {
            if cap > 0.0 {
                for v in &mut raw {
                    *v = cap * (*v / cap).tanh();
                }
            }
            raw
        };

        let layer_gamma_beta: Vec<Vec<f32>> = self
            .layer_mod
            .iter()
            .map(|lm| apply_cap(lm.fwd(&c_silu, b_sz)))
            .collect();
        let out_gamma_beta = apply_cap(self.out_mod.fwd(&c_silu, b_sz));

        CondBatchTape {
            b_sz,
            has_any,
            active_masks: active_masks.to_vec(),
            prop_feats,
            pe_h0_ln,
            pe_h0_act,
            pe_h1_ln,
            cat_props,
            gm0_ln,
            gm0_act,
            gm1_ln,
            global_c,
            c_silu,
            root_proj_raw,
            root_delta,
            layer_gamma_beta,
            out_gamma_beta,
        }
    }

    pub fn backward_batch(
        &self,
        tape: &CondBatchTape,
        d_root_delta: &[f32],
        d_layer_gb: &[Vec<f32>],
        d_out_gb: &[f32],
        grad: &mut ConditionerGrad,
    ) {
        let b_sz = tape.b_sz;
        let d = self.out_mod.in_dim;
        let p_cnt = self.cfg.num_props;
        let cap = self.cfg.modulation_cap;

        let uncap_grad = |d_gb: &[f32], gb_post: &[f32]| -> Vec<f32> {
            if cap > 0.0 {
                d_gb.iter()
                    .zip(gb_post.iter())
                    .map(|(&g, &y)| {
                        let t = y / cap;
                        g * (1.0 - t * t)
                    })
                    .collect()
            } else {
                d_gb.to_vec()
            }
        };

        let mut dc_silu = self.out_mod.bwd(
            &tape.c_silu,
            &uncap_grad(d_out_gb, &tape.out_gamma_beta),
            b_sz,
            Some(&mut grad.out_mod),
        );
        for (l, lm) in self.layer_mod.iter().enumerate() {
            let dl_raw = uncap_grad(&d_layer_gb[l], &tape.layer_gamma_beta[l]);
            let dc = lm.bwd(&tape.c_silu, &dl_raw, b_sz, Some(&mut grad.layer_mod[l]));
            for i in 0..dc_silu.len() {
                dc_silu[i] += dc[i];
            }
        }

        // `root_delta = root_gate * root_proj(global_c)`
        for i in 0..d_root_delta.len() {
            grad.d_root_gate += d_root_delta[i] * tape.root_proj_raw[i];
        }
        let d_rp_raw: Vec<f32> = d_root_delta.iter().map(|&g| g * self.root_gate).collect();
        let mut d_global_c =
            self.root_proj
                .bwd(&tape.global_c, &d_rp_raw, b_sz, Some(&mut grad.root_proj));
        for i in 0..d_global_c.len() {
            d_global_c[i] += dc_silu[i] * silu_grad(tape.global_c[i]);
        }

        let mut d_gm1_ln = vec![0.0f32; b_sz * d];
        for b in 0..b_sz {
            if tape.has_any[b] {
                for c in 0..d {
                    let idx = b * d + c;
                    d_gm1_ln[idx] = d_global_c[idx] * silu_grad(tape.gm1_ln.0[idx]);
                }
            }
        }
        let d_gm1_pre = self.global_ln1.bwd(
            &d_gm1_ln,
            &tape.gm1_ln.1,
            &tape.gm1_ln.2,
            b_sz,
            Some(&mut grad.global_ln1),
        );
        let d_gm0_act =
            self.global_fc1
                .bwd(&tape.gm0_act, &d_gm1_pre, b_sz, Some(&mut grad.global_fc1));
        let d_gm0_ln: Vec<f32> = d_gm0_act
            .iter()
            .zip(tape.gm0_ln.0.iter())
            .map(|(&g, &x)| g * silu_grad(x))
            .collect();
        let d_gm0_pre = self.global_ln0.bwd(
            &d_gm0_ln,
            &tape.gm0_ln.1,
            &tape.gm0_ln.2,
            b_sz,
            Some(&mut grad.global_ln0),
        );
        let d_cat =
            self.global_fc0
                .bwd(&tape.cat_props, &d_gm0_pre, b_sz, Some(&mut grad.global_fc0));

        for k in 0..p_cnt {
            let pe = &self.prop_encoders[k];
            let peg = &mut grad.prop_encoders[k];
            let mut d_pe1_ln = vec![0.0f32; b_sz * d];
            for b in 0..b_sz {
                let is_act = tape
                    .active_masks
                    .get(b)
                    .and_then(|m| m.get(k))
                    .copied()
                    .unwrap_or(false);
                let src = &d_cat[(b * p_cnt + k) * d..(b * p_cnt + k + 1) * d];
                if is_act {
                    for c in 0..d {
                        grad.d_prop_id_emb[k * d + c] += src[c];
                        d_pe1_ln[b * d + c] = src[c] * silu_grad(tape.pe_h1_ln[k].0[b * d + c]);
                    }
                } else {
                    for c in 0..d {
                        grad.d_inactive_prop_emb[k * d + c] += src[c];
                    }
                }
            }
            let d_z1 = pe.ln1.bwd(
                &d_pe1_ln,
                &tape.pe_h1_ln[k].1,
                &tape.pe_h1_ln[k].2,
                b_sz,
                Some(&mut peg.ln1),
            );
            let d_a0 = pe.fc1.bwd(&tape.pe_h0_act[k], &d_z1, b_sz, Some(&mut peg.fc1));
            let d_pe0_ln: Vec<f32> = d_a0
                .iter()
                .zip(tape.pe_h0_ln[k].0.iter())
                .map(|(&g, &x)| g * silu_grad(x))
                .collect();
            let d_z0 = pe.ln0.bwd(
                &d_pe0_ln,
                &tape.pe_h0_ln[k].1,
                &tape.pe_h0_ln[k].2,
                b_sz,
                Some(&mut peg.ln0),
            );
            let _ = pe.fc0.bwd(&tape.prop_feats[k], &d_z0, b_sz, Some(&mut peg.fc0));
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableValenceHead {
    pub norm: DenseLayerNorm,
    pub proj_num_h: DenseLinear,  // (D -> 5)
    pub proj_charge: DenseLinear, // (D -> 3)
}

#[derive(Debug, Clone)]
pub struct ValenceHeadGrad {
    pub norm: DenseLayerNormGrad,
    pub proj_num_h: DenseLinearGrad,
    pub proj_charge: DenseLinearGrad,
}

impl TrainableValenceHead {
    pub fn new(emb_dim: usize, rng: &mut DetRng) -> Self {
        Self {
            norm: DenseLayerNorm::ones(emb_dim),
            proj_num_h: DenseLinear::kaiming_init(emb_dim, 5, true, rng),
            proj_charge: DenseLinear::kaiming_init(emb_dim, 3, true, rng),
        }
    }

    pub fn zero_grad(&self) -> ValenceHeadGrad {
        ValenceHeadGrad {
            norm: self.norm.zero_grad(),
            proj_num_h: self.proj_num_h.zero_grad(),
            proj_charge: self.proj_charge.zero_grad(),
        }
    }

    /// Evaluate `loss_valence = CE(num_h) + 0.5 * CE(charge_cls)` on `rows` heavy-atom hidden vectors
    /// and return `(loss, d_hidden)`.
    pub fn forward_and_bwd(
        &self,
        h_heavy: &[f32],
        targets: &[(usize, usize)],
        loss_weight: f32,
        mut grad: Option<&mut ValenceHeadGrad>,
    ) -> (f32, Vec<f32>) {
        let rows = targets.len();
        let d = self.norm.dim;
        if rows == 0 || loss_weight == 0.0 {
            return (0.0, vec![0.0; rows * d]);
        }
        let (hn, x_hat, inv_std) = self.norm.fwd(h_heavy, rows);
        let logits_h = self.proj_num_h.fwd(&hn, rows);
        let logits_q = self.proj_charge.fwd(&hn, rows);

        let inv_r = 1.0 / (rows as f32);
        let mut ce_h = 0.0f32;
        let mut ce_q = 0.0f32;
        let mut dl_h = vec![0.0f32; rows * 5];
        let mut dl_q = vec![0.0f32; rows * 3];

        for r in 0..rows {
            let (gt_h, gt_q) = targets[r];
            ce_h += cross_entropy_row_grad(
                &logits_h[r * 5..(r + 1) * 5],
                gt_h.min(4),
                loss_weight * inv_r,
                &mut dl_h[r * 5..(r + 1) * 5],
            );
            ce_q += cross_entropy_row_grad(
                &logits_q[r * 3..(r + 1) * 3],
                gt_q.min(2),
                0.5 * loss_weight * inv_r,
                &mut dl_q[r * 3..(r + 1) * 3],
            );
        }

        let dhn_1 = self
            .proj_num_h
            .bwd(&hn, &dl_h, rows, grad.as_deref_mut().map(|g| &mut g.proj_num_h));
        let dhn_2 = self
            .proj_charge
            .bwd(&hn, &dl_q, rows, grad.as_deref_mut().map(|g| &mut g.proj_charge));
        let dhn: Vec<f32> = dhn_1.iter().zip(dhn_2.iter()).map(|(&a, &b)| a + b).collect();
        let dh = self
            .norm
            .bwd(&dhn, &x_hat, &inv_std, rows, grad.map(|g| &mut g.norm));

        ((ce_h + 0.5 * ce_q) * inv_r, dh)
    }
}

/// Stable single-row CrossEntropy returning unweighted `-log p[label]` while writing `weight * (p - one_hot)` into `d_logits`.
#[inline]
pub fn cross_entropy_row_grad(logits: &[f32], label: usize, grad_scale: f32, d_logits: &mut [f32]) -> f32 {
    let mut max_l = f32::NEG_INFINITY;
    for &l in logits {
        if l.is_finite() && l > max_l {
            max_l = l;
        }
    }
    if !max_l.is_finite() {
        d_logits.fill(0.0);
        return 0.0;
    }
    let mut sum_e = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        if l.is_finite() {
            let e = (l - max_l).exp();
            d_logits[i] = e;
            sum_e += e;
        } else {
            d_logits[i] = 0.0;
        }
    }
    let inv_s = 1.0 / sum_e.max(1e-20);
    let nll = if logits[label].is_finite() {
        -(logits[label] - max_l) + sum_e.max(1e-20).ln()
    } else {
        20.0
    };
    for i in 0..logits.len() {
        let p = d_logits[i] * inv_s;
        let target = if i == label { 1.0 } else { 0.0 };
        d_logits[i] = (p - target) * grad_scale;
    }
    nll
}

// ============================================================================
// Complete `LoomLoraAdapter` + `AdamW` Optimizer + `.loom_lora` Serialization
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoomLoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub include_ffn: bool,
    pub include_xyz_heads: bool,
    pub train_conditioner: bool,
    pub use_valence_head: bool,
}

impl Default for LoomLoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            include_ffn: true,
            include_xyz_heads: false,
            train_conditioner: true,
            use_valence_head: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoomLoraAdapter {
    pub cfg: LoomLoraConfig,
    /// Map from `.u3m` linear prefix (e.g. `"bb.layers.0.attn.q_proj"`, `"bb.tree_head.dense"`) to `LoraPair`.
    pub loras: HashMap<String, LoraPair>,
    pub conditioner: TrainableConditioner,
    pub valence_head: Option<TrainableValenceHead>,
}

pub struct LoomAdapterGrads {
    pub loras: HashMap<String, LoraGrad>,
    pub cond: Option<ConditionerGrad>,
    pub valence: Option<ValenceHeadGrad>,
}

impl LoomLoraAdapter {
    pub fn new_for_bundle(
        bundle: &ModelBundle,
        cfg: LoomLoraConfig,
        custom_targets: Option<(Vec<String>, Vec<String>, Vec<f32>, Vec<f32>)>,
        seed: u64,
    ) -> Result<Self, String> {
        let mut rng = DetRng::new(seed ^ 0x51F7_6BB0);
        let bb = &bundle.header.backbone;
        let d = bb.emb_dim;
        let ffn_d = bb.ffn_dim;
        let mut loras = HashMap::new();

        for l in 0..bb.num_layers {
            for p_name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                let key = format!("bb.layers.{l}.attn.{p_name}");
                loras.insert(
                    key,
                    LoraPair::new_zero_init(cfg.rank, cfg.alpha, d, d, &mut rng),
                );
            }
            if cfg.include_ffn {
                loras.insert(
                    format!("bb.layers.{l}.ffn.w1"),
                    LoraPair::new_zero_init(cfg.rank, cfg.alpha, d, ffn_d, &mut rng),
                );
                loras.insert(
                    format!("bb.layers.{l}.ffn.w3"),
                    LoraPair::new_zero_init(cfg.rank, cfg.alpha, d, ffn_d, &mut rng),
                );
                loras.insert(
                    format!("bb.layers.{l}.ffn.w2"),
                    LoraPair::new_zero_init(cfg.rank, cfg.alpha, ffn_d, d, &mut rng),
                );
            }
        }

        // Tree & Element heads
        loras.insert(
            "bb.tree_head.dense".into(),
            LoraPair::new_zero_init(cfg.rank, cfg.alpha, d, d, &mut rng),
        );
        loras.insert(
            "bb.tree_head.out".into(),
            LoraPair::new_zero_init(cfg.rank, cfg.alpha, d, 255, &mut rng),
        );
        loras.insert(
            "bb.atom_head.pred.0.dense".into(),
            LoraPair::new_zero_init(cfg.rank, cfg.alpha, d, d, &mut rng),
        );
        loras.insert(
            "bb.atom_head.pred.0.out".into(),
            LoraPair::new_zero_init(cfg.rank, cfg.alpha, d, bb.num_atom_vocab, &mut rng),
        );

        if cfg.include_xyz_heads {
            for ax in 1..=3 {
                loras.insert(
                    format!("bb.atom_head.pred.{ax}.dense"),
                    LoraPair::new_zero_init(cfg.rank, cfg.alpha, d, d, &mut rng),
                );
                loras.insert(
                    format!("bb.atom_head.pred.{ax}.out"),
                    LoraPair::new_zero_init(cfg.rank, cfg.alpha, d, bb.num_xyz, &mut rng),
                );
            }
        }

        let conditioner = TrainableConditioner::from_bundle_or_fresh(bundle, custom_targets, &mut rng)?;
        let valence_head = if cfg.use_valence_head {
            Some(TrainableValenceHead::new(d, &mut rng))
        } else {
            None
        };

        Ok(Self {
            cfg,
            loras,
            conditioner,
            valence_head,
        })
    }

    /// Stage 2.5B helper: ensure `XYZ` head LoRA adapters (`pred.1..3`) exist, freeze all topology/backbone/cond
    /// parameters, and unfreeze ONLY `bb.atom_head.pred.{1,2,3}.{dense,out}` (`111,744` scalars at `rank=16`)!
    pub fn set_xyz_distill_only_mode(&mut self, emb_dim: usize, num_xyz: usize, seed: u64) {
        let mut rng = DetRng::new(seed ^ 0x25B0_0001);
        for lp in self.loras.values_mut() {
            lp.trainable = false;
        }
        for ax in 1..=3 {
            let k_dense = format!("bb.atom_head.pred.{ax}.dense");
            self.loras
                .entry(k_dense)
                .or_insert_with(|| LoraPair::new_zero_init(self.cfg.rank, self.cfg.alpha, emb_dim, emb_dim, &mut rng))
                .trainable = true;
            let k_out = format!("bb.atom_head.pred.{ax}.out");
            self.loras
                .entry(k_out)
                .or_insert_with(|| LoraPair::new_zero_init(self.cfg.rank, self.cfg.alpha, emb_dim, num_xyz, &mut rng))
                .trainable = true;
        }
        self.cfg.include_xyz_heads = true;
        self.cfg.train_conditioner = false;
    }

    /// Stage 2.5A (`TopoGRPO`) / Stage 0 (`SFT`) mode: unfreeze backbone/tree/elem LoRA + conditioner,
    /// while keeping `XYZ` sub-voxel heads frozen so fine `0.01 Å` geometry never drifts during RL!
    pub fn set_topology_policy_mode(&mut self, train_conditioner: bool) {
        for (k, lp) in &mut self.loras {
            let is_xyz = k.starts_with("bb.atom_head.pred.1")
                || k.starts_with("bb.atom_head.pred.2")
                || k.starts_with("bb.atom_head.pred.3");
            lp.trainable = !is_xyz;
        }
        self.cfg.train_conditioner = train_conditioner;
    }

    pub fn num_trainable_parameters(&self) -> usize {
        let mut total = 0usize;
        for lp in self.loras.values() {
            if lp.trainable {
                total += lp.a.len() + lp.b.len();
            }
        }
        if self.cfg.train_conditioner {
            let c = &self.conditioner;
            total += c.prop_id_emb.len() + c.inactive_prop_emb.len() + 1;
            for pe in &c.prop_encoders {
                total += pe.fc0.w.len() + pe.fc0.b.as_ref().map_or(0, |b| b.len());
                total += 2 * pe.ln0.dim;
                total += pe.fc1.w.len() + pe.fc1.b.as_ref().map_or(0, |b| b.len());
                total += 2 * pe.ln1.dim;
            }
            total += c.global_fc0.w.len() + c.global_fc0.b.as_ref().map_or(0, |b| b.len()) + 2 * c.global_ln0.dim;
            total += c.global_fc1.w.len() + c.global_fc1.b.as_ref().map_or(0, |b| b.len()) + 2 * c.global_ln1.dim;
            total += c.root_proj.w.len();
            for lm in &c.layer_mod {
                total += lm.w.len() + lm.b.as_ref().map_or(0, |b| b.len());
            }
            total += c.out_mod.w.len() + c.out_mod.b.as_ref().map_or(0, |b| b.len());
        }
        if let Some(ref vh) = self.valence_head {
            if self.cfg.train_conditioner {
                total += 2 * vh.norm.dim
                    + vh.proj_num_h.w.len()
                    + 5
                    + vh.proj_charge.w.len()
                    + 3;
            }
        }
        total
    }

    pub fn zero_grads(&self) -> LoomAdapterGrads {
        let mut loras = HashMap::with_capacity(self.loras.len());
        for (k, lp) in &self.loras {
            if lp.trainable {
                loras.insert(k.clone(), lp.zero_grad());
            }
        }
        let cond = if self.cfg.train_conditioner {
            Some(self.conditioner.zero_grad())
        } else {
            None
        };
        let valence = if self.cfg.train_conditioner {
            self.valence_head.as_ref().map(|vh| vh.zero_grad())
        } else {
            None
        };
        LoomAdapterGrads {
            loras,
            cond,
            valence,
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        const BIN_MAGIC: &[u8; 8] = b"LOOMLR02";
        #[derive(Serialize)]
        struct BinHeader {
            shell: LoomLoraAdapter,
            specs: Vec<(usize, bool)>, // (num_elems, is_f32)
        }

        let mut shell = self.clone();
        let mut specs = Vec::new();
        let mut payload = Vec::with_capacity(48 * 1024 * 1024);

        let mut pack_vec = |v: &mut Vec<f32>, as_f32: bool| {
            specs.push((v.len(), as_f32));
            if as_f32 {
                payload.extend_from_slice(bytemuck::cast_slice(v.as_slice()));
            } else {
                for &val in v.iter() {
                    payload.extend_from_slice(&f16::from_f32(val).to_bits().to_le_bytes());
                }
            }
            *v = Vec::new();
        };

        let mut keys: Vec<String> = shell.loras.keys().cloned().collect();
        keys.sort();
        for k in &keys {
            let lp = shell.loras.get_mut(k).unwrap();
            pack_vec(&mut lp.a, true);
            pack_vec(&mut lp.b, true);
        }
        {
            let c = &mut shell.conditioner;
            pack_vec(&mut c.prop_id_emb, false);
            pack_vec(&mut c.inactive_prop_emb, false);
            for pe in &mut c.prop_encoders {
                pack_vec(&mut pe.fc0.w, false);
                if let Some(ref mut b) = pe.fc0.b {
                    pack_vec(b, true);
                }
                pack_vec(&mut pe.ln0.w, true);
                pack_vec(&mut pe.ln0.b, true);
                pack_vec(&mut pe.fc1.w, false);
                if let Some(ref mut b) = pe.fc1.b {
                    pack_vec(b, true);
                }
                pack_vec(&mut pe.ln1.w, true);
                pack_vec(&mut pe.ln1.b, true);
            }
            pack_vec(&mut c.global_fc0.w, false);
            if let Some(ref mut b) = c.global_fc0.b {
                pack_vec(b, true);
            }
            pack_vec(&mut c.global_ln0.w, true);
            pack_vec(&mut c.global_ln0.b, true);
            pack_vec(&mut c.global_fc1.w, false);
            if let Some(ref mut b) = c.global_fc1.b {
                pack_vec(b, true);
            }
            pack_vec(&mut c.global_ln1.w, true);
            pack_vec(&mut c.global_ln1.b, true);
            pack_vec(&mut c.root_proj.w, false);
            for lm in &mut c.layer_mod {
                pack_vec(&mut lm.w, false);
                if let Some(ref mut b) = lm.b {
                    pack_vec(b, true);
                }
            }
            pack_vec(&mut c.out_mod.w, false);
            if let Some(ref mut b) = c.out_mod.b {
                pack_vec(b, true);
            }
        }
        if let Some(ref mut vh) = shell.valence_head {
            pack_vec(&mut vh.norm.w, true);
            pack_vec(&mut vh.norm.b, true);
            pack_vec(&mut vh.proj_num_h.w, true);
            if let Some(ref mut b) = vh.proj_num_h.b {
                pack_vec(b, true);
            }
            pack_vec(&mut vh.proj_charge.w, true);
            if let Some(ref mut b) = vh.proj_charge.b {
                pack_vec(b, true);
            }
        }

        let hdr_json =
            serde_json::to_vec(&BinHeader { shell, specs }).map_err(|e| format!("Serialize .loom_lora header: {e}"))?;
        let mut out = Vec::with_capacity(16 + hdr_json.len() + payload.len());
        out.extend_from_slice(BIN_MAGIC);
        out.extend_from_slice(&(hdr_json.len() as u64).to_le_bytes());
        out.extend_from_slice(&hdr_json);
        out.extend_from_slice(&payload);
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 {
            return Err("Invalid .loom_lora magic header".into());
        }
        if &bytes[0..8] == LOOM_LORA_MAGIC {
            let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
            if bytes.len() < 16 + len {
                return Err("Truncated .loom_lora v1 payload".into());
            }
            return serde_json::from_slice(&bytes[16..16 + len])
                .map_err(|e| format!("Invalid .loom_lora JSON: {e}"));
        }
        if &bytes[0..8] != b"LOOMLR02" {
            return Err("Invalid .loom_lora magic header".into());
        }
        #[derive(Deserialize)]
        struct BinHeader {
            shell: LoomLoraAdapter,
            specs: Vec<(usize, bool)>,
        }
        let hdr_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        if bytes.len() < 16 + hdr_len {
            return Err("Truncated .loom_lora header".into());
        }
        let hdr: BinHeader = serde_json::from_slice(&bytes[16..16 + hdr_len])
            .map_err(|e| format!("Invalid .loom_lora header JSON: {e}"))?;
        let mut adapter = hdr.shell;
        let payload = &bytes[16 + hdr_len..];
        let mut cursor = 0usize;
        let mut spec_idx = 0usize;

        let mut unpack_vec = |dst: &mut Vec<f32>| -> Result<(), String> {
            let &(len, is_f32) = hdr
                .specs
                .get(spec_idx)
                .ok_or_else(|| "Corrupt .loom_lora slice spec count".to_string())?;
            spec_idx += 1;
            let nbytes = len * if is_f32 { 4 } else { 2 };
            if cursor + nbytes > payload.len() {
                return Err("Truncated .loom_lora binary tensor payload".into());
            }
            let mut v = Vec::with_capacity(len);
            if is_f32 {
                for chunk in payload[cursor..cursor + nbytes].chunks_exact(4) {
                    v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            } else {
                for chunk in payload[cursor..cursor + nbytes].chunks_exact(2) {
                    v.push(f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32());
                }
            }
            cursor += nbytes;
            *dst = v;
            Ok(())
        };

        let mut keys: Vec<String> = adapter.loras.keys().cloned().collect();
        keys.sort();
        for k in &keys {
            let lp = adapter.loras.get_mut(k).unwrap();
            unpack_vec(&mut lp.a)?;
            unpack_vec(&mut lp.b)?;
        }
        {
            let c = &mut adapter.conditioner;
            unpack_vec(&mut c.prop_id_emb)?;
            unpack_vec(&mut c.inactive_prop_emb)?;
            for pe in &mut c.prop_encoders {
                unpack_vec(&mut pe.fc0.w)?;
                if let Some(ref mut b) = pe.fc0.b {
                    unpack_vec(b)?;
                }
                unpack_vec(&mut pe.ln0.w)?;
                unpack_vec(&mut pe.ln0.b)?;
                unpack_vec(&mut pe.fc1.w)?;
                if let Some(ref mut b) = pe.fc1.b {
                    unpack_vec(b)?;
                }
                unpack_vec(&mut pe.ln1.w)?;
                unpack_vec(&mut pe.ln1.b)?;
            }
            unpack_vec(&mut c.global_fc0.w)?;
            if let Some(ref mut b) = c.global_fc0.b {
                unpack_vec(b)?;
            }
            unpack_vec(&mut c.global_ln0.w)?;
            unpack_vec(&mut c.global_ln0.b)?;
            unpack_vec(&mut c.global_fc1.w)?;
            if let Some(ref mut b) = c.global_fc1.b {
                unpack_vec(b)?;
            }
            unpack_vec(&mut c.global_ln1.w)?;
            unpack_vec(&mut c.global_ln1.b)?;
            unpack_vec(&mut c.root_proj.w)?;
            for lm in &mut c.layer_mod {
                unpack_vec(&mut lm.w)?;
                if let Some(ref mut b) = lm.b {
                    unpack_vec(b)?;
                }
            }
            unpack_vec(&mut c.out_mod.w)?;
            if let Some(ref mut b) = c.out_mod.b {
                unpack_vec(b)?;
            }
        }
        if let Some(ref mut vh) = adapter.valence_head {
            unpack_vec(&mut vh.norm.w)?;
            unpack_vec(&mut vh.norm.b)?;
            unpack_vec(&mut vh.proj_num_h.w)?;
            if let Some(ref mut b) = vh.proj_num_h.b {
                unpack_vec(b)?;
            }
            unpack_vec(&mut vh.proj_charge.w)?;
            if let Some(ref mut b) = vh.proj_charge.b {
                unpack_vec(b)?;
            }
        }
        Ok(adapter)
    }
}

// ============================================================================
// `AdamW` Optimizer over `LoomLoraAdapter`
// ============================================================================

pub struct LoomAdamW {
    pub step_t: usize,
    m_buf: Vec<f32>,
    v_buf: Vec<f32>,
}

impl LoomAdamW {
    pub fn new() -> Self {
        Self {
            step_t: 0,
            m_buf: Vec::new(),
            v_buf: Vec::new(),
        }
    }

    /// Clip global gradient norm to `max_grad_norm` and apply `AdamW` update if finite!
    /// Returns `(grad_norm, did_step)`.
    pub fn step(
        &mut self,
        adapter: &mut LoomLoraAdapter,
        grads: &LoomAdapterGrads,
        lr: f32,
        weight_decay: f32,
        max_grad_norm: f32,
    ) -> (f32, bool) {
        // Stable sorted key order over `adapter.loras`
        let mut lora_keys: Vec<String> = adapter
            .loras
            .iter()
            .filter(|(_, lp)| lp.trainable)
            .map(|(k, _)| k.clone())
            .collect();
        lora_keys.sort();

        // 1. Compute global L2 norm & check finiteness
        let mut sq_sum = 0.0f64;
        let mut total_cnt = 0usize;
        let mut all_finite = true;

        let mut acc_slice = |s: &[f32]| {
            total_cnt += s.len();
            for &g in s {
                if !g.is_finite() {
                    all_finite = false;
                } else {
                    sq_sum += (g as f64) * (g as f64);
                }
            }
        };

        for k in &lora_keys {
            if let Some(lg) = grads.loras.get(k) {
                acc_slice(&lg.da);
                acc_slice(&lg.db);
            }
        }
        if let Some(ref cg) = grads.cond {
            acc_slice(&cg.d_prop_id_emb);
            acc_slice(&cg.d_inactive_prop_emb);
            for peg in &cg.prop_encoders {
                acc_slice(&peg.fc0.dw);
                if let Some(ref db) = peg.fc0.db {
                    acc_slice(db);
                }
                acc_slice(&peg.ln0.dw);
                acc_slice(&peg.ln0.db);
                acc_slice(&peg.fc1.dw);
                if let Some(ref db) = peg.fc1.db {
                    acc_slice(db);
                }
                acc_slice(&peg.ln1.dw);
                acc_slice(&peg.ln1.db);
            }
            acc_slice(&cg.global_fc0.dw);
            if let Some(ref db) = cg.global_fc0.db {
                acc_slice(db);
            }
            acc_slice(&cg.global_ln0.dw);
            acc_slice(&cg.global_ln0.db);
            acc_slice(&cg.global_fc1.dw);
            if let Some(ref db) = cg.global_fc1.db {
                acc_slice(db);
            }
            acc_slice(&cg.global_ln1.dw);
            acc_slice(&cg.global_ln1.db);
            acc_slice(&cg.root_proj.dw);
            acc_slice(std::slice::from_ref(&cg.d_root_gate));
            for lmg in &cg.layer_mod {
                acc_slice(&lmg.dw);
                if let Some(ref db) = lmg.db {
                    acc_slice(db);
                }
            }
            acc_slice(&cg.out_mod.dw);
            if let Some(ref db) = cg.out_mod.db {
                acc_slice(db);
            }
        }
        if let Some(ref vg) = grads.valence {
            acc_slice(&vg.norm.dw);
            acc_slice(&vg.norm.db);
            acc_slice(&vg.proj_num_h.dw);
            if let Some(ref db) = vg.proj_num_h.db {
                acc_slice(db);
            }
            acc_slice(&vg.proj_charge.dw);
            if let Some(ref db) = vg.proj_charge.db {
                acc_slice(db);
            }
        }

        let gnorm = (sq_sum.sqrt()) as f32;
        if !all_finite || !gnorm.is_finite() || total_cnt == 0 {
            return (gnorm, false);
        }

        if self.m_buf.len() != total_cnt {
            self.m_buf.assign_or_resize(total_cnt);
            self.v_buf.assign_or_resize(total_cnt);
            self.step_t = 0;
        }
        self.step_t += 1;
        let bc1 = 1.0 - 0.9f32.powi(self.step_t as i32);
        let bc2 = 1.0 - 0.999f32.powi(self.step_t as i32);
        let clip_coef = if gnorm > max_grad_norm {
            max_grad_norm / (gnorm + 1e-8)
        } else {
            1.0
        };

        let mut cursor = 0usize;
        let m_buf = &mut self.m_buf;
        let v_buf = &mut self.v_buf;

        let mut apply_upd = |p_sl: &mut [f32], g_sl: &[f32], decay: bool| {
            for i in 0..p_sl.len() {
                let g = g_sl[i] * clip_coef;
                if decay && weight_decay > 0.0 {
                    p_sl[i] *= 1.0 - lr * weight_decay;
                }
                let idx = cursor + i;
                m_buf[idx] = 0.9 * m_buf[idx] + 0.1 * g;
                v_buf[idx] = 0.999 * v_buf[idx] + 0.001 * g * g;
                let mh = m_buf[idx] / bc1;
                let vh = v_buf[idx] / bc2;
                p_sl[i] -= lr * mh / (vh.sqrt() + 1e-8);
            }
            cursor += p_sl.len();
        };

        for k in &lora_keys {
            if let (Some(lp), Some(lg)) = (adapter.loras.get_mut(k), grads.loras.get(k)) {
                apply_upd(&mut lp.a, &lg.da, true);
                apply_upd(&mut lp.b, &lg.db, true);
            }
        }
        if let Some(ref cg) = grads.cond {
            let c = &mut adapter.conditioner;
            apply_upd(&mut c.prop_id_emb, &cg.d_prop_id_emb, false);
            apply_upd(&mut c.inactive_prop_emb, &cg.d_inactive_prop_emb, false);
            for (pe, peg) in c.prop_encoders.iter_mut().zip(cg.prop_encoders.iter()) {
                apply_upd(&mut pe.fc0.w, &peg.fc0.dw, true);
                if let (Some(ref mut b), Some(ref db)) = (&mut pe.fc0.b, &peg.fc0.db) {
                    apply_upd(b, db, false);
                }
                apply_upd(&mut pe.ln0.w, &peg.ln0.dw, false);
                apply_upd(&mut pe.ln0.b, &peg.ln0.db, false);
                apply_upd(&mut pe.fc1.w, &peg.fc1.dw, true);
                if let (Some(ref mut b), Some(ref db)) = (&mut pe.fc1.b, &peg.fc1.db) {
                    apply_upd(b, db, false);
                }
                apply_upd(&mut pe.ln1.w, &peg.ln1.dw, false);
                apply_upd(&mut pe.ln1.b, &peg.ln1.db, false);
            }
            apply_upd(&mut c.global_fc0.w, &cg.global_fc0.dw, true);
            if let (Some(ref mut b), Some(ref db)) = (&mut c.global_fc0.b, &cg.global_fc0.db) {
                apply_upd(b, db, false);
            }
            apply_upd(&mut c.global_ln0.w, &cg.global_ln0.dw, false);
            apply_upd(&mut c.global_ln0.b, &cg.global_ln0.db, false);
            apply_upd(&mut c.global_fc1.w, &cg.global_fc1.dw, true);
            if let (Some(ref mut b), Some(ref db)) = (&mut c.global_fc1.b, &cg.global_fc1.db) {
                apply_upd(b, db, false);
            }
            apply_upd(&mut c.global_ln1.w, &cg.global_ln1.dw, false);
            apply_upd(&mut c.global_ln1.b, &cg.global_ln1.db, false);
            apply_upd(&mut c.root_proj.w, &cg.root_proj.dw, true);
            apply_upd(
                std::slice::from_mut(&mut c.root_gate),
                std::slice::from_ref(&cg.d_root_gate),
                false,
            );
            for (lm, lmg) in c.layer_mod.iter_mut().zip(cg.layer_mod.iter()) {
                apply_upd(&mut lm.w, &lmg.dw, true);
                if let (Some(ref mut b), Some(ref db)) = (&mut lm.b, &lmg.db) {
                    apply_upd(b, db, false);
                }
            }
            apply_upd(&mut c.out_mod.w, &cg.out_mod.dw, true);
            if let (Some(ref mut b), Some(ref db)) = (&mut c.out_mod.b, &cg.out_mod.db) {
                apply_upd(b, db, false);
            }
        }
        if let (Some(ref mut vh), Some(ref vg)) = (&mut adapter.valence_head, &grads.valence) {
            apply_upd(&mut vh.norm.w, &vg.norm.dw, false);
            apply_upd(&mut vh.norm.b, &vg.norm.db, false);
            apply_upd(&mut vh.proj_num_h.w, &vg.proj_num_h.dw, true);
            if let (Some(ref mut b), Some(ref db)) = (&mut vh.proj_num_h.b, &vg.proj_num_h.db) {
                apply_upd(b, db, false);
            }
            apply_upd(&mut vh.proj_charge.w, &vg.proj_charge.dw, true);
            if let (Some(ref mut b), Some(ref db)) = (&mut vh.proj_charge.b, &vg.proj_charge.db) {
                apply_upd(b, db, false);
            }
        }

        (gnorm, true)
    }
}

trait VecZeroResize {
    fn assign_or_resize(&mut self, len: usize);
}
impl VecZeroResize for Vec<f32> {
    fn assign_or_resize(&mut self, len: usize) {
        self.clear();
        self.resize(len, 0.0);
    }
}

// ============================================================================
// `wgpu` GPU Training Accelerator (`LoomGpuTrainContext`)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    f0: f32,
    f1: f32,
}

struct GpuF16Weight {
    out_d: usize,
    in_d: usize,
    pairs_per_row: usize,
    buf: wgpu::Buffer,
    bias: Option<Vec<f32>>,
}

struct SingleLayerRecompute {
    x_norm: Vec<f32>,
    attn_inv_rms: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_cat: Vec<f32>,
    lse: Vec<f32>,
    h_mid: Vec<f32>,
    u_norm: Vec<f32>,
    ffn_inv_rms: Vec<f32>,
    gate_pre: Vec<f32>,
    up: Vec<f32>,
    swiglu: Vec<f32>,
    ffn_out: Vec<f32>,
}

pub struct LoomGpuTrainContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: String,
    bgl_mm: wgpu::BindGroupLayout,
    pipe_mm_fwd: wgpu::ComputePipeline,
    pipe_mm_bwd: wgpu::ComputePipeline,
    bgl_el: wgpu::BindGroupLayout,
    pipe_lora_xa: wgpu::ComputePipeline,
    pipe_lora_xb: wgpu::ComputePipeline,
    pipe_rmsnorm: wgpu::ComputePipeline,
    pipe_rope_qk: wgpu::ComputePipeline,
    pipe_elem_op: wgpu::ComputePipeline,
    bgl_mha_fwd: wgpu::BindGroupLayout,
    pipe_mha_fwd: wgpu::ComputePipeline,
    bgl_mha_bwd: wgpu::BindGroupLayout,
    pipe_mha_bwd_dq: wgpu::ComputePipeline,
    pipe_mha_bwd_dkv: wgpu::ComputePipeline,
    dummy_f32: wgpu::Buffer,
    dummy_u32: wgpu::Buffer,
    dummy_rw1: wgpu::Buffer,
    base_weights: HashMap<String, GpuF16Weight>,
    norm_weights: HashMap<String, wgpu::Buffer>,
    has_any_layer_bias: bool,
}

impl LoomGpuTrainContext {
    pub async fn create(bundle: &ModelBundle) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "No WebGPU/Metal/Vulkan adapter available".to_string())?;

        let info = adapter.get_info();
        let adapter_info = format!("{} ({:?})", info.name, info.backend);
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("loom-rs-train-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("wgpu request_device failed: {e}"))?;

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("loom-rs-train-shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let uni = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl_mm = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_mm"),
            entries: &[uni, ro(1), ro(2), ro(3), rw(4)],
        });
        let bgl_el = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_el"),
            entries: &[uni, ro(1), ro(2), ro(3), ro(4), rw(5), rw(6)],
        });
        let bgl_mha_fwd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_mha_fwd"),
            entries: &[uni, ro(1), ro(2), ro(3), ro(4), rw(5), rw(6)],
        });
        let bgl_mha_bwd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_mha_bwd"),
            entries: &[uni, ro(1), ro(2), ro(3), ro(4), ro(5), ro(6), rw(7)],
        });

        let make_pipe = |bgl: &wgpu::BindGroupLayout, entry: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipe_mm_fwd = make_pipe(&bgl_mm, "mm_f16_fwd");
        let pipe_mm_bwd = make_pipe(&bgl_mm, "mm_f16_bwd_dx");
        let pipe_lora_xa = make_pipe(&bgl_el, "lora_xa_fwd");
        let pipe_lora_xb = make_pipe(&bgl_el, "lora_xb_fwd");
        let pipe_rmsnorm = make_pipe(&bgl_el, "rmsnorm_fwd");
        let pipe_rope_qk = make_pipe(&bgl_el, "rope_fwd_qk");
        let pipe_elem_op = make_pipe(&bgl_el, "elem_op");
        let pipe_mha_fwd = make_pipe(&bgl_mha_fwd, "causal_mha_fwd");
        let pipe_mha_bwd_dq = make_pipe(&bgl_mha_bwd, "causal_mha_bwd_dq");
        let pipe_mha_bwd_dkv = make_pipe(&bgl_mha_bwd, "causal_mha_bwd_dkv");

        if let Some(err) = device.pop_error_scope().await {
            return Err(format!("loom-rs WGSL validation error: {err}"));
        }

        let dummy_f32 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dummy_f32"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_u32 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dummy_u32"),
            contents: bytemuck::cast_slice(&[0u32; 4]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_rw1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dummy_rw1"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Upload frozen backbone + head Linear weights in compact `f16` (`~170 MB` total) + RMSNorm weights
        let bb = &bundle.header.backbone;
        let mut prefixes = Vec::new();
        let mut norm_names = vec!["bb.norm.weight".to_string()];
        for l in 0..bb.num_layers {
            norm_names.push(format!("bb.layers.{l}.attn_norm.weight"));
            norm_names.push(format!("bb.layers.{l}.ffn_norm.weight"));
            for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                prefixes.push(format!("bb.layers.{l}.attn.{p}"));
            }
            for w in ["w1", "w2", "w3"] {
                prefixes.push(format!("bb.layers.{l}.ffn.{w}"));
            }
        }
        prefixes.push("bb.tree_head.dense".into());
        prefixes.push("bb.tree_head.out".into());
        for ax in 0..4 {
            prefixes.push(format!("bb.atom_head.pred.{ax}.dense"));
            prefixes.push(format!("bb.atom_head.pred.{ax}.out"));
        }

        let mut has_any_layer_bias = false;
        let mut base_weights = HashMap::with_capacity(prefixes.len());
        for pfx in prefixes {
            let (out_d, in_d, packed, bias) = bundle.dequantize_linear_f16_u32(&pfx)?;
            if pfx.starts_with("bb.layers.") && bias.is_some() {
                has_any_layer_bias = true;
            }
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&pfx),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
            base_weights.insert(
                pfx,
                GpuF16Weight {
                    out_d,
                    in_d,
                    pairs_per_row: (in_d + 1) / 2,
                    buf,
                    bias,
                },
            );
        }

        let mut norm_weights = HashMap::with_capacity(norm_names.len());
        for nn in norm_names {
            let sl = bundle.f32_slice(&nn)?;
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&nn),
                contents: bytemuck::cast_slice(sl),
                usage: wgpu::BufferUsages::STORAGE,
            });
            norm_weights.insert(nn, buf);
        }

        Ok(Self {
            device,
            queue,
            adapter_info,
            bgl_mm,
            pipe_mm_fwd,
            pipe_mm_bwd,
            bgl_el,
            pipe_lora_xa,
            pipe_lora_xb,
            pipe_rmsnorm,
            pipe_rope_qk,
            pipe_elem_op,
            bgl_mha_fwd,
            pipe_mha_fwd,
            bgl_mha_bwd,
            pipe_mha_bwd_dq,
            pipe_mha_bwd_dkv,
            dummy_f32,
            dummy_u32,
            dummy_rw1,
            base_weights,
            norm_weights,
            has_any_layer_bias,
        })
    }

    async fn readback_staging_f32(&self, staging: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
        let slice = staging.slice(..);
        let (tx, rx) = futures_oneshot();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await
            .map_err(|_| "wgpu map_async channel closed".to_string())?
            .map_err(|e| format!("wgpu map_async failed: {e}"))?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    async fn readback_f32(&self, src: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
        let nbytes = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: nbytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, nbytes);
        self.queue.submit(Some(enc.finish()));
        self.readback_staging_f32(&staging, count).await
    }

    #[inline]
    fn make_uniform(&self, p: GpuParams) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&p),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn record_rmsnorm(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        in_buf: &wgpu::Buffer,
        norm_name: &str,
        out_buf: &wgpu::Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        let nw_buf = self
            .norm_weights
            .get(norm_name)
            .ok_or_else(|| format!("Missing GPU norm weight '{norm_name}'"))?;
        let u_buf = self.make_uniform(GpuParams {
            p0: rows as u32,
            p1: cols as u32,
            p2: 0,
            p3: 0,
            p4: 0,
            p5: 0,
            f0: 1e-6,
            f1: 0.0,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_el,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: nw_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.dummy_f32.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.dummy_u32.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.dummy_rw1.as_entire_binding() },
            ],
        });
        pass.set_pipeline(&self.pipe_rmsnorm);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(((rows + 63) / 64) as u32, 1, 1);
        Ok(())
    }

    fn record_elem_op(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        mode: u32,
        b_sz: usize,
        max_s: usize,
        cols: usize,
        in0: &wgpu::Buffer,
        in1: &wgpu::Buffer,
        in2: &wgpu::Buffer,
        lens: &wgpu::Buffer,
        out0: &wgpu::Buffer,
    ) {
        let total = b_sz * max_s * cols;
        let u_buf = self.make_uniform(GpuParams {
            p0: mode,
            p1: b_sz as u32,
            p2: max_s as u32,
            p3: cols as u32,
            p4: 0,
            p5: 0,
            f0: 0.0,
            f1: 0.0,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_el,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: in0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: in1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: in2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: lens.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.dummy_rw1.as_entire_binding() },
            ],
        });
        pass.set_pipeline(&self.pipe_elem_op);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(((total + 255) / 256) as u32, 1, 1);
    }

    fn record_linear_with_lora_fwd(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        prefix: &str,
        x_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        lora_xa_buf: &wgpu::Buffer,
        lora_delta_buf: &wgpu::Buffer,
        rows: usize,
        lora_pair: Option<&LoraPair>,
        lora_scale_mult: f32,
    ) -> Result<(), String> {
        let gw = self
            .base_weights
            .get(prefix)
            .ok_or_else(|| format!("Missing GPU base weight '{prefix}'"))?;

        let active_lora = lora_pair.filter(|lp| {
            let sc = lp.scaling * lora_scale_mult;
            sc != 0.0 && lp.b.iter().any(|&v| v != 0.0)
        });

        let has_add = if let Some(lp) = active_lora {
            let sc = lp.scaling * lora_scale_mult;
            let a_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&lp.a),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let b_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&lp.b),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let u_xa = self.make_uniform(GpuParams {
                p0: rows as u32,
                p1: lp.rank as u32,
                p2: gw.in_d as u32,
                p3: 0,
                p4: 0,
                p5: 0,
                f0: 0.0,
                f1: 0.0,
            });
            let bg_xa = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl_el,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u_xa.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: a_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.dummy_f32.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.dummy_u32.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: lora_xa_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.dummy_rw1.as_entire_binding() },
                ],
            });
            pass.set_pipeline(&self.pipe_lora_xa);
            pass.set_bind_group(0, &bg_xa, &[]);
            pass.dispatch_workgroups(((rows * lp.rank + 63) / 64) as u32, 1, 1);

            let u_xb = self.make_uniform(GpuParams {
                p0: rows as u32,
                p1: gw.out_d as u32,
                p2: lp.rank as u32,
                p3: 0,
                p4: 0,
                p5: 0,
                f0: sc,
                f1: 0.0,
            });
            let bg_xb = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl_el,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u_xb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: lora_xa_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.dummy_f32.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.dummy_u32.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: lora_delta_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.dummy_rw1.as_entire_binding() },
                ],
            });
            pass.set_pipeline(&self.pipe_lora_xb);
            pass.set_bind_group(0, &bg_xb, &[]);
            pass.dispatch_workgroups(((rows * gw.out_d + 255) / 256) as u32, 1, 1);
            1u32
        } else {
            0u32
        };

        let u_mm = self.make_uniform(GpuParams {
            p0: rows as u32,
            p1: gw.out_d as u32,
            p2: gw.in_d as u32,
            p3: gw.pairs_per_row as u32,
            p4: has_add,
            p5: 0,
            f0: 0.0,
            f1: 0.0,
        });
        let add_ref = if has_add != 0 { lora_delta_buf } else { &self.dummy_f32 };
        let bg_mm = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_mm,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_mm.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gw.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: add_ref.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
            ],
        });
        pass.set_pipeline(&self.pipe_mm_fwd);
        pass.set_bind_group(0, &bg_mm, &[]);
        pass.dispatch_workgroups(((gw.out_d + 15) / 16) as u32, ((rows + 15) / 16) as u32, 1);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_transformer_block(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        l: usize,
        h_in_out: &wgpu::Buffer,
        h_mid_buf: &wgpu::Buffer,
        x_norm_buf: &wgpu::Buffer,
        u_norm_buf: &wgpu::Buffer,
        q_buf: &wgpu::Buffer,
        k_buf: &wgpu::Buffer,
        v_buf: &wgpu::Buffer,
        attn_cat_buf: &wgpu::Buffer,
        lse_buf: &wgpu::Buffer,
        proj_buf: &wgpu::Buffer,
        gate_buf: &wgpu::Buffer,
        up_buf: &wgpu::Buffer,
        swiglu_buf: &wgpu::Buffer,
        ffn_out_buf: &wgpu::Buffer,
        lora_xa_buf: &wgpu::Buffer,
        lora_delta_d_buf: &wgpu::Buffer,
        lora_delta_ffn_buf: &wgpu::Buffer,
        rope_cs_buf: &wgpu::Buffer,
        seq_lens_buf: &wgpu::Buffer,
        gb_l_buf: &wgpu::Buffer,
        b_sz: usize,
        max_seq: usize,
        n_heads: usize,
        hd: usize,
        ffn_d: usize,
        adapter: &LoomLoraAdapter,
        lora_scale_mult: f32,
    ) -> Result<(), String> {
        let rows = b_sz * max_seq;
        let d = n_heads * hd;
        let lp = format!("bb.layers.{l}");

        self.record_rmsnorm(pass, h_in_out, &format!("{lp}.attn_norm.weight"), x_norm_buf, rows, d)?;
        self.record_linear_with_lora_fwd(
            pass,
            &format!("{lp}.attn.q_proj"),
            x_norm_buf,
            q_buf,
            lora_xa_buf,
            lora_delta_d_buf,
            rows,
            adapter.loras.get(&format!("{lp}.attn.q_proj")),
            lora_scale_mult,
        )?;
        self.record_linear_with_lora_fwd(
            pass,
            &format!("{lp}.attn.k_proj"),
            x_norm_buf,
            k_buf,
            lora_xa_buf,
            lora_delta_d_buf,
            rows,
            adapter.loras.get(&format!("{lp}.attn.k_proj")),
            lora_scale_mult,
        )?;
        self.record_linear_with_lora_fwd(
            pass,
            &format!("{lp}.attn.v_proj"),
            x_norm_buf,
            v_buf,
            lora_xa_buf,
            lora_delta_d_buf,
            rows,
            adapter.loras.get(&format!("{lp}.attn.v_proj")),
            lora_scale_mult,
        )?;

        let u_rope = self.make_uniform(GpuParams {
            p0: rows as u32,
            p1: d as u32,
            p2: n_heads as u32,
            p3: hd as u32,
            p4: 0,
            p5: 0,
            f0: 0.0,
            f1: 0.0,
        });
        let bg_rope = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_el,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_rope.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rope_cs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.dummy_f32.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.dummy_f32.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.dummy_u32.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: k_buf.as_entire_binding() },
            ],
        });
        pass.set_pipeline(&self.pipe_rope_qk);
        pass.set_bind_group(0, &bg_rope, &[]);
        pass.dispatch_workgroups(((rows + 63) / 64) as u32, 1, 1);

        let total_mha_rows = b_sz * n_heads * max_seq;
        let u_mha = self.make_uniform(GpuParams {
            p0: b_sz as u32,
            p1: max_seq as u32,
            p2: n_heads as u32,
            p3: hd as u32,
            p4: 0,
            p5: 0,
            f0: 1.0 / (hd as f32).sqrt(),
            f1: 0.0,
        });
        let bg_mha = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_mha_fwd,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_mha.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: seq_lens_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: attn_cat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: lse_buf.as_entire_binding() },
            ],
        });
        pass.set_pipeline(&self.pipe_mha_fwd);
        pass.set_bind_group(0, &bg_mha, &[]);
        pass.dispatch_workgroups(((total_mha_rows + 63) / 64) as u32, 1, 1);

        self.record_linear_with_lora_fwd(
            pass,
            &format!("{lp}.attn.out_proj"),
            attn_cat_buf,
            proj_buf,
            lora_xa_buf,
            lora_delta_d_buf,
            rows,
            adapter.loras.get(&format!("{lp}.attn.out_proj")),
            lora_scale_mult,
        )?;
        self.record_elem_op(pass, 0, b_sz, max_seq, d, h_in_out, proj_buf, &self.dummy_f32, &self.dummy_u32, h_mid_buf);

        self.record_rmsnorm(pass, h_mid_buf, &format!("{lp}.ffn_norm.weight"), u_norm_buf, rows, d)?;
        self.record_linear_with_lora_fwd(
            pass,
            &format!("{lp}.ffn.w1"),
            u_norm_buf,
            gate_buf,
            lora_xa_buf,
            lora_delta_ffn_buf,
            rows,
            adapter.loras.get(&format!("{lp}.ffn.w1")),
            lora_scale_mult,
        )?;
        self.record_linear_with_lora_fwd(
            pass,
            &format!("{lp}.ffn.w3"),
            u_norm_buf,
            up_buf,
            lora_xa_buf,
            lora_delta_ffn_buf,
            rows,
            adapter.loras.get(&format!("{lp}.ffn.w3")),
            lora_scale_mult,
        )?;
        self.record_elem_op(pass, 1, b_sz, max_seq, ffn_d, gate_buf, up_buf, &self.dummy_f32, &self.dummy_u32, swiglu_buf);
        self.record_linear_with_lora_fwd(
            pass,
            &format!("{lp}.ffn.w2"),
            swiglu_buf,
            ffn_out_buf,
            lora_xa_buf,
            lora_delta_d_buf,
            rows,
            adapter.loras.get(&format!("{lp}.ffn.w2")),
            lora_scale_mult,
        )?;
        self.record_elem_op(pass, 2, b_sz, max_seq, d, h_mid_buf, ffn_out_buf, gb_l_buf, seq_lens_buf, h_in_out);
        Ok(())
    }

    /// Execute ALL 12 Transformer layers + final RMSNorm + `out_mod` in ONE single GPU `CommandEncoder` submission!
    #[allow(clippy::too_many_arguments)]
    async fn run_full_forward_12layers(
        &self,
        h0: &[f32],
        ropes: &[(Vec<f32>, Vec<f32>)],
        seq_lens: &[u32],
        cond_tape: &CondBatchTape,
        adapter: &LoomLoraAdapter,
        num_layers: usize,
        b_sz: usize,
        max_seq: usize,
        n_heads: usize,
        hd: usize,
        ffn_d: usize,
        lora_scale_mult: f32,
        save_checkpoints: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let rows = b_sz * max_seq;
        let d = n_heads * hd;
        let half = hd / 2;
        let mut rope_cs = vec![0.0f32; rows * hd];
        for r in 0..rows {
            rope_cs[r * hd..r * hd + half].copy_from_slice(&ropes[r].0);
            rope_cs[r * hd + half..(r + 1) * hd].copy_from_slice(&ropes[r].1);
        }

        let max_rank = adapter.loras.values().map(|lp| lp.rank).max().unwrap_or(8).max(1);
        let mk_rw = |elems: usize, src_copy: bool| {
            let mut u = wgpu::BufferUsages::STORAGE;
            if src_copy {
                u |= wgpu::BufferUsages::COPY_SRC;
            }
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: ((elems * 4) as u64).max(4),
                usage: u,
                mapped_at_creation: false,
            })
        };

        let h_a = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("h_a"),
            contents: bytemuck::cast_slice(h0),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let h_mid_buf = mk_rw(rows * d, true);
        let x_norm_buf = mk_rw(rows * d, true);
        let u_norm_buf = mk_rw(rows * d, false);
        let q_buf = mk_rw(rows * d, false);
        let k_buf = mk_rw(rows * d, false);
        let v_buf = mk_rw(rows * d, false);
        let attn_cat_buf = mk_rw(rows * d, false);
        let lse_buf = mk_rw(b_sz * n_heads * max_seq, false);
        let proj_buf = mk_rw(rows * d, false);
        let gate_buf = mk_rw(rows * ffn_d, false);
        let up_buf = mk_rw(rows * ffn_d, false);
        let swiglu_buf = mk_rw(rows * ffn_d, false);
        let lora_xa_buf = mk_rw(rows * max_rank, false);
        let lora_delta_d_buf = mk_rw(rows * d, false);
        let lora_delta_ffn_buf = mk_rw(rows * ffn_d, false);

        let rope_cs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&rope_cs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let seq_lens_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(seq_lens),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let gb_bufs: Vec<wgpu::Buffer> = (0..num_layers)
            .map(|l| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&cond_tape.layer_gamma_beta[l]),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        let out_gb_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&cond_tape.out_gamma_beta),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let rd_bytes = (rows * d * 4) as u64;
        let total_read_elems = if save_checkpoints {
            (num_layers + 3) * rows * d
        } else {
            rows * d
        };
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fwd12_staging"),
            size: ((total_read_elems * 4) as u64).max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fwd12") });
        for l in 0..num_layers {
            if save_checkpoints {
                enc.copy_buffer_to_buffer(&h_a, 0, &staging, (l as u64) * rd_bytes, rd_bytes);
            }
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            self.record_transformer_block(
                &mut pass,
                l,
                &h_a,
                &h_mid_buf,
                &x_norm_buf,
                &u_norm_buf,
                &q_buf,
                &k_buf,
                &v_buf,
                &attn_cat_buf,
                &lse_buf,
                &proj_buf,
                &gate_buf,
                &up_buf,
                &swiglu_buf,
                &proj_buf,
                &lora_xa_buf,
                &lora_delta_d_buf,
                &lora_delta_ffn_buf,
                &rope_cs_buf,
                &seq_lens_buf,
                &gb_bufs[l],
                b_sz,
                max_seq,
                n_heads,
                hd,
                ffn_d,
                adapter,
                lora_scale_mult,
            )?;
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            self.record_rmsnorm(&mut pass, &h_a, "bb.norm.weight", &x_norm_buf, rows, d)?;
            self.record_elem_op(
                &mut pass,
                3,
                b_sz,
                max_seq,
                d,
                &x_norm_buf,
                &self.dummy_f32,
                &out_gb_buf,
                &seq_lens_buf,
                &h_mid_buf,
            );
        }

        if save_checkpoints {
            let off_pre = (num_layers as u64) * rd_bytes;
            enc.copy_buffer_to_buffer(&h_a, 0, &staging, off_pre, rd_bytes);
            enc.copy_buffer_to_buffer(&x_norm_buf, 0, &staging, off_pre + rd_bytes, rd_bytes);
            enc.copy_buffer_to_buffer(&h_mid_buf, 0, &staging, off_pre + 2 * rd_bytes, rd_bytes);
        } else {
            enc.copy_buffer_to_buffer(&h_mid_buf, 0, &staging, 0, rd_bytes);
        }

        self.queue.submit(Some(enc.finish()));
        let flat = self.readback_staging_f32(&staging, total_read_elems).await?;

        let rd = rows * d;
        if save_checkpoints {
            let mut layer_inputs = Vec::with_capacity(num_layers);
            for l in 0..num_layers {
                layer_inputs.push(flat[l * rd..(l + 1) * rd].to_vec());
            }
            let final_pre_norm = flat[num_layers * rd..(num_layers + 1) * rd].to_vec();
            let final_normed = flat[(num_layers + 1) * rd..(num_layers + 2) * rd].to_vec();
            let hidden_out = flat[(num_layers + 2) * rd..(num_layers + 3) * rd].to_vec();
            let inv_d = 1.0 / (d as f32);
            let mut final_inv_rms = vec![0.0f32; rows];
            for r in 0..rows {
                let xr = &final_pre_norm[r * d..(r + 1) * d];
                let sq = xr.iter().map(|&v| v * v).sum::<f32>() * inv_d;
                final_inv_rms[r] = 1.0 / (sq + 1e-6).sqrt();
            }
            Ok((layer_inputs, final_pre_norm, final_inv_rms, final_normed, hidden_out))
        } else {
            Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), flat))
        }
    }

    /// Recompute all intermediates of layer `l` in ONE single GPU `CommandEncoder` submission!
    #[allow(clippy::too_many_arguments)]
    async fn recompute_single_layer_gpu(
        &self,
        l: usize,
        h_in: &[f32],
        rope_cs_buf: &wgpu::Buffer,
        seq_lens_buf: &wgpu::Buffer,
        gb_l: &[f32],
        adapter: &LoomLoraAdapter,
        b_sz: usize,
        max_seq: usize,
        n_heads: usize,
        hd: usize,
        ffn_d: usize,
        lora_scale_mult: f32,
    ) -> Result<SingleLayerRecompute, String> {
        let rows = b_sz * max_seq;
        let d = n_heads * hd;
        let mha_rows = b_sz * n_heads * max_seq;
        let max_rank = adapter.loras.values().map(|lp| lp.rank).max().unwrap_or(8).max(1);

        let mk_rw = |elems: usize, src_copy: bool| {
            let mut u = wgpu::BufferUsages::STORAGE;
            if src_copy {
                u |= wgpu::BufferUsages::COPY_SRC;
            }
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: ((elems * 4) as u64).max(4),
                usage: u,
                mapped_at_creation: false,
            })
        };

        let h_work = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(h_in),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let h_mid_buf = mk_rw(rows * d, true);
        let x_norm_buf = mk_rw(rows * d, true);
        let u_norm_buf = mk_rw(rows * d, true);
        let q_buf = mk_rw(rows * d, true);
        let k_buf = mk_rw(rows * d, true);
        let v_buf = mk_rw(rows * d, true);
        let attn_cat_buf = mk_rw(rows * d, true);
        let lse_buf = mk_rw(mha_rows, true);
        let proj_buf = mk_rw(rows * d, false);
        let gate_buf = mk_rw(rows * ffn_d, true);
        let up_buf = mk_rw(rows * ffn_d, true);
        let swiglu_buf = mk_rw(rows * ffn_d, true);
        let ffn_out_buf = mk_rw(rows * d, true);
        let lora_xa_buf = mk_rw(rows * max_rank, false);
        let lora_delta_d_buf = mk_rw(rows * d, false);
        let lora_delta_ffn_buf = mk_rw(rows * ffn_d, false);
        let gb_l_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(gb_l),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Pack 8 `(rows * d)` + 3 `(rows * ffn_d)` + 1 `(mha_rows)` into 1 staging readback!
        let rd = rows * d;
        let rf = rows * ffn_d;
        let total_elems = 8 * rd + 3 * rf + mha_rows;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((total_elems * 4) as u64).max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            self.record_transformer_block(
                &mut pass,
                l,
                &h_work,
                &h_mid_buf,
                &x_norm_buf,
                &u_norm_buf,
                &q_buf,
                &k_buf,
                &v_buf,
                &attn_cat_buf,
                &lse_buf,
                &proj_buf,
                &gate_buf,
                &up_buf,
                &swiglu_buf,
                &ffn_out_buf,
                &lora_xa_buf,
                &lora_delta_d_buf,
                &lora_delta_ffn_buf,
                rope_cs_buf,
                seq_lens_buf,
                &gb_l_buf,
                b_sz,
                max_seq,
                n_heads,
                hd,
                ffn_d,
                adapter,
                lora_scale_mult,
            )?;
        }

        let mut byte_off = 0u64;
        for (src, elems) in [
            (&x_norm_buf, rd),
            (&q_buf, rd),
            (&k_buf, rd),
            (&v_buf, rd),
            (&attn_cat_buf, rd),
            (&h_mid_buf, rd),
            (&u_norm_buf, rd),
            (&ffn_out_buf, rd),
            (&gate_buf, rf),
            (&up_buf, rf),
            (&swiglu_buf, rf),
            (&lse_buf, mha_rows),
        ] {
            let nb = (elems * 4) as u64;
            enc.copy_buffer_to_buffer(src, 0, &staging, byte_off, nb);
            byte_off += nb;
        }
        self.queue.submit(Some(enc.finish()));
        let flat = self.readback_staging_f32(&staging, total_elems).await?;

        let mut c = 0usize;
        let mut take = |n: usize| -> Vec<f32> {
            let v = flat[c..c + n].to_vec();
            c += n;
            v
        };
        let x_norm = take(rd);
        let q = take(rd);
        let k = take(rd);
        let v = take(rd);
        let attn_cat = take(rd);
        let h_mid = take(rd);
        let u_norm = take(rd);
        let ffn_out = take(rd);
        let gate_pre = take(rf);
        let up = take(rf);
        let swiglu = take(rf);
        let lse = take(mha_rows);

        let inv_d = 1.0 / (d as f32);
        let mut attn_inv_rms = vec![0.0f32; rows];
        let mut ffn_inv_rms = vec![0.0f32; rows];
        for r in 0..rows {
            let sq_a = h_in[r * d..(r + 1) * d].iter().map(|&x| x * x).sum::<f32>() * inv_d;
            attn_inv_rms[r] = 1.0 / (sq_a + 1e-6).sqrt();
            let sq_f = h_mid[r * d..(r + 1) * d].iter().map(|&x| x * x).sum::<f32>() * inv_d;
            ffn_inv_rms[r] = 1.0 / (sq_f + 1e-6).sqrt();
        }

        Ok(SingleLayerRecompute {
            x_norm,
            attn_inv_rms,
            q,
            k,
            v,
            attn_cat,
            lse,
            h_mid,
            u_norm,
            ffn_inv_rms,
            gate_pre,
            up,
            swiglu,
            ffn_out,
        })
    }

    /// Batched multi-projection `dX = sum_k (dY_k @ W_k) + dx_init` in ONE GPU submission!
    async fn run_linear_bwd_multi_sum_gpu(
        &self,
        specs: &[(&str, &[f32])],
        rows: usize,
        dx_init: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let first_gw = self
            .base_weights
            .get(specs[0].0)
            .ok_or_else(|| format!("Missing GPU base weight '{}'", specs[0].0))?;
        let in_d = first_gw.in_d;
        let out_len = rows * in_d;

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut cur_accum: Option<wgpu::Buffer> = dx_init.map(|ad| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(ad),
                usage: wgpu::BufferUsages::STORAGE,
            })
        });

        for &(pfx, dy) in specs {
            let gw = self
                .base_weights
                .get(pfx)
                .ok_or_else(|| format!("Missing GPU base weight '{pfx}'"))?;
            let u_buf = self.make_uniform(GpuParams {
                p0: rows as u32,
                p1: gw.out_d as u32,
                p2: gw.in_d as u32,
                p3: gw.pairs_per_row as u32,
                p4: if cur_accum.is_some() { 1 } else { 0 },
                p5: 0,
                f0: 0.0,
                f1: 0.0,
            });
            let dy_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(dy),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let next_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: ((out_len * 4) as u64).max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let add_ref = cur_accum.as_ref().unwrap_or(&self.dummy_f32);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl_mm,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dy_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: gw.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: add_ref.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: next_buf.as_entire_binding() },
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipe_mm_bwd);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(((gw.in_d + 15) / 16) as u32, ((rows + 15) / 16) as u32, 1);
            }
            cur_accum = Some(next_buf);
        }

        let final_buf = cur_accum.unwrap();
        let nbytes = (out_len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: nbytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&final_buf, 0, &staging, 0, nbytes);
        self.queue.submit(Some(enc.finish()));
        self.readback_staging_f32(&staging, out_len).await
    }

    pub async fn run_linear_fwd(
        &self,
        prefix: &str,
        x: &[f32],
        rows: usize,
        add_delta: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let gw = self
            .base_weights
            .get(prefix)
            .ok_or_else(|| format!("Missing GPU base weight '{prefix}'"))?;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let out_len = rows * gw.out_d;
        let u_buf = self.make_uniform(GpuParams {
            p0: rows as u32,
            p1: gw.out_d as u32,
            p2: gw.in_d as u32,
            p3: gw.pairs_per_row as u32,
            p4: if add_delta.is_some() { 1 } else { 0 },
            p5: 0,
            f0: 0.0,
            f1: 0.0,
        });
        let a_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let add_tmp;
        let add_ref = if let Some(ad) = add_delta {
            add_tmp = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(ad),
                usage: wgpu::BufferUsages::STORAGE,
            });
            &add_tmp
        } else {
            &self.dummy_f32
        };
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((out_len * 4) as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_mm,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gw.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: add_ref.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: c_buf.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipe_mm_fwd);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(((gw.out_d + 15) / 16) as u32, ((rows + 15) / 16) as u32, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let mut out = self.readback_f32(&c_buf, out_len).await?;
        if let Some(ref bv) = gw.bias {
            for r in 0..rows {
                let row = &mut out[r * gw.out_d..(r + 1) * gw.out_d];
                for c in 0..gw.out_d {
                    row[c] += bv[c];
                }
            }
        }
        Ok(out)
    }

    pub async fn run_linear_bwd_dx(
        &self,
        prefix: &str,
        dy: &[f32],
        rows: usize,
        dx_init: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        self.run_linear_bwd_multi_sum_gpu(&[(prefix, dy)], rows, dx_init)
            .await
    }

    pub async fn run_causal_mha_fwd(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_lens: &[u32],
        b_sz: usize,
        max_seq: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let total_rows = b_sz * n_heads * max_seq;
        let total_elems = b_sz * max_seq * n_heads * head_dim;
        let u_buf = self.make_uniform(GpuParams {
            p0: b_sz as u32,
            p1: max_seq as u32,
            p2: n_heads as u32,
            p3: head_dim as u32,
            p4: 0,
            p5: 0,
            f0: 1.0 / (head_dim as f32).sqrt(),
            f1: 0.0,
        });
        let q_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(q),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(k),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let len_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(seq_lens),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let o_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((total_elems * 4) as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let lse_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((total_rows * 4) as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_mha_fwd,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: len_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: o_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: lse_buf.as_entire_binding() },
            ],
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (((total_elems + total_rows) * 4) as u64).max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipe_mha_fwd);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(((total_rows + 63) / 64) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&o_buf, 0, &staging, 0, (total_elems * 4) as u64);
        enc.copy_buffer_to_buffer(&lse_buf, 0, &staging, (total_elems * 4) as u64, (total_rows * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        let both = self.readback_staging_f32(&staging, total_elems + total_rows).await?;
        Ok((both[..total_elems].to_vec(), both[total_elems..].to_vec()))
    }

    pub async fn run_causal_mha_bwd(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        o: &[f32],
        d_o: &[f32],
        lse: &[f32],
        seq_lens: &[u32],
        b_sz: usize,
        max_seq: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let total_rows = b_sz * n_heads * max_seq;
        let emb_d = n_heads * head_dim;
        let total_elems = b_sz * max_seq * emb_d;

        // Precompute `delta = dot(O, dO)` packed right after `lse`
        let mut lse_delta = Vec::with_capacity(2 * total_rows);
        lse_delta.extend_from_slice(lse);
        for b in 0..b_sz {
            let sl = seq_lens[b] as usize;
            for h in 0..n_heads {
                for qi in 0..max_seq {
                    if qi >= sl {
                        lse_delta.push(0.0);
                    } else {
                        let base = (b * max_seq + qi) * emb_d + h * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += o[base + d] * d_o[base + d];
                        }
                        lse_delta.push(dot);
                    }
                }
            }
        }

        let u_buf = self.make_uniform(GpuParams {
            p0: b_sz as u32,
            p1: max_seq as u32,
            p2: n_heads as u32,
            p3: head_dim as u32,
            p4: 0,
            p5: 0,
            f0: 1.0 / (head_dim as f32).sqrt(),
            f1: 0.0,
        });
        let q_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(q),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(k),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let do_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(d_o),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ld_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&lse_delta),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let len_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(seq_lens),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dq_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((total_elems * 4) as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dkv_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((2 * total_elems * 4) as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg_dq = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_mha_bwd,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: do_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ld_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: len_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: dq_buf.as_entire_binding() },
            ],
        });
        let bg_dkv = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_mha_bwd,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: do_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ld_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: len_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: dkv_buf.as_entire_binding() },
            ],
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((3 * total_elems * 4) as u64).max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipe_mha_bwd_dq);
            pass.set_bind_group(0, &bg_dq, &[]);
            pass.dispatch_workgroups(((total_rows + 63) / 64) as u32, 1, 1);
            pass.set_pipeline(&self.pipe_mha_bwd_dkv);
            pass.set_bind_group(0, &bg_dkv, &[]);
            pass.dispatch_workgroups(((total_rows + 63) / 64) as u32, 1, 1);
        }
        let eb = (total_elems * 4) as u64;
        enc.copy_buffer_to_buffer(&dq_buf, 0, &staging, 0, eb);
        enc.copy_buffer_to_buffer(&dkv_buf, 0, &staging, eb, 2 * eb);
        self.queue.submit(Some(enc.finish()));

        let all3 = self.readback_staging_f32(&staging, 3 * total_elems).await?;
        Ok((
            all3[..total_elems].to_vec(),
            all3[total_elems..2 * total_elems].to_vec(),
            all3[2 * total_elems..].to_vec(),
        ))
    }
}

/// Thread-safe (`Send + Sync`) oneshot channel compatible with both native `wgpu` and `wasm32`!
struct OneshotSender<T>(std::sync::Arc<std::sync::Mutex<OneshotInner<T>>>);
struct OneshotReceiver<T>(std::sync::Arc<std::sync::Mutex<OneshotInner<T>>>);
struct OneshotInner<T> {
    val: Option<T>,
    waker: Option<std::task::Waker>,
}
fn futures_oneshot<T>() -> (OneshotSender<T>, OneshotReceiver<T>) {
    let arc = std::sync::Arc::new(std::sync::Mutex::new(OneshotInner { val: None, waker: None }));
    (OneshotSender(arc.clone()), OneshotReceiver(arc))
}
impl<T> OneshotSender<T> {
    fn send(self, v: T) -> Result<(), ()> {
        let mut inner = self.0.lock().unwrap();
        inner.val = Some(v);
        if let Some(w) = inner.waker.take() {
            w.wake();
        }
        Ok(())
    }
}
impl<T> std::future::Future for OneshotReceiver<T> {
    type Output = Result<T, ()>;
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        let mut inner = self.0.lock().unwrap();
        if let Some(v) = inner.val.take() {
            std::task::Poll::Ready(Ok(v))
        } else {
            inner.waker = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}

// ============================================================================
// Batched Right-Padded Sequence Forward & Activation-Checkpointed Backward
// ============================================================================

#[inline]
fn rmsnorm_fwd_rows(x: &[f32], w: &[f32], rows: usize, d: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; rows * d];
    let mut inv_rms = vec![0.0f32; rows];
    let inv_d = 1.0 / (d as f32);
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let sq = xr.iter().map(|&v| v * v).sum::<f32>() * inv_d;
        let ir = 1.0 / (sq + eps).sqrt();
        inv_rms[r] = ir;
        let yr = &mut y[r * d..(r + 1) * d];
        for c in 0..d {
            yr[c] = xr[c] * ir * w[c];
        }
    }
    (y, inv_rms)
}

#[inline]
fn rmsnorm_bwd_rows(dy: &[f32], x: &[f32], inv_rms: &[f32], w: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; rows * d];
    let inv_d = 1.0 / (d as f32);
    for r in 0..rows {
        let dyr = &dy[r * d..(r + 1) * d];
        let xr = &x[r * d..(r + 1) * d];
        let ir = inv_rms[r];
        let mut dot = 0.0f32;
        for c in 0..d {
            dot += dyr[c] * w[c] * xr[c];
        }
        let coef = ir * ir * ir * inv_d * dot;
        let dxr = &mut dx[r * d..(r + 1) * d];
        for c in 0..d {
            dxr[c] = ir * dyr[c] * w[c] - xr[c] * coef;
        }
    }
    dx
}

/// Evaluate `Y = X @ W_base^T + lora_scale * LoRA(X)` on either `wgpu` or CPU (`Q4/Q8/F16/F32`)!
pub async fn eval_base_plus_lora_fwd(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    prefix: &str,
    x: &[f32],
    rows: usize,
    out_dim: usize,
    lora_scale_mult: f32,
) -> Result<Vec<f32>, String> {
    let delta_opt = adapter
        .loras
        .get(prefix)
        .filter(|_| lora_scale_mult != 0.0)
        .map(|lp| lp.forward_delta(x, rows, lora_scale_mult));

    if let Some(g) = gpu {
        if rows >= 32 && g.base_weights.contains_key(prefix) {
            return g.run_linear_fwd(prefix, x, rows, delta_opt.as_deref()).await;
        }
    }
    let mut out = vec![0.0f32; rows * out_dim];
    bundle.eval_linear(prefix, x, rows, &mut out)?;
    if let Some(dl) = delta_opt {
        for i in 0..out.len() {
            out[i] += dl[i];
        }
    }
    Ok(out)
}

pub async fn eval_base_plus_lora_bwd(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    prefix: &str,
    x: &[f32],
    dy: &[f32],
    rows: usize,
    in_dim: usize,
    lora_scale_mult: f32,
    grads: Option<&mut LoomAdapterGrads>,
) -> Result<Vec<f32>, String> {
    let mut dx_lora = vec![0.0f32; rows * in_dim];
    if let Some(lp) = adapter.loras.get(prefix) {
        let lg_opt = grads.and_then(|g| g.loras.get_mut(prefix));
        lp.backward_accum(x, dy, rows, lora_scale_mult, &mut dx_lora, lg_opt);
    }

    if let Some(g) = gpu {
        if rows >= 32 && g.base_weights.contains_key(prefix) {
            return g.run_linear_bwd_dx(prefix, dy, rows, Some(&dx_lora)).await;
        }
    }
    bundle.eval_linear_bwd_dx(prefix, dy, rows, &mut dx_lora)?;
    Ok(dx_lora)
}

fn cpu_causal_mha_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_lens: &[u32],
    b_sz: usize,
    max_seq: usize,
    n_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let emb_d = n_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; b_sz * max_seq * emb_d];
    let mut lse = vec![0.0f32; b_sz * n_heads * max_seq];

    for b in 0..b_sz {
        let sl = seq_lens[b] as usize;
        for qi in 0..sl {
            for h in 0..n_heads {
                let qb = (b * max_seq + qi) * emb_d + h * head_dim;
                let qs = &q[qb..qb + head_dim];
                let mut m_cur = -1.0e30f32;
                let mut l_cur = 0.0f32;
                let os = &mut out[qb..qb + head_dim];
                for kj in 0..=qi {
                    let kb = (b * max_seq + kj) * emb_d + h * head_dim;
                    let ks = &k[kb..kb + head_dim];
                    let vs = &v[kb..kb + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += qs[d] * ks[d];
                    }
                    let s = dot * scale;
                    let m_new = m_cur.max(s);
                    let ep = (m_cur - m_new).exp();
                    let es = (s - m_new).exp();
                    l_cur = l_cur * ep + es;
                    for d in 0..head_dim {
                        os[d] = os[d] * ep + es * vs[d];
                    }
                    m_cur = m_new;
                }
                let inv_l = 1.0 / l_cur.max(1e-20);
                for d in 0..head_dim {
                    os[d] *= inv_l;
                }
                lse[(b * n_heads + h) * max_seq + qi] = m_cur + l_cur.max(1e-20).ln();
            }
        }
    }
    (out, lse)
}

fn cpu_causal_mha_bwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &[f32],
    d_o: &[f32],
    lse: &[f32],
    seq_lens: &[u32],
    b_sz: usize,
    max_seq: usize,
    n_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let emb_d = n_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut dq = vec![0.0f32; b_sz * max_seq * emb_d];
    let mut dk = vec![0.0f32; b_sz * max_seq * emb_d];
    let mut dv = vec![0.0f32; b_sz * max_seq * emb_d];

    for b in 0..b_sz {
        let sl = seq_lens[b] as usize;
        for h in 0..n_heads {
            for qi in 0..sl {
                let qb = (b * max_seq + qi) * emb_d + h * head_dim;
                let lse_i = lse[(b * n_heads + h) * max_seq + qi];
                let mut delta = 0.0f32;
                for d in 0..head_dim {
                    delta += o[qb + d] * d_o[qb + d];
                }
                for kj in 0..=qi {
                    let kb = (b * max_seq + kj) * emb_d + h * head_dim;
                    let mut qk = 0.0f32;
                    let mut dov = 0.0f32;
                    for d in 0..head_dim {
                        qk += q[qb + d] * k[kb + d];
                        dov += d_o[qb + d] * v[kb + d];
                    }
                    let p = (qk * scale - lse_i).exp();
                    let ds = p * (dov - delta) * scale;
                    for d in 0..head_dim {
                        dv[kb + d] += p * d_o[qb + d];
                        dq[qb + d] += ds * k[kb + d];
                        dk[kb + d] += ds * q[qb + d];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}

pub struct SeqForwardTape {
    pub b_sz: usize,
    pub max_seq: usize,
    pub seq_lens: Vec<u32>,
    pub ropes: Vec<(Vec<f32>, Vec<f32>)>, // [b_sz * max_seq]
    pub cond_tape: CondBatchTape,
    /// Per-layer input residual states `[num_layers][b_sz * max_seq * emb_dim]` (Activation Checkpointing!)
    pub layer_inputs: Vec<Vec<f32>>,
    pub final_pre_norm: Vec<f32>,
    pub final_inv_rms: Vec<f32>,
    pub final_normed: Vec<f32>,
    /// Post-`out_mod` hidden states `[b_sz * max_seq * emb_dim]`
    pub hidden_out: Vec<f32>,
}

/// Batched right-padded causal sequence forward pass (`[B, max_seq, emb_dim]`).
/// When `save_checkpoints == false` (e.g. `no_grad` reference KL or Stage 2.5B detached XYZ distill),
/// per-layer input checkpoints are NOT saved, reducing memory even further!
pub async fn forward_sequence_batch(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    seqs: &[EncodedOctreeSeq],
    raw_targets: &[Vec<f32>],
    active_masks: &[Vec<bool>],
    lora_scale_mult: f32,
    save_checkpoints: bool,
) -> Result<SeqForwardTape, String> {
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let n_heads = bb.num_head;
    let hd = bb.head_dim;
    let ffn_d = bb.ffn_dim;
    let b_sz = seqs.len();
    let max_seq = seqs.iter().map(|s| s.tokens.len()).max().unwrap_or(1).max(1);
    let rows = b_sz * max_seq;

    let cond_tape = adapter.conditioner.forward_batch(raw_targets, active_masks);

    let mut seq_lens = vec![0u32; b_sz];
    let mut h = vec![0.0f32; rows * d];
    let mut ropes = vec![(vec![1.0f32; hd / 2], vec![0.0f32; hd / 2]); rows];

    for b in 0..b_sz {
        let toks = &seqs[b].tokens;
        seq_lens[b] = toks.len() as u32;
        let root_d = &cond_tape.root_delta[b * d..(b + 1) * d];
        for (s, tok) in toks.iter().enumerate() {
            let r_idx = b * max_seq + s;
            let emb = embed_octree_token(
                bundle,
                tok.typ,
                tok.xyz_bins,
                tok.level,
                tok.rem_atoms,
                if tok.is_root { Some(root_d) } else { None },
            )?;
            h[r_idx * d..(r_idx + 1) * d].copy_from_slice(&emb);
            ropes[r_idx] = compute_rope_3d(tok.phy_pos, bb.xyz_resolution, hd, bb.rope_theta);
        }
    }

    if let Some(g) = gpu {
        if !g.has_any_layer_bias {
            let (layer_inputs, final_pre_norm, final_inv_rms, final_normed, hidden_out) = g
                .run_full_forward_12layers(
                    &h,
                    &ropes,
                    &seq_lens,
                    &cond_tape,
                    adapter,
                    bb.num_layers,
                    b_sz,
                    max_seq,
                    n_heads,
                    hd,
                    ffn_d,
                    lora_scale_mult,
                    save_checkpoints,
                )
                .await?;
            return Ok(SeqForwardTape {
                b_sz,
                max_seq,
                seq_lens,
                ropes,
                cond_tape,
                layer_inputs,
                final_pre_norm,
                final_inv_rms,
                final_normed,
                hidden_out,
            });
        }
    }

    let mut layer_inputs = Vec::with_capacity(if save_checkpoints { bb.num_layers } else { 0 });

    for l in 0..bb.num_layers {
        if save_checkpoints {
            layer_inputs.push(h.clone());
        }
        let lp = format!("bb.layers.{l}");
        let attn_nw = bundle.f32_slice(&format!("{lp}.attn_norm.weight"))?;
        let (x_norm, _) = rmsnorm_fwd_rows(&h, attn_nw, rows, d, 1e-6);

        let mut q = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.attn.q_proj"), &x_norm, rows, d, lora_scale_mult).await?;
        let mut k = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.attn.k_proj"), &x_norm, rows, d, lora_scale_mult).await?;
        let v = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.attn.v_proj"), &x_norm, rows, d, lora_scale_mult).await?;

        for b in 0..b_sz {
            for s in 0..(seq_lens[b] as usize) {
                let r_idx = b * max_seq + s;
                apply_rope_in_place(&mut q[r_idx * d..(r_idx + 1) * d], n_heads, hd, &ropes[r_idx].0, &ropes[r_idx].1);
                apply_rope_in_place(&mut k[r_idx * d..(r_idx + 1) * d], n_heads, hd, &ropes[r_idx].0, &ropes[r_idx].1);
            }
        }

        let (attn_cat, _) = if let Some(g) = gpu {
            g.run_causal_mha_fwd(&q, &k, &v, &seq_lens, b_sz, max_seq, n_heads, hd).await?
        } else {
            cpu_causal_mha_fwd(&q, &k, &v, &seq_lens, b_sz, max_seq, n_heads, hd)
        };

        let attn_proj = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.attn.out_proj"), &attn_cat, rows, d, lora_scale_mult).await?;
        for i in 0..(rows * d) {
            h[i] += attn_proj[i];
        }

        let ffn_nw = bundle.f32_slice(&format!("{lp}.ffn_norm.weight"))?;
        let (u_norm, _) = rmsnorm_fwd_rows(&h, ffn_nw, rows, d, 1e-6);
        let gate_pre = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.ffn.w1"), &u_norm, rows, ffn_d, lora_scale_mult).await?;
        let up = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.ffn.w3"), &u_norm, rows, ffn_d, lora_scale_mult).await?;
        let swiglu: Vec<f32> = gate_pre
            .iter()
            .zip(up.iter())
            .map(|(&gv, &uv)| silu(gv) * uv)
            .collect();
        let ffn_out = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.ffn.w2"), &swiglu, rows, d, lora_scale_mult).await?;

        // Post-block AdaLN-Zero modulation: `x = (x + ffn_out) * (1 + gamma) + beta`
        let gb_l = &cond_tape.layer_gamma_beta[l];
        for b in 0..b_sz {
            let sl = seq_lens[b] as usize;
            let gamma = &gb_l[b * 2 * d..b * 2 * d + d];
            let beta = &gb_l[b * 2 * d + d..(b + 1) * 2 * d];
            for s in 0..sl {
                let row = &mut h[(b * max_seq + s) * d..(b * max_seq + s + 1) * d];
                let fo = &ffn_out[(b * max_seq + s) * d..(b * max_seq + s + 1) * d];
                for c in 0..d {
                    row[c] = (row[c] + fo[c]) * (1.0 + gamma[c]) + beta[c];
                }
            }
            for s in sl..max_seq {
                h[(b * max_seq + s) * d..(b * max_seq + s + 1) * d].fill(0.0);
            }
        }
    }

    let final_nw = bundle.f32_slice("bb.norm.weight")?;
    let (final_normed, final_inv_rms) = rmsnorm_fwd_rows(&h, final_nw, rows, d, 1e-6);
    let mut hidden_out = vec![0.0f32; rows * d];
    for b in 0..b_sz {
        let sl = seq_lens[b] as usize;
        let gamma = &cond_tape.out_gamma_beta[b * 2 * d..b * 2 * d + d];
        let beta = &cond_tape.out_gamma_beta[b * 2 * d + d..(b + 1) * 2 * d];
        for s in 0..sl {
            let src = &final_normed[(b * max_seq + s) * d..(b * max_seq + s + 1) * d];
            let dst = &mut hidden_out[(b * max_seq + s) * d..(b * max_seq + s + 1) * d];
            for c in 0..d {
                dst[c] = src[c] * (1.0 + gamma[c]) + beta[c];
            }
        }
    }

    Ok(SeqForwardTape {
        b_sz,
        max_seq,
        seq_lens,
        ropes,
        cond_tape,
        layer_inputs,
        final_pre_norm: h,
        final_inv_rms,
        final_normed,
        hidden_out,
    })
}

/// Activation-checkpointed reverse-mode VJP through `out_mod -> final RMSNorm -> 12 Transformer Blocks -> LoomConditioner`!
pub async fn backward_sequence_batch(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    tape: &SeqForwardTape,
    d_hidden_out: &[f32],
    lora_scale_mult: f32,
    grads: &mut LoomAdapterGrads,
) -> Result<(), String> {
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let n_heads = bb.num_head;
    let hd = bb.head_dim;
    let ffn_d = bb.ffn_dim;
    let b_sz = tape.b_sz;
    let max_seq = tape.max_seq;
    let rows = b_sz * max_seq;

    // 1. Final `out_mod` + `bb.norm` VJP
    let mut d_out_gb = vec![0.0f32; b_sz * 2 * d];
    let mut d_final_normed = vec![0.0f32; rows * d];
    for b in 0..b_sz {
        let sl = tape.seq_lens[b] as usize;
        let gamma = &tape.cond_tape.out_gamma_beta[b * 2 * d..b * 2 * d + d];
        let (d_gm, d_bt) = d_out_gb[b * 2 * d..(b + 1) * 2 * d].split_at_mut(d);
        for s in 0..sl {
            let dy = &d_hidden_out[(b * max_seq + s) * d..(b * max_seq + s + 1) * d];
            let xn = &tape.final_normed[(b * max_seq + s) * d..(b * max_seq + s + 1) * d];
            let dxn = &mut d_final_normed[(b * max_seq + s) * d..(b * max_seq + s + 1) * d];
            for c in 0..d {
                dxn[c] = dy[c] * (1.0 + gamma[c]);
                d_gm[c] += dy[c] * xn[c];
                d_bt[c] += dy[c];
            }
        }
    }
    let final_nw = bundle.f32_slice("bb.norm.weight")?;
    let mut dh = rmsnorm_bwd_rows(&d_final_normed, &tape.final_pre_norm, &tape.final_inv_rms, final_nw, rows, d);

    // 2. Reverse loop over `l = num_layers - 1 .. 0` with single-layer recomputation
    let mut d_layer_gb = vec![vec![0.0f32; b_sz * 2 * d]; bb.num_layers];

    // Pre-upload RoPE + seq_lens buffers once per backward pass when running on GPU!
    let gpu_bwd_aux = gpu.filter(|g| !g.has_any_layer_bias).map(|g| {
        let half = hd / 2;
        let mut rope_cs = vec![0.0f32; rows * hd];
        for r in 0..rows {
            rope_cs[r * hd..r * hd + half].copy_from_slice(&tape.ropes[r].0);
            rope_cs[r * hd + half..(r + 1) * hd].copy_from_slice(&tape.ropes[r].1);
        }
        let r_buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&rope_cs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let s_buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&tape.seq_lens),
            usage: wgpu::BufferUsages::STORAGE,
        });
        (g, r_buf, s_buf)
    });

    for l in (0..bb.num_layers).rev() {
        let h_in = &tape.layer_inputs[l];
        let lp = format!("bb.layers.{l}");
        let attn_nw = bundle.f32_slice(&format!("{lp}.attn_norm.weight"))?;
        let ffn_nw = bundle.f32_slice(&format!("{lp}.ffn_norm.weight"))?;
        let gb_l = &tape.cond_tape.layer_gamma_beta[l];

        // Recompute single-layer forward activations from `h_in` (1 fused GPU submission or CPU fallback)
        let rec = if let Some((g, ref r_buf, ref s_buf)) = gpu_bwd_aux {
            g.recompute_single_layer_gpu(
                l,
                h_in,
                r_buf,
                s_buf,
                gb_l,
                adapter,
                b_sz,
                max_seq,
                n_heads,
                hd,
                ffn_d,
                lora_scale_mult,
            )
            .await?
        } else {
            let (x_norm, attn_inv_rms) = rmsnorm_fwd_rows(h_in, attn_nw, rows, d, 1e-6);
            let mut q = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.attn.q_proj"), &x_norm, rows, d, lora_scale_mult).await?;
            let mut k = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.attn.k_proj"), &x_norm, rows, d, lora_scale_mult).await?;
            let v = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.attn.v_proj"), &x_norm, rows, d, lora_scale_mult).await?;
            for b in 0..b_sz {
                for s in 0..(tape.seq_lens[b] as usize) {
                    let r_idx = b * max_seq + s;
                    apply_rope_in_place(&mut q[r_idx * d..(r_idx + 1) * d], n_heads, hd, &tape.ropes[r_idx].0, &tape.ropes[r_idx].1);
                    apply_rope_in_place(&mut k[r_idx * d..(r_idx + 1) * d], n_heads, hd, &tape.ropes[r_idx].0, &tape.ropes[r_idx].1);
                }
            }
            let (attn_cat, lse) = cpu_causal_mha_fwd(&q, &k, &v, &tape.seq_lens, b_sz, max_seq, n_heads, hd);
            let attn_proj = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.attn.out_proj"), &attn_cat, rows, d, lora_scale_mult).await?;
            let mut h_mid = vec![0.0f32; rows * d];
            for i in 0..(rows * d) {
                h_mid[i] = h_in[i] + attn_proj[i];
            }
            let (u_norm, ffn_inv_rms) = rmsnorm_fwd_rows(&h_mid, ffn_nw, rows, d, 1e-6);
            let gate_pre = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.ffn.w1"), &u_norm, rows, ffn_d, lora_scale_mult).await?;
            let up = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.ffn.w3"), &u_norm, rows, ffn_d, lora_scale_mult).await?;
            let mut swiglu = vec![0.0f32; rows * ffn_d];
            for i in 0..(rows * ffn_d) {
                swiglu[i] = silu(gate_pre[i]) * up[i];
            }
            let ffn_out = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{lp}.ffn.w2"), &swiglu, rows, d, lora_scale_mult).await?;
            SingleLayerRecompute {
                x_norm,
                attn_inv_rms,
                q,
                k,
                v,
                attn_cat,
                lse,
                h_mid,
                u_norm,
                ffn_inv_rms,
                gate_pre,
                up,
                swiglu,
                ffn_out,
            }
        };

        // --- Backward through post-block AdaLN-Zero modulation ---
        let mut d_post_ffn = vec![0.0f32; rows * d];
        for b in 0..b_sz {
            let sl = tape.seq_lens[b] as usize;
            let gamma = &gb_l[b * 2 * d..b * 2 * d + d];
            let (d_gm, d_bt) = d_layer_gb[l][b * 2 * d..(b + 1) * 2 * d].split_at_mut(d);
            for s in 0..sl {
                let r_off = (b * max_seq + s) * d;
                for c in 0..d {
                    let pre_mod = rec.h_mid[r_off + c] + rec.ffn_out[r_off + c];
                    let g = dh[r_off + c];
                    d_post_ffn[r_off + c] = g * (1.0 + gamma[c]);
                    d_gm[c] += g * pre_mod;
                    d_bt[c] += g;
                }
            }
        }

        // --- Backward through SwiGLU FFN ---
        let d_swiglu = eval_base_plus_lora_bwd(
            bundle,
            gpu,
            adapter,
            &format!("{lp}.ffn.w2"),
            &rec.swiglu,
            &d_post_ffn,
            rows,
            ffn_d,
            lora_scale_mult,
            Some(grads),
        )
        .await?;

        let mut d_gate_pre = vec![0.0f32; rows * ffn_d];
        let mut d_up = vec![0.0f32; rows * ffn_d];
        for i in 0..(rows * ffn_d) {
            let g = d_swiglu[i];
            d_gate_pre[i] = g * rec.up[i] * silu_grad(rec.gate_pre[i]);
            d_up[i] = g * silu(rec.gate_pre[i]);
        }

        let k_w1 = format!("{lp}.ffn.w1");
        let k_w3 = format!("{lp}.ffn.w3");
        let du_norm = if let Some((g, _, _)) = gpu_bwd_aux {
            if rows >= 32 {
                let mut du_lora = vec![0.0f32; rows * d];
                if let Some(lp1) = adapter.loras.get(&k_w1) {
                    lp1.backward_accum(&rec.u_norm, &d_gate_pre, rows, lora_scale_mult, &mut du_lora, grads.loras.get_mut(&k_w1));
                }
                if let Some(lp3) = adapter.loras.get(&k_w3) {
                    lp3.backward_accum(&rec.u_norm, &d_up, rows, lora_scale_mult, &mut du_lora, grads.loras.get_mut(&k_w3));
                }
                g.run_linear_bwd_multi_sum_gpu(
                    &[(&k_w1, &d_gate_pre), (&k_w3, &d_up)],
                    rows,
                    Some(&du_lora),
                )
                .await?
            } else {
                let du_1 = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_w1, &rec.u_norm, &d_gate_pre, rows, d, lora_scale_mult, Some(grads)).await?;
                let du_3 = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_w3, &rec.u_norm, &d_up, rows, d, lora_scale_mult, Some(grads)).await?;
                du_1.iter().zip(du_3.iter()).map(|(&a, &b)| a + b).collect()
            }
        } else {
            let du_1 = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_w1, &rec.u_norm, &d_gate_pre, rows, d, lora_scale_mult, Some(grads)).await?;
            let du_3 = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_w3, &rec.u_norm, &d_up, rows, d, lora_scale_mult, Some(grads)).await?;
            du_1.iter().zip(du_3.iter()).map(|(&a, &b)| a + b).collect()
        };

        let mut d_h_mid = rmsnorm_bwd_rows(&du_norm, &rec.h_mid, &rec.ffn_inv_rms, ffn_nw, rows, d);
        for i in 0..(rows * d) {
            d_h_mid[i] += d_post_ffn[i];
        }

        // --- Backward through Causal Self-Attention ---
        let d_attn_cat = eval_base_plus_lora_bwd(
            bundle,
            gpu,
            adapter,
            &format!("{lp}.attn.out_proj"),
            &rec.attn_cat,
            &d_h_mid,
            rows,
            d,
            lora_scale_mult,
            Some(grads),
        )
        .await?;

        let (mut dq, mut dk, dv) = if let Some(g) = gpu {
            g.run_causal_mha_bwd(
                &rec.q,
                &rec.k,
                &rec.v,
                &rec.attn_cat,
                &d_attn_cat,
                &rec.lse,
                &tape.seq_lens,
                b_sz,
                max_seq,
                n_heads,
                hd,
            )
            .await?
        } else {
            cpu_causal_mha_bwd(
                &rec.q,
                &rec.k,
                &rec.v,
                &rec.attn_cat,
                &d_attn_cat,
                &rec.lse,
                &tape.seq_lens,
                b_sz,
                max_seq,
                n_heads,
                hd,
            )
        };

        for b in 0..b_sz {
            for s in 0..(tape.seq_lens[b] as usize) {
                let r_idx = b * max_seq + s;
                apply_rope_bwd_in_place(&mut dq[r_idx * d..(r_idx + 1) * d], n_heads, hd, &tape.ropes[r_idx].0, &tape.ropes[r_idx].1);
                apply_rope_bwd_in_place(&mut dk[r_idx * d..(r_idx + 1) * d], n_heads, hd, &tape.ropes[r_idx].0, &tape.ropes[r_idx].1);
            }
        }

        let k_q = format!("{lp}.attn.q_proj");
        let k_k = format!("{lp}.attn.k_proj");
        let k_v = format!("{lp}.attn.v_proj");
        let dx_norm = if let Some((g, _, _)) = gpu_bwd_aux {
            if rows >= 32 {
                let mut dx_lora = vec![0.0f32; rows * d];
                if let Some(lpq) = adapter.loras.get(&k_q) {
                    lpq.backward_accum(&rec.x_norm, &dq, rows, lora_scale_mult, &mut dx_lora, grads.loras.get_mut(&k_q));
                }
                if let Some(lpk) = adapter.loras.get(&k_k) {
                    lpk.backward_accum(&rec.x_norm, &dk, rows, lora_scale_mult, &mut dx_lora, grads.loras.get_mut(&k_k));
                }
                if let Some(lpv) = adapter.loras.get(&k_v) {
                    lpv.backward_accum(&rec.x_norm, &dv, rows, lora_scale_mult, &mut dx_lora, grads.loras.get_mut(&k_v));
                }
                g.run_linear_bwd_multi_sum_gpu(
                    &[(&k_q, &dq), (&k_k, &dk), (&k_v, &dv)],
                    rows,
                    Some(&dx_lora),
                )
                .await?
            } else {
                let dxn_q = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_q, &rec.x_norm, &dq, rows, d, lora_scale_mult, Some(grads)).await?;
                let dxn_k = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_k, &rec.x_norm, &dk, rows, d, lora_scale_mult, Some(grads)).await?;
                let dxn_v = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_v, &rec.x_norm, &dv, rows, d, lora_scale_mult, Some(grads)).await?;
                (0..rows * d).map(|i| dxn_q[i] + dxn_k[i] + dxn_v[i]).collect()
            }
        } else {
            let dxn_q = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_q, &rec.x_norm, &dq, rows, d, lora_scale_mult, Some(grads)).await?;
            let dxn_k = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_k, &rec.x_norm, &dk, rows, d, lora_scale_mult, Some(grads)).await?;
            let dxn_v = eval_base_plus_lora_bwd(bundle, gpu, adapter, &k_v, &rec.x_norm, &dv, rows, d, lora_scale_mult, Some(grads)).await?;
            (0..rows * d).map(|i| dxn_q[i] + dxn_k[i] + dxn_v[i]).collect()
        };

        let mut d_h_in = rmsnorm_bwd_rows(&dx_norm, h_in, &rec.attn_inv_rms, attn_nw, rows, d);
        for i in 0..(rows * d) {
            d_h_in[i] += d_h_mid[i];
        }
        dh = d_h_in;
    }

    // 3. Root token `0` delta gradient -> `LoomConditioner` VJP
    if let Some(ref mut cg) = grads.cond {
        let mut d_root_delta = vec![0.0f32; b_sz * d];
        for b in 0..b_sz {
            if tape.seq_lens[b] > 0 {
                d_root_delta[b * d..(b + 1) * d].copy_from_slice(&dh[(b * max_seq) * d..(b * max_seq + 1) * d]);
            }
        }
        adapter
            .conditioner
            .backward_batch(&tape.cond_tape, &d_root_delta, &d_layer_gb, &d_out_gb, cg);
    }

    Ok(())
}

// ============================================================================
// TaskHead (`Linear -> GELU -> Linear`) Forward & Backward Helper
// ============================================================================

pub async fn eval_task_head_with_lora_fwd(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    prefix: &str,
    h: &[f32],
    rows: usize,
    out_dim: usize,
    lora_scale_mult: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let d = bundle.header.backbone.emb_dim;
    let mid_pre = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{prefix}.dense"), h, rows, d, lora_scale_mult).await?;
    let mid_act: Vec<f32> = mid_pre.iter().map(|&v| gelu_exact(v)).collect();
    let logits = eval_base_plus_lora_fwd(bundle, gpu, adapter, &format!("{prefix}.out"), &mid_act, rows, out_dim, lora_scale_mult).await?;
    Ok((logits, mid_pre, mid_act))
}

pub async fn eval_task_head_with_lora_bwd(
    bundle: &ModelBundle,
    gpu: Option<&LoomGpuTrainContext>,
    adapter: &LoomLoraAdapter,
    prefix: &str,
    h: &[f32],
    mid_pre: &[f32],
    mid_act: &[f32],
    d_logits: &[f32],
    rows: usize,
    lora_scale_mult: f32,
    mut grads: Option<&mut LoomAdapterGrads>,
) -> Result<Vec<f32>, String> {
    let d = bundle.header.backbone.emb_dim;
    let d_mid_act = eval_base_plus_lora_bwd(
        bundle,
        gpu,
        adapter,
        &format!("{prefix}.out"),
        mid_act,
        d_logits,
        rows,
        d,
        lora_scale_mult,
        grads.as_deref_mut(),
    )
    .await?;
    let d_mid_pre: Vec<f32> = d_mid_act
        .iter()
        .zip(mid_pre.iter())
        .map(|(&g, &x)| g * gelu_grad(x))
        .collect();
    eval_base_plus_lora_bwd(
        bundle,
        gpu,
        adapter,
        &format!("{prefix}.dense"),
        h,
        &d_mid_pre,
        rows,
        d,
        lora_scale_mult,
        grads,
    )
    .await
}

// ============================================================================
// LoRA-Aware Rollout Trajectory Sampler (for Inference & Online `TopoGRPO`!)
// ============================================================================

#[derive(Debug, Clone)]
pub struct RolloutOptions {
    pub cfg_scale: f32,
    pub xyz_cfg_scale: f32,
    pub cfg_tree_decay: f32,
    pub temperatures: [f32; 3],
    pub geometric_guard: bool,
    pub even_electron_parity: bool,
    pub reconcile_valence: bool,
    pub max_decode_steps: usize,
    pub seed: u64,
}

impl Default for RolloutOptions {
    fn default() -> Self {
        Self {
            cfg_scale: 1.75,
            xyz_cfg_scale: 1.0,
            cfg_tree_decay: 0.25,
            temperatures: [0.85, 0.30, 0.25],
            geometric_guard: true,
            even_electron_parity: true,
            reconcile_valence: true,
            max_decode_steps: 240,
            seed: 42,
        }
    }
}

/// Apply real-time 3D steric clash & covalent coordination guard bias onto element logits `(602,)`
/// matching `_element_guard_logit_bias` in `uni3dar_slot.py`!
pub fn apply_element_geometric_guard(
    logits: &mut [f32],
    voxel_center: [f32; 3],
    prior_elems: &[usize],
    prior_pos: &[[f32; 3]],
    prior_coord_deg: &[i32],
    atom_null_id: usize,
) {
    if prior_elems.is_empty() {
        return;
    }
    let mut min_d = f32::INFINITY;
    for p in prior_pos {
        let dx = voxel_center[0] - p[0];
        let dy = voxel_center[1] - p[1];
        let dz = voxel_center[2] - p[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d < min_d {
            min_d = d;
        }
    }
    // Extreme steric overlap (`< 0.62 Å`): force `[NULL]` if allowed
    if min_d < 0.62 && atom_null_id < logits.len() && logits[atom_null_id].is_finite() {
        for (eid, l) in logits.iter_mut().enumerate() {
            if eid != atom_null_id {
                *l = f32::NEG_INFINITY;
            }
        }
        return;
    }

    for eid in 0..118.min(logits.len()) {
        if !logits[eid].is_finite() {
            continue;
        }
        let sym = elem_id_to_symbol(eid);
        let rc_cand = rcov_angstrom(sym);
        let mut penalty = 0.0f32;
        for (j, &pe) in prior_elems.iter().enumerate() {
            let dx = voxel_center[0] - prior_pos[j][0];
            let dy = voxel_center[1] - prior_pos[j][1];
            let dz = voxel_center[2] - prior_pos[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let sym_p = elem_id_to_symbol(pe);
            let rc_sum = rc_cand + rcov_angstrom(sym_p);
            if d < 0.58 * rc_sum {
                penalty += 18.0;
            } else if d <= 1.24 * rc_sum {
                // Would form a covalent bond to saturated neighbor `j`!
                if prior_coord_deg[j] >= max_coord_element(sym_p) {
                    penalty += 10.0;
                }
            }
        }
        logits[eid] -= penalty;
    }
}

/// Allowed element vocabulary mask (`[602]` bool): true ONLY for `bb.allowed_elements` + `atom_null_id`.
pub fn build_allowed_atom_vocab_mask(bundle: &ModelBundle) -> Vec<bool> {
    let bb = &bundle.header.backbone;
    let mut mask = vec![false; bb.num_atom_vocab];
    if bb.atom_null_id < mask.len() {
        mask[bb.atom_null_id] = true;
    }
    if bb.allowed_elements.is_empty() {
        for e in 0..118.min(mask.len()) {
            mask[e] = true;
        }
    } else {
        for sym in &bb.allowed_elements {
            if let Some(eid) = symbol_to_elem_id(sym) {
                if eid < mask.len() {
                    mask[eid] = true;
                }
            }
        }
    }
    mask
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AlignedQ8Block {
    scale: f32,
    qs: [i8; 32],
}

pub struct MergedQ8Linear {
    pub in_dim: usize,
    pub pad_in: usize,
    pub blocks_per_row: usize,
    pub out_dim: usize,
    blocks: Vec<AlignedQ8Block>,
    bias: Option<Vec<f32>>,
}

impl MergedQ8Linear {
    pub fn from_bundle_and_lora(
        bundle: &ModelBundle,
        prefix: &str,
        lora_opt: Option<&LoraPair>,
    ) -> Result<Self, String> {
        let (out_d, in_d, packed, bias) = bundle.dequantize_linear_f16_u32(prefix)?;
        let pairs = (in_d + 1) / 2;
        let bpr = (in_d + 31) / 32;
        let pad_in = bpr * 32;
        let mut blocks = vec![
            AlignedQ8Block {
                scale: 0.0,
                qs: [0i8; 32],
            };
            out_d * bpr
        ];

        let active_lp = lora_opt.filter(|lp| lp.scaling != 0.0 && lp.b.iter().any(|&v| v != 0.0));
        let mut row_buf = vec![0.0f32; pad_in];

        for o in 0..out_d {
            let p_row = &packed[o * pairs..(o + 1) * pairs];
            for p in 0..(in_d / 2) {
                let w = p_row[p];
                row_buf[2 * p] = f16::from_bits((w & 0xFFFF) as u16).to_f32();
                row_buf[2 * p + 1] = f16::from_bits((w >> 16) as u16).to_f32();
            }
            if (in_d & 1) != 0 {
                row_buf[in_d - 1] = f16::from_bits((p_row[pairs - 1] & 0xFFFF) as u16).to_f32();
            }
            for c in in_d..pad_in {
                row_buf[c] = 0.0;
            }
            if let Some(lp) = active_lp {
                let bo = &lp.b[o * lp.rank..(o + 1) * lp.rank];
                let sc = lp.scaling;
                for r in 0..lp.rank {
                    let br = bo[r] * sc;
                    if br == 0.0 {
                        continue;
                    }
                    let ar = &lp.a[r * in_d..(r + 1) * in_d];
                    for c in 0..in_d {
                        row_buf[c] += br * ar[c];
                    }
                }
            }
            let out_blks = &mut blocks[o * bpr..(o + 1) * bpr];
            for bi in 0..bpr {
                let seg = &row_buf[bi * 32..bi * 32 + 32];
                let mut m_abs = 0.0f32;
                for &v in seg {
                    let av = v.abs();
                    if av > m_abs {
                        m_abs = av;
                    }
                }
                if m_abs < 1e-12 {
                    out_blks[bi] = AlignedQ8Block {
                        scale: 0.0,
                        qs: [0i8; 32],
                    };
                } else {
                    let scale = m_abs / 127.0;
                    let inv = 127.0 / m_abs;
                    let mut qs = [0i8; 32];
                    for j in 0..32 {
                        qs[j] = (seg[j] * inv).round().clamp(-127.0, 127.0) as i8;
                    }
                    out_blks[bi] = AlignedQ8Block { scale, qs };
                }
            }
        }

        Ok(Self {
            in_dim: in_d,
            pad_in,
            blocks_per_row: bpr,
            out_dim: out_d,
            blocks,
            bias,
        })
    }

    #[inline]
    pub fn eval_rows(&self, x: &[f32], rows: usize, x_pad_scratch: &mut Vec<f32>, y: &mut [f32]) {
        let p_in = self.pad_in;
        let in_d = self.in_dim;
        let out_d = self.out_dim;
        let bpr = self.blocks_per_row;
        if x_pad_scratch.len() < rows * p_in {
            x_pad_scratch.resize(rows * p_in, 0.0);
        }
        for r in 0..rows {
            x_pad_scratch[r * p_in..r * p_in + in_d].copy_from_slice(&x[r * in_d..(r + 1) * in_d]);
            for c in in_d..p_in {
                x_pad_scratch[r * p_in + c] = 0.0;
            }
        }

        if rows == 4 {
            let (r0, rest) = x_pad_scratch[..4 * p_in].split_at(p_in);
            let (r1, rest) = rest.split_at(p_in);
            let (r2, r3) = rest.split_at(p_in);
            for o in 0..out_d {
                let b0 = self.bias.as_ref().map_or(0.0, |b| b[o]);
                let mut acc0 = b0;
                let mut acc1 = b0;
                let mut acc2 = b0;
                let mut acc3 = b0;
                let r_blks = &self.blocks[o * bpr..(o + 1) * bpr];
                for bi in 0..bpr {
                    let blk = &r_blks[bi];
                    let sc = blk.scale;
                    let x0 = &r0[bi * 32..bi * 32 + 32];
                    let x1 = &r1[bi * 32..bi * 32 + 32];
                    let x2 = &r2[bi * 32..bi * 32 + 32];
                    let x3 = &r3[bi * 32..bi * 32 + 32];
                    let (mut s0a, mut s0b, mut s0c, mut s0d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let (mut s1a, mut s1b, mut s1c, mut s1d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let (mut s2a, mut s2b, mut s2c, mut s2d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let (mut s3a, mut s3b, mut s3c, mut s3d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    for k in 0..8 {
                        let b = 4 * k;
                        let (q0, q1, q2, q3) = (
                            blk.qs[b] as f32,
                            blk.qs[b + 1] as f32,
                            blk.qs[b + 2] as f32,
                            blk.qs[b + 3] as f32,
                        );
                        s0a += x0[b] * q0; s0b += x0[b + 1] * q1; s0c += x0[b + 2] * q2; s0d += x0[b + 3] * q3;
                        s1a += x1[b] * q0; s1b += x1[b + 1] * q1; s1c += x1[b + 2] * q2; s1d += x1[b + 3] * q3;
                        s2a += x2[b] * q0; s2b += x2[b + 1] * q1; s2c += x2[b + 2] * q2; s2d += x2[b + 3] * q3;
                        s3a += x3[b] * q0; s3b += x3[b + 1] * q1; s3c += x3[b + 2] * q2; s3d += x3[b + 3] * q3;
                    }
                    acc0 += ((s0a + s0b) + (s0c + s0d)) * sc;
                    acc1 += ((s1a + s1b) + (s1c + s1d)) * sc;
                    acc2 += ((s2a + s2b) + (s2c + s2d)) * sc;
                    acc3 += ((s3a + s3b) + (s3c + s3d)) * sc;
                }
                y[o] = acc0;
                y[out_d + o] = acc1;
                y[2 * out_d + o] = acc2;
                y[3 * out_d + o] = acc3;
            }
        } else if rows == 2 {
            let (r0, r1) = x_pad_scratch[..2 * p_in].split_at(p_in);
            for o in 0..out_d {
                let b0 = self.bias.as_ref().map_or(0.0, |b| b[o]);
                let mut acc0 = b0;
                let mut acc1 = b0;
                let r_blks = &self.blocks[o * bpr..(o + 1) * bpr];
                for bi in 0..bpr {
                    let blk = &r_blks[bi];
                    let sc = blk.scale;
                    let x0 = &r0[bi * 32..bi * 32 + 32];
                    let x1 = &r1[bi * 32..bi * 32 + 32];
                    let (mut s0a, mut s0b, mut s0c, mut s0d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let (mut s1a, mut s1b, mut s1c, mut s1d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    for k in 0..8 {
                        let b = 4 * k;
                        let (q0, q1, q2, q3) = (
                            blk.qs[b] as f32,
                            blk.qs[b + 1] as f32,
                            blk.qs[b + 2] as f32,
                            blk.qs[b + 3] as f32,
                        );
                        s0a += x0[b] * q0; s0b += x0[b + 1] * q1; s0c += x0[b + 2] * q2; s0d += x0[b + 3] * q3;
                        s1a += x1[b] * q0; s1b += x1[b + 1] * q1; s1c += x1[b + 2] * q2; s1d += x1[b + 3] * q3;
                    }
                    acc0 += ((s0a + s0b) + (s0c + s0d)) * sc;
                    acc1 += ((s1a + s1b) + (s1c + s1d)) * sc;
                }
                y[o] = acc0;
                y[out_d + o] = acc1;
            }
        } else {
            for r in 0..rows {
                let xr = &x_pad_scratch[r * p_in..(r + 1) * p_in];
                let yr = &mut y[r * out_d..(r + 1) * out_d];
                for o in 0..out_d {
                    let mut acc = self.bias.as_ref().map_or(0.0, |b| b[o]);
                    let r_blks = &self.blocks[o * bpr..(o + 1) * bpr];
                    for bi in 0..bpr {
                        let blk = &r_blks[bi];
                        let xs = &xr[bi * 32..bi * 32 + 32];
                        let (mut sa, mut sb, mut sc0, mut sd) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                        for k in 0..8 {
                            let b = 4 * k;
                            sa += xs[b] * (blk.qs[b] as f32);
                            sb += xs[b + 1] * (blk.qs[b + 1] as f32);
                            sc0 += xs[b + 2] * (blk.qs[b + 2] as f32);
                            sd += xs[b + 3] * (blk.qs[b + 3] as f32);
                        }
                        acc += ((sa + sb) + (sc0 + sd)) * blk.scale;
                    }
                    yr[o] = acc;
                }
            }
        }
    }
}

pub struct MergedLayerWeights {
    pub attn_norm_w: Vec<f32>,
    pub q_proj: MergedQ8Linear,
    pub k_proj: MergedQ8Linear,
    pub v_proj: MergedQ8Linear,
    pub out_proj: MergedQ8Linear,
    pub ffn_norm_w: Vec<f32>,
    pub w1: MergedQ8Linear,
    pub w2: MergedQ8Linear,
    pub w3: MergedQ8Linear,
}

/// Pre-merged aligned `Block-Q8` weights (`W_base + LoRA`) built ONCE per rollout batch (`~25 ms`)
/// to accelerate KV-cached trajectory generation (`TopoGRPO`, `GP Atlas`, `Best-of-M`) by `30x–60x`!
pub struct MergedRolloutWeights {
    pub layers: Vec<MergedLayerWeights>,
    pub final_norm_w: Vec<f32>,
    pub tree_dense: MergedQ8Linear,
    pub tree_out: MergedQ8Linear,
    pub atom_dense: [MergedQ8Linear; 4],
    pub atom_out: [MergedQ8Linear; 4],
}

impl MergedRolloutWeights {
    pub fn build(bundle: &ModelBundle, adapter: &LoomLoraAdapter) -> Result<Self, String> {
        let bb = &bundle.header.backbone;
        let mk = |pfx: &str| MergedQ8Linear::from_bundle_and_lora(bundle, pfx, adapter.loras.get(pfx));
        let mut layers = Vec::with_capacity(bb.num_layers);
        for l in 0..bb.num_layers {
            let lp = format!("bb.layers.{l}");
            layers.push(MergedLayerWeights {
                attn_norm_w: bundle.f32_slice(&format!("{lp}.attn_norm.weight"))?.to_vec(),
                q_proj: mk(&format!("{lp}.attn.q_proj"))?,
                k_proj: mk(&format!("{lp}.attn.k_proj"))?,
                v_proj: mk(&format!("{lp}.attn.v_proj"))?,
                out_proj: mk(&format!("{lp}.attn.out_proj"))?,
                ffn_norm_w: bundle.f32_slice(&format!("{lp}.ffn_norm.weight"))?.to_vec(),
                w1: mk(&format!("{lp}.ffn.w1"))?,
                w2: mk(&format!("{lp}.ffn.w2"))?,
                w3: mk(&format!("{lp}.ffn.w3"))?,
            });
        }
        Ok(Self {
            layers,
            final_norm_w: bundle.f32_slice("bb.norm.weight")?.to_vec(),
            tree_dense: mk("bb.tree_head.dense")?,
            tree_out: mk("bb.tree_head.out")?,
            atom_dense: [
                mk("bb.atom_head.pred.0.dense")?,
                mk("bb.atom_head.pred.1.dense")?,
                mk("bb.atom_head.pred.2.dense")?,
                mk("bb.atom_head.pred.3.dense")?,
            ],
            atom_out: [
                mk("bb.atom_head.pred.0.out")?,
                mk("bb.atom_head.pred.1.out")?,
                mk("bb.atom_head.pred.2.out")?,
                mk("bb.atom_head.pred.3.out")?,
            ],
        })
    }
}

#[derive(Clone, Copy)]
struct RollQueueCell {
    pos: [i32; 3],
    level: i32,
}

/// Sample a single conditional 3D trajectory with KV-cached dual-branch batched SIMD decoding under `MergedRolloutWeights`,
/// recording the exact `EncodedOctreeSeq` token & decision trace for `TopoGRPO`!
pub fn sample_rollout_trajectory_kv(
    bundle: &ModelBundle,
    adapter: &LoomLoraAdapter,
    merged: &MergedRolloutWeights,
    raw_targets: &[f32],
    active_mask: &[bool],
    opts: &RolloutOptions,
) -> Result<(EncodedOctreeSeq, MolRecord, SampledUni3darMolecule), String> {
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let n_heads = bb.num_head;
    let hd = bb.head_dim;
    let ffn_d = bb.ffn_dim;
    let scale_mha = 1.0 / (hd as f32).sqrt();

    let any_active = active_mask.iter().any(|&b| b);
    let do_cfg = any_active && ((opts.cfg_scale - 1.0).abs() > 1e-4 || (opts.xyz_cfg_scale - 1.0).abs() > 1e-4);

    let cond_t = adapter
        .conditioner
        .forward_batch(&[raw_targets.to_vec()], &[active_mask.to_vec()]);
    let uncond_t = if do_cfg {
        Some(adapter.conditioner.forward_batch(
            &[vec![0.0; raw_targets.len()]],
            &[vec![false; active_mask.len()]],
        ))
    } else {
        None
    };

    let cap_kv = (opts.max_decode_steps * 2 + 8) * d;
    let mut kv_cond_k: Vec<Vec<f32>> = (0..bb.num_layers).map(|_| Vec::with_capacity(cap_kv)).collect();
    let mut kv_cond_v: Vec<Vec<f32>> = (0..bb.num_layers).map(|_| Vec::with_capacity(cap_kv)).collect();
    let mut kv_unc_k: Vec<Vec<f32>> = (0..bb.num_layers).map(|_| Vec::with_capacity(if do_cfg { cap_kv } else { 0 })).collect();
    let mut kv_unc_v: Vec<Vec<f32>> = (0..bb.num_layers).map(|_| Vec::with_capacity(if do_cfg { cap_kv } else { 0 })).collect();

    // Reusable zero-alloc scratch buffers across all decode steps!
    let mut x_pad = Vec::with_capacity(4 * ffn_d.max(d));
    let mut h_buf = vec![0.0f32; 4 * d];
    let mut xn_buf = vec![0.0f32; 4 * d];
    let mut q_buf = vec![0.0f32; 4 * d];
    let mut k_buf = vec![0.0f32; 4 * d];
    let mut v_buf = vec![0.0f32; 4 * d];
    let mut ac_buf = vec![0.0f32; 4 * d];
    let mut ap_buf = vec![0.0f32; 4 * d];
    let mut gp_buf = vec![0.0f32; 4 * ffn_d];
    let mut up_buf = vec![0.0f32; 4 * ffn_d];
    let mut sw_buf = vec![0.0f32; 4 * ffn_d];
    let mut scores = vec![0.0f32; opts.max_decode_steps * 2 + 16];
    let mut head_mid = vec![0.0f32; 2 * d];

    let attn_branch = |q_sl: &[f32],
                       kb: &[f32],
                       vb: &[f32],
                       prev_len: usize,
                       seq_q: usize,
                       scores_buf: &mut [f32],
                       out_sl: &mut [f32]| {
        out_sl.fill(0.0);
        for qi in 0..seq_q {
            let max_kj = prev_len + qi;
            for head in 0..n_heads {
                let qv = &q_sl[qi * d + head * hd..qi * d + (head + 1) * hd];
                let mut ms = f32::NEG_INFINITY;
                for kj in 0..=max_kj {
                    let kv = &kb[kj * d + head * hd..kj * d + (head + 1) * hd];
                    let mut dot = 0.0f32;
                    for c in 0..hd {
                        dot += qv[c] * kv[c];
                    }
                    let s = dot * scale_mha;
                    scores_buf[kj] = s;
                    if s > ms {
                        ms = s;
                    }
                }
                let mut se = 0.0f32;
                for kj in 0..=max_kj {
                    scores_buf[kj] = (scores_buf[kj] - ms).exp();
                    se += scores_buf[kj];
                }
                let inv_se = 1.0 / se.max(1e-20);
                let os = &mut out_sl[qi * d + head * hd..qi * d + (head + 1) * hd];
                for kj in 0..=max_kj {
                    let w = scores_buf[kj] * inv_se;
                    let vv = &vb[kj * d + head * hd..kj * d + (head + 1) * hd];
                    for c in 0..hd {
                        os[c] += w * vv[c];
                    }
                }
            }
        }
    };

    let rmsnorm_in_place = |src: &[f32], w: &[f32], rows: usize, dst: &mut [f32]| {
        let inv_d = 1.0 / (d as f32);
        for r in 0..rows {
            let xr = &src[r * d..(r + 1) * d];
            let sq = xr.iter().map(|&v| v * v).sum::<f32>() * inv_d;
            let ir = 1.0 / (sq + 1e-6).sqrt();
            let yr = &mut dst[r * d..(r + 1) * d];
            for c in 0..d {
                yr[c] = xr[c] * ir * w[c];
            }
        }
    };

    let allowed_mask = build_allowed_atom_vocab_mask(bundle);
    let mut rng = DetRng::new(opts.seed);
    let ml = bb.merge_level as i32;
    let cls_p = bb.cls_phy_pos();

    let mut queue = vec![RollQueueCell { pos: [0, 0, 0], level: ml }];
    let mut q_head = 0usize;
    let mut seq_tokens = Vec::with_capacity(256);
    let mut tree_targets = Vec::new();
    let mut atom_targets = Vec::new();

    let mut placed_elems: Vec<usize> = Vec::new();
    let mut placed_pos: Vec<[f32; 3]> = Vec::new();
    let mut placed_deg: Vec<i32> = Vec::new();
    let mut acc_z_parity = 0usize;

    // Step 0: push root query `[MASK]`
    seq_tokens.push(SeqToken {
        typ: bb.atom_mask_id,
        xyz_bins: [bb.xyz_null_id; 3],
        level: ml,
        rem_atoms: 0,
        phy_pos: [cls_p, cls_p, cls_p],
        is_root: true,
    });
    let mut feed_start = 0usize;

    while q_head < queue.len() && (q_head < opts.max_decode_steps) {
        let cell = queue[q_head];
        q_head += 1;

        let toks = &seq_tokens[feed_start..];
        let seq_q = toks.len();
        let branches = if do_cfg { 2usize } else { 1usize };
        let tot_r = branches * seq_q;
        let prev_len = kv_cond_k[0].len() / d;
        if scores.len() < prev_len + seq_q + 4 {
            scores.resize(prev_len + seq_q + 64, 0.0);
        }
        let q_token_pos = seq_tokens.len() - 1;

        let mut ropes = Vec::with_capacity(seq_q);
        for (i, t) in toks.iter().enumerate() {
            let emb_c = embed_octree_token(
                bundle,
                t.typ,
                t.xyz_bins,
                t.level,
                t.rem_atoms,
                if t.is_root { Some(&cond_t.root_delta[..d]) } else { None },
            )?;
            h_buf[i * d..(i + 1) * d].copy_from_slice(&emb_c);
            if let Some(ref ut) = uncond_t {
                let emb_u = embed_octree_token(
                    bundle,
                    t.typ,
                    t.xyz_bins,
                    t.level,
                    t.rem_atoms,
                    if t.is_root { Some(&ut.root_delta[..d]) } else { None },
                )?;
                h_buf[(seq_q + i) * d..(seq_q + i + 1) * d].copy_from_slice(&emb_u);
            }
            ropes.push(compute_rope_3d(t.phy_pos, bb.xyz_resolution, hd, bb.rope_theta));
        }

        // Batched 12-layer step across both `cond` and `uncond` branches simultaneously (`tot_r = 2` or `4`!)
        for l in 0..bb.num_layers {
            let lw = &merged.layers[l];
            rmsnorm_in_place(&h_buf[..tot_r * d], &lw.attn_norm_w, tot_r, &mut xn_buf[..tot_r * d]);
            lw.q_proj.eval_rows(&xn_buf[..tot_r * d], tot_r, &mut x_pad, &mut q_buf[..tot_r * d]);
            lw.k_proj.eval_rows(&xn_buf[..tot_r * d], tot_r, &mut x_pad, &mut k_buf[..tot_r * d]);
            lw.v_proj.eval_rows(&xn_buf[..tot_r * d], tot_r, &mut x_pad, &mut v_buf[..tot_r * d]);

            for br in 0..branches {
                for qi in 0..seq_q {
                    let r_idx = br * seq_q + qi;
                    apply_rope_in_place(&mut q_buf[r_idx * d..(r_idx + 1) * d], n_heads, hd, &ropes[qi].0, &ropes[qi].1);
                    apply_rope_in_place(&mut k_buf[r_idx * d..(r_idx + 1) * d], n_heads, hd, &ropes[qi].0, &ropes[qi].1);
                }
            }

            kv_cond_k[l].extend_from_slice(&k_buf[..seq_q * d]);
            kv_cond_v[l].extend_from_slice(&v_buf[..seq_q * d]);
            attn_branch(&q_buf[..seq_q * d], &kv_cond_k[l], &kv_cond_v[l], prev_len, seq_q, &mut scores, &mut ac_buf[..seq_q * d]);

            if do_cfg {
                kv_unc_k[l].extend_from_slice(&k_buf[seq_q * d..2 * seq_q * d]);
                kv_unc_v[l].extend_from_slice(&v_buf[seq_q * d..2 * seq_q * d]);
                attn_branch(
                    &q_buf[seq_q * d..2 * seq_q * d],
                    &kv_unc_k[l],
                    &kv_unc_v[l],
                    prev_len,
                    seq_q,
                    &mut scores,
                    &mut ac_buf[seq_q * d..2 * seq_q * d],
                );
            }

            lw.out_proj.eval_rows(&ac_buf[..tot_r * d], tot_r, &mut x_pad, &mut ap_buf[..tot_r * d]);
            for i in 0..(tot_r * d) {
                h_buf[i] += ap_buf[i];
            }

            rmsnorm_in_place(&h_buf[..tot_r * d], &lw.ffn_norm_w, tot_r, &mut xn_buf[..tot_r * d]);
            lw.w1.eval_rows(&xn_buf[..tot_r * d], tot_r, &mut x_pad, &mut gp_buf[..tot_r * ffn_d]);
            lw.w3.eval_rows(&xn_buf[..tot_r * d], tot_r, &mut x_pad, &mut up_buf[..tot_r * ffn_d]);
            for i in 0..(tot_r * ffn_d) {
                sw_buf[i] = silu(gp_buf[i]) * up_buf[i];
            }
            lw.w2.eval_rows(&sw_buf[..tot_r * ffn_d], tot_r, &mut x_pad, &mut ap_buf[..tot_r * d]);

            let gbc = &cond_t.layer_gamma_beta[l];
            for qi in 0..seq_q {
                let row = &mut h_buf[qi * d..(qi + 1) * d];
                let fr = &ap_buf[qi * d..(qi + 1) * d];
                for c in 0..d {
                    row[c] = (row[c] + fr[c]) * (1.0 + gbc[c]) + gbc[d + c];
                }
            }
            if let Some(ref ut) = uncond_t {
                let gbu = &ut.layer_gamma_beta[l];
                for qi in 0..seq_q {
                    let idx_r = seq_q + qi;
                    let row = &mut h_buf[idx_r * d..(idx_r + 1) * d];
                    let fr = &ap_buf[idx_r * d..(idx_r + 1) * d];
                    for c in 0..d {
                        row[c] = (row[c] + fr[c]) * (1.0 + gbu[c]) + gbu[d + c];
                    }
                }
            }
        }

        let mut h_last = vec![0.0f32; branches * d];
        rmsnorm_in_place(&h_buf[(seq_q - 1) * d..seq_q * d], &merged.final_norm_w, 1, &mut h_last[..d]);
        for c in 0..d {
            h_last[c] = h_last[c] * (1.0 + cond_t.out_gamma_beta[c]) + cond_t.out_gamma_beta[d + c];
        }
        if let Some(ref ut) = uncond_t {
            rmsnorm_in_place(&h_buf[(2 * seq_q - 1) * d..2 * seq_q * d], &merged.final_norm_w, 1, &mut h_last[d..2 * d]);
            for c in 0..d {
                h_last[d + c] = h_last[d + c] * (1.0 + ut.out_gamma_beta[c]) + ut.out_gamma_beta[d + c];
            }
        }
        feed_start = seq_tokens.len();

        let mut eval_head_pair = |dense: &MergedQ8Linear, out_lin: &MergedQ8Linear, z_in: &[f32], br_cnt: usize| -> Vec<f32> {
            dense.eval_rows(z_in, br_cnt, &mut x_pad, &mut head_mid[..br_cnt * d]);
            for v in &mut head_mid[..br_cnt * d] {
                *v = gelu_exact(*v);
            }
            let mut out = vec![0.0f32; br_cnt * out_lin.out_dim];
            out_lin.eval_rows(&head_mid[..br_cnt * d], br_cnt, &mut x_pad, &mut out);
            out
        };

        if cell.level > 0 {
            let both_l = eval_head_pair(&merged.tree_dense, &merged.tree_out, &h_last, branches);
            let mut logits = both_l[..255].to_vec();
            if do_cfg {
                let lu = &both_l[255..510];
                let decay_f = ((3 - cell.level).clamp(0, 2) as f32) * 0.5;
                let s_eff = 1.0 + (opts.cfg_scale - 1.0) * (1.0 - opts.cfg_tree_decay * decay_f);
                for i in 0..255 {
                    logits[i] = lu[i] + s_eff * (logits[i] - lu[i]);
                }
            }
            let (chosen_lbl, _) = rng.sample_logits_with_score(&logits, opts.temperatures[0]);
            let bitmask = chosen_lbl + 1;
            tree_targets.push(TreeDecisionTarget {
                token_pos: q_token_pos,
                level: cell.level,
                label_255: chosen_lbl,
            });

            // Push resolved `[TREE_bitmask]` odd token
            let q_tok = seq_tokens[q_token_pos];
            seq_tokens.push(SeqToken {
                typ: bb.tree0_split_id + bitmask,
                xyz_bins: [bb.xyz_null_id; 3],
                level: cell.level,
                rem_atoms: 0,
                phy_pos: q_tok.phy_pos,
                is_root: false,
            });

            // Enqueue occupied child octants in Morton Z-order
            let step = 1i32 << (cell.level - 1);
            for k in 0..8 {
                if (bitmask & (1 << k)) != 0 {
                    queue.push(RollQueueCell {
                        pos: [
                            cell.pos[0] + step * SUBCELL_ORDERS[k][0],
                            cell.pos[1] + step * SUBCELL_ORDERS[k][1],
                            cell.pos[2] + step * SUBCELL_ORDERS[k][2],
                        ],
                        level: cell.level - 1,
                    });
                }
            }
        } else {
            // Leaf atom (`level == 0`)
            let vox_c = seq_tokens[q_token_pos].phy_pos;
            let is_terminal = q_head == queue.len();
            let va = bb.num_atom_vocab;

            let both0 = eval_head_pair(&merged.atom_dense[0], &merged.atom_out[0], &h_last, branches);
            let mut logits0 = both0[..va].to_vec();
            if do_cfg {
                let lu0 = &both0[va..2 * va];
                for i in 0..va {
                    logits0[i] = lu0[i] + opts.cfg_scale * (logits0[i] - lu0[i]);
                }
            }
            for i in 0..va {
                if !allowed_mask[i] {
                    logits0[i] = f32::NEG_INFINITY;
                }
            }
            if opts.geometric_guard {
                apply_element_geometric_guard(
                    &mut logits0,
                    vox_c,
                    &placed_elems,
                    &placed_pos,
                    &placed_deg,
                    bb.atom_null_id,
                );
            }
            if opts.even_electron_parity && is_terminal {
                for eid in 0..118.min(va) {
                    if ((acc_z_parity + eid + 1) & 1) != 0 {
                        logits0[eid] = f32::NEG_INFINITY;
                    }
                }
                if (acc_z_parity & 1) != 0 && bb.atom_null_id < va {
                    logits0[bb.atom_null_id] = f32::NEG_INFINITY;
                }
            }

            let (elem_id, _) = rng.sample_logits_with_score(&logits0, opts.temperatures[1]);
            let mut u_bins = [bb.xyz_null_id; 3];
            let mut fine_p = vox_c;

            if elem_id < 342 {
                let do_xyz_cfg = do_cfg && (opts.xyz_cfg_scale - 1.0).abs() > 1e-4;
                let xyz_br = if do_xyz_cfg { 2usize } else { 1usize };
                let mut z_xyz = h_last[..xyz_br * d].to_vec();
                for ax in 1..=3 {
                    let prev_tok = if ax == 1 { elem_id } else { u_bins[ax - 2] };
                    let emb_ax = bundle.f32_slice(&format!("bb.atom_head.emb.{}", ax - 1))?;
                    let safe_id = prev_tok.min((emb_ax.len() / d).saturating_sub(1));
                    for br in 0..xyz_br {
                        for c in 0..d {
                            z_xyz[br * d + c] += emb_ax[safe_id * d + c];
                        }
                    }
                    let both_ax = eval_head_pair(&merged.atom_dense[ax], &merged.atom_out[ax], &z_xyz, xyz_br);
                    let mut lax = both_ax[..bb.num_xyz].to_vec();
                    if do_xyz_cfg {
                        let lu = &both_ax[bb.num_xyz..2 * bb.num_xyz];
                        for i in 0..bb.num_xyz {
                            lax[i] = lu[i] + opts.xyz_cfg_scale * (lax[i] - lu[i]);
                        }
                    }
                    let (ub, _) = rng.sample_logits_with_score(&lax, opts.temperatures[2]);
                    u_bins[ax - 1] = ub;
                }
                fine_p = [
                    (cell.pos[0] as f32) * bb.grid_len + (u_bins[0] as f32) * bb.xyz_resolution,
                    (cell.pos[1] as f32) * bb.grid_len + (u_bins[1] as f32) * bb.xyz_resolution,
                    (cell.pos[2] as f32) * bb.grid_len + (u_bins[2] as f32) * bb.xyz_resolution,
                ];
            }

            atom_targets.push(AtomDecisionTarget {
                token_pos: q_token_pos,
                elem_id,
                xyz_bins: u_bins,
                heavy_valence: None,
                is_terminal_leaf: is_terminal,
                prior_z_parity: acc_z_parity,
                voxel_center: vox_c,
            });

            if elem_id < 342 {
                acc_z_parity = (acc_z_parity + elem_id + 1) & 1;
                let mut my_deg = 0i32;
                let rc_me = rcov_angstrom(elem_id_to_symbol(elem_id));
                for j in 0..placed_elems.len() {
                    let dx = fine_p[0] - placed_pos[j][0];
                    let dy = fine_p[1] - placed_pos[j][1];
                    let dz = fine_p[2] - placed_pos[j][2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    let rc_sum = rc_me + rcov_angstrom(elem_id_to_symbol(placed_elems[j]));
                    if dist <= 1.24 * rc_sum {
                        my_deg += 1;
                        placed_deg[j] += 1;
                    }
                }
                placed_elems.push(elem_id);
                placed_pos.push(fine_p);
                placed_deg.push(my_deg);
            }

            seq_tokens.push(SeqToken {
                typ: elem_id,
                xyz_bins: u_bins,
                level: 0,
                rem_atoms: 0,
                phy_pos: fine_p,
                is_root: false,
            });
        }

        // Push next cell's `[MASK]` query token if queue still has pending cells!
        if q_head < queue.len() && q_head < opts.max_decode_steps {
            let nxt = queue[q_head];
            let half_e = 0.5f32 * ((1usize << (nxt.level as usize)) as f32) * bb.grid_len;
            seq_tokens.push(SeqToken {
                typ: bb.atom_mask_id,
                xyz_bins: [bb.xyz_null_id; 3],
                level: nxt.level,
                rem_atoms: 0,
                phy_pos: [
                    (nxt.pos[0] as f32) * bb.grid_len + half_e,
                    (nxt.pos[1] as f32) * bb.grid_len + half_e,
                    (nxt.pos[2] as f32) * bb.grid_len + half_e,
                ],
                is_root: false,
            });
        }
    }

    // Append trailing `[MASK]` + `[CLS]` pair (`level = -1`)
    seq_tokens.push(SeqToken {
        typ: bb.atom_mask_id,
        xyz_bins: [bb.xyz_null_id; 3],
        level: -1,
        rem_atoms: 0,
        phy_pos: [cls_p, cls_p, cls_p],
        is_root: false,
    });
    seq_tokens.push(SeqToken {
        typ: bb.atom_cls_id,
        xyz_bins: [bb.xyz_null_id; 3],
        level: -1,
        rem_atoms: 0,
        phy_pos: [cls_p, cls_p, cls_p],
        is_root: false,
    });

    let rg = if opts.reconcile_valence {
        reconcile_point_cloud(&placed_elems, &placed_pos, None, 24, true)
            .unwrap_or_else(|| build_raw_point_cloud_graph(&placed_elems, &placed_pos))
    } else {
        reconcile_point_cloud(&placed_elems, &placed_pos, None, 24, false)
            .filter(|g| g.delta_h_count == 0)
            .unwrap_or_else(|| build_raw_point_cloud_graph(&placed_elems, &placed_pos))
    };
    let sampled_mol = finalize_sampled_molecule(
        rg,
        None,
        None,
        std::collections::BTreeMap::new(),
        [1, 1, 1],
        q_head,
    );
    let rec = MolRecord::from_uni3dar(&sampled_mol);

    let enc = EncodedOctreeSeq {
        n_real_atoms: placed_elems.len(),
        tokens: seq_tokens,
        tree_targets,
        atom_targets,
        targets_raw: raw_targets.to_vec(),
        active_mask: active_mask.to_vec(),
    };

    Ok((enc, rec, sampled_mol))
}
