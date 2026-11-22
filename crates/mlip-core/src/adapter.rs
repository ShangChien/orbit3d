//! Shared-backbone 3D Distance-Biased Attention Property Oracle Adapter
//! (`_DPA4EquivariantPropertyAdapter` + `_BootstrapAdapters` from `src/loom/oracles/dpa4_oracle.py`).
//!
//! Consumes frozen per-atom equivariant descriptors `(N, desc_dim)` + 3D coordinates `(N, 3)`
//! from `DPA-4` or `HyFF-L2` and outputs calibrated ensemble `mean` and epistemic `std`.

use serde::{Deserialize, Serialize};

/// Target inverse-normalization specification matching `inverse_single_scaler_torch`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TargetScaler {
    Identity,
    /// Covers `StandardScaler` (`y * scale + mean`), `RobustScaler` (`y * scale + center`),
    /// `MaxAbsScaler`, and `MinMaxScaler`.
    Affine {
        mean: f32,
        scale: f32,
    },
    /// Covers `PowerTransformer(method="yeo-johnson")`.
    YeoJohnson {
        lambda: f32,
        mean: f32,
        scale: f32,
        standardize: bool,
    },
}

impl TargetScaler {
    #[inline]
    pub fn inverse(&self, y_norm: f32) -> f32 {
        match *self {
            TargetScaler::Identity => y_norm,
            TargetScaler::Affine { mean, scale } => y_norm * scale + mean,
            TargetScaler::YeoJohnson {
                lambda,
                mean,
                scale,
                standardize,
            } => {
                let x = if standardize {
                    y_norm * scale + mean
                } else {
                    y_norm
                };
                if x >= 0.0 {
                    if lambda.abs() < 1e-7 {
                        x.exp_m1()
                    } else {
                        ((lambda * x).ln_1p() / lambda).exp_m1()
                    }
                } else {
                    let comp = 2.0 - lambda;
                    if comp.abs() < 1e-7 {
                        -((-x).exp_m1())
                    } else {
                        -(((-comp * x).ln_1p() / comp).exp_m1())
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearWeights {
    /// Row-major `(out_features, in_features)` matching PyTorch `nn.Linear.weight`.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl LinearWeights {
    pub fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), rows * self.in_dim);
        let mut out = vec![0.0f32; rows * self.out_dim];
        for r in 0..rows {
            let x_row = &x[r * self.in_dim..(r + 1) * self.in_dim];
            let out_row = &mut out[r * self.out_dim..(r + 1) * self.out_dim];
            for o in 0..self.out_dim {
                let w_row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
                let mut acc = self.bias.get(o).copied().unwrap_or(0.0);
                for i in 0..self.in_dim {
                    acc += x_row[i] * w_row[i];
                }
                out_row[o] = acc;
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct LayerNormWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl LayerNormWeights {
    pub fn apply_in_place(&self, x: &mut [f32], dim: usize) {
        for row in x.chunks_exact_mut(dim) {
            let mut mean = 0.0f64;
            for &v in row.iter() {
                mean += v as f64;
            }
            mean /= dim as f64;
            let mut var = 0.0f64;
            for &v in row.iter() {
                let d = (v as f64) - mean;
                var += d * d;
            }
            var /= dim as f64;
            let inv_std = 1.0 / (var + (self.eps as f64)).sqrt();
            for i in 0..dim {
                let norm = ((row[i] as f64) - mean) * inv_std;
                row[i] = (norm as f32) * self.weight[i] + self.bias[i];
            }
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[derive(Clone, Debug)]
pub struct AdapterHeadWeights {
    pub gate: LinearWeights,       // (H -> 1)
    pub head_norm: LayerNormWeights, // (2*H)
    pub head_fc0: LinearWeights,   // (2*H -> H)
    pub head_fc1: LinearWeights,   // (H -> H/2)
    pub head_fc2: LinearWeights,   // (H/2 -> out_dim)
}

/// Exact Rust counterpart of `_DPA4EquivariantPropertyAdapter`.
#[derive(Clone, Debug)]
pub struct EquivariantAdapterMember {
    pub desc_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub rbf_centers: [f32; 16],
    pub in_proj: LinearWeights,
    pub dist_rbf: LinearWeights, // (16 -> 4)
    pub qkv: LinearWeights,      // (H -> 3*H)
    pub out_proj: LinearWeights, // (H -> H)
    pub norm1: LayerNormWeights,
    pub norm2: LayerNormWeights,
    pub ffn0: LinearWeights,     // (H -> 2*H)
    pub ffn1: LinearWeights,     // (2*H -> H)
    pub heads: Vec<AdapterHeadWeights>,
}

impl EquivariantAdapterMember {
    pub fn forward_mean(&self, atom_repr: &[f32], coords_a: &[[f64; 3]]) -> Result<Vec<f32>, String> {
        let n = coords_a.len();
        if n == 0 {
            return Err("Empty molecule passed to property adapter".into());
        }
        if atom_repr.len() != n * self.desc_dim {
            return Err(format!(
                "Descriptor length mismatch: expected {} ({}x{}), got {}",
                n * self.desc_dim,
                n,
                self.desc_dim,
                atom_repr.len()
            ));
        }

        // Center f64 coordinates before f32 pairwise distance computation (matches _properties_from_descriptor)
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

        let h_dim = self.hidden_dim;
        let head_dim = h_dim / 4;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // 1. h = norm1(in_proj(atom_repr))
        let mut h = self.in_proj.apply(atom_repr, n);
        self.norm1.apply_in_place(&mut h, h_dim);

        // 2. Pairwise 16-Gaussian RBF -> 4-head distance bias: (N*N, 16) -> (N*N, 4)
        let mut rbf_flat = vec![0.0f32; n * n * 16];
        for i in 0..n {
            for j in 0..n {
                let dx = coords_f32[i][0] - coords_f32[j][0];
                let dy = coords_f32[i][1] - coords_f32[j][1];
                let dz = coords_f32[i][2] - coords_f32[j][2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let base = (i * n + j) * 16;
                for k in 0..16 {
                    let z = (dist - self.rbf_centers[k]) / 0.4f32;
                    rbf_flat[base + k] = (-0.5f32 * z * z).exp();
                }
            }
        }
        let attn_bias_ij4 = self.dist_rbf.apply(&rbf_flat, n * n);

        // 3. 4-head 3D distance-biased self-attention
        let qkv = self.qkv.apply(&h, n); // (N, 3*H)
        let mut ctx = vec![0.0f32; n * h_dim];
        for hd in 0..4 {
            let hd_off = hd * head_dim;
            for i in 0..n {
                let q_slice = &qkv[i * 3 * h_dim + hd_off..i * 3 * h_dim + hd_off + head_dim];
                let mut scores = vec![0.0f32; n];
                let mut max_s = f32::NEG_INFINITY;
                for j in 0..n {
                    let k_slice =
                        &qkv[j * 3 * h_dim + h_dim + hd_off..j * 3 * h_dim + h_dim + hd_off + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_slice[d] * k_slice[d];
                    }
                    let s = dot * scale + attn_bias_ij4[(i * n + j) * 4 + hd];
                    scores[j] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut denom = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    denom += *s;
                }
                let inv_denom = 1.0 / denom.max(1e-20);
                let ctx_slice = &mut ctx[i * h_dim + hd_off..i * h_dim + hd_off + head_dim];
                for j in 0..n {
                    let p = scores[j] * inv_denom;
                    let v_slice = &qkv[j * 3 * h_dim + 2 * h_dim + hd_off
                        ..j * 3 * h_dim + 2 * h_dim + hd_off + head_dim];
                    for d in 0..head_dim {
                        ctx_slice[d] += p * v_slice[d];
                    }
                }
            }
        }

        // h = h + out_proj(ctx)
        let ctx_proj = self.out_proj.apply(&ctx, n);
        for idx in 0..h.len() {
            h[idx] += ctx_proj[idx];
        }

        // h = h + ffn(norm2(h))
        let mut h_norm2 = h.clone();
        self.norm2.apply_in_place(&mut h_norm2, h_dim);
        let mut ffn_mid = self.ffn0.apply(&h_norm2, n);
        for v in &mut ffn_mid {
            *v = silu(*v);
        }
        let ffn_out = self.ffn1.apply(&ffn_mid, n);
        for idx in 0..h.len() {
            h[idx] += ffn_out[idx];
        }

        // 4. Mean pool + Gated attention pool per head
        let mut mean_pool = vec![0.0f32; h_dim];
        let inv_n = 1.0 / (n as f32);
        for i in 0..n {
            for d in 0..h_dim {
                mean_pool[d] += h[i * h_dim + d] * inv_n;
            }
        }

        let mut head_sum = vec![0.0f32; self.output_dim];
        for hd in &self.heads {
            let gate_logits = hd.gate.apply(&h, n); // (N, 1)
            let max_g = gate_logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut weights = vec![0.0f32; n];
            let mut w_sum = 0.0f32;
            for i in 0..n {
                weights[i] = (gate_logits[i] - max_g).exp();
                w_sum += weights[i];
            }
            let inv_w = 1.0 / w_sum.max(1e-20);

            let mut mol_vec = vec![0.0f32; 2 * h_dim];
            mol_vec[..h_dim].copy_from_slice(&mean_pool);
            for i in 0..n {
                let w = weights[i] * inv_w;
                for d in 0..h_dim {
                    mol_vec[h_dim + d] += h[i * h_dim + d] * w;
                }
            }

            hd.head_norm.apply_in_place(&mut mol_vec, 2 * h_dim);
            let mut m0 = hd.head_fc0.apply(&mol_vec, 1);
            for v in &mut m0 {
                *v = silu(*v);
            }
            let mut m1 = hd.head_fc1.apply(&m0, 1);
            for v in &mut m1 {
                *v = silu(*v);
            }
            let pred = hd.head_fc2.apply(&m1, 1);
            for k in 0..self.output_dim {
                head_sum[k] += pred[k];
            }
        }

        let inv_heads = 1.0 / (self.heads.len().max(1) as f32);
        for v in &mut head_sum {
            *v *= inv_heads;
        }
        Ok(head_sum)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PropertyPrediction {
    pub target_names: Vec<String>,
    pub target_units: Vec<String>,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// Ensemble bootstrap property oracle (`_BootstrapAdapters` + per-target `TargetScaler`).
#[derive(Clone, Debug)]
pub struct BootstrapPropertyOracle {
    pub target_names: Vec<String>,
    pub target_units: Vec<String>,
    pub scalers: Vec<TargetScaler>,
    pub members: Vec<EquivariantAdapterMember>,
}

impl BootstrapPropertyOracle {
    pub fn predict(
        &self,
        atom_repr: &[f32],
        coords_a: &[[f64; 3]],
    ) -> Result<PropertyPrediction, String> {
        if self.members.is_empty() {
            return Err("BootstrapPropertyOracle has zero ensemble members".into());
        }
        let out_dim = self.members[0].output_dim;
        let m_count = self.members.len();
        let mut member_phys: Vec<Vec<f32>> = Vec::with_capacity(m_count);

        for member in &self.members {
            let norm_pred = member.forward_mean(atom_repr, coords_a)?;
            let mut phys = vec![0.0f32; out_dim];
            for k in 0..out_dim {
                let sc = self.scalers.get(k).unwrap_or(&TargetScaler::Identity);
                phys[k] = sc.inverse(norm_pred[k]);
            }
            member_phys.push(phys);
        }

        let mut mean = vec![0.0f32; out_dim];
        let inv_m = 1.0 / (m_count as f32);
        for row in &member_phys {
            for k in 0..out_dim {
                mean[k] += row[k] * inv_m;
            }
        }

        let mut std = vec![0.0f32; out_dim];
        for row in &member_phys {
            for k in 0..out_dim {
                let diff = row[k] - mean[k];
                std[k] += diff * diff * inv_m;
            }
        }
        for v in &mut std {
            *v = v.sqrt();
        }

        Ok(PropertyPrediction {
            target_names: self.target_names.clone(),
            target_units: self.target_units.clone(),
            mean,
            std,
        })
    }
}
