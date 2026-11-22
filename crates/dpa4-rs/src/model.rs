//! Zero-copy `.dpa4` (`DPA4RS01`) model configuration, pre-assembled equivariant operator tables,
//! and optional `BootstrapPropertyOracle` loader for DeePMD `DPA-4` (`SeZM`: `Nano`, `Mini`, `Neo`, `Air`, `Plus`).

use std::collections::HashMap;

use mlip_core::{
    AdapterHeadWeights, BootstrapPropertyOracle, BundleArchive, EquivariantAdapterMember,
    LayerNormWeights, LinearWeights, TargetScaler,
};
use serde::{Deserialize, Serialize};

pub const DPA4_MAGIC: &[u8; 8] = b"DPA4RS01";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dpa4Config {
    pub tier: String,
    pub type_map: Vec<String>,
    pub ntypes: usize,
    pub rcut: f64,
    pub env_exp: [usize; 2],
    pub n_radial: usize,
    pub channels: usize,
    pub lmax: usize,
    pub mmax: usize,
    pub kmax: usize,
    pub n_blocks: usize,
    pub mixing_layers: usize,
    pub n_focus: usize,
    pub radial_so2_mode: String,
    pub radial_so2_rank: usize,
    pub ffn_dim: usize,
    pub n_grid: usize,
    pub eps: f32,
    pub version: f64,
    pub deg_norm_floor: f32,
    pub default_chg_spin: [f32; 2],
    pub weight_dtype: String,
}

#[derive(Clone, Debug)]
pub struct So2LayerWeights {
    /// Pre-permuted `(F, W0, W0)` where `W0 = (lmax + 1) * C`.
    pub w_m0_f: Vec<f32>,
    /// Transposed `(F, W0, W0)` (`(out, in)`) for SIMD AXPY backward!
    pub w_m0_ft: Vec<f32>,
    /// Pre-assembled 2x2 SO(2) block `(F, 2*W1, 2*W1)` where `W1 = lmax * C`:
    /// `[ [W_u, W_v], [-W_v, W_u] ]` matching `SO2Linear._build_so2_weight`.
    pub w_m1_f: Vec<f32>,
    /// Transposed `(F, 2*W1, 2*W1)` (`(out, in)`) for SIMD AXPY backward!
    pub w_m1_ft: Vec<f32>,
    /// Optional `GatedActivation` weight `(C, F * lmax * C)` for `k < mixing_layers - 1`.
    pub gate_w: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct Dpa4BlockWeights {
    pub post_so2_norm_scale: Vec<f32>, // (L+1, C)
    pub post_so2_norm_bias: Vec<f32>,  // (C,)
    pub pre_focus_w: Vec<f32>,         // (L+1, C, F*C)
    pub post_focus_w: Vec<f32>,        // (L+1, F*C, C)

    pub rad_hid_proj_w: Option<Vec<f32>>,  // (C, F*C) for F > 1 (Neo)
    pub focus_comp_norm: Option<Vec<f32>>, // (F, C) for F > 1 (Neo)
    pub focus_comp_w: Option<Vec<f32>>,    // (C, F) for F > 1 (Neo)

    pub rad_deg_w: Option<Vec<f32>>,     // ((L+1)*C_wide, K_deg * R)
    pub rad_deg_basis: Option<Vec<f32>>, // (R, C_wide)

    pub so2_layers: Vec<So2LayerWeights>,

    pub attn_qk_norm: Vec<f32>,   // (F, C)
    pub attn_q_w: Vec<f32>,       // (C, F*C)
    pub attn_k_w: Vec<f32>,       // (C, F*C)
    pub attn_logit_w: Vec<f32>,   // (C * F,) reshaped from (C, F, 1)
    pub attn_zeta: Vec<f32>,      // (F,) precomputed softplus(z_raw) + 1e-7
    pub attn_gate_norm: Vec<f32>, // (F, C)
    pub attn_gate_w: Vec<f32>,    // (C * F,) reshaped from (C, F, 1)

    pub mn_expand_w: Vec<f32>,   // (L+1, C, 3*C)
    pub mn_contract_w: Vec<f32>, // (L+1, 3*C, C)
    pub mn_sc_gate_w: Vec<f32>,  // (2*C, F*C)
    pub mn_res_scale: Vec<f32>,  // (F, C)

    pub pre_ffn_norm_scale: Vec<f32>, // (L+1, C)
    pub pre_ffn_norm_bias: Vec<f32>,  // (C,)
    pub ffn_l1_w: Vec<f32>,           // (L+1, C, 6*H)
    pub ffn_sc_gate_w: Vec<f32>,      // (2*H, H)
    pub ffn_left_w: Vec<f32>,         // (H, H)
    pub ffn_right_w: Vec<f32>,        // (H, H)
    pub ffn_out_w: Vec<f32>,          // (H, H)
    pub ffn_l2_w: Vec<f32>,           // (L+1, 3*H, C)
}

#[derive(Clone, Debug)]
pub struct Dpa4OutFfnWeights {
    pub l1_w: Vec<f32>,      // (L+1, C, 6*H)
    pub sc_gate_w: Vec<f32>, // (2*H, H)
    pub left_w: Vec<f32>,    // (2*H, 2*H)
    pub right_w: Vec<f32>,   // (2*H, 2*H)
    pub out_w: Vec<f32>,     // (2*H, H)
    pub l2_w: Vec<f32>,      // (L+1, 3*H, C)
}

#[derive(Clone, Debug)]
pub struct Dpa4Weights {
    pub cfg: Dpa4Config,
    pub symbol_to_atype: HashMap<String, usize>,

    // Precomputed indexing & SO(3)/Wigner tables
    pub dim_d: usize,                // (lmax + 1)^2
    pub dim_m: usize,                // 3 * lmax + 1
    pub degree_of_d: Vec<usize>,     // (D,)
    pub coeff_index_m: Vec<usize>,   // (D_m,)
    pub degree_index_m: Vec<usize>,  // (D_m,)
    pub rotate_inv_rescale: Vec<f32>, // (D,)
    pub balance_weight: Vec<f32>,    // (D,)

    pub to_grid_mat: Vec<f32>,   // (G, 3*D)
    pub from_grid_mat: Vec<f32>, // (3*D, G)

    pub wigner_l2_coeff: Option<Vec<f32>>,   // (35, 25)
    pub wigner_exp_l2: Option<Vec<[u8; 4]>>, // (35,)
    pub wigner_c_l3: Option<Vec<f32>>,       // (49, 84)
    pub wigner_exp_l3: Option<Vec<[u8; 4]>>, // (84,)
    pub wigner_c_l3l4: Option<Vec<f32>>,     // (130, 165)
    pub wigner_exp_l4: Option<Vec<[u8; 4]>>, // (165,)

    // Energy statistics (FP64 lossless!)
    pub out_bias: Vec<f64>,    // (ntypes,)
    pub bias_atom_e: Vec<f64>, // (ntypes,)

    // Initial Embeddings
    pub type_embed: Vec<f32>,       // (119, C)
    pub chg_embed: Vec<f32>,        // (200, C)
    pub spin_embed: Vec<f32>,       // (100, C)
    pub cs_mix_w: Vec<f32>,         // (2*C, C)
    pub cs_mix_b: Vec<f32>,         // (C,)
    pub bessel_freqs: Vec<f32>,     // (16,)
    pub rad_w0: Vec<f32>,           // (16, C)
    pub rad_norm_scale: Vec<f32>,   // (C,)
    pub rad_w3: Vec<f32>,           // (C, (L+1)*C)
    pub env_rbf_w1: Vec<f32>,       // (16, 32)
    pub env_rbf_w2: Vec<f32>,       // (32, 32)
    pub env_type_embed: Vec<f32>,   // (119, d_type)
    pub env_g_w1: Vec<f32>,         // (32 + 2*d_type, d_hid)
    pub env_g_w2: Vec<f32>,         // (d_hid, d_emb)
    pub env_out_proj: Vec<f32>,     // (d_emb * d_axis, 2*C)
    pub film_scale_norm: Vec<f32>,  // (C,)
    pub film_shift_norm: Vec<f32>,  // (C,)
    pub film_scale_strength: f32,
    pub film_shift_strength: f32,

    pub blocks: Vec<Dpa4BlockWeights>,
    pub out_ffn: Dpa4OutFfnWeights,

    pub fit_glu_w: Vec<f32>, // (C, 2*H)
    pub fit_glu_b: Vec<f32>, // (2*H,)
    pub fit_out_w: Vec<f32>, // (H,)

    pub oracle: Option<BootstrapPropertyOracle>,
}

#[derive(Deserialize)]
struct OracleHeaderMeta {
    desc_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    n_members: usize,
    n_heads_per_member: usize,
    target_names: Vec<String>,
    target_units: Vec<String>,
    scalers: Vec<TargetScaler>,
}

fn parse_exponents(flat: &[i32]) -> Vec<[u8; 4]> {
    flat.chunks_exact(4)
        .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])
        .collect()
}

/// Pre-assemble `SO2Linear` weights into focus-major `(F, W0, W0)` and `(F, 2*W1, 2*W1)` blocks.
fn assemble_so2_layer(
    w_m0_raw: &[f32],
    w_m1_raw: &[f32],
    gate_w: Option<Vec<f32>>,
    lmax: usize,
    c: usize,
    n_focus: usize,
) -> So2LayerWeights {
    let w0 = (lmax + 1) * c;
    // w_m0_raw has shape (w0, F * w0) -> view (w0, F, w0) -> permute (F, w0, w0)
    let mut w_m0_f = vec![0.0f32; n_focus * w0 * w0];
    for i in 0..w0 {
        for f in 0..n_focus {
            for o in 0..w0 {
                w_m0_f[(f * w0 + i) * w0 + o] = w_m0_raw[i * (n_focus * w0) + f * w0 + o];
            }
        }
    }

    let w1 = lmax * c;
    let two_w1 = 2 * w1;
    // w_m1_raw has shape (w1, F * 2 * w1) -> view (w1, F, 2*w1), split into W_u (:w1) and W_v (w1:)
    // Assembled 2x2 block per focus (2*w1, 2*w1):
    //   neg_in -> neg_out: +W_u,  neg_in -> pos_out: +W_v
    //   pos_in -> neg_out: -W_v,  pos_in -> pos_out: +W_u
    let mut w_m1_f = vec![0.0f32; n_focus * two_w1 * two_w1];
    for f in 0..n_focus {
        let base_f = f * two_w1 * two_w1;
        for i in 0..w1 {
            let src_row = &w_m1_raw[i * (n_focus * two_w1) + f * two_w1
                ..i * (n_focus * two_w1) + (f + 1) * two_w1];
            for o in 0..w1 {
                let wu = src_row[o];
                let wv = src_row[w1 + o];
                // neg_in (i), neg_out (o)
                w_m1_f[base_f + i * two_w1 + o] = wu;
                // neg_in (i), pos_out (w1 + o)
                w_m1_f[base_f + i * two_w1 + (w1 + o)] = wv;
                // pos_in (w1 + i), neg_out (o)
                w_m1_f[base_f + (w1 + i) * two_w1 + o] = -wv;
                // pos_in (w1 + i), pos_out (w1 + o)
                w_m1_f[base_f + (w1 + i) * two_w1 + (w1 + o)] = wu;
            }
        }
    }

    let mut w_m0_ft = vec![0.0f32; n_focus * w0 * w0];
    for f in 0..n_focus {
        for i in 0..w0 {
            for o in 0..w0 {
                w_m0_ft[(f * w0 + o) * w0 + i] = w_m0_f[(f * w0 + i) * w0 + o];
            }
        }
    }

    let mut w_m1_ft = vec![0.0f32; n_focus * two_w1 * two_w1];
    for f in 0..n_focus {
        let base_f = f * two_w1 * two_w1;
        for i in 0..two_w1 {
            for o in 0..two_w1 {
                w_m1_ft[base_f + o * two_w1 + i] = w_m1_f[base_f + i * two_w1 + o];
            }
        }
    }

    So2LayerWeights {
        w_m0_f,
        w_m0_ft,
        w_m1_f,
        w_m1_ft,
        gate_w,
    }
}

fn load_oracle_from_archive(ar: &BundleArchive<'_>) -> Result<Option<BootstrapPropertyOracle>, String> {
    let Some(oracle_val) = ar.extra.get("oracle") else {
        return Ok(None);
    };
    if oracle_val.is_null() {
        return Ok(None);
    }
    let meta: OracleHeaderMeta = serde_json::from_value(oracle_val.clone())
        .map_err(|e| format!("Invalid oracle metadata JSON: {e}"))?;

    let rd_lin = |prefix: &str, in_d: usize, out_d: usize| -> Result<LinearWeights, String> {
        Ok(LinearWeights {
            weight: ar.read_f32(&format!("{prefix}.weight"))?,
            bias: ar.read_f32(&format!("{prefix}.bias"))?,
            in_dim: in_d,
            out_dim: out_d,
        })
    };
    let rd_ln = |prefix: &str| -> Result<LayerNormWeights, String> {
        Ok(LayerNormWeights {
            weight: ar.read_f32(&format!("{prefix}.weight"))?,
            bias: ar.read_f32(&format!("{prefix}.bias"))?,
            eps: 1e-5,
        })
    };

    let h = meta.hidden_dim;
    let mut members = Vec::with_capacity(meta.n_members);
    for m in 0..meta.n_members {
        let op = format!("oracle.m{m}");
        let rbf_vec = ar.read_f32(&format!("{op}.rbf_centers"))?;
        let mut rbf_centers = [0.0f32; 16];
        rbf_centers.copy_from_slice(&rbf_vec[..16]);

        let mut heads = Vec::with_capacity(meta.n_heads_per_member);
        for hd in 0..meta.n_heads_per_member {
            let hp = format!("{op}.h{hd}");
            heads.push(AdapterHeadWeights {
                gate: rd_lin(&format!("{hp}.gate"), h, 1)?,
                head_norm: rd_ln(&format!("{hp}.norm"))?,
                head_fc0: rd_lin(&format!("{hp}.fc0"), 2 * h, h)?,
                head_fc1: rd_lin(&format!("{hp}.fc1"), h, h / 2)?,
                head_fc2: rd_lin(&format!("{hp}.fc2"), h / 2, meta.output_dim)?,
            });
        }

        members.push(EquivariantAdapterMember {
            desc_dim: meta.desc_dim,
            hidden_dim: h,
            output_dim: meta.output_dim,
            rbf_centers,
            in_proj: rd_lin(&format!("{op}.in_proj"), meta.desc_dim, h)?,
            dist_rbf: rd_lin(&format!("{op}.dist_rbf"), 16, 4)?,
            qkv: rd_lin(&format!("{op}.qkv"), h, 3 * h)?,
            out_proj: rd_lin(&format!("{op}.out_proj"), h, h)?,
            norm1: rd_ln(&format!("{op}.norm1"))?,
            norm2: rd_ln(&format!("{op}.norm2"))?,
            ffn0: rd_lin(&format!("{op}.ffn0"), h, 2 * h)?,
            ffn1: rd_lin(&format!("{op}.ffn1"), 2 * h, h)?,
            heads,
        });
    }

    Ok(Some(BootstrapPropertyOracle {
        target_names: meta.target_names,
        target_units: meta.target_units,
        scalers: meta.scalers,
        members,
    }))
}

impl Dpa4Weights {
    pub fn from_bytes(raw: &[u8]) -> Result<Self, String> {
        let ar = BundleArchive::parse(raw, DPA4_MAGIC)?;
        let cfg: Dpa4Config = serde_json::from_value(ar.config.clone())
            .map_err(|e| format!("Invalid DPA-4 config JSON: {e}"))?;

        let mut symbol_to_atype = HashMap::with_capacity(cfg.type_map.len());
        for (idx, sym) in cfg.type_map.iter().enumerate() {
            symbol_to_atype.insert(sym.clone(), idx);
        }

        let lmax = cfg.lmax;
        let c = cfg.channels;
        let dim_d = (lmax + 1) * (lmax + 1);
        let dim_m = 3 * lmax + 1;

        let mut degree_of_d = vec![0usize; dim_d];
        let mut rotate_inv_rescale = vec![1.0f32; dim_d];
        let mut balance_weight = vec![0.0f32; dim_d];
        for l in 0..=lmax {
            let rescale = if l <= 1 {
                1.0f32
            } else {
                (((2 * l + 1) as f32) / 3.0).sqrt()
            };
            let bw = 1.0f32 / (((2 * l + 1) * (lmax + 1) * c) as f32);
            for m in -(l as isize)..=(l as isize) {
                let d = ((l * l + l) as isize + m) as usize;
                degree_of_d[d] = l;
                rotate_inv_rescale[d] = rescale;
                balance_weight[d] = bw;
            }
        }

        // m-major reduced layout for mmax = 1:
        //   [0..=lmax]: m = 0 (d = l*(l+1))
        //   [lmax+1..=2*lmax]: m = -1 (d = l*(l+1) - 1, l = 1..=lmax)
        //   [2*lmax+1..=3*lmax]: m = +1 (d = l*(l+1) + 1, l = 1..=lmax)
        let mut coeff_index_m = Vec::with_capacity(dim_m);
        let mut degree_index_m = Vec::with_capacity(dim_m);
        for l in 0..=lmax {
            coeff_index_m.push(l * (l + 1));
            degree_index_m.push(l);
        }
        for l in 1..=lmax {
            coeff_index_m.push(l * (l + 1) - 1);
            degree_index_m.push(l);
        }
        for l in 1..=lmax {
            coeff_index_m.push(l * (l + 1) + 1);
            degree_index_m.push(l);
        }

        let to_grid_mat = ar.read_f32("const.to_grid_mat")?;
        let from_grid_mat = ar.read_f32("const.from_grid_mat")?;

        let (wigner_l2_coeff, wigner_exp_l2) = if lmax >= 2 {
            (
                Some(ar.read_f32("const.wigner_l2_coeff")?),
                Some(parse_exponents(&ar.read_i32("const.wigner_exp_l2")?)),
            )
        } else {
            (None, None)
        };
        let (wigner_c_l3, wigner_exp_l3) = if lmax == 3 {
            (
                Some(ar.read_f32("const.wigner_c_l3")?),
                Some(parse_exponents(&ar.read_i32("const.wigner_exp_l3")?)),
            )
        } else {
            (None, None)
        };
        let (wigner_c_l3l4, wigner_exp_l4) = if lmax == 4 {
            (
                Some(ar.read_f32("const.wigner_c_l3l4")?),
                Some(parse_exponents(&ar.read_i32("const.wigner_exp_l4")?)),
            )
        } else {
            (None, None)
        };

        let film_logs = ar.read_f32("film_strengths_log")?;
        let film_scale_strength = film_logs[0].exp();
        let film_shift_strength = film_logs[1].exp();

        let mut blocks = Vec::with_capacity(cfg.n_blocks);
        for b in 0..cfg.n_blocks {
            let sp = format!("b{b}");
            let mut so2_layers = Vec::with_capacity(cfg.mixing_layers);
            for k in 0..cfg.mixing_layers {
                let w_m0 = ar.read_f32(&format!("{sp}.so2_l{k}.w_m0"))?;
                let w_m1 = ar.read_f32(&format!("{sp}.so2_l{k}.w_m1"))?;
                let gate_w = if k < cfg.mixing_layers - 1 {
                    Some(ar.read_f32(&format!("{sp}.so2_l{k}.gate_w"))?)
                } else {
                    None
                };
                so2_layers.push(assemble_so2_layer(&w_m0, &w_m1, gate_w, lmax, c, cfg.n_focus));
            }

            let z_raw = ar.read_f32(&format!("{sp}.attn_z_raw"))?;
            let attn_zeta: Vec<f32> = z_raw
                .iter()
                .map(|&z| {
                    // numerically stable softplus(z) + 1e-7
                    let sp_val = if z > 20.0 {
                        z
                    } else {
                        (1.0 + z.exp()).ln()
                    };
                    sp_val + 1e-7
                })
                .collect();

            blocks.push(Dpa4BlockWeights {
                post_so2_norm_scale: ar.read_f32(&format!("{sp}.post_so2_norm.scale"))?,
                post_so2_norm_bias: ar.read_f32(&format!("{sp}.post_so2_norm.bias"))?,
                pre_focus_w: ar.read_f32(&format!("{sp}.pre_focus_w"))?,
                post_focus_w: ar.read_f32(&format!("{sp}.post_focus_w"))?,
                rad_hid_proj_w: if cfg.n_focus > 1 {
                    Some(ar.read_f32(&format!("{sp}.rad_hid_proj_w"))?)
                } else {
                    None
                },
                focus_comp_norm: if cfg.n_focus > 1 {
                    Some(ar.read_f32(&format!("{sp}.focus_comp_norm"))?)
                } else {
                    None
                },
                focus_comp_w: if cfg.n_focus > 1 {
                    Some(ar.read_f32(&format!("{sp}.focus_comp_w"))?)
                } else {
                    None
                },
                rad_deg_w: if cfg.radial_so2_mode == "degree_channel" {
                    Some(ar.read_f32(&format!("{sp}.rad_deg_w"))?)
                } else {
                    None
                },
                rad_deg_basis: if cfg.radial_so2_mode == "degree_channel" {
                    Some(ar.read_f32(&format!("{sp}.rad_deg_basis"))?)
                } else {
                    None
                },
                so2_layers,
                attn_qk_norm: ar.read_f32(&format!("{sp}.attn_qk_norm"))?,
                attn_q_w: ar.read_f32(&format!("{sp}.attn_q_w"))?,
                attn_k_w: ar.read_f32(&format!("{sp}.attn_k_w"))?,
                attn_logit_w: ar.read_f32(&format!("{sp}.attn_logit_w"))?,
                attn_zeta,
                attn_gate_norm: ar.read_f32(&format!("{sp}.attn_gate_norm"))?,
                attn_gate_w: ar.read_f32(&format!("{sp}.attn_gate_w"))?,
                mn_expand_w: ar.read_f32(&format!("{sp}.mn_expand_w"))?,
                mn_contract_w: ar.read_f32(&format!("{sp}.mn_contract_w"))?,
                mn_sc_gate_w: ar.read_f32(&format!("{sp}.mn_sc_gate_w"))?,
                mn_res_scale: ar.read_f32(&format!("{sp}.mn_res_scale"))?,
                pre_ffn_norm_scale: ar.read_f32(&format!("{sp}.pre_ffn_norm.scale"))?,
                pre_ffn_norm_bias: ar.read_f32(&format!("{sp}.pre_ffn_norm.bias"))?,
                ffn_l1_w: ar.read_f32(&format!("{sp}.ffn_l1_w"))?,
                ffn_sc_gate_w: ar.read_f32(&format!("{sp}.ffn_sc_gate_w"))?,
                ffn_left_w: ar.read_f32(&format!("{sp}.ffn_left_w"))?,
                ffn_right_w: ar.read_f32(&format!("{sp}.ffn_right_w"))?,
                ffn_out_w: ar.read_f32(&format!("{sp}.ffn_out_w"))?,
                ffn_l2_w: ar.read_f32(&format!("{sp}.ffn_l2_w"))?,
            });
        }

        let out_ffn = Dpa4OutFfnWeights {
            l1_w: ar.read_f32("out_ffn.l1_w")?,
            sc_gate_w: ar.read_f32("out_ffn.sc_gate_w")?,
            left_w: ar.read_f32("out_ffn.left_w")?,
            right_w: ar.read_f32("out_ffn.right_w")?,
            out_w: ar.read_f32("out_ffn.out_w")?,
            l2_w: ar.read_f32("out_ffn.l2_w")?,
        };

        let oracle = load_oracle_from_archive(&ar)?;

        Ok(Self {
            cfg,
            symbol_to_atype,
            dim_d,
            dim_m,
            degree_of_d,
            coeff_index_m,
            degree_index_m,
            rotate_inv_rescale,
            balance_weight,
            to_grid_mat,
            from_grid_mat,
            wigner_l2_coeff,
            wigner_exp_l2,
            wigner_c_l3,
            wigner_exp_l3,
            wigner_c_l3l4,
            wigner_exp_l4,
            out_bias: ar.read_f64("out_bias")?,
            bias_atom_e: ar.read_f64("bias_atom_e")?,
            type_embed: ar.read_f32("type_embed")?,
            chg_embed: ar.read_f32("chg_embed")?,
            spin_embed: ar.read_f32("spin_embed")?,
            cs_mix_w: ar.read_f32("cs_mix_w")?,
            cs_mix_b: ar.read_f32("cs_mix_b")?,
            bessel_freqs: ar.read_f32("bessel_freqs")?,
            rad_w0: ar.read_f32("rad_w0")?,
            rad_norm_scale: ar.read_f32("rad_norm_scale")?,
            rad_w3: ar.read_f32("rad_w3")?,
            env_rbf_w1: ar.read_f32("env_rbf_w1")?,
            env_rbf_w2: ar.read_f32("env_rbf_w2")?,
            env_type_embed: ar.read_f32("env_type_embed")?,
            env_g_w1: ar.read_f32("env_g_w1")?,
            env_g_w2: ar.read_f32("env_g_w2")?,
            env_out_proj: ar.read_f32("env_out_proj")?,
            film_scale_norm: ar.read_f32("film_scale_norm")?,
            film_shift_norm: ar.read_f32("film_shift_norm")?,
            film_scale_strength,
            film_shift_strength,
            blocks,
            out_ffn,
            fit_glu_w: ar.read_f32("fit.glu_w")?,
            fit_glu_b: ar.read_f32("fit.glu_b")?,
            fit_out_w: ar.read_f32("fit.out_w")?,
            oracle,
        })
    }

    pub fn map_atomic_numbers(&self, atomic_numbers: &[u8]) -> Result<Vec<usize>, String> {
        let mut out = Vec::with_capacity(atomic_numbers.len());
        for &z in atomic_numbers {
            let sym = mlip_core::symbol_from_z(z)?;
            let Some(&idx) = self.symbol_to_atype.get(sym) else {
                return Err(format!(
                    "Element {sym} (Z={z}) not present in DPA-4 type_map"
                ));
            };
            out.push(idx);
        }
        Ok(out)
    }
}
