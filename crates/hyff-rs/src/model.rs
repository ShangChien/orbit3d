//! Zero-copy `.hyff` (`HYFFRS01`) model configuration and weight loader for `HyFF-L2`
//! (`Presto`, `Vivace`, `Moderato`).

use mlip_core::BundleArchive;
use serde::{Deserialize, Serialize};

pub const HYFF_MAGIC: &[u8; 8] = b"HYFFRS01";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyffConfig {
    pub model_id: String,
    pub tier: String,
    pub backbone: String,
    pub hidden: usize,
    pub tensor_hidden: usize,
    pub edge_hidden: usize,
    pub expert_hidden: usize,
    pub radial_hidden: usize,
    pub interactions: usize,
    pub n_rbf: usize,
    pub expert_count: usize,
    pub cutoff: f64,
    pub layer_cutoffs: Vec<f64>,
    pub switch_width: f64,
    pub condition_spin: bool,
    pub higher_invariants: bool,
    pub weight_dtype: String,
}

#[derive(Clone, Debug)]
pub struct HyffBlockWeights {
    pub block_index: usize,
    pub input_is_scalar: bool,
    pub local_cross: bool,
    pub terminal: bool,
    pub msg_chunks: usize, // 3 for b == 0, 7 for b > 0
    pub out_dim: usize,    // H + 2*T for terminal, H + 4*T otherwise
    pub layer_cutoff: f32,

    pub scalar_norm_w: Vec<f32>, // (H,)
    pub scalar_norm_b: Vec<f32>, // (H,)
    pub update_norm_w: Vec<f32>, // (H,)
    pub update_norm_b: Vec<f32>, // (H,)
    pub readout_norm_w: Vec<f32>, // (H,)
    pub readout_norm_b: Vec<f32>, // (H,)

    pub vector_tensor_gain: Option<Vec<f32>>, // (T,) iff local_cross
    pub vector_vector_gain: Option<Vec<f32>>, // (T,) iff local_cross

    pub sm0_w: Vec<f32>, // (H, H)
    pub sm0_b: Vec<f32>, // (H,)
    pub sm2_w: Vec<f32>, // (M_b * C, H)
    pub sm2_b: Vec<f32>, // (M_b * C,)

    pub rf0_w: Vec<f32>, // (R_H, K)
    pub rf0_b: Vec<f32>, // (R_H,)
    pub rf2_w: Vec<f32>, // (M_b * C, R_H)
    pub rf2_b: Vec<f32>, // (M_b * C,)

    pub vector_u_w: Vec<f32>, // (T, T)
    pub vector_v_w: Vec<f32>, // (T, T)
    pub tensor_u_w: Vec<f32>, // (T, T)
    pub tensor_v_w: Vec<f32>, // (T, T)

    pub inv_vec_out_w: Vec<f32>, // (H, T)
    pub inv_ten_out_w: Vec<f32>, // (H, T)

    pub edge_vec_in_w: Option<Vec<f32>>, // (C, T) for b > 0
    pub edge_ten_in_w: Option<Vec<f32>>, // (C, T) for b > 0

    pub edge_s_out_w: Vec<f32>, // (H, C)
    pub edge_v_out_w: Vec<f32>, // (T, C)
    pub edge_t_out_w: Vec<f32>, // (T, C)

    pub expert_w0: Vec<f32>, // (83, E_H, H + 5*T)
    pub expert_b0: Vec<f32>, // (83, E_H)
    pub expert_w2: Vec<f32>, // (83, O_b, E_H)
    pub expert_b2: Vec<f32>, // (83, O_b)

    pub ro0_w: Vec<f32>, // (H, H)
    pub ro0_b: Vec<f32>, // (H,)
    pub ro2_w: Vec<f32>, // (1, H)
    pub ro2_b: f32,      // scalar
}

#[derive(Clone, Debug)]
pub struct HyffWeights {
    pub cfg: HyffConfig,
    pub element_lut: Vec<i32>,       // (119,)
    pub embedding_w: Vec<f32>,       // (119, H)
    pub rbf_centers: Vec<f32>,       // (32,)
    pub rbf_gamma: f32,              // scalar
    pub element_offset_f64: Vec<f64>, // (119,)
    pub pair_coeff_w: Vec<f32>,      // (6889, 32)
    pub state_proj_w: Vec<f32>,      // (H, 2)
    pub blocks: Vec<HyffBlockWeights>,
}

impl HyffWeights {
    pub fn from_bytes(raw: &[u8]) -> Result<Self, String> {
        let ar = BundleArchive::parse(raw, HYFF_MAGIC)?;
        let cfg: HyffConfig = serde_json::from_value(ar.config.clone())
            .map_err(|e| format!("Invalid HyFF config JSON: {e}"))?;

        let element_lut = ar.read_i32("element_lut")?;
        let embedding_w = ar.read_f32("embedding.weight")?;
        let rbf_centers = ar.read_f32("rbf.centers")?;
        let rbf_gamma = ar.read_f32("rbf.gamma")?[0];
        let element_offset_f64 = ar.read_f64("element_offset.weight")?;
        let pair_coeff_w = ar.read_f32("pair_coeff.weight")?;
        let state_proj_w = ar.read_f32("state_projection.weight")?;

        let n_int = cfg.interactions;
        let h = cfg.hidden;
        let t = cfg.tensor_hidden;
        let mut blocks = Vec::with_capacity(n_int);

        for b in 0..n_int {
            let p = format!("blocks.{b}");
            let input_is_scalar = b == 0;
            let local_cross = n_int >= 2 && b == n_int - 2;
            let terminal = b == n_int - 1;
            let msg_chunks = if input_is_scalar { 3 } else { 7 };
            let out_dim = if terminal { h + 2 * t } else { h + 4 * t };

            let vector_tensor_gain = if ar.has(&format!("{p}.vector_tensor_gain")) {
                Some(ar.read_f32(&format!("{p}.vector_tensor_gain"))?)
            } else {
                None
            };
            let vector_vector_gain = if ar.has(&format!("{p}.vector_vector_gain")) {
                Some(ar.read_f32(&format!("{p}.vector_vector_gain"))?)
            } else {
                None
            };

            let edge_vec_in_w = if b > 0 {
                Some(ar.read_f32(&format!("{p}.edge_vector_in.weight"))?)
            } else {
                None
            };
            let edge_ten_in_w = if b > 0 {
                Some(ar.read_f32(&format!("{p}.edge_tensor_in.weight"))?)
            } else {
                None
            };

            let ro2_b = ar.read_f32(&format!("readouts.{b}.2.bias"))?[0];

            blocks.push(HyffBlockWeights {
                block_index: b,
                input_is_scalar,
                local_cross,
                terminal,
                msg_chunks,
                out_dim,
                layer_cutoff: cfg.layer_cutoffs[b] as f32,
                scalar_norm_w: ar.read_f32(&format!("{p}.scalar_norm.weight"))?,
                scalar_norm_b: ar.read_f32(&format!("{p}.scalar_norm.bias"))?,
                update_norm_w: ar.read_f32(&format!("{p}.update_norm.weight"))?,
                update_norm_b: ar.read_f32(&format!("{p}.update_norm.bias"))?,
                readout_norm_w: ar.read_f32(&format!("readout_norms.{b}.weight"))?,
                readout_norm_b: ar.read_f32(&format!("readout_norms.{b}.bias"))?,
                vector_tensor_gain,
                vector_vector_gain,
                sm0_w: ar.read_f32(&format!("{p}.scalar_message.0.weight"))?,
                sm0_b: ar.read_f32(&format!("{p}.scalar_message.0.bias"))?,
                sm2_w: ar.read_f32(&format!("{p}.scalar_message.2.weight"))?,
                sm2_b: ar.read_f32(&format!("{p}.scalar_message.2.bias"))?,
                rf0_w: ar.read_f32(&format!("{p}.radial_filter.0.weight"))?,
                rf0_b: ar.read_f32(&format!("{p}.radial_filter.0.bias"))?,
                rf2_w: ar.read_f32(&format!("{p}.radial_filter.2.weight"))?,
                rf2_b: ar.read_f32(&format!("{p}.radial_filter.2.bias"))?,
                vector_u_w: ar.read_f32(&format!("{p}.vector_u.weight"))?,
                vector_v_w: ar.read_f32(&format!("{p}.vector_v.weight"))?,
                tensor_u_w: ar.read_f32(&format!("{p}.tensor_u.weight"))?,
                tensor_v_w: ar.read_f32(&format!("{p}.tensor_v.weight"))?,
                inv_vec_out_w: ar.read_f32(&format!("{p}.invariant_vector_out.weight"))?,
                inv_ten_out_w: ar.read_f32(&format!("{p}.invariant_tensor_out.weight"))?,
                edge_vec_in_w,
                edge_ten_in_w,
                edge_s_out_w: ar.read_f32(&format!("{p}.edge_scalar_out.weight"))?,
                edge_v_out_w: ar.read_f32(&format!("{p}.edge_vector_out.weight"))?,
                edge_t_out_w: ar.read_f32(&format!("{p}.edge_tensor_out.weight"))?,
                expert_w0: ar.read_f32(&format!("{p}.expert_w0"))?,
                expert_b0: ar.read_f32(&format!("{p}.expert_b0"))?,
                expert_w2: ar.read_f32(&format!("{p}.expert_w2"))?,
                expert_b2: ar.read_f32(&format!("{p}.expert_b2"))?,
                ro0_w: ar.read_f32(&format!("readouts.{b}.0.weight"))?,
                ro0_b: ar.read_f32(&format!("readouts.{b}.0.bias"))?,
                ro2_w: ar.read_f32(&format!("readouts.{b}.2.weight"))?,
                ro2_b,
            });
        }

        Ok(Self {
            cfg,
            element_lut,
            embedding_w,
            rbf_centers,
            rbf_gamma,
            element_offset_f64,
            pair_coeff_w,
            state_proj_w,
            blocks,
        })
    }

    pub fn validate_elements(&self, atomic_numbers: &[u8]) -> Result<Vec<usize>, String> {
        let mut experts = Vec::with_capacity(atomic_numbers.len());
        for &z in atomic_numbers {
            let lut = self
                .element_lut
                .get(z as usize)
                .copied()
                .unwrap_or(-1);
            if lut < 0 || (lut as usize) >= self.cfg.expert_count {
                return Err(format!(
                    "Atomic number Z={z} is outside HyFF-L2 supported 83-element vocabulary (Z=1..83)"
                ));
            }
            experts.push(lut as usize);
        }
        Ok(experts)
    }
}
