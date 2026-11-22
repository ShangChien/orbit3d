use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub offset: usize,
    pub nbytes: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub orig_cols: usize,
    #[serde(default)]
    pub pad_cols: usize,
    #[serde(default)]
    pub blocks_per_row: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackboneConfig {
    pub emb_dim: usize,
    pub num_head: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub ffn_dim: usize,
    pub merge_level: usize,
    pub grid_len: f32,
    pub xyz_resolution: f32,
    pub num_xyz: usize,
    pub xyz_null_id: usize,
    pub num_atom_vocab: usize,
    pub atom_cls_id: usize,
    pub atom_unk_id: usize,
    pub atom_mask_id: usize,
    pub atom_null_id: usize,
    pub tree0_split_id: usize,
    pub rope_theta: f32,
    #[serde(default)]
    pub allowed_elements: Vec<String>,
    #[serde(default = "default_true")]
    pub non_trapping_budget: bool,
    #[serde(default)]
    pub has_budget_emb: bool,
}

fn default_true() -> bool {
    true
}

impl BackboneConfig {
    pub fn cls_phy_pos(&self) -> f32 {
        ((1usize << self.merge_level) as f32) * self.grid_len * 0.5
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionerConfig {
    pub num_props: usize,
    pub num_rbf: usize,
    pub num_freqs: usize,
    pub num_prefix_tokens: usize,
    pub modulation_cap: f32,
    pub rbf_gamma: f32,
    #[serde(default)]
    pub rope_center_prefix: bool,
    pub root_gate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct RefinerConfig {
    pub num_atom_classes: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub n_rbf: usize,
    pub d_pair: usize,
    pub max_disp_A: f32,
    pub rbf_gamma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage1CriticConfig {
    pub emb_dim: usize,
    pub num_targets: usize,
    pub hidden_dim: usize,
    pub geom_dim: usize,
    pub y_mean: Vec<f32>,
    pub y_scale: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage2CriticConfig {
    pub emb_dim: usize,
    pub num_targets: usize,
    pub num_atom_classes: usize,
    pub num_h_pools: usize,
    pub d_node: usize,
    pub n_rbf: usize,
    pub n_gnn_layers: usize,
    pub hidden_dim: usize,
    pub aux_hid: usize,
    pub cloud_feat_dim: usize,
    pub rbf_min: f32,
    pub rbf_max: f32,
    pub rbf_gamma: f32,
    pub edge_value_gate: bool,
    pub equivariant_multipole: bool,
    pub h_gate_scale: f32,
    pub y_mean: Vec<f32>,
    pub y_scale: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleHeader {
    pub format: String,
    pub quant_dtype: String,
    #[serde(default)]
    pub lora_merged_count: usize,
    #[serde(default)]
    pub target_cols: Vec<String>,
    #[serde(default)]
    pub target_units: Vec<String>,
    #[serde(default)]
    pub target_mean: Vec<f32>,
    #[serde(default)]
    pub target_scale: Vec<f32>,
    pub backbone: BackboneConfig,
    pub conditioner: Option<ConditionerConfig>,
    pub refiner: Option<RefinerConfig>,
    pub stage1_critic: Option<Stage1CriticConfig>,
    pub stage2_critic: Option<Stage2CriticConfig>,
    pub tensors: HashMap<String, TensorMeta>,
}

#[derive(Debug, Clone)]
pub struct PrecomputedCond {
    pub targets_norm: Vec<f32>,
    pub active_mask: Vec<bool>,
    pub has_any: bool,
    pub root_delta: Vec<f32>,            // (emb_dim,)
    pub prefix_tokens: Vec<f32>,         // (P * emb_dim)
    pub num_prefix: usize,
    pub layer_scale: Vec<Vec<f32>>,      // [L][emb_dim] (already 1.0 + scale)
    pub layer_shift: Vec<Vec<f32>>,      // [L][emb_dim]
    pub out_scale: Vec<f32>,             // (emb_dim,) (already 1.0 + scale)
    pub out_shift: Vec<f32>,             // (emb_dim,)
}
