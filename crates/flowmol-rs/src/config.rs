use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMolConfig {
    pub atom_types: Vec<String>,
    pub formal_charges: Vec<i32>,
    pub n_charges: usize,
    pub n_bond_types: usize,
    pub n_hidden_scalars: usize,
    pub n_hidden_edge_feats: usize,
    pub n_vec_channels: usize,
    pub n_cp_feats: usize,
    pub n_molecule_updates: usize,
    pub convs_per_update: usize,
    pub n_message_gvps: usize,
    pub n_update_gvps: usize,
    pub n_pos_update_gvps: usize,
    #[serde(default)]
    pub use_dst_feats: bool,
    #[serde(default = "default_true")]
    pub separate_mol_updaters: bool,
    #[serde(default = "default_true")]
    pub update_edge_w_distance: bool,
    #[serde(default = "default_msg_norm")]
    pub message_norm: serde_json::Value,
    pub rbf_dim: usize,
    pub rbf_dmax: f32,
    pub time_emb_dim: usize,
    #[serde(default = "default_true")]
    pub use_periodic_features: bool,
    #[serde(default = "default_true")]
    pub self_conditioning: bool,
    #[serde(default = "default_schedule")]
    pub schedule_type: String,
    #[serde(default = "default_one")]
    pub nu_x: f32,
    #[serde(default = "default_two")]
    pub nu_a: f32,
    #[serde(default = "default_two")]
    pub nu_c: f32,
    #[serde(default = "default_one_half")]
    pub nu_e: f32,
    #[serde(default = "default_prior_std")]
    pub prior_std: f32,
    #[serde(default = "default_true")]
    pub size_scale_prior: bool,
}

fn default_true() -> bool {
    true
}
fn default_msg_norm() -> serde_json::Value {
    serde_json::json!(100.0)
}
fn default_schedule() -> String {
    "cosine".to_string()
}
fn default_one() -> f32 {
    1.0
}
fn default_two() -> f32 {
    2.0
}
fn default_one_half() -> f32 {
    1.5
}
fn default_prior_std() -> f32 {
    1.15
}

impl FlowMolConfig {
    pub fn n_atom_types(&self) -> usize {
        self.atom_types.len()
    }

    pub fn message_norm_factor(&self, n_nodes: usize) -> f32 {
        if let Some(s) = self.message_norm.as_str() {
            if s == "mean" {
                return (n_nodes.saturating_sub(1)).max(1) as f32;
            }
            if s == "sum" {
                return 1.0;
            }
        }
        self.message_norm.as_f64().unwrap_or(100.0) as f32
    }

    pub fn node_in_dim(&self) -> usize {
        let time_out_dim = 1 + 2 * (self.time_emb_dim / 2);
        let mut d = (self.n_atom_types() + 1) + (self.n_charges + 1) + time_out_dim;
        if self.use_periodic_features {
            d += 34;
        }
        if self.self_conditioning {
            d += self.n_atom_types() + self.n_charges;
            if self.use_periodic_features {
                d += 34;
            }
        }
        d
    }

    pub fn edge_in_dim(&self) -> usize {
        let mut d = self.n_bond_types + 1;
        if self.self_conditioning {
            d += self.n_bond_types + self.rbf_dim;
        }
        d
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionerConfig {
    pub emb_dim: usize,
    pub num_props: usize,
    pub num_layers: usize,
    pub num_rbf: usize,
    pub num_freqs: usize,
    pub rbf_gamma: f32,
    pub modulation_cap: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub orig_cols: usize,
    #[serde(default)]
    pub blocks_per_row: usize,
    pub offset: usize,
    pub nbytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleHeader {
    pub format: String,
    pub quant_dtype: String,
    #[serde(default)]
    pub lora_merged_count: usize,
    #[serde(default)]
    pub step: usize,
    #[serde(default)]
    pub samples_seen: u64,
    pub config: FlowMolConfig,
    #[serde(default)]
    pub conditioner: Option<ConditionerConfig>,
    #[serde(default)]
    pub active_elements: Vec<String>,
    #[serde(default)]
    pub active_charges: Vec<i32>,
    #[serde(default)]
    pub n_atoms_hist: HashMap<String, u64>,
    #[serde(default)]
    pub per_source_n_atoms_hist: HashMap<String, HashMap<String, u64>>,
    #[serde(default)]
    pub per_source_active_elements: HashMap<String, Vec<String>>,
    pub tensors: HashMap<String, TensorMeta>,
}

#[derive(Debug, Clone)]
pub struct ForwardOutput {
    pub x: Vec<[f32; 3]>,
    pub a_logits: Vec<f32>, // (N, n_atom_types)
    pub c_logits: Vec<f32>, // (N, n_charges)
    pub e_logits: Vec<f32>, // (E_up, n_bond_types)
}

#[derive(Debug, Clone)]
pub struct SelfCondState {
    pub x: Vec<[f32; 3]>,
    pub a_probs: Vec<f32>, // (N, n_atom_types)
    pub c_probs: Vec<f32>, // (N, n_charges)
    pub e_probs: Vec<f32>, // (E_up, n_bond_types)
}

#[derive(Debug, Clone)]
pub struct PrecomputedModulations {
    pub layer_scale: Vec<Vec<f32>>, // [L][S]
    pub layer_shift: Vec<Vec<f32>>, // [L][S]
    pub out_scale: Vec<f32>,        // [S]
    pub out_shift: Vec<f32>,        // [S]
}
