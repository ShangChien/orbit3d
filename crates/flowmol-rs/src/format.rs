use crate::config::{BundleHeader, TensorMeta};
use half::f16;

pub const MAGIC: &[u8; 8] = b"FM118\x01\x00\x00";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    F32 = 0,
    F16 = 1,
    Q8_0 = 2,
}

#[derive(Debug, Clone)]
pub struct ModelBundle {
    pub header: BundleHeader,
    pub payload_offset: usize,
    raw: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearView<'a> {
    pub mode: QuantMode,
    pub out_features: usize,
    pub orig_cols: usize,
    pub pad_cols: usize,
    pub blocks_per_row: usize,
    pub weight_bytes: &'a [u8],
    pub bias: Option<&'a [f32]>,
}

impl ModelBundle {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.len() < 16 || &bytes[0..8] != MAGIC {
            return Err("Invalid .fm118 magic header".to_string());
        }
        let hdr_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let payload_offset = 16 + hdr_len;
        if bytes.len() < payload_offset {
            return Err("Truncated .fm118 header".to_string());
        }
        let header: BundleHeader = serde_json::from_slice(&bytes[16..payload_offset])
            .map_err(|e| format!("Failed to parse .fm118 JSON header: {e}"))?;
        if header.format != "flowmol_rs_v1" {
            return Err(format!("Unsupported bundle format '{}'", header.format));
        }
        Ok(Self {
            header,
            payload_offset,
            raw: bytes,
        })
    }

    pub fn payload(&self) -> &[u8] {
        &self.raw[self.payload_offset..]
    }

    pub fn tensor_meta(&self, name: &str) -> Result<&TensorMeta, String> {
        self.header
            .tensors
            .get(name)
            .ok_or_else(|| format!("Missing tensor '{name}' in .fm118 bundle"))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], String> {
        let m = self.tensor_meta(name)?;
        let start = self.payload_offset + m.offset;
        let end = start + m.nbytes;
        if end > self.raw.len() {
            return Err(format!("Tensor '{name}' out of bounds ({end} > {})", self.raw.len()));
        }
        Ok(&self.raw[start..end])
    }

    pub fn f32_slice(&self, name: &str) -> Result<&[f32], String> {
        let m = self.tensor_meta(name)?;
        if m.dtype != "f32" {
            return Err(format!("Expected f32 tensor for '{name}', got '{}'", m.dtype));
        }
        let b = self.tensor_bytes(name)?;
        Ok(bytemuck::cast_slice(b))
    }

    pub fn f32_slice_opt(&self, name: &str) -> Option<&[f32]> {
        if self.header.tensors.contains_key(name) {
            self.f32_slice(name).ok()
        } else {
            None
        }
    }

    pub fn linear_view(&self, prefix: &str) -> Result<LinearView<'_>, String> {
        let w_name = format!("{prefix}.weight");
        let m = self.tensor_meta(&w_name)?;
        let wb = self.tensor_bytes(&w_name)?;
        let mode = match m.dtype.as_str() {
            "f32" => QuantMode::F32,
            "f16" => QuantMode::F16,
            "q8_0" => QuantMode::Q8_0,
            other => return Err(format!("Unknown weight dtype '{other}' on '{w_name}'")),
        };
        let b_name = format!("{prefix}.bias");
        let bias = self.f32_slice_opt(&b_name);
        Ok(LinearView {
            mode,
            out_features: m.shape[0],
            orig_cols: m.orig_cols,
            pad_cols: m.shape[1],
            blocks_per_row: m.blocks_per_row,
            weight_bytes: wb,
            bias,
        })
    }
}

/// Evaluate `y = x @ W.T + bias` on CPU for `[rows, orig_cols] -> [rows, out_features]`.
pub fn eval_linear_cpu(lin: &LinearView<'_>, x: &[f32], rows: usize, out: &mut [f32]) {
    let in_c = lin.orig_cols;
    let out_c = lin.out_features;
    debug_assert_eq!(x.len(), rows * in_c);
    debug_assert_eq!(out.len(), rows * out_c);

    match lin.mode {
        QuantMode::F32 => {
            let w: &[f32] = bytemuck::cast_slice(lin.weight_bytes);
            for r in 0..rows {
                let xr = &x[r * in_c..(r + 1) * in_c];
                let yr = &mut out[r * out_c..(r + 1) * out_c];
                for o in 0..out_c {
                    let wr = &w[o * in_c..(o + 1) * in_c];
                    let mut acc = lin.bias.map_or(0.0, |b| b[o]);
                    for k in 0..in_c {
                        acc += xr[k] * wr[k];
                    }
                    yr[o] = acc;
                }
            }
        }
        QuantMode::F16 => {
            let w16: &[u16] = bytemuck::cast_slice(lin.weight_bytes);
            let pad_c = lin.pad_cols;
            // Unpack each weight row once across all `rows` when `rows > 1`? Even better: direct f16->f32!
            // Wait: on E=2000 rows, unpacking `w` per row vs looping `o` outer and `r` inner reuses unpacked `wr` 2000 times!
            let mut wr_buf = vec![0.0f32; in_c];
            for o in 0..out_c {
                let w_row = &w16[o * pad_c..o * pad_c + in_c];
                for k in 0..in_c {
                    wr_buf[k] = f16::from_bits(w_row[k]).to_f32();
                }
                let b0 = lin.bias.map_or(0.0, |b| b[o]);
                for r in 0..rows {
                    let xr = &x[r * in_c..(r + 1) * in_c];
                    let mut acc = b0;
                    for k in 0..in_c {
                        acc += xr[k] * wr_buf[k];
                    }
                    out[r * out_c + o] = acc;
                }
            }
        }
        QuantMode::Q8_0 => {
            let bpr = lin.blocks_per_row;
            let mut wr_buf = vec![0.0f32; lin.pad_cols];
            for o in 0..out_c {
                for b in 0..bpr {
                    let blk_off = (o * bpr + b) * 36;
                    let scale = f32::from_le_bytes(lin.weight_bytes[blk_off..blk_off + 4].try_into().unwrap());
                    let qs = &lin.weight_bytes[blk_off + 4..blk_off + 36];
                    let base = b * 32;
                    for j in 0..32 {
                        wr_buf[base + j] = (qs[j] as i8 as f32) * scale;
                    }
                }
                let wr = &wr_buf[..in_c];
                let b0 = lin.bias.map_or(0.0, |b| b[o]);
                for r in 0..rows {
                    let xr = &x[r * in_c..(r + 1) * in_c];
                    let mut acc = b0;
                    for k in 0..in_c {
                        acc += xr[k] * wr[k];
                    }
                    out[r * out_c + o] = acc;
                }
            }
        }
    }
}

pub fn eval_layernorm_cpu(x: &mut [f32], rows: usize, cols: usize, weight: &[f32], bias: &[f32], eps: f32) {
    let inv_n = 1.0 / (cols as f32);
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let mut mean = 0.0f32;
        for &v in row.iter() {
            mean += v;
        }
        mean *= inv_n;
        let mut var = 0.0f32;
        for &v in row.iter() {
            let d = v - mean;
            var += d * d;
        }
        var *= inv_n;
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            row[c] = (row[c] - mean) * inv_std * weight[c] + bias[c];
        }
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
