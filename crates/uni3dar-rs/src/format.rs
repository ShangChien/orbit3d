use crate::config::{BundleHeader, TensorMeta};
use half::f16;

pub const MAGIC: &[u8; 8] = b"U3DM\x01\x00\x00\x00";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    F32 = 0,
    F16 = 1,
    Q8_0 = 2,
    Q4_0 = 3,
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

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// High-accuracy `erf(x)` (Abramowitz & Stegun 7.1.26, max abs error `< 1.5e-7`) for exact `F.gelu`.
#[inline]
pub fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

#[inline]
pub fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
}

impl ModelBundle {
    pub fn from_bytes(raw: Vec<u8>) -> Result<Self, String> {
        if raw.len() < 16 {
            return Err("Bundle too small (< 16 bytes)".to_string());
        }
        if &raw[0..8] != MAGIC {
            return Err("Invalid .u3m magic header (expected U3DM v1)".to_string());
        }
        let hdr_len = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
        let payload_offset = 16 + hdr_len;
        if raw.len() < payload_offset {
            return Err("Truncated .u3m header".to_string());
        }
        let header: BundleHeader =
            serde_json::from_slice(&raw[16..payload_offset]).map_err(|e| format!("Invalid .u3m JSON header: {e}"))?;
        Ok(Self {
            header,
            payload_offset,
            raw,
        })
    }

    pub fn tensor_meta(&self, name: &str) -> Result<&TensorMeta, String> {
        self.header
            .tensors
            .get(name)
            .ok_or_else(|| format!("Missing tensor '{name}' in .u3m bundle"))
    }

    pub fn raw_slice(&self, meta: &TensorMeta) -> Result<&[u8], String> {
        let start = self.payload_offset + meta.offset;
        let end = start + meta.nbytes;
        if end > self.raw.len() {
            return Err(format!("Tensor slice out of bounds ({start}..{end} > {})", self.raw.len()));
        }
        Ok(&self.raw[start..end])
    }

    pub fn f32_slice(&self, name: &str) -> Result<&[f32], String> {
        let m = self.tensor_meta(name)?;
        if m.dtype != "f32" {
            return Err(format!("Expected f32 tensor '{name}', got '{}'", m.dtype));
        }
        Ok(bytemuck::cast_slice(self.raw_slice(m)?))
    }

    pub fn f32_slice_opt(&self, name: &str) -> Option<&[f32]> {
        self.header
            .tensors
            .get(name)
            .and_then(|m| if m.dtype == "f32" { self.raw_slice(m).ok() } else { None })
            .map(bytemuck::cast_slice)
    }

    pub fn linear_view(&self, prefix: &str) -> Result<LinearView<'_>, String> {
        let wm = self.tensor_meta(&format!("{prefix}.weight"))?;
        let mode = match wm.dtype.as_str() {
            "f32" => QuantMode::F32,
            "f16" => QuantMode::F16,
            "q8_0" => QuantMode::Q8_0,
            "q4_0" => QuantMode::Q4_0,
            other => return Err(format!("Unsupported linear dtype '{other}' on '{prefix}'")),
        };
        let out_features = wm.shape[0];
        let orig_cols = if wm.orig_cols > 0 { wm.orig_cols } else { wm.shape[1] };
        let pad_cols = if wm.pad_cols > 0 { wm.pad_cols } else { orig_cols };
        let weight_bytes = self.raw_slice(wm)?;
        let bias = self.f32_slice_opt(&format!("{prefix}.bias"));
        Ok(LinearView {
            mode,
            out_features,
            orig_cols,
            pad_cols,
            blocks_per_row: wm.blocks_per_row,
            weight_bytes,
            bias,
        })
    }

    /// Evaluate `Y = X @ W^T + bias` on CPU (`X`: `(rows, in_cols)` -> `out`: `(rows, out_cols)`).
    pub fn eval_linear(&self, prefix: &str, x: &[f32], rows: usize, out: &mut [f32]) -> Result<(), String> {
        let lv = self.linear_view(prefix)?;
        let in_c = lv.orig_cols;
        let out_c = lv.out_features;
        debug_assert_eq!(x.len(), rows * in_c, "eval_linear input size mismatch on {prefix}");
        debug_assert_eq!(out.len(), rows * out_c, "eval_linear output size mismatch on {prefix}");

        match lv.mode {
            QuantMode::F32 => {
                let w: &[f32] = bytemuck::cast_slice(lv.weight_bytes);
                for r in 0..rows {
                    let xr = &x[r * in_c..(r + 1) * in_c];
                    let orow = &mut out[r * out_c..(r + 1) * out_c];
                    for c in 0..out_c {
                        let wr = &w[c * lv.pad_cols..c * lv.pad_cols + in_c];
                        let mut acc = lv.bias.map_or(0.0, |b| b[c]);
                        for k in 0..in_c {
                            acc += xr[k] * wr[k];
                        }
                        orow[c] = acc;
                    }
                }
            }
            QuantMode::F16 => {
                let w: &[u16] = bytemuck::cast_slice(lv.weight_bytes);
                for r in 0..rows {
                    let xr = &x[r * in_c..(r + 1) * in_c];
                    let orow = &mut out[r * out_c..(r + 1) * out_c];
                    for c in 0..out_c {
                        let wr = &w[c * lv.pad_cols..c * lv.pad_cols + in_c];
                        let mut acc = lv.bias.map_or(0.0, |b| b[c]);
                        for k in 0..in_c {
                            acc += xr[k] * f16::from_bits(wr[k]).to_f32();
                        }
                        orow[c] = acc;
                    }
                }
            }
            QuantMode::Q8_0 => {
                let bpr = lv.blocks_per_row;
                let wb = lv.weight_bytes;
                for r in 0..rows {
                    let xr = &x[r * in_c..(r + 1) * in_c];
                    let orow = &mut out[r * out_c..(r + 1) * out_c];
                    for c in 0..out_c {
                        let mut acc = lv.bias.map_or(0.0, |b| b[c]);
                        for bi in 0..bpr {
                            let boff = (c * bpr + bi) * 36;
                            let scale = f32::from_le_bytes(wb[boff..boff + 4].try_into().unwrap());
                            let k_base = bi * 32;
                            let k_lim = (in_c - k_base).min(32);
                            let mut b_acc = 0.0f32;
                            for j in 0..k_lim {
                                let q = wb[boff + 4 + j] as i8;
                                b_acc += xr[k_base + j] * (q as f32);
                            }
                            acc += b_acc * scale;
                        }
                        orow[c] = acc;
                    }
                }
            }
            QuantMode::Q4_0 => {
                let bpr = lv.blocks_per_row;
                let wb = lv.weight_bytes;
                for r in 0..rows {
                    let xr = &x[r * in_c..(r + 1) * in_c];
                    let orow = &mut out[r * out_c..(r + 1) * out_c];
                    for c in 0..out_c {
                        let mut acc = lv.bias.map_or(0.0, |b| b[c]);
                        for bi in 0..bpr {
                            let boff = (c * bpr + bi) * 20;
                            let scale = f32::from_le_bytes(wb[boff..boff + 4].try_into().unwrap());
                            let k_base = bi * 32;
                            let k_lim = (in_c - k_base).min(32);
                            let mut b_acc = 0.0f32;
                            for j in 0..k_lim {
                                let byte = wb[boff + 4 + (j >> 1)];
                                let nib = if (j & 1) == 0 { byte & 0x0F } else { byte >> 4 };
                                // Sign-extend 4-bit two's complement nibble [-8..+7]
                                let q = ((nib as i8) << 4) >> 4;
                                b_acc += xr[k_base + j] * (q as f32);
                            }
                            acc += b_acc * scale;
                        }
                        orow[c] = acc;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn eval_rmsnorm_in_place(&self, weight_name: &str, x: &mut [f32], rows: usize, cols: usize, eps: f32) -> Result<(), String> {
        let w = self.f32_slice(weight_name)?;
        for r in 0..rows {
            let row = &mut x[r * cols..(r + 1) * cols];
            let sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / (cols as f32);
            let inv = 1.0 / (sq + eps).sqrt();
            for c in 0..cols {
                row[c] = row[c] * inv * w[c];
            }
        }
        Ok(())
    }

    pub fn eval_layernorm_in_place(&self, prefix: &str, x: &mut [f32], rows: usize, cols: usize, eps: f32) -> Result<(), String> {
        let w = self.f32_slice(&format!("{prefix}.weight"))?;
        let b = self.f32_slice(&format!("{prefix}.bias"))?;
        for r in 0..rows {
            let row = &mut x[r * cols..(r + 1) * cols];
            let mean = row.iter().sum::<f32>() / (cols as f32);
            let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (cols as f32);
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..cols {
                row[c] = (row[c] - mean) * inv * w[c] + b[c];
            }
        }
        Ok(())
    }

    #[inline]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw
    }

    /// Accumulate input gradient `dX += dY @ W` directly across `F32 / F16 / Q8_0 / Q4_0`
    /// (`dY`: `(rows, out_cols)`, `dx_accum`: `(rows, in_cols)`) for zero-copy `QLoRA` backward!
    pub fn eval_linear_bwd_dx(&self, prefix: &str, dy: &[f32], rows: usize, dx_accum: &mut [f32]) -> Result<(), String> {
        let lv = self.linear_view(prefix)?;
        let in_c = lv.orig_cols;
        let out_c = lv.out_features;
        debug_assert_eq!(dy.len(), rows * out_c, "eval_linear_bwd_dx dy mismatch on {prefix}");
        debug_assert_eq!(dx_accum.len(), rows * in_c, "eval_linear_bwd_dx dx mismatch on {prefix}");

        match lv.mode {
            QuantMode::F32 => {
                let w: &[f32] = bytemuck::cast_slice(lv.weight_bytes);
                for r in 0..rows {
                    let dyr = &dy[r * out_c..(r + 1) * out_c];
                    let dxr = &mut dx_accum[r * in_c..(r + 1) * in_c];
                    for (c, &g) in dyr.iter().enumerate() {
                        if g == 0.0 {
                            continue;
                        }
                        let wr = &w[c * lv.pad_cols..c * lv.pad_cols + in_c];
                        for k in 0..in_c {
                            dxr[k] += g * wr[k];
                        }
                    }
                }
            }
            QuantMode::F16 => {
                let w: &[u16] = bytemuck::cast_slice(lv.weight_bytes);
                for r in 0..rows {
                    let dyr = &dy[r * out_c..(r + 1) * out_c];
                    let dxr = &mut dx_accum[r * in_c..(r + 1) * in_c];
                    for (c, &g) in dyr.iter().enumerate() {
                        if g == 0.0 {
                            continue;
                        }
                        let wr = &w[c * lv.pad_cols..c * lv.pad_cols + in_c];
                        for k in 0..in_c {
                            dxr[k] += g * f16::from_bits(wr[k]).to_f32();
                        }
                    }
                }
            }
            QuantMode::Q8_0 => {
                let bpr = lv.blocks_per_row;
                let wb = lv.weight_bytes;
                for r in 0..rows {
                    let dyr = &dy[r * out_c..(r + 1) * out_c];
                    let dxr = &mut dx_accum[r * in_c..(r + 1) * in_c];
                    for (c, &g) in dyr.iter().enumerate() {
                        if g == 0.0 {
                            continue;
                        }
                        for bi in 0..bpr {
                            let boff = (c * bpr + bi) * 36;
                            let scale = g * f32::from_le_bytes(wb[boff..boff + 4].try_into().unwrap());
                            let k_base = bi * 32;
                            let k_lim = (in_c - k_base).min(32);
                            for j in 0..k_lim {
                                let q = wb[boff + 4 + j] as i8;
                                dxr[k_base + j] += scale * (q as f32);
                            }
                        }
                    }
                }
            }
            QuantMode::Q4_0 => {
                let bpr = lv.blocks_per_row;
                let wb = lv.weight_bytes;
                for r in 0..rows {
                    let dyr = &dy[r * out_c..(r + 1) * out_c];
                    let dxr = &mut dx_accum[r * in_c..(r + 1) * in_c];
                    for (c, &g) in dyr.iter().enumerate() {
                        if g == 0.0 {
                            continue;
                        }
                        for bi in 0..bpr {
                            let boff = (c * bpr + bi) * 20;
                            let scale = g * f32::from_le_bytes(wb[boff..boff + 4].try_into().unwrap());
                            let k_base = bi * 32;
                            let k_lim = (in_c - k_base).min(32);
                            for j in 0..k_lim {
                                let byte = wb[boff + 4 + (j >> 1)];
                                let nib = if (j & 1) == 0 { byte & 0x0F } else { byte >> 4 };
                                let q = ((nib as i8) << 4) >> 4;
                                dxr[k_base + j] += scale * (q as f32);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Dequantize any `LinearView` (`F32 / F16 / Q8_0 / Q4_0`) into packed `u32` pairs of IEEE-754 `f16`
    /// (`out_features * ((orig_cols + 1) / 2)` words) for compact `wgpu` training shaders (`unpack2x16float`).
    pub fn dequantize_linear_f16_u32(&self, prefix: &str) -> Result<(usize, usize, Vec<u32>, Option<Vec<f32>>), String> {
        let lv = self.linear_view(prefix)?;
        let in_c = lv.orig_cols;
        let out_c = lv.out_features;
        let pairs_per_row = (in_c + 1) / 2;
        let mut packed = vec![0u32; out_c * pairs_per_row];
        let mut row_f32 = vec![0.0f32; in_c];

        for c in 0..out_c {
            row_f32.fill(0.0);
            match lv.mode {
                QuantMode::F32 => {
                    let w: &[f32] = bytemuck::cast_slice(lv.weight_bytes);
                    row_f32.copy_from_slice(&w[c * lv.pad_cols..c * lv.pad_cols + in_c]);
                }
                QuantMode::F16 => {
                    let w: &[u16] = bytemuck::cast_slice(lv.weight_bytes);
                    let wr = &w[c * lv.pad_cols..c * lv.pad_cols + in_c];
                    for k in 0..in_c {
                        row_f32[k] = f16::from_bits(wr[k]).to_f32();
                    }
                }
                QuantMode::Q8_0 => {
                    let bpr = lv.blocks_per_row;
                    let wb = lv.weight_bytes;
                    for bi in 0..bpr {
                        let boff = (c * bpr + bi) * 36;
                        let scale = f32::from_le_bytes(wb[boff..boff + 4].try_into().unwrap());
                        let k_base = bi * 32;
                        let k_lim = (in_c - k_base).min(32);
                        for j in 0..k_lim {
                            row_f32[k_base + j] = (wb[boff + 4 + j] as i8 as f32) * scale;
                        }
                    }
                }
                QuantMode::Q4_0 => {
                    let bpr = lv.blocks_per_row;
                    let wb = lv.weight_bytes;
                    for bi in 0..bpr {
                        let boff = (c * bpr + bi) * 20;
                        let scale = f32::from_le_bytes(wb[boff..boff + 4].try_into().unwrap());
                        let k_base = bi * 32;
                        let k_lim = (in_c - k_base).min(32);
                        for j in 0..k_lim {
                            let byte = wb[boff + 4 + (j >> 1)];
                            let nib = if (j & 1) == 0 { byte & 0x0F } else { byte >> 4 };
                            let q = ((nib as i8) << 4) >> 4;
                            row_f32[k_base + j] = (q as f32) * scale;
                        }
                    }
                }
            }
            let out_row = &mut packed[c * pairs_per_row..(c + 1) * pairs_per_row];
            for p in 0..pairs_per_row {
                let lo = f16::from_f32(row_f32[2 * p]).to_bits() as u32;
                let hi = if 2 * p + 1 < in_c {
                    f16::from_f32(row_f32[2 * p + 1]).to_bits() as u32
                } else {
                    0u32
                };
                out_row[p] = lo | (hi << 16);
            }
        }
        Ok((out_c, in_c, packed, lv.bias.map(|b| b.to_vec())))
    }
}
