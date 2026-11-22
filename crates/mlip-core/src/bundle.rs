//! Shared zero-copy binary weight bundle reader (`BundleArchive`) supporting `F32`, `F16`, `F64`,
//! and `I32` tensors with 64-byte alignment for both `.dpa4` and `.hyff` model packages.
//!
//! Layout:
//! - `[0..8]`: 8-byte ASCII magic (e.g. `b"DPA4RS01"` or `b"HYFFRS01"`)
//! - `[8..12]`: `u32` little-endian format version (`1`)
//! - `[12..16]`: `u32` little-endian JSON header byte length `H`
//! - `[16..16+H]`: UTF-8 JSON header containing `"config"`, `"tensors"`, and optional `"oracle"`
//! - `[aligned_64(16+H)..]`: raw little-endian tensor payloads at 64-byte-aligned `offset`s

use std::collections::HashMap;

use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BundleDtype {
    F32,
    F16,
    F64,
    I32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: BundleDtype,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
}

impl TensorMeta {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Clone, Debug, Deserialize)]
struct RawHeader {
    pub config: Value,
    pub tensors: Vec<TensorMeta>,
    #[serde(default)]
    pub extra: Value,
}

pub struct BundleArchive<'a> {
    pub magic: [u8; 8],
    pub version: u32,
    pub config: Value,
    pub extra: Value,
    pub data_start: usize,
    raw: &'a [u8],
    index: HashMap<String, TensorMeta>,
}

impl<'a> BundleArchive<'a> {
    pub fn parse(raw: &'a [u8], expected_magic: &[u8; 8]) -> Result<Self, String> {
        if raw.len() < 16 {
            return Err(format!("Bundle too small ({} bytes)", raw.len()));
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&raw[0..8]);
        if &magic != expected_magic {
            return Err(format!(
                "Invalid bundle magic {:?}, expected {:?}",
                String::from_utf8_lossy(&magic),
                String::from_utf8_lossy(expected_magic)
            ));
        }
        let version = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        if version != 1 {
            return Err(format!("Unsupported bundle version {version} (expected 1)"));
        }
        let header_len = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
        let header_end = 16usize
            .checked_add(header_len)
            .ok_or_else(|| "Header length overflow".to_string())?;
        if header_end > raw.len() {
            return Err(format!(
                "Header extends past EOF ({header_end} > {})",
                raw.len()
            ));
        }

        let parsed: RawHeader = serde_json::from_slice(&raw[16..header_end])
            .map_err(|e| format!("Invalid bundle JSON header: {e}"))?;
        let data_start = (header_end + 63) & !63;

        let mut index = HashMap::with_capacity(parsed.tensors.len());
        for meta in parsed.tensors {
            let end = data_start
                .checked_add(meta.offset)
                .and_then(|s| s.checked_add(meta.nbytes))
                .ok_or_else(|| format!("Tensor '{}' byte range overflow", meta.name))?;
            if end > raw.len() {
                return Err(format!(
                    "Tensor '{}' out of bounds ({end} > {})",
                    meta.name,
                    raw.len()
                ));
            }
            index.insert(meta.name.clone(), meta);
        }

        Ok(Self {
            magic,
            version,
            config: parsed.config,
            extra: parsed.extra,
            data_start,
            raw,
            index,
        })
    }

    #[inline]
    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn meta(&self, name: &str) -> Result<&TensorMeta, String> {
        self.index
            .get(name)
            .ok_or_else(|| format!("Missing required tensor '{name}' in bundle"))
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.index.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    pub fn raw_bytes(&self, name: &str) -> Result<&'a [u8], String> {
        let m = self.meta(name)?;
        let start = self.data_start + m.offset;
        Ok(&self.raw[start..start + m.nbytes])
    }

    /// Read any `F32` / `F16` / `F64` tensor into a contiguous `Vec<f32>`.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let m = self.meta(name)?;
        let bytes = self.raw_bytes(name)?;
        let n = m.numel();
        match m.dtype {
            BundleDtype::F32 => {
                if bytes.len() != n * 4 {
                    return Err(format!("Tensor '{name}' F32 byte size mismatch"));
                }
                if (bytes.as_ptr() as usize) % 4 == 0 {
                    Ok(bytemuck::cast_slice::<u8, f32>(bytes).to_vec())
                } else {
                    Ok(bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect())
                }
            }
            BundleDtype::F16 => {
                if bytes.len() != n * 2 {
                    return Err(format!("Tensor '{name}' F16 byte size mismatch"));
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect())
            }
            BundleDtype::F64 => {
                if bytes.len() != n * 8 {
                    return Err(format!("Tensor '{name}' F64 byte size mismatch"));
                }
                Ok(bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect())
            }
            BundleDtype::I32 => Err(format!("Tensor '{name}' has dtype I32, expected float")),
        }
    }

    /// Read any `F64` / `F32` / `F16` tensor into a contiguous `Vec<f64>` (lossless for atomic energy offsets).
    pub fn read_f64(&self, name: &str) -> Result<Vec<f64>, String> {
        let m = self.meta(name)?;
        let bytes = self.raw_bytes(name)?;
        let n = m.numel();
        match m.dtype {
            BundleDtype::F64 => {
                if bytes.len() != n * 8 {
                    return Err(format!("Tensor '{name}' F64 byte size mismatch"));
                }
                if (bytes.as_ptr() as usize) % 8 == 0 {
                    Ok(bytemuck::cast_slice::<u8, f64>(bytes).to_vec())
                } else {
                    Ok(bytes
                        .chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                        .collect())
                }
            }
            BundleDtype::F32 => Ok(self
                .read_f32(name)?
                .into_iter()
                .map(|v| v as f64)
                .collect()),
            BundleDtype::F16 => Ok(self
                .read_f32(name)?
                .into_iter()
                .map(|v| v as f64)
                .collect()),
            BundleDtype::I32 => Err(format!("Tensor '{name}' has dtype I32, expected float")),
        }
    }

    /// Read an `I32` tensor into `Vec<i32>`.
    pub fn read_i32(&self, name: &str) -> Result<Vec<i32>, String> {
        let m = self.meta(name)?;
        if m.dtype != BundleDtype::I32 {
            return Err(format!("Tensor '{name}' is {:?}, expected I32", m.dtype));
        }
        let bytes = self.raw_bytes(name)?;
        if bytes.len() != m.numel() * 4 {
            return Err(format!("Tensor '{name}' I32 byte size mismatch"));
        }
        if (bytes.as_ptr() as usize) % 4 == 0 {
            Ok(bytemuck::cast_slice::<u8, i32>(bytes).to_vec())
        } else {
            Ok(bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
    }
}
