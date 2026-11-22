pub mod config;
pub mod cpu_backend;
pub mod format;
pub mod gpu_backend;
pub mod refiner_critic;
pub mod sampler;
pub mod valence;

use crate::format::ModelBundle;
use crate::gpu_backend::WgpuDecoderBackend;
use crate::sampler::{sample_molecule_async, Uni3darSampleOptions};
use crate::valence::SampledUni3darMolecule;

pub struct Uni3darEngine {
    pub bundle: ModelBundle,
    pub gpu: Option<WgpuDecoderBackend>,
}

impl Uni3darEngine {
    pub async fn from_bytes_async(bytes: Vec<u8>, use_gpu: bool) -> Result<Self, String> {
        let bundle = ModelBundle::from_bytes(bytes)?;
        let gpu = if use_gpu {
            Some(WgpuDecoderBackend::new_async(&bundle).await?)
        } else {
            None
        };
        Ok(Self { bundle, gpu })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_bytes_blocking(bytes: Vec<u8>, use_gpu: bool) -> Result<Self, String> {
        pollster::block_on(Self::from_bytes_async(bytes, use_gpu))
    }

    pub fn backend_description(&self) -> String {
        if let Some(ref g) = self.gpu {
            format!("wgpu/{} ({})", g.backend_api, g.adapter_name)
        } else {
            "cpu-fp32".to_string()
        }
    }

    pub async fn sample_async<F>(&mut self, opts: &Uni3darSampleOptions, on_step: F) -> Result<SampledUni3darMolecule, String>
    where
        F: FnMut(usize, usize, &[[f32; 3]], &[usize]),
    {
        let gpu_ref = if opts.force_cpu { None } else { self.gpu.as_mut() };
        sample_molecule_async(&self.bundle, gpu_ref, opts, on_step).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn sample_blocking<F>(&mut self, opts: &Uni3darSampleOptions, on_step: F) -> Result<SampledUni3darMolecule, String>
    where
        F: FnMut(usize, usize, &[[f32; 3]], &[usize]),
    {
        pollster::block_on(self.sample_async(opts, on_step))
    }
}

// ============================================================================
// Browser WebAssembly + WebGPU API (`wasm-pack build --target web --release`)
// ============================================================================
#[cfg(target_arch = "wasm32")]
pub mod wasm_api {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmUni3darEngine {
        inner: Uni3darEngine,
        gpu_error: Option<String>,
    }

    #[wasm_bindgen]
    impl WasmUni3darEngine {
        #[wasm_bindgen(js_name = create)]
        pub async fn create(model_bytes: Vec<u8>, use_gpu: bool) -> Result<WasmUni3darEngine, JsValue> {
            let bundle = ModelBundle::from_bytes(model_bytes).map_err(|e| JsValue::from_str(&e))?;
            let mut gpu_error = None;
            let gpu = if use_gpu {
                match WgpuDecoderBackend::new_async(&bundle).await {
                    Ok(g) => Some(g),
                    Err(e) => {
                        gpu_error = Some(e);
                        None
                    }
                }
            } else {
                None
            };
            Ok(WasmUni3darEngine {
                inner: Uni3darEngine { bundle, gpu },
                gpu_error,
            })
        }

        #[wasm_bindgen(js_name = backendInfo)]
        pub fn backend_info(&self) -> String {
            let h = &self.inner.bundle.header;
            serde_json::json!({
                "engine_family": "uni3dar",
                "backend": self.inner.backend_description(),
                "is_gpu": self.inner.gpu.is_some(),
                "gpu_error": self.gpu_error,
                "quant_dtype": h.quant_dtype,
                "lora_merged_count": h.lora_merged_count,
                "merge_level": h.backbone.merge_level,
                "num_layers": h.backbone.num_layers,
                "emb_dim": h.backbone.emb_dim,
                "allowed_elements": h.backbone.allowed_elements,
                "has_conditioner": h.conditioner.is_some(),
                "has_refiner": h.refiner.is_some(),
                "has_critics": h.stage1_critic.is_some() && h.stage2_critic.is_some(),
                "target_cols": h.target_cols,
                "target_units": h.target_units,
                "target_mean": h.target_mean,
                "target_scale": h.target_scale,
            })
            .to_string()
        }

        #[wasm_bindgen(js_name = sampleMolecule)]
        pub async fn sample_molecule(
            &mut self,
            opts_json: &str,
            on_progress: Option<js_sys::Function>,
        ) -> Result<String, JsValue> {
            let opts: Uni3darSampleOptions = serde_json::from_str(opts_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid Uni3darSampleOptions JSON: {e}")))?;

            let mol = self
                .inner
                .sample_async(&opts, |step, total, x, a| {
                    if let Some(ref cb) = on_progress {
                        let flat_x: &[f32] = bytemuck::cast_slice(x);
                        let x_js = js_sys::Float32Array::from(flat_x);
                        let a_u32: Vec<u32> = a.iter().map(|&v| v as u32).collect();
                        let a_js = js_sys::Uint32Array::from(a_u32.as_slice());
                        let _ = cb.call4(
                            &JsValue::NULL,
                            &JsValue::from_f64(step as f64),
                            &JsValue::from_f64(total as f64),
                            &x_js,
                            &a_js,
                        );
                    }
                })
                .await
                .map_err(|e| JsValue::from_str(&e))?;

            let sdf = mol.to_sdf("uni3dar-rs-webgpu");
            let out = serde_json::json!({
                "molecule": mol,
                "sdf": sdf,
            });
            Ok(out.to_string())
        }
    }
}
