pub mod chem;
pub mod config;
pub mod cpu_backend;
pub mod format;
pub mod gpu_backend;
pub mod sampler;

use crate::chem::SampledMolecule;
use crate::config::{ForwardOutput, PrecomputedModulations, SelfCondState};
use crate::cpu_backend::{forward_cpu, GraphTopology};
use crate::format::ModelBundle;
use crate::gpu_backend::WgpuBackend;
use crate::sampler::{sample_molecule_async, SampleOptions};

pub struct FlowMolEngine {
    pub bundle: ModelBundle,
    pub gpu: Option<WgpuBackend>,
}

impl FlowMolEngine {
    pub async fn from_bytes_async(bytes: Vec<u8>, use_gpu: bool, initial_n_cap: usize) -> Result<Self, String> {
        let bundle = ModelBundle::from_bytes(bytes)?;
        let gpu = if use_gpu {
            Some(WgpuBackend::new_async(&bundle, initial_n_cap).await?)
        } else {
            None
        };
        Ok(Self { bundle, gpu })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_bytes_blocking(bytes: Vec<u8>, use_gpu: bool, initial_n_cap: usize) -> Result<Self, String> {
        pollster::block_on(Self::from_bytes_async(bytes, use_gpu, initial_n_cap))
    }

    pub fn backend_description(&self) -> String {
        if let Some(ref g) = self.gpu {
            format!("wgpu/{} ({})", g.backend_api, g.adapter_name)
        } else {
            "cpu-fp32".to_string()
        }
    }

    pub async fn forward_single_async(
        &mut self,
        x_t: &[[f32; 3]],
        a_t: &[usize],
        c_t: &[usize],
        e_t_upper: &[usize],
        t_val: f32,
        sc: Option<&SelfCondState>,
        mods: Option<&PrecomputedModulations>,
        node_cond_flags: Option<&[[f32; 2]]>,
        force_cpu: bool,
    ) -> Result<ForwardOutput, String> {
        let topo = GraphTopology::complete(x_t.len());
        if !force_cpu {
            if let Some(ref mut g) = self.gpu {
                return g
                    .forward_async(&self.bundle, &topo, x_t, a_t, c_t, e_t_upper, t_val, sc, mods, node_cond_flags)
                    .await;
            }
        }
        forward_cpu(&self.bundle, &topo, x_t, a_t, c_t, e_t_upper, t_val, sc, mods, node_cond_flags)
    }

    pub async fn sample_async<F>(&mut self, opts: &SampleOptions, on_step: F) -> Result<SampledMolecule, String>
    where
        F: FnMut(usize, usize, &[[f32; 3]], &[usize]),
    {
        let gpu_ref = if opts.force_cpu { None } else { self.gpu.as_mut() };
        sample_molecule_async(&self.bundle, gpu_ref, opts, on_step).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn sample_blocking<F>(&mut self, opts: &SampleOptions, on_step: F) -> Result<SampledMolecule, String>
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
    pub struct WasmFlowMolEngine {
        inner: FlowMolEngine,
        gpu_error: Option<String>,
    }

    #[wasm_bindgen]
    impl WasmFlowMolEngine {
        /// Create a WebGPU-accelerated engine without cloning `model_bytes` (falls back to WASM CPU and records `gpu_error` if WebGPU fails).
        #[wasm_bindgen(js_name = create)]
        pub async fn create(model_bytes: Vec<u8>, use_gpu: bool) -> Result<WasmFlowMolEngine, JsValue> {
            let bundle = ModelBundle::from_bytes(model_bytes).map_err(|e| JsValue::from_str(&e))?;
            let mut gpu_error = None;
            let gpu = if use_gpu {
                match WgpuBackend::new_async(&bundle, 64).await {
                    Ok(g) => Some(g),
                    Err(e) => {
                        gpu_error = Some(e);
                        None
                    }
                }
            } else {
                None
            };
            Ok(WasmFlowMolEngine {
                inner: FlowMolEngine { bundle, gpu },
                gpu_error,
            })
        }

        #[wasm_bindgen(js_name = backendInfo)]
        pub fn backend_info(&self) -> String {
            let h = &self.inner.bundle.header;
            serde_json::json!({
                "backend": self.inner.backend_description(),
                "is_gpu": self.inner.gpu.is_some(),
                "gpu_error": self.gpu_error,
                "quant_dtype": h.quant_dtype,
                "step": h.step,
                "n_molecule_updates": h.config.n_molecule_updates,
                "n_hidden_scalars": h.config.n_hidden_scalars,
                "n_hidden_edge_feats": h.config.n_hidden_edge_feats,
                "has_conditioner": h.conditioner.is_some(),
                "domains": h.per_source_n_atoms_hist.keys().collect::<Vec<_>>(),
            })
            .to_string()
        }

        /// Sample a 3D molecule in the browser; `on_progress` (optional JS function `(step, totalSteps, xFlatFloat32, atomIdxU32)`)
        /// streams live 3D denoising frames directly to WebGL/Canvas!
        #[wasm_bindgen(js_name = sampleMolecule)]
        pub async fn sample_molecule(
            &mut self,
            opts_json: &str,
            on_progress: Option<js_sys::Function>,
        ) -> Result<String, JsValue> {
            let opts: SampleOptions =
                serde_json::from_str(opts_json).map_err(|e| JsValue::from_str(&format!("Invalid SampleOptions JSON: {e}")))?;

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

            let sdf = mol.to_sdf("flowmol-rs-webgpu");
            let out = serde_json::json!({
                "molecule": mol,
                "sdf": sdf,
            });
            Ok(out.to_string())
        }
    }
}
