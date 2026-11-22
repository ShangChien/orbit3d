//! Browser WebAssembly (`wasm-bindgen`) bindings for DeePMD `DPA-4` (`SeZM`: `Nano`, `Mini`, `Neo`, `Air`, `Plus`).

#[cfg(target_arch = "wasm32")]
use js_sys::Function;
#[cfg(target_arch = "wasm32")]
use mlip_core::{
    relax_async, run_md_async, AsyncPotentialBackend, MdOptions, MolecularSystem,
    PropertyPrediction, RelaxOptions,
};
#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::gpu::{Dpa4Engine, Dpa4ExecMode};

#[cfg(target_arch = "wasm32")]
#[derive(Serialize)]
struct SingleEvalPayload {
    energy_ev: f64,
    forces_ev_a: Vec<[f64; 3]>,
    descriptor: Option<Vec<f32>>,
    desc_dim: usize,
    oracle_prediction: Option<PropertyPrediction>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmDpa4Engine {
    inner: Dpa4Engine,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmDpa4Engine {
    /// Load a `.dpa4` bundle (`Nano`, `Mini`, `Neo`, `Air`, or `Plus`) in a Browser Worker.
    pub async fn create(bundle_bytes: &[u8], prefer_gpu: bool) -> Result<WasmDpa4Engine, JsValue> {
        let mode = if prefer_gpu {
            Dpa4ExecMode::Auto
        } else {
            Dpa4ExecMode::Cpu
        };
        let inner = Dpa4Engine::create(bundle_bytes, mode)
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = backendInfo)]
    pub fn backend_info(&self) -> String {
        self.inner.backend_info().to_string()
    }

    #[wasm_bindgen(js_name = hasPropertyOracle)]
    pub fn has_property_oracle(&self) -> bool {
        self.inner.weights.oracle.is_some()
    }

    /// Single-point energy (`eV`), conservative forces (`eV/Å`), optional `(N, C)` descriptor,
    /// and automatic shared-backbone property oracle prediction when bundled!
    #[wasm_bindgen(js_name = evalSingle)]
    pub async fn eval_single(
        &mut self,
        atomic_numbers: &[u8],
        coords_flat: &[f64],
        charge: f32,
        multiplicity: u32,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<JsValue, JsValue> {
        let sys = parse_system(atomic_numbers, coords_flat, charge, multiplicity)?;
        let want_desc = need_descriptor || self.inner.weights.oracle.is_some();
        let ev = self
            .inner
            .eval_async(&sys, need_forces, want_desc)
            .await
            .map_err(|e| JsValue::from_str(&e))?;

        let oracle_prediction = match (&ev.descriptor, &self.inner.weights.oracle) {
            (Some(desc), Some(oracle)) => Some(
                oracle
                    .predict(desc, &sys.coords_a)
                    .map_err(|e| JsValue::from_str(&e))?,
            ),
            _ => None,
        };

        let payload = SingleEvalPayload {
            energy_ev: ev.energy_ev,
            forces_ev_a: ev.forces_ev_a,
            descriptor: if need_descriptor { ev.descriptor } else { None },
            desc_dim: ev.desc_dim,
            oracle_prediction,
        };
        let json =
            serde_json::to_string(&payload).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(JsValue::from_str(&json))
    }

    /// 3D Geometry Relaxation (`L-BFGS` or `FIRE`, full or H-only `freeze_heavy=true`) with live frame callbacks.
    #[wasm_bindgen(js_name = relaxGeometry)]
    pub async fn relax_geometry(
        &mut self,
        atomic_numbers: &[u8],
        coords_flat: &[f64],
        charge: f32,
        multiplicity: u32,
        options_json: &str,
        on_frame: Option<Function>,
    ) -> Result<JsValue, JsValue> {
        let sys = parse_system(atomic_numbers, coords_flat, charge, multiplicity)?;
        let opts: RelaxOptions = if options_json.trim().is_empty() {
            RelaxOptions::default()
        } else {
            serde_json::from_str(options_json).map_err(|e| JsValue::from_str(&e.to_string()))?
        };

        let res = relax_async(&mut self.inner, &sys, &opts, false, |frame| {
            if let Some(ref cb) = on_frame {
                if let Ok(s) = serde_json::to_string(frame) {
                    let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&s));
                }
            }
        })
        .await
        .map_err(|e| JsValue::from_str(&e))?;

        let json = serde_json::to_string(&res).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(JsValue::from_str(&json))
    }

    /// Live 3D Molecular Dynamics (`NVT BAOAB Langevin` or `NVE Velocity Verlet`) with live frame callbacks.
    #[wasm_bindgen(js_name = runMd)]
    pub async fn run_md(
        &mut self,
        atomic_numbers: &[u8],
        coords_flat: &[f64],
        charge: f32,
        multiplicity: u32,
        options_json: &str,
        on_frame: Option<Function>,
    ) -> Result<JsValue, JsValue> {
        let sys = parse_system(atomic_numbers, coords_flat, charge, multiplicity)?;
        let opts: MdOptions = if options_json.trim().is_empty() {
            MdOptions::default()
        } else {
            serde_json::from_str(options_json).map_err(|e| JsValue::from_str(&e.to_string()))?
        };

        let res = run_md_async(&mut self.inner, &sys, &opts, |frame| {
            if let Some(ref cb) = on_frame {
                if let Ok(s) = serde_json::to_string(frame) {
                    let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&s));
                }
            }
        })
        .await
        .map_err(|e| JsValue::from_str(&e))?;

        let json = serde_json::to_string(&res).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(JsValue::from_str(&json))
    }
}

#[cfg(target_arch = "wasm32")]
fn parse_system(
    atomic_numbers: &[u8],
    coords_flat: &[f64],
    charge: f32,
    multiplicity: u32,
) -> Result<MolecularSystem, JsValue> {
    if coords_flat.len() != atomic_numbers.len() * 3 {
        return Err(JsValue::from_str(&format!(
            "coords_flat length ({}) != 3 * atomic_numbers length ({})",
            coords_flat.len(),
            atomic_numbers.len() * 3
        )));
    }
    let coords_a: Vec<[f64; 3]> = coords_flat
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    MolecularSystem::new(atomic_numbers.to_vec(), coords_a, charge, multiplicity)
        .map_err(|e| JsValue::from_str(&e))
}
