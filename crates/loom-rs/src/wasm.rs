//! Browser WebAssembly + WebGPU Bindings (`WasmLoomEngine`) for `loom-rs`.

#[cfg(target_arch = "wasm32")]
mod wasm_impl {
    use crate::chem_rules::{HeavyMolGraph, parse_sdf_records};
    use crate::grpo::TopoGrpoOptions;
    use crate::lora_model::{LoomLoraAdapter, LoomLoraConfig};
    use crate::oracle_train::OracleTrainOptions;
    use crate::rsi::run_incremental_calibration;
    use crate::sft::{SftOptions, XyzDistillOptions};
    use crate::spec::TaskWeavingSpec;
    use crate::{build_builtin_orbital_demo_sdf, LoomEngine, WeaveSampleOptions};
    use mlip_core::MolecularSystem;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmLoomEngine {
        inner: LoomEngine,
    }

    #[wasm_bindgen]
    impl WasmLoomEngine {
        #[wasm_bindgen(constructor)]
        pub fn new() -> WasmLoomEngine {
            WasmLoomEngine {
                inner: LoomEngine::new(),
            }
        }

        #[wasm_bindgen(js_name = builtinOrbitalDemoSdf)]
        pub fn builtin_orbital_demo_sdf() -> String {
            build_builtin_orbital_demo_sdf()
        }

        #[wasm_bindgen(js_name = loadUni3darBundle)]
        pub async fn load_uni3dar_bundle(
            &mut self,
            u3m_bytes: Vec<u8>,
            rank: usize,
            prefer_gpu: bool,
            custom_targets_json: Option<String>,
        ) -> Result<String, JsValue> {
            let r = rank.max(2);
            let cfg = LoomLoraConfig {
                rank: r,
                alpha: (2 * r) as f32,
                include_ffn: true,
                include_xyz_heads: false,
                train_conditioner: true,
                use_valence_head: true,
            };
            let custom = if let Some(js) = custom_targets_json.filter(|s| !s.trim().is_empty()) {
                let parsed: (Vec<String>, Vec<String>, Vec<f32>, Vec<f32>) =
                    serde_json::from_str(&js).map_err(|e| JsValue::from_str(&format!("Invalid custom_targets JSON: {e}")))?;
                Some(parsed)
            } else {
                None
            };
            self.inner
                .load_uni3dar_bundle_async(u3m_bytes, cfg, custom, prefer_gpu, 42)
                .await
                .map_err(|e| JsValue::from_str(&e))
        }

        #[wasm_bindgen(js_name = loadMlipBundle)]
        pub async fn load_mlip_bundle(
            &mut self,
            mlip_bytes: &[u8],
            prefer_gpu: bool,
        ) -> Result<String, JsValue> {
            self.inner
                .load_mlip_bundle_async(mlip_bytes, prefer_gpu)
                .await
                .map_err(|e| JsValue::from_str(&e))
        }

        #[wasm_bindgen(js_name = setTaskSpecJson)]
        pub fn set_task_spec_json(&mut self, spec_json: &str) -> Result<(), JsValue> {
            let spec: TaskWeavingSpec = serde_json::from_str(spec_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid TaskWeavingSpec JSON: {e}")))?;
            self.inner.spec = spec;
            Ok(())
        }

        #[wasm_bindgen(js_name = trainOracleFromSdf)]
        pub async fn train_oracle_from_sdf(
            &mut self,
            asset_id: &str,
            sdf_text: &str,
            prop_names_json: &str,
            prop_units_json: &str,
            is_audit_only: bool,
            opts_json: &str,
            on_progress: Option<js_sys::Function>,
        ) -> Result<String, JsValue> {
            let pnames: Vec<String> = serde_json::from_str(prop_names_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid prop_names: {e}")))?;
            let punits: Vec<String> = serde_json::from_str(prop_units_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid prop_units: {e}")))?;
            let opts: OracleTrainOptions = if opts_json.trim().is_empty() {
                OracleTrainOptions::default()
            } else {
                serde_json::from_str(opts_json)
                    .map_err(|e| JsValue::from_str(&format!("Invalid OracleTrainOptions: {e}")))?
            };

            let t0 = js_sys::Date::now();
            let mut rep = self
                .inner
                .train_oracle_from_sdf_async(
                    asset_id,
                    sdf_text,
                    pnames,
                    punits,
                    is_audit_only,
                    &opts,
                    |m, m_tot, ep, val_mae| {
                        if let Some(ref cb) = on_progress {
                            let info = serde_json::json!({
                                "member": m,
                                "n_members": m_tot,
                                "epoch": ep,
                                "total_epochs": opts.epochs,
                                "val_norm_mae": val_mae,
                            });
                            let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&info.to_string()));
                        }
                    },
                )
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            rep.elapsed_ms = js_sys::Date::now() - t0;
            serde_json::to_string(&rep).map_err(|e| JsValue::from_str(&e.to_string()))
        }

        #[wasm_bindgen(js_name = calibrateOracleFeedbackSdf)]
        pub async fn calibrate_oracle_feedback_sdf(
            &mut self,
            asset_id: &str,
            feedback_sdf: &str,
        ) -> Result<String, JsValue> {
            let recs = parse_sdf_records(feedback_sdf).map_err(|e| JsValue::from_str(&e))?;
            let mlip = self
                .inner
                .mlip
                .as_mut()
                .ok_or_else(|| JsValue::from_str("MLIP not loaded"))?;
            let mo = self
                .inner
                .oracles
                .get_mut(asset_id)
                .ok_or_else(|| JsValue::from_str(&format!("Oracle '{asset_id}' not found")))?;
            let pnames = mo.oracle.inner.target_names.clone();

            let mut graphs = Vec::new();
            let mut yb_rows = Vec::new();
            let mut yt_rows = Vec::new();
            for r in &recs {
                if !r.valence_valid {
                    continue;
                }
                let Ok(sys) = MolecularSystem::new(r.atomic_numbers.clone(), r.positions.clone(), 0.0, 1) else {
                    continue;
                };
                let Ok(ev) = mlip.eval_async(&sys, false, true).await else {
                    continue;
                };
                let Some(desc) = ev.descriptor else {
                    continue;
                };
                let Ok(preds) = mo.oracle.predict(&desc, &r.positions) else {
                    continue;
                };
                let yb: Vec<f64> = preds.mean.iter().map(|&v| v as f64).collect();
                let yt: Vec<f64> = pnames
                    .iter()
                    .map(|n| r.properties.get(n).copied().unwrap_or(f64::NAN))
                    .collect();
                if yt.iter().any(|v| v.is_finite()) {
                    graphs.push(HeavyMolGraph::from_record(r));
                    yb_rows.push(yb);
                    yt_rows.push(yt);
                }
            }

            let cal = mo
                .calibrator
                .get_or_insert_with(|| crate::gp::ResidualCalibratedAsset::new(pnames, 0.38));
            let rep = run_incremental_calibration(cal, &graphs, &yb_rows, &yt_rows, &[], &[], &[], &[]);
            serde_json::to_string(&rep).map_err(|e| JsValue::from_str(&e.to_string()))
        }

        #[wasm_bindgen(js_name = trainLoraSft)]
        pub async fn train_lora_sft(
            &mut self,
            sdf_text: Option<String>,
            opts_json: &str,
            on_step: Option<js_sys::Function>,
        ) -> Result<String, JsValue> {
            let opts: SftOptions = if opts_json.trim().is_empty() {
                SftOptions::default()
            } else {
                serde_json::from_str(opts_json)
                    .map_err(|e| JsValue::from_str(&format!("Invalid SftOptions: {e}")))?
            };
            let t0 = js_sys::Date::now();
            let mut rep = self
                .inner
                .train_lora_sft_async(sdf_text.as_deref(), &opts, |m| {
                    if let Some(ref cb) = on_step {
                        if let Ok(s) = serde_json::to_string(m) {
                            let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&s));
                        }
                    }
                })
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            rep.elapsed_ms = js_sys::Date::now() - t0;
            serde_json::to_string(&rep).map_err(|e| JsValue::from_str(&e.to_string()))
        }

        #[wasm_bindgen(js_name = distillXyzHeads)]
        pub async fn distill_xyz_heads(
            &mut self,
            sdf_text: Option<String>,
            opts_json: &str,
            on_step: Option<js_sys::Function>,
        ) -> Result<String, JsValue> {
            let opts: XyzDistillOptions = if opts_json.trim().is_empty() {
                XyzDistillOptions::default()
            } else {
                serde_json::from_str(opts_json)
                    .map_err(|e| JsValue::from_str(&format!("Invalid XyzDistillOptions: {e}")))?
            };
            let t0 = js_sys::Date::now();
            let mut rep = self
                .inner
                .distill_xyz_heads_async(sdf_text.as_deref(), &opts, |step, tot, l_xyz| {
                    if let Some(ref cb) = on_step {
                        let js = serde_json::json!({ "step": step, "total_steps": tot, "loss_xyz": l_xyz });
                        let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&js.to_string()));
                    }
                })
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            rep.elapsed_ms = js_sys::Date::now() - t0;
            serde_json::to_string(&rep).map_err(|e| JsValue::from_str(&e.to_string()))
        }

        #[wasm_bindgen(js_name = trainOnlineTopoGrpo)]
        pub async fn train_online_topo_grpo(
            &mut self,
            opts_json: &str,
            on_step: Option<js_sys::Function>,
        ) -> Result<String, JsValue> {
            let opts: TopoGrpoOptions = if opts_json.trim().is_empty() {
                TopoGrpoOptions::default()
            } else {
                serde_json::from_str(opts_json)
                    .map_err(|e| JsValue::from_str(&format!("Invalid TopoGrpoOptions: {e}")))?
            };
            let t0 = js_sys::Date::now();
            let mut rep = self
                .inner
                .train_online_topo_grpo_async(&opts, |m| {
                    if let Some(ref cb) = on_step {
                        if let Ok(s) = serde_json::to_string(m) {
                            let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&s));
                        }
                    }
                })
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            rep.elapsed_ms = js_sys::Date::now() - t0;
            serde_json::to_string(&rep).map_err(|e| JsValue::from_str(&e.to_string()))
        }

        #[wasm_bindgen(js_name = buildAtlasAndDiagnoseRsi)]
        pub async fn build_atlas_and_diagnose_rsi(
            &mut self,
            grid_json: &str,
            samples_per_cell: usize,
            cfg_scale: f32,
            seed: u64,
        ) -> Result<String, JsValue> {
            let grid: Vec<f64> = serde_json::from_str(grid_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid grid JSON: {e}")))?;
            let (curve, rsi, acq) = self
                .inner
                .build_atlas_and_diagnose_rsi_async(&grid, samples_per_cell, cfg_scale, seed)
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            let out = serde_json::json!({
                "atlas_curve": curve,
                "rsi_report": rsi,
                "acquisition_manifest": acq,
            });
            Ok(out.to_string())
        }

        #[wasm_bindgen(js_name = weaveSampleBestOfM)]
        pub async fn weave_sample_best_of_m(&mut self, opts_json: &str) -> Result<String, JsValue> {
            let opts: WeaveSampleOptions = if opts_json.trim().is_empty() {
                WeaveSampleOptions::default()
            } else {
                serde_json::from_str(opts_json)
                    .map_err(|e| JsValue::from_str(&format!("Invalid WeaveSampleOptions: {e}")))?
            };
            let res = self
                .inner
                .weave_sample_best_of_m_async(&opts)
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            serde_json::to_string(&res).map_err(|e| JsValue::from_str(&e.to_string()))
        }

        #[wasm_bindgen(js_name = exportLoraBytes)]
        pub fn export_lora_bytes(&self) -> Result<Vec<u8>, JsValue> {
            let ad = self
                .inner
                .adapter
                .as_ref()
                .ok_or_else(|| JsValue::from_str("No LoRA adapter loaded"))?;
            ad.to_bytes().map_err(|e| JsValue::from_str(&e))
        }

        #[wasm_bindgen(js_name = importLoraBytes)]
        pub fn import_lora_bytes(&mut self, bytes: &[u8]) -> Result<String, JsValue> {
            let ad = LoomLoraAdapter::from_bytes(bytes).map_err(|e| JsValue::from_str(&e))?;
            let n = ad.num_trainable_parameters();
            self.inner.adapter = Some(ad);
            Ok(format!("Loaded .loom_lora adapter ({:.2}M params)", (n as f64) / 1e6))
        }

        #[wasm_bindgen(js_name = exportOracleJson)]
        pub fn export_oracle_json(&self, asset_id: &str) -> Result<String, JsValue> {
            self.inner
                .export_oracle_json(asset_id)
                .map_err(|e| JsValue::from_str(&e))
        }

        #[wasm_bindgen(js_name = importOracleJson)]
        pub fn import_oracle_json(
            &mut self,
            asset_id: &str,
            json_str: &str,
            is_audit_only: bool,
        ) -> Result<String, JsValue> {
            self.inner
                .import_oracle_json(asset_id, json_str, is_audit_only)
                .map_err(|e| JsValue::from_str(&e))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_impl::WasmLoomEngine;
