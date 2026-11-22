//! `loom-rs`: Unified Client-Side Rust + `wgpu` (`Desktop Metal / Vulkan / DX12` + `Browser WASM / WebGPU`)
//! Weaving, Fine-Tuning (`Oracle`, `LoRA SFT`, `XYZ Distillation`, `Online TopoGRPO`), & `Grounded RSI` Engine.

pub mod chem_rules;
pub mod critic_bus;
pub mod gp;
pub mod grpo;
pub mod lora_model;
pub mod octree_encode;
pub mod oracle_train;
pub mod rsi;
pub mod sft;
pub mod spec;
pub mod wasm;

use crate::chem_rules::{
    formal_charge_signature, parse_sdf_records, HeavyMolGraph, MolRecord,
};
use crate::critic_bus::{
    evaluate_molecules_on_bus, refine_coords_physics_constrained, CoordRefinerOptions, MlipSlot,
    MoleculeCriticEvaluation, MorganDomain, MountedOracle,
};
use crate::gp::{HeteroscedasticGPAtlas1D, ProposalAtlasSample, ResidualCalibratedAsset};
use crate::grpo::{run_online_topo_grpo, TopoGrpoOptions, TopoGrpoReport, TopoGrpoStepMetrics};
use crate::lora_model::{
    sample_rollout_trajectory_kv, LoomGpuTrainContext, LoomLoraAdapter, LoomLoraConfig,
    MergedRolloutWeights, RolloutOptions,
};
use crate::oracle_train::{
    scaffold_key_for_graph, CachedDescriptorSample, OracleTrainOptions, OracleTrainReport,
    SerializedOracleBundle, TrainableBootstrapOracle,
};
use crate::rsi::{
    select_outer_rsi_acquisition_batch, AcquisitionBatchManifest, RSIDiagnoser,
    RSIDiagnosticReport1D,
};
use crate::sft::{
    run_lora_sft, run_xyz_head_distillation, SftOptions, SftStepMetrics, SftTrainReport,
    XyzDistillOptions,
};
use crate::spec::TaskWeavingSpec;
use dpa4_rs::{Dpa4Engine, Dpa4ExecMode, DPA4_MAGIC};
use hyff_rs::{HyffEngine, HyffExecMode, HYFF_MAGIC};
use mlip_core::MolecularSystem;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uni3dar_rs::format::ModelBundle;
use uni3dar_rs::valence::SampledUni3darMolecule;

/// Top-level stateful `Uni-Loom v2.0` Runtime & Fine-Tuning Engine.
pub struct LoomEngine {
    pub generator_bundle: Option<ModelBundle>,
    pub gpu_train: Option<LoomGpuTrainContext>,
    pub adapter: Option<LoomLoraAdapter>,
    pub mlip: Option<MlipSlot>,
    pub oracles: HashMap<String, MountedOracle>,
    pub replay_records: Vec<MolRecord>,
    pub spec: TaskWeavingSpec,
    pub atlas_1d: Option<HeteroscedasticGPAtlas1D>,
    pub last_rsi_report: Option<RSIDiagnosticReport1D>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaveSampleOptions {
    #[serde(default = "default_goals")]
    pub goals: Vec<f64>,
    #[serde(default = "default_m")]
    pub n_candidates: usize,
    #[serde(default = "default_cfg")]
    pub cfg_scale: f32,
    #[serde(default = "default_true")]
    pub geometric_guard: bool,
    #[serde(default = "default_true")]
    pub reconcile_valence: bool,
    #[serde(default = "default_true")]
    pub coord_refine: bool,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_goals() -> Vec<f64> {
    vec![-6.0]
}
fn default_m() -> usize {
    4
}
fn default_cfg() -> f32 {
    2.0
}
fn default_true() -> bool {
    true
}
fn default_seed() -> u64 {
    42
}

impl Default for WeaveSampleOptions {
    fn default() -> Self {
        Self {
            goals: vec![-6.0],
            n_candidates: 4,
            cfg_scale: 2.0,
            geometric_guard: true,
            reconcile_valence: true,
            coord_refine: true,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaveSampleResult {
    pub best_molecule: SampledUni3darMolecule,
    pub best_sdf: String,
    pub best_eval: MoleculeCriticEvaluation,
    pub candidates_evaluated: usize,
    pub valid_count: usize,
    pub clean_hit_count: usize,
    pub coord_refined: bool,
    pub gatekeeper: Option<crate::gp::Gatekeeper1DRecommendation>,
}

impl Default for LoomEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LoomEngine {
    pub fn new() -> Self {
        Self {
            generator_bundle: None,
            gpu_train: None,
            adapter: None,
            mlip: None,
            oracles: HashMap::new(),
            replay_records: Vec::new(),
            spec: TaskWeavingSpec::default_synthesizable_orbital("homo", "reward_oracle"),
            atlas_1d: None,
            last_rsi_report: None,
        }
    }

    /// Attach a `.u3m` `Uni-3DAR-Mol` backbone bundle and initialize zero-perturbation `LoRA` + `LoomConditioner`.
    pub async fn load_uni3dar_bundle_async(
        &mut self,
        u3m_bytes: Vec<u8>,
        lora_cfg: LoomLoraConfig,
        custom_targets: Option<(Vec<String>, Vec<String>, Vec<f32>, Vec<f32>)>,
        prefer_gpu: bool,
        seed: u64,
    ) -> Result<String, String> {
        let bundle = ModelBundle::from_bytes(u3m_bytes)?;
        let gpu_train = if prefer_gpu {
            LoomGpuTrainContext::create(&bundle).await.ok()
        } else {
            None
        };
        let adapter = LoomLoraAdapter::new_for_bundle(&bundle, lora_cfg, custom_targets, seed)?;
        let n_trainable = adapter.num_trainable_parameters();
        let backend = gpu_train
            .as_ref()
            .map(|g| format!("wgpu ({})", g.adapter_info))
            .unwrap_or_else(|| "SIMD CPU".to_string());

        let t_cols = adapter.conditioner.target_cols.clone();
        if let Some(first_col) = t_cols.first() {
            self.spec = TaskWeavingSpec::default_synthesizable_orbital(first_col, "reward_oracle");
        }

        self.generator_bundle = Some(bundle);
        self.gpu_train = gpu_train;
        self.adapter = Some(adapter);

        Ok(format!(
            "Uni-3DAR LoRA ready [{backend}, trainable={:.2}M params, targets={:?}]",
            (n_trainable as f64) / 1e6,
            t_cols
        ))
    }

    /// Attach a `.hyff` or `.dpa4` Quantum MLIP bundle as the active `PotentialSlot` & descriptor provider.
    pub async fn load_mlip_bundle_async(
        &mut self,
        mlip_bytes: &[u8],
        prefer_gpu: bool,
    ) -> Result<String, String> {
        if mlip_bytes.len() < 8 {
            return Err("MLIP bundle too small".into());
        }
        if &mlip_bytes[0..8] == HYFF_MAGIC {
            let mode = if prefer_gpu { HyffExecMode::Auto } else { HyffExecMode::Cpu };
            let eng = HyffEngine::create(mlip_bytes, mode).await?;
            let desc = format!("{} (desc_dim={})", eng.backend_info(), eng.weights.cfg.hidden);
            self.mlip = Some(MlipSlot::Hyff(eng));
            Ok(desc)
        } else if &mlip_bytes[0..8] == DPA4_MAGIC {
            let mode = if prefer_gpu { Dpa4ExecMode::Auto } else { Dpa4ExecMode::Cpu };
            let eng = Dpa4Engine::create(mlip_bytes, mode).await?;
            let desc = format!("{} (desc_dim={})", eng.backend_info(), eng.weights.cfg.channels);
            if let Some(ref bundled_ora) = eng.weights.oracle {
                if let Ok(tb) = TrainableBootstrapOracle::from_mlip_oracle("dpa4", bundled_ora.clone()) {
                    self.oracles.insert(
                        "reward_oracle".to_string(),
                        MountedOracle {
                            oracle: tb,
                            calibrator: None,
                            domain: None,
                            is_audit_only: false,
                        },
                    );
                }
            }
            self.mlip = Some(MlipSlot::Dpa4(eng));
            Ok(desc)
        } else {
            Err("Unrecognized MLIP magic (expected .hyff or .dpa4)".into())
        }
    }

    /// Train a shared-backbone 5-fold Bootstrap 3D Property Oracle + `MorganDomain` + `ResidualCalibratedAsset`
    /// directly from `.sdf` text!
    pub async fn train_oracle_from_sdf_async<F>(
        &mut self,
        asset_id: &str,
        sdf_text: &str,
        property_names: Vec<String>,
        property_units: Vec<String>,
        is_audit_only: bool,
        opts: &OracleTrainOptions,
        on_progress: F,
    ) -> Result<OracleTrainReport, String>
    where
        F: FnMut(usize, usize, usize, f32),
    {
        let mlip = self
            .mlip
            .as_mut()
            .ok_or("Load a HyFF-L2 or DPA-4 MLIP bundle before training a 3D Oracle")?;
        let records = parse_sdf_records(sdf_text)?;
        if records.len() < 4 {
            return Err(format!("Parsed only {} molecules from SDF (need >= 4)", records.len()));
        }

        let desc_dim = mlip.desc_dim();
        let fam = mlip.family().to_string();
        let mut cached = Vec::new();
        let mut kept_recs = Vec::new();
        let mut kept_graphs = Vec::new();

        for rec in &records {
            if !rec.valence_valid || formal_charge_signature(rec) != "neutral" {
                continue;
            }
            let mut t_vec = Vec::with_capacity(property_names.len());
            let mut all_present = true;
            for p in &property_names {
                if let Some(&v) = rec.properties.get(p) {
                    if v.is_finite() {
                        t_vec.push(v as f32);
                    } else {
                        all_present = false;
                        break;
                    }
                } else {
                    all_present = false;
                    break;
                }
            }
            if !all_present {
                continue;
            }
            let Ok(sys) = MolecularSystem::new(rec.atomic_numbers.clone(), rec.positions.clone(), 0.0, 1) else {
                continue;
            };
            let Ok(ev) = mlip.eval_async(&sys, false, true).await else {
                continue;
            };
            let Some(atom_repr) = ev.descriptor else {
                continue;
            };
            let g = HeavyMolGraph::from_record(rec);
            let scaf = scaffold_key_for_graph(&g);
            cached.push(CachedDescriptorSample {
                atom_repr,
                coords_a: rec.positions.clone(),
                targets: t_vec,
                scaffold_sig: scaf,
            });
            kept_recs.push(rec.clone());
            kept_graphs.push(g);
        }

        let (oracle, report) = TrainableBootstrapOracle::fit_from_cached_descriptors(
            &fam,
            property_names.clone(),
            property_units,
            desc_dim,
            &cached,
            opts,
            on_progress,
        )?;

        // Seed `MorganDomain` & `ResidualCalibratedAsset` with training anchors!
        let domain = MorganDomain::from_records(&kept_recs, 0.25, &property_names);
        let mut calibrator = ResidualCalibratedAsset::new(property_names.clone(), 0.38);
        let mut y_base_rows = Vec::with_capacity(cached.len());
        let mut y_true_rows = Vec::with_capacity(cached.len());
        for s in &cached {
            let preds = oracle.predict(&s.atom_repr, &s.coords_a)?;
            y_base_rows.push(preds.mean.iter().map(|&v| v as f64).collect());
            y_true_rows.push(s.targets.iter().map(|&v| v as f64).collect());
        }
        calibrator.ingest_samples(&kept_graphs, &y_base_rows, &y_true_rows, false);

        if !is_audit_only {
            self.replay_records = kept_recs;
            if let Some(first_p) = property_names.first() {
                self.spec = TaskWeavingSpec::default_synthesizable_orbital(first_p, asset_id);
            }
        }

        self.oracles.insert(
            asset_id.to_string(),
            MountedOracle {
                oracle,
                calibrator: Some(calibrator),
                domain: Some(domain),
                is_audit_only,
            },
        );

        Ok(report)
    }

    pub fn export_oracle_json(&self, asset_id: &str) -> Result<String, String> {
        let mo = self
            .oracles
            .get(asset_id)
            .ok_or_else(|| format!("Oracle '{asset_id}' not found"))?;
        let b: SerializedOracleBundle = mo.oracle.to_serialized_bundle();
        serde_json::to_string(&b).map_err(|e| format!("Oracle serialize error: {e}"))
    }

    pub fn import_oracle_json(&mut self, asset_id: &str, json_str: &str, is_audit_only: bool) -> Result<String, String> {
        let b: SerializedOracleBundle =
            serde_json::from_str(json_str).map_err(|e| format!("Invalid oracle JSON: {e}"))?;
        let pnames = b.property_names.clone();
        let oracle = TrainableBootstrapOracle::from_serialized_bundle(b)?;
        self.oracles.insert(
            asset_id.to_string(),
            MountedOracle {
                oracle,
                calibrator: None,
                domain: None,
                is_audit_only,
            },
        );
        Ok(format!("Mounted oracle '{asset_id}' ({:?})", pnames))
    }

    /// Run Stage 0/1 `LoRA SFT` on `.sdf` records (or currently cached `replay_records` if `sdf_text` is empty)!
    pub async fn train_lora_sft_async<F>(
        &mut self,
        sdf_text: Option<&str>,
        opts: &SftOptions,
        on_step: F,
    ) -> Result<SftTrainReport, String>
    where
        F: FnMut(&SftStepMetrics),
    {
        let bundle = self
            .generator_bundle
            .as_ref()
            .ok_or("Load a .u3m Uni-3DAR bundle first")?;
        let adapter = self
            .adapter
            .as_mut()
            .ok_or("LoRA adapter not initialized")?;

        let parsed_owned;
        let recs: &[MolRecord] = if let Some(txt) = sdf_text.filter(|t| !t.trim().is_empty()) {
            parsed_owned = parse_sdf_records(txt)?;
            self.replay_records = parsed_owned.clone();
            &parsed_owned
        } else {
            &self.replay_records
        };

        run_lora_sft(bundle, self.gpu_train.as_ref(), adapter, recs, opts, on_step).await
    }

    /// Run Stage 2.5B Detached Sub-Voxel `0.01 Å XYZ` Head Equilibrium Distillation!
    pub async fn distill_xyz_heads_async<F>(
        &mut self,
        sdf_text: Option<&str>,
        opts: &XyzDistillOptions,
        on_step: F,
    ) -> Result<SftTrainReport, String>
    where
        F: FnMut(usize, usize, f32),
    {
        let bundle = self
            .generator_bundle
            .as_ref()
            .ok_or("Load a .u3m Uni-3DAR bundle first")?;
        let adapter = self
            .adapter
            .as_mut()
            .ok_or("LoRA adapter not initialized")?;

        let parsed_owned;
        let recs: &[MolRecord] = if let Some(txt) = sdf_text.filter(|t| !t.trim().is_empty()) {
            parsed_owned = parse_sdf_records(txt)?;
            &parsed_owned
        } else {
            &self.replay_records
        };

        run_xyz_head_distillation(bundle, self.gpu_train.as_ref(), adapter, recs, opts, on_step).await
    }

    /// Run Stage 2 / 2.5A Online `TopoGRPO` closed-loop training!
    pub async fn train_online_topo_grpo_async<F>(
        &mut self,
        opts: &TopoGrpoOptions,
        on_step: F,
    ) -> Result<TopoGrpoReport, String>
    where
        F: FnMut(&TopoGrpoStepMetrics),
    {
        let bundle = self
            .generator_bundle
            .as_ref()
            .ok_or("Load a .u3m Uni-3DAR bundle first")?;
        let adapter = self
            .adapter
            .as_mut()
            .ok_or("LoRA adapter not initialized")?;

        run_online_topo_grpo(
            bundle,
            self.gpu_train.as_ref(),
            adapter,
            &self.spec,
            self.mlip.as_mut(),
            &self.oracles,
            &self.replay_records,
            opts,
            on_step,
        )
        .await
    }

    /// Build a 1D `HeteroscedasticGPAtlas1D`, diagnose 4-quadrant `Grounded RSI` states,
    /// and export an active-learning `OuterRSIAcquisition` `.sdf` batch!
    pub async fn build_atlas_and_diagnose_rsi_async(
        &mut self,
        grid_targets: &[f64],
        samples_per_cell: usize,
        cfg_scale: f32,
        seed: u64,
    ) -> Result<(Vec<crate::gp::Atlas1DPoint>, RSIDiagnosticReport1D, AcquisitionBatchManifest), String> {
        let bundle = self
            .generator_bundle
            .as_ref()
            .ok_or("Load a .u3m Uni-3DAR bundle first")?;
        let adapter = self
            .adapter
            .as_ref()
            .ok_or("LoRA adapter not initialized")?;
        if grid_targets.len() < 2 {
            return Err("Provide >= 2 target grid points for GP Atlas".into());
        }

        let p_cnt = adapter.conditioner.cfg.num_props;
        let m_cell = samples_per_cell.max(2);
        let merged_rollout = MergedRolloutWeights::build(bundle, adapter)?;
        let mut all_recs = Vec::with_capacity(grid_targets.len() * m_cell);
        let mut req_goals = Vec::with_capacity(grid_targets.len() * m_cell);
        let mut req_scalar = Vec::with_capacity(grid_targets.len() * m_cell);
        let mut act_masks = Vec::with_capacity(grid_targets.len() * m_cell);

        for (gi, &t) in grid_targets.iter().enumerate() {
            let mut g_f32 = vec![0.0f32; p_cnt];
            let mut g_f64 = vec![0.0f64; p_cnt];
            let mut am = vec![false; p_cnt];
            g_f32[0] = t as f32;
            g_f64[0] = t;
            am[0] = true;

            for k in 0..m_cell {
                let r_opts = RolloutOptions {
                    cfg_scale,
                    seed: seed.wrapping_add(((gi * 1024) + k) as u64),
                    ..RolloutOptions::default()
                };
                let (_, rec, _) = sample_rollout_trajectory_kv(bundle, adapter, &merged_rollout, &g_f32, &am, &r_opts)?;
                all_recs.push(rec);
                req_goals.push(g_f64.clone());
                req_scalar.push(t);
                act_masks.push(am.clone());
            }
        }

        let evals = evaluate_molecules_on_bus(
            &self.spec,
            &all_recs,
            &req_goals,
            &act_masks,
            self.mlip.as_mut(),
            &self.oracles,
            None,
            false,
            false,
        )
        .await?;

        let atlas_samples: Vec<ProposalAtlasSample> = (0..evals.len())
            .map(|i| ProposalAtlasSample {
                requested_targets: req_goals[i].clone(),
                evaluated: evals[i].evaluated,
                reward_pred: evals[i].objective_preds.clone(),
                audit_pred: evals[i].audit_preds.clone(),
                sigma_cal_est: Some(evals[i].sigma_cal_est.clone()),
            })
            .collect();

        let axis_name = self
            .spec
            .objectives
            .first()
            .map(|o| o.name.as_str())
            .unwrap_or("homo");
        let tol = self.spec.objectives.first().map_or(0.25, |o| o.tolerance);

        let atlas = HeteroscedasticGPAtlas1D::fit(axis_name, tol, &atlas_samples, 0)?;
        let curve = atlas.query_curve(81);

        let src_vals: Vec<f64> = self
            .replay_records
            .iter()
            .filter_map(|r| r.properties.get(axis_name).copied())
            .collect();
        let diagnoser = RSIDiagnoser::default();
        let rsi_rep = diagnoser.diagnose_1d(
            &atlas,
            grid_targets,
            if src_vals.len() >= 2 { Some(&src_vals) } else { None },
        );

        let acq = select_outer_rsi_acquisition_batch(
            &all_recs,
            &evals,
            &req_scalar,
            &rsi_rep.blindspot_targets,
            16,
            3.5,
            0.22,
        );

        self.atlas_1d = Some(atlas);
        self.last_rsi_report = Some(rsi_rep.clone());
        Ok((curve, rsi_rep, acq))
    }

    /// Calibrated `Best-of-M` Weaving Inference + `CompositeCriticBus` + `PhysicsConstrainedCoordRefiner`!
    pub async fn weave_sample_best_of_m_async(
        &mut self,
        opts: &WeaveSampleOptions,
    ) -> Result<WeaveSampleResult, String> {
        let bundle = self
            .generator_bundle
            .as_ref()
            .ok_or("Load a .u3m Uni-3DAR bundle first")?;
        let adapter = self
            .adapter
            .as_ref()
            .ok_or("LoRA adapter not initialized")?;

        let gatekeeper = if let (Some(ref atlas), Some(&g0)) = (&self.atlas_1d, opts.goals.first()) {
            Some(atlas.recommend_target(g0, 0.45))
        } else {
            None
        };

        let p_cnt = adapter.conditioner.cfg.num_props;
        let mut g_f32 = vec![0.0f32; p_cnt];
        let mut g_f64 = vec![0.0f64; p_cnt];
        let mut am = vec![false; p_cnt];
        for k in 0..p_cnt.min(opts.goals.len()) {
            g_f32[k] = opts.goals[k] as f32;
            g_f64[k] = opts.goals[k];
            am[k] = true;
        }

        let m_cand = opts.n_candidates.max(1);
        let merged_rollout = MergedRolloutWeights::build(bundle, adapter)?;
        let mut recs = Vec::with_capacity(m_cand);
        let mut mols = Vec::with_capacity(m_cand);
        let mut goals_rep = Vec::with_capacity(m_cand);
        let mut am_rep = Vec::with_capacity(m_cand);

        for k in 0..m_cand {
            let r_opts = RolloutOptions {
                cfg_scale: opts.cfg_scale,
                geometric_guard: opts.geometric_guard,
                reconcile_valence: opts.reconcile_valence,
                seed: opts.seed.wrapping_add(k as u64),
                ..RolloutOptions::default()
            };
            let (_, rec, mol) = sample_rollout_trajectory_kv(bundle, adapter, &merged_rollout, &g_f32, &am, &r_opts)?;
            recs.push(rec);
            mols.push(mol);
            goals_rep.push(g_f64.clone());
            am_rep.push(am.clone());
        }

        let evals = evaluate_molecules_on_bus(
            &self.spec,
            &recs,
            &goals_rep,
            &am_rep,
            self.mlip.as_mut(),
            &self.oracles,
            None,
            false,
            false,
        )
        .await?;

        let valid_count = evals.iter().filter(|e| e.legal).count();
        let clean_hit_count = evals.iter().filter(|e| e.is_clean_hit).count();

        let best_i = (0..m_cand)
            .max_by(|&a, &b| {
                let rank_a = (evals[a].is_clean_hit as i32 as f64) * 100.0
                    + (evals[a].legal as i32 as f64) * 10.0
                    + evals[a].prop_score
                    + 0.25 * evals[a].phys_score;
                let rank_b = (evals[b].is_clean_hit as i32 as f64) * 100.0
                    + (evals[b].legal as i32 as f64) * 10.0
                    + evals[b].prop_score
                    + 0.25 * evals[b].phys_score;
                rank_a.partial_cmp(&rank_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        let mut best_mol = mols[best_i].clone();
        let mut best_rec = recs[best_i].clone();
        let mut best_eval = evals[best_i].clone();
        let mut coord_refined = false;

        // Optional `PhysicsConstrainedCoordRefiner` on the winning candidate!
        if opts.coord_refine && best_eval.legal {
            if let (Some(ospec), Some(&goal0)) = (self.spec.objectives.first(), opts.goals.first()) {
                if let (Some(ref mut mlip), Some(mo)) = (&mut self.mlip, self.oracles.get(&ospec.asset_id)) {
                    if mo.oracle.backbone_family == mlip.family() {
                        if let Some(c_idx) = mo.channel_index(ospec.resolved_channel()) {
                            let ref_opts = CoordRefinerOptions {
                                tolerance: ospec.tolerance,
                                ..CoordRefinerOptions::default()
                            };
                            if let Ok(r_res) = refine_coords_physics_constrained(
                                &best_rec,
                                mlip,
                                &mo.oracle,
                                c_idx,
                                goal0,
                                &ref_opts,
                            )
                            .await
                            {
                                if r_res.steps_taken > 0 {
                                    coord_refined = true;
                                    best_rec.positions = r_res.positions.clone();
                                    best_mol.positions = r_res
                                        .positions
                                        .iter()
                                        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                                        .collect();
                                    best_eval.f_max_ev_a = r_res.final_fmax_ev_a;
                                    if !best_eval.objective_preds.is_empty() && r_res.final_pred.is_finite() {
                                        best_eval.objective_preds[0] = r_res.final_pred;
                                        best_eval.is_hit = ospec.is_hit(r_res.final_pred, goal0);
                                        best_eval.is_clean_hit =
                                            best_eval.is_hit && best_eval.clean_charge && best_eval.constraints_met;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (k, ospec) in self.spec.objectives.iter().enumerate() {
            if let Some(&yp) = best_eval.objective_preds.get(k) {
                if yp.is_finite() {
                    best_mol.predicted_props.insert(ospec.name.clone(), yp as f32);
                }
            }
        }
        let best_sdf = best_mol.to_sdf("loom-rs-weaved");

        Ok(WeaveSampleResult {
            best_molecule: best_mol,
            best_sdf,
            best_eval,
            candidates_evaluated: m_cand,
            valid_count,
            clean_hit_count,
            coord_refined,
            gatekeeper,
        })
    }
}

/// Self-contained 3D conjugated organic library (`16` diverse molecules across `HOMO = -7.65 .. -5.05 eV`)
/// so users in Browser WebGPU or CLI can test end-to-end `Oracle + LoRA SFT + XYZ Distill + TopoGRPO`
/// with 1 click even without uploading a local `.sdf` file!
pub fn build_builtin_orbital_demo_sdf() -> String {
    let ring6 = |r: f64| -> [[f64; 3]; 6] {
        let mut out = [[0.0; 3]; 6];
        for k in 0..6 {
            let a = (k as f64) * std::f64::consts::PI / 3.0;
            out[k] = [r * a.cos(), r * a.sin(), 0.0];
        }
        out
    };
    let c6 = ring6(1.395);
    let h6 = ring6(2.480);

    let mut entries: Vec<(String, f64, f64, Vec<(&str, [f64; 3])>)> = Vec::new();

    // 1. Benzene (-6.65, -0.45)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
            v.push(("H", h6[k]));
        }
        entries.push(("Benzene".into(), -6.65, -0.45, v));
    }
    // 2. Fluorobenzene (-6.78, -0.62)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("F", [2.75, 0.0, 0.0]));
        for k in 1..6 {
            v.push(("H", h6[k]));
        }
        entries.push(("Fluorobenzene".into(), -6.78, -0.62, v));
    }
    // 3. 1,4-Difluorobenzene (-6.92, -0.85)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("F", [2.75, 0.0, 0.0]));
        v.push(("F", [-2.75, 0.0, 0.0]));
        for k in [1, 2, 4, 5] {
            v.push(("H", h6[k]));
        }
        entries.push(("1,4-Difluorobenzene".into(), -6.92, -0.85, v));
    }
    // 4. Chlorobenzene (-6.72, -0.68)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("Cl", [3.12, 0.0, 0.0]));
        for k in 1..6 {
            v.push(("H", h6[k]));
        }
        entries.push(("Chlorobenzene".into(), -6.72, -0.68, v));
    }
    // 5. Phenol (-5.95, -0.35)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("O", [2.76, 0.0, 0.0]));
        v.push(("H", [3.32, 0.78, 0.0]));
        for k in 1..6 {
            v.push(("H", h6[k]));
        }
        entries.push(("Phenol".into(), -5.95, -0.35, v));
    }
    // 6. Aniline (-5.38, -0.22)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("N", [2.78, 0.0, 0.0]));
        v.push(("H", [3.29, 0.86, 0.12]));
        v.push(("H", [3.29, -0.86, -0.12]));
        for k in 1..6 {
            v.push(("H", h6[k]));
        }
        entries.push(("Aniline".into(), -5.38, -0.22, v));
    }
    // 7. 1,4-Phenylenediamine (-5.05, -0.10)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("N", [2.78, 0.0, 0.0]));
        v.push(("H", [3.29, 0.86, 0.10]));
        v.push(("H", [3.29, -0.86, -0.10]));
        v.push(("N", [-2.78, 0.0, 0.0]));
        v.push(("H", [-3.29, 0.86, -0.10]));
        v.push(("H", [-3.29, -0.86, 0.10]));
        for k in [1, 2, 4, 5] {
            v.push(("H", h6[k]));
        }
        entries.push(("p-Phenylenediamine".into(), -5.05, -0.10, v));
    }
    // 8. Hydroquinone (-5.58, -0.28)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("O", [2.76, 0.0, 0.0]));
        v.push(("H", [3.32, 0.78, 0.0]));
        v.push(("O", [-2.76, 0.0, 0.0]));
        v.push(("H", [-3.32, -0.78, 0.0]));
        for k in [1, 2, 4, 5] {
            v.push(("H", h6[k]));
        }
        entries.push(("Hydroquinone".into(), -5.58, -0.28, v));
    }
    // 9. Pyridine (-6.88, -0.95)
    {
        let mut v = Vec::new();
        v.push(("N", c6[0]));
        for k in 1..6 {
            v.push(("C", c6[k]));
            v.push(("H", h6[k]));
        }
        entries.push(("Pyridine".into(), -6.88, -0.95, v));
    }
    // 10. Pyrimidine (-7.18, -1.28)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            if k == 0 || k == 2 {
                v.push(("N", c6[k]));
            } else {
                v.push(("C", c6[k]));
                v.push(("H", h6[k]));
            }
        }
        entries.push(("Pyrimidine".into(), -7.18, -1.28, v));
    }
    // 11. Pyrazine (-7.12, -1.35)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            if k == 0 || k == 3 {
                v.push(("N", c6[k]));
            } else {
                v.push(("C", c6[k]));
                v.push(("H", h6[k]));
            }
        }
        entries.push(("Pyrazine".into(), -7.12, -1.35, v));
    }
    // 12. Benzonitrile (-7.24, -1.42)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("C", [2.83, 0.0, 0.0]));
        v.push(("N", [3.99, 0.0, 0.0]));
        for k in 1..6 {
            v.push(("H", h6[k]));
        }
        entries.push(("Benzonitrile".into(), -7.24, -1.42, v));
    }
    // 13. Terephthalonitrile (-7.62, -2.15)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("C", [2.83, 0.0, 0.0]));
        v.push(("N", [3.99, 0.0, 0.0]));
        v.push(("C", [-2.83, 0.0, 0.0]));
        v.push(("N", [-3.99, 0.0, 0.0]));
        for k in [1, 2, 4, 5] {
            v.push(("H", h6[k]));
        }
        entries.push(("Terephthalonitrile".into(), -7.62, -2.15, v));
    }
    // 14. Benzaldehyde (-6.98, -1.65)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("C", [2.86, 0.0, 0.0]));
        v.push(("O", [3.48, 1.06, 0.0]));
        v.push(("H", [3.42, -0.96, 0.0]));
        for k in 1..6 {
            v.push(("H", h6[k]));
        }
        entries.push(("Benzaldehyde".into(), -6.98, -1.65, v));
    }
    // 15. Thiophenol (-5.82, -0.52)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("S", [3.16, 0.0, 0.0]));
        v.push(("H", [3.52, 1.28, 0.0]));
        for k in 1..6 {
            v.push(("H", h6[k]));
        }
        entries.push(("Thiophenol".into(), -5.82, -0.52, v));
    }
    // 16. p-Aminobenzonitrile (-5.72, -1.08)
    {
        let mut v = Vec::new();
        for k in 0..6 {
            v.push(("C", c6[k]));
        }
        v.push(("C", [2.83, 0.0, 0.0]));
        v.push(("N", [3.99, 0.0, 0.0]));
        v.push(("N", [-2.78, 0.0, 0.0]));
        v.push(("H", [-3.29, 0.86, 0.10]));
        v.push(("H", [-3.29, -0.86, -0.10]));
        for k in [1, 2, 4, 5] {
            v.push(("H", h6[k]));
        }
        entries.push(("p-Aminobenzonitrile".into(), -5.72, -1.08, v));
    }

    let mut sdf = String::new();
    for (name, homo, lumo, atoms) in entries {
        let mut zs = Vec::with_capacity(atoms.len());
        let mut pos = Vec::with_capacity(atoms.len());
        for (sym, p) in atoms {
            let eid = uni3dar_rs::valence::symbol_to_elem_id(sym).unwrap_or(5);
            zs.push((eid + 1) as u8);
            pos.push(p);
        }
        let mut rec = MolRecord::from_point_cloud(&name, &zs, &pos, false);
        rec.properties.insert("homo".into(), homo);
        rec.properties.insert("lumo".into(), lumo);
        rec.properties.insert("gap".into(), lumo - homo);
        sdf.push_str(&rec.to_sdf());
    }
    sdf
}
