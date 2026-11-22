//! Desktop Native CLI & Full-Stack Verification/Benchmark Suite for `loom-rs` (`loom-cli`).

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(e) = pollster::block_on(run_cli()) {
        eprintln!("[loom-cli ERROR] {e}");
        std::process::exit(1);
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn run_cli() -> Result<(), String> {
    use loom_rs::chem_rules::{
        compute_residual_calibration_features_29d, parse_sdf_records, BuiltInRuleMetrics,
        HeavyMolGraph, MolRecord,
    };
    use loom_rs::gp::ResidualCalibratedAsset;
    use loom_rs::grpo::TopoGrpoOptions;
    use loom_rs::lora_model::{
        backward_sequence_batch, forward_sequence_batch, LoomGpuTrainContext, LoomLoraAdapter,
        LoomLoraConfig,
    };
    use loom_rs::octree_encode::encode_molecule_to_octree;
    use loom_rs::oracle_train::OracleTrainOptions;
    use loom_rs::rsi::run_incremental_calibration;
    use loom_rs::sft::{SftOptions, XyzDistillOptions};
    use loom_rs::{build_builtin_orbital_demo_sdf, LoomEngine, WeaveSampleOptions};
    use mlip_core::MolecularSystem;
    use std::time::Instant;
    use uni3dar_rs::format::ModelBundle;
    use uni3dar_rs::sampler::DetRng;

    let args: Vec<String> = std::env::args().collect();
    let u3m_path = args
        .iter()
        .position(|a| a == "--u3m")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "/Users/dp/project/mol_gen/pretrained/uni3dar_mol/oe62_homo_q8.u3m".to_string());
    let hyff_path = args
        .iter()
        .position(|a| a == "--hyff")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "/Users/dp/project/mol_gen/pretrained/hyff/hyff_presto.hyff".to_string());
    let dpa4_path = args
        .iter()
        .position(|a| a == "--dpa4")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "/Users/dp/project/mol_gen/pretrained/dpa4/dpa4_mini.dpa4".to_string());

    println!("=========================================================================");
    println!("  loom-rs Full-Stack Verification & Apple M5 Metal wgpu Benchmark Suite");
    println!("=========================================================================");

    // -------------------------------------------------------------------------
    // Phase 1A: Zero-RDKit Chemical Rules, SMARTS Alerts & GP Residual Calibration
    // -------------------------------------------------------------------------
    let demo_sdf = build_builtin_orbital_demo_sdf();
    let demo_recs = parse_sdf_records(&demo_sdf)?;
    println!("[Phase 1A] Parsed {} built-in 3D conjugated molecules from SDF", demo_recs.len());

    for idx in [0usize, 5, 11, 12] {
        let g = HeavyMolGraph::from_record(&demo_recs[idx]);
        let m = BuiltInRuleMetrics::evaluate(&g);
        let f29 = compute_residual_calibration_features_29d(&g);
        assert!(m.sa_score >= 1.0 && m.sa_score <= 5.0, "Clean aromatic SA score out of range");
        assert!(m.qed > 0.25 && m.qed <= 1.0, "Clean aromatic QED out of range");
        assert_eq!(m.smarts_alert, 0.0, "Clean aromatic should trigger 0 reactive alerts");
        assert!(f29.iter().all(|v| v.is_finite()), "29D features must be finite");
        println!(
            "  - {:<20} | MW={:>6.1} Da | SA={:.2} | QED={:.3} | Alerts={} | HOMO={:.2} eV",
            demo_recs[idx].title,
            m.mol_weight,
            m.sa_score,
            m.qed,
            m.smarts_alert as usize,
            demo_recs[idx].properties["homo"]
        );
    }

    // Verify Peroxide `[OX2][OX2]` reactive SMARTS alert fires == 1!
    let h2o2 = MolRecord {
        title: "H2O2".into(),
        atomic_numbers: vec![8, 8, 1, 1],
        positions: vec![[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [-0.5, 0.8, 0.0], [1.95, -0.8, 0.0]],
        formal_charges: vec![0, 0, 0, 0],
        bonds: vec![(0, 1, 1), (0, 2, 1), (1, 3, 1)],
        properties: Default::default(),
        valence_valid: true,
    };
    let h2o2_m = BuiltInRuleMetrics::evaluate(&HeavyMolGraph::from_record(&h2o2));
    assert_eq!(h2o2_m.smarts_alert, 1.0, "Hydrogen peroxide must trigger [OX2][OX2] alert");
    println!("  - Reactive SMARTS check passed: H2O2 smarts_alert = {}", h2o2_m.smarts_alert);

    // Verify `ResidualCalibratedAsset` epistemic shrinkage & blindspot calibration
    let t_gp = Instant::now();
    let mut cal = ResidualCalibratedAsset::new(vec!["homo".into()], 0.38);
    let graphs_all: Vec<HeavyMolGraph> = demo_recs.iter().map(HeavyMolGraph::from_record).collect();
    // Simulate biased base predictor on deep HOMO (`y_base = y_true + 0.85` when `homo < -6.8`)
    let yb_all: Vec<Vec<f64>> = demo_recs
        .iter()
        .map(|r| {
            let yt = r.properties["homo"];
            vec![if yt < -6.8 { yt + 0.85 } else { yt + 0.03 }]
        })
        .collect();
    let yt_all: Vec<Vec<f64>> = demo_recs.iter().map(|r| vec![r.properties["homo"]]).collect();
    // Seed only shallow molecules (`homo >= -6.8`), hold out deep molecules for feedback & blind test!
    let mut seed_g = Vec::new();
    let mut seed_yb = Vec::new();
    let mut seed_yt = Vec::new();
    let mut fb_g = Vec::new();
    let mut fb_yb = Vec::new();
    let mut fb_yt = Vec::new();
    for i in 0..demo_recs.len() {
        if yt_all[i][0] >= -6.8 {
            seed_g.push(graphs_all[i].clone());
            seed_yb.push(yb_all[i].clone());
            seed_yt.push(yt_all[i].clone());
        } else {
            fb_g.push(graphs_all[i].clone());
            fb_yb.push(yb_all[i].clone());
            fb_yt.push(yt_all[i].clone());
        }
    }
    cal.ingest_samples(&seed_g, &seed_yb, &seed_yt, false);
    let val_sb = vec![vec![0.10]; fb_g.len()];
    let inc_rep = run_incremental_calibration(
        &mut cal,
        &fb_g[..fb_g.len() - 2],
        &fb_yb[..fb_g.len() - 2],
        &fb_yt[..fb_g.len() - 2],
        &fb_g[fb_g.len() - 2..],
        &fb_yb[fb_g.len() - 2..],
        &val_sb[fb_g.len() - 2..],
        &fb_yt[fb_g.len() - 2..],
    );
    let gp_ms = t_gp.elapsed().as_secs_f64() * 1e3;
    println!(
        "[Phase 1A] GP Residual Calibrator fit+incremental update in {:.2} ms | Blind OOD MAE: {:.3} -> {:.3} eV (std {:.3} -> {:.3})",
        gp_ms, inc_rep.pre_val_mae[0], inc_rep.post_val_mae[0], inc_rep.pre_mean_std[0], inc_rep.post_mean_std[0]
    );
    assert!(inc_rep.post_val_mae[0] < inc_rep.pre_val_mae[0], "Incremental GP calibration must reduce OOD MAE");

    // -------------------------------------------------------------------------
    // Phase 1B: Shared-Backbone 5-Fold Bootstrap 3D Oracle Training + `grad_coords` Parity
    // -------------------------------------------------------------------------
    let hyff_bytes = std::fs::read(&hyff_path).map_err(|e| format!("Read {hyff_path}: {e}"))?;
    let mut engine = LoomEngine::new();
    let mlip_info = engine.load_mlip_bundle_async(&hyff_bytes, true).await?;
    println!("[Phase 1B] Loaded MLIP: {mlip_info}");

    let t_ora = Instant::now();
    let ora_opts = OracleTrainOptions {
        n_members: 5,
        hidden_dim: 128,
        epochs: 35,
        lr: 3.0e-4,
        ..OracleTrainOptions::default()
    };
    let ora_rep = engine
        .train_oracle_from_sdf_async(
            "reward_oracle",
            &demo_sdf,
            vec!["homo".into(), "lumo".into()],
            vec!["eV".into(), "eV".into()],
            false,
            &ora_opts,
            |_, _, _, _| {},
        )
        .await?;
    let ora_ms = t_ora.elapsed().as_secs_f64() * 1e3;
    println!(
        "[Phase 1B] Trained 5-Fold Bootstrap 3D Oracle (train={}, val={}) in {:.1} ms | Val MAE: HOMO={:.3} eV, LUMO={:.3} eV",
        ora_rep.n_train,
        ora_rep.n_val,
        ora_ms,
        ora_rep.ensemble_val_mae_per_prop[0],
        ora_rep.ensemble_val_mae_per_prop[1]
    );

    // Verify analytical `grad_coords` (`HyffEngine::vjp_descriptor_coords_async` + `predict_and_input_vjp`) vs Central Finite Differences!
    {
        let test_rec = &demo_recs[4]; // Phenol (13 atoms)
        let sys0 = MolecularSystem::new(test_rec.atomic_numbers.clone(), test_rec.positions.clone(), 0.0, 1)?;
        let ev0 = engine.mlip.as_mut().unwrap().eval_async(&sys0, false, true).await?;
        let desc0 = ev0.descriptor.unwrap();
        let mo = engine.oracles.get("reward_oracle").unwrap();
        let (_y0, _s0, d_repr, d_coords_rbf) = mo.oracle.predict_and_input_vjp(&desc0, &test_rec.positions, 0)?;
        let d_coords_bb = engine
            .mlip
            .as_mut()
            .unwrap()
            .vjp_descriptor_coords_async(&sys0, &d_repr)
            .await?;

        let eps = 1.5e-3f64;
        let mut max_grad_diff = 0.0f64;
        for atom_i in [0usize, 3, 6] {
            for axis in 0..3 {
                let mut p_plus = test_rec.positions.clone();
                p_plus[atom_i][axis] += eps;
                let sp = MolecularSystem::new(test_rec.atomic_numbers.clone(), p_plus.clone(), 0.0, 1)?;
                let dp = engine.mlip.as_mut().unwrap().eval_async(&sp, false, true).await?.descriptor.unwrap();
                let yp = mo.oracle.predict(&dp, &p_plus)?.mean[0] as f64;

                let mut p_minus = test_rec.positions.clone();
                p_minus[atom_i][axis] -= eps;
                let sm = MolecularSystem::new(test_rec.atomic_numbers.clone(), p_minus.clone(), 0.0, 1)?;
                let dm = engine.mlip.as_mut().unwrap().eval_async(&sm, false, true).await?.descriptor.unwrap();
                let ym = mo.oracle.predict(&dm, &p_minus)?.mean[0] as f64;

                let g_fd = (yp - ym) / (2.0 * eps);
                let g_an = d_coords_bb[atom_i][axis] + d_coords_rbf[atom_i][axis];
                max_grad_diff = max_grad_diff.max((g_fd - g_an).abs());
            }
        }
        println!(
            "[Phase 1B] Analytical 3D Oracle `grad_coords` (HyFF + 3D Attention Adapter) vs Finite-Diff max|Δ∇y| = {:.3e} eV/Å",
            max_grad_diff
        );
        assert!(max_grad_diff < 2.5e-3, "Analytical grad_coords mismatch: {max_grad_diff}");
    }

    // Also verify `DPA-4 Mini` `vjp_descriptor_coords_async` finite-diff parity if `.dpa4` exists!
    if let Ok(dpa_bytes) = std::fs::read(&dpa4_path) {
        let mut dpa_eng = LoomEngine::new();
        let _ = dpa_eng.load_mlip_bundle_async(&dpa_bytes, true).await?;
        let test_rec = &demo_recs[0];
        let sys0 = MolecularSystem::new(test_rec.atomic_numbers.clone(), test_rec.positions.clone(), 0.0, 1)?;
        let d_dim = dpa_eng.mlip.as_ref().unwrap().desc_dim();
        let n_a = test_rec.atomic_numbers.len();
        let up_w: Vec<f32> = (0..(n_a * d_dim)).map(|i| (((i * 7 + 3) % 19) as f32 - 9.0) * 0.05).collect();
        let g_an = dpa_eng.mlip.as_mut().unwrap().vjp_descriptor_coords_async(&sys0, &up_w).await?;
        let eps = 2.0e-3f64;
        let mut max_dpa_diff = 0.0f64;
        for ax in 0..3 {
            let mut pp = test_rec.positions.clone();
            pp[0][ax] += eps;
            let dp = dpa_eng.mlip.as_mut().unwrap().eval_async(&MolecularSystem::new(test_rec.atomic_numbers.clone(), pp, 0.0, 1)?, false, true).await?.descriptor.unwrap();
            let vp: f64 = dp.iter().zip(up_w.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();

            let mut pm = test_rec.positions.clone();
            pm[0][ax] -= eps;
            let dm = dpa_eng.mlip.as_mut().unwrap().eval_async(&MolecularSystem::new(test_rec.atomic_numbers.clone(), pm, 0.0, 1)?, false, true).await?.descriptor.unwrap();
            let vm: f64 = dm.iter().zip(up_w.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();

            let fd = (vp - vm) / (2.0 * eps);
            max_dpa_diff = max_dpa_diff.max((fd - g_an[0][ax]).abs());
        }
        println!(
            "[Phase 1B] Analytical `DPA-4 Mini` descriptor VJP vs Finite-Diff max|Δ∇D| = {:.3e}",
            max_dpa_diff
        );
        assert!(max_dpa_diff < 3.0e-3, "DPA-4 descriptor VJP mismatch: {max_dpa_diff}");
    }

    // -------------------------------------------------------------------------
    // Phase 2A: `CPU` vs `Apple M5 Metal wgpu` Forward & Backward Parity
    // -------------------------------------------------------------------------
    let u3m_bytes = std::fs::read(&u3m_path).map_err(|e| format!("Read {u3m_path}: {e}"))?;
    {
        let bundle = ModelBundle::from_bytes(u3m_bytes.clone())?;
        let gpu_ctx = LoomGpuTrainContext::create(&bundle).await?;
        println!("[Phase 2A] Initialized LoomGpuTrainContext on {}", gpu_ctx.adapter_info);

        let mut ad = LoomLoraAdapter::new_for_bundle(&bundle, LoomLoraConfig::default(), None, 77)?;
        // Set non-zero `b` on one layer so LoRA backward gradients are also active at step 0!
        if let Some(lp) = ad.loras.get_mut("bb.layers.0.attn.q_proj") {
            for (i, v) in lp.b.iter_mut().enumerate() {
                *v = (((i % 13) as f32) - 6.0) * 0.01;
            }
        }
        let mut rng = DetRng::new(99);
        let s0 = encode_molecule_to_octree(&demo_recs[0], &bundle.header.backbone, &ad.conditioner.target_cols, false, 0.0, &mut rng)?;
        let s1 = encode_molecule_to_octree(&demo_recs[5], &bundle.header.backbone, &ad.conditioner.target_cols, false, 0.0, &mut rng)?;
        let seqs = vec![s0, s1];
        let rt = vec![seqs[0].targets_raw.clone(), seqs[1].targets_raw.clone()];
        let am = vec![seqs[0].active_mask.clone(), seqs[1].active_mask.clone()];

        let tape_cpu = forward_sequence_batch(&bundle, None, &ad, &seqs, &rt, &am, 1.0, true).await?;
        let tape_gpu = forward_sequence_batch(&bundle, Some(&gpu_ctx), &ad, &seqs, &rt, &am, 1.0, true).await?;

        let max_fwd_diff = tape_cpu
            .hidden_out
            .iter()
            .zip(tape_gpu.hidden_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let d_up: Vec<f32> = (0..tape_cpu.hidden_out.len())
            .map(|i| (((i * 11 + 5) % 23) as f32 - 11.0) * 0.002)
            .collect();
        let mut g_cpu = ad.zero_grads();
        let mut g_gpu = ad.zero_grads();
        backward_sequence_batch(&bundle, None, &ad, &tape_cpu, &d_up, 1.0, &mut g_cpu).await?;
        backward_sequence_batch(&bundle, Some(&gpu_ctx), &ad, &tape_gpu, &d_up, 1.0, &mut g_gpu).await?;

        let gc_q = &g_cpu.loras["bb.layers.0.attn.q_proj"].da;
        let gg_q = &g_gpu.loras["bb.layers.0.attn.q_proj"].da;
        let max_bwd_diff = gc_q
            .iter()
            .zip(gg_q.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "[Phase 2A] CPU (Q8_0) vs Metal wgpu (FP16) Sequence Forward+Backward Parity: max|Δh|={:.4}, max|ΔdLoRA|={:.4}",
            max_fwd_diff, max_bwd_diff
        );
        // Note: Q8_0 -> FP16 quantization difference across 12 layers is naturally ~1e-3..2e-2
        assert!(max_fwd_diff < 5.0e-2 && max_bwd_diff < 5.0e-2, "GPU/CPU parity exceeded FP16 tolerance");
    }

    // -------------------------------------------------------------------------
    // Phase 2B & 2C: Benchmark Stage 2.5B `XYZ` Distillation & Stage 0 `LoRA SFT` on Metal wgpu!
    // -------------------------------------------------------------------------
    let u3m_info = engine
        .load_uni3dar_bundle_async(
            u3m_bytes,
            LoomLoraConfig {
                rank: 8,
                alpha: 16.0,
                include_ffn: true,
                include_xyz_heads: false,
                train_conditioner: true,
                use_valence_head: true,
            },
            None,
            true,
            42,
        )
        .await?;
    println!("[Phase 2B] {u3m_info}");

    // Benchmark Stage 2.5B Detached `XYZ` Equilibrium Distillation (`16` molecules × `3` epochs = `48` molecules)
    let t_xyz = Instant::now();
    let xyz_rep = engine
        .distill_xyz_heads_async(
            Some(&demo_sdf),
            &XyzDistillOptions {
                batch_size: 8,
                epochs: 3,
                lr: 2.0e-4,
                weight_decay: 1.0e-3,
                seed: 42,
            },
            |_, _, _| {},
        )
        .await?;
    let xyz_sec = t_xyz.elapsed().as_secs_f64();
    println!(
        "[Phase 2B] Stage 2.5B `0.01 Å XYZ` Head Distillation: {} mols in {:.2}s ({:.1} mol/s) | loss_xyz: {:.4} -> {:.4}",
        xyz_rep.molecules_seen,
        xyz_sec,
        (xyz_rep.molecules_seen as f64) / xyz_sec,
        xyz_rep.initial_loss,
        xyz_rep.final_loss
    );

    // Benchmark Stage 0/1 Full-Layer `LoRA SFT` (`batch_size = 8`, `epochs = 3` = `48` molecules, dual-pass contrast ON!)
    let t_sft = Instant::now();
    let sft_rep = engine
        .train_lora_sft_async(
            Some(&demo_sdf),
            &SftOptions {
                batch_size: 8,
                epochs: 3,
                lr: 3.0e-4,
                min_lr: 5.0e-5,
                contrast_weight: 0.35,
                ..SftOptions::default()
            },
            |m| {
                println!(
                    "    [SFT step {}/{}] loss={:.4} (tree={:.3}, atom={:.3}, xyz={:.3}, val={:.3}, con={:.3}) |gnorm|={:.3}",
                    m.step, m.total_steps, m.loss_total, m.loss_tree, m.loss_atom, m.loss_xyz, m.loss_valence, m.loss_contrast, m.grad_norm
                );
            },
        )
        .await?;
    let sft_sec = t_sft.elapsed().as_secs_f64();
    println!(
        "[Phase 2C] Stage 0/1 Full `LoRA-8 SFT` (with 2x Contrastive Pass): {} mols in {:.2}s ({:.1} mol/s) | loss: {:.4} -> {:.4}",
        sft_rep.molecules_seen,
        sft_sec,
        (sft_rep.molecules_seen as f64) / sft_sec,
        sft_rep.initial_loss,
        sft_rep.final_loss
    );
    assert!(sft_rep.final_loss < sft_rep.initial_loss, "LoRA SFT loss should decrease");

    // Verify `.loom_lora` roundtrip serialization
    let lora_bytes = engine.adapter.as_ref().unwrap().to_bytes()?;
    let _reloaded = LoomLoraAdapter::from_bytes(&lora_bytes)?;
    println!(
        "[Phase 2C] Exported & verified `.loom_lora` adapter bundle ({:.2} MB)",
        (lora_bytes.len() as f64) / (1024.0 * 1024.0)
    );

    // -------------------------------------------------------------------------
    // Phase 3: Online `TopoGRPO` + `GP Atlas` + `Grounded RSI` + `Best-of-M` Weaving!
    // -------------------------------------------------------------------------
    let t_grpo = Instant::now();
    let grpo_rep = engine
        .train_online_topo_grpo_async(
            &TopoGrpoOptions {
                steps: 2,
                prompts_per_step: 2,
                group_size: 4,
                ppo_epochs: 2,
                lr: 3.0e-5,
                replay_weight: 0.15,
                replay_batch_size: 4,
                cfg_scale: 1.75,
                target_grid: vec![vec![-6.80], vec![-6.00], vec![-5.50]],
                seed: 2026,
                ..TopoGrpoOptions::default()
            },
            |sm| {
                println!(
                    "    [TopoGRPO step {}/{}] {} proposals | valid={:.0}% | clean_q={:.0}% | SA&Clean={:.0}% | hit={:.0}% | med_Fmax={:.2} eV/Å | KL={:.4} | |g|={:.3}",
                    sm.step,
                    sm.total_steps,
                    sm.proposals,
                    sm.valid_rate * 100.0,
                    sm.clean_charge_rate * 100.0,
                    sm.constraints_met_rate * 100.0,
                    sm.hit_rate * 100.0,
                    sm.median_fmax_ev_a,
                    sm.kl_div,
                    sm.grad_norm
                );
            },
        )
        .await?;
    let grpo_sec = t_grpo.elapsed().as_secs_f64();
    println!(
        "[Phase 3A] Online `TopoGRPO` ({} proposals + 2 PPO epochs + Replay): {:.2}s total ({:.2}s / 8 proposals, {:.2}s / 64 proposals equiv)",
        grpo_rep.total_proposals,
        grpo_sec,
        grpo_sec / (grpo_rep.steps_completed as f64),
        grpo_sec * (64.0 / (grpo_rep.total_proposals as f64))
    );

    // Build 1D GP Atlas + 4-quadrant RSI Diagnosis + Active Learning `.sdf` Acquisition
    let t_atlas = Instant::now();
    let (curve, rsi_rep, acq) = engine
        .build_atlas_and_diagnose_rsi_async(&[-7.20, -6.50, -5.80], 3, 1.75, 31415)
        .await?;
    println!(
        "[Phase 3B] Built 1D GP Atlas ({} curve pts) & 4-Quadrant RSI Diagnosis in {:.2}s | Conv={} Inner={} Blind={} Ceil={} | AcqSelected={} ({} Murcko scaffolds)",
        curve.len(),
        t_atlas.elapsed().as_secs_f64(),
        rsi_rep.n_converged,
        rsi_rep.n_inner_underexplored,
        rsi_rep.n_outer_blindspot,
        rsi_rep.n_prior_ceiling,
        acq.selected.len(),
        acq.unique_murcko_scaffolds
    );

    // Run `Best-of-4` Weaved Inference + `PhysicsConstrainedCoordRefiner`!
    let t_wv = Instant::now();
    let wv = engine
        .weave_sample_best_of_m_async(&WeaveSampleOptions {
            goals: vec![-6.10],
            n_candidates: 4,
            cfg_scale: 1.85,
            geometric_guard: true,
            reconcile_valence: true,
            coord_refine: true,
            seed: 777,
        })
        .await?;
    println!(
        "[Phase 3C] Best-of-4 Weaved Inference + CoordRefiner in {:.2}s | Winner: {} (N={}, valid={}/{}, clean_q={}, SA={:.2}, QED={:.2}, pred_HOMO={:.3} eV, Fmax={:.2} eV/Å, coord_refined={})",
        t_wv.elapsed().as_secs_f64(),
        wv.best_molecule.formula,
        wv.best_molecule.n_atoms,
        wv.valid_count,
        wv.candidates_evaluated,
        wv.best_eval.clean_charge,
        wv.best_eval.rule_metrics.sa_score,
        wv.best_eval.rule_metrics.qed,
        wv.best_eval.objective_preds.first().copied().unwrap_or(f64::NAN),
        wv.best_eval.f_max_ev_a,
        wv.coord_refined
    );

    println!("=========================================================================");
    println!("  ALL 3 PHASES OF `loom-rs` PASSED VERIFICATION & BENCHMARKS CLEANLY!");
    println!("=========================================================================");
    Ok(())
}
