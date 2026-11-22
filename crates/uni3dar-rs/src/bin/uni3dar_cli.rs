use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use uni3dar_rs::cpu_backend::{
    add_atom_head_step_emb, embed_octree_token, eval_atom_head_axis, eval_tree_head, forward_decoder_step_cpu,
    precompute_condition, CpuKvSlot,
};
use uni3dar_rs::gpu_backend::StepHeadMode;
use uni3dar_rs::refiner_critic::{
    compute_coarse_voxel_geometry, compute_element_cloud_features, eval_stage1_critic, eval_stage2_critic,
    run_post_tree_refiner,
};
use uni3dar_rs::sampler::Uni3darSampleOptions;
use uni3dar_rs::Uni3darEngine;

#[derive(Deserialize)]
struct ParityFixture {
    raw_targets: Vec<f32>,
    active_mask: Vec<bool>,
    budget_atoms: i32,
    step0_last_hidden: Vec<f32>,
    step0_tree_logits: Vec<f32>,
    step1_prev_type: usize,
    step1_last_hidden: Vec<f32>,
    step1_tree_logits: Vec<f32>,
    atom_logits_0: Vec<f32>,
    xyz_logits_x: Vec<f32>,
    xyz_logits_y: Vec<f32>,
    xyz_logits_z: Vec<f32>,
    test_elems: Vec<usize>,
    test_pos: Vec<[f32; 3]>,
    refiner_out: Option<RefinerFixture>,
    stage1_out: Option<Stage1Fixture>,
    stage2_out: Option<Stage2Fixture>,
}

#[derive(Deserialize)]
struct RefinerFixture {
    refined_pos: Vec<[f32; 3]>,
    bond_logits_01: Vec<f32>,
    bond_logits_02: Vec<f32>,
}

#[derive(Deserialize)]
struct Stage1Fixture {
    geom_14: Vec<f32>,
    mu_norm: Vec<f32>,
    logit_legal: f32,
    logit_constr: f32,
    utility: f32,
}

#[derive(Deserialize)]
struct Stage2Fixture {
    cloud_42: Vec<f32>,
    mu_norm: Vec<f32>,
    logit_legal: f32,
    logit_constr: f32,
    utility: f32,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn print_usage() {
    eprintln!(
        "Usage:
  uni3dar-cli info   --model <path.u3m>
  uni3dar-cli parity --model <path.u3m> --fixture <parity.json> [--cpu]
  uni3dar-cli sample --model <path.u3m> [--budget 26] [--k 2] [--cfg 2.0] [--targets v0,v1,...] [--seed 42] [--out mol.sdf] [--cpu]
  uni3dar-cli bench  --model <path.u3m> [--budget 26] [--k 1] [--runs 4] [--cpu]"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    let subcmd = args[1].as_str();
    let mut model_path: Option<PathBuf> = None;
    let mut fixture_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut budget: Option<usize> = Some(26);
    let mut spec_k: usize = 2;
    let mut cfg_scale: f32 = 2.0;
    let mut seed: u64 = 42;
    let mut runs: usize = 4;
    let mut force_cpu = false;
    let mut no_reconcile = false;
    let mut no_refiner = false;
    let mut targets_str: Option<String> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--fixture" => {
                fixture_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--budget" => {
                let v: isize = args[i + 1].parse().expect("Invalid --budget");
                budget = if v > 0 { Some(v as usize) } else { None };
                i += 2;
            }
            "--k" => {
                spec_k = args[i + 1].parse().expect("Invalid --k");
                i += 2;
            }
            "--cfg" => {
                cfg_scale = args[i + 1].parse().expect("Invalid --cfg");
                i += 2;
            }
            "--targets" => {
                targets_str = Some(args[i + 1].clone());
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("Invalid --seed");
                i += 2;
            }
            "--runs" => {
                runs = args[i + 1].parse().expect("Invalid --runs");
                i += 2;
            }
            "--cpu" => {
                force_cpu = true;
                i += 1;
            }
            "--no-reconcile" => {
                no_reconcile = true;
                i += 1;
            }
            "--no-refiner" => {
                no_refiner = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
    }

    let mp = model_path.expect("Missing --model <path.u3m>");
    let raw_bytes = fs::read(&mp).expect("Failed to read .u3m model file");
    let mb_size = (raw_bytes.len() as f64) / 1e6;

    match subcmd {
        "info" => {
            let eng = Uni3darEngine::from_bytes_blocking(raw_bytes, !force_cpu).expect("Engine init failed");
            let h = &eng.bundle.header;
            println!("=== uni3dar-rs Bundle Info ===");
            println!("File        : {} ({:.2} MB)", mp.display(), mb_size);
            println!("Backend     : {}", eng.backend_description());
            println!("Quant Dtype : {} (LoRA merged: {})", h.quant_dtype, h.lora_merged_count);
            println!(
                "Backbone    : {} layers, emb_dim={}, heads={}x{}, ffn={}, merge_level={}",
                h.backbone.num_layers,
                h.backbone.emb_dim,
                h.backbone.num_head,
                h.backbone.head_dim,
                h.backbone.ffn_dim,
                h.backbone.merge_level
            );
            println!("Elements    : {:?}", h.backbone.allowed_elements);
            println!("Targets     : {:?} (units: {:?})", h.target_cols, h.target_units);
            println!(
                "Companions  : conditioner={}, refiner={}, stage1_critic={}, stage2_critic={}",
                h.conditioner.is_some(),
                h.refiner.is_some(),
                h.stage1_critic.is_some(),
                h.stage2_critic.is_some()
            );
        }

        "parity" => {
            let fp = fixture_path.expect("Missing --fixture <parity.json>");
            let fx: ParityFixture =
                serde_json::from_str(&fs::read_to_string(&fp).expect("Failed to read parity JSON")).expect("Invalid JSON");
            let mut eng = Uni3darEngine::from_bytes_blocking(raw_bytes, !force_cpu).expect("Engine init failed");
            println!(
                "=== uni3dar-rs Parity Verification ({}, dtype={}) ===",
                eng.backend_description(),
                eng.bundle.header.quant_dtype
            );

            let bb = eng.bundle.header.backbone.clone();
            let cond = precompute_condition(&eng.bundle, Some(&fx.raw_targets), Some(&fx.active_mask)).unwrap();
            let cls_p = bb.cls_phy_pos();
            let p_cnt = cond.num_prefix;

            // Step 0: [prefix_0..P-1, root_MASK]
            let root_emb = embed_octree_token(
                &eng.bundle,
                bb.atom_mask_id,
                [bb.xyz_null_id; 3],
                bb.merge_level as i32,
                fx.budget_atoms,
                Some(&cond.root_delta),
            )
            .unwrap();
            let mut x0 = Vec::with_capacity((p_cnt + 1) * bb.emb_dim);
            let mut pos0 = Vec::with_capacity(p_cnt + 1);
            for p in 0..p_cnt {
                x0.extend_from_slice(&cond.prefix_tokens[p * bb.emb_dim..(p + 1) * bb.emb_dim]);
                let off = ((p as f32) - (p_cnt as f32)) * 16.0 * bb.xyz_resolution;
                let c = cls_p + off;
                pos0.push([c, c, c]);
            }
            x0.extend_from_slice(&root_emb);
            pos0.push([cls_p, cls_p, cls_p]);

            let (h0, tl0, h1, tl1) = if let Some(ref mut g) = eng.gpu {
                g.upload_branch_modulations(&[&cond]).unwrap();
                let o0 = pollster::block_on(g.forward_step_batch_async(
                    &x0,
                    &pos0,
                    &[(0, 0)],
                    p_cnt + 1,
                    2,
                    256,
                    StepHeadMode::HiddenAndTree,
                ))
                .unwrap();

                let e_prev = embed_octree_token(
                    &eng.bundle,
                    fx.step1_prev_type,
                    [bb.xyz_null_id; 3],
                    bb.merge_level as i32,
                    fx.budget_atoms,
                    None,
                )
                .unwrap();
                let e_mask = embed_octree_token(
                    &eng.bundle,
                    bb.atom_mask_id,
                    [bb.xyz_null_id; 3],
                    (bb.merge_level - 1) as i32,
                    fx.budget_atoms,
                    None,
                )
                .unwrap();
                let mut x1 = Vec::with_capacity(2 * bb.emb_dim);
                x1.extend_from_slice(&e_prev);
                x1.extend_from_slice(&e_mask);
                let child_c = 0.5 * ((1usize << (bb.merge_level - 1)) as f32) * bb.grid_len;
                let pos1 = [[cls_p, cls_p, cls_p], [child_c, child_c, child_c]];

                let o1 = pollster::block_on(g.forward_step_batch_async(
                    &x1,
                    &pos1,
                    &[(0, (p_cnt + 1) as u32)],
                    2,
                    2,
                    256,
                    StepHeadMode::HiddenAndTree,
                ))
                .unwrap();
                (o0.last_hidden, o0.tree_logits.unwrap(), o1.last_hidden, o1.tree_logits.unwrap())
            } else {
                let mut kv = CpuKvSlot::new(bb.num_layers, 256, bb.emb_dim);
                let h0 = forward_decoder_step_cpu(&eng.bundle, &x0, &pos0, &cond, &mut kv).unwrap();
                let tl0 = eval_tree_head(&eng.bundle, &h0, 1).unwrap();

                let e_prev = embed_octree_token(
                    &eng.bundle,
                    fx.step1_prev_type,
                    [bb.xyz_null_id; 3],
                    bb.merge_level as i32,
                    fx.budget_atoms,
                    None,
                )
                .unwrap();
                let e_mask = embed_octree_token(
                    &eng.bundle,
                    bb.atom_mask_id,
                    [bb.xyz_null_id; 3],
                    (bb.merge_level - 1) as i32,
                    fx.budget_atoms,
                    None,
                )
                .unwrap();
                let mut x1 = Vec::with_capacity(2 * bb.emb_dim);
                x1.extend_from_slice(&e_prev);
                x1.extend_from_slice(&e_mask);
                let child_c = 0.5 * ((1usize << (bb.merge_level - 1)) as f32) * bb.grid_len;
                let pos1 = [[cls_p, cls_p, cls_p], [child_c, child_c, child_c]];

                let h1 = forward_decoder_step_cpu(&eng.bundle, &x1, &pos1, &cond, &mut kv).unwrap();
                let tl1 = eval_tree_head(&eng.bundle, &h1, 1).unwrap();
                (h0, tl0, h1, tl1)
            };

            println!("  Step-0 Hidden max_abs_diff     : {:.6e}", max_abs_diff(&h0, &fx.step0_last_hidden));
            println!("  Step-0 TreeLogits max_abs_diff : {:.6e}", max_abs_diff(&tl0, &fx.step0_tree_logits));
            println!("  Step-1 KV Hidden max_abs_diff  : {:.6e}", max_abs_diff(&h1, &fx.step1_last_hidden));
            println!("  Step-1 TreeLogits max_abs_diff : {:.6e}", max_abs_diff(&tl1, &fx.step1_tree_logits));

            // 4-axis AtomHead on PyTorch `step1_last_hidden` (isolates AtomHead parity cleanly)
            let mut af = fx.step1_last_hidden.clone();
            let al0 = eval_atom_head_axis(&eng.bundle, 0, &af, 1).unwrap();
            add_atom_head_step_emb(&eng.bundle, 0, 5, &mut af).unwrap();
            let xlx = eval_atom_head_axis(&eng.bundle, 1, &af, 1).unwrap();
            add_atom_head_step_emb(&eng.bundle, 1, 7, &mut af).unwrap();
            let xly = eval_atom_head_axis(&eng.bundle, 2, &af, 1).unwrap();
            add_atom_head_step_emb(&eng.bundle, 2, 15, &mut af).unwrap();
            let xlz = eval_atom_head_axis(&eng.bundle, 3, &af, 1).unwrap();
            println!(
                "  AtomHead (elem/x/y/z) max_diff : {:.6e} / {:.6e} / {:.6e} / {:.6e}",
                max_abs_diff(&al0, &fx.atom_logits_0),
                max_abs_diff(&xlx, &fx.xyz_logits_x),
                max_abs_diff(&xly, &fx.xyz_logits_y),
                max_abs_diff(&xlz, &fx.xyz_logits_z)
            );

            if let Some(rfx) = fx.refiner_out {
                let ro = run_post_tree_refiner(&eng.bundle, &fx.test_elems, &fx.test_pos)
                    .unwrap()
                    .unwrap();
                let mut max_p = 0.0f32;
                for (a, b) in ro.refined_pos.iter().zip(rfx.refined_pos.iter()) {
                    for c in 0..3 {
                        max_p = max_p.max((a[c] - b[c]).abs());
                    }
                }
                let k = fx.test_elems.len();
                let b01 = &ro.bond_logits[(0 * k + 1) * 5..(0 * k + 1 + 1) * 5];
                let b02 = &ro.bond_logits[(0 * k + 2) * 5..(0 * k + 2 + 1) * 5];
                println!(
                    "  PostTreeGraphRefiner max_diff  : pos={:.6e} A, bond01={:.6e}, bond02={:.6e}",
                    max_p,
                    max_abs_diff(b01, &rfx.bond_logits_01),
                    max_abs_diff(b02, &rfx.bond_logits_02)
                );
            }

            if let Some(s1fx) = fx.stage1_out {
                let g14 = compute_coarse_voxel_geometry(
                    &[1, 1, 2],
                    &[344, 344, 344],
                    &[[1.2, 2.4, 3.6], [2.8, 1.9, 4.1], [2.0, 2.0, 3.8]],
                    &[-0.25, -0.40, -0.15],
                    344,
                );
                let mut h_s1 = Vec::with_capacity(4 * bb.emb_dim);
                h_s1.extend_from_slice(&fx.step0_last_hidden);
                h_s1.extend_from_slice(&fx.step1_last_hidden);
                h_s1.extend_from_slice(&fx.step1_last_hidden);
                h_s1.extend_from_slice(&fx.step0_last_hidden);
                let c1 = eval_stage1_critic(
                    &eng.bundle,
                    &h_s1,
                    &g14,
                    &cond.targets_norm,
                    &cond.active_mask,
                    0.80,
                    0.55,
                )
                .unwrap()
                .unwrap();
                println!(
                    "  Stage-1 Critic max_diff        : geom14={:.6e}, mu={:.6e}, leg={:.6e}, con={:.6e}, util={:.6e}",
                    max_abs_diff(&g14, &s1fx.geom_14),
                    max_abs_diff(&c1.mu_norm, &s1fx.mu_norm),
                    (c1.logit_legal - s1fx.logit_legal).abs(),
                    (c1.logit_constr - s1fx.logit_constr).abs(),
                    (c1.utility - s1fx.utility).abs()
                );

                if let Some(s2fx) = fx.stage2_out {
                    let cloud42 = compute_element_cloud_features(&fx.test_elems, &fx.test_pos, &g14);
                    let mut h_s2 = Vec::with_capacity(6 * bb.emb_dim);
                    h_s2.extend_from_slice(&fx.step0_last_hidden);
                    h_s2.extend_from_slice(&fx.step1_last_hidden);
                    h_s2.extend_from_slice(&fx.step1_last_hidden);
                    h_s2.extend_from_slice(&fx.step0_last_hidden);
                    h_s2.extend_from_slice(&fx.step1_last_hidden);
                    h_s2.extend_from_slice(&fx.step0_last_hidden);
                    let c2 = eval_stage2_critic(
                        &eng.bundle,
                        &h_s2,
                        &fx.test_elems,
                        &fx.test_pos,
                        &cloud42,
                        &cond.targets_norm,
                        &cond.active_mask,
                        1.05,
                        0.70,
                    )
                    .unwrap()
                    .unwrap();
                    println!(
                        "  Stage-2 Critic max_diff        : cloud42={:.6e}, mu={:.6e}, leg={:.6e}, con={:.6e}, util={:.6e}",
                        max_abs_diff(&cloud42, &s2fx.cloud_42),
                        max_abs_diff(&c2.mu_norm, &s2fx.mu_norm),
                        (c2.logit_legal - s2fx.logit_legal).abs(),
                        (c2.logit_constr - s2fx.logit_constr).abs(),
                        (c2.utility - s2fx.utility).abs()
                    );
                }
            }
        }

        "sample" => {
            let mut eng = Uni3darEngine::from_bytes_blocking(raw_bytes, !force_cpu).expect("Engine init failed");
            let (cond_targets, cond_active) = parse_cli_targets(&eng.bundle.header.target_cols, targets_str.as_deref());
            let opts = Uni3darSampleOptions {
                budget_atoms: budget,
                speculative_scale: spec_k,
                cfg_scale,
                use_refiner: !no_refiner,
                reconcile_valence: !no_reconcile,
                cond_targets,
                cond_active,
                seed,
                force_cpu,
                ..Default::default()
            };
            let t0 = Instant::now();
            let mol = eng.sample_blocking(&opts, |_, _, _, _| {}).expect("Sampling failed");
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            println!(
                "[uni3dar-rs] {} | formula={} (N={}, heavy={}) | bonds={} | connected={} | valence_valid={} | status={} (RMSD_heavy={:.4} A, dH={:+}) | funnel={:?} | {:.1} ms",
                eng.backend_description(),
                mol.formula,
                mol.n_atoms,
                mol.n_heavy,
                mol.bonds.len(),
                mol.connected,
                mol.valence_valid,
                mol.reconcile_status,
                mol.heavy_rmsd_a,
                mol.delta_h_count,
                mol.funnel_spec,
                dt
            );
            if !mol.predicted_props.is_empty() {
                println!("  Critic predicted props: {:?}", mol.predicted_props);
            }
            if let Some(op) = out_path {
                if op.extension().and_then(|s| s.to_str()) == Some("xyz") {
                    fs::write(&op, mol.to_xyz("uni3dar-rs")).expect("Failed to write .xyz");
                } else {
                    fs::write(&op, mol.to_sdf("uni3dar-rs")).expect("Failed to write .sdf");
                }
                println!("  Wrote molecule -> {}", op.display());
            }
        }

        "bench" => {
            let mut eng = Uni3darEngine::from_bytes_blocking(raw_bytes, !force_cpu).expect("Engine init failed");
            let (cond_targets, cond_active) = parse_cli_targets(&eng.bundle.header.target_cols, targets_str.as_deref());
            println!(
                "=== uni3dar-rs Benchmark ({}, K={}, budget={:?}, runs={}) ===",
                eng.backend_description(),
                spec_k,
                budget,
                runs
            );
            // Warmup
            let w_opts = Uni3darSampleOptions {
                budget_atoms: budget,
                speculative_scale: spec_k,
                cfg_scale,
                cond_targets: cond_targets.clone(),
                cond_active: cond_active.clone(),
                seed,
                force_cpu,
                ..Default::default()
            };
            let _ = eng.sample_blocking(&w_opts, |_, _, _, _| {}).expect("Warmup failed");

            let mut times = Vec::with_capacity(runs);
            let mut valid_cnt = 0usize;
            for r in 0..runs {
                let mut o = w_opts.clone();
                o.seed = seed + (r as u64) + 1;
                let t0 = Instant::now();
                let m = eng.sample_blocking(&o, |_, _, _, _| {}).expect("Bench sample failed");
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                if m.valence_valid && m.connected {
                    valid_cnt += 1;
                }
                println!(
                    "  Run {:>2}: {:>7.1} ms | {:<14} (N={:>2}, steps={:>3}, status={})",
                    r + 1,
                    ms,
                    m.formula,
                    m.n_atoms,
                    m.decode_steps,
                    m.reconcile_status
                );
                times.push(ms);
            }
            let mean_ms = times.iter().sum::<f64>() / (times.len().max(1) as f64);
            println!(
                "Mean latency: {:.1} ms/mol | Connected+Valid: {}/{}",
                mean_ms, valid_cnt, runs
            );
        }

        _ => {
            print_usage();
            std::process::exit(1);
        }
    }
}

fn parse_cli_targets(cols: &[String], spec: Option<&str>) -> (Option<Vec<f32>>, Option<Vec<bool>>) {
    let Some(raw) = spec else {
        return (None, None);
    };
    let n = cols.len();
    if n == 0 {
        return (None, None);
    }
    let mut vals = vec![0.0f32; n];
    let mut mask = vec![false; n];
    for (i, part) in raw.split(',').enumerate() {
        if i >= n {
            break;
        }
        let t = part.trim();
        if t.is_empty() || t == "*" || t.eq_ignore_ascii_case("none") {
            continue;
        }
        if let Ok(v) = t.parse::<f32>() {
            vals[i] = v;
            mask[i] = true;
        }
    }
    (Some(vals), Some(mask))
}
