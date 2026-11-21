#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod cli {
    use flowmol_rs::config::{ForwardOutput, SelfCondState};
    use flowmol_rs::cpu_backend::precompute_condition_modulations;
    use flowmol_rs::sampler::SampleOptions;
    use flowmol_rs::FlowMolEngine;
    use serde::Deserialize;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    #[derive(Deserialize)]
    struct ParityFixture {
        n_atoms: usize,
        x_t: Vec<[f32; 3]>,
        a_t: Vec<usize>,
        c_t: Vec<usize>,
        e_t_upper: Vec<usize>,
        #[serde(default)]
        cond_values: Option<Vec<f32>>,
        #[serde(default)]
        cond_active: Option<Vec<bool>>,
        #[serde(default)]
        node_cond_flags: Option<Vec<[f32; 2]>>,
        t1: f32,
        out1: HashMap<String, serde_json::Value>,
        t2: f32,
        out2: HashMap<String, serde_json::Value>,
    }

    fn flatten_2d(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32))
            .collect()
    }

    fn max_and_mean_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
        assert_eq!(a.len(), b.len());
        let mut mx = 0.0f32;
        let mut s = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > mx {
                mx = d;
            }
            s += d as f64;
        }
        (mx, (s / (a.len().max(1) as f64)) as f32)
    }

    fn softmax_2d(logits: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let lr = &logits[r * cols..(r + 1) * cols];
            let mx = lr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for c in 0..cols {
                let e = (lr[c] - mx).exp();
                out[r * cols + c] = e;
                sum += e;
            }
            for c in 0..cols {
                out[r * cols + c] /= sum;
            }
        }
        out
    }

    fn compare_out(label: &str, got: &ForwardOutput, exp: &HashMap<String, serde_json::Value>) -> f32 {
        let got_x: &[f32] = bytemuck::cast_slice(&got.x);
        let exp_x = flatten_2d(&exp["x"]);
        let exp_a = flatten_2d(&exp["a"]);
        let exp_c = flatten_2d(&exp["c"]);
        let exp_e = flatten_2d(&exp["e"]);

        let (mx_x, mn_x) = max_and_mean_diff(got_x, &exp_x);
        let (mx_a, mn_a) = max_and_mean_diff(&got.a_logits, &exp_a);
        let (mx_c, mn_c) = max_and_mean_diff(&got.c_logits, &exp_c);
        let (mx_e, mn_e) = max_and_mean_diff(&got.e_logits, &exp_e);
        println!(
            "  [{label}] x: max={mx_x:.6} mean={mn_x:.6} | a: max={mx_a:.6} mean={mn_a:.6} | c: max={mx_c:.6} mean={mn_c:.6} | e: max={mx_e:.6} mean={mn_e:.6}"
        );
        mx_x.max(mx_a).max(mx_c).max(mx_e)
    }

    pub fn run() -> Result<(), String> {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            println!("Usage: flowmol-cli <info|parity|sample|bench> --model <path.fm118> [options]");
            return Ok(());
        }
        let cmd = args[1].as_str();
        let get_arg = |flag: &str| -> Option<String> {
            args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
        };
        let has_flag = |flag: &str| -> bool { args.iter().any(|a| a == flag) };

        let model_path = PathBuf::from(get_arg("--model").ok_or("Missing --model <path.fm118>")?);
        let force_cpu = has_flag("--cpu");
        let t_load = Instant::now();
        let bytes = fs::read(&model_path).map_err(|e| format!("Failed to read {}: {e}", model_path.display()))?;
        let mb = (bytes.len() as f64) / 1e6;
        let mut engine = FlowMolEngine::from_bytes_blocking(bytes, !force_cpu, 64)?;
        println!(
            "[flowmol-rs] Loaded {} ({mb:.2} MB, quant={}) on {} in {:.1} ms",
            model_path.display(),
            engine.bundle.header.quant_dtype,
            engine.backend_description(),
            t_load.elapsed().as_secs_f64() * 1e3
        );

        match cmd {
            "info" => {
                let h = &engine.bundle.header;
                println!("  format      : {}", h.format);
                println!("  step        : {} ({} samples)", h.step, h.samples_seen);
                println!(
                    "  backbone    : L={} S={} E={} V={} cp={} rbf={}",
                    h.config.n_molecule_updates,
                    h.config.n_hidden_scalars,
                    h.config.n_hidden_edge_feats,
                    h.config.n_vec_channels,
                    h.config.n_cp_feats,
                    h.config.rbf_dim
                );
                println!("  tensors     : {}", h.tensors.len());
                println!("  conditioner : {}", h.conditioner.is_some());
            }
            "parity" => {
                let fix_path = PathBuf::from(get_arg("--fixture").ok_or("Missing --fixture <parity.json>")?);
                let fix: ParityFixture = serde_json::from_str(&fs::read_to_string(&fix_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
                let n = fix.n_atoms;
                let n_up = n * (n - 1) / 2;
                let a_num = engine.bundle.header.config.n_atom_types();
                let c_num = engine.bundle.header.config.n_charges;
                let b_num = engine.bundle.header.config.n_bond_types;

                let mods = match (&fix.cond_values, &fix.cond_active) {
                    (Some(cv), Some(ca)) => precompute_condition_modulations(&engine.bundle, cv, ca)?,
                    _ => None,
                };
                for (mode_lbl, use_cpu) in [("CPU", true), ("GPU (wgpu)", false)] {
                    if !use_cpu && engine.gpu.is_none() {
                        continue;
                    }
                    println!("=== Parity Check: {mode_lbl} vs PyTorch (N={n}) ===");
                    let out1 = pollster::block_on(engine.forward_single_async(
                        &fix.x_t,
                        &fix.a_t,
                        &fix.c_t,
                        &fix.e_t_upper,
                        fix.t1,
                        None,
                        mods.as_ref(),
                        fix.node_cond_flags.as_deref(),
                        use_cpu,
                    ))?;
                    let err1 = compare_out("Pass 1 (no-SC)", &out1, &fix.out1);

                    // Use exact PyTorch Pass-1 outputs to test Pass-2 SC isolation cleanly
                    let exp1_x: Vec<[f32; 3]> = fix.out1["x"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|r| {
                            let a = r.as_array().unwrap();
                            [
                                a[0].as_f64().unwrap() as f32,
                                a[1].as_f64().unwrap() as f32,
                                a[2].as_f64().unwrap() as f32,
                            ]
                        })
                        .collect();
                    let sc = SelfCondState {
                        x: exp1_x,
                        a_probs: softmax_2d(&flatten_2d(&fix.out1["a"]), n, a_num),
                        c_probs: softmax_2d(&flatten_2d(&fix.out1["c"]), n, c_num),
                        e_probs: softmax_2d(&flatten_2d(&fix.out1["e"]), n_up, b_num),
                    };
                    let out2 = pollster::block_on(engine.forward_single_async(
                        &fix.x_t,
                        &fix.a_t,
                        &fix.c_t,
                        &fix.e_t_upper,
                        fix.t2,
                        Some(&sc),
                        mods.as_ref(),
                        fix.node_cond_flags.as_deref(),
                        use_cpu,
                    ))?;
                    let err2 = compare_out("Pass 2 (with-SC)", &out2, &fix.out2);
                    println!("  -> Worst max_abs_diff ({mode_lbl}) = {:.6}", err1.max(err2));
                }
            }
            "sample" => {
                let n_mols: usize = get_arg("--n-mols").and_then(|s| s.parse().ok()).unwrap_or(1);
                let mut base_opts = SampleOptions::default();
                base_opts.n_atoms = get_arg("--n-atoms").and_then(|s| s.parse().ok());
                base_opts.domain = get_arg("--domain");
                if let Some(st) = get_arg("--steps").and_then(|s| s.parse().ok()) {
                    base_opts.n_steps = st;
                }
                if let Some(sv) = get_arg("--solver") {
                    base_opts.solver = sv;
                }
                if let Some(gr) = get_arg("--grid") {
                    base_opts.grid_type = gr;
                }
                if let Some(sd) = get_arg("--seed").and_then(|s| s.parse().ok()) {
                    base_opts.seed = sd;
                }
                if let Some(al) = get_arg("--allow") {
                    base_opts.allowed_elements = Some(al.split(',').map(|s| s.trim().to_string()).collect());
                }
                if let Some(ch) = get_arg("--charges") {
                    base_opts.allowed_charges = Some(ch.split(',').filter_map(|s| s.trim().parse().ok()).collect());
                }
                if let Some(rq) = get_arg("--require") {
                    base_opts.required_elements = rq.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                }
                base_opts.force_cpu = force_cpu;

                let mut sdf_all = String::new();
                for idx in 0..n_mols {
                    let mut opts = base_opts.clone();
                    opts.seed = base_opts.seed + (idx as u64) * 101;
                    let t0 = Instant::now();
                    let mol = engine.sample_blocking(&opts, |_, _, _, _| {})?;
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    println!(
                        "  [#{idx}] {:18} | N={:2} (heavy={:2}) | valid={} conn={} (frags={}) | d_min={:.3} A | elems={:?} | {:.1} ms",
                        mol.formula,
                        mol.n_atoms,
                        mol.n_heavy,
                        mol.valence_valid,
                        mol.connected,
                        mol.n_fragments,
                        mol.min_dist,
                        mol.elements,
                        ms
                    );
                    for mc in &mol.metals {
                        let nb_s: Vec<String> = mc.neighbors.iter().map(|n| format!("{}:{:.2}A", n.symbol, n.distance)).collect();
                        println!("        metal {} (CN={}): [{}]", mc.symbol, mc.coordination_number, nb_s.join(", "));
                    }
                    sdf_all.push_str(&mol.to_sdf(&format!("mol_{idx}")));
                }
                if let Some(out_p) = get_arg("--out") {
                    fs::write(&out_p, sdf_all).map_err(|e| e.to_string())?;
                    println!("[flowmol-rs] Saved {n_mols} molecules to {out_p}");
                }
            }
            "bench" => {
                let n_atoms: usize = get_arg("--n-atoms").and_then(|s| s.parse().ok()).unwrap_or(40);
                let steps: usize = get_arg("--steps").and_then(|s| s.parse().ok()).unwrap_or(25);
                let solver = get_arg("--solver").unwrap_or_else(|| "heun".to_string());
                let warmup: usize = get_arg("--warmup").and_then(|s| s.parse().ok()).unwrap_or(1);
                let iters: usize = get_arg("--iters").and_then(|s| s.parse().ok()).unwrap_or(3);

                let mut opts = SampleOptions {
                    n_atoms: Some(n_atoms),
                    n_steps: steps,
                    solver: solver.clone(),
                    force_cpu,
                    ..Default::default()
                };
                for w in 0..warmup {
                    opts.seed = 10 + w as u64;
                    let _ = engine.sample_blocking(&opts, |_, _, _, _| {})?;
                }
                let mut times = Vec::with_capacity(iters);
                for it in 0..iters {
                    opts.seed = 1000 + it as u64;
                    let t0 = Instant::now();
                    let _ = engine.sample_blocking(&opts, |_, _, _, _| {})?;
                    times.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                let mean_ms = times.iter().sum::<f64>() / (times.len() as f64);
                let nfe = if solver == "heun" { 2 * steps - 1 } else { steps };
                println!(
                    "[bench] N={n_atoms} solver={solver} steps={steps} ({nfe} NFE) | mean={mean_ms:.1} ms/mol ({:.2} ms/NFE, {:.2} mol/s)",
                    mean_ms / (nfe as f64),
                    1000.0 / mean_ms
                );
            }
            other => return Err(format!("Unknown command '{other}'")),
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(e) = cli::run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
