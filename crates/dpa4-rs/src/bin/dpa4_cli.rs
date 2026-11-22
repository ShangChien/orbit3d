//! Native CLI (`dpa4-cli`) for DeePMD `DPA-4` (`SeZM`: `Nano`, `Mini`, `Neo`, `Air`, `Plus`):
//! - `info <bundle.dpa4>`
//! - `parity <bundle.dpa4> <fixtures.json> <tier> [--gpu|--cpu]`
//! - `relax <bundle.dpa4> [--algo lbfgs|fire] [--freeze-heavy]`
//! - `md <bundle.dpa4> [--steps 50] [--temp 300]`
//! - `bench <bundle.dpa4> [--iters 20] [--gpu|--cpu]`

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod cli_impl {
    use std::env;
    use std::fs;
    use std::time::Instant;

    use dpa4_rs::{Dpa4Engine, Dpa4ExecMode, Dpa4Weights};
    use mlip_core::{
        relax_sync, run_md_sync, MdEnsemble, MdOptions, MolecularSystem, PotentialBackend,
        RelaxAlgorithm, RelaxOptions,
    };
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct FixtureCase {
        name: String,
        symbols: Vec<String>,
        coords_a: Vec<[f64; 3]>,
        charge: f32,
        multiplicity: u32,
        energy_ev: f64,
        forces_ev_a: Vec<[f64; 3]>,
        desc_dim: usize,
        descriptor: Vec<f64>,
    }

    #[derive(Deserialize)]
    struct FixtureFile {
        tiers: std::collections::HashMap<String, Vec<FixtureCase>>,
    }

    fn default_salicylaldehyde() -> MolecularSystem {
        MolecularSystem::from_symbols(
            &[
                "C", "C", "C", "C", "C", "C", "C", "O", "O", "H", "H", "H", "H", "H", "H",
            ]
            .map(String::from),
            vec![
                [0.0000, 1.3950, 0.0120],
                [1.2120, 0.6980, -0.0210],
                [1.2080, -0.7010, 0.0150],
                [-0.0040, -1.3980, -0.0180],
                [-1.2100, -0.6950, 0.0250],
                [-1.2090, 0.7020, -0.0110],
                [2.4850, 1.4250, 0.0450],
                [2.5600, 2.6420, -0.0320],
                [2.3820, -1.3920, -0.0280],
                [-0.0020, 2.4810, 0.0340],
                [-0.0060, -2.4840, -0.0410],
                [-2.1510, -1.2390, 0.0520],
                [-2.1480, 1.2480, -0.0290],
                [3.3950, 0.8120, 0.1280],
                [3.1250, -0.7520, -0.0650],
            ],
            0.0,
            1,
        )
        .unwrap()
    }

    fn parse_mode(args: &[String]) -> Dpa4ExecMode {
        if args.iter().any(|a| a == "--gpu") {
            Dpa4ExecMode::Gpu
        } else if args.iter().any(|a| a == "--cpu") {
            Dpa4ExecMode::Cpu
        } else {
            Dpa4ExecMode::Auto
        }
    }

    pub fn run() -> Result<(), String> {
        let args: Vec<String> = env::args().collect();
        if args.len() < 3 {
            eprintln!(
                "Usage:\n  dpa4-cli info <bundle.dpa4>\n  dpa4-cli parity <bundle.dpa4> <fixtures.json> <tier> [--gpu|--cpu]\n  dpa4-cli relax <bundle.dpa4> [--algo lbfgs|fire] [--freeze-heavy]\n  dpa4-cli md <bundle.dpa4> [--steps 50] [--temp 300]\n  dpa4-cli bench <bundle.dpa4> [--iters 20] [--gpu|--cpu]"
            );
            std::process::exit(1);
        }

        let cmd = args[1].as_str();
        let bundle_path = &args[2];
        let raw =
            fs::read(bundle_path).map_err(|e| format!("Failed to read {bundle_path}: {e}"))?;

        match cmd {
            "info" => {
                let w = Dpa4Weights::from_bytes(&raw)?;
                println!(
                    "DPA-4 Bundle: tier={} | ntypes={} | C={} lmax={} n_blocks={} mix={} F={} rad_mode={} (rank={}) H={} G={} | rcut={:.1} A | oracle={}",
                    w.cfg.tier,
                    w.cfg.ntypes,
                    w.cfg.channels,
                    w.cfg.lmax,
                    w.cfg.n_blocks,
                    w.cfg.mixing_layers,
                    w.cfg.n_focus,
                    w.cfg.radial_so2_mode,
                    w.cfg.radial_so2_rank,
                    w.cfg.ffn_dim,
                    w.cfg.n_grid,
                    w.cfg.rcut,
                    w.oracle.is_some()
                );
            }
            "parity" => {
                if args.len() < 5 {
                    return Err(
                        "Usage: dpa4-cli parity <bundle.dpa4> <fixtures.json> <tier> [--gpu|--cpu]"
                            .into(),
                    );
                }
                let fix_raw = fs::read_to_string(&args[3])
                    .map_err(|e| format!("Failed to read {}: {e}", args[3]))?;
                let fix_file: FixtureFile = serde_json::from_str(&fix_raw)
                    .map_err(|e| format!("Invalid fixture JSON: {e}"))?;
                let tier = &args[4];
                let cases = fix_file
                    .tiers
                    .get(tier)
                    .ok_or_else(|| format!("Tier '{tier}' not found in fixtures"))?;

                let mode = parse_mode(&args);
                let mut engine = pollster::block_on(Dpa4Engine::create(&raw, mode))?;
                println!("=== DPA-4 Parity Check: {} ===", engine.backend_info());

                for (case_idx, tc) in cases.iter().enumerate() {
                    let sys = MolecularSystem::from_symbols(
                        &tc.symbols,
                        tc.coords_a.clone(),
                        tc.charge,
                        tc.multiplicity,
                    )?;
                    let out = engine.eval(&sys, true, true)?;
                    let e_err = (out.energy_ev - tc.energy_ev).abs();

                    let mut max_f_err = 0.0f64;
                    for (fa, fb) in out.forces_ev_a.iter().zip(tc.forces_ev_a.iter()) {
                        for d in 0..3 {
                            max_f_err = max_f_err.max((fa[d] - fb[d]).abs());
                        }
                    }

                    let desc = out.descriptor.as_ref().unwrap();
                    assert_eq!(out.desc_dim, tc.desc_dim);
                    let mut max_d_err = 0.0f64;
                    for (&da, &db) in desc.iter().zip(tc.descriptor.iter()) {
                        max_d_err = max_d_err.max(((da as f64) - db).abs());
                    }

                    // Central finite-difference verification on atom 0
                    let mut max_fd_err = 0.0f64;
                    if case_idx == 0 {
                        let eps_fd = 2e-3f64;
                        for axis in 0..3 {
                            let mut sp = sys.clone();
                            let mut sm = sys.clone();
                            sp.coords_a[0][axis] += eps_fd;
                            sm.coords_a[0][axis] -= eps_fd;
                            let ep = engine.eval(&sp, false, false)?.energy_ev;
                            let em = engine.eval(&sm, false, false)?.energy_ev;
                            let f_fd = -(ep - em) / (2.0 * eps_fd);
                            max_fd_err = max_fd_err.max((out.forces_ev_a[0][axis] - f_fd).abs());
                        }
                    }

                    println!(
                        "  [{}] |dE| = {:.3e} eV | max|dF_pytorch| = {:.3e} eV/A | max|dDesc| = {:.3e}{}",
                        tc.name,
                        e_err,
                        max_f_err,
                        max_d_err,
                        if case_idx == 0 {
                            format!(" | max|F_analyt - F_fd| = {:.3e} eV/A", max_fd_err)
                        } else {
                            String::new()
                        }
                    );

                    if e_err > 1e-4 || max_f_err > 2e-4 {
                        return Err(format!(
                            "DPA-4 parity threshold exceeded on {}: |dE|={e_err:.3e}, max|dF|={max_f_err:.3e}",
                            tc.name
                        ));
                    }
                    if case_idx == 0 && max_fd_err > 5e-3 {
                        return Err(format!(
                            "DPA-4 analytical VJP vs finite-difference mismatch: {max_fd_err:.3e} eV/A"
                        ));
                    }
                }
                println!(
                    "PASS: All {} parity & analytical VJP checks succeeded!",
                    cases.len()
                );
            }
            "relax" => {
                let mode = parse_mode(&args);
                let mut engine = pollster::block_on(Dpa4Engine::create(&raw, mode))?;
                let algo = if args.iter().any(|a| a == "fire") {
                    RelaxAlgorithm::Fire
                } else {
                    RelaxAlgorithm::Lbfgs
                };
                let freeze_heavy = args.iter().any(|a| a == "--freeze-heavy");
                let sys = default_salicylaldehyde();
                let t0 = Instant::now();
                let res = relax_sync(
                    &mut engine,
                    &sys,
                    &RelaxOptions {
                        algorithm: algo,
                        max_steps: 60,
                        fmax_ev_a: 0.03,
                        freeze_heavy,
                        stream_every: 10,
                        ..Default::default()
                    },
                    false,
                    |fr| {
                        println!(
                            "  step {:3} | E = {:13.5} eV | fmax = {:8.4} eV/A",
                            fr.step, fr.energy_ev, fr.fmax_ev_a
                        );
                    },
                )?;
                println!(
                    "Relaxation finished in {:.1} ms ({} steps, converged={}): dE = {:.4} eV, fmax {:.4} -> {:.4} eV/A, heavy_RMSD = {:.4} A",
                    t0.elapsed().as_secs_f64() * 1000.0,
                    res.steps_taken,
                    res.converged,
                    res.delta_energy_ev,
                    res.initial_fmax_ev_a,
                    res.final_fmax_ev_a,
                    res.heavy_rmsd_a
                );
            }
            "md" => {
                let mode = parse_mode(&args);
                let mut engine = pollster::block_on(Dpa4Engine::create(&raw, mode))?;
                let sys = default_salicylaldehyde();
                let t0 = Instant::now();
                let res = run_md_sync(
                    &mut engine,
                    &sys,
                    &MdOptions {
                        ensemble: MdEnsemble::NvtLangevinBaoab,
                        n_steps: 50,
                        dt_fs: 0.5,
                        temperature_k: 300.0,
                        stream_every: 10,
                        ..Default::default()
                    },
                    |fr| {
                        println!(
                            "  t = {:5.1} fs | E_pot = {:13.4} eV | E_kin = {:6.4} eV | T = {:6.1} K",
                            fr.time_fs,
                            fr.potential_energy_ev,
                            fr.kinetic_energy_ev,
                            fr.temperature_k
                        );
                    },
                )?;
                println!(
                    "MD finished in {:.1} ms: mean T = {:.1} K, final E_pot = {:.4} eV",
                    t0.elapsed().as_secs_f64() * 1000.0,
                    res.mean_temperature_k,
                    res.final_potential_ev
                );
            }
            "bench" => {
                let mode = parse_mode(&args);
                let mut engine = pollster::block_on(Dpa4Engine::create(&raw, mode))?;
                let sys = default_salicylaldehyde();
                let _ = engine.eval(&sys, true, false)?;
                let iters = 20usize;
                let t0 = Instant::now();
                for _ in 0..iters {
                    let _ = engine.eval(&sys, true, false)?;
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
                println!(
                    "{} | 15-atom E+F single-point: {:.3} ms/step ({:.1} steps/sec)",
                    engine.backend_info(),
                    ms,
                    1000.0 / ms
                );
            }
            other => return Err(format!("Unknown subcommand '{other}'")),
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), String> {
    cli_impl::run()
}
