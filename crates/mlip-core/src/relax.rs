//! Monomorphized 3D geometry relaxation engines:
//! - `FIRE` (`Fast Inertial Relaxation Engine`, Bitzek et al., PRL 2006) — robust from rough generated geometries
//! - `L-BFGS` (Two-loop limited-memory quasi-Newton with per-atom trust radius & uphill backtracking) — superlinear local convergence
//!
//! Both support `freeze_heavy = true` (`Z > 1` fixed -> `RMSD_heavy == 0.0000 Å` quantum H-polishing!)
//! and live frame streaming callbacks for 60fps WebGPU interactive visualization.

use serde::{Deserialize, Serialize};

use crate::types::{
    block_on_ready, AsyncPotentialBackend, MolecularSystem, PotentialBackend, PotentialEval,
    SyncAsAsync,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RelaxAlgorithm {
    Fire,
    Lbfgs,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelaxOptions {
    pub algorithm: RelaxAlgorithm,
    /// Maximum optimizer iterations (default `80`).
    pub max_steps: usize,
    /// Convergence threshold on `max_i ||F_i||_2` in `eV/Å` (default `0.03`).
    pub fmax_ev_a: f64,
    /// Maximum per-atom coordinate displacement per step in `Å` (default `0.15`).
    pub max_step_a: f64,
    /// Optional per-atom steric-clash force magnitude cap in `eV/Å` (default `Some(25.0)`).
    pub max_force_clip_ev_a: Option<f64>,
    /// When `true`, all heavy atoms (`Z > 1`) are strictly frozen (`RMSD_heavy == 0.0000 Å`)
    /// so only hydrogen positions relax on the quantum potential energy surface.
    pub freeze_heavy: bool,
    /// Optional explicit per-atom freeze mask (`true` = frozen). Combined with `freeze_heavy`.
    pub frozen_mask: Option<Vec<bool>>,
    /// Remove net translational force when no atoms are frozen (default `true`).
    pub zero_com: bool,
    /// Emit streaming frame callback every `stream_every` steps (`0` disables intermediate callbacks).
    pub stream_every: usize,
    /// L-BFGS history memory size `m` (default `10`).
    pub lbfgs_memory: usize,
}

impl Default for RelaxOptions {
    fn default() -> Self {
        Self {
            algorithm: RelaxAlgorithm::Lbfgs,
            max_steps: 80,
            fmax_ev_a: 0.03,
            max_step_a: 0.15,
            max_force_clip_ev_a: Some(25.0),
            freeze_heavy: false,
            frozen_mask: None,
            zero_com: true,
            stream_every: 1,
            lbfgs_memory: 10,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelaxFrame {
    pub step: usize,
    pub energy_ev: f64,
    pub fmax_ev_a: f64,
    pub frms_ev_a: f64,
    pub coords_a: Vec<[f64; 3]>,
    pub forces_ev_a: Vec<[f64; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelaxResult {
    pub converged: bool,
    pub steps_taken: usize,
    pub force_evals: usize,
    pub initial_energy_ev: f64,
    pub final_energy_ev: f64,
    pub delta_energy_ev: f64,
    pub initial_fmax_ev_a: f64,
    pub final_fmax_ev_a: f64,
    pub heavy_rmsd_a: f64,
    pub all_rmsd_a: f64,
    pub final_system: MolecularSystem,
    pub final_forces_ev_a: Vec<[f64; 3]>,
    pub final_descriptor: Option<Vec<f32>>,
    pub desc_dim: usize,
    pub trajectory_energies_ev: Vec<f64>,
    pub trajectory_fmax_ev_a: Vec<f64>,
}

fn build_effective_frozen_mask(sys: &MolecularSystem, opts: &RelaxOptions) -> Vec<bool> {
    let n = sys.n_atoms();
    let mut mask = vec![false; n];
    if opts.freeze_heavy {
        for (i, &z) in sys.atomic_numbers.iter().enumerate() {
            if z > 1 {
                mask[i] = true;
            }
        }
    }
    if let Some(ref user_mask) = opts.frozen_mask {
        for (i, &frozen) in user_mask.iter().take(n).enumerate() {
            if frozen {
                mask[i] = true;
            }
        }
    }
    mask
}

fn sanitize_eval(
    mut ev: PotentialEval,
    frozen_mask: &[bool],
    any_frozen: bool,
    opts: &RelaxOptions,
) -> PotentialEval {
    if let Some(clip) = opts.max_force_clip_ev_a {
        ev.clip_forces(clip);
    }
    if opts.zero_com && !any_frozen {
        ev.zero_com_forces(None);
    } else if any_frozen {
        for (i, &frz) in frozen_mask.iter().enumerate() {
            if frz {
                ev.forces_ev_a[i] = [0.0, 0.0, 0.0];
            }
        }
    }
    ev
}

fn cap_per_atom_step(step: &mut [[f64; 3]], frozen_mask: &[bool], max_step_a: f64) {
    let mut max_disp = 0.0f64;
    for (i, d) in step.iter_mut().enumerate() {
        if frozen_mask[i] {
            *d = [0.0, 0.0, 0.0];
            continue;
        }
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if norm > max_disp {
            max_disp = norm;
        }
    }
    if max_disp > max_step_a && max_disp > 0.0 {
        let scale = max_step_a / max_disp;
        for d in step.iter_mut() {
            d[0] *= scale;
            d[1] *= scale;
            d[2] *= scale;
        }
    }
}

/// Synchronous 3D geometry relaxation over any [`PotentialBackend`] (zero async/vtable overhead).
pub fn relax_sync<B, F>(
    backend: &mut B,
    initial_system: &MolecularSystem,
    opts: &RelaxOptions,
    need_final_descriptor: bool,
    on_frame: F,
) -> Result<RelaxResult, String>
where
    B: PotentialBackend,
    F: FnMut(&RelaxFrame),
{
    let mut adapter = SyncAsAsync(backend);
    block_on_ready(relax_async(
        &mut adapter,
        initial_system,
        opts,
        need_final_descriptor,
        on_frame,
    ))
}

/// Asynchronous 3D geometry relaxation over any [`AsyncPotentialBackend`] (Browser WebGPU + Native).
pub async fn relax_async<B, F>(
    backend: &mut B,
    initial_system: &MolecularSystem,
    opts: &RelaxOptions,
    need_final_descriptor: bool,
    mut on_frame: F,
) -> Result<RelaxResult, String>
where
    B: AsyncPotentialBackend,
    F: FnMut(&RelaxFrame),
{
    let n = initial_system.n_atoms();
    let frozen_mask = build_effective_frozen_mask(initial_system, opts);
    let any_frozen = frozen_mask.iter().any(|&b| b);
    let all_frozen = frozen_mask.iter().all(|&b| b);

    let mut cur_sys = initial_system.clone();
    let mut force_evals = 1usize;
    let raw_init = backend
        .eval_async(&cur_sys, true, need_final_descriptor && (opts.max_steps == 0 || all_frozen))
        .await?;
    let mut cur_eval = sanitize_eval(raw_init, &frozen_mask, any_frozen, opts);

    let initial_energy_ev = cur_eval.energy_ev;
    let initial_fmax_ev_a = cur_eval.max_force_norm(Some(&frozen_mask));
    let mut cur_fmax = initial_fmax_ev_a;
    let mut traj_e = vec![cur_eval.energy_ev];
    let mut traj_f = vec![cur_fmax];

    if opts.stream_every > 0 {
        on_frame(&RelaxFrame {
            step: 0,
            energy_ev: cur_eval.energy_ev,
            fmax_ev_a: cur_fmax,
            frms_ev_a: cur_eval.rms_force(Some(&frozen_mask)),
            coords_a: cur_sys.coords_a.clone(),
            forces_ev_a: cur_eval.forces_ev_a.clone(),
        });
    }

    if n == 0 || all_frozen || cur_fmax <= opts.fmax_ev_a || opts.max_steps == 0 {
        if need_final_descriptor && cur_eval.descriptor.is_none() {
            let desc_ev = backend.eval_async(&cur_sys, false, true).await?;
            cur_eval.descriptor = desc_ev.descriptor;
            cur_eval.desc_dim = desc_ev.desc_dim;
            force_evals += 1;
        }
        return Ok(RelaxResult {
            converged: cur_fmax <= opts.fmax_ev_a || all_frozen,
            steps_taken: 0,
            force_evals,
            initial_energy_ev,
            final_energy_ev: cur_eval.energy_ev,
            delta_energy_ev: cur_eval.energy_ev - initial_energy_ev,
            initial_fmax_ev_a,
            final_fmax_ev_a: cur_fmax,
            heavy_rmsd_a: initial_system.heavy_atom_rmsd(&cur_sys),
            all_rmsd_a: initial_system.all_atom_rmsd(&cur_sys),
            final_system: cur_sys,
            final_forces_ev_a: cur_eval.forces_ev_a,
            final_descriptor: cur_eval.descriptor,
            desc_dim: cur_eval.desc_dim,
            trajectory_energies_ev: traj_e,
            trajectory_fmax_ev_a: traj_f,
        });
    }

    let mut steps_taken = 0usize;

    match opts.algorithm {
        RelaxAlgorithm::Fire => {
            // Exact FIRE parameters (ASE / Bitzek et al. 2006)
            let n_min = 5usize;
            let f_inc = 1.1f64;
            let f_dec = 0.5f64;
            let alpha_start = 0.1f64;
            let f_alpha = 0.99f64;
            let mut dt = 0.10f64;
            let dt_max = 1.0f64;
            let dt_min = 0.002f64;
            let mut alpha = alpha_start;
            let mut n_pos = 0usize;
            let mut vel = vec![[0.0f64; 3]; n];

            for step in 1..=opts.max_steps {
                // Power P = F . v
                let mut power = 0.0f64;
                let mut v_norm_sq = 0.0f64;
                let mut f_norm_sq = 0.0f64;
                for i in 0..n {
                    if frozen_mask[i] {
                        continue;
                    }
                    let f = cur_eval.forces_ev_a[i];
                    let v = vel[i];
                    power += f[0] * v[0] + f[1] * v[1] + f[2] * v[2];
                    v_norm_sq += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    f_norm_sq += f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
                }

                if power > 0.0 {
                    let ratio = if f_norm_sq > 1e-30 {
                        (v_norm_sq / f_norm_sq).sqrt()
                    } else {
                        0.0
                    };
                    for i in 0..n {
                        if frozen_mask[i] {
                            vel[i] = [0.0, 0.0, 0.0];
                        } else {
                            let f = cur_eval.forces_ev_a[i];
                            vel[i][0] = (1.0 - alpha) * vel[i][0] + alpha * f[0] * ratio;
                            vel[i][1] = (1.0 - alpha) * vel[i][1] + alpha * f[1] * ratio;
                            vel[i][2] = (1.0 - alpha) * vel[i][2] + alpha * f[2] * ratio;
                        }
                    }
                    if n_pos > n_min {
                        dt = (dt * f_inc).min(dt_max);
                        alpha *= f_alpha;
                    }
                    n_pos += 1;
                } else {
                    for v in &mut vel {
                        *v = [0.0, 0.0, 0.0];
                    }
                    alpha = alpha_start;
                    dt = (dt * f_dec).max(dt_min);
                    n_pos = 0;
                }

                // Semi-implicit Euler velocity kick + displacement
                let mut disp = vec![[0.0f64; 3]; n];
                for i in 0..n {
                    if !frozen_mask[i] {
                        let f = cur_eval.forces_ev_a[i];
                        vel[i][0] += dt * f[0];
                        vel[i][1] += dt * f[1];
                        vel[i][2] += dt * f[2];
                        disp[i] = [dt * vel[i][0], dt * vel[i][1], dt * vel[i][2]];
                    }
                }
                cap_per_atom_step(&mut disp, &frozen_mask, opts.max_step_a);

                let prev_coords = cur_sys.coords_a.clone();
                let prev_energy = cur_eval.energy_ev;
                for i in 0..n {
                    cur_sys.coords_a[i][0] += disp[i][0];
                    cur_sys.coords_a[i][1] += disp[i][1];
                    cur_sys.coords_a[i][2] += disp[i][2];
                }

                let cand_raw = backend.eval_async(&cur_sys, true, false).await?;
                force_evals += 1;
                let cand_eval = sanitize_eval(cand_raw, &frozen_mask, any_frozen, opts);

                // Uphill energy safeguard: if E jumped up significantly, damp & take conservative steepest step
                if cand_eval.energy_ev > prev_energy + 1e-3 {
                    for v in &mut vel {
                        *v = [0.0, 0.0, 0.0];
                    }
                    alpha = alpha_start;
                    dt = (dt * f_dec).max(dt_min);
                    n_pos = 0;

                    for i in 0..n {
                        cur_sys.coords_a[i][0] = prev_coords[i][0] + 0.25 * disp[i][0];
                        cur_sys.coords_a[i][1] = prev_coords[i][1] + 0.25 * disp[i][1];
                        cur_sys.coords_a[i][2] = prev_coords[i][2] + 0.25 * disp[i][2];
                    }
                    let retry_raw = backend.eval_async(&cur_sys, true, false).await?;
                    force_evals += 1;
                    let retry_eval = sanitize_eval(retry_raw, &frozen_mask, any_frozen, opts);
                    if retry_eval.energy_ev <= cand_eval.energy_ev {
                        cur_eval = retry_eval;
                    } else {
                        cur_eval = cand_eval;
                    }
                } else {
                    cur_eval = cand_eval;
                }

                cur_fmax = cur_eval.max_force_norm(Some(&frozen_mask));
                traj_e.push(cur_eval.energy_ev);
                traj_f.push(cur_fmax);
                steps_taken = step;

                let converged = cur_fmax <= opts.fmax_ev_a;
                if opts.stream_every > 0 && (step % opts.stream_every == 0 || converged || step == opts.max_steps) {
                    on_frame(&RelaxFrame {
                        step,
                        energy_ev: cur_eval.energy_ev,
                        fmax_ev_a: cur_fmax,
                        frms_ev_a: cur_eval.rms_force(Some(&frozen_mask)),
                        coords_a: cur_sys.coords_a.clone(),
                        forces_ev_a: cur_eval.forces_ev_a.clone(),
                    });
                }
                if converged {
                    break;
                }
            }
        }
        RelaxAlgorithm::Lbfgs => {
            // Limited-memory BFGS with two-loop recursion on g = -F (dimension 3*N)
            let dim = 3 * n;
            let mem = opts.lbfgs_memory.max(1);
            let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(mem);
            let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(mem);
            let mut rho_hist: Vec<f64> = Vec::with_capacity(mem);

            let flatten_grad = |forces: &[[f64; 3]], mask: &[bool]| -> Vec<f64> {
                let mut g = vec![0.0f64; dim];
                for i in 0..n {
                    if !mask[i] {
                        g[3 * i] = -forces[i][0];
                        g[3 * i + 1] = -forces[i][1];
                        g[3 * i + 2] = -forces[i][2];
                    }
                }
                g
            };

            let dot = |a: &[f64], b: &[f64]| -> f64 {
                a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
            };

            let mut g_cur = flatten_grad(&cur_eval.forces_ev_a, &frozen_mask);

            for step in 1..=opts.max_steps {
                // L-BFGS two-loop recursion for search direction p = -H_k * g_cur
                let mut q = g_cur.clone();
                let k_len = s_hist.len();
                let mut alphas = vec![0.0f64; k_len];
                for idx in (0..k_len).rev() {
                    let a = rho_hist[idx] * dot(&s_hist[idx], &q);
                    alphas[idx] = a;
                    for d in 0..dim {
                        q[d] -= a * y_hist[idx][d];
                    }
                }

                // Initial inverse Hessian scale H_0 = gamma * I
                let gamma = if k_len > 0 {
                    let sy = dot(&s_hist[k_len - 1], &y_hist[k_len - 1]);
                    let yy = dot(&y_hist[k_len - 1], &y_hist[k_len - 1]);
                    if yy > 1e-20 && sy > 0.0 {
                        (sy / yy).clamp(1e-3, 0.25)
                    } else {
                        0.02
                    }
                } else {
                    // Initial steepest-descent inverse curvature (~1 / 50 eV/Å²)
                    (0.02f64).min(opts.max_step_a / cur_fmax.max(1e-6))
                };

                let mut r: Vec<f64> = q.iter().map(|&v| v * gamma).collect();
                for idx in 0..k_len {
                    let b = rho_hist[idx] * dot(&y_hist[idx], &r);
                    let diff = alphas[idx] - b;
                    for d in 0..dim {
                        r[d] += s_hist[idx][d] * diff;
                    }
                }

                // Search direction p = -r; verify descent p . g_cur < 0, otherwise reset history
                let mut p: Vec<f64> = r.iter().map(|&v| -v).collect();
                if dot(&p, &g_cur) >= 0.0 {
                    s_hist.clear();
                    y_hist.clear();
                    rho_hist.clear();
                    let fallback_scale = (0.02f64).min(opts.max_step_a / cur_fmax.max(1e-6));
                    for d in 0..dim {
                        p[d] = -fallback_scale * g_cur[d];
                    }
                }

                let mut disp = vec![[0.0f64; 3]; n];
                for i in 0..n {
                    if !frozen_mask[i] {
                        disp[i] = [p[3 * i], p[3 * i + 1], p[3 * i + 2]];
                    }
                }
                cap_per_atom_step(&mut disp, &frozen_mask, opts.max_step_a);

                let prev_coords = cur_sys.coords_a.clone();
                let prev_energy = cur_eval.energy_ev;

                // Up to 3 backtracking line-search trials if energy increases non-trivially
                let mut scale_try = 1.0f64;
                let mut accepted_eval: Option<PotentialEval> = None;
                let mut accepted_coords = prev_coords.clone();
                for trial in 0..3 {
                    let mut cand_sys = cur_sys.clone();
                    for i in 0..n {
                        cand_sys.coords_a[i][0] = prev_coords[i][0] + scale_try * disp[i][0];
                        cand_sys.coords_a[i][1] = prev_coords[i][1] + scale_try * disp[i][1];
                        cand_sys.coords_a[i][2] = prev_coords[i][2] + scale_try * disp[i][2];
                    }
                    let cand_raw = backend.eval_async(&cand_sys, true, false).await?;
                    force_evals += 1;
                    let cand_eval = sanitize_eval(cand_raw, &frozen_mask, any_frozen, opts);

                    if cand_eval.energy_ev <= prev_energy + 1e-4 || trial == 2 {
                        accepted_coords = cand_sys.coords_a;
                        accepted_eval = Some(cand_eval);
                        break;
                    }
                    // Reset quasi-Newton history on overshoot & shrink trial step
                    if trial == 0 {
                        s_hist.clear();
                        y_hist.clear();
                        rho_hist.clear();
                    }
                    scale_try *= 0.35;
                }

                cur_sys.coords_a = accepted_coords;
                cur_eval = accepted_eval.unwrap();
                let g_next = flatten_grad(&cur_eval.forces_ev_a, &frozen_mask);

                // Update (s_k, y_k) curvature pair
                let mut s_vec = vec![0.0f64; dim];
                let mut y_vec = vec![0.0f64; dim];
                for i in 0..n {
                    if !frozen_mask[i] {
                        s_vec[3 * i] = cur_sys.coords_a[i][0] - prev_coords[i][0];
                        s_vec[3 * i + 1] = cur_sys.coords_a[i][1] - prev_coords[i][1];
                        s_vec[3 * i + 2] = cur_sys.coords_a[i][2] - prev_coords[i][2];
                        y_vec[3 * i] = g_next[3 * i] - g_cur[3 * i];
                        y_vec[3 * i + 1] = g_next[3 * i + 1] - g_cur[3 * i + 1];
                        y_vec[3 * i + 2] = g_next[3 * i + 2] - g_cur[3 * i + 2];
                    }
                }
                let sy = dot(&s_vec, &y_vec);
                let s_norm = dot(&s_vec, &s_vec).sqrt();
                let y_norm = dot(&y_vec, &y_vec).sqrt();
                if sy > 1e-10 * s_norm * y_norm {
                    if s_hist.len() == mem {
                        s_hist.remove(0);
                        y_hist.remove(0);
                        rho_hist.remove(0);
                    }
                    s_hist.push(s_vec);
                    y_hist.push(y_vec);
                    rho_hist.push(1.0 / sy);
                }

                g_cur = g_next;
                cur_fmax = cur_eval.max_force_norm(Some(&frozen_mask));
                traj_e.push(cur_eval.energy_ev);
                traj_f.push(cur_fmax);
                steps_taken = step;

                let converged = cur_fmax <= opts.fmax_ev_a;
                if opts.stream_every > 0 && (step % opts.stream_every == 0 || converged || step == opts.max_steps) {
                    on_frame(&RelaxFrame {
                        step,
                        energy_ev: cur_eval.energy_ev,
                        fmax_ev_a: cur_fmax,
                        frms_ev_a: cur_eval.rms_force(Some(&frozen_mask)),
                        coords_a: cur_sys.coords_a.clone(),
                        forces_ev_a: cur_eval.forces_ev_a.clone(),
                    });
                }
                if converged {
                    break;
                }
            }
        }
    }

    if !any_frozen && opts.zero_com && cur_sys.pbc_cell.is_none() {
        cur_sys.zero_centroid();
    }

    if need_final_descriptor && cur_eval.descriptor.is_none() {
        let desc_ev = backend.eval_async(&cur_sys, false, true).await?;
        cur_eval.descriptor = desc_ev.descriptor;
        cur_eval.desc_dim = desc_ev.desc_dim;
        force_evals += 1;
    }

    Ok(RelaxResult {
        converged: cur_fmax <= opts.fmax_ev_a,
        steps_taken,
        force_evals,
        initial_energy_ev,
        final_energy_ev: cur_eval.energy_ev,
        delta_energy_ev: cur_eval.energy_ev - initial_energy_ev,
        initial_fmax_ev_a,
        final_fmax_ev_a: cur_fmax,
        heavy_rmsd_a: initial_system.heavy_atom_rmsd(&cur_sys),
        all_rmsd_a: initial_system.all_atom_rmsd(&cur_sys),
        final_system: cur_sys,
        final_forces_ev_a: cur_eval.forces_ev_a,
        final_descriptor: cur_eval.descriptor,
        desc_dim: cur_eval.desc_dim,
        trajectory_energies_ev: traj_e,
        trajectory_fmax_ev_a: traj_f,
    })
}
