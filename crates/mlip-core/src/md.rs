//! Monomorphized 3D Molecular Dynamics (`NVE Velocity Verlet` & `NVT BAOAB Langevin`)
//! in physical units (`Å`, `fs`, `amu`, `eV`, `K`).

use serde::{Deserialize, Serialize};

use crate::elements::mass_amu_from_z;
use crate::types::{
    block_on_ready, AsyncPotentialBackend, MolecularSystem, PotentialBackend, SyncAsAsync,
};

/// CODATA conversion `eV / (amu * Å² / fs²)`:
/// `a [Å/fs²] = (F [eV/Å] / m [amu]) * EV_PER_A_AMU_TO_A_PER_FS2`.
pub const EV_PER_A_AMU_TO_A_PER_FS2: f64 = 0.009648533212331002;

/// Inverse conversion `(amu * Å² / fs²) -> eV`:
/// `E_kin [eV] = 0.5 * AMU_A2_FS2_TO_EV * sum_i m_i * ||v_i||²`.
pub const AMU_A2_FS2_TO_EV: f64 = 103.64269656262174;

/// Boltzmann constant `k_B` in `eV / K`.
pub const KB_EV_PER_K: f64 = 8.617333262145e-5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MdEnsemble {
    NveVelocityVerlet,
    NvtLangevinBaoab,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MdOptions {
    pub ensemble: MdEnsemble,
    /// Number of MD integration steps.
    pub n_steps: usize,
    /// Integration timestep `dt` in femtoseconds (`fs`, e.g. `0.5` fs for organic H-containing molecules).
    pub dt_fs: f64,
    /// Target bath temperature `T` in Kelvin (`K`, default `300.0`).
    pub temperature_k: f64,
    /// Langevin friction coefficient `gamma` in `ps⁻¹` (default `1.0 ps⁻¹` = `0.001 fs⁻¹`).
    pub friction_per_ps: f64,
    /// Initialize velocities from Maxwell-Boltzmann distribution at `temperature_k` (default `true`).
    pub init_velocities: bool,
    /// Remove net COM linear & angular momentum for isolated molecules (default `true`).
    pub remove_com_motion: bool,
    /// Deterministic RNG seed for reproducible thermal trajectories across platforms.
    pub seed: u64,
    /// Optional force norm clip (`eV/Å`) for numerical safety on highly strained initial states.
    pub max_force_clip_ev_a: Option<f64>,
    /// Emit live trajectory frame every `stream_every` steps.
    pub stream_every: usize,
}

impl Default for MdOptions {
    fn default() -> Self {
        Self {
            ensemble: MdEnsemble::NvtLangevinBaoab,
            n_steps: 100,
            dt_fs: 0.5,
            temperature_k: 300.0,
            friction_per_ps: 2.0,
            init_velocities: true,
            remove_com_motion: true,
            seed: 42,
            max_force_clip_ev_a: Some(50.0),
            stream_every: 2,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MdFrame {
    pub step: usize,
    pub time_fs: f64,
    pub potential_energy_ev: f64,
    pub kinetic_energy_ev: f64,
    pub total_energy_ev: f64,
    pub temperature_k: f64,
    pub fmax_ev_a: f64,
    pub coords_a: Vec<[f64; 3]>,
    pub forces_ev_a: Vec<[f64; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MdResult {
    pub steps_taken: usize,
    pub time_fs: f64,
    pub initial_potential_ev: f64,
    pub final_potential_ev: f64,
    pub final_kinetic_ev: f64,
    pub final_temperature_k: f64,
    pub mean_temperature_k: f64,
    pub total_energy_drift_ev: f64,
    pub final_system: MolecularSystem,
    pub final_velocities_a_fs: Vec<[f64; 3]>,
    pub final_forces_ev_a: Vec<[f64; 3]>,
    pub potential_trace_ev: Vec<f64>,
    pub kinetic_trace_ev: Vec<f64>,
    pub temperature_trace_k: Vec<f64>,
}

/// Deterministic portable SplitMix64 + Box-Muller Gaussian RNG (identical across macOS/Linux/WASM).
struct GaussianRng {
    state: u64,
    spare: Option<f64>,
}

impl GaussianRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
            spare: None,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64_open(&mut self) -> f64 {
        // Uniform in (0, 1) strictly open
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        u.clamp(1e-15, 1.0 - 1e-15)
    }

    fn next_normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = self.next_f64_open();
        let u2 = self.next_f64_open();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

fn kinetic_and_temp(masses_amu: &[f64], vel_a_fs: &[[f64; 3]], remove_com: bool) -> (f64, f64) {
    let n = masses_amu.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut mv2 = 0.0f64;
    for (m, v) in masses_amu.iter().zip(vel_a_fs.iter()) {
        mv2 += m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    let e_kin_ev = 0.5 * AMU_A2_FS2_TO_EV * mv2;
    let dof = if remove_com && n > 2 {
        (3 * n - 6) as f64
    } else if remove_com && n == 2 {
        (3 * n - 5) as f64
    } else {
        (3 * n) as f64
    };
    let temp_k = (2.0 * e_kin_ev) / (dof.max(1.0) * KB_EV_PER_K);
    (e_kin_ev, temp_k)
}

/// Remove center-of-mass linear velocity and rigid-body angular velocity (`L = I * omega = 0`).
fn remove_com_linear_and_angular(
    masses_amu: &[f64],
    coords_a: &[[f64; 3]],
    vel_a_fs: &mut [[f64; 3]],
) {
    let n = masses_amu.len();
    if n == 0 {
        return;
    }
    let mut m_tot = 0.0f64;
    let mut r_com = [0.0f64; 3];
    let mut p_tot = [0.0f64; 3];
    for i in 0..n {
        let m = masses_amu[i];
        m_tot += m;
        r_com[0] += m * coords_a[i][0];
        r_com[1] += m * coords_a[i][1];
        r_com[2] += m * coords_a[i][2];
        p_tot[0] += m * vel_a_fs[i][0];
        p_tot[1] += m * vel_a_fs[i][1];
        p_tot[2] += m * vel_a_fs[i][2];
    }
    if m_tot <= 0.0 {
        return;
    }
    let inv_m = 1.0 / m_tot;
    r_com[0] *= inv_m;
    r_com[1] *= inv_m;
    r_com[2] *= inv_m;
    let v_com = [p_tot[0] * inv_m, p_tot[1] * inv_m, p_tot[2] * inv_m];
    for v in vel_a_fs.iter_mut() {
        v[0] -= v_com[0];
        v[1] -= v_com[1];
        v[2] -= v_com[2];
    }

    if n < 2 {
        return;
    }

    // Angular momentum L = sum_i m_i (dr_i x v_i) and inertia tensor I_ab
    let mut l_vec = [0.0f64; 3];
    let mut inertia = [[0.0f64; 3]; 3];
    for i in 0..n {
        let m = masses_amu[i];
        let dx = coords_a[i][0] - r_com[0];
        let dy = coords_a[i][1] - r_com[1];
        let dz = coords_a[i][2] - r_com[2];
        let vx = vel_a_fs[i][0];
        let vy = vel_a_fs[i][1];
        let vz = vel_a_fs[i][2];

        l_vec[0] += m * (dy * vz - dz * vy);
        l_vec[1] += m * (dz * vx - dx * vz);
        l_vec[2] += m * (dx * vy - dy * vx);

        let r2 = dx * dx + dy * dy + dz * dz;
        inertia[0][0] += m * (r2 - dx * dx);
        inertia[1][1] += m * (r2 - dy * dy);
        inertia[2][2] += m * (r2 - dz * dz);
        inertia[0][1] -= m * dx * dy;
        inertia[0][2] -= m * dx * dz;
        inertia[1][2] -= m * dy * dz;
    }
    inertia[1][0] = inertia[0][1];
    inertia[2][0] = inertia[0][2];
    inertia[2][1] = inertia[1][2];

    if let Some(omega) = solve_3x3_symmetric(&inertia, &l_vec) {
        for i in 0..n {
            let dx = coords_a[i][0] - r_com[0];
            let dy = coords_a[i][1] - r_com[1];
            let dz = coords_a[i][2] - r_com[2];
            // v_rot = omega x dr
            vel_a_fs[i][0] -= omega[1] * dz - omega[2] * dy;
            vel_a_fs[i][1] -= omega[2] * dx - omega[0] * dz;
            vel_a_fs[i][2] -= omega[0] * dy - omega[1] * dx;
        }
    }
}

fn solve_3x3_symmetric(m: &[[f64; 3]; 3], rhs: &[f64; 3]) -> Option<[f64; 3]> {
    // Small Tikhonov regularization handles collinear molecules cleanly without blow-up.
    let trace = m[0][0] + m[1][1] + m[2][2];
    if trace < 1e-10 {
        return None;
    }
    let reg = 1e-9 * trace;
    let a = [
        [m[0][0] + reg, m[0][1], m[0][2]],
        [m[1][0], m[1][1] + reg, m[1][2]],
        [m[2][0], m[2][1], m[2][2] + reg],
    ];
    let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let c01 = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]);
    let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    if det.abs() < 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    let c11 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    let c12 = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]);
    let c22 = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    let inv = [
        [c00 * inv_det, c01 * inv_det, c02 * inv_det],
        [c01 * inv_det, c11 * inv_det, c12 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ];
    Some([
        inv[0][0] * rhs[0] + inv[0][1] * rhs[1] + inv[0][2] * rhs[2],
        inv[1][0] * rhs[0] + inv[1][1] * rhs[1] + inv[1][2] * rhs[2],
        inv[2][0] * rhs[0] + inv[2][1] * rhs[1] + inv[2][2] * rhs[2],
    ])
}

/// Synchronous 3D Molecular Dynamics runner over any [`PotentialBackend`].
pub fn run_md_sync<B, F>(
    backend: &mut B,
    initial_system: &MolecularSystem,
    opts: &MdOptions,
    on_frame: F,
) -> Result<MdResult, String>
where
    B: PotentialBackend,
    F: FnMut(&MdFrame),
{
    let mut adapter = SyncAsAsync(backend);
    block_on_ready(run_md_async(&mut adapter, initial_system, opts, on_frame))
}

/// Asynchronous 3D Molecular Dynamics runner (`NVE Velocity Verlet` & `NVT BAOAB Langevin`).
pub async fn run_md_async<B, F>(
    backend: &mut B,
    initial_system: &MolecularSystem,
    opts: &MdOptions,
    mut on_frame: F,
) -> Result<MdResult, String>
where
    B: AsyncPotentialBackend,
    F: FnMut(&MdFrame),
{
    let n = initial_system.n_atoms();
    let mut sys = initial_system.clone();
    let remove_com = opts.remove_com_motion && sys.pbc_cell.is_none();
    let masses: Vec<f64> = sys
        .atomic_numbers
        .iter()
        .map(|&z| mass_amu_from_z(z))
        .collect();

    let mut rng = GaussianRng::new(opts.seed);
    let mut vel = vec![[0.0f64; 3]; n];

    if opts.init_velocities && opts.temperature_k > 0.0 && n > 0 {
        for i in 0..n {
            let sigma =
                ((KB_EV_PER_K * opts.temperature_k * EV_PER_A_AMU_TO_A_PER_FS2) / masses[i]).sqrt();
            vel[i] = [
                sigma * rng.next_normal(),
                sigma * rng.next_normal(),
                sigma * rng.next_normal(),
            ];
        }
        if remove_com {
            remove_com_linear_and_angular(&masses, &sys.coords_a, &mut vel);
        }
        // Rescale to exact target temperature after COM projection
        let (_, cur_t) = kinetic_and_temp(&masses, &vel, remove_com);
        if cur_t > 1e-8 {
            let scale = (opts.temperature_k / cur_t).sqrt();
            for v in &mut vel {
                v[0] *= scale;
                v[1] *= scale;
                v[2] *= scale;
            }
        }
    }

    let mut cur_eval = backend.eval_async(&sys, true, false).await?;
    if let Some(clip) = opts.max_force_clip_ev_a {
        cur_eval.clip_forces(clip);
    }
    if remove_com {
        cur_eval.zero_com_forces(None);
    }

    let (init_kin, init_temp) = kinetic_and_temp(&masses, &vel, remove_com);
    let init_pot = cur_eval.energy_ev;
    let init_tot = init_pot + init_kin;

    let mut pot_trace = Vec::with_capacity(opts.n_steps + 1);
    let mut kin_trace = Vec::with_capacity(opts.n_steps + 1);
    let mut temp_trace = Vec::with_capacity(opts.n_steps + 1);
    pot_trace.push(init_pot);
    kin_trace.push(init_kin);
    temp_trace.push(init_temp);

    if opts.stream_every > 0 {
        on_frame(&MdFrame {
            step: 0,
            time_fs: 0.0,
            potential_energy_ev: init_pot,
            kinetic_energy_ev: init_kin,
            total_energy_ev: init_tot,
            temperature_k: init_temp,
            fmax_ev_a: cur_eval.max_force_norm(None),
            coords_a: sys.coords_a.clone(),
            forces_ev_a: cur_eval.forces_ev_a.clone(),
        });
    }

    let dt = opts.dt_fs;
    let half_dt = 0.5 * dt;
    let gamma_fs = (opts.friction_per_ps.max(0.0)) * 1e-3;
    let c1_baoab = (-gamma_fs * dt).exp();
    let c2_baoab = (1.0 - c1_baoab * c1_baoab).max(0.0).sqrt();

    for step in 1..=opts.n_steps {
        match opts.ensemble {
            MdEnsemble::NveVelocityVerlet => {
                // Half-kick B + full drift A
                for i in 0..n {
                    let inv_m_acc = EV_PER_A_AMU_TO_A_PER_FS2 / masses[i];
                    let f = cur_eval.forces_ev_a[i];
                    vel[i][0] += half_dt * f[0] * inv_m_acc;
                    vel[i][1] += half_dt * f[1] * inv_m_acc;
                    vel[i][2] += half_dt * f[2] * inv_m_acc;

                    sys.coords_a[i][0] += dt * vel[i][0];
                    sys.coords_a[i][1] += dt * vel[i][1];
                    sys.coords_a[i][2] += dt * vel[i][2];
                }

                cur_eval = backend.eval_async(&sys, true, false).await?;
                if let Some(clip) = opts.max_force_clip_ev_a {
                    cur_eval.clip_forces(clip);
                }
                if remove_com {
                    cur_eval.zero_com_forces(None);
                }

                // Second half-kick B
                for i in 0..n {
                    let inv_m_acc = EV_PER_A_AMU_TO_A_PER_FS2 / masses[i];
                    let f = cur_eval.forces_ev_a[i];
                    vel[i][0] += half_dt * f[0] * inv_m_acc;
                    vel[i][1] += half_dt * f[1] * inv_m_acc;
                    vel[i][2] += half_dt * f[2] * inv_m_acc;
                }
            }
            MdEnsemble::NvtLangevinBaoab => {
                // B: half-step velocity kick
                // A: half-step coordinate drift
                // O: exact Ornstein-Uhlenbeck thermostat update
                // A: second half-step coordinate drift
                for i in 0..n {
                    let inv_m_acc = EV_PER_A_AMU_TO_A_PER_FS2 / masses[i];
                    let f = cur_eval.forces_ev_a[i];
                    vel[i][0] += half_dt * f[0] * inv_m_acc;
                    vel[i][1] += half_dt * f[1] * inv_m_acc;
                    vel[i][2] += half_dt * f[2] * inv_m_acc;

                    sys.coords_a[i][0] += half_dt * vel[i][0];
                    sys.coords_a[i][1] += half_dt * vel[i][1];
                    sys.coords_a[i][2] += half_dt * vel[i][2];

                    let sigma = ((KB_EV_PER_K * opts.temperature_k * EV_PER_A_AMU_TO_A_PER_FS2)
                        / masses[i])
                        .sqrt();
                    vel[i][0] = c1_baoab * vel[i][0] + c2_baoab * sigma * rng.next_normal();
                    vel[i][1] = c1_baoab * vel[i][1] + c2_baoab * sigma * rng.next_normal();
                    vel[i][2] = c1_baoab * vel[i][2] + c2_baoab * sigma * rng.next_normal();

                    sys.coords_a[i][0] += half_dt * vel[i][0];
                    sys.coords_a[i][1] += half_dt * vel[i][1];
                    sys.coords_a[i][2] += half_dt * vel[i][2];
                }

                cur_eval = backend.eval_async(&sys, true, false).await?;
                if let Some(clip) = opts.max_force_clip_ev_a {
                    cur_eval.clip_forces(clip);
                }
                if remove_com {
                    cur_eval.zero_com_forces(None);
                }

                // Final B: half-step velocity kick
                for i in 0..n {
                    let inv_m_acc = EV_PER_A_AMU_TO_A_PER_FS2 / masses[i];
                    let f = cur_eval.forces_ev_a[i];
                    vel[i][0] += half_dt * f[0] * inv_m_acc;
                    vel[i][1] += half_dt * f[1] * inv_m_acc;
                    vel[i][2] += half_dt * f[2] * inv_m_acc;
                }

                if remove_com && step % 5 == 0 {
                    remove_com_linear_and_angular(&masses, &sys.coords_a, &mut vel);
                    sys.zero_centroid();
                }
            }
        }

        let (kin_ev, temp_k) = kinetic_and_temp(&masses, &vel, remove_com);
        let pot_ev = cur_eval.energy_ev;
        pot_trace.push(pot_ev);
        kin_trace.push(kin_ev);
        temp_trace.push(temp_k);

        if opts.stream_every > 0 && (step % opts.stream_every == 0 || step == opts.n_steps) {
            on_frame(&MdFrame {
                step,
                time_fs: (step as f64) * dt,
                potential_energy_ev: pot_ev,
                kinetic_energy_ev: kin_ev,
                total_energy_ev: pot_ev + kin_ev,
                temperature_k: temp_k,
                fmax_ev_a: cur_eval.max_force_norm(None),
                coords_a: sys.coords_a.clone(),
                forces_ev_a: cur_eval.forces_ev_a.clone(),
            });
        }
    }

    let final_pot = *pot_trace.last().unwrap_or(&init_pot);
    let final_kin = *kin_trace.last().unwrap_or(&init_kin);
    let final_temp = *temp_trace.last().unwrap_or(&init_temp);
    let mean_temp = if temp_trace.len() > 1 {
        temp_trace[1..].iter().sum::<f64>() / ((temp_trace.len() - 1) as f64)
    } else {
        final_temp
    };

    Ok(MdResult {
        steps_taken: opts.n_steps,
        time_fs: (opts.n_steps as f64) * dt,
        initial_potential_ev: init_pot,
        final_potential_ev: final_pot,
        final_kinetic_ev: final_kin,
        final_temperature_k: final_temp,
        mean_temperature_k: mean_temp,
        total_energy_drift_ev: (final_pot + final_kin) - init_tot,
        final_system: sys,
        final_velocities_a_fs: vel,
        final_forces_ev_a: cur_eval.forces_ev_a,
        potential_trace_ev: pot_trace,
        kinetic_trace_ev: kin_trace,
        temperature_trace_k: temp_trace,
    })
}
