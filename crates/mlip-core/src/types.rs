//! Core data structures and zero-overhead static generic traits (`PotentialBackend` &
//! `AsyncPotentialBackend`) shared across `dpa4-rs`, `hyff-rs`, optimizers, and MD integrators.

use std::future::Future;
use serde::{Deserialize, Serialize};

use crate::elements::{symbol_from_z, z_from_symbol};

/// Canonical 3D molecular or periodic atomic system in physical units (`Å`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MolecularSystem {
    /// Atomic numbers `Z` (`1..=118`), length `N`.
    pub atomic_numbers: Vec<u8>,
    /// Cartesian atomic coordinates in `Å`, length `N`.
    pub coords_a: Vec<[f64; 3]>,
    /// Total formal molecular charge (default `0.0`).
    pub charge: f32,
    /// Spin multiplicity `2S + 1` (`1` = closed-shell singlet, `2` = doublet, etc.).
    pub multiplicity: u32,
    /// Optional periodic cell vectors `[a, b, c]` in `Å` (row-major lattice vectors).
    pub pbc_cell: Option<[[f64; 3]; 3]>,
}

impl MolecularSystem {
    pub fn new(
        atomic_numbers: Vec<u8>,
        coords_a: Vec<[f64; 3]>,
        charge: f32,
        multiplicity: u32,
    ) -> Result<Self, String> {
        if atomic_numbers.len() != coords_a.len() {
            return Err(format!(
                "atomic_numbers length ({}) != coords_a length ({})",
                atomic_numbers.len(),
                coords_a.len()
            ));
        }
        for &z in &atomic_numbers {
            if !(1..=118).contains(&z) {
                return Err(format!("Atomic number Z={z} out of supported range 1..=118"));
            }
        }
        Ok(Self {
            atomic_numbers,
            coords_a,
            charge,
            multiplicity: multiplicity.max(1),
            pbc_cell: None,
        })
    }

    pub fn from_symbols(
        symbols: &[String],
        coords_a: Vec<[f64; 3]>,
        charge: f32,
        multiplicity: u32,
    ) -> Result<Self, String> {
        let mut zs = Vec::with_capacity(symbols.len());
        for s in symbols {
            zs.push(z_from_symbol(s)?);
        }
        Self::new(zs, coords_a, charge, multiplicity)
    }

    #[inline]
    pub fn n_atoms(&self) -> usize {
        self.atomic_numbers.len()
    }

    pub fn symbols(&self) -> Vec<&'static str> {
        self.atomic_numbers
            .iter()
            .map(|&z| symbol_from_z(z).unwrap_or("X"))
            .collect()
    }

    /// Translate geometric centroid (or active subset) to origin `(0, 0, 0)`.
    pub fn zero_centroid(&mut self) {
        let n = self.coords_a.len();
        if n == 0 {
            return;
        }
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut cz = 0.0f64;
        for c in &self.coords_a {
            cx += c[0];
            cy += c[1];
            cz += c[2];
        }
        let inv = 1.0 / (n as f64);
        cx *= inv;
        cy *= inv;
        cz *= inv;
        for c in &mut self.coords_a {
            c[0] -= cx;
            c[1] -= cy;
            c[2] -= cz;
        }
    }

    /// Compute heavy-atom (`Z > 1`) coordinate RMSD (`Å`) without alignment (direct frame RMSD).
    pub fn heavy_atom_rmsd(&self, other: &MolecularSystem) -> f64 {
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for ((&z, a), b) in self
            .atomic_numbers
            .iter()
            .zip(self.coords_a.iter())
            .zip(other.coords_a.iter())
        {
            if z > 1 {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                sum_sq += dx * dx + dy * dy + dz * dz;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum_sq / (count as f64)).sqrt()
        }
    }

    /// Compute all-atom coordinate RMSD (`Å`).
    pub fn all_atom_rmsd(&self, other: &MolecularSystem) -> f64 {
        let n = self.coords_a.len().min(other.coords_a.len());
        if n == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0f64;
        for (a, b) in self.coords_a.iter().zip(other.coords_a.iter()) {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            sum_sq += dx * dx + dy * dy + dz * dz;
        }
        (sum_sq / (n as f64)).sqrt()
    }
}

/// Unified physical evaluation output from any MLIP backend (`DPA-4` or `HyFF-L2`).
///
/// Units are strictly standardized:
/// - `energy_ev`: potential energy in `eV` (`f64` to preserve large atomic reference offsets)
/// - `forces_ev_a`: conservative atomic forces `F_i = -dE/dr_i` in `eV/Å`
/// - `descriptor`: optional per-atom invariant/equivariant readout features `(N, desc_dim)`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PotentialEval {
    pub energy_ev: f64,
    pub forces_ev_a: Vec<[f64; 3]>,
    pub descriptor: Option<Vec<f32>>,
    pub desc_dim: usize,
}

impl PotentialEval {
    /// Maximum per-atom Euclidean force norm `max_i ||F_i||_2` in `eV/Å` over active (non-frozen) atoms.
    pub fn max_force_norm(&self, frozen_mask: Option<&[bool]>) -> f64 {
        let mut fmax = 0.0f64;
        for (i, f) in self.forces_ev_a.iter().enumerate() {
            if frozen_mask.and_then(|m| m.get(i).copied()).unwrap_or(false) {
                continue;
            }
            let norm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            if norm > fmax {
                fmax = norm;
            }
        }
        fmax
    }

    /// Root-mean-square per-atom force norm `sqrt(1/N_free * sum_i ||F_i||^2)` in `eV/Å`.
    pub fn rms_force(&self, frozen_mask: Option<&[bool]>) -> f64 {
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for (i, f) in self.forces_ev_a.iter().enumerate() {
            if frozen_mask.and_then(|m| m.get(i).copied()).unwrap_or(false) {
                continue;
            }
            sum_sq += f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            (sum_sq / (count as f64)).sqrt()
        }
    }

    /// Remove net translational force drift (`sum_i F_i = 0`) across free atoms.
    pub fn zero_com_forces(&mut self, frozen_mask: Option<&[bool]>) {
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        let mut sz = 0.0f64;
        let mut n_free = 0usize;
        for (i, f) in self.forces_ev_a.iter().enumerate() {
            if frozen_mask.and_then(|m| m.get(i).copied()).unwrap_or(false) {
                continue;
            }
            sx += f[0];
            sy += f[1];
            sz += f[2];
            n_free += 1;
        }
        if n_free == 0 {
            return;
        }
        let inv = 1.0 / (n_free as f64);
        sx *= inv;
        sy *= inv;
        sz *= inv;
        for (i, f) in self.forces_ev_a.iter_mut().enumerate() {
            if frozen_mask.and_then(|m| m.get(i).copied()).unwrap_or(false) {
                f[0] = 0.0;
                f[1] = 0.0;
                f[2] = 0.0;
            } else {
                f[0] -= sx;
                f[1] -= sy;
                f[2] -= sz;
            }
        }
    }

    /// Clip individual atomic force magnitudes to `max_norm_ev_a` while preserving direction.
    pub fn clip_forces(&mut self, max_norm_ev_a: f64) {
        if max_norm_ev_a <= 0.0 || !max_norm_ev_a.is_finite() {
            return;
        }
        for f in &mut self.forces_ev_a {
            let norm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            if norm > max_norm_ev_a {
                let scale = max_norm_ev_a / norm;
                f[0] *= scale;
                f[1] *= scale;
                f[2] *= scale;
            }
        }
    }
}

/// Zero-overhead synchronous interatomic potential trait (monomorphized at compile time).
pub trait PotentialBackend {
    /// Model identifier (e.g. `"dpa4-mini"` or `"hyff-vivace"`).
    fn model_name(&self) -> &str;

    /// Maximum interaction cutoff radius in `Å`.
    fn cutoff_a(&self) -> f64;

    /// Evaluate potential energy (`eV`), optional conservative forces (`eV/Å`), and optional
    /// per-atom native equivariant descriptor `(N, desc_dim)`.
    fn eval(
        &mut self,
        system: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<PotentialEval, String>;
}

/// Zero-overhead asynchronous interatomic potential trait (monomorphized for both WebGPU WASM
/// async buffer readback and native GPU/CPU backends).
pub trait AsyncPotentialBackend {
    fn model_name(&self) -> &str;
    fn cutoff_a(&self) -> f64;

    fn eval_async(
        &mut self,
        system: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> impl Future<Output = Result<PotentialEval, String>>;
}

/// Zero-cost adapter lifting any synchronous [`PotentialBackend`] into an [`AsyncPotentialBackend`].
pub struct SyncAsAsync<'a, B: PotentialBackend>(pub &'a mut B);

impl<'a, B: PotentialBackend> AsyncPotentialBackend for SyncAsAsync<'a, B> {
    #[inline]
    fn model_name(&self) -> &str {
        self.0.model_name()
    }

    #[inline]
    fn cutoff_a(&self) -> f64 {
        self.0.cutoff_a()
    }

    #[inline]
    async fn eval_async(
        &mut self,
        system: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<PotentialEval, String> {
        self.0.eval(system, need_forces, need_descriptor)
    }
}

/// Block on an immediately-ready future generated by [`SyncAsAsync`] without pulling in an async runtime.
pub(crate) fn block_on_ready<F: Future>(fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn noop_clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    fn noop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut ctx = Context::from_waker(&waker);
    let mut pinned = Box::pin(fut);
    match Pin::as_mut(&mut pinned).poll(&mut ctx) {
        Poll::Ready(val) => val,
        Poll::Pending => unreachable!("SyncAsAsync future never yields Pending"),
    }
}
