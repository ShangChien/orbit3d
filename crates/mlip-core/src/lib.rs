//! `mlip-core`: Zero-overhead shared physics abstractions, Verlet neighbor lists,
//! `FIRE`/`L-BFGS` 3D geometry optimizers, `NVE`/`BAOAB-NVT` molecular dynamics integrators,
//! and 3D distance-biased equivariant property adapters for `dpa4-rs` and `hyff-rs`.

pub mod adapter;
pub mod bundle;
pub mod elements;
pub mod md;
pub mod neighbor;
pub mod relax;
pub mod types;

pub use adapter::{
    AdapterHeadWeights, BootstrapPropertyOracle, EquivariantAdapterMember, LayerNormWeights,
    LinearWeights, PropertyPrediction, TargetScaler,
};
pub use bundle::{BundleArchive, BundleDtype, TensorMeta};
pub use elements::{
    covalent_radius_a_from_z, mass_amu_from_z, symbol_from_z, z_from_symbol, ATOMIC_MASSES_AMU,
    COVALENT_RADII_A, ELEMENT_SYMBOLS,
};
pub use md::{run_md_async, run_md_sync, MdEnsemble, MdFrame, MdOptions, MdResult};
pub use neighbor::{DirectedEdge, NeighborCache, NeighborList};
pub use relax::{
    relax_async, relax_sync, RelaxAlgorithm, RelaxFrame, RelaxOptions, RelaxResult,
};
pub use types::{
    AsyncPotentialBackend, MolecularSystem, PotentialBackend, PotentialEval, SyncAsAsync,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytical pairwise Morse potential `V(r) = D_e * (1 - exp(-a * (r - r0)))^2`
    /// for rigorous testing of `FIRE`, `L-BFGS`, `freeze_heavy`, `NVE` energy conservation,
    /// `BAOAB NVT` thermalization, and `NeighborCache` Verlet skin reuse.
    struct MorseTestBackend {
        de: f64,
        alpha: f64,
        r0_ch: f64,
        r0_cc: f64,
        cutoff: f64,
    }

    impl Default for MorseTestBackend {
        fn default() -> Self {
            Self {
                de: 4.2,
                alpha: 1.8,
                r0_ch: 1.09,
                r0_cc: 1.52,
                cutoff: 6.0,
            }
        }
    }

    impl PotentialBackend for MorseTestBackend {
        fn model_name(&self) -> &str {
            "morse-test"
        }

        fn cutoff_a(&self) -> f64 {
            self.cutoff
        }

        fn eval(
            &mut self,
            sys: &MolecularSystem,
            need_forces: bool,
            need_descriptor: bool,
        ) -> Result<PotentialEval, String> {
            let n = sys.n_atoms();
            let nl = NeighborList::build(&sys.coords_a, sys.pbc_cell.as_ref(), self.cutoff);
            let mut energy = 0.0f64;
            let mut forces = vec![[0.0f64; 3]; n];

            for e in &nl.edges {
                let i = e.src as usize;
                let j = e.dst as usize;
                if i >= j {
                    continue;
                }
                let zi = sys.atomic_numbers[i];
                let zj = sys.atomic_numbers[j];
                let r0 = if zi > 1 && zj > 1 {
                    self.r0_cc
                } else if zi == 1 && zj == 1 {
                    1.78
                } else {
                    self.r0_ch
                };
                let r = e.dist_a;
                let exp_term = (-self.alpha * (r - r0)).exp();
                let one_minus = 1.0 - exp_term;
                energy += self.de * one_minus * one_minus;

                if need_forces {
                    // dV/dr = 2 * De * (1 - exp) * alpha * exp
                    let dv_dr = 2.0 * self.de * one_minus * self.alpha * exp_term;
                    let scale = dv_dr / r;
                    // vec_a = r_j - r_i => dV/dr_i = -dv_dr * (vec_a / r) => F_i = -dV/dr_i = +scale * vec_a
                    let fx = scale * e.vec_a[0];
                    let fy = scale * e.vec_a[1];
                    let fz = scale * e.vec_a[2];
                    forces[i][0] += fx;
                    forces[i][1] += fy;
                    forces[i][2] += fz;
                    forces[j][0] -= fx;
                    forces[j][1] -= fy;
                    forces[j][2] -= fz;
                }
            }

            let descriptor = if need_descriptor {
                Some(vec![0.25f32; n * 8])
            } else {
                None
            };

            Ok(PotentialEval {
                energy_ev: energy,
                forces_ev_a: if need_forces { forces } else { Vec::new() },
                descriptor,
                desc_dim: if need_descriptor { 8 } else { 0 },
            })
        }
    }

    fn distorted_ch4() -> MolecularSystem {
        MolecularSystem::new(
            vec![6, 1, 1, 1, 1],
            vec![
                [0.05, -0.02, 0.03],
                [0.78, 0.61, 0.59],
                [-0.71, -0.68, 0.62],
                [-0.64, 0.72, -0.58],
                [0.69, -0.59, -0.74],
            ],
            0.0,
            1,
        )
        .unwrap()
    }

    #[test]
    fn test_lbfgs_and_fire_relaxation_and_freeze_heavy() {
        let mut backend = MorseTestBackend::default();
        let sys = distorted_ch4();

        // 1. Full L-BFGS relaxation
        let opts_lbfgs = RelaxOptions {
            algorithm: RelaxAlgorithm::Lbfgs,
            max_steps: 80,
            fmax_ev_a: 1e-3,
            ..Default::default()
        };
        let res_lbfgs = relax_sync(&mut backend, &sys, &opts_lbfgs, true, |_| {}).unwrap();
        assert!(res_lbfgs.converged, "L-BFGS failed to converge: fmax={}", res_lbfgs.final_fmax_ev_a);
        assert!(res_lbfgs.final_energy_ev < res_lbfgs.initial_energy_ev - 0.1);
        assert!(res_lbfgs.final_descriptor.is_some());

        // 2. Full FIRE relaxation
        let opts_fire = RelaxOptions {
            algorithm: RelaxAlgorithm::Fire,
            max_steps: 200,
            fmax_ev_a: 5e-3,
            ..Default::default()
        };
        let res_fire = relax_sync(&mut backend, &sys, &opts_fire, false, |_| {}).unwrap();
        assert!(res_fire.converged, "FIRE failed to converge: fmax={}", res_fire.final_fmax_ev_a);
        assert!((res_fire.final_energy_ev - res_lbfgs.final_energy_ev).abs() < 1e-3);

        // 3. Frozen-heavy-skeleton H-only relaxation (`RMSD_heavy == 0.0000 Å`!)
        let opts_h_only = RelaxOptions {
            algorithm: RelaxAlgorithm::Lbfgs,
            max_steps: 80,
            fmax_ev_a: 1e-3,
            freeze_heavy: true,
            ..Default::default()
        };
        let res_h = relax_sync(&mut backend, &sys, &opts_h_only, false, |_| {}).unwrap();
        assert!(res_h.converged);
        assert_eq!(res_h.heavy_rmsd_a, 0.0, "Heavy atom moved under freeze_heavy=true!");
        assert!(res_h.all_rmsd_a > 0.01);
    }

    #[test]
    fn test_nve_energy_conservation_and_nvt_baoab() {
        let mut backend = MorseTestBackend::default();
        let relaxed = relax_sync(
            &mut backend,
            &distorted_ch4(),
            &RelaxOptions {
                fmax_ev_a: 1e-4,
                ..Default::default()
            },
            false,
            |_| {},
        )
        .unwrap()
        .final_system;

        // Symplectic NVE Velocity Verlet at 300K, dt = 0.25 fs
        let nve_res = run_md_sync(
            &mut backend,
            &relaxed,
            &MdOptions {
                ensemble: MdEnsemble::NveVelocityVerlet,
                n_steps: 200,
                dt_fs: 0.25,
                temperature_k: 300.0,
                seed: 12345,
                max_force_clip_ev_a: None,
                stream_every: 0,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        assert!(
            nve_res.total_energy_drift_ev.abs() < 1e-3,
            "NVE symplectic oscillation at dt=0.25fs too large: {} eV",
            nve_res.total_energy_drift_ev
        );
        // Verify 2nd-order O(dt^2) symplectic scaling at 5x finer dt = 0.05 fs (~25x smaller oscillation)
        let nve_fine = run_md_sync(
            &mut backend,
            &relaxed,
            &MdOptions {
                ensemble: MdEnsemble::NveVelocityVerlet,
                n_steps: 200,
                dt_fs: 0.05,
                temperature_k: 300.0,
                seed: 12345,
                max_force_clip_ev_a: None,
                stream_every: 0,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        assert!(
            nve_fine.total_energy_drift_ev.abs() < 4e-5,
            "NVE O(dt^2) scaling failed at dt=0.05fs: {} eV",
            nve_fine.total_energy_drift_ev
        );

        // BAOAB Langevin NVT at 300K
        let nvt_res = run_md_sync(
            &mut backend,
            &relaxed,
            &MdOptions {
                ensemble: MdEnsemble::NvtLangevinBaoab,
                n_steps: 200,
                dt_fs: 0.5,
                temperature_k: 300.0,
                friction_per_ps: 5.0,
                seed: 777,
                stream_every: 0,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        assert!(
            nvt_res.mean_temperature_k > 100.0 && nvt_res.mean_temperature_k < 600.0,
            "Unexpected NVT mean temperature: {} K",
            nvt_res.mean_temperature_k
        );
    }

    #[test]
    fn test_neighbor_cache_verlet_skin_reuse() {
        let mut sys = distorted_ch4();
        let mut cache = NeighborCache::new(4.5, 0.6);
        let nl0 = cache.update(&sys.coords_a, None);
        assert_eq!(cache.rebuild_count, 1);
        assert_eq!(cache.reuse_count, 0);

        // Small step < 0.3 Å -> reuses skin candidates!
        sys.coords_a[1][0] += 0.05;
        let nl1 = cache.update(&sys.coords_a, None);
        let nl_exact = NeighborList::build(&sys.coords_a, None, 4.5);
        assert_eq!(cache.rebuild_count, 1);
        assert_eq!(cache.reuse_count, 1);
        assert_eq!(nl1.edges.len(), nl_exact.edges.len());
        for (a, b) in nl1.edges.iter().zip(nl_exact.edges.iter()) {
            assert_eq!((a.src, a.dst), (b.src, b.dst));
            assert!((a.dist_a - b.dist_a).abs() < 1e-12);
        }
        assert_eq!(nl0.n_atoms, 5);
    }
}
