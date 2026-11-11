"""IP-tuned LC-wHPBE/def2-SVP evaluation following OCELOT's Jh protocol.

The H in Gaussian LC-wHPBE denotes the Henderson exchange hole, not 25%
short-range HF exchange. Libxc 486 has the required zero/100% SR/LR HF limits;
487 (LC-wPBEh) is a different short-range hybrid and must not be substituted.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import minimize_scalar

HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class OcelotTheory:
    xc: str = "HYB_GGA_XC_LC_WPBE_WHS"
    basis: str = "def2-svp"
    omega_bounds: tuple = (0.05, 0.5)
    omega_tolerance: float = 0.001
    omega_geometry_tolerance: float = 0.01
    max_outer_cycles: int = 5
    scf_tolerance: float = 1e-9
    max_scf_cycles: int = 100
    radial_grid: int = 99
    angular_grid: int = 590
    threads: int = 4
    memory_mb: int = 4000
    max_geometry_steps: int = 80


def ip_objective(neutral_energy, cation_energy, neutral_homo):
    """Jh in Hartree: abs(epsilon_HOMO + E(N-1) - E(N))."""
    return abs(neutral_homo + cation_energy - neutral_energy)


def frontier_orbitals(energies, occupations):
    energies, occupations = np.asarray(energies), np.asarray(occupations)
    return float(energies[occupations > 0].max()), float(energies[occupations == 0].min())


class OcelotEvaluator:
    def __init__(self, theory, workdir):
        from pyscf import lib

        self.theory = theory
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        lib.num_threads(theory.threads)
        self.trace = []
        self.counter = 0

    def scf(self, atoms, coords, omega, charge=0, spin=0):
        from pyscf import gto, dft

        s = self.theory
        self.counter += 1
        mol = gto.M(
            atom=list(zip(atoms, np.asarray(coords).tolist())),
            unit="Angstrom",
            charge=charge,
            spin=spin,
            basis=s.basis,
            verbose=0,
            max_memory=s.memory_mb,
        )
        # OCELOT's source uses uLC-wHPBE for neutral and charged species.
        mf = dft.UKS(mol)
        mf.xc = s.xc
        mf.omega = round(float(omega), 4)  # Gaussian IOp precision in original workflow.
        mf.grids.atom_grid = (s.radial_grid, s.angular_grid)
        mf.conv_tol = s.scf_tolerance
        mf.max_cycle = s.max_scf_cycles
        mf.chkfile = str(self.workdir / f"scf_{self.counter:04d}.chk")
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"SCF did not converge at omega={mf.omega}, charge={charge}")
        return mf

    def tune(self, atoms, coords, bounds):
        cache = {}

        def objective(omega):
            omega = round(float(omega), 4)
            if omega not in cache:
                n = self.scf(atoms, coords, omega)
                c = self.scf(atoms, coords, omega, charge=1, spin=1)
                homo, lumo = frontier_orbitals(n.mo_energy, n.mo_occ)
                j = ip_objective(n.e_tot, c.e_tot, homo)
                row = {
                    "omega": omega,
                    "neutral_E_Ha": float(n.e_tot),
                    "cation_E_Ha": float(c.e_tot),
                    "homo_eV": homo * HARTREE_EV,
                    "lumo_eV": lumo * HARTREE_EV,
                    "Jh_eV": j * HARTREE_EV,
                    "cation_S2": float(c.spin_square()[0]),
                }
                cache[omega] = j
                self.trace.append(row)
                with (self.workdir / "tuning.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print("omega", omega, "Jh_eV", row["Jh_eV"], flush=True)
            return cache[omega]

        result = minimize_scalar(
            objective,
            bounds=bounds,
            method="bounded",
            options={"xatol": self.theory.omega_tolerance, "maxiter": 40},
        )
        if not result.success:
            raise RuntimeError("IP tuning did not converge")
        return round(float(result.x), 4)

    def evaluate(self, molecule, optimize_geometry=True, excitations=False):
        import pyscf
        from pyscf.geomopt.geometric_solver import kernel
        from rdkit import Chem
        from ..generator.octree_tokenizer import rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol
        from ..utils.dataset_io import write_precise_sdf

        started = time.perf_counter()
        if Chem.GetFormalCharge(molecule) or any(a.GetNumRadicalElectrons() for a in molecule.GetAtoms()):
            raise ValueError("This OCELOT validation protocol requires a neutral closed-shell input")
        atoms = [a.GetSymbol() for a in molecule.GetAtoms()]
        original = molecule.GetConformer().GetPositions()
        coords = original.copy()
        omega = 0.3
        outer = []
        converged = not optimize_geometry
        for cycle in range(self.theory.max_outer_cycles if optimize_geometry else 1):
            bounds = self.theory.omega_bounds if cycle == 0 else (max(0.001, omega - 0.2), omega + 0.2)
            previous = omega
            omega = self.tune(atoms, coords, bounds)
            mf = self.scf(atoms, coords, omega)
            if optimize_geometry:
                geo_ok, optimized = kernel(
                    mf,
                    maxsteps=self.theory.max_geometry_steps,
                    convergence_energy=1e-6,
                    convergence_grms=3e-4,
                    convergence_gmax=4.5e-4,
                    convergence_drms=1.2e-3,
                    convergence_dmax=1.8e-3,
                )
                if not geo_ok:
                    raise RuntimeError("Geometry optimization did not converge")
                coords = optimized.atom_coords(unit="Angstrom")
            outer.append({"cycle": cycle, "omega": omega, "omega_change": abs(omega - previous)})
            if not optimize_geometry or (
                cycle > 0 and abs(omega - previous) <= self.theory.omega_geometry_tolerance
            ):
                converged = True
                break
        if not converged:
            raise RuntimeError("Geometry/omega outer loop did not converge")
        final = self.scf(atoms, coords, omega)
        homo, lumo = frontier_orbitals(final.mo_energy, final.mo_occ)
        forces = -final.nuc_grad_method().kernel() * HARTREE_EV / BOHR_ANGSTROM
        result = {
            "theory": asdict(self.theory),
            "pyscf_version": pyscf.__version__,
            "omega": omega,
            "omega_geometry_cycles": outer,
            "geometry_optimized": optimize_geometry,
            "energy_Ha": float(final.e_tot),
            "homo_eV": homo * HARTREE_EV,
            "lumo_eV": lumo * HARTREE_EV,
            "S2": float(final.spin_square()[0]),
            "rms_force_eV_A": float(np.sqrt(np.mean(np.sum(forces**2, axis=-1)))),
            "max_force_eV_A": float(np.linalg.norm(forces, axis=-1).max()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if excitations:
            td = final.TDDFT()
            td.nstates = 3
            energies, _ = td.kernel()
            result["linear_response_excitation_eV"] = (np.asarray(energies) * HARTREE_EV).tolist()
        out = Chem.Mol(molecule)
        for i, xyz in enumerate(coords):
            out.GetConformer().SetAtomPosition(i, xyz.tolist())
        inst = rdkit_mol_to_mol3d_instance(out, np.array([homo, lumo]) * HARTREE_EV)
        mol3d_to_validated_rdkit_mol(inst)
        # Check recovered graph as well as the supplied graph after optimization.
        from dataclasses import replace

        recovered = mol3d_to_validated_rdkit_mol(replace(inst, graph=None))
        result["graph_preserved"] = Chem.MolToSmiles(
            Chem.RemoveHs(recovered), isomericSmiles=True
        ) == Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True)
        write_precise_sdf(self.workdir / "optimized.sdf", [out])
        (self.workdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
