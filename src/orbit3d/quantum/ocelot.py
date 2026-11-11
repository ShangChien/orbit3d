"""LC-wHPBE/def2-SVP diagnostic using the public OCELOT API's default Jh.

The production dataset's tuning objective remains unconfirmed. The Jh reference
cross-check disagrees with its CSV and must not be labelled reference validation.

The Gaussian name alone does not establish a Libxc mapping. Both 478 and 486
have zero/100% SR/LR HF limits but differ in their exchange holes. Public Gaussian
reference checks are required; the initial 486 mapping has not passed them.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class OcelotTheory:
    engine: str = "cpu"
    xc: str = "HYB_GGA_XC_LC_WPBE"
    basis: str = "def2-svp"
    tuning_objective: str = 'Jh'
    omega_bounds: tuple = (0.05, 0.5)
    omega_tolerance: float = 0.001
    omega_geometry_tolerance: float = 0.01
    max_outer_cycles: int = 5
    scf_tolerance: float = 1e-9
    max_scf_cycles: int = 200
    radial_grid: int = 99
    angular_grid: int = 590
    threads: int = 4
    memory_mb: int = 4000
    max_geometry_steps: int = 80
    ecp_policy: str = "matching_def2"


def theory_from_record(settings):
    """Older saved calculations omitted ECPs; never relabel their Hamiltonian."""
    return OcelotTheory(**{'ecp_policy': 'none', **settings})


def make_quantum_molecule(atoms, coords, theory, charge=0, spin=0):
    """Build the declared basis and its scalar def2 ECP as one Hamiltonian."""
    from pyscf import gto

    ecp = {}
    if theory.ecp_policy == 'matching_def2':
        if theory.basis.lower().replace('-', '').startswith('def2'):
            for symbol in sorted(set(atoms)):
                parameters = gto.basis.load_ecp('def2-svp', symbol)
                if parameters:
                    ecp[symbol] = parameters
    elif theory.ecp_policy != 'none':
        raise ValueError(f'Unknown ECP policy: {theory.ecp_policy}')
    return gto.M(atom=list(zip(atoms, np.asarray(coords).tolist())), unit='Angstrom',
                 charge=charge, spin=spin, basis=theory.basis, ecp=ecp,
                 verbose=0, max_memory=theory.memory_mb)


def ecp_core_electrons(molecule):
    return {molecule.atom_symbol(i): int(molecule.atom_nelec_core(i))
            for i in range(molecule.natm) if molecule.atom_nelec_core(i)}


def ip_objective(neutral_energy, cation_energy, neutral_homo):
    """Jh in Hartree: abs(epsilon_HOMO + E(N-1) - E(N))."""
    return abs(neutral_homo + cation_energy - neutral_energy)


def tuning_objective(scheme, en, ec, homo, lumo, ea=None, anion_homo=None):
    h = homo + ec - en
    if scheme == 'Jh':
        return abs(h)
    if scheme not in {'J2', 'Jn2'} or ea is None:
        raise ValueError('Supported OCELOT API objectives are Jh, J2 and Jn2; the latter require anion energy')
    second = anion_homo if scheme == 'J2' else lumo
    if second is None:
        raise ValueError('J2 requires the anion HOMO')
    return h**2 + (second + en - ea)**2


def frontier_orbitals(energies, occupations):
    energies, occupations = host_array(energies), host_array(occupations)
    return float(energies[occupations > 0].max()), float(energies[occupations == 0].min())


def host_array(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def record_geometry_step(workdir, cycle, omega, environment):
    """Preserve every evaluated geometry, including steps from a failed optimization."""
    coords = host_array(environment['coords']).reshape(-1, 3) * BOHR_ANGSTROM
    gradients = host_array(environment['gradients']).reshape(-1, 3)
    forces = -gradients * HARTREE_EV / BOHR_ANGSTROM
    record = {'outer_cycle': cycle, 'omega': omega, 'energy_Ha': float(environment['energy']),
              'coordinates_A': coords.tolist(), 'raw_forces_eV_A': forces.tolist(),
              'rms_gradient_Ha_bohr': float(np.sqrt(np.mean(np.sum(gradients**2, axis=-1)))),
              'max_gradient_Ha_bohr': float(np.linalg.norm(gradients, axis=-1).max()),
              'scf_converged': bool(environment['g_scanner'].converged)}
    with (Path(workdir) / f'geometry_cycle_{cycle:02d}.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    return record


def quantum_geometry_molecule(source, coordinates, status):
    """Keep input annotations distinct from properties of a new quantum geometry."""
    from rdkit import Chem
    molecule = Chem.Mol(source)
    preserved = {'SourceID', 'CandidateID', 'TotalCharge', 'SpinMultiplicity'}
    properties = {key: molecule.GetProp(key) for key in molecule.GetPropNames() if key not in preserved}
    for key in properties:
        molecule.ClearProp(key)
    molecule.SetProp('QuantumSourceProperties', json.dumps(properties))
    molecule.SetProp('QuantumGraphCheck', status)
    molecule.GetConformer().SetPositions(np.asarray(coordinates, dtype=float))
    return molecule


def inspect_optimized_graph(molecule):
    """Report geometry/graph consistency without treating bond inference as DFT truth."""
    from dataclasses import replace

    from rdkit import Chem

    from ..generator.octree_tokenizer import mol3d_to_validated_rdkit_mol, rdkit_mol_to_mol3d_instance

    elements = tuple(dict.fromkeys(atom.GetSymbol() for atom in molecule.GetAtoms()))
    inst = rdkit_mol_to_mol3d_instance(molecule, np.empty(0), center=False, element_table=elements)
    report = {'source_smiles': Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True),
              'source_graph_compatible': False, 'recovered_smiles': None}
    try:
        mol3d_to_validated_rdkit_mol(inst)
        report['source_graph_compatible'] = True
    except ValueError as exc:
        report['source_graph_error'] = str(exc)
    try:
        recovered = mol3d_to_validated_rdkit_mol(replace(inst, graph=None))
        report['recovered_smiles'] = Chem.MolToSmiles(Chem.RemoveHs(recovered), isomericSmiles=True)
        # Atom identities are retained by coordinate reconstruction. Distinguish
        # new/removed contacts from bond-order and formal-charge reassignment.
        def bonds(mol):
            return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondTypeAsDouble()
                    for b in mol.GetBonds()}
        source_bonds, inferred_bonds = bonds(molecule), bonds(recovered)
        coords = molecule.GetConformer().GetPositions()
        report['changed_bonds'] = [
            {'atoms': list(pair), 'source': source_bonds.get(pair), 'inferred': inferred_bonds.get(pair),
             'distance_A': float(np.linalg.norm(coords[pair[0]] - coords[pair[1]]))}
            for pair in sorted(source_bonds.keys() | inferred_bonds.keys())
            if source_bonds.get(pair) != inferred_bonds.get(pair)]
        report['changed_atom_charges'] = [
            {'atom_index': i, 'element': left.GetSymbol(), 'source': left.GetFormalCharge(),
             'inferred': right.GetFormalCharge()}
            for i, (left, right) in enumerate(zip(molecule.GetAtoms(), recovered.GetAtoms()))
            if left.GetFormalCharge() != right.GetFormalCharge()]

    except ValueError as exc:
        report['reconstruction_error'] = str(exc)
    report['graph_preserved'] = report['source_graph_compatible'] and report['recovered_smiles'] == report['source_smiles']
    report['graph_check_status'] = ('consistent' if report['graph_preserved'] else
        'incompatible_source_graph' if not report['source_graph_compatible'] else 'reconstruction_disagrees')
    return report


class OcelotEvaluator:
    def __init__(self, theory, workdir):
        from pyscf import lib

        self.theory = theory
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        lib.num_threads(theory.threads)
        self.trace = []
        self.counter = 0
        self.density_guesses = {}

    def scf(self, atoms, coords, omega, charge=0, spin=0):
        from pyscf import dft

        s = self.theory
        self.counter += 1
        mol = make_quantum_molecule(atoms, coords, s, charge=charge, spin=spin)
        # OCELOT's source uses uLC-wHPBE for neutral and charged species.
        if s.engine == "gpu":
            from .gpu_engine import gpu_scf

            mf = gpu_scf(mol)
        elif s.engine == "cpu":
            mf = dft.UKS(mol)
        else:
            raise ValueError(f"Unknown quantum engine: {s.engine}")
        mf.xc = s.xc
        mf.omega = round(float(omega), 4)  # Gaussian IOp precision in original workflow.
        mf.grids.atom_grid = (s.radial_grid, s.angular_grid)
        mf.conv_tol = s.scf_tolerance
        mf.max_cycle = s.max_scf_cycles
        mf.chkfile = str(self.workdir / f"scf_{self.counter:04d}.chk")
        geometry_key = (np.asarray(coords, dtype=np.float64).tobytes(), charge, spin)
        dm0 = self.density_guesses.get(geometry_key)
        if dm0 is not None and s.engine == "gpu":
            import cupy

            dm0 = cupy.asarray(dm0)
        mf.kernel(dm0=dm0)
        (self.workdir / f'scf_{self.counter:04d}.json').write_text(json.dumps({
            'omega': mf.omega, 'charge': charge, 'spin_2S': spin, 'xc': s.xc, 'basis': s.basis,
            'ecp_policy': s.ecp_policy, 'ecp_core_electrons': ecp_core_electrons(mol),
            'converged': bool(mf.converged), 'energy_Ha': float(mf.e_tot),
            'max_scf_cycles': s.max_scf_cycles, 'convergence_tolerance': s.scf_tolerance,
        }, indent=2) + '\n')
        if not mf.converged:
            raise RuntimeError(f"SCF did not converge at omega={mf.omega}, charge={charge}")
        self.density_guesses[geometry_key] = host_array(mf.make_rdm1())
        return mf

    def tune(self, atoms, coords, bounds):
        cache = {}

        def objective(omega):
            omega = round(float(omega), 4)
            if omega not in cache:
                n = self.scf(atoms, coords, omega)
                c = self.scf(atoms, coords, omega, charge=1, spin=1)
                homo, lumo = frontier_orbitals(n.mo_energy, n.mo_occ)
                anion = self.scf(atoms, coords, omega, charge=-1, spin=1) if self.theory.tuning_objective != 'Jh' else None
                anion_homo = frontier_orbitals(anion.mo_energy, anion.mo_occ)[0] if anion is not None else None
                j = tuning_objective(self.theory.tuning_objective, n.e_tot, c.e_tot, homo, lumo,
                                     anion.e_tot if anion is not None else None, anion_homo)
                row = {
                    "omega": omega,
                    "neutral_E_Ha": float(n.e_tot),
                    "cation_E_Ha": float(c.e_tot),
                    "homo_eV": homo * HARTREE_EV,
                    "lumo_eV": lumo * HARTREE_EV,
                    "Jh_eV": ip_objective(n.e_tot, c.e_tot, homo) * HARTREE_EV,
                    "objective": self.theory.tuning_objective,
                    "objective_atomic_units": float(j),
                    "objective_units": 'Ha' if self.theory.tuning_objective == 'Jh' else 'Ha^2',
                    "anion_E_Ha": float(anion.e_tot) if anion is not None else None,
                    "anion_homo_eV": anion_homo * HARTREE_EV if anion_homo is not None else None,
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

        from ..utils.dataset_io import write_precise_sdf

        started = time.perf_counter()
        if Chem.GetFormalCharge(molecule) or any(a.GetNumRadicalElectrons() for a in molecule.GetAtoms()) or (molecule.HasProp('SpinMultiplicity') and molecule.GetIntProp('SpinMultiplicity') != 1):
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
                geometry_steps = []
                def callback(environment):
                    geometry_steps.append(record_geometry_step(self.workdir, cycle, omega, environment))
                geo_ok, optimized = kernel(
                    mf,
                    maxsteps=self.theory.max_geometry_steps,
                    callback=callback,
                    convergence_energy=1e-6,
                    convergence_grms=3e-4,
                    convergence_gmax=4.5e-4,
                    convergence_drms=1.2e-3,
                    convergence_dmax=1.8e-3,
                )
                coords = optimized.atom_coords(unit="Angstrom")
                if not geo_ok:
                    failed = {'status': 'geometry_not_converged', 'theory': asdict(self.theory),
                              'outer_cycle': cycle, 'omega': omega, 'atom_symbols': atoms,
                              'returned_coordinates_A': coords.tolist(), 'evaluated_steps': len(geometry_steps),
                              'last_evaluated_geometry': geometry_steps[-1] if geometry_steps else None,
                              'elapsed_seconds': time.perf_counter() - started}
                    (self.workdir / 'geometry_failure.json').write_text(json.dumps(failed, indent=2) + '\n')
                    diagnostic = quantum_geometry_molecule(molecule, coords, 'geometry_not_converged')
                    write_precise_sdf(self.workdir / 'unconverged_input_graph.sdf', [diagnostic])
                    raise RuntimeError("Geometry optimization did not converge")
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
        forces = -host_array(final.nuc_grad_method().kernel()) * HARTREE_EV / BOHR_ANGSTROM
        result = {
            "theory": asdict(self.theory),
            "pyscf_version": pyscf.__version__,
            "omega": omega,
            "omega_geometry_cycles": outer,
            "geometry_optimized": optimize_geometry,
            "ecp_core_electrons": ecp_core_electrons(final.mol),
            "energy_Ha": float(final.e_tot),
            "homo_eV": homo * HARTREE_EV,
            "lumo_eV": lumo * HARTREE_EV,
            "S2": float(final.spin_square()[0]),
            "rms_force_eV_A": float(np.sqrt(np.mean(np.sum(forces**2, axis=-1)))),
            "max_force_eV_A": float(np.linalg.norm(forces, axis=-1).max()),
            "elapsed_seconds": time.perf_counter() - started,
            "ground_state_elapsed_seconds": time.perf_counter() - started,
            "atom_symbols": atoms,
            "optimized_coordinates_A": coords.tolist(),
            "raw_forces_eV_A": forces.tolist(),
        }
        out = Chem.Mol(molecule)
        for i, xyz in enumerate(coords):
            out.GetConformer().SetAtomPosition(i, xyz.tolist())
        result.update(inspect_optimized_graph(out))
        # Preserve failed coordinates separately from graph-validated outputs.
        out = quantum_geometry_molecule(molecule, coords, result['graph_check_status'])
        for name, value in {'QuantumHOMO_eV': result['homo_eV'], 'QuantumLUMO_eV': result['lumo_eV'],
                            'QuantumEnergy_Ha': result['energy_Ha'], 'QuantumOmega': omega,
                            'QuantumRMSForce_eV_A': result['rms_force_eV_A']}.items():
            out.SetProp(name, str(value))
        write_precise_sdf(self.workdir / ("optimized.sdf" if result['graph_preserved'] else "diagnostic_input_graph.sdf"), [out])
        (self.workdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result['graph_preserved']:
            raise ValueError('Quantum geometry did not pass graph-preservation checks: ' + result['graph_check_status'])
        if excitations:
            from .excited_states import vertical_excitations

            result['vertical_excitations'] = vertical_excitations(final.mol, omega, self.theory,
                                                                  host_array(final.make_rdm1()).sum(0), float(final.e_tot))
            result['elapsed_seconds'] = time.perf_counter() - started
            (self.workdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
