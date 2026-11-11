"""Check def2 electron counts, CPU/GPU parity and real ECP force derivatives."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from orbit3d.quantum.ocelot import (
    BOHR_ANGSTROM,
    HARTREE_EV,
    OcelotEvaluator,
    OcelotTheory,
    ecp_core_electrons,
    host_array,
)
from orbit3d.utils.run_evidence import freeze_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scf-tolerance', type=float, default=1e-9)
    parser.add_argument('--grid-response', action='store_true')
    args = parser.parse_args()
    freeze_run(args.output, __file__)
    theory = OcelotTheory(engine='cpu', threads=2, max_scf_cycles=100, scf_tolerance=args.scf_tolerance)
    omega = .3
    cases = [('HI', ['H', 'I'], [[0, 0, 0], [0, 0, 1.61]], 26),
             ('TeH2', ['Te', 'H', 'H'], [[0, 0, 0], [1.7, 0, 0], [0, 1.7, 0]], 26)]
    records = []
    for name, atoms, coordinates, electrons in cases:
        coords = np.array(coordinates, dtype=float)
        models, values = {}, {}
        for engine in ['cpu', 'gpu']:
            work = args.output / name / engine
            work.mkdir(parents=True)
            evaluator = OcelotEvaluator(replace(theory, engine=engine), work)
            mf = evaluator.scf(atoms, coords, omega)
            if mf.mol.nelectron != electrons:
                raise AssertionError('Incorrect explicit electron count for the def2 ECP')
            derivative = mf.nuc_grad_method()
            derivative.grid_response = args.grid_response
            force = -host_array(derivative.kernel()) * HARTREE_EV / BOHR_ANGSTROM
            models[engine] = evaluator
            values[engine] = {'energy_Ha': float(mf.e_tot), 'forces_eV_A': force.tolist(),
                              'explicit_electrons': mf.mol.nelectron, 'ecp_core_electrons': ecp_core_electrons(mf.mol)}
        difference_e = abs(values['cpu']['energy_Ha'] - values['gpu']['energy_Ha'])
        difference_f = float(np.abs(np.array(values['cpu']['forces_eV_A']) - values['gpu']['forces_eV_A']).max())
        coordinate = int(np.abs(values['gpu']['forces_eV_A']).argmax())
        step = .001
        energies = []
        for sign in [-1, 1]:
            displaced = coords.copy()
            displaced.flat[coordinate] += sign * step
            energies.append(float(models['gpu'].scf(atoms, displaced, omega).e_tot))
        numerical_force = -(energies[1] - energies[0]) * HARTREE_EV / (2 * step)
        analytic_force = float(np.asarray(values['gpu']['forces_eV_A']).flat[coordinate])
        fd_error = abs(numerical_force - analytic_force)
        checks = {'cpu_gpu_energy': difference_e <= 1e-7, 'cpu_gpu_force': difference_f <= 1e-5,
                  'force_finite_difference': fd_error <= 2e-4}
        records.append({'molecule': name, 'coordinates_A': coords.tolist(), 'engines': values,
                        'cpu_gpu_energy_difference_Ha': difference_e, 'cpu_gpu_force_max_difference_eV_A': difference_f,
                        'finite_difference_coordinate': coordinate, 'finite_difference_step_A': step,
                        'analytic_force_eV_A': analytic_force, 'finite_difference_force_eV_A': numerical_force,
                        'finite_difference_error_eV_A': fd_error, 'checks': checks})
        passed = len(records) == len(cases) and all(all(r['checks'].values()) for r in records)
        (args.output / 'summary.json').write_text(json.dumps({'passed': passed,
            'completed_cases': len(records), 'expected_cases': len(cases), 'theory': asdict(theory), 'omega': omega,
            'grid_response': args.grid_response,
            'thresholds': {'cpu_gpu_energy_Ha': 1e-7, 'cpu_gpu_force_eV_A': 1e-5, 'force_finite_difference_eV_A': 2e-4},
            'scope': 'Fixed-omega same-level CPU/GPU Hamiltonian and force check, not tuned OCELOT label accuracy.',
            'records': records}, indent=2) + '\n')
        print(name, difference_e, difference_f, fd_error, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
