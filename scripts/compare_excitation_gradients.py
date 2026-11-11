"""Compare surrogate derivatives with fixed-omega full-TDDFT excitation gradients."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyscf import lib
from rdkit import Chem

from orbit3d.quantum.excited_states import response, restricted_reference
from orbit3d.quantum.ocelot import (
    BOHR_ANGSTROM,
    HARTREE_EV,
    host_array,
    make_quantum_molecule,
    theory_from_record,
)
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantum-run', type=Path, required=True)
    parser.add_argument('--surrogate-gradients', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze_run(args.output, __file__)
    cohort = json.loads((args.quantum_run / 'summary.json').read_text())
    completed = [r for r in cohort['records'] if r['status'] == 'completed']
    models, inputs = {}, {'quantum_cohort': args.quantum_run / 'summary.json',
                          'quantum_protocol': args.quantum_run / 'manifest.json'}
    for path in args.surrogate_gradients:
        metadata = json.loads((path / 'summary.json').read_text())
        backend = metadata['backend']
        if backend in models:
            raise ValueError('Each surrogate backend may occur only once')
        if [r['quantum_index'] for r in metadata['records']] != [r['index'] for r in completed]:
            raise ValueError('Surrogate and quantum cohorts differ')
        models[backend] = (metadata, np.load(path / 'gradients.npz'))
        inputs[backend + '_gradients'] = path / 'gradients.npz'
        inputs[backend + '_metadata'] = path / 'summary.json'
    manifest = json.loads((args.quantum_run / 'manifest.json').read_text())
    theory = theory_from_record(manifest['theory'])
    lib.num_threads(theory.threads)
    records, arrays = [], {}
    for result in completed:
        index = result['index']
        geometry = args.quantum_run / f'molecule_{index:04d}' / 'optimized.sdf'
        inputs[f'geometry_{index}'] = geometry
    record_input_hashes(args.output, inputs)
    for result in completed:
        index = result['index']
        started = time.perf_counter()
        row = {'quantum_index': index, 'omega': result['omega'], 'status': 'failed'}
        try:
            molecule = next(iter(Chem.SDMolSupplier(str(inputs[f'geometry_{index}']), removeHs=False)))
            xyz = molecule.GetConformer().GetPositions()
            symbols = [a.GetSymbol() for a in molecule.GetAtoms()]
            for metadata, stored in models.values():
                np.testing.assert_array_equal(xyz, stored[f'{index}_coordinates_A'])
                entry = next(r for r in metadata['records'] if r['quantum_index'] == index)
                if entry['atom_symbols'] != symbols:
                    raise ValueError('Surrogate and quantum atom orders differ')
            mol = make_quantum_molecule(symbols, xyz, theory)
            mf = restricted_reference(mol, result['omega'], theory)
            reference_energy = result['vertical_excitations']['ground_energy_Ha']
            if abs(float(mf.e_tot) - reference_energy) > 1e-6:
                raise ValueError('Recomputed ground energy differs from the stored restricted reference')
            ground_gradient = host_array(mf.nuc_grad_method().kernel())
            comparisons, excitations = {}, {}
            for target, singlet in [('s0s1', True), ('s0t1', False)]:
                td, energies = response(mf, singlet)
                reference = result['vertical_excitations']['singlet_eV' if singlet else 'triplet_eV'][0]
                value = float(energies[0]) * HARTREE_EV
                if abs(value - reference) > 1e-4:
                    raise ValueError(f'{target} root differs from the stored full-TDDFT result')
                total_gradient = host_array(td.nuc_grad_method().kernel(state=1))
                quantum_gradient = (total_gradient - ground_gradient) * HARTREE_EV / BOHR_ANGSTROM
                arrays[f'{index}_{target}_quantum_eV_A'] = quantum_gradient
                excitations[target] = value
                comparisons[target] = {}
                for backend, (_, stored) in models.items():
                    surrogate = stored[f'{index}_{target}_eV_A']
                    norm_q, norm_s = float(np.linalg.norm(quantum_gradient)), float(np.linalg.norm(surrogate))
                    dot = float(np.sum(quantum_gradient * surrogate))
                    comparisons[target][backend] = {
                        'cosine': dot / (norm_q * norm_s) if norm_q and norm_s else None,
                        'quantum_norm_eV_A': norm_q, 'surrogate_norm_eV_A': norm_s,
                        'quantum_slope_along_unit_negative_surrogate_gradient_eV_A': -dot / norm_s if norm_s else None,
                        'cartesian_mean_absolute_disagreement_eV_A': float(np.abs(quantum_gradient - surrogate).mean()),
                    }
                print(index, target, comparisons[target], flush=True)
            row.update(status='completed', ground_energy_Ha=float(mf.e_tot), vertical_excitations_eV=excitations,
                       comparison=comparisons)
        except (ValueError, RuntimeError, FloatingPointError, AssertionError, np.linalg.LinAlgError) as exc:
            row['error'] = str(exc)
            print(index, 'failed', str(exc), flush=True)
        row['elapsed_seconds'] = time.perf_counter() - started
        records.append(row)
        report = {'attempted_quantum_cohort': cohort['attempted'], 'gradient_cohort': len(completed),
                  'original_failed_indices': [r['index'] for r in cohort['records'] if r['status'] != 'completed'],
                  'completed_gradients': sum(r['status'] == 'completed' for r in records),
                  'theory': manifest['theory'], 'records': records,
                  'definition': 'Gradient of the vertical excitation = total excited-state gradient minus S0 gradient; fixed original per-molecule omega and geometry, full TDDFT, no TDA or density-fitting substitution.',
                  'scope': 'All five completed generated-boron cases are OOD. These cross-model directional diagnostics do not validate in-domain OCELOT accuracy or adiabatic state gradients.'}
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        np.savez_compressed(args.output / 'quantum_gradients.npz', **arrays)
    if any(r['status'] != 'completed' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
