"""Pair frozen surrogate predictions with completed, unchanged quantum geometries."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from orbit3d.generator.octree_tokenizer import ATOM_TYPES
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.pipeline.evaluators import load_evaluators
from orbit3d.utils.property_metrics import paired_energy_difference_metrics
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def indexed_graph(molecule):
    atoms = [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope(), a.GetNumRadicalElectrons())
             for a in molecule.GetAtoms()]
    bonds = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    b.GetBondTypeAsDouble()) for b in molecule.GetBonds())
    return atoms, bonds, Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ['quantum-run', 'source-sdf', 'config', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fingerprint-model', type=Path)
    args = parser.parse_args()
    freeze_run(args.output, __file__)
    lock_deterministic_environment(42)
    torch.set_num_threads(2)
    quantum = json.loads((args.quantum_run / 'summary.json').read_text())
    manifest = json.loads((args.quantum_run / 'manifest.json').read_text())
    if hashlib.sha256(args.source_sdf.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Source SDF does not match the predeclared quantum cohort')
    cfg = json.loads(args.config.read_text())
    if cfg['oracle_config']['target_cols'] != ['homo', 'lumo', 's0s1', 's0t1']:
        raise ValueError('This quantum comparison uses HOMO/LUMO/vertical-S1/vertical-T1 in that order')
    if cfg['oracle_config']['target_units'] != ['eV'] * 4:
        raise ValueError('The quantum and surrogate target units must be eV')
    if len(quantum['records']) != quantum['attempted'] or len({r['index'] for r in quantum['records']}) != quantum['attempted']:
        raise ValueError('Quantum cohort has missing or duplicated records')
    source = list(Chem.SDMolSupplier(str(args.source_sdf), removeHs=False))
    completed = [r for r in quantum['records'] if r['status'] == 'completed']
    if not completed:
        raise ValueError('There are no completed quantum geometries to compare')
    inputs = {'quantum_results': args.quantum_run / 'summary.json',
              'quantum_protocol': args.quantum_run / 'manifest.json', 'source_structures': args.source_sdf,
              'configuration': args.config, 'property_model': cfg['mount_paths']['property_checkpoint_dir'],
              'forcefield': cfg['mount_paths']['forcefield_path'],
              'domain': cfg['oracle_config']['applicability_domain_path']}
    molecules = []
    for row in completed:
        path = args.quantum_run / f'molecule_{row["index"]:04d}' / 'optimized.sdf'
        exported = list(Chem.SDMolSupplier(str(path), removeHs=False))
        if len(exported) != 1 or exported[0] is None:
            raise ValueError(f'Expected one readable optimized molecule: {path}')
        original = source[row['index']]
        if original is None or indexed_graph(original) != indexed_graph(exported[0]):
            raise ValueError(f'Quantum geometry changed indexed source identity: {row["index"]}')
        molecules.append(exported[0])
        inputs[f'optimized_geometry_{row["index"]}'] = path
    _, oracle, domain = load_evaluators(cfg, ATOM_TYPES)
    evaluated = oracle.evaluate_3d_mols_batched(molecules)
    if any(evaluated['errors']):
        raise RuntimeError(f'Surrogate evaluation failed: {evaluated["errors"]}')
    names = oracle.target_cols
    predictions = {cfg['oracle_config']['oracle_mode']: np.asarray(evaluated['mean'])}
    if args.fingerprint_model:
        independent = FingerprintOracle.load_from_dir(args.fingerprint_model)
        if independent.target_cols != names:
            raise ValueError('Fingerprint and 3D property target orders differ')
        predictions['fingerprint'], _ = independent.predict_3d_mols(molecules)
        inputs['fingerprint_model'] = args.fingerprint_model
    record_input_hashes(args.output, inputs)
    reference = np.array([[r['homo_eV'], r['lumo_eV'], r['vertical_excitations']['singlet_eV'][0],
                           r['vertical_excitations']['triplet_eV'][0]] for r in completed])
    rows, forces = [], {}
    for index, (mol, result) in enumerate(zip(molecules, completed)):
        force = np.asarray(evaluated['forces_eV_A'][index])
        forces[f'quantum_index_{result["index"]}'] = force
        rows.append({'quantum_index': result['index'], 'source_properties': result['properties'],
                     'smiles': Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True), 'omega': result['omega'],
                     'reference_eV': dict(zip(names, reference[index].tolist())),
                     'predictions_eV': {name: dict(zip(names, values[index].tolist())) for name, values in predictions.items()},
                     'domain': domain.assess(mol),
                     'surrogate_rms_force_eV_A': float(np.sqrt(np.square(force).sum(-1).mean())),
                     'quantum_rms_force_eV_A': result['rms_force_eV_A']})
    aggregate = {name: {'paired_molecules': len(rows),
                       'mean_absolute_disagreement_eV': dict(zip(names, np.abs(prediction - reference).mean(0).tolist())),
                       'paired_vertical_gap_disagreement': paired_energy_difference_metrics(prediction, reference, names, 's0s1', 's0t1')}
                 for name, prediction in predictions.items()}
    report = {'attempted_quantum_cohort': quantum['attempted'], 'completed_paired': len(rows),
              'failed_quantum_indices': [r['index'] for r in quantum['records'] if r['status'] != 'completed'],
              'target_names': names, 'quantum_manifest': manifest, 'records': rows, 'aggregate': aggregate,
              'scope': 'Same saved quantum geometries, no optimization or retuning. Conditional disagreement on completed cases, not OCELOT reference MAE. Surrogate and quantum forces are cross-potential diagnostics, not identical-theory force errors.'}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output / 'surrogate_raw_forces.npz', **forces)
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
