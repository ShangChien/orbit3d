"""Save surrogate excitation gradients on an already compared quantum cohort."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from orbit3d.generator.octree_tokenizer import ATOM_TYPES
from orbit3d.pipeline.evaluators import load_evaluators
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def main():
    parser = argparse.ArgumentParser()
    for name in ['quantum-run', 'config', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    freeze_run(args.output, __file__)
    lock_deterministic_environment(42)
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text())
    _, oracle, domain = load_evaluators(cfg, ATOM_TYPES)
    quantum = json.loads((args.quantum_run / 'summary.json').read_text())
    completed = [r for r in quantum['records'] if r['status'] == 'completed']
    inputs = {'quantum_cohort': args.quantum_run / 'summary.json', 'configuration': args.config,
              'property_model': cfg['mount_paths']['property_checkpoint_dir'],
              'forcefield': cfg['mount_paths']['forcefield_path'],
              'domain': cfg['oracle_config']['applicability_domain_path']}
    arrays, records = {}, []
    for row in completed:
        index = row['index']
        path = args.quantum_run / f'molecule_{index:04d}' / 'optimized.sdf'
        mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=False)))
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        xyz = mol.GetConformer().GetPositions()
        coords = torch.tensor(xyz, dtype=oracle.coordinate_dtype, device=oracle.device, requires_grad=True)
        prediction, _ = oracle.forward_differentiable_3d(coords, symbols)
        values = prediction.reshape(-1)
        for target in ['s0s1', 's0t1']:
            gradient = torch.autograd.grad(values[oracle.target_cols.index(target)], coords,
                                           retain_graph=target == 's0s1')[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Nonfinite {target} gradient at quantum index {index}')
            arrays[f'{index}_{target}_eV_A'] = gradient.detach().cpu().numpy()
        arrays[f'{index}_coordinates_A'] = xyz
        records.append({'quantum_index': index, 'atom_symbols': symbols, 'domain': domain.assess(mol),
                        'prediction_eV': dict(zip(oracle.target_cols, values.detach().cpu().tolist()))})
        inputs[f'geometry_{index}'] = path
    record_input_hashes(args.output, inputs)
    np.savez_compressed(args.output / 'gradients.npz', **arrays)
    (args.output / 'summary.json').write_text(json.dumps({
        'backend': cfg['oracle_config']['oracle_mode'], 'attempted_quantum_cohort': quantum['attempted'],
        'records': records, 'units': 'eV/Angstrom',
        'definition': 'Autograd derivatives of predicted vertical excitation energies with respect to Cartesian coordinates.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
