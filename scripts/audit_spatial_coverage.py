"""Measure saved proposal spans without selecting new samples or changing validity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from loom.generator.octree_backend import trajectory_to_instance
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run, record_input_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze_run(args.output, __file__)
    inputs, rows, extended = {}, [], []
    for run in args.runs:
        for name in ['raw_trajectories.npz', 'raw_symbols.json', 'candidates.csv', 'summary.json']:
            inputs[f'{run.parent.name}/{run.name}/{name}'] = run / name
        table = pd.read_csv(run / 'candidates.csv').set_index('candidate_id')
        symbols = json.loads((run / 'raw_symbols.json').read_text())
        summary = json.loads((run / 'summary.json').read_text())
        if len(table) != summary['attempted'] or len(symbols) != len(table) or not table.index.is_unique:
            raise ValueError('The attempt ledger and raw trajectories differ')
        with np.load(run / 'raw_trajectories.npz') as arrays:
            for index, atoms in enumerate(symbols):
                coords = arrays[f'coordinates_{index}'].astype(np.float64)
                if coords.shape != (len(atoms), 3) or not np.isfinite(coords).all():
                    raise ValueError('Saved raw coordinates are malformed')
                span = np.ptp(coords, axis=0) if len(coords) else np.zeros(3)
                diameter = float(np.linalg.norm(coords[:, None] - coords[None, :], axis=-1).max()) if len(coords) else 0.
                row = {
                    'run': f'{run.parent.name}/{run.name}', 'candidate_id': index,
                    'atoms': len(atoms), 'axis_spans_A': span.tolist(), 'diameter_A': diameter,
                    'raw_valid': bool(table.loc[index, 'raw_valid']),
                    'exceeds_old_rotation_safe_training_span': diameter > .24 * 64 - .56,
                    'exceeds_old_box_in_saved_orientation': bool((span > .24 * 64).any()),
                }
                if row['raw_valid'] and row['exceeds_old_rotation_safe_training_span']:
                    instance = trajectory_to_instance(
                        {'atom_symbols': atoms, 'positions_np': coords}, element_table=tuple(dict.fromkeys(atoms)))
                    mol = instance.graph
                    mol.SetProp('SourceRun', row['run'])
                    mol.SetProp('CandidateID', str(index))
                    mol.SetProp('DiameterAngstrom', str(diameter))
                    periodic = Chem.GetPeriodicTable()
                    bond_ratios = [float(np.linalg.norm(instance.coords[b.GetBeginAtomIdx()] - instance.coords[b.GetEndAtomIdx()])) /
                                   (periodic.GetRcovalent(b.GetBeginAtom().GetAtomicNum()) + periodic.GetRcovalent(b.GetEndAtom().GetAtomicNum()))
                                   for b in mol.GetBonds()]
                    row['revalidated_graph'] = {
                        'smiles': instance.smiles, 'components': len(Chem.GetMolFrags(mol)),
                        'charged_atoms': sum(a.GetFormalCharge() != 0 for a in mol.GetAtoms()),
                        'min_bond_length_over_covalent_radii': min(bond_ratios),
                        'max_bond_length_over_covalent_radii': max(bond_ratios),
                    }
                    extended.append(mol)
                rows.append(row)
    record_input_hashes(args.output, inputs)
    counts = {}
    for label, selected in [('all', rows), ('chemically_valid', [r for r in rows if r['raw_valid']])]:
        counts[label] = {
            'n': len(selected),
            'above_old_atom_limit_40': sum(r['atoms'] > 40 for r in selected),
            'above_old_rotation_safe_span': sum(r['exceeds_old_rotation_safe_training_span'] for r in selected),
            'above_old_box_in_saved_orientation': sum(r['exceeds_old_box_in_saved_orientation'] for r in selected),
            'max_axis_span_A': max((max(r['axis_spans_A']) for r in selected), default=None),
            'max_diameter_A': max((r['diameter_A'] for r in selected), default=None),
        }
    report = {
        'scope': 'Saved generation cohorts, including every failed and filtered attempt; validity is the original chemically checked bit, not recomputed with relaxed rules.',
        'old_box_side_A': .24 * 64, 'old_rotation_safe_training_span_A': .24 * 64 - .56,
        'interpretation': 'More than 40 atoms does not imply a larger spatial span. Exceeding the old span is not evidence of useful target-conditioned generation.',
        'counts': counts, 'records': rows,
    }
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    write_precise_sdf(args.output / 'valid_extended.sdf', extended)
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
