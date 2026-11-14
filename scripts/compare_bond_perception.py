"""Separate coordinate-generation errors from graph-perception sensitivity."""
import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds

from loom.generator.molecular_state import (
    Mol3DInstance,
    mol3d_to_validated_rdkit_mol,
    rdkit_mol_to_mol3d_instance,
)
from loom.utils.molecular_identity import compare_molecular_identity
from loom.utils.run_evidence import freeze_run, record_input_hashes

METHODS = {'connect_the_dots': {}, 'vdw_1.3': {'useVdw': True, 'covFactor': 1.3},
           'extended_hueckel': {'useHueckel': True}}


def perceive(instance, parameters):
    graph = Chem.RWMol()
    for symbol in instance.symbols:
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        graph.AddAtom(atom)
    graph = graph.GetMol()
    conformer = Chem.Conformer(len(instance.symbols))
    conformer.SetPositions(np.asarray(instance.coords, dtype=np.float64))
    graph.AddConformer(conformer)
    rdDetermineBonds.DetermineBonds(graph, charge=instance.charge,
                                  allowChargedFragments=True, **parameters)
    return mol3d_to_validated_rdkit_mol(replace(instance, graph=graph))


def compare(instance, source=None):
    results, molecules = {}, {}
    for name, parameters in METHODS.items():
        try:
            mol = perceive(instance, parameters)
            molecules[name] = mol
            results[name] = {'valid': True, 'smiles': Chem.MolToSmiles(Chem.RemoveHs(mol)),
                             'charged_atoms': sum(a.GetFormalCharge() != 0 for a in mol.GetAtoms())}
            if source is not None:
                results[name].update(compare_molecular_identity(source, mol))
        except (ValueError, RuntimeError) as error:
            results[name] = {'valid': False, 'reason': str(error)}
    baseline = molecules.get('connect_the_dots')
    if baseline is not None:
        for name, mol in molecules.items():
            results[name]['comparison_to_default_graph'] = compare_molecular_identity(baseline, mol)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-sdf', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not rdDetermineBonds.hueckelEnabled():
        raise RuntimeError('This comparison requires RDKit YAeHMOP support')
    freeze_run(args.output, __file__)
    inputs = {'known_validation_structures': args.validation_sdf}
    rows = []
    for index, mol in enumerate(Chem.SDMolSupplier(str(args.validation_sdf), removeHs=False)):
        if mol is None:
            raise ValueError(f'Unreadable validation structure {index}')
        table = tuple(dict.fromkeys(a.GetSymbol() for a in mol.GetAtoms()))
        instance = rdkit_mol_to_mol3d_instance(mol, np.empty(0), center=False, element_table=table,
                                             coordinate_dtype=np.float64)
        rows.append({'cohort': 'known_validation', 'index': index,
                     'source_id': mol.GetProp('SourceID'),
                     'source_charged_atoms': sum(a.GetFormalCharge() != 0 for a in mol.GetAtoms()),
                     'methods': compare(instance, mol)})
    for run in args.runs:
        for filename in ['raw_symbols.json', 'raw_trajectories.npz', 'candidates.csv']:
            inputs[f'{run.name}/{filename}'] = run / filename
        symbols = json.loads((run / 'raw_symbols.json').read_text())
        ledger = pd.read_csv(run / 'candidates.csv').set_index('candidate_id')
        if set(ledger.index) != set(range(len(symbols))):
            raise ValueError('Candidate ledger and raw coordinate indices differ')
        with np.load(run / 'raw_trajectories.npz') as arrays:
            for index, atoms in enumerate(symbols):
                table = tuple(dict.fromkeys(atoms))
                instance = Mol3DInstance(coords=arrays[f'coordinates_{index}'],
                                         atom_types=np.array([table.index(s) for s in atoms]),
                                         props=np.empty(0), smiles='', element_table=table)
                result = compare(instance)
                if result['connect_the_dots']['valid'] != bool(ledger.loc[index, 'raw_valid']):
                    raise ValueError(f'Default reconstruction no longer matches the original ledger: {run.name}/{index}')
                rows.append({'cohort': 'generated', 'run': run.name, 'index': index, 'methods': result})
    record_input_hashes(args.output, inputs)
    aggregate = {}
    for cohort in ['known_validation', 'generated']:
        group = [r for r in rows if r['cohort'] == cohort]
        aggregate[cohort] = {'attempted': len(group), 'methods': {}}
        for name in METHODS:
            values = [r['methods'][name] for r in group]
            valid = [r for r in values if r['valid']]
            paired = [r['comparison_to_default_graph'] for r in valid if 'comparison_to_default_graph' in r]
            aggregate[cohort]['methods'][name] = {
                'valid': len(valid), 'valid_without_formal_atomic_charges': sum(r['charged_atoms'] == 0 for r in valid),
                'source_identity_status': dict(Counter(r.get('identity_status', 'invalid' if not r['valid'] else 'no_reference_graph') for r in values)),
                'paired_valid_with_default': len(paired),
                'constitution_changes_from_default': sum(not r['constitution_match'] for r in paired),
            }
    report = {'rdkit_version': rdBase.rdkitVersion, 'methods': METHODS, 'aggregate': aggregate, 'records': rows,
              'scope': 'Fixed validation geometries and all saved seed-42 total-20000 proposals; no new sampling, graph replacement in original runs, or property-score selection. Generated graphs have no known reference identity. Same chemical validator for every perception method.'}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
