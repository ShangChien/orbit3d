"""Freeze every accepted terminal proposal; never rank by predicted properties."""
import argparse, hashlib, json
from pathlib import Path
from rdkit import Chem
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def freeze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    freeze_run(output, __file__)
    arms = ['point_frozen','terminal_sft','terminal_structure','terminal_physical','terminal_property']
    expected = [177,160,167,182,162]
    records, molecules, inputs, counts = [], [], {}, {}
    for arm, count in zip(arms, expected):
        folder = root / arm
        report = json.loads((folder/'summary.json').read_text())
        inputs[arm+'/summary'] = folder/'summary.json'
        inputs[arm+'/coordinates'] = folder/'graph_preserved.sdf'
        mols = {m.GetProp('TerminalID'): m for m in Chem.SDMolSupplier(str(inputs[arm+'/coordinates']),removeHs=False)}
        rows = [r for r in report['records'] if r['physical_accepted']]
        assert len(report['records']) == 384 and len(rows) == count
        counts[arm] = count
        for row in rows:
            mol = mols[row['coordinates_key']]
            mol.SetProp('QuantumIndex', str(len(records)))
            records.append(dict(index=len(records), target='homo', target_eV=row['goal_eV'],
                quantile=row['quantile'], arm=arm, selection='all_terminal_accepted', training_seed=42,
                inference_seed=row['inference_seed'],candidate_id=row['candidate_id'],
                domain=row.get('domain'), source_terminal=str(folder), coordinates_key=row['coordinates_key'],
                dpa_prediction_eV=row['final']['prediction_eV'], smiles=Chem.MolToSmiles(mol)))
            molecules.append(mol)
    write_precise_sdf(output/'cohort.sdf', molecules)
    record_input_hashes(output, inputs)
    manifest=dict(selection='All 848 accepted E12 proposals; no score selection; repeated graphs retain proposal identity.',
        arms=arms,targets=['homo'], expected=848, selected=len(records), missing=[],
        proposal_denominators={arm:384 for arm in arms},planned_per_group={f'homo/{a}':n for a,n in counts.items()},
        records=records,sdf_sha256=hashlib.sha256((output/'cohort.sdf').read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(counts, flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();freeze(a.root,a.output)
