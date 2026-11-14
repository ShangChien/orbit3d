"""Measure the given-graph versus XYZ-only admission gap on fixed 3D datasets."""
import argparse
from collections import Counter
import json
from pathlib import Path
from rdkit import Chem, RDLogger
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.generator.molecular_state import mol_to_clean_smiles
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__);RDLogger.DisableLog('rdApp.*')
    inputs={'datasets':a.datasets};reports={}
    for spec in json.loads(a.datasets.read_text()):
        path=Path(spec['sdf']);inputs[spec['name']]=path;rows=[];counts=Counter()
        for i,mol in enumerate(Chem.SDMolSupplier(str(path),removeHs=False)):
            if i>=spec['limit']:break
            assert mol is not None
            symbols=[atom.GetSymbol() for atom in mol.GetAtoms()]
            original=mol_to_clean_smiles(mol)
            row=dict(index=i,source_id=mol.GetProp('SourceID') if mol.HasProp('SourceID') else None,
                original_smiles=original,atoms=mol.GetNumAtoms());rows.append(row);counts['attempted']+=1
            try:
                instance=trajectory_to_instance(dict(atom_symbols=symbols,positions_np=mol.GetConformer().GetPositions()),element_table=tuple(sorted(set(symbols))))
            except ValueError as error:
                row.update(status='recovery_failed',reason=str(error));counts['recovery_failed']+=1;continue
            recovered=instance.graph;row.update(status='recovered',recovered_smiles=instance.smiles)
            counts['recovered']+=1
            original_graph=Chem.MolToSmiles(Chem.RemoveHs(mol),isomericSmiles=False)
            recovered_graph=Chem.MolToSmiles(Chem.RemoveHs(recovered),isomericSmiles=False)
            row['same_nonstereo_graph']=original_graph==recovered_graph
            row['same_isomeric_graph']=original==instance.smiles
            counts['same_nonstereo_graph']+=int(row['same_nonstereo_graph']);counts['same_isomeric_graph']+=int(row['same_isomeric_graph'])
            row['original_formal_charges']=[at.GetFormalCharge() for at in mol.GetAtoms()]
            row['recovered_formal_charges']=[at.GetFormalCharge() for at in recovered.GetAtoms()]
        reports[spec['name']]=dict(scope=spec['scope'],counts=dict(counts),records=rows)
        print(spec['name'],dict(counts),flush=True)
    record_input_hashes(a.output,inputs)
    result=dict(scope='Known coordinates, XYZ-only graph recovery with production float32 centering. No octree quantization, training, predictor calls, relaxation or new DFT. Canonical graph differences can include resonance representations; they are not all incorrect chemistry.',datasets=reports)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
