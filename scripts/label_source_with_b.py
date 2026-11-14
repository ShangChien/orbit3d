"""Attach exact frozen-B labels to existing source coordinates, without refitting B."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from rdkit import Chem

from loom.generator.molecular_state import rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol
from loom.oracles.unimol2_task import FrozenUniMol2Task, support_exclusion
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run, record_input_hashes
from loom.utils.reproducibility import lock_deterministic_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    lock_deterministic_environment(42);torch.set_num_threads(4);model=FrozenUniMol2Task(a.oracle,a.selection)
    inputs={'specification':a.inputs,'oracle':a.oracle,'selection':a.selection};reports={};start=time.perf_counter()
    for spec in json.loads(a.inputs.read_text()):
        path=Path(spec['sdf']);inputs[spec['name']]=path;rows=[];mols=[];positions=[];kept=[]
        for i,mol in enumerate(Chem.SDMolSupplier(str(path),removeHs=False)):
            row=dict(index=i,accepted=False,split=spec['split'],origin=spec['origin']);rows.append(row)
            if mol is None:row['reason']='sdf_parse';continue
            row['source_id']=mol.GetProp('SourceID') if mol.HasProp('SourceID') else str(i)
            exclusion=support_exclusion(mol)
            if exclusion:row['reason']=exclusion;continue
            try:
                inst=rdkit_mol_to_mol3d_instance(mol,np.zeros(2),center=False,
                    element_table=tuple(sorted({at.GetSymbol() for at in mol.GetAtoms()})),coordinate_dtype=np.float64)
                validated=mol3d_to_validated_rdkit_mol(inst)
            except ValueError as error:row['reason']=str(error);continue
            np.testing.assert_array_equal(validated.GetConformer().GetPositions(),mol.GetConformer().GetPositions())
            row.update(accepted=True,smiles=inst.smiles,atoms=mol.GetNumAtoms())
            mols.append(mol);positions.append(i)
        for begin in range(0,len(mols),256):
            chunk=mols[begin:begin+256];prediction=model.predict(chunk)
            for mol,index,values in zip(chunk,positions[begin:begin+256],prediction):
                mol=Chem.Mol(mol);row=rows[index];row['B_prediction_eV']=values.tolist()
                for target,value in zip(['homo','lumo'],values):
                    if mol.HasProp(target):mol.SetProp('OriginalLabel_'+target,mol.GetProp(target))
                    mol.SetProp(target,str(float(value)));mol.SetProp('B_'+target,str(float(value)))
                if mol.HasProp('LabelTheory'):mol.SetProp('OriginalLabelTheory',mol.GetProp('LabelTheory'))
                mol.SetProp('LabelTheory','Frozen Uni-Mol2 1.1B prediction on unchanged source coordinates')
                mol.SetProp('PropertyLabelOrigin','frozen_B_task');mol.SetProp('BModelSHA256',model.contract['hashes']['model.pt'])
                kept.append(mol)
            print(spec['name'],min(begin+256,len(mols)),len(mols),flush=True)
        write_precise_sdf(a.output/(spec['name']+'.sdf'),kept)
        report=dict(input_count=len(rows),accepted=len(kept),exclusions=dict(Counter(r['reason'] for r in rows if not r['accepted'])),records=rows)
        reports[spec['name']]=report
        (a.output/(spec['name']+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    record_input_hashes(a.output,inputs)
    (a.output/'summary.json').write_text(json.dumps(dict(predictor=model.contract,datasets=reports,
        B_molecule_calls=model.molecule_calls,B_forward_calls=model.forward_calls,elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')


if __name__=='__main__':main()
