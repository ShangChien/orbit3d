"""Frozen B on the existing BRICS raw conformers; no new conformer or force call."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from rdkit import Chem
from diagnose_brics_terminal import frozen_instance
from loom.generator.molecular_state import mol3d_to_validated_rdkit_mol
from loom.oracles.unimol2_task import FrozenUniMol2Task, support_exclusion, graph_set, summarize_task
from loom.utils.run_evidence import freeze_run, record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--source-train',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__);torch.set_num_threads(4)
    model=FrozenUniMol2Task(a.oracle,a.selection);known=graph_set(a.source_train)
    source=json.loads((a.source/'summary.json').read_text());manifest=json.loads(a.manifest.read_text())
    mols={m.GetProp('ProposalID'):m for m in Chem.SDMolSupplier(str(a.source/'raw_valid.sdf'),removeHs=False)}
    rows=[];batch=[];indices=[];start=time.perf_counter()
    with np.load(a.source/'raw_conformers.npz') as arrays:
        for original in source['records']:
            row=dict(candidate_id=original['candidate_id'],seed=original['seed'],atoms=original.get('proposed_atoms'),legal=False,evaluated=False);rows.append(row)
            try:
                inst=frozen_instance(original,mols,arrays,tuple(manifest['elements']));mol=mol3d_to_validated_rdkit_mol(inst)
            except ValueError as error:row['rejection']=str(error);continue
            row.update(legal=True,smiles=inst.smiles,atoms=mol.GetNumAtoms())
            exclusion=support_exclusion(mol)
            if exclusion:row['support_exclusion']=exclusion;continue
            batch.append(mol);indices.append(len(rows)-1)
    predictions=model.predict(batch)
    for i,pred in zip(indices,predictions):rows[i].update(evaluated=True,prediction_eV=pred.tolist())
    metrics={target:{q:summarize_task(rows,target,goal,known,known) for q,goal in points.items()}
             for target,points in manifest['single_target_points'].items() if target in ['homo','lumo']}
    report=dict(scope='Existing 192 unconditional BRICS attempts; each original conformer evaluated once. Goal rescoring does not add proposals. Different fragment/size prior from octree. No force or DFT gate.',
                predictor=model.contract,records=rows,by_request=metrics,B_molecule_calls=model.molecule_calls,B_forward_calls=model.forward_calls,seconds=time.perf_counter()-start)
    record_input_hashes(a.output,{'source_summary':a.source/'summary.json','source_sdf':a.source/'raw_valid.sdf','source_xyz':a.source/'raw_conformers.npz','manifest':a.manifest,'source_train':a.source_train,'oracle':a.oracle,'selection':a.selection})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('BRICS',len(rows),len(batch),flush=True)


if __name__=='__main__':main()
