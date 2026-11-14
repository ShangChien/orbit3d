"""Relabel a dedicated raw B-scored recovery pool on its exact scored coordinates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from loom.generator.octree_backend import trajectory_to_instance
from loom.generator.molecular_state import mol3d_to_validated_rdkit_mol
from loom.oracles.unimol2_task import graph_set
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run, record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',type=Path,required=True);p.add_argument('--source-data',type=Path,required=True)
    p.add_argument('--source-train',type=Path,required=True);p.add_argument('--exclude-sdfs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    source=graph_set(a.source_train);heldout=set();inputs={'source':a.source_train,'pool':a.pool/'summary.json'}
    for split in ['valid','calibration','test']:
        path=a.source_data/(split+'.csv');heldout.update(pd.read_csv(path).scaffold);inputs[split]=path
    scaffold=lambda m:MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(m),includeChirality=False) or '<ACYCLIC>'
    for path in a.exclude_sdfs:
        heldout.update(scaffold(m) for m in Chem.SDMolSupplier(str(path),removeHs=False));inputs[str(path)]=path
    candidates=[];counts=Counter();report=json.loads((a.pool/'summary.json').read_text())
    for pool in sorted(a.pool.glob('eval_*/raw/summary.json')):
        parent=pool.parent.parent;raw=json.loads(pool.read_text());run=parent/'run'
        assert raw['control']=='correct'
        symbols=json.loads((run/'raw_symbols.json').read_text());inputs[str(pool)]=pool
        inputs[str(run/'raw_symbols.json')]=run/'raw_symbols.json';inputs[str(run/'raw_trajectories.npz')]=run/'raw_trajectories.npz'
        with np.load(run/'raw_trajectories.npz') as arrays:
            for row in raw['records']:
                counts['attempted']+=1
                if not row['evaluated']:counts['not_evaluated']+=1;continue
                i=row['candidate_id'];xyz=arrays[f'coordinates_{i}']
                inst=trajectory_to_instance({'atom_symbols':symbols[i],'positions_np':xyz},element_table=tuple(sorted(set(symbols[i]))))
                mol=mol3d_to_validated_rdkit_mol(inst);assert inst.smiles==row['smiles']
                if scaffold(mol) in heldout:counts['heldout_scaffold']+=1;continue
                if inst.smiles in source:counts['source_graph']+=1;continue
                key=hashlib.sha256((inst.smiles+hashlib.sha256(xyz.tobytes()).hexdigest()).encode()).hexdigest()
                for name,value in zip(['homo','lumo'],row['prediction_eV']):
                    mol.SetProp(name,str(float(value)));mol.SetProp('B_'+name,str(float(value)))
                mol.SetProp('LabelTheory','Frozen Uni-Mol2 1.1B on original generated coordinates')
                mol.SetProp('PropertyLabelOrigin','B_hindsight_prediction_not_DFT');mol.SetProp('SourcePool',str(parent))
                mol.SetProp('CandidateID',str(i));mol.SetProp('OriginalRequest',json.dumps(raw['target_point']))
                mol.SetProp('BModelSHA256',report['predictor']['hashes']['model.pt'])
                candidates.append((key,inst.smiles,mol,dict(pool=str(parent),candidate_id=i,smiles=inst.smiles,scaffold=scaffold(mol),prediction_eV=row['prediction_eV'])))
    seen=set();selected=[]
    for key,graph,mol,row in sorted(candidates,key=lambda v:v[0]):
        if graph in seen:counts['duplicate_graph']+=1;continue
        seen.add(graph);selected.append((mol,row))
    assert selected
    write_precise_sdf(a.output/'train.sdf',[m for m,_ in selected]);record_input_hashes(a.output,inputs)
    summary=dict(counts=dict(counts),selected=len(selected),selection='All eligible graph-unique molecules, one geometry by hash; no request-hit selection. Equal-width capped sampling in trainer.',
        predictor=report['predictor'],records=[r for _,r in selected],sdf_sha256=hashlib.sha256((a.output/'train.sdf').read_bytes()).hexdigest())
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print({'selected':len(selected),'counts':dict(counts)})


if __name__=='__main__':main()
