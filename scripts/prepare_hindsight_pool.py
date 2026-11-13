"""Relabel frozen native-coordinate proposals using their recorded A predictions."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import record_input_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--target',choices=['homo','lumo'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=512);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.data/'manifest.json').read_text());source=json.loads((a.data/'records.json').read_text())
    heldout=[r for r in source if r.get('status')=='accepted' and r['split']!='train']
    forbidden_graphs={r['smiles'] for r in heldout};forbidden_scaffolds={r['scaffold'] for r in heldout}
    train=list(Chem.SDMolSupplier(str(a.data/'train.sdf'),removeHs=False))
    boundaries=np.quantile([float(m.GetProp(a.target)) for m in train],[.1,.25,.5,.75,.9])
    bins=[[] for _ in range(6)];seen=set();counts=Counter();inputs={'source_split':a.data/'records.json','manifest':a.data/'manifest.json','train':a.data/'train.sdf'}
    for pool in sorted(a.campaign.glob('eval_frozen_p*_s*/raw/summary.json')):
        parent=pool.parent.parent;run=parent/'run';raw=json.loads(pool.read_text())
        inputs[str(parent.name)+'/summary']=pool;inputs[str(parent.name)+'/raw_manifest']=pool.parent/'input_manifest.json'
        inputs[str(parent.name)+'/coordinates']=run/'raw_trajectories.npz';inputs[str(parent.name)+'/symbols']=run/'raw_symbols.json'
        symbols=json.loads((run/'raw_symbols.json').read_text())
        with np.load(run/'raw_trajectories.npz') as arrays:
            for r in raw['records']:
                counts['attempted']+=1
                if not (r['evaluated'] and r.get('domain',{}).get('in_domain')):continue
                i=r['candidate_id'];inst=trajectory_to_instance({'atom_symbols':symbols[i],'positions_np':arrays[f'coordinates_{i}']},element_table=tuple(manifest['elements']))
                assert inst.smiles==r['smiles']
                mol=inst.graph;graph=Chem.MolToSmiles(Chem.RemoveHs(mol))
                scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(mol),includeChirality=False) or '<ACYCLIC>'
                if graph in forbidden_graphs or scaffold in forbidden_scaffolds:counts['heldout_excluded']+=1;continue
                if graph in seen:counts['duplicate_graph']+=1;continue
                seen.add(graph)
                prediction=r['prediction'];assert np.isfinite(prediction).all()
                for name,value in zip(['homo','lumo'],prediction):mol.SetProp(name,str(float(value)))
                mol.SetProp('LabelTheory','Frozen DPA adapter prediction on this generated coordinate set')
                mol.SetProp('PropertyLabelOrigin','surrogate_hindsight_not_DFT')
                mol.SetProp('SourcePool',str(parent));mol.SetProp('CandidateID',str(i))
                mol.SetProp('OriginalRequest',json.dumps(raw['target_point']))
                label=prediction[['homo','lumo'].index(a.target)];bucket=int(np.searchsorted(boundaries,label))
                key=hashlib.sha256(f'E10-hindsight:{graph}'.encode()).hexdigest()
                bins[bucket].append((key,mol,{'pool':str(parent),'candidate_id':i,'smiles':graph,'scaffold':scaffold,'prediction_eV':prediction,'bin':bucket}))
    for bucket in bins:bucket.sort(key=lambda row:row[0])
    selected=[]
    while len(selected)<a.limit and any(bins):
        for bucket in bins:
            if bucket and len(selected)<a.limit:selected.append(bucket.pop(0))
    assert selected, 'No eligible frozen candidates'
    write_precise_sdf(a.output/'train.sdf',[r[1] for r in selected])
    report={'label_origin':'surrogate_hindsight_not_DFT','target':a.target,'counts':dict(counts),'selected':len(selected),
            'bin_boundaries_eV':boundaries.tolist(),'selected_bins':dict(Counter(r[2]['bin'] for r in selected)),
            'selection':'Round-robin source-property bins; hash order within bins, graph unique, no B or DFT selection',
            'coordinates':'Production trajectory_to_instance transform; same native coordinates used by original A prediction',
            'records':[r[2] for r in selected],'sdf_sha256':hashlib.sha256((a.output/'train.sdf').read_bytes()).hexdigest()}
    record_input_hashes(a.output,inputs);(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
