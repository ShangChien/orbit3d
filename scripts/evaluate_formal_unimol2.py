"""Cross-evaluate full formal pools with frozen Uni-Mol2; no new proposal or selection."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from loom.generator.octree_backend import trajectory_to_instance
from loom.generator.molecular_state import mol3d_to_validated_rdkit_mol
from loom.utils.molecular_identity import compare_molecular_identity
from loom.oracles.unimol2_oracle import InternalUniMol2Oracle
from loom.utils.run_evidence import freeze_run,record_input_hashes
from evaluate_independent_3d_campaign import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);freeze_run(a.output,__file__);selection=json.loads(a.selection.read_text())
    with (a.oracle/'model.pt').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==selection['hashes'][selection['selected_unimol']]['model.pt']
    assert hashlib.sha256((a.oracle/"oracle_meta.json").read_bytes()).hexdigest()==selection["hashes"][selection["selected_unimol"]]["oracle_meta.json"]
    elements=tuple(json.loads(a.manifest.read_text())['elements']);model=InternalUniMol2Oracle.load_from_dir(a.oracle)
    inputs={'oracle':a.oracle,'selection':a.selection,'manifest':a.manifest};pools=[];start=time.perf_counter()
    for root in a.sources:
        for path in sorted(root.glob('eval_*/raw/summary.json')):
            original=json.loads(path.read_text());pool=path.parent.parent;name=pool.name
            target=next(k for k,v in original['target_point'].items() if v is not None);goal=original['target_point'][target]
            symbols=json.loads((pool/'run/raw_symbols.json').read_text());rows=[];mols=[];indices=[];started=time.perf_counter()
            with np.load(pool/'run/raw_trajectories.npz') as coords:
                for source in original['records']:
                    i=source['candidate_id'];row={'candidate_id':i,'valid':source['valid'],'original_in_domain':bool(source.get('domain') and source['domain']['in_domain']),
                        'prediction_attempted':False,'unimol2_evaluated':False};rows.append(row)
                    if not source['valid']:continue
                    if len(symbols[i])>80:row['support_exclusion']='More than 80 source atoms';continue
                    if 'Te' in symbols[i]:row['support_exclusion']='Te absent from shared training support';continue
                    inst=trajectory_to_instance({'atom_symbols':symbols[i],'positions_np':coords[f'coordinates_{i}']},element_table=elements)
                    assert compare_molecular_identity(Chem.MolFromSmiles(source['smiles']),Chem.MolFromSmiles(inst.smiles))['identity_match']
                    mols.append(mol3d_to_validated_rdkit_mol(inst));indices.append(len(rows)-1);row['prediction_attempted']=True
            prediction,_=model.predict_3d_mols(mols,batch_size=16)
            for i,values in zip(indices,prediction):rows[i].update(unimol2_evaluated=True,unimol2_prediction_eV=values.tolist())
            result={'source_pool':str(pool),'target':target,'name':name,'target_eV':goal,'attempted':len(rows),'native_evaluation_attempts':len(mols),
                'support_exclusions':sum('support_exclusion' in r for r in rows),'wall_seconds':time.perf_counter()-started,
                'original_domain':metrics(rows,target,goal,True,prefix='unimol2'),'records':rows}
            dest=a.output/(target+'_'+name);dest.mkdir();(dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n');pools.append(result)
            for label,file in [('original',path),('coordinates',pool/'run/raw_trajectories.npz'),('symbols',pool/'run/raw_symbols.json')]:inputs[target+'/'+name+'/'+label]=file
            print(target,name,len(mols),round(result['wall_seconds'],2),flush=True)
    report={'evaluator':'Uni-Mol2 1.1B','prediction_prefix':'unimol2','scope':'Frozen source-heavy-atom Uni-Mol2; original XYZ, original domains and all proposals retained. Element and source-size support exclusions separate; no reward, selection or generator changes.',
        'wall_seconds':time.perf_counter()-start,'pools':pools}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');record_input_hashes(a.output,inputs)
    print('FORMAL UNIMOL2 COMPLETE',len(pools),flush=True)


if __name__=='__main__':main()
