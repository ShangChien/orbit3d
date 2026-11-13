"""Independent 3D reassessment of frozen generated pools, without selection or retraining."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.generator.molecular_state import mol3d_to_validated_rdkit_mol
from orbit3d.physics.factory import build_forcefield
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.utils.molecular_identity import compare_molecular_identity
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def metrics(rows,target,goal,domain,prefix="hyff"):
    column=['homo','lumo'].index(target)
    selected=[r for r in rows if r.get(prefix+'_evaluated') and (not domain or r['original_in_domain'])]
    errors=[abs(r[prefix+'_prediction_eV'][column]-goal) for r in selected]
    return {'attempted':len(rows),'evaluated':len(selected),'coverage':len(selected)/len(rows),
            'target_mae_eV':float(np.mean(errors)) if errors else None,
            'mean_prediction_eV':float(np.mean([r[prefix+'_prediction_eV'][column] for r in selected])) if selected else None,
            'property_hits':sum(e<=.25 for e in errors),'property_hit_rate_all_attempts':sum(e<=.25 for e in errors)/len(rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',type=Path,required=True)
    p.add_argument('--forcefield',required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads);freeze_run(a.output,__file__)
    manifest=json.loads(a.manifest.read_text());elements=tuple(manifest['elements'])
    model=InternalDPA4PropertyOracle.load_from_dir(str(a.oracle),build_forcefield('hyff',a.forcefield,elements,'cpu',tier='presto',route='reference'))
    inputs={'oracle':a.oracle,'forcefield':a.forcefield,'source_manifest':a.manifest};results=[];start=time.perf_counter()
    for root in a.sources:
        for path in sorted(root.glob('eval_*/raw/summary.json')):
            pool=path.parent.parent;name=pool.name
            if not any(name.startswith(f'eval_{arm}_') for arm in ['frozen','continue400','grpo200','continue_wall']):continue
            original=json.loads(path.read_text());target=next(k for k,v in original['target_point'].items() if v is not None)
            goal=original['target_point'][target];column=['homo','lumo'].index(target)
            symbols=json.loads((pool/'run/raw_symbols.json').read_text());rows=[];mols=[];indices=[];started=time.perf_counter()
            with np.load(pool/'run/raw_trajectories.npz') as coords:
                for r in original['records']:
                    row={'candidate_id':r['candidate_id'],'valid':r['valid'],'hyff_evaluated':False,
                         'original_in_domain':bool(r.get('domain') and r['domain']['in_domain'])};rows.append(row)
                    if not r['valid']:continue
                    i=r['candidate_id'];instance=trajectory_to_instance({'atom_symbols':symbols[i],'positions_np':coords[f'coordinates_{i}']},element_table=elements)
                    assert compare_molecular_identity(Chem.MolFromSmiles(r['smiles']),Chem.MolFromSmiles(instance.smiles))['identity_match']
                    mols.append(mol3d_to_validated_rdkit_mol(instance));indices.append(len(rows)-1)
            evaluated=model.evaluate_3d_mols_batched(mols)
            for i,pred,std,force,error in zip(indices,evaluated['mean'],evaluated['std'],evaluated['forces_eV_A'],evaluated['errors']):
                if error:rows[i]['hyff_error']=error;continue
                rows[i].update(hyff_evaluated=True,hyff_prediction_eV=pred.tolist(),hyff_std_eV=std.tolist(),
                               hyff_rms_force_eVA=float(np.sqrt(np.mean(np.sum(force**2,axis=-1)))))
            result={'source_pool':str(pool),'target':target,'name':name,'target_eV':goal,'attempted':len(rows),
                    'all_evaluable':metrics(rows,target,goal,False),'original_domain':metrics(rows,target,goal,True),
                    'native_evaluation_attempts':len(mols),'wall_seconds':time.perf_counter()-started,'records':rows}
            if '_off_' in name:
                result['off_rescoring']={q:metrics(rows,target,center,True) for q,center in manifest['single_target_points'][target].items()}
            results.append(result);dest=a.output/(target+'_'+name);dest.mkdir();(dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            inputs[target+'/'+name+'/original']=path;inputs[target+'/'+name+'/coordinates']=pool/'run/raw_trajectories.npz';inputs[target+'/'+name+'/symbols']=pool/'run/raw_symbols.json'
            print(target,name,len(mols),result['wall_seconds'],flush=True)
    record_input_hashes(a.output,inputs)
    (a.output/'summary.json').write_text(json.dumps({'scope':'HyFF independent source-3D head, same frozen XYZ and original domain masks. No selection/reward/model changes. CPU assessment; primary fingerprint results remain separate.',
        'wall_seconds':time.perf_counter()-start,'pools':results},indent=2)+'\n');print('Independent 3D reassessment complete',flush=True)


if __name__=='__main__':main()
