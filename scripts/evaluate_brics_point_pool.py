"""Unconditional training-fragment baseline, with one ETKDG attempt and frozen point scoring."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from orbit3d.baselines.brics import build_library,FragmentSampler
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.generator.molecular_state import rdkit_mol_to_mol3d_instance,mol3d_to_validated_rdkit_mol
from orbit3d.physics.factory import build_forcefield
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.oracles.applicability import MorganDomain
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def summarize_scores(scores):
    aggregate=[]
    for target in ['homo','lumo']:
        for scope in ['raw','top16']:
            rows=[r for r in scores if r['target']==target and r['scope']==scope]
            for model in rows[0]['models']:
                quantiles=[]
                for q in ['10','25','50','75','90']:
                    group=[r['models'][model] for r in rows if r['quantile']==q]
                    n=sum(r['evaluated'] for r in group)
                    quantiles.append(sum(r['target_absolute_error_sum_eV'] for r in group)/n if n else None)
                aggregate.append({'target':target,'scope':scope,'model':model,
                    'task_proposal_count':sum(r['attempted'] for r in rows),
                    'evaluated_task_pairs':sum(r['models'][model]['evaluated'] for r in rows),
                    'target_mae_equal_quantile_eV':float(np.mean(quantiles)) if all(v is not None for v in quantiles) else None,
                    'property_hits':sum(r['models'][model]['property_hits'] for r in rows),
                    'property_and_dpa_force_hits':sum(r['models'][model]['property_and_dpa_force_hits'] for r in rows)})
    return aggregate


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--domain',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);freeze_run(a.output,__file__);manifest=json.loads((a.data/'manifest.json').read_text());spec=json.loads(a.models.read_text())
    source=list(Chem.SDMolSupplier(str(a.data/'train.sdf'),removeHs=False));elements=tuple(manifest['elements'])
    known={Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True) for m in source};start=time.perf_counter();library=build_library(source)
    library_path=a.output/'fragment_library.json';library_path.write_text(json.dumps(library,indent=2)+'\n');library_seconds=time.perf_counter()-start
    sampler=FragmentSampler(library);domain=MorganDomain.load(a.domain);models={};inputs={'training':a.data/'train.sdf','targets':a.data/'manifest.json','domain':a.domain,'model_protocol':a.models,'fragment_library':library_path}
    for name,entry in spec.items():
        inputs[name+'/oracle']=entry['oracle']
        if entry['backend']=='fingerprint':models[name]=FingerprintOracle.load_from_dir(entry['oracle'])
        else:
            inputs[name+'/forcefield']=entry['forcefield'];ff=build_forcefield(entry['backend'],entry['forcefield'],elements,'cpu',tier='presto',route='reference')
            models[name]=InternalDPA4PropertyOracle.load_from_dir(entry['oracle'],ff)
    assert all(model.target_cols==['homo','lumo'] for model in models.values())
    record_input_hashes(a.output,inputs);records=[];pool_costs=[];molecules=[];raw_coordinates={}
    for seed in [2026,2027,2028]:
        start=time.perf_counter();rows=[];mols=[];indices=[]
        for i in range(64):
            stream=candidate_seed(seed,i);row={'seed':seed,'candidate_id':i,'valid':False,'domain':False};rows.append(row)
            try:
                smiles,action=sampler.sample(stream);row.update(smiles=smiles,action=action)
                mol=Chem.AddHs(Chem.MolFromSmiles(smiles));row['proposed_atoms']=mol.GetNumAtoms()
                if mol.GetNumAtoms()>80:raise ValueError('atom_limit')
                if set(at.GetSymbol() for at in mol.GetAtoms())-set(elements):raise ValueError('element_limit')
                params=AllChem.ETKDGv3();params.randomSeed=int(stream%(2**31-1));params.maxIterations=300
                if AllChem.EmbedMolecule(mol,params):raise ValueError('etkdg_failed')
                xyz=mol.GetConformer().GetPositions()
                key=f'coordinates_{seed}_{i}';raw_coordinates[key]=xyz.copy()
                row.update(conformer_coordinates_key=key,conformer_symbols=[at.GetSymbol() for at in mol.GetAtoms()])
                if np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1).max()>manifest['rotation_safe_span_A']:raise ValueError('spatial_limit')
                instance=rdkit_mol_to_mol3d_instance(mol,np.empty(0),element_table=elements)
                if instance is None:raise ValueError('geometry_admission')
                mol=mol3d_to_validated_rdkit_mol(instance);row.update(valid=True,smiles=instance.smiles,atoms=len(instance.coords),novel=instance.smiles not in known)
                row['domain']=domain.assess(mol)['in_domain'];indices.append(i);mols.append(mol)
                mol.SetProp('ProposalID',f'{seed}:{i}');mol.SetProp('GeometryOrigin','ETKDGv3 only; no UFF/MMFF post-optimization or MLIP optimization');molecules.append(mol)
            except (ValueError,RuntimeError) as error:row['proposal_error']=str(error)
        proposal_seconds=time.perf_counter()-start;evaluation_start=time.perf_counter();evaluation_costs={}
        for name,model in models.items():
            started=time.perf_counter();result=model.evaluate_graph_mols(mols) if isinstance(model,FingerprintOracle) else model.evaluate_3d_mols_batched(mols)
            for i,pred,error in zip(indices,result['mean'],result['errors']):
                if error:rows[i][name+'_error']=error
                else:rows[i][name+'_prediction_eV']=pred.tolist()
            if name=='dpa':
                for i,force,error in zip(indices,result['forces_eV_A'],result['errors']):
                    if not error:rows[i]['dpa_rms_force_eVA']=float(np.sqrt(np.mean(np.sum(force**2,axis=-1))))
            evaluation_costs[name]={'attempts':len(mols),'wall_seconds':time.perf_counter()-started}
        records.extend(rows);pool_costs.append({'seed':seed,'proposal_seconds':proposal_seconds,'evaluation_seconds':time.perf_counter()-evaluation_start,'models':evaluation_costs})
        print(seed,'valid',len(mols),'domain',sum(r['domain'] for r in rows),'proposal_seconds',proposal_seconds,flush=True)
        (a.output/'records.json').write_text(json.dumps(records,indent=2)+'\n')
        np.savez_compressed(a.output/'raw_conformers.npz',**raw_coordinates)
    scores=[]
    for column,target in enumerate(['homo','lumo']):
        for q,goal in manifest['single_target_points'][target].items():
            for seed in [2026,2027,2028]:
                pool=[r for r in records if r['seed']==seed];eligible=[r for r in pool if r['domain'] and 'dpa_prediction_eV' in r]
                order=sorted(eligible,key=lambda r:(abs(r['dpa_prediction_eV'][column]-goal),r['candidate_id']));unique=[];seen=set()
                for r in order:
                    if r['smiles'] not in seen:seen.add(r['smiles']);unique.append(r)
                for scope,subset in [('raw',eligible),('top16',unique[:16])]:
                    result={'target':target,'quantile':q,'goal_eV':goal,'seed':seed,'scope':scope,'attempted':64,'selected':len(subset),'models':{}}
                    for model in models:
                        measured=[r for r in subset if model+'_prediction_eV' in r];error=[abs(r[model+'_prediction_eV'][column]-goal) for r in measured]
                        result['models'][model]={'evaluated':len(error),'target_absolute_error_sum_eV':sum(error),'target_mae_eV':float(np.mean(error)) if error else None,
                            'property_hits':sum(e<=.25 for e in error),'property_and_dpa_force_hits':sum(e<=.25 and r['dpa_rms_force_eVA']<=.1 for e,r in zip(error,measured))}
                    scores.append(result)
    write_precise_sdf(a.output/'raw_valid.sdf',molecules)
    (a.output/'summary.json').write_text(json.dumps({'scope':'Three unconditional pools, 192 unique attempts reused for ten point tasks. One BRICS pair and one ETKDGv3 attempt per proposal; no UFF/MMFF, force relaxation, resampling or property-conditioned proposal. All property models frozen.',
        'attempted':len(records),'valid':sum(r['valid'] for r in records),'domain':sum(r['domain'] for r in records),
        'aggregate':summarize_scores(scores),'library_fragments':len(library['fragments']),'library_seconds':library_seconds,'pool_costs':pool_costs,'scores':scores,'records':records},indent=2)+'\n')
    print('BRICS point baseline complete',flush=True)


if __name__=='__main__':main()
