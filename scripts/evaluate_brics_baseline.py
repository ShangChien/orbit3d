"""Training-only graph recombination with the mainline Oracle/refinement budgets."""
import argparse,copy,gc,hashlib,json,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from orbit3d.baselines.brics import build_library,FragmentSampler
from orbit3d.constraints import TargetIntervals
from orbit3d.generator.molecular_state import rdkit_mol_to_mol3d_instance,mol3d_to_validated_rdkit_mol
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.pipeline.evaluators import load_evaluators
from orbit3d.physics.batched_relaxer import relax_conformers_iso_property
from orbit3d.utils.conformers import ocelot_conformer
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment


def main():
    p=argparse.ArgumentParser()
    for name in ['training-sdf','reference-checkpoint','config','windows','independent-oracle','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    p.add_argument('--max-atoms',type=int,default=80)
    a=p.parse_args();freeze_run(a.output,__file__)
    record_input_hashes(a.output,{'training_structures':a.training_sdf,'reference_scaler':a.reference_checkpoint,
                                  'configuration':a.config,'target_windows':a.windows,'independent_model':a.independent_oracle})
    torch.set_num_threads(2);lock_deterministic_environment(42)
    cfg=json.loads(a.config.read_text());windows=json.loads(a.windows.read_text())
    state=torch.load(a.reference_checkpoint,map_location='cpu',weights_only=False,mmap=True)
    names,scale=state['target_cols'],np.asarray(state['scale'])
    assert names==cfg['oracle_config']['target_cols']
    elements=tuple(state['representation_config']['allowed_elements'])
    diameter_limit=.24*2**state['representation_config']['merge_level']-.56
    del state
    source=list(Chem.SDMolSupplier(str(a.training_sdf),removeHs=False));start=time.perf_counter()
    training_smiles={Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True) for m in source}
    library=build_library(source);library_seconds=time.perf_counter()-start
    library_path=a.output/'fragment_library.json';library_path.write_text(json.dumps(library,indent=2)+'\n')
    library_hash=hashlib.sha256(library_path.read_bytes()).hexdigest()
    sampler=FragmentSampler(library)
    print('library',len(source),len(library['fragments']),library_seconds,flush=True)
    ff,oracle,domain=load_evaluators(cfg,elements)
    count=int(cfg['inference_config']['num_candidates']);budget=int(cfg['inference_config']['oracle_candidates'])
    requested=int(cfg['inference_config']['num_molecules'])
    assert (count,budget,requested)==(64,32,16)
    runs=[]
    for seed in a.seeds:
        proposals=[];base_records=[];start=time.perf_counter();raw_mols=[]
        for index in range(count):
            stream=candidate_seed(seed,index)
            row={'candidate_id':index,'sampling_seed':stream,'status':'invalid','raw_graph_valid':False,
                 'raw_valid':False,'joint_success':False,'smiles':None,'initial_force':None,'final_force':None,'y_pred':None}
            try:
                smiles,action=sampler.sample(stream);row.update(action=action,smiles=smiles,raw_graph_valid=True)
                graph=Chem.AddHs(Chem.MolFromSmiles(smiles))
                if graph.GetNumAtoms()>a.max_atoms:raise ValueError('Proposed graph exceeds the common atom limit')
                if set(atom.GetSymbol() for atom in graph.GetAtoms())-set(elements):raise ValueError('Unsupported proposal element')
                molecule=ocelot_conformer(smiles,seed=stream%(2**31-1),method='uff')
                xyz=molecule.GetConformer().GetPositions()
                if np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1).max()>diameter_limit:
                    raise ValueError('Proposed conformer exceeds the common rotation-safe spatial span')
                inst=rdkit_mol_to_mol3d_instance(molecule,np.empty(0),element_table=elements)
                inst=replace(inst,graph=mol3d_to_validated_rdkit_mol(inst))
                row.update(raw_valid=True,status='not_selected',
                           charged_atoms=sum(atom.GetFormalCharge()!=0 for atom in inst.graph.GetAtoms()),
                           atom_count=len(inst.coords),novel_vs_training=inst.smiles not in training_smiles)
                inst.graph.SetProp('CandidateID',str(index));raw_mols.append(inst.graph)
                assessment=domain.assess(inst.graph) if domain else {'in_domain':True}
                row['domain_assessment']=assessment
                if assessment['in_domain']:proposals.append((index,inst))
                else:row['status']='out_of_oracle_domain'
            except (ValueError,RuntimeError) as error:
                row['reason']=str(error)
            base_records.append(row)
        proposal_seconds=time.perf_counter()-start
        pool=a.output/f'proposal_seed{seed}';pool.mkdir()
        write_precise_sdf(pool/'raw_valid.sdf',raw_mols)
        (pool/'records.json').write_text(json.dumps(base_records,indent=2)+'\n')
        for window,bounds in windows.items():
            started=time.perf_counter();run=a.output/f'{window}_seed{seed}';freeze_run(run,__file__)
            current=copy.deepcopy(cfg);current['mode']='predict'
            current['generator_config']={'backend':'brics_single_cut_baseline','label_usage':'none'}
            current['inference_config'].update(target_interval=bounds,base_seed=seed)
            current['mount_paths']['output_dir']=str(run)
            (run/'run_config.json').write_text(json.dumps(current,indent=2)+'\n')
            records=copy.deepcopy(base_records);intervals=TargetIntervals.parse(bounds,names)
            shortlist=proposals[:budget];ranked=[]
            if shortlist:
                evaluated=oracle.evaluate_3d_mols_batched([inst.graph for _,inst in shortlist])
                for (index,inst),prediction,std,error in zip(shortlist,evaluated['mean'],evaluated['std'],evaluated['errors']):
                    if error:records[index].update(status='oracle_failed',reason=error);continue
                    score=float(np.square(intervals.violations(prediction).numpy()/scale).sum())
                    ranked.append((score,float(np.mean(std/scale)),index,inst))
                    records[index].update(oracle_rank_score=score,oracle_rank_prediction=np.asarray(prediction).tolist())
            selected=[];seen=set()
            for score,spread,index,inst in sorted(ranked,key=lambda row:row[:3]):
                if inst.smiles in seen:records[index]['status']='duplicate'
                elif len(selected)<requested:selected.append((index,inst));seen.add(inst.smiles)
            results=relax_conformers_iso_property([inst for _,inst in selected],ff,oracle,bounds,
                      n_steps=30,step_size=.004,property_tolerance=0.,energy_tolerance=0.)
            exported=[];qualified=[];traces=[]
            for (index,inst),(out,info) in zip(selected,results):
                traces.append({'candidate_id':index,**info})
                records[index].update(evaluation_calls=info.get('evaluation_calls',0),gradient_calls=info.get('gradient_calls',0))
                if out is None:records[index].update(status='refinement_failed',reason=info['reason']);continue
                mol=mol3d_to_validated_rdkit_mol(out)
                success=bool(info['interval_hit'] and info['final_force_eva']<=current['inference_config']['max_rms_force_eva'])
                for key,value in {'CandidateID':str(index),'PredictedValue':json.dumps(info['y_pred']),'JointSuccess':str(success)}.items():mol.SetProp(key,value)
                records[index].update(status='refined',joint_success=success,y_pred=json.dumps(info['y_pred']),
                                      initial_force=info['init_force_eva'],final_force=info['final_force_eva'])
                exported.append(mol)
                if success:qualified.append(mol)
            write_precise_sdf(run/'generated.sdf',exported);write_precise_sdf(run/'qualified.sdf',qualified)
            if exported:
                reloaded=list(Chem.SDMolSupplier(str(run/'generated.sdf'),removeHs=False));prediction,_=oracle.predict_3d_mols(reloaded)
                np.testing.assert_allclose(prediction,[json.loads(m.GetProp('PredictedValue')) for m in exported],atol=1e-5,rtol=1e-6)
                for before,after in zip(exported,reloaded):
                    np.testing.assert_allclose(before.GetConformer().GetPositions(),after.GetConformer().GetPositions(),atol=1e-9,rtol=0)
            pd.DataFrame(records).to_csv(run/'candidates.csv',index=False)
            (run/'proposal_records.json').write_text(json.dumps(records,indent=2)+'\n')
            (run/'relaxation_traces.json').write_text(json.dumps(traces,indent=2)+'\n')
            summary={'backend':'brics_single_cut_baseline','window':window,'inference_seed':seed,'attempted':count,
                     'raw_graph_valid':sum(r['raw_graph_valid'] for r in records),'raw_valid':len(raw_mols),
                     'domain_eligible':len(proposals),'external_ranking_calls':len(shortlist),'retained':len(exported),
                     'qualified':len(qualified),'joint_success_rate':len(qualified)/count,
                     'refinement_evaluation_calls':sum(r.get('evaluation_calls',0) for r in records),
                     'refinement_gradient_calls':sum(r.get('gradient_calls',0) for r in records),
                     'export_recheck_calls':len(exported),'unique_raw_valid':len({r['smiles'] for r in records if r['raw_valid']}),
                     'retained_novel_vs_training':sum(Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True) not in training_smiles for m in exported),
                     'export_prediction_recheck_passed':True,'proposal_seconds':proposal_seconds,
                     'ranking_and_refinement_seconds':time.perf_counter()-started,'library_build_seconds_once':library_seconds,
                     'fragment_library_sha256':library_hash,'records':records,
                     'scope':'Unconditional compatible fragment pairs; same seed pool reused across windows. UFF geometry cost included. One graph proposal per candidate, no geometry retries.'}
            (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_generated.py')),'--oracle',str(a.independent_oracle),'--runs',str(run)],check=True)
            summary['independent']=json.loads((run/'independent_evaluation.json').read_text());runs.append(summary)
            (a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
            print(window,seed,summary['raw_valid'],summary['domain_eligible'],summary['qualified'],flush=True)
            gc.collect()
    print('BRICS baseline complete',flush=True)


if __name__=='__main__':main()
