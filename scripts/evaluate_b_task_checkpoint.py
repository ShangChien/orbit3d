"""Fresh point-conditioned trajectories evaluated by B on original coordinates only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from rdkit import Chem

from orbit3d.constraints import TargetPoints
from orbit3d.generator.octree_backend import load_octree_checkpoint, trajectory_to_instance
from orbit3d.generator.molecular_state import mol3d_to_validated_rdkit_mol
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.oracles.unimol2_task import FrozenUniMol2Task, support_exclusion, graph_set, summarize_task
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--source-train',type=Path,required=True);p.add_argument('--asset-sdfs',type=Path,nargs='*',default=[])
    p.add_argument('--target',choices=['homo','lumo'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--quantiles',type=int,nargs='+',default=[10,25,35,50,65,75,90]);p.add_argument('--controls',nargs='+',choices=['correct','off','shuffled'],default=['correct'])
    p.add_argument('--attempts',type=int,default=128);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();lock_deterministic_environment(42);torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    (a.output/'run_config.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
    model,state=load_octree_checkpoint(a.checkpoint,'cuda:0',a.batch_size);model.float().eval()
    if model.condition_version!=4 or state['active_targets']!=[a.target]:raise ValueError('Requires matching single-property point checkpoint')
    judge=FrozenUniMol2Task(a.oracle,a.selection);source=graph_set(a.source_train);assets=set(source)|set(state.get('training_smiles',[]))
    for path in a.asset_sdfs:assets.update(graph_set(path))
    targets=json.loads(a.targets.read_text());points=targets['single_target_points'][a.target];names=state['target_cols'];reports=[];start=time.perf_counter()
    jobs=[(control,q,seed) for control in a.controls for q in ([50] if control=='off' else a.quantiles) for seed in a.seeds]
    for control,q,seed in jobs:
        goal=points[str(q)];input_q=a.quantiles[(a.quantiles.index(q)+len(a.quantiles)//2)%len(a.quantiles)] if control=='shuffled' else q
        task=TargetPoints.parse({a.target:points[str(input_q)]},{a.target:.25},names)
        low,high,active=task.arrays(state['mean'],state['scale'])
        if control=='off':active=np.zeros_like(active)
        name=f'eval_{a.arm}_p{q}_s{seed}_{control}';pool=a.output/name;run=pool/'run';run.mkdir(parents=True);(pool/'raw').mkdir()
        rows=[];xyz={};symbols=[];generation_seconds=0.;scoring_seconds=0.
        for begin in range(0,a.attempts,a.batch_size):
            n=min(a.batch_size,a.attempts-begin);torch.cuda.synchronize();t=time.perf_counter()
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                trajectories=model.generate_conditional_trajectories(torch.tensor(low,device='cuda').repeat(n,1),torch.tensor(high,device='cuda').repeat(n,1),
                    torch.tensor(active,device='cuda').unsqueeze(0).expand(n,*active.shape),cfg_scale=1.,tree_temp=.85,atom_temp=.3,xyz_temp=.25,
                    evaluate_self_consistency=False,row_seeds=[candidate_seed(seed,begin+j) for j in range(n)],electron_parity=0,cache_mode='concat')
            torch.cuda.synchronize();generation_seconds+=time.perf_counter()-t;t=time.perf_counter();mols=[];indices=[]
            for j,traj in enumerate(trajectories):
                i=begin+j;symbols.append(traj['atom_symbols']);xyz[f'coordinates_{i}']=traj['positions_np']
                xyz[f'tokens_{i}']=traj['decoder_feat'].cpu().numpy()
                row=dict(candidate_id=i,atoms=len(traj['atom_symbols']),valid=False,legal=False,evaluated=False,
                         token_sha256=hashlib.sha256(traj['decoder_feat'].cpu().numpy().tobytes()).hexdigest());rows.append(row)
                try:
                    inst=trajectory_to_instance(traj,element_table=tuple(model.args.allow_atoms.split(',')));mol=mol3d_to_validated_rdkit_mol(inst)
                except ValueError as error:row['rejection']=str(error);continue
                row.update(legal=True,valid=True,smiles=inst.smiles);exclusion=support_exclusion(mol)
                if exclusion:row['support_exclusion']=exclusion;continue
                mols.append(mol);indices.append(i)
            pred=judge.predict(mols)
            for i,values in zip(indices,pred):rows[i].update(evaluated=True,prediction_eV=values.tolist())
            scoring_seconds+=time.perf_counter()-t
        np.savez_compressed(run/'raw_trajectories.npz',**xyz);(run/'raw_symbols.json').write_text(json.dumps(symbols)+'\n')
        (run/'run_config.json').write_text(json.dumps(dict(inference_config=dict(target_point={a.target:goal},input_target_point={a.target:points[str(input_q)]},condition_control=control),B_contract=judge.contract),indent=2)+'\n')
        metrics=summarize_task(rows,a.target,goal,source,assets)
        summary=dict(arm=a.arm,target=a.target,quantile=q,goal_eV=goal,input_goal_eV=points[str(input_q)],seed=seed,control=control,attempted=len(rows),
            generation_seconds=generation_seconds,scoring_seconds=scoring_seconds,metrics=metrics,records=rows,target_point={a.target:goal})
        column=names.index(a.target);ranked=sorted([r for r in rows if r['evaluated']],key=lambda r:(abs(r['prediction_eV'][column]-goal),r['candidate_id']))
        summary['top_k']={str(k):dict(requested=k,**summarize_task(ranked[:k],a.target,goal,source,assets)) for k in [4,8]}
        (pool/'raw/summary.json').write_text(json.dumps(summary,indent=2)+'\n');reports.append(summary)
        print(name,metrics,flush=True)
    inputs={'checkpoint':a.checkpoint,'oracle':a.oracle,'selection':a.selection,'targets':a.targets,'source_train':a.source_train}
    inputs.update({str(p):p for p in a.asset_sdfs});record_input_hashes(a.output,inputs)
    (a.output/'summary.json').write_text(json.dumps(dict(predictor=judge.contract,scope='Fresh raw B-defined task. Off and shuffled controls explicitly labelled. No DPA calls or DFT.',pools=reports,
        elapsed_seconds=time.perf_counter()-start,B_molecule_calls=judge.molecule_calls,B_forward_calls=judge.forward_calls),indent=2)+'\n')


if __name__=='__main__':main()
