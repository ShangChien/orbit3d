"""Point GRPO against the same frozen B used for labels and acceptance."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from rdkit import Chem
from orbit3d.constraints import TargetPoints
from orbit3d.generator.octree_backend import load_octree_checkpoint
from orbit3d.generator.octree_alignment import align_octree
from orbit3d.oracles.unimol2_task import FrozenUniMol2Task, file_sha256
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--source-train',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True);p.add_argument('--target',choices=['homo','lumo'],required=True)
    p.add_argument('--groups',type=int,default=256);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int)
    p.add_argument('--save-groups',type=int,nargs='*',default=[])
    a=p.parse_args();started=time.perf_counter();lock_deterministic_environment(a.seed);torch.set_num_threads(4)
    # A resumed segment gets its own immutable source/input receipt.
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    (a.output/'run_config.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
    model,state=load_octree_checkpoint(a.checkpoint,'cuda:0',32);model.float().eval()
    task=FrozenUniMol2Task(a.oracle,a.selection)
    if state.get('fixed_predictor_contract')!=task.contract or state['active_targets']!=[a.target]:
        raise ValueError('Alignment requires a matching B-labelled point checkpoint')
    molecules=list(Chem.SDMolSupplier(str(a.source_train),removeHs=False))
    labels=np.asarray([[float(m.GetProp(n)) for n in state['target_cols']] for m in molecules],dtype=np.float32)
    raw=json.loads(a.targets.read_text())['single_target_points'][a.target]
    points=[TargetPoints.parse({a.target:raw[str(q)]},{a.target:.25},state['target_cols']) for q in [10,25,50,75,90]]
    inputs={'checkpoint':a.checkpoint,'train':a.source_train,'oracle':a.oracle,'selection':a.selection,'targets':a.targets,'resume':a.resume}
    record_input_hashes(a.output,inputs)
    resume=torch.load(a.resume,map_location='cuda:0',weights_only=False) if a.resume else None
    previous_seconds=resume.get('training_wall_seconds',0.) if resume else 0.
    previous_calls=resume.get('B_molecule_calls',0) if resume else 0
    previous_forwards=resume.get('B_forward_calls',0) if resume else 0
    cost_curve=list(resume.get('cost_curve',[])) if resume else []
    latest_training_state=None
    def save(path, history):
        result={**state,'model_state_dict':model.state_dict(),'initial_weights_sha256':file_sha256(a.checkpoint),
            'grpo_history':history,'alignment_training_state':latest_training_state,
            'fixed_predictor_contract':task.contract,'train_backbone':False,
            'training_smiles':sorted(set(state.get('training_smiles',[]))|{r['smiles'] for h in history for r in h['rollout_records'] if r['valid']}),
            'training_wall_seconds':previous_seconds+time.perf_counter()-started,
            'B_molecule_calls':previous_calls+task.molecule_calls,'B_forward_calls':previous_forwards+task.forward_calls,
            'cost_curve':cost_curve,
            'optimizer_updates':sum(h['optimizer_updates'] for h in history)}
        temporary=path.with_suffix('.tmp');torch.save(result,temporary);temporary.replace(path)
    def checkpoint(step, model, history, training_state):
        nonlocal latest_training_state
        latest_training_state=training_state
        cost_curve.append(dict(groups=step,raw_proposals=step*32,
            B_molecule_calls=previous_calls+task.molecule_calls,B_forward_calls=previous_forwards+task.forward_calls,
            elapsed_seconds=previous_seconds+time.perf_counter()-started))
        if step in a.save_groups:
            save(a.output/f'octree_group_{step}.pt',history)
        if step%16==0 or step==(a.stop_after or a.groups):
            save(a.output/'resume.pt',history)
            (a.output/'progress.json').write_text(json.dumps(dict(groups=step,raw_proposals=step*32,B_molecule_calls=task.molecule_calls,
                B_forward_calls=task.forward_calls,process_seconds=time.perf_counter()-started),indent=2)+'\n')
    history=align_octree(model,molecules,labels,state['mean'],state['scale'],points[0].intervals(),None,None,
        steps=a.groups,group_size=32,lr=2e-5,seed=a.seed,kl_beta=.02,physics_weight=0,uncertainty_weight=0,replay_weight=.1,
        electron_parity=0,points=points[0],point_pool=points,point_advantage_mode='conditional_property_v1',point_task=task,
        checkpoint_callback=checkpoint,resume_checkpoint=resume,stop_after=a.stop_after)
    save(a.output/'octree_conditional.pt',history)
    (a.output/'costs.json').write_text(json.dumps(dict(segment_seconds=time.perf_counter()-started,
        process_seconds=previous_seconds+time.perf_counter()-started,groups=len(history),
        raw_proposals=len(history)*32,B_molecule_calls=previous_calls+task.molecule_calls,
        B_forward_calls=previous_forwards+task.forward_calls),indent=2)+'\n')
    (a.output/'cost_curve.json').write_text(json.dumps(cost_curve,indent=2)+'\n')


if __name__=='__main__':main()
