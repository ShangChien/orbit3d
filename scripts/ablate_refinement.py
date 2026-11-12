"""Compare refinement rules on the exact same pre-refinement selected structures."""
import argparse,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from orbit3d.constraints import TargetIntervals
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.generator.molecular_state import ATOM_TYPES,mol3d_to_validated_rdkit_mol
from orbit3d.physics.factory import build_forcefield
from orbit3d.physics.conformer_relaxer import relax_conformer_iso_property,relax_conformer_energy_only,project_force,cap_molecular_displacement
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
p=argparse.ArgumentParser()
p.add_argument('--runs',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--step-size',type=float,help='Override the source run step size; default uses its configuration')
a=p.parse_args();freeze_run(a.output,__file__);lock_deterministic_environment(42);torch.set_num_threads(2)
config=json.loads((a.runs[0]/'run_config.json').read_text());paths=config['mount_paths']
ff=build_forcefield('dpa4',paths['forcefield_path'],ATOM_TYPES,'cuda:0')
oracle=InternalDPA4PropertyOracle.load_from_dir(paths['property_checkpoint_dir'],ff)
targets=TargetIntervals.parse(config['inference_config']['target_interval'],oracle.target_cols)
inputs={'forcefield':paths['forcefield_path'],'oracle':paths['property_checkpoint_dir']}
for i,run in enumerate(a.runs):
 for name in ['raw_trajectories.npz','raw_symbols.json','generated.sdf','candidates.csv','run_config.json']:
  inputs[f'run{i}_{name}']=run/name
record_input_hashes(a.output,inputs)
settings=dict(n_steps=config['stage5_relax_config'].get('relax_steps',30),
              step_size=a.step_size if a.step_size is not None else config['stage5_relax_config'].get('step_size',.004),
              max_atom_disp=.025,prop_grad_weight=.25,property_tolerance=0.,energy_tolerance=0.)
force_threshold=config['inference_config']['max_rms_force_eva']

def evaluate(inst):
 mol3d_to_validated_rdkit_mol(inst)
 x=torch.tensor(inst.coords,device='cuda:0',dtype=torch.float32)
 with torch.no_grad():y,std,energy,force=oracle.forward_shared_force_and_property(x,inst.symbols)
 if not all(torch.isfinite(v).all() for v in [y,std,energy,force]):raise FloatingPointError('Nonfinite physical evaluation')
 return {'prediction':y.cpu().tolist(),'energy_eV':float(energy),'rms_force_eV_A':float(force.square().sum(-1).mean().sqrt()),'property_hit':targets.hit(y)}

def projection_only(inst):
 x=torch.tensor(inst.coords,device='cuda:0',dtype=torch.float32);history=[]
 for step in range(settings['n_steps']):
  xv=x.detach().requires_grad_(True)
  y,_=oracle.forward_differentiable_3d(xv,inst.symbols)
  loss=targets.violations(y).square().sum()
  g=torch.autograd.grad(loss,xv)[0].detach()
  _,_,_,force=oracle.forward_shared_force_and_property(x.detach(),inst.symbols)
  d=project_force(force,g)-settings['prop_grad_weight']*g
  x=x.detach()+cap_molecular_displacement(settings['step_size']*d,settings['max_atom_disp'])
  proposed=replace(inst,coords=x.cpu().numpy())
  value=evaluate(proposed)  # Record failure, never release an invalid final state.
  history.append({'step':step,**value})
 return proposed,{'history':history}

records=[];attempted=0;selected=0
for run in a.runs:
 rows=pd.read_csv(run/'candidates.csv');attempted+=len(rows)
 archive=np.load(run/'raw_trajectories.npz');symbols=json.loads((run/'raw_symbols.json').read_text())
 exports=list(Chem.SDMolSupplier(str(run/'generated.sdf'),removeHs=False))
 ids=[int(m.GetProp('CandidateID')) for m in exports]
 selected+=len(ids)
 for index in ids:
  inst=trajectory_to_instance({'atom_symbols':symbols[index],'positions_np':archive[f'coordinates_{index}']})
  initial=evaluate(inst)
  for rule in ['no_refinement','energy_only','projection_only','full_acceptance']:
   start=time.perf_counter();row={'run':run.name,'candidate_id':index,'rule':rule,'initial':initial}
   try:
    if rule=='no_refinement':result,trace=inst,{}
    elif rule=='energy_only':result,trace=relax_conformer_energy_only(inst,ff,**settings)
    elif rule=='projection_only':result,trace=projection_only(inst)
    else:result,trace=relax_conformer_iso_property(inst,ff,oracle,config['inference_config']['target_interval'],**settings)
    final=evaluate(result)
    row.update(status='completed',final=final,trace=trace,joint_success=final['property_hit'] and final['rms_force_eV_A']<=force_threshold)
    dest=a.output/run.name/rule;dest.mkdir(parents=True,exist_ok=True)
    write_precise_sdf(dest/f'candidate_{index}.sdf',[mol3d_to_validated_rdkit_mol(result)])
   except (ValueError,FloatingPointError) as exc:
    row.update(status='failed',reason=str(exc),joint_success=False)
   row['elapsed_seconds']=time.perf_counter()-start;records.append(row)
   print(run.name,index,rule,row['status'],row['joint_success'],flush=True)
  (a.output/'records.json').write_text(json.dumps(records,indent=2)+'\n')
summary={'attempted_per_rule':attempted,'selected_per_rule':selected,'selection':'Exact pre-refinement coordinates of every exported candidate in each frozen source run; all other attempts remain denominator failures','settings':settings,'max_rms_force_eV_A':force_threshold,'rules':{}}
for rule in ['no_refinement','energy_only','projection_only','full_acceptance']:
 rows=[r for r in records if r['rule']==rule];ok=[r for r in rows if r['status']=='completed']
 summary['rules'][rule]={'completed':len(ok),'failed':len(rows)-len(ok),'joint_successes':sum(r['joint_success'] for r in rows),'property_hits':sum(r['final']['property_hit'] for r in ok),'mean_final_rms_force_eV_A':float(np.mean([r['final']['rms_force_eV_A'] for r in ok])) if ok else None}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
