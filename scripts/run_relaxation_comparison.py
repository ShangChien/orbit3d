"""Matched starting-point comparison of current refinement and standard energy minimizers."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from orbit3d.constraints import TargetPoints
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.generator.molecular_state import mol3d_to_validated_rdkit_mol
from orbit3d.physics.factory import build_forcefield
from orbit3d.physics.conformer_relaxer import relax_conformer_iso_property,relax_conformer_energy_only
from orbit3d.physics.standard_relaxer import relax_conformer_standard_energy,check_relaxed_geometry
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def trace_result(entry,energy,force):
    f=torch.as_tensor(force).detach().cpu().numpy();norms=np.linalg.norm(f,axis=-1)
    entry.update(completed=True,energy_eV=float(torch.as_tensor(energy).detach().cpu().sum()),
                 rms_force_eVA=float(np.sqrt(np.mean(norms**2))),max_force_eVA=float(norms.max()))


class TracedField:
    def __init__(self,base,history):
        self.base=base;self.history=history;self.device_str=base.device_str;self.atom_type_map=base.atom_type_map
    def compute_energy_and_forces(self,coords,types,**settings):
        entry={'call':len(self.history)+1,'positions_A':coords.detach().cpu().tolist(),'completed':False};self.history.append(entry)
        energy,force=self.base.compute_energy_and_forces(coords,types,**settings);trace_result(entry,energy,force)
        return energy,force


class TracedOracle:
    def __init__(self,base,history):
        self.base=base;self.history=history;self.target_cols=base.target_cols;self.dpa_guidance=base.dpa_guidance;self.gradient_calls=0
    def validate_molecule(self,mol):return self.base.validate_molecule(mol)
    def forward_shared_force_and_property(self,coords,symbols):
        entry={'call':len(self.history)+1,'positions_A':coords.detach().cpu().tolist(),'completed':False};self.history.append(entry)
        y,s,e,f=self.base.forward_shared_force_and_property(coords,symbols);trace_result(entry,e,f)
        return y,s,e,f
    def forward_differentiable_3d(self,*args,**kwargs):
        self.gradient_calls+=1
        return self.base.forward_differentiable_3d(*args,**kwargs)


def evaluate(model,molecule):
    result=model.evaluate_3d_mols_batched([molecule])
    if result['errors'][0]:raise ValueError(result['errors'][0])
    forces=result['forces_eV_A'][0];norms=np.linalg.norm(forces,axis=-1)
    return {'prediction_eV':result['mean'][0].tolist(),'std_eV':result['std'][0].tolist(),
            'energy_eV':float(result['energy_eV'][0]),'rms_force_eVA':float(np.sqrt(np.mean(norms**2))),
            'max_force_eVA':float(norms.max())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--forcefield',required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--independent-forcefield',required=True);p.add_argument('--independent-oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--limit',type=int)
    p.add_argument('--methods',nargs='+',choices=['none','full','energy_current','lbfgs','fire'],default=['none','full','energy_current','lbfgs','fire'])
    a=p.parse_args();lock_deterministic_environment(42);torch.set_num_threads(a.threads);freeze_run(a.output,__file__)
    (a.output/'run_config.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
    record_input_hashes(a.output,{'cohort':a.cohort/'cohort.json','coordinates':a.cohort/'initial_coordinates.npz',
        'dpa':a.forcefield,'property_oracle':a.oracle,'independent_forcefield':a.independent_forcefield,'independent_oracle':a.independent_oracle})
    cohort=json.loads((a.cohort/'cohort.json').read_text());samples=cohort['samples'][:a.limit];elements=tuple(cohort['elements'])
    ff=build_forcefield('dpa4',a.forcefield,elements,a.device)
    oracle=InternalDPA4PropertyOracle.load_from_dir(str(a.oracle),ff)
    independent_ff=build_forcefield('hyff',a.independent_forcefield,elements,a.device,tier='presto',route='reference')
    independent=InternalDPA4PropertyOracle.load_from_dir(str(a.independent_oracle),independent_ff)
    assert oracle.target_cols==independent.target_cols==cohort['target_names']
    records=[];exports=[];coordinates={};start_all=time.perf_counter()
    with np.load(a.cohort/'initial_coordinates.npz') as source:
        for sample in samples:
            instance=trajectory_to_instance({'atom_symbols':sample['symbols'],'positions_np':source[sample['coordinates_key']]},element_table=elements)
            assert instance.smiles==sample['smiles']
            initial_graph=mol3d_to_validated_rdkit_mol(instance)
            initial={'dpa':evaluate(oracle,initial_graph),'hyff':evaluate(independent,initial_graph)}
            point=TargetPoints.parse(sample['target_point'],sample['target_tolerance'],oracle.target_cols)
            column=oracle.target_cols.index(sample['target']);goal=sample['target_point'][sample['target']]
            for method in a.methods:
                started=time.perf_counter();history=[];row={'id':sample['id'],'target':sample['target'],'policy':sample['policy'],
                    'inference_seed':sample['inference_seed'],'candidate_id':sample['candidate_id'],'method':method,
                    'initial':initial,'completed':False,'joint_success':False,'energy_force_calls':0,'property_gradient_calls':0,'post_property_calls':0,
                    'initial_prediction_max_diff_from_gpu_eV':float(np.max(np.abs(np.array(initial['dpa']['prediction_eV'])-sample['reference_prediction_eV']))),
                    'initial_rms_force_diff_from_gpu_eVA':abs(initial['dpa']['rms_force_eVA']-sample['reference_rms_force_eVA'])}
                try:
                    if method=='none':
                        refined=instance;info={'status':'unchanged','evaluation_calls':0}
                    elif method=='full':
                        traced=TracedOracle(oracle,history)
                        refined,info=relax_conformer_iso_property(instance,ff,traced,dict(zip(point.names,point.intervals().bounds)),n_steps=30,step_size=.004)
                        row['property_gradient_calls']=traced.gradient_calls
                    elif method=='energy_current':
                        refined,info=relax_conformer_energy_only(instance,TracedField(ff,history),n_steps=30,step_size=.004)
                    else:
                        refined,info=relax_conformer_standard_energy(instance,ff,method,max_evaluations=150,fmax=.1,max_step=.2,history=history)
                    info=dict(info);info.pop('history',None);row['optimization']=info
                    row.update(check_relaxed_geometry(initial_graph,refined))
                    key=f"sample_{sample['id']}_{method}";coordinates[key]=refined.coords;row['coordinates_key']=key
                    if row['geometry_valid'] and row['connectivity_preserved']:
                        final_mol=mol3d_to_validated_rdkit_mol(refined)
                        final=initial if method=='none' else {}
                        if method!='none':
                            for backend,model in [('dpa',oracle),('hyff',independent)]:
                                row['post_property_calls']+=1
                                final[backend]=evaluate(model,final_mol)
                        row['final']=final;row['completed']=True
                        for backend in ['dpa','hyff']:
                            row[f'{backend}_target_error_eV']=abs(final[backend]['prediction_eV'][column]-goal)
                            row[f'{backend}_property_hit']=point.intervals().hit(final[backend]['prediction_eV'])
                        row['joint_success']=row['dpa_property_hit'] and row['hyff_property_hit'] and final['dpa']['rms_force_eVA']<=.1
                        final_mol.SetProp('ComparisonID',key);final_mol.SetProp('GeometryOrigin','MLIP optimization of generated proposal; no quantum labels')
                        for backend in ['dpa','hyff']:
                            for name,value in zip(oracle.target_cols,final[backend]['prediction_eV']):final_mol.SetProp(f'{backend}_prediction_{name}_eV',str(value))
                        exports.append(final_mol)
                except (ValueError,FloatingPointError) as error:row['error']=str(error)
                row.update(energy_force_calls=len(history),wall_seconds=time.perf_counter()-started,
                           evaluation_trace=history)
                records.append(row)
                print(sample['id'],method,row['completed'],row.get('final',{}).get('dpa',{}).get('rms_force_eVA'),len(history),flush=True)
                (a.output/'records.json').write_text(json.dumps(records,indent=2)+'\n')
            np.savez_compressed(a.output/'final_coordinates.npz',**coordinates)
    write_precise_sdf(a.output/'graph_preserved.sdf',exports)
    summary={'scope':'Same frozen 48 proposals; original Full/energy descent at 30 steps, standard LBFGS/FIRE at up to 150 energy/force calls; no independent-model selection',
        'executed_samples':len(samples),'methods':a.methods,'device':a.device,'threads':a.threads,'wall_seconds':time.perf_counter()-start_all,
        'shared_initial_evaluation_calls':{'dpa':len(samples),'hyff':len(samples)},'records':records}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Matched geometry comparison complete',flush=True)


if __name__=='__main__':main()
