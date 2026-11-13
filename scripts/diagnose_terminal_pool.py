"""Freeze unranked point-SFT proposals and measure a budgeted terminal DPA gate."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.generator.molecular_state import mol3d_to_validated_rdkit_mol
from orbit3d.pipeline.evaluators import load_evaluators
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.physics.terminal_evaluation import evaluate_terminal


def summarize(records):
    valid=[r for r in records if r['initial_valid']]
    accepted=[r for r in records if r.get('physical_accepted')]
    evaluated=[r for r in accepted if 'final' in r]
    paired=[r for r in records if 'initial' in r and 'final' in r]
    error=lambda r,stage:abs(r[stage]['prediction_eV'][['homo','lumo'].index(r['target'])]-r['goal_eV'])
    return {'attempted':len(records),'initial_valid':len(valid),'physical_accepted':len(accepted),
        'accepted_property_evaluated':len(evaluated),'joint_hits':sum(error(r,'final')<=.25 for r in evaluated),
        'accepted_mae_eV':float(np.mean([error(r,'final') for r in evaluated])) if evaluated else None,
        'paired_geometry_valid_count':len(paired),
        'paired_initial_mae_eV':float(np.mean([error(r,'initial') for r in paired])) if paired else None,
        'paired_final_mae_eV':float(np.mean([error(r,'final') for r in paired])) if paired else None,
        'energy_force_calls':sum(r['total_dpa_calls'] for r in records),
        'seconds':sum(r['seconds'] for r in records),
        'seconds_per_physical_accepted':sum(r['seconds'] for r in records)/len(accepted) if accepted else None,
        'zero_or_one_accepted_groups':sum(sum(r.get('physical_accepted',False) for r in records if r['group']==g)<=1 for g in {r['group'] for r in records}),
        'groups':len({r['group'] for r in records})}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--target',choices=['homo','lumo'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--arm',default='point_frozen');p.add_argument('--training-seed',type=int,default=42)
    p.add_argument('--inference-seeds',type=int,nargs='+',default=[2071]);a=p.parse_args()
    lock_deterministic_environment(42);torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    source=json.loads(a.manifest.read_text());elements=tuple(source['elements']);inputs={'manifest':a.manifest};pools=[]
    for q in [25,50,75]:
        for seed in a.inference_seeds:
            pool=a.root/f'eval_{a.target}_seed{a.training_seed}'/f'eval_{a.arm}_p{q}_wNone_s{seed}_correct'
            for name in ['run_config.json','raw_symbols.json','raw_trajectories.npz']:inputs[f'{q}/{seed}/{name}']=pool/'run'/name
            pools.append((q,seed,pool))
    cfg=json.loads((pools[0][2]/'run/run_config.json').read_text());cfg['device']=a.device
    inputs.update(forcefield=cfg['mount_paths']['forcefield_path'],oracle=cfg['mount_paths']['property_checkpoint_dir'])
    record_input_hashes(a.output,inputs)
    (a.output/'protocol.json').write_text(json.dumps({'selection':'All 64 proposals per P25/P50/P75 pool. No score, domain or physical preselection.','arm':a.arm,'training_seed':a.training_seed,'inference_seeds':a.inference_seeds,'attempted':192*len(a.inference_seeds),'method':'lbfgs','maximum_total_dpa_calls':150,'optimizer_calls':148,'endpoint_measurement_calls':2,'fmax_eVA':.1,'maxstep_A':.2,'gate':'Initial chemistry/geometry valid; optimizer force converged and graph/stereo preserved. Property benefit only after physical gate.','target':a.target},indent=2)+'\n')
    field,oracle,domain=load_evaluators(cfg,elements);records=[];exports=[];coordinates={}
    for q,seed,pool in pools:
        symbols=json.loads((pool/'run/raw_symbols.json').read_text());assert len(symbols)==64
        with np.load(pool/'run/raw_trajectories.npz') as arrays:
            for i,atoms in enumerate(symbols):
                started=time.perf_counter();row={'target':a.target,'quantile':q,'candidate_id':i,'group':f'{q}/{seed}/{i//32}','inference_seed':seed,'source_pool':str(pool),'goal_eV':source['single_target_points'][a.target][str(q)],'atoms':len(atoms),'initial_valid':False,'physical_accepted':False,'total_dpa_calls':0}
                try:
                    inst=trajectory_to_instance({'atom_symbols':atoms,'positions_np':arrays[f'coordinates_{i}']},element_table=elements)
                    graph=mol3d_to_validated_rdkit_mol(inst);row.update(initial_valid=True,smiles=inst.smiles)
                    try:row['domain']=domain.assess(graph) if domain else None
                    except ValueError as error:row['domain_error']=str(error)
                    final,terminal=evaluate_terminal(inst,field,oracle);row.update(terminal)
                    key=f'p{q}_s{seed}_{i}';coordinates[key]=final.coords;row['coordinates_key']=key
                    info=row.get('optimization',{})
                    if info.get('geometry_valid') and info.get('connectivity_preserved'):
                        mol=mol3d_to_validated_rdkit_mol(final)
                        mol.SetProp('TerminalID',key);mol.SetProp('PhysicalAccepted',str(row['physical_accepted']));exports.append(mol)
                except (ValueError,FloatingPointError,RuntimeError) as error:
                    row['error']=str(error)
                assert row['total_dpa_calls']<=150
                row['seconds']=time.perf_counter()-started;records.append(row)
                (a.output/'records.json').write_text(json.dumps(records,indent=2)+'\n')
                print(q,i,row['initial_valid'],row['physical_accepted'],row['total_dpa_calls'],flush=True)
        np.savez_compressed(a.output/'final_coordinates.npz',**coordinates)
    write_precise_sdf(a.output/'graph_preserved.sdf',exports)
    report={'scope':'Unranked frozen source proposals; MLIP physical gate is not DFT validation. Nonconverged but graph-preserved endpoint predictions are diagnostic only.','target':a.target,'all':summarize(records),'per_quantile':{str(q):summarize([r for r in records if r['quantile']==q]) for q in [25,50,75]},'records':records}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report['all'],flush=True)


if __name__=='__main__':main()
