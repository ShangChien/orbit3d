"""Evaluate every saved raw proposal, before selection or refinement."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.constraints import TargetPoints
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.pipeline.evaluators import load_evaluators
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--independent',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
    freeze_run(a.output,__file__);started=time.perf_counter()
    cfg=json.loads((a.run/'run_config.json').read_text());cfg['device']=a.device
    names=cfg['oracle_config']['target_cols']
    point=TargetPoints.parse(cfg['inference_config']['target_point'],cfg['inference_config']['target_tolerance'],names)
    symbols=json.loads((a.run/'raw_symbols.json').read_text())
    state=torch.load(cfg['mount_paths']['checkpoint_load_path'],map_location='cpu',weights_only=False,mmap=True)
    elements=state['representation_config']['allowed_elements'];training=set(state['training_smiles']);del state
    ff,oracle,domain=load_evaluators(cfg,elements);independent=FingerprintOracle.load_from_dir(a.independent)
    assert independent.target_cols==names
    record_input_hashes(a.output,{'run_config':a.run/'run_config.json','raw_coordinates':a.run/'raw_trajectories.npz',
                                 'raw_symbols':a.run/'raw_symbols.json','independent':a.independent,
                                 'oracle':cfg['mount_paths']['property_checkpoint_dir']})
    records=[];mols=[];indices=[]
    with np.load(a.run/'raw_trajectories.npz') as arrays:
        for i,atoms in enumerate(symbols):
            row={'candidate_id':i,'valid':False,'evaluated':False,'independent_evaluated':False}
            try:
                inst=trajectory_to_instance({'atom_symbols':atoms,'positions_np':arrays[f'coordinates_{i}']},element_table=tuple(elements))
                row.update(valid=True,smiles=inst.smiles,atoms=len(atoms),novel_vs_sft_training=inst.smiles not in training,
                           domain=domain.assess(inst.graph) if domain else None)
                mols.append(inst.graph);indices.append(i)
            except ValueError as e:row['reason']=str(e)
            records.append(row)
    if mols:
        ev=oracle.evaluate_3d_mols_batched(mols)
        fp,_=independent.predict_3d_mols(mols)
        for j,i in enumerate(indices):
            row=records[i]
            if np.isfinite(fp[j]).all():row.update(independent_evaluated=True,independent_prediction=fp[j].tolist())
            if ev['errors'][j]:row['evaluation_error']=ev['errors'][j];continue
            pred=ev['mean'][j];force=ev['forces_eV_A'][j]
            row.update(evaluated=True,prediction=np.asarray(pred).tolist(),rms_force=float(np.sqrt(np.square(force).sum(-1).mean())))
    stats={}
    for label,flag,field in [('dpa','evaluated','prediction'),('fingerprint','independent_evaluated','independent_prediction')]:
        eligible=[r for r in records if r[flag]]
        stats[label]={'evaluated':len(eligible),'coverage':len(eligible)/len(records),
                      'property_hits':sum(point.intervals().hit(r[field]) for r in eligible)}
        stats[label]['property_hit_rate_all_attempts']=stats[label]['property_hits']/len(records)
        stats[label]['mean_distance']=float(np.mean([point.distance(r[field]) for r in eligible])) if eligible else None
        stats[label]['targets']={n:{'mae_eV':float(np.mean([abs(r[field][k]-t) for r in eligible])),
                                    'mean_prediction_eV':float(np.mean([r[field][k] for r in eligible]))}
                                  for k,(n,t) in enumerate(zip(names,point.values)) if t is not None and eligible}
        stats[label]['joint_domain_force_hits']=sum(point.intervals().hit(r[field]) and r.get('evaluated',False)
                    and r['rms_force']<=.1 and bool(r.get('domain') and r['domain']['in_domain']) for r in eligible)
    report={'scope':'Raw pool, no ranking or relaxation; all attempts retained; OOD predictions diagnostic only',
            'attempted':len(records),'valid':len(mols),'unique_valid':len({r['smiles'] for r in records if r['valid']}),
            'novel_valid':sum(r.get('novel_vs_sft_training',False) for r in records),
            'oracle_calls':len(mols),'independent_calls':len(mols),'elapsed_seconds':time.perf_counter()-started,
            'metrics':stats,'records':records}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(stats,flush=True)


if __name__=='__main__':main()
