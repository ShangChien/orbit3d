"""Rescore unchanged exported molecules with a separately trained 3D evaluator."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from orbit3d.constraints import TargetIntervals
from orbit3d.generator.octree_tokenizer import ATOM_TYPES
from orbit3d.pipeline.evaluators import load_evaluators
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--weight',type=Path,required=True)
    p.add_argument('--domain',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['dpa4','hyff'],required=True);p.add_argument('--hyff-tier',default='presto')
    p.add_argument('--device',default='cuda:0');p.add_argument('--runs',type=Path,nargs='+',required=True)
    a=p.parse_args();freeze_run(a.output,__file__);lock_deterministic_environment(42);torch.set_num_threads(2)
    if len({p.name for p in a.runs})!=len(a.runs):raise ValueError('Run names must be unique within one comparison')
    meta=json.loads((a.oracle/'oracle_meta.json').read_text());names=meta['target_cols'];units=meta['target_units']
    cfg={'device':a.device,'mount_paths':{'forcefield_path':str(a.weight),'property_checkpoint_dir':str(a.oracle)},
         'oracle_config':{'oracle_mode':'internal_'+a.backend,'target_cols':names,'target_units':units,
                          'applicability_domain_path':str(a.domain),'hyff_tier':a.hyff_tier}}
    ff,oracle,domain=load_evaluators(cfg,ATOM_TYPES)
    inputs={'property_model':a.oracle,'forcefield':a.weight,'domain':a.domain}
    for run in a.runs:
        for filename in ['generated.sdf','candidates.csv','run_config.json','summary.json']:
            inputs[run.name+'/'+filename]=run/filename
    record_input_hashes(a.output,inputs)
    summaries=[]
    for run in a.runs:
        source=json.loads((run/'run_config.json').read_text())
        source_names=source['oracle_config']['target_cols'];source_units=source['oracle_config']['target_units']
        indices=[names.index(name) for name in source_names]
        if [units[i] for i in indices]!=source_units:raise ValueError('Independent and source target units differ')
        intervals=TargetIntervals.parse(source['inference_config']['target_interval'],names)
        table=pd.read_csv(run/'candidates.csv').set_index('candidate_id')
        attempted=json.loads((run/'summary.json').read_text())['attempted']
        if len(table)!=attempted or not table.index.is_unique:raise ValueError('Source attempt ledger is incomplete or duplicated')
        sdf=run/'generated.sdf'
        molecules=list(Chem.SDMolSupplier(str(sdf),removeHs=False)) if sdf.stat().st_size else []
        if any(m is None for m in molecules):raise ValueError('Unreadable source export')
        evaluation=oracle.evaluate_3d_mols_batched(molecules) if molecules else None
        records=[];arrays={};threshold=source['inference_config']['max_rms_force_eva']
        for index,mol in enumerate(molecules):
            cid=int(mol.GetProp('CandidateID'));old=np.atleast_1d(json.loads(mol.GetProp('PredictedValue')))
            if old.shape!=(len(source_names),):raise ValueError('Source target order/shape is inconsistent')
            row={'candidate_id':cid,'source_prediction':dict(zip(source_names,old.tolist())),
                 'source_rms_force_eV_A':float(table.loc[cid,'final_force']),'evaluation_error':evaluation['errors'][index],
                 'property_hit':False,'joint_with_source_force':False,'joint_with_independent_force':False}
            force=evaluation['forces_eV_A'][index]
            if force is not None:arrays['forces_'+str(cid)]=np.asarray(force)
            assessment=domain.assess(mol);row['independent_domain']=assessment
            if not row['evaluation_error']:
                prediction=evaluation['mean'][index]
                rms=float(np.sqrt(np.square(np.asarray(force)).sum(-1).mean()))
                hit=intervals.hit(prediction)
                row.update(prediction=dict(zip(names,prediction.tolist())),std=dict(zip(names,evaluation['std'][index].tolist())),
                           independent_rms_force_eV_A=rms,property_hit=hit,
                           joint_with_source_force=bool(hit and assessment['in_domain'] and row['source_rms_force_eV_A']<=threshold),
                           joint_with_independent_force=bool(hit and assessment['in_domain'] and rms<=threshold),
                           absolute_disagreement_eV=dict(zip(source_names,np.abs(prediction[indices]-old).tolist())))
            records.append(row)
        valid=[r for r in records if not r['evaluation_error']]
        summary={'source_run':run.name,'evaluator':'independently trained '+a.backend+' property adapter; no DFT truth',
                 'attempted':len(table),'exported':len(molecules),'prediction_failures':len(records)-len(valid),
                 'independent_domain_eligible':sum(r['independent_domain']['in_domain'] for r in records),
                 'independent_property_hits_without_domain_gate':sum(r['property_hit'] for r in records),
                 'independent_joint_with_source_force':sum(r['joint_with_source_force'] for r in records),
                 'independent_joint_with_own_force':sum(r['joint_with_independent_force'] for r in records),
                 'force_threshold_eV_A':threshold,'target_intervals':source['inference_config']['target_interval'],
                 'mean_absolute_disagreement_eV':{n:float(np.mean([r['absolute_disagreement_eV'][n] for r in valid])) for n in source_names} if valid else None,
                 'records':records}
        np.savez_compressed(a.output/(run.name+'_raw_forces.npz'),**arrays)
        (a.output/(run.name+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
        summaries.append(summary);(a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print(run.name,summary['independent_joint_with_source_force'],summary['independent_joint_with_own_force'],flush=True)


if __name__=='__main__':main()
