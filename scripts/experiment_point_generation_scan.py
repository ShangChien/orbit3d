"""Fresh point-conditioned generation across training quantiles, with stage timings.

Run with the recorded fixed-sampling runtime snapshot, not an older sampler.
Only one trained property mask is activated per checkpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from dataclasses import replace

import numpy as np
import torch
from rdkit import Chem
from experiment_relaxation_priority import Experiment
from orbit3d.constraints import TargetPoints
from orbit3d.generator.octree_backend import load_octree_checkpoint, trajectory_to_instance
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.generator.molecular_state import mol3d_to_validated_rdkit_mol
from orbit3d.pipeline.evaluators import load_evaluators
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def target_metrics(rows, index, target, field='prediction', in_domain=True):
    eligible=[r for r in rows if field in r and (not in_domain or r.get('domain',{}).get('in_domain',False))]
    pred=np.array([r[field][index] for r in eligible])
    error=np.abs(pred-target)
    return {'attempted':len(rows),'evaluated':len(eligible),'coverage':len(eligible)/len(rows) if rows else 0,
            'mae_eV':float(error.mean()) if len(error) else None,
            'mean_prediction_eV':float(pred.mean()) if len(pred) else None,
            'prediction_p10_p50_p90_eV':np.quantile(pred,[.1,.5,.9]).tolist() if len(pred) else None,
            'hits':{str(t):int((error<=t).sum()) for t in [.1,.25,.5]},
            'hits_rms01':sum(abs(r[field][index]-target)<=.25 and r.get('rms_force',float('inf'))<=.1 for r in eligible)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--target',choices=['homo','lumo'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime-commit',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    p.add_argument('--quantiles',nargs='+',default=['10','25','50','75','90'])
    p.add_argument('--candidates',type=int,default=64)
    p.add_argument('--retain',type=int,default=8)
    p.add_argument('--speed-replays',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(2);lock_deterministic_environment(42)
    freeze_run(a.output,__file__)
    helper=Path(__file__).with_name('experiment_relaxation_priority.py');shutil.copy2(helper,a.output/helper.name)
    cfg_path=a.root/f'point_{a.target}_seed42/configs/sft.json'
    cfg=json.loads(cfg_path.read_text());cfg['device']='cuda:0'
    checkpoint=a.root/f'point_{a.target}_seed42/sft/weights/octree_conditional.pt'
    manifest_path=a.root/'data_final/manifest.json';manifest=json.loads(manifest_path.read_text())
    points={q:manifest['single_target_points'][a.target][q] for q in a.quantiles}
    model,state=load_octree_checkpoint(checkpoint,'cuda:0',8)
    assert model.condition_version==4 and state['active_targets']==[a.target]
    assert not state.get('representation_migration',{}).get('requires_adaptation',False)
    assert not (state.get('condition_migration') or {}).get('requires_condition_adaptation',False)
    names=state['target_cols'];index=names.index(a.target)
    mean,scale=state['mean'],state['scale'];training=set(state['training_smiles'])
    elements=tuple(model.args.allow_atoms.split(','));ff,oracle,domain=load_evaluators(cfg,elements)
    fp_dir=a.root/'oracle_fingerprint/oracle';fp=FingerprintOracle.load_from_dir(fp_dir)
    assert fp.target_cols==oracle.target_cols==names
    record_input_hashes(a.output,{'checkpoint':checkpoint,'data_manifest':manifest_path,'sft_config':cfg_path,
        'forcefield':cfg['mount_paths']['forcefield_path'],'oracle':cfg['mount_paths']['property_checkpoint_dir'],
        'domain':cfg['oracle_config']['applicability_domain_path'],'independent':fp_dir,'optimizer_helper':helper})
    protocol={'target':a.target,'points_eV':points,'seeds':a.seeds,'candidates_per_pool':a.candidates,
        'retain':a.retain,'runtime_commit':a.runtime_commit,'trained_steps':len(state.get('history',[])),
        'active_targets':state['active_targets'],'label_theory':state.get('label_theory'),
        'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_name':torch.cuda.get_device_name(),
        'sampling':{'batch_size':8,'precision':'float32','tree_temperature':.85,'atom_temperature':.3,'xyz_temperature':.25,'cfg_scale':1.,'electron_parity':0},
        'scope':'Fresh trajectories per target and seed; common random streams across targets. One frozen trained checkpoint per property, not independent training repetitions.',
        'timing':'CUDA synchronized stage wall time; model load and warmup excluded. Raw evaluation includes in-domain and OOD valid molecules. Geometry only on up to 8 DPA-ranked in-domain correct-condition candidates.',
        'off_control':'One independent unconditional pool per seed, rescored against every requested target; no fake new draws per rescoring.',
        'geometry':'Fixed graph L-BFGS maxstep .025 A, max150 steps; stop RMS<=.1 and maxforce<=.15 eV/A. No target gradient.'}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    del state

    def generate(value,seed,off=False,count=None):
        count=a.candidates if count is None else count
        task=TargetPoints.parse({a.target:value},{a.target:.25},names)
        lo,hi,active=task.arrays(mean,scale)
        if off:active=np.zeros_like(active)
        lock_deterministic_environment(seed);trajectories=[]
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():
            for begin in range(0,count,8):
                n=min(8,count-begin)
                trajectories.extend(model.generate_conditional_trajectories(
                    torch.as_tensor(lo,device='cuda:0').repeat(n,1),torch.as_tensor(hi,device='cuda:0').repeat(n,1),
                    torch.as_tensor(active,device='cuda:0').unsqueeze(0).expand(n,*active.shape),
                    cfg_scale=1.,tree_temp=.85,atom_temp=.3,xyz_temp=.25,electron_parity=0,
                    cache_mode='concat',evaluate_self_consistency=True,
                    row_seeds=[candidate_seed(seed,begin+j) for j in range(n)]))
        torch.cuda.synchronize();return trajectories,time.perf_counter()-start

    # Excluded warmup: generator plus one genuine source structure for the Oracle.
    generate(manifest['single_target_points'][a.target]['50'],1909,count=8)
    first=next(m for m in Chem.SDMolSupplier(str(a.root/'data_final/train.sdf'),removeHs=False) if m is not None)
    oracle.evaluate_3d_mols_batched([first]);fp.predict_3d_mols([first]);torch.cuda.synchronize()
    runs=[];speed=[];reference_tokens=None
    for round_index,seed in enumerate(a.seeds):
        order=list(a.quantiles)+['off']
        order=order[round_index%len(order):]+order[:round_index%len(order)]
        for q in order:
            off=q=='off';value=manifest['single_target_points'][a.target]['50'] if off else points[q]
            dest=a.output/f'seed{seed}_{q}';dest.mkdir()
            trajectories,seconds=generate(value,seed,off)
            raw=[];instances={};arrays={};tokens=[]
            validation_start=time.perf_counter()
            for i,traj in enumerate(trajectories):
                tok=traj['decoder_feat'].detach().cpu().numpy();token_hash=hashlib.sha256(tok.tobytes()).hexdigest()
                tokens.append(token_hash);arrays[f'coords_{i}']=traj['positions_np'];arrays[f'tokens_{i}']=tok
                row={'candidate_id':i,'sampling_seed':candidate_seed(seed,i),'valid':False,'token_sha256':token_hash,'symbols':traj['atom_symbols']}
                try:
                    inst=trajectory_to_instance(traj,element_table=elements)
                    row.update(valid=True,smiles=inst.smiles,atoms=len(inst.coords),novel=inst.smiles not in training,domain=domain.assess(inst.graph))
                    instances[i]=inst
                except ValueError as exc:row['reason']=str(exc)
                raw.append(row)
            np.savez_compressed(dest/'trajectories.npz',**arrays)
            validation_seconds=time.perf_counter()-validation_start
            mols=[v.graph for v in instances.values()];ids=list(instances)
            torch.cuda.synchronize();eval_start=time.perf_counter()
            if mols:
                ev=oracle.evaluate_3d_mols_batched(mols)
                for j,i in enumerate(ids):
                    if ev['errors'][j]:raw[i]['evaluation_error']=ev['errors'][j];continue
                    pred=np.asarray(ev['mean'][j]);f=np.asarray(ev['forces_eV_A'][j])
                    if not np.isfinite(pred).all() or not np.isfinite(f).all():raise FloatingPointError('Nonfinite raw evaluation')
                    raw[i].update(prediction=pred.tolist(),rms_force=float(np.sqrt(np.square(f).sum(-1).mean())))
            torch.cuda.synchronize();evaluation_seconds=time.perf_counter()-eval_start
            fp_start=time.perf_counter()
            if mols:
                fpy,_=fp.predict_3d_mols(mols)
                for i,pred in zip(ids,fpy):
                    if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite independent prediction')
                    raw[i]['independent_prediction']=pred.tolist()
            independent_seconds=time.perf_counter()-fp_start
            task=TargetPoints.parse({a.target:value},{a.target:.25},names)
            ranked=sorted([r for r in raw if 'prediction' in r and r['domain']['in_domain']],key=lambda r:(abs(r['prediction'][index]-value),r['candidate_id']))
            selected=[];seen=set()
            for r in ranked:
                if r['smiles'] in seen:continue
                selected.append(r);seen.add(r['smiles'])
                if len(selected)==a.retain:break
            refined=[];geometry_start=time.perf_counter();torch.cuda.synchronize()
            if not off:
                for row in selected:
                    i=row['candidate_id'];inst=instances[i];exp=Experiment(inst,ff,oracle,task.intervals(),'cuda:0')
                    result={'candidate_id':i,'domain':row['domain'],'before_prediction':row['prediction']}
                    try:
                        coords,trace=exp.minimize(inst.coords,'energy')
                        final=trace['final'];outmol=mol3d_to_validated_rdkit_mol(replace(inst,coords=coords))
                        independent,_=fp.predict_3d_mols([outmol])
                        result.update(status='completed',prediction=final['prediction'],rms_force=final['rms_force_eV_A'],
                                      independent_prediction=independent[0].tolist(),converged=trace['converged'],history=trace['history'])
                        write_precise_sdf(dest/f'refined_{i}.sdf',[outmol])
                    except (ValueError,FloatingPointError,RuntimeError) as exc:result.update(status='failed',reason=str(exc))
                    result['evaluation_calls']=exp.evaluations;refined.append(result)
            torch.cuda.synchronize();geometry_seconds=time.perf_counter()-geometry_start
            report={'seed':seed,'quantile':q,'target_name':a.target,'input_value':None if off else value,
                'scoring_value':value,'condition':'off' if off else 'correct','attempted':len(raw),'valid':len(instances),
                'unique_valid':len({r['smiles'] for r in raw if r['valid']}),'novel_valid':sum(r.get('novel',False) for r in raw),
                'mean_valid_atoms':float(np.mean([r['atoms'] for r in raw if r['valid']])) if instances else None,
                'stage_seconds':{'generation':seconds,'validation_and_raw_export':validation_seconds,'dpa_evaluation':evaluation_seconds,'independent_evaluation':independent_seconds,'geometry':geometry_seconds},
                'raw_metrics':{f'{model_name}/{subset}':target_metrics(raw,index,value,field,subset=='in_domain') for model_name,field in [('dpa','prediction'),('fingerprint','independent_prediction')] for subset in ['in_domain','all_evaluable']},
                'selected_ids':[r['candidate_id'] for r in selected], 'raw_records':raw,'refined_records':refined,
                'refined_metrics':{key:target_metrics(refined,index,value,field) for key,field in [('dpa','prediction'),('fingerprint','independent_prediction')]}}
            (dest/'summary.json').write_text(json.dumps(report,indent=2));runs.append(report)
            if seed==a.seeds[0] and q=='50':
                reference_tokens=tokens;speed.append({'kind':'original_p50','seed':seed,'generation_seconds':seconds})
            print(a.target,seed,q,'valid',len(instances),'domain',report['raw_metrics']['dpa/in_domain']['evaluated'],'MAE',report['raw_metrics']['dpa/in_domain']['mae_eV'],flush=True)
            (a.output/'progress.json').write_text(json.dumps({'completed_pools':len(runs),'expected_pools':len(a.seeds)*(len(a.quantiles)+1)}))
    if '50' in points:
        for repeat in range(a.speed_replays):
            tr,seconds=generate(points['50'],a.seeds[0]);hashes=[hashlib.sha256(t['decoder_feat'].detach().cpu().numpy().tobytes()).hexdigest() for t in tr]
            assert hashes==reference_tokens,'Same-seed speed replay changed trajectories'
            speed.append({'kind':'same_seed_p50_replay','repeat':repeat,'generation_seconds':seconds,'tokens_equal':True})
    final={'protocol':protocol,'runs':runs,'speed_replays':speed}
    (a.output/'summary.json').write_text(json.dumps(final,indent=2))
    print('COMPLETE',a.target,len(runs),flush=True)


if __name__=='__main__':main()
