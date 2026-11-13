"""Evaluate mean control and preference-width response with frozen raw pools."""
import argparse,copy,json,time
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import torch
from orbit3d.generator.octree_backend import load_octree_checkpoint
from orbit3d.pipeline.octree_runner import run_octree_from_config
from orbit3d.pipeline.evaluators import load_evaluators
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
from evaluate_point_pool import evaluate_pool


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--extra-checkpoints',type=Path);p.add_argument('--independent',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--candidates',type=int,default=64)
    p.add_argument('--seeds',nargs='+',type=int,default=[2061,2062]);p.add_argument('--quantiles',nargs='+',default=['10','25','50','75','90'])
    a=p.parse_args();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    checkpoints=json.loads((a.campaign/'checkpoints.json').read_text())
    if a.extra_checkpoints:checkpoints.update(json.loads(a.extra_checkpoints.read_text()))
    base=json.loads((a.campaign/'configs/gaussian.json').read_text());base.update(mode='predict',device=a.device)
    base['generator_config'].update(grpo_steps=0,batch_size=a.candidates)
    base['mount_paths'].pop('hindsight_data_path',None)
    base['inference_config'].update(num_candidates=a.candidates,oracle_candidates=a.candidates,batch_size=a.candidates)
    target=base['generator_config']['active_targets'][0];manifest=json.loads(a.manifest.read_text());points=manifest['single_target_points'][target]
    record_input_hashes(a.output,{'manifest':a.manifest,'fingerprint':a.independent,**checkpoints})
    evaluators=load_evaluators(base,manifest['elements']);fp=FingerprintOracle.load_from_dir(a.independent)
    matrix=[]
    for arm,checkpoint in checkpoints.items():
        model,state=load_octree_checkpoint(checkpoint,a.device,a.candidates);prepared=(model,state,checkpoint)
        requests=[(q,width,seed,'correct') for width in ([.1,.25,.5] if model.condition_version==5 else [None]) for q in a.quantiles for seed in a.seeds]
        requests += [('50',.25 if model.condition_version==5 else None,seed,'off') for seed in a.seeds]
        for q,width,seed,control in requests:
            name=f'eval_{arm}_p{q}_w{width}_s{seed}_{control}';cfg=copy.deepcopy(base)
            cfg['mount_paths'].update(checkpoint_load_path=str(checkpoint),output_dir=str(a.output/name/'run'),checkpoint_save_dir=str(a.output/name/'weights'))
            cfg['inference_config'].update(target_point={target:points[q]},target_tolerance={target:.25},base_seed=seed,condition_control=control)
            if width is not None:cfg['inference_config']['preference_sigma_eV']=width
            path=a.output/(name+'.json');path.write_text(json.dumps(cfg,indent=2)+'\n');start=time.perf_counter()
            with (a.output/(name+'.log')).open('x') as log,redirect_stdout(log),redirect_stderr(log):
                run_octree_from_config(cfg,str(path),prepared_evaluators=evaluators,prepared_generator=prepared)
                evaluate_pool(a.output/name/'run',a.output/name/'raw',a.independent,a.device,
                    prepared_evaluators=evaluators,independent_model=fp,checkpoint_state=state)
            matrix.append({'arm':arm,'target':target,'quantile':q,'goal_eV':points[q],'preference_sigma_eV':width,
                           'seed':seed,'control':control,'pool':name,'attempted':a.candidates,'wall_seconds':time.perf_counter()-start})
            (a.output/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n');print(name,matrix[-1]['wall_seconds'],flush=True)
        del model,state,prepared
    (a.output/'completed.json').write_text(json.dumps({'pools':len(matrix),'attempted':sum(r['attempted'] for r in matrix)})+'\n')


if __name__=='__main__':main()
