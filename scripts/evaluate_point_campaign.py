"""Evaluate fixed policies with reusable read-only models on one GPU worker."""
import argparse,copy,json,time
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import torch
from orbit3d.generator.octree_backend import load_octree_checkpoint
from orbit3d.pipeline.octree_runner import run_octree_from_config
from orbit3d.pipeline.evaluators import load_evaluators
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
from evaluate_point_pool import evaluate_pool


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--continued',type=Path,required=True)
    p.add_argument('--independent',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--candidates',type=int,default=64)
    p.add_argument('--arms',nargs='+',default=['frozen','continue400','grpo200','continue100','continue200','grpo50','grpo100'])
    p.add_argument('--quantiles',nargs='+')
    p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    p.add_argument('--skip-off',action='store_true')
    a=p.parse_args();lock_deterministic_environment(42);torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    (a.output/'worker_config.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
    base=json.loads((a.campaign/'configs/grpo.json').read_text());base['mode']='predict';base['device']=a.device
    base['generator_config']['grpo_steps']=0;base['generator_config']['batch_size']=a.batch_size
    base['mount_paths'].pop('alignment_resume_path',None)
    data=Path(base['mount_paths']['train_data_path']).parent
    manifest=json.loads((data/'manifest.json').read_text())
    active=base['generator_config']['active_targets'];assert len(active)==1
    target=active[0];points=manifest['single_target_points'][target]
    base['inference_config'].update(batch_size=a.batch_size,num_candidates=a.candidates,oracle_candidates=a.candidates)
    checkpoints={'frozen':a.frozen,'continue400':a.continued,
                 'grpo200':a.campaign/'grpo/weights/octree_aligned.pt',
                 'continue100':a.continued.parent/'octree_step_100.pt',
                 'continue200':a.continued.parent/'octree_step_200.pt',
                 'grpo50':a.campaign/'grpo_part100/weights/octree_group_50.pt',
                 'grpo100':a.campaign/'grpo_part100/weights/octree_group_100.pt'}
    record_input_hashes(a.output,{'source_manifest':data/'manifest.json',
                                 **{name:checkpoints[name] for name in a.arms}})
    costs=[]
    def cost(name,start):
        costs.append({'name':name,'wall_seconds':time.perf_counter()-start,'returncode':0,'device':a.device})
        (a.output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
        print(name,costs[-1]['wall_seconds'],flush=True)
    started=time.perf_counter()
    evaluators=load_evaluators(base,manifest['elements'])
    independent=FingerprintOracle.load_from_dir(a.independent)
    cost('initialize_evaluators',started)
    config_dir=a.output/'configs';config_dir.mkdir()

    def evaluate(name,checkpoint,prepared,quantile,seed,control='correct'):
        cfg=copy.deepcopy(base)
        cfg['mount_paths'].update(checkpoint_load_path=str(checkpoint),output_dir=str(a.output/name/'run'),
                                 checkpoint_save_dir=str(a.output/name/'weights'))
        cfg['inference_config'].update(target_point={target:points[quantile]},base_seed=seed,condition_control=control)
        path=config_dir/f'{name}.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
        started=time.perf_counter()
        with (a.output/f'{name}.log').open('x') as log,redirect_stdout(log),redirect_stderr(log):
            run_octree_from_config(cfg,str(path),prepared_evaluators=evaluators,prepared_generator=prepared)
        cost(name,started)
        started=time.perf_counter()
        with (a.output/f'{name}_raw.log').open('x') as log,redirect_stdout(log),redirect_stderr(log):
            evaluate_pool(a.output/name/'run',a.output/name/'raw',a.independent,a.device,
                          prepared_evaluators=evaluators,independent_model=independent,checkpoint_state=prepared[1])
        cost(name+'_raw',started)

    for arm in a.arms:
        started=time.perf_counter();checkpoint=checkpoints[arm]
        model,state=load_octree_checkpoint(checkpoint,a.device,a.batch_size)
        prepared=(model,state,checkpoint);cost('load_'+arm,started)
        quantiles=a.quantiles or (['25','50','75'] if arm in ['continue100','continue200','grpo50','grpo100'] else ['10','25','50','75','90'])
        for q in quantiles:
            for seed in a.seeds:
                evaluate(f'eval_{arm}_p{q}_s{seed}',checkpoint,prepared,q,seed)
        del model,state,prepared
    if not a.skip_off:
        started=time.perf_counter();model,state=load_octree_checkpoint(a.frozen,a.device,a.batch_size)
        prepared=(model,state,a.frozen);cost('load_frozen_off',started)
        for seed in a.seeds:
            evaluate(f'eval_frozen_off_s{seed}',a.frozen,prepared,'50',seed,'off')
    print('Evaluation matrix complete',flush=True)


if __name__=='__main__':main()
