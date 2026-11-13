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
    p.add_argument('--extra-checkpoints',type=Path,help='Named arm -> checkpoint JSON; also permits explicit control reuse')
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--continued',type=Path,required=True)
    p.add_argument('--continued-wall',type=Path)
    p.add_argument('--independent',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--candidates',type=int,default=64)
    p.add_argument('--arms',nargs='+',default=['frozen','continue400','grpo200','continue100','continue200','grpo50','grpo100'])
    p.add_argument('--quantiles',nargs='+')
    p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    p.add_argument('--skip-off',action='store_true')
    p.add_argument('--reuse-from',type=Path)
    p.add_argument('--time-control',action='store_true')
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
                 'continue_wall':a.continued_wall,
                 'grpo200':a.campaign/'grpo/weights/octree_aligned.pt',
                 'continue100':a.continued.parent/'octree_step_100.pt',
                 'continue200':a.continued.parent/'octree_step_200.pt',
                 'grpo50':a.campaign/'grpo_part100/weights/octree_group_50.pt',
                 'grpo100':a.campaign/'grpo_part100/weights/octree_group_100.pt'}
    if a.extra_checkpoints:
        checkpoints.update({name:Path(path) for name,path in json.loads(a.extra_checkpoints.read_text()).items()})
    if 'continue_wall' in a.arms and (a.continued_wall is None or a.time_control):
        p.error('Evaluate a saved continue_wall arm with --continued-wall; --time-control trains and evaluates it separately')
    inputs=record_input_hashes(a.output,{'source_manifest':data/'manifest.json',
                                 'forcefield':base['mount_paths']['forcefield_path'],
                                 'property_model':base['mount_paths']['property_checkpoint_dir'],
                                 'applicability_domain':base['oracle_config']['applicability_domain_path'],
                                 'independent':a.independent,'frozen':a.frozen,
                                 **{name:checkpoints[name] for name in a.arms}})
    costs=[]
    prior_costs={r['name']:r for r in json.loads((a.reuse_from/'costs.json').read_text()) if r['returncode']==0} if a.reuse_from else {}
    reused={}
    def cost(name,start):
        costs.append({'name':name,'wall_seconds':time.perf_counter()-start,'returncode':0,'device':a.device})
        (a.output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
        print(name,costs[-1]['wall_seconds'],flush=True)
    started=time.perf_counter()
    evaluators=load_evaluators(base,manifest['elements'])
    independent=FingerprintOracle.load_from_dir(a.independent)
    cost('initialize_evaluators',started)
    config_dir=a.output/'configs';config_dir.mkdir()

    def reuse(name,cfg,arm):
        if arm=='continue_wall' or not a.reuse_from or name not in prior_costs or name+'_raw' not in prior_costs:
            return False
        source=a.reuse_from/name
        old=json.loads((source/'run/run_config.json').read_text())
        def inference(config):
            value=copy.deepcopy(config['inference_config']);value.pop('target_interval',None)
            value.setdefault('condition_control','correct');return value
        assert inference(old)==inference(cfg),'Existing inference settings differ'
        assert old['oracle_config']==cfg['oracle_config'] and old['stage5_relax_config']==cfg['stage5_relax_config']
        original=json.loads((source/'run/input_manifest.json').read_text())
        assert original['generator']==inputs[arm]
        for role in ['forcefield','property_model','applicability_domain']:
            assert original[role]==inputs[role],role
        raw_inputs=json.loads((source/'raw/input_manifest.json').read_text())
        assert raw_inputs['independent']==inputs['independent'] and raw_inputs['oracle']==inputs['property_model']
        for stage in ['run','raw']:
            assert json.loads((source/stage/'summary.json').read_text())['attempted']==a.candidates
        (a.output/name).symlink_to(source.resolve(),target_is_directory=True)
        reused[name]={'source':str(source),'settings_and_model_hashes_verified':True}
        (a.output/'reuse_manifest.json').write_text(json.dumps(reused,indent=2)+'\n')
        for key in [name,name+'_raw']:
            costs.append({**prior_costs[key],'reused_from':str(a.reuse_from)})
        (a.output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
        print('reused',name,flush=True)
        return True

    def evaluate(name,checkpoint,prepared,quantile,seed,control='correct',arm='frozen'):
        cfg=copy.deepcopy(base)
        cfg['mount_paths'].update(checkpoint_load_path=str(checkpoint),output_dir=str(a.output/name/'run'),
                                 checkpoint_save_dir=str(a.output/name/'weights'))
        cfg['inference_config'].update(target_point={target:points[quantile]},base_seed=seed,condition_control=control)
        path=config_dir/f'{name}.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
        if reuse(name,cfg,arm):return
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
                evaluate(f'eval_{arm}_p{q}_s{seed}',checkpoint,prepared,q,seed,arm=arm)
        del model,state,prepared
    if not a.skip_off:
        started=time.perf_counter();model,state=load_octree_checkpoint(a.frozen,a.device,a.batch_size)
        prepared=(model,state,a.frozen);cost('load_frozen_off',started)
        for seed in a.seeds:
            evaluate(f'eval_frozen_off_s{seed}',a.frozen,prepared,'50',seed,'off')
        del model,state,prepared
    if a.time_control:
        training_costs=json.loads((a.campaign/'costs.json').read_text())
        budget=sum(next(c['wall_seconds'] for c in training_costs if c['name']==name and c['returncode']==0)
                   for name in ['grpo_part100','grpo'])
        cfg=copy.deepcopy(base);cfg['mode']='train'
        cfg['mount_paths'].update(checkpoint_load_path=str(a.frozen),
                                 checkpoint_save_dir=str(a.output/'continue_wall/weights'),
                                 output_dir=str(a.output/'continue_wall/run'))
        cfg['generator_config'].update(sft_steps=1000000,sft_wall_seconds=budget,sft_lr=2e-5,
                                      batch_size=8,train_backbone=False,train_property_head=False,lr_schedule='constant')
        path=config_dir/'continue_wall.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
        started=time.perf_counter()
        with (a.output/'continue_wall.log').open('x') as log,redirect_stdout(log),redirect_stderr(log):
            result=run_octree_from_config(cfg,str(path))
        cost('continue_wall',started)
        checkpoint=Path(result['checkpoint']);started=time.perf_counter()
        model,state=load_octree_checkpoint(checkpoint,a.device,a.batch_size)
        prepared=(model,state,checkpoint);cost('load_continue_wall',started)
        for q in ['10','25','50','75','90']:
            for seed in a.seeds:
                evaluate(f'eval_continue_wall_p{q}_s{seed}',checkpoint,prepared,q,seed,arm='continue_wall')
    print('Evaluation matrix complete',flush=True)


if __name__=='__main__':main()
