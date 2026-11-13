"""Run the frozen seed-level mean/width SFT and reciprocal/hindsight factorial."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-config',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--hindsight',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--base-steps',type=int,default=2000);p.add_argument('--arm-steps',type=int,default=1000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    record_input_hashes(a.output,{'base_config':a.base_config,'initial':a.initial,'hindsight':a.hindsight})
    scripts=Path(__file__).resolve().parent;costs=[]
    def execute(name,cmd):
        start=time.perf_counter()
        with (a.output/f'{name}.log').open('x') as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        costs.append({'name':name,'seconds':time.perf_counter()-start,'returncode':result.returncode,'command':cmd})
        (a.output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n');result.check_returncode()
        print(name,'completed',costs[-1]['seconds'],flush=True)
    migrated=a.output/'initial_gaussian.pt'
    execute('migrate',[sys.executable,str(scripts/'migrate_gaussian_conditioner.py'),'--checkpoint',str(a.initial),'--output',str(migrated)])
    base=json.loads(a.base_config.read_text());base.update(mode='train',device=a.device,seed=42)
    base['generator_config'].update(condition_version=5,train_backbone=False,train_property_head=False,
        batch_size=8,sft_lr=2e-5,lr_schedule='constant',grpo_steps=0,checkpoint_every=0,
        understanding_weight=0,preference_sigma_range_eV=[.1,.5])
    base['mount_paths'].pop('alignment_resume_path',None)
    base['generator_config'].pop('sft_wall_seconds',None)
    configs=a.output/'configs';configs.mkdir()
    checkpoints={}
    for name,contrast,hindsight in [('gaussian',False,False),('sft',False,False),('contrast',True,False),('hindsight',False,True),('both',True,True)]:
        cfg=copy.deepcopy(base)
        cfg['generator_config'].update(sft_steps=a.base_steps if name=='gaussian' else a.arm_steps,condition_contrast_weight=1. if contrast else 0.)
        cfg['mount_paths'].update(checkpoint_load_path=str(migrated if name=='gaussian' else checkpoints['gaussian']),
            checkpoint_save_dir=str(a.output/name/'weights'),output_dir=str(a.output/name/'run'))
        if hindsight:cfg['mount_paths']['hindsight_data_path']=str(a.hindsight)
        path=configs/f'{name}.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
        execute(name,[sys.executable,'-u','-m','orbit3d.cli','--config',str(path)])
        checkpoints[name]=str(a.output/name/'weights/octree_conditional.pt')
        (a.output/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2)+'\n')


if __name__=='__main__':main()
