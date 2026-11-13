"""Isolate reciprocal conditioning from Gaussian label noise, with step/time SFT controls."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
import torch
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-config',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=1000)
    a=p.parse_args();initial_state=torch.load(a.initial,map_location='cpu',weights_only=False,mmap=True)
    if initial_state['backend']!='uni3dar_octree_v4':raise ValueError('Point bridge requires a v4 point checkpoint')
    del initial_state
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    record_input_hashes(a.output,{'base_config':a.base_config,'initial':a.initial})
    base=json.loads(a.base_config.read_text());base.update(mode='train',device='cuda:0',seed=a.seed)
    base['generator_config'].update(condition_version=4,grpo_steps=0,train_backbone=False,
        train_property_head=False,understanding_weight=0,batch_size=8,sft_lr=2e-5,
        lr_schedule='constant',checkpoint_every=0,sft_steps=a.steps,contrast_pairing_sigma_eV=.25)
    base['mount_paths'].pop('alignment_resume_path',None)
    base['generator_config'].pop('sft_wall_seconds',None)
    configs=a.output/'configs';configs.mkdir();costs=[];checkpoints={'point_frozen':str(a.initial)};budget=None
    for name,contrast in [('point_sft',False),('point_contrast',True),('point_time',False)]:
        cfg=copy.deepcopy(base);cfg['generator_config']['condition_contrast_weight']=float(contrast)
        if name=='point_time':cfg['generator_config'].update(sft_steps=1000000,sft_wall_seconds=budget)
        cfg['mount_paths'].update(checkpoint_load_path=str(a.initial),checkpoint_save_dir=str(a.output/name/'weights'),output_dir=str(a.output/name/'run'))
        path=configs/f'{name}.json';path.write_text(json.dumps(cfg,indent=2)+'\n');started=time.perf_counter()
        with (a.output/f'{name}.log').open('x') as stream:
            result=subprocess.run([sys.executable,'-u','-m','orbit3d.cli','--config',str(path)],stdout=stream,stderr=subprocess.STDOUT)
        costs.append({'name':name,'wall_seconds':time.perf_counter()-started,'returncode':result.returncode})
        (a.output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n');result.check_returncode()
        checkpoint=a.output/name/'weights/octree_conditional.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        costs[-1].update(optimizer_updates=state['optimizer_updates'],pipeline_training_seconds=state['training_wall_seconds'])
        if contrast:budget=state['training_wall_seconds']
        del state
        checkpoints[name]=str(checkpoint)
        (a.output/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
        (a.output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
        print(name,costs[-1],flush=True)


if __name__=='__main__':main()
