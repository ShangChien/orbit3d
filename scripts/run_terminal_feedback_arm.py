"""Run one predeclared terminal-feedback arm from a common point-SFT checkpoint."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import torch
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['structure','physical','property','sft'],required=True)
    p.add_argument('--groups',type=int,required=True);p.add_argument('--time-reference',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    inputs={'base_config':a.base_config,'initial':a.initial,'manifest':a.manifest,'time_reference':a.time_reference}
    record_input_hashes(a.output,inputs)
    cfg=json.loads(a.base_config.read_text());cfg.update(mode='train' if a.arm=='sft' else 'align',seed=42,device='cuda:0')
    points=json.loads(a.manifest.read_text())['single_target_points']['homo']
    settings=cfg['generator_config'];settings.update(train_backbone=False,train_property_head=False,understanding_weight=0,
        batch_size=8,sft_lr=2e-5,lr_schedule='constant',checkpoint_every=0,group_size=32,
        grpo_steps=0 if a.arm=='sft' else a.groups,grpo_lr=2e-5,kl_beta=.02,replay_weight=.1,
        physics_weight=0,uncertainty_weight=0,point_advantage_mode='joint',point_reward_type='inverse_point_distance',
        alignment_checkpoint_every=4,alignment_checkpoints=[],active_targets=['homo'],
        point_target_pool=[{'homo':points[str(q)]} for q in [25,50,75]])
    if a.arm=='sft':
        if a.time_reference is None:p.error('Time-matched SFT requires the completed property-arm checkpoint')
        state=torch.load(a.time_reference,map_location='cpu',weights_only=False,mmap=True)
        settings.update(sft_steps=1000000,sft_wall_seconds=state['alignment_invocation_seconds']);del state
        settings.pop('terminal_feedback_level',None)
    else:settings['terminal_feedback_level']=a.arm
    cfg['mount_paths'].pop('alignment_resume_path',None)
    cfg['mount_paths'].update(checkpoint_load_path=str(a.initial),checkpoint_save_dir=str(a.output/'weights'),output_dir=str(a.output/'run'))
    cfg['inference_config'].update(target_point={'homo':points['50']},target_tolerance={'homo':.25})
    config=a.output/'config.json';config.write_text(json.dumps(cfg,indent=2)+'\n');started=time.perf_counter()
    with (a.output/'train.log').open('x') as log:
        result=subprocess.run([sys.executable,'-u','-m','orbit3d.cli','--config',str(config)],stdout=log,stderr=subprocess.STDOUT)
    costs={'arm':a.arm,'process_wall_seconds':time.perf_counter()-started,'returncode':result.returncode,'groups':a.groups if a.arm!='sft' else None}
    (a.output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n');result.check_returncode()
    path=a.output/'weights'/('octree_conditional.pt' if a.arm=='sft' else 'octree_aligned.pt')
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    costs['pipeline_seconds']=state['training_wall_seconds'] if a.arm=='sft' else state['alignment_invocation_seconds']
    costs['optimizer_updates']=state['optimizer_updates'] if a.arm=='sft' else sum(r['optimizer_updates'] for r in state['grpo_history'])
    (a.output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n');print(costs,flush=True)


if __name__=='__main__':main()
