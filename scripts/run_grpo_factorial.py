"""Complete explicitly listed reward/seed controls without repeating SFT or exporting training-time proposals."""
import argparse,copy,gc,json,subprocess,sys
from pathlib import Path
import torch
from orbit3d.pipeline.octree_runner import run_octree_from_config
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--tasks',type=Path,required=True,help='JSON list of {variant, training_seed}')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--independent-oracle',type=Path,required=True)
a=p.parse_args();freeze_run(a.output,__file__);record_input_hashes(a.output,{'config':a.config,'tasks':a.tasks,'independent_oracle':a.independent_oracle})
base=json.loads(a.config.read_text());tasks=json.loads(a.tasks.read_text())
treatments={'full':{},'no_kl':{'kl_beta':0.},'no_physics':{'physics_weight':0.},'no_replay':{'replay_weight':0.},'uncertainty_on':{'uncertainty_weight':1.}}
base['generator_config'].update(grpo_steps=50,group_size=16,grpo_lr=2e-5,kl_beta=.02,physics_weight=.2,replay_weight=.1,uncertainty_weight=0.)
base['inference_config'].update(electron_parity=0,num_candidates=64,num_molecules=16,oracle_candidates=32,precision='float32',sampling_mode='independent',ranking='external')
base['stage5_relax_config'].update(enabled=True,batch_refinement=True,relax_steps=30,property_tolerance=0.,energy_tolerance=0.)
records=[]
for task in tasks:
 variant,seed=task['variant'],task['training_seed'];setting=treatments[variant]
 cfg=copy.deepcopy(base);cfg.update(seed=seed,mode='align');cfg['generator_config'].update(setting)
 stem=f'{variant}_train{seed}'
 cfg['mount_paths'].update(output_dir=str(a.output/stem/'training'),checkpoint_save_dir=str(a.output/stem/'checkpoints'))
 if cfg['generator_config']['replay_weight']==0:cfg['mount_paths'].pop('train_data_path')
 trained=run_octree_from_config(cfg);checkpoint=trained['checkpoint']
 (a.output/stem/'training'/'alignment_summary.json').write_text(json.dumps({'checkpoint':checkpoint,'updates':len(trained['history'])},indent=2)+'\n')
 gc.collect();torch.cuda.empty_cache()
 for inference_seed in [2026,2027,2028]:
  evaluation=copy.deepcopy(cfg);evaluation['mode']='predict';evaluation['generator_config']['grpo_steps']=0
  evaluation['mount_paths'].pop('train_data_path',None)
  run=a.output/f'{stem}_infer{inference_seed}'
  evaluation['mount_paths'].update(checkpoint_load_path=checkpoint,output_dir=str(run))
  evaluation['inference_config']['base_seed']=inference_seed
  result=run_octree_from_config(evaluation)
  subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_generated.py')),'--oracle',str(a.independent_oracle),'--runs',str(run)],check=True)
  records.append({'variant':variant,'training_seed':seed,'inference_seed':inference_seed,**result,'independent':json.loads((run/'independent_evaluation.json').read_text())})
  (a.output/'runs.json').write_text(json.dumps(records,indent=2)+'\n');gc.collect();torch.cuda.empty_cache()
