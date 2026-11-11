"""Repeat full-reward alignment with independent training seeds and fixed evaluation budgets."""
import argparse,copy,gc,json,subprocess,sys
from pathlib import Path
import torch
from orbit3d.pipeline.octree_runner import run_octree_from_config
from orbit3d.utils.run_evidence import freeze_run
p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--training-seeds',type=int,nargs='+',default=[43,44])
p.add_argument('--independent-oracle',type=Path,required=True)
a=p.parse_args();freeze_run(a.output,__file__)
base=json.loads(a.config.read_text());records=[]
base['mode']='predict'
base['generator_config'].update(grpo_steps=50,group_size=16,grpo_lr=2e-5,kl_beta=.02,physics_weight=.2,uncertainty_weight=0.,replay_weight=.1)
base['inference_config'].update(electron_parity=0,num_candidates=64,num_molecules=16,oracle_candidates=32,precision='float32',sampling_mode='independent',ranking='external')
base['stage5_relax_config'].update(enabled=True,batch_refinement=True,relax_steps=30,property_tolerance=0.,energy_tolerance=0.)
for seed in a.training_seeds:
 cfg=copy.deepcopy(base);cfg['seed']=seed
 cfg['mount_paths'].update(output_dir=str(a.output/f'train_seed{seed}'),checkpoint_save_dir=str(a.output/f'train_seed{seed}'/'checkpoints'))
 cfg['inference_config']['base_seed']=2026
 trained=run_octree_from_config(cfg);checkpoint=trained['checkpoint']
 gc.collect();torch.cuda.empty_cache()
 for inference_seed in [2026,2027,2028]:
  eval_cfg=copy.deepcopy(cfg)
  eval_cfg['generator_config']['grpo_steps']=0
  eval_cfg['mount_paths'].pop('train_data_path')
  run=a.output/f'train{seed}_infer{inference_seed}'
  eval_cfg['mount_paths'].update(output_dir=str(run),checkpoint_load_path=checkpoint)
  eval_cfg['inference_config']['base_seed']=inference_seed
  result=run_octree_from_config(eval_cfg)
  subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_generated.py')),'--oracle',str(a.independent_oracle),'--runs',str(run)],check=True)
  records.append({'training_seed':seed,'inference_seed':inference_seed,**result,'independent':json.loads((run/'independent_evaluation.json').read_text())})
  (a.output/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
  gc.collect();torch.cuda.empty_cache()
