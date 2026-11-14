"""Matched unconditional-prior versus interval-SFT controls using verified upstream weights."""
import argparse,copy,gc,hashlib,json,subprocess,sys
from pathlib import Path
import torch
from loom.generator.octree_backend import load_pretrained,checkpoint_contract
from loom.pipeline.octree_runner import run_octree_from_config
from loom.utils.run_evidence import freeze_run,record_input_hashes
from loom.utils.reproducibility import lock_deterministic_environment
p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--pretrained',type=Path,required=True)
p.add_argument('--sft',type=Path,required=True)
p.add_argument('--independent-oracle',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();freeze_run(a.output,__file__)
record_input_hashes(a.output,{'upstream_prior':a.pretrained,'sft':a.sft,'independent_oracle':a.independent_oracle})
lock_deterministic_environment(42);torch.set_num_threads(2)
reference=torch.load(a.sft,map_location='cpu',weights_only=False)
names,units,mean,scale=[reference[k] for k in ['target_cols','target_units','mean','scale']]
del reference
model=load_pretrained(a.pretrained,'cuda:0',len(names),16,condition_version=3)
# No randomly initialized head may rank proposals or masquerade as a predictor.
# A zero conditioner is the identity transformation of the loaded structural prior.
with torch.no_grad():
 for parameter in model.interval_conditioner.parameters():parameter.zero_()
 for parameter in model.prop_understanding_head.parameters():parameter.zero_()
prior=a.output/'unconditional_prior.pt'
with a.pretrained.open('rb') as stream:source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
torch.save({'model_state_dict':model.state_dict(),'backend':'uni3dar_octree_v3','generator_role':'unconditional_pretrained_baseline','conditioning_trained':False,'self_ranking_trained':False,'pretrained_sha256':source_hash,'target_cols':names,'target_units':units,'mean':mean,'scale':scale,'contract':checkpoint_contract(model,names,units,mean,scale),'training_smiles':[]},prior)
del model;gc.collect();torch.cuda.empty_cache()
base=json.loads(a.config.read_text());base['mode']='predict';base['generator_config']['grpo_steps']=0
base['mount_paths'].pop('train_data_path',None)
base['inference_config'].update(num_candidates=64,num_molecules=16,oracle_candidates=32,electron_parity=0,sampling_mode='independent',precision='float32',ranking='external')
base['stage5_relax_config'].update(enabled=True,batch_refinement=True,relax_steps=30,property_tolerance=0.,energy_tolerance=0.)
records=[]
for variant,checkpoint in [('unconditional_prior',prior),('interval_sft',a.sft)]:
 for seed in [2026,2027,2028]:
  cfg=copy.deepcopy(base);run=a.output/f'{variant}_seed{seed}'
  cfg['experiment_role']=variant
  cfg['mount_paths'].update(checkpoint_load_path=str(checkpoint),output_dir=str(run))
  cfg['inference_config']['base_seed']=seed
  result=run_octree_from_config(cfg)
  subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_generated.py')),'--oracle',str(a.independent_oracle),'--runs',str(run)],check=True)
  # Prior pretraining identities are unavailable; do not report empty-set novelty.
  if variant=='unconditional_prior':result['retained_novel_vs_adaptation']=None
  records.append({'variant':variant,'inference_seed':seed,**result,'independent':json.loads((run/'independent_evaluation.json').read_text())})
  (a.output/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
  gc.collect();torch.cuda.empty_cache()
(a.output/'protocol.json').write_text(json.dumps({'prior_conditioning':'disabled by zero identity modulation; backbone loaded from verified upstream EMA','prior_self_ranking':'disabled by constant zero head; candidate-ID order before external evaluation budget','comparison':'Stage bundle (interval conditioning and trained self-shortlist), not a pure conditioner-only ablation','prior_pretraining_novelty':'not measured','attempts_per_variant':192},indent=2)+'\n')
