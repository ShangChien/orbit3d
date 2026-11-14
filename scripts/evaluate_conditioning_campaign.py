"""Matched-budget factorial evaluation of conditioner versions and electron parity."""
import argparse
import copy
import gc
import json
from pathlib import Path
import numpy as np
import torch
from loom.pipeline.octree_runner import run_octree_from_config
p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--models',type=Path,required=True,help='JSON mapping variant names to checkpoints')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
p.add_argument('--parity',choices=['both','even','off'],default='both')
a=p.parse_args()
base=json.loads(a.config.read_text());models=json.loads(a.models.read_text())
base['mode']='predict';base['generator_config']['grpo_steps']=0
base['mount_paths'].pop('train_data_path',None)
base['mount_paths'].pop('generator_pretrained_path',None)
base['inference_config'].update(num_candidates=64,num_molecules=16,oracle_candidates=32,
                                sampling_mode='independent',precision='float32',ranking='external')
base['stage5_relax_config'].update(batch_refinement=True,enabled=True)
parities={'both':[None,0],'even':[0],'off':[None]}[a.parity]
records=[]
for name,checkpoint in models.items():
    for parity in parities:
        for seed in a.seeds:
            config=copy.deepcopy(base)
            config['mount_paths'].update(checkpoint_load_path=checkpoint,
                output_dir=str(a.output/f'{name}_parity{parity}_seed{seed}'))
            config['inference_config'].update(base_seed=seed,electron_parity=parity)
            summary=run_octree_from_config(config)
            summary.pop('checkpoint')
            records.append({'variant':name,'electron_parity':parity,'inference_seed':seed,**summary})
            (a.output/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
aggregate=[]
for name in models:
    for parity in parities:
        selected=[r for r in records if r['variant']==name and r['electron_parity']==parity]
        row={'variant':name,'electron_parity':parity,'inference_seeds':a.seeds}
        for metric in ['raw_valid','domain_eligible','retained','qualified','joint_success_rate']:
            x=np.array([r[metric] for r in selected])
            row[metric]={'mean':float(x.mean()),'seed_sd':float(x.std(ddof=1)) if len(x)>1 else None}
        aggregate.append(row)
(a.output/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
print(json.dumps(aggregate,indent=2))
