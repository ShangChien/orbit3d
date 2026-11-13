"""Verify the actual intervention and budgets of a frozen point-contrast campaign."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaigns', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();torch.set_num_threads(4);runs=[]
    for campaign in a.campaigns:
        checkpoints=json.loads((campaign/'checkpoints.json').read_text())
        initial=Path(checkpoints['point_frozen'])
        digest=hashlib.sha256(initial.read_bytes()).hexdigest()
        base=torch.load(initial,map_location='cpu',weights_only=False,mmap=True)
        row={'campaign':str(campaign),'initial_sha256':digest,
             'training_costs':json.loads((campaign/'costs.json').read_text()),'arms':{}}
        for arm in ['point_sft','point_contrast','point_time']:
            path=Path(checkpoints[arm]);state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert state['initial_weights_sha256']==digest
            assert state['backend']=='uni3dar_octree_v4' and not state['train_backbone']
            changed=[]
            for key,value in base['model_state_dict'].items():
                if not torch.equal(value,state['model_state_dict'][key]):changed.append(key)
            assert changed and all(key.startswith('interval_conditioner.') for key in changed), 'Parameters outside declared intervention changed'
            assert np.array_equal(state['mean'],base['mean']) and np.array_equal(state['scale'],base['scale'])
            history=state['history']
            row['arms'][arm]={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'optimizer_updates':state['optimizer_updates'],'pipeline_training_seconds':state['training_wall_seconds'],
                'requested_wall_seconds':state['requested_wall_seconds'],
                'changed_conditioner_tensors':len(changed),'nonconditioner_tensors_unchanged':True,
                'scaler_unchanged':True,'active_targets':state['active_targets'],
                'mean_contrast_loss':float(np.mean([r['loss_contrast'] for r in history])) if arm=='point_contrast' else None,
                'mean_eligible_pairs_per_batch':float(np.mean([r['contrast_eligible'] for r in history])) if arm=='point_contrast' else None}
            del state
        runs.append(row);del base
    a.output.write_text(json.dumps({'scope':'Exact checkpoint tensor comparisons; time budget is pipeline time, process wall reported separately. No evaluation-based checkpoint selection.','runs':runs},indent=2)+'\n')


if __name__=='__main__':main()
