"""Measure eligibility and within-eligible advantage components of frozen rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def analyze(history):
    groups=[]
    for step in history:
        rows=step['rollout_records']
        reward=np.array([r['reward'] for r in rows],dtype=float)
        eligible=np.array([r.get('evaluation_status')=='completed' for r in rows])
        advantage=np.array([r['advantage'] for r in rows]) if all('advantage' in r for r in rows) else (reward-reward.mean())/max(reward.std(),1e-6)
        category=np.zeros_like(advantage)
        for mask in [eligible,~eligible]:
            if mask.any():category[mask]=advantage[mask].mean()
        residual=advantage-category
        groups.append({'step':step['step'],'eligible':int(eligible.sum()),'attempted':len(rows),
            'category_advantage_square':float(category@category),'within_category_advantage_square':float(residual@residual),
            'eligible_advantage_std':float(advantage[eligible].std()) if eligible.any() else None,
            'eligible_negative_advantages':int((advantage[eligible]<0).sum()),
            'eligible_reward_range':float(np.ptp(reward[eligible])) if eligible.any() else None,
            'gradient_norm':step['gradient_norm'],'parameter_update_l2':step['parameter_update_l2'],
            'policy_kl':step['policy_kl_before_last_update']})
    between=sum(r['category_advantage_square'] for r in groups)
    within=sum(r['within_category_advantage_square'] for r in groups)
    return {'groups':len(groups),'attempted':sum(r['attempted'] for r in groups),
        'eligible':sum(r['eligible'] for r in groups),
        'zero_eligible_groups':sum(r['eligible']==0 for r in groups),
        'one_eligible_groups':sum(r['eligible']==1 for r in groups),
        'two_plus_eligible_groups':sum(r['eligible']>=2 for r in groups),
        'at_most_half_eligible_groups':sum(2*r['eligible']<=r['attempted'] for r in groups),
        'within_category_advantage_fraction':within/(between+within) if between+within else None,
        'eligible_negative_advantages':sum(r['eligible_negative_advantages'] for r in groups),
        'per_group':groups}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,required=True,help='Name to checkpoint JSON mapping')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();results={}
    for name,path in json.loads(a.checkpoints.read_text()).items():
        path=Path(path);state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        results[name]={'checkpoint':str(path),'sha256':digest,'reference_sha256':state['alignment_reference_sha256'],
            'alignment_complete':state['alignment_complete'],**analyze(state['grpo_history'])}
    a.output.write_text(json.dumps({'scope':'Orthogonal decomposition of realized trajectory advantages; descriptive signal magnitude, not causal gradient attribution. Zero and one eligible groups cannot rank eligible properties.',
        'checkpoints':results},indent=2)+'\n')


if __name__=='__main__':main()
