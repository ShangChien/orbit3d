"""Audit actual parameter scope and channel separation after the frozen five-arm run."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);base=torch.load(a.initial,map_location='cpu',weights_only=False,mmap=True)
    with a.initial.open('rb') as f:initial_hash=hashlib.file_digest(f,'sha256').hexdigest()
    report={'initial_sha256':initial_hash,'arms':{}}
    for arm in ['structure','physical','property','sft']:
        path=a.root/arm/'weights'/('octree_conditional.pt' if arm=='sft' else 'octree_aligned.pt')
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert state['initial_weights_sha256' if arm=='sft' else 'alignment_reference_sha256']==initial_hash
        changed=[k for k,v in base['model_state_dict'].items() if not torch.equal(v,state['model_state_dict'][k])]
        assert changed and all(k.startswith('interval_conditioner.') for k in changed)
        row={'nonconditioner_tensors_unchanged':True,'changed_conditioner_tensors':len(changed)}
        with path.open('rb') as f:row['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        if arm!='sft':
            history=state['grpo_history'];assert len(history)==24 and all(r['group_size']==32 for r in history)
            counts=[];negative=0
            for group in history:
                rows=group['rollout_records'];eligible=[r for r in rows if r.get('evaluation_status')=='completed'];counts.append(len(eligible))
                for r in rows:
                    channels=r['terminal_advantages'];assert abs(sum(channels.values())-r['advantage'])<1e-5
                    if arm!='property' or len(eligible)<2:assert channels['property']==0
                    if arm=='structure':assert channels['physical']==0
                    negative+=channels['property']<0
                assert np.std([r['advantage'] for r in rows])<=1.00001
            row.update(groups=24,proposals=768,optimizer_updates=sum(r['optimizer_updates'] for r in history),
                evaluated_per_group=counts,negative_property_components=negative,
                zero_or_one_property_groups=sum(n<=1 for n in counts),
                energy_force_calls=sum(r.get('terminal',{}).get('total_dpa_calls',0) for g in history for r in g['rollout_records']))
        else:row['optimizer_updates']=state['optimizer_updates']
        report['arms'][arm]=row;del state
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
