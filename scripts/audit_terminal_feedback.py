"""Audit actual parameter scope and channel separation after the frozen five-arm run."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=['structure','physical','property','sft'])
    p.add_argument('--sft-arm',default='sft');p.add_argument('--groups',type=int,default=24);a=p.parse_args()
    torch.set_num_threads(4);base=torch.load(a.initial,map_location='cpu',weights_only=False,mmap=True)
    with a.initial.open('rb') as f:initial_hash=hashlib.file_digest(f,'sha256').hexdigest()
    report={'initial_sha256':initial_hash,'arms':{}}
    for arm in a.arms:
        is_sft=arm==a.sft_arm
        path=a.root/arm/'weights'/('octree_conditional.pt' if is_sft else 'octree_aligned.pt')
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert state['initial_weights_sha256' if is_sft else 'alignment_reference_sha256']==initial_hash
        changed=[k for k,v in base['model_state_dict'].items() if not torch.equal(v,state['model_state_dict'][k])]
        assert changed and all(k.startswith('interval_conditioner.') for k in changed)
        assert np.array_equal(state['mean'],base['mean']) and np.array_equal(state['scale'],base['scale'])
        assert state['target_cols']==base['target_cols'] and state['target_units']==base['target_units']
        row={'nonconditioner_tensors_unchanged':True,'target_contract_and_scaler_unchanged':True,'changed_conditioner_tensors':len(changed)}
        delta=sum((state['model_state_dict'][k].double()-base['model_state_dict'][k].double()).square().sum().item() for k in changed)
        norm=sum(v.double().square().sum().item() for k,v in base['model_state_dict'].items() if k.startswith('interval_conditioner.'))
        row['conditioner_delta_l2']=float(np.sqrt(delta));row['conditioner_relative_delta_l2']=float(np.sqrt(delta/norm))
        with path.open('rb') as f:row['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        if not is_sft:
            history=state['grpo_history'];assert len(history)==a.groups and all(r['group_size']==32 for r in history)
            counts=[];negative=0;energies=[]
            for group in history:
                rows=group['rollout_records'];eligible=[r for r in rows if r.get('evaluation_status')=='completed'];counts.append(len(eligible))
                for r in rows:
                    channels=r['terminal_advantages'];assert abs(sum(channels.values())-r['advantage'])<1e-5
                    if group['terminal_feedback_level']!='property' or len(eligible)<2:assert channels['property']==0
                    if group['terminal_feedback_level']=='structure':assert channels['physical']==0
                    negative+=channels['property']<0
                assert np.std([r['advantage'] for r in rows])<=1.00001
                vectors=np.array([[r['terminal_advantages'][name] for name in ['structure','physical','property']] for r in rows])
                gram=vectors.T@vectors/len(rows)
                assert np.max(np.abs(gram-np.diag(np.diag(gram))))<1e-5
                energies.append(np.diag(gram))
            energy=np.mean(energies,axis=0)
            row['advantage_vector_energy']={name:float(value) for name,value in zip(['structure','physical','property'],energy)}
            row['energy_fraction']={name:float(value/energy.sum()) if energy.sum() else None for name,value in zip(['structure','physical','property'],energy)}
            row['energy_scope']='Orthogonal nested advantage vectors in rollout space, not orthogonal parameter gradients or causal attribution.'
            row['update_diagnostics']={key:{'min':float(np.min(values)),'median':float(np.median(values)),'max':float(np.max(values))}
                for key in ['gradient_norm','parameter_update_l2','policy_kl_before_last_update']
                if (values:=[g[key] for g in history if g[key] is not None])}
            row.update(groups=len(history),proposals=sum(g['group_size'] for g in history),optimizer_updates=sum(r['optimizer_updates'] for r in history),
                evaluated_per_group=counts,negative_property_components=negative,
                zero_or_one_property_groups=sum(n<=1 for n in counts),
                energy_force_calls=sum(r.get('terminal',{}).get('total_dpa_calls',0) for g in history for r in g['rollout_records']))
        else:row['optimizer_updates']=state['optimizer_updates']
        report['arms'][arm]=row;del state
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
