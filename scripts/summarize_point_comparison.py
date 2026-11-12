"""Summarize raw pools; reuse fixed random streams for shuffled/off comparisons."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from orbit3d.constraints import TargetPoints
from evaluate_point_pool import summarize_records


def reward_signal(history):
    """Decompose observed group reward variance, without claiming gradient causality."""
    between=[];within=[];eligible_rewards=[]
    for row in history:
        records=row['rollout_records']
        values=np.array([r['reward'] for r in records],dtype=float)
        eligible=np.array([r.get('evaluation_status')=='completed' for r in records])
        p=float(eligible.mean())
        means=[float(values[mask].mean()) if mask.any() else 0. for mask in [eligible,~eligible]]
        between.append(p*(1-p)*(means[0]-means[1])**2)
        within.append(sum(float(mask.mean())*float(values[mask].var()) for mask in [eligible,~eligible] if mask.any()))
        eligible_rewards.extend(values[eligible].tolist())
    total=sum(between)+sum(within)
    return {'definition':'Population reward variance within each rollout group; summed between evaluable/ineligible and within-category components',
            'between_evaluability_fraction':sum(between)/total if total else None,
            'within_category_fraction':sum(within)/total if total else None,
            'evaluable_proposals':len(eligible_rewards),
            'mean_reward_given_evaluable':float(np.mean(eligible_rewards)) if eligible_rewards else None,
            'interpretation':'Descriptive reward contrast, not a causal decomposition of policy gradients'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--alignment-campaign',type=Path)
    p.add_argument('--sft-campaign',type=Path)
    p.add_argument('--time-control-campaign',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--target',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    points=json.loads(a.manifest.read_text())['single_target_points'][a.target]
    groups={};sources=[];source_hashes={}
    for path in sorted(a.campaign.glob('eval_*/raw/summary.json')):
        name=path.parent.parent.name; data=json.loads(path.read_text())
        group=name.rsplit('_s',1)[0]
        groups.setdefault(group,[]).append(data)
        sources.append(str(path))
        source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[]
    def score(group,scoring_quantile,control,input_quantile=None):
        pools=groups[group];records=[r for pool in pools for r in pool['records']]
        point=TargetPoints.parse({a.target:points[scoring_quantile]},{a.target:.25},['homo','lumo'])
        valid=[r for r in records if r['valid']]
        mols=[Chem.MolFromSmiles(r['smiles']) for r in valid]
        generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
        fps=[generator.GetFingerprint(m) for m in mols]
        similarities=[v for i,fp in enumerate(fps) for v in DataStructs.BulkTanimotoSimilarity(fp,fps[:i])]
        forces=[r['rms_force'] for r in records if r['evaluated'] and r.get('domain') and r['domain']['in_domain']]
        target_index=['homo','lumo'].index(a.target)
        common=[r for r in records if r['evaluated'] and r['independent_evaluated'] and r.get('domain') and r['domain']['in_domain']]
        rows.append({'group':group,'condition_control':control,'scoring_quantile':scoring_quantile,
                     'scoring_target_eV':points[scoring_quantile],
                     'input_quantile':input_quantile,'input_target_eV':points[input_quantile] if input_quantile else None,
                     'inference_repeats':len(pools),'attempted':len(records),
                     'valid':sum(r['valid'] for r in records),'valid_rate':sum(r['valid'] for r in records)/len(records),
                     'unique_valid':len({r['smiles'] for r in records if r['valid']}),
                     'unique_scaffolds':len({MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols}),
                     'internal_diversity_valid':float(1-np.mean(similarities)) if similarities else None,
                     'domain_valid':sum(bool(r.get('domain') and r['domain']['in_domain']) for r in valid),
                     'novel_valid':sum(r.get('novel_vs_sft_training',False) for r in records),
                     'unique_novel_valid':len({r['smiles'] for r in valid if r['novel_vs_sft_training']}),
                     'valid_atom_quantiles':np.quantile([r['atoms'] for r in valid],[0,.25,.5,.75,1]).tolist() if valid else None,
                     'domain_rms_force_p10_p50_p90':np.quantile(forces,[.1,.5,.9]).tolist() if forces else None,
                     'domain_mean_abs_evaluator_disagreement_eV':float(np.mean([abs(r['prediction'][target_index]-r['independent_prediction'][target_index]) for r in common])) if common else None,
                     'charge_separated_valid':sum(any(atom.GetFormalCharge()!=0 for atom in m.GetAtoms()) for m in mols),
                     'metrics':summarize_records(records,point),
                     'per_inference_seed_metrics':[summarize_records(pool['records'],point) for pool in pools]})
    for group in sorted(groups):
        if '_off' not in group:
            q=group.rsplit('_p',1)[1];score(group,q,'correct',q)
    # P50 receives P10, so every shifted center lies outside the original ±0.25 window.
    for original,supplied in [('25','75'),('50','10'),('75','25')]:
        group=f'eval_frozen_p{supplied}'
        if group in groups:score(group,original,'shuffled',supplied)
    if 'eval_frozen_off' in groups:
        for q in ['25','50','75']:score('eval_frozen_off',q,'off')
    arm_summary={}
    for arm in sorted({r['group'][5:].rsplit('_p',1)[0] for r in rows if r['condition_control']=='correct'}):
        selected=[r for r in rows if r['condition_control']=='correct' and r['group'].startswith(f'eval_{arm}_p')]
        count=sum(r['attempted'] for r in selected)
        summary={'quantiles':[r['scoring_quantile'] for r in selected],
                 'inference_repeats_per_quantile':[r['inference_repeats'] for r in selected],
                 'attempted':count,'valid':sum(r['valid'] for r in selected),
                 'domain_valid':sum(r['domain_valid'] for r in selected),
                 'metrics':{}}
        for backend in ['dpa','fingerprint']:
            values=[r['metrics'][backend]['in_domain'] for r in selected]
            maes=[r['targets'][a.target]['mae_eV'] for r in values if a.target in r['targets']]
            summary['metrics'][backend]={'domain_target_mae_equal_quantile_eV':float(np.mean(maes)) if len(maes)==len(values) else None,
                'domain_property_hits':sum(r['property_hits'] for r in values),
                'domain_property_hit_rate_all_attempts':sum(r['property_hits'] for r in values)/count,
                'domain_force_property_hits':sum(r['joint_domain_force_hits'] for r in values)}
        arm_summary[arm]=summary
    checkpoint=(a.alignment_campaign or a.campaign)/'grpo/weights/octree_aligned.pt'
    training=None
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        history=state['grpo_history'];saved=state.get('alignment_training_state')
        training={'groups':len(history),'attempted':sum(r['group_size'] for r in history),
                  'optimizer_updates':sum(r['optimizer_updates'] for r in history),
                  'complete':state.get('alignment_complete'),
                  'reference_sha256':state['alignment_reference_sha256'],
                  'zero_evaluable_groups':sum(r['evaluated']==0 for r in history),
                  'one_evaluable_groups':sum(r['evaluated']==1 for r in history),
                  'two_or_more_evaluable_groups':sum(r['evaluated']>=2 for r in history),
                  'last_policy_kl':history[-1]['policy_kl_before_last_update'],
                  'optimizer_state_steps':sorted({int(v['step']) for v in saved['optimizer']['state'].values()}) if saved else None,
                  'reward_signal':reward_signal(history),
                  'windows':{}}
        for name,window in [('first50',history[:50]),('last50',history[-50:])]:
            training['windows'][name]={'groups':len(window),
                'mean_valid':float(np.mean([r['valid'] for r in window])),
                'mean_evaluable':float(np.mean([r['evaluated'] for r in window])),
                'mean_group_reward':float(np.mean([r['reward_mean'] for r in window])),
                'reward_signal':reward_signal(window),
                'mean_gradient_norm':float(np.mean([r['gradient_norm'] for r in window]))}
    costs=[];cost_roots=set()
    for root,training_only in [(a.campaign,False),(a.alignment_campaign,True),(a.sft_campaign,True),(a.time_control_campaign,True)]:
        if root is None or root.resolve() in cost_roots:continue
        cost_roots.add(root.resolve())
        for row in json.loads((root/'costs.json').read_text()):
            if row['returncode'] or (training_only and row['name'] not in ['sft','continue','grpo_part100','grpo','continue_wall']):continue
            costs.append({key:row[key] for key in ['name','wall_seconds','reused_from'] if key in row} | {'campaign':str(root)})
    training_config=(a.alignment_campaign or a.campaign)/'configs/grpo.json'
    training_seed=json.loads(training_config.read_text())['seed'] if training_config.exists() else None
    time_control=None
    wall_checkpoint=(a.time_control_campaign or a.campaign)/'continue_wall/weights/octree_conditional.pt'
    if wall_checkpoint.exists():
        wall_state=torch.load(wall_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        time_control={key:wall_state[key] for key in ['optimizer_updates','requested_wall_seconds',
                      'training_wall_seconds','initial_weights_sha256','train_backbone']}
        with wall_checkpoint.open('rb') as stream:time_control['checkpoint_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    a.output.write_text(json.dumps({'scope':'Raw pools before ranking/refinement; reused pools are not independent repeats',
                                   'target':a.target,'training_seed':training_seed,
                                   'sources':sources,'source_sha256':source_hashes,'training_signal':training,
                                   'time_matched_sft':time_control,
                                   'completed_job_costs':costs,
                                   'arm_summary':arm_summary,
                                   'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
