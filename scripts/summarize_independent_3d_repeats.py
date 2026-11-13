"""Paired training-seed comparison for the independently evaluated native-3D pools."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--summaries',nargs='+',type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=['frozen','continue400','grpo200','continue_wall'])
    p.add_argument('--reference-arm',default='grpo200')
    p.add_argument('--target',choices=['homo','lumo'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=[json.loads(p.read_text()) for p in a.summaries];seeds=[r['training_seed'] for r in runs];assert len(set(seeds))==len(seeds)
    evaluators={r.get('evaluator','HyFF source-3D property head') for r in runs};assert len(evaluators)==1
    evaluator=next(iter(evaluators))
    arms=a.arms;controls=[arm for arm in arms if arm!=a.reference_arm];per_seed=[];counts=set()
    if a.reference_arm not in arms:p.error('Reference arm must be included in arms')
    for run in runs:
        row={'training_seed':run['training_seed'],'arms':{},'paired_grpo_minus_control':{}}
        for arm in arms:
            r=next(r for r in run['arms'] if r['target']==a.target and r['arm']==arm);counts.add(r['attempted'])
            conditions={v['condition']:v['target_mae_equal_quantile_eV'] for v in run['condition_controls'] if v['target']==a.target and v['arm']==arm}
            row['arms'][arm]={'independent_target_mae_eV':r['target_mae_equal_quantile_eV'],'domain_rate':r['evaluated']/r['attempted'],
                             'independent_hit_rate':r['property_hits']/r['attempted'],'independent_mean_response_slope':r['mean_response_slope'],
                             'independent_condition_specificity_eV':conditions['shuffled']-conditions['correct'],
                             'independent_alignment_target_mae_eV':float(np.mean([v['target_mae_eV'] for v in r['per_quantile'] if v['quantile'] in ['25','50','75']])),
                             'independent_other_target_mae_eV':float(np.mean([v['target_mae_eV'] for v in r['per_quantile'] if v['quantile'] in ['10','90']]))}
        for control in controls:
            row['paired_grpo_minus_control'][control]={k:row['arms'][a.reference_arm][k]-v for k,v in row['arms'][control].items()}
        per_seed.append(row)
    assert len(counts)==1,'Do not merge pilot and formal proposal budgets'
    def describe(values):return {'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'min':min(values),'max':max(values)}
    report={'reference_arm':a.reference_arm,'target':a.target,'evaluator':evaluator,'training_seeds':seeds,'attempts_per_arm_per_training_seed':next(iter(counts)),
        'scope':f'Independent training seeds are the repeat unit. Same native XYZ and original domain masks; {evaluator} never used for reward or selection. Evaluator support exclusions remain in proposal denominators.',
        'per_training_seed':per_seed,'arms':{arm:{k:describe([r['arms'][arm][k] for r in per_seed]) for k in per_seed[0]['arms'][arm]} for arm in arms},
        'paired_grpo_minus_control':{control:{k:describe([r['paired_grpo_minus_control'][control][k] for r in per_seed]) for k in per_seed[0]['paired_grpo_minus_control'][control]} for control in controls},
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.summaries}}
    if a.reference_arm!='grpo200':
        report['paired_reference_minus_control']=report.pop('paired_grpo_minus_control')
        for row in report['per_training_seed']:row['paired_reference_minus_control']=row.pop('paired_grpo_minus_control')
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
