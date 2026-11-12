"""Paired training-seed comparison for the independently evaluated native-3D pools."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--summaries',nargs='+',type=Path,required=True)
    p.add_argument('--target',choices=['homo','lumo'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=[json.loads(p.read_text()) for p in a.summaries];seeds=[r['training_seed'] for r in runs];assert len(set(seeds))==len(seeds)
    arms=['frozen','continue400','grpo200','continue_wall'];per_seed=[];counts=set()
    for run in runs:
        row={'training_seed':run['training_seed'],'arms':{},'paired_grpo_minus_control':{}}
        for arm in arms:
            r=next(r for r in run['arms'] if r['target']==a.target and r['arm']==arm);counts.add(r['attempted'])
            row['arms'][arm]={'independent_target_mae_eV':r['target_mae_equal_quantile_eV'],'domain_rate':r['evaluated']/r['attempted'],
                             'independent_hit_rate':r['property_hits']/r['attempted'],'independent_mean_response_slope':r['mean_response_slope']}
        for control in ['frozen','continue400','continue_wall']:
            row['paired_grpo_minus_control'][control]={k:row['arms']['grpo200'][k]-v for k,v in row['arms'][control].items()}
        per_seed.append(row)
    assert len(counts)==1,'Do not merge pilot and formal proposal budgets'
    def describe(values):return {'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'min':min(values),'max':max(values)}
    report={'target':a.target,'evaluator':'HyFF source-3D property head','training_seeds':seeds,'attempts_per_arm_per_training_seed':next(iter(counts)),
        'scope':'Independent training seeds are the repeat unit. Same native XYZ and original domain masks; HyFF never used for reward or selection.',
        'per_training_seed':per_seed,'arms':{arm:{k:describe([r['arms'][arm][k] for r in per_seed]) for k in per_seed[0]['arms'][arm]} for arm in arms},
        'paired_grpo_minus_control':{control:{k:describe([r['paired_grpo_minus_control'][control][k] for r in per_seed]) for k in per_seed[0]['paired_grpo_minus_control'][control]} for control in ['frozen','continue400','continue_wall']},
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.summaries}}
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
