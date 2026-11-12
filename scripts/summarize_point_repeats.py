"""Summarize paired policy differences over independent training seeds."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--summaries',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();runs=[json.loads(path.read_text()) for path in a.summaries]
    targets={run['target'] for run in runs};assert len(targets)==1
    seeds=[run['training_seed'] for run in runs]
    assert None not in seeds and len(set(seeds))==len(seeds),'Require distinct known training seeds'
    target=next(iter(targets));arms=['frozen','continue400','grpo200','continue_wall']
    per_seed=[];counts=set()
    for run in runs:
        row={'training_seed':run['training_seed'],'arms':{},'paired_grpo_minus_control':{}}
        for arm in arms:
            summary=run['arm_summary'][arm]
            assert summary['quantiles']==['10','25','50','75','90']
            assert summary['inference_repeats_per_quantile']==[3]*5
            counts.add(summary['attempted'])
            responses=[r for r in run['rows'] if r['condition_control']=='correct' and r['group'].startswith(f'eval_{arm}_p')]
            x=[r['scoring_target_eV'] for r in responses]
            y=[r['metrics']['fingerprint']['in_domain']['targets'][target]['mean_prediction_eV'] for r in responses]
            correct=[r for r in responses if r['scoring_quantile'] in ['25','50','75']]
            shuffled=[r for r in run['rows'] if r['condition_control']=='shuffled' and r['group'].startswith(f'eval_{arm}_p')]
            assert len(correct)==len(shuffled)==3
            condition_error=lambda records:float(np.mean([r['metrics']['fingerprint']['in_domain']['targets'][target]['mae_eV'] for r in records]))
            row['arms'][arm]={
                'independent_target_mae_eV':summary['metrics']['fingerprint']['domain_target_mae_equal_quantile_eV'],
                'optimization_oracle_target_mae_eV':summary['metrics']['dpa']['domain_target_mae_equal_quantile_eV'],
                'valid_rate':summary['valid']/summary['attempted'],
                'domain_rate':summary['domain_valid']/summary['attempted'],
                'independent_hit_rate':summary['metrics']['fingerprint']['domain_property_hit_rate_all_attempts'],
                'independent_mean_response_slope':float(np.polyfit(x,y,1)[0]),
                'independent_condition_specificity_eV':condition_error(shuffled)-condition_error(correct),
                'independent_alignment_target_mae_eV':condition_error(correct),
                'independent_other_target_mae_eV':condition_error([r for r in responses if r['scoring_quantile'] in ['10','90']]),
                'mean_internal_diversity':float(np.mean([r['internal_diversity_valid'] for r in responses])),
                'mean_unique_valid_fraction':float(np.mean([r['unique_valid']/r['valid'] for r in responses])),
            }
        for control in ['frozen','continue400','continue_wall']:
            row['paired_grpo_minus_control'][control]={name:row['arms']['grpo200'][name]-value
                                                       for name,value in row['arms'][control].items()}
        per_seed.append(row)
    assert len(counts)==1,'Do not combine pilot and larger candidate budgets'
    def describe(values):
        return {'values':values,'mean':float(np.mean(values)),
                'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
                'min':min(values),'max':max(values)}
    arm_stats={arm:{name:describe([row['arms'][arm][name] for row in per_seed])
                    for name in per_seed[0]['arms'][arm]} for arm in arms}
    differences={control:{name:describe([row['paired_grpo_minus_control'][control][name] for row in per_seed])
                          for name in per_seed[0]['paired_grpo_minus_control'][control]}
                 for control in ['frozen','continue400','continue_wall']}
    a.output.write_text(json.dumps({'target':target,'training_seeds':seeds,'attempts_per_arm_per_training_seed':next(iter(counts)),
        'scope':'Independent training seeds are the repeat unit; descriptive paired differences, no significance claim',
        'mae_definition':'Equal weight over five target quantiles; domain-evaluable molecules within each quantile',
        'sources':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in a.summaries},
        'per_training_seed':per_seed,'arms':arm_stats,'paired_grpo_minus_control':differences},indent=2)+'\n')


if __name__=='__main__':main()
