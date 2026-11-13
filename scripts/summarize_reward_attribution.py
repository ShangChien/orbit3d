"""Separate proposal yield changes into evaluable supply and conditional hit fractions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def decompose(control, treatment):
    """Symmetric exact accounting; this is not causal mediation."""
    pc=control['evaluated']/control['attempted'];pt=treatment['evaluated']/treatment['attempted']
    hc=control['hits']/control['attempted'];ht=treatment['hits']/treatment['attempted']
    qc=control['hits']/control['evaluated'] if control['evaluated'] else None
    qt=treatment['hits']/treatment['evaluated'] if treatment['evaluated'] else None
    supply=(pt-pc)*(qt+qc)/2 if qc is not None and qt is not None else None
    precision=(qt-qc)*(pt+pc)/2 if qc is not None and qt is not None else None
    return {'control_coverage':pc,'treatment_coverage':pt,'control_conditional_hit_fraction':qc,
        'treatment_conditional_hit_fraction':qt,'yield_delta':ht-hc,'supply_component':supply,
        'conditional_hit_component':precision,'target_mae_delta_eV':treatment['mae']-control['mae'] if treatment['mae'] is not None and control['mae'] is not None else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--a-summaries',nargs='+',type=Path,required=True)
    p.add_argument('--b-summaries',nargs='+',type=Path,required=True)
    p.add_argument('--treatment',default='grpo200');p.add_argument('--controls',nargs='+',default=['eligibility200','continue_wall','frozen'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a_runs=[json.loads(path.read_text()) for path in a.a_summaries]
    b_runs=[json.loads(path.read_text()) for path in a.b_summaries]
    seeds=[r['training_seed'] for r in a_runs];assert len(set(seeds))==len(seeds)
    assert sorted(seeds)==sorted(r['training_seed'] for r in b_runs)
    targets={r['target'] for r in a_runs};assert len(targets)==1;target=targets.pop()
    comparisons=[]
    for run in a_runs:
        b=next(r for r in b_runs if r['training_seed']==run['training_seed'])
        for evaluator in ['dpa','fingerprint','unimol2']:
            def score(arm,q):
                if evaluator=='unimol2':
                    arm_row=next(r for r in b['arms'] if r['target']==target and r['arm']==arm)
                    r=next(r for r in arm_row['per_quantile'] if str(r['quantile'])==q)
                    return {'attempted':r['attempted'],'evaluated':r['evaluated'],'hits':r['property_hits'],'mae':r['target_mae_eV']}
                r=next(r for r in run['rows'] if r['condition_control']=='correct' and r['group']==f'eval_{arm}_p{q}')
                m=r['metrics'][evaluator]['in_domain']
                return {'attempted':r['attempted'],'evaluated':m['evaluated'],'hits':m['property_hits'],'mae':m['targets'].get(target,{}).get('mae_eV')}
            for control in a.controls:
                cells=[]
                for q in ['10','25','50','75','90']:
                    c=score(control,q);t=score(a.treatment,q);assert c['attempted']==t['attempted']
                    cells.append({'quantile':q,'control':c,'treatment':t,**decompose(c,t)})
                comparisons.append({'training_seed':run['training_seed'],'evaluator':evaluator,'control':control,
                    'per_quantile':cells,'mean':{k:float(np.mean([r[k] for r in cells])) if all(r[k] is not None for r in cells) else None
                    for k in ['yield_delta','supply_component','conditional_hit_component','target_mae_delta_eV']}})
    aggregate=[]
    for evaluator in ['dpa','fingerprint','unimol2']:
        for control in a.controls:
            rows=[r for r in comparisons if r['evaluator']==evaluator and r['control']==control]
            aggregate.append({'evaluator':evaluator,'control':control,'mean':{
                key:float(np.mean([r['mean'][key] for r in rows])) if all(r['mean'][key] is not None for r in rows) else None
                for key in rows[0]['mean']}})
    report={'target':target,'training_seeds':seeds,'treatment':a.treatment,
        'definition':'Hit yield = evaluable coverage * hit fraction among evaluable proposals. Delta splits exactly into delta coverage * average conditional hit fraction + delta conditional hit fraction * average coverage. Per-quantile, then equal quantile and training-seed means.',
        'interpretation':'Accounting identity, not a causal mediation estimate. Policy arms generate different candidate populations; raw coverage, conditional MAE and independent evaluators remain necessary.',
        'comparisons':comparisons,'aggregate':aggregate,
        'sources':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in a.a_summaries+a.b_summaries}}
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
