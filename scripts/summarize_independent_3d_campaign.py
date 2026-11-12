"""Reduce native-3D reassessment to paired arm and condition-control summaries."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from evaluate_independent_3d_campaign import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.summary.read_text());manifest=json.loads(a.manifest.read_text());arms=[];controls=[]
    for target in ['homo','lumo']:
        for arm in ['frozen','continue400','grpo200','continue_wall']:
            per_target=[]
            for q,goal in manifest['single_target_points'][target].items():
                pools=[r for r in data['pools'] if r['target']==target and r['name'].startswith(f'eval_{arm}_p{q}_')]
                assert len(pools)==3
                rows=sum([r['records'] for r in pools],[])
                per_target.append({'quantile':q,'goal_eV':goal,**metrics(rows,target,goal,True)})
            arms.append({'target':target,'arm':arm,'attempted':sum(r['attempted'] for r in per_target),
                'evaluated':sum(r['evaluated'] for r in per_target),'property_hits':sum(r['property_hits'] for r in per_target),
                'target_mae_equal_quantile_eV':float(np.mean([r['target_mae_eV'] for r in per_target])),
                'mean_response_slope':float(np.polyfit([r['goal_eV'] for r in per_target],[r['mean_prediction_eV'] for r in per_target],1)[0]),
                'per_quantile':per_target})
        for mode in ['correct','off','shuffled']:
            scored=[]
            for q,supplied in [('25','75'),('50','10'),('75','25')]:
                pattern='eval_frozen_off_' if mode=='off' else f'eval_frozen_p{supplied if mode=="shuffled" else q}_'
                pools=[r for r in data['pools'] if r['target']==target and r['name'].startswith(pattern)];assert len(pools)==3
                rows=sum([r['records'] for r in pools],[]);goal=manifest['single_target_points'][target][q]
                scored.append({'scoring_quantile':q,**metrics(rows,target,goal,True)})
            controls.append({'target':target,'condition':mode,'target_mae_equal_quantile_eV':float(np.mean([r['target_mae_eV'] for r in scored])),
                             'per_quantile':scored})
    output={'scope':data['scope'],'mae_definition':'Equal target-quantile weighting; original-domain evaluable proposals. Each property/arm has 960 attempts. Conditions reuse frozen pools and are not new training repeats.',
        'arms':arms,'condition_controls':controls,'wall_seconds':data['wall_seconds'],
        'native_evaluation_attempts':sum(r['native_evaluation_attempts'] for r in data['pools']),
        'prediction_failures':sum(sum(r['valid'] and not r['hyff_evaluated'] for r in pool['records']) for pool in data['pools']),
        'pools':[{k:v for k,v in r.items() if k!='records'} for r in data['pools']],
        'raw_summary_sha256':hashlib.sha256(a.summary.read_bytes()).hexdigest(),'input_manifest':json.loads(a.inputs.read_text())}
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    for r in controls:print(r['target'],r['condition'],r['target_mae_equal_quantile_eV'])


if __name__=='__main__':main()
