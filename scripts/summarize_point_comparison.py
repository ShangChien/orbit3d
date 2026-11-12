"""Summarize raw pools; reuse fixed random streams for shuffled/off comparisons."""
import argparse
import json
from pathlib import Path
from orbit3d.constraints import TargetPoints
from evaluate_point_pool import summarize_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--target',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    points=json.loads(a.manifest.read_text())['single_target_points'][a.target]
    groups={};sources=[]
    for path in sorted(a.campaign.glob('eval_*/raw/summary.json')):
        name=path.parent.parent.name; data=json.loads(path.read_text())
        group=name.rsplit('_s',1)[0]
        groups.setdefault(group,[]).append(data)
        sources.append(str(path))
    rows=[]
    def score(group,scoring_quantile,control,input_quantile=None):
        pools=groups[group];records=[r for pool in pools for r in pool['records']]
        point=TargetPoints.parse({a.target:points[scoring_quantile]},{a.target:.25},['homo','lumo'])
        rows.append({'group':group,'condition_control':control,'scoring_quantile':scoring_quantile,
                     'scoring_target_eV':points[scoring_quantile],
                     'input_quantile':input_quantile,'input_target_eV':points[input_quantile] if input_quantile else None,
                     'inference_repeats':len(pools),'attempted':len(records),
                     'valid':sum(r['valid'] for r in records),'valid_rate':sum(r['valid'] for r in records)/len(records),
                     'unique_valid':len({r['smiles'] for r in records if r['valid']}),
                     'novel_valid':sum(r.get('novel_vs_sft_training',False) for r in records),
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
    a.output.write_text(json.dumps({'scope':'Raw pools before ranking/refinement; reused pools are not independent repeats',
                                   'target':a.target,'sources':sources,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
