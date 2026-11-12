"""Summarize raw pools; reuse fixed random streams for shuffled/off comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
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
        valid=[r for r in records if r['valid']]
        mols=[Chem.MolFromSmiles(r['smiles']) for r in valid]
        generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
        fps=[generator.GetFingerprint(m) for m in mols]
        similarities=[v for i,fp in enumerate(fps) for v in DataStructs.BulkTanimotoSimilarity(fp,fps[:i])]
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
