"""Frozen terminal-pool selection utility; DFT never participates in ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def select(rows, model, property_index, goal, k):
    eligible=[r for r in rows if r.get(model) is not None]
    chosen=sorted(eligible,key=lambda r:(abs(r[model][property_index]-goal),r['candidate_id']))[:k]
    paired=[r for r in chosen if r['truth'] is not None]
    errors=[abs(r['truth'][property_index]-goal) for r in paired]
    return dict(requested=k,selected=len(chosen),quantum_verified=len(paired),hits=sum(e<=.25 for e in errors),
                error_sum_eV=float(sum(errors)),mae_eV=float(np.mean(errors)) if errors else None,
                selected_indices=[r['index'] for r in chosen],quantum_failed_selected=[r['index'] for r in chosen if r['truth'] is None])


def aggregate(cells):
    names=['requested','selected','quantum_verified','hits'];s={n:sum(c[n] for c in cells) for n in names}
    target_maes=[]
    for q in sorted({c['quantile'] for c in cells}):
        local=[c for c in cells if c['quantile']==q];n=sum(c['quantum_verified'] for c in local)
        target_maes.append(sum(c['error_sum_eV'] for c in local)/n if n else None)
    s.update(delivery_coverage=s['selected']/s['requested'],hit_rate_requested_slots=s['hits']/s['requested'],
             quantum_coverage_selected=s['quantum_verified']/s['selected'] if s['selected'] else None,
             retained_hit_yield_all_proposals_equal_goal=s['hits']/(len(cells)*64),
             mae_equal_goal_eV=float(np.mean(target_maes)) if all(v is not None for v in target_maes) else None,
             mae_quantum_verified_selected_eV=sum(c['error_sum_eV'] for c in cells)/s['quantum_verified'] if s['quantum_verified'] else None,
             unique_selected_endpoints=len({i for c in cells for i in c['selected_indices']}))
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--transfer',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    transfer=json.loads(a.transfer.read_text());goals=json.loads(a.targets.read_text())['single_target_points'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.transfer,a.targets]};indexed={};arms={};seeds=[2091,2092];quantiles=[10,25,50,75,90]
    for target in ['homo','lumo']:
        for control in ['correct','off']:
            path=a.root/f'e14_cohort_{target}_{control}/manifest.json';manifest=json.loads(path.read_text());hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();arms[target]=manifest['arms']
            for r in manifest['records']:indexed[(target,control,r['index'])]=r
    rows=[{**r,**{k:indexed[(r['target'],r['control'],r['index'])][k] for k in ['quantile','inference_seed','candidate_id']}} for r in transfer['records']]
    reports=[]
    for target in ['homo','lumo']:
        j=['homo','lumo'].index(target)
        for arm in arms[target]:
            for control in ['correct','off']:
                candidates=[r for r in rows if r['target']==target and r['arm']==arm and r['control']==control]
                for model in ['baseline','affine','adapted']:
                    for k in [4,8]:
                        cells=[]
                        for q in quantiles:
                            for seed in seeds:
                                pool=[r for r in candidates if r['inference_seed']==seed and (control=='off' or r['quantile']==q)]
                                cells.append(dict(quantile=q,inference_seed=seed,**select(pool,model,j,goals[target][str(q)],k)))
                        reports.append(dict(target=target,arm=arm,control=control,model=model,k=k,unique_raw_proposal_budget=640 if control=='correct' else 128,metrics=aggregate(cells),cells=cells))
    report={'scope':'E15 prespecified top-4/top-8 by predicted point distance, ties by original candidate ID. Only physically accepted rankable endpoints; no replacement of SCF failures or missing slots. Off pools reused across five requests, not 640 unique proposals. Counts over request cells are occurrences; unique endpoint count reported separately. DFT labels never rank or fit here. Does not change original whole-pool generation yields.','reports':reports,'source_sha256':hashes}
    a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
