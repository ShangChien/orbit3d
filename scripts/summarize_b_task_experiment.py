"""Summarize fresh B-task pools, preserving proposal and shared-off denominators."""
import argparse
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from orbit3d.oracles.unimol2_task import file_sha256


def aggregate(pools):
    metrics = [p['metrics'] for p in pools]
    rows = [r for p in pools for r in p['records']]
    legal = [r for r in rows if r['legal']]
    graphs = {r['smiles'] for r in legal}
    scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) or '<ACYCLIC>' for s in graphs}
    return dict(attempted=len(rows), legal=len(legal), evaluated=sum(m['evaluated'] for m in metrics),
        hits=sum(m['hits'] for m in metrics), hit_yield=sum(m['hits'] for m in metrics)/len(rows),
        mae_equal_pool_eV=float(np.mean([m['mae_eV'] for m in metrics])),
        unique_source_novel_hits_per128=float(np.mean([128*m['unique_source_novel_hits']/m['attempted'] for m in metrics])),
        unique_all_asset_novel_hits_per128=float(np.mean([128*m['unique_all_asset_novel_hits']/m['attempted'] for m in metrics])),
        response_slope=float(np.polyfit([p['goal_eV'] for p in pools], [m['prediction_mean_eV'] for m in metrics], 1)[0]),
        std_equal_pool_eV=float(np.mean([m['prediction_std_eV'] for m in metrics])),
        mean_legal_atoms=float(np.mean([r['atoms'] for r in legal])),
        unique_legal_graphs=len(graphs), unique_legal_scaffolds=len(scaffolds),
        top_k={str(k):dict(requested=sum(p['top_k'][str(k)]['requested'] for p in pools),
                          hits=sum(p['top_k'][str(k)]['hits'] for p in pools)) for k in [4,8]},
        cells=[{k:p[k] for k in ['quantile','seed','goal_eV','input_goal_eV','metrics']} for p in pools])


def paired_hit_bootstrap(first, second, repeats=4000):
    key=lambda p:(p['quantile'],p['seed'])
    second={key(p):p for p in second}
    differences={}
    for pool in first:
        other=second[key(pool)]
        assert pool['goal_eV']==other['goal_eV'] and pool['attempted']==other['attempted']
        col=['homo','lumo'].index(pool['target']);goal=pool['goal_eV']
        hit=lambda r:int(r['evaluated'] and abs(r['prediction_eV'][col]-goal)<=.25)
        assert [r['candidate_id'] for r in pool['records']]==[r['candidate_id'] for r in other['records']]
        differences.setdefault(pool['seed'], []).append(np.array([hit(a)-hit(b) for a,b in zip(pool['records'],other['records'])]))
    rng=np.random.default_rng(20261114)
    # Candidate streams are reused across requests, so resample an index jointly
    # across all requests within a seed instead of pretending those are independent.
    streams=[np.stack(ds).mean(0) for ds in differences.values()]
    samples=sum(d[rng.integers(len(d),size=(repeats,len(d)))].mean(1) for d in streams)/len(streams)
    return dict(difference_pp=float(100*np.mean([d.mean() for d in streams])),
        paired_sampling_95_pp=(100*np.quantile(samples,[.025,.975])).tolist(),
        scope='Candidate-stream paired bootstrap preserving cross-request stream reuse within each fixed inference seed; no training-seed uncertainty')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',default='oa_source');a=p.parse_args()
    reports={d.parent.name:json.loads(d.read_text()) for d in sorted((a.root/'stage1_evaluation').glob('*/summary.json'))}
    groups={};results={}
    for name,report in reports.items():
        for control in sorted({p['control'] for p in report['pools']}):
            if control=='off':continue
            pools=[p for p in report['pools'] if p['control']==control]
            groups[(name,control)]=pools;results[name+':'+control]=aggregate(pools)
    reference=groups[(a.reference,'correct')]
    comparisons={name:paired_hit_bootstrap(pools,reference) for (name,control),pools in groups.items() if control=='correct' and name!=a.reference}
    if (a.reference,'shuffled') in groups:
        correct={(p['goal_eV'],p['seed']):p for p in reference}
        for pool in groups[(a.reference,'shuffled')]:
            same_input=correct[(pool['input_goal_eV'],pool['seed'])]
            assert [r['token_sha256'] for r in pool['records']]==[r['token_sha256'] for r in same_input['records']]
        comparisons['correct_minus_shuffled']=paired_hit_bootstrap(reference,groups[(a.reference,'shuffled')])
    shared=reports.get('shared_off')
    off_summary=None
    if shared:
        goals={p['quantile']:p['goal_eV'] for p in reference};off=shared['pools'];cells=[]
        for q,goal in sorted(goals.items()):
            for pool in off:
                rows=pool['records'];col=['homo','lumo'].index(pool['target'])
                scored=[r for r in rows if r['evaluated']];hits=[r for r in scored if abs(r['prediction_eV'][col]-goal)<=.25]
                cells.append(dict(quantile=q,seed=pool['seed'],goal_eV=goal,attempted=len(rows),hits=len(hits),
                    mae_eV=float(np.mean([abs(r['prediction_eV'][col]-goal) for r in scored]))))
        off_summary=dict(unique_proposals=sum(p['attempted'] for p in off),
            hit_yield_equal_request=float(np.mean([c['hits']/c['attempted'] for c in cells])),
            mae_equal_request_eV=float(np.mean([c['mae_eV'] for c in cells])),cells=cells)
    costs={d.parent.name:json.loads(d.read_text()) for d in sorted((a.root/'training').glob('*/costs.json'))}
    report=dict(scope='Fresh raw B-defined task; no force gate or new DFT. Unique novel hits are deduplicated within each 128-proposal request/seed pool and averaged; cross-request unique graph counts are separate. Shuffled controls exactly permute the same conditional inputs and random streams, so are not independent new proposals. B is both training target and acceptance, not independent accuracy.',
        predictor=next(iter(reports.values()))['predictor'],results=results,paired_differences=comparisons,
        shared_off=off_summary,training_costs=costs,
        evaluation_costs={name:{k:r[k] for k in ['elapsed_seconds','B_molecule_calls','B_forward_calls']} for name,r in reports.items()},
        inputs={str(d):file_sha256(d) for d in sorted((a.root/'stage1_evaluation').glob('*/summary.json'))})
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    for name,result in results.items():print(name,{k:v for k,v in result.items() if k not in {'cells','top_k'}})
    print('paired',comparisons);print('shared_off',off_summary)


if __name__=='__main__':main()
