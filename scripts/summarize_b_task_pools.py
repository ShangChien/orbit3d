"""Aggregate original requests and explicitly rescored off controls under frozen B."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from loom.oracles.unimol2_task import graph_set, summarize_task, file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--specs',type=Path,required=True)
    p.add_argument('--source-train',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=json.loads(a.report.read_text());source=graph_set(a.source_train);assets={}
    for spec in json.loads(a.specs.read_text()):
        known=set(source)
        for path in spec.get('asset_graph_files',[]):known.update(json.loads(Path(path).read_text()))
        for path in spec.get('asset_sdfs',[]):known.update(graph_set(path))
        assets[spec['name']]=known
    groups=defaultdict(list)
    for pool in raw['pools']:groups[(pool['campaign'],pool['target'],pool['arm'],pool['control'])].append(pool)
    results=[]
    def cell(rows,target,goal,known):
        s=summarize_task(rows,target,goal,source,known)
        s['loss_partition']=dict(structure_failed=len(rows)-s['legal'],unsupported=s['unsupported'],
            prediction_failed=s['legal']-s['unsupported']-s['evaluated'],property_miss=s['evaluated']-s['hits'],
            repeated_hit=s['hits']-s['unique_hit_graphs'],known_asset_unique_hit=s['unique_hit_graphs']-s['unique_all_asset_novel_hits'],
            unique_novel_hit=s['unique_all_asset_novel_hits'])
        assert sum(s['loss_partition'].values())==len(rows)
        s['by_size']={name:summarize_task([r for r in rows if lo<=r['atoms']<=hi],target,goal,source,known)
                      for name,lo,hi in [('empty',0,0),('1-20',1,20),('21-40',21,40),('41-80',41,80),('over80',81,100000)]}
        return s
    for (campaign,target,arm,control),pools in groups.items():
        if control!='correct':continue
        requests=sorted({(int(pool['quantile']),float(pool['goal_eV'])) for pool in pools})
        off_key=('E16_shared_off',target,'shared_off','off') if campaign.startswith('E16_') else (campaign,target,arm,'off')
        off_pools=groups[off_key];off_rows=[row for pool in off_pools for row in pool['records']]
        cells={};off={};all_rows=[row for pool in pools for row in pool['records']]
        for q,goal in requests:
            rows=[row for pool in pools if int(pool['quantile'])==q for row in pool['records']]
            cells[str(q)]=dict(goal_eV=goal,**cell(rows,target,goal,assets[campaign]))
            off[str(q)]=dict(goal_eV=goal,**cell(off_rows,target,goal,assets[campaign]))
        def aggregate(cs):
            return dict(hit_yield=float(np.mean([c['hit_yield'] for c in cs.values()])),
                mae_equal_request_eV=float(np.mean([c['mae_eV'] for c in cs.values()])),
                unique_source_novel_hits_per128=float(np.mean([128*c['unique_source_novel_hits']/c['attempted'] for c in cs.values()])),
                unique_all_asset_novel_hits_per128=float(np.mean([128*c['unique_all_asset_novel_hits']/c['attempted'] for c in cs.values()])),
                response_slope=float(np.polyfit([c['goal_eV'] for c in cs.values()],[c['prediction_mean_eV'] for c in cs.values()],1)[0]),
                std_equal_request_eV=float(np.mean([c['prediction_std_eV'] for c in cs.values()])))
        off_selection={}
        column=['homo','lumo'].index(target)
        for k in [4,8]:
            chosen=[];hits=0
            for _,goal in requests:
                for pool in off_pools:
                    selected=sorted([r for r in pool['records'] if r['evaluated']],key=lambda r:(abs(r['prediction_eV'][column]-goal),r['candidate_id']))[:k]
                    chosen.extend(selected);hits+=sum(abs(r['prediction_eV'][column]-goal)<=.25 for r in selected)
            off_selection[str(k)]=dict(requested=len(requests)*len(off_pools)*k,selected=len(chosen),hits=hits,
                unique_graphs_across_requests=len({r['smiles'] for r in chosen}),unique_raw_proposals=len(off_rows))
        result=dict(campaign=campaign,target=target,arm=arm,attempted=len(all_rows),legal=sum(r['legal'] for r in all_rows),
            evaluated=sum(r['evaluated'] for r in all_rows),metrics=aggregate(cells),cells=cells,
            shared_off=dict(unique_proposals=len(off_rows),metrics=aggregate(off),cells=off,top_k=off_selection),
            top_k={str(k):dict(requested=sum(pool['top_k'][str(k)]['requested'] for pool in pools),
                              selected=sum(pool['top_k'][str(k)]['attempted'] for pool in pools),
                              hits=sum(pool['top_k'][str(k)]['hits'] for pool in pools)) for k in [4,8]})
        results.append(result)
    report=dict(scope='Historical raw coordinates; same fixed B defines success. Request-equal metrics; unique hits deduplicated within each request across inference seeds. Off pool reused across goals, not independent new proposals. Training novelty references include declared source and per-arm rollout/critic fitting assets; foundation pretraining overlap unknown.',
                predictor=raw['predictor'],unique_proposals=sum(pool['attempted'] for pool in raw['pools']),
                B_molecule_calls=raw['B_molecule_calls'],B_forward_calls=raw['B_forward_calls'],elapsed_seconds=raw['elapsed_seconds'],comparisons=results,
                source_sha256={str(p):file_sha256(p) for p in [a.report,a.specs,a.source_train]})
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    for row in results:print(row['campaign'],row['arm'],row['metrics'],flush=True)


if __name__=='__main__':main()
