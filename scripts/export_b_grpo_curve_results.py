"""Archive completed E21/E22 evidence without copying weights or training trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np

from orbit3d.oracles.unimol2_task import file_sha256


def timing_summary(root):
    rows=json.loads((root/'training/stage_timings.json').read_text())
    steady=[r for r in rows if r.get('profiled') and r['group']>1]
    keys=sorted({k for r in steady for k in r['stage_seconds']})
    costs_by_group={r['groups']:r for r in json.loads((root/'training/cost_curve.json').read_text())}
    checkpoints={g:{**costs_by_group[g], 'checkpoint_receipt':json.loads((root/f'training/octree_group_{g}.receipt.json').read_text())}
                 for g in [64,128,256,512,1024]}
    return dict(
        costs=json.loads((root/'training/costs.json').read_text()),
        cost_at_checkpoints=checkpoints,
        group_wall_median_p95_seconds=np.quantile([r['group_wall_seconds'] for r in rows if r['event']=='group'],[.5,.95]).tolist(),
        steady_profile_groups=[r['group'] for r in steady],
        stage_median_seconds={k:float(np.median([r['stage_seconds'][k] for r in steady if k in r['stage_seconds']])) for k in keys},
        probes=[r for r in rows if r.get('gradient_probes')],
        scope='Synchronized sequential stage walls on GPU0. Medians exclude group1 warmup; sparse probe groups, not all updates. Evaluation is excluded from performance claims; device routing is recorded in the protocols and execution addenda.')


def validation_structure_summary(root):
    out={}
    for group in [0,1024]:
        pools=json.loads((root/f'validation/group{group}/summary.json').read_text())['pools']
        rows=[]
        for pool in pools:
            for r in pool['records']:
                rows.append({**r,'hit':bool(r['evaluated'] and abs(r['prediction_eV'][0]-pool['goal_eV'])<=.25)})
        bins={}
        for low,high in [(0,15),(16,25),(26,40),(41,80),(81,1000000)]:
            subset=[r for r in rows if low<=r['atoms']<=high]
            bins[f'{low}-{high}' if high<1000000 else '81+']=dict(
                attempted=len(subset),legal=sum(r['legal'] for r in subset),
                evaluated=sum(r['evaluated'] for r in subset),hits=sum(r['hit'] for r in subset))
        legal=[r for r in rows if r['legal']]
        out[group]=dict(raw_atom_bins=bins,
            legal_atoms_min_q25_median_q75_max=np.quantile([r['atoms'] for r in legal],[0,.25,.5,.75,1]).tolist(),
            scope='Counts retain failures and use raw generated total atom count, including hydrogen. Descriptive strata, not size-matched causal evidence.')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direct',type=Path,required=True);p.add_argument('--margin',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    roots={'direct':a.direct,'margin':a.margin}
    assert all((r/'all_done').exists() for r in roots.values())
    report=dict(status='Complete: both1024-group trajectories, all development checkpoints, fresh validation and paired comparison',
        comparison=json.loads((a.margin/'paired_comparison.json').read_text()),
        protocols={k:json.loads((r/'protocol.json').read_text()) for k,r in roots.items()},
        selections={k:json.loads((r/'checkpoint_selection.json').read_text()) for k,r in roots.items()},
        per_route_statistics={k:{phase:json.loads((r/f'{phase}_statistics.json').read_text())
                                for phase in ['development','validation']} for k,r in roots.items()},
        timing={k:timing_summary(r) for k,r in roots.items()},
        validation_structure={k:validation_structure_summary(r) for k,r in roots.items()},
        input_hashes={},export_script_sha256=file_sha256(Path(__file__)))
    for root in roots.values():
        for rel in ['protocol.json','checkpoint_selection.json','development_statistics.json','validation_statistics.json',
                    'training/costs.json','training/cost_curve.json','training/stage_timings.json']:
            path=root/rel;report['input_hashes'][str(path)]=file_sha256(path)
        for g in [64,128,256,512,1024]:
            path=root/f'training/octree_group_{g}.receipt.json';report['input_hashes'][str(path)]=file_sha256(path)
    path=a.margin/'paired_comparison.json';report['input_hashes'][str(path)]=file_sha256(path)
    report['execution_addenda']={}
    for name,root in roots.items():
        path=root/'execution_addendum.json'
        if path.exists():
            report['execution_addenda'][name]=json.loads(path.read_text())
            report['input_hashes'][str(path)]=file_sha256(path)
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
