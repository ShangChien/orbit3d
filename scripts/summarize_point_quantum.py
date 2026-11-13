"""Summarize every preselected formal point-control candidate, retaining failures."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def errors(values,denominator):
    x=np.asarray(values,dtype=float)
    return {'evaluated':len(x),'planned':denominator,'coverage':len(x)/denominator if denominator else None,
            'mae_eV':float(np.abs(x).mean()) if len(x) else None,
            'bias_eV':float(x.mean()) if len(x) else None,
            'hits':{str(t):int((np.abs(x)<=t).sum()) for t in [.1,.25,.5]},
            'hit_rate_planned':{str(t):float((np.abs(x)<=t).sum()/denominator) if denominator else None for t in [.1,.25,.5]}}


def summarize(rows,planned=None,optimization_requested=True):
    planned=len(rows) if planned is None else planned
    initial=[r for r in rows if 'singlepoint' in r]
    final=[r for r in rows if 'optimized_singlepoint' in r and r['geometry']['converged'] and r['geometry']['graph_preserved']]
    def delta(r,key):return r[key]['prediction_eV'][['homo','lumo'].index(r['target'])]-r['target_eV']
    result={'optimization_requested':optimization_requested,'planned':planned,'selected':len(rows),'singlepoint_completed':len(initial),'geometry_converged':sum(r.get('geometry',{}).get('converged',False) for r in rows),
        'graph_preserved':sum(r.get('geometry',{}).get('graph_preserved',False) for r in rows),'accepted_optimized':len(final),
        'input_target':errors([delta(r,'singlepoint') for r in initial],planned),
        'input_target_same_accepted_subset':errors([delta(r,'singlepoint') for r in final],planned),
        'optimized_target':errors([delta(r,'optimized_singlepoint') for r in final],planned),
        'dpa_minus_quantum_input':errors([r['dpa_prediction_eV'][['homo','lumo'].index(r['target'])]-r['singlepoint']['prediction_eV'][['homo','lumo'].index(r['target'])] for r in initial if r.get('dpa_prediction_eV') is not None],planned),
        'dpa_target_input':errors([r['dpa_prediction_eV'][['homo','lumo'].index(r['target'])]-r['target_eV'] for r in rows if r.get('dpa_prediction_eV') is not None],planned),
        'failed_or_unaccepted_indices':[r['index'] for r in rows if r not in final],
        'failed_singlepoint_indices':[r['index'] for r in rows if r not in initial]}
    if not optimization_requested:
        for key in ['geometry_converged','graph_preserved','accepted_optimized','input_target_same_accepted_subset','optimized_target','failed_or_unaccepted_indices']:result[key]=None
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());completed=json.loads((a.results/'completed.json').read_text());assert completed['attempted']==manifest['selected']
    optimize=completed.get('optimization_requested',any('--optimize' in job['command'] for job in completed['jobs']))
    rows=[];hashes={str(a.manifest):hashlib.sha256(a.manifest.read_bytes()).hexdigest(),str(a.results/'completed.json'):hashlib.sha256((a.results/'completed.json').read_bytes()).hexdigest()}
    for source in manifest['records']:
        i=source['index'];path=a.results/f'job{i:02d}/result.json'
        result=json.loads(path.read_text()) if path.exists() else {'status':'worker_failed','reason':'Worker produced no result; consult preserved job log'}
        if path.exists():
            assert result['input_sha256']==manifest['sdf_sha256'] and result['index']==i
            assert result['status']!='running','Queue has an unfinished result'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if 'steps' in result.get('geometry',{}):
            # The frozen initial worker named callback evaluations 'steps'.
            result['geometry']['gradient_evaluations']=result['geometry'].pop('steps')
        rows.append({**source,**result})
    full_pool='proposal_denominators' in manifest
    selection_rules=['all_terminal_accepted'] if full_pool else ['top1','fixed_hash']
    report={'scope':('Entire physically accepted pool; these rates condition on terminal selection. Use the full-proposal report for original proposal yields.' if full_pool else 'Pre-frozen exploratory cohort; no candidate replacement, no population quantum success claim. Target/seed assignment is matched across policies but not independently crossed.'),
        'optimization_requested':optimize,
        'protocol':'PBE0/def2-TZVP/DF(def2-universal-jkfit), grid3, SCF1e-8'+('; PBE-D3BJ/def2-TZVP optimization max80' if optimize else '; input-geometry singlepoints only')+'; distinct from source FHI-aims Tier2/PBE-TS.',
        'selection':{k:v for k,v in manifest.items() if k!='records'},'elapsed_seconds':completed['elapsed_seconds'],
        'execution_metadata':{k:v for k,v in completed.items() if k!='jobs'},
        'all':summarize(rows,manifest['expected'],optimize),'groups':{f'{target}/{arm}':summarize([r for r in rows if r['target']==target and r['arm']==arm],manifest.get('planned_per_group',{}).get(f'{target}/{arm}'),optimize) for target in manifest.get('targets',['homo','lumo']) for arm in manifest.get('arms',['continue_wall','grpo200'])},
        'by_selection':{rule:summarize([r for r in rows if r['selection']==rule],manifest['expected'] if full_pool else manifest['expected']//2,optimize) for rule in selection_rules},
        'records':rows,'source_sha256':hashes}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['groups'],indent=2))


if __name__=='__main__':main()
