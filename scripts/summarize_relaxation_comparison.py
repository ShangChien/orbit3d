"""Summarize fixed-cohort geometry outcomes without dropping chemical failures."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def distribution(values):
    return {'n':len(values),'mean':float(np.mean(values)),'p10_p50_p90':np.quantile(values,[.1,.5,.9]).tolist()} if values else {'n':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.run/'summary.json';data=json.loads(raw.read_text());cohort=json.loads(a.cohort.read_text())
    samples={r['id']:r for r in cohort['samples']};records=data['records'];aggregate=[]
    for r in records:
        sample=samples[r['id']];column=['homo','lumo'].index(r['target']);goal=sample['target_point'][r['target']]
        r['graph_preserved']=bool(r.get('geometry_valid') and r.get('connectivity_preserved'))
        r['numerical_final_rms_force_eVA']=r['initial']['dpa']['rms_force_eVA'] if r['method']=='none' else r.get('optimization',{}).get('final_force_eva')
        r['numerical_final_max_force_eVA']=r['initial']['dpa']['max_force_eVA'] if r['method']=='none' else r.get('optimization',{}).get('final_max_force_eva')
        for backend in ['dpa','hyff']:
            r[f'{backend}_initial_target_error_eV']=abs(r['initial'][backend]['prediction_eV'][column]-goal)
        # Keep scalar force traces; full evaluated coordinates remain in the hashed remote summary.
        for entry in r['evaluation_trace']:entry.pop('positions_A')
        if 'optimization' in r:
            for key in ['initial_forces_eva','final_forces_eva']:r['optimization'].pop(key,None)
    for target in ['all','homo','lumo']:
        for method in data['methods']:
            rows=[r for r in records if r['method']==method and (target=='all' or r['target']==target)]
            eligible=[r for r in rows if r['completed'] and r['graph_preserved']]
            force=lambda r:r['numerical_final_rms_force_eVA']
            item={'target':target,'method':method,'attempted':len(rows),'graph_preserved':sum(r['graph_preserved'] for r in rows),
                  'property_evaluated':len(eligible),'numerical_rms_below_0.1':sum(force(r) is not None and force(r)<=.1 for r in rows),
                  'graph_and_rms_below_0.1':sum(force(r)<=.1 for r in eligible),
                  'graph_and_rms_below_0.5':sum(force(r)<=.5 for r in eligible),
                  'joint_success':sum(r['joint_success'] for r in rows),'energy_force_calls':sum(r['energy_force_calls'] for r in rows),
                  'property_gradient_calls':sum(r['property_gradient_calls'] for r in rows),
                  'post_property_calls':sum(r['post_property_calls'] for r in rows),
                  'wall_seconds':sum(r['wall_seconds'] for r in rows),
                  'final_rms_force_eVA':distribution([force(r) for r in rows if force(r) is not None]),
                  'graph_preserved_rms_force_eVA':distribution([force(r) for r in eligible])}
            for backend in ['dpa','hyff']:
                before=[r[f'{backend}_initial_target_error_eV'] for r in eligible];after=[r[f'{backend}_target_error_eV'] for r in eligible]
                item[backend]={'initial_error_same_evaluable_subset_eV':distribution(before),'final_error_eV':distribution(after),
                               'paired_error_change_eV':distribution((np.asarray(after)-before).tolist()),
                               'property_hits':sum(r[f'{backend}_property_hit'] for r in eligible)}
            aggregate.append(item)
    summary={k:v for k,v in data.items() if k!='records'}
    summary.update(aggregate=aggregate,records=records,
        interpretation='All frozen starting points remain in denominators. Target MAE uses graph-preserved evaluable cases, paired with their own initial values. Numerical convergence alone does not imply chemical success.',
        force_trace_interpretation='Actual energy/force evaluations, including rejected trials in current Full/energy descent. These traces are not accepted-state or graph-validity curves.',
        input_manifest=json.loads((a.run/'input_manifest.json').read_text()),
        source_manifest=json.loads((a.run/'run_manifest.json').read_text()),
        raw_summary_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),cohort_sha256=hashlib.sha256(a.cohort.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    for r in aggregate:
        if r['target']=='all':print(r['method'],r['attempted'],r['graph_preserved'],r['graph_and_rms_below_0.1'],r['joint_success'],r['energy_force_calls'])


if __name__=='__main__':main()
