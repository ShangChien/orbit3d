"""Aggregate fresh point scans without multiplying reused control pools."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiment_point_generation_scan import target_metrics

def select_candidates(rows,index,target,retain):
    ranked=sorted([r for r in rows if 'prediction' in r and r.get('domain',{}).get('in_domain',False)],
                  key=lambda r:(abs(r['prediction'][index]-target),r['candidate_id']))
    chosen=[];seen=set()
    for row in ranked:
        if row['smiles'] in seen:continue
        chosen.append(row);seen.add(row['smiles'])
        if len(chosen)==retain:break
    return chosen

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--plot',type=Path)
a=p.parse_args()
report={'scope':'Two separately trained single-property OE62 point policies; three inference seeds, not independent training seeds. Raw-pool evidence separated from ranking/relaxation.',
        'runtime_commit':'1fcd1bd','gpu_indices':[2,3],'properties':{},'generation_attempts':0,
        'remote_artifacts':'/sjtu-combustion/wjw/chensq/orbit_generation_scan_5fe5/campaign'}
for prop,index in [('homo',0),('lumo',1)]:
    root=a.input/prop;d=json.loads((root/'summary.json').read_text());runs=d['runs'];points=d['protocol']['points_eV'];qs=list(points)
    assert len(runs)==18 and len({(r['seed'],r['quantile']) for r in runs})==18
    assert d['protocol']['visible_devices']==str(2 if prop=='homo' else 3)
    assert d['protocol']['active_targets']==[prop] and d['protocol']['trained_steps']==5000
    assert all(r['attempted']==64 and len(r['raw_records'])==64 for r in runs)
    report['generation_attempts']+=sum(r['attempted'] for r in runs)
    out={'protocol':d['protocol'],'targets':{},'speed_replays':d['speed_replays'],
         'input_manifest':json.loads((root/'input_manifest.json').read_text()),
         'run_manifest':json.loads((root/'run_manifest.json').read_text()),
         'raw_summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()}
    off=[row for r in runs if r['quantile']=='off' for row in r['raw_records']]
    assert len(off)==192
    for qi,q in enumerate(qs):
        value=points[q];group=[r for r in runs if r['quantile']==q]
        raw=[row for r in group for row in r['raw_records']]
        selected=[r['raw_records'][i] for r in group for i in r['selected_ids']]
        refined=[row for r in group for row in r['refined_records']]
        for pool in group:
            assert [r['candidate_id'] for r in select_candidates(pool['raw_records'],index,value,d['protocol']['retain'])]==pool['selected_ids']
        off_selected=[row for pool in runs if pool['quantile']=='off' for row in select_candidates(pool['raw_records'],index,value,d['protocol']['retain'])]
        wrong_q=qs[(qi+2)%len(qs)]
        wrong=[row for r in runs if r['quantile']==wrong_q for row in r['raw_records']]
        item={'value_eV':value,'attempted':len(raw),'valid':sum(r['valid'] for r in raw),
              'unique_valid':len({r['smiles'] for r in raw if r['valid']}),'novel_valid':sum(r.get('novel',False) for r in raw),
              'mean_valid_atoms':float(np.mean([r['atoms'] for r in raw if r['valid']])),
              'shuffled_input_quantile':wrong_q,'shuffled_input_value_eV':points[wrong_q],
              'controls_note':'Off pool reused across targets; shuffled is another already generated target pool with identical seeds, scored against this requested value.',
              'selected':len(selected),'refinement_completed':sum(r['status']=='completed' for r in refined),
              'refinement_failure_reasons':[r['reason'] for r in refined if r['status']!='completed'],
              'metrics':{}}
        for stage,rows in [('raw',raw),('off_rescored',off),('off_selected',off_selected),('shuffled_rescored',wrong),('selected',selected),('refined',refined)]:
            item['metrics'][stage]={}
            for model,field in [('dpa','prediction'),('fingerprint','independent_prediction')]:
                item['metrics'][stage][model]={}
                for subset,domain in [('in_domain',True),('all_evaluable_diagnostic',False)]:
                    m=target_metrics(rows,index,value,field,domain)
                    m['hit_rate_per_generation_attempt']=m['hits']['0.25']/192
                    m['joint_rms01_rate_per_generation_attempt']=m['hits_rms01']/192
                    item['metrics'][stage][model][subset]=m
        item['off_selected_count']=len(off_selected)
        pairs=[(pool['raw_records'][r['candidate_id']],r) for pool in group for r in pool['refined_records'] if r['status']=='completed']
        item['paired_refinement_change']={}
        for model,field in [('dpa','prediction'),('fingerprint','independent_prediction')]:
            delta=[abs(after[field][index]-value)-abs(before[field][index]-value) for before,after in pairs]
            item['paired_refinement_change'][model]={'paired_count':len(pairs),'mean_after_minus_before_abs_target_error_eV':float(np.mean(delta)) if delta else None,
                'new_hits':sum(abs(b[field][index]-value)>.25 and abs(c[field][index]-value)<=.25 for b,c in pairs),
                'lost_hits':sum(abs(b[field][index]-value)<=.25 and abs(c[field][index]-value)>.25 for b,c in pairs),
                'max_abs_prediction_change_eV':max([abs(c[field][index]-b[field][index]) for b,c in pairs],default=None)}
        item['stage_timing_seconds']={stage:{'values':[r['stage_seconds'][stage] for r in group],
                    'median':float(np.median([r['stage_seconds'][stage] for r in group]))} for stage in group[0]['stage_seconds']}
        item['seed_metrics']=[{'seed':r['seed'],'valid':r['valid'],
            'dpa':target_metrics(r['raw_records'],index,value),'fingerprint':target_metrics(r['raw_records'],index,value,'independent_prediction')} for r in group]
        item['tokens_equal_to_off']=sum(row['token_sha256']==next(r for r in runs if r['quantile']=='off' and r['seed']==pool['seed'])['raw_records'][row['candidate_id']]['token_sha256'] for pool in group for row in pool['raw_records'])
        out['targets'][q]=item
    response={}
    for model in ['dpa','fingerprint']:
        for subset in ['in_domain','all_evaluable_diagnostic']:
            pairs=[(points[q],out['targets'][q]['metrics']['raw'][model][subset]['mean_prediction_eV']) for q in qs]
            usable=[p for p in pairs if p[1] is not None]
            response[model+'/'+subset]={'requested_and_mean_predicted':pairs,
                'descriptive_slope':float(np.polyfit(np.array(usable)[:,0],np.array(usable)[:,1],1)[0]) if len(usable)>1 else None}
    for model in ['dpa','fingerprint']:
        for subset in ['in_domain','all_evaluable_diagnostic']:
            slopes={}
            source_subset='in_domain' if subset=='in_domain' else 'all_evaluable'
            for seed in d['protocol']['seeds']:
                xy=[(points[q],next(r for r in runs if r['seed']==seed and r['quantile']==q)['raw_metrics'][model+'/'+source_subset]['mean_prediction_eV']) for q in qs]
                xy=[pair for pair in xy if pair[1] is not None]
                slopes[str(seed)]=float(np.polyfit(np.array(xy)[:,0],np.array(xy)[:,1],1)[0]) if len(xy)>1 else None
            response[model+'/'+subset]['descriptive_slope_by_sampling_seed']=slopes
    out['response']=response
    out['task_aggregate']={'correct_generation_attempts':960,'off_unique_generation_attempts':192,'off_task_evaluations':960,
        'note':'Off candidates are reused across five requested targets; totals are task scores, not counts of unique molecules or independent repeated controls.',
        'valid_correct':sum(t['valid'] for t in out['targets'].values()),
        'domain_correct':sum(t['metrics']['raw']['dpa']['in_domain']['evaluated'] for t in out['targets'].values())}
    for model in ['dpa','fingerprint']:
        out['task_aggregate'][model]={stage:{'hits':sum(t['metrics'][stage][model]['in_domain']['hits']['0.25'] for t in out['targets'].values()),
            'macro_target_mae_eV':float(np.mean([t['metrics'][stage][model]['in_domain']['mae_eV'] for t in out['targets'].values()]))} for stage in ['raw','off_rescored','shuffled_rescored']}
    off_seconds=[r['stage_seconds']['generation'] for r in runs if r['quantile']=='off']
    replay=[r['generation_seconds'] for r in d['speed_replays']]
    assert len(replay)==3 and all(r.get('tokens_equal',True) for r in d['speed_replays'])
    out['timing_control']={'off_generation_seconds':off_seconds,'off_median':float(np.median(off_seconds)),
        'same_p50_seed2026_replay_seconds':replay,'same_p50_median':float(np.median(replay)),
        'same_p50_molecules_per_second':64/float(np.median(replay)),
        'note':'Repeated P50 timing uses identical trajectories. Off/other targets have different atom counts and lengths; this is not a point-versus-interval architecture speedup.'}
    # Keep every proposal outcome without repeating bulky symbol lists or optimizer histories.
    out['runs']=[]
    for r in runs:
        c={k:v for k,v in r.items() if k not in ['raw_records','refined_records']}
        c['raw_records']=[{k:v for k,v in row.items() if k!='symbols'} for row in r['raw_records']]
        c['refined_records']=[{k:v for k,v in row.items() if k!='history'} for row in r['refined_records']]
        out['runs'].append(c)
    report['properties'][prop]=out
assert report['generation_attempts']==2304
jobs=json.loads((a.input/'jobs.json').read_text());assert {j['gpu'] for j in jobs}=={2,3} and all(j['exit_code']==0 for j in jobs)
report['jobs']=jobs;report['campaign_completed']=json.loads((a.input/'completed.json').read_text())
samples=[json.loads(line) for line in (a.input/'gpu_samples.jsonl').read_text().splitlines()]
concurrent=[];observed={}
for sample in samples:
    relevant=set()
    for line in sample['gpu_csv'].splitlines():
        cols=[s.strip() for s in line.split(',')]
        if cols and cols[0] in ['2','3']:relevant.add(cols[1])
    current={uuid:[] for uuid in relevant}
    for line in sample['process_csv'].splitlines():
        cols=[s.strip() for s in line.split(',')]
        if len(cols)>1 and cols[0] in relevant:
            current[cols[0]].append(cols[1]);observed.setdefault(cols[0],set()).add(cols[1])
    for uuid,pids in current.items():
        if len(pids)>1:concurrent.append({'unix':sample['unix'],'gpu_uuid':uuid,'host_pids':pids})
report['gpu_monitor']={'samples':len(samples),'interval_seconds':10,'concurrent_process_observations_on_reserved_gpus':concurrent,'observed_host_pids':{k:sorted(v) for k,v in observed.items()},
    'sha256':hashlib.sha256((a.input/'gpu_samples.jsonl').read_bytes()).hexdigest(),'note':'GPU0/1 activity is outside this experiment; GPUs share a host. NVIDIA reports host PIDs, which cannot be directly compared with container job PIDs. Sampling cannot rule out brief unobserved contention.'}
a.output.write_text(json.dumps(report,indent=2)+'\n')
for prop,data in report['properties'].items():
    print(prop,'timing',data['timing_control'])
    for q,item in data['targets'].items():
        m=item['metrics'];print(q,item['value_eV'],'valid',item['valid'],'domain',m['raw']['dpa']['in_domain']['evaluated'],
            'MAE correct/off/wrong',[m[k]['dpa']['in_domain']['mae_eV'] for k in ['raw','off_rescored','shuffled_rescored']],
            'DPA hits',m['raw']['dpa']['in_domain']['hits']['0.25'],'FP hits',m['raw']['fingerprint']['in_domain']['hits']['0.25'])
    print('response',data['response'])
print('GPU monitor',report['gpu_monitor'])

if a.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for ax,prop in zip(axes,['homo','lumo']):
        data=report['properties'][prop];items=list(data['targets'].values())
        x=np.array([r['value_eV'] for r in items])
        ax.plot(x,x,color='#9ca3af',linestyle='--',label='Perfect mean tracking')
        for model,color,label in [('dpa','#c45132','DPA mean'),('fingerprint','#2c6caa','Independent FP mean')]:
            y=[r['metrics']['raw'][model]['in_domain']['mean_prediction_eV'] for r in items]
            ax.plot(x,y,'o-',color=color,label=label,linewidth=2,markersize=5)
        off=items[0]['metrics']['off_rescored']['dpa']['in_domain']['mean_prediction_eV']
        if off is not None:ax.axhline(off,color='#c45132',linestyle=':',label='Condition off: DPA')
        ax.set(xlabel=f'Requested {prop.upper()} (eV)',ylabel=f'Predicted {prop.upper()} (eV)',title=prop.upper()+'-only policy')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.055))
    fig.suptitle('Fresh point-conditioned generation: target response',fontsize=14)
    fig.text(.5,.02,'In-domain raw proposals only; 192 attempts/target; 3 sampling seeds; frozen SFT; surrogate evaluation.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.18,1,.94]);a.plot.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.plot,dpi=180,bbox_inches='tight');plt.close(fig)
