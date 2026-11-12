"""Summarize fixed-SFT inference controls; retain failed proposal denominators."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ARMS=['baseline','cfg3','cool','cfg3_cool']
FIELDS={'dpa':'prediction','fingerprint':'independent_prediction'}

def error_metrics(rows,index,target,field,in_domain=True):
    rows=[r for r in rows if field in r and (not in_domain or r.get('domain',{}).get('in_domain'))]
    err=np.array([r[field][index]-target for r in rows])
    if not len(err):return {'evaluated':0,'mae':None,'rmse':None,'bias':None,'std':None,'hits025':0,'hits010':0,'p90_abs':None}
    return {'evaluated':len(err),'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.square(err).mean())),
            'bias':float(err.mean()),'std':float(err.std()),'hits025':int((abs(err)<=.25).sum()),
            'hits010':int((abs(err)<=.1).sum()),'p90_abs':float(np.quantile(abs(err),.9))}

def diversity(rows):
    smiles=sorted({r['smiles'] for r in rows if r.get('valid')})
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fps=[fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    distances=[1-v for i,f in enumerate(fps) for v in DataStructs.BulkTanimotoSimilarity(f,fps[:i])]
    return {'unique':len(smiles),'unique_pairwise_tanimoto_distance':float(np.mean(distances)) if distances else None}

def select(rows,index,target,k=4):
    candidates=sorted([r for r in rows if 'prediction' in r and r['domain']['in_domain']],key=lambda r:(abs(r['prediction'][index]-target),r['candidate_id']))
    chosen=[];seen=set()
    for r in candidates:
        if r['smiles'] in seen:continue
        chosen.append(r);seen.add(r['smiles'])
        if len(chosen)==k:break
    return chosen

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path,required=True)
    p.add_argument('--baseline-reference',type=Path)
    a=p.parse_args();report={'scope':'Frozen SFT, three inference seeds, three targets, 32 attempts/cell. No training or coordinate optimization. Surrogate errors, not quantum truth.',
      'novelty_scope':'Relative to 8192 SFT training_smiles; not all pretraining data.',
      'runtime':'1fcd1bd','arms':ARMS,'properties':{},'raw_predictions':{},'total_attempts':0,
      'remote_artifacts':'/sjtu-combustion/wjw/chensq/orbit_generation_scan_5fe5/concentration',
      'protocol_note':'Worker off_control/geometry prose inherited original-scan defaults; commands and outcomes verify --skip-off --retain 0. No off pools or geometry steps occurred. Future worker prose corrected.',
      'ranking':'Post-hoc top1/top4 unique in-domain molecules per 32 attempts using DPA distance; same-pool baseline, no extra generation. DPA improvement by selection is expected; FP is independent but not ground truth.',
      'uncertainty':'Three matched inference seeds only; seed-wise differences reported, no molecule-independent p-values. Temperature changes all three token heads simultaneously.'}
    sources={}
    for prop,index in [('homo',0),('lumo',1)]:
        dest={};report['properties'][prop]=dest;report['raw_predictions'][prop]={}
        common=None;source=None
        for arm in ARMS:
            path=a.input/f'{prop}_{arm}';d=json.loads((path/'summary.json').read_text())
            manifest=json.loads((path/'input_manifest.json').read_text());runtime=json.loads((path/'run_manifest.json').read_text())
            if common is None:common=manifest;source=runtime['source_sha256']
            assert manifest==common and runtime['source_sha256']==source
            assert d['protocol']['visible_devices']==str(2 if prop=='homo' else 3)
            assert d['protocol']['trained_steps']==5000 and len(d['runs'])==9
            assert {(r['seed'],r['quantile']) for r in d['runs']}=={(s,q) for s in [2026,2027,2028] for q in ['25','50','75']}
            entry={'sampling':d['protocol']['sampling'],'targets':{},'summary_sha256':hashlib.sha256((path/'summary.json').read_bytes()).hexdigest()}
            dest[arm]=entry;report['raw_predictions'][prop][arm]=[]
            for q in ['25','50','75']:
                runs=[r for r in d['runs'] if r['quantile']==q];rows=[r for run in runs for r in run['raw_records']];value=runs[0]['scoring_value']
                assert len(rows)==96 and all(not r['refined_records'] and r['attempted']==32 for r in runs)
                report['total_attempts']+=len(rows)
                chosen=[r for run in runs for r in select(run['raw_records'],index,value)]
                e={'value':value,'attempted':len(rows),'valid':sum(r['valid'] for r in rows),'novel':sum(r.get('novel',False) for r in rows),
                   'domain':sum(r.get('domain',{}).get('in_domain',False) for r in rows),'diversity':diversity(rows),
                   'domain_mean_atoms':float(np.mean([r['atoms'] for r in rows if r.get('domain',{}).get('in_domain')])),
                   'domain_rms_force_p50_p90':np.quantile([r['rms_force'] for r in rows if 'rms_force' in r and r.get('domain',{}).get('in_domain')],[.5,.9]).tolist(),
                   'domain_DPA_FP_disagreement_mae':float(np.mean([abs(r['prediction'][index]-r['independent_prediction'][index]) for r in rows if 'prediction' in r and 'independent_prediction' in r and r.get('domain',{}).get('in_domain')])),
                   'mean_atoms':float(np.mean([r['atoms'] for r in rows if r['valid']])) if any(r['valid'] for r in rows) else None,
                   'generation_seconds':[r['stage_seconds']['generation'] for r in runs],
                   'metrics':{model:error_metrics(rows,index,value,field) for model,field in FIELDS.items()},
                   'all_evaluable_diagnostic':{model:error_metrics(rows,index,value,field,False) for model,field in FIELDS.items()},
                   'top4':{'retained':len(chosen),'available_slots':12,'metrics':{model:error_metrics(chosen,index,value,field) for model,field in FIELDS.items()}},
                   'seeds':{str(run['seed']):{model:error_metrics(run['raw_records'],index,value,field) for model,field in FIELDS.items()} for run in runs}}
                e['selection_budget_curve']={}
                for budget in [8,16,32]:
                    for keep in [1,4]:
                        retained=[r for run in runs for r in select(run['raw_records'][:budget],index,value,keep)]
                        e['selection_budget_curve'][f'{budget}_top{keep}']={'attempted':budget*3,'retained':len(retained),'available_slots':keep*3,
                            'metrics':{model:error_metrics(retained,index,value,field) for model,field in FIELDS.items()}}
                entry['targets'][q]=e
                for run in runs:
                    for row in run['raw_records']:
                        report['raw_predictions'][prop][arm].append(dict(seed=run['seed'],quantile=q,target=value,**{k:v for k,v in row.items() if k!='symbols'}))
            sources[prop]={'input_manifest':manifest,'runtime_source_sha256':source}
        for arm in ARMS:
            cells=list(dest[arm]['targets'].values())
            dest[arm]['aggregate']={'attempted':288,'valid':sum(c['valid'] for c in cells),'domain':sum(c['domain'] for c in cells),
                'generation_seconds':sum(sum(c['generation_seconds']) for c in cells),
                'metrics':{model:{'equal_target_mae':float(np.mean([c['metrics'][model]['mae'] for c in cells])),
                                  'equal_target_std':float(np.mean([c['metrics'][model]['std'] for c in cells])),
                                  'hits025':sum(c['metrics'][model]['hits025'] for c in cells)} for model in FIELDS}}
            if arm!='baseline':
                dest[arm]['paired_seed_deltas']={model:[{'seed':seed,'mean_target_mae_delta':float(np.mean([dest[arm]['targets'][q]['seeds'][str(seed)][model]['mae']-dest['baseline']['targets'][q]['seeds'][str(seed)][model]['mae'] for q in ['25','50','75']]))} for seed in [2026,2027,2028]] for model in FIELDS}
            print(prop,arm,json.dumps(dest[arm]['aggregate']))
    assert report['total_attempts']==2304
    report['inputs']=sources;report['completed']=json.loads((a.input/'completed.json').read_text())
    assert len(report['completed']['jobs'])==8 and all(j['gpu'] in [2,3] and j['exit_code']==0 for j in report['completed']['jobs'])
    samples=[json.loads(line) for line in (a.input/'gpu_samples.jsonl').read_text().splitlines()]
    cooccupancy=[]
    for sample in samples:
        reserved={line.split(',')[1].strip() for line in sample['gpus'].splitlines() if line.split(',')[0].strip() in ['2','3']}
        for uuid in reserved:
            processes=[line for line in sample['processes'].splitlines() if line.split(',')[0].strip()==uuid]
            if len(processes)>1:cooccupancy.append({'unix':sample['unix'],'gpu_uuid':uuid,'processes':processes})
    report['gpu_monitor']={'samples':len(samples),'cooccupancy':cooccupancy,
        'sha256':hashlib.sha256((a.input/'gpu_samples.jsonl').read_bytes()).hexdigest(),
        'note':'Jobs explicitly use CUDA_VISIBLE_DEVICES=2 or 3. Ten-second snapshots cannot rule out brief unobserved cooccupancy.'}
    if a.baseline_reference:
        checks={}
        for prop in ['homo','lumo']:
            reference=json.loads((a.baseline_reference/prop/'summary.json').read_text())['runs']
            comparisons=[]
            for row in report['raw_predictions'][prop]['baseline']:
                run=next(r for r in reference if r['seed']==row['seed'] and r['quantile']==row['quantile'])
                old=run['raw_records'][row['candidate_id']]
                assert old['token_sha256']==row['token_sha256'] and old['valid']==row['valid']
                if 'prediction' in row:
                    comparisons.append(float(np.max(np.abs(np.array(row['prediction'])-old['prediction']))))
            checks[prop]={'identical_token_and_validity_outcomes':288,'maximum_DPA_prediction_difference_eV':max(comparisons)}
        report['baseline_prefix_reproduction']=checks
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    plot(report,a.plot)

def plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(14,8.6))
    for row,(prop,index) in enumerate([('homo',0),('lumo',1)]):
        records=[r for arm in ARMS for r in report['raw_predictions'][prop][arm] if 'prediction' in r and r.get('domain',{}).get('in_domain')]
        values=[r['prediction'][index] for r in records]+[r['independent_prediction'][index] for r in records]+[r['target'] for r in records]
        lo=np.floor(min(values)*2)/2-.5;hi=np.ceil(max(values)*2)/2+.5
        for col,arm in enumerate(ARMS):
            ax=axes[row,col];rows=[r for r in report['raw_predictions'][prop][arm] if 'prediction' in r and r.get('domain',{}).get('in_domain')]
            ax.scatter([r['target'] for r in rows],[r['prediction'][index] for r in rows],s=13,alpha=.45,color='#c45132',label='DPA')
            ax.scatter([r['target'] for r in rows],[r['independent_prediction'][index] for r in rows],s=13,alpha=.45,color='#2c6caa',marker='x',label='Independent FP')
            ax.plot([lo,hi],[lo,hi],'--',color='#9ca3af');ax.set(xlim=(lo,hi),ylim=(lo,hi))
            ticks=np.arange(np.ceil(lo),hi+.01,1.);ax.set_xticks(ticks);ax.set_yticks(ticks)
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
            m=report['properties'][prop][arm]['aggregate'];mae=m['metrics']['dpa']['equal_target_mae']
            ax.set_title(f'{prop.upper()} | {arm}\nDPA MAE {mae:.3f}; AD {m["domain"]}/288',fontsize=10)
            ax.set_xlabel('Requested (eV)');ax.set_ylabel('Prediction (eV)')
    span=max(ax.get_xlim()[1]-ax.get_xlim()[0] for ax in axes.flat)
    for ax in axes.flat:
        lo=ax.get_xlim()[0];ax.set_xlim(lo,lo+span);ax.set_ylim(lo,lo+span)
        ax.lines[0].set_data([lo,lo+span],[lo,lo+span])
        ticks=np.arange(np.ceil(lo),lo+span+.01,1.);ax.set_xticks(ticks);ax.set_yticks(ticks)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,frameon=False)
    fig.suptitle('Individual predictions: CFG and temperature controls (frozen SFT)')
    fig.text(.5,.025,'Three targets × three inference seeds × 32 attempts/arm/property. In-domain points shown; no coordinate optimization.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.91]);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=180);plt.close(fig)

if __name__=='__main__':main()
