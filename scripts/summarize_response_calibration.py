"""New-target validation and same-pool Gaussian selection with fixed output budget."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from summarize_concentration_scan import error_metrics, diversity

ARMS=['original','linear','gp']
METHODS=['top4','gaussian025','gaussian050','uniform']
FIELDS={'dpa':'prediction','fingerprint':'independent_prediction'}

def unique_eligible(rows):
    result=[];seen=set()
    for row in rows:
        if 'prediction' not in row or 'independent_prediction' not in row or not row.get('domain',{}).get('in_domain'):continue
        if row['smiles'] in seen:continue
        result.append(row);seen.add(row['smiles'])
    return result

def choose(rows,index,target,method,seed,k=4):
    rows=unique_eligible(rows)
    if not rows:return []
    error=np.array([r['prediction'][index]-target for r in rows])
    if method=='top4':order=np.argsort(abs(error),kind='stable')
    else:
        sigma={'gaussian025':.25,'gaussian050':.5}.get(method)
        if method!='uniform' and sigma is None:raise ValueError(method)
        logw=np.zeros(len(rows)) if sigma is None else -.5*np.square(error/sigma)
        # Gumbel top-k is weighted sampling without replacement; no probability underflow fallback.
        score=logw+np.random.default_rng(seed).gumbel(size=len(rows))
        order=np.argsort(-score,kind='stable')
    return [rows[i] for i in order[:k]]

def stats(rows,index,target,denominator):
    return {'attempted':denominator,'valid':sum(r.get('valid',False) for r in rows),
        'domain':sum(r.get('domain',{}).get('in_domain',False) for r in rows),
        'retained':len(rows),'unique':len({r['smiles'] for r in rows if r.get('valid')}),
        'novel':sum(r.get('novel',False) for r in rows),
        'metrics':{m:error_metrics(rows,index,target,f) for m,f in FIELDS.items()}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path,required=True);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());result={'protocol':protocol,'properties':{},'raw_runs':{},
        'scope':'Surrogate evaluation of frozen SFT with new target values and new inference seeds. No geometry optimization, no GRPO. Conditional-error metrics always accompanied by coverage.',
        'selection_scope':'First occurrence per graph retained before ALL selection methods. At most 4 per 64 attempts; no replenishment. 100 selection repetitions do not count as new generated molecules.',
        'remote_artifacts':'/sjtu-combustion/wjw/chensq/orbit_generation_scan_5fe5/response_calibration'}
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);fps={}
    def pair_div(rows):
        smi=sorted({r['smiles'] for r in rows})
        for s in smi:
            if s not in fps:fps[s]=fpgen.GetFingerprint(Chem.MolFromSmiles(s))
        ds=[1-DataStructs.TanimotoSimilarity(fps[s],fps[t]) for i,s in enumerate(smi) for t in smi[:i]]
        return float(np.mean(ds)) if ds else None
    for prop,index in [('homo',0),('lumo',1)]:
        root=a.input/prop;data=json.loads((root/'summary.json').read_text());runs=data['runs']
        assert len(runs)==27 and data['protocol']['visible_devices']==str(2 if prop=='homo' else 3)
        assert data['protocol']['trained_steps']==5000 and data['protocol']['seeds']==[2031,2032,2033]
        assert all(r['attempted']==64 and not r['refined_records'] for r in runs)
        result['raw_runs'][prop]=[{**r,'raw_records':[{k:v for k,v in row.items() if k!='symbols'} for row in r['raw_records']]} for r in runs]
        out={'arms':{},'input_manifest':json.loads((root/'input_manifest.json').read_text()),
            'source_manifest':json.loads((root/'run_manifest.json').read_text()),'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()}
        result['properties'][prop]=out
        plan={r['key']:r for r in protocol['properties'][prop]['experiments']}
        for arm in ARMS:
            cells={}
            for ti in range(3):
                key=f't{ti}_{arm}';cellruns=[r for r in runs if r['quantile']==key];assert len(cellruns)==3
                t=plan[key]['target_value'];u=plan[key]['input_value']
                assert all(r['scoring_value']==t and r['input_value']==u and r['arm']==arm for r in cellruns)
                rows=[v for r in cellruns for v in r['raw_records']]
                cell=stats(rows,index,t,192);cell.update(target=t,input=u,at_boundary=plan[key]['at_boundary'],diversity=diversity(rows),domain_diversity=pair_div(unique_eligible(rows)),
                    mean_atoms=float(np.mean([r['atoms'] for r in rows if r['valid']])),
                    all_evaluable={m:error_metrics(rows,index,t,f,False) for m,f in FIELDS.items()},
                    generation_seconds=sum(r['stage_seconds']['generation'] for r in cellruns),
                    seeds={str(r['seed']):stats(r['raw_records'],index,t,64) for r in cellruns},selection={})
                for method in METHODS:
                    reps=[]
                    for rep in range(1 if method=='top4' else 100):
                        chosen=[v for run in cellruns for v in choose(run['raw_records'],index,t,method,900000+1000*run['seed']+rep)]
                        s=stats(chosen,index,t,192);s['diversity']=pair_div(chosen);reps.append(s)
                    cell['selection'][method]={'available_slots':12,'repetitions':len(reps),'retained':reps[0]['retained'],
                        'mean_diversity':float(np.mean([r['diversity'] for r in reps])),
                        'mean_unique':float(np.mean([r['unique'] for r in reps])),
                        'metrics':{m:{k:float(np.mean([r['metrics'][m][k] for r in reps])) for k in ['mae','rmse','bias','std','hits025','hits010']} for m in FIELDS},
                        'selection_only_mae_p10_p90':{m:np.quantile([r['metrics'][m]['mae'] for r in reps],[.1,.9]).tolist() for m in FIELDS}}
                cells[str(ti)]=cell
            out['arms'][arm]={'targets':cells,'aggregate':{'attempted':576,'valid':sum(c['valid'] for c in cells.values()),'domain':sum(c['domain'] for c in cells.values()),
                'generation_seconds':sum(c['generation_seconds'] for c in cells.values()),
                'metrics':{m:{k:float(np.mean([c['metrics'][m][k] for c in cells.values()])) for k in ['mae','rmse','bias','std']} for m in FIELDS}}}
        for arm in ARMS:
            cells=list(out['arms'][arm]['targets'].values())
            for m in FIELDS:
                agg=out['arms'][arm]['aggregate']['metrics'][m]
                agg['mean_abs_bias']=float(np.mean([abs(c['metrics'][m]['bias']) for c in cells]))
                agg['mean_squared_bias']=float(np.mean([c['metrics'][m]['bias']**2 for c in cells]))
                agg['mean_variance']=float(np.mean([c['metrics'][m]['std']**2 for c in cells]))
                agg['rmse']=float(np.sqrt(agg['mean_squared_bias']+agg['mean_variance']))
        for arm in ['linear','gp']:
            out['arms'][arm]['seed_mae_delta']={m:{str(s):float(np.mean([out['arms'][arm]['targets'][str(ti)]['seeds'][str(s)]['metrics'][m]['mae']-out['arms']['original']['targets'][str(ti)]['seeds'][str(s)]['metrics'][m]['mae'] for ti in range(3)])) for s in [2031,2032,2033]} for m in FIELDS}
        for arm,d in out['arms'].items():print(prop,arm,d['aggregate'])
    result['completed']=json.loads((a.input/'completed.json').read_text());assert all(j['gpu'] in [2,3] and j['exit_code']==0 for j in result['completed']['jobs'])
    samples=[json.loads(s) for s in (a.input/'gpu_samples.jsonl').read_text().splitlines()];co=[]
    for s in samples:
        reserved={line.split(',')[1].strip() for line in s['gpu_csv'].splitlines() if line.split(',')[0].strip() in ['2','3']}
        for uuid in reserved:
            ps=[line for line in s['process_csv'].splitlines() if line.split(',')[0].strip()==uuid]
            if len(ps)>1:co.append({'time':s['unix'],'gpu':uuid,'processes':ps})
    result['gpu_monitor']={'samples':len(samples),'cooccupancy':co,'sha256':hashlib.sha256((a.input/'gpu_samples.jsonl').read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(result,indent=2)+'\n');plot(result,a.plot)

def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,8.5));bounds={}
    for ri,(prop,index) in enumerate([('homo',0),('lumo',1)]):
        vals=[]
        for arm in ARMS:
            runs=[r for r in result['raw_runs'][prop] if r['arm']==arm]
            for run in runs:
                vals.append(run['scoring_value'])
                for r in run['raw_records']:
                    if 'prediction' in r and r.get('domain',{}).get('in_domain'):vals.extend([r['prediction'][index],r['independent_prediction'][index]])
        bounds[prop]=(np.floor(min(vals)*2)/2-.5,np.ceil(max(vals)*2)/2+.5)
    span=max(hi-lo for lo,hi in bounds.values())
    for ri,(prop,index) in enumerate([('homo',0),('lumo',1)]):
        lo=bounds[prop][0]
        for ci,arm in enumerate(ARMS):
            ax=axes[ri,ci]
            for field,color,marker,label in [('prediction','#c45132','o','DPA'),('independent_prediction','#2c6caa','x','Independent FP')]:
                xy=[(run['scoring_value'],r[field][index]) for run in result['raw_runs'][prop] if run['arm']==arm for r in run['raw_records'] if field in r and r.get('domain',{}).get('in_domain')]
                ax.scatter(*np.array(xy).T,s=12,alpha=.4,color=color,marker=marker,label=label)
            ax.plot([lo,lo+span],[lo,lo+span],'--',color='gray');ax.set_xlim(lo,lo+span);ax.set_ylim(lo,lo+span)
            ticks=np.arange(np.ceil(lo),lo+span+.01,1);ax.set_xticks(ticks);ax.set_yticks(ticks);ax.set_aspect('equal',adjustable='box')
            m=result['properties'][prop]['arms'][arm]['aggregate'];ax.set_title(f'{prop.upper()} | {arm}\nDPA MAE {m["metrics"]["dpa"]["mae"]:.3f}; AD {m["domain"]}/576',fontsize=10)
            ax.set_xlabel('Desired target (eV)');ax.set_ylabel('Prediction (eV)');ax.grid(alpha=.15)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False)
    fig.suptitle('Held-out targets and seeds: original vs linear vs GP input correction')
    fig.text(.5,.02,'Raw in-domain proposals; 192 attempts/target/arm; frozen SFT; no geometry optimization.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.92]);fig.savefig(path,dpi=180);plt.close(fig)

if __name__=='__main__':main()
