"""Equal-target/equal-seed comparisons of frozen A-optimized and B-evaluated pools."""
import argparse,gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

MODELS=['A_historical','A_common','B_unimol']

def macro(pools,view='raw'):
 targets=sorted({p['target'] for p in pools});out={'pools':len(pools),'targets':targets,'attempted':sum(p['attempted'] for p in pools),'models':{}}
 for model in MODELS:
  values={}
  for metric in ['mae','std','bias','p90_absolute_error']:
   bytarget=[]
   for t in targets:
    xs=[]
    for p in pools:
     if p['target']!=t:continue
     data=p[view]['models'][model] if view in ['raw','retained_geometry'] else p['selections'][view]['metrics'][model]
     if data[metric] is not None:xs.append(abs(data[metric]) if metric=='bias' else data[metric])
    if xs:bytarget.append(float(np.mean(xs)))
   values['mean_abs_bias' if metric=='bias' else metric]=float(np.mean(bytarget)) if bytarget else None
  values['hits025']=float(sum((p[view]['models'][model] if view in ['raw','retained_geometry'] else p['selections'][view]['metrics'][model])['hits025'] or 0 for p in pools))
  out['models'][model]=values
 out['valid']=sum(p['diversity']['valid'] for p in pools);out['raw_matched_domain']=sum(p['raw']['matched_domain_evaluated'] for p in pools)
 out['mean_unique_fraction']=float(np.mean([p['diversity']['unique_fraction_valid'] for p in pools if p['diversity']['unique_fraction_valid'] is not None]))
 out['mean_scaffolds_per_pool']=float(np.mean([p['diversity']['scaffolds'] for p in pools]));out['mean_morgan_distance']=float(np.mean([p['diversity']['morgan_distance'] for p in pools if p['diversity']['morgan_distance'] is not None]))
 out['mean_raw_A_B_disagreement']=float(np.mean([p['raw']['A_B_mean_absolute_disagreement'] for p in pools if p['raw'].get('A_B_mean_absolute_disagreement') is not None]))
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=json.loads((a.input/'reevaluation/summary.json').read_text());snapshot=json.loads((a.input/'candidate_snapshot/manifest.json').read_text())
 pools=r['pools'];report={'scope':'A historical optimization scores unchanged; comparison restricted to mutually evaluated original-AD candidates, equal weight to targets and matched seeds. All-attempt denominators retained. Main headline uses P25/P50/P75 so GRPO stages share support.',
 'source_pool_count':len(pools),'source_attempts_in_distinct_protocol_pools':r['source_attempts'],'unique_evaluated_raw_geometries':r['unique_evaluated_raw_geometries'],
 'generator_lineage':json.loads((a.input/'generator_lineage.json').read_text()),
 'snapshot_sha256':hashlib.sha256((a.input/'candidate_snapshot/manifest.json').read_bytes()).hexdigest(),'model_lock':r['model_lock'],'B_status':dict(sum((Counter(p['B_status']) for p in pools),Counter())),
 'comparisons':{},'geometry_diagnostic':r['geometry_diagnostic'],
 'raw_prediction_sha256':hashlib.sha256((a.input/'reevaluation/candidate_predictions.jsonl').read_bytes()).hexdigest()}
 for group in sorted({p['group'] for p in pools}):
  ps=[p for p in pools if p['group']==group and p['arm']!='off']
  if group.startswith('main:'):ps=[p for p in ps if any(f'_p{q}_' in p['id'] for q in [25,50,75])]
  baseline='frozen' if group.startswith('main:') else 'baseline' if group.startswith('concentration:') else 'original'
  key=lambda p:(p['target'],p['seed'],p['attempted'])
  base={key(p):p for p in ps if p['arm']==baseline};groupout={}
  for arm in sorted({p['arm'] for p in ps if p['arm']!=baseline}):
   treatment={key(p):p for p in ps if p['arm']==arm};common=sorted(base.keys()&treatment.keys())
   if group.startswith('main:'):common=[k for k in common if base[k]['sampling']==treatment[k]['sampling']]
   if not common:continue
   left=[base[k] for k in common];right=[treatment[k] for k in common];entry={'matched_cells':len(common),'baseline':macro(left),'treatment':macro(right),'selections':{}}
   methods=set.intersection(*(set(p['selections']) for p in left+right))
   for method in sorted(methods):
    if method=='original_selected' and not any(p['selections'][method]['retained'] for p in left+right):continue
    entry['selections'][method]={'baseline':macro(left,method),'treatment':macro(right,method)}
   entry['retained_geometry']={'baseline':macro(left,'retained_geometry'),'treatment':macro(right,'retained_geometry')}
   groupout[arm]=entry
   print(group,arm,'cells',len(common),'A/B',[[round(e['models'][m]['mae'],3) for m in ['A_historical','B_unimol']] for e in [entry['baseline'],entry['treatment']]])
  report['comparisons'][group]=groupout
 # Quantify disagreement for exactly the pre-frozen selection IDs, never selecting with B.
 bypool=defaultdict(dict)
 for line in (a.input/'reevaluation/candidate_predictions.jsonl').read_text().splitlines():
  row=json.loads(line);bypool[row['pool']][row['candidate_id']]=row
 disagreements={}
 for pool in snapshot['pools']:
  rows=bypool[pool['id']];value={}
  for name,ids in pool['selections'].items():
   reps=ids if ids and isinstance(ids[0],list) else [ids];scores=[];idx=['homo','lumo'].index(pool['property'])
   for rep in reps:
    e=[abs(rows[i]['prediction'][idx]-rows[i]['unimol_prediction'][idx]) for i in rep if 'prediction' in rows[i] and 'unimol_prediction' in rows[i]]
    if e:scores.append(float(np.mean(e)))
   value[name]=float(np.mean(scores)) if scores else None
  disagreements[pool['id']]=value
 report['frozen_selection_disagreement']=disagreements
 pool_file=a.output.with_suffix('.pools.json.gz')
 with gzip.open(pool_file,'wt') as f:json.dump(pools,f,separators=(',',':'))
 report['per_pool_evidence']={'file':pool_file.name,'sha256':hashlib.sha256(pool_file.read_bytes()).hexdigest(),'includes':'Per-target/seed metrics, retained geometry predictions and selection statistics'}
 a.output.write_text(json.dumps(report,indent=2)+'\n')
 with gzip.open(a.output.with_suffix('.jsonl.gz'),'wt') as f:f.write((a.input/'reevaluation/candidate_predictions.jsonl').read_text())
 print('status',report['B_status'])

if __name__=='__main__':main()
