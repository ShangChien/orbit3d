"""Freeze completed candidate pools and A-only selections before reading model B outputs."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def top_ids(rows,index,target,k,first_unique=False):
    valid=[r for r in rows if 'prediction' in r and r.get('domain',{}).get('in_domain')]
    if first_unique:
        seen=set();valid=[r for r in valid if not (r['smiles'] in seen or seen.add(r['smiles']))]
    ranked=sorted(valid,key=lambda r:(abs(r['prediction'][index]-target),r['candidate_id']))
    selected=[];seen=set()
    for r in ranked:
        if r['smiles'] in seen:continue
        seen.add(r['smiles']);selected.append(r['candidate_id'])
        if len(selected)==k:break
    return selected


def main():
 p=argparse.ArgumentParser();p.add_argument('--scan-root',type=Path,required=True);p.add_argument('--main-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);pools=[];omitted=[]
 for campaign in ['campaign','concentration','response_calibration']:
  for f in sorted((a.scan_root/campaign).glob('*/seed*/summary.json')):
   d=json.loads(f.read_text());lane=f.parent.parent.name;prop=d['target_name'];idx=['homo','lumo'].index(prop);t=d['scoring_value'];rows=d['raw_records'];q=d['quantile']
   group=campaign+':'+prop;arm=(lane.split(prop+'_',1)[1] if campaign=='concentration' else d.get('arm','original'))
   if d['condition']=='off':arm='off'
   selected={'original_selected':d['selected_ids'],'distance_top4':top_ids(rows,idx,t,4)}
   if campaign=='response_calibration':
    from summarize_response_calibration import choose
    selected['distance_top4']=[r['candidate_id'] for r in choose(rows,idx,t,'top4',0)]
    for method in ['gaussian025','gaussian050','uniform']:
     selected[method]=[[r['candidate_id'] for r in choose(rows,idx,t,method,900000+1000*d['seed']+rep)] for rep in range(100)]
   paths={'summary':f,'coordinates':f.parent/'trajectories.npz'}
   pools.append({'id':f'{campaign}/{lane}/{f.parent.name}','format':'scan','group':group,'arm':arm,'target':t,'property':prop,'seed':d['seed'],'attempted':d['attempted'],'paths':{k:str(v) for k,v in paths.items()},'sha256':{k:digest(v) for k,v in paths.items()},'records':rows,'selections':selected})
 for prop in ['homo','lumo']:
  lane=a.main_root/f'fast_{prop}_seed42'
  for folder in sorted(lane.glob('eval_*')):
   if not folder.is_dir():continue
   raw=folder/'raw/summary.json';run=folder/'run'
   if not raw.exists() or not (run/'summary.json').exists():omitted.append(str(folder));continue
   d=json.loads(raw.read_text());cfg=json.loads((run/'run_config.json').read_text());rc=json.loads((run/'summary.json').read_text());rows=d['records']
   if len(rows)!=d['attempted'] or d['attempted']!=cfg['inference_config']['num_candidates']:raise ValueError(str(folder))
   inf=cfg['inference_config'];t=inf['target_point'][prop];idx=['homo','lumo'].index(prop)
   arm=re.sub(r'_p\d+_s\d+$','',folder.name.removeprefix('eval_'))
   if inf.get('condition_control','correct')!='correct':arm='off'
   paths={'summary':raw,'coordinates':run/'raw_trajectories.npz','symbols':run/'raw_symbols.json','config':run/'run_config.json','retained':run/'generated.sdf','candidates':run/'candidates.csv','run_summary':run/'summary.json'}
   # Only completed, immutable pool files are selected; lane-level training continues elsewhere.
   pools.append({'id':f'main/{prop}/{folder.name}','format':'main','group':'main:'+prop,'arm':arm,'target':t,'property':prop,'seed':inf['base_seed'],'attempted':d['attempted'],'paths':{k:str(v) for k,v in paths.items()},'sha256':{k:digest(v) for k,v in paths.items()},'records':rows,'selections':{'distance_top4':top_ids(rows,idx,t,4)},'checkpoint_sha256':rc['checkpoint_sha256'],'sampling':{k:inf[k] for k in ['tree_temperature','atom_temperature','xyz_temperature','cfg_scale','batch_size','precision']}})
 for pool in pools:
  for row in pool['records']:
   row.pop('symbols',None) if pool['format']=='main' else None
 report={'pools':pools,'omitted_incomplete':omitted,'scope':'Snapshot of completed pools only; no model B predictions used. Separate original budgets/groups/targets. A predictions and previously frozen selections retained.',
         'attempts_including_distinct_protocol_pools':sum(p['attempted'] for p in pools),'pool_count':len(pools),'coordinate_hashes_unique':len({p['sha256']['coordinates'] for p in pools})}
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='pools'},flush=True)

if __name__=='__main__':main()
