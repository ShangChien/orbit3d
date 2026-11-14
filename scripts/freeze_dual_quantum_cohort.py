"""A-only, pre-quantum selection of 24 raw geometries from matched existing SFT/GRPO pools."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rdkit import Chem
from loom.generator.octree_backend import trajectory_to_instance
from loom.utils.dataset_io import write_precise_sdf
from loom.oracles.uniqsar_oracle import UNIMOL_SYM2IDX

p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
d=json.loads(a.snapshot.read_text());pools=[x for x in d['pools'] if x['group'].startswith('main:')]
lookup={}
for pool in pools:
 if '_p25_' in pool['id'] or '_p50_' in pool['id'] or '_p75_' in pool['id']:
  lookup[(pool['property'],pool['arm'],pool['target'],pool['seed'])]=pool
expected={prop:sorted({(x['target'],x['seed']) for x in pools if x['property']==prop and x['arm']=='frozen' and any(f'_p{q}_' in x['id'] for q in [25,50,75])}) for prop in ['homo','lumo']}
assert all(len(v)==9 for v in expected.values())

def eligible(pool):
 symbols=json.loads(Path(pool['paths']['symbols']).read_text())
 return [r for r in pool['records'] if r.get('valid') and 'prediction' in r and r.get('domain',{}).get('in_domain') and len(symbols[r['candidate_id']])<=80 and not(set(symbols[r['candidate_id']])-set(UNIMOL_SYM2IDX))]

scores={}
for arm in ['grpo50','grpo100','grpo200']:
 if any((prop,arm,t,s) not in lookup for prop,keys in expected.items() for t,s in keys):continue
 cells=[]
 for prop,keys in expected.items():
  idx=['homo','lumo'].index(prop)
  for t,s in keys:
   base=lookup[(prop,'frozen',t,s)];pool=lookup[(prop,arm,t,s)]
   assert pool['attempted']==base['attempted']==64 and pool['sampling']==base['sampling']
   rows=eligible(pool);assert rows
   cells.append(float(np.mean([abs(r['prediction'][idx]-t) for r in rows])))
 scores[arm]=float(np.mean(cells))
assert scores;chosen=min(scores,key=scores.get);cohort=[];mols=[];missing=[]
for prop,keys in expected.items():
 idx=['homo','lumo'].index(prop)
 for target in sorted({t for t,s in keys}):
  for arm in ['frozen',chosen]:
   candidates=[]
   for t,seed in keys:
    if t!=target:continue
    pool=lookup[(prop,arm,t,seed)]
    for row in eligible(pool):candidates.append((pool,row))
   candidates.sort(key=lambda pr:(abs(pr[1]['prediction'][idx]-target),pr[0]['id'],pr[1]['candidate_id']))
   unique=[];seen=set()
   for item in candidates:
    if item[1]['smiles'] in seen:continue
    unique.append(item);seen.add(item[1]['smiles'])
   selected=[]
   if unique:selected.append(('A_top1',unique[0]))
   if len(unique)>1:
    draw=min(unique[1:],key=lambda pr:hashlib.sha256(('quantum24-v1:'+pr[0]['id']+':'+str(pr[1]['candidate_id'])).encode()).hexdigest());selected.append(('fixed_random',draw))
   if len(selected)<2:missing.append({'property':prop,'target':target,'arm':arm,'missing':2-len(selected)})
   for rule,(pool,row) in selected:
    syms=json.loads(Path(pool['paths']['symbols']).read_text())[row['candidate_id']]
    with np.load(pool['paths']['coordinates']) as z:coords=z[f'coordinates_{row["candidate_id"]}']
    elements=tuple(sorted(set(syms)));inst=trajectory_to_instance({'atom_symbols':syms,'positions_np':coords},element_table=elements)
    assert inst.smiles==row['smiles'];mol=inst.graph;cid=f'q{len(cohort):02d}';mol.SetProp('CohortID',cid)
    entry={'id':cid,'property':prop,'target':target,'method':arm,'rule':rule,'pool':pool['id'],'candidate_id':row['candidate_id'],'smiles':row['smiles'],'atoms':len(syms),'elements':sorted(set(syms)),'A_prediction':row['prediction'],'initial_rms_force':row.get('rms_force'),'input_coordinate_file_sha256':pool['sha256']['coordinates']}
    cohort.append(entry);mols.append(mol)
write_precise_sdf(a.output/'cohort.sdf',mols)
report={'selection_rule':'Choose completed GRPO stage with lowest macro A-only raw MAE on 18 matched development cells. Freeze A-top1 and a fixed hash-uniform draw from remaining unique eligible graphs per task/target/method before model B or quantum evaluation.',
 'eligibility':'Original chemical validity + original AD + common vocabulary + <=80 all-source atoms; no target-hit or low-force filter.',
 'stage_scores':scores,'chosen_improvement_candidate':chosen,'expected':24,'selected':len(cohort),'missing':missing,'records':cohort,
 'snapshot_sha256':hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),'sdf_sha256':hashlib.sha256((a.output/'cohort.sdf').read_bytes()).hexdigest()}
(a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('stages',scores,'selected',chosen,'cohort',len(cohort),'atoms',[r['atoms'] for r in cohort],flush=True)
