"""Cross-evaluate frozen pools/selections without any model-B re-ranking."""
import argparse,hashlib,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np,torch
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from profile_dual_oracle import load_models,predict
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.oracles.uniqsar_oracle import UNIMOL_SYM2IDX
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes

FIELDS={'A_historical':'prediction','A_common':'dpa_common_prediction','B_unimol':'unimol_prediction'}

def target_stats(rows,index,t,attempted,matched=True):
 eligible=[r for r in rows if r.get('domain',{}).get('in_domain') and (not matched or all(f in r for f in FIELDS.values()))]
 out={'attempted':attempted,'rows':len(rows),'matched_domain_evaluated':len(eligible),'models':{}}
 for name,field in FIELDS.items():
  rr=[r for r in eligible if field in r];e=np.array([r[field][index]-t for r in rr])
  out['models'][name]={'evaluated':len(e),'mae':float(abs(e).mean()) if len(e) else None,'bias':float(e.mean()) if len(e) else None,'std':float(e.std()) if len(e) else None,'p90_absolute_error':float(np.quantile(abs(e),.9)) if len(e) else None,'rmse':float(np.sqrt(np.square(e).mean())) if len(e) else None,'hits025':int((abs(e)<=.25).sum()),'hits010':int((abs(e)<=.1).sum())}
 if eligible:
  gap=np.array([r['unimol_prediction'][index]-r['prediction'][index] for r in eligible if 'unimol_prediction' in r and 'prediction' in r])
  out['A_B_mean_absolute_disagreement']=float(abs(gap).mean()) if len(gap) else None
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',default='profile');a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2','3'];torch.set_num_threads(2);freeze_run(a.output,__file__)
 snapshot=a.root/'candidate_snapshot/manifest.json';data=json.loads(snapshot.read_text());frozen=json.loads((a.root/a.profile/'models_frozen.json').read_text());choice=frozen['selected_unimol']
 ff,models=load_models(a.root,choice)
 record_input_hashes(a.output,{'candidate_manifest':snapshot,'model_lock':a.root/a.profile/'models_frozen.json','unimol':a.root/'training'/choice/'oracle','dpa_common':a.root/'training/dpa_common/oracle'})
 # Freeze auxiliary retained geometry hashes before any candidate B prediction.
 auxiliary={}
 for pool in data['pools']:
  if pool['format']=='scan':
   s=json.loads(Path(pool['paths']['summary']).read_text())
   for row in s['refined_records']:
    if row['status']=='completed':
     path=Path(pool['paths']['summary']).parent/f'refined_{row["candidate_id"]}.sdf';auxiliary[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 (a.output/'retained_geometry_hashes.json').write_text(json.dumps(auxiliary,indent=2))
 report={'pool_count':len(data['pools']),'source_attempts':data['attempts_including_distinct_protocol_pools'],'scope':'Frozen A-only selections; B never selects candidates. Historical A scores remain unchanged; common-support A is an additional matched-training baseline. Pool budgets are not combined across experiments.', 'model_lock':frozen,'pools':[]};cache={};fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);fps={};scaffolds={}
 def diversity(rows):
  valid=[r for r in rows if r.get('valid')];smis=sorted({r['smiles'] for r in valid})
  for s in smis:
   if s not in fps:
    m=Chem.MolFromSmiles(s);fps[s]=fpgen.GetFingerprint(m);scaffolds[s]=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) or '<ACYCLIC>'
  ds=[1-v for i,s in enumerate(smis) for v in DataStructs.BulkTanimotoSimilarity(fps[s],[fps[t] for t in smis[:i]])]
  return {'valid':len(valid),'unique':len(smis),'unique_fraction_valid':len(smis)/len(valid) if valid else None,'scaffolds':len({scaffolds[s] for s in smis}),'morgan_distance':float(np.mean(ds)) if ds else None,'novel_vs_sft':sum(r.get('novel',r.get('novel_vs_sft_training',False)) for r in valid),'atom_quantiles':np.quantile([r['atoms'] for r in valid],[0,.25,.5,.75,1]).tolist() if valid else None}
 def evaluate(ms,rows):
  if not ms:return
  pa,_=predict(models['dpa_common'],ms);pb,_=predict(models['unimol'],ms)
  assert np.isfinite(pa).all() and np.isfinite(pb).all()
  for row,x,y in zip(rows,pa,pb):row.update(dpa_common_prediction=x.tolist(),unimol_prediction=y.tolist(),B_status='evaluated')
 for number,pool in enumerate(data['pools']):
  start=time.perf_counter()
  for k,path in pool['paths'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pool['sha256'][k],pool['id']+'/'+k
  idx=['homo','lumo'].index(pool['property']);rows=[dict(r) for r in pool['records']];symbols=json.loads(Path(pool['paths']['symbols']).read_text()) if pool['format']=='main' else [r['symbols'] for r in rows]
  ms=[];todo=[];keys={}
  with np.load(pool['paths']['coordinates']) as z:
   for row in rows:
    cid=row['candidate_id'];row['B_status']='invalid_original'
    if not row.get('valid'):continue
    syms=symbols[cid]
    if len(syms)>80 or set(syms)-set(UNIMOL_SYM2IDX):row['B_status']='outside_common_support';continue
    coords=z[f'coordinates_{cid}' if pool['format']=='main' else f'coords_{cid}'];key=hashlib.sha256(json.dumps(syms).encode()+np.asarray(coords,dtype=np.float64).tobytes()).hexdigest();keys[cid]=key
    if key in cache:row.update(cache[key]);continue
    try:
     inst=trajectory_to_instance({'atom_symbols':syms,'positions_np':coords},element_table=tuple(sorted(set(syms))))
     if inst.smiles!=row['smiles']:raise ValueError('Stored graph and coordinate reconstruction differ')
    except (ValueError,RuntimeError) as error:row.update(B_status='geometry_reconstruction_failed',B_error=str(error));continue
    ms.append(inst.graph);todo.append(row)
  evaluate(ms,todo)
  for row in todo:cache[keys[row['candidate_id']]]={k:row[k] for k in ['dpa_common_prediction','unimol_prediction','B_status']}
  byid={r['candidate_id']:r for r in rows};selections={}
  for name,selected in pool['selections'].items():
   repetitions=selected if selected and isinstance(selected[0],list) else [selected]
   stats=[target_stats([byid[i] for i in ids],idx,pool['target'],pool['attempted']) for ids in repetitions]
   selections[name]={'frozen_repetitions':len(repetitions),'retained':len(repetitions[0]),'mean_B_evaluated':float(np.mean([s['models']['B_unimol']['evaluated'] for s in stats])),
    'metrics':{model:{key:float(np.mean([s['models'][model][key] for s in stats if s['models'][model][key] is not None])) if any(s['models'][model][key] is not None for s in stats) else None for key in ['mae','bias','std','p90_absolute_error','hits025']} for model in FIELDS},
    'diversity_first_repetition':diversity([byid[i] for i in repetitions[0]])}
  retained=[];retmols=[];retrows=[]
  if pool['format']=='main':
   for m in (Chem.SDMolSupplier(pool['paths']['retained'],removeHs=False) if Path(pool['paths']['retained']).stat().st_size else []):
    if m is None:raise ValueError('Invalid frozen retained SDF')
    cid=int(m.GetProp('CandidateID'));r=dict(byid[cid]);r['input_rms_force']=r.pop('rms_force',None);r['prediction']=json.loads(m.GetProp('PredictedValue'));r.pop('unimol_prediction',None);r.pop('dpa_common_prediction',None);r['B_status']='outside_common_support'
    retained.append(r)
    if m.GetNumAtoms()<=80 and not({x.GetSymbol() for x in m.GetAtoms()}-set(UNIMOL_SYM2IDX)):retmols.append(m);retrows.append(r)
  else:
   original=json.loads(Path(pool['paths']['summary']).read_text())
   for rr in original['refined_records']:
    if rr['status']!='completed':continue
    path=Path(pool['paths']['summary']).parent/f'refined_{rr["candidate_id"]}.sdf';m=next(iter(Chem.SDMolSupplier(str(path),removeHs=False)));r=dict(byid[rr['candidate_id']]);r['input_rms_force']=r.pop('rms_force',None);r['rms_force']=rr['rms_force'];r['prediction']=rr['prediction'];r.pop('unimol_prediction',None);r.pop('dpa_common_prediction',None);r['B_status']='outside_common_support';retained.append(r)
    if m.GetNumAtoms()<=80 and not({x.GetSymbol() for x in m.GetAtoms()}-set(UNIMOL_SYM2IDX)):retmols.append(m);retrows.append(r)
  evaluate(retmols,retrows)
  entry={k:pool[k] for k in ['id','group','arm','target','property','seed','attempted']};entry.update(raw=target_stats(rows,idx,pool['target'],pool['attempted']),A_original_domain=target_stats(rows,idx,pool['target'],pool['attempted'],matched=False),diversity=diversity(rows),B_status=dict(Counter(r['B_status'] for r in rows)),selections=selections,retained_geometry=target_stats(retained,idx,pool['target'],pool['attempted']),retained_records=retained,seconds=time.perf_counter()-start)
  if 'sampling' in pool:entry['sampling']=pool['sampling']
  # Each raw record remains available for matched re-analysis; no replaced A values.
  with (a.output/'candidate_predictions.jsonl').open('a') as stream:
   for row in rows:stream.write(json.dumps({'pool':pool['id'],**{k:v for k,v in row.items() if k!='symbols'}})+'\n')
  report['pools'].append(entry);(a.output/'summary.json').write_text(json.dumps(report,indent=2));print(number+1,len(data['pools']),pool['id'],entry['B_status'],flush=True)
 # Frozen source geometry stability cohort: prediction changes, never accuracy versus old labels at changed geometry.
 originals=list(Chem.SDMolSupplier(str(a.root/'geometry_diagnostic/original.sdf'),removeHs=False));relaxed=list(Chem.SDMolSupplier(str(a.root/'geometry_diagnostic/relaxed.sdf'),removeHs=False));lookup={m.GetProp('SourceID'):m for m in originals};geometry={}
 for name,model in models.items():
  before,_=predict(model,[lookup[m.GetProp('SourceID')] for m in relaxed]);after,_=predict(model,relaxed);geometry[name]={'mean_absolute_change':np.mean(abs(after-before),axis=0).tolist(),'p90_absolute_change':np.quantile(abs(after-before),.9,axis=0).tolist(),'before':before.tolist(),'after':after.tolist()}
 report['geometry_diagnostic']={'attempted':24,'completed':len(relaxed),'ids':[m.GetProp('SourceID') for m in relaxed],'scope':'No new quantum labels; stability only','models':geometry}
 qms=list(Chem.SDMolSupplier(str(a.root/'quantum_cohort/cohort.sdf'),removeHs=False));qp={name:predict(model,qms)[0].tolist() for name,model in models.items()};qmanifest=json.loads((a.root/'quantum_cohort/manifest.json').read_text());old=np.array([r['A_prediction'] for r in qmanifest['records']]);delta=float(np.max(abs(old-np.array(qp['dpa_historical']))));assert delta<1e-4
 (a.output/'quantum_cohort_predictions.json').write_text(json.dumps({'models':qp,'historical_A_recheck_max_abs_difference':delta,'cohort_manifest_sha256':hashlib.sha256((a.root/'quantum_cohort/manifest.json').read_bytes()).hexdigest()},indent=2))
 report['unique_evaluated_raw_geometries']=len(cache);(a.output/'summary.json').write_text(json.dumps(report,indent=2));(a.output/'completed.json').write_text(json.dumps({'pools':len(report['pools']),'visible_gpu':os.environ['CUDA_VISIBLE_DEVICES']}));print('REEVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()
