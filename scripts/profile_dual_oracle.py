"""Freeze predictor selection on development; then calibrate and open locked source test once."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np,torch
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from loom.physics.factory import build_forcefield
from loom.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from loom.oracles.unimol_oracle import InternalUniMolOracle
from loom.utils.run_evidence import freeze_run,record_input_hashes

NAMES=['homo','lumo']
def metrics(y,p):
 if not len(y):return {'n':0}
 e=p-y
 return {'n':len(y),'mae':np.abs(e).mean(0).tolist(),'rmse':np.sqrt(np.square(e).mean(0)).tolist(),
   'bias':e.mean(0).tolist(),'absolute_error_p90':np.quantile(abs(e),.9,axis=0).tolist(),
   'slope':[float(np.polyfit(y[:,k],p[:,k],1)[0]) for k in range(2)] if len(y)>2 and np.std(y,axis=0).min()>1e-8 else None,
   'intercept':[float(np.polyfit(y[:,k],p[:,k],1)[1]) for k in range(2)] if len(y)>2 and np.std(y,axis=0).min()>1e-8 else None}

def predict(model,ms,batch=32):
 if isinstance(model,InternalUniMolOracle):return model.predict_3d_mols(ms,batch_size=batch)
 return model.predict_3d_mols_batched(ms,batch_size=batch)

def load_models(root,selection):
 manifest=json.loads((root/'data/manifest.json').read_text())
 ff=build_forcefield('dpa4','/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt',manifest['splits']['train']['elements'],'cuda:0')
 model_dir=root/'training'/selection/'oracle'
 meta=json.loads((model_dir/'oracle_meta.json').read_text())
 if meta['oracle_type']=='internal_unimol2':
  from loom.oracles.unimol2_oracle import InternalUniMol2Oracle
  evaluator=InternalUniMol2Oracle.load_from_dir(model_dir,'cuda:0')
 else:evaluator=InternalUniMolOracle.load_from_dir(str(model_dir),'cuda:0')
 models={'dpa_common':InternalDPA4PropertyOracle.load_from_dir(str(root/'training/dpa_common/oracle'),ff),
  'dpa_historical':InternalDPA4PropertyOracle.load_from_dir('/sjtu-combustion/wjw/chensq/orbit3d_oe62_20261112/oracle_dpa/oracle',ff),
  'unimol':evaluator}
 return ff,models

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',type=Path,help='Development-only version/precision comparison, frozen before test evaluation');a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2','3']
 torch.set_num_threads(2);freeze_run(a.output,__file__)
 selection=json.loads(a.selection.read_text()) if a.selection else None
 names=selection['candidates'].keys() if selection else ['unimol_lr1e4','unimol_lr3e5']
 choices={name:json.loads((a.root/'training'/name/'oracle/oracle_meta.json').read_text()) for name in names}
 choice=selection['selected_unimol'] if selection else min(choices,key=lambda n:choices[n]['best_valid_normalized_mae'])
 if selection:
  checkpoint=a.root/'training'/choice/'oracle/model.pt'
  assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==selection['candidates'][choice]['checkpoint_sha256']
 ff,models=load_models(a.root,choice);manifest=json.loads((a.root/'data/manifest.json').read_text())
 train=list(Chem.SDMolSupplier(str(a.root/'data/train.sdf'),removeHs=False));valid=list(Chem.SDMolSupplier(str(a.root/'data/valid.sdf'),removeHs=False))
 target=lambda ms:np.array([[float(m.GetProp(k)) for k in NAMES] for m in ms])
 yt=target(train);yv=target(valid);cuts=np.quantile(yt,[.2,.4,.6,.8],axis=0)
 dev={};devpred={};devtime={}
 for name,model in models.items():
  torch.cuda.synchronize();start=time.perf_counter();pred,spread=predict(model,valid);torch.cuda.synchronize();devtime[name]=time.perf_counter()-start
  assert np.isfinite(pred).all();dev[name]=metrics(yv,pred);devpred[name]=pred
 b=dev['unimol'];bins=[np.mean(abs(devpred['unimol'][np.searchsorted(cuts[:,k],yv[:,k],side='right')==j,k]-yv[np.searchsorted(cuts[:,k],yv[:,k],side='right')==j,k])) for k in range(2) for j in range(5)]
 passed=bool(max(b['mae'])<=.20 and max(abs(np.array(b['bias'])))<=.10 and all(.75<=s<=1.25 for s in b['slope']) and max(bins)<=.35)
 development={'selected_unimol':choice,'selection_rule':selection['selection_rule'] if selection else 'Minimum normalized development MAE across two predefined learning rates; no calibration/test/candidate B predictions used',
  'candidates':{n:{'best_epoch':v['best_epoch'],'best_valid_normalized_mae':v['best_valid_normalized_mae']} for n,v in choices.items()},'metrics':dev,'gate':{'passed':passed,'max_mae':.20,'max_abs_bias':.10,'slope_range':[.75,1.25],'max_quintile_mae':.35,'observed_quintile_mae':bins}}
 (a.output/'development.json').write_text(json.dumps(development,indent=2)+'\n');print('DEVELOPMENT',development,flush=True)
 if not passed:raise RuntimeError('Uni-Mol development gate failed; keep locked test closed and improve predictor first')
 inputs={'split_manifest':a.root/'data/manifest.json','dpa_common':a.root/'training/dpa_common/oracle','dpa_historical':Path('/sjtu-combustion/wjw/chensq/orbit3d_oe62_20261112/oracle_dpa/oracle'),'unimol':a.root/'training'/choice/'oracle'}
 hashes=record_input_hashes(a.output,inputs)
 (a.output/'models_frozen.json').write_text(json.dumps({'selected_unimol':choice,'hashes':hashes,'roles':{'optimization_A':'historical DPA for existing candidates, predictions unchanged','fair_predictor_A':'DPA common-support refit','cross_evaluator_B':'selected Uni-Mol'},'test_unsealed_after_this_lock':not bool(selection),'test_history':'Previously reported for v1; v2 selection uses development only' if selection else 'First source test profile','development_selection':selection},indent=2)+'\n')
 fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
 def scaf(m):return MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(m))
 train_scaffolds={Chem.MolToSmiles(scaf(m),isomericSmiles=False) for m in train};trainfps=[fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_scaffolds]
 report={'development':development,'dataset':manifest,'split_metrics':{},'timing':{},'scope':'Common-support source 3D prediction. Historical DPA retained separately; common DPA and Uni-Mol share exact splits. Calibration intervals are empirical, not guaranteed conditional probabilities.'};calq={}
 for split in ['valid','calibration','test']:
  ms=valid if split=='valid' else list(Chem.SDMolSupplier(str(a.root/'data'/f'{split}.sdf'),removeHs=False));y=target(ms);ids=[m.GetProp('SourceID') for m in ms];preds={};spreads={};out={}
  for name,model in models.items():
   if split=='valid':pred=devpred[name];spread=np.zeros_like(pred);seconds=devtime[name]
   else:
    torch.cuda.synchronize();start=time.perf_counter();pred,spread=predict(model,ms);torch.cuda.synchronize();seconds=time.perf_counter()-start
   assert np.isfinite(pred).all();preds[name]=pred;spreads[name]=spread;out[name]=metrics(y,pred);report['timing'][split+'/'+name]={'seconds':seconds,'molecules_per_second':len(ms)/seconds}
   if split=='calibration':calq[name]=np.quantile(abs(pred-y),min(1,np.ceil((len(y)+1)*.9)/len(y)),axis=0,method='higher')
   if split=='test':out[name]['calibrated_marginal90']={'half_width_eV':calq[name].tolist(),'width_eV':(2*calq[name]).tolist(),'coverage':(abs(pred-y)<=calq[name]).mean(0).tolist(),'both_covered_fraction':float(np.all(abs(pred-y)<=calq[name],axis=1).mean())}
  atoms=np.array([m.GetNumAtoms() for m in ms]);sim=np.array([max(DataStructs.BulkTanimotoSimilarity(fpgen.GetFingerprint(scaf(m)),trainfps)) for m in ms]);novel=np.array([Chem.MolToSmiles(scaf(m),isomericSmiles=False) not in train_scaffolds for m in ms])
  groups={**{f'atoms_{lo}_{hi}':(atoms>=lo)&(atoms<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]},
    **{f'contains_{s}':np.array([any(x.GetSymbol()==s for x in m.GetAtoms()) for m in ms]) for s in manifest['splits']['train']['elements']},
    'novel_scaffold':novel,'seen_scaffold':~novel,'scaffold_similarity_low':sim<.4,'scaffold_similarity_mid':(sim>=.4)&(sim<.7),'scaffold_similarity_high':sim>=.7}
  for k,name in enumerate(NAMES):
   assigned=np.searchsorted(cuts[:,k],y[:,k],side='right')
   for j in range(5):groups[f'{name}_quintile{j+1}']=assigned==j
  out['groups']={g:{name:metrics(y[mask],pred[mask]) for name,pred in preds.items()} for g,mask in groups.items()}
  ea=preds['dpa_common']-y;eb=preds['unimol']-y
  out['pair']={NAMES[k]:{'residual_correlation':float(np.corrcoef(ea[:,k],eb[:,k])[0,1]),'absolute_error_correlation':float(np.corrcoef(abs(ea[:,k]),abs(eb[:,k]))[0,1]),'both_error_gt_025':float(np.mean((abs(ea[:,k])>.25)&(abs(eb[:,k])>.25))),'A_error_gt_025':float(np.mean(abs(ea[:,k])>.25)),'B_error_gt_025':float(np.mean(abs(eb[:,k])>.25)),'both_error_gt_050':float(np.mean((abs(ea[:,k])>.5)&(abs(eb[:,k])>.5)))} for k in range(2)}
  disagreement=np.max(abs(preds['dpa_common']-preds['unimol']),axis=1);top=np.argsort(-disagreement)[:20]
  out['largest_disagreements']=[{'source_id':ids[i],'labels':y[i].tolist(),'A':preds['dpa_common'][i].tolist(),'B':preds['unimol'][i].tolist(),'atoms':int(atoms[i])} for i in top]
  np.savez_compressed(a.output/f'{split}_predictions.npz',ids=ids,labels=y,atoms=atoms,scaffold_similarity=sim,**preds)
  report['split_metrics'][split]=out;print(split,{n:out[n]['mae'] for n in models},flush=True)
  (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 # Fixed 128-molecule development throughput benchmark, after warmup; report resident and incremental memory.
 report['batch_profile']={}
 for name,model in models.items():
  predict(model,valid[:8],8);rows=[]
  for batch in [8,16,32]:
   resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
   for _ in range(3):
    torch.cuda.synchronize();start=time.perf_counter();pred,_=predict(model,valid[:128],batch);torch.cuda.synchronize();times.append(time.perf_counter()-start)
   rows.append({'batch_size':batch,'seconds':times,'median_molecules_per_second':128/float(np.median(times)),'resident_bytes':resident,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'max_abs_difference_vs_full_valid':float(np.max(abs(pred-devpred[name][:128])))})
  report['batch_profile'][name]=rows
 report['calibration90_half_width_eV']={k:v.tolist() for k,v in calq.items()};(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 (a.output/'completed.json').write_text(json.dumps({'selected_unimol':choice,'visible_gpu':os.environ['CUDA_VISIBLE_DEVICES']}));print('PROFILE COMPLETE',flush=True)

if __name__=='__main__':main()
