"""Fixed-split OE62 predictors. Development-only selection; never reads calibration/test labels."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import joblib,numpy as np,torch
from rdkit import Chem
from torch.utils.data import DataLoader
from loom.oracles.unimol_oracle import _UniMol3DDataset,_collate_unimol_3d,load_pretrained_unimol_backbone
from loom.oracles.uniqsar_oracle import StandaloneUniMolModel
from loom.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from loom.physics.factory import build_forcefield
from loom.utils.dataset_io import fit_target_scalers,transform_with_scalers
from loom.utils.reproducibility import lock_deterministic_environment
from loom.utils.run_evidence import freeze_run,record_input_hashes


def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--backend',choices=['unimol','dpa'],required=True);p.add_argument('--weight',required=True)
 p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=1e-4)
 p.add_argument('--patience',type=int,default=8);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2','3']
 torch.set_num_threads(2);lock_deterministic_environment(a.seed);freeze_run(a.output,__file__)
 (a.output/'config.json').write_text(json.dumps(vars(a),default=str,indent=2))
 record_input_hashes(a.output,{'train':a.data/'train.sdf','valid':a.data/'valid.sdf','manifest':a.data/'manifest.json','pretrained':a.weight})
 ms={s:list(Chem.SDMolSupplier(str(a.data/f'{s}.sdf'),removeHs=False)) for s in ['train','valid']}
 ys={s:np.array([[float(m.GetProp(k)) for k in ['homo','lumo']] for m in ms[s]],dtype=np.float32) for s in ms}
 manifest=json.loads((a.data/'manifest.json').read_text());out=a.output/'oracle';out.mkdir();start=time.perf_counter()
 if a.backend=='dpa':
  elements=manifest['splits']['train']['elements'];ff=build_forcefield('dpa4',a.weight,elements,'cuda:0')
  model=InternalDPA4PropertyOracle(ff,2,['homo','lumo'],n_ensemble_heads=3);model.target_units=['eV','eV'];model.geometry_method='source_oe62_pbe_vdw';model.label_theory=manifest['theory']
  model.fit_from_dataset(ms['train'],ys['train'],str(out),epochs=a.epochs,lr=a.lr,batch_size=a.batch_size,seed=a.seed,validation_mols=ms['valid'],validation_targets=ys['valid'])
 else:
  scalers=fit_target_scalers(ys['train'],mode='standard')
  datasets={s:_UniMol3DDataset(ms[s],transform_with_scalers(ys[s],scalers),False,80) for s in ms}
  loaders={s:DataLoader(datasets[s],batch_size=a.batch_size,shuffle=s=='train',collate_fn=_collate_unimol_3d,num_workers=0) for s in ms}
  model=StandaloneUniMolModel(2,dropout=.1).cuda();loaded=load_pretrained_unimol_backbone(model,a.weight)
  opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);total=a.epochs*len(loaders['train']);warmup=max(1,int(.06*total));step=0
  best=float('inf');stale=0;history=[];torch.cuda.reset_peak_memory_stats()
  for epoch in range(a.epochs):
   model.train();loss_sum=0.;epoch_start=time.perf_counter()
   for tok,dist,edge,y in loaders['train']:
    step+=1;factor=step/warmup if step<=warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))
    for group in opt.param_groups:group['lr']=a.lr*factor
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     pred=model(tok.cuda(),dist.cuda(),edge.cuda());loss=torch.nn.functional.smooth_l1_loss(pred,y.cuda(),beta=.1)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();loss_sum+=float(loss.detach())
   model.eval();errors=[];physical=[]
   with torch.no_grad():
    for tok,dist,edge,y in loaders['valid']:
     pred=model(tok.cuda(),dist.cuda(),edge.cuda()).float().cpu().numpy();errors.append(abs(pred-y.numpy()))
     physical.append(np.column_stack([scalers[k].inverse_transform(pred[:,k,None]).ravel() for k in range(2)]))
   score=float(np.concatenate(errors).mean());mae=np.mean(abs(np.concatenate(physical)-ys['valid']),axis=0).tolist()
   if not math.isfinite(score):raise FloatingPointError('Nonfinite development metric')
   if score<best:
    best=score;stale=0;torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},out/'unimol_fold_0.pth');best_epoch=epoch+1
   else:stale+=1
   entry={'epoch':epoch+1,'train_loss':loss_sum/len(loaders['train']),'valid_normalized_mae':score,'valid_mae_eV':mae,'seconds':time.perf_counter()-epoch_start};history.append(entry)
   print(json.dumps(entry),flush=True);(a.output/'progress.json').write_text(json.dumps({'history':history,'best_epoch':best_epoch,'best':best,'peak_memory_bytes':torch.cuda.max_memory_allocated()},indent=2))
   if stale>=a.patience:break
  joblib.dump({'fold_scalers':[scalers]},out/'target_scaler.ss')
  h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
  meta={'schema_version':2,'oracle_type':'internal_unimol','ensemble_kind':'single_fixed_split','scaler_scope':'fixed_training_only',
   'n_folds':1,'output_dim':2,'target_cols':['homo','lumo'],'target_units':['eV','eV'],'label_theory':manifest['theory'],
   'remove_hs':False,'max_atoms':80,'training_elements':manifest['splits']['train']['elements'],'electronic_states':[{'charge':0,'multiplicity':1}],
   'input_geometry_method':'source_oe62_pbe_vdw','pretrained_sha256':h(a.weight),'checkpoint_sha256':{'unimol_fold_0.pth':h(out/'unimol_fold_0.pth')},
   'scaler_sha256':h(out/'target_scaler.ss'),'uncertainty':'Single model; zero ensemble spread is not certainty. Use held-out residual calibration.',
   'train_records':len(ms['train']),'validation_records':len(ms['valid']),'split_manifest_sha256':h(a.data/'manifest.json'),
   'seed':a.seed,'best_epoch':best_epoch,'best_valid_normalized_mae':best,'history':history,'loaded_backbone_tensors':loaded,'training_dropout':.1}
  (out/'oracle_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
 (a.output/'completed.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-start,'backend':a.backend,'visible_device':os.environ['CUDA_VISIBLE_DEVICES']},indent=2))
 print('COMPLETED',a.backend,flush=True)

if __name__=='__main__':main()
