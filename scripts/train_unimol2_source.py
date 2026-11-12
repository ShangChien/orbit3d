"""Full Uni-Mol2 adaptation with official inputs, EMA, and development-only selection."""
import argparse,copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np,torch
from rdkit import Chem
from orbit3d.oracles.unimol2_oracle import build_unimol2,load_pretrained,SourceUniMol2Dataset,collate_unimol2,grouped_batches
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def update_count(batches,budget):
    n=steps=0
    for ids in batches:
        n+=len(ids)
        if n>=budget:steps+=1;n=0
    return steps+bool(n)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--size',choices=['310M','1.1B'],required=True)
 p.add_argument('--batch-size',type=int,required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--seed',type=int,default=42);p.add_argument('--patience',type=int,default=10);p.add_argument('--probe',action='store_true');a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2','3'];torch.set_num_threads(2);lock_deterministic_environment(a.seed);freeze_run(a.output,__file__)
 weight=a.root/'assets/unimol2'/a.size/'checkpoint.pt';data=a.root/'data';manifest=json.loads((data/'manifest.json').read_text());(a.output/'config.json').write_text(json.dumps(vars(a),default=str,indent=2))
 record_input_hashes(a.output,{'train':data/'train.sdf','valid':data/'valid.sdf','manifest':data/'manifest.json','pretrained':weight})
 ms={s:list(Chem.SDMolSupplier(str(data/f'{s}.sdf'),removeHs=False)) for s in ['train','valid']};ys={s:np.array([[float(m.GetProp(k)) for k in ['homo','lumo']] for m in ms[s]],dtype=np.float32) for s in ms}
 mean=ys['train'].mean(0);scale=ys['train'].std(0);ds={s:SourceUniMol2Dataset(ms[s],(ys[s]-mean)/scale) for s in ms}
 model=build_unimol2(a.size,dropout=0.);loaded=load_pretrained(model,weight);print('PRETRAINED',loaded,flush=True);model=model.cuda();ema=copy.deepcopy(model).eval().requires_grad_(False)
 opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);params=[p for p in model.parameters()];ema_params=[p for p in ema.parameters()]
 epoch_batches=[grouped_batches(ds['train'].lengths,a.batch_size,a.seed+e) for e in range(a.epochs)];total=sum(update_count(b,32) for b in epoch_batches);warmup=max(1,int(.06*total));step=0
 if a.probe:
  ids=np.argsort(-ds['train'].lengths)[:a.batch_size];batch,y=collate_unimol2([ds['train'][i] for i in ids]);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=model({k:v.cuda() for k,v in batch.items()},classification_head_name='homo_lumo',features_only=True)[0];loss=torch.nn.functional.smooth_l1_loss(pred,y.cuda(),beta=.1)
  loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
  with torch.no_grad():torch._foreach_mul_(ema_params,.999);torch._foreach_add_(ema_params,params,alpha=.001)
  torch.cuda.synchronize();result={'finite':bool(torch.isfinite(loss)),'batch':len(ids),'max_heavy_atoms':int(ds['train'].lengths.max()),'peak_memory_bytes':torch.cuda.max_memory_allocated(),'seconds':time.perf_counter()-started}
  (a.output/'probe.json').write_text(json.dumps(result,indent=2));print(result,flush=True);return
 out=a.output/'oracle';out.mkdir();best=float('inf');best_state=None;history=[];stale=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 def evaluate(net):
  net.eval();pred=np.empty_like(ys['valid'])
  with torch.no_grad():
   for ids in grouped_batches(ds['valid'].lengths,a.batch_size):
    batch,_=collate_unimol2([ds['valid'][i] for i in ids]);batch={k:v.cuda() for k,v in batch.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):value=net(batch,classification_head_name='homo_lumo',features_only=True)[0]
    pred[ids]=value.float().cpu().numpy()
  errors=abs(pred-ds['valid'].targets);return float(errors.mean()),(errors*scale).mean(0).tolist()
 for epoch,batches in enumerate(epoch_batches):
  model.train();opt.zero_grad(set_to_none=True);accum=0;loss_sum=0.;epoch_started=time.perf_counter()
  for bi,ids in enumerate(batches):
   batch,y=collate_unimol2([ds['train'][i] for i in ids]);batch={k:v.cuda() for k,v in batch.items()}
   with torch.autocast('cuda',dtype=torch.bfloat16):
    pred=model(batch,classification_head_name='homo_lumo',features_only=True)[0];loss=torch.nn.functional.smooth_l1_loss(pred,y.cuda(),beta=.1,reduction='sum')/2
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
   loss.backward();accum+=len(ids);loss_sum+=float(loss.detach())
   if accum>=32 or bi==len(batches)-1:
    step+=1;factor=step/warmup if step<=warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))
    for group in opt.param_groups:group['lr']=a.lr*factor
    for param in params:
     if param.grad is not None:param.grad.div_(accum)
    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);accum=0
    with torch.no_grad():torch._foreach_mul_(ema_params,.999);torch._foreach_add_(ema_params,params,alpha=.001)
  current,current_mae=evaluate(model);ema_score,ema_mae=evaluate(ema);selected=ema if ema_score<current else model;score=min(current,ema_score)
  if score<best:
   best=score;stale=0;best_epoch=epoch+1;best_kind='ema' if selected is ema else 'raw';best_state={k:v.detach().cpu().clone() for k,v in selected.state_dict().items()}
  else:stale+=1
  entry={'epoch':epoch+1,'train_loss':loss_sum/len(ds['train']),'raw_valid_normalized_mae':current,'raw_valid_mae_eV':current_mae,'ema_valid_normalized_mae':ema_score,'ema_valid_mae_eV':ema_mae,'seconds':time.perf_counter()-epoch_started,'optimizer_steps':step};history.append(entry)
  if (epoch+1)%5==0:torch.save(best_state,out/'model.pt')
  (a.output/'progress.json').write_text(json.dumps({'history':history,'best_epoch':best_epoch,'best_kind':best_kind,'best':best,'peak_memory_bytes':torch.cuda.max_memory_allocated()},indent=2));print(json.dumps(entry),flush=True)
  if stale>=a.patience:break
 torch.save(best_state,out/'model.pt');h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 meta={'oracle_type':'internal_unimol2','backbone_version':'Uni-Mol2','model_size':a.size,'upstream_commit':'90f52c41299a1a582da0f9765e9f87aa21faa16a','mean':mean.tolist(),'scale':scale.tolist(),'target_cols':['homo','lumo'],'target_units':['eV','eV'],'label_theory':manifest['theory'],'train_records':len(ms['train']),'validation_records':len(ms['valid']),'pretrained_sha256':h(weight),'checkpoint_sha256':h(out/'model.pt'),'data_manifest_sha256':h(data/'manifest.json'),'best_valid_normalized_mae':best,'best_epoch':best_epoch,'best_kind':best_kind,'history':history,'loaded_backbone':loaded,'input_contract':{'source_geometry':True,'remove_hydrogen':True,'center_coordinates':True,'crop':False,'max_source_atoms':80,'drop_feat_prob':1.,'padding':'group by canonical per-molecule padded length, matching train/inference'},'uncertainty':'Single chosen checkpoint; use residual calibration, not zero spread as certainty','validation_precision':'BF16 autocast; final evaluation also records FP32 precision sensitivity','seed':a.seed}
 (out/'oracle_meta.json').write_text(json.dumps(meta,indent=2)+'\n');(a.output/'completed.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-started,'visible_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'best':best},indent=2));print('COMPLETED',a.size,flush=True)

if __name__=='__main__':main()
