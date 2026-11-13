"""Two-card predictor campaign. No generator or GRPO training."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--python',required=True);a=p.parse_args()
r=a.root;out=r/'training';out.mkdir(exist_ok=False)
queue={2:[('unimol_lr1e4','unimol',str(r/'assets/mol_pre_all_h_220816.pt'),40,32,1e-4)],
       3:[('dpa_common','dpa','/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt',35,16,2e-4),
          ('unimol_lr3e5','unimol',str(r/'assets/mol_pre_all_h_220816.pt'),40,32,3e-5)]}
active={};jobs=[]
with (out/'gpu_monitor.jsonl').open('w') as monitor:
 while any(queue.values()) or active:
  for gpu in [2,3]:
   if gpu in active or not queue[gpu]:continue
   name,backend,weight,epochs,batch,lr=queue[gpu].pop(0)
   cmd=[a.python,'-u',str(r/'source/scripts/train_dual_oracle.py'),'--data',str(r/'data'),'--output',str(out/name),'--backend',backend,'--weight',weight,'--epochs',str(epochs),'--batch-size',str(batch),'--lr',str(lr)]
   log=(out/f'{name}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(r/'source/src'),OMP_NUM_THREADS='2')
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   job={'name':name,'gpu':gpu,'pid':proc.pid,'command':cmd,'start':time.time()};jobs.append(job);active[gpu]=(proc,log,job)
  for gpu,(proc,log,job) in list(active.items()):
   if proc.poll() is not None:
    job.update(exit_code=proc.returncode,end=time.time());log.close();del active[gpu]
    if proc.returncode:queue[gpu]=[]
  monitor.write(json.dumps({'time':time.time(),'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)})+'\n');monitor.flush()
  (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
  if active or any(queue.values()):time.sleep(10)
(out/'completed.json').write_text(json.dumps(jobs,indent=2));assert len(jobs)==3 and all(j['exit_code']==0 for j in jobs)
