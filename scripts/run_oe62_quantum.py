"""Run a frozen SDF queue on GPUs 2 and 3; failures remain in the queue denominator."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from rdkit import Chem
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--optimize',action='store_true');p.add_argument('--reuse-first',type=Path);p.add_argument('--gpu3-after',type=Path);p.add_argument('--gpus',nargs='+',type=int,default=[2,3]);a=p.parse_args();assert set(a.gpus)<= {2,3} and len(set(a.gpus))==len(a.gpus)
a.output.mkdir(exist_ok=False,parents=True);count=len(list(Chem.SDMolSupplier(str(a.input),removeHs=False)));pending=list(range(count));active={};jobs=[];start=time.time()
if a.reuse_first:
 result=json.loads((a.reuse_first/'result.json').read_text())
 assert result['index']==0 and result['input_sha256']==hashlib.sha256(a.input.read_bytes()).hexdigest() and 'optimized_singlepoint' in result
 shutil.copytree(a.reuse_first,a.output/'job00');pending.remove(0);jobs.append({'index':0,'gpu':2,'exit_code':0,'reused_from':str(a.reuse_first),'elapsed_seconds':result['elapsed_seconds']})
while pending or active:
 for gpu in a.gpus:
  if gpu in active or not pending:continue
  if gpu==3 and a.gpu3_after and not a.gpu3_after.exists():continue
  index=pending.pop(0);dest=a.output/f'job{index:02d}';cmd=['/sjtu-combustion/wjw/chensq/orbit3d_quantum_20261111/run_quantum_gpu19_python.sh',str(a.root/'source/scripts/oe62_quantum_singlepoint.py'),'--input',str(a.input),'--index',str(index),'--output',str(dest)]
  if a.optimize:cmd+=['--optimize']
  log=(a.output/f'job{index:02d}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(a.root/'source/src'));proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  job={'index':index,'gpu':gpu,'pid':proc.pid,'start':time.time(),'command':cmd};jobs.append(job);active[gpu]=(proc,log,job)
 for gpu,(proc,log,job) in list(active.items()):
  if proc.poll() is not None:job.update(exit_code=proc.returncode,end=time.time());log.close();del active[gpu]
 (a.output/'jobs.json').write_text(json.dumps(jobs,indent=2))
 if active:time.sleep(5)
(a.output/'completed.json').write_text(json.dumps({'attempted':count,'elapsed_seconds':time.time()-start,'jobs':jobs},indent=2))
