"""Two-GPU launcher for frozen HOMO/LUMO point-generation scans."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--python',required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--runtime-commit',required=True)
p.add_argument('--gpus',nargs=2,type=int,default=[2,3])
a=p.parse_args();assert len(set(a.gpus))==2
assert all(g>=0 for g in a.gpus)
a.output.mkdir(parents=True,exist_ok=False)
start=time.time();jobs=[];active=[]
for target,gpu in zip(['homo','lumo'],a.gpus):
    command=[a.python,'-u',str(a.source/'scripts/experiment_point_generation_scan.py'),
             '--root',str(a.root),'--target',target,'--output',str(a.output/target),'--runtime-commit',a.runtime_commit]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(a.source/'src'),OMP_NUM_THREADS='2')
    log=(a.output/f'{target}.log').open('w')
    proc=subprocess.Popen(command,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    job={'target':target,'gpu':gpu,'pid':proc.pid,'command':command,'start_unix':time.time()}
    jobs.append(job);active.append((proc,log,job));print('START',target,gpu,proc.pid,flush=True)
with (a.output/'gpu_samples.jsonl').open('w') as monitor:
    while active:
        sample={'unix':time.time(),'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
                'process_csv':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)}
        monitor.write(json.dumps(sample)+'\n');monitor.flush()
        remaining=[]
        for proc,log,job in active:
            code=proc.poll()
            if code is None:remaining.append((proc,log,job))
            else:
                job.update(exit_code=code,end_unix=time.time());log.close()
                print('END',job['target'],code,flush=True)
        active=remaining;(a.output/'jobs.json').write_text(json.dumps(jobs,indent=2))
        if active:time.sleep(10)
(a.output/'completed.json').write_text(json.dumps({'jobs':jobs,'elapsed_seconds':time.time()-start},indent=2))
assert all(j['exit_code']==0 for j in jobs),'A lane failed; retain outputs for diagnosis'
