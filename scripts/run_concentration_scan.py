"""Inference-only 2x2 CFG/temperature ablation. Never allocates GPU 0/1."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ARMS={'baseline':(1.,[.85,.3,.25]),'cfg3':(3.,[.85,.3,.25]),
      'cool':(1.,[.65,.2,.18]),'cfg3_cool':(3.,[.65,.2,.18])}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    pending={2:[('homo',arm) for arm in ARMS],3:[('lumo',arm) for arm in reversed(ARMS)]}
    active={};jobs=[];start=time.time()
    with (a.output/'gpu_samples.jsonl').open('w') as monitor:
        while any(pending.values()) or active:
            for gpu in (2,3):
                if gpu in active or not pending[gpu]:continue
                prop,arm=pending[gpu].pop(0);cfg,temps=ARMS[arm]
                cmd=[a.python,'-u',str(a.source/'scripts/experiment_point_generation_scan.py'),
                     '--root',str(a.root),'--target',prop,'--output',str(a.output/f'{prop}_{arm}'),
                     '--runtime-commit','1fcd1bd','--quantiles','25','50','75','--candidates','32',
                     '--retain','0','--speed-replays','0','--skip-off','--cfg-scale',str(cfg),
                     '--temperatures',*[str(t) for t in temps]]
                log=(a.output/f'{prop}_{arm}.log').open('w')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(a.source/'src'),OMP_NUM_THREADS='2')
                proc=subprocess.Popen(cmd,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT)
                job={'property':prop,'arm':arm,'gpu':gpu,'pid':proc.pid,'command':cmd,'start':time.time()}
                jobs.append(job);active[gpu]=(proc,log,job)
            for gpu,(proc,log,job) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    log.close();job.update(exit_code=code,end=time.time());del active[gpu]
                    if code:pending[gpu]=[]
            monitor.write(json.dumps({'unix':time.time(),'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)})+'\n');monitor.flush()
            (a.output/'jobs.json').write_text(json.dumps(jobs,indent=2))
            if active or any(pending.values()):time.sleep(10)
    (a.output/'completed.json').write_text(json.dumps({'elapsed_seconds':time.time()-start,'jobs':jobs},indent=2))
    assert len(jobs)==8 and all(j['exit_code']==0 for j in jobs)

if __name__=='__main__':main()
