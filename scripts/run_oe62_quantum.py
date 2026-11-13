"""Execute a frozen SDF cohort, explicit GPU slots or bounded CPU workers."""
import argparse,json,os,subprocess,time
from pathlib import Path
from rdkit import Chem


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True,help='Python executable or runtime wrapper')
    p.add_argument('--engine',choices=['cpu','gpu'],default='gpu');p.add_argument('--cpu-workers',type=int,default=4)
    p.add_argument('--scratch-root',type=Path,help='Optional local or RAM-backed temporary root; results stay under --output')
    p.add_argument('--gpus',nargs='+',type=int,default=[0,1]);p.add_argument('--optimize',action='store_true');a=p.parse_args()
    if len(set(a.gpus))!=len(a.gpus):p.error('GPU indices must be distinct')
    if a.cpu_workers<1:p.error('CPU worker count must be positive')
    slots=range(a.cpu_workers) if a.engine=='cpu' else a.gpus
    a.output.mkdir(parents=True,exist_ok=False)
    scratch=(a.scratch_root/a.output.name if a.scratch_root else a.output/'scratch')
    scratch.mkdir(parents=True,exist_ok=False)
    pending=list(range(len(list(Chem.SDMolSupplier(str(a.input),removeHs=False)))))
    active={};jobs=[];start=time.time();count=len(pending)
    while pending or active:
        for gpu in slots:
            if gpu in active or not pending:continue
            i=pending.pop(0);cmd=[str(a.python),str(Path(__file__).with_name('oe62_quantum_singlepoint.py')),'--input',str(a.input),'--index',str(i),'--output',str(a.output/f'job{i:02d}'),'--engine',a.engine]
            if a.optimize:cmd.append('--optimize')
            log=(a.output/f'job{i:02d}.log').open('w');proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if a.engine=='cpu' else str(gpu),TMPDIR=str(scratch.resolve())),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            job={'index':i,'engine':a.engine,'gpu':gpu if a.engine=='gpu' else None,'worker':gpu,'scratch_dir':str(scratch.resolve()),'pid':proc.pid,'command':cmd,'start':time.time()};jobs.append(job);active[gpu]=(proc,log,job)
        for gpu,(proc,log,job) in list(active.items()):
            if proc.poll() is not None:
                job.update(exit_code=proc.returncode,end=time.time());log.close();del active[gpu]
                print('Finished',job['index'],'exit',job['exit_code'],flush=True)
        (a.output/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
        if active:time.sleep(2)
    (a.output/'completed.json').write_text(json.dumps({'attempted':count,'engine':a.engine,'optimization_requested':a.optimize,'elapsed_seconds':time.time()-start,'jobs':jobs},indent=2)+'\n')


if __name__=='__main__':main()
