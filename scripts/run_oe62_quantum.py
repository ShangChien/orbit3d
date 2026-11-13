"""Execute a frozen SDF cohort, one worker per explicitly assigned GPU."""
import argparse,json,os,subprocess,time
from pathlib import Path
from rdkit import Chem


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True,help='Python executable or runtime wrapper')
    p.add_argument('--gpus',nargs='+',type=int,default=[0,1]);p.add_argument('--optimize',action='store_true');a=p.parse_args()
    if len(set(a.gpus))!=len(a.gpus):p.error('GPU indices must be distinct')
    a.output.mkdir(parents=True,exist_ok=False);pending=list(range(len(list(Chem.SDMolSupplier(str(a.input),removeHs=False)))))
    active={};jobs=[];start=time.time();count=len(pending)
    while pending or active:
        for gpu in a.gpus:
            if gpu in active or not pending:continue
            i=pending.pop(0);cmd=[str(a.python),str(Path(__file__).with_name('oe62_quantum_singlepoint.py')),'--input',str(a.input),'--index',str(i),'--output',str(a.output/f'job{i:02d}')]
            if a.optimize:cmd.append('--optimize')
            log=(a.output/f'job{i:02d}.log').open('w');proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            job={'index':i,'gpu':gpu,'pid':proc.pid,'command':cmd,'start':time.time()};jobs.append(job);active[gpu]=(proc,log,job)
        for gpu,(proc,log,job) in list(active.items()):
            if proc.poll() is not None:
                job.update(exit_code=proc.returncode,end=time.time());log.close();del active[gpu]
                print('Finished',job['index'],'exit',job['exit_code'],flush=True)
        (a.output/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
        if active:time.sleep(2)
    (a.output/'completed.json').write_text(json.dumps({'attempted':count,'elapsed_seconds':time.time()-start,'jobs':jobs},indent=2)+'\n')


if __name__=='__main__':main()
