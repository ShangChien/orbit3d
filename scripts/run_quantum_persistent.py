"""Reuse one GPU context for a frozen cohort; preserve the existing SCF calculation."""
import argparse,json,os,time
from pathlib import Path
from rdkit import Chem
from oe62_quantum_singlepoint import calculate


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch-root',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);scratch=a.scratch_root/a.output.name;scratch.mkdir(parents=True,exist_ok=False)
    os.environ['TMPDIR']=str(scratch.resolve())
    count=len(Chem.SDMolSupplier(str(a.input),removeHs=False));jobs=[];start=time.time()
    for i in range(count):
        job=dict(index=i,engine='gpu',visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),start=time.time(),command=[],pid=os.getpid())
        calculate(argparse.Namespace(input=a.input,index=i,output=a.output/f'job{i:02d}',engine='gpu',optimize=False))
        job.update(end=time.time(),exit_code=0);jobs.append(job)
        (a.output/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    (a.output/'completed.json').write_text(json.dumps(dict(attempted=count,engine='gpu',optimization_requested=False,execution='persistent context; unchanged calculate function',elapsed_seconds=time.time()-start,jobs=jobs),indent=2)+'\n')

if __name__=='__main__':main()
