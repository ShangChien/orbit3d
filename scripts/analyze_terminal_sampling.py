"""Paired sampling uncertainty on fixed E12 checkpoints, not training-seed uncertainty."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ARMS=['point_frozen','terminal_sft','terminal_structure','terminal_physical','terminal_property']

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    vectors={};hashes={};summary={}
    for arm in ARMS:
        path=a.root/'independent'/arm/(arm+'.json');raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest()
        rows=json.loads(raw)['records'];rows=sorted(rows,key=lambda r:(r['quantile'],r['inference_seed'],r['candidate_id']))
        assert len(rows)==384
        array=np.asarray([[float(r['physical_accepted']),float(r['evaluated'] and abs(r['prediction_eV'][0]-r['goal_eV'])<=.25)] for r in rows]).reshape(3,2,64,2)
        vectors[arm]=array
        summary[arm]={'physical_yield':float(array[...,0].mean()),'b_hit_yield':float(array[...,1].mean()),'prediction_std_by_target':{str(q):float(np.std([r['prediction_eV'][0] for r in rows if r['quantile']==q and r['evaluated']])) for q in [25,50,75]}}
    rng=np.random.default_rng(1314);differences={arm:[] for arm in ARMS[1:]}
    for _ in range(4000):
        seeds=rng.integers(0,2,size=2);indices=rng.integers(0,64,size=(3,2,64))
        sample={arm:np.stack([v[q,seeds[j],indices[q,j]] for q in range(3) for j in range(2)]).mean(axis=(0,1)) for arm,v in vectors.items()}
        for arm in differences:differences[arm].append(sample[arm]-sample['point_frozen'])
    report={'scope':'Paired seed-block plus within-target proposal bootstrap, 4000 replicates, RNG1314. Only two inference seeds on ONE trained checkpoint per arm; intervals do not estimate training variance or confirm method superiority. Metrics use all proposals. Frozen B is developer evidence.',
        'arms':summary,'difference_vs_frozen':{arm:{'physical_yield_delta_95':np.quantile(values,[.025,.975],axis=0)[:,0].tolist(),'b_hit_yield_delta_95':np.quantile(values,[.025,.975],axis=0)[:,1].tolist()} for arm,values in differences.items()},'source_sha256':hashes}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['difference_vs_frozen'],indent=2))

if __name__=='__main__':main()
