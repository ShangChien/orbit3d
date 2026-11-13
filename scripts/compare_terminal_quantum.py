"""Compare frozen A/B with quantum labels on exactly the selected terminal coordinates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--quantum',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    hashes={}
    def read(path):
        raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    quantum=read(a.quantum);rows=[]
    for q in quantum['records']:
        source=Path(q['source_terminal']);terminal=read(source/'summary.json')
        b_root=a.root/'independent'/q['arm'];b=read(b_root/(q['arm']+'.json'))
        bound=read(b_root/'input_manifest.json')
        assert bound[str(source)+'/summary']['summary.json']==hashes[str(source/'summary.json')]
        match=lambda r:(r['quantile'],r['inference_seed'],r['candidate_id'])==(q['quantile'],q['inference_seed'],q['candidate_id'])
        original=next(r for r in terminal['records'] if match(r));independent=next(r for r in b['records'] if match(r))
        if 'coordinates_A' in q:assert np.allclose(q['coordinates_A'],original['coordinates_A'],atol=1e-10,rtol=0)
        row={k:q[k] for k in ['index','arm','quantile','selection','target_eV']}
        row.update(dpa_eV=q['dpa_prediction_eV'][0],b_eV=independent['prediction_eV'][0] if independent['evaluated'] else None,
                   quantum_eV=q['singlepoint']['prediction_eV'][0] if 'singlepoint' in q else None,status=q['status'])
        rows.append(row)
    summary={}
    for model in ['dpa','b']:
        paired=[r for r in rows if r[model+'_eV'] is not None and r['quantum_eV'] is not None]
        summary[model]={'paired':len(paired),'mae_against_quantum_eV':float(np.mean([abs(r[model+'_eV']-r['quantum_eV']) for r in paired])) if paired else None,
                        'predicted_hits':sum(abs(r[model+'_eV']-r['target_eV'])<=.25 for r in paired),
                        'confirmed_predicted_hits':sum(abs(r[model+'_eV']-r['target_eV'])<=.25 and abs(r['quantum_eV']-r['target_eV'])<=.25 for r in paired)}
    a.output.write_text(json.dumps({'scope':'Selected terminal cohort only; same coordinates verified, no independent-score selection. Missing quantum and B evaluations remain explicit.','models':summary,'records':rows,'source_sha256':hashes,'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
