"""Pair frozen A/B predictions with the same terminal coordinates and DFT outcomes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def summarize(rows):
    output={}
    for model in ['dpa','unimol2']:
        paired=[r for r in rows if r['truth_eV'] is not None and r[model] is not None]
        if not paired:output[model]={'paired':0};continue
        error=np.asarray([np.asarray(r[model])-r['truth_eV'] for r in paired]);tp=fp=fn=0
        for r in paired:
            j=['homo','lumo'].index(r['target']);truth=abs(r['truth_eV'][j]-r['target_eV'])<=.25;pred=abs(r[model][j]-r['target_eV'])<=.25
            tp+=int(truth and pred);fp+=int(pred and not truth);fn+=int(truth and not pred)
        output[model]={'paired':len(paired),'surrogate_vs_dft_mae_eV':abs(error).mean(0).tolist(),'surrogate_minus_dft_bias_eV':error.mean(0).tolist(),'true_positive':tp,'false_positive':fp,'false_negative':fn,'quantum_missing':sum(r['truth_eV'] is None for r in rows)}
    common=[r for r in rows if r['truth_eV'] is not None and all(r[m] is not None for m in ['dpa','unimol2'])]
    if len(common)>1:
        errors=[np.asarray([np.asarray(r[m])-r['truth_eV'] for r in common]) for m in ['dpa','unimol2']]
        output['paired_residual_correlation']={'n':len(common),'homo_lumo':[float(np.corrcoef(errors[0][:,j],errors[1][:,j])[0,1]) if all(np.std(e[:,j])>0 for e in errors) else None for j in range(2)]}
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--quantum',type=Path,required=True);group=p.add_mutually_exclusive_group(required=True);group.add_argument('--legacy',type=Path);group.add_argument('--flat-independent',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    quantum=json.loads(a.quantum.read_text());rows=[];cache={};hashes={str(a.quantum):hashlib.sha256(a.quantum.read_bytes()).hexdigest()};lock=None
    for source in quantum['records']:
        arm=source['arm'];terminal=Path(source['source_terminal']);broot=a.flat_independent if a.flat_independent else a.legacy/'independent'/arm
        if arm not in cache:
            paths=[terminal/'summary.json',terminal/'graph_preserved.sdf',broot/(arm+'.json'),broot/'input_manifest.json']
            for path in paths:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            inputs=json.loads((broot/'input_manifest.json').read_text())
            assert inputs[str(terminal)+'/summary']['summary.json']==hashes[str(terminal/'summary.json')]
            assert inputs[str(terminal)+'/sdf']['graph_preserved.sdf']==hashes[str(terminal/'graph_preserved.sdf')]
            if lock is None:lock=inputs['oracle']
            assert lock==inputs['oracle'] and lock['model.pt']=='108b50b8d9910144cd840446b86b2f95500bec5c27cf8a08de20b5cc2701d385'
            key=lambda r:(r['quantile'],r['inference_seed'],r['candidate_id'])
            cache[arm]=({key(r):r for r in json.loads((terminal/'summary.json').read_text())['records']},{key(r):r for r in json.loads((broot/(arm+'.json')).read_text())['records']})
        key=(source['quantile'],source['inference_seed'],source['candidate_id']);raw,b=(c[key] for c in cache[arm])
        np.testing.assert_allclose(raw['coordinates_A'],source['coordinates_A'],atol=1e-10,rtol=0)
        np.testing.assert_array_equal(raw['final']['prediction_eV'],source['dpa_prediction_eV'])
        rows.append(dict(index=source['index'],arm=arm,target=source['target'],target_eV=source['target_eV'],quantile=source['quantile'],inference_seed=source['inference_seed'],candidate_id=source['candidate_id'],in_domain=bool(raw.get('domain') and raw['domain']['in_domain']),dpa=source['dpa_prediction_eV'],unimol2=b['prediction_eV'] if b['evaluated'] else None,quantum_status=source['status'],truth_eV=source['singlepoint']['prediction_eV'] if source['status']=='completed' else None))
    report={'scope':'Frozen models, exact terminal coordinate pairing, all selected endpoints. Property errors and confusion use only successful quantum pairs; SCF failures remain explicit, not labelled property negatives. Off confusion uses the stored P50 request, without duplicating endpoints across targets. No new inference or score selection.','b_model_lock':lock,'all':summarize(rows),'domain':{str(flag):summarize([r for r in rows if r['in_domain']==flag]) for flag in [True,False]},'arms':{arm:summarize([r for r in rows if r['arm']==arm]) for arm in cache},'records':rows,'source_sha256':hashes}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['all'],indent=2))

if __name__=='__main__':main()
