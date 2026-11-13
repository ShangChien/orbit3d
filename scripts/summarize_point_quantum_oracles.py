"""Compare frozen predictors and forces on exactly paired quantum geometries."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--quantum',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--failure-replay',type=Path);a=p.parse_args()
    q=json.loads(a.quantum.read_text());r=json.loads(a.predictions.read_text());r['input_manifest']=json.loads(a.inputs.read_text());r['comparison']={}
    accepted={v['index'] for v in q['records'] if 'optimized_singlepoint' in v and v['geometry']['converged'] and v['geometry']['graph_preserved']}
    for model,rows in r['models'].items():
        stats={}
        for stage,label in [('input','input'),('input','input_same_accepted_subset'),('optimized','optimized')]:
            chosen=[v for v in rows if v['stage']==stage and not v['error'] and (label=='input' or v['index'] in accepted)]
            key='singlepoint' if stage=='input' else 'optimized_singlepoint';deltas=[]
            for v in chosen:
                source=q['records'][v['index']];k=['homo','lumo'].index(source['target']);deltas.append(v['prediction_eV'][k]-source[key]['prediction_eV'][k])
            stats[label]={'n':len(chosen),'mae_against_quantum_eV':float(np.abs(deltas).mean()) if chosen else None}
            if model!='fingerprint_v1':
                forces=[v['rms_force_eVA'] for v in chosen]
                stats[label].update(rms_force_p10_p50_p90_eVA=np.quantile(forces,[.1,.5,.9]).tolist(),rms_force_le_0_1=sum(v<=.1 for v in forces))
        r['comparison'][model]=stats
    forces=[q['records'][i]['geometry']['trace'][-1]['rms_gradient_Ha_Bohr']*np.sqrt(3)*27.211386245988/.529177210903 for i in sorted(accepted)]
    r['quantum_geometry_force']={'scope':'Final PBE-D3BJ gradients on converged and graph-preserved endpoints; RMS per atom, not per Cartesian component.',
        'n':len(forces),'rms_force_p10_p50_p90_eVA':np.quantile(forces,[.1,.5,.9]).tolist(),
        'maximum_rms_force_eVA':float(max(forces)),'rms_force_le_0_1':int(sum(v<=.1 for v in forces))}
    r['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.quantum,a.predictions,a.inputs]}
    if a.failure_replay:
        fix=json.loads(a.failure_replay.read_text());old=q['records'][fix['index']]
        delta=float(np.abs(np.asarray(fix['singlepoint']['prediction_eV'])-old['singlepoint']['prediction_eV']).max())
        assert fix['status']==old['status']=='failed' and fix['geometry']['trace'] and delta<1e-5
        r['failure_recording_regression']={'scope':'Separate replay; primary failure is retained unchanged.','index':fix['index'],'status':fix['status'],
            'completed_gradient_records':len(fix['geometry']['trace']),'max_singlepoint_difference_eV':delta,
            'input_sha256':fix['input_sha256'],'script_sha256':fix['script_sha256']}
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['comparison'],indent=2))


if __name__=='__main__':main()
