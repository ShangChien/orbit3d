"""Compare already frozen raw-pool predictions with asynchronous quantum singlepoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from orbit3d.generator.octree_backend import trajectory_to_instance


def verify_geometry(record, symbols, coordinates, element_table):
    if record['symbols']!=symbols:raise ValueError('Quantum input does not match the frozen raw candidate')
    # Raw octree coordinates are in the positive grid box. Both pool oracles
    # and SDF exports consume the shared trajectory conversion, which centers them.
    inst=trajectory_to_instance({'atom_symbols':symbols,'positions_np':coordinates},element_table=element_table)
    expected=np.asarray(record['coordinates_A']);actual=np.asarray(inst.coords)
    if expected.shape!=actual.shape or not np.allclose(expected,actual,atol=1e-10,rtol=0):
        raise ValueError('Quantum input does not match the frozen raw candidate')
    return float(np.abs(expected-actual).max())


def summarize(rows,model):
    paired=[r for r in rows if 'quantum_eV' in r and model in r['predictions_eV']]
    error=[r['predictions_eV'][model]-r['quantum_eV'] for r in paired]
    agreement={'true_positive':0,'false_positive':0,'false_negative':0,'true_negative':0}
    for r in paired:
        predicted=abs(r['predictions_eV'][model]-r['target_eV'])<=.25
        actual=abs(r['quantum_eV']-r['target_eV'])<=.25
        agreement[('true_' if predicted==actual else 'false_')+('positive' if predicted else 'negative')]+=1
    return {'selected':len(rows),'paired':len(paired),'active_property_mae_eV':float(np.abs(error).mean()) if error else None,
        'active_property_bias_eV':float(np.mean(error)) if error else None,'target_hit_agreement_025':agreement}


def main():
    p=argparse.ArgumentParser();p.add_argument('--quantum',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    q=json.loads(a.quantum.read_text());rows=[];sources={};b_models={}
    def read(path):
        content=path.read_bytes();sources[str(path)]=hashlib.sha256(content).hexdigest();return json.loads(content)
    elements=tuple(read(a.manifest)['elements'])
    for record in q['records']:
        run=Path(record['source_run']);pool=run.parent;target=record['target'];k=['homo','lumo'].index(target);i=record['candidate_id']
        raw=read(pool/'raw/summary.json');symbols=read(run/'raw_symbols.json')
        coordinates=run/'raw_trajectories.npz'
        sources[str(coordinates)]=hashlib.sha256(coordinates.read_bytes()).hexdigest()
        with np.load(coordinates) as arrays:delta=verify_geometry(record,symbols[i],arrays[f'coordinates_{i}'],elements)
        original=next(r for r in raw['records'] if r['candidate_id']==i)
        b_root=a.campaign_root/f'unimol2_{target}_seed{record["training_seed"]}'
        b=read(b_root/f'{target}_{pool.name}/summary.json');assert Path(b['source_pool']).resolve()==pool.resolve()
        inputs=read(b_root/'input_manifest.json');assert inputs['manifest'][a.manifest.name]==sources[str(a.manifest)]
        role=f'{target}/{pool.name}'
        for field,path in [('original',pool/'raw/summary.json'),('coordinates',coordinates),('symbols',run/'raw_symbols.json')]:
            assert inputs[f'{role}/{field}'][path.name]==sources[str(path)],'Frozen B inputs changed'
        b_models[str(b_root)]=inputs['oracle'];independent=next(r for r in b['records'] if r['candidate_id']==i)
        predictions={}
        if original['evaluated']:predictions['dpa']=original['prediction'][k]
        if original['independent_evaluated']:predictions['fingerprint']=original['independent_prediction'][k]
        if independent['unimol2_evaluated']:predictions['unimol2']=independent['unimol2_prediction_eV'][k]
        row={key:record[key] for key in ['index','target','target_eV','arm','selection','training_seed','inference_seed','candidate_id']}
        row.update(predictions_eV=predictions,coordinate_match_max_delta_A=delta,quantum_status=record['status'])
        if 'singlepoint' in record:row['quantum_eV']=record['singlepoint']['prediction_eV'][k]
        if independent.get('support_exclusion'):row['unimol2_support_exclusion']=independent['support_exclusion']
        rows.append(row)
    sources[str(a.quantum)]=hashlib.sha256(a.quantum.read_bytes()).hexdigest()
    report={'scope':'Existing frozen predictions reused only after shared trajectory centering, atom-order, coordinate and input-hash checks. Active property only; selected quantum cohort, not a population calibration estimate. No new model inference or policy updates.',
        'selected':len(rows),'planned':q['selection']['expected'],'b_model_hashes':b_models,
        'models':{name:summarize(rows,name) for name in ['dpa','fingerprint','unimol2']},
        'by_target_arm':{f'{target}/{arm}':{name:summarize([r for r in rows if r['target']==target and r['arm']==arm],name) for name in ['dpa','fingerprint','unimol2']}
            for target in ['homo','lumo'] for arm in dict.fromkeys(r['arm'] for r in rows)},
        'records':rows,'source_sha256':sources}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['models'],indent=2))


if __name__=='__main__':main()
