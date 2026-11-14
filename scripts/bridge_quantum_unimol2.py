"""Frozen Uni-Mol2 reassessment of the formal GRPO200/time-SFT quantum cohort."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from loom.oracles.unimol2_oracle import InternalUniMol2Oracle
from loom.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--quantum',type=Path,required=True);p.add_argument('--sdf',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);freeze_run(a.output,__file__)
    selected=json.loads(a.selection.read_text());expected=selected['hashes'][selected['selected_unimol']]['model.pt']
    with (a.oracle/'model.pt').open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected
    assert hashlib.sha256((a.oracle/"oracle_meta.json").read_bytes()).hexdigest()==selected["hashes"][selected["selected_unimol"]]["oracle_meta.json"]
    q=json.loads(a.quantum.read_text());assert hashlib.sha256(a.sdf.read_bytes()).hexdigest()==q['selection']['sdf_sha256']
    source=list(Chem.SDMolSupplier(str(a.sdf),removeHs=False));mols=[];items=[]
    for row in q['records']:
        i=row['index'];m=source[i];assert int(m.GetProp('QuantumIndex'))==i
        assert not any(at.GetSymbol()=='Te' for at in m.GetAtoms()),'Outside frozen common support'
        mols.append(m);items.append({'index':i,'stage':'input','target':row['target'],'target_eV':row['target_eV'],'arm':row['arm'],'quantum_eV':row['singlepoint']['prediction_eV']})
        if 'optimized_singlepoint' in row and row['geometry']['converged'] and row['geometry']['graph_preserved']:
            out=Chem.Mol(m);out.GetConformer().SetPositions(np.array(row['geometry']['coordinates_A']))
            mols.append(out);items.append({**items[-1],'stage':'optimized','quantum_eV':row['optimized_singlepoint']['prediction_eV']})
    start=time.perf_counter();oracle=InternalUniMol2Oracle.load_from_dir(a.oracle);prediction,_=oracle.predict_3d_mols(mols,batch_size=8)
    probe,_=oracle.predict_3d_mols(mols[:4],batch_size=1);replay_diff=float(np.abs(probe-prediction[:4]).max());assert replay_diff<1e-4
    for item,pred in zip(items,prediction):item['prediction_eV']=pred.tolist()
    accepted={i['index'] for i in items if i['stage']=='optimized'}
    def stats(rows):
        deltas=[];targets=[]
        for r in rows:
            k=['homo','lumo'].index(r['target']);deltas.append(r['prediction_eV'][k]-r['quantum_eV'][k]);targets.append(r['prediction_eV'][k]-r['target_eV'])
        return {'n':len(rows),'mae_against_quantum_eV':float(np.abs(deltas).mean()),'target_mae_eV':float(np.abs(targets).mean()),'target_hits_025':sum(abs(x)<=.25 for x in targets)}
    report={'scope':'Same frozen formal GRPO200/time-SFT quantum cohort; selected B never changes candidates, primary predictions, geometry admission or quantum outcomes.',
        'selected_model':selected['selected_unimol'],'checkpoint_sha256':actual,'input_scope':'Official source heavy coordinates only; hydrogen-only motion is invisible.',
        'batch_replay_max_difference_eV':replay_diff,'wall_seconds':time.perf_counter()-start,'records':items,
        'comparison':{'input':stats([r for r in items if r['stage']=='input']),
            'input_same_accepted_subset':stats([r for r in items if r['stage']=='input' and r['index'] in accepted]),
            'optimized':stats([r for r in items if r['stage']=='optimized'])},
        'by_target':{f'{target}/{stage}':stats([r for r in items if r['target']==target and r['stage']==stage and r['index'] in accepted]) for target in ['homo','lumo'] for stage in ['input','optimized']}}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    record_input_hashes(a.output,{'quantum':a.quantum,'sdf':a.sdf,'selection':a.selection,'oracle':a.oracle})
    print(json.dumps(report['comparison'],indent=2))


if __name__=='__main__':main()
