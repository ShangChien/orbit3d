"""One-time frozen-model evaluation on the complete held-out OE62 scaffold split."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.physics.factory import build_forcefield
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.oracles.calibration import ResidualCalibration
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def errors(prediction,truth,names,mask):
    delta=prediction[mask]-truth[mask]
    return {'n':int(mask.sum()),'targets':{name:{'mae_eV':float(np.abs(delta[:,i]).mean()),
        'bias_eV':float(delta[:,i].mean()),'rmse_eV':float(np.sqrt(np.mean(delta[:,i]**2))),
        'absolute_error_p50_p90_p95_eV':np.quantile(np.abs(delta[:,i]),[.5,.9,.95]).tolist()} for i,name in enumerate(names)}} if mask.any() else {'n':0,'targets':{}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads);freeze_run(a.output,__file__);spec=json.loads(a.models.read_text());manifest=json.loads((a.data/'manifest.json').read_text())
    inputs={'model_protocol':a.models,'data_manifest':a.data/'manifest.json','test':a.data/'test.sdf'}
    for name,entry in spec.items():
        inputs[name+'/oracle']=entry['oracle'];inputs[name+'/calibration']=entry['calibration']
        if 'forcefield' in entry:inputs[name+'/forcefield']=entry['forcefield']
    # Freeze model and data hashes before touching test labels or predicting.
    record_input_hashes(a.output,inputs)
    mols=list(Chem.SDMolSupplier(str(a.data/'test.sdf'),removeHs=False));assert all(m is not None for m in mols)
    ids=[m.GetProp('SourceID') for m in mols];assert ids==manifest['selection_ids']['test']
    assert not set(ids)&set(sum([manifest['selection_ids'][k] for k in ['train','valid','calibration']],[]))
    names=['homo','lumo'];truth=np.array([[float(m.GetProp(n)) for n in names] for m in mols]);atoms=np.array([m.GetNumAtoms() for m in mols])
    element_sets=[{at.GetSymbol() for at in m.GetAtoms()} for m in mols];results={};predictions={'truth':truth,'source_ids':ids}
    for name,entry in spec.items():
        started=time.perf_counter()
        if entry['backend']=='fingerprint':
            model=FingerprintOracle.load_from_dir(entry['oracle']);evaluated=model.evaluate_graph_mols(mols)
        else:
            ff=build_forcefield(entry['backend'],entry['forcefield'],tuple(manifest['elements']),'cpu',tier='presto',route='reference')
            model=InternalDPA4PropertyOracle.load_from_dir(entry['oracle'],ff);evaluated=model.evaluate_3d_mols_batched(mols)
        assert model.target_cols==names
        prediction,std=evaluated['mean'],evaluated['std'];ok=np.array([not err for err in evaluated['errors']])
        assert np.isfinite(prediction[ok]).all()
        calibration=ResidualCalibration(**json.loads(Path(entry['calibration']).read_text()));assert calibration.target_cols==names
        lower,upper=calibration.intervals(prediction,std);covered=(truth>=lower)&(truth<=upper)
        marginal_lower,marginal_upper=calibration.intervals(prediction,std,simultaneous=False)
        result={'attempted':len(mols),'evaluated':int(ok.sum()),'coverage':float(ok.mean()),'overall':errors(prediction,truth,names,ok),
                'by_atom_count':{f'{lo}-{hi}':errors(prediction,truth,names,ok&(atoms>=lo)&(atoms<=hi)) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]},
                'by_element_presence':{el:errors(prediction,truth,names,ok&np.array([el in es for es in element_sets])) for el in manifest['elements']},
                'target_neighborhoods':{target:{q:errors(prediction,truth,names,ok&(np.abs(truth[:,j]-center)<=.25)) for q,center in manifest['single_target_points'][target].items()} for j,target in enumerate(names)},
                'calibrated_90pct_joint_coverage':float(covered[ok].all(-1).mean()),
                'calibrated_90pct_marginal_coverage':dict(zip(names,((truth>=marginal_lower)&(truth<=marginal_upper))[ok].mean(0).tolist())),
                'failures':[{'source_id':identifier,'error':error} for identifier,error in zip(ids,evaluated['errors']) if error],
                'wall_seconds':time.perf_counter()-started}
        results[name]=result;predictions[name+'_mean']=prediction;predictions[name+'_std']=std;predictions[name+'_evaluated']=ok
        print(name,result['overall'],result['evaluated'],result['wall_seconds'],flush=True)
        (a.output/'summary.json').write_text(json.dumps({'scope':'Frozen models and calibrations; full untouched source-3D scaffold holdout. No fitting or model selection on this split. Empirical coverage, not a scaffold-exchangeability guarantee.',
            'n':len(mols),'label_theory':manifest['theory'],'models':results},indent=2)+'\n')
    np.savez_compressed(a.output/'predictions.npz',**predictions);print('Frozen source holdout complete',flush=True)


if __name__=='__main__':main()
