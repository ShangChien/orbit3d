"""Train frozen-MLIP property heads directly on audited source SDF coordinates."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.physics.factory import build_forcefield
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.oracles.validation import compare_oracle_reload
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.oracles.applicability import MorganDomain
from orbit3d.oracles.calibration import ResidualCalibration
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weight');p.add_argument('--backend',choices=['dpa4','fingerprint'],default='dpa4');p.add_argument('--device',default='cuda:0')
    p.add_argument('--targets',nargs='+',default=['homo','lumo']);p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();started=time.perf_counter();torch.set_num_threads(2);lock_deterministic_environment(a.seed)
    freeze_run(a.output,__file__)
    (a.output/'run_config.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
    record_input_hashes(a.output,{'forcefield':a.weight,'source_manifest':a.data/'manifest.json',
                               **{k:a.data/f'{k}.sdf' for k in ['train','valid','calibration']}})
    mols={k:list(Chem.SDMolSupplier(str(a.data/f'{k}.sdf'),removeHs=False)) for k in ['train','valid','calibration']}
    for split,ms in mols.items():
        if not ms or any(m is None for m in ms):raise ValueError(f'Unreadable {split} SDF')
    theories={m.GetProp('LabelTheory') for ms in mols.values() for m in ms}
    if len(theories)!=1:raise ValueError('Mixed label calculation levels')
    values={k:np.array([[float(m.GetProp(n)) for n in a.targets] for m in ms]) for k,ms in mols.items()}
    elements=sorted({atom.GetSymbol() for ms in mols.values() for m in ms for atom in m.GetAtoms()})
    if a.backend == 'dpa4':
        if not a.weight:raise ValueError('DPA training requires --weight')
        ff=build_forcefield('dpa4',a.weight,elements,a.device)
        model=InternalDPA4PropertyOracle(ff,len(a.targets),a.targets,n_ensemble_heads=3)
        model.target_units=['eV']*len(a.targets);model.geometry_method='source_oe62_pbe_vdw'
        model.label_theory=next(iter(theories))
        model.fit_from_dataset(mols['train'],values['train'],str(a.output/'oracle'),epochs=a.epochs,
                               seed=a.seed,batch_size=a.batch_size,validation_mols=mols['valid'],validation_targets=values['valid'])
        restored=InternalDPA4PropertyOracle.load_from_dir(str(a.output/'oracle'),ff)
    else:
        model=FingerprintOracle(a.targets)
        model.label_theory=next(iter(theories))
        model.fit(mols['train'],values['train'],a.output/'oracle',seed=a.seed,n_models=3,n_trees=128,n_jobs=4)
        restored=FingerprintOracle.load_from_dir(a.output/'oracle')
    results={}
    for split in ['valid','calibration']:
        if a.backend == 'dpa4':
            prediction,std,reload=compare_oracle_reload(model,restored,mols[split])
        else:
            prediction,std=model.predict_3d_mols(mols[split]);again,_=restored.predict_3d_mols(mols[split])
            np.testing.assert_allclose(prediction,again,atol=0,rtol=0)
            reload={'prediction_max_abs_diff_eV':float(np.abs(prediction-again).max())}
        results[split]={'n':len(prediction),'mae_eV':dict(zip(a.targets,np.abs(prediction-values[split]).mean(0).tolist())),
                        'reload':reload,'source_ids':[m.GetProp('SourceID') for m in mols[split]]}
        np.savez(a.output/f'{split}_predictions.npz',truth=values[split],prediction=prediction,std=std)
    ResidualCalibration.fit(a.targets,values['calibration'],prediction,std).save(a.output/'calibration.json')
    domain=MorganDomain([Chem.MolToSmiles(Chem.RemoveHs(m)) for m in mols['train']],0)
    domain.cutoff=float(np.quantile([domain.score(m) for m in mols['calibration']],.05))
    domain.save(a.output/'applicability.json')
    (a.output/'summary.json').write_text(json.dumps({'backend':a.backend,'elapsed_seconds':time.perf_counter()-started,'targets':a.targets,'label_theory':theories.pop(),'metrics':results,
                                                   'training_ids':[m.GetProp('SourceID') for m in mols['train']]},indent=2)+'\n')
    print({split:{k:v for k,v in values.items() if k!='source_ids'}
           for split,values in results.items()},flush=True)


if __name__=='__main__':main()
