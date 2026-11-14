"""Warm-start the native DPA property adapter on source anchors and terminal DFT labels."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from loom.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from loom.oracles.validation import compare_oracle_reload
from loom.physics.factory import build_forcefield
from loom.utils.reproducibility import lock_deterministic_environment
from loom.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--source-test',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--weight',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=35);a=p.parse_args()
    lock_deterministic_environment(42);torch.set_num_threads(4);start=time.perf_counter();freeze_run(a.output,__file__)
    meta=json.loads((a.parent/'oracle_meta.json').read_text());manifest=json.loads((a.data/'manifest.json').read_text())
    field=build_forcefield('dpa4',str(a.weight),meta['training_elements'],a.device)
    model=InternalDPA4PropertyOracle.load_from_dir(str(a.parent),field)
    mols={k:list(Chem.SDMolSupplier(str(a.data/f'{k}.sdf'),removeHs=False)) for k in ['train','train_generated','valid','valid_generated','test_generated']}
    mols['source_test']=list(Chem.SDMolSupplier(str(a.source_test),removeHs=False))
    assert all(all(m is not None for m in ms) for ms in mols.values())
    values={k:np.array([[float(m.GetProp(n)) for n in model.target_cols] for m in ms]) for k,ms in mols.items()}
    record_input_hashes(a.output,{'parent':a.parent,'weight':a.weight,'manifest':a.data/'manifest.json',**{k:(a.source_test if k=='source_test' else a.data/f'{k}.sdf') for k in mols}})
    baseline={};baseline_predictions={}
    for k in ['train_generated','valid_generated','test_generated','source_test']:
        pred,_=model.predict_3d_mols_batched(mols[k],batch_size=16)
        baseline[k]=np.abs(pred-values[k]).mean(0).tolist();baseline_predictions[k]=pred
    coefficients=np.stack([np.linalg.lstsq(np.column_stack([baseline_predictions['train_generated'][:,j],np.ones(len(mols['train_generated']))]),values['train_generated'][:,j],rcond=None)[0] for j in range(model.output_dim)])
    model.geometry_method='native source anchors and graph-preserving DPA terminal coordinates'
    model.label_theory='Mixed, explicitly recorded: OE62 PBE0/Tier2 source anchors plus generated PBE0/def2-TZVP/DF terminal labels'
    model.fit_from_dataset(mols['train'],values['train'],str(a.output/'oracle'),epochs=a.epochs,lr=2e-5,seed=42,batch_size=16,
        validation_mols=mols['valid'],validation_targets=values['valid'],warm_start=True,validation_weights=manifest['validation_weights'])
    restored=InternalDPA4PropertyOracle.load_from_dir(str(a.output/'oracle'),field);metrics={}
    for k in ['valid_generated','test_generated','source_test']:
        pred,std,reload=compare_oracle_reload(model,restored,mols[k])
        metrics[k]={'n':len(pred),'baseline_mae_eV':baseline[k],'adapted_mae_eV':np.abs(pred-values[k]).mean(0).tolist(),'affine_mae_eV':np.abs(baseline_predictions[k]*coefficients[:,0]+coefficients[:,1]-values[k]).mean(0).tolist(),'reload':reload}
        np.savez(a.output/f'{k}_predictions.npz',truth=values[k],prediction=pred,std=std,baseline=baseline_predictions[k],affine=baseline_predictions[k]*coefficients[:,0]+coefficients[:,1])
    (a.output/'summary.json').write_text(json.dumps({'scope':'Generated DFT labels scaffold split before fitting. Shared source-pretrained initialization; E14 not used. Best epochs use 50:50 generated/source validation weights. Source native labels and generated quantum labels retain their distinct calculation levels.','affine_coefficients':coefficients.tolist(),'metrics':metrics,'elapsed_seconds':time.perf_counter()-start},indent=2)+'\n')
    print(metrics,flush=True)

if __name__=='__main__':main()
