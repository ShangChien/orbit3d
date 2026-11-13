"""Test terminal property calibration on E14 pools excluded from calibration fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from orbit3d.physics.factory import build_forcefield
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def metrics(rows):
    output={}
    for name in ['baseline','affine','adapted']:
        paired=[r for r in rows if r.get('truth') is not None and r.get(name) is not None]
        if not paired:output[name]={'paired':0};continue
        error=np.asarray([np.asarray(r[name])-r['truth'] for r in paired])
        tp=fp=fn=0
        for r in paired:
            j=['homo','lumo'].index(r['target']);actual=abs(r['truth'][j]-r['target_eV'])<=.25;predicted=abs(r[name][j]-r['target_eV'])<=.25
            tp+=int(actual and predicted);fp+=int(predicted and not actual);fn+=int(actual and not predicted)
        output[name]={'paired':len(paired),'mae_eV':abs(error).mean(0).tolist(),'rmse_eV':np.sqrt((error**2).mean(0)).tolist(),'true_positive':tp,'false_positive':fp,'false_negative':fn}
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--weight',type=Path,required=True);p.add_argument('--training-manifest',type=Path,required=True);p.add_argument('--fit-summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);freeze_run(a.output,__file__);meta=json.loads((a.oracle/'oracle_meta.json').read_text())
    field=build_forcefield('dpa4',str(a.weight),meta['training_elements'],'cuda:0');model=InternalDPA4PropertyOracle.load_from_dir(str(a.oracle),field);parent=InternalDPA4PropertyOracle.load_from_dir(str(a.parent),field)
    train=json.loads(a.training_manifest.read_text());seen={r['scaffold'] for r in train['records'] if r['status']=='accepted' and r['split']=='train'}
    coefficients=np.asarray(json.loads(a.fit_summary.read_text())['affine_coefficients']);inputs={'oracle':a.oracle,'parent':a.parent,'weight':a.weight,'training':a.training_manifest,'fit':a.fit_summary};rows=[]
    for target in ['homo','lumo']:
        for control in ['correct','off']:
            cohort=a.root/f'e14_cohort_{target}_{control}';results=a.root/f'e14_quantum_{target}_{control}'
            manifest=json.loads((cohort/'manifest.json').read_text());assert json.loads((results/'completed.json').read_text())['attempted']==manifest['selected']
            assert hashlib.sha256((cohort/'cohort.sdf').read_bytes()).hexdigest()==manifest['sdf_sha256']
            molecules=list(Chem.SDMolSupplier(str(cohort/'cohort.sdf'),removeHs=False));assert len(molecules)==manifest['selected']
            inputs[str(cohort)+'/manifest']=cohort/'manifest.json';inputs[str(cohort)+'/sdf']=cohort/'cohort.sdf'
            prediction=model.evaluate_3d_mols_batched(molecules,batch_size=16)
            previous=parent.evaluate_3d_mols_batched(molecules,batch_size=16)
            for item,mol,pred,error,old,old_error in zip(manifest['records'],molecules,prediction['mean'],prediction['errors'],previous['mean'],previous['errors']):
                path=results/f"job{item['index']:02d}"/'result.json';q=json.loads(path.read_text()) if path.exists() else {'status':'worker_failed'}
                if path.exists():
                    inputs[str(path)]=path;assert q['input_sha256']==manifest['sdf_sha256']
                    np.testing.assert_allclose(mol.GetConformer().GetPositions(),q['coordinates_A'],atol=1e-10,rtol=0)
                scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(mol),includeChirality=False) or '<ACYCLIC>'
                baseline=None if old_error else old.tolist()
                rows.append(dict(cohort=cohort.name,index=item['index'],arm=item['arm'],target=target,target_eV=item['target_eV'],control=control,
                    seen_generated_train_scaffold=scaffold in seen,baseline=baseline,
                    affine=(np.asarray(baseline)*coefficients[:,0]+coefficients[:,1]).tolist() if baseline is not None else None,
                    adapted=None if error else pred.tolist(),prediction_error=error,quantum_status=q['status'],truth=q['singlepoint']['prediction_eV'] if q['status']=='completed' else None))
    record_input_hashes(a.output,inputs)
    report={'scope':'E14 pools excluded from fitting and early stopping. Seen/unseen refers only to added generated-label training scaffolds; source-pretrained exposure is shared. This evaluates property predictions, not changes to generator yield. Off pools are unique coordinates, not duplicated target requests; off hit diagnostics use their stored P50 request.','parent_oracle':str(a.parent),'records_total':len(rows),'quantum_completed':sum(r['truth'] is not None for r in rows),'all':metrics(rows),'common_complete':metrics([r for r in rows if r['truth'] is not None and all(r[k] is not None for k in ['baseline','affine','adapted'])]),'unseen_generated_scaffold':metrics([r for r in rows if not r['seen_generated_train_scaffold']]),'by_cohort':{key:metrics([r for r in rows if r['cohort']==key]) for key in sorted({r['cohort'] for r in rows})},'records':rows}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report['all'],flush=True)

if __name__=='__main__':main()
