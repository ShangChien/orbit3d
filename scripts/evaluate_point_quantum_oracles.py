"""Re-evaluate frozen point-study predictors on exactly the quantum input/end geometries."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.molecular_state import Mol3DInstance,mol3d_to_validated_rdkit_mol
from orbit3d.physics.factory import build_forcefield
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--quantum',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);freeze_run(a.output,__file__)
    spec=json.loads(a.models.read_text());elements=tuple(json.loads(a.manifest.read_text())['elements']);q=json.loads(a.quantum.read_text())
    source=list(Chem.SDMolSupplier(str(a.cohort),removeHs=False));mols=[];keys=[];excluded=[]
    for row in q['records']:
        i=row['index'];mol=source[i];mols.append(mol);keys.append((i,'input'))
        if 'coordinates_A' not in row.get('geometry',{}):continue
        coords=np.array(row['geometry']['coordinates_A']);inst=Mol3DInstance(coords,np.array([elements.index(at.GetSymbol()) for at in mol.GetAtoms()]),graph=mol,element_table=elements)
        try:valid=mol3d_to_validated_rdkit_mol(inst)
        except ValueError as error:excluded.append({'index':i,'stage':'optimized','reason':str(error)});continue
        mols.append(valid);keys.append((i,'optimized'))
    inputs={'cohort':a.cohort,'quantum':a.quantum,'models':a.models,'data_manifest':a.manifest};models={}
    for name in ['dpa','fingerprint_v1','hyff']:
        entry=spec[name];inputs[name+'/oracle']=entry['oracle']
        if entry['backend']=='fingerprint':model=FingerprintOracle.load_from_dir(entry['oracle']);evaluated=model.evaluate_graph_mols(mols)
        else:
            inputs[name+'/forcefield']=entry['forcefield'];ff=build_forcefield(entry['backend'],entry['forcefield'],elements,'cpu',tier='presto',route='reference')
            model=InternalDPA4PropertyOracle.load_from_dir(entry['oracle'],ff);evaluated=model.evaluate_3d_mols_batched(mols)
        assert model.target_cols==['homo','lumo']
        models[name]=[{'index':i,'stage':stage,'prediction_eV':np.asarray(pred).tolist() if not error else None,'error':error} for (i,stage),pred,error in zip(keys,evaluated['mean'],evaluated['errors'])]
        print(name,len(mols),sum(bool(e) for e in evaluated['errors']),flush=True)
    record_input_hashes(a.output,inputs)
    (a.output/'summary.json').write_text(json.dumps({'scope':'Same coordinates as PySCF input/endpoints; original frozen models, no fitting or selection; graph failures excluded only from endpoint prediction and retained in cohort denominator.',
        'models':models,'excluded':excluded},indent=2)+'\n')


if __name__=='__main__':main()
