"""Measure source-to-MLIP geometry/prediction drift on fixed development IDs."""
import argparse,json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import torch
from loom.physics.factory import build_forcefield
from loom.physics.conformer_relaxer import relax_conformer_energy_only
from loom.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from loom.generator.molecular_state import rdkit_mol_to_mol3d_instance,mol3d_to_validated_rdkit_mol
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run,record_input_hashes
from loom.utils.reproducibility import lock_deterministic_environment


def edges(mol):
    return {tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in mol.GetBonds()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--forcefield',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();lock_deterministic_environment(42);torch.set_num_threads(2)
    freeze_run(a.output,__file__);record_input_hashes(a.output,{'source':a.data/'valid.sdf','oracle':a.oracle,'forcefield':a.forcefield})
    mols=list(Chem.SDMolSupplier(str(a.data/'valid.sdf'),removeHs=False))[:32]
    elements=json.loads((a.data/'manifest.json').read_text())['elements']
    ff=build_forcefield('dpa4',a.forcefield,elements,a.device)
    oracle=InternalDPA4PropertyOracle.load_from_dir(str(a.oracle),ff)
    initial,_=oracle.predict_3d_mols(mols);records=[];exports=[]
    for i,mol in enumerate(mols):
        truth=np.array([float(mol.GetProp(n)) for n in oracle.target_cols])
        row={'source_id':mol.GetProp('SourceID'),'source_labels_eV':truth.tolist(),
             'source_prediction_eV':initial[i].tolist(),'completed':False}
        try:
            inst=rdkit_mol_to_mol3d_instance(mol,truth,center=False,element_table=tuple(elements),coordinate_dtype=np.float64)
            refined,info=relax_conformer_energy_only(inst,ff,n_steps=10,step_size=.004)
            final=mol3d_to_validated_rdkit_mol(refined)
            inferred=Chem.MolFromXYZBlock(Chem.MolToXYZBlock(final))
            rdDetermineBonds.DetermineConnectivity(inferred)
            pred,_=oracle.predict_3d_mols([final])
            row.update(completed=True,connectivity_preserved=edges(mol)==edges(inferred),
                       final_prediction_eV=pred[0].tolist(),prediction_change_eV=(pred[0]-initial[i]).tolist(),
                       coordinate_rms_displacement_A=float(np.sqrt(np.square(refined.coords-inst.coords).sum(-1).mean())),
                       optimization=info)
            for name in oracle.target_cols:
                final.SetProp('SourceReference_'+name,mol.GetProp(name));final.ClearProp(name)
            final.SetProp('GeometryOrigin','10-step DPA energy-only refinement of OE62 source; quantum labels not recomputed')
            exports.append(final)
        except (ValueError,FloatingPointError) as error:
            row['error']=str(error)
        records.append(row);print(row['source_id'],row['completed'],flush=True)
    write_precise_sdf(a.output/'refined.sdf',exports)
    report={'scope':'First 32 frozen development IDs; source labels do not apply as truth on refined coordinates',
            'steps':10,'step_size':.004,'attempted':len(records),'completed':len(exports),'records':records}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
