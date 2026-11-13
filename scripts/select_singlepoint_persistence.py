"""Freeze a geometry-persistence cohort conditional on previously established DFT hits."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rdkit import Chem
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--quantum',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    freeze_run(a.output,__file__);source=json.loads(a.quantum.read_text());manifest=json.loads((a.cohort/'manifest.json').read_text());sdf=a.cohort/'cohort.sdf'
    assert hashlib.sha256(sdf.read_bytes()).hexdigest()==manifest['sdf_sha256']
    mols=list(Chem.SDMolSupplier(str(sdf),removeHs=False));points=json.loads(a.targets.read_text())['single_target_points']['homo'];chosen=[];export=[];used=set();salt='E13-BRICS-SP-hit-persistence-20261114'
    for q in [25,50,75]:
        goal=points[str(q)];eligible=[r for r in source['records'] if r['arm']=='brics' and r['status']=='completed' and abs(r['singlepoint']['prediction_eV'][0]-goal)<=.25]
        eligible.sort(key=lambda r:hashlib.sha256(f'{salt}:{q}:{r["index"]}:{r["smiles"]}'.encode()).hexdigest())
        count=0
        for row in eligible:
            mol=Chem.Mol(mols[row['index']]);identity=Chem.MolToSmiles(Chem.RemoveHs(mol))
            if identity in used:continue
            used.add(identity);np.testing.assert_allclose(mol.GetConformer().GetPositions(),row['coordinates_A'],atol=1e-10,rtol=0)
            mol.SetProp('QuantumIndex',str(len(chosen)));export.append(mol)
            chosen.append(dict(index=len(chosen),original_index=row['index'],target='homo',target_eV=goal,quantile=q,arm='brics',selection='singlepoint_hit_fixed_hash',smiles=row['smiles'],dpa_prediction_eV=row['dpa_prediction_eV'],original_singlepoint=row['singlepoint'],inference_seed=row['inference_seed'],candidate_id=row['candidate_id']))
            count+=1
            if count==4:break
        assert count==4
    write_precise_sdf(a.output/'cohort.sdf',export)
    report={'evaluation_scope':'Geometry persistence conditional on already-known BRICS DFT singlepoint hits; NOT a fresh proposal hit-rate experiment. Four fixed-hash hits at each of three central HOMO targets; unique molecular graphs across requests; all failures retained, no replacement.','selection_salt':salt,'selected':12,'expected':12,'arms':['brics'],'targets':['homo'],'planned_per_group':{'homo/brics':12},'planned_by_selection':{'singlepoint_hit_fixed_hash':12},'records':chosen,'sdf_sha256':hashlib.sha256((a.output/'cohort.sdf').read_bytes()).hexdigest()}
    record_input_hashes(a.output,{'source_quantum':a.quantum,'source_manifest':a.cohort/'manifest.json','source_sdf':sdf,'targets':a.targets})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print([(r['quantile'],r['original_index']) for r in chosen])

if __name__=='__main__':main()
