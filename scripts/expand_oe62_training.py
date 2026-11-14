"""Expand only the frozen training split, preserving source geometries and test IDs."""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
from rdkit import Chem
from loom.utils.oe62 import adapt_record
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run, record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old=json.loads((a.frozen/'manifest.json').read_text())
    assert hashlib.sha256(a.source.read_bytes()).hexdigest()==old['source_sha256']
    table=pd.read_csv(a.frozen/'train.csv');ids=table.identifier.tolist()
    assert len(ids)==len(set(ids))==43951 and set(old['selection_ids']['train']).issubset(ids)
    for split in ['valid','calibration','test']:
        assert not set(ids).intersection(pd.read_csv(a.frozen/f'{split}.csv').identifier)
    frame=pd.read_json(a.source,orient='split').set_index('refcode_csd',drop=False)
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    molecules=[]
    for i,identifier in enumerate(ids):
        mol=adapt_record(frame.loc[identifier].to_dict(),tuple(old['elements']),old['max_atoms']);mol.SetProp('Split','train')
        expected=table.iloc[i]
        assert abs(float(mol.GetProp('homo'))-expected.homo)<1e-7
        assert abs(float(mol.GetProp('lumo'))-expected.lumo)<1e-7
        molecules.append(mol)
        if i%5000==0:print('expanded',i,flush=True)
    write_precise_sdf(a.output/'train.sdf',molecules)
    old['selection_ids']['train']=ids
    old['training_expansion']='All frozen training IDs; original evaluation targets and holdouts unchanged.'
    old['files']={'train.sdf':hashlib.sha256((a.output/'train.sdf').read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(old,indent=2)+'\n')
    record_input_hashes(a.output,{'source':a.source,'frozen_manifest':a.frozen/'manifest.json','train_table':a.frozen/'train.csv'})
    print('complete',len(molecules),flush=True)

if __name__=='__main__':main()
