"""Split new terminal DFT labels by scaffold; keep E14 and source holdouts out of fitting."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run,record_input_hashes


def scaffold(mol):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(mol),includeChirality=False) or '<ACYCLIC>'


def assignment(key):
    u=int(hashlib.sha256(('E15-20261114:'+key).encode()).hexdigest()[:16],16)/2**64
    return 'train' if u<.7 else 'valid' if u<.8 else 'test'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    heldout=set();inputs={};groups={k:[] for k in ['train','valid','test']};records=[];seen=set()
    for split in ['valid','calibration','test']:
        path=a.source/f'{split}.csv';inputs[split+'_table']=path;heldout.update(pd.read_csv(path).scaffold)
    for cohort,results in [('cohort848','quantum848'),('off_cohort','off_quantum'),('brics_cohort','brics_quantum')]:
        folder=a.root/cohort;manifest=json.loads((folder/'manifest.json').read_text())
        assert json.loads((a.root/results/'completed.json').read_text())['attempted']==manifest['selected']
        inputs[cohort+'/manifest']=folder/'manifest.json';inputs[cohort+'/sdf']=folder/'cohort.sdf'
        assert hashlib.sha256((folder/'cohort.sdf').read_bytes()).hexdigest()==manifest['sdf_sha256']
        mols=list(Chem.SDMolSupplier(str(folder/'cohort.sdf'),removeHs=False));assert len(mols)==manifest['selected']
        for item,mol in zip(manifest['records'],mols):
            path=a.root/results/f"job{item['index']:02d}"/'result.json'
            if path.exists():inputs[str(path)]=path
            record=dict(cohort=cohort,index=item['index']);records.append(record)
            if not path.exists():record.update(status='excluded',reason='worker_failed');continue
            result=json.loads(path.read_text());assert result['input_sha256']==manifest['sdf_sha256']
            if result['status']!='completed':record.update(status='excluded',reason=result['status']);continue
            assert mol is not None
            np.testing.assert_allclose(mol.GetConformer().GetPositions(),result['coordinates_A'],atol=1e-10,rtol=0)
            key=scaffold(mol);smiles=Chem.MolToSmiles(Chem.RemoveHs(mol));record.update(scaffold=key,smiles=smiles)
            identity=(smiles,tuple(at.GetSymbol() for at in mol.GetAtoms()),hashlib.sha256(mol.GetConformer().GetPositions().tobytes()).hexdigest())
            if identity in seen:record.update(status='excluded',reason='exact_geometry_repeat');continue
            seen.add(identity)
            if key in heldout:record.update(status='excluded',reason='source_holdout_scaffold');continue
            split=assignment(key);record.update(status='accepted',split=split)
            mol=Chem.Mol(mol);mol.SetProp('SourceID',f'E15:{cohort}:{item["index"]}')
            mol.SetProp('LabelTheory','PBE0/def2-TZVP/DF(def2-universal-jkfit), grid3, SCF1e-8')
            mol.SetProp('GeometryOrigin','Frozen DPA terminal geometry; no re-embedding')
            for name,value in zip(['homo','lumo'],result['singlepoint']['prediction_eV']):mol.SetProp(name,str(value))
            groups[split].append(mol)
    assert all(groups.values())
    reserved={scaffold(m) for k in ['valid','test'] for m in groups[k]}
    anchors=[m for m in Chem.SDMolSupplier(str(a.source/'train.sdf'),removeHs=False) if scaffold(m) not in reserved]
    anchors.sort(key=lambda m:hashlib.sha256(('E15-anchor:'+m.GetProp('SourceID')).encode()).hexdigest())
    anchors=anchors[:len(groups['train'])];assert len(anchors)==len(groups['train'])
    source_valid=list(Chem.SDMolSupplier(str(a.source/'valid.sdf'),removeHs=False))
    write_precise_sdf(a.output/'train_generated.sdf',groups['train'])
    write_precise_sdf(a.output/'train.sdf',groups['train']+anchors)
    write_precise_sdf(a.output/'valid_generated.sdf',groups['valid'])
    write_precise_sdf(a.output/'test_generated.sdf',groups['test'])
    write_precise_sdf(a.output/'valid.sdf',groups['valid']+source_valid)
    weights=[.5/len(groups['valid'])]*len(groups['valid'])+[.5/len(source_valid)]*len(source_valid)
    inputs['source_anchors']=a.source/'train.sdf';inputs['source_valid']=a.source/'valid.sdf';record_input_hashes(a.output,inputs)
    report=dict(scope='New generated labels are scaffold-disjoint across 70/10/20 hash groups. Original supervised pretraining is shared by all methods; no claim of unseen pretraining scaffolds. Source holdout scaffolds excluded from fitting pool. Source anchors exclude generated valid/test scaffolds. E14 labels never used in fitting or early stopping.',counts={k:len(v) for k,v in groups.items()},source_anchor_count=len(anchors),validation_weights=weights,records=records,anchor_ids=[m.GetProp('SourceID') for m in anchors])
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(report['counts'],flush=True)

if __name__=='__main__':main()
