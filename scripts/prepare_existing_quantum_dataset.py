"""Reuse completed same-protocol quantum geometries as an explicit data extension."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from loom.generator.molecular_state import mol_to_clean_smiles
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run, record_input_hashes
from loom.oracles.unimol2_task import file_sha256, graph_set


def scaffold(mol):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(mol),includeChirality=False) or '<ACYCLIC>'


def split_for_scaffold(key):
    value=int(hashlib.sha256(('E17-existing-DFT:'+key).encode()).hexdigest()[:16],16)/2**64
    return 'train' if value<.8 else 'valid' if value<.9 else 'test'


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohorts',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--source-train',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    inputs={'cohorts':a.cohorts,'source_train':a.source_train};source_graphs=graph_set(a.source_train)
    heldout=set()
    for split in ['valid','calibration','test']:
        path=a.source/(split+'.csv');heldout.update(pd.read_csv(path).scaffold);inputs[split]=path
    records=[];candidates=[]
    for spec in json.loads(a.cohorts.read_text()):
        cohort,results=Path(spec['cohort']),Path(spec['results'])
        manifest=json.loads((cohort/'manifest.json').read_text());sdf=cohort/'cohort.sdf'
        digest=file_sha256(sdf);assert digest==manifest['sdf_sha256']
        assert json.loads((results/'completed.json').read_text())['attempted']==manifest['selected']
        inputs[str(cohort)+'/manifest']=cohort/'manifest.json';inputs[str(sdf)]=sdf
        molecules=list(Chem.SDMolSupplier(str(sdf),removeHs=False));assert len(molecules)==manifest['selected']
        for item,mol in zip(manifest['records'],molecules):
            index=item['index'];path=results/f'job{index:02d}'/'result.json';q=json.loads(path.read_text());inputs[str(path)]=path
            row=dict(cohort=str(cohort),index=index,status='excluded',quantum_status=q['status']);records.append(row)
            assert q['input_sha256']==digest and mol is not None
            if q['status']!='completed':row['reason']=q['status'];continue
            if q['protocol']!='PBE0/def2-TZVP/DF(def2-universal-jkfit), grid level3, SCF1e-8; approximate OE62 reference':
                raise ValueError('Mixed quantum protocols require separate dataset construction')
            np.testing.assert_allclose(mol.GetConformer().GetPositions(),q['coordinates_A'],atol=1e-10,rtol=0)
            graph=mol_to_clean_smiles(mol);key=scaffold(mol);row.update(smiles=graph,scaffold=key)
            if key in heldout:row['reason']='source_holdout_scaffold';continue
            if graph in source_graphs:row['reason']='already_source_training_graph';continue
            rank=hashlib.sha256((graph+json.dumps(q['coordinates_A'],separators=(',',':'))).encode()).hexdigest()
            candidates.append((rank,len(records)-1,mol,q))
    seen=set();groups={name:[] for name in ['train','valid','test']}
    for rank,position,mol,q in sorted(candidates,key=lambda v:v[0]):
        row=records[position];graph=row['smiles']
        if graph in seen:row['reason']='duplicate_graph';continue
        seen.add(graph);split=split_for_scaffold(row['scaffold']);row.update(status='accepted',split=split,geometry_selection_sha256=rank)
        mol=Chem.Mol(mol);mol.SetProp('SourceID','ExistingDFT:'+rank)
        mol.SetProp('DFTSourceCohort',row['cohort']);mol.SetIntProp('DFTSourceIndex',row['index'])
        mol.SetProp('DFTProtocol',q['protocol']);mol.SetProp('LabelTheory',q['protocol'])
        mol.SetProp('PropertyLabelOrigin','existing_DFT_singlepoint')
        mol.SetProp('GeometryOrigin','Existing frozen MLIP terminal; exact quantum-input coordinates, not DFT-optimized')
        for target,value in zip(['homo','lumo'],q['singlepoint']['prediction_eV']):
            mol.SetProp(target,str(float(value)));mol.SetProp('DFT_'+target,str(float(value)))
        groups[split].append(mol)
    for name,molecules in groups.items():write_precise_sdf(a.output/(name+'.sdf'),molecules)
    record_input_hashes(a.output,inputs)
    report=dict(scope='Previously computed E13/E14/E16 terminal single points; no new DFT. One geometry per canonical graph selected by geometry hash, never property success. Source holdout scaffolds and existing source training graphs excluded. New 80/10/10 scaffold hash split. Historical cohorts are development assets and no longer independent policy tests after training use. B labels are attached in a separate unchanged-coordinate pass.',
                records=records,counts={k:len(v) for k,v in groups.items()},exclusions=dict(Counter(r['reason'] for r in records if r['status']=='excluded')))
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(report['counts'],report['exclusions'])


if __name__=='__main__':main()
