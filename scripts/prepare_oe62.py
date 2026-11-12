"""Adapt official OE62 split-oriented JSON without regenerating coordinates."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
ELEMENTS = ("H", "C", "N", "O", "F", "B", "Si", "P", "S", "Cl", "Br", "I", "Se", "As", "Te")
from orbit3d.utils.oe62 import adapt_record, THEORY
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run


def rank(value, seed=42):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def prepare(source, output, max_atoms=80, seed=42):
    output.mkdir(parents=True, exist_ok=False)
    freeze_run(output,__file__)
    frame = pd.read_json(source, orient='split')
    assert frame.refcode_csd.is_unique
    records, molecules, seen = [], {}, {}
    for index, row in enumerate(frame.to_dict('records')):
        identifier = row['refcode_csd']
        try:
            mol = adapt_record(row, ELEMENTS, max_atoms)
            xyz = mol.GetConformer().GetPositions()
            diameter = float(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1).max())
            if diameter > .24 * 2**7 - .56:
                raise ValueError('outside_depth7_rotation_safe_span')
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
            # Same constitution stays in one group even when source stereo differs.
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(mol), includeChirality=False) or '<ACYCLIC>'
            value = int(rank(scaffold, seed)[:16], 16) / 2**64
            split = 'train' if value < .8 else ('valid' if value < .9 else 'test')
            if split == 'valid':
                split = 'valid' if int(rank('cal:'+scaffold, seed)[:16],16)/2**64 < .5 else 'calibration'
            if smiles in seen:
                records.append({'source_id': identifier, 'status':'duplicate', 'duplicate_of':seen[smiles]})
                continue
            seen[smiles] = identifier
            mol.SetProp('Split', split)
            molecules[identifier] = mol
            records.append({'source_id':identifier,'status':'accepted','split':split,'smiles':smiles,'scaffold':scaffold,
                            'atoms':mol.GetNumAtoms(),'heavy_atoms':mol.GetNumHeavyAtoms(),'diameter_A':diameter,
                            'homo':float(mol.GetProp('homo')),'lumo':float(mol.GetProp('lumo'))})
        except (ValueError, RuntimeError) as error:
            records.append({'source_id':identifier,'status':'excluded','reason':str(error)})
        if index % 2000 == 0:print('adapted',index,'accepted',len(molecules),flush=True)
    accepted = pd.DataFrame([r for r in records if r['status']=='accepted'])
    assert accepted.groupby('scaffold').split.nunique().max()==1
    selection={}
    for split,limit in [('train',8192),('valid',512),('calibration',512),('test',None)]:
        rows=accepted[accepted.split==split].copy()
        rows['rank']=rows.source_id.map(lambda s:rank('selection:'+s,seed))
        rows=rows.sort_values('rank')
        rows.rename(columns={'source_id':'identifier','smiles':'canonical_smiles'}).to_csv(output/f'{split}.csv',index=False)
        chosen=rows if limit is None else rows.head(limit)
        selection[split]=chosen.source_id.tolist()
        write_precise_sdf(output/f'{split}.sdf',[molecules[s] for s in selection[split]])
    train=accepted.set_index('source_id').loc[selection['train']]
    targets={k:{str(q):float(train[k].quantile(q/100)) for q in [10,25,50,75,90]} for k in ['homo','lumo']}
    report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_rows':len(frame),
            'theory':THEORY,'units':'eV','generator_merge_level':7,'rotation_safe_span_A':.24*2**7-.56,'rdkit_version':rdBase.rdkitVersion,'max_atoms':max_atoms,'seed':seed,
            'split_counts':accepted.split.value_counts().to_dict(),'selection_ids':selection,'single_target_points':targets,
            'status_counts':dict(Counter(r['status'] for r in records)),
            'exclusion_counts':dict(Counter(r.get('reason') for r in records if r['status']=='excluded')),
            'atom_quantiles':np.quantile(accepted.atoms,[0,.25,.5,.75,1]).tolist(),
            'elements':sorted({a.GetSymbol() for m in molecules.values() for a in m.GetAtoms()}),
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}}
    (output/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ['selection_ids','files']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-atoms',type=int,default=80);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();prepare(a.source,a.output,a.max_atoms,a.seed)
