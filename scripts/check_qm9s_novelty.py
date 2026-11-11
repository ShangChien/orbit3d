"""Check generated graph identities against all supplied QM9s source SMILES."""

import argparse
import json
from pathlib import Path
import lmdb
from rdkit import Chem, RDLogger
from orbit3d.utils.qm9s import read_record, canonical_record

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
RDLogger.DisableLog('rdApp.*')
corpus = set()
counts = {}
for split in ['train', 'valid']:
    env = lmdb.open(str(a.source / f'{split}.lmdb'), subdir=False, readonly=True, lock=False)
    total = rejected = 0
    with env.begin() as tx:
        for _, raw in tx.cursor():
            total += 1
            try:
                corpus.add(canonical_record(read_record(raw)))
            except ValueError:
                rejected += 1
    env.close()
    counts[split] = {'records': total, 'unparseable_source_smiles': rejected}
    print(split, counts[split], flush=True)
records = []
for mol in Chem.SDMolSupplier(str(a.run / 'generated.sdf'), removeHs=False):
    assert mol is not None
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    records.append({'candidate_id': int(mol.GetProp('CandidateID')), 'smiles': smiles,
                    'present_in_supplied_qm9s': smiles in corpus})
summary = {'reference': 'All parseable supplied QM9s train/valid SMILES; not a verified copy of the pretrained model corpus',
           'source_counts': counts, 'unique_reference_structures': len(corpus),
           'exported': len(records), 'not_found_in_reference': sum(not r['present_in_supplied_qm9s'] for r in records),
           'records': records}
(a.run / 'full_qm9s_novelty.json').write_text(json.dumps(summary, indent=2)+'\n')
print({k:v for k,v in summary.items() if k!='records'})
