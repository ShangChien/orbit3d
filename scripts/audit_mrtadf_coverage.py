import argparse,json,hashlib
from collections import Counter
from pathlib import Path
import lmdb
from rdkit import Chem,RDLogger
from loom.utils.qm9s import read_record,adapt_record,canonical_record
RDLogger.DisableLog('rdApp.*')
p=argparse.ArgumentParser(description='Audit complete closed-shell MR-TADF element coverage without repairing labels')
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.source;out=a.output;out.mkdir(parents=True)
report={};elements=tuple(Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1,84))
for group in ['de_novo_gen','local_gen']:
 for split in (['train','valid'] if group=='de_novo_gen' else ['train']):
  source=root/group/'mrtadf_norm_350'/f'{split}.lmdb';env=lmdb.open(str(source),subdir=False,readonly=True,lock=False,readahead=False)
  counts=Counter();raw=Counter();accepted=Counter();examples=[];seen=set()
  with env.begin() as txn:
   for key,value in txn.cursor():
    counts['records']+=1
    try:
     record=read_record(value);smiles=canonical_record(record);m=Chem.MolFromSmiles(smiles)
     present={atom.GetSymbol() for atom in m.GetAtoms()};raw.update(present)
     if Chem.GetFormalCharge(m) or any(atom.GetNumRadicalElectrons() for atom in m.GetAtoms()):counts['charged_or_radical']+=1;continue
     inst=adapt_record(record,max_atoms=180,element_table=elements,spectrum_size=450)
     if smiles in seen:counts['duplicate']+=1;continue
     seen.add(smiles);accepted.update(present);counts['accepted']+=1
     if present-{'H','C','N','O','F','B'}:
      examples.append({'key':key.decode(),'smiles':smiles,'elements':sorted(present),'atoms':len(inst.coords)})
    except (ValueError,RuntimeError,AssertionError) as exc:counts[str(exc).split(':')[0]]+=1
  with source.open('rb') as stream:source_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
  report[f'{group}/{split}']={'source_sha256':source_sha256,'criteria':{'max_atoms':180,'spectrum_channels':450,'charge':0,'radical_atoms':0},'counts':dict(counts),'raw_element_records':dict(raw),'accepted_unique_element_records':dict(accepted),'wide_examples':examples}
  print(group,split,counts,accepted,flush=True);env.close()
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
