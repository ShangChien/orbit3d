"""Reuse audited property conformers with atom-conserving octree checks."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rdkit import Chem
from loom.generator.octree_model import build_default_uni3dar_args,AtomFeatDict
from loom.generator.octree_backend import OctreeDataset
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run,record_input_hashes
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--elements',nargs='+',required=True)
p.add_argument('--merge-level',type=int,choices=[6,7,8],default=7)
a=p.parse_args();freeze_run(a.output,__file__);record_input_hashes(a.output,{'source_structures':a.source});torch.set_num_threads(4)
args=build_default_uni3dar_args(1);args.merge_level=a.merge_level;args.allow_atoms=','.join(a.elements)
representation=SimpleNamespace(args=args,dictionary=AtomFeatDict(args))
molecules=list(Chem.SDMolSupplier(str(a.source),removeHs=False));assert all(m is not None for m in molecules)
dataset=OctreeDataset(molecules,representation,training=False)
accepted=[];records=[]
for index,mol in enumerate(molecules):
 row={'source_id':mol.GetProp('SourceID'),'source_index':index}
 try:
  xyz=mol.GetConformer().GetPositions();diameter=float(np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1).max())
  if diameter>args.grid_len*2**args.merge_level-.56:raise ValueError('outside rotation-safe spatial span')
  packed=dataset.collate([index])
  row.update(status='accepted',atoms=mol.GetNumAtoms(),diameter_A=diameter,tokens=int(packed['decoder_level'].numel()))
  accepted.append(mol)
 except (ValueError,RuntimeError,AssertionError) as error:row.update(status='excluded',reason=str(error))
 records.append(row)
write_precise_sdf(a.output/'train.sdf',accepted)
report={'input_records':len(molecules),'accepted':len(accepted),'excluded':len(molecules)-len(accepted),'merge_level':a.merge_level,'allowed_elements':a.elements,'observed_elements':sorted({atom.GetSymbol() for mol in accepted for atom in mol.GetAtoms()}),'records':records,'geometry':'Copied from the audited source SDF without re-embedding, changing labels or changing split membership'}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='records'})
assert accepted and set(a.elements)<=set(report['observed_elements'])
