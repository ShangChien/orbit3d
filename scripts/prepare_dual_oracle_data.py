"""Audit actual selected OE62 SDFs and freeze a shared supported split, without geometry changes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from loom.oracles.uniqsar_oracle import UNIMOL_SYM2IDX
from loom.utils.dataset_io import write_precise_sdf

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);original=json.loads((a.data/'manifest.json').read_text());report={'source_manifest_sha256':hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),'theory':original['theory'],'units':'eV','hydrogens':'All explicit source hydrogens, no embedding or atom addition','max_atoms':80,'splits':{},'overlaps':{},'excluded':[],'locked_test':'Original scaffold-held-out test, IDs and geometry frozen before fitting; labels not used for selection','generator_overlap_scope':'Known SFT training identities only; pretraining corpus overlap unknown'}
sets={};source_train=set()
for split in ['train','valid','calibration','test']:
 ms=list(Chem.SDMolSupplier(str(a.data/f'{split}.sdf'),removeHs=False));assert all(m is not None for m in ms)
 assert [m.GetProp('SourceID') for m in ms]==original['selection_ids'][split]
 ids=[];mols=[];identities=set();scaffolds=set();elements=set();atoms=[]
 for m in ms:
  identity=Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=False)
  if split=='train':source_train.add(identity)
  unsupported=sorted({x.GetSymbol() for x in m.GetAtoms()}-set(UNIMOL_SYM2IDX))
  if unsupported:report['excluded'].append({'split':split,'source_id':m.GetProp('SourceID'),'elements':unsupported});continue
  assert m.GetNumAtoms()<=80 and Chem.GetFormalCharge(m)==0 and not any(x.GetNumRadicalElectrons() for x in m.GetAtoms())
  assert m.GetProp('LabelTheory')==original['theory'] and m.GetProp('TargetUnits')=='eV'
  mols.append(m);ids.append(m.GetProp('SourceID'));identities.add(identity)
  scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(m),includeChirality=False) or '<ACYCLIC>')
  elements.update(x.GetSymbol() for x in m.GetAtoms());atoms.append(m.GetNumAtoms())
 write_precise_sdf(a.output/f'{split}.sdf',mols)
 loaded=list(Chem.SDMolSupplier(str(a.output/f'{split}.sdf'),removeHs=False))
 for before,after in zip(mols,loaded):np.testing.assert_allclose(before.GetConformer().GetPositions(),after.GetConformer().GetPositions(),rtol=0,atol=1e-10)
 sets[split]=(identities,scaffolds);report['splits'][split]={'source_count':len(ms),'count':len(mols),'ids':ids,'unique_constitutions':len(identities),'scaffolds':len(scaffolds),'elements':sorted(elements),'atom_quantiles':np.quantile(atoms,[0,.25,.5,.75,1]).tolist(),'sha256':hashlib.sha256((a.output/f'{split}.sdf').read_bytes()).hexdigest()}
for i,left in enumerate(sets):
 for right in list(sets)[i+1:]:
  overlap={'identity':len(sets[left][0]&sets[right][0]),'scaffold':len(sets[left][1]&sets[right][1])};report['overlaps'][left+'/'+right]=overlap;assert not any(overlap.values())
report['known_generator_train_overlap']={k:len(v[0]&source_train) for k,v in sets.items()}
(a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:{f:v for f,v in d.items() if f!='ids'} for k,d in report['splits'].items()},flush=True)
