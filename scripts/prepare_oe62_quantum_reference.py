"""Freeze source-coordinate protocol references from development, never the locked test."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rdkit import Chem
from orbit3d.utils.dataset_io import write_precise_sdf
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'quantum_reference';out.mkdir(exist_ok=False)
cohort=json.loads((a.root/'quantum_cohort/manifest.json').read_text());allowed={s for r in cohort['records'] for s in r['elements']}
train=list(Chem.SDMolSupplier(str(a.root/'data/train.sdf'),removeHs=False));valid=list(Chem.SDMolSupplier(str(a.root/'data/valid.sdf'),removeHs=False));valid=[m for m in valid if not {x.GetSymbol() for x in m.GetAtoms()}-allowed]
selected=[];seen=set();rules=[]
for name in ['homo','lumo']:
 for q in [.25,.5,.75]:
  t=float(np.quantile([float(m.GetProp(name)) for m in train],q))
  for lo,hi in [(10,30),(31,55)]:
   eligible=[m for m in valid if lo<=m.GetNumAtoms()<=hi and m.GetProp('SourceID') not in seen]
   m=min(eligible,key=lambda m:(abs(float(m.GetProp(name))-t),hashlib.sha256(m.GetProp('SourceID').encode()).hexdigest()))
   selected.append(m);seen.add(m.GetProp('SourceID'));rules.append({'property':name,'quantile':q,'atom_bin':[lo,hi]})
for e in ['B','P']:
 if e not in allowed or any(any(x.GetSymbol()==e for x in m.GetAtoms()) for m in selected):continue
 eligible=[m for m in valid if any(x.GetSymbol()==e for x in m.GetAtoms()) and m.GetProp('SourceID') not in seen]
 if eligible:
  m=min(eligible,key=lambda m:(m.GetNumAtoms(),m.GetProp('SourceID')));selected.append(m);seen.add(m.GetProp('SourceID'));rules.append({'diagnostic_element':e})
records=[]
for i,(m,rule) in enumerate(zip(selected,rules)):
 m.SetProp('ReferenceID',f'r{i:02d}');records.append({'id':f'r{i:02d}','source_id':m.GetProp('SourceID'),'atoms':m.GetNumAtoms(),'elements':sorted({x.GetSymbol() for x in m.GetAtoms()}),'source_labels':[float(m.GetProp(k)) for k in ['homo','lumo']],'rule':rule})
write_precise_sdf(out/'source.sdf',selected)
manifest={'source':'Common development source 3D coordinates, unchanged; no locked-test labels used','selection':'12 property/size stratified references plus missing B/P diagnostics to cover the frozen cohort; not a random full-OE62 accuracy estimate','records':records,'sdf_sha256':hashlib.sha256((out/'source.sdf').read_bytes()).hexdigest(),
 'protocol':{'engine':'GPU4PySCF 1.9.1 / PySCF 2.12.1','single_point':'PBE0/def2-TZVP, density fitting def2-universal-jkfit','grid_level':3,'scf_tolerance':1e-8,'charge':0,'multiplicity':1,'geometry':'unchanged OE62 PBE+TS source coordinates','scope':'Approximate basis/protocol comparison, not FHI-aims Tier2 equivalence'}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(records,flush=True)
