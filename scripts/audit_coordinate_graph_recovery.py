"""Audit reconstruction on trusted geometries before interpreting generated graphs."""
import argparse,json
from collections import Counter
from dataclasses import replace
from pathlib import Path
import numpy as np
from rdkit import Chem,RDLogger
from orbit3d.generator.octree_tokenizer import rdkit_mol_to_mol3d_instance,mol3d_to_validated_rdkit_mol
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
from orbit3d.utils.molecular_identity import compare_molecular_identity
p=argparse.ArgumentParser()
p.add_argument('--sdf',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();freeze_run(a.output,__file__);record_input_hashes(a.output,{'structures':a.sdf})
RDLogger.DisableLog('rdApp.*');records=[]
for i,molecule in enumerate(Chem.SDMolSupplier(str(a.sdf),removeHs=False)):
 assert molecule is not None
 original=Chem.MolToSmiles(Chem.RemoveHs(molecule),isomericSmiles=True)
 elements=tuple(dict.fromkeys(atom.GetSymbol() for atom in molecule.GetAtoms()))
 inst=rdkit_mol_to_mol3d_instance(molecule,np.empty(0),center=False,element_table=elements)
 row={'index':i,'source_id':molecule.GetProp('SourceID') if molecule.HasProp('SourceID') else None,'source_smiles':original,'source_charged_atoms':sum(atom.GetFormalCharge()!=0 for atom in molecule.GetAtoms()),'elements':list(elements)}
 for name,coords in [('exact_input_coordinates',inst.coords),('rounded_0.01A',np.rint(inst.coords/.01)*.01)]:
  try:
   recovered=mol3d_to_validated_rdkit_mol(replace(inst,graph=None,coords=coords))
   smiles=Chem.MolToSmiles(Chem.RemoveHs(recovered),isomericSmiles=True)
   row[name]={'valid':True,**compare_molecular_identity(molecule,recovered),'recovered_smiles':smiles,'charged_atoms':sum(atom.GetFormalCharge()!=0 for atom in recovered.GetAtoms())}
  except ValueError as exc:row[name]={'valid':False,'identity_match':False,'reason':str(exc)}
 records.append(row)
summary={'records':len(records),'source_charge_count_histogram':dict(Counter(row['source_charged_atoms'] for row in records)), 'reconstruction':{name:{'valid':sum(row[name]['valid'] for row in records),'identity_matches':sum(row[name]['identity_match'] for row in records),'identity_status_counts':dict(Counter(row[name].get('identity_status','invalid') for row in records))} for name in ['exact_input_coordinates','rounded_0.01A']},'scope':'Known input geometries; rounded coordinates are a sensitivity test, not the full octree transform'}
(a.output/'records.json').write_text(json.dumps(records,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
