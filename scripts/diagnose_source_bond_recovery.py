"""Test bond recovery on known source geometry and exact encoded octree leaves."""
import argparse,json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rdkit import Chem,RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from orbit3d.generator.octree_backend import OctreeDataset
from orbit3d.generator.octree_model import AtomFeatDict,build_default_uni3dar_args
from orbit3d.generator.molecular_state import Mol3DInstance,rdkit_mol_to_mol3d_instance,mol3d_to_validated_rdkit_mol
from orbit3d.vendor.uni3dar.data.atom_data import atom_list
from orbit3d.utils.molecular_identity import compare_molecular_identity
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--sdf',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();freeze_run(a.output,__file__);record_input_hashes(a.output,{'sdf':a.sdf,'manifest':a.manifest})
    torch.set_num_threads(2);RDLogger.DisableLog('rdApp.*')
    manifest=json.loads(a.manifest.read_text());elements=tuple(manifest['elements'])
    mols=list(Chem.SDMolSupplier(str(a.sdf),removeHs=False));assert all(m is not None for m in mols)
    args=build_default_uni3dar_args(1);args.merge_level=7;args.allow_atoms=','.join(elements)
    model=SimpleNamespace(args=args,dictionary=AtomFeatDict(args));dataset=OctreeDataset(mols,model,seed=42)
    rows=[]
    for i,source in enumerate(mols):
        source_id=source.GetProp('SourceID');print('checking',i,source_id,flush=True)
        normalized_source=Chem.MolToSmiles(rdMolStandardize.Normalize(Chem.RemoveHs(source)))
        original=rdkit_mol_to_mol3d_instance(source,np.empty(0),center=False,element_table=elements,coordinate_dtype=np.float64)
        variants={'source_xyz':replace(original,graph=None)}
        for epoch in [1,2]:
            dataset.grid.set_epoch(epoch);packed=dataset.collate([i])
            selected=(packed['decoder_level']==0)&packed['decoder_is_second_atom']
            symbols=[atom_list[k] for k in packed['decoder_type'][selected].tolist()]
            variants[f'encoded_epoch{epoch}']=Mol3DInstance(coords=packed['decoder_phy_pos'][selected].numpy(),
                atom_types=np.array([elements.index(s) for s in symbols]),element_table=elements)
        for name,instance in variants.items():
            row={'source_id':source_id,'geometry':name,'valid':False}
            try:
                parsed=mol3d_to_validated_rdkit_mol(instance)
                row.update(valid=True,smiles=Chem.MolToSmiles(Chem.RemoveHs(parsed)),
                           charged_atoms=sum(atom.GetFormalCharge()!=0 for atom in parsed.GetAtoms()),
                           normalized_graph_match=normalized_source==Chem.MolToSmiles(rdMolStandardize.Normalize(Chem.RemoveHs(parsed))),
                           **compare_molecular_identity(source,parsed))
            except ValueError as error:row['reason']=str(error)
            rows.append(row)
        (a.output/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    aggregate={}
    for name in ['source_xyz','encoded_epoch1','encoded_epoch2']:
        group=[r for r in rows if r['geometry']==name]
        aggregate[name]={'attempted':len(group),'valid':sum(r['valid'] for r in group),
            'identity_status':dict(Counter(r.get('identity_status','invalid') for r in group)),
            'normalized_graph_matches':sum(r.get('normalized_graph_match',False) for r in group),
            'with_formal_charges':sum(r.get('charged_atoms',0)>0 for r in group)}
    (a.output/'summary.json').write_text(json.dumps({'scope':'Fixed validation IDs; source geometry vs exact depth-7 octree encoding; no model sampling or graph replacement',
        'aggregate':aggregate,'records':rows},indent=2)+'\n')
    print(aggregate,flush=True)


if __name__=='__main__':main()
