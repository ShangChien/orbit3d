"""Replay native coordinates through the free-generation graph recovery path."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rdkit import Chem
from loom.generator.molecular_state import Mol3DInstance, mol3d_to_validated_rdkit_mol
from loom.vendor.uni3dar.data.base_grid_dataset import BaseGridDataset
from loom.oracles.graph_features import fingerprint_graph


def graph_key(mol):
    return Chem.MolToSmiles(fingerprint_graph(mol), isomericSmiles=False)


def charge_states(mol):
    return {f'{a.GetSymbol()}:{a.GetFormalCharge():+}' for a in fingerprint_graph(mol).GetAtoms()}


def quantize(coords):
    # Inference-time encoder: mean centering plus half a 0.24 A leaf, followed
    # by the actual cell/fine-coordinate conversion. Subsequent whole-cell
    # translations and atom reordering cannot change pairwise geometry.
    grid = SimpleNamespace(grid_len=.24, xyz_resolution=.01, num_xyz=24)
    pos, fine = BaseGridDataset.phy_pos_to_grid_pos(grid, coords-coords.mean(0)+.12)
    decoded = BaseGridDataset.grid_pos_to_phy_pos(grid, pos, fine)
    return (decoded-decoded.mean(0)).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    known=set(json.loads(a.support.read_text())['training_state_counts']);rows=[]
    for index,mol in enumerate(Chem.SDMolSupplier(str(a.source),removeHs=False)):
        symbols=[at.GetSymbol() for at in mol.GetAtoms()];elements=tuple(sorted(set(symbols)));types=np.array([elements.index(s) for s in symbols]);coords=mol.GetConformer().GetPositions();row={'index':index,'source_smiles':graph_key(mol),'source_states':sorted(charge_states(mol)),'atoms':len(types),'cases':{}}
        for name,positions,graph in [('supplied_graph',(coords-coords.mean(0)).astype(np.float32),mol),('coordinates_only',(coords-coords.mean(0)).astype(np.float32),None),('quantized_coordinates_only',quantize(coords),None)]:
            inst=Mol3DInstance(coords=positions,atom_types=types,element_table=elements,graph=graph,charge=Chem.GetFormalCharge(mol),multiplicity=1)
            try:
                recovered=mol3d_to_validated_rdkit_mol(inst);key=graph_key(recovered)
                row['cases'][name]=dict(accepted=True,normalized_constitution_preserved=key==row['source_smiles'],recovered_smiles=key,unseen_states=sorted(charge_states(recovered)-known))
            except (ValueError,RuntimeError) as error:row['cases'][name]=dict(accepted=False,error=str(error))
        rows.append(row)
        if (index+1)%64==0:print('Completed',index+1,flush=True)
    summary={}
    for name in ['supplied_graph','coordinates_only','quantized_coordinates_only']:
        values=[r['cases'][name] for r in rows];accepted=[r for r in values if r['accepted']]
        summary[name]=dict(attempted=len(rows),accepted=len(accepted),normalized_constitution_preserved=sum(r['normalized_constitution_preserved'] for r in accepted),accepted_unseen_states=sum(bool(r['unseen_states']) for r in accepted),failure_reasons=dict(Counter(r['error'].split(':')[0] for r in values if not r['accepted'])))
    report=dict(scope='Post hoc source-coordinate replay after E14 charge-shift observation. All structures in the fixed 512-molecule native valid SDF; no generated-pool outcomes selected samples. Neutral closed-shell native graphs compared with coordinate-only free-proposal recovery and actual 0.01 A grid conversion. Constitution uses normalized graphs without stereo; geometric admission still applies normal stereo checks when a graph is supplied. No training, filtering, source labels or current experiments changed.',summary=summary,records=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.source,a.support]})
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
