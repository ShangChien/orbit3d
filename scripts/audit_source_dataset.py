"""Check selected source geometries, labels, and full-molecule octree encoding."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from orbit3d.generator.octree_backend import OctreeDataset
from orbit3d.generator.octree_model import AtomFeatDict, build_default_uni3dar_args
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);freeze_run(a.output,__file__)
    record_input_hashes(a.output,{'source':a.source,'manifest':a.data/'manifest.json'})
    source=pd.read_json(a.source,orient='split').set_index('refcode_csd')
    manifest=json.loads((a.data/'manifest.json').read_text())
    args=build_default_uni3dar_args(1);args.merge_level=7;args.allow_atoms=','.join(manifest['elements'])
    model=SimpleNamespace(args=args,dictionary=AtomFeatDict(args));report={}
    for split in ['train','valid','calibration']:
        mols=list(Chem.SDMolSupplier(str(a.data/f'{split}.sdf'),removeHs=False))
        assert all(m is not None for m in mols)
        assert [m.GetProp('SourceID') for m in mols]==manifest['selection_ids'][split]
        dataset=OctreeDataset(mols,model,training=True,seed=42)
        max_error=0.;elements=Counter();histogram=Counter();tokens=[]
        for i,mol in enumerate(mols):
            row=source.loc[mol.GetProp('SourceID')]
            original=Chem.MolFromXYZBlock(row.xyz_pbe_relaxed)
            assert [x.GetSymbol() for x in original.GetAtoms()]==[x.GetSymbol() for x in mol.GetAtoms()]
            delta=float(np.max(np.abs(original.GetConformer().GetPositions()-mol.GetConformer().GetPositions())))
            max_error=max(max_error,delta);assert delta<=1e-10
            assert float(mol.GetProp('homo'))==row.energies_occ_pbe0_vac_tier2[-1]
            assert float(mol.GetProp('lumo'))==row.energies_unocc_pbe0_vac_tier2[0]
            for epoch in [1,2]:
                dataset.grid.set_epoch(epoch)
                packed=dataset.collate([i]);tokens.append(packed['decoder_level'].numel())
            elements.update({x.GetSymbol() for x in mol.GetAtoms()})
            histogram[min((mol.GetNumAtoms()-1)//10*10+10,80)]+=1
        report[split]={'records':len(mols),'source_coordinate_max_abs_error_A':max_error,
                       'source_label_exact':True,'octree_epochs_checked':[1,2],
                       'atom_quantiles':np.quantile([m.GetNumAtoms() for m in mols],[0,.25,.5,.75,1]).tolist(),
                       'heavy_atom_quantiles':np.quantile([m.GetNumHeavyAtoms() for m in mols],[0,.25,.5,.75,1]).tolist(),
                       'atom_bins_upper_bound':dict(sorted(histogram.items())),
                       'molecules_containing_element':dict(elements),'max_decoder_tokens':max(tokens)}
        print(split,report[split],flush=True)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
