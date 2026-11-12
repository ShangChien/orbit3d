"""Save fixed-geometry property/energy/force outputs for cross-runtime comparison."""
import argparse
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.molecular_state import ATOM_TYPES
from orbit3d.physics.dpa_forcefield import DPAPotentialGuidance
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.utils.run_evidence import freeze_run
from orbit3d.utils.reproducibility import lock_deterministic_environment
p=argparse.ArgumentParser()
p.add_argument('--sdf', required=True)
p.add_argument('--oracle', required=True)
p.add_argument('--weight', required=True)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
freeze_run(a.output, __file__)
lock_deterministic_environment(42)
torch.set_num_threads(2)
ff=DPAPotentialGuidance(a.weight, atom_type_map=ATOM_TYPES, device='cuda:0')
prop=InternalDPA4PropertyOracle.load_from_dir(a.oracle, ff)
mols=list(Chem.SDMolSupplier(a.sdf, removeHs=False))[:32]
r=prop.evaluate_3d_mols_batched(mols)
assert not any(r['errors'])
np.savez(a.output/'values.npz', mean=r['mean'], std=r['std'], energy=r['energy_eV'],
         **{f'force_{i}':f for i,f in enumerate(r['forces_eV_A'])})
