"""Small real-DPA integration experiment; no surrogate property accuracy claim."""

import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from orbit3d.generator.octree_tokenizer import (
    rdkit_mol_to_mol3d_instance,
    ATOM_TYPES,
    mol3d_to_validated_rdkit_mol,
)
from orbit3d.physics.dpa_forcefield import DPAPotentialGuidance
from orbit3d.physics.conformer_relaxer import relax_conformer_iso_property
from orbit3d.oracles.base import PropertyOracleBase


class DistanceOracle(PropertyOracleBase):
    target_cols = ["bond_length"]

    def predict_3d_mols(self, mols):
        return np.array(
            [
                np.linalg.norm(m.GetConformer().GetPositions()[1] - m.GetConformer().GetPositions()[0])
                for m in mols
            ]
        ), np.zeros(len(mols))

    def forward_differentiable_3d(self, x, *args, **kw):
        return (x[1] - x[0]).norm(), torch.zeros(1, device=x.device)


p = argparse.ArgumentParser()
p.add_argument("--weight", required=True)
p.add_argument("--device", default="cpu")
p.add_argument("--output", required=True)
args = p.parse_args()
torch.set_num_threads(2)
started = time.perf_counter()
ff = DPAPotentialGuidance(args.weight, atom_type_map=ATOM_TYPES, device=args.device)
mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
AllChem.EmbedMolecule(mol, randomSeed=42)
inst = rdkit_mol_to_mol3d_instance(mol, np.array([0.0]))
d = float(np.linalg.norm(inst.coords[1] - inst.coords[0]))
interval = (d - 0.005, d + 0.005)
out, info = relax_conformer_iso_property(inst, ff, DistanceOracle(), interval, n_steps=5)
assert interval[0] <= info["y_pred"] <= interval[1]
assert Chem.MolToSmiles(mol3d_to_validated_rdkit_mol(out)) == Chem.MolToSmiles(mol)
assert all(np.isfinite([info["init_force_eva"], info["final_force_eva"]]))
import orbit3d

root = Path(orbit3d.__file__).parent
source_hashes = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.py")
}
report = {
    "source_sha256": source_hashes,
    "torch_version": torch.__version__,
    "experiment": "real_dpa_ethane_constrained_relaxation",
    "device": args.device,
    "weight_sha256": hashlib.sha256(Path(args.weight).read_bytes()).hexdigest(),
    "interval": interval,
    "elapsed_seconds": time.perf_counter() - started,
    "result": info,
}
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
