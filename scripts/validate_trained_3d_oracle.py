"""Compare true prediction, differentiable prediction, finite-difference gradient and refinement."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.octree_tokenizer import ATOM_TYPES, rdkit_mol_to_mol3d_instance
from orbit3d.physics.dpa_forcefield import DPAPotentialGuidance
from orbit3d.physics.conformer_relaxer import relax_conformer_iso_property
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle

p = argparse.ArgumentParser()
p.add_argument("--oracle", required=True)
p.add_argument("--molecules", required=True)
p.add_argument("--weight", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
torch.set_num_threads(2)
ff = DPAPotentialGuidance(a.weight, atom_type_map=ATOM_TYPES, device="cpu")
oracle = InternalDPA4PropertyOracle.load_from_dir(a.oracle, ff)
mols = [m for m in Chem.SDMolSupplier(a.molecules, removeHs=False) if m is not None][:4]
rows = []
for mol in mols:
    x = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, requires_grad=True)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    pred, std = oracle.forward_differentiable_3d(x, atoms)
    plain, _ = oracle.predict_3d_mols([mol])
    np.testing.assert_allclose(pred.detach().numpy(), plain[0], atol=1e-5, rtol=1e-5)
    gradient = torch.autograd.grad(pred.sum(), x)[0]
    assert torch.isfinite(gradient).all()
    idx = int(gradient.abs().argmax())
    h = 0.002
    plus = x.detach().clone()
    minus = x.detach().clone()
    plus.reshape(-1)[idx] += h
    minus.reshape(-1)[idx] -= h
    yp, _ = oracle.forward_differentiable_3d(plus, atoms)
    ym, _ = oracle.forward_differentiable_3d(minus, atoms)
    fd = float((yp.sum() - ym.sum()) / (2 * h))
    auto = float(gradient.reshape(-1)[idx])
    assert abs(fd - auto) <= max(0.01, abs(auto) * 0.1), (auto, fd)
    intervals = {
        name: (float(pred[i]) - 0.03, float(pred[i]) + 0.03) for i, name in enumerate(oracle.target_cols)
    }
    inst = rdkit_mol_to_mol3d_instance(mol, plain[0])
    out, info = relax_conformer_iso_property(inst, ff, oracle, intervals, n_steps=5, step_size=0.004)
    assert info["interval_hit"]
    rows.append(
        {
            "id": mol.GetProp("SourceID"),
            "prediction_path_max_difference": float(np.abs(pred.detach().numpy() - plain[0]).max()),
            "gradient": auto,
            "finite_difference": fd,
            "relaxation": info,
        }
    )
Path(a.output).write_text(json.dumps({"records": rows}, indent=2) + "\n")
print(json.dumps({"records": rows}, indent=2))
