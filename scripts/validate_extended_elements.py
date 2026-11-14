"""Verify explicit element identities and real DPA forces for new organic elements."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from loom.generator.molecular_state import rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol
from loom.physics.dpa_forcefield import DPAPotentialGuidance
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run, record_input_hashes
from loom.utils.reproducibility import lock_deterministic_environment

p = argparse.ArgumentParser()
p.add_argument("--weight", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
freeze_run(a.output, __file__)
record_input_hashes(a.output, {"forcefield": a.weight})
lock_deterministic_environment(42)
torch.set_num_threads(2)
elements = ["H", "C", "O", "B", "Si", "Se", "As", "Te"]
ff = DPAPotentialGuidance(a.weight, atom_type_map=elements, device="cuda:0")
rows = []
molecules = []
for smiles in ["B(O)O", "C[Si](C)(C)C", "C[Se]C", "C[As](C)C", "C[Te]C"]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    instance = rdkit_mol_to_mol3d_instance(mol, np.empty(0), element_table=tuple(reversed(elements)))
    checked = mol3d_to_validated_rdkit_mol(instance)
    assert instance.atomic_numbers.tolist() == [a.GetAtomicNum() for a in checked.GetAtoms()]
    coords = torch.tensor(instance.coords, device="cuda:0")
    indices = torch.tensor([ff.atom_type_map.index(s) for s in instance.symbols], device="cuda:0")
    e, f = ff.compute_energy_and_forces(coords, indices, clip_forces=False, zero_com_forces=False)
    coordinate = int(f.abs().argmax())
    h = 0.002
    plus, minus = coords.clone(), coords.clone()
    plus.reshape(-1)[coordinate] += h
    minus.reshape(-1)[coordinate] -= h
    ep, _ = ff.compute_energy_and_forces(plus, indices, clip_forces=False, zero_com_forces=False)
    em, _ = ff.compute_energy_and_forces(minus, indices, clip_forces=False, zero_com_forces=False)
    fd = float(-(ep.sum() - em.sum()) / (2 * h))
    analytic = float(f.reshape(-1)[coordinate])
    assert abs(fd - analytic) < max(0.05, 0.05 * abs(analytic))
    ff.set_atom_type_map(list(reversed(elements)))
    other = torch.tensor([ff.atom_type_map.index(s) for s in instance.symbols], device="cuda:0")
    e2, f2 = ff.compute_energy_and_forces(coords, other, clip_forces=False, zero_com_forces=False)
    torch.testing.assert_close(e, e2, atol=1e-6, rtol=0)
    torch.testing.assert_close(f, f2, atol=1e-5, rtol=0)
    ff.set_atom_type_map(elements)
    rows.append(
        {
            "smiles": smiles,
            "atomic_numbers": instance.atomic_numbers.tolist(),
            "energy_eV": float(e.sum()),
            "analytic_force_eV_A": analytic,
            "finite_difference_force_eV_A": fd,
            "type_order_energy_error_eV": float(abs(e - e2).max()),
            "type_order_force_error_eV_A": float(abs(f - f2).max()),
        }
    )
    molecules.append(checked)
report = {
    "passed": True,
    "force_repeat_tolerance_eV_A": 1e-5,
    "records": rows,
    "scope": "Element identity and finite DPA force execution; not broad-element generator training or property accuracy.",
}
write_precise_sdf(a.output / "structures.sdf", molecules)
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
