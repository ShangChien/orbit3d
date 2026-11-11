"""Energy-only, fixed-graph validation of every generated molecule in an element cohort."""

import argparse
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from orbit3d.generator.octree_tokenizer import rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol
from orbit3d.physics.dpa_forcefield import DPAPotentialGuidance
from orbit3d.physics.conformer_relaxer import relax_conformer_energy_only
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment
import torch

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--weight", required=True)
p.add_argument("--element", default="B")
p.add_argument("--steps", type=int, default=50)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
freeze_run(a.output, __file__)
record_input_hashes(
    a.output,
    {
        "forcefield": a.weight,
        "generated_sdf": a.run / "generated.sdf",
        "generator_summary": a.run / "summary.json",
    },
)
lock_deterministic_environment(42)
torch.set_num_threads(2)
source = json.loads((a.run / "summary.json").read_text())
all_mols = list(Chem.SDMolSupplier(str(a.run / "generated.sdf"), removeHs=False))
assert all(m is not None for m in all_mols)
mols = [m for m in all_mols if any(atom.GetSymbol() == a.element for atom in m.GetAtoms())]
elements = sorted({atom.GetSymbol() for mol in mols for atom in mol.GetAtoms()})
if not mols:
    raise ValueError("The declared generated-element cohort is empty")
ff = DPAPotentialGuidance(a.weight, atom_type_map=elements, device="cuda:0")
records = []
exports = []
for mol in mols:
    row = {"candidate_id": int(mol.GetProp("CandidateID")), "atoms": mol.GetNumAtoms(), "status": "started"}
    try:
        inst = rdkit_mol_to_mol3d_instance(mol, np.empty(0), center=False, element_table=tuple(elements))
        refined, info = relax_conformer_energy_only(inst, ff, n_steps=a.steps, step_size=0.004)
        final = mol3d_to_validated_rdkit_mol(refined)
        final.SetIntProp("CandidateID", row["candidate_id"])
        row.update(info)
        row.update(status="completed", smiles=refined.smiles, refinement_mode=info["status"])
        exports.append(final)
    except (ValueError, FloatingPointError) as error:
        row.update(status="failed", reason=str(error))
    records.append(row)
    (a.output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in row.items()
            if k in ["candidate_id", "status", "init_force_eva", "final_force_eva", "reason"]
        },
        flush=True,
    )
write_precise_sdf(a.output / "relaxed.sdf", exports)
report = {
    "generator_checkpoint_sha256": source["checkpoint_sha256"],
    "all_generator_attempts": source["attempted"],
    "all_raw_valid": len(all_mols),
    "selected_element": a.element,
    "cohort_size": len(mols),
    "selection_rule": "Every chemically valid generated molecule containing the specified element; no property/force preselection",
    "completed": len(exports),
    "steps": a.steps,
    "records": records,
    "scope": "Real MLIP energy-only fixed-graph relaxation. No property Oracle, target-hit, quantum accuracy or synthesis claim.",
}
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
