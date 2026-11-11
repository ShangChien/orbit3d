"""Real Oracle/force-field check of explicit scaffold graph-growth proposals."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from orbit3d.generator.scaffold import ScaffoldState
from orbit3d.generator.octree_tokenizer import ATOM_TYPES, rdkit_mol_to_mol3d_instance
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.oracles.applicability import MorganDomain
from orbit3d.physics.dpa_forcefield import DPAPotentialGuidance
from orbit3d.physics.batched_relaxer import relax_conformers_iso_property
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes

p = argparse.ArgumentParser()
p.add_argument("--weight", required=True)
p.add_argument("--oracle", required=True)
p.add_argument("--domain", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()
freeze_run(a.output, __file__)
record_input_hashes(a.output, {"forcefield": a.weight, "oracle": a.oracle, "domain": a.domain})
torch.set_num_threads(2)
rng = np.random.default_rng(a.seed)
fragments = ["C", "N", "O", "CC", "C#N", "C=O"]
scaffolds = [("benzene", "c1ccccc1", 0), ("naphthalene", "c1ccc2ccccc2c1", 0), ("pyridine", "c1ccncc1", 0)]
records = []
instances = []
frozen = []
states = []
owners = []
for name, smiles, site in scaffolds:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=42):
        raise RuntimeError("Scaffold embedding failed")
    AllChem.MMFFOptimizeMolecule(mol)
    mol.GetConformer().SetPositions(mol.GetConformer().GetPositions() + [10, 20, 30])
    source = ScaffoldState.from_molecule(mol, [site])
    for fragment in fragments:
        row = {"candidate_id": len(records), "scaffold": name, "fragment": fragment, "status": "proposed"}
        try:
            state = source.attach(
                fragment, next(iter(source.ports)), seed=a.seed, torsion=float(rng.uniform(-np.pi, np.pi))
            )
            instance = rdkit_mol_to_mol3d_instance(state.molecule, np.zeros(2), center=False)
            ids = {atom.GetAtomMapNum(): atom.GetIdx() for atom in state.molecule.GetAtoms()}
            instances.append(instance)
            frozen.append([ids[k] for k in state.fixed_positions])
            states.append(state)
            owners.append(len(records))
            row.update(status="chemically_valid", actions=state.actions)
        except ValueError as error:
            row.update(status="generation_failed", reason=str(error))
        records.append(row)
ff = DPAPotentialGuidance(a.weight, atom_type_map=ATOM_TYPES, device="cuda:0")
oracle = InternalDPA4PropertyOracle.load_from_dir(a.oracle, ff)
domain = MorganDomain.load(a.domain)
# Diagnostic evaluates every valid proposal and labels OOD explicitly; no
# out-of-domain prediction is counted as a qualified design.
results = relax_conformers_iso_property(
    instances, ff, oracle, {"homo": [-8.2, -7.2], "lumo": [-0.6, 0.8]}, n_steps=15, frozen_indices=frozen
)
exported = []
traces = []
for index, instance, state, indices, (refined, info) in zip(owners, instances, states, frozen, results):
    row = records[index]
    if refined is None:
        row.update(status="refinement_failed", reason=info["reason"])
        continue
    delta = float(abs(refined.coords[indices] - instance.coords[indices]).max())
    assert delta == 0.0, delta
    reference = np.array(
        [
            state.fixed_positions[atom.GetAtomMapNum()]
            for atom in state.molecule.GetAtoms()
            if atom.GetAtomMapNum() in state.fixed_positions
        ]
    )
    total_delta = float(abs(refined.coords[indices] - reference).max())
    assert total_delta < 2e-6, total_delta
    assessment = domain.assess(refined.graph)
    similarity = assessment['similarity']
    row.update(
        status="refined",
        frozen_coordinate_shift_A=delta,
        world_frame_error_A=total_delta,
        domain_similarity=similarity,
        domain_eligible=assessment['in_domain'],
        unseen_oracle_training_elements=assessment['unseen_training_elements'],
        y_pred=info["y_pred"],
        final_rms_force_eV_A=info["final_force_eva"],
        joint_success=bool(
            assessment['in_domain'] and info["interval_hit"] and info["final_force_eva"] <= 0.5
        ),
    )
    out = Chem.Mol(refined.graph)
    out.GetConformer().SetPositions(refined.coords.astype(np.float64))
    out.SetProp("CandidateID", str(index))
    out.SetProp("ScaffoldID", row["scaffold"])
    out.SetProp("AttachmentActions", json.dumps(row["actions"]))
    out.SetProp("FixedAtomMaps", json.dumps(list(state.fixed_positions)))
    exported.append(out)
    traces.append({"candidate_id": index, **info})
report = {
    "backend": "explicit_scaffold_graph_growth",
    "attempted": len(records),
    "chemically_valid": len(instances),
    "refined": len(exported),
    "joint_successes": sum(r.get("joint_success", False) for r in records),
    "domain_cutoff": domain.cutoff,
    "records": records,
    "scope": "Finite fragment menu with explicit graph actions; not learned octree scaffold completion or a synthesis route.",
    "coordinate_precision": "Exact graph-growth scaffold coordinates; MLIP float32 conversion <2e-6 A; fixed coordinates bitwise unchanged during refinement.",
}
write_precise_sdf(a.output / "generated.sdf", exported)
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
(a.output / "relaxation_traces.json").write_text(json.dumps(traces, indent=2) + "\n")
print({k: v for k, v in report.items() if k != "records"})
