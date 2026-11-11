"""Structure-only adaptation for an expanded octree vocabulary; no property labels."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
from orbit3d.generator.octree_backend import (
    load_octree_checkpoint,
    OctreeDataset,
    checkpoint_contract,
    trajectory_to_instance,
    required_condition_elements,
)
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--train", type=Path, required=True)
p.add_argument("--validation", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--steps", type=int, default=500)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
if a.steps <= 0:
    raise ValueError("Structural adaptation needs positive training steps")
freeze_run(a.output, __file__)
record_input_hashes(
    a.output, {"source_checkpoint": a.checkpoint, "train": a.train, "validation": a.validation}
)
lock_deterministic_environment(a.seed)
torch.set_num_threads(4)
RDLogger.DisableLog("rdApp.*")
model, state = load_octree_checkpoint(a.checkpoint, "cuda:0", a.batch_size)
mols = list(Chem.SDMolSupplier(str(a.train), removeHs=False))
valid = list(Chem.SDMolSupplier(str(a.validation), removeHs=False))
assert mols and valid and all(m is not None for m in mols + valid)
canonical = lambda m: Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True)
train_smiles = {canonical(m) for m in mols}
assert not train_smiles & {canonical(m) for m in valid}
observed = {atom.GetSymbol() for m in mols for atom in m.GetAtoms()}
migration = state.get("representation_migration", {})
new_elements = set(migration.get("new_elements", [])) - set(migration.get("old_elements", []))
if new_elements - observed:
    raise ValueError(f"Uncovered new elements: {new_elements - observed}")
training = OctreeDataset(mols, model, seed=a.seed)
validation = OctreeDataset(valid, model, training=False, seed=a.seed)
model.requires_grad_(True)
model.interval_conditioner.requires_grad_(False)
model.prop_understanding_head.requires_grad_(False)
parameters = [v for v in model.parameters() if v.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=1e-4)
rng = np.random.default_rng(a.seed)
n_targets = len(state["target_cols"])


def structural_loss(dataset, indices):
    data = dataset.collate(indices)
    zero = torch.zeros((len(indices), n_targets), device="cuda:0")
    mask = torch.zeros_like(zero, dtype=torch.bool)
    return model.compute_sft_and_understanding_loss(data, zero, zero, zero, mask, understanding_weight=0.0)


def validation_loss():
    model.eval()
    values = []
    with torch.no_grad():
        for i in range(0, len(valid), a.batch_size):
            loss, _ = structural_loss(validation, list(range(i, min(i + a.batch_size, len(valid)))))
            values.append(float(loss))
    return float(np.mean(values))


initial_validation = validation_loss()
history = []
for step in range(a.steps):
    model.train()
    training.grid.set_epoch(step + 1)
    indices = rng.integers(0, len(mols), size=a.batch_size)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, metrics = structural_loss(training, indices)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite structural loss")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    history.append({"step": step + 1, **metrics})
    if (step + 1) % 25 == 0:
        print("structure-only SFT", history[-1], flush=True)
        (a.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
final_validation = validation_loss()
source_hash = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
state.update(
    model_state_dict=model.state_dict(),
    structural_history=history,
    initial_weights_sha256=source_hash,
    training_smiles=sorted(set(state.get("training_smiles", [])) | train_smiles),
    representation_config={
        "merge_level": model.args.merge_level,
        "allowed_elements": model.args.allow_atoms.split(","),
    },
    representation_migration={
        **migration,
        "requires_adaptation": False,
        "training_elements": sorted(observed),
    },
    condition_migration={
        "requires_condition_adaptation": True,
        "required_elements": sorted(required_condition_elements(state)),
        "reason": "Structure-only adaptation; property conditioner and understanding head were frozen",
    },
    contract=checkpoint_contract(
        model, state["target_cols"], state["target_units"], state["mean"], state["scale"]
    ),
)
checkpoint = a.output / "octree_structural.pt"
torch.save(state, checkpoint)
model.eval()
rows = []
arrays = {}
exports = []
for begin in range(0, 64, 8):
    zero = torch.zeros((8, n_targets), device="cuda:0")
    trajectories = model.generate_conditional_trajectories(
        zero,
        zero,
        torch.zeros_like(zero, dtype=torch.bool),
        evaluate_self_consistency=False,
        row_seeds=[candidate_seed(2026, begin + i) for i in range(8)],
        electron_parity=0,
    )
    for j, traj in enumerate(trajectories):
        i = begin + j
        symbols = traj["atom_symbols"]
        tokens = traj["decoder_feat"].cpu().numpy()
        arrays[f"tokens_{i}"] = tokens
        arrays[f"coordinates_{i}"] = traj["positions_np"]
        row = {
            "candidate_id": i,
            "symbols": symbols,
            "atom_count": len(symbols),
            "valid": False,
            "token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        }
        try:
            inst = trajectory_to_instance(traj, element_table=tuple(model.args.allow_atoms.split(",")))
            row.update(
                valid=True,
                smiles=inst.smiles,
                novel_vs_adaptation=inst.smiles not in state["training_smiles"],
            )
            inst.graph.SetProp("CandidateID", str(i))
            exports.append(inst.graph)
        except ValueError as error:
            row["reason"] = str(error)
        rows.append(row)
    print("unconditional candidates", len(rows), flush=True)
np.savez_compressed(a.output / "raw_trajectories.npz", **arrays)
write_precise_sdf(a.output / "generated.sdf", exports)
report = {
    "task": "Structure-only vocabulary adaptation; no spectra/HOMO/LUMO labels used",
    "train_records": len(mols),
    "validation_records": len(valid),
    "steps": a.steps,
    "initial_validation_mean_batch_loss": initial_validation,
    "final_validation_mean_batch_loss": final_validation,
    "validation_loss_definition": "FP32 mean of the model weighted tree/atom/XYZ generation loss over fixed batches; not unweighted token NLL",
    "observed_elements": sorted(observed),
    "allowed_elements": model.args.allow_atoms.split(","),
    "attempted": len(rows),
    "valid": sum(r["valid"] for r in rows),
    "boron_containing_valid": sum(r["valid"] and "B" in r["symbols"] for r in rows),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "records": rows,
    "conditional_inference_requires_further_labelled_adaptation": True,
}
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print({k: v for k, v in report.items() if k != "records"}, flush=True)
