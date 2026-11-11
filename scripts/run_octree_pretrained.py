"""Measure the actual pretrained octree prior before adding conditioning."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
from orbit3d.generator.octree_backend import load_pretrained, trajectory_to_instance
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run

p = argparse.ArgumentParser()
p.add_argument("--weight", required=True)
p.add_argument("--training-sdf", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", default="cuda:0")
p.add_argument("--samples", type=int, default=64)
a = p.parse_args()
freeze_run(a.output, __file__)
torch.set_num_threads(4)
torch.manual_seed(42)
RDLogger.DisableLog("rdApp.*")
model = load_pretrained(a.weight, a.device)
train = {
    Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True)
    for m in Chem.SDMolSupplier(a.training_sdf, removeHs=False)
    if m is not None
}
records = []
mols = []
started = time.perf_counter()
for start in range(0, a.samples, 16):
    count = min(16, a.samples - start)
    zeros = torch.zeros(count, 2, device=a.device)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=a.device.split(":")[0], dtype=torch.bfloat16, enabled=a.device.startswith("cuda")
        ),
    ):
        trajectories = model.generate_conditional_trajectories(
            zeros, zeros, torch.zeros_like(zeros, dtype=torch.bool), evaluate_self_consistency=False
        )
    for i, trajectory in enumerate(trajectories):
        row = {"sample_id": start + i}
        try:
            inst = trajectory_to_instance(trajectory)
            mols.append(inst.graph)
            row.update(valid=True, smiles=inst.smiles, novel_vs_adaptation=inst.smiles not in train)
        except ValueError as exc:
            row.update(valid=False, reason=str(exc))
        records.append(row)
    print("generated", len(records), flush=True)
write_precise_sdf(a.output / "generated.sdf", mols)
report = {
    "backend": "pretrained_uni3dar_octree",
    "attempted": len(records),
    "valid": len(mols),
    "unique_valid": len({r["smiles"] for r in records if r["valid"]}),
    "novel_vs_adaptation": sum(r.get("novel_vs_adaptation", False) for r in records),
    "elapsed_seconds": time.perf_counter() - started,
    "records": records,
}
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print({k: v for k, v in report.items() if k != "records"})
