"""Evaluate saved property checkpoints and separate weights/scalers from backbone replay."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from loom.generator.molecular_state import ATOM_TYPES
from loom.physics.factory import build_forcefield
from loom.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from loom.oracles.validation import compare_oracle_reload
from loom.utils.run_evidence import freeze_run, record_input_hashes
from loom.utils.reproducibility import lock_deterministic_environment
import torch

p = argparse.ArgumentParser()
p.add_argument("--trained-run", type=Path, required=True)
p.add_argument("--weight", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--compare-run", type=Path)
p.add_argument("--backend", choices=['dpa4', 'hyff'], default='dpa4')
p.add_argument("--hyff-tier", default='vivace')
a = p.parse_args()
freeze_run(a.output, __file__)
record_input_hashes(
    a.output,
    {
        "oracle": a.trained_run / "oracle",
        "forcefield": a.weight,
        **{s + "_sdf": a.trained_run / (s + ".sdf") for s in ["train", "valid", "test"]},
        "comparison_oracle": a.compare_run / "oracle" if a.compare_run else None,
        "comparison_test": a.compare_run / "test.sdf" if a.compare_run else None,
    },
)
lock_deterministic_environment(42)
torch.set_num_threads(2)
ff = build_forcefield(a.backend, a.weight, ATOM_TYPES, "cuda:0", tier=a.hyff_tier)
first = InternalDPA4PropertyOracle.load_from_dir(str(a.trained_run / "oracle"), ff)
second = InternalDPA4PropertyOracle.load_from_dir(str(a.trained_run / "oracle"), ff)
cohorts = {
    s: list(Chem.SDMolSupplier(str(a.trained_run / (s + ".sdf")), removeHs=False))
    for s in ["train", "valid", "test"]
}
assert all(m is not None for group in cohorts.values() for m in group)
names = first.target_cols
train_mean = np.array([[float(m.GetProp(n)) for n in names] for m in cohorts["train"]]).mean(0)
metrics = {}
for split in ["valid", "test"]:
    molecules = cohorts[split]
    pred, std, replay = compare_oracle_reload(first, second, molecules)
    truth = np.array([[float(m.GetProp(n)) for n in names] for m in molecules])
    metrics[split] = {
        "n": len(molecules),
        "mae_eV": dict(zip(names, abs(pred - truth).mean(0).tolist())),
        "train_mean_baseline_mae_eV": dict(zip(names, abs(train_mean - truth).mean(0).tolist())),
        "reload": replay,
        "mean_std_eV": std.mean(0).tolist(),
    }
    frame = pd.DataFrame({"identifier": [m.GetProp("SourceID") for m in molecules]})
    for i, n in enumerate(names):
        frame[n + "_true"] = truth[:, i]
        frame[n + "_pred"] = pred[:, i]
        frame[n + "_std"] = std[:, i]
    frame.to_csv(a.output / (split + "_predictions.csv"), index=False)
comparison = None
if a.compare_run:
    molecules = list(Chem.SDMolSupplier(str(a.compare_run / "test.sdf"), removeHs=False))
    canonical = lambda m: Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True)
    assert not {canonical(m) for m in molecules} & {canonical(m) for m in cohorts["train"]}
    baseline = InternalDPA4PropertyOracle.load_from_dir(str(a.compare_run / "oracle"), ff)
    older, _ = baseline.predict_3d_mols_batched(molecules)
    newer, _ = first.predict_3d_mols_batched(molecules)
    truth = np.array([[float(m.GetProp(n)) for n in names] for m in molecules])
    comparison = {
        "n": len(molecules),
        "same_structures": True,
        "training_overlap": 0,
        "previous_model_mae_eV": dict(zip(names, abs(older - truth).mean(0).tolist())),
        "new_model_mae_eV": dict(zip(names, abs(newer - truth).mean(0).tolist())),
    }
report = {
    "task": "Saved OCELOT property model evaluation",
    "training_records": len(cohorts["train"]),
    "target_names": names,
    "metrics": metrics,
    "matched_previous_test": comparison,
    "training_ids": [m.GetProp("SourceID") for m in cohorts["train"]],
    "valid_ids": [m.GetProp("SourceID") for m in cohorts["valid"]],
    "test_ids": [m.GetProp("SourceID") for m in cohorts["test"]],
    "geometry_origin": "ETKDGv3/MMFF94 reconstructed geometries; not original DFT geometries",
}
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"metrics": metrics, "matched_previous_test": comparison}, indent=2))
