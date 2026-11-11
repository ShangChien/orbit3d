"""Replay an octree pipeline twice and compare all candidates and exported molecules."""

import argparse, json, copy
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from orbit3d.pipeline.octree_runner import run_octree_from_config

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--reference-run", type=Path, help="Compare the live post-training run with one reload")
a = p.parse_args()
cfg = json.loads(a.config.read_text())
cfg["mode"] = "predict"
cfg["generator_config"]["grpo_steps"] = 0
cfg["mount_paths"]["checkpoint_load_path"] = a.checkpoint
cfg["mount_paths"].pop("generator_pretrained_path", None)
cfg["mount_paths"].pop("train_data_path", None)
summaries = []
first_path = a.reference_run or a.output / "first"
second_path = a.output / "second"
for path in ([second_path] if a.reference_run else [first_path, second_path]):
    current = copy.deepcopy(cfg)
    current["mount_paths"]["output_dir"] = str(path)
    summaries.append(run_octree_from_config(current))
first = pd.read_csv(first_path / "candidates.csv")
second = pd.read_csv(second_path / "candidates.csv")
# Compare semantic fields exactly, numeric model/force results within declared tolerances.
for field in ["candidate_id", "status", "joint_success", "smiles", "raw_valid"]:
    pd.testing.assert_series_equal(first[field].fillna(""), second[field].fillna(""))
if "token_sha256" in first:
    pd.testing.assert_series_equal(first.token_sha256, second.token_sha256)
    with np.load(first_path / "raw_trajectories.npz") as x, np.load(second_path / "raw_trajectories.npz") as y:
        assert set(x.files) == set(y.files)
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key])
for field in ["self_score", "initial_force", "final_force"]:
    np.testing.assert_allclose(first[field], second[field], equal_nan=True, atol=1e-5, rtol=1e-5)
for x, y in zip(first.y_pred, second.y_pred):
    if pd.isna(x):
        assert pd.isna(y)
    else:
        np.testing.assert_allclose(json.loads(x), json.loads(y), atol=1e-5, rtol=1e-5)
mols = []
for path in [first_path, second_path]:
    mols.append(
        [
            m
            for m in Chem.SDMolSupplier(str(path / "generated.sdf"), removeHs=False)
            if m is not None
        ]
    )
assert len(mols[0]) == len(mols[1])
for x, y in zip(*mols):
    assert Chem.MolToSmiles(x) == Chem.MolToSmiles(y)
    np.testing.assert_allclose(
        x.GetConformer().GetPositions(), y.GetConformer().GetPositions(), atol=1e-5, rtol=0
    )
traces = [json.loads((p / "relaxation_traces.json").read_text()) for p in [first_path, second_path]]
for x, y in zip(*traces):
    assert x['candidate_id'] == y['candidate_id']
    for key in ['initial_forces_eva', 'final_forces_eva']:
        if key in x:
            np.testing.assert_allclose(x[key], y[key], atol=1e-5, rtol=1e-5)
(a.output / "reload_summary.json").write_text(
    json.dumps(
        {
            "passed": True,
            "all_candidates": len(first),
            "exported_molecules": len(mols[0]),
            "prediction_atol": 1e-5,
            "coordinate_atol_A": 1e-5,
            "raw_tokens_compared": "token_sha256" in first,
            "upstream_pretrained_and_training_data_required": False,
        },
        indent=2,
    )
    + "\n"
)
