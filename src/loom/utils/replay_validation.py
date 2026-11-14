"""Replay an octree pipeline twice and compare all candidates and exported molecules."""

import argparse, json, copy
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from ..pipeline.octree_runner import run_octree_from_config


def main():
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
    for path in [second_path] if a.reference_run else [first_path, second_path]:
        current = copy.deepcopy(cfg)
        current["mount_paths"]["output_dir"] = str(path)
        summaries.append(run_octree_from_config(current))
    report = compare_run_outputs(first_path, second_path)
    (a.output / "reload_summary.json").write_text(json.dumps(report, indent=2) + "\n")


def compare_scaffold_records(first, second):
    """Exact graph/placement decisions; model and force values use replay tolerances."""
    assert len(first) == len(second), 'Scaffold proposal count differs'
    numerical = {'property_score', 'proposal_prediction', 'y_pred', 'initial_force', 'final_force'}
    for left, right in zip(first, second):
        assert set(left) == set(right), 'Scaffold proposal fields differ'
        for key in left:
            if key in numerical:
                np.testing.assert_allclose(left[key], right[key], atol=1e-5, rtol=1e-5)
            else:
                assert left[key] == right[key], f'Scaffold proposal {left["candidate_id"]}: {key} differs'


def compare_run_outputs(first_path, second_path):
    """Verify two existing runs without repeating generation or changing their files."""
    first_path, second_path = Path(first_path), Path(second_path)
    first = pd.read_csv(first_path / "candidates.csv")
    second = pd.read_csv(second_path / "candidates.csv")
    # Columns for refinement exist only if at least one molecule reached that stage.
    # A reproducible all-rejected run still has a complete raw-candidate ledger.
    assert set(first.columns) == set(second.columns)
    # Compare semantic fields exactly, numeric model/force results within declared tolerances.
    for field in ["candidate_id", "status", "joint_success", "smiles", "raw_valid"]:
        if field in first:
            pd.testing.assert_series_equal(first[field].fillna(""), second[field].fillna(""))
    if "token_sha256" in first:
        pd.testing.assert_series_equal(first.token_sha256, second.token_sha256)
        with (
            np.load(first_path / "raw_trajectories.npz") as x,
            np.load(second_path / "raw_trajectories.npz") as y,
        ):
            assert set(x.files) == set(y.files)
            for key in x.files:
                np.testing.assert_array_equal(x[key], y[key])
    for field in ["self_score", "initial_force", "final_force"]:
        if field in first:
            np.testing.assert_allclose(first[field], second[field], equal_nan=True, atol=1e-5, rtol=1e-5)
    for x, y in zip(first.y_pred, second.y_pred) if "y_pred" in first else []:
        if pd.isna(x):
            assert pd.isna(y)
        else:
            np.testing.assert_allclose(json.loads(x), json.loads(y), atol=1e-5, rtol=1e-5)
    mols = []
    for path in [first_path, second_path]:
        sdf = path / "generated.sdf"
        mols.append(list(Chem.SDMolSupplier(str(sdf), removeHs=False)) if sdf.stat().st_size else [])
        assert all(m is not None for m in mols[-1])
    assert len(mols[0]) == len(mols[1])
    assert json.loads((first_path/'raw_symbols.json').read_text()) == json.loads((second_path/'raw_symbols.json').read_text())
    for x, y in zip(*mols):
        assert Chem.MolToSmiles(x) == Chem.MolToSmiles(y)
        for name in ['CandidateID', 'TotalCharge', 'SpinMultiplicity']:
            assert x.HasProp(name) == y.HasProp(name)
            if x.HasProp(name):
                assert x.GetProp(name) == y.GetProp(name)
        np.testing.assert_allclose(json.loads(x.GetProp('PredictedValue')), json.loads(y.GetProp('PredictedValue')), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            x.GetConformer().GetPositions(), y.GetConformer().GetPositions(), atol=1e-5, rtol=0
        )
    traces = [json.loads((p / "relaxation_traces.json").read_text()) for p in [first_path, second_path]]
    assert len(traces[0]) == len(traces[1]), "Relaxation trace count differs"
    for x, y in zip(*traces):
        assert x["candidate_id"] == y["candidate_id"]
        assert x.get('status') == y.get('status')
        assert [(r['accepted'], r['backtracks']) for r in x.get('history', [])] == [(r['accepted'], r['backtracks']) for r in y.get('history', [])]
        for key in ["initial_forces_eva", "final_forces_eva"]:
            if key in x:
                np.testing.assert_allclose(x[key], y[key], atol=1e-5, rtol=1e-5)
    scaffold_records = first_path/'proposal_records.json'
    compare_scaffold = scaffold_records.exists() or (second_path/'proposal_records.json').exists()
    if compare_scaffold:
        compare_scaffold_records(json.loads(scaffold_records.read_text()),
                                 json.loads((second_path/'proposal_records.json').read_text()))
    return {
        "scaffold_proposals_compared": compare_scaffold,
        "passed": True,
        "all_candidates": len(first),
        "exported_molecules": len(mols[0]),
        "prediction_atol": 1e-5,
        "coordinate_atol_A": 1e-5,
        "raw_tokens_compared": "token_sha256" in first,
        "upstream_pretrained_and_training_data_required": False,
    }
