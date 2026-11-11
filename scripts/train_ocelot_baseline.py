"""Train a real OCELOT graph-property baseline on fixed scaffold splits."""

import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.property_metrics import paired_energy_difference_metrics

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--trees", type=int, default=128)
p.add_argument("--models", type=int, default=3)
p.add_argument("--jobs", type=int, default=4)
p.add_argument("--reuse-oracle", action="store_true")
p.add_argument("--targets", nargs="+", default=["homo", "lumo"])
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
freeze_run(a.output, __file__)
record_input_hashes(a.output, {s: a.data / f"{s}.csv" for s in ["train", "valid", "test"]})
(a.output / "run_config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2) + "\n")
start = time.perf_counter()
names = a.targets
if len(set(names)) != len(names):
    raise ValueError('Target names must be unique')
frames = {s: pd.read_csv(a.data / f"{s}.csv") for s in ["train", "valid", "test"]}
for first, second in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    assert not set(frames[first].canonical_smiles) & set(frames[second].canonical_smiles)
    assert not set(frames[first].scaffold) & set(frames[second].scaffold)
if a.reuse_oracle:
    oracle = FingerprintOracle.load_from_dir(a.output / "oracle")
else:
    oracle = FingerprintOracle(names)
    oracle.fit(
        [Chem.MolFromSmiles(s) for s in frames["train"].canonical_smiles],
        frames["train"][names].to_numpy(),
        a.output / "oracle",
        n_models=a.models,
        n_trees=a.trees,
        n_jobs=a.jobs,
    )
if oracle.target_cols != names:
    raise ValueError('Reused Oracle target order differs')
reloaded = FingerprintOracle.load_from_dir(a.output / "oracle")
metrics = {}
for split in ["valid", "test"]:
    frame = frames[split]
    mols = [Chem.MolFromSmiles(s) for s in frame.canonical_smiles]
    pred, std = oracle.predict_3d_mols(mols)
    again, _ = reloaded.predict_3d_mols(mols)
    np.testing.assert_allclose(pred, again, rtol=0, atol=0)
    y = frame[names].to_numpy()
    err = pred - y
    metrics[split] = {
        "n": len(y),
        "mae_eV": dict(zip(names, np.abs(err).mean(0).tolist())),
        "rmse_eV": dict(zip(names, np.sqrt(np.square(err).mean(0)).tolist())),
        "train_mean_baseline_mae_eV": dict(zip(names, np.abs(y - oracle.training_mean).mean(0).tolist())),

        "reload_max_abs_diff": float(np.abs(pred - again).max()),
    }
    if {'homo','lumo'} <= set(names):
        metrics[split]['gap_from_homo_lumo_mae_eV'] = float(np.abs(
            pred[:,names.index('lumo')]-pred[:,names.index('homo')]-frame.hl.to_numpy()).mean())
    if {'s0s1','s0t1'} <= set(names):
        metrics[split]['vertical_singlet_triplet_gap'] = paired_energy_difference_metrics(
            pred, y, names, 's0s1', 's0t1')
        metrics[split]['vertical_singlet_triplet_gap']['geometry_definition'] = 'Both excitations vertical at the source S0 reference geometry; not an adiabatic gap'
    metrics[split]['element_subgroups'] = {}
    for element in oracle.training_elements or []:
        indices = [i for i, mol in enumerate(mols) if any(atom.GetSymbol() == element for atom in Chem.AddHs(mol).GetAtoms())]
        if indices:
            metrics[split]['element_subgroups'][element] = {
                'n': len(indices), 'mae_eV': dict(zip(names, np.abs(err[indices]).mean(0).tolist()))}
    out = frame[["identifier", "canonical_smiles"]].copy()
    for i, name in enumerate(names):
        out[f"{name}_true"] = y[:, i]
        out[f"{name}_pred"] = pred[:, i]
        out[f"{name}_std"] = std[:, i]
    out.to_csv(a.output / f"{split}_predictions.csv", index=False)
report = {
    "task": "OCELOT 2D fingerprint property baseline",
    "targets": names,
    "units": "eV",
    "split": "fixed scaffold holdouts",
    "training_records": len(frames["train"]),
    "training_elements": oracle.training_elements,
    "elapsed_seconds": time.perf_counter() - start,
    "reused_trained_oracle": a.reuse_oracle,
    "metrics": metrics,
    "data_sha256": {s: hashlib.sha256((a.data / f"{s}.csv").read_bytes()).hexdigest() for s in frames},
    "note": "Predictor validation only; no coordinate gradient or calibrated uncertainty claim",
}
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
