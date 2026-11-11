"""Small scaffold-separated 3D property pilot with independently bootstrapped DPA adapters."""

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orbit3d.generator.octree_tokenizer import ATOM_TYPES
from orbit3d.physics.factory import build_forcefield
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.oracles.validation import compare_oracle_reload
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.conformers import ocelot_conformer

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--weight", required=True)
p.add_argument("--device", default="cpu")
p.add_argument("--backend", choices=['dpa4', 'hyff'], default='dpa4')
p.add_argument("--hyff-tier", default='vivace')
p.add_argument('--geometry-method', choices=['mmff94', 'uff'], default='mmff94')
p.add_argument('--prioritize-elements', nargs='*', default=[], help='Include rare-element structures before the deterministic hash sample in each split')
p.add_argument("--train", type=int, default=512)
p.add_argument("--eval", type=int, default=64)
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
freeze_run(a.output, __file__)
(a.output / 'run_config.json').write_text(json.dumps(vars(a), default=str, indent=2)+'\n')
record_input_hashes(a.output, {'forcefield': a.weight, **{split+'_csv': a.data / (split+'.csv') for split in ['train','valid','test']}})
torch.set_num_threads(2)
lock_deterministic_environment(a.seed)
started = time.perf_counter()
frames = {}
molecules = {}
failures = []
names = ["homo", "lumo"]
for split, limit in [("train", a.train), ("valid", a.eval), ("test", a.eval)]:
    frame = pd.read_csv(a.data / f"{split}.csv")
    frame["rank"] = frame.identifier.astype(str).map(
        lambda s: hashlib.sha256(("42:" + s).encode()).hexdigest()
    )
    if a.prioritize_elements:
        from rdkit import Chem
        wanted = set(a.prioritize_elements)
        frame['priority'] = frame.canonical_smiles.map(lambda s: bool(wanted & {atom.GetSymbol() for atom in Chem.MolFromSmiles(s).GetAtoms()}))
        frame = frame.sort_values(['priority', 'rank'], ascending=[False, True])
    else:
        frame = frame.sort_values("rank")
    rows = []
    mols = []
    for row in frame.to_dict("records"):
        if len(rows) >= limit:
            break
        try:
            mol = ocelot_conformer(row['canonical_smiles'], method=a.geometry_method)
        except ValueError as exc:
            failures.append({'split': split, 'id': row['identifier'], 'reason': 'geometry_preparation: ' + str(exc)})
            continue
        mol.SetProp("SourceID", str(row["identifier"]))
        for name in names:
            mol.SetProp(name, str(row[name]))
        mols.append(mol)
        rows.append(row)
    write_precise_sdf(a.output / f"{split}.sdf", mols)
    frames[split] = pd.DataFrame(rows)
    molecules[split] = mols
    print(split, len(mols), "conformers", flush=True)
ff = build_forcefield(a.backend, a.weight, ATOM_TYPES, a.device, tier=a.hyff_tier)
oracle = InternalDPA4PropertyOracle(ff, 2, names, n_ensemble_heads=3)
oracle.target_units = ['eV', 'eV']
oracle.geometry_method = a.geometry_method
oracle.fit_from_dataset(
    molecules["train"],
    frames["train"][names].to_numpy(),
    str(a.output / "oracle"),
    epochs=a.epochs,
    batch_size=a.batch_size,
    seed=a.seed,
    validation_mols=molecules["valid"],
    validation_targets=frames["valid"][names].to_numpy(),
)
restored = InternalDPA4PropertyOracle.load_from_dir(str(a.output / "oracle"), ff)
metrics = {}
for split in ["valid", "test"]:
    pred, std, reload_report = compare_oracle_reload(oracle, restored, molecules[split])
    true = frames[split][names].to_numpy()
    metrics[split] = {
        "n": len(true),
        "mae_eV": dict(zip(names, np.abs(pred - true).mean(0).tolist())),
        "train_mean_baseline_mae_eV": dict(
            zip(names, np.abs(true - frames["train"][names].mean().to_numpy()).mean(0).tolist())
        ),
        "reload": reload_report,
        "mean_std_eV": std.mean(0).tolist(),
        "element_subgroups": {
            element: {'n': int(mask.sum()), 'mae_eV': dict(zip(names, np.abs(pred[mask]-true[mask]).mean(0).tolist()))}
            for element in sorted({atom.GetSymbol() for m in molecules[split] for atom in m.GetAtoms()})
            if (mask := np.array([any(atom.GetSymbol() == element for atom in m.GetAtoms()) for m in molecules[split]])).any()
        },
    }
    result = frames[split][["identifier"]].copy()
    for i, name in enumerate(names):
        result[name + "_true"] = true[:, i]
        result[name + "_pred"] = pred[:, i]
        result[name + "_std"] = std[:, i]
    result.to_csv(a.output / f"{split}_predictions.csv", index=False)
report = {
    "task": f"OCELOT 3D bootstrap-{a.backend} training",
    "forcefield_backend": a.backend,
    "seed": a.seed,
    "batch_size": a.batch_size,
    "geometry_origin": f"ETKDGv3+{a.geometry_method.upper()}; original DFT coordinates unavailable",
    "metrics": metrics,
    "embedding_failures": failures,
    "elapsed_seconds": time.perf_counter() - started,
    "device": a.device,
    "training_ids": frames["train"].identifier.tolist(),
    "valid_ids": frames["valid"].identifier.tolist(),
    "test_ids": frames["test"].identifier.tolist(),
}
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(metrics, indent=2), flush=True)
