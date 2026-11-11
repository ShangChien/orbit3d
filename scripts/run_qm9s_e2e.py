"""Actual QM9s -> OCELOT-trained oracle -> generator -> DPA refinement -> reload run.

Generator conditions are explicitly oracle pseudo-labels, not QM9s quantum labels.
"""

import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.pipeline.runner import run_from_config
from orbit3d.utils.run_evidence import freeze_run
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.physics.dpa_forcefield import DPAPotentialGuidance
from orbit3d.generator.octree_tokenizer import ATOM_TYPES
from orbit3d.utils.dataset_io import write_precise_sdf
import shutil

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--oracle", type=Path, required=True)
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--train-size", type=int, default=256)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--samples", type=int, default=32)
p.add_argument("--device", default="cpu")
p.add_argument("--oracle-mode", choices=["fingerprint", "internal_dpa4"], default="fingerprint")
p.add_argument("--temperature", type=float, default=0.5)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
freeze_run(a.output, __file__)
torch.set_num_threads(4)
started = time.perf_counter()
if a.oracle_mode == "fingerprint":
    oracle = FingerprintOracle.load_from_dir(a.oracle)
else:
    ff = DPAPotentialGuidance(
        str(a.weights / "DPA4-Mini-OMol25-100M-v20260820.pt"), atom_type_map=ATOM_TYPES, device=a.device
    )
    oracle = InternalDPA4PropertyOracle.load_from_dir(str(a.oracle), ff)
    shutil.copytree(a.oracle, a.output / "checkpoints" / "internal_dpa4_oracle", dirs_exist_ok=True)
mols = list(Chem.SDMolSupplier(str(a.data / "train.sdf"), removeHs=False))[: a.train_size]
assert len(mols) == a.train_size and all(m is not None for m in mols)
y, std = oracle.predict_3d_mols(mols)
source = a.output / "train_pseudolabels.sdf"

for index, m in enumerate(mols):
    for k, name in enumerate(oracle.target_cols):
        m.SetProp(name, str(y[index, k]))
    m.SetProp("PropertyLabelOrigin", f"OCELOT {a.oracle_mode} predictions; NOT QM9s ground truth")
write_precise_sdf(source, mols)
interval = {
    name: [float(np.quantile(y[:, i], 0.1)), float(np.quantile(y[:, i], 0.9))]
    for i, name in enumerate(oracle.target_cols)
}
config = {
    "mode": "train_and_predict",
    "device": a.device,
    "seed": 42,
    "mount_paths": {
        "train_data_path": str(source.resolve()),
        "checkpoint_save_dir": str((a.output / "checkpoints").resolve()),
        "output_dir": str((a.output / "generated").resolve()),
        "builtin_pretrained_dir": str(a.weights.resolve()),
        "external_fingerprint_dir": str(a.oracle.resolve()),
    },
    "oracle_config": {"oracle_mode": a.oracle_mode, "target_cols": oracle.target_cols, "train_oracle": False},
    "generator_config": {
        "backend": "dense_grid_v2",
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "max_atoms": 40,
        "max_seq_len": 162,
        "sft_epochs": a.epochs,
        "sft_lr": 0.001,
        "batch_size": 32,
        "grpo_steps": 2,
        "group_size": 8,
        "grpo_mode": "single_window_specialist",
    },
    "stage5_relax_config": {"enabled": True, "dpa4_scale": "Mini", "relax_steps": 8, "step_size": 0.004},
    "inference_config": {
        "num_molecules": a.samples,
        "target_interval": interval,
        "base_seed": 2026,
        "temperature": a.temperature,
        "top_p": 0.9,
        "max_rms_force_eva": 0.5,
    },
}
path = a.output / "config.json"
path.write_text(json.dumps(config, indent=2) + "\n")
summary = run_from_config(str(path))
first = pd.read_csv(summary["output_csv"])
config["mode"] = "predict"
config["mount_paths"]["output_dir"] = str((a.output / "reloaded").resolve())
path2 = a.output / "config_reload.json"
path2.write_text(json.dumps(config, indent=2) + "\n")
reloaded = run_from_config(str(path2))
second = pd.read_csv(reloaded["output_csv"])
pd.testing.assert_frame_equal(first.fillna(""), second.fillna(""), check_exact=False, atol=1e-6, rtol=1e-6)
first_mols = [m for m in Chem.SDMolSupplier(summary["output_sdf"], removeHs=False) if m is not None]
second_mols = [m for m in Chem.SDMolSupplier(reloaded["output_sdf"], removeHs=False) if m is not None]
assert len(first_mols) == len(second_mols)
for m, n in zip(first_mols, second_mols):
    assert Chem.MolToSmiles(m) == Chem.MolToSmiles(n)
    np.testing.assert_allclose(
        m.GetConformer().GetPositions(), n.GetConformer().GetPositions(), atol=1e-5, rtol=0
    )
train_smiles = {Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True) for m in mols}
generated_smiles = [Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True) for m in first_mols]
report = {
    "task": "QM9s / OCELOT oracle / DPA end-to-end integration",
    "oracle_mode": a.oracle_mode,
    "training_records": len(mols),
    "generator_label_origin": "oracle pseudo-labels",
    "summary": summary,
    "all_record_reload_passed": True,
    "valid_unique": len(set(generated_smiles)),
    "novel_vs_training": sum(s not in train_smiles for s in generated_smiles),
    "elapsed_seconds": time.perf_counter() - started,
    "limitations": [
        "QM9s and OCELOT are different chemistry distributions",
        "Property success measured by optimization oracle, not independent DFT",
        "Small generator training subset; no general superiority claim",
    ],
}
(a.output / "e2e_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
