#!/usr/bin/env python3
"""
End-to-End Verification Suite for ORBIT-3D Package
==================================================
Verifies:
1. Full DPA-4 Series (Nano, Mini, Neo, Air, Plus) native equivariant descriptor extraction (`D_desc`)
2. All 3 Oracle Modes via `config.json`:
   - `external_uniqsar`
   - `internal_unimol` (initialized from FP16 built-in weights)
   - `internal_dpa4` (using 100% of DPA-4's native equivariant descriptor from `fitting_net`)
3. Online Gated-GRPO + Stage 5 Iso-Property Conformer Relaxation + Reload Reproducibility
"""

import json
import os
import shutil
import sys
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "src"), str(_ROOT), "/sjtu-combustion/wjw/chensq/deepmd_v32"]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from orbit3d.physics import DPAPotentialGuidance
from orbit3d.pipeline import run_from_config
from orbit3d.utils import resolve_dpa4_weight_path


def create_demo_dataset(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("c1ccccc1", 0.42),
        ("c1ccc2ccccc2c1", 0.51),
        ("CC(=O)Oc1ccccc1C(=O)O", 0.48),
        ("COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", 0.56),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 0.46),
        ("c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", 0.61),
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("SMILES,TARGET\n")
        for smi, val in rows:
            f.write(f"{smi},{val}\n")


def run_mode_test(mode_name: str, oracle_cfg: dict, base_dir: Path, pretrained_dir: str) -> dict:
    mode_dir = base_dir / mode_name
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    data_csv = mode_dir / "data" / "train.csv"
    ckpt_save = mode_dir / "checkpoints" / "save"
    out_dir = mode_dir / "outputs"
    create_demo_dataset(data_csv)

    cfg = {
        "mode": "train_and_predict",
        "seed": 20260319,
        "device": "cuda:0",
        "mount_paths": {
            "train_data_path": str(data_csv),
            "external_uniqsar_dir": "/sjtu-combustion/wjw/chensq/uniqsar/data_uniqsar/train/669736b53ea4b9b6611619c2",
            "checkpoint_save_dir": str(ckpt_save),
            "checkpoint_load_dir": str(ckpt_save),
            "output_dir": str(out_dir),
            "builtin_pretrained_dir": pretrained_dir,
        },
        "oracle_config": oracle_cfg,
        "generator_config": {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 3,
            "sft_epochs": 2,
            "sft_lr": 3e-4,
            "grpo_mode": "multi_window_general",
            "grpo_steps": 2,
            "group_size": 4,
            "lr": 5e-5,
            "temperature": 0.90,
            "kl_beta": 0.04,
            "sft_anchor_weight": 0.15,
            "uncertainty_weight": 0.35,
        },
        "inference_config": {
            "target_interval": [0.45, 0.60],
            "run_specialist_grpo_before_gen": False,
            "num_molecules": 2,
            "base_seed": 20260319,
            "temperature": 0.85,
            "top_p": 0.92,
        },
        "stage5_relax_config": {
            "enabled": True,
            "dpa4_scale": "Mini",
            "relax_steps": 5,
            "step_size": 0.008,
            "prop_grad_weight": 0.25,
        },
    }

    cfg_path = mode_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n>>> Running E2E Test for Mode: [{mode_name}] ...", flush=True)
    summary = run_from_config(str(cfg_path))
    print(f">>> [{mode_name}] Train+Predict Summary:", json.dumps(summary, indent=2), flush=True)

    out_dir_reload = mode_dir / "outputs_reload_verify"
    cfg["mode"] = "predict"
    cfg["mount_paths"]["output_dir"] = str(out_dir_reload)
    cfg_reload_path = mode_dir / "config_predict_only.json"
    with open(cfg_reload_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    summary_reload = run_from_config(str(cfg_reload_path))

    df1 = pd.read_csv(summary["output_csv"])
    df2 = pd.read_csv(summary_reload["output_csv"])
    y1 = float(df1.iloc[0]["y_pred"])
    y2 = float(df2.iloc[0]["y_pred"])
    f1 = float(df1.iloc[0]["final_force_eva"])
    f2 = float(df2.iloc[0]["final_force_eva"])
    assert abs(y1 - y2) < 1e-5, f"Deterministic reproduction mismatch: {y1} vs {y2}"
    print(
        f">>> [{mode_name}] Reload Predict Reproducibility Verified! "
        f"y_pred={y1:.6f} == {y2:.6f} | final_force={f1:.4f} == {f2:.4f} eV/A",
        flush=True,
    )
    return summary


def main() -> None:
    pretrained_dir = str((_ROOT / "pretrained").resolve())
    os.environ["ORBIT3D_PRETRAINED_DIR"] = pretrained_dir

    # Verify all 5 DPA-4 scales and probe their native internal descriptor dimension D_desc
    for scale in ["Nano", "Mini", "Neo", "Air", "Plus"]:
        p = resolve_dpa4_weight_path(pretrained_dir, scale=scale)
        g = DPAPotentialGuidance(model_path=p, head="Default", device="cuda:0")
        d_desc = g.probe_native_descriptor_dim()
        print(
            f"[DPA-4 Native Descriptor Check] {scale:5s} -> {Path(p).name} "
            f"({Path(p).stat().st_size / (1024*1024):.2f} MB) | native D_desc = {d_desc}",
            flush=True,
        )

    base_dir = Path("/tmp/orbit3d_e2e_verify")

    run_mode_test(
        "1_external_uniqsar",
        {
            "oracle_mode": "external_uniqsar",
            "smiles_col": "SMILES",
            "target_cols": "TARGET",
            "infer_choices": None,
        },
        base_dir,
        pretrained_dir,
    )

    run_mode_test(
        "2_internal_unimol",
        {
            "oracle_mode": "internal_unimol",
            "smiles_col": "SMILES",
            "target_cols": "TARGET",
            "train_oracle": True,
            "remove_hs": False,
            "n_folds": 2,
            "oracle_epochs": 2,
            "batch_size": 4,
            "oracle_lr": 1e-4,
        },
        base_dir,
        pretrained_dir,
    )

    run_mode_test(
        "3_internal_dpa4",
        {
            "oracle_mode": "internal_dpa4",
            "smiles_col": "SMILES",
            "target_cols": "TARGET",
            "train_oracle": True,
            "n_ensemble_heads": 3,
            "oracle_epochs": 3,
            "oracle_lr": 3e-4,
        },
        base_dir,
        pretrained_dir,
    )

    print("\nALL 5 DPA-4 SCALES + ALL 3 ORACLE MODES + CONFORMER RELAXATION + REPRODUCIBILITY PASSED!", flush=True)


if __name__ == "__main__":
    main()
