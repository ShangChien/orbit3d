"""
Unified Config-Driven Pipeline Orchestrator (`runner.py`)
=========================================================
Executes `train`, `predict` (`generate`), or `train_and_predict` workflows from a single `config.json`.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem

from ..generator import (
    ATOM_TYPES,
    ConditionalUni3DAR,
    Mol3DInstance,
    encode_mol_to_octree_tokens,
    mol3d_to_validated_rdkit_mol,
    rdkit_mol_to_mol3d_instance,
    sample_molecule_deterministic_by_id,
    train_gated_grpo_with_oracle,
)
from ..oracles import (
    InternalDPA4PropertyOracle,
    InternalUniMolOracle,
    UniQSARTrainedBridge,
)
from ..physics import DPAPotentialGuidance, HyFFPotentialGuidance, relax_conformer_iso_property
from ..utils import (
    load_3d_molecular_dataset,
    lock_deterministic_environment,
    resolve_dpa4_weight_path,
    resolve_hyff_weight_path,
    resolve_unimol_weight_path,
)


def run_from_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    mode = str(cfg.get("mode", "predict")).lower()
    seed = int(cfg.get("seed", 42))
    device = str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    lock_deterministic_environment(seed)

    mounts = cfg.get("mount_paths", {})
    train_data_path = mounts.get("train_data_path", "/workspace/data/train.csv")
    ckpt_save_dir = mounts.get("checkpoint_save_dir", "/workspace/checkpoints/save")
    ckpt_load_dir = mounts.get("checkpoint_load_dir", ckpt_save_dir)
    output_dir = mounts.get("output_dir", "/workspace/outputs")
    ext_uniqsar_dir = mounts.get("external_uniqsar_dir", "")
    builtin_dir = mounts.get("builtin_pretrained_dir", "/opt/orbit3d/pretrained")

    os.makedirs(ckpt_save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    oracle_cfg = cfg.get("oracle_config", {})
    oracle_mode = str(oracle_cfg.get("oracle_mode", "external_uniqsar")).lower()
    target_cols = oracle_cfg.get("target_cols", "TARGET")
    smiles_col = oracle_cfg.get("smiles_col", "SMILES")

    stage5_cfg = cfg.get("stage5_relax_config", {})
    mlip_backend = str(stage5_cfg.get("mlip_backend", oracle_cfg.get("mlip_backend", "dpa4"))).lower()
    if mlip_backend == "hyff" or oracle_mode == "internal_hyff":
        hyff_tier = str(stage5_cfg.get("hyff_tier", oracle_cfg.get("hyff_tier", "vivace")))
        dpa4_scale = f"HyFF-{hyff_tier}"
        hyff_w = resolve_hyff_weight_path(builtin_dir, tier=hyff_tier)
        dpa_guidance = HyFFPotentialGuidance(
            tier=hyff_tier,
            weights=hyff_w,
            atom_type_map=ATOM_TYPES,
            device=device,
            default_charge=float(stage5_cfg.get("charge", 0.0)),
            default_multiplicity=int(stage5_cfg.get("multiplicity", 1)),
        )
    else:
        dpa4_scale = str(stage5_cfg.get("dpa4_scale", oracle_cfg.get("dpa4_scale", "Mini")))
        dpa4_path = resolve_dpa4_weight_path(builtin_dir, scale=dpa4_scale)
        dpa_guidance = DPAPotentialGuidance(
            model_path=dpa4_path,
            head="Default",
            atom_type_map=ATOM_TYPES,
            device=device,
        )

    # 1. Load dataset & initialize/train selected Property Oracle
    mols_3d: List[Chem.Mol] = []
    targets_2d: Optional[np.ndarray] = None

    if mode in ["train", "train_and_predict"] and os.path.exists(train_data_path):
        mols_3d, targets_2d, target_col_list = load_3d_molecular_dataset(
            train_data_path,
            smiles_col=smiles_col,
            target_cols=target_cols,
            dpa_guidance=None,
            seed=seed,
        )
        prop_mean = float(np.mean(targets_2d[:, 0]))
        prop_std = float(np.std(targets_2d[:, 0]) + 1e-6)
    else:
        prop_mean = float(cfg.get("dataset_stats", {}).get("prop_mean", 540.0))
        prop_std = float(cfg.get("dataset_stats", {}).get("prop_std", 120.0))
        target_col_list = [c.strip() for c in str(target_cols).split(",")]

    if oracle_mode == "external_uniqsar":
        if not ext_uniqsar_dir or not os.path.exists(ext_uniqsar_dir):
            raise FileNotFoundError(f"external_uniqsar_dir not found: {ext_uniqsar_dir}")
        oracle = UniQSARTrainedBridge(
            model_dir=ext_uniqsar_dir,
            device=device,
            infer_choices=oracle_cfg.get("infer_choices", None),
        )
        if hasattr(oracle.scaler, "mean_"):
            prop_mean = float(np.asarray(oracle.scaler.mean_).reshape(-1)[0])
            prop_std = float(np.asarray(oracle.scaler.scale_).reshape(-1)[0])
    elif oracle_mode == "internal_unimol":
        oracle_subdir_save = os.path.join(ckpt_save_dir, "internal_unimol_oracle")
        oracle_subdir_load = os.path.join(ckpt_load_dir, "internal_unimol_oracle")
        if mode in ["train", "train_and_predict"] and oracle_cfg.get("train_oracle", True):
            remove_hs = bool(oracle_cfg.get("remove_hs", False))
            pretrain_unimol = resolve_unimol_weight_path(builtin_dir, remove_hs=remove_hs)
            oracle = InternalUniMolOracle(
                output_dim=len(target_col_list),
                n_folds=int(oracle_cfg.get("n_folds", 5)),
                remove_hs=remove_hs,
                target_cols=target_col_list,
                pretrain_weight_path=pretrain_unimol,
                device=device,
            )
            oracle.fit_from_dataset(
                mols_3d,
                targets_2d,
                save_dir=oracle_subdir_save,
                epochs=int(oracle_cfg.get("oracle_epochs", 25)),
                batch_size=int(oracle_cfg.get("batch_size", 16)),
                lr=float(oracle_cfg.get("oracle_lr", 5e-5)),
                seed=seed,
            )
        else:
            oracle = InternalUniMolOracle.load_from_dir(oracle_subdir_load, device=device)
    elif oracle_mode in ("internal_dpa4", "internal_hyff"):
        subdir_name = "internal_hyff_oracle" if oracle_mode == "internal_hyff" else "internal_dpa4_oracle"
        oracle_subdir_save = os.path.join(ckpt_save_dir, subdir_name)
        oracle_subdir_load = os.path.join(ckpt_load_dir, subdir_name)
        if mode in ["train", "train_and_predict"] and oracle_cfg.get("train_oracle", True):
            oracle = InternalDPA4PropertyOracle(
                dpa_guidance=dpa_guidance,
                output_dim=len(target_col_list),
                target_cols=target_col_list,
                n_ensemble_heads=int(oracle_cfg.get("n_ensemble_heads", 5)),
            )
            oracle.fit_from_dataset(
                mols_3d,
                targets_2d,
                save_dir=oracle_subdir_save,
                epochs=int(oracle_cfg.get("oracle_epochs", 35)),
                lr=float(oracle_cfg.get("oracle_lr", 2e-4)),
                seed=seed,
            )
        else:
            oracle = InternalDPA4PropertyOracle.load_from_dir(oracle_subdir_load, dpa_guidance=dpa_guidance)
    else:
        raise ValueError(f"Unknown oracle_mode: {oracle_mode}")

    # 2. Build / Train / Load ConditionalUni3DAR Generator
    gen_cfg = cfg.get("generator_config", {})
    uni3dar = ConditionalUni3DAR(
        d_model=int(gen_cfg.get("d_model", 256)),
        n_heads=int(gen_cfg.get("n_heads", 8)),
        n_layers=int(gen_cfg.get("n_layers", 12)),
    ).to(device)

    sft_mols: List[Mol3DInstance] = []
    if mols_3d and targets_2d is not None:
        for m, yv in zip(mols_3d, targets_2d):
            inst = rdkit_mol_to_mol3d_instance(m, yv)
            if inst is not None:
                sft_mols.append(inst)

    if mode in ["train", "train_and_predict"]:
        sft_epochs = int(gen_cfg.get("sft_epochs", 20))
        sft_lr = float(gen_cfg.get("sft_lr", 3e-4))
        sft_seqs = [encode_mol_to_octree_tokens(m) for m in sft_mols]
        opt_sft = torch.optim.AdamW(uni3dar.parameters(), lr=sft_lr, weight_decay=1e-2)
        rng = np.random.default_rng(seed)

        for _ in range(sft_epochs):
            uni3dar.train()
            perm = rng.permutation(len(sft_seqs))
            for i in range(0, len(perm), 8):
                b_idx = perm[i : i + 8]
                max_l = max(len(sft_seqs[j]) for j in b_idx)
                tok_b = torch.zeros((len(b_idx), max_l), dtype=torch.long, device=device)
                cond_b = torch.zeros((len(b_idx), 7), dtype=torch.float32, device=device)
                y_true_b = torch.zeros((len(b_idx), 3), dtype=torch.float32, device=device)
                for bi, j in enumerate(b_idx):
                    seq = sft_seqs[j]
                    tok_b[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                    y0_norm = (float(sft_mols[j].props[0]) - prop_mean) / max(prop_std, 1e-6)
                    d1, d2 = float(rng.uniform(0.05, 0.20)), float(rng.uniform(0.05, 0.20))
                    cond_b[bi, 0] = y0_norm - d1
                    cond_b[bi, 1] = y0_norm + d2
                    cond_b[bi, 6] = 1.0
                    y_true_b[bi, 0] = y0_norm

                logits, pred_p = uni3dar(tok_b[:, :-1], cond_b)
                loss_ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tok_b[:, 1:].reshape(-1), ignore_index=0
                )
                loss_p = F.smooth_l1_loss(pred_p[:, 0], y_true_b[:, 0])
                loss = loss_ce + 0.5 * loss_p
                opt_sft.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(uni3dar.parameters(), 1.0)
                opt_sft.step()

        sft_ckpt_path = os.path.join(ckpt_save_dir, "uni3dar_sft.pt")
        torch.save(
            {"model_state_dict": uni3dar.state_dict(), "prop_mean": prop_mean, "prop_std": prop_std},
            sft_ckpt_path,
        )

        ref_model = copy.deepcopy(uni3dar).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        inf_cfg = cfg.get("inference_config", {})
        if "desired_labels" in inf_cfg and hasattr(oracle, "target_labels_to_prob_intervals"):
            raw_iv = oracle.target_labels_to_prob_intervals(inf_cfg["desired_labels"])
        else:
            raw_iv = inf_cfg.get("target_interval", [prop_mean + 0.5 * prop_std, prop_mean + 0.8 * prop_std])
        target_iv_tuple = (float(raw_iv[0]), float(raw_iv[1])) if isinstance(raw_iv, (list, tuple)) and len(raw_iv) == 2 and isinstance(raw_iv[0], (int, float)) else raw_iv

        grpo_info = train_gated_grpo_with_oracle(
            model=uni3dar,
            ref_model=ref_model,
            sft_mols=sft_mols,
            oracle=oracle,
            prop_mean=prop_mean,
            prop_std=prop_std,
            grpo_cfg=gen_cfg,
            target_interval_physical=target_iv_tuple,
            dpa_guidance=dpa_guidance,
            device=device,
        )
        grpo_ckpt_path = os.path.join(ckpt_save_dir, "uni3dar_grpo.pt")
        torch.save(
            {
                "model_state_dict": uni3dar.state_dict(),
                "prop_mean": prop_mean,
                "prop_std": prop_std,
                "grpo_info": grpo_info,
            },
            grpo_ckpt_path,
        )

    # 3. Inference / Targeted Interval Generation + Stage 5 DPA-4 Conformer Relaxation
    if mode in ["predict", "generate", "train_and_predict"]:
        ckpt_file = os.path.join(ckpt_load_dir, "uni3dar_grpo.pt")
        if not os.path.exists(ckpt_file):
            ckpt_file = os.path.join(ckpt_load_dir, "uni3dar_sft.pt")
        if os.path.exists(ckpt_file) and mode != "train_and_predict":
            ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
            uni3dar.load_state_dict(ckpt_data["model_state_dict"], strict=True)
            prop_mean = float(ckpt_data.get("prop_mean", prop_mean))
            prop_std = float(ckpt_data.get("prop_std", prop_std))

        inf_cfg = cfg.get("inference_config", {})
        if "desired_labels" in inf_cfg and hasattr(oracle, "target_labels_to_prob_intervals"):
            raw_iv = oracle.target_labels_to_prob_intervals(inf_cfg["desired_labels"])
        else:
            raw_iv = inf_cfg.get("target_interval", [620.0, 640.0])
        if isinstance(raw_iv, (list, tuple)) and len(raw_iv) == 2 and isinstance(raw_iv[0], (int, float)):
            lo_phys, hi_phys = float(raw_iv[0]), float(raw_iv[1])
            target_interval_arg: Any = (lo_phys, hi_phys)
        elif isinstance(raw_iv, dict):
            first_v = list(raw_iv.values())[0]
            lo_phys, hi_phys = float(first_v[0]), float(first_v[1])
            target_interval_arg = {k: (float(v[0]), float(v[1])) for k, v in raw_iv.items()}
        elif isinstance(raw_iv, list) and len(raw_iv) > 0 and isinstance(raw_iv[0], (list, tuple)):
            lo_phys, hi_phys = float(raw_iv[0][0]), float(raw_iv[0][1])
            target_interval_arg = [(float(v[0]), float(v[1])) if v is not None else None for v in raw_iv]
        else:
            lo_phys, hi_phys = 620.0, 640.0
            target_interval_arg = (lo_phys, hi_phys)

        if bool(inf_cfg.get("run_specialist_grpo_before_gen", False)) and len(sft_mols) > 0:
            ref_model = copy.deepcopy(uni3dar).eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            spec_cfg = dict(gen_cfg)
            spec_cfg["grpo_mode"] = "single_window_specialist"
            spec_cfg["grpo_steps"] = int(inf_cfg.get("specialist_grpo_steps", 25))
            train_gated_grpo_with_oracle(
                model=uni3dar,
                ref_model=ref_model,
                sft_mols=sft_mols,
                oracle=oracle,
                prop_mean=prop_mean,
                prop_std=prop_std,
                grpo_cfg=spec_cfg,
                target_interval_physical=target_interval_arg,
                dpa_guidance=dpa_guidance,
                device=device,
            )
            spec_ckpt = os.path.join(ckpt_save_dir, f"uni3dar_specialist_{int(lo_phys)}_{int(hi_phys)}.pt")
            spec_payload = {"model_state_dict": uni3dar.state_dict(), "prop_mean": prop_mean, "prop_std": prop_std}
            torch.save(spec_payload, spec_ckpt)
            if mode == "train_and_predict":
                torch.save(spec_payload, os.path.join(ckpt_save_dir, "uni3dar_grpo.pt"))

        uni3dar.eval()
        num_molecules = int(inf_cfg.get("num_molecules", 16))
        base_seed = int(inf_cfg.get("base_seed", 2026))
        sampling_temp = float(inf_cfg.get("temperature", 0.85))
        sampling_top_p = float(inf_cfg.get("top_p", 0.92))

        cond_raw = torch.zeros(7, dtype=torch.float32, device=device)
        cond_raw[0] = (lo_phys - prop_mean) / max(prop_std, 1e-6)
        cond_raw[1] = (hi_phys - prop_mean) / max(prop_std, 1e-6)
        cond_raw[6] = 1.0

        enable_stage5 = bool(stage5_cfg.get("enabled", True))
        relax_steps = int(stage5_cfg.get("relax_steps", 35))
        relax_step_size = float(stage5_cfg.get("step_size", 0.008))
        prop_grad_w = float(stage5_cfg.get("prop_grad_weight", 0.25))

        out_mols_rdkit: List[Chem.Mol] = []
        records: List[Dict[str, Any]] = []

        for mol_id in range(num_molecules):
            inst_raw, _ = sample_molecule_deterministic_by_id(
                model=uni3dar,
                cond_raw=cond_raw,
                mol_id=mol_id,
                base_seed=base_seed,
                temperature=sampling_temp,
                top_p=sampling_top_p,
                device=device,
            )
            if enable_stage5:
                inst_final, s5_info = relax_conformer_iso_property(
                    inst=inst_raw,
                    dpa_guidance=dpa_guidance,
                    oracle=oracle,
                    target_interval=target_interval_arg,
                    n_steps=relax_steps,
                    step_size=relax_step_size,
                    prop_grad_weight=prop_grad_w,
                    tangent_projection=True,
                )
            else:
                inst_final = inst_raw
                rd_tmp = mol3d_to_validated_rdkit_mol(inst_final)
                yp_a, ys_a = oracle.predict_3d_mols([rd_tmp])
                s5_info = {
                    "init_force_eva": 0.0,
                    "final_force_eva": 0.0,
                    "y_pred": float(np.squeeze(yp_a)),
                    "y_std": float(np.squeeze(ys_a)),
                }

            rd_mol = mol3d_to_validated_rdkit_mol(inst_final)
            y_pred_val = s5_info["y_pred"]
            y_std_val = s5_info["y_std"]
            y_scalar = float(y_pred_val[0]) if isinstance(y_pred_val, list) else float(y_pred_val)
            is_hit = int(lo_phys <= y_scalar <= hi_phys)

            rd_mol.SetProp("MolID", str(mol_id))
            rd_mol.SetProp("BaseSeed", str(base_seed))
            rd_mol.SetProp("OracleMode", oracle_mode)
            rd_mol.SetProp("DPA4Scale", dpa4_scale)
            rd_mol.SetProp("TargetInterval", f"{lo_phys}-{hi_phys}")
            rd_mol.SetProp("PredictedValue", str(y_pred_val))
            rd_mol.SetProp("PredictedStd", str(y_std_val))
            rd_mol.SetProp("IntervalHit", str(is_hit))
            rd_mol.SetProp("FinalForceNorm_eVA", f"{s5_info['final_force_eva']:.4f}")
            out_mols_rdkit.append(rd_mol)

            records.append(
                {
                    "mol_id": mol_id,
                    "base_seed": base_seed,
                    "smiles": inst_final.smiles,
                    "target_interval": f"[{lo_phys}, {hi_phys}]",
                    "y_pred": y_pred_val,
                    "y_std": y_std_val,
                    "is_hit": is_hit,
                    "init_force_eva": s5_info["init_force_eva"],
                    "final_force_eva": s5_info["final_force_eva"],
                }
            )

        df_out = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, "generated_molecules.csv")
        sdf_path = os.path.join(output_dir, "generated_molecules.sdf")
        summary_path = os.path.join(output_dir, "generation_summary.json")

        df_out.to_csv(csv_path, index=False)
        writer = Chem.SDWriter(sdf_path)
        for m in out_mols_rdkit:
            writer.write(m)
        writer.close()

        summary = {
            "mode": mode,
            "oracle_mode": oracle_mode,
            "dpa4_scale": dpa4_scale,
            "target_interval": [lo_phys, hi_phys],
            "num_molecules": num_molecules,
            "base_seed": base_seed,
            "hit_rate": float(df_out["is_hit"].mean()) if len(df_out) > 0 else 0.0,
            "mean_final_force_eva": float(df_out["final_force_eva"].mean()) if len(df_out) > 0 else 0.0,
            "output_csv": csv_path,
            "output_sdf": sdf_path,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if hasattr(dpa_guidance, "close"):
            dpa_guidance.close()
        print(f"[ORBIT-3D Complete] HitRate={summary['hit_rate']*100:.1f}%, saved to {output_dir}")
        return summary

    if hasattr(dpa_guidance, "close"):
        dpa_guidance.close()
    return {"mode": mode, "oracle_mode": oracle_mode, "checkpoint_save_dir": ckpt_save_dir}
