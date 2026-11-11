"""
Unified Config-Driven Pipeline Orchestrator (`runner.py`)
=========================================================
Executes `train`, `predict` (`generate`), or `train_and_predict` workflows from a single `config.json`.
"""

from __future__ import annotations

import copy
import hashlib
from ..constraints import TargetIntervals
from ..generator.candidate_selection import select_candidates
from ..oracles.fingerprint_oracle import FingerprintOracle
from ..utils.dataset_io import write_precise_sdf
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

    if cfg.get("generator_config", {}).get("backend") == "uni3dar_octree":
        from .octree_runner import run_octree_from_config

        return run_octree_from_config(cfg, config_path)

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

    if mode not in {"train", "predict", "generate", "train_and_predict"}:
        raise ValueError("Unknown execution mode")
    if mode in {"train", "train_and_predict"} and not os.path.isfile(train_data_path):
        raise FileNotFoundError(train_data_path)
    if mode in {"predict", "generate"} and not any(
        os.path.isfile(os.path.join(ckpt_load_dir, n)) for n in ["uni3dar_grpo.pt", "uni3dar_sft.pt"]
    ):
        raise FileNotFoundError("No trained generator checkpoint found")
    if cfg.get("generator_config", {}).get("backend", "dense_grid_v2") != "dense_grid_v2":
        raise ValueError("This CLI supports dense_grid_v2; upstream octree is a separate backend")

    os.makedirs(ckpt_save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    oracle_cfg = cfg.get("oracle_config", {})
    oracle_mode = str(oracle_cfg.get("oracle_mode", "external_uniqsar")).lower()
    target_cols = oracle_cfg.get("target_cols", "TARGET")
    smiles_col = oracle_cfg.get("smiles_col", "SMILES")

    stage5_cfg = cfg.get("stage5_relax_config", {})
    if stage5_cfg.get("charge", 0) != 0 or stage5_cfg.get("multiplicity", 1) != 1:
        raise ValueError("dense_grid_v2 generation currently supports neutral singlets only")
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
        prop_mean = np.mean(targets_2d, axis=0)
        prop_std = np.std(targets_2d, axis=0) + 1e-6
    else:
        prop_mean = float(cfg.get("dataset_stats", {}).get("prop_mean", 540.0))
        prop_std = float(cfg.get("dataset_stats", {}).get("prop_std", 120.0))
        target_col_list = (
            [c.strip() for c in target_cols.split(",")] if isinstance(target_cols, str) else list(target_cols)
        )

    if oracle_mode == "fingerprint":
        oracle = FingerprintOracle.load_from_dir(mounts["external_fingerprint_dir"])
    elif oracle_mode == "external_uniqsar":
        if not ext_uniqsar_dir or not os.path.exists(ext_uniqsar_dir):
            raise FileNotFoundError(f"external_uniqsar_dir not found: {ext_uniqsar_dir}")
        oracle = UniQSARTrainedBridge(
            model_dir=ext_uniqsar_dir,
            device=device,
            infer_choices=oracle_cfg.get("infer_choices", None),
        )
        if hasattr(oracle.scaler, "mean_"):
            prop_mean = np.asarray(oracle.scaler.mean_).reshape(-1)
            prop_std = np.asarray(oracle.scaler.scale_).reshape(-1)
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

    if list(oracle.target_cols) != target_col_list:
        raise ValueError("Oracle target names/order do not match requested targets")

    # 2. Build / Train / Load ConditionalUni3DAR Generator
    gen_cfg = cfg.get("generator_config", {})
    uni3dar = ConditionalUni3DAR(
        d_model=int(gen_cfg.get("d_model", 256)),
        n_heads=int(gen_cfg.get("n_heads", 8)),
        n_layers=int(gen_cfg.get("n_layers", 12)),
        cond_dim=4 * len(target_col_list),
        output_dim=len(target_col_list),
        max_atoms=int(gen_cfg.get("max_atoms", 40)),
        max_seq_len=int(gen_cfg.get("max_seq_len", 256)),
    ).to(device)

    model_meta = {
        "backend": "dense_grid_v2",
        "d_model": uni3dar.d_model,
        "n_layers": len(uni3dar.blocks),
        "n_heads": uni3dar.blocks[0].attn.num_heads,
        "targets": target_col_list,
        "max_atoms": uni3dar.max_atoms,
        "max_seq_len": uni3dar.max_seq_len,
        "coordinate_bounds": [-6.4, 6.4],
        "coordinate_bins": 256,
        "condition_schema": "bounds_masks_v2",
    }
    sft_mols: List[Mol3DInstance] = []
    if mols_3d and targets_2d is not None:
        for m, yv in zip(mols_3d, targets_2d):
            inst = rdkit_mol_to_mol3d_instance(m, yv)
            if inst is not None:
                sft_mols.append(inst)

    if mode in ["train", "train_and_predict"]:
        if not sft_mols:
            raise ValueError("No usable SFT molecules")
        sft_epochs = int(gen_cfg.get("sft_epochs", 20))
        sft_lr = float(gen_cfg.get("sft_lr", 3e-4))
        sft_seqs = [encode_mol_to_octree_tokens(m, uni3dar.max_atoms) for m in sft_mols]
        opt_sft = torch.optim.AdamW(uni3dar.parameters(), lr=sft_lr, weight_decay=1e-2)
        rng = np.random.default_rng(seed)

        batch_size = int(gen_cfg.get("batch_size", 8))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for epoch in range(sft_epochs):
            epoch_losses = []
            uni3dar.train()
            perm = rng.permutation(len(sft_seqs))
            for i in range(0, len(perm), batch_size):
                b_idx = perm[i : i + batch_size]
                max_l = max(len(sft_seqs[j]) for j in b_idx)
                tok_b = torch.zeros((len(b_idx), max_l), dtype=torch.long, device=device)
                cond_b = torch.zeros(
                    (len(b_idx), 4 * len(target_col_list)), dtype=torch.float32, device=device
                )
                y_true_b = torch.zeros((len(b_idx), len(target_col_list)), dtype=torch.float32, device=device)
                for bi, j in enumerate(b_idx):
                    seq = sft_seqs[j]
                    tok_b[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                    y_norm = (sft_mols[j].props - prop_mean) / prop_std
                    d1, d2 = (
                        rng.uniform(0.05, 0.2, len(target_col_list)),
                        rng.uniform(0.05, 0.2, len(target_col_list)),
                    )
                    iv = TargetIntervals.parse(
                        dict(
                            zip(
                                target_col_list,
                                zip(sft_mols[j].props - d1 * prop_std, sft_mols[j].props + d2 * prop_std),
                            )
                        ),
                        target_col_list,
                    )
                    cond_b[bi] = torch.tensor(iv.condition(prop_mean, prop_std), device=device)
                    y_true_b[bi] = torch.tensor(y_norm, device=device)

                logits, _ = uni3dar(tok_b[:, :-1], cond_b)
                loss_ce = F.cross_entropy(
                    uni3dar.constrained_logits(logits).flatten(0, 1), tok_b[:, 1:].flatten(), ignore_index=0
                )
                _, pred_p = uni3dar(tok_b, torch.zeros_like(cond_b))
                loss_p = F.smooth_l1_loss(pred_p, y_true_b)

                loss = loss_ce + 0.5 * loss_p
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"SFT diverged at epoch {epoch + 1}, batch {i // batch_size + 1}"
                    )
                opt_sft.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(uni3dar.parameters(), 1.0)
                opt_sft.step()
                epoch_losses.append(float(loss.detach()))
            print(f"SFT epoch {epoch + 1}/{sft_epochs} loss {np.mean(epoch_losses):.5f}", flush=True)

        sft_ckpt_path = os.path.join(ckpt_save_dir, "uni3dar_sft.pt")
        torch.save(
            {
                "model_state_dict": uni3dar.state_dict(),
                "model_meta": model_meta,
                "prop_mean": prop_mean,
                "prop_std": prop_std,
            },
            sft_ckpt_path,
        )

        ref_model = copy.deepcopy(uni3dar).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        inf_cfg = cfg.get("inference_config", {})
        if "desired_labels" in inf_cfg and hasattr(oracle, "target_labels_to_prob_intervals"):
            raw_iv = oracle.target_labels_to_prob_intervals(inf_cfg["desired_labels"])
        else:
            raw_iv = inf_cfg.get(
                "target_interval",
                dict(zip(target_col_list, zip(prop_mean + 0.5 * prop_std, prop_mean + 0.8 * prop_std))),
            )
        target_iv_tuple = (
            (float(raw_iv[0]), float(raw_iv[1]))
            if isinstance(raw_iv, (list, tuple)) and len(raw_iv) == 2 and isinstance(raw_iv[0], (int, float))
            else raw_iv
        )

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
                "model_meta": model_meta,
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
            if ckpt_data.get("model_meta") != model_meta:
                raise ValueError(
                    "Incompatible or unversioned generator checkpoint; retraining/migration required"
                )
            uni3dar.load_state_dict(ckpt_data["model_state_dict"], strict=True)
            prop_mean = np.asarray(ckpt_data["prop_mean"])
            prop_std = np.asarray(ckpt_data["prop_std"])

        inf_cfg = cfg.get("inference_config", {})
        if "desired_labels" in inf_cfg and hasattr(oracle, "target_labels_to_prob_intervals"):
            raw_iv = oracle.target_labels_to_prob_intervals(inf_cfg["desired_labels"])
        else:
            raw_iv = inf_cfg.get("target_interval", [620.0, 640.0])
        intervals = TargetIntervals.parse(raw_iv, target_col_list)
        target_interval_arg = dict(zip(target_col_list, intervals.bounds))
        first = next(iv for iv in intervals.bounds if iv is not None)
        lo_phys, hi_phys = first

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
            spec_ckpt = os.path.join(ckpt_save_dir, "uni3dar_specialist.pt")
            spec_payload = {
                "model_state_dict": uni3dar.state_dict(),
                "model_meta": model_meta,
                "prop_mean": prop_mean,
                "prop_std": prop_std,
            }
            torch.save(spec_payload, spec_ckpt)
            if mode == "train_and_predict":
                torch.save(spec_payload, os.path.join(ckpt_save_dir, "uni3dar_grpo.pt"))

        uni3dar.eval()
        num_molecules = int(inf_cfg.get("num_molecules", 16))
        base_seed = int(inf_cfg.get("base_seed", 2026))
        sampling_temp = float(inf_cfg.get("temperature", 0.85))
        sampling_top_p = float(inf_cfg.get("top_p", 0.92))

        cond_raw = torch.tensor(intervals.condition(prop_mean, prop_std), device=device)

        enable_stage5 = bool(stage5_cfg.get("enabled", True))
        relax_steps = int(stage5_cfg.get("relax_steps", 35))
        relax_step_size = float(stage5_cfg.get("step_size", 0.008))
        prop_grad_w = float(stage5_cfg.get("prop_grad_weight", 0.25))
        force_threshold = inf_cfg.get("max_rms_force_eva")
        if force_threshold is not None and (not np.isfinite(force_threshold) or force_threshold <= 0):
            raise ValueError("Force threshold must be finite and positive")

        out_mols_rdkit: List[Chem.Mol] = []
        selected, records = select_candidates(
            sample_molecule_deterministic_by_id,
            uni3dar,
            cond_raw,
            intervals,
            prop_mean,
            prop_std,
            num_molecules,
            pool_factor=int(inf_cfg.get("candidate_pool_factor", 1)),
            selection_mode=str(inf_cfg.get("selection_mode", "self_rank")),
            deduplicate=bool(inf_cfg.get("deduplicate", False)),
            base_seed=base_seed,
            temperature=sampling_temp,
            top_p=sampling_top_p,
            device=device,
        )

        for candidate in selected:
            mol_id = candidate.mol_id
            try:
                inst_raw = candidate.instance
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
                        "init_force_eva": None,
                        "final_force_eva": None,
                        "y_pred": np.asarray(yp_a).reshape(-1).tolist(),
                        "y_std": np.asarray(ys_a).reshape(-1).tolist(),
                    }

                rd_mol = mol3d_to_validated_rdkit_mol(inst_final)
                y_pred_val = s5_info["y_pred"]
                y_std_val = s5_info["y_std"]
                if not np.isfinite(y_pred_val).all() or not np.isfinite(y_std_val).all():
                    raise FloatingPointError("Nonfinite exported property prediction")
                is_hit = int(intervals.hit(y_pred_val))
                joint_success = int(
                    bool(
                        is_hit
                        and enable_stage5
                        and force_threshold is not None
                        and s5_info["final_force_eva"] <= force_threshold
                    )
                )

                rd_mol.SetProp("MolID", str(mol_id))
                rd_mol.SetProp("BaseSeed", str(base_seed))
                rd_mol.SetProp("OracleMode", oracle_mode)
                rd_mol.SetProp("DPA4Scale", dpa4_scale)
                rd_mol.SetProp("TargetInterval", str(target_interval_arg))
                rd_mol.SetProp("PredictedValue", str(y_pred_val))
                rd_mol.SetProp("PredictedStd", str(y_std_val))
                rd_mol.SetProp("IntervalHit", str(is_hit))
                rd_mol.SetProp("JointSuccess", str(joint_success))
                rd_mol.SetProp("FinalForceNorm_eVA", str(s5_info["final_force_eva"]))
                out_mols_rdkit.append(rd_mol)

                records[mol_id].update(
                    {
                        "mol_id": mol_id,
                        "base_seed": base_seed,
                        "smiles": inst_final.smiles,
                        "target_interval": str(target_interval_arg),
                        "status": "valid",
                        "y_pred": y_pred_val,
                        "y_std": y_std_val,
                        "is_hit": is_hit,
                        "joint_success": joint_success,
                        "init_force_eva": s5_info["init_force_eva"],
                        "final_force_eva": s5_info["final_force_eva"],
                    }
                )
            except (ValueError, FloatingPointError) as exc:
                records[mol_id].update(
                    {
                        "mol_id": mol_id,
                        "base_seed": base_seed,
                        "status": "failed",
                        "failure_reason": str(exc),
                        "is_hit": 0,
                    }
                )

        df_out = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, "generated_molecules.csv")
        sdf_path = os.path.join(output_dir, "generated_molecules.sdf")
        summary_path = os.path.join(output_dir, "generation_summary.json")

        df_out.to_csv(csv_path, index=False)
        write_precise_sdf(sdf_path, out_mols_rdkit)
        qualified_path = os.path.join(output_dir, "qualified_molecules.sdf")
        write_precise_sdf(qualified_path, [m for m in out_mols_rdkit if m.GetProp("JointSuccess") == "1"])
        exported = (
            [m for m in Chem.SDMolSupplier(sdf_path, removeHs=False) if m is not None]
            if out_mols_rdkit
            else []
        )
        if len(exported) != len(out_mols_rdkit):
            raise ValueError("SDF export lost records")
        export_checks = []
        for before, after in zip(out_mols_rdkit, exported):
            if Chem.MolToSmiles(before) != Chem.MolToSmiles(after):
                raise ValueError("SDF export changed molecular graph")
            np.testing.assert_allclose(
                before.GetConformer().GetPositions(), after.GetConformer().GetPositions(), atol=1e-9, rtol=0
            )
            expected = np.asarray(json.loads(before.GetProp("PredictedValue"))).reshape(-1)
            actual, _ = oracle.predict_3d_mols([after])
            actual = np.asarray(actual).reshape(-1)
            np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-6)
            export_checks.append(
                {
                    "mol_id": int(before.GetProp("MolID")),
                    "prediction_max_difference": float(np.abs(expected - actual).max()),
                }
            )
        with open(os.path.join(output_dir, "export_validation.json"), "w") as export_file:
            json.dump({"records": export_checks, "passed": True}, export_file, indent=2)

        summary = {
            "mode": mode,
            "oracle_mode": oracle_mode,
            "dpa4_scale": dpa4_scale,
            "target_interval": str(target_interval_arg),
            "backend": "dense_grid_v2",
            "valid_count": len(out_mols_rdkit),
            "failed_count": sum(r["status"] == "failed" for r in records),
            "num_attempted": len(records),
            "force_threshold_eva": force_threshold,
            "retained_hit_rate": sum(r.get("is_hit", 0) for r in records) / len(out_mols_rdkit)
            if out_mols_rdkit
            else 0.0,
            "joint_success_rate": sum(r.get("joint_success", 0) for r in records) / len(records)
            if force_threshold is not None and enable_stage5
            else None,
            "qualified_output_sdf": qualified_path,
            "raw_valid_count": sum(r.get("raw_valid", False) for r in records),
            "not_selected_count": sum(r["status"] in {"not_selected", "duplicate"} for r in records),
            "status": "completed" if out_mols_rdkit else "no_valid_outputs",
            "force_evaluated": enable_stage5,
            "config_sha256": hashlib.sha256(open(config_path, "rb").read()).hexdigest(),
            "num_molecules": num_molecules,
            "base_seed": base_seed,
            "hit_rate": float(df_out["is_hit"].mean()) if len(df_out) > 0 else 0.0,
            "mean_final_force_eva": float(df_out["final_force_eva"].mean())
            if "final_force_eva" in df_out and df_out["final_force_eva"].notna().any()
            else None,
            "output_csv": csv_path,
            "output_sdf": sdf_path,
        }
        used_checkpoint = (
            os.path.join(ckpt_save_dir, "uni3dar_grpo.pt") if mode == "train_and_predict" else ckpt_file
        )
        summary["generator_checkpoint_sha256"] = hashlib.sha256(
            open(used_checkpoint, "rb").read()
        ).hexdigest()
        with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as manifest:
            json.dump(cfg, manifest, indent=2, allow_nan=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
        if hasattr(dpa_guidance, "close"):
            dpa_guidance.close()
        print(f"[ORBIT-3D Complete] HitRate={summary['hit_rate'] * 100:.1f}%, saved to {output_dir}")
        return summary

    if hasattr(dpa_guidance, "close"):
        dpa_guidance.close()
    return {"mode": mode, "oracle_mode": oracle_mode, "checkpoint_save_dir": ckpt_save_dir}

    if mode == "train":
        if hasattr(dpa_guidance, "close"):
            dpa_guidance.close()
        return {
            "mode": "train",
            "checkpoint_save_dir": ckpt_save_dir,
            "model_meta": model_meta,
            "grpo_info": grpo_info,
        }
