"""Unified 5-Dimensional `Uni-Loom` Evaluator (`1,792`-proposal protocol + AuditOracle + PotentialSlot).

Directly comparable to E21/E22 (`7 quantiles × 2 seeds [24411, 24412] × 128 attempts = 1,792 proposals`)
while reporting all 5 `Uni-Loom` metrics:
  1. Calibration slope, intercept, R^2 across `P10..P90`
  2. Within-target spread `sigma_in` (eV) & conditional MAE
  3. Structural validity, unseen formal-charge rate, and zero-step `F_max` (`PotentialSlot`)
  4. End-to-end `±0.25 eV` hit yield over all attempted proposals (+ novelty vs source)
  5. Cross-foundation `AuditOracleSlot` (`E15` DFT-calibrated `DPA-4`) anti-reward-hacking gap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
import numpy as np
import torch
from rdkit import Chem

from loom.generator.molecular_state import mol_to_clean_smiles
from loom.generator.octree_backend import trajectory_to_instance
from loom.oracles.unimol2_task import B_WEIGHT_SHA256
from loom.weaving import (
    AuditOracleSlot,
    LoomUni3DARGenerator,
    PotentialSlot,
    RewardOracleSlot,
    compute_loom_metrics,
    formal_charge_signature,
    is_clean_organic_charge_state,
)


def extract_source_sets(source_train_sdf: Path) -> Tuple[set[str], FrozenSet[Tuple[Tuple[str, int], ...]]]:
    graphs = set()
    charges = set()
    for m in Chem.SDMolSupplier(str(source_train_sdf), removeHs=False):
        if m is not None:
            graphs.add(mol_to_clean_smiles(m))
            charges.add(formal_charge_signature(m))
    return graphs, frozenset(charges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--source-train-sdf", type=Path, required=True)
    parser.add_argument("--reward-oracle-dir", type=Path, required=True)
    parser.add_argument("--audit-oracle-dir", type=Path, default=None)
    parser.add_argument("--potential-backend", choices=["none", "dpa4", "hyff"], default="dpa4")
    parser.add_argument("--potential-weights", type=Path, default=None)
    parser.add_argument("--dpa4-forcefield", type=Path, default=None)
    parser.add_argument("--target", choices=["homo", "lumo"], default="homo")
    parser.add_argument("--quantiles", type=int, nargs="+", default=[10, 25, 35, 50, 65, 75, 90])
    parser.add_argument("--seeds", type=int, nargs="+", default=[24411, 24412])
    parser.add_argument("--attempts-per-seed", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cfg-scales", type=float, nargs="+", default=[1.0, 1.75])
    parser.add_argument("--xyz-cfg-scale", type=float, default=1.0)
    parser.add_argument("--check-relaxation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-proposals-npz", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.set_num_threads(4)
    gen, ckpt_meta = LoomUni3DARGenerator.load_checkpoint(args.checkpoint, device=args.device, batch_size=args.batch_size)
    gen.eval()

    targets_cfg = json.loads(args.targets_json.read_text())
    q_points = targets_cfg["single_target_points"][args.target]
    prop_idx = gen.target_cols.index(args.target)

    print(f"[Loom-Eval] Loading source reference sets from {args.source_train_sdf}...", flush=True)
    source_graphs, seen_charges = extract_source_sets(args.source_train_sdf)

    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )
    rew_col_idx = reward_slot.target_cols.index(args.target)

    audit_slot: Optional[AuditOracleSlot] = None
    aud_col_idx = 0
    if args.audit_oracle_dir is not None and args.audit_oracle_dir.exists():
        audit_slot = AuditOracleSlot(
            backend="dpa4",
            checkpoint_dir=args.audit_oracle_dir,
            device=args.device,
            forcefield_path=str(args.dpa4_forcefield) if args.dpa4_forcefield else None,
        )
        aud_col_idx = audit_slot.target_cols.index(args.target)

    pot_slot: Optional[PotentialSlot] = None
    if args.potential_backend != "none" and args.potential_weights is not None:
        pot_slot = PotentialSlot(
            backend=args.potential_backend,
            weights=str(args.potential_weights),
            device=args.device,
        )

    allowed_elems = tuple(gen.args.allow_atoms.split(","))
    device = torch.device(args.device)

    all_cfg_reports: Dict[str, Any] = {}
    best_cfg_key = None
    best_hits = -1
    best_arrays: Optional[Dict[str, np.ndarray]] = None

    for cfg_val in args.cfg_scales:
        t_cfg0 = time.perf_counter()
        req_list: List[float] = []
        mols_all: List[Optional[Chem.Mol]] = []
        legal_list: List[bool] = []
        smiles_list: List[Optional[str]] = []
        unseen_charge_list: List[bool] = []
        gen_seconds = 0.0

        for q in args.quantiles:
            goal_eV = float(q_points[str(q)])
            for sd in args.seeds:
                n_rem = args.attempts_per_seed
                offset = 0
                while n_rem > 0:
                    cur_b = min(args.batch_size, n_rem)
                    t_mat = np.zeros((cur_b, gen.num_props), dtype=np.float32)
                    t_mat[:, prop_idx] = goal_eV
                    t_norm = gen.normalize_targets(t_mat)
                    m_act = torch.zeros((cur_b, gen.num_props), dtype=torch.bool, device=device)
                    m_act[:, prop_idx] = True

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    tg0 = time.perf_counter()
                    trajs = gen.sample(
                        t_norm,
                        m_act,
                        cfg_scale=cfg_val,
                        xyz_cfg_scale=args.xyz_cfg_scale,
                        temperatures=(0.85, 0.30, 0.25),
                        seed=int(sd) * 10007 + int(q) * 131 + offset,
                        use_bf16=(device.type == "cuda"),
                        electron_parity=0,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    gen_seconds += time.perf_counter() - tg0

                    for tr in trajs:
                        req_list.append(goal_eV)
                        try:
                            inst = trajectory_to_instance(tr, element_table=allowed_elems)
                            mol = inst.graph
                            mols_all.append(mol)
                            legal_list.append(True)
                            smiles_list.append(inst.smiles)
                            unseen_charge_list.append(not is_clean_organic_charge_state(mol))
                        except ValueError:
                            mols_all.append(None)
                            legal_list.append(False)
                            smiles_list.append(None)
                            unseen_charge_list.append(False)

                    n_rem -= cur_b
                    offset += cur_b

        # Batched RewardOracle evaluation
        rew_preds_full, _, rew_ev_mask, _ = reward_slot.evaluate_candidates(mols_all, batch_size=32)
        rew_preds_1d = rew_preds_full[:, rew_col_idx]

        # Batched AuditOracle evaluation
        aud_preds_1d = None
        aud_ev_mask = None
        if audit_slot is not None:
            aud_full, _, aud_ev_mask, _ = audit_slot.evaluate_candidates(mols_all, batch_size=32)
            aud_preds_1d = aud_full[:, aud_col_idx]

        # Batched PotentialSlot zero-step force evaluation
        f_max_all = np.full(len(mols_all), np.nan, dtype=np.float64)
        relax_summary = None
        valid_indices = [i for i, m in enumerate(mols_all) if m is not None]
        valid_mols = [mols_all[i] for i in valid_indices]  # type: ignore[misc]
        if pot_slot is not None and valid_mols:
            p_out = pot_slot.evaluate_molecules(valid_mols, chunk_size=32)
            for loc_i, glob_i in enumerate(valid_indices):
                f_max_all[glob_i] = p_out["f_max_eV_A"][loc_i]

            if args.check_relaxation:
                # Subsample up to 256 valid molecules evenly to audit zero-step vs 15-step MLIP relaxation
                step_sub = max(len(valid_mols) // 256, 1)
                sub_idx = valid_indices[::step_sub][:256]
                sub_mols = [mols_all[i] for i in sub_idx]  # type: ignore[misc]
                rel = pot_slot.relax_molecules(sub_mols, n_steps=15, chunk_size=32)
                preserved_mask = rel["graph_preserved"]
                rel_mols_eval = [rm if ok else None for rm, ok in zip(rel["relaxed_mols"], preserved_mask)]
                rel_rew_full, _, rel_ev, _ = reward_slot.evaluate_candidates(rel_mols_eval, batch_size=32)
                sub_orig_rew = rew_preds_1d[sub_idx]
                sub_req = np.asarray(req_list)[sub_idx]
                both_rel = rew_ev_mask[sub_idx] & rel_ev
                post_relax_aud_mae = None
                post_relax_rew_aud_gap = None
                post_relax_clean_rew_aud_gap = None
                if audit_slot is not None:
                    rel_aud_full, _, rel_aud_ev, _ = audit_slot.evaluate_candidates(rel_mols_eval, batch_size=32)
                    both_aud = both_rel & rel_aud_ev
                    sub_clean = np.asarray([not unseen_charge_list[i] for i in sub_idx], dtype=bool)
                    both_aud_clean = both_aud & sub_clean
                    if both_aud.any():
                        post_relax_aud_mae = float(np.mean(np.abs(rel_aud_full[both_aud, aud_col_idx] - sub_req[both_aud])))
                        post_relax_rew_aud_gap = float(
                            np.mean(np.abs(rel_rew_full[both_aud, rew_col_idx] - rel_aud_full[both_aud, aud_col_idx]))
                        )
                    if both_aud_clean.any():
                        post_relax_clean_rew_aud_gap = float(
                            np.mean(np.abs(rel_rew_full[both_aud_clean, rew_col_idx] - rel_aud_full[both_aud_clean, aud_col_idx]))
                        )
                relax_summary = {
                    "audited_subset": len(sub_mols),
                    "graph_preserved_rate": float(np.mean(preserved_mask)),
                    "median_strain_drop_eV": float(np.median(rel["init_energy_eV"] - rel["final_energy_eV"])),
                    "init_f_max_median_eV_A": float(np.median(f_max_all[sub_idx])),
                    "relaxed_f_max_median_eV_A": float(np.median(rel["final_f_max_eV_A"])),
                    "zero_step_mae_on_subset_eV": float(np.mean(np.abs(sub_orig_rew[both_rel] - sub_req[both_rel]))) if both_rel.any() else None,
                    "post_relax_mae_on_subset_eV": float(np.mean(np.abs(rel_rew_full[both_rel, rew_col_idx] - sub_req[both_rel]))) if both_rel.any() else None,
                    "mean_abs_pred_shift_from_relaxation_eV": float(np.mean(np.abs(rel_rew_full[both_rel, rew_col_idx] - sub_orig_rew[both_rel]))) if both_rel.any() else None,
                    "post_relax_audit_mae_on_subset_eV": post_relax_aud_mae,
                    "post_relax_reward_audit_disagreement_eV": post_relax_rew_aud_gap,
                    "post_relax_clean_reward_audit_disagreement_eV": post_relax_clean_rew_aud_gap,
                }

        loom_m = compute_loom_metrics(
            requested_targets_eV=req_list,
            reward_predictions_eV=rew_preds_1d,
            legal_mask=legal_list,
            evaluated_mask=rew_ev_mask,
            audit_predictions_eV=aud_preds_1d,
            audit_evaluated_mask=aud_ev_mask,
            f_max_eV_A=f_max_all,
            unseen_charge_mask=unseen_charge_list,
            tolerance_eV=0.25,
        )

        # Novelty & uniqueness among hits + zwitterion-clean hits
        hit_smiles = [
            smiles_list[i]
            for i in range(len(req_list))
            if rew_ev_mask[i] and smiles_list[i] is not None and abs(rew_preds_1d[i] - req_list[i]) <= 0.25
        ]
        clean_hit_count = sum(
            1
            for i in range(len(req_list))
            if rew_ev_mask[i] and not unseen_charge_list[i] and abs(rew_preds_1d[i] - req_list[i]) <= 0.25
        )
        unique_hits = set(hit_smiles)
        novel_unique_hits = unique_hits - source_graphs

        loom_m.update({
            "cfg_scale": cfg_val,
            "xyz_cfg_scale": args.xyz_cfg_scale,
            "clean_hits": int(clean_hit_count),
            "clean_hit_yield": float(clean_hit_count / max(len(req_list), 1)),
            "unique_hit_graphs": len(unique_hits),
            "novel_unique_hit_graphs": len(novel_unique_hits),
            "generation_seconds": gen_seconds,
            "generation_mol_per_sec": len(req_list) / max(gen_seconds, 1e-6),
            "total_eval_wall_seconds": time.perf_counter() - t_cfg0,
            "relaxation_audit": relax_summary,
        })
        cfg_key = f"cfg_{cfg_val:.2f}"
        all_cfg_reports[cfg_key] = loom_m
        print(
            f"[Loom-Eval {args.target} CFG={cfg_val:.2f}] "
            f"Hits={loom_m['hits']}/{loom_m['attempted']} ({loom_m['hit_yield']*100:.2f}%), "
            f"Slope={loom_m['calibration_slope']:.3f}, Sigma_in={loom_m['within_target_std_eV']:.3f} eV, "
            f"MAE={loom_m['conditional_mae_eV']:.3f} eV, Legal={loom_m['legal_rate']*100:.1f}%, "
            f"UnseenCharge={(loom_m['unseen_formal_charge_rate_among_legal'] or 0)*100:.1f}%, "
            f"Fmax_med={loom_m['zero_step_f_max_median_eV_A'] or -1:.2f} eV/A, "
            f"AuditMAE={(loom_m['audit'] or {}).get('audit_mae_eV', float('nan')):.3f} eV "
            f"({loom_m['generation_mol_per_sec']:.1f} mol/s)",
            flush=True,
        )

        if loom_m["hits"] > best_hits:
            best_hits = loom_m["hits"]
            best_cfg_key = cfg_key
            best_arrays = {
                "requested_eV": np.asarray(req_list, dtype=np.float64),
                "reward_preds_full_eV": rew_preds_full,
                "reward_pred_1d_eV": rew_preds_1d,
                "evaluated_mask": rew_ev_mask,
                "legal_mask": np.asarray(legal_list, dtype=bool),
                "f_max_eV_A": f_max_all,
                "audit_pred_1d_eV": aud_preds_1d if aud_preds_1d is not None else np.full(len(req_list), np.nan),
            }

    out_payload = {
        "checkpoint": str(args.checkpoint),
        "target": args.target,
        "quantiles": args.quantiles,
        "seeds": args.seeds,
        "attempts_per_seed": args.attempts_per_seed,
        "best_cfg_key": best_cfg_key,
        "by_cfg": all_cfg_reports,
        "checkpoint_metadata": ckpt_meta.get("metadata", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_payload, indent=2) + "\n")
    if args.save_proposals_npz is not None and best_arrays is not None:
        args.save_proposals_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_proposals_npz, **best_arrays)


if __name__ == "__main__":
    main()
