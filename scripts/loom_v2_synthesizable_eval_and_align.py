"""`loom v2.0` Phase 1 & Phase 2 End-to-End Validation Runner:
  1. Dual-Track Exotic Zwitterion Elimination (`Neutral Valence Guard` + `HeavyAtomValenceHead` + `ValenceReconciler`)
     reporting BOTH `Raw Point-Cloud Metrics` and `Valence-Reconciled Metrics` side-by-side over 1,792 proposals.
  2. Zero-Retrain Multi-Asset Declarative Weaving (`configs/tasks/oe62_orbital_synthesizable.yaml`):
     simultaneously aligning frozen external `Uni-Mol2` orbital targets with built-in inequality constraints
     (`SA_Score <= 3.5`, `mol_weight <= 550`, `smarts_alert <= 0`) and `DPA-4` atomic force internalization,
     audited blindly by cross-foundation `E15 DPA-4 AuditOracle`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from rdkit import Chem

from loom.chemistry import ValenceReconciler
from loom.generator.molecular_state import mol_to_clean_smiles
from loom.generator.octree_backend import trajectory_to_instance
from loom.oracles.unimol2_task import B_WEIGHT_SHA256
from loom.weaving import (
    AssetRegistry,
    AuditOracleSlot,
    BuiltInRuleAsset,
    CompositeCriticBus,
    LoomGRPOTrainer,
    LoomUni3DARGenerator,
    PotentialSlot,
    RewardOracleSlot,
    SlotOracleAsset,
    TaskWeavingSpec,
    compute_loom_metrics,
    formal_charge_signature,
    is_clean_organic_charge_state,
)
from scripts.loom_train_sft import load_sdf_with_targets


def evaluate_1792_dual_track(
    gen: LoomUni3DARGenerator,
    bus: CompositeCriticBus,
    q_points: Dict[str, float],
    source_graphs: set[str],
    *,
    prop_name: str = "homo",
    quantiles: Sequence[int] = (10, 25, 35, 50, 65, 75, 90),
    seeds: Sequence[int] = (24411, 24412),
    attempts_per_seed: int = 128,
    batch_size: int = 128,
    cfg_scale: float = 2.00,
    cfg_tree_decay: float = 0.30,
    geometric_guard: bool = True,
) -> Dict[str, Any]:
    from typing import Sequence  # local type import safety

    t0 = time.perf_counter()
    device = next(gen.parameters()).device
    prop_idx = gen.target_cols.index(prop_name)
    obj_idx = bus.spec.objective_names.index(prop_name)
    allowed_elems = tuple(gen.args.allow_atoms.split(","))
    reconciler = ValenceReconciler(allowed_elements=allowed_elems)

    req_list: List[float] = []
    trajs_all: List[Dict[str, Any]] = []
    gen_sec = 0.0

    for q in quantiles:
        goal = float(q_points[str(q)])
        for sd in seeds:
            rem = attempts_per_seed
            offset = 0
            while rem > 0:
                cur_b = min(batch_size, rem)
                t_mat = np.zeros((cur_b, gen.num_props), dtype=np.float32)
                t_mat[:, prop_idx] = goal
                t_norm = gen.normalize_targets(t_mat)
                m_act = torch.zeros((cur_b, gen.num_props), dtype=torch.bool, device=device)
                m_act[:, prop_idx] = True

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tg0 = time.perf_counter()
                trajs = gen.sample(
                    t_norm,
                    m_act,
                    cfg_scale=cfg_scale,
                    xyz_cfg_scale=1.0,
                    cfg_tree_decay=cfg_tree_decay,
                    geometric_guard=geometric_guard,
                    temperatures=(0.85, 0.30, 0.25),
                    seed=int(sd) * 10007 + int(q) * 131 + offset,
                    use_bf16=(device.type == "cuda"),
                    electron_parity=0,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                gen_sec += time.perf_counter() - tg0

                for tr in trajs:
                    req_list.append(goal)
                    trajs_all.append(tr)
                rem -= cur_b
                offset += cur_b

    n_total = len(trajs_all)
    req_mat = np.zeros((n_total, len(bus.spec.objectives)), dtype=np.float64)
    req_mat[:, obj_idx] = np.asarray(req_list, dtype=np.float64)
    act_mat = np.zeros((n_total, len(bus.spec.objectives)), dtype=bool)
    act_mat[:, obj_idx] = True

    # Track 1: Raw 3D Point-Cloud Evaluation (no ValenceReconciler)
    raw_mols: List[Optional[Chem.Mol]] = [None] * n_total
    raw_legal = np.zeros(n_total, dtype=bool)
    raw_exotic = np.zeros(n_total, dtype=bool)
    for i, tr in enumerate(trajs_all):
        try:
            inst = trajectory_to_instance(tr, element_table=allowed_elems)
            raw_mols[i] = inst.graph
            raw_legal[i] = True
            raw_exotic[i] = not is_clean_organic_charge_state(inst.graph)
        except ValueError:
            pass

    raw_ev = bus.evaluate_molecules(
        raw_mols, requested_targets=req_mat, active_mask=act_mat, include_audit=True, for_training=False, batch_size=32
    )
    raw_metrics = compute_loom_metrics(
        requested_targets_eV=req_list,
        reward_predictions_eV=raw_ev["objective_preds"][:, obj_idx],
        legal_mask=raw_legal,
        evaluated_mask=np.isfinite(raw_ev["objective_preds"][:, obj_idx]) & raw_legal,
        audit_predictions_eV=raw_ev["audit_preds"].get(prop_name),
        f_max_eV_A=raw_ev["f_max_eV_A"],
        unseen_charge_mask=raw_exotic,
        tolerance_eV=0.25,
    )
    raw_sa = raw_ev["constraint_values"].get("synthesizable_sa", np.full(n_total, np.nan))
    raw_clean_hits = int(
        np.sum(
            raw_legal
            & (~raw_exotic)
            & np.isfinite(raw_ev["objective_preds"][:, obj_idx])
            & (np.abs(raw_ev["objective_preds"][:, obj_idx] - req_mat[:, obj_idx]) <= 0.25)
        )
    )
    raw_synth_clean_hits = int(np.sum(raw_ev["constrained_clean_hits"]))
    raw_metrics.update(
        {
            "clean_hits": raw_clean_hits,
            "clean_hit_yield": raw_clean_hits / n_total,
            "synthesizable_clean_hits": raw_synth_clean_hits,
            "synthesizable_clean_hit_yield": raw_synth_clean_hits / n_total,
            "constraints_met_rate_among_legal": (
                float(raw_ev["constraints_met"][raw_legal].mean()) if raw_legal.any() else 0.0
            ),
            "median_sa_score_among_legal": (
                float(np.nanmedian(raw_sa[raw_legal])) if np.isfinite(raw_sa[raw_legal]).any() else None
            ),
            "sa_leq_3_5_rate_among_legal": (
                float(np.mean(raw_sa[raw_legal] <= 3.5)) if raw_legal.any() else 0.0
            ),
        }
    )

    # Track 2: Valence-Reconciled Evaluation (heavy-atom coordinates frozen!)
    rec_mols: List[Optional[Chem.Mol]] = [None] * n_total
    rec_legal = np.zeros(n_total, dtype=bool)
    rec_exotic = np.zeros(n_total, dtype=bool)
    rec_was_fixed = np.zeros(n_total, dtype=bool)
    heavy_rmsds: List[float] = []
    status_counts: Dict[str, int] = {}

    for i, tr in enumerate(trajs_all):
        if raw_legal[i] and not raw_exotic[i] and raw_mols[i] is not None:
            rec_mols[i] = raw_mols[i]
            rec_legal[i] = True
            rec_exotic[i] = False
            status_counts["clean_raw"] = status_counts.get("clean_raw", 0) + 1
        else:
            res = reconciler.reconcile_point_cloud(
                tr.get("atom_symbols", []),
                tr.get("positions_np", np.zeros((0, 3), dtype=np.float32)),
                potential_slot=None,
            )
            status_counts[res.status] = status_counts.get(res.status, 0) + 1
            if res.reconciled_legal and res.mol is not None:
                rec_mols[i] = res.mol
                rec_legal[i] = True
                rec_exotic[i] = res.reconciled_exotic_charge
                rec_was_fixed[i] = res.was_reconciled
                if np.isfinite(res.heavy_rmsd_A):
                    heavy_rmsds.append(res.heavy_rmsd_A)

    if bus.potential_slot is not None and rec_was_fixed.any():
        reconciler.batch_polish_hydrogens_mlip(
            [rec_mols[i] for i in range(n_total) if rec_was_fixed[i] and rec_mols[i] is not None],
            bus.potential_slot,
            n_steps=5,
            chunk_size=32,
        )

    rec_ev = bus.evaluate_molecules(
        rec_mols, requested_targets=req_mat, active_mask=act_mat, include_audit=True, for_training=False, batch_size=32
    )
    rec_metrics = compute_loom_metrics(
        requested_targets_eV=req_list,
        reward_predictions_eV=rec_ev["objective_preds"][:, obj_idx],
        legal_mask=rec_legal,
        evaluated_mask=np.isfinite(rec_ev["objective_preds"][:, obj_idx]) & rec_legal,
        audit_predictions_eV=rec_ev["audit_preds"].get(prop_name),
        f_max_eV_A=rec_ev["f_max_eV_A"],
        unseen_charge_mask=rec_exotic,
        tolerance_eV=0.25,
    )
    rec_sa = rec_ev["constraint_values"].get("synthesizable_sa", np.full(n_total, np.nan))
    rec_clean_hits = int(
        np.sum(
            rec_legal
            & (~rec_exotic)
            & np.isfinite(rec_ev["objective_preds"][:, obj_idx])
            & (np.abs(rec_ev["objective_preds"][:, obj_idx] - req_mat[:, obj_idx]) <= 0.25)
        )
    )
    rec_synth_clean_hits = int(np.sum(rec_ev["constrained_clean_hits"]))
    synth_hit_smiles = {
        mol_to_clean_smiles(rec_mols[i])
        for i in range(n_total)
        if rec_ev["constrained_clean_hits"][i] and rec_mols[i] is not None
    }
    novel_synth_hits = synth_hit_smiles - source_graphs

    rec_metrics.update(
        {
            "clean_hits": rec_clean_hits,
            "clean_hit_yield": rec_clean_hits / n_total,
            "synthesizable_clean_hits": rec_synth_clean_hits,
            "synthesizable_clean_hit_yield": rec_synth_clean_hits / n_total,
            "unique_synthesizable_hit_graphs": len(synth_hit_smiles),
            "novel_unique_synthesizable_hit_graphs": len(novel_synth_hits),
            "constraints_met_rate_among_legal": (
                float(rec_ev["constraints_met"][rec_legal].mean()) if rec_legal.any() else 0.0
            ),
            "median_sa_score_among_legal": (
                float(np.nanmedian(rec_sa[rec_legal])) if np.isfinite(rec_sa[rec_legal]).any() else None
            ),
            "sa_leq_3_5_rate_among_legal": (
                float(np.mean(rec_sa[rec_legal] <= 3.5)) if rec_legal.any() else 0.0
            ),
            "reconciled_molecule_count": int(rec_was_fixed.sum()),
            "max_heavy_atom_rmsd_A": float(max(heavy_rmsds)) if heavy_rmsds else 0.0,
            "status_breakdown": status_counts,
        }
    )

    return {
        "cfg_scale": cfg_scale,
        "generation_seconds": gen_sec,
        "total_eval_seconds": time.perf_counter() - t0,
        "raw_point_cloud": raw_metrics,
        "valence_reconciled": rec_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--replay-sdf", type=Path, required=True)
    parser.add_argument("--source-train-sdf", type=Path, required=True)
    parser.add_argument("--reward-oracle-dir", type=Path, required=True)
    parser.add_argument("--audit-oracle-dir", type=Path, required=True)
    parser.add_argument("--potential-weights", type=Path, required=True)
    parser.add_argument("--grpo-steps", type=int, default=96, help="96 steps * 4 prompts = 384 groups")
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(4)

    spec = TaskWeavingSpec.from_file(args.task_spec)
    registry = AssetRegistry(register_builtin_rules=True)
    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )
    audit_slot = AuditOracleSlot(
        backend="dpa4",
        checkpoint_dir=args.audit_oracle_dir,
        device=args.device,
        forcefield_path=str(args.potential_weights),
    )
    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.potential_weights), device=args.device)

    registry.register_predictor(SlotOracleAsset(reward_slot, asset_id="ext_unimol2", units=["eV", "eV"]))
    registry.register_predictor(SlotOracleAsset(audit_slot, asset_id="audit_dpa4", units=["eV", "eV"]))
    registry.register_predictor(BuiltInRuleAsset(asset_id="builtin_rules"))
    registry.register_potential("mlip_dpa4", pot_slot)

    print("[Loom-v2] Loading source & replay reference sets...", flush=True)
    source_graphs = {
        mol_to_clean_smiles(m)
        for m in Chem.SDMolSupplier(str(args.source_train_sdf), removeHs=False)
        if m is not None
    }
    replay_mols_all, replay_labels_all = load_sdf_with_targets(args.replay_sdf, ["homo", "lumo"])
    seen_charges = frozenset(formal_charge_signature(m) for m in replay_mols_all)

    bus = CompositeCriticBus(spec=spec, registry=registry, seen_charge_signatures=seen_charges)

    # Filter replay pool through declarative `CompositeCriticBus` (`SA_Score <= 3.5` & zero reactive alerts!)
    replay_mols, replay_labels, relabel_summary = bus.relabel_conformer_pool(replay_mols_all, max_fmax_eV_A=4.5)
    print(f"[Loom-v2] Declarative Replay Admission: {relabel_summary}", flush=True)

    targets_cfg = json.loads(args.targets_json.read_text())
    q_points = targets_cfg["single_target_points"]["homo"]

    gen, _ = LoomUni3DARGenerator.load_checkpoint(
        args.init_checkpoint, device=args.device, batch_size=128
    )

    # 1. Pre-Alignment Baseline Evaluation (Raw vs ValenceReconciler on v1.1 checkpoint)
    print("[Loom-v2] Evaluating Baseline v1.1 checkpoint (Raw vs Valence-Reconciled)...", flush=True)
    pre_report = evaluate_1792_dual_track(
        gen, bus, q_points, source_graphs, prop_name="homo", cfg_scale=2.00, cfg_tree_decay=0.30, geometric_guard=True
    )
    print(
        f"[Baseline Raw] Legal={pre_report['raw_point_cloud']['legal_rate']*100:.1f}%, "
        f"ExoticQ={(pre_report['raw_point_cloud']['unseen_formal_charge_rate_among_legal'] or 0)*100:.1f}%, "
        f"SA<=3.5={pre_report['raw_point_cloud']['sa_leq_3_5_rate_among_legal']*100:.1f}%, "
        f"SynthCleanHits={pre_report['raw_point_cloud']['synthesizable_clean_hits']}/1792\n"
        f"[Baseline Reconciled] Legal={pre_report['valence_reconciled']['legal_rate']*100:.1f}%, "
        f"ExoticQ={(pre_report['valence_reconciled']['unseen_formal_charge_rate_among_legal'] or 0)*100:.2f}%, "
        f"SA<=3.5={pre_report['valence_reconciled']['sa_leq_3_5_rate_among_legal']*100:.1f}%, "
        f"SynthCleanHits={pre_report['valence_reconciled']['synthesizable_clean_hits']}/1792",
        flush=True,
    )

    # 2. Upgrade to v2.0 (`HeavyAtomValenceHead` + Neutral Valence Guard + Multi-Asset `CompositeCriticBus` GRPO)
    trainable = gen.enable_v20_valence_head()
    print(f"[Loom-v2] Enabled HeavyAtomValenceHead; trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    h_p06, h_p94 = np.percentile(replay_labels[:, 0], [6.0, 94.0])

    def _sample_prompts(step_idx: int, rng: np.random.Generator):
        del step_idx
        p = args.prompts_per_step
        t_mat = np.zeros((p, 2), dtype=np.float32)
        strata = (np.arange(p, dtype=np.float64) + rng.random(p)) / p
        rng.shuffle(strata)
        t_mat[:, 0] = (h_p06 + strata * (h_p94 - h_p06)).astype(np.float32)
        mask = np.zeros((p, 2), dtype=bool)
        mask[:, 0] = True
        return t_mat, mask

    trainer = LoomGRPOTrainer(
        generator=gen,
        critic_bus=bus,
        replay_molecules=replay_mols,
        replay_targets_eV=replay_labels,
        lr=1.8e-5,
        kl_beta=0.04,
        replay_weight=0.15,
        w_valid=0.85,
        w_phys=0.35,
        w_prop=1.05,
        sharp_concentration=True,
        cfg_scale=1.80,
        cfg_tree_decay=0.30,
        geometric_guard=True,
        topology_only_policy=True,
        seed=2026,
    )
    grpo_hist = trainer.train_loop(
        total_steps=args.grpo_steps,
        prompt_sampler_fn=_sample_prompts,
        group_size=args.group_size,
        log_every=12,
    )

    ckpt_out = args.output_dir / "loom_v2_synthesizable_homo.pt"
    gen.save_checkpoint(
        ckpt_out,
        extra_metadata={
            "version": "2.0.0",
            "task_spec": spec.to_dict(),
            "groups_completed": args.grpo_steps * args.prompts_per_step,
        },
    )

    # 3. Post-Alignment v2.0 Evaluation (Raw vs Valence-Reconciled)
    print("[Loom-v2] Evaluating Aligned v2.0 checkpoint (Raw vs Valence-Reconciled)...", flush=True)
    post_report = evaluate_1792_dual_track(
        gen, bus, q_points, source_graphs, prop_name="homo", cfg_scale=2.00, cfg_tree_decay=0.30, geometric_guard=True
    )
    print(
        f"[v2.0 Aligned Raw] Legal={post_report['raw_point_cloud']['legal_rate']*100:.1f}%, "
        f"ExoticQ={(post_report['raw_point_cloud']['unseen_formal_charge_rate_among_legal'] or 0)*100:.1f}%, "
        f"SA<=3.5={post_report['raw_point_cloud']['sa_leq_3_5_rate_among_legal']*100:.1f}%, "
        f"SynthCleanHits={post_report['raw_point_cloud']['synthesizable_clean_hits']}/1792\n"
        f"[v2.0 Aligned Reconciled] Legal={post_report['valence_reconciled']['legal_rate']*100:.1f}%, "
        f"ExoticQ={(post_report['valence_reconciled']['unseen_formal_charge_rate_among_legal'] or 0)*100:.2f}%, "
        f"SA<=3.5={post_report['valence_reconciled']['sa_leq_3_5_rate_among_legal']*100:.1f}%, "
        f"SynthCleanHits={post_report['valence_reconciled']['synthesizable_clean_hits']}/1792",
        flush=True,
    )

    summary = {
        "task_spec": spec.to_dict(),
        "registered_assets": registry.list_assets(),
        "replay_pool_filtering": relabel_summary,
        "pre_alignment_v11": pre_report,
        "post_alignment_v20": post_report,
        "grpo_groups_completed": args.grpo_steps * args.prompts_per_step,
        "grpo_history_tail": grpo_hist[-8:],
        "saved_checkpoint": str(ckpt_out),
    }
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[Loom-v2] Saved Phase 1 & Phase 2 report to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
