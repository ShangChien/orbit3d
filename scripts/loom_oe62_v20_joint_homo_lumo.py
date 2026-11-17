"""`Uni-Loom v2.0` + `OctreeProcessCritic` Joint Dual-Objective `[HOMO, LUMO]` OE62 Benchmark.

Closes the multi-objective generality loop on 12-element, up-to-80-atom scaffold-disjoint OE62:
  1. `--phase align_v20`: Upgrades `stage25_joint_v11_grpo/final_grpo_checkpoint.pt` (`["homo", "lumo"]`)
     to `Uni-Loom v2.0` (`HeavyAtomValenceHead` + Neutral Valence Guard + `oe62_orbital_synthesizable.yaml`
     declarative constraints `SA_Score <= 3.5`, `smarts_alert <= 0`, `mol_weight <= 550`, `DPA-4` `F_phys`),
     training strictly WITHOUT `AuditOracleSlot`.
  2. `--phase train_critic`: Collects `8,192` joint `[HOMO, LUMO]` rollouts (`return_coarse_summary=True`),
     labels with training-side `ext_unimol2` (`[HOMO, LUMO]`) + `builtin_rules` (ZERO `audit_dpa4` leakage),
     and fits a 2-objective `OctreeProcessCritic` (`oe62_v20_joint_process_critic.pt`).
  3. `--phase eval_joint`: Clean single-GPU uncontended evaluation across the 49-cell `7x7` `[HOMO, LUMO]`
     2D quantile grid (`49 cells x 32 proposals = 1,568` joint proposals/arm) comparing:
       - `v11_Raw_M1` & `v11_Reconciled_M1` (`v1.1` baseline)
       - `v20_Raw_M1` & `v20_Outcome_M1, M2, M4, M8` (`v2.0` + `ValenceReconciler`)
       - `v20_Process_8_to_1_nosurr`, `v20_Process_8_to_2`, `v20_Process_16_to_4` (`OctreeProcessCritic` funnels)
     under both `Uni-Mol2` + `SA_Score` and strictly held-out blind `E15 DPA-4 AuditOracle` (`[HOMO, LUMO]`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    OctreeProcessCritic,
    PotentialSlot,
    RewardOracleSlot,
    SlotOracleAsset,
    TaskWeavingSpec,
    fit_octree_process_critic,
    formal_charge_signature,
    is_clean_organic_charge_state,
)
from scripts.loom_train_sft import load_sdf_with_targets


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_joint_bus(args: argparse.Namespace, *, include_blind_auditor: bool) -> Tuple[CompositeCriticBus, set[str]]:
    spec_dict = TaskWeavingSpec.from_file(args.oe62_task_spec).to_dict()
    # Ensure BOTH `homo` and `lumo` cross-oracle auditors are registered for blind 2D `[HOMO, LUMO]` audit!
    spec_dict["auditors"] = [
        {
            "name": "dpa4_homo_audit",
            "objective_name": "homo",
            "mode": "cross_oracle",
            "auditor_asset_id": "audit_dpa4",
            "auditor_channel": "homo",
            "disagreement_cap": 0.35,
        },
        {
            "name": "dpa4_lumo_audit",
            "objective_name": "lumo",
            "mode": "cross_oracle",
            "auditor_asset_id": "audit_dpa4",
            "auditor_channel": "lumo",
            "disagreement_cap": 0.35,
        },
    ]
    spec = TaskWeavingSpec.from_dict(spec_dict)

    registry = AssetRegistry(register_builtin_rules=True)
    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.oe62_reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )
    registry.register_predictor(SlotOracleAsset(reward_slot, asset_id="ext_unimol2", units=["eV", "eV"]))
    registry.register_predictor(BuiltInRuleAsset(asset_id="builtin_rules"))
    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.oe62_potential_weights), device=args.device)
    registry.register_potential("mlip_dpa4", pot_slot)

    if include_blind_auditor:
        audit_slot = AuditOracleSlot(
            backend="dpa4",
            checkpoint_dir=args.oe62_audit_oracle_dir,
            device=args.device,
            forcefield_path=str(args.oe62_potential_weights),
        )
        registry.register_predictor(SlotOracleAsset(audit_slot, asset_id="audit_dpa4", units=["eV", "eV"]))

    source_graphs = {
        mol_to_clean_smiles(m)
        for m in Chem.SDMolSupplier(str(args.oe62_source_train_sdf), removeHs=False)
        if m is not None
    }
    replay_mols_all, _ = load_sdf_with_targets(args.oe62_replay_sdf, ["homo", "lumo"])
    seen_charges = frozenset(formal_charge_signature(m) for m in replay_mols_all)
    bus = CompositeCriticBus(spec=spec, registry=registry, seen_charge_signatures=seen_charges)
    return bus, source_graphs


def run_align_v20(args: argparse.Namespace) -> None:
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(4)

    out_dir = args.work_dir / "v20_synthesizable_joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = out_dir / "loom_v2_synthesizable_joint_homo_lumo.pt"

    # Strictly NO `audit_dpa4` during GRPO alignment!
    bus, _ = _build_joint_bus(args, include_blind_auditor=False)
    replay_mols_all, _ = load_sdf_with_targets(args.oe62_replay_sdf, ["homo", "lumo"])
    replay_mols, replay_labels, relabel_summary = bus.relabel_conformer_pool(replay_mols_all, max_fmax_eV_A=4.5)
    print(f"[Joint-v2.0] Replay pool admission: {relabel_summary}", flush=True)

    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.oe62_v11_joint_ckpt, device=args.device, batch_size=128)
    trainable = gen.enable_v20_valence_head()
    print(f"[Joint-v2.0] Enabled HeavyAtomValenceHead; trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    h_p05, h_p95 = np.percentile(replay_labels[:, 0], [5.0, 95.0])
    l_p05, l_p95 = np.percentile(replay_labels[:, 1], [5.0, 95.0])
    n_emp = len(replay_labels)

    def _sample_joint_prompts(step_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        del step_idx
        p = int(args.prompts_per_step)
        t_mat = np.zeros((p, 2), dtype=np.float32)
        for j in range(p):
            if j % 2 == 0:
                idx = int(rng.integers(0, n_emp))
                th = float(replay_labels[idx, 0] + rng.normal(0.0, 0.08))
                tl = float(replay_labels[idx, 1] + rng.normal(0.0, 0.08))
            else:
                th = float(rng.uniform(h_p05, h_p95))
                tl = float(rng.uniform(l_p05, l_p95))
            if tl - th < 1.50:
                tl = th + 1.50
            t_mat[j, 0] = th
            t_mat[j, 1] = tl
        mask = np.ones((p, 2), dtype=bool)
        return t_mat, mask

    t0 = time.perf_counter()
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
        total_steps=int(args.grpo_steps),
        prompt_sampler_fn=_sample_joint_prompts,
        group_size=int(args.group_size),
        log_every=12,
    )
    align_sec = time.perf_counter() - t0
    gen.save_checkpoint(
        ckpt_out,
        extra_metadata={
            "version": "2.0.0-joint",
            "task_spec": bus.spec.to_dict(),
            "groups_completed": int(args.grpo_steps) * int(args.prompts_per_step),
            "align_wall_seconds": round(align_sec, 2),
        },
    )
    meta = {
        "init_checkpoint": str(args.oe62_v11_joint_ckpt),
        "saved_checkpoint": str(ckpt_out),
        "replay_admission": relabel_summary,
        "groups_completed": int(args.grpo_steps) * int(args.prompts_per_step),
        "align_wall_seconds": round(align_sec, 2),
        "grpo_history_tail": grpo_hist[-8:],
    }
    (out_dir / "v20_joint_align_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[Joint-v2.0 ALIGN DONE] Saved {ckpt_out} ({align_sec:.1f}s)", flush=True)


def run_train_critic(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    out_dir = args.work_dir / "octree_process_critic"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = out_dir / "oe62_v20_joint_process_critic.pt"
    v20_ckpt = args.work_dir / "v20_synthesizable_joint" / "loom_v2_synthesizable_joint_homo_lumo.pt"

    # Strictly NO `audit_dpa4` during Process Critic training!
    bus, _ = _build_joint_bus(args, include_blind_auditor=False)
    gen, _ = LoomUni3DARGenerator.load_checkpoint(v20_ckpt, device=args.device, batch_size=args.batch_size)
    gen.eval()

    _, train_labels = load_sdf_with_targets(args.oe62_master_train_sdf, ["homo", "lumo"])
    h_p04, h_p96 = np.percentile(train_labels[:, 0], [4.0, 96.0])
    l_p04, l_p96 = np.percentile(train_labels[:, 1], [4.0, 96.0])
    rng = np.random.default_rng(202617)

    n_rollouts = int(args.joint_rollouts)
    half = n_rollouts // 2
    emp_idx = rng.choice(len(train_labels), size=half, replace=False)
    emp_pairs = train_labels[emp_idx].astype(np.float32)
    uni_h = rng.uniform(h_p04, h_p96, size=n_rollouts - half).astype(np.float32)
    uni_l = np.maximum(rng.uniform(l_p04, l_p96, size=n_rollouts - half).astype(np.float32), uni_h + 1.50)
    uni_pairs = np.stack([uni_h, uni_l], axis=-1)
    goals_2d = np.concatenate([emp_pairs, uni_pairs], axis=0)
    rng.shuffle(goals_2d, axis=0)

    h_idx = gen.target_cols.index("homo")
    l_idx = gen.target_cols.index("lumo")
    allowed_elems = tuple(gen.args.allow_atoms.split(","))
    reconciler = ValenceReconciler(allowed_elements=allowed_elems)

    all_h: List[torch.Tensor] = []
    all_geom: List[torch.Tensor] = []
    all_t_norm: List[torch.Tensor] = []
    all_m_act: List[torch.Tensor] = []
    rec_mols: List[Optional[Chem.Mol]] = []
    rec_clean: List[bool] = []

    t0 = time.perf_counter()
    for s in range(0, n_rollouts, args.batch_size):
        e = min(s + args.batch_size, n_rollouts)
        cb = e - s
        t_mat = np.zeros((cb, gen.num_props), dtype=np.float32)
        t_mat[:, h_idx] = goals_2d[s:e, 0]
        t_mat[:, l_idx] = goals_2d[s:e, 1]
        t_n = gen.normalize_targets(t_mat)
        m_a = torch.zeros((cb, gen.num_props), dtype=torch.bool, device=device)
        m_a[:, h_idx] = True
        m_a[:, l_idx] = True

        trajs = gen.sample(
            t_n,
            m_a,
            cfg_scale=2.00,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.30,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=660000 + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            return_coarse_summary=True,
        )
        all_t_norm.append(t_n.detach().cpu())
        all_m_act.append(m_a.detach().cpu())
        for tr in trajs:
            all_h.append(tr["coarse_h_concat"])
            all_geom.append(tr["coarse_geom_feats"])
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and res.mol is not None:
                rec_mols.append(res.mol)
                rec_clean.append(not bool(res.reconciled_exotic_charge))
            else:
                rec_mols.append(None)
                rec_clean.append(False)

    ev = bus.evaluate_molecules(rec_mols, include_audit=False, for_training=True, batch_size=64)
    clean_arr = np.asarray(rec_clean, dtype=bool)
    constr_ok = ev["constraints_met"]
    clean_and_synth = clean_arr & constr_ok

    obj_h = bus.spec.objective_names.index("homo")
    obj_l = bus.spec.objective_names.index("lumo")
    y_full = np.full((n_rollouts, gen.num_props), np.nan, dtype=np.float32)
    y_full[:, h_idx] = ev["objective_preds"][:, obj_h].astype(np.float32)
    y_full[:, l_idx] = ev["objective_preds"][:, obj_l].astype(np.float32)
    valid_y = clean_arr & np.isfinite(y_full).all(axis=1)
    y_full = np.nan_to_num(y_full, nan=0.0)

    critic = OctreeProcessCritic(
        emb_dim=int(gen.args.emb_dim), num_targets=gen.num_props, hidden_dim=512, dropout=0.10
    ).to(device)
    critic.set_normalization(gen.target_mean.detach().cpu(), gen.target_scale.detach().cpu())

    fit_t0 = time.perf_counter()
    stats = fit_octree_process_critic(
        critic,
        h_concat=torch.stack(all_h, dim=0),
        geom_feats=torch.stack(all_geom, dim=0),
        targets_norm=torch.cat(all_t_norm, dim=0),
        active_mask=torch.cat(all_m_act, dim=0),
        y_surrogate_norm=critic.normalize_y(torch.from_numpy(y_full)).detach().cpu(),
        y_valid_mask=torch.from_numpy(valid_y),
        is_legal=torch.from_numpy(clean_arr),
        is_constr_met=torch.from_numpy(clean_and_synth),
        epochs=24,
        batch_size=256,
        lr=1.5e-3,
        seed=2026,
    )
    fit_sec = time.perf_counter() - fit_t0
    meta = {
        "total_rollouts": n_rollouts,
        "clean_rate": round(float(clean_arr.mean()), 4),
        "clean_and_synthesizable_rate": round(float(clean_and_synth.mean()), 4),
        "critic_params": sum(p.numel() for p in critic.parameters()),
        "rollout_and_surrogate_wall_sec": round(time.perf_counter() - t0 - fit_sec, 2),
        "critic_fit_wall_sec": round(fit_sec, 2),
        "train_stats": stats.__dict__,
    }
    critic.save_checkpoint(ckpt_out, extra=meta)
    (out_dir / "oe62_v20_joint_process_critic_train.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[Joint ProcCritic TRAIN DONE] Saved {ckpt_out} | meta={meta}", flush=True)


def _select_joint_from_pool(
    mols_grid: List[List[Optional[Chem.Mol]]],
    legal_grid: np.ndarray,
    exotic_grid: np.ndarray,
    req_2d: np.ndarray,
    bus: CompositeCriticBus,
    m_keep: int,
    use_surrogate: bool,
    device: torch.device,
) -> Tuple[List[Optional[Chem.Mol]], np.ndarray, np.ndarray, float]:
    n_prompts = len(req_2d)
    if not use_surrogate or m_keep == 1:
        return [mols_grid[i][0] for i in range(n_prompts)], legal_grid[:, 0].copy(), exotic_grid[:, 0].copy(), 0.0

    flat_mols = [mols_grid[i][k] for i in range(n_prompts) for k in range(m_keep)]
    _sync(device)
    ts0 = time.perf_counter()
    pred_asset = bus.registry.get_predictor("ext_unimol2")
    rule_asset = bus.registry.get_predictor("builtin_rules")
    u_out = pred_asset.predict_mols(flat_mols, batch_size=64)
    r_out = rule_asset.predict_mols(flat_mols, batch_size=256)
    _sync(device)
    surr_wall = time.perf_counter() - ts0

    homo_p = u_out["mean"][:, pred_asset.channel_index("homo")].reshape(n_prompts, m_keep)
    lumo_p = u_out["mean"][:, pred_asset.channel_index("lumo")].reshape(n_prompts, m_keep)
    sa_p = r_out["mean"][:, rule_asset.channel_index("sa_score")].reshape(n_prompts, m_keep)
    mw_p = r_out["mean"][:, rule_asset.channel_index("mol_weight")].reshape(n_prompts, m_keep)
    sm_p = r_out["mean"][:, rule_asset.channel_index("smarts_alert")].reshape(n_prompts, m_keep)

    sel_mols: List[Optional[Chem.Mol]] = []
    sel_leg = np.zeros(n_prompts, dtype=bool)
    sel_exo = np.zeros(n_prompts, dtype=bool)

    for i in range(n_prompts):
        gh, gl = float(req_2d[i, 0]), float(req_2d[i, 1])
        best_k = 0
        best_key = (99, float("inf"))
        for k in range(m_keep):
            if not bool(legal_grid[i, k]) or mols_grid[i][k] is None:
                ck = (4, float(k))
            else:
                cln = not bool(exotic_grid[i, k])
                c_ok = (sa_p[i, k] <= 3.5) and (sm_p[i, k] <= 0.0) and (mw_p[i, k] <= 550.0)
                eh = abs(float(homo_p[i, k]) - gh) if math.isfinite(float(homo_p[i, k])) else 99.0
                el = abs(float(lumo_p[i, k]) - gl) if math.isfinite(float(lumo_p[i, k])) else 99.0
                # Max-norm + L1 joint distance favors simultaneous `[HOMO, LUMO]` box hits (`<= 0.25 eV`)
                j_err = 0.5 * (eh + el) + 0.35 * max(eh, el)
                sa_pen = max(0.0, float(sa_p[i, k]) - 3.5) * 0.25
                tier = 0 if (cln and c_ok) else (1 if cln else 2)
                ck = (tier, j_err + sa_pen)
            if ck < best_key:
                best_key = ck
                best_k = k
        sel_mols.append(mols_grid[i][best_k] if bool(legal_grid[i, best_k]) else None)
        sel_leg[i] = bool(legal_grid[i, best_k])
        sel_exo[i] = bool(exotic_grid[i, best_k])

    return sel_mols, sel_leg, sel_exo, surr_wall


def _score_joint_cohort(
    selected_mols: List[Optional[Chem.Mol]],
    selected_legal: np.ndarray,
    selected_exotic: np.ndarray,
    req_2d: np.ndarray,
    bus: CompositeCriticBus,
    source_graphs: set[str],
    gen_wall_sec: float,
    reconcile_wall_sec: float,
    surrogate_wall_sec: float,
    tolerance_eV: float = 0.25,
) -> Dict[str, Any]:
    n_total = len(selected_mols)
    obj_h = bus.spec.objective_names.index("homo")
    obj_l = bus.spec.objective_names.index("lumo")
    req_mat = np.zeros((n_total, len(bus.spec.objectives)), dtype=np.float64)
    req_mat[:, obj_h] = req_2d[:, 0]
    req_mat[:, obj_l] = req_2d[:, 1]
    act_mat = np.ones((n_total, len(bus.spec.objectives)), dtype=bool)

    ev = bus.evaluate_molecules(
        selected_mols,
        requested_targets=req_mat,
        active_mask=act_mat,
        include_audit=True,
        for_training=False,
        batch_size=32,
    )
    rew_h = ev["objective_preds"][:, obj_h]
    rew_l = ev["objective_preds"][:, obj_l]
    aud_h = ev["audit_preds"].get("homo")
    aud_l = ev["audit_preds"].get("lumo")
    sa_vals = ev["constraint_values"].get("synthesizable_sa", np.full(n_total, np.nan))
    fmax = ev["f_max_eV_A"]

    clean_mask = selected_legal & (~selected_exotic)
    ev_rew = clean_mask & np.isfinite(rew_h) & np.isfinite(rew_l)
    ev_aud = (
        clean_mask & np.isfinite(aud_h) & np.isfinite(aud_l)
        if (aud_h is not None and aud_l is not None)
        else np.zeros(n_total, dtype=bool)
    )

    hit_h = ev_rew & (np.abs(rew_h - req_2d[:, 0]) <= tolerance_eV)
    hit_l = ev_rew & (np.abs(rew_l - req_2d[:, 1]) <= tolerance_eV)
    joint_clean_hit = hit_h & hit_l
    joint_synth_clean_hit = joint_clean_hit & ev["constraints_met"]

    aud_joint_clean_hit = (
        ev_aud
        & (np.abs(aud_h - req_2d[:, 0]) <= tolerance_eV)
        & (np.abs(aud_l - req_2d[:, 1]) <= tolerance_eV)
        if (aud_h is not None and aud_l is not None)
        else np.zeros(n_total, dtype=bool)
    )
    aud_joint_synth_clean_hit = aud_joint_clean_hit & ev["constraints_met"]

    synth_smiles = {
        mol_to_clean_smiles(selected_mols[i])
        for i in range(n_total)
        if joint_synth_clean_hit[i] and selected_mols[i] is not None
    }

    def _safe_mae(p: Optional[np.ndarray], t: np.ndarray, m: np.ndarray) -> Optional[float]:
        if p is None or not m.any():
            return None
        return round(float(np.mean(np.abs(p[m] - t[m]))), 4)

    def _safe_slope(p: Optional[np.ndarray], t: np.ndarray, m: np.ndarray) -> Optional[float]:
        if p is None or int(m.sum()) < 4 or np.std(t[m]) < 1e-8:
            return None
        return round(float(np.polyfit(t[m], p[m], 1)[0]), 4)

    rew_h_mae = _safe_mae(rew_h, req_2d[:, 0], ev_rew)
    rew_l_mae = _safe_mae(rew_l, req_2d[:, 1], ev_rew)
    rew_2d_mae = round(0.5 * (rew_h_mae + rew_l_mae), 4) if (rew_h_mae is not None and rew_l_mae is not None) else None

    aud_h_mae = _safe_mae(aud_h, req_2d[:, 0], ev_aud)
    aud_l_mae = _safe_mae(aud_l, req_2d[:, 1], ev_aud)
    aud_2d_mae = round(0.5 * (aud_h_mae + aud_l_mae), 4) if (aud_h_mae is not None and aud_l_mae is not None) else None

    valid_f = fmax[selected_legal & np.isfinite(fmax)]
    end_wall = gen_wall_sec + reconcile_wall_sec + surrogate_wall_sec

    return {
        "attempted": n_total,
        "legal": int(selected_legal.sum()),
        "legal_rate": round(float(selected_legal.mean()), 5),
        "unseen_formal_charge_rate_among_legal": (
            round(float(selected_exotic[selected_legal].mean()), 5) if selected_legal.any() else 0.0
        ),
        "constraints_met_rate_among_legal": (
            round(float(ev["constraints_met"][selected_legal].mean()), 5) if selected_legal.any() else 0.0
        ),
        "sa_leq_3_5_rate_among_legal": (
            round(float(np.mean(sa_vals[selected_legal] <= 3.5)), 5) if selected_legal.any() else 0.0
        ),
        "median_sa_score_among_legal": (
            round(float(np.nanmedian(sa_vals[selected_legal])), 4)
            if (selected_legal.any() and np.isfinite(sa_vals[selected_legal]).any())
            else None
        ),
        "zero_step_f_max_median_eV_A": round(float(np.median(valid_f)), 4) if len(valid_f) else None,
        "zero_step_f_max_p90_eV_A": round(float(np.percentile(valid_f, 90)), 4) if len(valid_f) else None,
        "homo_mae_eV": rew_h_mae,
        "lumo_mae_eV": rew_l_mae,
        "joint_2d_mae_eV": rew_2d_mae,
        "homo_slope": _safe_slope(rew_h, req_2d[:, 0], ev_rew),
        "lumo_slope": _safe_slope(rew_l, req_2d[:, 1], ev_rew),
        "homo_marginal_clean_hits": int(hit_h.sum()),
        "lumo_marginal_clean_hits": int(hit_l.sum()),
        "joint_2d_clean_hits": int(joint_clean_hit.sum()),
        "joint_2d_clean_hit_yield": round(float(joint_clean_hit.mean()), 5),
        "joint_2d_synthesizable_clean_hits": int(joint_synth_clean_hit.sum()),
        "joint_2d_synthesizable_clean_hit_yield": round(float(joint_synth_clean_hit.mean()), 5),
        "blind_dpa4_homo_mae_eV": aud_h_mae,
        "blind_dpa4_lumo_mae_eV": aud_l_mae,
        "blind_dpa4_joint_2d_mae_eV": aud_2d_mae,
        "blind_dpa4_homo_slope": _safe_slope(aud_h, req_2d[:, 0], ev_aud),
        "blind_dpa4_lumo_slope": _safe_slope(aud_l, req_2d[:, 1], ev_aud),
        "blind_dpa4_joint_2d_clean_hits": int(aud_joint_clean_hit.sum()),
        "blind_dpa4_joint_2d_clean_hit_yield": round(float(aud_joint_clean_hit.mean()), 5),
        "blind_dpa4_joint_2d_synthesizable_clean_hits": int(aud_joint_synth_clean_hit.sum()),
        "blind_dpa4_joint_2d_synthesizable_clean_hit_yield": round(float(aud_joint_synth_clean_hit.mean()), 5),
        "unique_synthesizable_hit_graphs": len(synth_smiles),
        "novel_unique_synthesizable_hit_graphs": len(synth_smiles - source_graphs),
        "gen_wall_seconds": round(gen_wall_sec, 3),
        "reconcile_wall_seconds": round(reconcile_wall_sec, 3),
        "surrogate_wall_seconds": round(surrogate_wall_sec, 3),
        "end_to_end_wall_seconds": round(end_wall, 3),
        "effective_output_mol_per_sec": round(float(n_total) / max(end_wall, 1e-6), 2),
    }


def run_eval_joint(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    v20_ckpt = args.work_dir / "v20_synthesizable_joint" / "loom_v2_synthesizable_joint_homo_lumo.pt"
    critic_ckpt = args.work_dir / "octree_process_critic" / "oe62_v20_joint_process_critic.pt"

    critic, critic_meta = OctreeProcessCritic.load_checkpoint(critic_ckpt, device=args.device)
    critic.eval()
    bus, source_graphs = _build_joint_bus(args, include_blind_auditor=True)

    _, src_labels = load_sdf_with_targets(args.oe62_source_train_sdf, ["homo", "lumo"])
    h_2d = np.linspace(np.percentile(src_labels[:, 0], 4), np.percentile(src_labels[:, 0], 96), 7)
    l_2d = np.linspace(np.percentile(src_labels[:, 1], 4), np.percentile(src_labels[:, 1], 96), 7)
    hh, ll = np.meshgrid(h_2d, l_2d, indexing="ij")
    cells = np.stack([hh.ravel(), ll.ravel()], axis=-1)
    valid_cells = [c for c in cells if float(c[0]) < float(c[1]) - 0.2]
    k_per_cell = int(args.proposals_per_cell)
    req_2d = np.repeat(np.asarray(valid_cells, dtype=np.float64), k_per_cell, axis=0)
    n_prompts = len(req_2d)  # 49 * 32 = 1,568

    arms_out: Dict[str, Any] = {}

    # 0. Reference v1.1 Joint Checkpoint (`Raw M=1` and `ValenceReconciled M=1`)
    gen_v11, _ = LoomUni3DARGenerator.load_checkpoint(args.oe62_v11_joint_ckpt, device=args.device, batch_size=args.batch_size)
    gen_v11.eval()
    allowed_elems = tuple(gen_v11.args.allow_atoms.split(","))
    reconciler = ValenceReconciler(allowed_elements=allowed_elems)
    h_idx = gen_v11.target_cols.index("homo")
    l_idx = gen_v11.target_cols.index("lumo")

    def _encode_2d(g_obj: LoomUni3DARGenerator, arr_2d: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        nb = len(arr_2d)
        mat = np.zeros((nb, g_obj.num_props), dtype=np.float32)
        mat[:, h_idx] = arr_2d[:, 0].astype(np.float32)
        mat[:, l_idx] = arr_2d[:, 1].astype(np.float32)
        tn = g_obj.normalize_targets(mat)
        ma = torch.ones((nb, g_obj.num_props), dtype=torch.bool, device=device)
        return tn, ma

    wt, wm = _encode_2d(gen_v11, req_2d[:16])
    _ = gen_v11.sample(wt, wm, cfg_scale=2.0, cfg_tree_decay=0.30, geometric_guard=True, seed=9, use_bf16=(device.type == "cuda"))
    _sync(device)

    tg0 = time.perf_counter()
    v11_trajs: List[Dict[str, Any]] = []
    for s in range(0, n_prompts, args.batch_size):
        e = min(s + args.batch_size, n_prompts)
        bt, bm = _encode_2d(gen_v11, req_2d[s:e])
        v11_trajs.extend(
            gen_v11.sample(
                bt,
                bm,
                cfg_scale=2.00,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.30,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=73001 + s,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
            )
        )
    _sync(device)
    v11_gen_w = time.perf_counter() - tg0
    del gen_v11
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tr0 = time.perf_counter()
    v11_raw_m: List[Optional[Chem.Mol]] = [None] * n_prompts
    v11_raw_l = np.zeros(n_prompts, dtype=bool)
    v11_raw_x = np.zeros(n_prompts, dtype=bool)
    for i, tr in enumerate(v11_trajs):
        try:
            inst = trajectory_to_instance(tr, element_table=allowed_elems)
            v11_raw_m[i] = inst.graph
            v11_raw_l[i] = True
            v11_raw_x[i] = not is_clean_organic_charge_state(inst.graph)
        except ValueError:
            pass
    v11_raw_rw = time.perf_counter() - tr0
    arms_out["v11_Raw_M1"] = _score_joint_cohort(
        v11_raw_m, v11_raw_l, v11_raw_x, req_2d, bus, source_graphs, v11_gen_w, v11_raw_rw, 0.0
    )

    tr0 = time.perf_counter()
    v11_rec_m: List[Optional[Chem.Mol]] = [None] * n_prompts
    v11_rec_l = np.zeros(n_prompts, dtype=bool)
    v11_rec_x = np.zeros(n_prompts, dtype=bool)
    for i, tr in enumerate(v11_trajs):
        res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
        if res.reconciled_legal and res.mol is not None:
            v11_rec_m[i] = res.mol
            v11_rec_l[i] = True
            v11_rec_x[i] = bool(res.reconciled_exotic_charge)
    if bus.potential_slot is not None:
        reconciler.batch_polish_hydrogens_mlip([m for m in v11_rec_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
    v11_rec_rw = time.perf_counter() - tr0
    arms_out["v11_Reconciled_M1"] = _score_joint_cohort(
        v11_rec_m, v11_rec_l, v11_rec_x, req_2d, bus, source_graphs, v11_gen_w, v11_rec_rw, 0.0
    )
    print(
        f"[Joint v1.1 M=1] Raw SynthClean={arms_out['v11_Raw_M1']['joint_2d_synthesizable_clean_hits']}/{n_prompts} "
        f"| Reconciled SynthClean={arms_out['v11_Reconciled_M1']['joint_2d_synthesizable_clean_hits']}/{n_prompts}",
        flush=True,
    )

    # 1. Aligned `v2.0` Outcome Best-of-M (`M = 1, 2, 4, 8`) + `v20_Raw_M1`
    gen_v20, _ = LoomUni3DARGenerator.load_checkpoint(v20_ckpt, device=args.device, batch_size=args.batch_size)
    gen_v20.eval()
    wt, wm = _encode_2d(gen_v20, req_2d[:16])
    _ = gen_v20.sample(wt, wm, cfg_scale=2.0, cfg_tree_decay=0.30, geometric_guard=True, seed=19, use_bf16=(device.type == "cuda"))
    _sync(device)

    m_max = 8
    round_trajs: List[List[Dict[str, Any]]] = []
    round_gen_walls: List[float] = []
    for m_i in range(m_max):
        _sync(device)
        tg0 = time.perf_counter()
        cur_tr: List[Dict[str, Any]] = []
        for s in range(0, n_prompts, args.batch_size):
            e = min(s + args.batch_size, n_prompts)
            bt, bm = _encode_2d(gen_v20, req_2d[s:e])
            cur_tr.extend(
                gen_v20.sample(
                    bt,
                    bm,
                    cfg_scale=2.00,
                    xyz_cfg_scale=1.0,
                    cfg_tree_decay=0.30,
                    geometric_guard=True,
                    temperatures=(0.85, 0.30, 0.25),
                    seed=844110 + m_i * 9973 + s,
                    use_bf16=(device.type == "cuda"),
                    electron_parity=0,
                )
            )
        _sync(device)
        round_gen_walls.append(time.perf_counter() - tg0)
        round_trajs.append(cur_tr)

    # v20_Raw_M1
    tr0 = time.perf_counter()
    v20_raw_m: List[Optional[Chem.Mol]] = [None] * n_prompts
    v20_raw_l = np.zeros(n_prompts, dtype=bool)
    v20_raw_x = np.zeros(n_prompts, dtype=bool)
    for i, tr in enumerate(round_trajs[0]):
        try:
            inst = trajectory_to_instance(tr, element_table=allowed_elems)
            v20_raw_m[i] = inst.graph
            v20_raw_l[i] = True
            v20_raw_x[i] = not is_clean_organic_charge_state(inst.graph)
        except ValueError:
            pass
    arms_out["v20_Raw_M1"] = _score_joint_cohort(
        v20_raw_m, v20_raw_l, v20_raw_x, req_2d, bus, source_graphs, round_gen_walls[0], time.perf_counter() - tr0, 0.0
    )

    tr0 = time.perf_counter()
    out_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(n_prompts)]
    out_leg = np.zeros((n_prompts, m_max), dtype=bool)
    out_exo = np.zeros((n_prompts, m_max), dtype=bool)
    for i in range(n_prompts):
        for m_i in range(m_max):
            tr = round_trajs[m_i][i]
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and res.mol is not None:
                out_grid[i][m_i] = res.mol
                out_leg[i, m_i] = True
                out_exo[i, m_i] = bool(res.reconciled_exotic_charge)
    rec_wall_total = time.perf_counter() - tr0

    for mb in (1, 2, 4, 8):
        sub_g = [row[:mb] for row in out_grid]
        sel_m, sel_l, sel_x, surr_w = _select_joint_from_pool(
            sub_g, out_leg[:, :mb], out_exo[:, :mb], req_2d, bus, mb, use_surrogate=(mb > 1), device=device
        )
        if bus.potential_slot is not None:
            reconciler.batch_polish_hydrogens_mlip([m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
        gw = float(sum(round_gen_walls[:mb]))
        rw = rec_wall_total * (float(mb) / float(m_max))
        res_m = _score_joint_cohort(sel_m, sel_l, sel_x, req_2d, bus, source_graphs, gw, rw, surr_w)
        arms_out[f"v20_Outcome_M{mb}"] = res_m
        b_mae = float("nan") if res_m["blind_dpa4_joint_2d_mae_eV"] is None else float(res_m["blind_dpa4_joint_2d_mae_eV"])
        print(
            f"[Joint v2.0 Outcome M={mb}] Eff={res_m['effective_output_mol_per_sec']:.2f} mol/s | "
            f"JointSynthClean={res_m['joint_2d_synthesizable_clean_hits']}/{n_prompts} ({res_m['joint_2d_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindJointSynthClean={res_m['blind_dpa4_joint_2d_synthesizable_clean_hits']} | Blind2DMAE={b_mae:.4f}",
            flush=True,
        )

    # 2. `OctreeProcessCritic` Joint Dual-Objective Funnels (`8->1` zero-surrogate, `8->2`, `16->4`)
    obj_col_mask = torch.ones((gen_v20.num_props,), dtype=torch.bool, device=device)
    funnels: List[Tuple[str, int, int, bool]] = [
        ("v20_Process_8_to_1_nosurr", 8, 1, False),
        ("v20_Process_8_to_2", 8, 2, True),
        ("v20_Process_16_to_4", 16, 4, True),
    ]
    for f_name, m_init, m_keep, use_surr in funnels:
        _sync(device)
        tg0 = time.perf_counter()
        groups_all: List[List[Dict[str, Any]]] = []
        for s in range(0, n_prompts, args.batch_size):
            e = min(s + args.batch_size, n_prompts)
            bt, bm = _encode_2d(gen_v20, req_2d[s:e])
            grp = gen_v20.sample_with_process_critic(
                bt,
                bm,
                critic,
                m_init=m_init,
                m_keep=m_keep,
                objective_col_mask=obj_col_mask,
                legal_weight=0.80,
                constr_weight=0.55,
                cfg_scale=2.00,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.30,
                geometric_guard=True,
                seed=910000 + m_init * 313 + m_keep * 37 + s,
                use_bf16=(device.type == "cuda"),
            )
            groups_all.extend(grp)
        _sync(device)
        gw = time.perf_counter() - tg0

        tr0 = time.perf_counter()
        p_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_keep for _ in range(n_prompts)]
        p_leg = np.zeros((n_prompts, m_keep), dtype=bool)
        p_exo = np.zeros((n_prompts, m_keep), dtype=bool)
        for i in range(n_prompts):
            for k in range(m_keep):
                tr = groups_all[i][k]
                res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
                if res.reconciled_legal and res.mol is not None:
                    p_grid[i][k] = res.mol
                    p_leg[i, k] = True
                    p_exo[i, k] = bool(res.reconciled_exotic_charge)
        rw = time.perf_counter() - tr0

        sel_m, sel_l, sel_x, surr_w = _select_joint_from_pool(
            p_grid, p_leg, p_exo, req_2d, bus, m_keep, use_surrogate=use_surr, device=device
        )
        if bus.potential_slot is not None:
            reconciler.batch_polish_hydrogens_mlip([m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
        res_f = _score_joint_cohort(sel_m, sel_l, sel_x, req_2d, bus, source_graphs, gw, rw, surr_w)
        arms_out[f_name] = res_f
        b_mae = float("nan") if res_f["blind_dpa4_joint_2d_mae_eV"] is None else float(res_f["blind_dpa4_joint_2d_mae_eV"])
        print(
            f"[Joint {f_name}] Eff={res_f['effective_output_mol_per_sec']:.2f} mol/s | "
            f"JointSynthClean={res_f['joint_2d_synthesizable_clean_hits']}/{n_prompts} ({res_f['joint_2d_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindJointSynthClean={res_f['blind_dpa4_joint_2d_synthesizable_clean_hits']} | Blind2DMAE={b_mae:.4f}",
            flush=True,
        )

    payload = {
        "experiment": "OE62 Joint Dual-Objective [HOMO, LUMO] Synthesizable Weaving (v2.0 + ValenceReconciler + OctreeProcessCritic)",
        "n_grid_cells": len(valid_cells),
        "proposals_per_cell": k_per_cell,
        "proposals_per_arm": n_prompts,
        "v11_checkpoint": str(args.oe62_v11_joint_ckpt),
        "v20_checkpoint": str(v20_ckpt),
        "process_critic_meta": critic_meta,
        "arms": arms_out,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Joint v2.0 + ProcCritic EVAL DONE] Wrote {args.output_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["align_v20", "train_critic", "eval_joint"], required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument(
        "--oe62-v11-joint-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/stage25_joint_v11_grpo/final_grpo_checkpoint.pt"),
    )
    p.add_argument(
        "--oe62-master-train-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"),
    )
    p.add_argument(
        "--oe62-source-train-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/source_labels/train_source.sdf"),
    )
    p.add_argument(
        "--oe62-replay-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_B.sdf"),
    )
    p.add_argument(
        "--oe62-reward-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"),
    )
    p.add_argument(
        "--oe62-audit-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_e15_calibration_20261114/full_fit/oracle"),
    )
    p.add_argument(
        "--oe62-potential-weights",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"),
    )
    p.add_argument("--grpo-steps", type=int, default=96)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--joint-rollouts", type=int, default=8192)
    p.add_argument("--proposals-per-cell", type=int, default=32)
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_v20_joint_homo_lumo_eval.json"),
    )
    args = p.parse_args()
    if args.phase == "align_v20":
        run_align_v20(args)
    elif args.phase == "train_critic":
        run_train_critic(args)
    elif args.phase == "eval_joint":
        run_eval_joint(args)


if __name__ == "__main__":
    main()
