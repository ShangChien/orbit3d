"""Unified `LoomService` Benchmark & Self-Driving Lab Epistemic Verification Suite (`OE62`).

Executes all final benchmarks through the standardized `LoomService` (`src/loom/weaving/service.py`)
contract (`EFFORT_TO_M = {"base": 1, "low": 2, "medium": 4, "high": 8}`), strictly enforcing:
  1. Zero PCA canonicalization (`random_rotation_prob = 0.80`);
  2. Zero `n_atoms` in `LoomConditioner`;
  3. Zero inference-time `DPA-4` relaxation or scoring (`potential_slot=None` inside `LoomService`);
  4. Zero `audit_dpa4` leakage during training, dev calibration, or Best-of-M selection.

Supported `--mode` phases:
  - `rollout_fphys_shard` / `refit_fphys_critic`:
      Upgrades `TwoStageOctreeProcessCritic` with continuous post-refiner training-time `DPA-4`
      `F_phys` (`zero_step_physical_reward`) group ranking targets and saves a unified bundle
      `v4_pro_bundled.pt`.
  - `fair30_uni3dar`:
      Symmetric 112-prompt zero-leakage `LoomService.calibrate_on_dev` + full 1,792-prompt
      30.4-atom evaluation across standardized `base (M=1)`, `low (M=2)`, `medium (M=4)`, and
      `high (M=8)` tiers + `GPAtlas` 4-quadrant epistemic confidence envelopes.
  - `fair30_pmf`:
      Applies the exact same `LoomService` 112-prompt calibration & `M=1,2,4,8` evaluation to
      `LoomFlowMolGenerator` (`PropMolFlow` checkpoints).
  - `buckets_uni3dar`:
      Controlled multi-atom scaling (`N in {20, 30, 40, 50, 60}`, 280 prompts/bucket) via
      `LoomService` across `base (M=1)`, `low (M=2)`, `medium (M=4)`, and `high (M=8)`.
  - `ablation_m4`:
      Fixed `medium (M=4)` internal architecture ablation (`4->4->4` vs `32->8->4` speculative
      octree pruning x `PostTreeGraphRefiner` seam healing).
  - `multi_objective_sdl`:
      Joint `(HOMO, LUMO) + SA <= 3.5` multi-objective `RSIDiagnoser.diagnose_nd` & partial-`NaN`
      assay calibration verification on real `OE62` DFT conformers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state
from loom.weaving import (
    EFFORT_TO_M,
    AssetRegistry,
    BuiltInRuleAsset,
    CompositeCriticBus,
    HeteroscedasticGPAtlas1D,
    IncrementalCalibrator,
    LoomFlowMolGenerator,
    LoomService,
    LoomUni3DARGenerator,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    PostTreeGraphRefiner,
    RSIDiagnoser,
    ResidualCalibratedAsset,
    RewardOracleSlot,
    SlotOracleAsset,
    TaskWeavingSpec,
    TwoStageOctreeProcessCritic,
    fit_octree_element_process_critic,
    fit_octree_process_critic,
    zero_step_physical_reward,
)
from scripts.loom_oe62_size_anchored import (
    _get_clean_oe62_pool,
    _get_source_novelty_sets,
    _score_oe62_size_cohort,
)
from scripts.loom_uni3dar_v4_step1_eval import (
    _build_calibrated_oe62_bus,
    build_fair_30atom_prompts,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_112_dev_prompts(clean_targets: np.ndarray, targets_json: Path, seed: int = 2026901) -> Tuple[np.ndarray, np.ndarray]:
    """Build standardized 112-prompt (`7` HOMO quantiles x `16`) zero-leakage calibration dev set."""
    q_points = json.loads(targets_json.read_text())["single_target_points"]["homo"]
    rng = np.random.default_rng(seed)
    sub_n = clean_targets[(clean_targets[:, 1] >= 20.0) & (clean_targets[:, 1] <= 42.0), 1]
    dev_h: List[float] = []
    dev_n: List[float] = []
    for q in (10, 25, 35, 50, 65, 75, 90):
        gh = float(q_points[str(q)])
        picked = rng.choice(sub_n, size=16, replace=True)
        dev_h.extend([gh] * 16)
        dev_n.extend(np.round(picked).tolist())
    return np.asarray(dev_h, dtype=np.float32), np.asarray(dev_n, dtype=np.float32)


def _compute_total_uni3dar_gpuh(pro_meta: Dict[str, Any], ref_meta: Dict[str, Any], critic_meta: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    sft_gpuh = float((pro_meta.get("sft_metadata") or {}).get("sft_total_gpuh", 0.0))
    awr_gpuh = float(pro_meta.get("hindsight_awr_gpuh", 0.0))
    grpo_gpuh = float(pro_meta.get("grpo_gpuh", 0.0))
    xyz_gpuh = float((pro_meta.get("xyz_distill_metadata") or {}).get("xyz_distill_gpuh", 0.0))
    ref_gpuh = float(ref_meta.get("refiner_gpuh", (pro_meta.get("final_refiner_metadata") or {}).get("refiner_gpuh", 0.0)))
    crit_gpuh = float(critic_meta.get("rollout_gpuh", 0.0))
    bd = {
        "pro_relaxed_sft_gpuh": round(sft_gpuh, 4),
        "pro_hindsight_awr_gpuh": round(awr_gpuh, 4),
        "pro_homo_grpo_gpuh": round(grpo_gpuh, 4),
        "pro_xyz_distill_gpuh": round(xyz_gpuh, 4),
        "pro_refiner_gpuh": round(ref_gpuh, 4),
        "pro_ranking_critic_gpuh": round(crit_gpuh, 4),
    }
    return round(sum(bd.values()), 4), bd


def run_rollout_fphys_shard(args: argparse.Namespace) -> None:
    """Collect grouped rollouts with continuous post-refiner `DPA-4` `F_phys` scores (`0` `audit_dpa4` calls)."""
    device = torch.device(args.device)
    pro_dir = args.work_dir / args.run_name
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    shard_out = pro_dir / f"pro_fphys_rollout_shard_{args.shard_id}.pt"

    gen, _ = LoomUni3DARGenerator.load_checkpoint(
        pro_dir / "v4_pro_final.pt", device=args.device, batch_size=512, auto_discover_companions=False
    )
    gen.eval()
    refiner, _ = PostTreeGraphRefiner.load_checkpoint(pro_dir / "v4_pro_post_tree_refiner.pt", device=args.device)
    refiner.eval()

    bus, _, _ = _build_calibrated_oe62_bus(args, include_blind_auditor=False, calib_cache_path=calib_cache)
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    _, clean_targets = _get_clean_oe62_pool(args, None)

    g_sz = int(args.critic_group_size)
    n_prompts_total = int(args.critic_rollouts) // g_sz
    n_shards = max(1, int(args.num_shards))
    p_per_shard = n_prompts_total // n_shards
    n_shard = p_per_shard * g_sz

    rng = np.random.default_rng(2026800 + int(args.shard_id) * 1009)
    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    q_vals = np.array([float(q_points[str(q)]) for q in (10, 25, 35, 50, 65, 75, 90)], dtype=np.float32)
    pool_mid = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 42.0), 1]
    pool_big = clean_targets[(clean_targets[:, 1] > 42.0) & (clean_targets[:, 1] <= 60.0), 1]

    p_big = int(round(p_per_shard * 0.25))
    p_mid = p_per_shard - p_big
    p_homo = np.clip(rng.choice(q_vals, size=p_per_shard, replace=True) + rng.normal(0.0, 0.22, size=p_per_shard), -8.2, -4.8).astype(np.float32)
    p_nat = np.concatenate([rng.choice(pool_mid, size=p_mid, replace=True), rng.choice(pool_big, size=p_big, replace=True)])
    rng.shuffle(p_nat)
    p_nat = np.clip(np.round(p_nat), 18.0, 60.0).astype(np.float32)

    req_homo = np.repeat(p_homo, g_sz)
    req_natoms = np.repeat(p_nat, g_sz)
    group_ids = np.repeat(np.arange(p_per_shard, dtype=np.int64) + int(args.shard_id) * p_per_shard, g_sz)
    cond_2d = np.stack([req_homo, np.full(n_shard, float(args.eval_sa_target), dtype=np.float32)], axis=-1)

    t0 = time.perf_counter()
    s1_h_l, s1_g_l, s2_h_l, s2_c_l, t_norm_l = [], [], [], [], []
    s2_e_sl, s2_p_sl = [], []
    mols_all: List[Optional[Chem.Mol]] = []
    is_clean = np.zeros(n_shard, dtype=bool)

    chunk = 256
    for s in range(0, n_shard, chunk):
        e = min(s + chunk, n_shard)
        nb = e - s
        tn = gen.normalize_targets(cond_2d[s:e])
        cm = torch.ones((nb, 2), dtype=torch.bool, device=device)
        roll = gen.sample_rollouts_with_two_stage_features(
            tn, cm, uncond_active_mask=None, cfg_scale=float(args.cfg_scale), xyz_cfg_scale=1.0,
            cfg_tree_decay=0.20, geometric_guard=True, temperatures=(0.85, 0.30, 0.25),
            seed=810000 + int(args.shard_id) * 60000 + s, use_bf16=(device.type == "cuda"),
            electron_parity=0, leaf_atom_budget=req_natoms[s:e], budget_guidance_scale=0.35,
        )
        s1_h_l.append(roll["s1_h_concat"])
        s1_g_l.append(roll["s1_geom_feats"])
        s2_h_l.append(roll["s2_h_stage2"])
        s2_c_l.append(roll["s2_cloud_feats"])
        t_norm_l.append(tn.detach().cpu())

        trajs_sub = roll["trajectories"]
        refiner.refine_trajectories_batch(trajs_sub)
        for loc_i, tr in enumerate(trajs_sub):
            s2_e_sl.append(tr["stage2_elem_ids"])
            s2_p_sl.append(tr["stage2_leaf_pos"])
            res = reconciler.reconcile_point_cloud(
                tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
            )
            g_i = s + loc_i
            if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
                mols_all.append(res.mol)
                is_clean[g_i] = True
            else:
                mols_all.append(None)

    pred_asset = bus.registry.get_predictor("ext_unimol2")
    rule_asset = bus.registry.get_predictor("builtin_rules")
    u_out = pred_asset.predict_mols(mols_all, batch_size=64)
    r_out = rule_asset.predict_mols(mols_all, batch_size=256)
    valid_mols = [m for m in mols_all if m is not None]
    pot_out = bus.potential_slot.evaluate_molecules(valid_mols, chunk_size=64) if (bus.potential_slot and valid_mols) else None

    fmax_all = np.full(n_shard, np.nan, dtype=np.float32)
    phys_all = np.zeros(n_shard, dtype=np.float32)
    if pot_out is not None:
        r_phys_sub = zero_step_physical_reward(pot_out["f_rms_eV_A"], pot_out["f_max_eV_A"]).numpy()
        cp = 0
        for g_i, m in enumerate(mols_all):
            if m is not None:
                fmax_all[g_i] = float(pot_out["f_max_eV_A"][cp])
                if math.isfinite(float(r_phys_sub[cp])):
                    phys_all[g_i] = float(r_phys_sub[cp])
                cp += 1

    homo_surr = u_out["mean"][:, pred_asset.channel_index("homo")].astype(np.float32)
    sa_vals = r_out["mean"][:, rule_asset.channel_index("sa_score")].astype(np.float32)
    mw_vals = r_out["mean"][:, rule_asset.channel_index("mol_weight")]
    sm_vals = r_out["mean"][:, rule_asset.channel_index("smarts_alert")]
    y_valid = is_clean & u_out["valid_mask"] & np.isfinite(homo_surr) & np.isfinite(sa_vals)
    low_strain = (~np.isfinite(fmax_all)) | (fmax_all <= 3.2)
    is_constr = y_valid & (sa_vals <= 3.45) & (mw_vals <= 550.0) & (sm_vals <= 0.0) & low_strain
    y_raw_2d = np.stack([np.nan_to_num(homo_surr, nan=-6.5), np.nan_to_num(sa_vals, nan=4.0)], axis=-1).astype(np.float32)
    y_surr_norm = gen.normalize_targets(y_raw_2d).detach().cpu()

    wall_s = time.perf_counter() - t0
    torch.save(
        {
            "s1_h_concat": torch.cat(s1_h_l, dim=0),
            "s1_geom_feats": torch.cat(s1_g_l, dim=0),
            "s2_h_stage2": torch.cat(s2_h_l, dim=0),
            "s2_cloud_feats": torch.cat(s2_c_l, dim=0),
            "s2_elem_slices": s2_e_sl,
            "s2_pos_slices": s2_p_sl,
            "targets_norm": torch.cat(t_norm_l, dim=0),
            "y_surrogate_norm": y_surr_norm,
            "y_valid_mask": torch.from_numpy(y_valid),
            "is_legal": torch.from_numpy(is_clean),
            "is_constr_met": torch.from_numpy(is_constr),
            "phys_scores": torch.from_numpy(phys_all),
            "group_ids": torch.from_numpy(group_ids),
            "fmax_median": float(np.nanmedian(fmax_all)) if np.isfinite(fmax_all).any() else float("nan"),
            "wall_sec": wall_s,
        },
        shard_out,
    )
    print(
        f"[Fphys Rollout Shard {args.shard_id}] N={n_shard} | clean={is_clean.mean()*100:.1f}% | "
        f"constr+low_strain={is_constr.mean()*100:.1f}% | med_Fmax={np.nanmedian(fmax_all):.2f} | "
        f"mean_Fphys={phys_all[is_clean].mean():.3f} | wall={wall_s:.1f}s",
        flush=True,
    )


def run_refit_fphys_critic(args: argparse.Namespace) -> None:
    """Fit `F_phys`-aware `TwoStageOctreeProcessCritic` and compare/bundle into `v4_pro_bundled.pt`."""
    device = torch.device(args.device)
    pro_dir = args.work_dir / args.run_name
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"

    gpu_list = [g.strip() for g in args.rollout_gpus.split(",") if g.strip()]
    n_shards = len(gpu_list)
    if not all((pro_dir / f"pro_fphys_rollout_shard_{i}.pt").exists() for i in range(n_shards)):
        t_r0 = time.perf_counter()
        procs = []
        for sh_i, gpu_id in enumerate(gpu_list):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "rollout_fphys_shard",
                "--device",
                f"cuda:{gpu_id}",
                "--shard-id",
                str(sh_i),
                "--num-shards",
                str(n_shards),
                "--critic-rollouts",
                str(args.critic_rollouts),
                "--critic-group-size",
                str(args.critic_group_size),
                "--eval-sa-target",
                str(args.eval_sa_target),
                "--cfg-scale",
                str(args.cfg_scale),
                "--work-dir",
                str(args.work_dir),
                "--run-name",
                str(args.run_name),
                "--step2-dir",
                str(args.step2_dir),
            ]
            if args.allow_tf32:
                cmd.append("--allow-tf32")
            procs.append(subprocess.Popen(cmd, text=True))
        for sh_i, pr in enumerate(procs):
            if pr.wait() != 0:
                raise RuntimeError(f"rollout_fphys_shard {sh_i} failed")
        roll_wall = time.perf_counter() - t_r0
    else:
        shards_meta = [torch.load(pro_dir / f"pro_fphys_rollout_shard_{i}.pt", map_location="cpu", weights_only=False) for i in range(n_shards)]
        roll_wall = max(float(sh["wall_sec"]) for sh in shards_meta)
        del shards_meta

    shards = [torch.load(pro_dir / f"pro_fphys_rollout_shard_{sh_i}.pt", map_location="cpu", weights_only=False) for sh_i in range(n_shards)]
    s1_h = torch.cat([sh["s1_h_concat"] for sh in shards], dim=0)
    s1_g = torch.cat([sh["s1_geom_feats"] for sh in shards], dim=0)
    s2_h = torch.cat([sh["s2_h_stage2"] for sh in shards], dim=0)
    s2_c = torch.cat([sh["s2_cloud_feats"] for sh in shards], dim=0)
    t_all = torch.cat([sh["targets_norm"] for sh in shards], dim=0)
    y_surr = torch.cat([sh["y_surrogate_norm"] for sh in shards], dim=0)
    y_ok = torch.cat([sh["y_valid_mask"] for sh in shards], dim=0)
    is_leg = torch.cat([sh["is_legal"] for sh in shards], dim=0)
    is_con = torch.cat([sh["is_constr_met"] for sh in shards], dim=0)
    phys_sc = torch.cat([sh["phys_scores"] for sh in shards], dim=0)
    gids = torch.cat([sh["group_ids"] for sh in shards], dim=0)
    fmax_med = float(np.nanmean([sh["fmax_median"] for sh in shards]))

    all_e = [es for sh in shards for es in sh["s2_elem_slices"]]
    all_p = [ps for sh in shards for ps in sh["s2_pos_slices"]]
    n_tot_r = len(all_e)
    max_k_r = max(1, max((int(es.numel()) for es in all_e), default=1))
    pad_e = torch.zeros((n_tot_r, max_k_r), dtype=torch.long)
    pad_p = torch.zeros((n_tot_r, max_k_r, 3), dtype=torch.float32)
    pad_v = torch.zeros((n_tot_r, max_k_r), dtype=torch.bool)
    for i, (es, ps) in enumerate(zip(all_e, all_p)):
        ki = int(es.numel())
        if ki > 0:
            pad_e[i, :ki] = es
            pad_p[i, :ki] = ps
            pad_v[i, :ki] = True
    del shards

    gen, gen_payload = LoomUni3DARGenerator.load_checkpoint(
        pro_dir / "v4_pro_final.pt", device=args.device, batch_size=256, auto_discover_companions=True
    )
    emb_d = int(gen.args.emb_dim)
    num_atom = int(gen.backbone.num_atom)
    t_mean_np = gen.target_mean.detach().cpu().numpy()
    t_scale_np = gen.target_scale.detach().cpu().numpy()

    t_f0 = time.perf_counter()
    act_all = torch.ones((n_tot_r, 2), dtype=torch.bool)
    rk_obj_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

    s1_c = OctreeProcessCritic(emb_dim=emb_d, num_targets=2, hidden_dim=512).to(device)
    s1_c.set_normalization(t_mean_np, t_scale_np)
    st1 = fit_octree_process_critic(
        s1_c,
        h_concat=s1_h,
        geom_feats=s1_g,
        targets_norm=t_all,
        active_mask=act_all,
        y_surrogate_norm=y_surr,
        y_valid_mask=y_ok,
        is_legal=is_leg,
        is_constr_met=is_con,
        phys_scores=phys_sc,
        phys_weight=0.70,
        group_ids=gids,
        ranking_weight=0.60,
        ranking_obj_mask=rk_obj_mask,
        epochs=26,
        batch_size=256,
        lr=1.4e-3,
        seed=202681,
    )
    s2_c_mod = OctreeElementProcessCritic(
        emb_dim=emb_d,
        num_targets=2,
        num_atom_classes=num_atom,
        d_node=192,
        n_rbf=24,
        n_gnn_layers=3,
        hidden_dim=512,
    ).to(device)
    s2_c_mod.set_normalization(t_mean_np, t_scale_np)
    st2 = fit_octree_element_process_critic(
        s2_c_mod,
        h_stage2=s2_h,
        elem_ids=pad_e,
        leaf_pos=pad_p,
        leaf_valid=pad_v,
        cloud_feats=s2_c,
        targets_norm=t_all,
        active_mask=act_all,
        y_surrogate_norm=y_surr,
        y_valid_mask=y_ok,
        is_legal=is_leg,
        is_constr_met=is_con,
        phys_scores=phys_sc,
        phys_weight=0.75,
        group_ids=gids,
        ranking_weight=0.70,
        ranking_obj_mask=rk_obj_mask,
        epochs=30,
        batch_size=256,
        lr=1.1e-3,
        seed=202682,
    )
    fit_wall = time.perf_counter() - t_f0
    fphys_critic = TwoStageOctreeProcessCritic(s1_c, s2_c_mod).to(device)
    fphys_critic.eval()
    fphys_meta = {
        "total_rollouts": n_tot_r,
        "group_size": int(args.critic_group_size),
        "n_gnn_layers": 3,
        "fphys_aware": True,
        "phys_weights": {"stage1": 0.70, "stage2": 0.75},
        "ranking_weights": {"stage1": 0.60, "stage2": 0.70},
        "rollout_gpuh": round((roll_wall * n_shards + fit_wall) / 3600.0, 4),
        "clean_rate": round(float(is_leg.float().mean().item()), 4),
        "constr_and_low_strain_rate": round(float(is_con.float().mean().item()), 4),
        "rollout_fmax_median_eV_A": round(fmax_med, 3),
        "stage1_stats": st1.__dict__,
        "stage2_stats": st2.__dict__,
    }
    fphys_ckpt = pro_dir / "v4_pro_two_stage_critic_fphys.pt"
    fphys_critic.save_checkpoint(fphys_ckpt, extra=fphys_meta)

    # Zero-leakage 112-prompt dev comparison against base `v4_pro_two_stage_critic.pt` (if present)
    bus_dev, _, _ = _build_calibrated_oe62_bus(args, include_blind_auditor=False, calib_cache_path=calib_cache)
    _, clean_targets = _get_clean_oe62_pool(args, None)
    dev_h, dev_n = build_112_dev_prompts(clean_targets, args.oe62_targets_json)

    chosen_critic = fphys_critic
    chosen_meta = fphys_meta
    if gen.process_critic is not None:
        svc_base = LoomService(gen, bus_dev)
        svc_base.calibrate_on_dev(dev_h, dev_atom_budget=dev_n, calib_effort="medium", seed=2026901)
        out_b = svc_base.generate(dev_h, atom_budget=dev_n, effort="medium", seed=2026905, include_epistemic_envelope=False)
        ev_b = bus_dev.evaluate_molecules(out_b["molecules"], dev_h[:, None], np.ones((len(dev_h), 1), dtype=bool), include_audit=False)
        hit_b = float(np.mean(ev_b["constrained_clean_hits"]))
        fmax_b = float(np.nanmedian(ev_b["f_max_eV_A"]))

        gen.process_critic = fphys_critic
        svc_fp = LoomService(gen, bus_dev)
        svc_fp.calibrate_on_dev(dev_h, dev_atom_budget=dev_n, calib_effort="medium", seed=2026901)
        out_f = svc_fp.generate(dev_h, atom_budget=dev_n, effort="medium", seed=2026905, include_epistemic_envelope=False)
        ev_f = bus_dev.evaluate_molecules(out_f["molecules"], dev_h[:, None], np.ones((len(dev_h), 1), dtype=bool), include_audit=False)
        hit_f = float(np.mean(ev_f["constrained_clean_hits"]))
        fmax_f = float(np.nanmedian(ev_f["f_max_eV_A"]))
        print(
            f"[Critic 112-Dev Check] base=(hit={hit_b*100:.1f}%, Fmax={fmax_b:.2f}) vs "
            f"fphys=(hit={hit_f*100:.1f}%, Fmax={fmax_f:.2f})",
            flush=True,
        )
        if (hit_b - 0.025 * fmax_b) > (hit_f - 0.025 * fmax_f):
            orig_c, orig_m = TwoStageOctreeProcessCritic.load_checkpoint(pro_dir / "v4_pro_two_stage_critic.pt", device=device)
            chosen_critic = orig_c
            chosen_meta = orig_m

    gen.process_critic = chosen_critic
    meta_out = dict(gen_payload.get("metadata") or {})
    meta_out["bundled_critic_meta"] = chosen_meta
    bundled_path = pro_dir / "v4_pro_bundled.pt"
    gen.save_checkpoint(bundled_path, extra_metadata=meta_out)
    print(f"[Bundled Checkpoint SAVED] {bundled_path} | critic_meta={chosen_meta}", flush=True)


def run_fair30_uni3dar(args: argparse.Namespace) -> None:
    """Experiment A: Fair 30.4-Atom (`N=1,792`) `base/low/medium/high` (`M=1,2,4,8`) via `LoomService`."""
    device = torch.device(args.device)
    pro_dir = args.work_dir / args.run_name
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    sdf_dir = pro_dir / "service_cohorts_sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    bundled_p = pro_dir / "v4_pro_bundled.pt"
    ckpt_to_load = bundled_p if bundled_p.exists() else (pro_dir / "v4_pro_final.pt")
    gen, gen_payload = LoomUni3DARGenerator.load_checkpoint(ckpt_to_load, device=args.device, batch_size=512)
    gen.eval()
    pro_meta = gen_payload.get("metadata") or {}
    ref_meta = pro_meta.get("final_refiner_metadata") or {}
    critic_meta = pro_meta.get("bundled_critic_meta") or {}
    if not critic_meta and (pro_dir / "v4_pro_two_stage_critic.pt").exists():
        _, critic_meta = TwoStageOctreeProcessCritic.load_checkpoint(pro_dir / "v4_pro_two_stage_critic.pt", device="cpu")
    total_gpuh, gpuh_bd = _compute_total_uni3dar_gpuh(pro_meta, ref_meta, critic_meta)

    bus_eval, _, calib_diag = _build_calibrated_oe62_bus(args, include_blind_auditor=True, calib_cache_path=calib_cache)
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)

    svc = LoomService(gen, bus_eval)
    dev_h, dev_n = build_112_dev_prompts(clean_targets, args.oe62_targets_json, seed=2026901)
    cal = svc.calibrate_on_dev(
        dev_h,
        dev_atom_budget=dev_n,
        calib_effort="medium",
        sa_anchor_candidates=(1.85, 1.95, 2.05, 2.15),
        cfg_scale=float(args.cfg_scale),
        high_stage1=int(args.high_stage1),
        seed=2026901,
    )
    print(f"[LoomService 112-Dev Calibration] {cal.to_dict()}", flush=True)

    # Warmup GPU kernels
    svc.generate(np.full(16, -6.4, dtype=np.float32), atom_budget=np.full(16, 30.0, dtype=np.float32), effort="low", seed=17, include_epistemic_envelope=False)
    _sync(device)

    req_30 = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
    arms_out: Dict[str, Any] = {}
    epistemic_report: Optional[Dict[str, Any]] = None

    for effort_name in ("base", "low", "medium", "high"):
        m_val = EFFORT_TO_M[effort_name]
        arm_key = f"{effort_name}_M{m_val}"
        out = svc.generate(
            req_30[:, 0],
            atom_budget=req_30[:, 1],
            effort=effort_name,
            seed=991000 + m_val * 131,
            cfg_scale=float(args.cfg_scale),
            high_stage1=int(args.high_stage1),
            include_epistemic_envelope=False,
        )
        sel_m = out["molecules"]
        sel_l = np.array([m is not None for m in sel_m], dtype=bool)
        sel_x = np.array([(not is_clean_organic_charge_state(m)) if m is not None else False for m in sel_m], dtype=bool)
        sdf_path = sdf_dir / f"UniLoom_Uni3DAR_{arm_key}.sdf"
        arm_res = _score_oe62_size_cohort(
            sel_m,
            sel_l,
            sel_x,
            req_30,
            bus_eval,
            source_graphs,
            source_scaffolds,
            float(out["gen_wall_sec"]),
            0.0,
            float(out["surrogate_wall_sec"]),
            export_sdf_path=sdf_path,
        )
        _, ms1, ms2, mk = gen.resolve_internal_funnel(effort=effort_name, high_stage1=int(args.high_stage1))
        arm_res["effort"] = effort_name
        arm_res["n_candidates"] = m_val
        arm_res["internal_funnel"] = {"m_stage1": ms1, "m_stage2": ms2, "m_keep": mk}
        arm_res["exported_sdf"] = str(sdf_path)
        arms_out[arm_key] = arm_res
        print(
            f"[LoomService Fair-30 {arm_key:10s} ({ms1}->{ms2}->{mk})] "
            f"Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
            f"Legal={arm_res['legal_rate']*100:.1f}% SA_ok={arm_res['sa_leq_3_5_rate_among_legal']*100:.1f}% "
            f"Fmax={arm_res['zero_step_f_max_median_eV_A']:.2f} | "
            f"RewSynth={arm_res['synthesizable_clean_hit_yield']*100:.1f}% | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} "
            f"({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindMAE={arm_res['audit_mae_eV']:.4f} eV",
            flush=True,
        )

        # Build zero-leakage GPAtlas epistemic confidence map on `high_M8` (using calibrated Uni-Mol2 std, 0 audit_dpa4!)
        if effort_name == "high":
            ev_zl = svc._evaluate_surrogate_zero_mlip(sel_m, req_30[:, 0:1])
            atlas_1d = HeteroscedasticGPAtlas1D.fit_from_proposals(
                req_30[:, 0],
                ev_zl["objective_preds"][:, 0],
                ev_zl["legal_mask"] & ev_zl["clean_charge_mask"],
                sigma_cal_samples=ev_zl["objective_stds"][:, 0],
                source_property_values_eV=clean_targets[:, 0],
                axis_name="homo",
                unit="eV",
            )
            svc.gp_atlases["homo"] = atlas_1d
            probe_grid = np.linspace(-8.6, -4.6, 17, dtype=np.float32)
            epistemic_report = svc.diagnose_requests(probe_grid)

    payload = {
        "experiment": "OE62 Fair Fixed-Prompt 30.4-Atom Standardized 4-Tier Benchmark — Uni-Loom Service [Uni-3DAR v4-Pro: base(M=1), low(M=2), medium(M=4), high(M=8)]",
        "benchmark_seed": 20261118,
        "n_prompts": len(req_30),
        "mean_requested_atoms": round(float(np.mean(req_30[:, 1])), 2),
        "total_training_gpuh": total_gpuh,
        "gpuh_breakdown": gpuh_bd,
        "generator_capabilities": svc.generator_capabilities,
        "symmetric_112_dev_calibration": cal.to_dict(),
        "zero_leakage_dft_calibration": calib_diag,
        "self_awareness_epistemic_grid": epistemic_report,
        "arms": arms_out,
    }
    args.fair_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.fair_output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Fair-30 Unified Service SAVED] {args.fair_output_json} (Total GPU-h={total_gpuh})", flush=True)


def run_fair30_pmf(args: argparse.Namespace) -> None:
    """Symmetric `LoomService` 112-Dev Calibration + 4-Tier (`M=1,2,4,8`) Fair 30.4-Atom Eval on `PropMolFlow`."""
    if args.pmf_ckpt is None:
        raise ValueError("--pmf-ckpt is required for --mode fair30_pmf")
    device = torch.device(args.device)
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"

    pmf_gen = LoomFlowMolGenerator.from_propmolflow_checkpoint(
        args.pmf_ckpt,
        device=args.device,
        data_dir=args.pmf_data_dir,
        property_name="homo",
        sample_timesteps=100,
    )
    pmf_gen.eval()

    bus_eval, _, calib_diag = _build_calibrated_oe62_bus(args, include_blind_auditor=True, calib_cache_path=calib_cache)
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)

    svc = LoomService(pmf_gen, bus_eval)
    dev_h, dev_n = build_112_dev_prompts(clean_targets, args.oe62_targets_json, seed=2026901)
    cal = svc.calibrate_on_dev(dev_h, dev_atom_budget=dev_n, calib_effort="medium", seed=2026901)
    print(f"[LoomService PMF 112-Dev Calibration] {cal.to_dict()}", flush=True)

    svc.generate(np.full(8, -6.4, dtype=np.float32), atom_budget=np.full(8, 30.0, dtype=np.float32), effort="base", seed=11, include_epistemic_envelope=False)
    _sync(device)

    req_30 = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
    arms_out: Dict[str, Any] = {}
    for effort_name in ("base", "low", "medium", "high"):
        m_val = EFFORT_TO_M[effort_name]
        arm_key = f"{effort_name}_M{m_val}"
        out = svc.generate(
            req_30[:, 0],
            atom_budget=req_30[:, 1],
            effort=effort_name,
            seed=881000 + m_val * 131,
            include_epistemic_envelope=False,
        )
        sel_m = out["molecules"]
        sel_l = np.array([m is not None for m in sel_m], dtype=bool)
        sel_x = np.array([(not is_clean_organic_charge_state(m)) if m is not None else False for m in sel_m], dtype=bool)
        arm_res = _score_oe62_size_cohort(
            sel_m, sel_l, sel_x, req_30, bus_eval, source_graphs, source_scaffolds,
            float(out["gen_wall_sec"]), 0.0, float(out["surrogate_wall_sec"]),
        )
        arm_res["effort"] = effort_name
        arm_res["n_candidates"] = m_val
        arms_out[arm_key] = arm_res
        print(
            f"[LoomService PMF Fair-30 {arm_key:10s}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Atoms={arm_res['mean_total_atoms']} | Legal={arm_res['legal_rate']*100:.1f}% "
            f"Fmax={arm_res['zero_step_f_max_median_eV_A']:.2f} | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}% | "
            f"BlindMAE={arm_res['audit_mae_eV']:.4f} eV",
            flush=True,
        )

    payload = {
        "experiment": "OE62 Fair Fixed-Prompt 30.4-Atom Standardized 4-Tier Benchmark — Uni-Loom Service [PropMolFlow]",
        "pmf_checkpoint": str(args.pmf_ckpt),
        "total_training_gpuh": round(float(pmf_gen.cum_gpu_hours), 4),
        "generator_capabilities": svc.generator_capabilities,
        "symmetric_112_dev_calibration": cal.to_dict(),
        "zero_leakage_dft_calibration": calib_diag,
        "arms": arms_out,
    }
    args.pmf_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.pmf_output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[PMF Fair-30 Unified Service SAVED] {args.pmf_output_json}", flush=True)


def run_buckets_uni3dar(args: argparse.Namespace) -> None:
    """Experiment B: Multi-Atom-Count Scaling (`N in {20, 30, 40, 50, 60}`, 280 prompts/bucket) via `LoomService`."""
    device = torch.device(args.device)
    pro_dir = args.work_dir / args.run_name
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"

    bundled_p = pro_dir / "v4_pro_bundled.pt"
    ckpt_to_load = bundled_p if bundled_p.exists() else (pro_dir / "v4_pro_final.pt")
    gen, gen_payload = LoomUni3DARGenerator.load_checkpoint(ckpt_to_load, device=args.device, batch_size=512)
    gen.eval()
    pro_meta = gen_payload.get("metadata") or {}
    ref_meta = pro_meta.get("final_refiner_metadata") or {}
    critic_meta = pro_meta.get("bundled_critic_meta") or {}
    if not critic_meta and (pro_dir / "v4_pro_two_stage_critic.pt").exists():
        _, critic_meta = TwoStageOctreeProcessCritic.load_checkpoint(pro_dir / "v4_pro_two_stage_critic.pt", device="cpu")
    total_gpuh, _ = _compute_total_uni3dar_gpuh(pro_meta, ref_meta, critic_meta)

    bus_eval, _, _ = _build_calibrated_oe62_bus(args, include_blind_auditor=True, calib_cache_path=calib_cache)
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)

    svc = LoomService(gen, bus_eval)
    dev_h, dev_n = build_112_dev_prompts(clean_targets, args.oe62_targets_json, seed=2026901)
    cal = svc.calibrate_on_dev(
        dev_h,
        dev_atom_budget=dev_n,
        calib_effort="medium",
        sa_anchor_candidates=(1.85, 1.95, 2.05, 2.15),
        cfg_scale=float(args.cfg_scale),
        high_stage1=int(args.high_stage1),
        seed=2026901,
    )
    svc.generate(np.full(16, -6.4, dtype=np.float32), atom_budget=np.full(16, 30.0, dtype=np.float32), effort="low", seed=19, include_epistemic_envelope=False)
    _sync(device)

    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    bucket_results: Dict[str, Any] = {}

    for n_exact in (20, 30, 40, 50, 60):
        req_b = np.asarray([(float(q_points[str(q)]), float(n_exact)) for q in quantiles for _ in range(40)], dtype=np.float32)
        b_arms: Dict[str, Any] = {}
        for effort_name in ("base", "low", "medium", "high"):
            m_val = EFFORT_TO_M[effort_name]
            arm_key = f"Uni3DAR_{effort_name}_M{m_val}"
            out = svc.generate(
                req_b[:, 0],
                atom_budget=req_b[:, 1],
                effort=effort_name,
                seed=74000 + m_val * 67 + n_exact,
                cfg_scale=float(args.cfg_scale),
                high_stage1=int(args.high_stage1),
                include_epistemic_envelope=False,
            )
            sel_m = out["molecules"]
            sel_l = np.array([m is not None for m in sel_m], dtype=bool)
            sel_x = np.array([(not is_clean_organic_charge_state(m)) if m is not None else False for m in sel_m], dtype=bool)
            b_res = _score_oe62_size_cohort(
                sel_m,
                sel_l,
                sel_x,
                req_b,
                bus_eval,
                source_graphs,
                source_scaffolds,
                float(out["gen_wall_sec"]),
                0.0,
                float(out["surrogate_wall_sec"]),
            )
            b_res["effort"] = effort_name
            b_res["n_candidates"] = m_val
            b_arms[arm_key] = b_res
            print(
                f"[Bucket N={n_exact} {arm_key:20s}] Eff={b_res['effective_output_mol_per_sec']:.2f} mol/s | "
                f"Atoms={b_res['mean_total_atoms']} (err={b_res['mean_natoms_abs_err']}) | "
                f"Legal={b_res['legal_rate']*100:.1f}% Fmax={b_res['zero_step_f_max_median_eV_A']:.2f} | "
                f"BlindSynth={b_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}% | "
                f"BlindMAE={b_res['audit_mae_eV']:.4f} eV",
                flush=True,
            )
        bucket_results[str(n_exact)] = {"target_n_atoms": n_exact, "n_prompts": len(req_b), "arms": b_arms}

    payload = {
        "experiment": "OE62 Controlled Multi-Atom-Count Scaling (N in {20, 30, 40, 50, 60}) — Uni-Loom Service [Uni-3DAR v4-Pro: M=1,2,4,8]",
        "total_training_gpuh": total_gpuh,
        "symmetric_112_dev_calibration": cal.to_dict(),
        "buckets": bucket_results,
    }
    args.buckets_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.buckets_output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Buckets Unified Service SAVED] {args.buckets_output_json}", flush=True)


def run_ablation_m4(args: argparse.Namespace) -> None:
    """Experiment C: Fixed `medium (M=4)` internal architecture ablation (`4->4->4` vs `32->8->4` x `Refiner`)."""
    device = torch.device(args.device)
    pro_dir = args.work_dir / args.run_name
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"

    bundled_p = pro_dir / "v4_pro_bundled.pt"
    ckpt_to_load = bundled_p if bundled_p.exists() else (pro_dir / "v4_pro_final.pt")
    gen, _ = LoomUni3DARGenerator.load_checkpoint(ckpt_to_load, device=args.device, batch_size=512)
    gen.eval()
    bus_eval, _, _ = _build_calibrated_oe62_bus(args, include_blind_auditor=True, calib_cache_path=calib_cache)
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)

    svc = LoomService(gen, bus_eval)
    dev_h, dev_n = build_112_dev_prompts(clean_targets, args.oe62_targets_json, seed=2026901)
    svc.calibrate_on_dev(
        dev_h,
        dev_atom_budget=dev_n,
        calib_effort="medium",
        sa_anchor_candidates=(1.85, 1.95, 2.05, 2.15),
        cfg_scale=float(args.cfg_scale),
        seed=2026901,
    )
    svc.generate(np.full(16, -6.4, dtype=np.float32), atom_budget=np.full(16, 30.0, dtype=np.float32), effort="low", seed=23, include_epistemic_envelope=False)
    _sync(device)

    req_30 = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
    ablation_configs = [
        ("Naive_Outer_4to4to4_NoRefiner", False, False),
        ("Naive_Outer_4to4to4_WithRefiner", False, True),
        ("SpeculativeTree_32to8to4_NoRefiner", True, False),
        ("Full_UniLoom_32to8to4_WithRefiner", True, True),
    ]
    arms_ab: Dict[str, Any] = {}
    for ab_name, use_tree, use_ref in ablation_configs:
        out = svc.generate(
            req_30[:, 0],
            atom_budget=req_30[:, 1],
            effort="medium",
            seed=995000,
            cfg_scale=float(args.cfg_scale),
            enable_internal_tree=use_tree,
            enable_post_tree_refiner=use_ref,
            include_epistemic_envelope=False,
        )
        sel_m = out["molecules"]
        sel_l = np.array([m is not None for m in sel_m], dtype=bool)
        sel_x = np.array([(not is_clean_organic_charge_state(m)) if m is not None else False for m in sel_m], dtype=bool)
        res = _score_oe62_size_cohort(
            sel_m, sel_l, sel_x, req_30, bus_eval, source_graphs, source_scaffolds,
            float(out["gen_wall_sec"]), 0.0, float(out["surrogate_wall_sec"]),
        )
        res["enable_internal_tree"] = use_tree
        res["enable_post_tree_refiner"] = use_ref
        arms_ab[ab_name] = res
        print(
            f"[Ablation M=4 {ab_name:36s}] Eff={res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Legal={res['legal_rate']*100:.1f}% Fmax={res['zero_step_f_max_median_eV_A']:.2f} | "
            f"BlindSynth={res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}% | "
            f"BlindMAE={res['audit_mae_eV']:.4f} eV",
            flush=True,
        )

    payload = {
        "experiment": "OE62 Fixed Medium-Effort (M=4) Internal Architecture Ablation — Speculative Octree Pruning (32->8->4 vs 4->4->4) x PostTreeGraphRefiner",
        "n_prompts": len(req_30),
        "arms": arms_ab,
    }
    args.ablation_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.ablation_output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Ablation M=4 SAVED] {args.ablation_output_json}", flush=True)


def run_multi_objective_sdl(args: argparse.Namespace) -> None:
    """Verify 2D `(HOMO, LUMO)` `RSIDiagnoser.diagnose_nd` & Partial-`NaN` Assay Calibration on real OE62 DFT data."""
    slot = RewardOracleSlot(backend="unimol2", checkpoint_dir=args.oe62_reward_oracle_dir, device=args.device)
    base_asset = SlotOracleAsset(slot, asset_id="unimol2_2d", units=["eV", "eV"])
    cal_asset = ResidualCalibratedAsset(base_asset, asset_id="unimol2_2d_cal")

    def _read_dft_sdf_2d(sdf_p: Path) -> Tuple[List[Chem.Mol], np.ndarray]:
        ms, ys = [], []
        for m in Chem.SDMolSupplier(str(sdf_p), removeHs=False, sanitize=True):
            if m is None:
                continue
            if m.HasProp("DFT_homo") and m.HasProp("DFT_lumo"):
                h_v = float(m.GetProp("DFT_homo"))
                l_v = float(m.GetProp("DFT_lumo"))
            elif m.HasProp("dft_homo_eV") and m.HasProp("dft_lumo_eV"):
                h_v = float(m.GetProp("dft_homo_eV"))
                l_v = float(m.GetProp("dft_lumo_eV"))
            elif m.HasProp("homo") and m.HasProp("lumo"):
                h_v = float(m.GetProp("homo"))
                l_v = float(m.GetProp("lumo"))
            else:
                continue
            if math.isfinite(h_v) and math.isfinite(l_v):
                ms.append(m)
                ys.append([h_v, l_v])
        return ms, np.asarray(ys, dtype=np.float64)

    tr_mols, tr_y = _read_dft_sdf_2d(args.quantum_train_dft_sdf)
    val_mols, val_y = _read_dft_sdf_2d(args.quantum_valid_dft_sdf)

    # Seed in-domain anchors (mid-band HOMO in [-7.0, -5.6], LUMO in [-2.8, -1.2]), hold out deep-acceptor blindspot!
    in_dom = (tr_y[:, 0] >= -7.0) & (tr_y[:, 0] <= -5.6) & (tr_y[:, 1] >= -2.8) & (tr_y[:, 1] <= -1.2)
    anc_m = [tr_mols[i] for i in range(len(tr_mols)) if bool(in_dom[i])]
    anc_y = tr_y[in_dom]
    cal_asset.seed_in_domain_anchors(anc_m, anc_y, max_anchors=256, seed=42)

    pre_out = cal_asset.predict_mols(val_mols, batch_size=64)
    atlas_h = HeteroscedasticGPAtlas1D.fit_from_proposals(
        val_y[:, 0], pre_out["mean"][:, 0], pre_out["valid_mask"],
        sigma_cal_samples=pre_out["std"][:, 0], source_property_values_eV=anc_y[:, 0],
        axis_name="homo", unit="eV",
    )
    atlas_l = HeteroscedasticGPAtlas1D.fit_from_proposals(
        val_y[:, 1], pre_out["mean"][:, 1], pre_out["valid_mask"],
        sigma_cal_samples=pre_out["std"][:, 1], source_property_values_eV=anc_y[:, 1],
        axis_name="lumo", unit="eV",
    )

    grid_2d = np.array(
        [
            [-6.30, -2.00],
            [-6.55, -2.35],
            [-6.80, -2.60],
            [-7.65, -2.20],  # Deep HOMO blindspot
            [-6.40, -3.65],  # Deep LUMO electron-acceptor blindspot
            [-7.80, -3.80],  # Dual deep blindspot
        ],
        dtype=np.float64,
    )
    diagnoser = RSIDiagnoser()
    diag_pre = diagnoser.diagnose_nd(
        [atlas_h.query(grid_2d[:, 0], tolerance_eV=0.25), atlas_l.query(grid_2d[:, 1], tolerance_eV=0.25)],
        tolerances=[0.25, 0.25],
        targets_nd=grid_2d,
    )

    # Simulate asynchronous Self-Driving Lab partial assay readouts (half HOMO-only, half LUMO-only, some joint!)
    ood_idx = np.where((tr_y[:, 0] < -7.0) | (tr_y[:, 1] < -2.8))[0][:96]
    fb_mols = [tr_mols[int(i)] for i in ood_idx]
    fb_partial = tr_y[ood_idx].copy()
    fb_partial[0::3, 1] = np.nan  # Optical/electrochemical assay returned HOMO only
    fb_partial[1::3, 0] = np.nan  # Reduction potential assay returned LUMO only

    blind_val_mask = (val_y[:, 0] < -7.0) | (val_y[:, 1] < -2.8)
    b_val_mols = [val_mols[i] for i in range(len(val_mols)) if bool(blind_val_mask[i])]
    b_val_y = val_y[blind_val_mask]
    cal_stats = IncrementalCalibrator.ingest_labeled_feedback(
        cal_asset, fb_mols, fb_partial, validation_mols=b_val_mols, validation_true_targets=b_val_y
    )

    post_out = cal_asset.predict_mols(val_mols, batch_size=64)
    atlas_h_post = HeteroscedasticGPAtlas1D.fit_from_proposals(
        val_y[:, 0], post_out["mean"][:, 0], post_out["valid_mask"],
        sigma_cal_samples=post_out["std"][:, 0], source_property_values_eV=tr_y[:, 0],
        axis_name="homo", unit="eV",
    )
    atlas_l_post = HeteroscedasticGPAtlas1D.fit_from_proposals(
        val_y[:, 1], post_out["mean"][:, 1], post_out["valid_mask"],
        sigma_cal_samples=post_out["std"][:, 1], source_property_values_eV=tr_y[:, 1],
        axis_name="lumo", unit="eV",
    )
    diag_post = diagnoser.diagnose_nd(
        [atlas_h_post.query(grid_2d[:, 0], tolerance_eV=0.25), atlas_l_post.query(grid_2d[:, 1], tolerance_eV=0.25)],
        tolerances=[0.25, 0.25],
        targets_nd=grid_2d,
    )

    payload = {
        "experiment": "Self-Driving Lab Multi-Objective (HOMO, LUMO) 4-Quadrant Epistemic Diagnosis & Asynchronous Partial-Assay Calibration",
        "grid_targets_eV": grid_2d.tolist(),
        "pre_acquisition_nd_diagnosis": diag_pre,
        "partial_assay_calibration_stats": cal_stats,
        "post_acquisition_nd_diagnosis": diag_post,
    }
    args.sdl_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.sdl_output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Multi-Objective SDL RSI SAVED] {args.sdl_output_json} | stats={cal_stats}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=[
            "rollout_fphys_shard",
            "refit_fphys_critic",
            "fair30_uni3dar",
            "fair30_pmf",
            "buckets_uni3dar",
            "ablation_m4",
            "multi_objective_sdl",
        ],
        required=True,
    )
    p.add_argument("--pmf-ckpt", type=Path, default=None)
    p.add_argument("--pmf-data-dir", type=Path, default=None)
    p.add_argument("--pmf-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_pmf_52h_awr_service_fair_30atom.json"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rollout-gpus", type=str, default="0,1")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--critic-rollouts", type=int, default=24576)
    p.add_argument("--critic-group-size", type=int, default=8)
    p.add_argument("--eval-sa-target", type=float, default=1.95)
    p.add_argument("--cfg-scale", type=float, default=2.20)
    p.add_argument("--high-stage1", type=int, default=64)
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="uni3dar_v4_pro")
    p.add_argument("--step2-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/uni3dar_v4_step2"))
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument("--oe62-master-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"))
    p.add_argument("--oe62-master-valid-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_valid_B.sdf"))
    p.add_argument("--oe62-source-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/source_labels/train_source.sdf"))
    p.add_argument("--oe62-replay-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_B.sdf"))
    p.add_argument("--quantum-train-dft-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_dft.sdf"))
    p.add_argument("--quantum-valid-dft-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_valid_dft.sdf"))
    p.add_argument("--oe62-targets-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/targets_B.json"))
    p.add_argument("--oe62-reward-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"))
    p.add_argument("--oe62-audit-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_e15_calibration_20261114/full_fit/oracle"))
    p.add_argument("--oe62-potential-weights", type=Path, default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"))
    p.add_argument("--fair-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_pro_fair_30atom.json"))
    p.add_argument("--buckets-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_pro_atom_buckets.json"))
    p.add_argument("--ablation-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_pro_m4_ablation.json"))
    p.add_argument("--sdl-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_sdl_multi_objective_epistemic_rsi.json"))
    args = p.parse_args()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.mode == "rollout_fphys_shard":
        run_rollout_fphys_shard(args)
    elif args.mode == "refit_fphys_critic":
        run_refit_fphys_critic(args)
    elif args.mode == "fair30_uni3dar":
        run_fair30_uni3dar(args)
    elif args.mode == "fair30_pmf":
        run_fair30_pmf(args)
    elif args.mode == "buckets_uni3dar":
        run_buckets_uni3dar(args)
    elif args.mode == "ablation_m4":
        run_ablation_m4(args)
    elif args.mode == "multi_objective_sdl":
        run_multi_objective_sdl(args)


if __name__ == "__main__":
    main()
