"""Phase A — Zero-Retrain Inference & Search Optimization on `Uni-3DAR v4` (`v4_step3_final.pt`).

Implements all 4 zero-retrain inference upgrades strictly obeying the four v4 Red Lines
(0 PCA, 0 `n_atoms` in `LoomConditioner`, 0 inference `DPA-4` relaxation, 0 `audit_dpa4` leakage):
  1. Zero-Leakage Damped Affine Prompt Inversion `(slope, intercept)` on `256` dev rollouts (`cal_unimol2` only).
  2. Relaxed `eval_sa_target` (`2.45..2.65`) + `HOMO-only CFG` (`uncond_active_mask=[False, True]`) with
     zero-leakage dev selection over `(eval_sa_target, cfg_scale, budget_guidance_scale)`.
  3. Stage-2 Trunk-Diverse MMR selection (`stage2_diversity_weight=0.45`) so the `m_keep` finalists sent
     to `cal_unimol2` cover distinct chemical scaffolds.
  4. Conservative external selection (`sd_weight=0.30` + `refiner_disp_rms_A` displacement penalty `0.40`)
     to suppress Winner's Curse between `cal_unimol2` and blind `audit_dpa4`.
  5. Full Fair Fixed-Prompt `30.4`-Atom (`N=1,792`, seed `20261118`) scaling funnel including
     `TwoStage_32_to_8_to_3`, `TwoStage_32_to_8_to_4`, and `TwoStage_48_to_12_to_4`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
from loom.weaving import (
    LoomUni3DARGenerator,
    PostTreeGraphRefiner,
    TwoStageOctreeProcessCritic,
)
from scripts.loom_oe62_size_anchored import (
    _get_clean_oe62_pool,
    _get_source_novelty_sets,
    _score_oe62_size_cohort,
    build_fair_30atom_prompts,
)
from scripts.loom_uni3dar_v4_step1_eval import (
    _build_calibrated_oe62_bus,
    _select_v4_step1_from_pool,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fit_damped_affine_prompt_map(
    gen: LoomUni3DARGenerator,
    refiner: PostTreeGraphRefiner,
    cal_unimol2: Any,
    rule_asset: Any,
    dev_homo: np.ndarray,
    dev_natoms: np.ndarray,
    *,
    sa_target: float,
    cfg_scale: float,
    cfg_tree_decay: float,
    homo_only_cfg: bool,
    budget_guidance_scale: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Zero-leakage damped affine prompt inversion fitted on `256` dev rollouts (`0` `audit_dpa4` calls)."""
    nb = len(dev_homo)
    mat = np.stack([dev_homo.astype(np.float32), np.full(nb, float(sa_target), dtype=np.float32)], axis=-1)
    tn = gen.normalize_targets(mat)
    cm = torch.ones((nb, 2), dtype=torch.bool, device=device)
    um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device) if homo_only_cfg else None

    trajs = gen.sample(
        tn,
        cm,
        uncond_active_mask=um,
        cfg_scale=float(cfg_scale),
        xyz_cfg_scale=1.0,
        cfg_tree_decay=float(cfg_tree_decay),
        geometric_guard=True,
        temperatures=(0.85, 0.30, 0.25),
        seed=884422,
        use_bf16=(device.type == "cuda"),
        electron_parity=0,
        leaf_atom_budget=dev_natoms,
        budget_guidance_scale=float(budget_guidance_scale),
    )
    refiner.refine_trajectories_batch(trajs, batch_size=256)
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    mols: List[Optional[Chem.Mol]] = []
    for tr in trajs:
        res = reconciler.reconcile_point_cloud(
            tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
        )
        mols.append(res.mol if (res.reconciled_legal and not res.reconciled_exotic_charge and res.mol is not None) else None)

    u_out = cal_unimol2.predict_mols(mols, batch_size=64)
    r_out = rule_asset.predict_mols(mols, batch_size=256)
    preds = u_out["mean"][:, cal_unimol2.channel_index("homo")]
    sa_v = r_out["mean"][:, rule_asset.channel_index("sa_score")]
    ok = u_out["valid_mask"] & np.isfinite(preds) & (sa_v <= 3.8)

    h_center = -6.60
    xs = dev_homo[ok].astype(np.float64)
    ys = preds[ok].astype(np.float64)
    n_ok = int(ok.sum())
    if n_ok >= 40 and float(np.std(xs)) > 0.15:
        slope, intercept_c = np.polyfit(xs - h_center, ys - h_center, deg=1)
        slope_clamped = float(np.clip(slope, 0.55, 1.15))
        # Damped affine inverse: h_prompt = h_center + alpha*(h_target - h_center) + beta
        alpha = float(np.clip(1.0 + 0.78 * ((1.0 / slope_clamped) - 1.0), 0.90, 1.48))
        beta = float(np.clip(-0.90 * intercept_c / slope_clamped, -0.60, 0.60))
        dev_mae_raw = float(np.mean(np.abs(ys - xs)))
        dev_sa_ok = float(np.mean(sa_v[u_out["valid_mask"]] <= 3.5)) if u_out["valid_mask"].any() else 0.0
        dev_nat = float(np.mean([m.GetNumAtoms() for m in mols if m is not None])) if any(m is not None for m in mols) else 0.0
        dev_nerr = float(np.mean([abs(m.GetNumAtoms() - float(dev_natoms[i])) for i, m in enumerate(mols) if m is not None]))
    else:
        slope, intercept_c, alpha, beta = 1.0, 0.0, 1.0, 0.0
        dev_mae_raw, dev_sa_ok, dev_nat, dev_nerr = 9.9, 0.0, 0.0, 9.9

    return {
        "h_center_eV": h_center,
        "n_valid": n_ok,
        "dev_legal_rate": round(float(np.mean([m is not None for m in mols])), 4),
        "dev_sa_ok_rate": round(dev_sa_ok, 4),
        "dev_mae_raw_eV": round(dev_mae_raw, 4),
        "dev_mean_atoms": round(dev_nat, 2),
        "dev_natoms_err": round(dev_nerr, 2),
        "dev_forward_slope": round(float(slope), 4),
        "dev_forward_bias_eV": round(float(intercept_c), 4),
        "prompt_alpha": round(alpha, 4),
        "prompt_beta_eV": round(beta, 4),
    }


def _apply_affine_prompt_map(homo_1d: np.ndarray, pmap: Dict[str, Any]) -> np.ndarray:
    h0 = float(pmap["h_center_eV"])
    alpha = float(pmap["prompt_alpha"])
    beta = float(pmap["prompt_beta_eV"])
    return np.clip(h0 + alpha * (homo_1d.astype(np.float32) - h0) + beta, -8.50, -4.40)


def _select_best_inference_knobs_on_dev(
    gen: LoomUni3DARGenerator,
    refiner: PostTreeGraphRefiner,
    critic: TwoStageOctreeProcessCritic,
    bus_train: Any,
    cal_unimol2: Any,
    clean_targets: np.ndarray,
    oe62_targets_json: Path,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Zero-leakage 140-prompt dev sweep (`0` `audit_dpa4` calls!) to lock `(sa_target, cfg_scale, guidance_scale)`."""
    q_points = json.loads(oe62_targets_json.read_text())["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    rng = np.random.default_rng(2026901)
    sub_n = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 42.0), 1]

    dev_rows = []
    for q in quantiles:
        gh = float(q_points[str(q)])
        picked = rng.choice(sub_n, size=36, replace=True)
        for ni in picked.tolist():
            dev_rows.append((gh, float(round(ni))))
    dev_req = np.asarray(dev_rows, dtype=np.float32)  # 252 prompts spanning all 7 quantiles
    rule_asset = bus_train.registry.get_predictor("builtin_rules")
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    obj_col_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

    candidates = [
        {"eval_sa_target": 2.55, "cfg_scale": 2.60, "homo_only_cfg": True, "budget_guidance_scale": 0.65},
        {"eval_sa_target": 2.65, "cfg_scale": 2.85, "homo_only_cfg": True, "budget_guidance_scale": 0.65},
        {"eval_sa_target": 2.40, "cfg_scale": 2.50, "homo_only_cfg": True, "budget_guidance_scale": 0.85},
        {"eval_sa_target": 2.55, "cfg_scale": 2.40, "homo_only_cfg": False, "budget_guidance_scale": 0.65},
    ]

    sweep_logs: List[Dict[str, Any]] = []
    best_cfg: Optional[Dict[str, Any]] = None
    best_pmap: Optional[Dict[str, Any]] = None
    best_score = -1e9

    # Test on `140` dev prompts (`20` per quantile) using `TwoStage 16->6->3` under `cal_unimol2` ONLY!
    dev_eval_idx = [q_i * 36 + k for q_i in range(7) for k in range(20)]
    dev_eval = dev_req[dev_eval_idx]
    nb = len(dev_eval)

    for cfg in candidates:
        pmap = _fit_damped_affine_prompt_map(
            gen,
            refiner,
            cal_unimol2,
            rule_asset,
            dev_req[:, 0],
            dev_req[:, 1],
            sa_target=cfg["eval_sa_target"],
            cfg_scale=cfg["cfg_scale"],
            cfg_tree_decay=0.25,
            homo_only_cfg=cfg["homo_only_cfg"],
            budget_guidance_scale=cfg["budget_guidance_scale"],
            device=device,
        )
        h_mapped = _apply_affine_prompt_map(dev_eval[:, 0], pmap)
        mat = np.stack([h_mapped, np.full(nb, float(cfg["eval_sa_target"]), dtype=np.float32)], axis=-1)
        tn = gen.normalize_targets(mat)
        cm = torch.ones((nb, 2), dtype=torch.bool, device=device)
        um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device) if cfg["homo_only_cfg"] else None

        grps = gen.sample_with_two_stage_process_critic(
            tn,
            cm,
            critic,
            m_stage1=16,
            m_stage2=6,
            m_keep=3,
            objective_col_mask=obj_col_mask,
            stage1_legal_weight=0.80,
            stage1_constr_weight=0.55,
            stage2_legal_weight=1.05,
            stage2_constr_weight=0.70,
            stage2_diversity_weight=0.45,
            coarse_chunk_size=1024,
            cfg_scale=float(cfg["cfg_scale"]),
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=55001,
            use_bf16=True,
            electron_parity=0,
            uncond_active_mask=um,
            leaf_atom_budget=dev_eval[:, 1],
            budget_guidance_scale=float(cfg["budget_guidance_scale"]),
        )
        refiner.refine_trajectories_batch([tr for g in grps for tr in g], batch_size=256)

        p_grid: List[List[Optional[Chem.Mol]]] = [[None] * 3 for _ in range(nb)]
        p_leg = np.zeros((nb, 3), dtype=bool)
        p_exo = np.zeros((nb, 3), dtype=bool)
        p_disp = np.full((nb, 3), np.nan, dtype=np.float32)
        for i in range(nb):
            for k in range(3):
                tr = grps[i][k]
                p_disp[i, k] = float(tr.get("refiner_disp_rms_A", np.nan))
                res = reconciler.reconcile_point_cloud(
                    tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
                )
                if res.reconciled_legal and res.mol is not None:
                    p_grid[i][k] = res.mol
                    p_leg[i, k] = True
                    p_exo[i, k] = bool(res.reconciled_exotic_charge)

        sel_m, sel_l, sel_x, _ = _select_v4_step1_from_pool(
            p_grid, p_leg, p_exo, dev_eval, bus_train, 3, use_surrogate=True, device=device,
            sd_weight=0.30, disp_rms_grid=p_disp, disp_penalty_weight=0.40,
        )
        u_out = cal_unimol2.predict_mols(sel_m, batch_size=64)
        r_out = rule_asset.predict_mols(sel_m, batch_size=256)
        hp = u_out["mean"][:, cal_unimol2.channel_index("homo")]
        hsd = u_out["std"][:, cal_unimol2.channel_index("homo")]
        sa = r_out["mean"][:, rule_asset.channel_index("sa_score")]
        ok = sel_l & (~sel_x) & u_out["valid_mask"] & np.isfinite(hp)
        hit = ok & (sa <= 3.5) & (np.abs(hp - dev_eval[:, 0]) <= 0.25)
        mae = float(np.mean(np.abs(hp[ok] - dev_eval[ok, 0]))) if ok.any() else 9.9
        mean_sd = float(np.mean(hsd[ok])) if ok.any() else 9.9
        nat = np.array([m.GetNumAtoms() for m in sel_m if m is not None], dtype=np.float32)
        mean_nat = float(np.mean(nat)) if len(nat) else 0.0
        n_err = float(np.mean([abs(m.GetNumAtoms() - float(dev_eval[i, 1])) for i, m in enumerate(sel_m) if m is not None]))
        # Score favors high cal_unimol2 synth hit yield, low MAE, low uncertainty, and tight 30.4-atom match
        sc = float(hit.mean()) - 0.35 * mae - 0.15 * mean_sd - 0.03 * max(0.0, abs(mean_nat - 30.38) - 0.6)
        rec = {
            **cfg,
            "pmap": pmap,
            "dev_16_6_3_hit_yield": round(float(hit.mean()), 4),
            "dev_16_6_3_mae_eV": round(mae, 4),
            "dev_16_6_3_mean_sd_eV": round(mean_sd, 4),
            "dev_16_6_3_sa_ok": round(float(np.mean(sa[ok] <= 3.5)) if ok.any() else 0.0, 4),
            "dev_16_6_3_mean_atoms": round(mean_nat, 2),
            "dev_16_6_3_natoms_err": round(n_err, 2),
            "dev_score": round(sc, 4),
        }
        sweep_logs.append(rec)
        print(f"[PhaseA Dev Sweep] {rec}", flush=True)
        if sc > best_score:
            best_score = sc
            best_cfg = cfg
            best_pmap = pmap

    assert best_cfg is not None and best_pmap is not None
    return best_cfg, best_pmap, sweep_logs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--run-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/uni3dar_v4_step3_clean"))
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
    p.add_argument("--output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_phaseA_zero_retrain_fair_30atom.json"))
    p.add_argument("--sd-weight", type=float, default=0.30)
    p.add_argument("--disp-penalty-weight", type=float, default=0.40)
    p.add_argument("--stage2-diversity-weight", type=float, default=0.45)
    args = p.parse_args()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    calib_cache = args.step2_dir / "residual_calibrated_unimol2.pt"
    sdf_dir = args.run_dir / "cohorts_sdf_phaseA"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    gen, gen_payload = LoomUni3DARGenerator.load_checkpoint(args.run_dir / "v4_step3_final.pt", device=args.device, batch_size=512)
    gen.eval()
    refiner, ref_meta = PostTreeGraphRefiner.load_checkpoint(args.run_dir / "v4_step3_post_tree_refiner.pt", device=args.device)
    refiner.eval()
    critic, critic_meta = TwoStageOctreeProcessCritic.load_checkpoint(args.run_dir / "v4_step3_two_stage_critic.pt", device=args.device)
    critic.eval()

    bus_eval, cal_unimol2, calib_diag = _build_calibrated_oe62_bus(
        args, include_blind_auditor=True, calib_cache_path=calib_cache
    )
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)

    best_cfg, best_pmap, sweep_logs = _select_best_inference_knobs_on_dev(
        gen, refiner, critic, bus_eval, cal_unimol2, clean_targets, args.oe62_targets_json, device
    )
    print(f"[PhaseA Selected Dev Config] cfg={best_cfg} | pmap={best_pmap}", flush=True)

    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    obj_col_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

    def _encode(homo_1d: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        nb = len(homo_1d)
        h_m = _apply_affine_prompt_map(homo_1d, best_pmap)
        mat = np.stack([h_m, np.full(nb, float(best_cfg["eval_sa_target"]), dtype=np.float32)], axis=-1)
        tn = gen.normalize_targets(mat)
        ma = torch.ones((nb, 2), dtype=torch.bool, device=device)
        um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device) if best_cfg["homo_only_cfg"] else None
        return tn, ma, um

    # Warmup
    wt, wm, wu = _encode(np.full(16, -6.4, dtype=np.float32))
    wg = gen.sample_with_two_stage_process_critic(
        wt, wm, critic, m_stage1=8, m_stage2=4, m_keep=2, objective_col_mask=obj_col_mask,
        stage2_diversity_weight=float(args.stage2_diversity_weight),
        cfg_scale=float(best_cfg["cfg_scale"]), cfg_tree_decay=0.25, geometric_guard=True, seed=19, use_bf16=True,
        uncond_active_mask=wu, leaf_atom_budget=np.full(16, 30.0, dtype=np.float32),
        budget_guidance_scale=float(best_cfg["budget_guidance_scale"]),
    )
    refiner.refine_trajectories_batch([tr for g in wg for tr in g])
    _sync(device)

    req_30 = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
    n_p30 = len(req_30)
    funnels: List[Tuple[str, int, int, int, bool]] = [
        ("Outcome_M1", 1, 1, 1, False),
        ("TwoStage_8_to_4_to_1", 8, 4, 1, False),
        ("TwoStage_8_to_4_to_2", 8, 4, 2, True),
        ("TwoStage_16_to_6_to_2", 16, 6, 2, True),
        ("TwoStage_24_to_8_to_2", 24, 8, 2, True),
        ("TwoStage_32_to_8_to_2", 32, 8, 2, True),
        ("TwoStage_32_to_8_to_3", 32, 8, 3, True),
        ("TwoStage_32_to_8_to_4", 32, 8, 4, True),
        ("TwoStage_48_to_12_to_4", 48, 12, 4, True),
    ]

    arms_out: Dict[str, Any] = {}
    for arm_name, ms1, ms2, mkeep, use_surr in funnels:
        _sync(device)
        tg0 = time.perf_counter()
        groups_all: List[List[Dict[str, Any]]] = []
        p_chunk = 512 if ms1 == 1 else (128 if ms1 >= 48 else 256)
        for s in range(0, n_p30, p_chunk):
            e = min(s + p_chunk, n_p30)
            bt, bm, bu = _encode(req_30[s:e, 0])
            groups_all.extend(
                gen.sample_with_two_stage_process_critic(
                    bt, bm, critic, m_stage1=ms1, m_stage2=ms2, m_keep=mkeep,
                    objective_col_mask=obj_col_mask, stage1_legal_weight=0.80, stage1_constr_weight=0.55,
                    stage2_legal_weight=1.05, stage2_constr_weight=0.70,
                    stage2_diversity_weight=float(args.stage2_diversity_weight),
                    coarse_chunk_size=1024,
                    cfg_scale=float(best_cfg["cfg_scale"]), xyz_cfg_scale=1.0, cfg_tree_decay=0.25,
                    geometric_guard=True, temperatures=(0.85, 0.30, 0.25),
                    seed=980000 + ms1 * 409 + ms2 * 61 + mkeep * 7 + s,
                    use_bf16=True, electron_parity=0, uncond_active_mask=bu,
                    leaf_atom_budget=req_30[s:e, 1],
                    budget_guidance_scale=float(best_cfg["budget_guidance_scale"]),
                )
            )
        refiner.refine_trajectories_batch([tr for g in groups_all for tr in g], batch_size=256)
        _sync(device)
        gen_w = time.perf_counter() - tg0

        tr0 = time.perf_counter()
        p_grid: List[List[Optional[Chem.Mol]]] = [[None] * mkeep for _ in range(n_p30)]
        p_leg = np.zeros((n_p30, mkeep), dtype=bool)
        p_exo = np.zeros((n_p30, mkeep), dtype=bool)
        p_disp = np.full((n_p30, mkeep), np.nan, dtype=np.float32)
        for i in range(n_p30):
            for k in range(mkeep):
                tr = groups_all[i][k]
                p_disp[i, k] = float(tr.get("refiner_disp_rms_A", np.nan))
                res = reconciler.reconcile_point_cloud(
                    tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
                )
                if res.reconciled_legal and res.mol is not None:
                    p_grid[i][k] = res.mol
                    p_leg[i, k] = True
                    p_exo[i, k] = bool(res.reconciled_exotic_charge)
        rec_w = time.perf_counter() - tr0

        sel_m, sel_l, sel_x, surr_w = _select_v4_step1_from_pool(
            p_grid, p_leg, p_exo, req_30, bus_eval, mkeep, use_surrogate=use_surr, device=device,
            sd_weight=float(args.sd_weight), disp_rms_grid=p_disp, disp_penalty_weight=float(args.disp_penalty_weight),
        )
        arm_res = _score_oe62_size_cohort(
            sel_m, sel_l, sel_x, req_30, bus_eval, source_graphs, source_scaffolds, gen_w, rec_w, surr_w,
            export_sdf_path=(sdf_dir / f"{arm_name}.sdf"),
        )
        arm_res["funnel"] = {"m_stage1": ms1, "m_stage2": ms2, "m_keep": mkeep, "use_surrogate": use_surr}
        arms_out[arm_name] = arm_res
        print(
            f"[PhaseA 30-Atom {arm_name}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
            f"Legal={arm_res['legal_rate']*100:.1f}% SA_ok={arm_res['sa_leq_3_5_rate_among_legal']*100:.1f}% Fmax={arm_res['zero_step_f_max_median_eV_A']:.2f} | "
            f"RewSynth={arm_res['synthesizable_clean_hit_yield']*100:.1f}% | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} ({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindMAE={arm_res['audit_mae_eV']:.4f} eV",
            flush=True,
        )

    step3_meta = gen_payload.get("metadata") or {}
    total_gpuh = round(
        float((step3_meta.get("synth_sft_metadata") or {}).get("synth_sft_gpuh", 0.0))
        + float(step3_meta.get("hindsight_awr_gpuh", 0.0))
        + float((step3_meta.get("xyz_distill_metadata") or {}).get("xyz_distill_gpuh", 0.0))
        + float(ref_meta.get("refiner_gpuh", 0.0))
        + float(step3_meta.get("grpo_gpuh", 0.0))
        + float(critic_meta.get("rollout_gpuh", 0.0)),
        4,
    )
    payload = {
        "experiment": "OE62 Fair 30.4-Atom — Uni-3DAR v4 Phase A (Zero-Retrain Inference Opt: Damped Affine Inversion + HOMO-Only CFG + Stage-2 MMR + Conservative sd_h/disp + 32->8->4 / 48->12->4)",
        "benchmark_seed": 20261118,
        "n_prompts": n_p30,
        "total_training_gpuh": total_gpuh,
        "selected_inference_cfg": best_cfg,
        "damped_affine_prompt_map": best_pmap,
        "dev_sweep_logs": sweep_logs,
        "sd_weight": float(args.sd_weight),
        "disp_penalty_weight": float(args.disp_penalty_weight),
        "stage2_diversity_weight": float(args.stage2_diversity_weight),
        "arms": arms_out,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[PhaseA ALL DONE] Saved {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
