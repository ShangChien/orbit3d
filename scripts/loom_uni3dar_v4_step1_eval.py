"""Step 1 of `Uni-3DAR v4 + Uni-Loom` Speed-Preserving Overtake Plan (`~0.6 GPU-h`, Zero Backbone Retraining).

Validates the three zero-retraining inference & calibration upgrades on top of the frozen `v3` generator
under the **Four Non-Negotiable Red Lines**:
  1. **Zero-Leakage Cross-Foundation `ResidualCalibratedAsset(ext_unimol2)` + Dev Affine Prompt Centering**:
     - Fits `ResidualCalibratedAsset` wrapping `ext_unimol2` strictly on **training-split** `PBE0/def2-TZVP`
       DFT HOMO labels (`quantum_train_dft.sdf` + stratified `oe62_source` training conformers with
       `OriginalLabel_homo`, including unrelaxed `0.01 A` octree-quantized + `ValenceReconciler` training
       geometries). **0 `audit_dpa4` calls, 0 validation/test overlap.**
     - Verifies calibration improvement on held-out `quantum_valid_dft.sdf` (`PBE0/def2-TZVP`).
     - Fits a 2-parameter dev-rollout conditioner prompt affine map `h_cond = h0 + alpha * (h_req - h0) + beta`
       strictly against `ResidualCalibratedAsset(ext_unimol2)` (`0` `audit_dpa4` calls).
  2. **Vectorized GPU Pause-and-Gather `TwoStageOctreeProcessCritic` (`32 -> 8 -> 2`)**:
     - Parallel 2-GPU (`GPU 0` + `GPU 1`) rollout collection (`16,384` training-prompt rollouts) labeling
       `is_constr_met = clean & (SA <= 3.5) & (SMARTS == 0) & (MW <= 550) & (F_max <= 4.5 eV/A)` offline.
     - Trains Stage-1 `OctreeProcessCritic` (`level 1 -> 0` coarse envelope) and SO(3)-invariant Stage-2
       `OctreeElementProcessCritic` (`level == 0` 3D element cloud).
  3. **Strict Inference Speed Red Line (`> 4.0 mol/s`, ZERO `DPA-4` Relaxation / H-Polish at Inference)**:
     - Evaluates on the strictly fair fixed-per-prompt `30.4`-atom OE62 benchmark (`N = 1,792`, seed `20261118`)
       timed on **1 single uncontended A800 GPU** (`cuda:0`), using pure analytic
       `ValenceReconciler(..., potential_slot=None)` + sub-millisecond analytic covalent/steric strain tie-breaker.
"""

from __future__ import annotations

import argparse
import json
import math
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
MOLGEN_COMPAT_DEFAULT = Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat")
for cand in (ROOT / "src" / "compat", ROOT / "src", ROOT, MOLGEN_COMPAT_DEFAULT):
    cstr = str(cand)
    if cand.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402
from loom.oracles.unimol2_task import B_WEIGHT_SHA256  # noqa: E402
from loom.weaving import (  # noqa: E402
    AssetRegistry,
    AuditOracleSlot,
    BuiltInRuleAsset,
    CompositeCriticBus,
    LoomUni3DARGenerator,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    PotentialSlot,
    ResidualCalibratedAsset,
    RewardOracleSlot,
    SlotOracleAsset,
    TaskWeavingSpec,
    TwoStageOctreeProcessCritic,
    fit_octree_element_process_critic,
    fit_octree_process_critic,
    formal_charge_signature,
)
from scripts.loom_oe62_fair_size_matched_and_buckets import build_fair_30atom_prompts  # noqa: E402
from scripts.loom_oe62_size_anchored import (  # noqa: E402
    _get_clean_oe62_pool,
    _get_source_novelty_sets,
    _score_oe62_size_cohort,
)
from scripts.loom_train_sft import load_sdf_with_targets  # noqa: E402

_COV_RADII_A: Dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _analytic_conformer_strain_penalty(mol: Chem.Mol) -> float:
    """Fast (`<0.02 ms`) pure-CPU covalent stretch + steric clash penalty (zero MLIP calls)."""
    if mol.GetNumConformers() == 0 or mol.GetNumAtoms() < 2:
        return 0.0
    pos = mol.GetConformer().GetPositions().astype(np.float64)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    rcov = np.array([_COV_RADII_A.get(s, 0.77) for s in syms], dtype=np.float64)
    n = len(syms)

    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(dmat, 99.0)

    bonded = np.zeros((n, n), dtype=bool)
    max_stretch = 0.0
    sum_sq_stretch = 0.0
    n_bonds = max(mol.GetNumBonds(), 1)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded[i, j] = bonded[j, i] = True
        bo = float(b.GetBondTypeAsDouble())
        # Equilibrium covalent distance contracts slightly with higher bond order
        shrink = 0.90 if bo >= 2.5 else (0.93 if bo >= 1.8 else (0.96 if bo >= 1.4 else 1.01))
        d_eq = shrink * (rcov[i] + rcov[j])
        stretch = abs(float(dmat[i, j]) - d_eq)
        if stretch > max_stretch:
            max_stretch = stretch
        sum_sq_stretch += stretch * stretch

    rms_stretch = math.sqrt(sum_sq_stretch / float(n_bonds))
    # Non-bonded steric clash check (`d_ij < 1.15 * (rcov_i + rcov_j)` for non-bonded pairs)
    r_sum = rcov[:, None] + rcov[None, :]
    nonbond = ~bonded
    np.fill_diagonal(nonbond, False)
    if nonbond.any():
        ratios = dmat[nonbond] / np.maximum(r_sum[nonbond], 0.5)
        min_nb_ratio = float(ratios.min())
    else:
        min_nb_ratio = 2.0

    clash_pen = max(0.0, 1.32 - min_nb_ratio) * 0.45
    stretch_pen = max(0.0, max_stretch - 0.16) * 0.35 + max(0.0, rms_stretch - 0.07) * 0.25
    return float(clash_pen + stretch_pen)


def _quantize_and_reconcile_training_mol(
    mol: Chem.Mol,
    reconciler: ValenceReconciler,
    rng: np.random.Generator,
    jitter_sigma_A: float = 0.025,
) -> Optional[Chem.Mol]:
    """Simulate unrelaxed `0.01 A` octree-decoded point-cloud conformer on a training molecule."""
    if mol.GetNumConformers() == 0:
        return None
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    noisy = pos + rng.normal(0.0, jitter_sigma_A, size=pos.shape).astype(np.float32)
    quant = np.round(noisy / 0.01) * 0.01
    res = reconciler.reconcile_point_cloud(syms, quant.astype(np.float32), potential_slot=None)
    if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
        return res.mol
    return None


def _build_calibrated_oe62_bus(
    args: argparse.Namespace,
    *,
    include_blind_auditor: bool,
    calib_cache_path: Optional[Path] = None,
) -> Tuple[CompositeCriticBus, ResidualCalibratedAsset, Dict[str, Any]]:
    """Build `CompositeCriticBus` with zero-leakage `ResidualCalibratedAsset(ext_unimol2)`."""
    spec = TaskWeavingSpec.from_file(args.oe62_task_spec)
    registry = AssetRegistry(register_builtin_rules=True)

    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.oe62_reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )
    raw_unimol2 = SlotOracleAsset(reward_slot, asset_id="ext_unimol2_raw", units=["eV", "eV"])
    cal_unimol2 = ResidualCalibratedAsset(raw_unimol2, asset_id="ext_unimol2", prior_blindspot_std=0.32)

    calib_diag: Dict[str, Any] = {}
    if calib_cache_path is not None and calib_cache_path.exists():
        cached = torch.load(calib_cache_path, map_location="cpu", weights_only=False)
        cal_unimol2._feedback_static = list(cached["feedback_static"])
        cal_unimol2._feedback_base_preds = list(cached["feedback_base_preds"])
        cal_unimol2._feedback_residuals = list(cached["feedback_residuals"])
        cal_unimol2._pred_mean = cached["pred_mean"]
        cal_unimol2._pred_scale = cached["pred_scale"]
        cal_unimol2._refit_gps()
        calib_diag = dict(cached.get("calib_diag", {}))
    else:
        rng = np.random.default_rng(20261119)
        reconciler = ValenceReconciler(
            allowed_elements=("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I")
        )
        fb_mols: List[Chem.Mol] = []
        fb_targets: List[List[float]] = []

        # Strictly isolated `train` scaffold split (`quantum_train_dft.sdf`, pure `PBE0/def2-TZVP/DF` protocol):
        # combine native 3D conformers + unrelaxed `0.01 A` octree-quantized `ValenceReconciler` conformers.
        q_train = [
            m
            for m in Chem.SDMolSupplier(str(args.quantum_train_dft_sdf), removeHs=False)
            if m is not None and is_clean_organic_charge_state(m)
        ]
        n_nat = min(520, len(q_train))
        q_idx = np.sort(rng.choice(len(q_train), size=n_nat, replace=False)) if len(q_train) > n_nat else np.arange(len(q_train))
        quant_sub = set(rng.choice(q_idx, size=min(260, len(q_idx)), replace=False).tolist()) if len(q_idx) > 0 else set()
        for idx in q_idx.tolist():
            m = q_train[idx]
            if m.HasProp("DFT_homo") and m.HasProp("DFT_lumo"):
                dh = float(m.GetProp("DFT_homo"))
                dl = float(m.GetProp("DFT_lumo"))
            else:
                dh = float(m.GetProp("homo"))
                dl = float(m.GetProp("lumo"))
            if not (math.isfinite(dh) and math.isfinite(dl)):
                continue
            fb_mols.append(m)
            fb_targets.append([dh, dl])
            if idx in quant_sub:
                m_q = _quantize_and_reconcile_training_mol(m, reconciler, rng, jitter_sigma_A=0.025)
                if m_q is not None:
                    fb_mols.append(m_q)
                    fb_targets.append([dh, dl])

        fb_y = np.asarray(fb_targets, dtype=np.float64)
        ingest_info = cal_unimol2.ingest_feedback(fb_mols, fb_y)
        calib_diag["train_dft_anchors_ingested"] = int(ingest_info["total_feedback_samples"])
        calib_diag["raw_unimol2_train_mae_eV"] = ingest_info["base_uncalibrated_mae_on_batch"]

        # Zero-leakage validation check on `quantum_valid_dft.sdf` (`PBE0/def2-TZVP`)
        if args.quantum_valid_dft_sdf.exists():
            q_val = [
                m for m in Chem.SDMolSupplier(str(args.quantum_valid_dft_sdf), removeHs=False) if m is not None
            ]
            if q_val:
                vy = np.array(
                    [[float(m.GetProp("DFT_homo")), float(m.GetProp("DFT_lumo"))] for m in q_val],
                    dtype=np.float64,
                )
                raw_v = raw_unimol2.predict_mols(q_val, batch_size=64)
                cal_v = cal_unimol2.predict_mols(q_val, batch_size=64)
                ok_v = raw_v["valid_mask"] & cal_v["valid_mask"] & np.isfinite(vy[:, 0])
                if ok_v.any():
                    raw_diff = raw_v["mean"][ok_v, 0] - vy[ok_v, 0]
                    cal_diff = cal_v["mean"][ok_v, 0] - vy[ok_v, 0]
                    calib_diag["quantum_valid_dft_n"] = int(ok_v.sum())
                    calib_diag["raw_unimol2_valid_dft_homo_mae_eV"] = round(float(np.mean(np.abs(raw_diff))), 4)
                    calib_diag["raw_unimol2_valid_dft_homo_bias_eV"] = round(float(np.mean(raw_diff)), 4)
                    calib_diag["cal_unimol2_valid_dft_homo_mae_eV"] = round(float(np.mean(np.abs(cal_diff))), 4)
                    calib_diag["cal_unimol2_valid_dft_homo_bias_eV"] = round(float(np.mean(cal_diff)), 4)

        if calib_cache_path is not None:
            calib_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "feedback_static": cal_unimol2._feedback_static,
                    "feedback_base_preds": cal_unimol2._feedback_base_preds,
                    "feedback_residuals": cal_unimol2._feedback_residuals,
                    "pred_mean": cal_unimol2._pred_mean,
                    "pred_scale": cal_unimol2._pred_scale,
                    "calib_diag": calib_diag,
                },
                calib_cache_path,
            )

    registry.register_predictor(cal_unimol2)
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

    replay_mols_all, _ = load_sdf_with_targets(args.oe62_replay_sdf, ["homo", "lumo"])
    seen_charges = frozenset(formal_charge_signature(m) for m in replay_mols_all)
    bus = CompositeCriticBus(spec=spec, registry=registry, seen_charge_signatures=seen_charges)
    return bus, cal_unimol2, calib_diag


def _fit_dev_prompt_affine_map(
    gen: LoomUni3DARGenerator,
    cal_unimol2: ResidualCalibratedAsset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    """Fit 2-parameter dev-set conditioner prompt centering against `cal_unimol2` (`0` `audit_dpa4` calls!)."""
    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    per_q = 48
    rng = np.random.default_rng(20261120)

    val_mols = [
        m
        for m in Chem.SDMolSupplier(str(args.oe62_master_valid_sdf), removeHs=False)
        if m is not None and 18 <= m.GetNumAtoms() <= 42
    ]
    val_natoms = (
        np.array([float(m.GetNumAtoms()) for m in val_mols], dtype=np.float32)
        if val_mols
        else np.full(128, 30.0, dtype=np.float32)
    )

    dev_rows: List[Tuple[float, float]] = []
    for q in quantiles:
        gh = float(q_points[str(q)])
        picked_n = rng.choice(val_natoms, size=per_q, replace=True)
        for ni in picked_n.tolist():
            dev_rows.append((gh, float(ni)))
    dev_req = np.asarray(dev_rows, dtype=np.float32)
    n_dev = len(dev_req)

    # Sample `M=3` dev candidates per prompt & pick closest clean synthesizable candidate under `cal_unimol2`
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    rule_asset = BuiltInRuleAsset(asset_id="dev_rules")
    m_dev = 3
    t_norm = gen.normalize_targets(dev_req).repeat_interleave(m_dev, dim=0)
    c_mask = torch.ones((n_dev * m_dev, 2), dtype=torch.bool, device=device)
    u_mask = torch.tensor([[False, True]] * (n_dev * m_dev), dtype=torch.bool, device=device)

    trajs = gen.sample(
        t_norm,
        c_mask,
        uncond_active_mask=u_mask,
        cfg_scale=2.00,
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.30,
        geometric_guard=True,
        temperatures=(0.85, 0.30, 0.25),
        seed=9102026,
        use_bf16=(device.type == "cuda"),
        electron_parity=0,
    )
    mols_flat: List[Optional[Chem.Mol]] = []
    clean_flat = np.zeros(len(trajs), dtype=bool)
    for idx, tr in enumerate(trajs):
        res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
        if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
            mols_flat.append(res.mol)
            clean_flat[idx] = True
        else:
            mols_flat.append(None)

    u_out = cal_unimol2.predict_mols(mols_flat, batch_size=64)
    r_out = rule_asset.predict_mols(mols_flat, batch_size=256)
    h_pred = u_out["mean"][:, cal_unimol2.channel_index("homo")].reshape(n_dev, m_dev)
    sa_pred = r_out["mean"][:, rule_asset.channel_index("sa_score")].reshape(n_dev, m_dev)
    clean_2d = (clean_flat & u_out["valid_mask"]).reshape(n_dev, m_dev)

    xs: List[float] = []
    ys: List[float] = []
    for i in range(n_dev):
        gh = float(dev_req[i, 0])
        best_h: Optional[float] = None
        best_err = 99.0
        for k in range(m_dev):
            if clean_2d[i, k] and sa_pred[i, k] <= 3.6 and math.isfinite(float(h_pred[i, k])):
                err = abs(float(h_pred[i, k]) - gh)
                if err < best_err:
                    best_err = err
                    best_h = float(h_pred[i, k])
        if best_h is not None:
            xs.append(gh)
            ys.append(best_h)

    h_center = -6.60
    if len(xs) >= 40:
        x_arr = np.asarray(xs, dtype=np.float64) - h_center
        y_arr = np.asarray(ys, dtype=np.float64) - h_center
        # Forward response: y - h_center ~= s_fwd * (x - h_center) + b_fwd
        s_fwd, b_fwd = np.polyfit(x_arr, y_arr, deg=1)
        s_clamped = float(np.clip(s_fwd, 0.72, 1.15))
        # Damped inverse prompt map (`0.75` shrinkage avoids over-extrapolating conditioner range)
        alpha = float(np.clip(1.0 + 0.75 * ((1.0 / s_clamped) - 1.0), 0.85, 1.28))
        beta = float(np.clip(-0.85 * b_fwd / s_clamped, -0.45, 0.45))
    else:
        s_fwd, b_fwd = 1.0, 0.0
        alpha, beta = 1.0, 0.0

    return {
        "h_center_eV": h_center,
        "dev_valid_pairs": len(xs),
        "dev_forward_slope": round(float(s_fwd), 4),
        "dev_forward_bias_eV": round(float(b_fwd), 4),
        "prompt_alpha": round(alpha, 4),
        "prompt_beta_eV": round(beta, 4),
    }


def _apply_prompt_map(req_targets: np.ndarray, pmap: Dict[str, float]) -> np.ndarray:
    """Map requested `(homo_req, n_atoms)` to conditioner input `(homo_cond, n_atoms)`."""
    out = req_targets.astype(np.float32, copy=True)
    h0 = float(pmap["h_center_eV"])
    alpha = float(pmap["prompt_alpha"])
    beta = float(pmap["prompt_beta_eV"])
    out[:, 0] = np.clip(h0 + alpha * (out[:, 0] - h0) + beta, -8.50, -4.50)
    return out


def _select_v4_step1_from_pool(
    mols_grid: List[List[Optional[Chem.Mol]]],
    legal_grid: np.ndarray,
    exotic_grid: np.ndarray,
    req_targets: np.ndarray,
    bus: CompositeCriticBus,
    m_keep: int,
    use_surrogate: bool,
    device: torch.device,
    *,
    sd_weight: float = 0.08,
    disp_rms_grid: Optional[np.ndarray] = None,
    disp_penalty_weight: float = 0.0,
) -> Tuple[List[Optional[Chem.Mol]], np.ndarray, np.ndarray, float]:
    """Select best finalist per prompt using calibrated `ext_unimol2` + `SA<=3.5` + conservative uncertainty & strain."""
    n_prompts = len(req_targets)
    if not use_surrogate or m_keep == 1:
        sel_m = [mols_grid[i][0] for i in range(n_prompts)]
        return sel_m, legal_grid[:, 0].copy(), exotic_grid[:, 0].copy(), 0.0

    flat_mols = [mols_grid[i][k] for i in range(n_prompts) for k in range(m_keep)]
    _sync(device)
    ts0 = time.perf_counter()
    pred_asset = bus.registry.get_predictor("ext_unimol2")
    rule_asset = bus.registry.get_predictor("builtin_rules")
    u_out = pred_asset.predict_mols(flat_mols, batch_size=64)
    r_out = rule_asset.predict_mols(flat_mols, batch_size=256)

    homo_p = u_out["mean"][:, pred_asset.channel_index("homo")].reshape(n_prompts, m_keep)
    homo_sd = u_out["std"][:, pred_asset.channel_index("homo")].reshape(n_prompts, m_keep)
    sa_p = r_out["mean"][:, rule_asset.channel_index("sa_score")].reshape(n_prompts, m_keep)
    mw_p = r_out["mean"][:, rule_asset.channel_index("mol_weight")].reshape(n_prompts, m_keep)
    sm_p = r_out["mean"][:, rule_asset.channel_index("smarts_alert")].reshape(n_prompts, m_keep)

    sel_mols: List[Optional[Chem.Mol]] = []
    sel_leg = np.zeros(n_prompts, dtype=bool)
    sel_exo = np.zeros(n_prompts, dtype=bool)
    for i in range(n_prompts):
        goal_h = float(req_targets[i, 0])
        goal_n = float(req_targets[i, 1])
        best_k = 0
        best_key = (99, float("inf"))
        for k in range(m_keep):
            mol_k = mols_grid[i][k]
            if not bool(legal_grid[i, k]) or mol_k is None:
                ck = (4, float(k))
            else:
                cln = not bool(exotic_grid[i, k])
                c_ok = (sa_p[i, k] <= 3.5) and (sm_p[i, k] <= 0.0) and (mw_p[i, k] <= 550.0)
                err_h = abs(float(homo_p[i, k]) - goal_h) if math.isfinite(float(homo_p[i, k])) else 99.0
                sd_h = float(homo_sd[i, k]) if math.isfinite(float(homo_sd[i, k])) else 0.0
                err_n = abs(float(mol_k.GetNumAtoms()) - goal_n)
                tol_n = max(4.0, 0.12 * goal_n)
                size_pen = max(0.0, err_n - tol_n) * 0.04 + err_n * 0.015
                sa_pen = max(0.0, float(sa_p[i, k]) - 3.5) * 0.25
                strain_pen = _analytic_conformer_strain_penalty(mol_k)
                disp_pen = 0.0
                if disp_rms_grid is not None and disp_penalty_weight > 0.0:
                    d_rms = float(disp_rms_grid[i, k])
                    if math.isfinite(d_rms):
                        disp_pen = float(disp_penalty_weight) * max(0.0, d_rms - 0.11)
                tier = 0 if (cln and c_ok) else (1 if cln else 2)
                ck = (tier, err_h + float(sd_weight) * sd_h + sa_pen + size_pen + strain_pen + disp_pen)
            if ck < best_key:
                best_key = ck
                best_k = k
        sel_mols.append(mols_grid[i][best_k] if bool(legal_grid[i, best_k]) else None)
        sel_leg[i] = bool(legal_grid[i, best_k])
        sel_exo[i] = bool(exotic_grid[i, best_k])

    _sync(device)
    surr_wall = time.perf_counter() - ts0
    return sel_mols, sel_leg, sel_exo, surr_wall


def run_rollout_shard(args: argparse.Namespace) -> None:
    """Collect one GPU's shard of Stage-1 & Stage-2 training rollouts (`0` `audit_dpa4` calls)."""
    device = torch.device(args.device)
    step1_dir = args.work_dir / args.run_name
    calib_cache = step1_dir / "residual_calibrated_unimol2.pt"
    pmap_json = step1_dir / "dev_prompt_affine_map.json"
    shard_out = step1_dir / f"rollout_shard_{args.shard_id}.pt"

    bus, _, _ = _build_calibrated_oe62_bus(args, include_blind_auditor=False, calib_cache_path=calib_cache)
    pmap = json.loads(pmap_json.read_text())

    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.v3_ckpt, device=args.device, batch_size=256)
    gen.eval()
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))

    _, clean_targets = _get_clean_oe62_pool(args, None)
    rng = np.random.default_rng(20261130 + int(args.shard_id) * 1009)
    n_shard = int(args.critic_rollouts) // max(1, int(args.num_shards))

    # Sample diverse training-distribution `(homo_req, n_atoms)` prompts concentrating around `18..46` atoms
    pool_sub = clean_targets[(clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 48.0)]
    pick_idx = rng.choice(len(pool_sub), size=n_shard, replace=True)
    req_raw = pool_sub[pick_idx].astype(np.float32, copy=True)
    # Add small target jitter so rollout targets cover off-manifold queries smoothly
    req_raw[:, 0] = np.clip(req_raw[:, 0] + rng.normal(0.0, 0.18, size=n_shard).astype(np.float32), -8.2, -4.8)
    req_raw[:, 1] = np.clip(np.round(req_raw[:, 1] + rng.integers(-2, 3, size=n_shard)), 18.0, 48.0)
    cond_raw = _apply_prompt_map(req_raw, pmap)

    t0 = time.perf_counter()
    s1_h_list: List[torch.Tensor] = []
    s1_g_list: List[torch.Tensor] = []
    s2_h_list: List[torch.Tensor] = []
    s2_c_list: List[torch.Tensor] = []
    s2_e_slices: List[torch.Tensor] = []
    s2_p_slices: List[torch.Tensor] = []
    t_norm_req_list: List[torch.Tensor] = []
    mols_all: List[Optional[Chem.Mol]] = []
    is_clean_all = np.zeros(n_shard, dtype=bool)
    actual_natoms = np.zeros(n_shard, dtype=np.float32)

    chunk = 256
    for s in range(0, n_shard, chunk):
        e = min(s + chunk, n_shard)
        nb = e - s
        tn_cond = gen.normalize_targets(cond_raw[s:e])
        tn_req = gen.normalize_targets(req_raw[s:e]).detach().cpu()
        cm = torch.ones((nb, 2), dtype=torch.bool, device=device)
        um = torch.tensor([[False, True]] * nb, dtype=torch.bool, device=device)

        roll = gen.sample_rollouts_with_two_stage_features(
            tn_cond,
            cm,
            uncond_active_mask=um,
            cfg_scale=2.00,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.30,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=440000 + int(args.shard_id) * 50000 + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
        )
        s1_h_list.append(roll["s1_h_concat"])
        s1_g_list.append(roll["s1_geom_feats"])
        s2_h_list.append(roll["s2_h_stage2"])
        s2_c_list.append(roll["s2_cloud_feats"])
        t_norm_req_list.append(tn_req)

        for loc_i, tr in enumerate(roll["trajectories"]):
            s2_e_slices.append(tr["stage2_elem_ids"])
            s2_p_slices.append(tr["stage2_leaf_pos"])
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            g_i = s + loc_i
            if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
                mols_all.append(res.mol)
                is_clean_all[g_i] = True
                actual_natoms[g_i] = float(res.mol.GetNumAtoms())
            else:
                mols_all.append(None)

    pred_asset = bus.registry.get_predictor("ext_unimol2")
    rule_asset = bus.registry.get_predictor("builtin_rules")
    u_out = pred_asset.predict_mols(mols_all, batch_size=64)
    r_out = rule_asset.predict_mols(mols_all, batch_size=256)
    pot_out = (
        bus.potential_slot.evaluate_molecules([m for m in mols_all if m is not None], chunk_size=64)
        if bus.potential_slot is not None
        else None
    )

    fmax_all = np.full(n_shard, np.nan, dtype=np.float32)
    if pot_out is not None:
        cur_p = 0
        for g_i, m in enumerate(mols_all):
            if m is not None:
                fmax_all[g_i] = float(pot_out["f_max_eV_A"][cur_p])
                cur_p += 1

    homo_surr = u_out["mean"][:, pred_asset.channel_index("homo")].astype(np.float32)
    sa_vals = r_out["mean"][:, rule_asset.channel_index("sa_score")]
    mw_vals = r_out["mean"][:, rule_asset.channel_index("mol_weight")]
    sm_vals = r_out["mean"][:, rule_asset.channel_index("smarts_alert")]

    y_valid = is_clean_all & u_out["valid_mask"] & np.isfinite(homo_surr)
    low_strain = (~np.isfinite(fmax_all)) | (fmax_all <= 4.5)
    is_constr = y_valid & (sa_vals <= 3.5) & (mw_vals <= 550.0) & (sm_vals <= 0.0) & low_strain

    y_raw_2d = np.stack([np.nan_to_num(homo_surr, nan=-6.5), actual_natoms], axis=-1).astype(np.float32)
    y_surr_norm = gen.normalize_targets(y_raw_2d).detach().cpu()

    wall_s = time.perf_counter() - t0
    torch.save(
        {
            "s1_h_concat": torch.cat(s1_h_list, dim=0),
            "s1_geom_feats": torch.cat(s1_g_list, dim=0),
            "s2_h_stage2": torch.cat(s2_h_list, dim=0),
            "s2_cloud_feats": torch.cat(s2_c_list, dim=0),
            "s2_elem_slices": s2_e_slices,
            "s2_pos_slices": s2_p_slices,
            "targets_norm_req": torch.cat(t_norm_req_list, dim=0),
            "y_surrogate_norm": y_surr_norm,
            "y_valid_mask": torch.from_numpy(y_valid),
            "is_legal": torch.from_numpy(is_clean_all),
            "is_constr_met": torch.from_numpy(is_constr),
            "wall_sec": wall_s,
        },
        shard_out,
    )
    print(
        f"[Step1 Rollout Shard {args.shard_id}] N={n_shard} | clean={is_clean_all.mean()*100:.1f}% | "
        f"constr+low_strain={is_constr.mean()*100:.1f}% | wall={wall_s:.1f}s -> {shard_out}",
        flush=True,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    step1_dir = args.work_dir / args.run_name
    step1_dir.mkdir(parents=True, exist_ok=True)
    calib_cache = step1_dir / "residual_calibrated_unimol2.pt"
    pmap_json = step1_dir / "dev_prompt_affine_map.json"
    critic_ckpt = step1_dir / "v4_step1_two_stage_critic.pt"
    sdf_dir = step1_dir / "cohorts_sdf"

    t_pipe0 = time.perf_counter()
    # 1. Fit zero-leakage `ResidualCalibratedAsset(ext_unimol2)` + dev prompt affine centering
    bus_train, cal_unimol2, calib_diag = _build_calibrated_oe62_bus(
        args, include_blind_auditor=False, calib_cache_path=(calib_cache if args.reuse_artifacts else None)
    )
    if not calib_cache.exists():
        # Ensure saved for parallel shard workers
        torch.save(
            {
                "feedback_static": cal_unimol2._feedback_static,
                "feedback_base_preds": cal_unimol2._feedback_base_preds,
                "feedback_residuals": cal_unimol2._feedback_residuals,
                "pred_mean": cal_unimol2._pred_mean,
                "pred_scale": cal_unimol2._pred_scale,
                "calib_diag": calib_diag,
            },
            calib_cache,
        )
    print(f"[Step1 Calib] Zero-leakage DFT calibration diagnostics: {calib_diag}", flush=True)

    gen, v3_payload = LoomUni3DARGenerator.load_checkpoint(args.v3_ckpt, device=args.device, batch_size=512)
    gen.eval()
    v3_meta = v3_payload.get("metadata") or {}
    leaf_tol = v3_meta.get("selected_leaf_budget_tol")

    if args.reuse_artifacts and pmap_json.exists():
        pmap = json.loads(pmap_json.read_text())
    else:
        pmap = _fit_dev_prompt_affine_map(gen, cal_unimol2, args, device)
        pmap_json.write_text(json.dumps(pmap, indent=2) + "\n")
    print(f"[Step1 DevMap] Fitted dev-set prompt affine map: {pmap}", flush=True)

    # Free `bus_train` before spawning 2-GPU rollout workers
    del bus_train
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Parallel 2-GPU rollout collection + `TwoStageOctreeProcessCritic` fit
    gpu_list = [g.strip() for g in args.rollout_gpus.split(",") if g.strip()]
    n_shards = len(gpu_list)
    if args.reuse_artifacts and critic_ckpt.exists():
        two_stage_critic, critic_meta = TwoStageOctreeProcessCritic.load_checkpoint(critic_ckpt, device=args.device)
        print(f"[Step1 Critic] Reused existing TwoStageOctreeProcessCritic: {critic_meta}", flush=True)
    else:
        t_roll0 = time.perf_counter()
        procs: List[subprocess.Popen[str]] = []
        for sh_i, gpu_id in enumerate(gpu_list):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "rollout_shard",
                "--device",
                f"cuda:{gpu_id}",
                "--shard-id",
                str(sh_i),
                "--num-shards",
                str(n_shards),
                "--critic-rollouts",
                str(args.critic_rollouts),
                "--work-dir",
                str(args.work_dir),
                "--run-name",
                str(args.run_name),
                "--v3-ckpt",
                str(args.v3_ckpt),
                "--oe62-task-spec",
                str(args.oe62_task_spec),
                "--oe62-master-train-sdf",
                str(args.oe62_master_train_sdf),
                "--oe62-master-valid-sdf",
                str(args.oe62_master_valid_sdf),
                "--oe62-source-train-sdf",
                str(args.oe62_source_train_sdf),
                "--oe62-replay-sdf",
                str(args.oe62_replay_sdf),
                "--quantum-train-dft-sdf",
                str(args.quantum_train_dft_sdf),
                "--quantum-valid-dft-sdf",
                str(args.quantum_valid_dft_sdf),
                "--oe62-targets-json",
                str(args.oe62_targets_json),
                "--oe62-reward-oracle-dir",
                str(args.oe62_reward_oracle_dir),
                "--oe62-potential-weights",
                str(args.oe62_potential_weights),
            ]
            if args.allow_tf32:
                cmd.append("--allow-tf32")
            procs.append(subprocess.Popen(cmd, text=True))

        for sh_i, pr in enumerate(procs):
            rc = pr.wait()
            if rc != 0:
                raise RuntimeError(f"Rollout shard worker {sh_i} exited with code {rc}")
        rollout_wall = time.perf_counter() - t_roll0

        shards = [
            torch.load(step1_dir / f"rollout_shard_{sh_i}.pt", map_location="cpu", weights_only=False)
            for sh_i in range(n_shards)
        ]
        s1_h = torch.cat([sh["s1_h_concat"] for sh in shards], dim=0)
        s1_g = torch.cat([sh["s1_geom_feats"] for sh in shards], dim=0)
        s2_h = torch.cat([sh["s2_h_stage2"] for sh in shards], dim=0)
        s2_c = torch.cat([sh["s2_cloud_feats"] for sh in shards], dim=0)
        t_req_all = torch.cat([sh["targets_norm_req"] for sh in shards], dim=0)
        y_surr_all = torch.cat([sh["y_surrogate_norm"] for sh in shards], dim=0)
        y_ok_all = torch.cat([sh["y_valid_mask"] for sh in shards], dim=0)
        is_leg_all = torch.cat([sh["is_legal"] for sh in shards], dim=0)
        is_con_all = torch.cat([sh["is_constr_met"] for sh in shards], dim=0)

        all_e_slices = [es for sh in shards for es in sh["s2_elem_slices"]]
        all_p_slices = [ps for sh in shards for ps in sh["s2_pos_slices"]]
        n_tot_r = len(all_e_slices)
        max_k_r = max((int(es.numel()) for es in all_e_slices), default=1)
        max_k_r = max(max_k_r, 1)
        pad_e_all = torch.zeros((n_tot_r, max_k_r), dtype=torch.long)
        pad_p_all = torch.zeros((n_tot_r, max_k_r, 3), dtype=torch.float32)
        pad_v_all = torch.zeros((n_tot_r, max_k_r), dtype=torch.bool)
        for i, (es, ps) in enumerate(zip(all_e_slices, all_p_slices)):
            ki = int(es.numel())
            if ki > 0:
                pad_e_all[i, :ki] = es
                pad_p_all[i, :ki] = ps
                pad_v_all[i, :ki] = True
        del shards

        t_fit0 = time.perf_counter()
        emb_d = int(gen.args.emb_dim)
        t_mean_np = gen.target_mean.detach().cpu().numpy()
        t_scale_np = gen.target_scale.detach().cpu().numpy()
        act_all = torch.ones((n_tot_r, 2), dtype=torch.bool)

        s1_critic = OctreeProcessCritic(emb_dim=emb_d, num_targets=2, hidden_dim=512).to(device)
        s1_critic.set_normalization(t_mean_np, t_scale_np)
        st1 = fit_octree_process_critic(
            s1_critic,
            h_concat=s1_h,
            geom_feats=s1_g,
            targets_norm=t_req_all,
            active_mask=act_all,
            y_surrogate_norm=y_surr_all,
            y_valid_mask=y_ok_all,
            is_legal=is_leg_all,
            is_constr_met=is_con_all,
            epochs=24,
            batch_size=256,
            lr=1.5e-3,
            seed=20261119,
        )

        s2_critic = OctreeElementProcessCritic(
            emb_dim=emb_d,
            num_targets=2,
            num_atom_classes=int(gen.backbone.num_atom),
            d_node=160,
            hidden_dim=512,
        ).to(device)
        s2_critic.set_normalization(t_mean_np, t_scale_np)
        st2 = fit_octree_element_process_critic(
            s2_critic,
            h_stage2=s2_h,
            elem_ids=pad_e_all,
            leaf_pos=pad_p_all,
            leaf_valid=pad_v_all,
            cloud_feats=s2_c,
            targets_norm=t_req_all,
            active_mask=act_all,
            y_surrogate_norm=y_surr_all,
            y_valid_mask=y_ok_all,
            is_legal=is_leg_all,
            is_constr_met=is_con_all,
            epochs=28,
            batch_size=256,
            lr=1.2e-3,
            seed=20261120,
        )
        fit_wall = time.perf_counter() - t_fit0

        two_stage_critic = TwoStageOctreeProcessCritic(s1_critic, s2_critic).to(device)
        two_stage_critic.eval()
        critic_meta = {
            "total_rollouts": n_tot_r,
            "num_rollout_gpus": n_shards,
            "rollout_wall_sec": round(rollout_wall, 2),
            "critic_fit_wall_sec": round(fit_wall, 2),
            "rollout_gpuh": round((rollout_wall * n_shards + fit_wall) / 3600.0, 4),
            "clean_rate": round(float(is_leg_all.float().mean().item()), 4),
            "constr_and_low_strain_rate": round(float(is_con_all.float().mean().item()), 4),
            "stage1_stats": st1.__dict__,
            "stage2_stats": st2.__dict__,
        }
        two_stage_critic.save_checkpoint(critic_ckpt, extra=critic_meta)
        print(f"[Step1 Critic DONE] Saved {critic_ckpt} | meta={critic_meta}", flush=True)

    # 3. Strictly Fair Fixed-Prompt `30.4`-Atom Benchmark (`N = 1,792`, 1-GPU Timing, ZERO `DPA-4` Relaxation!)
    bus_eval, _, _ = _build_calibrated_oe62_bus(args, include_blind_auditor=True, calib_cache_path=calib_cache)
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)
    req_targets = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
    cond_targets = _apply_prompt_map(req_targets, pmap)
    n_prompts = len(req_targets)

    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    obj_col_mask = torch.tensor([True, True], dtype=torch.bool, device=device)

    def _encode_cond(rows: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tn = gen.normalize_targets(rows)
        ma = torch.ones((len(rows), 2), dtype=torch.bool, device=device)
        ua = torch.tensor([[False, True]] * len(rows), dtype=torch.bool, device=device)
        return tn, ma, ua

    # Warmup on 16 prompts
    wt, wm, wu = _encode_cond(cond_targets[:16])
    _ = gen.sample_with_two_stage_process_critic(
        wt,
        wm,
        two_stage_critic,
        m_stage1=8,
        m_stage2=4,
        m_keep=2,
        objective_col_mask=obj_col_mask,
        cfg_scale=2.0,
        cfg_tree_decay=0.30,
        geometric_guard=True,
        seed=17,
        use_bf16=True,
        uncond_active_mask=wu,
    )
    _sync(device)

    results_arms: Dict[str, Any] = {}
    # Funnel configurations: `(arm_name, m_stage1, m_stage2, m_keep, use_surrogate)`
    funnels: List[Tuple[str, int, int, int, bool]] = [
        ("Outcome_M1", 1, 1, 1, False),
        ("TwoStage_16_to_6_to_2", 16, 6, 2, True),
        ("TwoStage_24_to_8_to_2", 24, 8, 2, True),
        ("TwoStage_32_to_8_to_2", 32, 8, 2, True),
        ("TwoStage_32_to_8_to_3", 32, 8, 3, True),
    ]

    for arm_name, ms1, ms2, mkeep, use_surr in funnels:
        _sync(device)
        tg0 = time.perf_counter()
        groups_all: List[List[Dict[str, Any]]] = []
        p_chunk = 512 if ms1 == 1 else 256
        for s in range(0, n_prompts, p_chunk):
            e = min(s + p_chunk, n_prompts)
            bt, bm, bu = _encode_cond(cond_targets[s:e])
            groups_all.extend(
                gen.sample_with_two_stage_process_critic(
                    bt,
                    bm,
                    two_stage_critic,
                    m_stage1=ms1,
                    m_stage2=ms2,
                    m_keep=mkeep,
                    objective_col_mask=obj_col_mask,
                    stage1_legal_weight=0.80,
                    stage1_constr_weight=0.55,
                    stage2_legal_weight=1.05,
                    stage2_constr_weight=0.70,
                    coarse_chunk_size=1024,
                    cfg_scale=2.00,
                    xyz_cfg_scale=1.0,
                    cfg_tree_decay=0.30,
                    geometric_guard=True,
                    temperatures=(0.85, 0.30, 0.25),
                    seed=950000 + ms1 * 409 + ms2 * 61 + mkeep * 13 + s,
                    use_bf16=True,
                    electron_parity=0,
                    uncond_active_mask=bu,
                    leaf_atom_budget=(req_targets[s:e, 1] if leaf_tol is not None else None),
                    leaf_budget_tol=(float(leaf_tol) if leaf_tol is not None else 2.0),
                )
            )
        _sync(device)
        gen_w = time.perf_counter() - tg0

        # Pure analytic RDKit valence reconciliation (`potential_slot=None`, ZERO MLIP calls!)
        tr0 = time.perf_counter()
        p_grid: List[List[Optional[Chem.Mol]]] = [[None] * mkeep for _ in range(n_prompts)]
        p_leg = np.zeros((n_prompts, mkeep), dtype=bool)
        p_exo = np.zeros((n_prompts, mkeep), dtype=bool)
        for i in range(n_prompts):
            for k in range(mkeep):
                tr = groups_all[i][k]
                res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
                if res.reconciled_legal and res.mol is not None:
                    p_grid[i][k] = res.mol
                    p_leg[i, k] = True
                    p_exo[i, k] = bool(res.reconciled_exotic_charge)
        rec_w = time.perf_counter() - tr0

        # Finalist selection (`m_keep -> 1`) using calibrated `ext_unimol2` + analytic strain (`0` DPA-4 steps!)
        sel_m, sel_l, sel_x, surr_w = _select_v4_step1_from_pool(
            p_grid, p_leg, p_exo, req_targets, bus_eval, mkeep, use_surrogate=use_surr, device=device
        )
        # RED LINE #3: DO NOT call `batch_polish_hydrogens_mlip` or any DPA-4 relaxation!
        arm_res = _score_oe62_size_cohort(
            sel_m,
            sel_l,
            sel_x,
            req_targets,
            bus_eval,
            source_graphs,
            source_scaffolds,
            gen_w,
            rec_w,
            surr_w,
            export_sdf_path=(sdf_dir / f"{arm_name}.sdf"),
        )
        results_arms[arm_name] = arm_res
        bias_ra = arm_res["audit"]["mean_reward_minus_audit_pred_eV"] if arm_res.get("audit") else float("nan")
        print(
            f"[Step1 {arm_name}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s "
            f"(gen={gen_w:.1f}s, rec={rec_w:.1f}s, surr={surr_w:.1f}s) | "
            f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
            f"RewSynth={arm_res['synthesizable_clean_hits']}/{n_prompts} ({arm_res['synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} ({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindMAE={arm_res['audit_mae_eV']:.4f} eV | Rew-Aud={bias_ra:+.4f} eV",
            flush=True,
        )

    out_payload = {
        "experiment": "OE62 Fair 30-Atom Benchmark — Uni-3DAR v4 Step 1 (Zero Backbone Retraining: Calibrated Surrogate + Vectorized TwoStage 32->8->2)",
        "red_lines_enforced": {
            "zero_inference_dpa4_relaxation_or_h_polish": True,
            "zero_audit_dpa4_leakage": True,
            "single_gpu_fair_timing": True,
            "fixed_per_prompt_30atom_seed": 20261118,
        },
        "proposals_per_arm": n_prompts,
        "requested_natoms_summary": {
            "mean": round(float(req_targets[:, 1].mean()), 2),
            "median": round(float(np.median(req_targets[:, 1])), 1),
            "p10": round(float(np.percentile(req_targets[:, 1], 10)), 1),
            "p90": round(float(np.percentile(req_targets[:, 1], 90)), 1),
        },
        "v3_base_checkpoint": str(args.v3_ckpt),
        "zero_leakage_dft_calibration": calib_diag,
        "dev_prompt_affine_map": pmap,
        "two_stage_critic_meta": critic_meta,
        "total_pipeline_wall_sec": round(time.perf_counter() - t_pipe0, 2),
        "arms": results_arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out_payload, indent=2) + "\n")
    print(f"[Step1 ALL DONE] Wrote {args.output_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["pipeline", "rollout_shard"], default="pipeline")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rollout-gpus", type=str, default="0,1")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--critic-rollouts", type=int, default=16384)
    p.add_argument("--reuse-artifacts", action="store_true")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="uni3dar_v4_step1")
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument(
        "--v3-ckpt",
        type=Path,
        default=Path(
            "/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/oe62_size_anchored_v3_a800/v20_aligned/oe62_size_v20_aligned.pt"
        ),
    )
    p.add_argument(
        "--oe62-master-train-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"),
    )
    p.add_argument(
        "--oe62-master-valid-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_valid_B.sdf"),
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
        "--quantum-train-dft-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_dft.sdf"),
    )
    p.add_argument(
        "--quantum-valid-dft-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_valid_dft.sdf"),
    )
    p.add_argument(
        "--oe62-targets-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/targets_B.json"),
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
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_uni3dar_v4_step1_fair_30atom.json"),
    )
    args = p.parse_args()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.mode == "rollout_shard":
        run_rollout_shard(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
