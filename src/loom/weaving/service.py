"""Top-Level Model- & Domain-Agnostic Self-Driving Lab Service Layer (`src/loom/weaving/service.py`).

`LoomService` unifies the three core platform pillars of `Uni-Loom`:
  1. **Capability-Adaptive Post-Training (`align`)**:
       Automatically inspects `GeneratorSlot.supports_exact_trajectory_logprob` to dispatch
       `Tree-Vine GRPO` (autoregressive octree `Uni-3DAR`) or `Loom-AWR` (ODE/CTMC `PropMolFlow`),
       followed by `fit_internal_post_processors(...)`.
  2. **Symmetric Dev Calibration & Effort-Tiered Generation (`calibrate_on_dev`, `generate`)**:
       Applies an identical zero-leakage (`0` audit calls, `0` inference `PotentialSlot` calls)
       dev-set calibration protocol across all generators and exposes standardized effort tiers:
         - `base`   (`M = 1`): pure single-sample generation (`0` outer surrogate calls)
         - `low`    (`M = 2`): fast screening (`16->4->2` in `Uni-3DAR` vs `M=2` in `PropMolFlow`)
         - `medium` (`M = 4`): balanced discovery (`32->8->4` in `Uni-3DAR` vs `M=4` in `PropMolFlow`)
         - `high`   (`M = 8`): SOTA precision (`64->16->8` in `Uni-3DAR` vs `M=8` in `PropMolFlow`)
  3. **Self-Aware Epistemic Confidence Envelope (`diagnose_requests`)**:
       Couples `GPAtlas` + `RSIDiagnoser` (`1D` or multi-objective `ND`) to attach causal
       4-quadrant epistemic diagnostics (`CONFIDENT`, `GENERATOR_DISPERSED_UPGRADE_TIER`,
       `ORACLE_BLINDSPOT_WARNING`, `PRIOR_CEILING_UNREACHABLE`) to every generation batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from rdkit import Chem
import torch

from .critic_bus import CompositeCriticBus
from .gp_atlas import HeteroscedasticGPAtlas1D
from .grpo import LoomGRPOTrainer, is_clean_organic_charge_state
from .rsi import RSIDiagnoser
from .slots import EFFORT_TO_M, GeneratorSlot, resolve_effort_candidates


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
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "I": 1.39,
}


def _analytic_conformer_strain_penalty(mol: Chem.Mol) -> float:
    """Fast (`<0.02 ms`) pure-CPU covalent stretch + steric clash penalty (`0` MLIP calls)."""
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
        shrink = 0.90 if bo >= 2.5 else (0.93 if bo >= 1.8 else (0.96 if bo >= 1.4 else 1.01))
        d_eq = shrink * (rcov[i] + rcov[j])
        stretch = abs(float(dmat[i, j]) - d_eq)
        if stretch > max_stretch:
            max_stretch = stretch
        sum_sq_stretch += stretch * stretch

    rms_stretch = math.sqrt(sum_sq_stretch / float(n_bonds))
    r_sum = rcov[:, None] + rcov[None, :]
    nonbond = ~bonded
    np.fill_diagonal(nonbond, False)
    min_nb_ratio = float((dmat[nonbond] / np.maximum(r_sum[nonbond], 0.5)).min()) if nonbond.any() else 2.0
    clash_pen = max(0.0, 1.32 - min_nb_ratio) * 0.45
    stretch_pen = max(0.0, max_stretch - 0.16) * 0.35 + max(0.0, rms_stretch - 0.07) * 0.25
    return float(clash_pen + stretch_pen)


ZONE_TO_SERVICE_STATUS: Dict[str, Dict[str, str]] = {
    "CONVERGED": {
        "epistemic_status": "CONFIDENT",
        "recommended_effort": "low",
        "lab_action": "DEPLOY_LOW_TIER",
    },
    "INNER_UNDEREXPLORED": {
        "epistemic_status": "GENERATOR_DISPERSED_UPGRADE_TIER",
        "recommended_effort": "high",
        "lab_action": "UPGRADE_EFFORT_AND_TRIGGER_INNER_RSI",
    },
    "OUTER_BLINDSPOT": {
        "epistemic_status": "ORACLE_BLINDSPOT_WARNING",
        "recommended_effort": "medium",
        "lab_action": "FREEZE_RL_AND_ACQUIRE_WETLAB_OR_DFT_ASSAY",
    },
    "PRIOR_CEILING": {
        "epistemic_status": "PRIOR_CEILING_UNREACHABLE",
        "recommended_effort": "base",
        "lab_action": "FLAG_OUT_OF_PRIOR_SUPPORT",
    },
}


@dataclass
class AffineConditionCalibration:
    """Symmetric zero-leakage damped affine prompt map & auxiliary channel anchor configuration."""

    objective_names: List[str]
    centers: List[float]
    alphas: List[float]
    betas: List[float]
    auxiliary_anchors: Dict[str, float] = field(default_factory=dict)
    dev_diagnostics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(
        cls,
        objective_names: Sequence[str],
        *,
        auxiliary_anchors: Optional[Dict[str, float]] = None,
    ) -> "AffineConditionCalibration":
        names = list(objective_names)
        k = len(names)
        return cls(
            objective_names=names,
            centers=[0.0] * k,
            alphas=[1.0] * k,
            betas=[0.0] * k,
            auxiliary_anchors=dict(auxiliary_anchors or {}),
        )

    def map_objectives(self, requested_targets: np.ndarray) -> np.ndarray:
        req = np.asarray(requested_targets, dtype=np.float32)
        if req.ndim == 1:
            req = req[:, None]
        out = req.copy()
        for j in range(min(out.shape[1], len(self.objective_names))):
            c_j = float(self.centers[j])
            a_j = float(self.alphas[j])
            b_j = float(self.betas[j])
            out[:, j] = c_j + a_j * (out[:, j] - c_j) + b_j
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LoomService:
    """Unified Self-Driving Lab service orchestrating `GeneratorSlot` + `CompositeCriticBus` + `GPAtlas`."""

    def __init__(
        self,
        generator: GeneratorSlot,
        critic_bus: CompositeCriticBus,
        *,
        calibration: Optional[AffineConditionCalibration] = None,
        default_auxiliary_anchors: Optional[Dict[str, float]] = None,
        gp_atlases: Optional[Dict[str, HeteroscedasticGPAtlas1D]] = None,
        rsi_diagnoser: Optional[RSIDiagnoser] = None,
        uncertainty_penalty_weight: float = 0.25,
        refiner_disp_penalty_weight: float = 0.35,
    ) -> None:
        self.generator = generator
        self.critic_bus = critic_bus
        default_aux = dict(default_auxiliary_anchors or {})
        if "sa_score" in getattr(generator, "target_cols", []) and "sa_score" not in default_aux:
            default_aux["sa_score"] = 1.95
        self.calibration = calibration or AffineConditionCalibration.identity(
            critic_bus.spec.objective_names, auxiliary_anchors=default_aux
        )
        self.gp_atlases = dict(gp_atlases) if gp_atlases is not None else {}
        self.rsi_diagnoser = rsi_diagnoser or RSIDiagnoser()
        self.uncertainty_penalty_weight = float(uncertainty_penalty_weight)
        self.refiner_disp_penalty_weight = float(refiner_disp_penalty_weight)

    @property
    def generator_capabilities(self) -> Dict[str, Any]:
        return {
            "backend_id": getattr(self.generator, "backend_id", type(self.generator).__name__),
            "generator_family": getattr(self.generator, "generator_family", "unknown"),
            "supports_exact_trajectory_logprob": bool(self.generator.supports_exact_trajectory_logprob),
            "supports_prefix_process_search": bool(getattr(self.generator, "supports_prefix_process_search", False)),
            "preferred_alignment_mode": (
                "grpo" if bool(self.generator.supports_exact_trajectory_logprob) else "awr"
            ),
        }

    def _sync_device(self) -> None:
        try:
            dev = next(self.generator.parameters()).device
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        except StopIteration:
            pass

    def _normalize_request_shape(
        self,
        requested_targets: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
        *,
        dtype: Any = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align 1D or subset-column requests `(B, K_in)` to full `spec.objectives` width `(B, K_obj)`."""
        req_in = np.asarray(requested_targets, dtype=dtype)
        if req_in.ndim == 1:
            req_in = req_in[:, None]
        bsz, k_in = req_in.shape
        k_obj = len(self.critic_bus.spec.objectives)
        if k_in == k_obj:
            req = req_in
            act = (
                np.asarray(active_mask, dtype=bool).reshape(bsz, k_obj)
                if active_mask is not None
                else np.ones((bsz, k_obj), dtype=bool)
            )
            return req, act
        if k_in < k_obj:
            req = np.zeros((bsz, k_obj), dtype=dtype)
            req[:, :k_in] = req_in
            if active_mask is not None:
                a_in = np.asarray(active_mask, dtype=bool)
                if a_in.size == bsz * k_obj:
                    act = a_in.reshape(bsz, k_obj)
                else:
                    act = np.zeros((bsz, k_obj), dtype=bool)
                    act[:, :k_in] = a_in.reshape(bsz, k_in)
            else:
                act = np.zeros((bsz, k_obj), dtype=bool)
                act[:, :k_in] = True
            return req, act
        raise ValueError(f"requested_targets has {k_in} columns, exceeding spec.objectives ({k_obj})")

    def encode_prompts(
        self,
        requested_targets: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
        *,
        calibration_override: Optional[AffineConditionCalibration] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map task-space objective requests `(B, K_obj)` into normalized generator conditioning tensors."""
        cal = calibration_override or self.calibration
        req, act_obj = self._normalize_request_shape(requested_targets, active_mask, dtype=np.float32)
        bsz = int(req.shape[0])
        mapped_obj = cal.map_objectives(req)

        gen_cols = list(getattr(self.generator, "target_cols", self.critic_bus.spec.objective_names))
        obj_name_to_idx = {name: i for i, name in enumerate(self.critic_bus.spec.objective_names)}

        gen_raw = np.zeros((bsz, len(gen_cols)), dtype=np.float32)
        gen_act = np.zeros((bsz, len(gen_cols)), dtype=bool)
        for g_j, col_name in enumerate(gen_cols):
            if col_name in obj_name_to_idx:
                o_j = obj_name_to_idx[col_name]
                gen_raw[:, g_j] = mapped_obj[:, o_j]
                gen_act[:, g_j] = act_obj[:, o_j]
            elif col_name in cal.auxiliary_anchors:
                gen_raw[:, g_j] = float(cal.auxiliary_anchors[col_name])
                gen_act[:, g_j] = True

        t_norm = self.generator.normalize_targets(gen_raw)  # type: ignore[attr-defined]
        m_tensor = torch.as_tensor(gen_act, dtype=torch.bool, device=t_norm.device)
        return t_norm, m_tensor

    def _evaluate_surrogate_zero_mlip(
        self,
        mols: Sequence[Optional[Chem.Mol]],
        requested_targets: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        *,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Evaluate objectives & rule constraints with ZERO `PotentialSlot` (`DPA-4`) and ZERO `AuditOracleSlot` calls."""
        saved_pot = self.critic_bus.potential_slot
        try:
            self.critic_bus.potential_slot = None
            return self.critic_bus.evaluate_molecules(
                mols,
                requested_targets=requested_targets,
                active_mask=active_mask,
                include_audit=False,
                for_training=False,
                batch_size=batch_size,
            )
        finally:
            self.critic_bus.potential_slot = saved_pot

    def select_best_of_m(
        self,
        candidates_grid: Sequence[Sequence[Optional[Chem.Mol]]],
        requested_targets: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
        *,
        atom_budget: Optional[np.ndarray | Sequence[float]] = None,
        batch_size: int = 64,
    ) -> Tuple[List[Optional[Chem.Mol]], Dict[str, Any]]:
        """Select Top-1 molecule per prompt using outer `CompositeCriticBus` (`0` MLIP calls, `0` audit calls)."""
        bsz = len(candidates_grid)
        if bsz == 0:
            return [], {"surrogate_wall_sec": 0.0, "m_candidates": 0}
        m_keep = len(candidates_grid[0])
        if m_keep <= 1:
            selected_m1 = [row[0] if len(row) > 0 else None for row in candidates_grid]
            return selected_m1, {"surrogate_wall_sec": 0.0, "m_candidates": m_keep}

        req, act = self._normalize_request_shape(requested_targets, active_mask, dtype=np.float64)
        b_1d = np.asarray(atom_budget, dtype=np.float64).reshape(bsz) if atom_budget is not None else None

        self._sync_device()
        t_s0 = time.perf_counter()
        flat_mols: List[Optional[Chem.Mol]] = [mol for row in candidates_grid for mol in row]
        rep_req = np.repeat(req, m_keep, axis=0)
        rep_act = np.repeat(act, m_keep, axis=0)
        ev = self._evaluate_surrogate_zero_mlip(flat_mols, rep_req, rep_act, batch_size=batch_size)

        preds = ev["objective_preds"]  # (B*M, K_obj)
        stds = np.nan_to_num(ev["objective_stds"], nan=0.0)
        legal = ev["legal_mask"] & ev["clean_charge_mask"]
        hard_ok = legal & ev["hard_constraints_met"]
        c_scores = ev["constraint_scores"]

        tols = np.array([max(float(o.tolerance), 1e-4) for o in self.critic_bus.spec.objectives], dtype=np.float64)
        weights = np.array([float(o.weight) for o in self.critic_bus.spec.objectives], dtype=np.float64)

        selected: List[Optional[Chem.Mol]] = [None] * bsz
        chosen_slots: List[int] = []
        for i in range(bsz):
            best_k = 0
            best_tier = -1
            best_utility = -1e18
            for k in range(m_keep):
                g_idx = i * m_keep + k
                mol_k = candidates_grid[i][k]
                if mol_k is None or not bool(legal[g_idx]):
                    continue
                act_i = act[i]
                pred_row = preds[g_idx]
                if not np.isfinite(pred_row[act_i]).all():
                    tier = 0
                    u_val = float(c_scores[g_idx]) - 50.0
                else:
                    tier = 2 if bool(hard_ok[g_idx]) else 1
                    err_vec = (
                        np.abs(pred_row - req[i]) + self.uncertainty_penalty_weight * stds[g_idx]
                    ) / tols
                    denom_w = float(np.sum(weights[act_i])) if act_i.any() else 1.0
                    norm_loss = float(np.sum(weights[act_i] * err_vec[act_i])) / max(denom_w, 1e-6)
                    strain_pen = _analytic_conformer_strain_penalty(mol_k) / 0.25
                    disp_pen = 0.0
                    if mol_k.HasProp("_loom_refiner_disp_rms_A"):
                        disp_rms = float(mol_k.GetDoubleProp("_loom_refiner_disp_rms_A"))
                        if math.isfinite(disp_rms):
                            disp_pen = (self.refiner_disp_penalty_weight * max(0.0, disp_rms - 0.10)) / 0.25
                    size_pen = 0.0
                    if b_1d is not None and b_1d[i] > 0:
                        err_n = abs(float(mol_k.GetNumAtoms()) - float(b_1d[i]))
                        tol_n = max(3.5, 0.10 * float(b_1d[i]))
                        size_pen = (max(0.0, err_n - tol_n) * 0.04 + err_n * 0.012) / 0.25
                    u_val = -norm_loss + 0.15 * float(c_scores[g_idx]) - strain_pen - disp_pen - size_pen

                if (tier > best_tier) or (tier == best_tier and u_val > best_utility):
                    best_tier = tier
                    best_utility = u_val
                    best_k = k

            selected[i] = candidates_grid[i][best_k]
            chosen_slots.append(best_k)

        self._sync_device()
        surr_wall = time.perf_counter() - t_s0
        return selected, {
            "surrogate_wall_sec": surr_wall,
            "m_candidates": m_keep,
            "chosen_candidate_indices": chosen_slots,
        }

    def calibrate_on_dev(
        self,
        dev_targets: np.ndarray,
        *,
        dev_atom_budget: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        calib_effort: str = "low",
        sa_anchor_candidates: Sequence[float] = (1.85, 1.95, 2.05, 2.15),
        alpha_bounds: Tuple[float, float] = (0.95, 1.08),
        slope_damping: float = 0.50,
        bias_damping: float = 0.85,
        beta_cap: float = 0.35,
        seed: int = 2026901,
        **gen_kwargs: Any,
    ) -> AffineConditionCalibration:
        """Symmetric 112-prompt zero-leakage dev calibration (`0` audit calls, `0` MLIP calls) for any `GeneratorSlot`."""
        req, act = self._normalize_request_shape(dev_targets, active_mask, dtype=np.float32)
        bsz = int(req.shape[0])
        obj_names = list(self.critic_bus.spec.objective_names)
        gen_cols = list(getattr(self.generator, "target_cols", obj_names))
        has_aux_sa = ("sa_score" in gen_cols) and ("sa_score" not in obj_names)

        cand_anchors: List[Dict[str, float]]
        if has_aux_sa and len(sa_anchor_candidates) > 0:
            cand_anchors = [{**self.calibration.auxiliary_anchors, "sa_score": float(sa_v)} for sa_v in sa_anchor_candidates]
        else:
            cand_anchors = [dict(self.calibration.auxiliary_anchors)]

        best_aux = cand_anchors[0]
        best_ev: Optional[Dict[str, Any]] = None
        best_score = -1e18
        sweep_logs: List[Dict[str, Any]] = []

        for idx_a, aux_map in enumerate(cand_anchors):
            probe_cal = AffineConditionCalibration.identity(obj_names, auxiliary_anchors=aux_map)
            tn, tm = self.encode_prompts(req, act, calibration_override=probe_cal)
            c_grid = self.generator.generate_candidates(
                tn,
                tm,
                atom_budget=dev_atom_budget,
                effort=calib_effort,
                seed=int(seed) + idx_a * 97,
                **gen_kwargs,
            )
            sel_mols, _ = self.select_best_of_m(c_grid, req, act, atom_budget=dev_atom_budget)
            ev = self._evaluate_surrogate_zero_mlip(sel_mols, req, act)
            leg = ev["legal_mask"] & ev["clean_charge_mask"]
            hits = float(np.mean(ev["constrained_clean_hits"])) if bsz > 0 else 0.0
            hard_rate = float(np.mean(ev["hard_constraints_met"][leg])) if leg.any() else 0.0
            preds_0 = ev["objective_preds"][:, 0]
            ok_0 = leg & np.isfinite(preds_0)
            mae_0 = float(np.mean(np.abs(preds_0[ok_0] - req[ok_0, 0]))) if ok_0.any() else 2.5
            # Strongly reward >=90% SA/constraint satisfaction + constrained hit yield - surrogate MAE
            sa_shortfall = max(0.0, 0.90 - hard_rate)
            sc = hits + 0.35 * hard_rate - 0.80 * sa_shortfall - 0.35 * mae_0
            sweep_logs.append(
                {
                    "auxiliary_anchors": dict(aux_map),
                    "dev_constrained_hit_yield": round(hits, 4),
                    "dev_legal_rate": round(float(np.mean(leg)), 4),
                    "dev_hard_constraint_rate_among_legal": round(hard_rate, 4),
                    "dev_obj0_mae": round(mae_0, 4),
                    "dev_composite": round(sc, 5),
                }
            )
            if sc > best_score:
                best_score = sc
                best_aux = aux_map
                best_ev = ev

        assert best_ev is not None
        leg_b = best_ev["legal_mask"] & best_ev["clean_charge_mask"]
        hard_b = leg_b & best_ev["hard_constraints_met"]
        centers: List[float] = []
        alphas: List[float] = []
        betas: List[float] = []
        per_obj_fit: Dict[str, Any] = {}

        for j, o_name in enumerate(obj_names):
            col_req = req[:, j].astype(np.float64)
            col_pred = best_ev["objective_preds"][:, j].astype(np.float64)
            c0 = float(np.median(col_req))
            fit_mask = hard_b & act[:, j] & np.isfinite(col_pred)
            if int(fit_mask.sum()) < 24:
                fit_mask = leg_b & act[:, j] & np.isfinite(col_pred)
            if int(fit_mask.sum()) >= 20 and float(np.std(col_req[fit_mask])) > 1e-4:
                xs = col_req[fit_mask] - c0
                ys = col_pred[fit_mask] - c0
                raw_slope, raw_bias = np.polyfit(xs, ys, deg=1)
                sl_clamped = float(np.clip(raw_slope, 0.75, 1.15))
                alpha_j = float(
                    np.clip(1.0 + float(slope_damping) * ((1.0 / sl_clamped) - 1.0), alpha_bounds[0], alpha_bounds[1])
                )
                beta_j = float(np.clip(-float(bias_damping) * float(raw_bias) / sl_clamped, -beta_cap, beta_cap))
            else:
                raw_slope, raw_bias, alpha_j, beta_j = 1.0, 0.0, 1.0, 0.0
            centers.append(round(c0, 4))
            alphas.append(round(alpha_j, 4))
            betas.append(round(beta_j, 4))
            per_obj_fit[o_name] = {
                "n_fit": int(fit_mask.sum()),
                "forward_slope": round(float(raw_slope), 4),
                "forward_bias": round(float(raw_bias), 4),
                "alpha": round(alpha_j, 4),
                "beta": round(beta_j, 4),
            }

        self.calibration = AffineConditionCalibration(
            objective_names=obj_names,
            centers=centers,
            alphas=alphas,
            betas=betas,
            auxiliary_anchors=dict(best_aux),
            dev_diagnostics={
                "calib_effort": calib_effort,
                "n_dev_prompts": bsz,
                "per_objective_fit": per_obj_fit,
                "anchor_sweep_logs": sweep_logs,
            },
        )
        return self.calibration

    def diagnose_requests(
        self,
        requested_targets: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attach 4-quadrant epistemic Self-Awareness diagnostics to requested objective vectors."""
        if not self.gp_atlases:
            return None
        req, act = self._normalize_request_shape(requested_targets, active_mask, dtype=np.float64)
        obj_specs = [
            o
            for o in self.critic_bus.spec.objectives
            if (o.name in self.gp_atlases) and bool(act[:, self.critic_bus.spec.objective_names.index(o.name)].any())
        ]
        if not obj_specs:
            return None

        queries: List[Dict[str, Any]] = []
        tols: List[float] = []
        for o in obj_specs:
            col_idx = self.critic_bus.spec.objective_names.index(o.name)
            q = self.gp_atlases[o.name].query(req[:, col_idx], tolerance_eV=float(o.tolerance))
            queries.append(q)
            tols.append(float(o.tolerance))

        if len(queries) == 1:
            diag = self.rsi_diagnoser.diagnose_1d(queries[0])
        else:
            sub_t = np.stack(
                [req[:, self.critic_bus.spec.objective_names.index(o.name)] for o in obj_specs], axis=-1
            )
            diag = self.rsi_diagnoser.diagnose_nd(queries, tolerances=tols, targets_nd=sub_t)

        zones = diag["rsi_zones"]
        envelopes = [ZONE_TO_SERVICE_STATUS.get(z, ZONE_TO_SERVICE_STATUS["INNER_UNDEREXPLORED"]) for z in zones]
        return {
            "rsi_zones": zones,
            "zone_counts": diag["zone_counts"],
            "service_envelopes": envelopes,
            "diagnosis": diag,
        }

    def generate(
        self,
        requested_targets: np.ndarray,
        *,
        atom_budget: Optional[np.ndarray | Sequence[float]] = None,
        active_mask: Optional[np.ndarray] = None,
        effort: str = "medium",
        n_candidates: Optional[int] = None,
        seed: Optional[int] = None,
        include_epistemic_envelope: bool = True,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        """Standardized `Uni-Loom` generation endpoint across `effort in {'base','low','medium','high'}`."""
        eff_name, m_keep = resolve_effort_candidates(effort=effort, n_candidates=n_candidates)
        req, act = self._normalize_request_shape(requested_targets, active_mask, dtype=np.float32)
        bsz = int(req.shape[0])

        self._sync_device()
        t_g0 = time.perf_counter()
        t_norm, m_act = self.encode_prompts(req, act)
        candidates_grid = self.generator.generate_candidates(
            t_norm,
            m_act,
            atom_budget=atom_budget,
            n_candidates=m_keep,
            effort=eff_name,
            seed=seed,
            **gen_kwargs,
        )
        self._sync_device()
        gen_wall = time.perf_counter() - t_g0

        selected_mols, sel_info = self.select_best_of_m(
            candidates_grid, req, act, atom_budget=atom_budget
        )
        surr_wall = float(sel_info["surrogate_wall_sec"])
        total_wall = gen_wall + surr_wall
        eff_mol_s = float(bsz) / max(total_wall, 1e-6)
        epistemic = self.diagnose_requests(req, act) if include_epistemic_envelope else None

        return {
            "molecules": selected_mols,
            "candidates_grid": candidates_grid,
            "effort": eff_name,
            "n_candidates": m_keep,
            "gen_wall_sec": gen_wall,
            "surrogate_wall_sec": surr_wall,
            "total_wall_sec": total_wall,
            "effective_output_mol_per_sec": eff_mol_s,
            "gpu_sec_per_1k_prompts": 1000.0 / max(eff_mol_s, 1e-6),
            "epistemic_envelope": epistemic,
            "calibration": self.calibration.to_dict(),
        }

    def align(
        self,
        *,
        replay_molecules: Sequence[Chem.Mol],
        replay_targets: np.ndarray,
        prompt_sampler_fn: Callable[[int, np.random.Generator], Any],
        total_steps: int = 120,
        group_size: int = 8,
        lr: float = 2.5e-5,
        kl_beta: float = 0.03,
        w_prop: float = 2.40,
        post_processor_kwargs: Optional[Dict[str, Any]] = None,
        equil_mols_for_post: Optional[Sequence[Chem.Mol]] = None,
        onpolicy_pairs_for_post: Optional[Sequence[Tuple[np.ndarray, Chem.Mol]]] = None,
        seed: int = 2026,
        **trainer_kwargs: Any,
    ) -> Dict[str, Any]:
        """Capability-adaptive post-training (`Tree-Vine GRPO` for `Uni-3DAR`, `Loom-AWR` for `PropMolFlow`)."""
        if hasattr(self.generator, "configure_trainable"):
            try:
                self.generator.configure_trainable(train_conditioner=True, train_lora=True)
            except TypeError:
                self.generator.configure_trainable()

        trainer = LoomGRPOTrainer(
            generator=self.generator,
            critic_bus=self.critic_bus,
            replay_molecules=replay_molecules,
            replay_targets_eV=replay_targets,
            alignment_mode="auto",
            lr=lr,
            kl_beta=kl_beta,
            w_prop=w_prop,
            seed=seed,
            **trainer_kwargs,
        )
        history = trainer.train_loop(
            total_steps=total_steps,
            prompt_sampler_fn=prompt_sampler_fn,
            group_size=group_size,
            log_every=max(1, total_steps // 6),
        )
        post_summary = self.generator.fit_internal_post_processors(
            equil_mols_for_post if equil_mols_for_post is not None else replay_molecules,
            onpolicy_pairs_for_post or [],
            self.critic_bus,
            **(post_processor_kwargs or {}),
        )
        return {
            "resolved_alignment_mode": trainer.alignment_mode,
            "total_steps": total_steps,
            "history": history,
            "internal_post_processors": post_summary,
        }
