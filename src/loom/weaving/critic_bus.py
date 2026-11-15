"""Layer 2B: Multi-Asset Composite Critic & Constraint Weaving Bus (`critic_bus.py`).

Unifies:
  - Heterogeneous external/internal `PredictorAsset` batch evaluation (caching each asset call once per batch);
  - Inequality `ConstraintChannelSpec` gating & smooth shaping (`SA_Score <= 3.5`, `QED >= 0.5`, etc.);
  - Dual-mode `AuditorChannelSpec` (`cross_oracle` disagreement OR single-predictor `MorganDomain` + MLIP strain OOD proxy);
  - Transparent side-by-side `Raw` vs `ValenceReconciler` admission & multi-model hindsight pool relabeling;
  - Strict isolation of `is_audit_only=True` assets so blind audit oracles never leak into training rewards.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple
import numpy as np
import torch
from rdkit import Chem

from ..chemistry.valence_reconciler import ReconciliationResult, ValenceReconciler
from ..generator.octree_backend import trajectory_to_instance
from ..oracles.applicability import MorganDomain
from .grpo import (
    compute_disentangled_group_advantages,
    formal_charge_signature,
    is_clean_organic_charge_state,
    property_distance_reward,
    zero_step_physical_reward,
)
from .registry import AssetRegistry, PredictorAsset
from .slots import PotentialSlot
from .spec import TaskWeavingSpec


class CompositeCriticBus:
    """Predictor-agnostic evaluation, constraint-gating, auditing, and hindsight relabeling bus."""

    def __init__(
        self,
        spec: TaskWeavingSpec,
        registry: AssetRegistry,
        *,
        applicability_domain: Optional[MorganDomain] = None,
        seen_charge_signatures: Optional[FrozenSet[Tuple[Tuple[str, int], ...]]] = None,
        allowed_elements: Sequence[str] = (
            "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"
        ),
    ) -> None:
        self.spec = spec
        self.registry = registry
        self.applicability_domain = applicability_domain
        self.seen_charge_signatures = seen_charge_signatures
        self.allowed_elements = tuple(allowed_elements)
        self.reconciler = ValenceReconciler(allowed_elements=self.allowed_elements)

        # Verify strict isolation: no audit_only asset may be used in objectives or constraints!
        for obj in self.spec.objectives:
            asset = self.registry.get_predictor(obj.asset_id)
            if asset.capabilities.is_audit_only:
                raise ValueError(
                    f"Audit-only asset '{obj.asset_id}' cannot be mounted as an ObjectiveChannel ('{obj.name}')"
                )
            asset.channel_index(obj.asset_channel)

        for con in self.spec.constraints:
            asset = self.registry.get_predictor(con.asset_id)
            if asset.capabilities.is_audit_only:
                raise ValueError(
                    f"Audit-only asset '{con.asset_id}' cannot be mounted as a ConstraintChannel ('{con.name}')"
                )
            asset.channel_index(con.asset_channel)

        self.potential_slot: Optional[PotentialSlot] = (
            self.registry.get_potential(self.spec.potential_asset_id)
            if self.spec.potential_asset_id
            else None
        )

    def admit_trajectories(
        self,
        trajectories: Sequence[Dict[str, Any]],
        *,
        reconcile_valence: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Admit sampled octree trajectories, returning both raw and reconciled molecule lists & stats."""
        use_rec = self.spec.reconcile_valence if reconcile_valence is None else bool(reconcile_valence)
        n = len(trajectories)
        raw_mols: List[Optional[Chem.Mol]] = [None] * n
        raw_legal = np.zeros(n, dtype=bool)
        raw_clean_q = np.zeros(n, dtype=bool)

        for i, tr in enumerate(trajectories):
            try:
                inst = trajectory_to_instance(tr, element_table=self.allowed_elements)
                mol = inst.graph
                raw_mols[i] = mol
                raw_legal[i] = True
                raw_clean_q[i] = is_clean_organic_charge_state(mol)
            except ValueError:
                pass

        if not use_rec:
            return {
                "mols": raw_mols,
                "raw_mols": raw_mols,
                "legal_mask": raw_legal,
                "clean_charge_mask": raw_clean_q,
                "raw_legal_mask": raw_legal,
                "raw_clean_charge_mask": raw_clean_q,
                "reconciled_flags": np.zeros(n, dtype=bool),
                "reconcile_results": None,
            }

        rec_results: List[ReconciliationResult] = []
        final_mols: List[Optional[Chem.Mol]] = [None] * n
        rec_legal = np.zeros(n, dtype=bool)
        rec_clean_q = np.zeros(n, dtype=bool)
        rec_flags = np.zeros(n, dtype=bool)

        for i, tr in enumerate(trajectories):
            if raw_legal[i] and raw_clean_q[i] and raw_mols[i] is not None:
                final_mols[i] = raw_mols[i]
                rec_legal[i] = True
                rec_clean_q[i] = True
                rec_results.append(
                    ReconciliationResult(
                        mol=raw_mols[i],
                        raw_mol=raw_mols[i],
                        status="clean_raw",
                        raw_legal=True,
                        raw_exotic_charge=False,
                        reconciled_legal=True,
                        reconciled_exotic_charge=False,
                        was_reconciled=False,
                        heavy_rmsd_A=0.0,
                        delta_h_count=0,
                        smiles=None,
                    )
                )
            else:
                res = self.reconciler.reconcile_point_cloud(
                    tr.get("atom_symbols", []),
                    tr.get("positions_np", np.zeros((0, 3), dtype=np.float32)),
                    potential_slot=None,
                )
                rec_results.append(res)
                if res.reconciled_legal and res.mol is not None:
                    final_mols[i] = res.mol
                    rec_legal[i] = True
                    rec_clean_q[i] = not res.reconciled_exotic_charge
                    rec_flags[i] = res.was_reconciled

        if self.potential_slot is not None and rec_flags.any():
            reconciled_subset = [final_mols[i] for i in range(n) if rec_flags[i] and final_mols[i] is not None]
            self.reconciler.batch_polish_hydrogens_mlip(reconciled_subset, self.potential_slot, n_steps=5, chunk_size=32)

        return {
            "mols": final_mols,
            "raw_mols": raw_mols,
            "legal_mask": rec_legal,
            "clean_charge_mask": rec_clean_q,
            "raw_legal_mask": raw_legal,
            "raw_clean_charge_mask": raw_clean_q,
            "reconciled_flags": rec_flags,
            "reconcile_results": rec_results,
        }

    def evaluate_molecules(
        self,
        mols: Sequence[Optional[Chem.Mol]],
        requested_targets: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        *,
        include_audit: bool = False,
        for_training: bool = False,
        sharp_concentration: bool = False,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Evaluate all objectives, inequality constraints, physics strain, and auditors in one batched pass."""
        n = len(mols)
        num_obj = len(self.spec.objectives)
        legal_mask = np.array([m is not None for m in mols], dtype=bool)
        clean_q_mask = np.array(
            [is_clean_organic_charge_state(m) if m is not None else False for m in mols], dtype=bool
        )

        # Gate exotic zwitterions out of predictor evaluation when training
        eval_input_mols: List[Optional[Chem.Mol]] = [
            (mols[i] if (legal_mask[i] and (clean_q_mask[i] or not for_training)) else None)
            for i in range(n)
        ]

        # 1. Deduplicated asset batch calls
        needed_asset_ids = {o.asset_id for o in self.spec.objectives} | {c.asset_id for c in self.spec.constraints}
        if include_audit:
            for aud in self.spec.auditors:
                if aud.mode == "cross_oracle" and aud.auditor_asset_id:
                    aud_asset = self.registry.get_predictor(aud.auditor_asset_id)
                    if for_training and aud_asset.capabilities.is_audit_only:
                        continue
                    needed_asset_ids.add(aud.auditor_asset_id)

        asset_cache: Dict[str, Dict[str, Any]] = {}
        for aid in sorted(needed_asset_ids):
            asset_obj: PredictorAsset = self.registry.get_predictor(aid)
            asset_cache[aid] = asset_obj.predict_mols(eval_input_mols, batch_size=batch_size)

        # 2. Extract objective predictions `(N, num_obj)`
        obj_preds = np.full((n, num_obj), np.nan, dtype=np.float64)
        obj_stds = np.full((n, num_obj), np.nan, dtype=np.float64)
        obj_valid = np.ones((n, num_obj), dtype=bool)
        for j, obj in enumerate(self.spec.objectives):
            cached = asset_cache[obj.asset_id]
            c_idx = self.registry.get_predictor(obj.asset_id).channel_index(obj.asset_channel)
            obj_preds[:, j] = cached["mean"][:, c_idx]
            obj_stds[:, j] = cached["std"][:, c_idx]
            obj_valid[:, j] = cached["valid_mask"] & np.isfinite(obj_preds[:, j])

        all_obj_evaluated = legal_mask & clean_q_mask & obj_valid.all(axis=-1)

        # 3. Evaluate inequality constraints (`SA_Score <= 3.5`, `QED >= 0.5`, etc.)
        constraints_met = np.ones(n, dtype=bool)
        hard_constraints_met = np.ones(n, dtype=bool)
        constraint_scores = np.ones(n, dtype=np.float64)
        constraint_values: Dict[str, np.ndarray] = {}

        if self.spec.constraints:
            weighted_c_sum = np.zeros(n, dtype=np.float64)
            total_c_w = 0.0
            for con in self.spec.constraints:
                cached = asset_cache[con.asset_id]
                c_idx = self.registry.get_predictor(con.asset_id).channel_index(con.asset_channel)
                vals = cached["mean"][:, c_idx]
                constraint_values[con.name] = vals
                sat = con.satisfied_mask(vals) & cached["valid_mask"]
                constraints_met &= sat
                if con.hard_gate:
                    hard_constraints_met &= sat
                weighted_c_sum += con.weight * con.smooth_satisfaction_score(vals)
                total_c_w += con.weight
            if total_c_w > 0:
                constraint_scores = weighted_c_sum / total_c_w

        # 4. Evaluate `PotentialSlot` zero-step atomic forces
        f_max = np.full(n, np.nan, dtype=np.float64)
        f_rms = np.full(n, np.nan, dtype=np.float64)
        phys_scores = np.zeros(n, dtype=np.float32)
        valid_indices = [i for i in range(n) if mols[i] is not None]
        if self.potential_slot is not None and valid_indices:
            sub_mols = [mols[i] for i in valid_indices]  # type: ignore[misc]
            p_out = self.potential_slot.evaluate_molecules(sub_mols, chunk_size=batch_size)
            r_phys = zero_step_physical_reward(p_out["f_rms_eV_A"], p_out["f_max_eV_A"]).numpy()
            for loc_i, glob_i in enumerate(valid_indices):
                f_max[glob_i] = p_out["f_max_eV_A"][loc_i]
                f_rms[glob_i] = p_out["f_rms_eV_A"][loc_i]
                if np.isfinite(r_phys[loc_i]):
                    phys_scores[glob_i] = r_phys[loc_i]

        # 5. Compute objective distance, hits, and constraint-shaped property scores
        prop_scores = np.zeros(n, dtype=np.float32)
        target_hits = np.zeros(n, dtype=bool)
        constrained_clean_hits = np.zeros(n, dtype=bool)
        mean_norm_err = np.full(n, np.nan, dtype=np.float64)

        if requested_targets is not None:
            req = np.asarray(requested_targets, dtype=np.float64).reshape(n, num_obj)
            act = (
                np.asarray(active_mask, dtype=bool).reshape(n, num_obj)
                if active_mask is not None
                else np.ones((n, num_obj), dtype=bool)
            )
            for i in range(n):
                if not (legal_mask[i] and clean_q_mask[i]):
                    continue
                act_i = act[i]
                if not act_i.any():
                    continue
                if not obj_valid[i, act_i].all():
                    continue

                per_ch_rewards = []
                per_ch_weights = []
                all_ch_hit = True
                norm_errs = []
                for j, obj in enumerate(self.spec.objectives):
                    if not act_i[j]:
                        continue
                    dist_j = float(obj.distance_to_goal(obj_preds[i, j], req[i, j]))
                    hit_j = bool(obj.is_hit(obj_preds[i, j], req[i, j]))
                    all_ch_hit = all_ch_hit and hit_j
                    norm_errs.append(dist_j / obj.tolerance)
                    r_j = float(
                        property_distance_reward(
                            dist_j, obj.tolerance, sharp_concentration=sharp_concentration
                        ).item()
                    )
                    per_ch_rewards.append(obj.weight * r_j)
                    per_ch_weights.append(obj.weight)

                raw_r = float(sum(per_ch_rewards) / max(sum(per_ch_weights), 1e-8))
                mean_norm_err[i] = float(np.mean(norm_errs))
                target_hits[i] = all_ch_hit
                constrained_clean_hits[i] = all_ch_hit and bool(constraints_met[i])

                # Enforce strict hierarchy: violating a hard constraint caps property score below clean misses
                if hard_constraints_met[i]:
                    prop_scores[i] = float(0.20 * constraint_scores[i] + 0.80 * raw_r)
                else:
                    prop_scores[i] = float(0.18 * constraint_scores[i])

        # 6. Auditor channels (cross-oracle disagreement OR single-oracle Applicability Domain + strain)
        audit_preds: Dict[str, np.ndarray] = {}
        sigma_cal_est = np.full((n, num_obj), np.nan, dtype=np.float64)
        in_ad_mask = np.ones(n, dtype=bool)

        if include_audit and self.spec.auditors:
            obj_name_to_idx = {o.name: idx for idx, o in enumerate(self.spec.objectives)}
            for aud in self.spec.auditors:
                if aud.objective_name not in obj_name_to_idx:
                    continue
                j_idx = obj_name_to_idx[aud.objective_name]
                if aud.mode == "cross_oracle" and aud.auditor_asset_id and aud.auditor_asset_id in asset_cache:
                    aud_cached = asset_cache[aud.auditor_asset_id]
                    ch_name = aud.auditor_channel or aud.objective_name
                    c_idx = self.registry.get_predictor(aud.auditor_asset_id).channel_index(ch_name)
                    a_vals = aud_cached["mean"][:, c_idx]
                    audit_preds[aud.objective_name] = a_vals
                    both = obj_valid[:, j_idx] & aud_cached["valid_mask"] & np.isfinite(a_vals)
                    sigma_cal_est[both, j_idx] = np.abs(obj_preds[both, j_idx] - a_vals[both])
                elif aud.mode == "single_oracle_ad":
                    # Single-oracle OOD proxy: ensemble/GP std + Morgan Tanimoto distance
                    # + calibrated target-support extrapolation + MLIP strain excess
                    obj_spec = self.spec.objectives[j_idx]
                    obj_asset = self.registry.get_predictor(obj_spec.asset_id)
                    cal_bounds = getattr(obj_asset, "calibrated_target_bounds", None)
                    c_idx = obj_asset.channel_index(obj_spec.asset_channel)
                    ch_bound = (
                        cal_bounds[c_idx]
                        if (cal_bounds is not None and c_idx < len(cal_bounds))
                        else None
                    )
                    req_col = (
                        np.asarray(requested_targets, dtype=np.float64).reshape(n, num_obj)[:, j_idx]
                        if requested_targets is not None
                        else None
                    )
                    for i in range(n):
                        if not obj_valid[i, j_idx] or mols[i] is None:
                            continue
                        ens_sd = float(obj_stds[i, j_idx]) if np.isfinite(obj_stds[i, j_idx]) else 0.12
                        ad_pen = 0.0
                        if self.applicability_domain is not None:
                            try:
                                info_ad = self.applicability_domain.assess(mols[i])
                                in_ad_mask[i] = in_ad_mask[i] and bool(info_ad["in_domain"])
                                sim = float(info_ad["similarity"])
                                ad_pen = max(0.0, aud.tanimoto_cutoff + 0.08 - sim) * 0.55
                                if info_ad["unseen_training_elements"]:
                                    ad_pen += 0.50
                            except Exception:
                                in_ad_mask[i] = False
                                ad_pen = 0.30
                        extrap_pen = 0.0
                        if ch_bound is not None and req_col is not None and np.isfinite(req_col[i]):
                            d_out = max(0.0, ch_bound[0] - float(req_col[i]), float(req_col[i]) - ch_bound[1])
                            if d_out > 0.02:
                                extrap_pen = min(0.45, 0.14 + 0.80 * d_out)
                        strain_pen = 0.0
                        if (
                            obj_asset.capabilities.modality != "2d_graph"
                            and np.isfinite(f_max[i])
                            and f_max[i] > 4.0
                        ):
                            strain_pen = min(0.40, 0.05 * (f_max[i] - 4.0))
                        sigma_cal_est[i, j_idx] = float(
                            np.sqrt(ens_sd ** 2 + ad_pen ** 2 + extrap_pen ** 2 + strain_pen ** 2)
                        )

        # 7. Structural validity scores for GRPO
        valid_scores = np.full(n, -1.0, dtype=np.float32)
        for i in range(n):
            if not legal_mask[i] or mols[i] is None:
                continue
            if not clean_q_mask[i]:
                valid_scores[i] = -0.35
            elif not hard_constraints_met[i]:
                valid_scores[i] = 0.35 * float(constraint_scores[i])
            elif self.seen_charge_signatures is not None:
                sig = formal_charge_signature(mols[i])  # type: ignore[arg-type]
                valid_scores[i] = 1.0 if sig in self.seen_charge_signatures else 0.80
            else:
                valid_scores[i] = 1.0

        return {
            "legal_mask": legal_mask,
            "clean_charge_mask": clean_q_mask,
            "evaluated_mask": all_obj_evaluated,
            "objective_preds": obj_preds,
            "objective_stds": obj_stds,
            "constraints_met": constraints_met,
            "hard_constraints_met": hard_constraints_met,
            "constraint_scores": constraint_scores,
            "constraint_values": constraint_values,
            "f_max_eV_A": f_max,
            "f_rms_eV_A": f_rms,
            "valid_scores": valid_scores,
            "phys_scores": phys_scores,
            "prop_scores": prop_scores,
            "target_hits": target_hits,
            "constrained_clean_hits": constrained_clean_hits,
            "mean_normalized_error": mean_norm_err,
            "audit_preds": audit_preds,
            "sigma_cal_estimate": sigma_cal_est,
            "in_applicability_domain": in_ad_mask,
        }

    def compute_group_advantages(
        self,
        eval_bundle: Dict[str, Any],
        *,
        group_size: int,
        device: torch.device,
        w_valid: float = 0.75,
        w_phys: float = 0.25,
        w_prop: float = 1.00,
    ) -> torch.Tensor:
        total_b = len(eval_bundle["valid_scores"])
        p_count = total_b // group_size
        valid_scores_t = torch.as_tensor(eval_bundle["valid_scores"], device=device, dtype=torch.float32)
        valid_mask_t = torch.as_tensor(eval_bundle["legal_mask"], device=device, dtype=torch.bool)
        phys_scores_t = torch.as_tensor(eval_bundle["phys_scores"], device=device, dtype=torch.float32)
        eval_mask_t = torch.as_tensor(eval_bundle["evaluated_mask"], device=device, dtype=torch.bool)
        prop_scores_t = torch.as_tensor(eval_bundle["prop_scores"], device=device, dtype=torch.float32)

        adv_all = torch.zeros(total_b, device=device, dtype=torch.float32)
        for p_idx in range(p_count):
            s = p_idx * group_size
            e = s + group_size
            adv_g, _ = compute_disentangled_group_advantages(
                valid_scores_t[s:e],
                valid_mask_t[s:e],
                phys_scores_t[s:e],
                eval_mask_t[s:e],
                prop_scores_t[s:e],
                w_valid=w_valid,
                w_phys=w_phys if self.potential_slot is not None else 0.0,
                w_prop=w_prop,
            )
            adv_all[s:e] = adv_g
        return adv_all

    def relabel_conformer_pool(
        self,
        mols: Sequence[Chem.Mol],
        *,
        max_fmax_eV_A: float = 4.0,
        batch_size: int = 32,
    ) -> Tuple[List[Chem.Mol], np.ndarray, Dict[str, Any]]:
        """Multi-asset pseudo-label completion & inequality-constraint filtering on any 3D conformer pool."""
        ev = self.evaluate_molecules(mols, include_audit=False, for_training=True, batch_size=batch_size)
        admitted_mols: List[Chem.Mol] = []
        admitted_labels: List[np.ndarray] = []

        for i, mol in enumerate(mols):
            if not ev["evaluated_mask"][i]:
                continue
            if not ev["hard_constraints_met"][i]:
                continue
            if self.potential_slot is not None and np.isfinite(ev["f_max_eV_A"][i]):
                if ev["f_max_eV_A"][i] > max_fmax_eV_A:
                    continue
            admitted_mols.append(mol)
            admitted_labels.append(ev["objective_preds"][i])

        labels_mat = (
            np.stack(admitted_labels, axis=0).astype(np.float32)
            if admitted_labels
            else np.zeros((0, len(self.spec.objectives)), dtype=np.float32)
        )
        summary = {
            "input_pool_size": len(mols),
            "evaluated_clean_count": int(ev["evaluated_mask"].sum()),
            "constraints_met_count": int((ev["evaluated_mask"] & ev["hard_constraints_met"]).sum()),
            "admitted_count": len(admitted_mols),
            "admitted_rate": len(admitted_mols) / max(len(mols), 1),
        }
        return admitted_mols, labels_mat, summary
