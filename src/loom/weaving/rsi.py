"""Layer 4: Grounded Recursive Self-Improvement (RSI) Engine (`src/loom/weaving/rsi.py`).

Implements the complete Weak-Zone Diagnostic -> Inner-Loop Curriculum Self-Improvement ->
Outer-Loop Active Acquisition -> Residual Incremental Calibration state machine without
model collapse or reward hacking:

  1. `RSIDiagnoser`:
       Classifies every target interval `t` on `GPAtlas` into 4 causal states:
         - `CONVERGED`           : High clean hit yield, low bias, trusted predictor (`sigma_cal <= cal_cap`)
         - `INNER_UNDEREXPLORED` : Trusted predictor (`sigma_cal <= cal_cap`), wide generator spread / low hit rate
         - `OUTER_BLINDSPOT`     : Valid clean molecules exist, but predictor disagreement/OOD `sigma_cal > cal_cap`
                                   (RL frozen here to prevent reward hacking; triggers active acquisition!)
         - `PRIOR_CEILING`       : Zero source support + severe mean pull-back (`|bias| > ceiling_bias_cap`)
  2. `InnerRSIRunner`:
       Zero-external-cost curriculum focusing (`80%` `INNER_UNDEREXPLORED` + `20%` `CONVERGED` anchor,
       `0%` uncalibrated `OUTER_BLINDSPOT`) with Hindsight Replay Augmentation + Topology-Decoupled GRPO
       + non-regression Acceptance Gate.
  3. `OuterRSIAcquisition` & `IncrementalCalibrator`:
       Selects clean, constraint-satisfying, scaffold-diverse molecules from `OUTER_BLINDSPOT` zones via
       uncertainty-weighted farthest-point Morgan Tanimoto selection, exports `acquisition_batch.sdf`,
       and ingests external high-fidelity (`DFT` / assay) feedback into `ResidualCalibratedAsset`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..generator.molecular_state import mol_to_clean_smiles
from .critic_bus import CompositeCriticBus
from .gp_atlas import HeteroscedasticGPAtlas1D
from .grpo import LoomGRPOTrainer, is_clean_organic_charge_state
from .registry import ResidualCalibratedAsset
from .slots import GeneratorSlot


@dataclass(frozen=True)
class RSIDiagnosticThresholds:
    hit_floor: float = 0.24
    bias_cap: float = 0.30
    cal_sigma_cap: float = 0.32
    spread_cap: float = 0.48
    eval_floor: float = 0.35
    ceiling_bias_cap: float = 0.55
    source_density_floor_ratio: float = 0.06


class RSIDiagnoser:
    """Four-quadrant weak-zone causal classifier & safe curriculum constructor."""

    def __init__(
        self,
        thresholds: Optional[RSIDiagnosticThresholds] = None,
        *,
        uniform_trusted_anchor: bool = False,
    ) -> None:
        self.thresholds = thresholds or RSIDiagnosticThresholds()
        self.uniform_trusted_anchor = bool(uniform_trusted_anchor)

    def diagnose_1d(self, atlas_query: Dict[str, Any]) -> Dict[str, Any]:
        th = self.thresholds
        targets = np.asarray(atlas_query["targets"], dtype=np.float64)
        bias = np.asarray(atlas_query["bias"], dtype=np.float64)
        sigma_gen = np.asarray(atlas_query["sigma_gen"], dtype=np.float64)
        sigma_cal = np.asarray(atlas_query["sigma_cal"], dtype=np.float64)
        p_eval = np.asarray(atlas_query["p_eval"], dtype=np.float64)
        p_hit = np.asarray(atlas_query["p_hit_reward"], dtype=np.float64)
        p_hit_cal = np.asarray(atlas_query["p_hit_dft_calibrated"], dtype=np.float64)
        kde = np.asarray(atlas_query["source_kde_density"], dtype=np.float64)
        kde_peak = float(np.max(kde)) if len(kde) > 0 else 0.0

        labels: List[str] = []
        deficits = np.zeros_like(targets, dtype=np.float64)
        acq_priority = np.zeros_like(targets, dtype=np.float64)

        for i in range(len(targets)):
            low_src = (kde_peak > 0.0) and (kde[i] <= th.source_density_floor_ratio * kde_peak)
            if low_src and abs(bias[i]) > th.ceiling_bias_cap:
                labels.append("PRIOR_CEILING")
            elif p_eval[i] >= th.eval_floor and sigma_cal[i] > th.cal_sigma_cap:
                labels.append("OUTER_BLINDSPOT")
                acq_priority[i] = float((sigma_cal[i] - th.cal_sigma_cap) * p_eval[i])
            elif (
                p_hit[i] >= th.hit_floor
                and abs(bias[i]) <= th.bias_cap
                and sigma_cal[i] <= th.cal_sigma_cap
                and sigma_gen[i] <= th.spread_cap
            ):
                labels.append("CONVERGED")
            else:
                labels.append("INNER_UNDEREXPLORED")
                deficits[i] = float(
                    max(0.0, th.hit_floor - p_hit[i])
                    + 0.40 * max(0.0, sigma_gen[i] - th.spread_cap)
                    + 0.30 * max(0.0, abs(bias[i]) - 0.5 * th.bias_cap)
                    + 0.05
                )

        # Safe Curriculum Sampling Distribution:
        # 80% on INNER_UNDEREXPLORED, 20% anchor on CONVERGED, 0% on OUTER_BLINDSPOT / PRIOR_CEILING
        n = len(targets)
        inner_mask = np.array([lb == "INNER_UNDEREXPLORED" for lb in labels], dtype=bool)
        conv_mask = np.array([lb == "CONVERGED" for lb in labels], dtype=bool)
        blind_mask = np.array([lb == "OUTER_BLINDSPOT" for lb in labels], dtype=bool)
        ceil_mask = np.array([lb == "PRIOR_CEILING" for lb in labels], dtype=bool)

        probs = np.zeros(n, dtype=np.float64)
        trusted_mask = (~blind_mask) & (~ceil_mask)
        if self.uniform_trusted_anchor and inner_mask.any() and trusted_mask.any():
            w_in = deficits[inner_mask]
            probs[inner_mask] = 0.78 * (w_in / np.sum(w_in))
            probs[trusted_mask] += 0.22 / int(trusted_mask.sum())
        elif inner_mask.any() and conv_mask.any():
            w_in = deficits[inner_mask]
            probs[inner_mask] = 0.80 * (w_in / np.sum(w_in))
            probs[conv_mask] = 0.20 / int(conv_mask.sum())
        elif inner_mask.any():
            w_in = deficits[inner_mask]
            probs[inner_mask] = w_in / np.sum(w_in)
        elif conv_mask.any():
            probs[conv_mask] = 1.0 / int(conv_mask.sum())
        else:
            # Fallback uniform over non-blindspot, non-ceiling targets if all points are extreme
            safe = (~blind_mask) & (~ceil_mask)
            if safe.any():
                probs[safe] = 1.0 / int(safe.sum())
            else:
                non_ceil = ~ceil_mask
                if non_ceil.any():
                    probs[non_ceil] = 1.0 / int(non_ceil.sum())
                else:
                    probs[:] = 1.0 / max(n, 1)

        counts = {
            k: int(sum(1 for lb in labels if lb == k))
            for k in ("CONVERGED", "INNER_UNDEREXPLORED", "OUTER_BLINDSPOT", "PRIOR_CEILING")
        }

        return {
            "axis_name": atlas_query.get("axis_name", "target"),
            "unit": atlas_query.get("unit", "eV"),
            "targets": targets.tolist(),
            "rsi_zones": labels,
            "zone_counts": counts,
            "curriculum_sampling_probs": probs.tolist(),
            "inner_deficit_scores": deficits.tolist(),
            "outer_acquisition_priority": acq_priority.tolist(),
            "mean_calibrated_hit_prob": float(np.mean(p_hit_cal)),
            "mean_sigma_cal": float(np.mean(sigma_cal)),
            "mean_sigma_gen": float(np.mean(sigma_gen)),
            "inner_target_intervals": targets[inner_mask].tolist(),
            "outer_blindspot_targets": targets[blind_mask].tolist(),
            "prior_ceiling_targets": targets[ceil_mask].tolist(),
        }

    def diagnose_nd(
        self,
        channel_queries: Sequence[Dict[str, Any]],
        *,
        tolerances: Optional[Sequence[float]] = None,
        targets_nd: Optional[np.ndarray] = None,
        cal_sigma_ratio_cap: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Multi-objective $K$-D weak-zone causal diagnosis for Self-Driving Labs.

        Given $K$ aligned `GPAtlas` query dictionaries over $N$ joint target vectors
        `t_i = (t_{i,1}, ..., t_{i,K})`:
          - Computes joint hit yield `prod_k P_hit_k(t_i)` and calibrated `prod_k P_hit_cal_k(t_i)`.
          - Flags `OUTER_BLINDSPOT` whenever ANY objective channel has normalized epistemic
            uncertainty `sigma_cal_k / tol_k > cal_sigma_ratio_cap` (default `cal_sigma_cap / 0.25`),
            freezing RL on `t_i` to prevent multi-objective Goodhart hacking and routing to active acquisition.
        """
        if not channel_queries:
            raise ValueError("diagnose_nd requires at least one channel query dict")
        th = self.thresholds
        k_dim = len(channel_queries)
        n = len(channel_queries[0]["targets"])
        if targets_nd is not None:
            t_mat = np.asarray(targets_nd, dtype=np.float64).reshape(n, k_dim)
        else:
            t_mat = np.stack([np.asarray(q["targets"], dtype=np.float64).reshape(n) for q in channel_queries], axis=-1)

        tols = (
            np.asarray(tolerances, dtype=np.float64).reshape(k_dim)
            if tolerances is not None
            else np.full(k_dim, 0.25, dtype=np.float64)
        )
        tols = np.maximum(tols, 1e-4)
        ratio_cap = float(cal_sigma_ratio_cap) if cal_sigma_ratio_cap is not None else float(th.cal_sigma_cap / 0.25)

        bias_mat = np.stack([np.asarray(q["bias"], dtype=np.float64).reshape(n) for q in channel_queries], axis=-1)
        sgen_mat = np.stack([np.asarray(q["sigma_gen"], dtype=np.float64).reshape(n) for q in channel_queries], axis=-1)
        scal_mat = np.stack([np.asarray(q["sigma_cal"], dtype=np.float64).reshape(n) for q in channel_queries], axis=-1)
        peval_mat = np.stack([np.asarray(q["p_eval"], dtype=np.float64).reshape(n) for q in channel_queries], axis=-1)
        phit_mat = np.stack([np.asarray(q["p_hit_reward"], dtype=np.float64).reshape(n) for q in channel_queries], axis=-1)
        pcal_mat = np.stack(
            [np.asarray(q["p_hit_dft_calibrated"], dtype=np.float64).reshape(n) for q in channel_queries], axis=-1
        )
        kde_mat = np.stack(
            [np.asarray(q.get("source_kde_density", np.ones(n)), dtype=np.float64).reshape(n) for q in channel_queries],
            axis=-1,
        )
        kde_peaks = np.max(kde_mat, axis=0)

        p_eval_joint = np.min(peval_mat, axis=-1)
        p_hit_joint = np.prod(np.clip(phit_mat, 0.0, 1.0), axis=-1)
        p_cal_joint = np.prod(np.clip(pcal_mat, 0.0, 1.0), axis=-1)
        scal_ratio = scal_mat / tols[None, :]
        bias_ratio = np.abs(bias_mat) / tols[None, :]
        sgen_ratio = sgen_mat / tols[None, :]

        hit_floor_nd = float(th.hit_floor ** max(1.0, 0.65 * k_dim))
        bias_ratio_cap = float(th.bias_cap / 0.25)
        spread_ratio_cap = float(th.spread_cap / 0.25)
        ceil_ratio_cap = float(th.ceiling_bias_cap / 0.25)

        labels: List[str] = []
        blindspot_channels: List[List[int]] = []
        deficits = np.zeros(n, dtype=np.float64)
        acq_priority = np.zeros(n, dtype=np.float64)

        for i in range(n):
            low_src_any = any(
                (kde_peaks[c] > 0.0)
                and (kde_mat[i, c] <= th.source_density_floor_ratio * kde_peaks[c])
                and (bias_ratio[i, c] > ceil_ratio_cap)
                for c in range(k_dim)
            )
            bad_cal_cols = [c for c in range(k_dim) if scal_ratio[i, c] > ratio_cap]
            blindspot_channels.append(bad_cal_cols)
            if low_src_any:
                labels.append("PRIOR_CEILING")
            elif p_eval_joint[i] >= th.eval_floor and len(bad_cal_cols) > 0:
                labels.append("OUTER_BLINDSPOT")
                excess = float(np.max(scal_ratio[i] - ratio_cap))
                acq_priority[i] = float(max(excess, 0.01) * p_eval_joint[i])
            elif (
                p_hit_joint[i] >= hit_floor_nd
                and float(np.max(bias_ratio[i])) <= bias_ratio_cap
                and len(bad_cal_cols) == 0
                and float(np.max(sgen_ratio[i])) <= spread_ratio_cap
            ):
                labels.append("CONVERGED")
            else:
                labels.append("INNER_UNDEREXPLORED")
                deficits[i] = float(
                    max(0.0, hit_floor_nd - p_hit_joint[i])
                    + 0.35 * max(0.0, float(np.mean(sgen_ratio[i])) - spread_ratio_cap)
                    + 0.25 * max(0.0, float(np.mean(bias_ratio[i])) - 0.5 * bias_ratio_cap)
                    + 0.05
                )

        inner_mask = np.array([lb == "INNER_UNDEREXPLORED" for lb in labels], dtype=bool)
        conv_mask = np.array([lb == "CONVERGED" for lb in labels], dtype=bool)
        blind_mask = np.array([lb == "OUTER_BLINDSPOT" for lb in labels], dtype=bool)
        ceil_mask = np.array([lb == "PRIOR_CEILING" for lb in labels], dtype=bool)
        trusted_mask = (~blind_mask) & (~ceil_mask)

        probs = np.zeros(n, dtype=np.float64)
        if self.uniform_trusted_anchor and inner_mask.any() and trusted_mask.any():
            w_in = deficits[inner_mask]
            probs[inner_mask] = 0.78 * (w_in / np.sum(w_in))
            probs[trusted_mask] += 0.22 / int(trusted_mask.sum())
        elif inner_mask.any() and conv_mask.any():
            w_in = deficits[inner_mask]
            probs[inner_mask] = 0.80 * (w_in / np.sum(w_in))
            probs[conv_mask] = 0.20 / int(conv_mask.sum())
        elif inner_mask.any():
            w_in = deficits[inner_mask]
            probs[inner_mask] = w_in / np.sum(w_in)
        elif conv_mask.any():
            probs[conv_mask] = 1.0 / int(conv_mask.sum())
        elif trusted_mask.any():
            probs[trusted_mask] = 1.0 / int(trusted_mask.sum())
        else:
            probs[:] = 1.0 / max(n, 1)

        counts = {
            k: int(sum(1 for lb in labels if lb == k))
            for k in ("CONVERGED", "INNER_UNDEREXPLORED", "OUTER_BLINDSPOT", "PRIOR_CEILING")
        }
        axis_names = [str(q.get("axis_name", f"obj_{c}")) for c, q in enumerate(channel_queries)]
        units = [str(q.get("unit", "unspecified")) for q in channel_queries]

        return {
            "axis_names": axis_names,
            "units": units,
            "tolerances": tols.tolist(),
            "targets": t_mat.tolist(),
            "rsi_zones": labels,
            "blindspot_channels": blindspot_channels,
            "zone_counts": counts,
            "curriculum_sampling_probs": probs.tolist(),
            "inner_deficit_scores": deficits.tolist(),
            "outer_acquisition_priority": acq_priority.tolist(),
            "joint_reward_hit_prob": p_hit_joint.tolist(),
            "joint_calibrated_hit_prob": p_cal_joint.tolist(),
            "mean_calibrated_hit_prob": float(np.mean(p_cal_joint)),
            "mean_max_sigma_cal_ratio": float(np.mean(np.max(scal_ratio, axis=-1))),
            "inner_target_vectors": t_mat[inner_mask].tolist(),
            "outer_blindspot_targets": t_mat[blind_mask].tolist(),
            "prior_ceiling_targets": t_mat[ceil_mask].tolist(),
        }


class OuterRSIAcquisition:
    """Scaffold-diverse active-acquisition batch selector & SDF exporter for `OUTER_BLINDSPOT`."""

    def __init__(self, *, max_fmax_eV_A: float = 3.5, fp_bits: int = 2048) -> None:
        self.max_fmax_eV_A = float(max_fmax_eV_A)
        self._fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_bits, includeChirality=True)

    def select_acquisition_batch(
        self,
        mols: Sequence[Optional[Chem.Mol]],
        requested_targets: np.ndarray,
        base_preds: np.ndarray,
        sigma_cal_vec: np.ndarray,
        *,
        constraints_met: Optional[np.ndarray] = None,
        f_max_eV_A: Optional[np.ndarray] = None,
        blindspot_target_values: Optional[Sequence[float]] = None,
        budget_k: int = 64,
        min_sigma_cal: float = 0.22,
    ) -> Tuple[List[Chem.Mol], List[Dict[str, Any]]]:
        """Greedy uncertainty-weighted farthest-point Tanimoto active-learning batch selection."""
        n = len(mols)
        req = np.asarray(requested_targets, dtype=np.float64).reshape(n, -1)
        preds = np.asarray(base_preds, dtype=np.float64).reshape(n, -1)
        sig = np.asarray(sigma_cal_vec, dtype=np.float64).reshape(n)
        c_ok = np.asarray(constraints_met, dtype=bool) if constraints_met is not None else np.ones(n, dtype=bool)
        f_arr = np.asarray(f_max_eV_A, dtype=np.float64) if f_max_eV_A is not None else np.zeros(n, dtype=np.float64)

        blind_arr = (
            np.asarray(blindspot_target_values, dtype=np.float64)
            if (blindspot_target_values is not None and len(blindspot_target_values) > 0)
            else None
        )

        # 1. Deduplicate clean, constraint-valid, low-strain candidates
        seen_smi: set[str] = set()
        cand_mols: List[Chem.Mol] = []
        cand_fps: List[Any] = []
        cand_meta: List[Dict[str, Any]] = []
        cand_unc: List[float] = []

        for i in range(n):
            mol = mols[i]
            if mol is None or not c_ok[i]:
                continue
            if not is_clean_organic_charge_state(mol):
                continue
            if not np.isfinite(sig[i]):
                continue
            if np.isfinite(f_arr[i]) and f_arr[i] > self.max_fmax_eV_A:
                continue

            in_blind_req = False
            if blind_arr is not None:
                in_blind_req = bool(np.min(np.abs(blind_arr - req[i, 0])) <= 0.35)
            if sig[i] < min_sigma_cal and not in_blind_req:
                continue

            smi = mol_to_clean_smiles(mol)
            if smi in seen_smi:
                continue
            seen_smi.add(smi)

            no_h = Chem.RemoveHs(Chem.Mol(mol))
            fp = self._fp_gen.GetFingerprint(no_h)
            try:
                scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(no_h))
            except Exception:
                scaf = ""

            bonus = 1.25 if in_blind_req else 1.0
            u_base = float(max(sig[i], 0.05) * bonus)
            cand_mols.append(Chem.Mol(mol))
            cand_fps.append(fp)
            cand_unc.append(u_base)
            cand_meta.append(
                {
                    "smiles": smi,
                    "murcko_scaffold": scaf,
                    "requested_target": req[i].tolist(),
                    "base_prediction": preds[i].tolist(),
                    "sigma_cal": float(sig[i]),
                    "f_max_eV_A": float(f_arr[i]) if np.isfinite(f_arr[i]) else None,
                    "in_blindspot_interval": in_blind_req,
                }
            )

        if not cand_mols:
            return [], []

        k_sel = min(int(budget_k), len(cand_mols))
        unc_arr = np.asarray(cand_unc, dtype=np.float64)
        min_tanimoto_dist = np.ones(len(cand_mols), dtype=np.float64)
        chosen_mask = np.zeros(len(cand_mols), dtype=bool)
        selected_mols: List[Chem.Mol] = []
        selected_records: List[Dict[str, Any]] = []

        for rank in range(k_sel):
            scores = np.where(chosen_mask, -1.0, unc_arr * (0.15 + min_tanimoto_dist))
            best_idx = int(np.argmax(scores))
            chosen_mask[best_idx] = True
            m_copy = Chem.Mol(cand_mols[best_idx])
            rec = dict(cand_meta[best_idx])
            rec["acquisition_rank"] = rank + 1
            rec["acquisition_utility"] = float(scores[best_idx])
            m_copy.SetProp("_Name", f"loom_rsi_acq_{rank+1:04d}")
            m_copy.SetProp("SMILES", rec["smiles"])
            m_copy.SetProp("SigmaCal", f"{rec['sigma_cal']:.4f}")
            m_copy.SetProp("RequestedTarget", json.dumps(rec["requested_target"]))
            m_copy.SetProp("BasePrediction", json.dumps(rec["base_prediction"]))
            selected_mols.append(m_copy)
            selected_records.append(rec)

            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(cand_fps[best_idx], cand_fps), dtype=np.float64)
            dists = 1.0 - sims
            min_tanimoto_dist = np.minimum(min_tanimoto_dist, dists)

        return selected_mols, selected_records

    @staticmethod
    def export_sdf(mols: Sequence[Chem.Mol], records: Sequence[Dict[str, Any]], sdf_path: Path | str) -> Dict[str, Any]:
        out = Path(sdf_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(out))
        for m in mols:
            writer.write(m)
        writer.close()

        scaffolds = {r["murcko_scaffold"] for r in records if r.get("murcko_scaffold")}
        manifest = {
            "sdf_path": str(out),
            "exported_count": len(mols),
            "unique_murcko_scaffolds": len(scaffolds),
            "mean_sigma_cal": float(np.mean([r["sigma_cal"] for r in records])) if records else 0.0,
            "records": list(records),
        }
        out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


class IncrementalCalibrator:
    """Ingests external `DFT` / wet-lab feedback labels into `ResidualCalibratedAsset`."""

    @staticmethod
    def ingest_labeled_feedback(
        calibrated_asset: ResidualCalibratedAsset,
        feedback_mols: Sequence[Chem.Mol],
        true_targets: np.ndarray,
        *,
        validation_mols: Optional[Sequence[Chem.Mol]] = None,
        validation_true_targets: Optional[np.ndarray] = None,
        save_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        pre_val_mae = None
        pre_val_std = None
        if validation_mols is not None and validation_true_targets is not None and len(validation_mols) > 0:
            v_pre = calibrated_asset.predict_mols(validation_mols)
            ok = v_pre["valid_mask"]
            if ok.any():
                vt = np.asarray(validation_true_targets, dtype=np.float64).reshape(len(validation_mols), -1)
                err_m = np.abs(v_pre["mean"][ok] - vt[ok])
                pre_val_mae = [
                    (float(np.nanmean(err_m[:, j])) if np.isfinite(err_m[:, j]).any() else None)
                    for j in range(err_m.shape[1])
                ]
                pre_val_std = np.mean(v_pre["std"][ok], axis=0).tolist()

        ingest_stats = calibrated_asset.ingest_feedback(feedback_mols, true_targets)
        if save_path is not None:
            calibrated_asset.save_calibrator(save_path)

        post_val_mae = None
        post_val_std = None
        if validation_mols is not None and validation_true_targets is not None and len(validation_mols) > 0:
            v_post = calibrated_asset.predict_mols(validation_mols)
            ok = v_post["valid_mask"]
            if ok.any():
                vt = np.asarray(validation_true_targets, dtype=np.float64).reshape(len(validation_mols), -1)
                err_m = np.abs(v_post["mean"][ok] - vt[ok])
                post_val_mae = [
                    (float(np.nanmean(err_m[:, j])) if np.isfinite(err_m[:, j]).any() else None)
                    for j in range(err_m.shape[1])
                ]
                post_val_std = np.mean(v_post["std"][ok], axis=0).tolist()

        return {
            **ingest_stats,
            "pre_calibration_blindspot_mae": pre_val_mae,
            "post_calibration_blindspot_mae": post_val_mae,
            "pre_calibration_mean_std": pre_val_std,
            "post_calibration_mean_std": post_val_std,
        }


class InnerRSIRunner:
    """Curriculum-focused Inner-Loop RSI runner with Hindsight Replay Augmentation + Acceptance Gate."""

    def __init__(
        self,
        generator: GeneratorSlot,
        critic_bus: CompositeCriticBus,
        *,
        prop_channel_idx: int = 0,
        active_channel_indices: Optional[Sequence[int]] = None,
        anchor_channel_values: Optional[Dict[int, float]] = None,
        alignment_mode: str = "auto",
        lr: float = 2.5e-5,
        kl_beta: float = 0.03,
        replay_weight: float = 0.20,
        cfg_scale: float = 1.65,
        geometric_guard: bool = True,
        topology_only_policy: bool = True,
        seed: int = 101,
    ) -> None:
        self.generator = generator
        self.critic_bus = critic_bus
        self.prop_channel_idx = int(prop_channel_idx)
        self.active_channel_indices = (
            tuple(int(i) for i in active_channel_indices)
            if active_channel_indices is not None
            else None
        )
        self.anchor_channel_values = dict(anchor_channel_values) if anchor_channel_values is not None else {}
        self.alignment_mode = str(alignment_mode)
        self.lr = float(lr)
        self.kl_beta = float(kl_beta)
        self.replay_weight = float(replay_weight)
        self.cfg_scale = float(cfg_scale)
        self.geometric_guard = bool(geometric_guard)
        self.topology_only_policy = bool(topology_only_policy)
        self.seed = int(seed)

    def build_curriculum_sampler(
        self,
        diagnosis: Dict[str, Any],
        *,
        prompts_per_step: int = 4,
    ) -> Callable[[int, np.random.Generator], Tuple[np.ndarray, np.ndarray]]:
        raw_grid = np.asarray(diagnosis["targets"], dtype=np.float64)
        probs = np.asarray(diagnosis["curriculum_sampling_probs"], dtype=np.float64)
        probs = probs / np.sum(probs)
        num_props = self.generator.num_props

        if raw_grid.ndim == 2:
            k_diag = int(raw_grid.shape[1])
            act_cols = (
                list(self.active_channel_indices)
                if self.active_channel_indices is not None
                else list(range(min(k_diag, num_props)))
            )
            jitters = np.array(
                [
                    (float(np.std(raw_grid[:, j])) * 0.12 if len(raw_grid) > 1 else 0.05)
                    for j in range(len(act_cols))
                ],
                dtype=np.float64,
            )

            def _sampler_nd(step_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
                del step_idx
                chosen = rng.choice(len(raw_grid), size=prompts_per_step, replace=True, p=probs)
                vals = raw_grid[chosen, : len(act_cols)] + rng.uniform(
                    -jitters[None, :], jitters[None, :], size=(prompts_per_step, len(act_cols))
                )
                t_mat = np.zeros((prompts_per_step, num_props), dtype=np.float32)
                m_act = np.zeros((prompts_per_step, num_props), dtype=bool)
                for loc_j, col_j in enumerate(act_cols):
                    t_mat[:, col_j] = vals[:, loc_j].astype(np.float32)
                    m_act[:, col_j] = True
                for anc_col, anc_val in self.anchor_channel_values.items():
                    if 0 <= int(anc_col) < num_props:
                        t_mat[:, int(anc_col)] = float(anc_val)
                        m_act[:, int(anc_col)] = True
                return t_mat, m_act

            return _sampler_nd

        grid_targets = raw_grid.reshape(-1)
        c_idx = self.prop_channel_idx
        step_jitter = float(np.median(np.diff(grid_targets))) * 0.35 if len(grid_targets) > 1 else 0.05

        def _sampler(step_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
            del step_idx
            chosen = rng.choice(len(grid_targets), size=prompts_per_step, replace=True, p=probs)
            vals = grid_targets[chosen] + rng.uniform(-step_jitter, step_jitter, size=prompts_per_step)
            t_mat = np.zeros((prompts_per_step, num_props), dtype=np.float32)
            t_mat[:, c_idx] = vals.astype(np.float32)
            m_act = np.zeros((prompts_per_step, num_props), dtype=bool)
            m_act[:, c_idx] = True
            for anc_col, anc_val in self.anchor_channel_values.items():
                if 0 <= int(anc_col) < num_props:
                    t_mat[:, int(anc_col)] = float(anc_val)
                    m_act[:, int(anc_col)] = True
            return t_mat, m_act

        return _sampler

    def run_inner_cycle(
        self,
        diagnosis: Dict[str, Any],
        *,
        replay_mols: Sequence[Chem.Mol],
        replay_targets: np.ndarray,
        steps: int = 32,
        prompts_per_step: int = 4,
        group_size: int = 8,
    ) -> Dict[str, Any]:
        """Run capability-adaptive Inner-RSI (`GRPO` or `Loom-AWR` via `alignment_mode='auto'`) and return summary."""
        backup_state = {k: v.detach().cpu().clone() for k, v in self.generator.state_dict().items()}
        if hasattr(self.generator, "configure_trainable"):
            try:
                self.generator.configure_trainable(train_conditioner=True, train_lora=True)
            except TypeError:
                self.generator.configure_trainable()
        else:
            self.generator.requires_grad_(True)

        trainer = LoomGRPOTrainer(
            generator=self.generator,
            critic_bus=self.critic_bus,
            replay_molecules=replay_mols,
            replay_targets_eV=replay_targets,
            alignment_mode=self.alignment_mode,
            lr=self.lr,
            kl_beta=self.kl_beta,
            replay_weight=self.replay_weight,
            sharp_concentration=True,
            cfg_scale=self.cfg_scale,
            geometric_guard=self.geometric_guard,
            topology_only_policy=self.topology_only_policy,
            seed=self.seed,
        )
        sampler = self.build_curriculum_sampler(diagnosis, prompts_per_step=prompts_per_step)
        history = trainer.train_loop(
            total_steps=steps,
            prompt_sampler_fn=sampler,
            group_size=group_size,
            log_every=max(steps // 4, 1),
        )
        return {
            "steps": steps,
            "groups_completed": steps * prompts_per_step,
            "history": history,
            "backup_state_dict": backup_state,
        }

    def acceptance_gate(
        self,
        pre_eval: Dict[str, Any],
        post_eval: Dict[str, Any],
        backup_state_dict: Dict[str, torch.Tensor],
        *,
        min_clean_rate_retention: float = -0.02,
    ) -> Dict[str, Any]:
        """Promote the Inner-RSI checkpoint iff constrained clean hit yield improves without structural collapse."""
        pre_hit = float(pre_eval["constrained_clean_hit_yield"])
        post_hit = float(post_eval["constrained_clean_hit_yield"])
        pre_weak_hit = float(pre_eval.get("weak_zone_hit_yield", pre_hit))
        post_weak_hit = float(post_eval.get("weak_zone_hit_yield", post_hit))
        pre_legal = float(pre_eval["legal_rate"])
        post_legal = float(post_eval["legal_rate"])

        improved = (post_weak_hit > pre_weak_hit + 1e-4) or (post_hit > pre_hit + 1e-4)
        safe_legality = (post_legal - pre_legal) >= min_clean_rate_retention
        accepted = bool(improved and safe_legality)

        if not accepted:
            device = next(self.generator.parameters()).device
            self.generator.load_state_dict({k: v.to(device) for k, v in backup_state_dict.items()}, strict=True)

        return {
            "accepted": accepted,
            "pre_constrained_clean_hit_yield": pre_hit,
            "post_constrained_clean_hit_yield": post_hit,
            "pre_weak_zone_hit_yield": pre_weak_hit,
            "post_weak_zone_hit_yield": post_weak_hit,
            "pre_legal_rate": pre_legal,
            "post_legal_rate": post_legal,
        }
