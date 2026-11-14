"""Disentangled Multi-Channel Online GRPO Trainer for `Uni-Loom`.

Combines:
  1. Channel 1 (`adv_valid`): 3D covalent valence validity + seen formal-charge preference.
  2. Channel 2 (`adv_phys`): Zero-step MLIP force/strain internalization via `PotentialSlot` (`DPA-4` or `HyFF`).
  3. Channel 3 (`adv_prop`): Sharp Gaussian + long-range Cauchy property distance via `RewardOracleSlot`.
  4. Multi-prompt batched rollouts (`prompts_per_step * group_size`) with strict per-group normalization,
     reference KL anchor, and supervised replay anchor to prevent structural drift.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple
import numpy as np
import torch
from rdkit import Chem

from ..generator.octree_alignment import categorical_policy_loss
from ..generator.octree_backend import OctreeDataset, trajectory_to_instance
from .slots import GeneratorSlot, PotentialSlot, RewardOracleSlot, assert_not_audit_oracle


def formal_charge_signature(mol: Chem.Mol) -> Tuple[Tuple[str, int], ...]:
    """Canonical formal-charge multiset signature of nonzero-charged atoms."""
    nonzero = sorted((a.GetSymbol(), int(a.GetFormalCharge())) for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
    return tuple(nonzero)


def compute_disentangled_group_advantages(
    valid_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    phys_scores: torch.Tensor,
    eval_mask: torch.Tensor,
    prop_scores: torch.Tensor,
    *,
    w_valid: float = 0.35,
    w_phys: float = 0.25,
    w_prop: float = 1.00,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute strictly disentangled group-normalized advantages for a single prompt group `(G,)`."""
    g = valid_scores.shape[0]
    device = valid_scores.device

    v_std = valid_scores.std(unbiased=False)
    if float(v_std.item()) > 1e-6:
        adv_valid = (valid_scores - valid_scores.mean()) / v_std.clamp_min(1e-6)
    else:
        adv_valid = torch.zeros(g, device=device, dtype=torch.float32)

    adv_phys = torch.zeros(g, device=device, dtype=torch.float32)
    n_valid = int(valid_mask.sum().item())
    if w_phys > 0.0 and n_valid >= 2:
        p_vals = phys_scores[valid_mask]
        p_std = p_vals.std(unbiased=False)
        if float(p_std.item()) > 1e-6:
            adv_phys[valid_mask] = (p_vals - p_vals.mean()) / p_std.clamp_min(0.05)

    adv_prop = torch.zeros(g, device=device, dtype=torch.float32)
    n_eval = int(eval_mask.sum().item())
    if w_prop > 0.0 and n_eval >= 2:
        r_vals = prop_scores[eval_mask]
        r_std = r_vals.std(unbiased=False)
        if float(r_std.item()) > 1e-6:
            adv_prop[eval_mask] = (r_vals - r_vals.mean()) / r_std.clamp_min(0.05)

    combined = w_valid * adv_valid + w_phys * adv_phys + w_prop * adv_prop
    scale = combined.std(unbiased=False).clamp_min(1.0)
    combined = combined / scale
    return combined, {
        "adv_valid": adv_valid,
        "adv_phys": adv_phys,
        "adv_prop": adv_prop,
    }


def property_distance_reward(distance_eV: np.ndarray | torch.Tensor, tolerance_eV: float = 0.25) -> torch.Tensor:
    """Hybrid Cauchy (long-range pull) + Gaussian (sharp ±tolerance concentration) reward in `(0, 1]`."""
    d = torch.as_tensor(distance_eV, dtype=torch.float32)
    u = d / max(tolerance_eV, 1e-4)
    return 0.5 / (1.0 + u) + 0.5 * torch.exp(-0.5 * u.square())


def zero_step_physical_reward(f_rms_eV_A: np.ndarray | torch.Tensor, f_max_eV_A: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Smooth bounded zero-step MLIP force/strain reward in `(0, 1]`."""
    f_rms = torch.as_tensor(f_rms_eV_A, dtype=torch.float32)
    f_max = torch.as_tensor(f_max_eV_A, dtype=torch.float32)
    r_rms = torch.exp(-torch.clamp(f_rms - 0.15, min=0.0) / 0.80)
    r_max = torch.exp(-torch.clamp(f_max - 0.40, min=0.0) / 2.00)
    return r_rms * r_max


class LoomGRPOTrainer:
    """Multi-prompt disentangled online GRPO engine operating strictly through `Uni-Loom` slots."""

    def __init__(
        self,
        generator: GeneratorSlot,
        reward_oracle: RewardOracleSlot,
        potential_slot: Optional[PotentialSlot] = None,
        *,
        seen_charge_signatures: Optional[FrozenSet[Tuple[Tuple[str, int], ...]]] = None,
        replay_molecules: Optional[Sequence[Chem.Mol]] = None,
        replay_targets_eV: Optional[np.ndarray] = None,
        lr: float = 3e-5,
        weight_decay: float = 1e-4,
        kl_beta: float = 0.02,
        clip_eps: float = 0.20,
        ppo_epochs: int = 2,
        replay_weight: float = 0.15,
        replay_batch_size: int = 8,
        w_valid: float = 0.35,
        w_phys: float = 0.25,
        w_prop: float = 1.00,
        tolerance_eV: float = 0.25,
        cfg_scale: float = 1.0,
        xyz_cfg_scale: float = 1.0,
        temperatures: Tuple[float, float, float] = (0.85, 0.30, 0.25),
        electron_parity: Optional[int] = 0,
        seed: int = 42,
    ) -> None:
        assert_not_audit_oracle(reward_oracle)
        if potential_slot is not None:
            assert_not_audit_oracle(potential_slot)

        self.generator = generator
        self.reward_oracle = reward_oracle
        self.potential_slot = potential_slot
        self.seen_charge_signatures = seen_charge_signatures

        self.device = next(generator.parameters()).device
        self.reference = copy.deepcopy(generator).eval().requires_grad_(False) if kl_beta > 0.0 else None
        self.trainable_params = [p for p in generator.parameters() if p.requires_grad]
        if not self.trainable_params:
            raise ValueError("Generator has no trainable parameters enabled for GRPO")

        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=lr, weight_decay=weight_decay)
        self.kl_beta = float(kl_beta)
        self.clip_eps = float(clip_eps)
        self.ppo_epochs = int(ppo_epochs)
        self.replay_weight = float(replay_weight)
        self.replay_batch_size = int(replay_batch_size)
        self.w_valid = float(w_valid)
        self.w_phys = float(w_phys) if potential_slot is not None else 0.0
        self.w_prop = float(w_prop)
        self.tolerance_eV = float(tolerance_eV)
        self.cfg_scale = float(cfg_scale)
        self.xyz_cfg_scale = float(xyz_cfg_scale)
        self.temperatures = temperatures
        self.electron_parity = electron_parity
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Map generator target_cols to reward_oracle target_cols indices
        self.oracle_col_indices = [self.reward_oracle.target_cols.index(col) for col in self.generator.target_cols]

        self.replay_dataset: Optional[OctreeDataset] = None
        self.replay_targets_eV: Optional[np.ndarray] = None
        if replay_molecules is not None and replay_targets_eV is not None and self.replay_weight > 0.0:
            self.replay_dataset = OctreeDataset(list(replay_molecules), self.generator.backbone, training=True, seed=seed)
            self.replay_targets_eV = np.asarray(replay_targets_eV, dtype=np.float32)

    def step(
        self,
        prompt_targets_eV: np.ndarray,
        prompt_active_mask: np.ndarray,
        *,
        group_size: int = 8,
        step_index: int = 0,
    ) -> Dict[str, Any]:
        """Run one multi-prompt batched GRPO rollout + policy update step.

        `prompt_targets_eV`: `(P, num_props)`
        `prompt_active_mask`: `(P, num_props)` bool
        Total batch generated in parallel: `B = P * group_size`.
        """
        t0 = time.perf_counter()
        p_count = prompt_targets_eV.shape[0]
        total_b = p_count * group_size

        rep_targets_eV = np.repeat(prompt_targets_eV, group_size, axis=0)
        rep_mask_np = np.repeat(prompt_active_mask.astype(bool), group_size, axis=0)

        targets_norm = self.generator.normalize_targets(rep_targets_eV)
        active_mask = torch.as_tensor(rep_mask_np, dtype=torch.bool, device=self.device)

        # 1. Batched rollout + old/ref policy distributions
        self.generator.eval()
        trajectories = self.generator.sample(
            targets_norm,
            active_mask,
            cfg_scale=self.cfg_scale,
            xyz_cfg_scale=self.xyz_cfg_scale,
            temperatures=self.temperatures,
            seed=self.seed * 100003 + step_index * 997,
            use_bf16=(self.device.type == "cuda"),
            electron_parity=self.electron_parity,
        )
        with torch.no_grad():
            old_packets = self.generator.trajectory_distributions(
                trajectories,
                targets_norm,
                active_mask,
                temperatures=self.temperatures,
                cfg_scale=self.cfg_scale,
                xyz_cfg_scale=self.xyz_cfg_scale,
                electron_parity=self.electron_parity,
            )
            ref_packets = (
                self.reference.trajectory_distributions(
                    trajectories,
                    targets_norm,
                    active_mask,
                    temperatures=self.temperatures,
                    cfg_scale=self.cfg_scale,
                    xyz_cfg_scale=self.xyz_cfg_scale,
                    electron_parity=self.electron_parity,
                )
                if self.reference is not None
                else None
            )

        # 2. Chemical & formal-charge admission
        allowed_elems = tuple(self.generator.args.allow_atoms.split(","))
        mols: List[Optional[Chem.Mol]] = [None] * total_b
        valid_mask_np = np.zeros(total_b, dtype=bool)
        seen_charge_np = np.ones(total_b, dtype=bool)
        valid_scores_np = np.full(total_b, -1.0, dtype=np.float32)

        valid_indices: List[int] = []
        valid_mols: List[Chem.Mol] = []
        for idx, traj in enumerate(trajectories):
            try:
                inst = trajectory_to_instance(traj, element_table=allowed_elems)
                mol = inst.graph
                mols[idx] = mol
                valid_mask_np[idx] = True
                valid_indices.append(idx)
                valid_mols.append(mol)
                if self.seen_charge_signatures is not None:
                    sig = formal_charge_signature(mol)
                    is_seen = sig in self.seen_charge_signatures
                    seen_charge_np[idx] = is_seen
                    valid_scores_np[idx] = 1.0 if is_seen else 0.25
                else:
                    valid_scores_np[idx] = 1.0
            except ValueError:
                pass

        # 3. Batched PotentialSlot zero-step force evaluation
        phys_scores_np = np.zeros(total_b, dtype=np.float32)
        f_max_np = np.full(total_b, np.nan, dtype=np.float64)
        f_rms_np = np.full(total_b, np.nan, dtype=np.float64)
        if self.potential_slot is not None and valid_mols:
            phys_out = self.potential_slot.evaluate_molecules(valid_mols, chunk_size=32)
            r_phys_valid = zero_step_physical_reward(phys_out["f_rms_eV_A"], phys_out["f_max_eV_A"]).numpy()
            for local_i, global_i in enumerate(valid_indices):
                f_max_np[global_i] = phys_out["f_max_eV_A"][local_i]
                f_rms_np[global_i] = phys_out["f_rms_eV_A"][local_i]
                if np.isfinite(r_phys_valid[local_i]):
                    phys_scores_np[global_i] = r_phys_valid[local_i]

        # 4. Batched RewardOracleSlot evaluation
        preds_all, _, eval_mask_np, _ = self.reward_oracle.evaluate_candidates(mols, batch_size=32)
        preds_aligned = preds_all[:, self.oracle_col_indices]  # (B, num_props)

        prop_scores_np = np.zeros(total_b, dtype=np.float32)
        abs_err_np = np.full(total_b, np.nan, dtype=np.float64)
        hits_np = np.zeros(total_b, dtype=bool)
        for idx in range(total_b):
            if eval_mask_np[idx]:
                act = rep_mask_np[idx]
                if act.any():
                    diffs = np.abs(preds_aligned[idx, act] - rep_targets_eV[idx, act])
                    mean_err = float(np.mean(diffs))
                    max_err = float(np.max(diffs))
                    abs_err_np[idx] = mean_err
                    hits_np[idx] = max_err <= self.tolerance_eV
                    prop_scores_np[idx] = float(property_distance_reward(mean_err, self.tolerance_eV).item())

        # 5. Per-prompt disentangled group-normalized advantages
        valid_scores_t = torch.as_tensor(valid_scores_np, device=self.device)
        valid_mask_t = torch.as_tensor(valid_mask_np, device=self.device)
        phys_scores_t = torch.as_tensor(phys_scores_np, device=self.device)
        eval_mask_t = torch.as_tensor(eval_mask_np, device=self.device)
        prop_scores_t = torch.as_tensor(prop_scores_np, device=self.device)

        adv_all = torch.zeros(total_b, device=self.device, dtype=torch.float32)
        for p_idx in range(p_count):
            s = p_idx * group_size
            e = s + group_size
            adv_g, _ = compute_disentangled_group_advantages(
                valid_scores_t[s:e],
                valid_mask_t[s:e],
                phys_scores_t[s:e],
                eval_mask_t[s:e],
                prop_scores_t[s:e],
                w_valid=self.w_valid,
                w_phys=self.w_phys,
                w_prop=self.w_prop,
            )
            adv_all[s:e] = adv_g

        # 6. Clipped PPO/GRPO policy updates + KL + Supervised Replay Anchor
        self.generator.train()
        has_signal = bool((adv_all.abs() > 1e-6).any().item()) or self.kl_beta > 0.0 or self.replay_dataset is not None
        n_updates = self.ppo_epochs if has_signal else 0
        last_loss = 0.0
        last_policy_loss = 0.0
        last_replay_loss = 0.0
        last_kl = 0.0
        grad_norm = 0.0

        autocast_train = self.device.type == "cuda"
        for _ in range(n_updates):
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=autocast_train):
                cur_packets = self.generator.trajectory_distributions(
                    trajectories,
                    targets_norm,
                    active_mask,
                    temperatures=self.temperatures,
                    cfg_scale=self.cfg_scale,
                    xyz_cfg_scale=self.xyz_cfg_scale,
                    electron_parity=self.electron_parity,
                )
                p_loss = categorical_policy_loss(
                    cur_packets, old_packets, ref_packets, adv_all, beta=self.kl_beta, clip=self.clip_eps
                )
                total_loss = p_loss

                if ref_packets is not None:
                    with torch.no_grad():
                        kl_terms = []
                        for (logp, _, _), (ref_logp, _, _) in zip(cur_packets, ref_packets):
                            delta = torch.where(torch.isfinite(logp), logp - ref_logp, torch.zeros_like(logp))
                            kl_terms.append((logp.exp() * delta).sum(-1))
                        last_kl = float(torch.cat(kl_terms).mean().item())

                if self.replay_dataset is not None and self.replay_targets_eV is not None:
                    r_indices = self.rng.integers(0, len(self.replay_dataset), size=self.replay_batch_size)
                    r_packed = self.replay_dataset.collate(r_indices)
                    r_targets_norm = self.generator.normalize_targets(self.replay_targets_eV[r_indices])
                    # Match active mask distribution of current prompt batch
                    r_mask_idx = self.rng.integers(0, p_count, size=self.replay_batch_size)
                    r_mask = torch.as_tensor(prompt_active_mask[r_mask_idx], dtype=torch.bool, device=self.device)
                    r_loss, _ = self.generator.compute_sft_loss(r_packed, r_targets_norm, r_mask, cfg_dropout_prob=0.10)
                    total_loss = total_loss + self.replay_weight * r_loss
                    last_replay_loss = float(r_loss.detach().item())

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0).item())
            self.optimizer.step()
            last_loss = float(total_loss.detach().item())
            last_policy_loss = float(p_loss.detach().item())

        valid_f = f_max_np[valid_mask_np & np.isfinite(f_max_np)]
        valid_err = abs_err_np[eval_mask_np & np.isfinite(abs_err_np)]
        elapsed = time.perf_counter() - t0
        return {
            "step": step_index + 1,
            "prompts": p_count,
            "group_size": group_size,
            "proposals": total_b,
            "valid": int(valid_mask_np.sum()),
            "valid_rate": float(valid_mask_np.mean()),
            "evaluated": int(eval_mask_np.sum()),
            "hits": int(hits_np.sum()),
            "hit_yield": float(hits_np.mean()),
            "unseen_charge_rate": float(1.0 - seen_charge_np[valid_mask_np].mean()) if valid_mask_np.any() else 0.0,
            "conditional_mae_eV": float(np.mean(valid_err)) if len(valid_err) > 0 else float("nan"),
            "f_max_median_eV_A": float(np.median(valid_f)) if len(valid_f) > 0 else float("nan"),
            "prop_reward_mean": float(prop_scores_np[eval_mask_np].mean()) if eval_mask_np.any() else 0.0,
            "phys_reward_mean": float(phys_scores_np[valid_mask_np].mean()) if valid_mask_np.any() else 0.0,
            "loss_total": last_loss,
            "loss_policy": last_policy_loss,
            "loss_replay": last_replay_loss,
            "kl": last_kl,
            "grad_norm": grad_norm,
            "wall_seconds": elapsed,
        }

    def train_loop(
        self,
        total_steps: int,
        prompt_sampler_fn: Callable[[int, np.random.Generator], Tuple[np.ndarray, np.ndarray]],
        *,
        group_size: int = 8,
        checkpoint_fn: Optional[Callable[[int, GeneratorSlot, List[Dict[str, Any]]], None]] = None,
        log_every: int = 5,
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for s_idx in range(total_steps):
            p_targets, p_mask = prompt_sampler_fn(s_idx, self.rng)
            metrics = self.step(p_targets, p_mask, group_size=group_size, step_index=s_idx)
            history.append(metrics)
            if (s_idx + 1) % log_every == 0 or s_idx == 0 or (s_idx + 1) == total_steps:
                print(
                    f"[Loom-GRPO {metrics['step']:04d}/{total_steps:04d}] "
                    f"hits={metrics['hits']}/{metrics['proposals']} ({metrics['hit_yield']*100:.1f}%) "
                    f"MAE={metrics['conditional_mae_eV']:.3f}eV "
                    f"valid={metrics['valid_rate']*100:.1f}% "
                    f"Fmax={metrics['f_max_median_eV_A']:.2f} "
                    f"KL={metrics['kl']:.4f} ({metrics['wall_seconds']:.1f}s)",
                    flush=True,
                )
            if checkpoint_fn is not None:
                checkpoint_fn(s_idx + 1, self.generator, history)
        return history
