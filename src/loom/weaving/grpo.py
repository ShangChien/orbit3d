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
import math
import time
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple
import numpy as np
import torch
from rdkit import Chem

from ..chemistry.valence_reconciler import (
    ALLOWED_DIPOLAR_PAIRS,
    formal_charge_signature,
    is_clean_organic_charge_state,
)
from ..generator.octree_alignment import categorical_policy_loss
from ..generator.octree_backend import trajectory_to_instance
from .slots import GeneratorSlot, PotentialSlot, RewardOracleSlot, assert_not_audit_oracle


def compute_disentangled_group_advantages(
    valid_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    phys_scores: torch.Tensor,
    eval_mask: torch.Tensor,
    prop_scores: torch.Tensor,
    *,
    w_valid: float = 0.75,
    w_phys: float = 0.25,
    w_prop: float = 1.00,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute strictly disentangled, monotonic group-normalized advantages for a prompt group `(G,)`.

    Enforces strict monotonic hierarchy within each group:
      `Clean Low-Strain Hit > Clean Miss > Exotic Zwitterion > Invalid Graph`
    """
    g = valid_scores.shape[0]
    device = valid_scores.device

    v_std = valid_scores.std(unbiased=False)
    if float(v_std.item()) > 1e-6:
        adv_valid = (valid_scores - valid_scores.mean()) / v_std.clamp_min(1e-6)
    else:
        adv_valid = torch.zeros(g, device=device, dtype=torch.float32)

    adv_phys = torch.zeros(g, device=device, dtype=torch.float32)
    n_valid = int(valid_mask.sum().item())
    if w_phys > 0.0 and n_valid >= 1:
        if n_valid >= 2:
            p_vals = phys_scores[valid_mask]
            p_std = p_vals.std(unbiased=False)
            if float(p_std.item()) > 1e-6:
                adv_phys[valid_mask] = ((p_vals - p_vals.mean()) / p_std.clamp_min(0.05)).clamp(-1.5, 2.0)
        else:
            adv_phys[valid_mask] = phys_scores[valid_mask].clamp(0.0, 1.0)
        if n_valid < g:
            adv_phys[~valid_mask] = -0.65
            adv_phys = adv_phys - adv_phys.mean()

    adv_prop = torch.zeros(g, device=device, dtype=torch.float32)
    n_eval = int(eval_mask.sum().item())
    if w_prop > 0.0 and n_eval >= 1:
        if n_eval >= 2:
            r_vals = prop_scores[eval_mask]
            r_std = r_vals.std(unbiased=False)
            if float(r_std.item()) > 1e-6:
                adv_prop[eval_mask] = ((r_vals - r_vals.mean()) / r_std.clamp_min(0.05)).clamp(-1.35, 2.0)
        else:
            adv_prop[eval_mask] = prop_scores[eval_mask].clamp(0.0, 1.0)
        if n_eval < g:
            adv_prop[~eval_mask] = -0.85
            adv_prop = adv_prop - adv_prop.mean()

    combined = w_valid * adv_valid + w_phys * adv_phys + w_prop * adv_prop
    combined = combined - combined.mean()
    scale = combined.std(unbiased=False).clamp_min(0.20)
    combined = combined / scale
    return combined, {
        "adv_valid": adv_valid,
        "adv_phys": adv_phys,
        "adv_prop": adv_prop,
    }


def compute_vine_counterfactual_advantages(
    group_adv: torch.Tensor,
    m_trunks: int,
    k_leaves: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose flat `(P * m_trunks * k_leaves,)` advantages into `(adv_tree, adv_leaf)` for Tree-Vine GRPO."""
    total_b = int(group_adv.shape[0])
    g = int(m_trunks * k_leaves)
    if m_trunks <= 1 or k_leaves <= 1 or total_b % g != 0:
        return group_adv, group_adv

    p_count = total_b // g
    adv_tree = torch.zeros_like(group_adv)
    adv_leaf = torch.zeros_like(group_adv)
    for p_idx in range(p_count):
        s = p_idx * g
        e = s + g
        gp = group_adv[s:e]
        mat = gp.view(m_trunks, k_leaves)
        t_mean = mat.mean(dim=1)
        t_std = t_mean.std(unbiased=False)
        if float(t_std.item()) > 1e-5:
            t_norm = ((t_mean - t_mean.mean()) / t_std.clamp_min(0.10)).clamp(-2.0, 2.0)
        else:
            t_norm = torch.zeros_like(t_mean)
        adv_tree[s:e] = t_norm.repeat_interleave(k_leaves)

        intra = mat - t_mean.unsqueeze(1)
        intra_std = intra.reshape(-1).std(unbiased=False)
        if float(intra_std.item()) > 1e-5:
            intra_norm = (intra.reshape(-1) / intra_std.clamp_min(0.10)).clamp(-2.0, 2.0)
            leaf_p = 0.60 * gp + 0.40 * intra_norm
        else:
            leaf_p = gp
        adv_leaf[s:e] = leaf_p
    return adv_tree, adv_leaf


def property_distance_reward(
    distance_eV: np.ndarray | torch.Tensor,
    tolerance_eV: float = 0.25,
    *,
    sharp_concentration: bool = False,
) -> torch.Tensor:
    """Hybrid Cauchy (long-range pull) + Gaussian (concentration) + smooth hit-window bonus in `(0, 1]`."""
    d = torch.as_tensor(distance_eV, dtype=torch.float32)
    tol = max(tolerance_eV, 1e-4)
    u = d / tol
    if sharp_concentration:
        # Tighter Gaussian core (`sigma = 0.65 * tol`) + steeper window edge to compress `sigma_in`
        base = 0.30 / (1.0 + u.square()) + 0.40 * torch.exp(-0.5 * (d / (0.65 * tol)).square())
        hit_bonus = 0.30 * torch.sigmoid((tol - d) / (0.12 * tol))
        return base + hit_bonus
    base = 0.40 / (1.0 + u) + 0.35 * torch.exp(-0.5 * u.square())
    hit_bonus = 0.25 * torch.sigmoid((tol - d) / (0.16 * tol))
    return base + hit_bonus


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
        reward_oracle: Optional[RewardOracleSlot] = None,
        potential_slot: Optional[PotentialSlot] = None,
        *,
        critic_bus: Optional[Any] = None,
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
        w_valid: float = 0.75,
        w_phys: float = 0.25,
        w_prop: float = 1.00,
        tolerance_eV: float = 0.25,
        sharp_concentration: bool = False,
        cfg_scale: float = 1.0,
        xyz_cfg_scale: float = 1.0,
        cfg_tree_decay: float = 0.0,
        geometric_guard: bool = False,
        topology_only_policy: bool = False,
        relax_distill_weight: float = 0.0,
        relax_distill_steps: int = 30,
        relax_distill_max_mols: int = 12,
        relax_distill_max_fmax: float = 1.5,
        temperatures: Tuple[float, float, float] = (0.85, 0.30, 0.25),
        electron_parity: Optional[int] = 0,
        size_anchored_cfg: bool = False,
        size_col_name: str = "n_atoms",
        cfg_anchor_cols: Sequence[str] = (),
        size_adherence_weight: float = 0.0,
        size_tolerance_atoms: float = 4.0,
        size_tolerance_frac: float = 0.12,
        alignment_mode: str = "auto",
        awr_temperature: float = 0.45,
        awr_max_weight: float = 6.0,
        awr_repeat_k: int = 2,
        awr_hindsight_fallback: bool = True,
        awr_require_native_legal: bool = False,
        awr_relax_steps: int = 0,
        awr_max_fmax: float = 2.5,
        grad_clip_mode: str = "norm",
        use_vine_rollouts: bool = False,
        m_trunks: int = 4,
        k_leaves: int = 2,
        post_tree_refiner: Optional[Any] = None,
        polish_hydrogens_mlip: bool = True,
        seed: int = 42,
    ) -> None:
        if reward_oracle is None and critic_bus is None:
            raise ValueError("LoomGRPOTrainer requires either `reward_oracle` or `critic_bus`")
        if reward_oracle is not None:
            assert_not_audit_oracle(reward_oracle)
        if potential_slot is None and critic_bus is not None:
            potential_slot = getattr(critic_bus, "potential_slot", None)
        if potential_slot is not None:
            assert_not_audit_oracle(potential_slot)

        self.generator = generator
        self.reward_oracle = reward_oracle
        self.critic_bus = critic_bus
        self.potential_slot = potential_slot
        self.seen_charge_signatures = seen_charge_signatures

        mode = str(alignment_mode).strip().lower()
        if mode == "auto":
            mode = "grpo" if bool(getattr(generator, "supports_exact_trajectory_logprob", True)) else "awr_sft"
        if mode not in ("grpo", "awr_sft"):
            raise ValueError(f"Unsupported alignment_mode '{alignment_mode}'. Expected 'auto', 'grpo', or 'awr_sft'.")
        self.alignment_mode = mode
        if self.alignment_mode == "grpo" and not bool(getattr(generator, "supports_exact_trajectory_logprob", True)):
            raise ValueError(f"{type(generator).__name__} does not support exact trajectory log-probs for 'grpo'.")
        self.awr_temperature = max(float(awr_temperature), 1e-4)
        self.awr_max_weight = max(float(awr_max_weight), 1.0)
        self.awr_repeat_k = max(1, int(awr_repeat_k))
        self.awr_hindsight_fallback = bool(awr_hindsight_fallback)
        self.awr_require_native_legal = bool(awr_require_native_legal)
        self.awr_relax_steps = max(0, int(awr_relax_steps))
        self.awr_max_fmax = float(awr_max_fmax)
        self.grad_clip_mode = str(grad_clip_mode).strip().lower()
        self.use_vine_rollouts = bool(use_vine_rollouts)
        self.m_trunks = max(1, int(m_trunks))
        self.k_leaves = max(1, int(k_leaves))
        self.post_tree_refiner = post_tree_refiner
        self.polish_hydrogens_mlip = bool(polish_hydrogens_mlip)

        self.device = next(generator.parameters()).device
        use_ref = (self.alignment_mode == "grpo") and (kl_beta > 0.0)
        self.reference = copy.deepcopy(generator).eval().requires_grad_(False) if use_ref else None
        self.trainable_params = [p for p in generator.parameters() if p.requires_grad]
        if not self.trainable_params:
            raise ValueError("Generator has no trainable parameters enabled for alignment")

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
        self.sharp_concentration = bool(sharp_concentration)
        self.cfg_scale = float(cfg_scale)
        self.xyz_cfg_scale = float(xyz_cfg_scale)
        self.cfg_tree_decay = float(cfg_tree_decay)
        self.geometric_guard = bool(geometric_guard)
        self.topology_only_policy = bool(topology_only_policy)
        self.relax_distill_weight = float(relax_distill_weight) if potential_slot is not None else 0.0
        self.relax_distill_steps = int(relax_distill_steps)
        self.relax_distill_max_mols = int(relax_distill_max_mols)
        self.relax_distill_max_fmax = float(relax_distill_max_fmax)
        self.temperatures = temperatures
        self.electron_parity = electron_parity
        self.size_anchored_cfg = bool(size_anchored_cfg)
        self.size_col_name = str(size_col_name)
        self.cfg_anchor_cols = tuple(str(c) for c in cfg_anchor_cols)
        self.size_adherence_weight = float(size_adherence_weight)
        self.size_tolerance_atoms = float(size_tolerance_atoms)
        self.size_tolerance_frac = float(size_tolerance_frac)
        gen_cols = list(getattr(self.generator, "target_cols", []))
        _atom_size_aliases = {"n_atoms", "num_atoms", "natoms"}
        self.size_col_idx: Optional[int] = (
            gen_cols.index(self.size_col_name)
            if (self.size_col_name in gen_cols and self.size_col_name in _atom_size_aliases)
            else None
        )
        anchor_idxs: List[int] = [gen_cols.index(c) for c in self.cfg_anchor_cols if c in gen_cols]
        if self.size_anchored_cfg and self.size_col_name in gen_cols:
            sc_i = gen_cols.index(self.size_col_name)
            if sc_i not in anchor_idxs:
                anchor_idxs.append(sc_i)
        self.cfg_anchor_indices: Tuple[int, ...] = tuple(anchor_idxs)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Map generator target_cols to reward_oracle target_cols indices when using legacy reward_oracle
        self.oracle_col_indices = (
            [self.reward_oracle.target_cols.index(col) for col in self.generator.target_cols]
            if self.reward_oracle is not None
            else list(range(len(self.generator.target_cols)))
        )

        self.replay_dataset: Optional[Any] = None
        self.replay_targets_eV: Optional[np.ndarray] = None
        if replay_molecules is not None and replay_targets_eV is not None and self.replay_weight > 0.0:
            self.replay_dataset = self.generator.build_replay_dataset(list(replay_molecules), seed=seed)
            self.replay_targets_eV = np.asarray(replay_targets_eV, dtype=np.float32)

    def _build_uncond_active_mask(self, active_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Return CFG unconditional mask keeping `cfg_anchor_indices` active (decoupled from atom-size target)."""
        if not self.cfg_anchor_indices:
            return None
        uncond_mask = torch.zeros_like(active_mask, dtype=torch.bool)
        for a_idx in self.cfg_anchor_indices:
            uncond_mask[:, a_idx] = active_mask[:, a_idx]
        return uncond_mask

    def _sample_replay_active_mask(self, prompt_active_mask: np.ndarray) -> torch.Tensor:
        """Sample supervised replay masks (`75% full / 15% anchor-only / 10% uncond` when CFG-anchored)."""
        r_mask_idx = self.rng.integers(0, prompt_active_mask.shape[0], size=self.replay_batch_size)
        r_mask_np = prompt_active_mask[r_mask_idx].astype(bool).copy()
        if self.cfg_anchor_indices:
            u = self.rng.random(size=self.replay_batch_size)
            anchor_only = (u >= 0.75) & (u < 0.90)
            full_drop = u >= 0.90
            if anchor_only.any():
                keep_cols = {a_idx: r_mask_np[anchor_only, a_idx].copy() for a_idx in self.cfg_anchor_indices}
                r_mask_np[anchor_only] = False
                for a_idx, vals in keep_cols.items():
                    r_mask_np[anchor_only, a_idx] = vals
            if full_drop.any():
                r_mask_np[full_drop] = False
        return torch.as_tensor(r_mask_np, dtype=torch.bool, device=self.device)

    def _compute_size_metrics_and_advantages(
        self,
        mols: Sequence[Optional[Chem.Mol]],
        valid_mask_np: np.ndarray,
        rep_targets_eV: np.ndarray,
        rep_mask_np: np.ndarray,
        group_size: int,
        prompt_atom_budgets: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """Compute atom counts, absolute size errors, adherence scores, and normalized group advantages."""
        total_b = len(mols)
        natoms_actual = np.full(total_b, np.nan, dtype=np.float64)
        natoms_abs_err = np.full(total_b, np.nan, dtype=np.float64)
        size_scores = np.zeros(total_b, dtype=np.float32)
        size_eval_mask = np.zeros(total_b, dtype=bool)
        adv_size = torch.zeros(total_b, device=self.device, dtype=torch.float32)

        for idx, mol in enumerate(mols):
            if mol is not None and valid_mask_np[idx]:
                natoms_actual[idx] = float(mol.GetNumAtoms())

        if prompt_atom_budgets is not None:
            rep_b = np.repeat(np.asarray(prompt_atom_budgets, dtype=np.float64).reshape(-1), group_size)
            size_active = rep_b > 0.0
            target_nat = rep_b
        elif self.size_col_idx is not None:
            size_active = rep_mask_np[:, self.size_col_idx]
            target_nat = rep_targets_eV[:, self.size_col_idx]
        else:
            return natoms_actual, natoms_abs_err, size_scores, size_eval_mask, adv_size
        for idx in range(total_b):
            if valid_mask_np[idx] and size_active[idx] and np.isfinite(natoms_actual[idx]):
                n_req = float(target_nat[idx])
                err = abs(float(natoms_actual[idx]) - n_req)
                tol = max(self.size_tolerance_atoms, self.size_tolerance_frac * max(n_req, 1.0))
                u = err / tol
                score = 0.50 * math.exp(-0.5 * u * u) + 0.50 / (1.0 + u * u) - 0.15 * max(0.0, u - 1.0)
                natoms_abs_err[idx] = err
                size_scores[idx] = float(np.clip(score, -1.5, 1.0))
                size_eval_mask[idx] = True

        if self.size_adherence_weight <= 0.0:
            return natoms_actual, natoms_abs_err, size_scores, size_eval_mask, adv_size

        p_count = total_b // group_size
        for p_idx in range(p_count):
            s = p_idx * group_size
            e = s + group_size
            if not size_active[s:e].any():
                continue
            ok = torch.as_tensor(size_eval_mask[s:e], dtype=torch.bool, device=self.device)
            n_ok = int(ok.sum().item())
            if n_ok == 0:
                continue
            g_adv = torch.zeros(group_size, device=self.device, dtype=torch.float32)
            scores_t = torch.as_tensor(size_scores[s:e], dtype=torch.float32, device=self.device)
            if n_ok >= 2:
                vals = scores_t[ok]
                std = vals.std(unbiased=False)
                if float(std.item()) > 1e-6:
                    g_adv[ok] = ((vals - vals.mean()) / std.clamp_min(0.05)).clamp(-1.5, 2.0)
            else:
                g_adv[ok] = scores_t[ok].clamp(-1.0, 1.0)
            if n_ok < group_size:
                inv_act = (~ok) & torch.as_tensor(size_active[s:e], dtype=torch.bool, device=self.device)
                g_adv[inv_act] = -0.85
            g_adv = g_adv - g_adv.mean()
            adv_size[s:e] = g_adv

        return natoms_actual, natoms_abs_err, size_scores, size_eval_mask, adv_size

    def step(
        self,
        prompt_targets_eV: np.ndarray,
        prompt_active_mask: np.ndarray,
        *,
        group_size: int = 8,
        step_index: int = 0,
        prompt_atom_budgets: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run one multi-prompt batched GRPO rollout + policy update step.

        `prompt_targets_eV`: `(P, num_props)`
        `prompt_active_mask`: `(P, num_props)` bool
        Total batch generated in parallel: `B = P * group_size`.
        """
        t0 = time.perf_counter()
        p_count = prompt_targets_eV.shape[0]
        if self.use_vine_rollouts:
            group_size = self.m_trunks * self.k_leaves
        total_b = p_count * group_size

        rep_targets_eV = np.repeat(prompt_targets_eV, group_size, axis=0)
        rep_mask_np = np.repeat(prompt_active_mask.astype(bool), group_size, axis=0)

        targets_norm = self.generator.normalize_targets(rep_targets_eV)
        active_mask = torch.as_tensor(rep_mask_np, dtype=torch.bool, device=self.device)

        # 1. Batched rollout + old/ref policy distributions
        self.generator.eval()
        uncond_mask = self._build_uncond_active_mask(active_mask)
        cfg_extra = {"uncond_active_mask": uncond_mask} if uncond_mask is not None else {}
        step_seed = self.seed * 100003 + step_index * 997

        if self.use_vine_rollouts and hasattr(self.generator, "sample_vine_rollouts"):
            p_targets_norm = self.generator.normalize_targets(prompt_targets_eV)
            p_mask_t = torch.as_tensor(prompt_active_mask.astype(bool), dtype=torch.bool, device=self.device)
            p_uncond = self._build_uncond_active_mask(p_mask_t)
            p_budgets_t = (
                torch.as_tensor(prompt_atom_budgets, dtype=torch.float32, device=self.device)
                if prompt_atom_budgets is not None
                else None
            )
            trajectories = self.generator.sample_vine_rollouts(
                p_targets_norm,
                p_mask_t,
                m_trunks=self.m_trunks,
                k_leaves=self.k_leaves,
                cfg_scale=self.cfg_scale,
                xyz_cfg_scale=self.xyz_cfg_scale,
                cfg_tree_decay=self.cfg_tree_decay,
                geometric_guard=self.geometric_guard,
                temperatures=self.temperatures,
                seed=step_seed,
                use_bf16=(self.device.type == "cuda"),
                electron_parity=self.electron_parity,
                uncond_active_mask=p_uncond,
                leaf_atom_budget=p_budgets_t,
            )
        else:
            sample_extra = dict(cfg_extra)
            if prompt_atom_budgets is not None:
                rep_b = np.repeat(np.asarray(prompt_atom_budgets, dtype=np.float32), group_size, axis=0)
                sample_extra["leaf_atom_budget"] = torch.as_tensor(rep_b, dtype=torch.float32, device=self.device)
                sample_extra["sort_by_budget"] = False
            trajectories = self.generator.sample(
                targets_norm,
                active_mask,
                cfg_scale=self.cfg_scale,
                xyz_cfg_scale=self.xyz_cfg_scale,
                cfg_tree_decay=self.cfg_tree_decay,
                geometric_guard=self.geometric_guard,
                temperatures=self.temperatures,
                seed=step_seed,
                use_bf16=(self.device.type == "cuda"),
                electron_parity=self.electron_parity,
                **sample_extra,
            )

        if self.post_tree_refiner is not None:
            self.post_tree_refiner.refine_trajectories_batch(trajectories)
        if self.alignment_mode == "grpo":
            with torch.no_grad():
                old_packets = self.generator.trajectory_distributions(
                    trajectories,
                    targets_norm,
                    active_mask,
                    temperatures=self.temperatures,
                    cfg_scale=self.cfg_scale,
                    xyz_cfg_scale=self.xyz_cfg_scale,
                    cfg_tree_decay=self.cfg_tree_decay,
                    geometric_guard=self.geometric_guard,
                    topology_only_policy=self.topology_only_policy,
                    electron_parity=self.electron_parity,
                    **cfg_extra,
                )
                ref_packets = (
                    self.reference.trajectory_distributions(
                        trajectories,
                        targets_norm,
                        active_mask,
                        temperatures=self.temperatures,
                        cfg_scale=self.cfg_scale,
                        xyz_cfg_scale=self.xyz_cfg_scale,
                        cfg_tree_decay=self.cfg_tree_decay,
                        geometric_guard=self.geometric_guard,
                        topology_only_policy=self.topology_only_policy,
                        electron_parity=self.electron_parity,
                        **cfg_extra,
                    )
                    if self.reference is not None
                    else None
                )
        else:
            old_packets = None
            ref_packets = None

        # 2-5. Evaluation & Advantage Routing (`CompositeCriticBus` OR legacy `RewardOracleSlot`)
        constraints_met_rate = 1.0
        if self.critic_bus is not None:
            admit_out = self.critic_bus.admit_trajectories(
                trajectories, polish_hydrogens_mlip=self.polish_hydrogens_mlip
            )
            raw_legal_np = np.asarray(admit_out.get("raw_legal_mask", admit_out["legal_mask"]), dtype=bool)
            raw_clean_np = np.asarray(admit_out.get("raw_clean_charge_mask", admit_out["clean_charge_mask"]), dtype=bool)
            if self.alignment_mode == "awr_sft" and self.awr_require_native_legal:
                raw_mols_list = admit_out.get("raw_mols", admit_out["mols"])
                mols = [
                    raw_mols_list[k] if (bool(raw_legal_np[k]) and bool(raw_clean_np[k])) else None
                    for k in range(total_b)
                ]
            else:
                mols = admit_out["mols"]
            bus_obj_names = self.critic_bus.spec.objective_names
            gen_cols = list(self.generator.target_cols)
            if len(bus_obj_names) == rep_targets_eV.shape[-1] and tuple(bus_obj_names) == tuple(gen_cols):
                bus_req = rep_targets_eV
                bus_act = rep_mask_np
            else:
                bus_req = np.zeros((total_b, len(bus_obj_names)), dtype=np.float32)
                bus_act = np.zeros((total_b, len(bus_obj_names)), dtype=bool)
                for i, nm in enumerate(bus_obj_names):
                    if nm in gen_cols:
                        gi = gen_cols.index(nm)
                        bus_req[:, i] = rep_targets_eV[:, gi]
                        bus_act[:, i] = rep_mask_np[:, gi]
            ev_bundle = self.critic_bus.evaluate_molecules(
                mols,
                requested_targets=bus_req,
                active_mask=bus_act,
                include_audit=False,
                for_training=True,
                sharp_concentration=self.sharp_concentration,
                batch_size=32,
            )
            valid_mask_np = ev_bundle["legal_mask"]
            clean_charge_np = ev_bundle["clean_charge_mask"]
            seen_charge_np = clean_charge_np.copy()
            valid_scores_np = ev_bundle["valid_scores"]
            phys_scores_np = ev_bundle["phys_scores"]
            f_max_np = ev_bundle["f_max_eV_A"]
            f_rms_np = ev_bundle["f_rms_eV_A"]
            eval_mask_np = ev_bundle["evaluated_mask"]
            prop_scores_np = ev_bundle["prop_scores"]
            hits_np = ev_bundle["constrained_clean_hits"]
            abs_err_np = ev_bundle["mean_normalized_error"] * self.tolerance_eV
            if valid_mask_np.any():
                constraints_met_rate = float(ev_bundle["constraints_met"][valid_mask_np].mean())
            adv_all = self.critic_bus.compute_group_advantages(
                ev_bundle,
                group_size=group_size,
                device=self.device,
                w_valid=self.w_valid,
                w_phys=self.w_phys,
                w_prop=self.w_prop,
            )
        else:
            assert self.reward_oracle is not None
            allowed_elems = self.generator.allowed_elements
            mols = [None] * total_b
            valid_mask_np = np.zeros(total_b, dtype=bool)
            clean_charge_np = np.zeros(total_b, dtype=bool)
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
                    clean_q = is_clean_organic_charge_state(mol)
                    clean_charge_np[idx] = clean_q
                    if not clean_q:
                        valid_scores_np[idx] = -0.35
                        seen_charge_np[idx] = False
                    elif self.seen_charge_signatures is not None:
                        sig = formal_charge_signature(mol)
                        is_seen = sig in self.seen_charge_signatures
                        seen_charge_np[idx] = is_seen
                        valid_scores_np[idx] = 1.0 if is_seen else 0.75
                    else:
                        valid_scores_np[idx] = 1.0
                except ValueError:
                    pass

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

            mols_for_prop = [m if clean_charge_np[i] else None for i, m in enumerate(mols)]
            preds_all, _, eval_mask_np, _ = self.reward_oracle.evaluate_candidates(mols_for_prop, batch_size=32)
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
                        prop_scores_np[idx] = float(
                            property_distance_reward(
                                mean_err, self.tolerance_eV, sharp_concentration=self.sharp_concentration
                            ).item()
                        )

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
            raw_legal_np = valid_mask_np.copy()

        natoms_actual_np, natoms_abs_err_np, size_scores_np, size_eval_mask_np, adv_size = (
            self._compute_size_metrics_and_advantages(
                mols,
                valid_mask_np,
                rep_targets_eV,
                rep_mask_np,
                group_size,
                prompt_atom_budgets=prompt_atom_budgets,
            )
        )
        if self.size_adherence_weight > 0.0 and (self.size_col_idx is not None or prompt_atom_budgets is not None):
            adv_all = adv_all + self.size_adherence_weight * adv_size

        if self.use_vine_rollouts:
            adv_tree_all, adv_leaf_all = compute_vine_counterfactual_advantages(
                adv_all, self.m_trunks, self.k_leaves
            )
        else:
            adv_tree_all = None
            adv_leaf_all = adv_all

        # 5b. Optional v1.1 Online Relaxed-Coordinate Distillation (`L_relax_SFT`)
        relax_packed: Optional[Dict[str, torch.Tensor]] = None
        relax_targets_norm: Optional[torch.Tensor] = None
        relax_mask_t: Optional[torch.Tensor] = None
        n_relax_admitted = 0
        if self.relax_distill_weight > 0.0 and self.potential_slot is not None:
            clean_eval_idx = [
                i
                for i in range(total_b)
                if eval_mask_np[i]
                and mols[i] is not None
                and np.isfinite(abs_err_np[i])
                and float(abs_err_np[i]) <= 1.20 * self.tolerance_eV
            ]
            clean_eval_idx.sort(key=lambda i: float(abs_err_np[i]))
            cand_idx = clean_eval_idx[: self.relax_distill_max_mols]
            if cand_idx:
                cand_mols = [mols[i] for i in cand_idx if mols[i] is not None]
                rel_out = self.potential_slot.relax_molecules(
                    cand_mols, n_steps=self.relax_distill_steps, chunk_size=16
                )
                kept_mols: List[Chem.Mol] = []
                kept_masks: List[np.ndarray] = []
                kept_req_targets: List[np.ndarray] = []
                for loc_i, orig_i in enumerate(cand_idx):
                    rm = rel_out["relaxed_mols"][loc_i]
                    if (
                        rel_out["graph_preserved"][loc_i]
                        and rm is not None
                        and np.isfinite(rel_out["final_f_max_eV_A"][loc_i])
                        and rel_out["final_f_max_eV_A"][loc_i] <= self.relax_distill_max_fmax
                        and is_clean_organic_charge_state(rm)
                    ):
                        kept_mols.append(rm)
                        kept_masks.append(rep_mask_np[orig_i])
                        kept_req_targets.append(rep_targets_eV[orig_i])
                if kept_mols:
                    if self.critic_bus is not None:
                        rel_ev = self.critic_bus.evaluate_molecules(kept_mols, for_training=True, batch_size=16)
                        rel_aligned = rel_ev["objective_preds"]
                        rel_ok = rel_ev["evaluated_mask"] & rel_ev["hard_constraints_met"]
                    else:
                        assert self.reward_oracle is not None
                        rel_preds, _, rel_ok, _ = self.reward_oracle.evaluate_candidates(kept_mols, batch_size=16)
                        rel_aligned = rel_preds[:, self.oracle_col_indices]
                    pass_mols: List[Chem.Mol] = []
                    pass_targets: List[np.ndarray] = []
                    pass_masks: List[np.ndarray] = []
                    gen_cols = list(self.generator.target_cols)
                    bus_names = list(self.critic_bus.spec.objective_names) if self.critic_bus is not None else []
                    for loc_j in range(len(kept_mols)):
                        if not rel_ok[loc_j]:
                            continue
                        req_j = kept_req_targets[loc_j]
                        msk_j = kept_masks[loc_j]
                        if self.critic_bus is not None:
                            errs = [
                                abs(float(rel_aligned[loc_j, b_i]) - float(req_j[gen_cols.index(nm)]))
                                for b_i, nm in enumerate(bus_names)
                                if nm in gen_cols and bool(msk_j[gen_cols.index(nm)])
                            ]
                            post_err = max(errs) if errs else 999.0
                        else:
                            post_err = float(np.max(np.abs(rel_aligned[loc_j][msk_j] - req_j[msk_j]))) if msk_j.any() else 999.0
                        if post_err > 1.25 * self.tolerance_eV:
                            continue
                        pass_mols.append(kept_mols[loc_j])
                        if post_err <= self.tolerance_eV:
                            mapped_t = req_j.copy()
                        elif self.critic_bus is not None:
                            mapped_t = req_j.copy()
                            for gc_i, gc_nm in enumerate(gen_cols):
                                if gc_nm in bus_names:
                                    mapped_t[gc_i] = rel_aligned[loc_j, bus_names.index(gc_nm)]
                        else:
                            mapped_t = rel_aligned[loc_j].copy()
                        if self.size_col_idx is not None:
                            mapped_t[self.size_col_idx] = float(kept_mols[loc_j].GetNumAtoms())
                        pass_targets.append(mapped_t)
                        pass_masks.append(msk_j)
                    if pass_mols:
                        relax_packed, kept_rel = self.generator.encode_molecules_batch(
                            pass_mols, seed=self.seed + step_index, repeat_k=1
                        )
                        if relax_packed is not None and kept_rel:
                            relax_targets_norm = self.generator.normalize_targets(
                                np.asarray([pass_targets[j] for j in kept_rel], dtype=np.float32)
                            )
                            relax_mask_t = torch.as_tensor(
                                np.asarray([pass_masks[j] for j in kept_rel], dtype=bool),
                                dtype=torch.bool,
                                device=self.device,
                            )
                            n_relax_admitted = len(kept_rel)
                        else:
                            relax_packed = None

        # 5c. Advantage-Weighted Flow/ML SFT batch construction (when `alignment_mode == 'awr_sft'`)
        awr_packed: Optional[Dict[str, Any]] = None
        awr_weights_t: Optional[torch.Tensor] = None
        awr_targets_norm: Optional[torch.Tensor] = None
        awr_mask_t: Optional[torch.Tensor] = None
        n_awr_admitted = 0
        if self.alignment_mode == "awr_sft":
            pos_idx = [
                i
                for i in range(total_b)
                if valid_mask_np[i]
                and clean_charge_np[i]
                and mols[i] is not None
                and float(adv_all[i].item()) > 1e-5
            ]
            cand_mols: List[Chem.Mol] = []
            cand_raw_w: List[float] = []
            cand_targets: List[np.ndarray] = []
            cand_masks: List[np.ndarray] = []
            for i in pos_idx:
                adv_val = float(adv_all[i].item())
                w_i = min(math.exp(adv_val / self.awr_temperature), self.awr_max_weight)
                cand_mols.append(mols[i])  # type: ignore[arg-type]
                cand_raw_w.append(w_i)
                cand_targets.append(rep_targets_eV[i])
                cand_masks.append(rep_mask_np[i])

            if len(cand_mols) < 2 and self.awr_hindsight_fallback and self.critic_bus is not None:
                used_set = set(pos_idx)
                gen_cols = list(self.generator.target_cols)
                bus_names = list(self.critic_bus.spec.objective_names)
                her_cands = [
                    i
                    for i in range(total_b)
                    if i not in used_set
                    and valid_mask_np[i]
                    and clean_charge_np[i]
                    and mols[i] is not None
                    and bool(eval_mask_np[i])
                    and bool(ev_bundle["constraints_met"][i])
                ]
                for i in her_cands[:4]:
                    mapped_t = rep_targets_eV[i].copy()
                    for gc_i, gc_nm in enumerate(gen_cols):
                        if gc_nm in bus_names:
                            val = float(ev_bundle["objective_preds"][i, bus_names.index(gc_nm)])
                            if math.isfinite(val):
                                mapped_t[gc_i] = val
                    cand_mols.append(mols[i])  # type: ignore[arg-type]
                    cand_raw_w.append(0.5)
                    cand_targets.append(mapped_t)
                    cand_masks.append(rep_mask_np[i])

            if cand_mols and self.awr_relax_steps > 0 and self.potential_slot is not None:
                rel_out = self.potential_slot.relax_molecules(
                    cand_mols, n_steps=self.awr_relax_steps, chunk_size=16
                )
                kept_rm: List[Chem.Mol] = []
                kept_w: List[float] = []
                kept_t: List[np.ndarray] = []
                kept_m: List[np.ndarray] = []
                for loc_k, rm in enumerate(rel_out["relaxed_mols"]):
                    f_end = float(rel_out["final_f_max_eV_A"][loc_k])
                    if (
                        rel_out["graph_preserved"][loc_k]
                        and rm is not None
                        and math.isfinite(f_end)
                        and f_end <= self.awr_max_fmax
                        and is_clean_organic_charge_state(rm)
                    ):
                        kept_rm.append(rm)
                        kept_w.append(cand_raw_w[loc_k])
                        kept_t.append(cand_targets[loc_k])
                        kept_m.append(cand_masks[loc_k])
                if kept_rm and self.critic_bus is not None:
                    rel_ev = self.critic_bus.evaluate_molecules(kept_rm, for_training=True, batch_size=16)
                    rel_aligned = rel_ev["objective_preds"]
                    rel_ok = rel_ev["evaluated_mask"] & rel_ev["hard_constraints_met"]
                    gen_cols = list(self.generator.target_cols)
                    bus_names = list(self.critic_bus.spec.objective_names)
                    v_mols: List[Chem.Mol] = []
                    v_w: List[float] = []
                    v_t: List[np.ndarray] = []
                    v_m: List[np.ndarray] = []
                    for j_rm in range(len(kept_rm)):
                        if not bool(rel_ok[j_rm]):
                            continue
                        req_j = kept_t[j_rm]
                        msk_j = kept_m[j_rm]
                        errs = [
                            abs(float(rel_aligned[j_rm, b_i]) - float(req_j[gen_cols.index(nm)]))
                            for b_i, nm in enumerate(bus_names)
                            if nm in gen_cols and bool(msk_j[gen_cols.index(nm)])
                        ]
                        post_err = max(errs) if errs else 0.0
                        if post_err > 1.40 * self.tolerance_eV:
                            continue
                        mapped_t = req_j.copy()
                        w_mult = 1.0
                        if post_err > self.tolerance_eV:
                            w_mult = 0.75
                            for gc_i, gc_nm in enumerate(gen_cols):
                                if gc_nm in bus_names:
                                    mapped_t[gc_i] = float(rel_aligned[j_rm, bus_names.index(gc_nm)])
                        if self.size_col_idx is not None:
                            mapped_t[self.size_col_idx] = float(kept_rm[j_rm].GetNumAtoms())
                        v_mols.append(kept_rm[j_rm])
                        v_w.append(kept_w[j_rm] * w_mult)
                        v_t.append(mapped_t)
                        v_m.append(msk_j)
                    cand_mols, cand_raw_w, cand_targets, cand_masks = v_mols, v_w, v_t, v_m
                else:
                    cand_mols, cand_raw_w, cand_targets, cand_masks = kept_rm, kept_w, kept_t, kept_m

            if cand_mols:
                awr_packed, kept_local = self.generator.encode_molecules_batch(
                    cand_mols, seed=self.seed + step_index, repeat_k=self.awr_repeat_k
                )
                if awr_packed is not None and kept_local:
                    w_arr = np.asarray([cand_raw_w[j] for j in kept_local], dtype=np.float32)
                    w_arr = w_arr / max(float(w_arr.sum()), 1e-8)
                    awr_weights_t = torch.as_tensor(w_arr, dtype=torch.float32, device=self.device)
                    awr_targets_norm = self.generator.normalize_targets(
                        np.asarray([cand_targets[j] for j in kept_local], dtype=np.float32)
                    )
                    awr_mask_t = torch.as_tensor(
                        np.asarray([cand_masks[j] for j in kept_local], dtype=bool),
                        dtype=torch.bool,
                        device=self.device,
                    )
                    n_awr_admitted = len(set(kept_local))

        # 6. Policy updates (`grpo` clipped PPO OR `awr_sft` Online RWR) + Replay Anchor + Relaxed-Coord Distillation
        self.generator.train()
        if self.alignment_mode == "grpo":
            has_signal = (
                bool((adv_all.abs() > 1e-6).any().item())
                or self.kl_beta > 0.0
                or self.replay_dataset is not None
                or relax_packed is not None
            )
        else:
            has_signal = (
                awr_packed is not None
                or self.replay_dataset is not None
                or relax_packed is not None
            )
        n_updates = self.ppo_epochs if has_signal else 0
        last_loss = 0.0
        last_policy_loss = 0.0
        last_replay_loss = 0.0
        last_relax_distill_loss = 0.0
        last_kl = 0.0
        grad_norm = 0.0

        autocast_train = (self.device.type == "cuda") and bool(
            getattr(self.generator, "supports_exact_trajectory_logprob", True)
        )
        for _ in range(n_updates):
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=autocast_train):
                if self.alignment_mode == "grpo":
                    cur_packets = self.generator.trajectory_distributions(
                        trajectories,
                        targets_norm,
                        active_mask,
                        temperatures=self.temperatures,
                        cfg_scale=self.cfg_scale,
                        xyz_cfg_scale=self.xyz_cfg_scale,
                        cfg_tree_decay=self.cfg_tree_decay,
                        geometric_guard=self.geometric_guard,
                        topology_only_policy=self.topology_only_policy,
                        electron_parity=self.electron_parity,
                        **cfg_extra,
                    )
                    p_loss = categorical_policy_loss(
                        cur_packets,
                        old_packets,
                        ref_packets,
                        adv_leaf_all,
                        beta=self.kl_beta,
                        clip=self.clip_eps,
                        tree_advantages=adv_tree_all,
                    )
                    total_loss = p_loss

                    if ref_packets is not None:
                        with torch.no_grad():
                            kl_terms = []
                            for (logp, _, _), (ref_logp, _, _) in zip(cur_packets, ref_packets):
                                both_fin = torch.isfinite(logp) & torch.isfinite(ref_logp)
                                s_lp = torch.where(both_fin, logp, torch.zeros_like(logp))
                                s_rf = torch.where(both_fin, ref_logp, torch.zeros_like(ref_logp))
                                pr = torch.where(both_fin, s_lp.exp(), torch.zeros_like(logp))
                                kl_terms.append((pr * (s_lp - s_rf).clamp(-30.0, 30.0)).sum(-1))
                            last_kl = float(torch.cat(kl_terms).mean().item())
                else:
                    if awr_packed is not None and awr_targets_norm is not None and awr_mask_t is not None and awr_weights_t is not None:
                        per_mol_loss, _ = self.generator.compute_sft_loss(
                            awr_packed,
                            awr_targets_norm,
                            awr_mask_t,
                            cfg_dropout_prob=0.0,
                            return_per_molecule=True,
                        )
                        p_loss = (per_mol_loss.float() * awr_weights_t).sum()
                    else:
                        p_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                    total_loss = p_loss

                if self.replay_dataset is not None and self.replay_targets_eV is not None:
                    r_packed = None
                    r_indices = None
                    for _ in range(5):
                        cand_r = self.rng.integers(0, len(self.replay_dataset), size=self.replay_batch_size)
                        try:
                            r_packed = self.replay_dataset.collate(cand_r)
                            r_indices = cand_r
                            break
                        except ValueError:
                            continue
                    if r_packed is not None and r_indices is not None:
                        r_targets_norm = self.generator.normalize_targets(self.replay_targets_eV[r_indices])
                        r_mask = self._sample_replay_active_mask(prompt_active_mask)
                        r_drop = (
                            0.0
                            if (self.alignment_mode == "awr_sft" or (self.size_anchored_cfg and self.size_col_idx is not None))
                            else 0.05
                        )
                        r_loss, _ = self.generator.compute_sft_loss(r_packed, r_targets_norm, r_mask, cfg_dropout_prob=r_drop)
                        total_loss = total_loss + self.replay_weight * r_loss
                        last_replay_loss = float(r_loss.detach().item())

                if relax_packed is not None and relax_targets_norm is not None and relax_mask_t is not None:
                    d_loss, _ = self.generator.compute_sft_loss(
                        relax_packed, relax_targets_norm, relax_mask_t, cfg_dropout_prob=0.0
                    )
                    total_loss = total_loss + self.relax_distill_weight * d_loss
                    last_relax_distill_loss = float(d_loss.detach().item())

            if torch.isfinite(total_loss) and total_loss.requires_grad:
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if self.grad_clip_mode == "value":
                    torch.nn.utils.clip_grad_value_(self.trainable_params, 1.0)
                    sq_sum = sum(
                        float(p.grad.detach().float().pow(2).sum().item())
                        for p in self.trainable_params
                        if p.grad is not None
                    )
                    gn = math.sqrt(sq_sum)
                else:
                    gn = float(torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0).item())
                if math.isfinite(gn):
                    self.optimizer.step()
                    grad_norm = gn
                    last_loss = float(total_loss.detach().item())
                    last_policy_loss = float(p_loss.detach().item())

        valid_f = f_max_np[valid_mask_np & np.isfinite(f_max_np)]
        valid_err = abs_err_np[eval_mask_np & np.isfinite(abs_err_np)]
        legal_atoms = natoms_actual_np[valid_mask_np & np.isfinite(natoms_actual_np)]
        valid_size_err = natoms_abs_err_np[size_eval_mask_np & np.isfinite(natoms_abs_err_np)]
        elapsed = time.perf_counter() - t0
        return {
            "step": step_index + 1,
            "alignment_mode": self.alignment_mode,
            "prompts": p_count,
            "group_size": group_size,
            "proposals": total_b,
            "valid": int(valid_mask_np.sum()),
            "valid_rate": float(valid_mask_np.mean()),
            "raw_valid_rate": float(raw_legal_np.mean()),
            "evaluated": int(eval_mask_np.sum()),
            "hits": int(hits_np.sum()),
            "hit_yield": float(hits_np.mean()),
            "constraints_met_rate_among_valid": float(constraints_met_rate),
            "clean_charge_rate_among_valid": float(clean_charge_np[valid_mask_np].mean()) if valid_mask_np.any() else 0.0,
            "exotic_charge_rate_among_valid": float(1.0 - clean_charge_np[valid_mask_np].mean()) if valid_mask_np.any() else 0.0,
            "unseen_charge_rate": float(1.0 - seen_charge_np[valid_mask_np].mean()) if valid_mask_np.any() else 0.0,
            "conditional_mae_eV": float(np.mean(valid_err)) if len(valid_err) > 0 else float("nan"),
            "mean_legal_atoms": float(np.mean(legal_atoms)) if len(legal_atoms) > 0 else float("nan"),
            "mean_natoms_abs_err": float(np.mean(valid_size_err)) if len(valid_size_err) > 0 else float("nan"),
            "size_reward_mean": float(size_scores_np[size_eval_mask_np].mean()) if size_eval_mask_np.any() else 0.0,
            "f_max_median_eV_A": float(np.median(valid_f)) if len(valid_f) > 0 else float("nan"),
            "prop_reward_mean": float(prop_scores_np[eval_mask_np].mean()) if eval_mask_np.any() else 0.0,
            "phys_reward_mean": float(phys_scores_np[valid_mask_np].mean()) if valid_mask_np.any() else 0.0,
            "awr_admitted": n_awr_admitted,
            "relax_distill_admitted": n_relax_admitted,
            "loss_total": last_loss,
            "loss_policy": last_policy_loss,
            "loss_replay": last_replay_loss,
            "loss_relax_distill": last_relax_distill_loss,
            "kl": last_kl,
            "grad_norm": grad_norm,
            "wall_seconds": elapsed,
        }

    def train_loop(
        self,
        total_steps: int,
        prompt_sampler_fn: Callable[[int, np.random.Generator], Any],
        *,
        group_size: int = 8,
        checkpoint_fn: Optional[Callable[[int, GeneratorSlot, List[Dict[str, Any]]], None]] = None,
        log_every: int = 5,
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for s_idx in range(total_steps):
            sampled = prompt_sampler_fn(s_idx, self.rng)
            if len(sampled) == 3:
                p_targets, p_mask, p_budgets = sampled
            else:
                p_targets, p_mask = sampled
                p_budgets = None
            metrics = self.step(
                p_targets,
                p_mask,
                group_size=group_size,
                step_index=s_idx,
                prompt_atom_budgets=p_budgets,
            )
            history.append(metrics)
            if (s_idx + 1) % log_every == 0 or s_idx == 0 or (s_idx + 1) == total_steps:
                tag = "Loom-AWR" if self.alignment_mode == "awr_sft" else "Loom-GRPO"
                tail = (
                    f"awr={metrics['awr_admitted']} loss={metrics['loss_total']:.4f}"
                    if self.alignment_mode == "awr_sft"
                    else f"KL={metrics['kl']:.4f}"
                )
                print(
                    f"[{tag} {metrics['step']:04d}/{total_steps:04d}] "
                    f"hits={metrics['hits']}/{metrics['proposals']} ({metrics['hit_yield']*100:.1f}%) "
                    f"MAE={metrics['conditional_mae_eV']:.3f}eV "
                    f"valid={metrics['valid_rate']*100:.1f}% "
                    f"SA_ok={metrics['constraints_met_rate_among_valid']*100:.1f}% "
                    f"Fmax={metrics['f_max_median_eV_A']:.2f} "
                    f"{tail} ({metrics['wall_seconds']:.1f}s)",
                    flush=True,
                )
            if checkpoint_fn is not None:
                checkpoint_fn(s_idx + 1, self.generator, history)
        return history
