"""Masked GRPO with differentiable reference KL and nonnegative gated utility."""

import numpy as np
import torch
import torch.nn.functional as F
from ..constraints import TargetIntervals
from .octree_tokenizer import (
    encode_mol_to_octree_tokens,
    decode_tokens_to_mol3d,
    mol3d_to_validated_rdkit_mol,
    mol_to_clean_smiles,
)


def gated_reward(violation, uncertainty, force_rms, uncertainty_weight=0.35):
    if not np.isfinite([violation, uncertainty, force_rms]).all():
        return -1.0
    return float(
        np.exp(
            -max(0.0, violation)
            - uncertainty_weight * max(0.0, uncertainty)
            - 0.08 * max(0.0, force_rms - 2.5)
        )
    )


def policy_objective(model, ref_model, seq, cond, old_logp, advantages, beta, temperature=1.0, clip=0.2):
    logits, _ = model(seq[:, :-1], cond)
    logp = model.constrained_logits(logits, temperature).log_softmax(-1)
    with torch.no_grad():
        ref, _ = ref_model(seq[:, :-1], cond)
        logref = ref_model.constrained_logits(ref, temperature).log_softmax(-1)
    valid = seq[:, 1:] != 0
    current = logp.gather(-1, seq[:, 1:, None]).squeeze(-1).masked_fill(~valid, 0)
    ratio = (current - old_logp.detach()).clamp(-20, 20).exp()
    adv = advantages.detach()[:, None]
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    # Outside the shared grammar support both distributions have logp=-inf.
    delta = torch.where(torch.isfinite(logp), logp - logref, torch.zeros_like(logp))
    kl = (logp.exp() * delta).sum(-1)
    denom = valid.sum(1).clamp_min(1)
    return (-(surrogate * valid).sum(1) / denom + beta * (kl * valid).sum(1) / denom).mean(), kl


def train_gated_grpo_with_oracle(
    model,
    ref_model,
    sft_mols,
    oracle,
    prop_mean,
    prop_std,
    grpo_cfg,
    target_interval_physical=(620.0, 640.0),
    dpa_guidance=None,
    device="cuda:0",
):
    if not sft_mols:
        raise ValueError("GRPO requires SFT replay examples")
    names = oracle.target_cols
    K = len(names)
    means = np.broadcast_to(np.asarray(prop_mean), (K,))
    scales = np.broadcast_to(np.asarray(prop_std), (K,))
    steps = int(grpo_cfg.get("grpo_steps", 30))
    group = int(grpo_cfg.get("group_size", 8))
    if group < 2:
        raise ValueError("GRPO group_size must be >=2")
    temperature = float(grpo_cfg.get("temperature", 0.9))
    opt = torch.optim.AdamW(model.parameters(), lr=float(grpo_cfg.get("lr", 2e-5)))
    ref_model.eval()
    ref_model.requires_grad_(False)
    rng = np.random.default_rng(int(grpo_cfg.get("seed", 42)))
    history = []
    for step in range(steps):
        if grpo_cfg.get("grpo_mode") == "single_window_specialist":
            intervals = TargetIntervals.parse(target_interval_physical, names)
        else:
            center = means + rng.uniform(-1, 1.4, K) * scales
            width = rng.uniform(0.08, 0.15, K) * scales
            intervals = TargetIntervals.parse(dict(zip(names, zip(center - width, center + width))), names)
        cond = torch.tensor(intervals.condition(means, scales), device=device).repeat(group, 1)
        model.eval()
        seq, old, _ = model.sample_autoregressive(cond, temperature=temperature, top_p=1.0)
        rewards = []
        hits = []
        smiles = []
        for row in seq:
            try:
                inst = decode_tokens_to_mol3d(row.tolist())
                mol = mol3d_to_validated_rdkit_mol(inst)
                yp, ys = oracle.predict_3d_mols([mol])
                yp = np.asarray(yp).reshape(-1)
                ys = np.asarray(ys).reshape(-1)
                if not np.isfinite(yp).all() or not np.isfinite(ys).all():
                    raise FloatingPointError("Oracle failed")
                force = 0.0
                if dpa_guidance is not None:
                    e, f = dpa_guidance.compute_energy_and_forces(
                        torch.tensor(inst.coords, device=device),
                        torch.tensor(inst.atom_types, device=device),
                        clip_forces=False,
                        zero_com_forces=False,
                    )
                    if not torch.isfinite(e).all() or not torch.isfinite(f).all():
                        raise FloatingPointError("MLIP failed")
                    force = float(f.square().sum(-1).mean().sqrt())
                d = intervals.violations(yp).numpy() / scales
                rewards.append(
                    gated_reward(
                        float(np.square(d).sum()),
                        float(np.mean(ys / scales)),
                        force,
                        float(grpo_cfg.get("uncertainty_weight", 0.35)),
                    )
                )
                hits.append(intervals.hit(yp))
                smiles.append(mol_to_clean_smiles(mol))
            except (ValueError, FloatingPointError):
                rewards.append(-1.0)
                hits.append(False)
        reward = torch.tensor(rewards, device=device)
        advantages = (reward - reward.mean()) / reward.std(unbiased=False).clamp_min(1e-6)
        loss, kl = policy_objective(
            model, ref_model, seq, cond, old, advantages, float(grpo_cfg.get("kl_beta", 0.04)), temperature
        )
        replay = sft_mols[int(rng.integers(len(sft_mols)))]
        tokens = torch.tensor([encode_mol_to_octree_tokens(replay, model.max_atoms)], device=device)
        y = (replay.props - means) / scales
        iv = TargetIntervals.parse(
            dict(zip(names, zip(replay.props - 0.15 * scales, replay.props + 0.15 * scales))), names
        )
        c = torch.tensor(iv.condition(means, scales), device=device)[None]
        logits, _ = model(tokens[:, :-1], c)
        anchor = F.cross_entropy(model.constrained_logits(logits).flatten(0, 1), tokens[:, 1:].flatten())
        total = loss + float(grpo_cfg.get("sft_anchor_weight", 0.15)) * anchor
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(
            {
                "step": step + 1,
                "hit_rate": float(np.mean(hits)),
                "valid_fraction": len(smiles) / group,
                "reward_mean": float(reward.mean()),
                "loss": float(total.detach()),
            }
        )
    model.eval()
    return {"steps_run": len(history), "history": history}
