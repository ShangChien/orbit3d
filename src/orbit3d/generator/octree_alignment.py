"""GRPO on complete octree trajectories, including all coordinate decisions."""

import copy
import hashlib
import numpy as np
import torch
from .octree_backend import trajectory_to_instance, OctreeDataset, interval_arrays
from .random_streams import candidate_seed
from ..oracles.dpa4_oracle import InternalDPA4PropertyOracle


def categorical_policy_loss(current, previous, reference, advantages, beta=0.02, clip=0.2):
    total = torch.zeros_like(advantages)
    counts = torch.zeros_like(advantages)
    for (logp, labels, owner), (old, _, _), (ref, _, _) in zip(current, previous, reference):
        selected = logp.gather(1, labels[:, None]).squeeze(1)
        old_selected = old.gather(1, labels[:, None]).squeeze(1)
        ratio = (selected - old_selected).clamp(-20, 20).exp()
        adv = advantages[owner]
        surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
        delta = torch.where(torch.isfinite(logp), logp - ref, torch.zeros_like(logp))
        kl = (logp.exp() * delta).sum(-1)
        total.index_add_(0, owner, -surrogate + beta * kl)
        counts.index_add_(0, owner, torch.ones_like(selected))
    return (total / counts.clamp_min(1)).mean()


def align_octree(
    model, molecules, labels, mean, scale, intervals, oracle, forcefield, steps=20, group_size=8, lr=2e-5,
    seed=42, kl_beta=0.02, physics_weight=0.2, uncertainty_weight=0.0, replay_weight=0.1,
):
    device = next(model.parameters()).device
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    model.requires_grad_(False)
    model.interval_conditioner.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    lower, upper, active = interval_arrays(intervals, mean, scale, model.condition_version)
    low = torch.tensor(lower, device=device).repeat(group_size, 1)
    high = torch.tensor(upper, device=device).repeat(group_size, 1)
    mask = torch.tensor(active, device=device).unsqueeze(0).expand(group_size, *active.shape)
    dataset = OctreeDataset(molecules, model, seed=seed)
    rng = np.random.default_rng(seed + 1)
    history = []
    for step in range(steps):
        model.eval()
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, enabled=False),
        ):
            trajectories = model.generate_conditional_trajectories(
                low,
                high,
                mask,
                cfg_scale=1.0,
                tree_temp=0.85,
                atom_temp=0.3,
                xyz_temp=0.25,
                evaluate_self_consistency=False,
                row_seeds=[candidate_seed(seed, step * group_size + i) for i in range(group_size)],
            )
            old = model.trajectory_distributions(trajectories, low, high, mask)
            ref = reference.trajectory_distributions(trajectories, low, high, mask)
        rewards = [-1.0] * len(trajectories)
        hits = [False] * len(trajectories)
        instances = []
        rollout_records = []
        for i, trajectory in enumerate(trajectories):
            tokens = trajectory['decoder_feat'].cpu().numpy()
            row = {'candidate_seed': candidate_seed(seed, step * group_size + i), 'valid': False,
                   'token_sha256': hashlib.sha256(tokens.tobytes()).hexdigest(),
                   'atom_symbols': trajectory['atom_symbols'], 'coordinates_A': trajectory['positions_np'].tolist()}
            try:
                inst = trajectory_to_instance(trajectory)
                row.update(valid=True, smiles=inst.smiles)
                instances.append((i, inst))
            except ValueError as exc:
                row['reason'] = str(exc)
            rollout_records.append(row)
        evaluation = None
        if instances and isinstance(oracle, InternalDPA4PropertyOracle):
            evaluation = oracle.evaluate_3d_mols_batched([inst.graph for _, inst in instances])
        for j, (i, inst) in enumerate(instances):
            try:
                if evaluation is not None:
                    raw_force = evaluation['forces_eV_A'][j]
                    rollout_records[i].update(energy_eV=float(evaluation['energy_eV'][j]),
                                              raw_forces_eV_A=raw_force.tolist() if raw_force is not None else None)
                    if evaluation['errors'][j]:
                        raise ValueError(evaluation['errors'][j])
                    prediction, std = evaluation['mean'][j], evaluation['std'][j]
                    energy = evaluation['energy_eV'][j]
                    force = torch.as_tensor(evaluation['forces_eV_A'][j])
                else:
                    prediction, std = oracle.predict_3d_mols([inst.graph])
                    prediction = np.asarray(prediction).reshape(-1)
                    std = np.asarray(std).reshape(-1)
                    energy, force = forcefield.compute_energy_and_forces(
                        torch.tensor(inst.coords, device=device), torch.tensor(inst.atom_types, device=device),
                        clip_forces=False, zero_com_forces=False,
                    )
                if (
                    not np.isfinite(prediction).all()
                    or not np.isfinite(std).all()
                    or not np.isfinite(float(energy))
                    or not torch.isfinite(force).all()
                ):
                    raise ValueError("Nonfinite evaluation")
                violation = intervals.violations(prediction).numpy() / scale
                rms = float(force.square().sum(-1).mean().sqrt())
                reward = float(
                    np.exp(
                        -np.square(violation).sum() - physics_weight * max(0.0, rms - 0.5) - uncertainty_weight * np.mean(std / scale)
                    )
                )
                rewards[i] = reward
                hits[i] = intervals.hit(prediction)
                rollout_records[i].update(prediction=prediction.tolist(), std=std.tolist(), rms_force=rms,
                                          energy_eV=float(energy), raw_forces_eV_A=force.cpu().tolist(), property_hit=hits[i],
                                          evaluation_status='completed')
            except (ValueError, FloatingPointError) as exc:
                rollout_records[i].update(reason=str(exc), evaluation_status='failed')
        for row, value in zip(rollout_records, rewards):
            row['reward'] = value
        reward = torch.tensor(rewards, device=device)
        adv = (reward - reward.mean()) / reward.std(unbiased=False).clamp_min(1e-6)
        for _ in range(2):
            with torch.autocast(device_type=device.type, enabled=False):
                current = model.trajectory_distributions(trajectories, low, high, mask)
                loss = categorical_policy_loss(current, old, ref, adv, beta=kl_beta)
                indices = rng.integers(0, len(molecules), size=4)
                packed = dataset.collate(indices)
                y = torch.tensor((labels[indices] - mean) / scale, device=device)
                replay, _ = model.compute_sft_and_understanding_loss(
                    packed,
                    y,
                    y - 0.2,
                    y + 0.2,
                    torch.ones_like(y, dtype=torch.bool),
                    understanding_weight=0.0,
                )
                loss = loss + replay_weight * replay
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite octree policy loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
        row = {
            "step": step + 1,
            "valid": len(instances),
            "group_size": group_size,
            "property_hits": sum(hits),
            "reward_mean": float(reward.mean()),
            "loss": float(loss.detach()),
            "precision": "float32",
            "weights": {'kl': kl_beta, 'physics': physics_weight, 'uncertainty': uncertainty_weight, 'replay': replay_weight},
            "rollout_records": rollout_records,
        }
        history.append(row)
        print("octree GRPO", {k: v for k, v in row.items() if k != 'rollout_records'}, flush=True)
    del reference
    return history
