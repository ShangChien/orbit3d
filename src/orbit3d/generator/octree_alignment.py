"""GRPO on complete octree trajectories, including all coordinate decisions."""

import copy
import hashlib
import numpy as np
import torch
from .octree_backend import trajectory_to_instance, OctreeDataset, interval_arrays
from .random_streams import candidate_seed
from ..oracles.dpa4_oracle import InternalDPA4PropertyOracle


def group_relative_advantages(reward, eligible, mode="joint"):
    """Separate conditional property comparisons from eligibility in experimental v1.

    The property channel averages over eligible candidates (K/m correction),
    with a fixed 0.1 reward-unit standard-deviation floor. The combined signal
    is scaled down to unit standard deviation, never amplified at this stage.
    Zero/one eligible candidates contribute no property comparison.
    """
    if mode == "joint":
        return (reward-reward.mean())/reward.std(unbiased=False).clamp_min(1e-6), None
    if mode != "conditional_property_v1":
        raise ValueError("Unknown point advantage mode")
    binary = torch.where(eligible, torch.ones_like(reward), -torch.ones_like(reward))
    eligibility = (binary-binary.mean())/binary.std(unbiased=False).clamp_min(1e-6)
    property_signal = torch.zeros_like(reward)
    count = int(eligible.sum())
    if count >= 2:
        values = reward[eligible]
        property_signal[eligible] = (len(reward)/count)*(values-values.mean())/values.std(unbiased=False).clamp_min(0.1)
    combined = eligibility+property_signal
    normalizer = combined.std(unbiased=False).clamp_min(1.0)
    return combined/normalizer, (eligibility/normalizer, property_signal/normalizer)


def categorical_policy_loss(current, previous, reference, advantages, beta=0.02, clip=0.2):
    if beta and reference is None:
        raise ValueError('A nonzero KL weight requires a reference policy')
    total = torch.zeros_like(advantages)
    counts = torch.zeros_like(advantages)
    for decision, ((logp, labels, owner), (old, _, _)) in enumerate(zip(current, previous)):
        selected = logp.gather(1, labels[:, None]).squeeze(1)
        old_selected = old.gather(1, labels[:, None]).squeeze(1)
        ratio = (selected - old_selected).clamp(-20, 20).exp()
        adv = advantages[owner]
        surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
        objective = -surrogate
        if beta:
            ref = reference[decision][0]
            delta = torch.where(torch.isfinite(logp), logp - ref, torch.zeros_like(logp))
            objective = objective + beta * (logp.exp() * delta).sum(-1)
        total.index_add_(0, owner, objective)
        counts.index_add_(0, owner, torch.ones_like(selected))
    return (total / counts.clamp_min(1)).mean()


def align_octree(
    model, molecules, labels, mean, scale, intervals, oracle, forcefield, steps=20, group_size=8, lr=2e-5,
    seed=42, kl_beta=0.02, physics_weight=0.2, uncertainty_weight=0.0, replay_weight=0.1, domain=None, electron_parity=None,
    points=None, point_pool=None, checkpoint_callback=None, resume_checkpoint=None, stop_after=None,
    point_reward_type="inverse_point_distance", point_advantage_mode="joint", terminal_feedback_level=None, terminal_max_atoms=None,
):
    if point_reward_type not in {"inverse_point_distance", "eligibility_only"}:
        raise ValueError("Unknown point reward type")
    if points is None and point_reward_type != "inverse_point_distance":
        raise ValueError("Point reward ablations require point conditions")
    if point_advantage_mode not in {"joint", "conditional_property_v1"}:
        raise ValueError("Unknown point advantage mode")
    if points is None and point_advantage_mode != "joint":
        raise ValueError("Conditional property advantages require point conditions")
    if points is not None and (physics_weight or uncertainty_weight):
        raise ValueError("Point GRPO primary comparison uses property-only reward")
    if terminal_feedback_level is not None and (points is None or terminal_feedback_level not in {'structure','physical','property'}):
        raise ValueError('Terminal feedback requires point conditions and a known channel level')
    device = next(model.parameters()).device
    reference = copy.deepcopy(model).eval().requires_grad_(False) if kl_beta else None
    model.requires_grad_(False)
    model.interval_conditioner.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    lower, upper, active = points.arrays(mean, scale) if points else interval_arrays(intervals, mean, scale, model.condition_version)
    low = torch.tensor(lower, device=device).repeat(group_size, 1)
    high = torch.tensor(upper, device=device).repeat(group_size, 1)
    mask = torch.tensor(active, device=device).unsqueeze(0).expand(group_size, *active.shape)
    dataset = OctreeDataset(molecules, model, seed=seed) if replay_weight else None
    rng = np.random.default_rng(seed + 1)
    history = []
    settings = {'seed':seed,'group_size':group_size,'lr':lr,'kl_beta':kl_beta,
                'physics_weight':physics_weight,'uncertainty_weight':uncertainty_weight,
                'replay_weight':replay_weight,'electron_parity':electron_parity,
                'targets':[(p.names,p.values,p.radii) for p in (point_pool or [points])] if points
                          else (intervals.names,intervals.bounds)}
    # Preserve legacy default resume settings; non-default rewards are explicit.
    if point_reward_type != "inverse_point_distance":settings["point_reward_type"] = point_reward_type
    if point_advantage_mode != "joint":settings["point_advantage_mode"] = point_advantage_mode
    if terminal_feedback_level is not None:settings['terminal_feedback_level']=terminal_feedback_level
    if terminal_max_atoms is not None:settings['terminal_max_atoms']=terminal_max_atoms
    if resume_checkpoint is not None:
        saved = resume_checkpoint['alignment_training_state']
        if saved['settings'] != settings:
            raise ValueError('Resume requires the same GRPO targets, optimizer and reward settings')
        # Reference was copied from the original SFT model above. Only the
        # current policy is replaced by the interrupted alignment weights.
        model.load_state_dict(resume_checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        history = list(resume_checkpoint['grpo_history'])
        if saved['completed_groups'] != len(history):
            raise ValueError('Resume step and rollout history differ')
    end = steps if stop_after is None else int(stop_after)
    if not len(history) < end <= steps:
        raise ValueError('Alignment segment must advance within the requested total groups')
    for step in range(len(history), end):
        if point_pool:
            points = point_pool[step % len(point_pool)]
            intervals = points.intervals()
            lower, upper, active = points.arrays(mean, scale)
            low = torch.tensor(lower, device=device).repeat(group_size, 1)
            high = torch.tensor(upper, device=device).repeat(group_size, 1)
            mask = torch.tensor(active, device=device).unsqueeze(0).expand(group_size, *active.shape)
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
                electron_parity=electron_parity,
            )
            old = model.trajectory_distributions(trajectories, low, high, mask, electron_parity=electron_parity)
            ref = reference.trajectory_distributions(trajectories, low, high, mask, electron_parity=electron_parity) if reference is not None else None
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
                inst = trajectory_to_instance(trajectory, element_table=tuple(model.args.allow_atoms.split(',')))
                row.update(valid=True, smiles=inst.smiles)
                assessment = domain.assess(inst.graph) if domain else None
                row['oracle_in_domain'] = assessment['in_domain'] if assessment else True
                row['oracle_domain_similarity'] = assessment['similarity'] if assessment else None
                row['unseen_oracle_training_elements'] = assessment['unseen_training_elements'] if assessment else []
                if assessment and not assessment['in_domain'] and terminal_feedback_level is None:
                    row.update(reason='out_of_oracle_domain', evaluation_status='not_evaluated')
                else:
                    instances.append((i, inst))
            except ValueError as exc:
                row['reason'] = str(exc)
            rollout_records.append(row)
        if terminal_feedback_level is not None:
            from ..physics.terminal_evaluation import evaluate_terminal
            for i,inst in instances:
                record=rollout_records[i]
                if terminal_feedback_level=='structure':
                    rewards[i]=1.;record['evaluation_status']='structural_only'
                    continue
                _,terminal=evaluate_terminal(inst,forcefield,oracle,max_atoms=terminal_max_atoms)
                record['terminal']=terminal
                if terminal['physical_accepted'] and 'final' in terminal:
                    prediction=np.asarray(terminal['final']['prediction_eV'])
                    distance=points.distance(prediction);rewards[i]=1./(1.+distance)
                    hits[i]=intervals.hit(prediction)
                    record.update(evaluation_status='completed',prediction=prediction.tolist(),point_distance=distance,property_hit=hits[i])
                else:record['evaluation_status']='terminal_not_eligible'
        else:
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
                            torch.tensor(inst.coords, device=device), torch.tensor([forcefield.atom_type_map.index(s) for s in inst.symbols], device=device),
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
                    if points is not None:
                        distance = points.distance(prediction)
                        rollout_records[i]["point_distance"] = distance
                        reward = 1.0 if point_reward_type == "eligibility_only" else 1.0 / (1.0 + distance)
                    else:
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
        eligible = torch.tensor([r.get('evaluation_status') == 'completed' for r in rollout_records],device=device)
        if terminal_feedback_level is not None:
            from .terminal_feedback import terminal_advantages
            structural=torch.tensor([r['valid'] for r in rollout_records],device=device)
            physical=torch.tensor([r.get('terminal',{}).get('physical_accepted',False) for r in rollout_records],device=device)
            adv,terminal_components=terminal_advantages(structural,physical,eligible,reward,terminal_feedback_level)
            components=None
            for k,record in enumerate(rollout_records):
                record['terminal_advantages']={name:float(value[k]) for name,value in terminal_components.items()}
        else:
            adv, components = group_relative_advantages(reward, eligible, point_advantage_mode)
        for record, advantage in zip(rollout_records, adv.tolist()):record["advantage"] = advantage
        if components is not None:
            for record, eligibility_part, property_part in zip(rollout_records,components[0].tolist(),components[1].tolist()):
                record.update(eligibility_advantage=eligibility_part,property_advantage=property_part)
        # With no ranking signal or auxiliary objective, Adam's momentum and
        # weight decay must not silently move the policy on an all-failed group.
        updates = 2 if bool(adv.any()) or kl_beta or replay_weight else 0
        loss = torch.zeros((), device=device)
        before = [p.detach().clone() for p in parameters]
        gradient_norm = 0.0
        policy_kl = None
        for _ in range(updates):
            with torch.autocast(device_type=device.type, enabled=False):
                current = model.trajectory_distributions(trajectories, low, high, mask, electron_parity=electron_parity)
                loss = categorical_policy_loss(current, old, ref, adv, beta=kl_beta)
                if ref is not None:
                    with torch.no_grad():
                        kl_terms = []
                        for (logp, _, _), (ref_logp, _, _) in zip(current, ref):
                            delta = torch.where(torch.isfinite(logp), logp - ref_logp, torch.zeros_like(logp))
                            kl_terms.append((logp.exp() * delta).sum(-1))
                        policy_kl = float(torch.cat(kl_terms).mean())
                if replay_weight:
                    indices = rng.integers(0, len(molecules), size=4)
                    packed = dataset.collate(indices)
                    y = torch.tensor((labels[indices] - mean) / scale, device=device)
                    if points is not None:
                        replay_lower = replay_upper = y
                        replay_mask = torch.tensor([v is not None for v in points.values], device=device).expand_as(y)
                    else:
                        replay_lower, replay_upper = y - 0.2, y + 0.2
                        replay_mask = torch.ones_like(y, dtype=torch.bool)
                    replay, _ = model.compute_sft_and_understanding_loss(
                        packed, y, replay_lower, replay_upper, replay_mask, understanding_weight=0.0)
                    loss = loss + replay_weight * replay
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite octree policy loss")
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            optimizer.step()
        row = {
            "step": step + 1,
            "valid": sum(row['valid'] for row in rollout_records),
            "domain_eligible": sum(r.get("oracle_in_domain",False) for r in rollout_records),
            "physical_accepted":sum(r.get("terminal",{}).get("physical_accepted",False) for r in rollout_records),
            "domain_cutoff": domain.cutoff if domain else None,
            "electron_parity": electron_parity,
            "group_size": group_size,
            "property_hits": sum(hits),
            "reward_mean": float(reward.mean()),
            "reward_std": float(reward.std(unbiased=False)),
            "evaluated": sum(r.get('evaluation_status') == 'completed' for r in rollout_records),
            "point_target": dict(zip(points.names, points.values)) if points else None,
            "reward_type": point_reward_type if points else 'exponential_interval_violation',
            "advantage_mode": "terminal_channels" if terminal_feedback_level else point_advantage_mode,
            "terminal_feedback_level":terminal_feedback_level,
            "advantage_max": float(adv.max()),
            "loss": float(loss.detach()),
            "precision": "float32",
            "optimizer_updates": updates,
            "gradient_norm": gradient_norm,
            "policy_kl_before_last_update": policy_kl,
            "parameter_update_l2": float(sum((p.detach()-v).square().sum() for p,v in zip(parameters,before)).sqrt()),
            "weights": {'kl': kl_beta, 'physics': physics_weight, 'uncertainty': uncertainty_weight, 'replay': replay_weight},
            "rollout_records": rollout_records,
        }
        history.append(row)
        if checkpoint_callback is not None:
            training_state = {'settings':settings,'completed_groups':step+1,
                              'optimizer':optimizer.state_dict(),'numpy_rng':rng.bit_generator.state,
                              'torch_rng':torch.get_rng_state(),
                              'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
            checkpoint_callback(step + 1, model, history, training_state)
        print("octree GRPO", {k: v for k, v in row.items() if k != 'rollout_records'}, flush=True)
    del reference
    return history
