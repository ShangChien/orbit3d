"""Pretrained octree adaptation and real-oracle/force-field evaluation."""

import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from ..constraints import TargetIntervals, TargetPoints
from ..generator.octree_backend import (
    load_pretrained,
    load_octree_checkpoint,
    OctreeDataset,
    trajectory_to_instance,
    interval_arrays,
    checkpoint_contract,
    required_condition_elements,
)
from ..generator.molecular_state import ATOM_TYPES
from ..generator.random_streams import candidate_seed
from ..generator.spatial_box import BoxSpec3D
from ..oracles.dpa4_oracle import InternalDPA4PropertyOracle
from ..oracles.fingerprint_oracle import FingerprintOracle
from .evaluators import load_evaluators
from ..physics.conformer_relaxer import relax_conformer_iso_property
from ..physics.batched_relaxer import relax_conformers_iso_property
from ..utils.dataset_io import write_precise_sdf
from ..utils.reproducibility import lock_deterministic_environment
from ..utils.run_evidence import freeze_run, record_input_hashes


def run_octree_from_config(cfg, config_path=None, *, prepared_evaluators=None, prepared_generator=None):
    electronic_state = cfg.get('stage5_relax_config', {})
    if electronic_state.get('charge', 0) != 0 or electronic_state.get('multiplicity', 1) != 1:
        raise ValueError('Current octree/property checkpoints support neutral closed-shell generation only')
    started = time.perf_counter()
    lock_deterministic_environment(int(cfg.get("seed", 42)))
    RDLogger.DisableLog("rdApp.*")
    device = cfg.get("device", "cuda:0")
    torch.set_num_threads(int(cfg.get("threads", 4)))
    paths = cfg["mount_paths"]
    settings = cfg.get("generator_config", {})
    inference = cfg.get("inference_config", {})
    spatial_box = BoxSpec3D(**cfg['spatial_box']) if cfg.get('spatial_box') else None
    output = Path(paths["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    freeze_run(output, config_path or __file__)
    (output / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    names = cfg["oracle_config"]["target_cols"]
    units = cfg['oracle_config'].get('target_units', ['eV' if n in {'homo', 'lumo'} else 'unspecified' for n in names])
    mode = cfg.get("mode", "train_and_predict")
    if mode not in {"train", "predict", "train_and_predict", "align"}:
        raise ValueError(f"Unknown mode: {mode}")
    if mode != 'predict' and (prepared_evaluators is not None or prepared_generator is not None):
        raise ValueError('Read-only prepared resources are for prediction only')
    if mode in {'train', 'train_and_predict'} and int(settings.get('sft_steps', 100)) <= 0:
        raise ValueError('Supervised training requires positive sft_steps; use align for GRPO-only training')
    if mode == 'align' and int(settings.get('grpo_steps', 0)) <= 0:
        raise ValueError('Alignment requires positive grpo_steps')
    checkpoint = Path(paths["checkpoint_save_dir"]) / "octree_conditional.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    uses_evaluators = (mode != 'train' or int(settings.get('grpo_steps', 0)) > 0
                       or settings.get('label_source', 'sdf') == 'oracle')
    ff, oracle, domain = ((prepared_evaluators if prepared_evaluators is not None else load_evaluators(cfg, ATOM_TYPES))
                         if uses_evaluators else (None, None, None))
    domain_path = cfg['oracle_config'].get('applicability_domain_path') if uses_evaluators else None
    if domain:
        domain.save(output / 'oracle_domain.json')
    needs_training = mode in {'train', 'train_and_predict'} or (int(settings.get('grpo_steps', 0)) > 0 and float(settings.get('replay_weight', .1)) > 0)
    generator_source = paths.get('checkpoint_load_path') or (
        checkpoint if mode in {'predict', 'align'} else paths.get('generator_pretrained_path'))
    record_input_hashes(output, {'generator': generator_source,
                                'alignment_resume': paths.get('alignment_resume_path'),
                                'forcefield': paths['forcefield_path'] if uses_evaluators else None,
                                'property_model': paths['property_checkpoint_dir'] if uses_evaluators else None,
                                'applicability_domain': domain_path,
                                'adaptation_data': paths.get('train_data_path') if needs_training else None,
                                'hindsight_data': paths.get('hindsight_data_path') if needs_training else None})
    training = []
    source_count = 0
    if needs_training and paths.get("train_data_path"):
        training = list(Chem.SDMolSupplier(paths["train_data_path"], removeHs=False))
        if any(m is None for m in training):
            raise ValueError("Training SDF contains unreadable molecules")
    source_count = len(training)
    if needs_training and paths.get('hindsight_data_path'):
        additions = list(Chem.SDMolSupplier(paths['hindsight_data_path'], removeHs=False))
        if not additions or any(m is None for m in additions):
            raise ValueError('Hindsight data must contain readable structures')
        training.extend(additions)
    train_smiles = {Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True) for m in training}
    history = []
    if mode in {"train", "train_and_predict"}:
        if not training:
            raise ValueError("Octree adaptation requires training molecules")
        initial_state = None
        if paths.get('checkpoint_load_path'):
            initial_path = Path(paths['checkpoint_load_path'])
            model, initial_state = load_octree_checkpoint(initial_path, device, settings.get('batch_size', 16))
            if initial_state['target_cols'] != names or initial_state.get('target_units', units) != units:
                raise ValueError('Continued adaptation requires matching target names and units')
            train_smiles.update(initial_state.get('training_smiles', []))
        else:
            initial_path = Path(paths['generator_pretrained_path'])
            model = load_pretrained(initial_path, device, len(names), settings.get('batch_size', 16),
                                    condition_version=int(settings.get('condition_version', 2)))
        label_source = settings.get("label_source", "sdf")
        if label_source == "oracle":
            required = required_condition_elements(initial_state or {})
            if required - set(getattr(oracle, 'training_elements', None) or []):
                raise ValueError('New-element pseudo-labels require documented Oracle training coverage')
            y, _ = oracle.predict_3d_mols(training)
            for mol, values in zip(training, y):
                for name, value in zip(names, values):
                    mol.SetProp(name, str(float(value)))
                mol.SetProp(
                    "PropertyLabelOrigin", "OCELOT-trained oracle pseudo-labels; NOT QM9s ground truth"
                )
            write_precise_sdf(output / "train_pseudolabels.sdf", training)
        elif label_source == "sdf":
            y = np.array([[float(m.GetProp(n)) for n in names] for m in training], dtype=np.float32)
        else:
            raise ValueError(f"Unknown label source: {label_source}")
        mean = initial_state['mean'] if initial_state else y.mean(0)
        scale = initial_state['scale'] if initial_state else y.std(0).clip(1e-6)
        yn = ((y - mean) / scale).astype(np.float32)
        dataset = OctreeDataset(training, model, seed=int(cfg.get("seed", 42)))
        weighting = settings.get('element_loss_weighting', 'none')
        atom_class_weights, weighting_metadata = None, {'scheme': 'none'}
        if weighting == 'inverse_sqrt_frequency':
            from ..generator.supervised_weights import inverse_sqrt_element_weights
            atom_class_weights, weighting_metadata = inverse_sqrt_element_weights(training, model.dictionary, device)
        elif weighting != 'none':
            raise ValueError(f'Unknown element loss weighting: {weighting}')
        train_backbone = bool(settings.get("train_backbone", False))
        if initial_state:
            migration = initial_state.get('representation_migration', {})
            if migration.get('requires_adaptation') and not train_backbone:
                raise ValueError('Representation migration requires backbone adaptation')
            if migration.get('requires_adaptation') or initial_state.get('condition_migration', {}).get('requires_condition_adaptation'):
                added = required_condition_elements(initial_state)
                covered = {atom.GetSymbol() for mol in training for atom in mol.GetAtoms()}
                if added - covered:
                    raise ValueError(f'Adaptation data lacks newly enabled elements: {sorted(added-covered)}')
        model.requires_grad_(train_backbone)
        model.interval_conditioner.requires_grad_(True)
        model.prop_understanding_head.requires_grad_(bool(settings.get("train_property_head", True)))
        trainable = [p for p in model.parameters() if p.requires_grad]
        steps = int(settings.get("sft_steps", 100))
        batch = int(settings.get("batch_size", 16))
        adapter_parameters = list(model.interval_conditioner.parameters()) + list(
            model.prop_understanding_head.parameters()
        )
        adapter_parameters = [p for p in adapter_parameters if p.requires_grad]
        adapter_ids = {id(p) for p in adapter_parameters}
        groups = [{"params": adapter_parameters, "lr": float(settings.get("sft_lr", 2e-4))}]
        if train_backbone:
            groups.append(
                {
                    "params": [p for p in trainable if id(p) not in adapter_ids],
                    "lr": float(settings.get("backbone_lr", 1e-5)),
                }
            )
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: 1.0 if settings.get("lr_schedule") == "constant" else 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps))
        )
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        initial_hash = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        def save_adaptation(save_path):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backend": f"uni3dar_octree_v{model.condition_version}",
                    "target_cols": names,
                    "target_units": units,
                    "contract": checkpoint_contract(model, names, units, mean, scale),
                    "mean": mean,
                    "scale": scale,
                    "initial_weights_sha256": initial_hash,
                    "initial_weights_kind": 'adapted_checkpoint' if initial_state else 'upstream_ema',
                    "optimizer_resumed": False,
                    "history": history,
                    "optimizer_updates": len(history),
                    "training_wall_seconds": time.perf_counter() - started,
                    "requested_sft_steps": steps,
                    "requested_wall_seconds": settings.get('sft_wall_seconds'),
                    "training_smiles": sorted(train_smiles),
                    "label_source": label_source,
                    "active_targets": settings.get("active_targets", names),
                    "conditional_training": {k: settings.get(k) for k in ["preference_sigma_range_eV", "condition_contrast_weight", "contrast_pairing_sigma_eV"]},
                    "hindsight_data_path": paths.get("hindsight_data_path"),
                    "label_theory": sorted({m.GetProp("LabelTheory") for m in training if m.HasProp("LabelTheory")}),
                    "train_backbone": train_backbone,
                    "element_loss_weighting": weighting_metadata,
                    "target_migration": ({**initial_state['target_migration'], 'new_target_predictions_trained': True}
                                         if initial_state and initial_state.get('target_migration') else {}),
                    "representation_config": {'merge_level': model.args.merge_level, 'allowed_elements': model.args.allow_atoms.split(',')},
                    "representation_migration": ({**initial_state.get('representation_migration', {}),
                        'requires_adaptation': steps <= 0} if initial_state and initial_state.get('representation_migration') else {}),
                },
                save_path,
            )

        for step in range(steps):
            model.train()
            dataset.grid.set_epoch(step + 1)
            indices = rng.integers(0, len(training), size=batch)
            if paths.get('hindsight_data_path'):
                indices[:batch//2] = rng.integers(0, source_count, size=batch//2)
                indices[batch//2:] = rng.integers(source_count, len(training), size=batch-batch//2)
            packed = dataset.collate(indices)
            target = torch.tensor(yn[indices], device=device)
            if model.condition_version >= 4:
                lower, upper = target, target
                enabled = settings.get('active_targets', names)
                if not enabled or set(enabled) - set(names):
                    raise ValueError('Point SFT requires known active target names')
                active = torch.tensor([n in enabled for n in names], device=device)
                keep = torch.rand((len(target), 1), device=device) >= 0.1
                mask = (active[None, :] & keep).unsqueeze(-1).expand(-1, -1, 2)
                if model.condition_version == 5:
                    from ..generator.gaussian_conditions import sample_gaussian_conditions
                    lower, upper, mask = sample_gaussian_conditions(target, scale, active,
                        tuple(settings.get('preference_sigma_range_eV', [.1, .5])))
            else:
                width = torch.rand_like(target) * 0.25 + 0.15
                lower, upper = target - width, target + width
                if model.condition_version >= 3:
                    # Conditions represent a set containing the label, not a narrow
                    # interval whose midpoint always leaks the regression target.
                    span = torch.exp(torch.empty_like(target).uniform_(np.log(.2), np.log(2.0)))
                    offset = torch.rand_like(target) * span
                    lower, upper = target - offset, target + span - offset
                mask = torch.rand_like(target) > 0.2
                if model.condition_version >= 2:
                    mask = mask.unsqueeze(-1) & (torch.rand(*target.shape, 2, device=device) > 0.2)
            with torch.autocast(
                device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")
            ):
                loss, metrics = model.compute_sft_and_understanding_loss(
                    packed, target, lower, upper, mask, understanding_weight=float(settings.get("understanding_weight", .5)),
                    return_per_molecule=bool(settings.get("condition_contrast_weight", 0)),
                    atom_class_weights=atom_class_weights
                )
            if settings.get('condition_contrast_weight', 0):
                if model.condition_version not in {4, 5}:
                    raise ValueError('Reciprocal contrast requires point or mean-width conditions')
                from ..generator.gaussian_conditions import reciprocal_pairs
                from ..generator.condition_contrast import condition_margin_loss
                pairing_width = (upper if model.condition_version == 5 else
                    torch.full_like(target, float(settings.get('contrast_pairing_sigma_eV', .25))) /
                    torch.as_tensor(scale, device=device, dtype=target.dtype))
                partner, eligible = reciprocal_pairs(target, lower, pairing_width, mask)
                positive = metrics.pop('per_molecule_generation_loss')
                contrast = positive.sum() * 0
                if eligible.any():
                    with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16,
                                        enabled=device.startswith('cuda')):
                        _, negative = model.compute_sft_and_understanding_loss(
                            packed, target, lower[partner], upper[partner], mask[partner],
                            understanding_weight=0, return_per_molecule=True)
                    contrast = condition_margin_loss(positive, negative['per_molecule_generation_loss'], eligible, .02)
                loss = loss + float(settings['condition_contrast_weight']) * contrast
                metrics.update(loss_contrast=float(contrast.detach()), contrast_eligible=int(eligible.sum()),
                               loss_total=float(loss.detach()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Octree SFT diverged at step {step + 1}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            history.append(metrics)
            if step % 10 == 0 or step == steps - 1:
                print("octree SFT", step + 1, metrics, flush=True)
            every = int(settings.get("checkpoint_every", 0))
            if every and (step + 1) % every == 0 and step + 1 < steps:
                save_adaptation(checkpoint.parent / f"octree_step_{step + 1}.pt")
            if settings.get('sft_wall_seconds') is not None and time.perf_counter()-started >= float(settings['sft_wall_seconds']):
                break
        save_adaptation(checkpoint)
    else:
        checkpoint = Path(paths.get("checkpoint_load_path", checkpoint))
        if prepared_generator is None:
            model, state = load_octree_checkpoint(checkpoint, device, settings.get("batch_size", 16))
        else:
            model,state,source = prepared_generator
            if Path(source).resolve() != checkpoint.resolve():
                raise ValueError('Prepared generator does not match the requested checkpoint')
        if state.get('representation_migration', {}).get('requires_adaptation'):
            raise ValueError('Migrated octree depth requires backbone adaptation before inference')
        if state.get('condition_migration', {}).get('requires_condition_adaptation'):
            raise ValueError('Migrated conditioner requires adaptation before conditional inference')
        if state["target_cols"] != names:
            raise ValueError("Octree checkpoint semantics differ")
        if 'target_units' in state and state['target_units'] != units:
            raise ValueError('Octree target units differ from the requested Oracle units')
        mean = state["mean"]
        scale = state["scale"]
        train_smiles = set(state.get("training_smiles", train_smiles))
    if mode == "train" and not int(settings.get("grpo_steps", 0)):
        return {"checkpoint": str(checkpoint), "history": history}
    model.eval()
    ff.set_atom_type_map(model.args.allow_atoms.split(','))
    if spatial_box:
        spatial_box.validate_model(model.args)
    points = None
    if model.condition_version >= 4:
        points = TargetPoints.parse(inference['target_point'], inference['target_tolerance'], names)
        trained_targets = (settings.get('active_targets', names) if mode in {'train','train_and_predict'}
                           else state.get('active_targets'))
        if trained_targets is not None and {n for n,v in zip(names,points.values) if v is not None} != set(trained_targets):
            raise ValueError('Point inference must use the property mask trained by this checkpoint')
        intervals = points.intervals()
        inference["target_interval"] = dict(zip(names, intervals.bounds))
        lower, upper, active_bounds = points.arrays(mean, scale)
    else:
        intervals = TargetIntervals.parse(inference['target_interval'], names)
        lower, upper, active_bounds = interval_arrays(intervals, mean, scale, model.condition_version)
    if model.condition_version == 5:
        sigma = inference['preference_sigma_eV']
        sigma_values = np.array([1. if value is None else float(sigma[n]) if isinstance(sigma, dict) else float(sigma)
                                 for n, value in zip(names, points.values)], dtype=np.float32)
        if not np.isfinite(sigma_values).all() or (sigma_values <= 0).any():
            raise ValueError('Preference width must be finite and positive')
        upper = (sigma_values / scale).astype(np.float32)
        if int(settings.get('grpo_steps', 0)):
            raise ValueError('v5 experiment uses supervised objectives; GRPO remains a separate v4 control')
    # Preserve the acceptance task separately from optional inference controls.
    if inference.get('condition_control', 'correct') == 'off':
        active_bounds = np.zeros_like(active_bounds, dtype=bool)
    elif inference.get('condition_control', 'correct') == 'shuffled':
        if points is None:
            raise ValueError('Shuffled controls require point conditioning')
        supplied = TargetPoints.parse(inference['input_target_point'], inference['target_tolerance'], names)
        if [x is not None for x in supplied.values] != [x is not None for x in points.values]:
            raise ValueError('Shuffled controls must preserve the active property mask')
        lower, supplied_upper, active_bounds = supplied.arrays(mean, scale)
        if model.condition_version != 5: upper = supplied_upper
    elif inference.get('condition_control', 'correct') != 'correct':
        raise ValueError('Unknown condition control')
    if int(settings.get('grpo_steps', 0)) and inference.get('condition_control', 'correct') != 'correct':
        raise ValueError('Condition controls are evaluation-only')
    (output / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if int(settings.get("grpo_steps", 0)):
        from ..generator.octree_alignment import align_octree

        if not training and float(settings.get("replay_weight", .1)) > 0:
            raise ValueError("GRPO replay requires training molecules")
        y = np.array([[float(m.GetProp(n)) for n in names] for m in training], dtype=np.float32) if training else None
        reference_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        resume = (torch.load(paths['alignment_resume_path'],map_location=device,weights_only=False)
                  if paths.get('alignment_resume_path') else None)
        if resume is not None and resume['alignment_reference_sha256'] != reference_hash:
            raise ValueError('GRPO resume must retain the original frozen SFT reference')
        last_training_state = None
        point_pool = [TargetPoints.parse(raw, inference['target_tolerance'], names)
                      for raw in settings.get('point_target_pool', [])] if points else None
        if points and any([v is not None for v in p.values] != [v is not None for v in points.values] for p in point_pool):
            raise ValueError('Point alignment target pool must preserve the trained property mask')
        def alignment_state(alignment_history):
            return {
                "model_state_dict": model.state_dict(),
                "backend": f"uni3dar_octree_v{model.condition_version}",
                "target_cols": names,
                "target_units": units,
                "contract": checkpoint_contract(model, names, units, mean, scale),
                "mean": mean,
                "scale": scale,
                "grpo_history": alignment_history,
                "alignment_training_state": last_training_state,
                "alignment_requested_groups": int(settings['grpo_steps']),
                "alignment_complete": len(alignment_history) == int(settings['grpo_steps']),
                "label_theory": state.get("label_theory", []) if mode == "align" else sorted({m.GetProp("LabelTheory") for m in training if m.HasProp("LabelTheory")}),
                "alignment_reference_sha256": reference_hash,
                "active_targets": ([n for n,v in zip(names,points.values) if v is not None] if points
                                   else settings.get('active_targets',names)),
                "representation_config": {'merge_level': model.args.merge_level, 'allowed_elements': model.args.allow_atoms.split(',')},
                "training_smiles": sorted(train_smiles),
            }

        def save_alignment(path, alignment_history):
            temporary = path.with_suffix('.pt.tmp')
            torch.save(alignment_state(alignment_history),temporary)
            temporary.replace(path)

        def save_alignment_step(step, aligned_model, alignment_history, training_state=None):
            nonlocal last_training_state
            last_training_state = training_state
            if step in settings.get('alignment_checkpoints', [50, 100]):
                save_alignment(Path(paths['checkpoint_save_dir']) / f'octree_group_{step}.pt',alignment_history)
            if step % int(settings.get('alignment_checkpoint_every',25)) == 0:
                save_alignment(Path(paths['checkpoint_save_dir']) / 'octree_resume_latest.pt',alignment_history)

        history = align_octree(
            model,
            training,
            y,
            mean,
            scale,
            intervals,
            oracle,
            ff,
            steps=int(settings["grpo_steps"]),
            lr=float(settings.get('grpo_lr', 2e-5)),
            group_size=int(settings.get("group_size", 8)),
            seed=int(cfg.get('seed', 42)),
            kl_beta=float(settings.get('kl_beta', 0.02)),
            physics_weight=float(settings.get('physics_weight', 0.2)),
            uncertainty_weight=float(settings.get('uncertainty_weight', 0.0)),
            replay_weight=float(settings.get('replay_weight', 0.1)),
            domain=domain,
            electron_parity=inference.get('electron_parity'),
            points=points,
            point_pool=point_pool,
            checkpoint_callback=save_alignment_step,
            resume_checkpoint=resume,
            stop_after=settings.get('alignment_stop_after'),
            point_reward_type=settings.get('point_reward_type','inverse_point_distance'),
            point_advantage_mode=settings.get('point_advantage_mode','joint'),
            terminal_feedback_level=settings.get('terminal_feedback_level'),
        )
        filename = 'octree_aligned.pt' if len(history)==int(settings['grpo_steps']) else f'octree_resume_{len(history)}.pt'
        checkpoint = Path(paths["checkpoint_save_dir"]) / filename
        save_alignment(checkpoint,history)
    if mode in {"train", "align"}:
        return {"checkpoint": str(checkpoint), "history": history}
    lock_deterministic_environment(int(inference.get("base_seed", 2026)))
    model.eval()
    total = int(inference.get("num_candidates", 64))
    requested = int(inference.get("num_molecules", 16))
    batch = int(inference.get("batch_size", 16))
    records = []
    candidates = []
    token_arrays = []
    coordinate_arrays = []
    raw_symbols = []
    stage_seconds = {"generation": 0.0, "validation": 0.0, "ranking": 0.0, "refinement": 0.0}
    ranking = inference.get("ranking", "external")
    if ranking not in {"first_valid", "self", "external"}:
        raise ValueError("ranking must be first_valid, self or external")
    independent_sampling = inference.get("sampling_mode", "legacy") == "independent"
    precision = inference.get("precision", "bfloat16")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("Inference precision must be float32 or bfloat16")
    for begin in range(0, total, batch):
        batch_start = time.perf_counter()
        count = min(batch, total - begin)
        low = torch.tensor(lower, device=device).repeat(count, 1)
        high = torch.tensor(upper, device=device).repeat(count, 1)
        active = torch.tensor(active_bounds, device=device).unsqueeze(0).expand(count, *active_bounds.shape)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.split(":")[0],
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda") and precision == "bfloat16",
            ),
        ):
            trajectories = model.generate_conditional_trajectories(
                low,
                high,
                active,
                cfg_scale=float(inference.get("cfg_scale", 1.0)),
                tree_temp=inference.get('tree_temperature'),
                atom_temp=inference.get('atom_temperature'),
                xyz_temp=inference.get('xyz_temperature'),
                electron_parity=inference.get('electron_parity'),
                cache_mode=inference.get('cache_mode', 'concat'),
                evaluate_self_consistency=True,
                row_seeds=[candidate_seed(inference.get("base_seed", 2026), begin + j) for j in range(count)]
                if independent_sampling
                else None,
                spatial_box=spatial_box,
            )
        for j, trajectory in enumerate(trajectories):
            if j == 0:
                stage_seconds["generation"] += time.perf_counter() - batch_start
                validation_start = time.perf_counter()
            idx = len(records)
            tokens = trajectory["decoder_feat"].detach().cpu().numpy()
            token_arrays.append(tokens)
            coordinate_arrays.append(np.asarray(trajectory["positions_np"]))
            raw_symbols.append(trajectory["atom_symbols"])
            row = {
                "candidate_id": idx,
                "status": "not_selected",
                "smiles": None,
                "raw_valid": False,
                "self_score": None,
                "initial_force": None,
                "final_force": None,
                "y_pred": None,
                "joint_success": False,
                "token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                "sampling_seed": candidate_seed(inference.get("base_seed", 2026), idx)
                if independent_sampling
                else None,
            }
            try:
                inst = trajectory_to_instance(trajectory, spatial_box, tuple(model.args.allow_atoms.split(',')))
                self_y = trajectory["self_pred_norm"].float().cpu().numpy() * scale + mean
                score = points.distance(self_y) if points else float(np.square(intervals.violations(self_y).numpy() / scale).sum())
                row.update(smiles=inst.smiles, self_score=score, raw_valid=True)
                try:
                    assessment = domain.assess(inst.graph) if domain else None
                except ValueError as error:
                    row.update(status='domain_failed',reason=str(error))
                    records.append(row)
                    continue
                row['oracle_domain_similarity'] = assessment['similarity'] if assessment else None
                row['unseen_oracle_training_elements'] = assessment['unseen_training_elements'] if assessment else []
                if assessment and not assessment['in_domain']:
                    row["status"] = "out_of_oracle_domain"
                else:
                    candidates.append((score, idx, inst))
            except ValueError as exc:
                row.update(status="invalid", raw_valid=False, reason=str(exc))
            records.append(row)
        stage_seconds["validation"] += time.perf_counter() - validation_start
        print("octree candidates", len(records), flush=True)
    np.savez_compressed(output / "raw_trajectories.npz",
                        **{f"tokens_{i}": t for i, t in enumerate(token_arrays)},
                        **{f"coordinates_{i}": c for i, c in enumerate(coordinate_arrays)})
    (output / "raw_symbols.json").write_text(json.dumps(raw_symbols) + "\n")
    pd.DataFrame(records).to_csv(output / "candidates.csv", index=False)
    ranking_start = time.perf_counter()
    if ranking != "first_valid":
        candidates.sort(key=lambda item: (item[0], item[1]))
    budget = int(inference.get("oracle_candidates", requested * 2))
    shortlist = candidates[:budget] if ranking == "external" and budget > 0 else candidates
    external_ranking_calls = 0
    if ranking == "external" and budget > 0 and shortlist:
        external_ranking_calls = len(shortlist)
        ranking_mols = [inst.graph for _, _, inst in shortlist]
        if isinstance(oracle, InternalDPA4PropertyOracle):
            evaluation = oracle.evaluate_3d_mols_batched(ranking_mols)
            predictions, uncertainty = evaluation['mean'], evaluation['std']
            errors = evaluation['errors']
        elif isinstance(oracle, FingerprintOracle):
            evaluation=oracle.evaluate_graph_mols(ranking_mols)
            predictions,uncertainty,errors=evaluation['mean'],evaluation['std'],evaluation['errors']
        else:
            predictions, uncertainty = oracle.predict_3d_mols(ranking_mols)
            errors = [''] * len(ranking_mols)
        reranked = []
        for (self_score, idx, inst), prediction, std, error in zip(shortlist, predictions, uncertainty, errors):
            if error:
                records[idx].update(status='oracle_failed', reason=error)
                continue
            deviation = intervals.violations(prediction).numpy() / scale
            score = points.distance(prediction) if points else float(np.square(deviation).sum())
            records[idx].update(
                oracle_rank_prediction=np.asarray(prediction).tolist(), oracle_rank_score=score
            )
            reranked.append((score, float(np.mean(np.asarray(std) / scale)), idx, inst))
        reranked.sort(key=lambda item: (item[0], item[1], item[2]))
        shortlist = [(score, idx, inst) for score, _, idx, inst in reranked]
    selected = []
    seen = set()
    for score, idx, inst in shortlist:
        if inst.smiles in seen:
            records[idx]["status"] = "duplicate"
            continue
        if len(selected) < requested:
            selected.append((idx, inst))
            seen.add(inst.smiles)
    stage_seconds["ranking"] = time.perf_counter() - ranking_start
    threshold = float(inference.get("max_rms_force_eva", 0.5))
    mols = []
    qualified = []
    traces = []
    relax_settings = cfg.get("stage5_relax_config", {})
    relax_steps = int(relax_settings.get("relax_steps", 8)) if relax_settings.get("enabled", True) else 0
    use_batch_refinement = bool(relax_settings.get('batch_refinement', False))
    acceptance_tolerances = {key: float(relax_settings.get(key, 0.))
                             for key in ['property_tolerance', 'energy_tolerance']}
    batch_results = None
    if use_batch_refinement:
        batch_start = time.perf_counter()
        batch_results = relax_conformers_iso_property(
            [inst for _, inst in selected], ff, oracle, dict(zip(names, intervals.bounds)),
            n_steps=relax_steps, step_size=float(relax_settings.get('step_size', .004)),
            **acceptance_tolerances,
            coordinate_bounds=spatial_box.world_bounds if spatial_box else None,
        )
        stage_seconds['refinement'] = time.perf_counter() - batch_start
    for selected_index, (idx, inst) in enumerate(selected):
        refine_start = time.perf_counter()
        if use_batch_refinement:
            out, info = batch_results[selected_index]
            if out is None:
                records[idx].update(info, status='refinement_failed')
                pd.DataFrame(records).to_csv(output / 'candidates.csv', index=False)
                continue
        else:
            try:
                out, info = relax_conformer_iso_property(
                    inst, ff, oracle, dict(zip(names, intervals.bounds)),
                    n_steps=relax_steps,
                    step_size=float(relax_settings.get("step_size", 0.004)),
                    **acceptance_tolerances,
                    coordinate_bounds=spatial_box.world_bounds if spatial_box else None,
                )
            except (ValueError, FloatingPointError) as exc:
                records[idx].update(status="refinement_failed", reason=str(exc))
                pd.DataFrame(records).to_csv(output / "candidates.csv", index=False)
                stage_seconds["refinement"] += time.perf_counter() - refine_start
                continue
            stage_seconds["refinement"] += time.perf_counter() - refine_start
        passed = info["interval_hit"] and info["final_force_eva"] <= threshold
        mol = out.graph
        mol.RemoveAllConformers()
        conf = Chem.Conformer(len(out.coords))
        for i, xyz in enumerate(out.coords):
            conf.SetAtomPosition(i, xyz.tolist())
        mol.AddConformer(conf)
        mol.SetProp("CandidateID", str(idx))
        mol.SetProp("PredictedValue", json.dumps(info["y_pred"]))
        mol.SetProp("JointSuccess", str(passed))
        mols.append(mol)
        if passed:
            qualified.append(mol)
        records[idx].update(
            status="refined",
            y_pred=info["y_pred"],
            initial_force=info["init_force_eva"],
            final_force=info["final_force_eva"],
            initial_max_force=info["init_max_force_eva"],
            final_max_force=info["final_max_force_eva"],
            refinement_seconds=None if use_batch_refinement else time.perf_counter() - refine_start,
            evaluation_calls=info["evaluation_calls"],
            gradient_calls=info["gradient_calls"],
            joint_success=passed,
        )
        traces.append({"candidate_id": idx, **info})
        pd.DataFrame(records).to_csv(output / "candidates.csv", index=False)
    write_precise_sdf(output / "generated.sdf", mols)
    write_precise_sdf(output / "qualified.sdf", qualified)
    exported = (
        [m for m in Chem.SDMolSupplier(str(output / "generated.sdf"), removeHs=False) if m is not None]
        if mols
        else []
    )
    for before, after in zip(mols, exported):
        np.testing.assert_allclose(
            before.GetConformer().GetPositions(), after.GetConformer().GetPositions(), atol=1e-9, rtol=0
        )
        pred, _ = oracle.predict_3d_mols([after])
        np.testing.assert_allclose(
            pred[0], json.loads(before.GetProp("PredictedValue")), atol=1e-5, rtol=1e-6
        )
    assert len(exported) == len(mols)
    fingerprints = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fps = [fingerprints.GetFingerprint(Chem.RemoveHs(m)) for m in mols]
    similarities = [s for i, fp in enumerate(fps) for s in DataStructs.BulkTanimotoSimilarity(fp, fps[:i])]
    summary = {
        "backend": f"uni3dar_octree_v{model.condition_version}",
        "cache_mode": inference.get('cache_mode', 'concat'),
        "sampling_temperatures": {
            name: (inference[name + '_temperature'] if inference.get(name + '_temperature') is not None
                   else list(getattr(model, name + '_temperature')))
            for name in ['tree', 'atom', 'xyz']
        },
        "attempted": len(records),
        "raw_valid": sum(bool(r.get("raw_valid", False)) for r in records),
        "domain_eligible": len(candidates),
        "out_of_oracle_domain": sum(r["status"] == "out_of_oracle_domain" for r in records),
        "ranking": ranking,
        "external_ranking_calls": external_ranking_calls,
        "refinement_evaluation_calls": sum(r.get("evaluation_calls", 0) for r in records),
        "refinement_gradient_calls": sum(r.get("gradient_calls", 0) for r in records),
        "export_recheck_calls": len(exported),
        "stage_seconds": stage_seconds,
        "batch_refinement": use_batch_refinement,
        "acceptance_tolerances": acceptance_tolerances,
        "retained_morgan_diversity": float(1 - np.mean(similarities)) if similarities else None,
        "refinement_failed": sum(r["status"] == "refinement_failed" for r in records),
        "domain_failed": sum(r["status"] == "domain_failed" for r in records),
        "oracle_failed": sum(r['status'] == 'oracle_failed' for r in records),
        "unique_raw_valid": len({r["smiles"] for r in records if r.get("raw_valid")}),
        "retained": len(mols),
        "qualified": len(qualified),
        "joint_success_rate": len(qualified) / len(records),
        "retained_novel_vs_adaptation": sum(r["smiles"] not in train_smiles for r in records if r["status"] == "refined")
        if train_smiles
        else None,
        "property_hit_retained": sum(intervals.hit(r["y_pred"]) for r in records if r["status"] == "refined"),
        "mean_final_force": float(np.mean([r["final_force"] for r in records if r["status"] == "refined"]))
        if mols
        else None,
        "maximum_retained_atom_force_eva": max((r["final_max_force"] for r in records if r["status"] == "refined"), default=None),
        "export_prediction_recheck_passed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    pd.DataFrame(records).to_csv(output / "candidates.csv", index=False)
    (output / "relaxation_traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
