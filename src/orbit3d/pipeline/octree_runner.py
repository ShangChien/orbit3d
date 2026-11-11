"""Pretrained octree adaptation and real-oracle/force-field evaluation."""

import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from ..constraints import TargetIntervals
from ..generator.octree_backend import (
    load_pretrained,
    load_octree_checkpoint,
    OctreeDataset,
    trajectory_to_instance,
    interval_arrays,
    checkpoint_contract,
    required_condition_elements,
)
from ..generator.octree_tokenizer import ATOM_TYPES
from ..generator.random_streams import candidate_seed
from ..generator.spatial_box import BoxSpec3D
from ..oracles.dpa4_oracle import InternalDPA4PropertyOracle
from .evaluators import load_evaluators
from ..physics.conformer_relaxer import relax_conformer_iso_property
from ..physics.batched_relaxer import relax_conformers_iso_property
from ..utils.dataset_io import write_precise_sdf
from ..utils.reproducibility import lock_deterministic_environment
from ..utils.run_evidence import freeze_run, record_input_hashes


def run_octree_from_config(cfg, config_path=None):
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
    if mode in {'train', 'train_and_predict'} and int(settings.get('sft_steps', 100)) <= 0:
        raise ValueError('Supervised training requires positive sft_steps; use align for GRPO-only training')
    if mode == 'align' and int(settings.get('grpo_steps', 0)) <= 0:
        raise ValueError('Alignment requires positive grpo_steps')
    checkpoint = Path(paths["checkpoint_save_dir"]) / "octree_conditional.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    uses_evaluators = (mode != 'train' or int(settings.get('grpo_steps', 0)) > 0
                       or settings.get('label_source', 'sdf') == 'oracle')
    ff, oracle, domain = load_evaluators(cfg, ATOM_TYPES) if uses_evaluators else (None, None, None)
    domain_path = cfg['oracle_config'].get('applicability_domain_path') if uses_evaluators else None
    if domain:
        domain.save(output / 'oracle_domain.json')
    needs_training = mode in {'train', 'train_and_predict'} or (int(settings.get('grpo_steps', 0)) > 0 and float(settings.get('replay_weight', .1)) > 0)
    generator_source = paths.get('checkpoint_load_path') or (
        checkpoint if mode in {'predict', 'align'} else paths.get('generator_pretrained_path'))
    record_input_hashes(output, {'generator': generator_source,
                                'forcefield': paths['forcefield_path'] if uses_evaluators else None,
                                'property_model': paths['property_checkpoint_dir'] if uses_evaluators else None,
                                'applicability_domain': domain_path,
                                'adaptation_data': paths.get('train_data_path') if needs_training else None})
    training = []
    if needs_training and paths.get("train_data_path"):
        training = list(Chem.SDMolSupplier(paths["train_data_path"], removeHs=False))
        if any(m is None for m in training):
            raise ValueError("Training SDF contains unreadable molecules")
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
        yn = (y - mean) / scale
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
        model.prop_understanding_head.requires_grad_(True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        steps = int(settings.get("sft_steps", 100))
        batch = int(settings.get("batch_size", 16))
        adapter_parameters = list(model.interval_conditioner.parameters()) + list(
            model.prop_understanding_head.parameters()
        )
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
            opt, lambda step: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps))
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
                    "training_smiles": sorted(train_smiles),
                    "label_source": label_source,
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
            packed = dataset.collate(indices)
            target = torch.tensor(yn[indices], device=device)
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
                    packed, target, lower, upper, mask, understanding_weight=0.5,
                    atom_class_weights=atom_class_weights
                )
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
        save_adaptation(checkpoint)
    else:
        checkpoint = Path(paths.get("checkpoint_load_path", checkpoint))
        model, state = load_octree_checkpoint(checkpoint, device, settings.get("batch_size", 16))
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
    intervals = TargetIntervals.parse(inference["target_interval"], names)
    lower, upper, active_bounds = interval_arrays(intervals, mean, scale, model.condition_version)
    if int(settings.get("grpo_steps", 0)):
        from ..generator.octree_alignment import align_octree

        if not training and float(settings.get("replay_weight", .1)) > 0:
            raise ValueError("GRPO replay requires training molecules")
        y = np.array([[float(m.GetProp(n)) for n in names] for m in training], dtype=np.float32) if training else None
        reference_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
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
        )
        checkpoint = Path(paths["checkpoint_save_dir"]) / "octree_aligned.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "backend": f"uni3dar_octree_v{model.condition_version}",
                "target_cols": names,
                "target_units": units,
                "contract": checkpoint_contract(model, names, units, mean, scale),
                "mean": mean,
                "scale": scale,
                "grpo_history": history,
                "alignment_reference_sha256": reference_hash,
                "representation_config": {'merge_level': model.args.merge_level, 'allowed_elements': model.args.allow_atoms.split(',')},
                "training_smiles": sorted(train_smiles),
            },
            checkpoint,
        )
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
                score = float(np.square(intervals.violations(self_y).numpy() / scale).sum())
                row.update(smiles=inst.smiles, self_score=score, raw_valid=True)
                assessment = domain.assess(inst.graph) if domain else None
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
        else:
            predictions, uncertainty = oracle.predict_3d_mols(ranking_mols)
            errors = [''] * len(ranking_mols)
        reranked = []
        for (self_score, idx, inst), prediction, std, error in zip(shortlist, predictions, uncertainty, errors):
            if error:
                records[idx].update(status='oracle_failed', reason=error)
                continue
            deviation = intervals.violations(prediction).numpy() / scale
            score = float(np.square(deviation).sum())
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
