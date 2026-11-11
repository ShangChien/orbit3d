"""Configured fixed-scaffold graph growth with the shared evaluator/refinement stack."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from ..constraints import TargetIntervals
from ..generator.scaffold import ScaffoldState
from ..generator.scaffold_geometry import propose_attachment
from ..generator.random_streams import candidate_seed
from ..generator.octree_tokenizer import rdkit_mol_to_mol3d_instance
from ..physics.batched_relaxer import relax_conformers_iso_property
from ..physics.conformer_relaxer import relax_conformer_iso_property
from ..utils.dataset_io import write_precise_sdf
from ..utils.reproducibility import lock_deterministic_environment
from ..utils.run_evidence import freeze_run, record_input_hashes
from .evaluators import load_evaluators


def read_scaffolds(path):
    states = []
    for index, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False)):
        if mol is None or not mol.GetNumConformers() or not mol.HasProp('OpenAtomIndices'):
            raise ValueError(f'Scaffold {index} requires a readable 3D graph and OpenAtomIndices')
        indices = json.loads(mol.GetProp('OpenAtomIndices'))
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < mol.GetNumAtoms() for i in indices):
            raise ValueError(f'Scaffold {index} has invalid open atom indices')
        states.append(ScaffoldState.from_molecule(mol, indices))
    if not states:
        raise ValueError('Scaffold input is empty')
    return states


def graph_smiles(molecule):
    mol = Chem.RemoveHs(Chem.Mol(molecule))
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def run_scaffold_from_config(cfg, config_path=None):
    if cfg.get('mode', 'predict') != 'predict':
        raise ValueError('Scaffold graph growth is an explicit proposal workflow; mode must be predict')
    started = time.perf_counter()
    paths, settings, inference = (cfg[key] for key in ['mount_paths', 'generator_config', 'inference_config'])
    count, retained = int(inference['num_candidates']), int(inference['num_molecules'])
    budget = int(inference.get('oracle_candidates', count))
    growth_steps = int(settings.get('attachments_per_candidate', 1))
    if count < 1 or retained < 0 or budget < retained or budget > count or growth_steps < 1:
        raise ValueError('Invalid proposal, external evaluation or growth budget')
    sources = read_scaffolds(paths['scaffolds_path'])
    menu = settings['fragments']
    if not menu or any(Chem.MolFromSmiles(smiles) is None for smiles in menu):
        raise ValueError('Fragment menu requires valid SMILES')
    elements = sorted({atom.GetSymbol() for source in sources for atom in source.molecule.GetAtoms()} |
                      {atom.GetSymbol() for smiles in menu for atom in Chem.AddHs(Chem.MolFromSmiles(smiles)).GetAtoms()})
    seed = int(cfg.get('seed', 2026))
    lock_deterministic_environment(seed)
    torch.set_num_threads(int(cfg.get('threads', 2)))
    output = Path(paths['output_dir'])
    freeze_run(output, config_path or __file__)
    (output/'run_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    record_input_hashes(output, {'scaffolds': paths['scaffolds_path'], 'forcefield': paths['forcefield_path'],
                                'oracle': paths['property_checkpoint_dir'],
                                'domain': cfg['oracle_config'].get('applicability_domain_path')})
    ff, oracle, domain = load_evaluators(cfg, elements, torch.float64)
    intervals = TargetIntervals.parse(inference['target_interval'], oracle.target_cols)
    if domain:
        domain.save(output/'oracle_domain.json')
    records, proposals, raw_symbols, raw_molecules = [], [], [], []
    for i in range(count):
        row_seed = candidate_seed(seed, i)
        rng = np.random.default_rng(row_seed)
        source_id = i % len(sources)
        state = sources[source_id]
        row = {'candidate_id': i, 'candidate_seed': row_seed, 'source_scaffold_index': source_id,
               'raw_valid': False, 'joint_success': False, 'status': 'generation_failed', 'growth': []}
        symbols = []
        try:
            for step in range(growth_steps):
                if not state.ports:
                    raise ValueError('No open port remains for the requested growth depth')
                port = int(rng.choice(sorted(state.ports)))
                fragment = menu[int(rng.integers(len(menu)))]
                state, proposal = propose_attachment(state, fragment, port, seed=candidate_seed(seed, i, step+1) % (2**31-1),
                    max_attempts=int(settings.get('max_attachment_attempts', 12)), bounds=settings.get('coordinate_bounds'))
                row['growth'].append({'step': step, 'port_map': port, 'fragment': fragment, **proposal})
                if state is None:
                    raise ValueError(proposal['reason'])
            mol = state.molecule
            symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            inst = rdkit_mol_to_mol3d_instance(mol, np.empty(0), center=False, element_table=elements, coordinate_dtype=np.float64)
            mol.SetProp('CandidateID', str(i))
            raw_molecules.append(mol)
            row.update(raw_valid=True, status='chemically_valid', smiles=graph_smiles(mol),
                       actions=list(state.actions), fixed_atom_maps=list(state.fixed_positions))
            assessment = domain.assess(mol) if domain else None
            row['domain_eligible'] = assessment['in_domain'] if assessment else True
            row['domain_assessment'] = assessment
            if row['domain_eligible']:
                proposals.append((i, inst, state))
            else:
                row['status'] = 'out_of_oracle_domain'
        except ValueError as error:
            row['reason'] = str(error)
        records.append(row)
        raw_symbols.append(symbols)
    (output/'raw_symbols.json').write_text(json.dumps(raw_symbols)+'\n')
    write_precise_sdf(output/'proposals.sdf', raw_molecules)
    # Candidate order controls the external-call budget. There is no learned self-head here.
    ranked = []
    for i, inst, state in proposals[:budget]:
        try:
            prediction, std = oracle.predict_3d_mols([inst.graph])
            prediction, std = np.asarray(prediction).reshape(-1), np.asarray(std).reshape(-1)
            if not np.isfinite(prediction).all() or not np.isfinite(std).all():
                raise ValueError('Nonfinite property evaluation')
            score = float(intervals.violations(prediction).square().sum())
            records[i].update(status='scored', proposal_prediction=prediction.tolist(), property_score=score)
            ranked.append((score, i, inst, state))
        except (ValueError, FloatingPointError) as error:
            records[i].update(status='oracle_failed', reason=str(error))
    selected, seen = [], set()
    for score, i, inst, state in sorted(ranked, key=lambda item: (item[0], item[1])):
        if records[i]['smiles'] in seen:
            records[i]['status'] = 'duplicate'
        elif len(selected) < retained:
            selected.append((i, inst, state))
            seen.add(records[i]['smiles'])
    frozen = []
    for _, _, state in selected:
        mapping = {atom.GetAtomMapNum(): atom.GetIdx() for atom in state.molecule.GetAtoms()}
        frozen.append([mapping[key] for key in state.fixed_positions])
    relaxation = cfg.get('stage5_relax_config', {})
    relax_args = dict(
        n_steps=int(relaxation.get('relax_steps', 30)) if relaxation.get('enabled', True) else 0,
        step_size=float(relaxation.get('step_size', .004)),
        property_tolerance=float(relaxation.get('property_tolerance', 0.)),
        energy_tolerance=float(relaxation.get('energy_tolerance', 0.)),
        coordinate_bounds=settings.get('coordinate_bounds'))
    target_bounds = dict(zip(intervals.names, intervals.bounds))
    if relaxation.get('batch_refinement', True):
        results = relax_conformers_iso_property([inst for _, inst, _ in selected], ff, oracle,
                                               target_bounds, frozen_indices=frozen, **relax_args)
    else:
        results = []
        for (_, inst, _), fixed in zip(selected, frozen):
            try:
                results.append(relax_conformer_iso_property(inst, ff, oracle, target_bounds,
                                                           frozen_indices=fixed, **relax_args))
            except (ValueError, FloatingPointError) as error:
                results.append((None, {'reason': str(error)}))
    exported, qualified, traces = [], [], []
    for (i, inst, state), fixed, (out, info) in zip(selected, frozen, results):
        if out is None:
            records[i].update(status='refinement_failed', reason=info['reason'])
            continue
        np.testing.assert_array_equal(out.coords[fixed], inst.coords[fixed])
        mol = Chem.Mol(out.graph)
        mol.GetConformer().SetPositions(out.coords.astype(float))
        passed = bool(info['interval_hit'] and info['final_force_eva'] <= inference['max_rms_force_eva'])
        for key, value in {'CandidateID': str(i), 'PredictedValue': json.dumps(info['y_pred']),
                           'JointSuccess': str(passed), 'FixedAtomMaps': json.dumps(list(state.fixed_positions)),
                           'AttachmentActions': json.dumps(state.actions)}.items():
            mol.SetProp(key, value)
        records[i].update(status='refined', joint_success=passed, y_pred=info['y_pred'],
                          final_force=info['final_force_eva'], initial_force=info['init_force_eva'],
                          frozen_coordinate_shift_A=0.)
        exported.append(mol)
        if passed:
            qualified.append(mol)
        traces.append({'candidate_id': i, **info})
    write_precise_sdf(output/'generated.sdf', exported)
    write_precise_sdf(output/'qualified.sdf', qualified)
    reloaded = list(Chem.SDMolSupplier(str(output/'generated.sdf'), removeHs=False)) if exported else []
    if reloaded:
        rechecked, _ = oracle.predict_3d_mols(reloaded)
        np.testing.assert_allclose(np.asarray(rechecked).reshape(len(exported), -1),
                                   [json.loads(m.GetProp('PredictedValue')) for m in exported], atol=1e-5, rtol=1e-5)
    pd.DataFrame(records).to_csv(output/'candidates.csv', index=False)
    (output/'proposal_records.json').write_text(json.dumps(records, indent=2)+'\n')
    (output/'relaxation_traces.json').write_text(json.dumps(traces, indent=2)+'\n')
    summary = {'backend': 'scaffold_graph_growth', 'attempted': count,
               'raw_valid': sum(row['raw_valid'] for row in records),
               'domain_eligible': len(proposals), 'external_ranking_calls': min(len(proposals), budget),
               'retained': len(exported), 'qualified': len(qualified), 'joint_success_rate': len(qualified)/count,
               'coordinate_precision': 'float64 world coordinates, fixed scaffold positions unchanged',
               'export_prediction_recheck_passed': True, 'elapsed_seconds': time.perf_counter()-started,
               'scope': 'Explicit fragment attachments; no learned octree scaffold conditioning or synthesis guarantee'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
