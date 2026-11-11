"""Pretrained Uni-3DAR backend and direct SDF-to-octree training data."""

from pathlib import Path
from copy import copy
import hashlib
import json
import numpy as np
import torch
from rdkit import Chem
from .octree_model import ConditionalUni3DAR, build_default_uni3dar_args, AtomFeatDict, GridDataset
from .octree_tokenizer import ATOM_TYPES, Mol3DInstance, mol3d_to_validated_rdkit_mol, mol_to_clean_smiles


class OctreeDataset:
    def __init__(self, molecules, model, training=True, seed=42):
        self.molecules = molecules
        self.allowed_elements = set(model.args.allow_atoms.split(','))
        args = copy(model.args)
        if not training:
            args.random_rotation_prob = 0.0
        self.grid = GridDataset(self, seed, model.dictionary, "atom_type", "atom_pos", training, args)
        self.grid.set_epoch(1)

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, index):
        mol = self.molecules[index]
        if any(a.GetSymbol() not in self.allowed_elements for a in mol.GetAtoms()):
            raise ValueError(f'Record {index} has elements outside the generator contract')
        return {
            "uid": index,
            "data_type": "molecule",
            "atom_type": [a.GetSymbol() for a in mol.GetAtoms()],
            "atom_pos": mol.GetConformer().GetPositions(),
        }

    def collate(self, indices):
        items = []
        for index in indices:
            item = self.grid[int(index)]
            # A property label belongs to the complete molecule. Upstream can
            # sample one atom per occupied cell; reject any such atom loss.
            expected = self.molecules[int(index)].GetNumAtoms()
            decoded = ((item['decoder_level'] == 0) & item['decoder_is_second_atom']).sum().item()
            if item['cutoff_atom_count'] or decoded != expected:
                raise ValueError(f'Record {index} lost atoms during octree encoding: {expected} -> {decoded}')
            items.append(item)
        return self.grid.collater(items)


def build_octree(device="cuda:0", targets=2, batch_size=16, condition_version=2, representation_config=None):
    args = build_default_uni3dar_args(batch_size)
    args.condition_version = condition_version
    if representation_config is not None:
        if 'merge_level' not in representation_config or set(representation_config) - {'merge_level', 'allowed_elements'}:
            raise ValueError('Grid migration currently changes tree depth only, at fixed physical resolution')
        depth = representation_config['merge_level']
        if not isinstance(depth, int) or not 6 <= depth <= 8:
            raise ValueError('Supported migrated octree depths are 6 through 8')
        args.merge_level = depth
        elements = representation_config.get('allowed_elements', args.allow_atoms.split(','))
        table = Chem.GetPeriodicTable()
        if not elements or len(set(elements)) != len(elements) or any(table.GetAtomicNumber(s) <= 0 for s in elements):
            raise ValueError('Allowed generator elements must be unique atomic symbols')
        args.allow_atoms = ','.join(elements)
    return ConditionalUni3DAR(args, AtomFeatDict(args), num_props=targets).to(device)


def checkpoint_contract(model, names, units, mean, scale):
    if len(names) != len(units) or len(set(names)) != len(names):
        raise ValueError('Checkpoint targets and units must have a unique, matching order')
    mean, scale = np.asarray(mean), np.asarray(scale)
    if mean.shape != (len(names),) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid target scaler in octree checkpoint')
    semantics = {'target_cols': list(names), 'target_units': list(units), 'mean': mean.tolist(), 'scale': scale.tolist(),
                 'condition_version': model.condition_version,
                 'token_dictionary': model.dictionary.atom_type,
                 'grid_len_A': model.args.grid_len, 'xyz_resolution_A': model.args.xyz_resolution,
                 'merge_level': model.args.merge_level, 'allowed_elements': model.args.allow_atoms.split(',')}
    digest = hashlib.sha256(json.dumps(semantics, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'schema_version': 1, 'semantics': semantics, 'sha256': digest}


def load_pretrained(path, device="cuda:0", targets=2, batch_size=16, condition_version=2):
    model = build_octree(device, targets, batch_size, condition_version)
    data = torch.load(path, map_location="cpu", weights_only=False)
    state = data["ema"]["params"]
    incompatible = model.load_state_dict(state, strict=False)
    expected = ("interval_conditioner.", "prop_understanding_head.")
    missing = [k for k in incompatible.missing_keys if not k.startswith(expected)]
    if missing or incompatible.unexpected_keys:
        raise ValueError(f"Pretrained model mismatch: {missing}, {incompatible.unexpected_keys}")
    return model.eval()


def load_octree_checkpoint(path, device="cuda:0", batch_size=16):
    state = torch.load(path, map_location="cpu", weights_only=False)
    versions = {"uni3dar_octree_v1": 1, "uni3dar_octree_v2": 2, "uni3dar_octree_v3": 3}
    if state["backend"] not in versions:
        raise ValueError("Unknown octree checkpoint contract")
    model = build_octree(device, len(state["target_cols"]), batch_size, versions[state["backend"]], state.get('representation_config'))
    units = state.get('target_units', ['unspecified'] * len(state['target_cols']))
    contract = checkpoint_contract(model, state['target_cols'], units, state['mean'], state['scale'])
    if 'contract' in state and state['contract'] != contract:
        raise ValueError('Octree checkpoint tokenizer, grid, target units or scaler contract mismatch')
    if not torch.equal(state['model_state_dict']['interval_conditioner.freqs'].cpu(), model.interval_conditioner.freqs.cpu()):
        raise ValueError('Condition frequencies differ from the declared version; migrate and retrain explicitly')
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.eval(), state


def interval_arrays(intervals, mean, scale, condition_version):
    bounds = intervals.condition(mean, scale).reshape(len(mean), 4)
    if condition_version == 1:
        if np.any(bounds[:, 2] != bounds[:, 3]):
            raise ValueError("One-sided intervals require a v2 conditioner trained with bound masks")
        active = bounds[:, 2].astype(bool)
    else:
        active = bounds[:, 2:].astype(bool)
    return bounds[:, 0], bounds[:, 1], active


def required_condition_elements(state):
    """Carry unfinished element supervision across consecutive checkpoint migrations."""
    condition = state.get('condition_migration', {})
    representation = state.get('representation_migration', {})
    required = set(condition.get('required_elements', []))
    if condition.get('requires_condition_adaptation') or representation.get('requires_adaptation'):
        required.update(set(representation.get('new_elements', [])) - set(representation.get('old_elements', [])))
    return required


def trajectory_to_instance(trajectory, spatial_box=None, element_table=tuple(ATOM_TYPES)):
    symbols = trajectory["atom_symbols"]
    types = np.array([element_table.index(s) for s in symbols], dtype=np.int64)
    coords = np.asarray(trajectory["positions_np"], dtype=np.float32)
    coords = spatial_box.to_world(coords) if spatial_box else coords - coords.mean(0)
    inst = Mol3DInstance(coords=coords, atom_types=types, element_table=tuple(element_table))
    molecule = mol3d_to_validated_rdkit_mol(inst)
    inst.graph = molecule
    inst.smiles = mol_to_clean_smiles(molecule)
    return inst
