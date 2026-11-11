"""Pretrained Uni-3DAR backend and direct SDF-to-octree training data."""

from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .octree_model import ConditionalUni3DAR, build_default_uni3dar_args, AtomFeatDict, GridDataset
from .octree_tokenizer import ATOM_TYPES, Mol3DInstance, mol3d_to_validated_rdkit_mol, mol_to_clean_smiles


class OctreeDataset:
    def __init__(self, molecules, model, training=True):
        self.molecules = molecules
        self.grid = GridDataset(self, 42, model.dictionary, "atom_type", "atom_pos", training, model.args)
        self.grid.set_epoch(1)

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, index):
        mol = self.molecules[index]
        return {
            "uid": index,
            "data_type": "molecule",
            "atom_type": [a.GetSymbol() for a in mol.GetAtoms()],
            "atom_pos": mol.GetConformer().GetPositions(),
        }

    def collate(self, indices):
        return self.grid.collater([self.grid[int(i)] for i in indices])


def build_octree(device="cuda:0", targets=2, batch_size=16):
    args = build_default_uni3dar_args(batch_size)
    return ConditionalUni3DAR(args, AtomFeatDict(args), num_props=targets).to(device)


def load_pretrained(path, device="cuda:0", targets=2, batch_size=16):
    model = build_octree(device, targets, batch_size)
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
    if state["backend"] != "uni3dar_octree_v1":
        raise ValueError("Expected a uni3dar_octree_v1 checkpoint")
    model = build_octree(device, len(state["target_cols"]), batch_size)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.eval(), state


def trajectory_to_instance(trajectory):
    symbols = trajectory["atom_symbols"]
    types = np.array([ATOM_TYPES.index(s) for s in symbols], dtype=np.int64)
    coords = np.asarray(trajectory["positions_np"], dtype=np.float32)
    inst = Mol3DInstance(coords=coords - coords.mean(0), atom_types=types)
    molecule = mol3d_to_validated_rdkit_mol(inst)
    inst.graph = molecule
    inst.smiles = mol_to_clean_smiles(molecule)
    return inst
