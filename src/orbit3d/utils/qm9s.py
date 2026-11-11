"""Explicit-H QM9s adaptation without label reinterpretation or atom reordering."""

import importlib
import io
import pickle
from dataclasses import replace
import numpy as np
from rdkit import Chem
from ..generator.octree_tokenizer import (
    rdkit_mol_to_mol3d_instance,
    encode_mol_to_octree_tokens,
    decode_tokens_to_mol3d,
    mol3d_to_validated_rdkit_mol,
    mol_to_clean_smiles,
    ATOM_TYPES,
)


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            numeric = importlib.import_module("numpy._core.numeric")
            multiarray = importlib.import_module("numpy._core.multiarray")
        except ImportError:
            numeric = importlib.import_module("numpy.core.numeric")
            multiarray = importlib.import_module("numpy.core.multiarray")
        allowed = {("numpy", "dtype"): np.dtype, ("numpy", "ndarray"): np.ndarray}
        for prefix in ["numpy.core", "numpy._core"]:
            allowed[(prefix + ".numeric", "_frombuffer")] = numeric._frombuffer
            allowed[(prefix + ".multiarray", "_reconstruct")] = multiarray._reconstruct
            allowed[(prefix + ".multiarray", "scalar")] = multiarray.scalar
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported constructor {module}.{name}")
        return allowed[(module, name)]


def read_record(raw):
    if len(raw) > 64 * 1024**2:
        raise ValueError("Oversized LMDB record")
    result = ArrayUnpickler(io.BytesIO(raw)).load()
    if not isinstance(result, dict):
        raise ValueError("LMDB record must be a mapping")
    return result


def canonical_record(record):
    mol = Chem.MolFromSmiles(str(record.get("smi", "")))
    if mol is None:
        raise ValueError("invalid_smiles")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def adapt_record(record, max_atoms=40, backend="uni3dar_octree", element_table=tuple(ATOM_TYPES), spectrum_size=240):
    mol = Chem.MolFromSmiles(str(record.get("smi", "")))
    if mol is None:
        raise ValueError("invalid_smiles")
    if Chem.GetFormalCharge(mol) != 0:
        raise ValueError("unsupported_charge")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("radical_representation")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("disconnected")
    mol = Chem.AddHs(mol)
    if [a.GetSymbol() for a in mol.GetAtoms()] != list(record["atoms"]):
        raise ValueError("atom_order_mismatch")
    coords = np.asarray(record["coordinates"], dtype=np.float64)
    if coords.ndim == 3:
        if len(coords) != 1:
            raise ValueError("multiple_conformers_require_explicit_selection")
        coords = coords[0]
    if coords.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(coords).all():
        raise ValueError("invalid_coordinates")
    spectrum = np.asarray(record["target"], dtype=np.float32)
    if spectrum.shape != (spectrum_size,) or not np.isfinite(spectrum).all():
        raise ValueError("invalid_spectrum")
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for index, xyz in enumerate(coords):
        conf.SetAtomPosition(index, xyz.tolist())
    mol.AddConformer(conf)
    inst = rdkit_mol_to_mol3d_instance(mol, spectrum, element_table=element_table)
    if mol.GetNumAtoms() > max_atoms:
        raise ValueError("oversized_molecule")
    mol3d_to_validated_rdkit_mol(inst)
    if backend == "uni3dar_octree":
        # Preserve source coordinates. Hierarchical tokenization belongs to the
        # octree dataset, not the unrelated dense-grid baseline representation.
        inferred = mol3d_to_validated_rdkit_mol(replace(inst, graph=None))
        if mol_to_clean_smiles(inferred) != mol_to_clean_smiles(mol):
            raise ValueError("source_graph_or_stereo_mismatch")
        return inst, None
    if backend != "dense_grid_v2":
        raise ValueError(f"Unknown backend: {backend}")
    tokens = encode_mol_to_octree_tokens(inst, max_atoms)
    decoded = mol3d_to_validated_rdkit_mol(decode_tokens_to_mol3d(tokens))
    if mol_to_clean_smiles(decoded) != mol_to_clean_smiles(mol):
        raise ValueError("quantized_graph_or_stereo_mismatch")
    return inst, tokens
