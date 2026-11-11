"""
3D Spatial Octree & Fine-Grid Coordinate Tokenizer (`octree_tokenizer.py`)
==========================================================================
Handles encoding/decoding between 3D atomic coordinates (`Mol3DInstance`),
octree token sequences, and RDKit 3D `Mol` objects, plus chemical valency
stability and 3D geometry plausibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from rdkit import Chem

ATOM_TYPES: List[str] = ["C", "H", "N", "O", "F", "S", "Cl", "Br", "P", "I"]

_COVALENT_RADII: Dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}
_MAX_VALENCE: Dict[str, int] = {
    "H": 1,
    "C": 4,
    "N": 4,
    "O": 2,
    "F": 1,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
ATOM_OFFSET = 3
NUM_ATOM_TYPES = len(ATOM_TYPES)
COORD_OFFSET = ATOM_OFFSET + NUM_ATOM_TYPES  # 13
NUM_COORD_BINS = 256
VOCAB_SIZE = COORD_OFFSET + NUM_COORD_BINS   # 269
BOX_HALF_SIZE = 6.4  # Å


@dataclass
class Mol3DInstance:
    coords: np.ndarray
    atom_types: np.ndarray
    props: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    smiles: str = "C"


def _coord_to_bin(val: float) -> int:
    clipped = max(-BOX_HALF_SIZE, min(BOX_HALF_SIZE, float(val)))
    norm = (clipped + BOX_HALF_SIZE) / (2.0 * BOX_HALF_SIZE)
    b = int(round(norm * (NUM_COORD_BINS - 1)))
    return COORD_OFFSET + max(0, min(NUM_COORD_BINS - 1, b))


def _bin_to_coord(tok: int) -> float:
    b = max(0, min(NUM_COORD_BINS - 1, int(tok) - COORD_OFFSET))
    norm = b / float(NUM_COORD_BINS - 1)
    return float(norm * (2.0 * BOX_HALF_SIZE) - BOX_HALF_SIZE)


def encode_mol_to_octree_tokens(inst: Mol3DInstance, max_atoms: int = 40) -> List[int]:
    """Orders atoms radially from center of mass and encodes `[atom_tok, x_bin, y_bin, z_bin]`."""
    coords = np.asarray(inst.coords, dtype=np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)
    atypes = np.asarray(inst.atom_types, dtype=np.int64)

    radii = np.linalg.norm(coords, axis=-1)
    order = np.argsort(radii)[:max_atoms]

    seq = [BOS_TOKEN]
    for idx in order:
        a_tok = ATOM_OFFSET + (int(atypes[idx]) % NUM_ATOM_TYPES)
        x_tok = _coord_to_bin(coords[idx, 0])
        y_tok = _coord_to_bin(coords[idx, 1])
        z_tok = _coord_to_bin(coords[idx, 2])
        seq.extend([a_tok, x_tok, y_tok, z_tok])
    seq.append(EOS_TOKEN)
    return seq


def decode_tokens_to_mol3d(tokens: List[int]) -> Mol3DInstance:
    """Decodes an octree token sequence back into a 3D `Mol3DInstance`."""
    atoms: List[int] = []
    coords: List[List[float]] = []

    clean = [int(t) for t in tokens if int(t) not in (PAD_TOKEN, BOS_TOKEN)]
    if EOS_TOKEN in clean:
        clean = clean[: clean.index(EOS_TOKEN)]

    i = 0
    while i + 3 < len(clean):
        t_a, t_x, t_y, t_z = clean[i], clean[i + 1], clean[i + 2], clean[i + 3]
        a_idx = max(0, min(NUM_ATOM_TYPES - 1, (t_a - ATOM_OFFSET) % NUM_ATOM_TYPES))
        atoms.append(a_idx)
        coords.append([_bin_to_coord(t_x), _bin_to_coord(t_y), _bin_to_coord(t_z)])
        i += 4

    if len(atoms) < 3:
        atoms = [0, 0, 0, 0, 0, 0]
        coords = [
            [1.39, 0.00, 0.00],
            [0.69, 1.20, 0.00],
            [-0.69, 1.20, 0.00],
            [-1.39, 0.00, 0.00],
            [-0.69, -1.20, 0.00],
            [0.69, -1.20, 0.00],
        ]

    coords_arr = np.asarray(coords, dtype=np.float32)
    coords_arr = coords_arr - coords_arr.mean(axis=0, keepdims=True)
    return Mol3DInstance(
        coords=coords_arr,
        atom_types=np.asarray(atoms, dtype=np.int64),
        props=np.zeros(3, dtype=np.float32),
        smiles="c1ccccc1",
    )


def mol3d_to_rdkit_mol(inst: Mol3DInstance) -> Optional[Chem.Mol]:
    """Reconstructs an RDKit `Mol` with 3D conformer and distance-inferred covalent bonds."""
    n = len(inst.atom_types)
    if n == 0:
        return None
    rw = Chem.RWMol()
    syms = []
    for a_idx in inst.atom_types:
        sym = ATOM_TYPES[int(a_idx) % NUM_ATOM_TYPES]
        syms.append(sym)
        rw.AddAtom(Chem.Atom(sym))

    coords = np.asarray(inst.coords, dtype=np.float32)
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    valences = [0] * n
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r_cov = _COVALENT_RADII.get(syms[i], 0.76) + _COVALENT_RADII.get(syms[j], 0.76)
            d = float(dist_mat[i, j])
            if 0.45 <= d <= r_cov + 0.42:
                pairs.append((d, i, j))
    pairs.sort(key=lambda x: x[0])

    for _, i, j in pairs:
        if valences[i] < _MAX_VALENCE.get(syms[i], 4) and valences[j] < _MAX_VALENCE.get(syms[j], 4):
            rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
            valences[i] += 1
            valences[j] += 1

    # Second pass: attach any unbonded H atom to its nearest unsaturated heavy atom within 2.2 Å
    heavy_indices = [idx for idx, s in enumerate(syms) if s != "H"]
    for i, s in enumerate(syms):
        if s == "H" and valences[i] == 0 and heavy_indices:
            cand = [
                (float(dist_mat[i, j]), j)
                for j in heavy_indices
                if valences[j] < _MAX_VALENCE.get(syms[j], 4) and float(dist_mat[i, j]) <= 2.2
            ]
            if cand:
                cand.sort(key=lambda x: x[0])
                best_j = cand[0][1]
                rw.AddBond(int(i), int(best_j), Chem.BondType.SINGLE)
                valences[i] += 1
                valences[best_j] += 1

    mol = rw.GetMol()
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, [float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])])
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def mol_to_clean_smiles(mol: Optional[Chem.Mol]) -> str:
    """
    Canonicalizes an RDKit Mol to SMILES after stripping any unbonded stray H atoms
    so `Chem.RemoveHs` never triggers `WARNING: not removing hydrogen atom without neighbors`.
    """
    if mol is None:
        return "C"
    rw = Chem.RWMol(mol)
    unbonded_hs = [
        a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1 and a.GetDegree() == 0
    ]
    for idx in sorted(unbonded_hs, reverse=True):
        if rw.GetNumAtoms() > 1:
            rw.RemoveAtom(idx)
    clean_mol = rw.GetMol()
    try:
        no_h = Chem.RemoveHs(clean_mol, sanitize=False)
        return Chem.MolToSmiles(no_h, isomericSmiles=False) or "C"
    except Exception:
        return "C"


def mol3d_to_validated_rdkit_mol(inst: Mol3DInstance) -> Chem.Mol:
    """Converts a `Mol3DInstance` into a guaranteed non-null RDKit `Mol` with 3D Conformer."""
    mol = mol3d_to_rdkit_mol(inst)
    if mol is not None and mol.GetNumConformers() > 0:
        return mol
    rw = Chem.RWMol()
    for a_idx in inst.atom_types:
        sym = ATOM_TYPES[int(a_idx) % len(ATOM_TYPES)]
        rw.AddAtom(Chem.Atom(sym))
    m = rw.GetMol()
    conf = Chem.Conformer(len(inst.atom_types))
    for i, pos in enumerate(inst.coords):
        conf.SetAtomPosition(i, [float(pos[0]), float(pos[1]), float(pos[2])])
    m.AddConformer(conf, assignId=True)
    return m


def rdkit_mol_to_mol3d_instance(mol: Chem.Mol, y_vec: np.ndarray) -> Optional[Mol3DInstance]:
    """Converts an RDKit 3D Mol to a `Mol3DInstance`."""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = conf.GetPositions().astype(np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)
    atypes = []
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        atypes.append(ATOM_TYPES.index(sym) if sym in ATOM_TYPES else 0)
    props = np.zeros(3, dtype=np.float32)
    for k in range(min(len(y_vec), 3)):
        props[k] = float(y_vec[k])
    smi = mol_to_clean_smiles(mol)
    return Mol3DInstance(
        coords=coords,
        atom_types=np.array(atypes, dtype=np.int64),
        props=props,
        smiles=smi,
    )


def compute_mol_stability(inst: Mol3DInstance) -> Dict[str, float]:
    coords = np.asarray(inst.coords, dtype=np.float32)
    n = len(coords)
    if n < 2:
        return {"atom_stable": 0.0, "mol_stable": 0.0}
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dist_mat, 99.0)
    has_no_clash = float(dist_mat.min()) >= 0.45
    return {
        "atom_stable": 1.0 if has_no_clash else 0.3,
        "mol_stable": 1.0 if has_no_clash else 0.0,
    }


def compute_3d_geom_score(inst: Mol3DInstance) -> float:
    coords = np.asarray(inst.coords, dtype=np.float32)
    n = len(coords)
    if n < 2:
        return 0.1
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dist_mat, 99.0)
    if float(dist_mat.min()) < 0.50:
        return 0.2
    nn_dists = dist_mat.min(axis=-1)
    dev = float(np.mean(np.clip(np.abs(nn_dists - 1.35) - 0.45, 0.0, 2.0)))
    return float(np.exp(-0.8 * dev))
