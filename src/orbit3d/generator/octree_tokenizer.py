"""Versioned dense-grid spatial representation (not hierarchical octree)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

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
VOCAB_SIZE = COORD_OFFSET + NUM_COORD_BINS  # 269
BOX_HALF_SIZE = 6.4  # Å


@dataclass
class Mol3DInstance:
    coords: np.ndarray
    atom_types: np.ndarray
    props: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    smiles: str = ""
    graph: Optional[Chem.Mol] = field(default=None, repr=False)
    charge: int = 0
    multiplicity: int = 1
    token_ids: List[int] = field(default_factory=list)
    element_table: tuple[str, ...] = tuple(ATOM_TYPES)

    @property
    def symbols(self):
        return [self.element_table[int(t)] for t in self.atom_types]

    @property
    def atomic_numbers(self):
        table = Chem.GetPeriodicTable()
        return np.array([table.GetAtomicNumber(s) for s in self.symbols], dtype=np.int64)


def _coord_to_bin(val: float) -> int:
    if not np.isfinite(val) or abs(val) > BOX_HALF_SIZE:
        raise ValueError("Coordinate outside dense_grid_v2 bounds")
    clipped = float(val)
    norm = (clipped + BOX_HALF_SIZE) / (2.0 * BOX_HALF_SIZE)
    b = int(round(norm * (NUM_COORD_BINS - 1)))
    return COORD_OFFSET + max(0, min(NUM_COORD_BINS - 1, b))


def _bin_to_coord(tok: int) -> float:
    if not COORD_OFFSET <= tok < VOCAB_SIZE:
        raise ValueError("Invalid coordinate token")
    b = int(tok) - COORD_OFFSET
    norm = b / float(NUM_COORD_BINS - 1)
    return float(norm * (2.0 * BOX_HALF_SIZE) - BOX_HALF_SIZE)


def encode_mol_to_octree_tokens(inst: Mol3DInstance, max_atoms: int = 40) -> List[int]:
    """Orders atoms radially from center of mass and encodes `[atom_tok, x_bin, y_bin, z_bin]`."""
    if tuple(inst.element_table) != tuple(ATOM_TYPES):
        raise ValueError('dense_grid_v2 requires its original element table; token offsets are unchanged')
    coords = np.asarray(inst.coords, dtype=np.float32)
    if len(coords) == 0 or len(coords) > max_atoms:
        raise ValueError("Empty or oversized molecule; no truncation allowed")
    if inst.charge != 0 or inst.multiplicity != 1:
        raise ValueError("dense_grid_v2 generation currently supports neutral singlets only")
    coords = coords - coords.mean(axis=0, keepdims=True)
    atypes = np.asarray(inst.atom_types, dtype=np.int64)

    if mol3d_to_rdkit_mol(inst) is None:
        raise ValueError("Training molecule failed chemical validation")

    radii = np.linalg.norm(coords, axis=-1)
    if coords.shape != (len(atypes), 3) or np.any((atypes < 0) | (atypes >= NUM_ATOM_TYPES)):
        raise ValueError("Invalid coordinates or atom indices")
    order = np.argsort(radii, kind="stable")

    seq = [BOS_TOKEN]
    for idx in order:
        a_tok = ATOM_OFFSET + int(atypes[idx])
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

    seq = [int(t) for t in tokens]
    if not seq or seq[0] != BOS_TOKEN or EOS_TOKEN not in seq:
        raise ValueError("Missing BOS/EOS")
    end = seq.index(EOS_TOKEN)
    if any(t != PAD_TOKEN for t in seq[end + 1 :]):
        raise ValueError("Unexpected tokens after EOS")
    clean = seq[1:end]
    if not clean or len(clean) % 4:
        raise ValueError("Incomplete or empty atom sequence")
    for i in range(0, len(clean), 4):
        t_a, t_x, t_y, t_z = clean[i : i + 4]
        if not ATOM_OFFSET <= t_a < COORD_OFFSET:
            raise ValueError("Invalid element token")
        atoms.append(t_a - ATOM_OFFSET)
        coords.append([_bin_to_coord(t_x), _bin_to_coord(t_y), _bin_to_coord(t_z)])

    coords_arr = np.asarray(coords, dtype=np.float32)
    coords_arr = coords_arr - coords_arr.mean(axis=0, keepdims=True)
    return Mol3DInstance(
        coords=coords_arr,
        atom_types=np.asarray(atoms, dtype=np.int64),
        props=np.zeros(3, dtype=np.float32),
        smiles="",
        token_ids=list(tokens),
    )


def _validated_molecule(inst: Mol3DInstance) -> Chem.Mol:
    coords = np.asarray(inst.coords, dtype=float)
    types = np.asarray(inst.atom_types)
    n = len(types)
    if n == 0 or coords.shape != (n, 3) or not np.isfinite(coords).all():
        raise ValueError('coordinates: empty, malformed or nonfinite')
    if np.any((types < 0) | (types >= len(inst.element_table))) or not np.equal(types, types.astype(int)).all():
        raise ValueError('elements: atom indices outside the representation')
    syms = inst.symbols
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    if n > 1 and distances.min() < 0.45:
        raise ValueError(f'collision: minimum separation {distances.min():.6f} A < 0.45 A')
    if inst.graph is not None:
        mol = Chem.Mol(inst.graph)
        if [a.GetSymbol() for a in mol.GetAtoms()] != syms:
            raise ValueError('graph_identity: supplied graph and coordinate element order differ')
    else:
        rw = Chem.RWMol()
        for sym in syms:
            atom = Chem.Atom(sym)
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        mol = rw.GetMol()
    mol.RemoveAllConformers()
    conf = Chem.Conformer(n)
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, xyz.tolist())
    mol.AddConformer(conf)
    if inst.graph is None:
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=int(inst.charge), allowChargedFragments=True)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f'bond_inference: {exc}') from exc
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f'valence_or_aromaticity: {exc}') from exc
    fragments = len(Chem.GetMolFrags(mol))
    if fragments != 1:
        raise ValueError(f'connectivity: {fragments} disconnected fragments')
    if Chem.GetFormalCharge(mol) != inst.charge:
        raise ValueError('charge: graph and requested total charge differ')
    electrons = sum(a.GetAtomicNum() + a.GetTotalNumHs() for a in mol.GetAtoms()) - inst.charge
    if inst.multiplicity < 1 or inst.multiplicity > electrons + 1 or (electrons - inst.multiplicity + 1) % 2:
        raise ValueError('electronic_state: multiplicity and electron-count parity differ')
    if inst.multiplicity == 1 and any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('radicals: closed-shell generation contains radical centers')
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        table = Chem.GetPeriodicTable()
        radii = [_COVALENT_RADII[s] if s in _COVALENT_RADII else table.GetRcovalent(table.GetAtomicNumber(s)) for s in (syms[i], syms[j])]
        radius = sum(radii)
        if not 0.6 * radius <= distances[i, j] <= 1.4 * radius:
            raise ValueError(f'bond_geometry: {i}-{j} ({syms[i]}-{syms[j]}) length {distances[i,j]:.6f} A')
    if inst.graph is not None:
        chiral_atoms = [(a.GetIdx(), a.GetChiralTag()) for a in mol.GetAtoms()
                       if a.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)]
        stereo_bonds = [(b.GetIdx(), b.GetStereo()) for b in mol.GetBonds()
                        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)]
        if chiral_atoms or stereo_bonds:
            geometric = Chem.Mol(mol)
            Chem.AssignStereochemistryFrom3D(geometric, replaceExistingTags=True)
            Chem.AssignStereochemistry(geometric, cleanIt=True, force=True)
            if any(geometric.GetAtomWithIdx(i).GetChiralTag() != tag for i, tag in chiral_atoms):
                raise ValueError('stereo_geometry: coordinates disagree with a specified tetrahedral center')
            if any(geometric.GetBondWithIdx(i).GetStereo() != tag for i, tag in stereo_bonds):
                raise ValueError('stereo_geometry: coordinates disagree with a specified double-bond configuration')
    mol.SetIntProp('TotalCharge', int(inst.charge))
    mol.SetIntProp('SpinMultiplicity', int(inst.multiplicity))
    return mol


def mol3d_to_rdkit_mol(inst: Mol3DInstance) -> Optional[Chem.Mol]:
    """Compatibility query returning None for chemical rejection."""
    try:
        return _validated_molecule(inst)
    except (ValueError, RuntimeError):
        return None


def mol_to_clean_smiles(mol: Optional[Chem.Mol]) -> str:
    if mol is None:
        raise ValueError("Cannot serialize invalid molecule")
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), isomericSmiles=True)


def mol3d_to_validated_rdkit_mol(inst: Mol3DInstance) -> Chem.Mol:
    return _validated_molecule(inst)


def rdkit_mol_to_mol3d_instance(mol: Chem.Mol, y_vec: np.ndarray, center=True, element_table=tuple(ATOM_TYPES), coordinate_dtype=np.float32) -> Optional[Mol3DInstance]:
    """Converts an RDKit 3D Mol to a `Mol3DInstance`."""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = conf.GetPositions().astype(coordinate_dtype)
    if center:
        coords = coords - coords.mean(axis=0, keepdims=True)
    atypes = []
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym not in element_table:
            raise ValueError(f"Unsupported generator element: {sym}")
        atypes.append(element_table.index(sym))
    props = np.asarray(y_vec, dtype=np.float32).copy()
    smi = mol_to_clean_smiles(mol)
    return Mol3DInstance(
        coords=coords,
        atom_types=np.array(atypes, dtype=np.int64),
        props=props,
        smiles=smi,
        graph=Chem.Mol(mol),
        charge=Chem.GetFormalCharge(mol),
        multiplicity=mol.GetIntProp("SpinMultiplicity") if mol.HasProp("SpinMultiplicity") else 1,
        element_table=tuple(element_table),
    )


def compute_mol_stability(inst: Mol3DInstance) -> Dict[str, float]:
    valid = mol3d_to_rdkit_mol(inst) is not None
    return {"atom_stable": float(valid), "mol_stable": float(valid)}


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
