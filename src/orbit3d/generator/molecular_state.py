"""Molecular state, graph preservation and geometry validation shared by proposal backends.

Hierarchical tokenization is implemented in octree_backend and the vendored
Uni-3DAR data modules.
"""

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


@dataclass
class Mol3DInstance:
    coords: np.ndarray
    atom_types: np.ndarray
    props: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    smiles: str = ""
    graph: Optional[Chem.Mol] = field(default=None, repr=False)
    charge: int = 0
    multiplicity: int = 1
    element_table: tuple[str, ...] = tuple(ATOM_TYPES)

    @property
    def symbols(self):
        return [self.element_table[int(t)] for t in self.atom_types]

    @property
    def atomic_numbers(self):
        table = Chem.GetPeriodicTable()
        return np.array([table.GetAtomicNumber(s) for s in self.symbols], dtype=np.int64)


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
    if inst.graph is None:
        # Bond perception can leave E/Z tags on bonds that RDKit does not
        # recognize as stereogenic. Clean inferred tags before freezing the graph;
        # supplied graph stereochemistry must still be checked, never erased.
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
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
        inferred = Chem.Mol(mol)
        rdDetermineBonds.DetermineConnectivity(inferred)
        edges = lambda graph: {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in graph.GetBonds()}
        expected, observed = edges(mol), edges(inferred)
        if observed != expected:
            raise ValueError(f'connectivity_geometry: coordinate edges differ (added={sorted(observed-expected)}, removed={sorted(expected-observed)})')
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
