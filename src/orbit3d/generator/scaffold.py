"""Explicit graph attachments that preserve a supplied scaffold and its coordinates.

This is a graph-constrained proposal backend, not a claim that the pretrained
unconditional octree has learned scaffold completion. Every bond is an action;
distance-based bond inference is never used to add scaffold connections.
"""

from dataclasses import dataclass
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .octree_tokenizer import rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol


@dataclass(frozen=True)
class OpenPort:
    atom_map: int
    replaceable_hydrogen_maps: tuple[int, ...]
    allowed_orders: tuple[int, ...] = (1,)


def _maps(molecule):
    return {a.GetAtomMapNum(): a.GetIdx() for a in molecule.GetAtoms()}


def _rotation(source, target):
    source, target = source / np.linalg.norm(source), target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.dot(source, target))
    if cosine < -1 + 1e-10:
        axis = np.eye(3)[np.argmin(abs(source))]
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew / max(1 + cosine, 1e-10)


class ScaffoldState:
    def __init__(self, molecule, ports, fixed_positions=None, scaffold_bonds=None, actions=()):
        self.molecule = Chem.Mol(molecule)
        self.ports = {p.atom_map: p for p in ports}
        ids = _maps(self.molecule)
        if 0 in ids or len(ids) != self.molecule.GetNumAtoms():
            raise ValueError("Every scaffold atom needs a unique positive atom map")
        self.fixed_positions = (
            fixed_positions
            if fixed_positions is not None
            else {
                key: self.molecule.GetConformer().GetPositions()[index].copy() for key, index in ids.items()
            }
        )
        self.scaffold_bonds = (
            scaffold_bonds
            if scaffold_bonds is not None
            else {
                tuple(
                    sorted([b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()])
                ): b.GetBondType()
                for b in self.molecule.GetBonds()
            }
        )
        self.actions = tuple(actions)
        for port in self.ports.values():
            if port.atom_map not in ids or len(set(port.replaceable_hydrogen_maps)) != len(
                port.replaceable_hydrogen_maps
            ):
                raise ValueError("Invalid attachment port")
            for h in port.replaceable_hydrogen_maps:
                if (
                    h not in ids
                    or self.molecule.GetAtomWithIdx(ids[h]).GetAtomicNum() != 1
                    or self.molecule.GetBondBetweenAtoms(ids[h], ids[port.atom_map]) is None
                ):
                    raise ValueError("A replaceable hydrogen must be bonded to its port")
            if any(
                order not in (1, 2, 3) or order > len(port.replaceable_hydrogen_maps)
                for order in port.allowed_orders
            ):
                raise ValueError("Port bond orders exceed its explicit hydrogen capacity")

    @classmethod
    def from_molecule(cls, molecule, open_atom_indices):
        """Open one single-bond substitution at each specified input atom index."""
        mol = Chem.AddHs(Chem.Mol(molecule), addCoords=True)
        if not mol.GetNumConformers():
            raise ValueError("A scaffold must include its fixed 3D coordinates")
        next_map = max(a.GetAtomMapNum() for a in mol.GetAtoms()) + 1
        for atom in mol.GetAtoms():
            if not atom.GetAtomMapNum():
                atom.SetAtomMapNum(next_map)
                next_map += 1
        ports = []
        for index in open_atom_indices:
            atom = mol.GetAtomWithIdx(index)
            hydrogens = [a.GetAtomMapNum() for a in atom.GetNeighbors() if a.GetAtomicNum() == 1]
            if not hydrogens:
                raise ValueError("Opening this atom requires replacing a hydrogen")
            ports.append(OpenPort(atom.GetAtomMapNum(), (min(hydrogens),)))
        return cls(mol, ports)

    def _port_hydrogens(self, atom_map, order):
        if atom_map not in self.ports:
            raise ValueError("Attachment at a non-open scaffold site")
        port = self.ports[atom_map]
        if order not in port.allowed_orders:
            raise ValueError("Bond order not allowed at this port")
        return port.replaceable_hydrogen_maps[:order]

    def _finish(self, combined, removals, connections, action, optimize=True):
        rw = Chem.RWMol(combined)
        for key in removals:
            rw.RemoveAtom(_maps(rw)[key])
        ids = _maps(rw)
        bond_types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        for left, right, order in connections:
            if rw.GetBondBetweenAtoms(ids[left], ids[right]) is not None:
                raise ValueError("Connection already exists")
            rw.AddBond(ids[left], ids[right], bond_types[order])
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        fixed = {k: v for k, v in self.fixed_positions.items() if k not in removals}
        if optimize:
            properties = AllChem.MMFFGetMoleculeProperties(mol)
            if properties is None:
                raise ValueError("Fragment geometry lacks MMFF parameters")
            ff = AllChem.MMFFGetMoleculeForceField(mol, properties)
            for key in fixed:
                ff.AddFixedPoint(_maps(mol)[key])
            ff.Initialize()
            ff.Minimize(maxIts=300)
        coords = mol.GetConformer().GetPositions()
        ids = _maps(mol)
        for key, position in fixed.items():
            if not np.array_equal(coords[ids[key]], position):
                raise ValueError("Scaffold coordinates changed")
        for (left, right), kind in self.scaffold_bonds.items():
            if left in ids and right in ids:
                bond = mol.GetBondBetweenAtoms(ids[left], ids[right])
                if bond is None or bond.GetBondType() != kind:
                    raise ValueError("Scaffold bond identity changed")
        inst = rdkit_mol_to_mol3d_instance(mol, np.empty(0), center=False)
        mol3d_to_validated_rdkit_mol(inst)
        ports = []
        for port in self.ports.values():
            remaining = tuple(h for h in port.replaceable_hydrogen_maps if h not in removals)
            orders = tuple(order for order in port.allowed_orders if order <= len(remaining))
            if orders:
                ports.append(OpenPort(port.atom_map, remaining, orders))
        return ScaffoldState(mol, ports, fixed, self.scaffold_bonds, self.actions + (action,))

    def attach(self, fragment_smiles, port_map, fragment_atom=0, order=1, seed=42, torsion=0.0):
        removals = self._port_hydrogens(port_map, order)
        fragment = Chem.MolFromSmiles(fragment_smiles)
        if fragment is None or len(Chem.GetMolFrags(fragment)) != 1:
            raise ValueError("Attachment fragment must be a connected molecule")
        fragment = Chem.AddHs(fragment)
        if AllChem.EmbedMolecule(fragment, randomSeed=seed):
            raise ValueError("Fragment embedding failed")
        anchor = fragment.GetAtomWithIdx(fragment_atom)
        hs = [a.GetIdx() for a in anchor.GetNeighbors() if a.GetAtomicNum() == 1]
        if len(hs) < order:
            raise ValueError("Fragment anchor lacks hydrogen capacity")
        offset = max(_maps(self.molecule))
        for i, atom in enumerate(fragment.GetAtoms()):
            atom.SetAtomMapNum(offset + i + 1)
        ids = _maps(self.molecule)
        positions = self.molecule.GetConformer().GetPositions()
        origin = positions[ids[port_map]]
        direction = positions[ids[removals[0]]] - origin
        direction /= np.linalg.norm(direction)
        xyz = fragment.GetConformer().GetPositions()
        rotation = _rotation(xyz[hs[0]] - xyz[fragment_atom], -direction)
        cross = np.array(
            [
                [0, -direction[2], direction[1]],
                [direction[2], 0, -direction[0]],
                [-direction[1], direction[0], 0],
            ]
        )
        twist = np.eye(3) + np.sin(torsion) * cross + (1 - np.cos(torsion)) * (cross @ cross)
        table = Chem.GetPeriodicTable()
        bond_length = (
            table.GetRcovalent(self.molecule.GetAtomWithIdx(ids[port_map]).GetAtomicNum())
            + table.GetRcovalent(anchor.GetAtomicNum())
        ) * {1: 1.0, 2: 0.88, 3: 0.80}[order]
        xyz = (xyz - xyz[fragment_atom]) @ (twist @ rotation).T + origin + direction * bond_length
        fragment.GetConformer().SetPositions(xyz)
        right = anchor.GetAtomMapNum()
        combined = Chem.CombineMols(self.molecule, fragment)
        action = {
            "type": "attach",
            "port_map": port_map,
            "fragment": fragment_smiles,
            "fragment_atom": fragment_atom,
            "new_anchor_map": right,
            "bond_order": order,
            "removed_hydrogen_maps": list(removals) + [offset + i + 1 for i in hs[:order]],
            "seed": seed,
            "torsion": torsion,
        }
        return self._finish(combined, action["removed_hydrogen_maps"], [(port_map, right, order)], action)

    def open_new_ports(self, atom_maps):
        """Explicitly enable single-bond growth on previously added fragment atoms."""
        ids = _maps(self.molecule)
        ports = list(self.ports.values())
        for key in atom_maps:
            if key in self.fixed_positions or key in self.ports:
                raise ValueError("Only newly added, unopened fragment atoms can be enabled")
            hydrogens = [
                a.GetAtomMapNum()
                for a in self.molecule.GetAtomWithIdx(ids[key]).GetNeighbors()
                if a.GetAtomicNum() == 1
            ]
            if not hydrogens:
                raise ValueError("New attachment site has no replaceable hydrogen")
            ports.append(OpenPort(key, (min(hydrogens),)))
        return ScaffoldState(self.molecule, ports, self.fixed_positions, self.scaffold_bonds, self.actions)

    def close_ring(self, left_map, right_map, order=1):
        if left_map == right_map:
            raise ValueError("A ring closure needs two distinct open sites")
        removed = list(self._port_hydrogens(left_map, order)) + list(self._port_hydrogens(right_map, order))
        action = {
            "type": "ring_closure",
            "left_map": left_map,
            "right_map": right_map,
            "bond_order": order,
            "removed_hydrogen_maps": removed,
        }
        return self._finish(self.molecule, removed, [(left_map, right_map, order)], action)
