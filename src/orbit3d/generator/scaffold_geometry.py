"""Joint XYZ support for explicit scaffold attachments on a finite local grid.

The catalog is bounded by sampled directions and bond lengths. Every prefix has
a complete XYZ continuation within that catalog; this is not exhaustive search
of the spatial lattice, learned octree conditioning, or whole-fragment validity.
"""
from dataclasses import dataclass
import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class AttachmentGrid:
    cells: np.ndarray
    resolution: float

    def allowed_bins(self, prefix=()):
        if len(prefix) >= 3:
            raise ValueError('XYZ prefix is already complete')
        available = self.cells
        for axis, value in enumerate(prefix):
            available = available[available[:, axis] == value]
        values = np.unique(available[:, len(prefix)])
        if not len(values):
            raise ValueError('Empty XYZ continuation for attachment prefix')
        return values

    def sample(self, rng):
        prefix, sizes = [], []
        for _ in range(3):
            choices = self.allowed_bins(prefix)
            sizes.append(len(choices))
            prefix.append(int(rng.choice(choices)))
        return np.array(prefix) * self.resolution, {'cells': prefix, 'support_sizes_xyz': sizes}


def attachment_grid(state, port_map, element, order=1, resolution=.01,
                    direction_count=512, cone_degrees=50., bounds=None):
    if resolution <= 0 or direction_count < 1 or not 0 < cone_degrees <= 180:
        raise ValueError('Invalid attachment grid parameters')
    removed = state._port_hydrogens(port_map, order)
    ids = {atom.GetAtomMapNum(): atom.GetIdx() for atom in state.molecule.GetAtoms()}
    positions = state.molecule.GetConformer().GetPositions()
    origin = positions[ids[port_map]]
    outward = positions[ids[removed[0]]] - origin
    outward /= np.linalg.norm(outward)
    table = Chem.GetPeriodicTable()
    radius = table.GetRcovalent(table.GetAtomicNumber(element))
    bond_length = (radius + table.GetRcovalent(state.molecule.GetAtomWithIdx(ids[port_map]).GetAtomicNum())) * {1: 1., 2: .88, 3: .8}[order]
    # Fibonacci directions bound construction cost independently of lattice size.
    index = np.arange(direction_count)
    z = 1 - 2 * (index + .5) / direction_count
    phi = index * np.pi * (3 - np.sqrt(5))
    radial = np.sqrt(1 - z*z)
    directions = np.column_stack((radial*np.cos(phi), radial*np.sin(phi), z))
    directions = directions[directions @ outward >= np.cos(np.deg2rad(cone_degrees))]
    points = origin + (directions[:, None, :] * (bond_length*np.array([.96, 1., 1.04]))[None, :, None]).reshape(-1, 3)
    cells = np.unique(np.rint(points/resolution).astype(np.int64), axis=0)
    points = cells * resolution
    length = np.linalg.norm(points-origin, axis=1)
    keep = (length >= .9*bond_length) & (length <= 1.1*bond_length)
    for atom in state.molecule.GetAtoms():
        if atom.GetAtomMapNum() in set(removed) | {port_map}:
            continue
        clearance = .75 * (radius + table.GetRcovalent(atom.GetAtomicNum()))
        keep &= np.linalg.norm(points-positions[atom.GetIdx()], axis=1) >= clearance
    if bounds is not None:
        lower, upper = np.asarray(bounds)
        keep &= ((points >= lower) & (points < upper)).all(axis=1)
    cells = cells[keep]
    if not len(cells):
        raise ValueError('Empty local attachment catalog after shell, direction, collision and bounds checks')
    return AttachmentGrid(cells, resolution)


def propose_attachment(state, fragment, port_map, *, fragment_atom=0, order=1,
                       seed=42, max_attempts=12, bounds=None):
    if max_attempts < 1:
        raise ValueError('Attachment requires a positive finite proposal budget')
    molecule = Chem.MolFromSmiles(fragment)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError('Attachment fragment must be connected')
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise ValueError('Hydrogen-replacement attachment requires a closed-shell fragment')
    if molecule.GetAtomWithIdx(fragment_atom).GetTotalNumHs(includeNeighbors=True) < order:
        raise ValueError('Fragment anchor lacks hydrogen capacity')
    element = molecule.GetAtomWithIdx(fragment_atom).GetSymbol()
    grid = attachment_grid(state, port_map, element, order, bounds=bounds)
    rng = np.random.default_rng(seed)
    attempts = []
    for trial in range(max_attempts):
        position, choice = grid.sample(rng)
        torsion = float(rng.uniform(-np.pi, np.pi))
        row = {'trial': trial, **choice, 'anchor_position_A': position.tolist(), 'torsion': torsion}
        attempts.append(row)
        try:
            result = state.attach(fragment, port_map, fragment_atom, order, seed=seed,
                                  torsion=torsion, anchor_position=position, optimize=False)
            if bounds is not None:
                xyz = result.molecule.GetConformer().GetPositions()
                lower, upper = np.asarray(bounds)
                if not ((xyz >= lower) & (xyz < upper)).all():
                    raise ValueError('Completed fragment lies outside the requested spatial bounds')
            row['status'] = 'accepted'
            return result, {'status': 'completed', 'catalog_size': len(grid.cells), 'attempts': attempts}
        except ValueError as exc:
            row.update(status='rejected', reason=str(exc))
    return None, {'status': 'failed', 'reason': 'attachment proposal budget exhausted',
                  'catalog_size': len(grid.cells), 'attempts': attempts}
