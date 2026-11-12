"""OE62 PBE+vdW source geometries with PBE0/vacuum/Tier2 orbital labels."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from .molecular_identity import compare_molecular_identity

THEORY = 'OE62:PBE0/vacuum/Tier2/tight//PBE+vdW/vacuum'


def connectivity_signature(molecule):
    graph = Chem.RWMol()
    for atom in molecule.GetAtoms():
        bare = Chem.Atom(atom.GetAtomicNum())
        bare.SetNoImplicit(True)
        graph.AddAtom(bare)
    for bond in molecule.GetBonds():
        graph.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return Chem.MolToSmiles(graph.GetMol(), canonical=True)


def adapt_record(row, allowed_elements, max_atoms=80):
    """Recover bonds in XYZ order; never embed, optimize, reorder or add atoms."""
    occupied = row['energies_occ_pbe0_vac_tier2']
    virtual = row['energies_unocc_pbe0_vac_tier2']
    if not isinstance(occupied, list) or not occupied or not isinstance(virtual, list) or not virtual:
        raise ValueError('missing_pbe0_frontier_labels')
    homo, lumo = float(occupied[-1]), float(virtual[0])
    if not np.isfinite([homo, lumo]).all() or homo > lumo:
        raise ValueError('invalid_frontier_labels')
    source = Chem.MolFromSmiles(row['canonical_smiles'].strip())
    if source is None:
        raise ValueError('invalid_source_smiles')
    if Chem.GetFormalCharge(source) or any(a.GetNumRadicalElectrons() for a in source.GetAtoms()):
        raise ValueError('unsupported_electronic_state')
    if len(Chem.GetMolFrags(source)) != 1:
        raise ValueError('disconnected_source')
    mol = Chem.MolFromXYZBlock(row['xyz_pbe_relaxed'])
    if mol is None:
        raise ValueError('invalid_source_xyz')
    if mol.GetNumAtoms() != int(row['number_of_atoms']) or mol.GetNumAtoms() != Chem.AddHs(source).GetNumAtoms():
        raise ValueError('source_atom_count_mismatch')
    if mol.GetNumAtoms() > max_atoms:
        raise ValueError('size_limit')
    if {a.GetSymbol() for a in mol.GetAtoms()} - set(allowed_elements):
        raise ValueError('unsupported_elements')
    coords = mol.GetConformer().GetPositions().copy()
    if not np.isfinite(coords).all():
        raise ValueError('nonfinite_coordinates')
    # The source already supplies bond orders. Match its topology to XYZ
    # connectivity instead of searching exponentially over possible valences.
    rdDetermineBonds.DetermineConnectivity(mol)
    template = Chem.AddHs(source)
    if mol.GetNumBonds() != template.GetNumBonds():
        raise ValueError('source_connectivity_mismatch')
    if connectivity_signature(mol) != connectivity_signature(template):
        raise ValueError('source_connectivity_mismatch')
    query = Chem.RWMol()
    for atom in template.GetAtoms():
        query.AddAtom(Chem.AtomFromSmarts(f'[#{atom.GetAtomicNum()};D{atom.GetDegree()}]'))
    for bond in template.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
        query.ReplaceBond(query.GetNumBonds()-1, Chem.BondFromSmarts('~'))
    mapping = mol.GetSubstructMatch(query.GetMol())
    if len(mapping) != mol.GetNumAtoms():
        raise ValueError('source_connectivity_mismatch')
    inverse = [0] * len(mapping)
    for source_index, xyz_index in enumerate(mapping):
        inverse[xyz_index] = source_index
    recovered = Chem.RenumberAtoms(template, inverse)
    recovered.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    Chem.RemoveStereochemistry(recovered)
    Chem.AssignStereochemistryFrom3D(recovered)
    mol = recovered
    identity = compare_molecular_identity(source, mol)
    if not identity['source_specified_stereo_preserved']:
        raise ValueError('source_identity:' + identity['identity_status'])
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('recovered_radical')
    np.testing.assert_array_equal(coords, mol.GetConformer().GetPositions())
    mol.SetProp('SourceID', str(row['refcode_csd']))
    mol.SetProp('SourceSMILES', row['canonical_smiles'].strip())
    mol.SetProp('SourceInChI', row['inchi'].strip())
    mol.SetProp('GeometryOrigin', 'OE62 xyz_pbe_relaxed; unchanged source coordinates')
    mol.SetProp('LabelTheory', THEORY)
    mol.SetProp('TargetUnits', 'eV')
    mol.SetProp('homo', str(homo))
    mol.SetProp('lumo', str(lumo))
    return mol
