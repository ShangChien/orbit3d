import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from loom.generator.scaffold import ScaffoldState, OpenPort


def geometry(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def identity(mol):
    mol = Chem.Mol(mol)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


@pytest.mark.parametrize(
    "smiles,site,expected",
    [
        ("c1ccccc1", 0, "Cc1ccccc1"),
        ("CC=O", 0, "CCC=O"),
        ("c1ccncc1", 0, "Cc1ccncc1"),
    ],
)
def test_explicit_attachment_preserves_every_surviving_scaffold_atom(smiles, site, expected):
    state = ScaffoldState.from_molecule(geometry(smiles), [site])
    before = state.molecule.GetConformer().GetPositions().copy()
    port = next(iter(state.ports))
    result = state.attach("C", port)
    assert identity(result.molecule) == Chem.MolToSmiles(Chem.MolFromSmiles(expected))
    ids = {a.GetAtomMapNum(): a.GetIdx() for a in result.molecule.GetAtoms()}
    for key, pos in result.fixed_positions.items():
        np.testing.assert_array_equal(result.molecule.GetConformer().GetPositions()[ids[key]], pos)
    np.testing.assert_array_equal(state.molecule.GetConformer().GetPositions(), before)
    assert not result.ports
    with pytest.raises(ValueError, match="non-open"):
        result.attach("C", port)


def test_multiple_ports_and_dynamic_fragment_growth_are_explicit():
    state = ScaffoldState.from_molecule(geometry("c1ccccc1"), [0, 3])
    first, second = state.ports
    grown = state.attach("C", first).attach("C", second)
    assert identity(grown.molecule) == "Cc1ccc(C)cc1"
    added = grown.actions[0]["new_anchor_map"]
    grown = grown.open_new_ports([added]).attach("C", added)
    assert identity(grown.molecule) == "CCc1ccc(C)cc1"
    with pytest.raises(ValueError, match="newly added"):
        grown.open_new_ports([first])


def test_double_bond_consumes_two_hydrogens_without_implicit_repair():
    source = ScaffoldState.from_molecule(geometry("C"), [0])
    atom = source.molecule.GetAtomWithIdx(0)
    hs = tuple(a.GetAtomMapNum() for a in atom.GetNeighbors() if a.GetAtomicNum() == 1)
    source = ScaffoldState(source.molecule, [OpenPort(atom.GetAtomMapNum(), hs[:2], (2,))])
    out = source.attach("C", atom.GetAtomMapNum(), order=2)
    assert identity(out.molecule) == "C=C"
    assert out.molecule.GetNumAtoms() == 6
    assert len(out.actions[0]["removed_hydrogen_maps"]) == 4


def test_invalid_ports_and_dead_end_do_not_mutate_parent():
    state = ScaffoldState.from_molecule(geometry("CO"), [0])
    prior = Chem.MolToMolBlock(state.molecule)
    with pytest.raises(ValueError, match="not allowed"):
        state.attach("C", next(iter(state.ports)), order=2)
    with pytest.raises(ValueError, match="hydrogen capacity"):
        state.attach("C(F)(F)(F)F", next(iter(state.ports)))
    assert Chem.MolToMolBlock(state.molecule) == prior
    with pytest.raises(ValueError, match="replacing a hydrogen"):
        ScaffoldState.from_molecule(geometry("CC=O"), [2])


def test_explicit_ring_closure_consumes_both_ports_and_preserves_fixed_core():
    source = ScaffoldState.from_molecule(geometry("C"), [0])
    atom = source.molecule.GetAtomWithIdx(0)
    hs = tuple(a.GetAtomMapNum() for a in atom.GetNeighbors() if a.GetAtomicNum() == 1)
    source = ScaffoldState(source.molecule, [OpenPort(atom.GetAtomMapNum(), hs[:2], (1,))])
    chain = source.attach("CCC", atom.GetAtomMapNum())
    end = chain.actions[0]["new_anchor_map"] + 2
    chain = chain.open_new_ports([end])
    closed = chain.close_ring(atom.GetAtomMapNum(), end)
    assert identity(closed.molecule) == "C1CCC1"
    assert closed.molecule.GetRingInfo().NumRings() == 1
    assert not closed.ports


def test_existing_maps_and_world_coordinates_survive_growth():
    mol = geometry("c1ccccc1")
    mol.GetConformer().SetPositions(mol.GetConformer().GetPositions() + [10, 20, 30])
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 101)
    state = ScaffoldState.from_molecule(mol, [0])
    assert next(iter(state.ports)) == 101
    result = state.attach("C", 101)
    for key, pos in result.fixed_positions.items():
        index = next(a.GetIdx() for a in result.molecule.GetAtoms() if a.GetAtomMapNum() == key)
        np.testing.assert_array_equal(result.molecule.GetConformer().GetPositions()[index], pos)
