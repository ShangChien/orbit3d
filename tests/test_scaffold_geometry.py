import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from loom.generator.scaffold import ScaffoldState
from loom.generator.scaffold_geometry import AttachmentGrid, attachment_grid, propose_attachment


def scaffold():
    molecule = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    AllChem.MMFFOptimizeMolecule(molecule)
    molecule.GetConformer().SetPositions(molecule.GetConformer().GetPositions() + [10, 20, 30])
    return ScaffoldState.from_molecule(molecule, [0])


def test_xyz_lookahead_keeps_complete_coordinate_support():
    grid = AttachmentGrid(np.array([[0, 0, 1], [0, 2, 3], [1, 1, 2]]), .01)
    assert grid.allowed_bins([1]).tolist() == [1]
    assert grid.allowed_bins([0, 2]).tolist() == [3]
    with pytest.raises(ValueError, match='Empty XYZ continuation'):
        grid.allowed_bins([1, 2])
    rng = np.random.default_rng(42)
    for _ in range(20):
        _, record = grid.sample(rng)
        assert tuple(record['cells']) in {(0, 0, 1), (0, 2, 3), (1, 1, 2)}


def test_local_attachment_preserves_world_frame_and_records_joint_grid_choice():
    source = scaffold()
    original = source.molecule.GetConformer().GetPositions().copy()
    result, trace = propose_attachment(source, 'C', next(iter(source.ports)), seed=2026)
    assert result is not None and trace['status'] == 'completed'
    assert 0 < trace['catalog_size'] <= 512 * 3
    assert len(trace['attempts']) <= 12
    selected = trace['attempts'][-1]
    np.testing.assert_allclose(np.array(selected['cells']) * .01, selected['anchor_position_A'], atol=0, rtol=0)
    maps = {atom.GetAtomMapNum(): atom.GetIdx() for atom in result.molecule.GetAtoms()}
    for key, position in result.fixed_positions.items():
        np.testing.assert_array_equal(result.molecule.GetConformer().GetPositions()[maps[key]], position)
    np.testing.assert_array_equal(source.molecule.GetConformer().GetPositions(), original)


def test_empty_attachment_region_and_fragment_dead_end_are_bounded(monkeypatch):
    source = scaffold()
    port = next(iter(source.ports))
    with pytest.raises(ValueError, match='Empty local attachment catalog'):
        attachment_grid(source, port, 'C', bounds=[[0, 0, 0], [1, 1, 1]])
    def invalid(*args, **kwargs):
        raise ValueError('Whole fragment collision after an admissible anchor')
    monkeypatch.setattr(source, 'attach', invalid)
    result, trace = propose_attachment(source, 'CC', port, max_attempts=3)
    assert result is None and trace['status'] == 'failed'
    assert len(trace['attempts']) == 3
    assert all(row['status'] == 'rejected' for row in trace['attempts'])


def test_explicit_graph_attachment_preserves_boron_element_identity():
    source = scaffold()
    result, trace = propose_attachment(source, 'B', next(iter(source.ports)), seed=2026)
    assert result is not None and trace['status'] == 'completed'
    assert sum(atom.GetSymbol() == 'B' for atom in result.molecule.GetAtoms()) == 1
