import pytest
from rdkit import Chem

from loom.utils.molecular_identity import compare_molecular_identity


@pytest.mark.parametrize('source,recovered,status', [
    ('CC=CC', 'C/C=C/C', 'stereo_specification_added'),
    ('C/C=C/C', 'C/C=C\\C', 'specified_stereo_changed'),
    ('C/C=C/C', 'CC=CC', 'specified_stereo_changed'),
    ('FC(Cl)Br', 'F[C@H](Cl)Br', 'stereo_specification_added'),
    ('F[C@H](Cl)Br', 'F[C@@H](Cl)Br', 'specified_stereo_changed'),
    ('[13CH3]CO', 'CCO', 'constitution_changed'),
    ('CCO', 'OCC', 'exact'),
    ('CC=O', 'C=CO', 'constitution_changed'),
    ('N=C(Nc1ccccc1)c1ccccc1Cl', '[H]/N=C(/Nc1ccccc1)c1ccccc1Cl', 'stereo_specification_added'),
    ('[H]/N=C(/C)C=C', '[H]/N=C(\\C)C=C', 'specified_stereo_changed'),
    ('[H]/N=C(/C)C=C', 'N=C(C)C=C', 'specified_stereo_changed'),
    ('[2H]/N=C(/C)C=C', '[H]/N=C(/C)C=C', 'constitution_changed'),
    ('N=C(C)C=C', 'NC(C)C=C', 'constitution_changed'),
])
def test_coordinate_recovery_identity_distinguishes_unspecified_stereo(source, recovered, status):
    result = compare_molecular_identity(Chem.MolFromSmiles(source), Chem.MolFromSmiles(recovered))
    assert result['identity_status'] == status
    assert result['identity_match'] == (status == 'exact')


def test_explicit_and_implicit_hydrogens_do_not_change_identity():
    source = Chem.MolFromSmiles('[H]/N=C(/C)C=C')
    explicit = Chem.AddHs(source)
    before = Chem.MolToMolBlock(source)
    assert compare_molecular_identity(source, explicit)['identity_status'] == 'exact'
    assert Chem.MolToMolBlock(source) == before


def test_specified_alkene_flip_with_many_equivalent_methyl_hydrogens():
    # Topology of OE62 AXAJUF; exercise stereo rejection without enumerating H permutations.
    source=r'N#C/C(=C(\c1n(CC)nc(c1Cl)C)/OC(=O)C(C)(C)C)/c1ccc(cc1)C(C)(C)C'
    flipped=source.replace(r'C(\c1',r'C(/c1')
    assert compare_molecular_identity(Chem.MolFromSmiles(source),Chem.MolFromSmiles(flipped))['identity_status']=='specified_stereo_changed'


def test_inferred_graph_stereo_can_be_revalidated_without_changing_geometry():
    import json
    from pathlib import Path
    import numpy as np
    from loom.generator.octree_backend import trajectory_to_instance
    from loom.generator.molecular_state import mol3d_to_validated_rdkit_mol
    case=json.loads((Path(__file__).parent/'fixtures/inferred_stereo_cleanup.json').read_text())
    instance=trajectory_to_instance({'atom_symbols':case['symbols'],'positions_np':case['coordinates']},
                                    element_table=tuple(case['elements']))
    # Actual generated P=O case: DetermineBonds left a Z tag which the
    # canonical graph does not support; the old second validation rejected it.
    assert compare_molecular_identity(Chem.MolFromSmiles(case['smiles']),instance.graph)['identity_match']
    before=instance.coords.copy()
    recovered=mol3d_to_validated_rdkit_mol(instance)
    np.testing.assert_array_equal(instance.coords,before)
    assert Chem.MolToSmiles(recovered)==Chem.MolToSmiles(instance.graph)
    assert all(b.GetStereo()==Chem.BondStereo.STEREONONE for b in recovered.GetBonds())


def test_existing_bond_length_bounds_cannot_hide_coordinate_connectivity_drift():
    import json
    from pathlib import Path
    from dataclasses import replace
    import numpy as np
    from loom.generator.octree_backend import trajectory_to_instance
    from loom.generator.molecular_state import mol3d_to_validated_rdkit_mol
    case=json.loads((Path(__file__).parent/'fixtures/connectivity_drift.json').read_text())
    instance=trajectory_to_instance({'atom_symbols':case['symbols'],'positions_np':case['initial']},
                                    element_table=tuple(case['elements']))
    mol3d_to_validated_rdkit_mol(instance)
    # Actual accepted Full endpoint: its N-O distance passed the old radius
    # bounds but lost an edge under the common coordinate-connectivity rule.
    moved=replace(instance,coords=np.array(case['refined'],dtype=np.float32))
    with pytest.raises(ValueError,match='connectivity_geometry:'):
        mol3d_to_validated_rdkit_mol(moved)
