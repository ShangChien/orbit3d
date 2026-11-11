import pytest
from rdkit import Chem
from orbit3d.utils.molecular_identity import compare_molecular_identity


@pytest.mark.parametrize('source,recovered,status', [
    ('CC=CC', 'C/C=C/C', 'stereo_specification_added'),
    ('C/C=C/C', 'C/C=C\\C', 'specified_stereo_changed'),
    ('C/C=C/C', 'CC=CC', 'specified_stereo_changed'),
    ('FC(Cl)Br', 'F[C@H](Cl)Br', 'stereo_specification_added'),
    ('F[C@H](Cl)Br', 'F[C@@H](Cl)Br', 'specified_stereo_changed'),
    ('[13CH3]CO', 'CCO', 'constitution_changed'),
    ('CCO', 'OCC', 'exact'),
    ('CC=O', 'C=CO', 'constitution_changed'),
])
def test_coordinate_recovery_identity_distinguishes_unspecified_stereo(source, recovered, status):
    result = compare_molecular_identity(Chem.MolFromSmiles(source), Chem.MolFromSmiles(recovered))
    assert result['identity_status'] == status
    assert result['identity_match'] == (status == 'exact')
