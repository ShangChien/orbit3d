import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from orbit3d.constraints import TargetPoints
from orbit3d.generator.molecular_state import rdkit_mol_to_mol3d_instance
from orbit3d.generator.point_task_feedback import evaluate_point_task


def test_raw_feedback_preserves_coordinates_and_uses_continuous_point_distance():
    mol=Chem.AddHs(Chem.MolFromSmiles('CC'))
    AllChem.EmbedMolecule(mol,randomSeed=42)
    instance=rdkit_mol_to_mol3d_instance(mol,np.empty(0),center=False)
    coordinates=instance.coords.copy()
    class Task:
        def predict(self, molecules):
            np.testing.assert_array_equal(molecules[0].GetConformer().GetPositions(),coordinates)
            return np.array([[-6.5,-1.2]])
    point=TargetPoints.parse({'homo':-7.},{'homo':.25},['homo','lumo'])
    row=evaluate_point_task([(3,instance)],Task(),point)[3]
    assert row['evaluation_status']=='completed'
    assert row['point_distance']==4 and row['reward']==1/5
    assert not row['property_hit']
    assert not any('force' in key or 'energy' in key for key in row)
    np.testing.assert_array_equal(instance.coords,coordinates)


def test_feedback_rejects_geometry_that_no_longer_matches_its_graph():
    mol=Chem.AddHs(Chem.MolFromSmiles('CC'))
    AllChem.EmbedMolecule(mol,randomSeed=42)
    instance=rdkit_mol_to_mol3d_instance(mol,np.empty(0),center=False)
    instance.coords[0] += 20
    class Task:
        def predict(self,molecules):
            assert not molecules
            return np.empty((0,2))
    point=TargetPoints.parse({'homo':-7.},{'homo':.25},['homo','lumo'])
    row=evaluate_point_task([(0,instance)],Task(),point)[0]
    assert row['valid'] is False and row['evaluation_status']=='invalid_structure'
