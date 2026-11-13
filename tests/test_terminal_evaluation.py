import numpy as np
from test_standard_relaxer import HarmonicField,helium
from orbit3d.physics.terminal_evaluation import evaluate_terminal


class Oracle:
    def __init__(self):self.positions=[]
    def evaluate_3d_mols_batched(self,molecules):
        x=molecules[0].GetConformer().GetPositions();self.positions.append(x.copy())
        return {'errors':[None],'mean':np.array([[float(np.linalg.norm(x)),0.]]),'std':np.zeros((1,2)),
                'forces_eV_A':[-x],'energy_eV':[float(.5*np.square(x).sum())]}


def test_terminal_property_uses_final_coordinates_and_preserves_initial():
    start=helium();field=HarmonicField();oracle=Oracle()
    final,row=evaluate_terminal(start,field,oracle)
    assert row['physical_accepted'] and row['total_dpa_calls']==len(field.calls)+2<=150
    np.testing.assert_array_equal(oracle.positions[-1],final.coords)
    np.testing.assert_array_equal(start.coords,[[1.,0.,0.]])
    assert row['final']['prediction_eV'][0]<row['initial']['prediction_eV'][0]


def test_scope_limit_preserves_oversize_proposal_without_force_or_property_calls():
    from rdkit import Chem
    from orbit3d.generator.molecular_state import Mol3DInstance
    start=Mol3DInstance(coords=np.array([[-.37,0.,0.],[.37,0.,0.]]),atom_types=np.zeros(2,dtype=int),element_table=('H',),graph=Chem.MolFromSmiles('[H][H]'))
    field=HarmonicField();oracle=Oracle()
    final,row=evaluate_terminal(start,field,oracle,max_atoms=1)
    assert not row['physical_accepted'] and row['scope_exclusion']=='atom_limit'
    assert row['total_dpa_calls']==0 and not field.calls and not oracle.positions
    np.testing.assert_array_equal(final.coords,start.coords)
    _,boundary=evaluate_terminal(helium(),HarmonicField(),Oracle(),max_atoms=1)
    assert boundary['physical_accepted'] and boundary['scope_supported']
