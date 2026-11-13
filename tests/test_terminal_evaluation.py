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
