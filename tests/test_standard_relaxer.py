import numpy as np
import pytest
import torch
from rdkit import Chem
from orbit3d.generator.molecular_state import Mol3DInstance
from orbit3d.physics.standard_relaxer import relax_conformer_standard_energy


class HarmonicField:
    device_str='cpu'
    atom_type_map=['He']
    def __init__(self):self.calls=[]
    def compute_energy_and_forces(self,x,types,**settings):
        assert not settings['clip_forces'] and not settings['zero_com_forces']
        assert settings['charges']==[0] and settings['multiplicities']==[1]
        self.calls.append(x.detach().numpy().copy())
        return .5*x.square().sum(),-x


def helium():
    return Mol3DInstance(coords=np.array([[1.,0.,0.]],dtype=np.float64),atom_types=np.array([0]),
                         element_table=('He',),graph=Chem.MolFromSmiles('[He]'))


@pytest.mark.parametrize('method',['lbfgs','fire'])
def test_standard_optimizers_minimize_energy_using_true_forces(method):
    initial=helium();field=HarmonicField()
    result,info=relax_conformer_standard_energy(initial,field,method,max_evaluations=150,fmax=1e-3)
    assert info['force_converged'] and info['connectivity_preserved']
    assert info['final_energy_ev']<info['initial_energy_ev']
    assert info['final_max_force_eva']<1e-3
    assert info['evaluation_calls']==len(field.calls)<=150
    np.testing.assert_array_equal(result.coords,field.calls[-1])
    np.testing.assert_array_equal(initial.coords,[[1.,0.,0.]])


def test_evaluation_budget_includes_initial_and_exact_exported_state():
    field=HarmonicField();result,info=relax_conformer_standard_energy(helium(),field,max_evaluations=2,fmax=1e-8)
    assert not info['force_converged'] and info['status']=='evaluation_budget'
    assert info['evaluation_calls']==len(field.calls)==2
    assert info['optimizer_steps']==1
    np.testing.assert_array_equal(result.coords,field.calls[-1])
    assert info['final_energy_ev']==float(.5*np.square(result.coords).sum())
