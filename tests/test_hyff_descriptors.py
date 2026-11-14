"""Check the adapter's molecule boundaries, electronic states and gradient graph."""
import torch
from torch import nn
from loom.physics.hyff_forcefield import HyFFPotentialGuidance
from rdkit import Chem


class AnalyticHyFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout_norms = nn.ModuleList([nn.Identity()])

    def energy(self, batch, edges):
        # Make the property descriptor depend on coordinates and each graph's Q/M.
        state = batch['total_charge'] + batch['multiplicity'] - 1
        features = self.readout_norms[0](batch['positions'] + state[batch['batch'], None])
        atomic = features.square().sum(-1).double() + 1e8
        return atomic.new_zeros(len(batch['ptr']) - 1).index_add(0, batch['batch'], atomic)


def test_hyff_batch_preserves_molecule_state_gradients_and_float64_energy():
    ff = HyFFPotentialGuidance.__new__(HyFFPotentialGuidance)
    nn.Module.__init__(ff)
    ff.device_str = 'cpu'
    ff._pt = Chem.GetPeriodicTable()
    ff.default_charge, ff.default_multiplicity = 0, 1
    ff.cutoff = 3.
    ff.ref_model = AnalyticHyFF()
    x = torch.tensor([[[0., 0., 0.], [1.4, .1, 0.]], [[0., 0., 0.], [1.1, 0., .2]]], requires_grad=True)
    symbols = [['C', 'C'], ['N', 'H']]
    desc, energy, force = ff.extract_native_batch(x, symbols, charges=[0, 1], multiplicities=[1, 2])
    assert energy.dtype == torch.float64
    torch.testing.assert_close(desc[0], x[0])
    torch.testing.assert_close(desc[1], x[1] + 2)
    gradient = torch.autograd.grad(desc[0].sum(), x)[0]
    assert torch.equal(gradient[0], torch.ones_like(x[0]))
    assert torch.equal(gradient[1], torch.zeros_like(x[1]))
    for i, (charge, multiplicity) in enumerate([(0, 1), (1, 2)]):
        single = ff.extract_native_descriptor_and_forces(x[i], symbols[i], charge, multiplicity)
        for actual, expected in zip(single, (desc[i], energy[i], force[i])):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    differentiable = ff.extract_differentiable_batch(x, symbols)
    assert torch.autograd.grad(differentiable.sum(), x)[0].abs().sum() > 0
