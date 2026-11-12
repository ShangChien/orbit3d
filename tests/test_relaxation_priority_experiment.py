"""Numerical contracts of the isolated relaxation experiment."""
from types import SimpleNamespace

import numpy as np
import torch

from scripts.experiment_relaxation_priority import Experiment, reverse_project


def test_reverse_projection_is_first_order_only():
    f = torch.tensor([[1., 0., 0.]])
    for g in [torch.tensor([[2., 3., 0.]]), torch.tensor([[-2., 3., 0.]])]:
        d = f - .25 * reverse_project(f, g)
        assert (f * d).sum() >= f.square().sum() - 1e-6
    g = torch.tensor([[1., 0., 0.]])
    step = -.01 * reverse_project(torch.zeros_like(g), g)
    # At a minimum the projector does not protect the stiff direction.
    assert torch.allclose(step, -.01 * g)
    assert torch.allclose(1000 * step.norm(), torch.tensor(10.))


def test_standard_optimizer_converges_with_bounded_steps():
    class Harmonic(Experiment):
        def evaluate(self, coords):
            self.positions.append(np.array(coords).copy())
            f = -2 * np.asarray(coords)
            return {'energy_eV': float(np.square(coords).sum()), 'forces': f.tolist(),
                    'rms_force_eV_A': float(np.sqrt(np.square(f).sum(-1).mean())),
                    'max_force_eV_A': float(np.linalg.norm(f, axis=-1).max())}
    exp = Harmonic(SimpleNamespace(symbols=['H', 'H']), None, None, None, 'cpu')
    exp.positions = []
    coords, result = exp.minimize(np.array([[.4, .2, 0.], [-.4, -.2, 0.]]), 'test')
    assert result['converged']
    assert np.square(coords).sum() < .01
    steps = np.diff(np.array(exp.positions), axis=0)
    assert np.linalg.norm(steps, axis=-1).max() <= .025 + 1e-9
