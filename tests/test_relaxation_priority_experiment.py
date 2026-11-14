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


def test_point_gradient_remains_active_inside_interval(monkeypatch):
    import importlib
    from pathlib import Path
    from loom.constraints import TargetIntervals
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    PointExperiment = importlib.import_module('experiment_point_refinement').PointExperiment
    target = [-7.25, -.63]
    intervals = TargetIntervals.parse({'homo': [-7.5,-7.0], 'lumo': [-.88,-.38]}, ['homo','lumo'])
    interval = Experiment(None,None,None,intervals,'cpu')
    point = PointExperiment(None,None,None,intervals,'cpu',point=target)
    pred = torch.tensor([-7.15,-.53],requires_grad=True)
    old = interval.guidance_loss(pred)
    new = point.guidance_loss(pred)
    assert old.item() == 0
    assert new.item() > 0
    assert torch.allclose(torch.autograd.grad(new,pred)[0],torch.tensor([.2,.2]),atol=1e-5)
    assert point.guidance_value({'prediction': target}) == 0
