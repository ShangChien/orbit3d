import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,MaxAbsScaler,PowerTransformer,FunctionTransformer
from orbit3d.utils.dataset_io import inverse_single_scaler_torch
from orbit3d.oracles.uniqsar_oracle import _inverse_single_scaler_torch


@pytest.mark.parametrize('scaler',[
    StandardScaler(with_mean=False),StandardScaler(with_std=False),StandardScaler(with_mean=False,with_std=False),
    RobustScaler(with_centering=False),RobustScaler(with_scaling=False),MinMaxScaler(feature_range=(-2,3)),MaxAbsScaler(),
    PowerTransformer(method='box-cox'),PowerTransformer(method='yeo-johnson'),
    FunctionTransformer(),FunctionTransformer(np.log,inverse_func=np.exp),
])
def test_differentiable_inverse_matches_the_actual_fitted_scaler(scaler):
    scaler.fit(np.array([[.1],[.3],[1.],[2.],[10.]]))
    x=torch.tensor([-.5,.1,1.2],dtype=torch.float64,requires_grad=True)
    expected=scaler.inverse_transform(x.detach().numpy()[:,None])[:,0]
    for inverse in [inverse_single_scaler_torch,_inverse_single_scaler_torch]:
        result=inverse(x,scaler)
        np.testing.assert_allclose(result.detach().numpy(),expected,atol=1e-10,rtol=1e-10)
        gradient=torch.autograd.grad(result.sum(),x)[0]
        assert torch.isfinite(gradient).all()


def test_invalid_power_inverse_is_not_clipped_into_a_finite_prediction():
    scaler=PowerTransformer(method='box-cox',standardize=False).fit(np.array([[1.],[2.],[4.]]))
    scaler.lambdas_=np.array([-.5])
    x=torch.tensor([1.,3.],dtype=torch.float64)
    with np.errstate(invalid='ignore'):
        expected=scaler.inverse_transform(x.numpy()[:,None])[:,0]
    actual=inverse_single_scaler_torch(x,scaler).numpy()
    np.testing.assert_allclose(actual,expected,equal_nan=True)
    assert np.isfinite(actual[0]) and np.isnan(actual[1])


def test_log_inverse_is_not_capped_and_unknown_functions_do_not_become_expm1():
    scaler=PowerTransformer(standardize=False).fit(np.array([[1.],[2.],[4.]]));scaler.lambdas_=np.array([0.])
    x=torch.tensor([20.],dtype=torch.float64)
    torch.testing.assert_close(inverse_single_scaler_torch(x,scaler),torch.expm1(x))
    with pytest.raises(NotImplementedError,match='FunctionTransformer'):
        _inverse_single_scaler_torch(x,FunctionTransformer(inverse_func=lambda y:y**3))


@pytest.mark.parametrize('ensemble', ['average', 'stacking_linear', 'vote'])
def test_bridge_nonlinear_scaling_matches_public_mean_spread_and_coordinate_gradients(ensemble):
    from types import SimpleNamespace
    from rdkit import Chem
    from orbit3d.oracles.uniqsar_oracle import UniQSARTrainedBridge

    bridge = UniQSARTrainedBridge.__new__(UniQSARTrainedBridge)
    bridge.device = torch.device('cpu')
    bridge.output_dim = 1
    bridge.task = 'regression'
    bridge.scaler = FunctionTransformer(np.log, inverse_func=np.exp)
    bridge.has_3d_unimol = True
    bridge.ensemble_method = ensemble
    bridge.selected_keys = ['ML01_TAB', 'UniMolModel']
    bridge.needs_fp = bridge.needs_handcrafts = False
    bridge.desc_calc = None
    bridge._1d2d_cache = {}
    bridge.ml_folds = {'ML01_TAB': [SimpleNamespace(_predict=lambda _: np.array([0.]))]}
    bridge.nn_folds = {}
    bridge.stacking_linear_weights = np.array([[[.5, .5]], [[1., 1.]]])
    bridge.stacking_linear_biases = np.zeros((2, 1))

    def coordinate_folds(key, coords, symbols):
        bridge._last_3d_folds = torch.stack([coords[1, 0:1], 2 * coords[1, 0:1]])
        return bridge._last_3d_folds.mean(0)

    bridge._forward_unimol_folds = coordinate_folds
    mol = Chem.MolFromSmiles('CC')
    conf = Chem.Conformer(2)
    conf.SetAtomPosition(1, (1., 0., 0.))
    mol.AddConformer(conf)
    coords = torch.tensor(conf.GetPositions(), dtype=torch.float64, requires_grad=True)
    mean, std = bridge.forward_differentiable_3d(coords, ['C', 'C'], 'CC')
    public_mean, public_std = bridge.predict_3d_mols([mol])
    np.testing.assert_allclose([mean.item(), std.item()], [public_mean[0], public_std[0]], rtol=2e-7)
    if ensemble == 'average':
        # Equal models, then equal folds: 0.5, 0.25, 0.25; NOT three equal votes.
        values, weights = np.exp([0., 1., 2.]), np.array([.5, .25, .25])
        expected_std = np.sqrt(np.sum(weights * (values - np.dot(values, weights)) ** 2))
        assert mean.item() == pytest.approx(np.exp(.75))  # preserve native point estimate
        assert std.item() == pytest.approx(expected_std)
    for value, index in [(mean, 0), (std, 1)]:
        analytic = torch.autograd.grad(value, coords, retain_graph=True)[0][1, 0].item()
        numeric = []
        for delta in [-1e-5, 1e-5]:
            shifted = coords.detach().clone()
            shifted[1, 0] += delta
            numeric.append(bridge.forward_differentiable_3d(shifted, ['C', 'C'], 'CC')[index].item())
        assert analytic == pytest.approx((numeric[1] - numeric[0]) / 2e-5, rel=1e-7)


@pytest.mark.parametrize('scaler', [StandardScaler(), MinMaxScaler(), PowerTransformer()])
def test_shared_multitarget_scaler_matches_numpy(scaler):
    from orbit3d.oracles.uniqsar_oracle import UniQSARTrainedBridge
    scaler.fit(np.array([[.1, 1.], [.3, 5.], [1., 3.], [2., 9.], [10., 13.]]))
    bridge = UniQSARTrainedBridge.__new__(UniQSARTrainedBridge)
    bridge.output_dim, bridge.task, bridge.scaler = 2, 'multilabel_regression', scaler
    values = torch.tensor([[.1, .2], [.4, .8]], dtype=torch.float64, requires_grad=True)
    real = bridge._inverse_scale_torch(values)
    np.testing.assert_allclose(real.detach(), scaler.inverse_transform(values.detach().numpy()), rtol=1e-10)
    assert torch.isfinite(torch.autograd.grad(real.sum(), values)[0]).all()
