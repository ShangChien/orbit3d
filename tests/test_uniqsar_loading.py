import json
from types import SimpleNamespace
import joblib
import numpy as np
import pytest
from rdkit import Chem
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from orbit3d.oracles.uniqsar_oracle import UniQSARTrainedBridge


class StoredRegressor:
    def __init__(self, value):
        self.value = value

    def _predict(self, x):
        return np.full((len(x), 1), self.value)


class StoredClassifier:
    def __init__(self, model):
        self.model = model

    def _predict(self, x):
        return self.model.predict_proba(x)[:, 1:]


def saved_model(path, task='regression', method='stacking'):
    cfg = {'Base': {'task': task}, 'Datahub': {'num_classes': 1, 'split_n': 2,
           'target_cols': ['value'], 'target_normalize': 'none'}, 'Ensembler': {'method': method}}
    (path/'config.yaml').write_text(json.dumps(cfg))
    ens = path/'EnsembleModel'; ens.mkdir()
    keys = ['ML01_TAB_fringerprint', 'ML02_TAB_fringerprint']
    (ens/'selected_keys.txt').write_text('\n'.join(keys)+'\n')
    for key, value in zip(keys, [.2, .8]):
        (path/key).mkdir()
        for i in range(2):
            joblib.dump(StoredRegressor(value+i*.1), path/key/f'model_{i}.pth')
    return cfg, keys


def test_missing_training_artifacts_do_not_change_model(tmp_path):
    with pytest.raises(FileNotFoundError, match='config.yaml'):
        UniQSARTrainedBridge(str(tmp_path), device='cpu')
    cfg, keys = saved_model(tmp_path)
    with pytest.raises(ValueError, match='Incomplete stacking folds'):
        UniQSARTrainedBridge(str(tmp_path), device='cpu')
    # Explicit model subset uses native averaging, regardless of trained stacking.
    bridge = UniQSARTrainedBridge(str(tmp_path), device='cpu', infer_choices=['ML01'])
    mean, _ = bridge.predict_3d_mols([Chem.MolFromSmiles('CC')])
    assert mean.item() == pytest.approx(.25)
    (tmp_path/keys[0]/'model_1.pth').unlink()
    with pytest.raises(ValueError, match='Incomplete fold set'):
        UniQSARTrainedBridge(str(tmp_path), device='cpu', infer_choices=['ML01'])
    cfg['Datahub']['target_normalize'] = 'standard'
    (tmp_path/'config.yaml').write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError, match='target_scaler.ss'):
        UniQSARTrainedBridge(str(tmp_path), device='cpu')


def test_missing_selected_member_is_not_filtered(tmp_path):
    _, keys = saved_model(tmp_path)
    (tmp_path/keys[0]/'model_0.pth').unlink()
    with pytest.raises(FileNotFoundError, match='ensemble members'):
        UniQSARTrainedBridge(str(tmp_path), device='cpu')


def test_calibrated_linear_classifier_uses_native_probabilities(tmp_path):
    saved_model(tmp_path, task='classification')
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.2, .5], [.8, .4]])
    y = [0, 0, 1, 1, 0, 1]
    models = [SVC(kernel='linear', probability=True, random_state=i).fit(x, y) for i in range(2)]
    for i, model in enumerate(models):
        joblib.dump(StoredClassifier(model), tmp_path/'EnsembleModel'/f'stacking_{i}.pth')
    bridge = UniQSARTrainedBridge(str(tmp_path), device='cpu')
    assert bridge.ensemble_method == 'stacking_general'
    prediction, _ = bridge.predict_3d_mols([Chem.MolFromSmiles('CC')])
    expected = np.mean([model.predict_proba([[.25, .85]])[0, 1] for model in models])
    assert prediction.item() == pytest.approx(expected)
    guessed = np.mean([1/(1+np.exp(-model.decision_function([[.25, .85]])[0])) for model in models])
    assert abs(expected-guessed) > .01


def test_verified_linear_regression_retains_native_prediction():
    bridge = UniQSARTrainedBridge.__new__(UniQSARTrainedBridge)
    bridge.task = 'regression'; bridge.output_dim = 1
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    model = Ridge().fit(x, [1., 2., 3., 4.])
    assert bridge._try_extract_linear_stacking([SimpleNamespace(model=model)], 2)
    actual = np.einsum('fkm,nm->fnk', bridge.stacking_linear_weights, x)+bridge.stacking_linear_biases[:, None]
    np.testing.assert_allclose(actual[0, :, 0], model.predict(x))
