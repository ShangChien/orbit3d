from pathlib import Path
import json
import numpy as np
import pytest
import torch
from rdkit import Chem
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from loom.oracles import unimol_oracle as module
from loom.oracles.uniqsar_oracle import tokenize_smiles_1d, SMI_TOKEN2IDX


class TinyUniMol(torch.nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(31, 4)
        self.encoder = torch.nn.Linear(4, 4)
        self.classification_head = torch.nn.Linear(4, output_dim)

    def forward(self, tokens, distances, edges):
        features = self.encoder(self.embed_tokens(tokens).mean(1))
        return self.classification_head(features) + .01*distances.mean((1, 2))[:, None]


def molecule(smiles):
    m = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetPositions(np.arange(m.GetNumAtoms()*3).reshape(-1, 3)*.1)
    m.AddConformer(conf)
    return m


def test_pretrained_loader_requires_the_complete_backbone_and_does_not_mutate_on_failure(tmp_path):
    model = TinyUniMol()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    weights = tmp_path/'partial.pt'
    torch.save({'model': {'embed_tokens.weight': torch.ones_like(model.embed_tokens.weight)}}, weights)
    with pytest.raises(ValueError, match='Incomplete'):
        module.load_pretrained_unimol_backbone(model, str(weights))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
    pretrained = {k: torch.full_like(v, .2) for k, v in original.items()}
    torch.save({'model': pretrained}, weights)
    count = module.load_pretrained_unimol_backbone(model, str(weights))
    assert count == 3
    assert torch.equal(model.classification_head.weight, original['classification_head.weight'])
    assert torch.equal(model.encoder.weight, pretrained['encoder.weight'])


def test_unimol_never_truncates_or_substitutes_unknown_elements():
    with pytest.raises(ValueError, match='supported range'):
        module._UniMol3DDataset([molecule('CCC')], np.zeros((1, 1)), max_atoms=2)
    with pytest.raises(ValueError, match='Te'):
        module._UniMol3DDataset([molecule('[TeH2]')], np.zeros((1, 1)))
    with pytest.raises(ValueError, match='row counts'):
        module._UniMol3DDataset([molecule('CC')], np.zeros((0, 1)))


def test_fold_scalers_are_fit_on_training_labels_and_reloaded_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'StandaloneUniMolModel', TinyUniMol)
    initial = tmp_path/'pretrained.pt'
    torch.save({'model': TinyUniMol().state_dict()}, initial)
    molecules = [molecule(s) for s in ['CC', 'CO', 'CN', 'CF']]
    labels = np.array([[1., -1.], [2., -2.], [30., -30.], [40., -40.]], dtype=np.float32)
    oracle = module.InternalUniMolOracle(output_dim=2, n_folds=2, device='cpu', pretrain_weight_path=str(initial))
    metadata = oracle.fit_from_dataset(molecules, labels, str(tmp_path/'trained'), epochs=1, batch_size=2)
    for scalers, split in zip(oracle.fold_scalers, metadata['fold_splits']):
        for k, scaler in enumerate(scalers):
            assert scaler.mean_[0] == pytest.approx(labels[split['train_indices'], k].mean())
            assert scaler.mean_[0] != labels[:, k].mean()
    restored = module.InternalUniMolOracle.load_from_dir(str(tmp_path/'trained'), device='cpu')
    before = oracle.predict_3d_mols(molecules)
    after = restored.predict_3d_mols(molecules)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    coords = torch.tensor(molecules[0].GetConformer().GetPositions(), dtype=torch.float32, requires_grad=True)
    mean, std = restored.forward_differentiable_3d(coords, ['C', 'C'])
    np.testing.assert_allclose(mean.detach(), after[0][0], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(std.detach(), after[1][0], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(torch.autograd.grad(mean.sum()+std.sum(), coords)[0]).all()
    path = tmp_path/'trained/unimol_fold_1.pth'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checkpoint checksum'):
        module.InternalUniMolOracle.load_from_dir(str(tmp_path/'trained'), device='cpu')


def test_unimol_aggregates_different_fold_scalers_in_physical_units():
    oracle = module.InternalUniMolOracle(output_dim=1, device='cpu')
    a = StandardScaler().fit([[1.], [3.]])
    b = FunctionTransformer(np.log, inverse_func=np.exp)
    oracle.fold_scalers = [[a], [b]]
    normalized = torch.tensor([[0.], [2.]], dtype=torch.float64, requires_grad=True)
    mean, std = oracle._physical_statistics(normalized)
    expected = np.array([2., np.exp(2.)])
    assert mean.item() == pytest.approx(expected.mean())
    assert std.item() == pytest.approx(expected.std())
    assert torch.isfinite(torch.autograd.grad(mean.sum()+std.sum(), normalized)[0]).all()


def test_smiles_stereo_tokens_and_length_are_not_silently_dropped():
    tokens = tokenize_smiles_1d('C/C=C\\C')
    assert SMI_TOKEN2IDX['\\'] in tokens and SMI_TOKEN2IDX['/'] in tokens
    with pytest.raises(ValueError, match='token count'):
        tokenize_smiles_1d('C'*201)
    with pytest.raises(ValueError, match='grammar'):
        tokenize_smiles_1d('C!C')
