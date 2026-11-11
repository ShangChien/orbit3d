import pickle
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from orbit3d.utils.qm9s import adapt_record, read_record
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle


def mol(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(m, randomSeed=42) == 0
    return m


def test_qm9s_adapter_preserves_spectrum_and_rejects_order_mismatch():
    m = mol("CCO")
    record = {
        "smi": "CCO",
        "atoms": [a.GetSymbol() for a in m.GetAtoms()],
        "coordinates": [m.GetConformer().GetPositions()],
        "target": np.arange(240, dtype=np.float32),
    }
    loaded = read_record(pickle.dumps(record, protocol=5))
    inst, tokens = adapt_record(loaded)
    np.testing.assert_array_equal(inst.props, np.arange(240))
    assert tokens is None  # Octree data must not pass through the dense-grid tokenizer.
    _, tokens = adapt_record(loaded, backend="dense_grid_v2")
    assert len(tokens) == 4 * m.GetNumAtoms() + 2
    record["smi"] = "COC"
    with pytest.raises(ValueError, match="atom_order"):
        adapt_record(record)
    with pytest.raises(pickle.UnpicklingError):
        read_record(b"cposix\nsystem\n.")


def test_fingerprint_reload_and_checksum(tmp_path):
    o = FingerprintOracle(["homo", "lumo"])
    ms = [mol("CC"), mol("CO"), mol("CN")]
    o.fit(ms, np.array([[-6.0, -1.0], [-5.0, -2.0], [-7.0, -3.0]]), tmp_path, n_models=2, n_trees=4, n_jobs=1)
    reloaded = FingerprintOracle.load_from_dir(tmp_path)
    assert reloaded.training_elements == ["C", "H", "N", "O"]
    np.testing.assert_array_equal(o.predict_3d_mols(ms)[0], reloaded.predict_3d_mols(ms)[0])
    path = tmp_path / "ensemble.joblib"
    path.write_bytes(path.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        FingerprintOracle.load_from_dir(tmp_path)


class DescriptorForce:
    device_str = "cpu"

    def probe_native_descriptor_dim(self):
        return 4

    def extract_native_descriptor_and_forces(self, x, atoms):
        return torch.cat([x, x.square().sum(-1, keepdim=True)], -1), x.square().sum(), -2 * x

    def extract_native_batch(self, x, atoms):
        return torch.cat([x, x.square().sum(-1, keepdim=True)], -1), x.square().sum((-1, -2)), -2 * x


def test_reload_check_separates_native_noise_from_changed_scalers():
    import copy
    from sklearn.preprocessing import StandardScaler
    from orbit3d.oracles.validation import compare_oracle_reload

    class NoisyDescriptor(DescriptorForce):
        calls = 0

        def extract_native_batch(self, x, atoms):
            desc, energy, force = super().extract_native_batch(x, atoms)
            self.calls += 1
            return desc + (self.calls % 2) * 1e-6, energy, force

    original = InternalDPA4PropertyOracle(NoisyDescriptor(), 2, ['homo', 'lumo'], 2)
    original.scalers = [StandardScaler().fit(np.array([[0.], [1.]])) for _ in range(2)]
    restored = copy.deepcopy(original)
    restored.dpa_guidance = original.dpa_guidance
    _, _, report = compare_oracle_reload(original, restored, [mol('CC')])
    assert report['same_descriptor_prediction_max_abs_diff'] == 0
    assert 0 < report['repeated_descriptor_max_abs_diff'] < 1e-5
    restored.scalers[0].mean_ += .1
    with pytest.raises(ValueError, match='Adapter/scaler reload differs'):
        compare_oracle_reload(original, restored, [mol('CC')])


def test_bootstrap_dpa_uses_train_only_scalers_and_reloads(tmp_path):
    ff = DescriptorForce()
    o = InternalDPA4PropertyOracle(ff, 2, ["homo", "lumo"], 2)
    train = [mol(s) for s in ["CC", "CO", "CN", "C#N"]]
    valid = [mol("CCC"), mol("CCO")]
    y = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
    meta = o.fit_from_dataset(
        train,
        y,
        str(tmp_path),
        epochs=1,
        validation_mols=valid,
        validation_targets=np.array([[100.0, 101.0], [110.0, 111.0]], dtype=np.float32),
    )
    assert o.scalers[0].mean_[0] == 1.5
    assert meta["history"][0]["bootstrap_indices_sha256"] != meta["history"][1]["bootstrap_indices_sha256"]
    restored = InternalDPA4PropertyOracle.load_from_dir(str(tmp_path), ff)
    np.testing.assert_allclose(o.predict_3d_mols(valid)[0], restored.predict_3d_mols(valid)[0], atol=1e-7)
    x = torch.tensor(valid[0].GetConformer().GetPositions(), dtype=torch.float32, requires_grad=True)
    pred, _ = restored.forward_differentiable_3d(x, [a.GetSymbol() for a in valid[0].GetAtoms()])
    gradient = torch.autograd.grad(pred.sum(), x)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    with (tmp_path / 'target_scaler.ss').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='scaler checksum'):
        InternalDPA4PropertyOracle.load_from_dir(str(tmp_path), ff)


def test_disconnected_descriptor_does_not_return_partial_coordinate_gradient():
    class DetachedForce(DescriptorForce):
        def extract_native_descriptor_and_forces(self, x, atoms):
            desc, e, f = super().extract_native_descriptor_and_forces(x, atoms)
            return desc.detach(), e, f

    oracle = InternalDPA4PropertyOracle(DetachedForce(), 2, ["homo", "lumo"], 2)
    molecule = mol("CC")
    coords = torch.tensor(molecule.GetConformer().GetPositions(), dtype=torch.float32, requires_grad=True)
    with pytest.raises(NotImplementedError, match="gradients are unavailable"):
        oracle.forward_differentiable_3d(coords, [a.GetSymbol() for a in molecule.GetAtoms()])


def test_candidate_ranking_keeps_attempt_denominator_and_deduplicates():
    from orbit3d.generator.candidate_selection import select_candidates
    from orbit3d.generator.octree_tokenizer import rdkit_mol_to_mol3d_instance
    from orbit3d.constraints import TargetIntervals

    structures = [mol("CC"), mol("CO"), mol("CO"), mol("CN")]
    scores = [3.0, 0.5, 0.2, 2.0]

    def sample(**kwargs):
        i = kwargs["mol_id"]
        return rdkit_mol_to_mol3d_instance(structures[i], np.array([1.0])), torch.tensor([scores[i]])

    selected, records = select_candidates(
        sample, None, None, TargetIntervals.parse((0, 1), ["y"]), [0], [1], 2, pool_factor=2, deduplicate=True
    )
    assert len(records) == 4 and len(selected) == 2
    assert selected[0].mol_id == 1
    assert records[2]["status"] == "duplicate"


def test_precise_sdf_roundtrips_coordinates(tmp_path):
    from orbit3d.utils.dataset_io import write_precise_sdf

    m = mol("CCO")
    m.SetProp("Value", "[1.1, 2.2]")
    path = tmp_path / "precise.sdf"
    write_precise_sdf(path, [m])
    reloaded = next(iter(Chem.SDMolSupplier(str(path), removeHs=False)))
    np.testing.assert_allclose(
        m.GetConformer().GetPositions(), reloaded.GetConformer().GetPositions(), atol=1e-10, rtol=0
    )
    assert reloaded.GetProp("Value") == "[1.1, 2.2]"


def test_calibration_uses_finite_sample_joint_residual_quantiles():
    from orbit3d.oracles.calibration import ResidualCalibration

    y = np.array([[i, 2 * i] for i in range(10)], dtype=float)
    mean = np.zeros_like(y)
    std = np.ones_like(y)
    c = ResidualCalibration.fit(["a", "b"], y, mean, std, coverage=0.8)
    assert c.marginal_q == [8.0, 16.0]
    assert c.joint_q == 16.0
    lo, hi = c.intervals(np.array([[2.0, 3.0]]), np.ones((1, 2)))
    np.testing.assert_array_equal(lo, [[-14.0, -13.0]])
    np.testing.assert_array_equal(hi, [[18.0, 19.0]])


def test_applicability_domain_preserves_reference_and_hydrogen_convention(tmp_path):
    from orbit3d.oracles.applicability import MorganDomain

    domain = MorganDomain(['c1ccccc1', 'CCO'], 0.5)
    path = tmp_path / 'domain.json'
    domain.save(path)
    loaded = MorganDomain.load(path)
    assert loaded.score(Chem.AddHs(Chem.MolFromSmiles('CCO'))) == 1.0
    assert loaded.score(Chem.MolFromSmiles('C#N')) < loaded.cutoff


def test_vectorized_adapter_keeps_frames_and_coordinate_gradients_independent():
    from orbit3d.oracles.dpa4_oracle import _BootstrapAdapters

    torch.manual_seed(7)
    adapter = _BootstrapAdapters(12, 2, 3, hidden_dim=16)
    features = torch.randn(3, 5, 12)
    coords = torch.randn(3, 5, 3, requires_grad=True)
    together = adapter.forward_members(features, coords)
    separate = torch.stack([adapter.forward_members(f, x) for f, x in zip(features, coords)], dim=1)
    torch.testing.assert_close(together, separate)
    gradient = torch.autograd.grad(together[:, 0].sum(), coords)[0]
    assert gradient[0].abs().sum() > 0
    assert torch.count_nonzero(gradient[1:]) == 0


@pytest.mark.parametrize('smiles', ['[NH4+]', '[CH2]'])
def test_ocelot_property_adapters_reject_untrained_electronic_states(smiles):
    with pytest.raises(ValueError, match='neutral closed-shell'):
        InternalDPA4PropertyOracle.validate_molecule(Chem.MolFromSmiles(smiles))


def test_phosphazene_mmff_failure_is_an_explicit_preparation_rejection():
    from orbit3d.utils.conformers import ocelot_conformer

    # OCELOT csd_TAVCOI: parses as a graph, but MMFF cannot kekulize its fused system.
    with pytest.raises(ValueError, match='MMFF aromaticity/valence unsupported'):
        ocelot_conformer('FP1(F)=NP(F)(F)=NP2(=N1)Oc1cc3ccccc3cc1O2')


def test_csv_loader_records_geometry_rejection_without_shifting_labels(tmp_path):
    import json
    import pandas as pd
    from orbit3d.utils.dataset_io import load_3d_molecular_dataset

    source = tmp_path / 'molecules.csv'
    pd.DataFrame({'SMILES': ['FP1(F)=NP(F)(F)=NP2(=N1)Oc1cc3ccccc3cc1O2', 'CC'],
                  'target': [9., 3.]}).to_csv(source, index=False)
    report = tmp_path / 'preparation.json'
    molecules, values, _ = load_3d_molecular_dataset(str(source), target_cols=['target'], report_path=str(report))
    assert values.tolist() == [[3.]] and molecules[0].GetProp('SourceRow') == '1'
    evidence = json.loads(report.read_text())
    assert evidence['accepted'] == 1 and evidence['rejected'] == 1
    assert 'MMFF aromaticity/valence unsupported' in evidence['records'][0]['reason']


def test_wide_element_geometry_uses_the_explicitly_selected_forcefield():
    from orbit3d.utils.conformers import ocelot_conformer

    smiles = 'B1Nc2ccccc2C1=Cc1ccccc1'
    with pytest.raises(ValueError, match='MMFF parameters unavailable'):
        ocelot_conformer(smiles)
    molecule = ocelot_conformer(smiles, method='uff')
    assert any(atom.GetSymbol() == 'B' for atom in molecule.GetAtoms())
    assert molecule.GetProp('GeometryMethod') == 'etkdgv3_uff'
    assert not molecule.HasProp('MMFFOptimizationStatus')
    assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_vendored_hyff_is_private_and_preserves_constructor_ids(monkeypatch):
    import json
    import sys
    import types
    from pathlib import Path
    import orbit3d.compat.hyff as bundled
    from orbit3d.compat.hyff import registry
    from orbit3d.compat.hyff.models.factory import build_model, factory_id

    monkeypatch.setitem(sys.modules, 'hyff', types.ModuleType('unrelated_hyff'))
    before=list(sys.path)
    assert factory_id('hyff-l2')=='hyff.models.base_model:build_model'
    config=json.loads((Path(bundled.__file__).parent/'checkpoints/presto/weights.json').read_text())
    spec={k:v for k,v in config.items() if k in {'format','model_id','model_config','numerical_execution'}}
    model=build_model(spec)
    assert type(model).__module__.startswith('orbit3d.compat.hyff.')
    assert set(registry.manifest()['models'])=={'presto','vivace','moderato','andante','largo'}
    assert sys.path==before


def test_mim_wide_element_adapter_requires_declared_spectrum_and_explicit_atoms():
    m=mol('B(O)O')
    row={'smi':'B(O)O','atoms':[a.GetSymbol() for a in m.GetAtoms()],
         'coordinates':[m.GetConformer().GetPositions()],'target':np.arange(450,dtype=np.float32)}
    with pytest.raises(ValueError,match='spectrum'):
        adapt_record(row)
    instance,_=adapt_record(row,element_table=('H','C','B','O'),spectrum_size=450)
    np.testing.assert_array_equal(instance.props,row['target'])
    assert 5 in instance.atomic_numbers
    row['smi']='[C]'
    with pytest.raises(ValueError,match='radical'):
        adapt_record(row,element_table=('H','C','B','O'),spectrum_size=450)


def test_fingerprint_similarity_does_not_certify_an_unseen_element(tmp_path):
    from orbit3d.oracles.applicability import MorganDomain
    domain=MorganDomain(['Nc1ccccc1','Oc1ccccc1'],.25)
    molecule=Chem.MolFromSmiles('OB(O)c1ccccc1')
    assessment=domain.assess(molecule)
    assert assessment['similarity']>=domain.cutoff
    assert assessment['unseen_training_elements']==['B'] and not assessment['in_domain']
    assert domain.assess(Chem.AddHs(Chem.MolFromSmiles('Oc1ccccc1')))['in_domain']
    domain.save(tmp_path/'domain.json')
    assert MorganDomain.load(tmp_path/'domain.json').assess(molecule)==assessment
