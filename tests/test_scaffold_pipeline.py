import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from orbit3d.pipeline import scaffold_runner
from orbit3d.pipeline.runner import run_from_config
from orbit3d.utils.dataset_io import write_precise_sdf


class SharedOracle:
    target_cols = ['homo', 'lumo']

    def __init__(self, forcefield):
        self.dpa_guidance = forcefield

    def predict_3d_mols(self, molecules):
        return np.zeros((len(molecules), 2)), np.zeros((len(molecules), 2))

    def forward_shared_batch(self, coords, symbols):
        relative = coords-coords.mean(1, keepdim=True)
        return (torch.zeros(len(coords), 2), torch.zeros(len(coords), 2),
                .005*relative.square().sum((1, 2)), -.01*relative)


def test_scaffold_cli_grows_multiple_ports_without_losing_world_coordinates(tmp_path, monkeypatch):
    source = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
    assert AllChem.EmbedMolecule(source, randomSeed=42) == 0
    AllChem.MMFFOptimizeMolecule(source)
    source.GetConformer().SetPositions(source.GetConformer().GetPositions()+[10000.123456789, -10000.987654321, 20000.246813579])
    source.SetProp('OpenAtomIndices', '[0, 3]')
    source.SetProp('homo', '-7.5')
    source.SetProp('spectrum', '[1, 2, 3]')
    sdf = tmp_path/'scaffolds.sdf'
    write_precise_sdf(sdf, [source])
    original = scaffold_runner.read_scaffolds(sdf)[0]
    forcefield = SimpleNamespace(device_str='cpu')
    oracle = SharedOracle(forcefield)
    monkeypatch.setattr(scaffold_runner, 'load_evaluators', lambda *args: (forcefield, oracle, None))
    cfg = {'mode': 'predict', 'device': 'cpu', 'seed': 2026,
           'mount_paths': {'scaffolds_path': str(sdf), 'output_dir': str(tmp_path/'out'),
                           'forcefield_path': str(sdf), 'property_checkpoint_dir': str(sdf)},
           'generator_config': {'backend': 'scaffold_graph_growth', 'fragments': ['C', 'N', 'C#N'],
                                'attachments_per_candidate': 2},
           'oracle_config': {'oracle_mode': 'internal_dpa4', 'target_cols': ['homo', 'lumo']},
           'inference_config': {'num_candidates': 4, 'num_molecules': 4, 'oracle_candidates': 4,
                                'target_interval': {'homo': [-1, 1], 'lumo': [-1, 1]}, 'max_rms_force_eva': 1.},
           'stage5_relax_config': {'relax_steps': 2, 'batch_refinement': True}}
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg))
    report = run_from_config(str(config))
    assert report['raw_valid'] == 4 and report['retained'] > 0
    assert report['export_prediction_recheck_passed']
    for mol in Chem.SDMolSupplier(str(tmp_path/'out/generated.sdf'), removeHs=False):
        assert len(json.loads(mol.GetProp('AttachmentActions'))) == 2
        assert not mol.HasProp('homo') and not mol.HasProp('spectrum')
        source_properties = json.loads(mol.GetProp('ScaffoldSourceProperties'))
        assert source_properties['homo'] == '-7.5' and source_properties['spectrum'] == '[1, 2, 3]'
        assert int(mol.GetProp('TotalCharge')) == Chem.GetFormalCharge(mol)
        mapping = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
        for key in json.loads(mol.GetProp('FixedAtomMaps')):
            np.testing.assert_array_equal(mol.GetConformer().GetPositions()[mapping[key]], original.fixed_positions[key])


def test_float64_adapter_geometry_is_translation_invariant_before_precision_conversion():
    from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle, _BootstrapAdapters
    oracle = InternalDPA4PropertyOracle.__new__(InternalDPA4PropertyOracle)
    torch.nn.Module.__init__(oracle)
    torch.manual_seed(42)
    oracle.adapter = _BootstrapAdapters(8, 2, 2, hidden_dim=16).eval()
    oracle.output_dim, oracle.scalers = 2, []
    coords = torch.tensor([[.123456789, .1, 0.], [1.987654321, .3, -.4]], dtype=torch.float64)
    features = torch.randn(2, 8)
    expected = oracle._properties_from_descriptor(features, coords)
    shifted = oracle._properties_from_descriptor(features, coords+100000.)
    for a, b in zip(expected, shifted):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)


def test_ordinary_oracle_preserves_the_same_float64_geometry_as_shared_forward():
    from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle, _BootstrapAdapters
    oracle = InternalDPA4PropertyOracle.__new__(InternalDPA4PropertyOracle)
    torch.nn.Module.__init__(oracle)
    oracle.device, oracle.coordinate_dtype = torch.device('cpu'), torch.float64
    torch.manual_seed(7)
    oracle.adapter = _BootstrapAdapters(8, 2, 2, hidden_dim=16).eval()
    oracle.output_dim, oracle.scalers = 2, []
    features = torch.randn(2, 8)
    oracle.dpa_guidance = SimpleNamespace(extract_native_descriptor_and_forces=lambda coords, symbols:
                                         (features, torch.zeros(()), torch.zeros_like(coords)))
    mol = Chem.MolFromSmiles('CC')
    coords = np.array([[.123456789, .1, 0.], [1.987654321, .3, -.4]]) + 100000.
    conf = Chem.Conformer(2)
    conf.SetPositions(coords)
    mol.AddConformer(conf)
    expected, _, _, _ = oracle.forward_shared_force_and_property(torch.tensor(coords), ['C', 'C'])
    actual, _ = oracle.predict_3d_mols([mol])
    np.testing.assert_allclose(actual[0], expected.detach().numpy(), atol=1e-7, rtol=1e-7)
