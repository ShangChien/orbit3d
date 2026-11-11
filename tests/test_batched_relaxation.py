from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from orbit3d.generator.octree_tokenizer import rdkit_mol_to_mol3d_instance
from orbit3d.physics.conformer_relaxer import relax_conformer_iso_property
from orbit3d.physics.batched_relaxer import relax_conformers_iso_property


class AnalyticOracle:
    target_cols = ['bond']

    def __init__(self, forcefield):
        self.dpa_guidance = forcefield

    def forward_differentiable_batch(self, coords, symbols):
        d = (coords[:, 1] - coords[:, 0]).norm(dim=-1, keepdim=True)
        return d, torch.zeros_like(d)

    def forward_differentiable_3d(self, coords, symbols, smiles=None):
        y, s = self.forward_differentiable_batch(coords[None], [symbols])
        return y[0], s[0]

    def forward_shared_batch(self, coords, symbols):
        with torch.enable_grad():
            x = coords.detach().clone().requires_grad_(True)
            y, std = self.forward_differentiable_batch(x, symbols)
            energy = 500 * (y[:, 0] - 1.6).square()
            force = -torch.autograd.grad(energy.sum(), x)[0]
        force[coords.mean((-1, -2)) > 5] = torch.nan
        return y.detach(), std.detach(), energy.detach(), force.detach()

    def forward_shared_force_and_property(self, coords, symbols):
        return tuple(x[0] for x in self.forward_shared_batch(coords[None], [symbols]))


def test_batch_refinement_matches_serial_and_isolates_nonfinite_frame():
    instances = []
    for smiles in ['CC', 'CO']:
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(m, randomSeed=42)
        instances.append(rdkit_mol_to_mol3d_instance(m, np.zeros(1)))
    instances.append(replace(instances[0], coords=instances[0].coords + 10))
    ff = SimpleNamespace(device_str='cpu')
    oracle = AnalyticOracle(ff)
    settings = dict(n_steps=8, step_size=.0004)
    batch = relax_conformers_iso_property(instances, ff, oracle, (1.49, 1.51),
                                         frozen_indices=[[0]] * 3, **settings)
    assert batch[2][0] is None and batch[2][1]['status'] == 'failed'
    for i in range(2):
        serial, trace = relax_conformer_iso_property(instances[i], ff, oracle, (1.49, 1.51),
                                                    frozen_indices=[0], **settings)
        actual, info = batch[i]
        np.testing.assert_allclose(actual.coords, serial.coords, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(actual.coords[0], instances[i].coords[0])
        assert [h['accepted'] for h in info['history']] == [h['accepted'] for h in trace['history']]
        assert info['interval_hit'] == trace['interval_hit']
        assert info['final_energy_ev'] <= info['initial_energy_ev']


def test_native_oracle_reports_per_frame_failures_without_erasing_raw_forces():
    from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle

    class Frames:
        device_str = 'cpu'

        def extract_native_batch(self, coords, symbols, check_finite):
            assert not check_finite
            energies = coords.mean((-1, -2))
            force = torch.ones_like(coords)
            force[energies > 5] = torch.nan
            return coords, energies, force

    oracle = InternalDPA4PropertyOracle.__new__(InternalDPA4PropertyOracle)
    torch.nn.Module.__init__(oracle)
    oracle.adapter = torch.nn.Identity()
    oracle.device = 'cpu'
    oracle.dpa_guidance = Frames()
    oracle.output_dim = 1
    oracle.target_cols = ['homo']
    oracle.target_units = ['eV']
    oracle._properties_from_descriptor = lambda features, coords: (coords.mean((-1, -2))[:, None], torch.zeros((len(coords), 1)))
    mols = []
    for smiles, shift in [('CC', 0), ('CO', 0), ('CC', 10), ('[NH4+]', 0)]:
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(m, randomSeed=42)
        m.GetConformer().SetPositions(m.GetConformer().GetPositions()+shift)
        mols.append(m)
    result = oracle.evaluate_3d_mols_batched(mols, batch_size=2)
    assert result['errors'][:2] == ['', '']
    assert 'Nonfinite' in result['errors'][2]
    assert 'neutral closed-shell' in result['errors'][3]
    assert np.isnan(result['forces_eV_A'][2]).all()
    assert result['forces_eV_A'][3] is None
    assert np.isfinite(result['mean'][:2]).all()
    assert result['target_units'] == ['eV']
