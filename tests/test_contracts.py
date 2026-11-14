import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from loom.generator.molecular_state import (
    ATOM_TYPES,
    Mol3DInstance,
    mol3d_to_rdkit_mol,
    mol3d_to_validated_rdkit_mol,
    rdkit_mol_to_mol3d_instance,
)


def embedded(smi):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


@pytest.mark.parametrize("smi", ["c1ccccc1", "CC=O", "C#N", "C[C@H](O)F"])
def test_graph_identity(smi):
    mol = embedded(smi)
    inst = rdkit_mol_to_mol3d_instance(mol, np.array([1.0, 2.0]))
    out = mol3d_to_validated_rdkit_mol(inst)
    assert Chem.MolToSmiles(out) == Chem.MolToSmiles(mol)


def test_unsupported_elements_are_not_substituted():
    mol = Chem.MolFromSmiles("[Ir+3]")
    mol.AddConformer(Chem.Conformer(1))
    with pytest.raises(ValueError):
        rdkit_mol_to_mol3d_instance(mol, np.array([1.0]))


def test_disconnected_invalid():
    inst = Mol3DInstance(np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]]), np.array([0, 0, 0]))
    assert mol3d_to_rdkit_mol(inst) is None


def test_charge_spin_and_fixed_coordinate_frame_are_preserved():
    from dataclasses import replace
    charged = embedded('[NH4+]')
    charged.GetConformer().SetPositions(charged.GetConformer().GetPositions() + [10, 20, 30])
    inst = rdkit_mol_to_mol3d_instance(charged, np.zeros(1), center=False)
    result = mol3d_to_validated_rdkit_mol(inst)
    assert Chem.GetFormalCharge(result) == 1
    np.testing.assert_allclose(result.GetConformer().GetPositions(), charged.GetConformer().GetPositions(), atol=2e-6)
    radical = embedded('[CH3]')
    radical.SetIntProp('SpinMultiplicity', 2)
    inst = rdkit_mol_to_mol3d_instance(radical, np.zeros(1))
    assert mol3d_to_validated_rdkit_mol(inst).GetIntProp('SpinMultiplicity') == 2
    assert mol3d_to_rdkit_mol(replace(inst, multiplicity=3)) is None
    assert mol3d_to_rdkit_mol(replace(inst, multiplicity=1)) is None




from types import SimpleNamespace

from loom.constraints import TargetIntervals
from loom.oracles.base import PropertyOracleBase
from loom.oracles.uniqsar_oracle import UniQSARTrainedBridge
from loom.physics.conformer_relaxer import project_force, relax_conformer_iso_property


def test_intervals_named_masked_and_joint():
    iv = TargetIntervals.parse({"b": (0, 1), "a": (0, 1)}, ["a", "b"])
    assert not iv.hit([0.5, 99])
    assert iv.hit([0.5, 0.9])
    iv = TargetIntervals.parse([None, (None, 1)], ["a", "b"])
    assert iv.hit([99, 0.9])
    assert iv.condition([0, 0], [1, 1]).tolist() == [0, 0, 0, 0, 0, 1, 0, 1]
    with pytest.raises(ValueError):
        TargetIntervals.parse({"bad": (0, 1)}, ["a"])




def test_molecular_force_projection():
    g = torch.tensor([[1.0, 0, 0], [2.0, 0, 0]])
    f = 3 * g
    assert torch.allclose(project_force(f, g), torch.zeros_like(f))


class BondOracle(PropertyOracleBase):
    def predict_3d_mols(self, mols):
        return np.array(
            [
                np.linalg.norm(m.GetConformer().GetPositions()[1] - m.GetConformer().GetPositions()[0])
                for m in mols
            ]
        ), np.zeros(len(mols))

    def forward_differentiable_3d(self, x, *args, **kwargs):
        return (x[1] - x[0]).norm(), torch.tensor(0.0)


class BondForce:
    device_str = "cpu"
    atom_type_map = ATOM_TYPES

    def __init__(self, target):
        self.target = target

    def compute_energy_and_forces(self, x, *args, **kwargs):
        with torch.enable_grad():
            v = x.detach().clone().requires_grad_(True)
            e = 500 * ((v[1] - v[0]).norm() - self.target).square()
            f = -torch.autograd.grad(e, v)[0]
        return e.detach().reshape(1), f


def test_relaxation_acceptance_and_fixed_scaffold():
    inst = rdkit_mol_to_mol3d_instance(embedded("CC"), np.array([1.0]))
    d = float(np.linalg.norm(inst.coords[1] - inst.coords[0]))
    result, info = relax_conformer_iso_property(
        inst,
        BondForce(d + 0.05),
        BondOracle(),
        (d - 0.001, d + 0.001),
        n_steps=8,
        step_size=0.0004,
        frozen_indices=[0],
    )
    assert abs(info["y_pred"] - d) <= 0.001 + 1e-7
    np.testing.assert_array_equal(result.coords[0], inst.coords[0])
    energies = [h["energy"] for h in info["history"]]
    assert all(b <= a for a, b in zip(energies, energies[1:]))


def bridge():
    b = UniQSARTrainedBridge.__new__(UniQSARTrainedBridge)
    b.device = torch.device("cpu")
    b.output_dim = b.num_classes = 1
    b.task = "regression"
    b.scaler = None
    b.has_3d_unimol = False
    b.aux_3d_head = None
    return b


def test_fold_uncertainty_and_stacking():
    b = bridge()
    b.selected_keys = ["ML01_TAB_foo"]
    b.ensemble_method = "average"
    b.needs_fp = b.needs_handcrafts = False
    b.desc_calc = None
    b._1d2d_cache = {}
    b.nn_folds = {}
    b.ml_folds = {
        "ML01_TAB_foo": [
            SimpleNamespace(_predict=lambda _: np.array([-1.0])),
            SimpleNamespace(_predict=lambda _: np.array([1.0])),
        ]
    }
    mean, std = b.predict_3d_mols([Chem.MolFromSmiles("CC")])
    assert std[0] == 1
    b.ensemble_method = "stacking_general"
    with pytest.raises(NotImplementedError):
        b.forward_differentiable_3d(torch.zeros(2, 3), ["C", "C"], "CC")






def test_sdf_sparse_indices_and_empty_data(tmp_path, monkeypatch):
    import pandas as pd

    from loom.utils.dataset_io import PandasTools, load_3d_molecular_dataset

    monkeypatch.setattr(
        PandasTools,
        "LoadSDF",
        lambda *a, **kw: pd.DataFrame({"ROMol": [embedded("CC")], "y": [1.0]}, index=[1]),
    )
    mols, y, names = load_3d_molecular_dataset(str(tmp_path / "sample.sdf"), target_cols=["y"])
    assert len(mols) == 1 and y.tolist() == [[1.0]]
    monkeypatch.setattr(
        PandasTools,
        "LoadSDF",
        lambda *a, **kw: pd.DataFrame({"ROMol": [embedded("CC")], "y": [np.inf]}, index=[1]),
    )
    with pytest.raises(ValueError, match="No usable"):
        load_3d_molecular_dataset(str(tmp_path / "sample.sdf"), target_cols=["y"])




def test_raw_nan_force_rejected():
    inst = rdkit_mol_to_mol3d_instance(embedded("CC"), np.array([1.0]))

    class BadForce:
        device_str = "cpu"
        atom_type_map = ATOM_TYPES

        def compute_energy_and_forces(self, x, *a, **kw):
            return torch.tensor([0.0]), torch.full_like(x, float("nan"))

    with pytest.raises(FloatingPointError):
        relax_conformer_iso_property(inst, BadForce(), BondOracle(), (0, 10))


def test_odd_electron_singlet_training_is_rejected():
    inst = rdkit_mol_to_mol3d_instance(embedded("[CH3]"), np.array([1.0]))
    with pytest.raises(ValueError, match="electronic_state"):
        mol3d_to_validated_rdkit_mol(inst)


def test_refinement_freezes_inferred_graph():
    original = rdkit_mol_to_mol3d_instance(embedded("CC"), np.array([1.0]))
    from dataclasses import replace
    inst = replace(original, graph=None)
    d = float(np.linalg.norm(inst.coords[1] - inst.coords[0]))
    result, _ = relax_conformer_iso_property(inst, BondForce(d), BondOracle(), (0, 10), n_steps=0)
    assert result.graph is not None
    assert Chem.MolToSmiles(Chem.RemoveHs(result.graph)) == "CC"


def test_hyff_never_substitutes_unknown_elements_with_carbon():
    from loom.physics.hyff_forcefield import HyFFPotentialGuidance
    ff = HyFFPotentialGuidance.__new__(HyFFPotentialGuidance)
    torch.nn.Module.__init__(ff)
    ff._pt = Chem.GetPeriodicTable()
    ff.device_str = 'cpu'
    ff.set_atom_type_map(['C', 'Si', 'Ir'])
    assert ff.dataset_to_atomic_num.tolist() == [6, 14, 77]
    with pytest.raises(ValueError, match='Unknown element'):
        ff.set_atom_type_map(['not_an_element'])
    with pytest.raises(ValueError, match='outside HyFF'):
        ff.set_atom_type_map(['U'])


def test_hyff_native_failure_is_not_silently_replaced_by_reference():
    from loom.physics.hyff_forcefield import HyFFPotentialGuidance
    ff = HyFFPotentialGuidance.__new__(HyFFPotentialGuidance)
    torch.nn.Module.__init__(ff)
    ff._pt = Chem.GetPeriodicTable()
    ff.device_str = 'cpu'
    ff.hyff_device = 'cuda'  # Exercise route dispatch without requiring a GPU kernel.
    ff.route = 'native'
    ff.tier = 'presto'
    ff.default_charge, ff.default_multiplicity = 0., 1
    ff.set_atom_type_map(['C'])
    def unavailable(*args):
        raise RuntimeError('Native capacity exceeded')
    ff._get_or_create_native_cp = unavailable
    ff._compute_pytorch_gpu = lambda **kwargs: pytest.fail('Silent engine replacement')
    with pytest.raises(RuntimeError, match='Native capacity'):
        ff.compute_energy_and_forces(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long))


def test_displacement_cap_preserves_molecular_tangent_and_batch_independence():
    from loom.physics.conformer_relaxer import cap_molecular_displacement
    gradient = torch.tensor([[2., 0, 0], [1., 0, 0]])
    tangent = torch.tensor([[1., 0, 0], [-2., 0, 0]])
    # Per-atom clipping would rotate this tangent and produce g dot d = +0.1.
    capped = cap_molecular_displacement(tangent, .1)
    assert abs(float((capped * gradient).sum())) < 1e-8
    assert capped.norm(dim=-1).max() <= .1
    batch = torch.stack([tangent, tangent * .01, torch.zeros_like(tangent)])
    result = cap_molecular_displacement(batch, .1)
    torch.testing.assert_close(result[0], capped)
    torch.testing.assert_close(result[1], batch[1])
    torch.testing.assert_close(result[2], batch[2])


@pytest.mark.parametrize('smiles', ['B(O)O', 'C[Si](C)(C)C', 'C[Se]C', 'C[As](C)C', 'C[Te]C'])
def test_explicit_element_table_preserves_wide_elements(smiles):
    mol = embedded(smiles)
    table = tuple(sorted({a.GetSymbol() for a in mol.GetAtoms()}))
    inst = rdkit_mol_to_mol3d_instance(mol, np.zeros(1), element_table=table)
    out = mol3d_to_validated_rdkit_mol(inst)
    assert [a.GetAtomicNum() for a in out.GetAtoms()] == inst.atomic_numbers.tolist()
    assert Chem.MolToSmiles(out) == Chem.MolToSmiles(mol)


def test_forcefield_indices_follow_its_own_element_table():
    from loom.oracles.base import PropertyOracleBase
    mol = embedded('C')
    inst = rdkit_mol_to_mol3d_instance(mol, np.zeros(1), element_table=('H','C'))
    class FixedOracle(PropertyOracleBase):
        target_cols = ['x']
        def predict_3d_mols(self, molecules):
            return np.zeros((len(molecules),1)), np.zeros((len(molecules),1))
    class FixedField:
        device_str = 'cpu'
        atom_type_map = ['C','H']
        def compute_energy_and_forces(self, coords, types, **kwargs):
            assert types.tolist() == [0,1,1,1,1]
            return torch.tensor(0.), torch.zeros_like(coords)
    out,_ = relax_conformer_iso_property(inst,FixedField(),FixedOracle(),[-1,1],n_steps=0)
    assert out.symbols == inst.symbols


def test_large_atomic_energy_offsets_do_not_erase_small_physical_energy_changes():
    from loom.physics.dpa_forcefield import DPAPotentialGuidance
    class AnalyticNative(torch.nn.Module):
        def forward(self, coords, atom_types):
            # The large atomic reference energy is common in heavy-element MLIPs.
            atom_energy = 1e8 + coords.double().square().sum(-1)
            return [{'atom_energy':atom_energy,'energy':atom_energy.sum(-1),'force':-2*coords}]
    ff=DPAPotentialGuidance.__new__(DPAPotentialGuidance)
    torch.nn.Module.__init__(ff)
    ff.wrapper=AnalyticNative();ff.coord_dtype=torch.float64
    ff.device_str='cpu';ff.batch_separation=100.;ff.atom_type_map=['C']
    ff.register_buffer('dataset_to_dpa_idx',torch.zeros(1,dtype=torch.long))
    x=torch.tensor([[-1.,0,0],[1.,0,0]],dtype=torch.float32)
    moved=x.clone();moved[:,0]*=1.001
    types=torch.zeros(2,dtype=torch.long)
    e,_=ff.compute_energy_and_forces(x,types,clip_forces=False,zero_com_forces=False)
    after,_=ff.compute_energy_and_forces(moved,types,clip_forces=False,zero_com_forces=False)
    expected=(moved.double().square()-x.double().square()).sum()
    assert e.dtype==torch.float64
    torch.testing.assert_close(after-e,expected.reshape(1),atol=1e-7,rtol=0)


def test_reflected_coordinates_cannot_keep_an_incompatible_stereocenter_label():
    from dataclasses import replace
    inst=rdkit_mol_to_mol3d_instance(embedded('C[C@H](O)F'),np.zeros(1))
    reflected=inst.coords.copy();reflected[:,0]*=-1
    with pytest.raises(ValueError,match='stereo_geometry'):
        mol3d_to_validated_rdkit_mol(replace(inst,coords=reflected))


def test_cis_geometry_cannot_retain_a_trans_graph_label():
    from dataclasses import replace
    trans=rdkit_mol_to_mol3d_instance(embedded('F/C=C/F'),np.zeros(1))
    cis=rdkit_mol_to_mol3d_instance(embedded('F/C=C\\F'),np.zeros(1))
    mol3d_to_validated_rdkit_mol(trans)
    mol3d_to_validated_rdkit_mol(cis)
    with pytest.raises(ValueError,match='stereo_geometry'):
        mol3d_to_validated_rdkit_mol(replace(trans,coords=cis.coords))


def test_energy_only_refinement_has_no_fake_property_success_and_forwards_state():
    from loom.physics.conformer_relaxer import relax_conformer_energy_only
    inst=rdkit_mol_to_mol3d_instance(embedded('CC'),np.zeros(1))
    out,info=relax_conformer_energy_only(inst,BondForce(1.6),n_steps=8,step_size=.0004)
    assert info['status']=='energy_only' and info['interval_hit'] is None
    assert info['y_pred'] is None and info['gradient_calls']==0
    assert info['final_energy_ev']<=info['initial_energy_ev']
    assert info['final_force_eva']<info['init_force_eva']
    charged=rdkit_mol_to_mol3d_instance(embedded('[NH4+]'),np.zeros(1))
    class StateAwareField:
        device_str='cpu'
        atom_type_map=ATOM_TYPES
        def compute_energy_and_forces(self,x,types,charges,multiplicities,**kwargs):
            assert charges==[1] and multiplicities==[1]
            return torch.tensor(0.,dtype=torch.float64),torch.zeros_like(x)
    out,info=relax_conformer_energy_only(charged,StateAwareField(),n_steps=0)
    assert out.charge==1


def test_dpa_state_is_rejected_before_any_neutral_calculation():
    from loom.physics.dpa_forcefield import DPAPotentialGuidance
    ff=DPAPotentialGuidance.__new__(DPAPotentialGuidance)
    torch.nn.Module.__init__(ff)
    for state in [{'charges':[1]},{'multiplicities':[3]}]:
        with pytest.raises(ValueError,match='supports'):
            ff.compute_energy_and_forces(torch.zeros(1,3),torch.zeros(1,dtype=torch.long),**state)


def test_refinement_preserves_float64_scaffold_world_coordinates():
    mol = embedded('CC')
    mol.GetConformer().SetPositions(mol.GetConformer().GetPositions()+[10000.123456789, 20000.987654321, -10000.246813579])
    inst = rdkit_mol_to_mol3d_instance(mol, np.array([1.]), center=False, coordinate_dtype=np.float64)
    distance = float(np.linalg.norm(inst.coords[1]-inst.coords[0]))
    result, _ = relax_conformer_iso_property(inst, BondForce(distance+.05), BondOracle(),
                                            (distance-.001, distance+.001), n_steps=2, step_size=.0004,
                                            frozen_indices=[0])
    assert result.coords.dtype == np.float64
    np.testing.assert_array_equal(result.coords[0], mol.GetConformer().GetPositions()[0])


@pytest.mark.parametrize("backend", [None, "uni3dar", "dense_grid_v2", "unrecognized"])
def test_generator_backend_must_be_supported_and_explicit(tmp_path, backend):
    import json

    from loom.pipeline.runner import run_from_config
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps({"generator_config": {"backend": backend}}))
    with pytest.raises(ValueError, match="Declare generator_config.backend"):
        run_from_config(str(path))


def test_packed_molecular_translations_preserve_float32_input_geometry():
    from loom.physics.dpa_forcefield import DPAPotentialGuidance

    class IndependentHarmonicPairs(torch.nn.Module):
        def forward(self, coords, atom_types):
            pairs=coords.reshape(-1,2,3)
            delta=pairs[:,1]-pairs[:,0]
            atom_energy=(.25*delta.square().sum(-1)).repeat_interleave(2)[None]
            forces=torch.stack([delta,-delta],dim=1).reshape_as(coords)
            return [{'atom_energy':atom_energy,'energy':atom_energy.sum(-1),'force':forces}]

    ff=DPAPotentialGuidance.__new__(DPAPotentialGuidance)
    torch.nn.Module.__init__(ff)
    ff.wrapper=IndependentHarmonicPairs();ff.coord_dtype=torch.float64
    ff.device_str='cpu';ff.batch_separation=60.;ff.atom_type_map=['C']
    ff.register_buffer('dataset_to_dpa_idx',torch.zeros(1,dtype=torch.long))
    pair=torch.tensor([[.017123,.15123,.61789],[1.123456,.76234,-.12345]],dtype=torch.float32)
    coords=pair.repeat(32,1);types=torch.zeros(64,dtype=torch.long)
    groups=torch.arange(32).repeat_interleave(2)
    energy,force=ff.compute_energy_and_forces(coords,types,node_batch_idx=groups,
                                            clip_forces=False,zero_com_forces=False)
    delta=pair[1].double()-pair[0].double()
    expected_force=torch.stack([delta,-delta]).float().repeat(32,1)
    torch.testing.assert_close(force,expected_force,atol=1e-7,rtol=0)
    torch.testing.assert_close(energy,torch.full((32,),.5*delta.square().sum(),dtype=torch.float64),atol=1e-10,rtol=0)
