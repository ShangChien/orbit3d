import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from orbit3d.generator.octree_tokenizer import *
from orbit3d.generator.uni3dar_model import ConditionalUni3DAR


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
    decoded = decode_tokens_to_mol3d(encode_mol_to_octree_tokens(inst))
    out = mol3d_to_validated_rdkit_mol(decoded)
    assert rdMolDescriptors.CalcMolFormula(out) == rdMolDescriptors.CalcMolFormula(mol)
    assert Chem.MolToSmiles(Chem.RemoveHs(out)) == Chem.MolToSmiles(Chem.RemoveHs(mol))


def test_no_substitution_or_truncation():
    mol = Chem.MolFromSmiles("[Ir+3]")
    mol.AddConformer(Chem.Conformer(1))
    with pytest.raises(ValueError):
        rdkit_mol_to_mol3d_instance(mol, np.array([1.0]))
    inst = rdkit_mol_to_mol3d_instance(embedded("C" * 20), np.array([1.0]))
    with pytest.raises(ValueError):
        encode_mol_to_octree_tokens(inst)
    with pytest.raises(ValueError):
        decode_tokens_to_mol3d([BOS_TOKEN, EOS_TOKEN])


def test_disconnected_invalid():
    inst = Mol3DInstance(np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]]), np.array([0, 0, 0]))
    assert compute_mol_stability(inst)["mol_stable"] == 0


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


def test_eos_and_length():
    model = ConditionalUni3DAR(d_model=16, n_heads=2, n_layers=1)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[EOS_TOKEN] = 1000
    seq, _, _ = model.sample_autoregressive(torch.zeros(1, 7))
    assert seq.shape[1] == 6


from orbit3d.constraints import TargetIntervals
from orbit3d.generator.grpo_trainer import policy_objective, gated_reward
from orbit3d.physics.conformer_relaxer import relax_conformer_iso_property, project_force
from orbit3d.oracles.base import PropertyOracleBase
from orbit3d.oracles.uniqsar_oracle import UniQSARTrainedBridge
from types import SimpleNamespace
import copy


def test_intervals_named_masked_and_joint():
    iv = TargetIntervals.parse({"b": (0, 1), "a": (0, 1)}, ["a", "b"])
    assert not iv.hit([0.5, 99])
    assert iv.hit([0.5, 0.9])
    iv = TargetIntervals.parse([None, (None, 1)], ["a", "b"])
    assert iv.hit([99, 0.9])
    assert iv.condition([0, 0], [1, 1]).tolist() == [0, 0, 0, 0, 0, 1, 0, 1]
    with pytest.raises(ValueError):
        TargetIntervals.parse({"bad": (0, 1)}, ["a"])


def test_kl_has_gradient_and_no_reference_gradient():
    torch.manual_seed(7)
    model = ConditionalUni3DAR(d_model=16, n_heads=2, n_layers=1, max_atoms=2)
    ref = copy.deepcopy(model)
    with torch.no_grad():
        ref.lm_head.bias.add_(torch.linspace(-2, 2, model.vocab_size))
    seq, old, _ = model.sample_autoregressive(torch.zeros(2, 7), temperature=1, top_p=1)
    loss, _ = policy_objective(model, ref, seq, torch.zeros(2, 7), old, torch.zeros(2), 1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.lm_head.bias.grad.abs().sum() > 0
    assert all(p.grad is None for p in ref.parameters())


def test_reward_and_projection():
    assert gated_reward(1, 0.5, 100) < gated_reward(1, 0.5, 1)
    assert gated_reward(1, 2, 1) < gated_reward(1, 0.5, 1)
    assert gated_reward(0, 0, float("nan")) == -1
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


def test_missing_generator_fails_before_loading_oracle(tmp_path):
    import json
    from orbit3d.pipeline.runner import run_from_config

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"mode": "predict", "mount_paths": {"checkpoint_load_dir": str(tmp_path / "missing")}})
    )
    with pytest.raises(FileNotFoundError, match="generator"):
        run_from_config(str(path))


def test_pipeline_train_reload_all_records(tmp_path, monkeypatch):
    import json, pandas as pd
    import orbit3d.pipeline.runner as runner

    class Oracle(PropertyOracleBase):
        def __init__(self, **kwargs):
            super().__init__()
            self.target_cols = ["a", "b"]
            self.output_dim = 2

        def fit_from_dataset(self, *a, **kw):
            return {}

        @classmethod
        def load_from_dir(cls, *a, **kw):
            return cls()

        def predict_3d_mols(self, mols):
            return np.tile([0.5, 99.0], (len(mols), 1)), np.zeros((len(mols), 2))

    monkeypatch.setattr(runner, "InternalDPA4PropertyOracle", Oracle)
    monkeypatch.setattr(runner, "resolve_dpa4_weight_path", lambda *a, **k: "mock")
    monkeypatch.setattr(runner, "DPAPotentialGuidance", lambda **k: SimpleNamespace())
    csv = tmp_path / "train.csv"
    pd.DataFrame({"SMILES": ["CC", "CO"], "a": [0.1, 0.2], "b": [0.3, 0.4]}).to_csv(csv, index=False)
    cfg = {
        "mode": "train_and_predict",
        "device": "cpu",
        "mount_paths": {
            "train_data_path": str(csv),
            "checkpoint_save_dir": str(tmp_path / "weights"),
            "output_dir": str(tmp_path / "out"),
        },
        "oracle_config": {"oracle_mode": "internal_dpa4", "target_cols": ["a", "b"]},
        "generator_config": {"d_model": 16, "n_layers": 1, "n_heads": 2, "sft_epochs": 1, "grpo_steps": 0},
        "stage5_relax_config": {"enabled": False},
        "inference_config": {"num_molecules": 3, "target_interval": {"b": [0, 1], "a": [0, 1]}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    # Real tiny generator trains and reloads; failures remain in the attempted denominator.
    summary = runner.run_from_config(str(path))
    before = pd.read_csv(summary["output_csv"]).fillna("")
    cfg["mode"] = "predict"
    path.write_text(json.dumps(cfg))
    summary = runner.run_from_config(str(path))
    pd.testing.assert_frame_equal(before, pd.read_csv(summary["output_csv"]).fillna(""))
    assert summary["num_molecules"] == 3
    # Controlled proposals separately exercise successful SDF/multi-target export.
    inst = rdkit_mol_to_mol3d_instance(embedded("CC"), np.array([0.5, 99.0]))
    monkeypatch.setattr(runner, "sample_molecule_deterministic_by_id", lambda **kw: (inst, torch.zeros(2)))
    summary = runner.run_from_config(str(path))
    assert summary["valid_count"] == 3 and summary["hit_rate"] == 0
    assert len(list(Chem.SDMolSupplier(summary["output_sdf"], removeHs=False))) == 3


def test_sdf_sparse_indices_and_empty_data(tmp_path, monkeypatch):
    import pandas as pd
    from orbit3d.utils.dataset_io import load_3d_molecular_dataset, PandasTools

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


@pytest.mark.parametrize("batch", [1, 4])
def test_incremental_logits_match_prefix(batch):
    torch.manual_seed(123)
    model = ConditionalUni3DAR(d_model=32, n_heads=4, n_layers=2).eval()
    seq = torch.randint(0, model.vocab_size, (batch, 17))
    cond = torch.randn(batch, 7)
    cache = None
    with torch.no_grad():
        for step in range(seq.shape[1]):
            full, _ = model(seq[:, : step + 1], cond)
            incremental, cache = model.forward_step(seq[:, step], cond, cache)
            torch.testing.assert_close(incremental[:, -1], full[:, -1], atol=2e-6, rtol=1e-5)


def test_raw_nan_force_rejected():
    inst = rdkit_mol_to_mol3d_instance(embedded("CC"), np.array([1.0]))

    class BadForce:
        device_str = "cpu"

        def compute_energy_and_forces(self, x, *a, **kw):
            return torch.tensor([0.0]), torch.full_like(x, float("nan"))

    with pytest.raises(FloatingPointError):
        relax_conformer_iso_property(inst, BadForce(), BondOracle(), (0, 10))


def test_radical_training_is_rejected():
    inst = rdkit_mol_to_mol3d_instance(embedded("[CH3]"), np.array([1.0]))
    with pytest.raises(ValueError, match="chemical validation"):
        encode_mol_to_octree_tokens(inst)


def test_refinement_freezes_inferred_graph():
    original = rdkit_mol_to_mol3d_instance(embedded("CC"), np.array([1.0]))
    inst = decode_tokens_to_mol3d(encode_mol_to_octree_tokens(original))
    d = float(np.linalg.norm(inst.coords[1] - inst.coords[0]))
    result, _ = relax_conformer_iso_property(inst, BondForce(d), BondOracle(), (0, 10), n_steps=0)
    assert result.graph is not None
    assert Chem.MolToSmiles(Chem.RemoveHs(result.graph)) == "CC"
