"""Tests for plug-and-play `GeneratorSlot` switching and Online Advantage-Weighted Flow SFT (`awr_sft`)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn

from loom.weaving import (
    AssetRegistry,
    BuiltInRuleAsset,
    CallablePredictorAsset,
    CompositeCriticBus,
    GeneratorSlot,
    LoomFlowMolGenerator,
    LoomGRPOTrainer,
    TaskWeavingSpec,
    build_generator_slot,
)
from loom.weaving.uni3dar_slot import LoomUni3DARGenerator


def _make_3d_mol(smiles: str, seed: int = 42) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


class _MockFlowBackbone(nn.Module):
    """Pure-PyTorch mock FlowMol backbone enabling CPU tests without DGL."""

    is_mock_flow_backbone: bool = True

    def __init__(self, target_homo_by_idx: Sequence[float] | list[float] | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3)
        self.base_mols = [
            _make_3d_mol("CCC", seed=1),        # clean alkane (SA low)
            _make_3d_mol("CCO", seed=2),        # clean alcohol (SA low)
            _make_3d_mol("c1ccccc1", seed=3),   # benzene
            _make_3d_mol("c1ccncc1", seed=4),   # pyridine
        ]
        self._prop_vals = list(target_homo_by_idx) if target_homo_by_idx is not None else [-7.2, -6.7, -6.5, -6.0]

    def sample_trajectories(
        self,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        n_atoms: list[int],
        seed: int | None = None,
    ) -> list[dict]:
        del active_mask, n_atoms, seed
        trajs = []
        for i in range(targets_norm.shape[0]):
            m = Chem.Mol(self.base_mols[i % len(self.base_mols)])
            syms = [a.GetSymbol() for a in m.GetAtoms()]
            pos = np.asarray(m.GetConformer().GetPositions(), dtype=np.float32)
            # Inject an intentional formal-charge OR open-shell radical defect to test admission & repair
            if i % 4 == 3:
                bad = Chem.Mol(m)
                bad.GetAtomWithIdx(0).SetFormalCharge(1)
                bad_rdmol = bad
            elif i % 4 == 2:
                rad = Chem.Mol(m)
                rad.GetAtomWithIdx(0).SetNumRadicalElectrons(1)
                bad_rdmol = rad
            else:
                bad_rdmol = m
            trajs.append({"atom_symbols": syms, "positions_np": pos, "rdkit_mol": bad_rdmol})
        return trajs

    def compute_flow_loss(
        self,
        packed_batch: dict,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        return_per_molecule: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        del active_mask
        bsz = int(packed_batch["batch_size"])
        node_batch = packed_batch["node_batch_idx"]
        x1 = packed_batch["x_1_true"].float()
        t_node = targets_norm[:, 0][node_batch].unsqueeze(-1)
        feat_in = torch.cat([x1, t_node], dim=-1)
        pred = self.proj(feat_in)
        node_mse = ((pred - x1) ** 2).mean(dim=-1)
        per_mol = torch.zeros(bsz, device=x1.device, dtype=torch.float32).index_add(0, node_batch, node_mse)
        cnts = torch.bincount(node_batch, minlength=bsz).float().clamp_min(1.0)
        per_mol = per_mol / cnts
        info = {"loss_x": float(per_mol.detach().mean().item()), "loss_total": float(per_mol.detach().mean().item())}
        if return_per_molecule:
            info["per_molecule_generation_loss"] = per_mol
            return per_mol, info
        return per_mol.mean(), info


class _MockAutoRegSlot(GeneratorSlot):
    supports_exact_trajectory_logprob: bool = True
    generator_family: str = "octree_autoregressive"

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.w = self.backbone.weight
        self.target_cols = ["homo"]
        self.register_buffer("target_mean", torch.tensor([-6.5]))
        self.register_buffer("target_scale", torch.tensor([0.7]))

    def normalize_targets(self, targets_eV):
        return (torch.as_tensor(targets_eV, dtype=torch.float32) - self.target_mean) / self.target_scale

    def build_replay_dataset(self, mols, *, seed: int = 42):
        del mols, seed
        return None

    def sample(self, targets_norm, active_mask, **kwargs):
        del targets_norm, active_mask, kwargs
        return []

    def compute_sft_loss(self, packed_batch, targets_norm, active_mask, *, cfg_dropout_prob=0.0, return_per_molecule=False, **kwargs):
        del packed_batch, targets_norm, active_mask, cfg_dropout_prob, kwargs
        z = (self.w ** 2).mean()
        return (z.unsqueeze(0) if return_per_molecule else z), {}

    def trajectory_distributions(self, trajectories, targets_norm, active_mask, **kwargs):
        del trajectories, targets_norm, active_mask, kwargs
        return []


def _build_test_bus() -> CompositeCriticBus:
    spec = TaskWeavingSpec.from_dict(
        {
            "task_name": "test_awr_flow",
            "objectives": [
                {
                    "name": "homo",
                    "asset_id": "toy_qm",
                    "asset_channel": "homo",
                    "mode": "point",
                    "tolerance": 0.25,
                    "unit": "eV",
                    "weight": 1.0,
                }
            ],
            "constraints": [
                {
                    "name": "synthesizable_sa",
                    "asset_id": "builtin_rules",
                    "asset_channel": "sa_score",
                    "op": "leq",
                    "threshold": 4.5,
                    "soft_margin": 0.5,
                    "weight": 0.5,
                    "hard_gate": True,
                }
            ],
            "reconcile_valence": True,
        }
    )

    def _predict_homo(mols):
        vals = np.full((len(mols), 1), np.nan, dtype=np.float64)
        for i, m in enumerate(mols):
            if m is not None:
                # Deterministic proxy distinguishing molecular size/aromaticity
                vals[i, 0] = -7.5 + 0.12 * m.GetNumHeavyAtoms()
        return vals

    reg = AssetRegistry(register_builtin_rules=True)
    reg.register_predictor(
        CallablePredictorAsset(
            asset_id="toy_qm",
            predict_fn=_predict_homo,
            target_names=["homo"],
            units=["eV"],
        )
    )
    return CompositeCriticBus(spec=spec, registry=reg)


def test_generator_slot_hotswap_and_awr_auto_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_flow_gen = LoomFlowMolGenerator(
        backbone=_MockFlowBackbone(),
        atom_type_map=["H", "C", "N", "O"],
        target_cols=["homo"],
        target_units=["eV"],
        mean=[-6.6],
        scale=[0.65],
        sft_time_draws=2,
    )
    mock_ar_gen = _MockAutoRegSlot()

    monkeypatch.setattr(LoomUni3DARGenerator, "load_checkpoint", classmethod(lambda cls, *a, **kw: (mock_ar_gen, {})))
    monkeypatch.setattr(
        LoomFlowMolGenerator,
        "from_propmolflow_checkpoint",
        classmethod(lambda cls, *a, **kw: mock_flow_gen),
    )

    g_ar = build_generator_slot("uni3dar", "/tmp/fake_uni3dar.pt", device="cpu")
    g_fl = build_generator_slot("propmolflow", "/tmp/fake_pmf.pt", device="cpu")
    assert g_ar.supports_exact_trajectory_logprob is True
    assert g_fl.supports_exact_trajectory_logprob is False
    assert g_fl.generator_family == "ctmc_flow_matching"

    with pytest.raises(ValueError, match="Unsupported generator backend"):
        build_generator_slot("unsupported_gen", "/tmp/x.pt", device="cpu")
    with pytest.raises(NotImplementedError):
        g_fl.trajectory_distributions([], torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool))

    bus = _build_test_bus()
    with pytest.raises(ValueError, match="does not support exact trajectory log-probs"):
        LoomGRPOTrainer(g_fl, critic_bus=bus, alignment_mode="grpo")

    trainer_ar = LoomGRPOTrainer(g_ar, critic_bus=bus, alignment_mode="auto", kl_beta=0.02)
    assert trainer_ar.alignment_mode == "grpo"
    assert trainer_ar.reference is not None

    replay_mols = [_make_3d_mol("CCC", seed=10), _make_3d_mol("c1ccccc1", seed=11)]
    replay_y = np.array([[-7.14], [-6.78]], dtype=np.float32)

    trainer_fl = LoomGRPOTrainer(
        g_fl,
        critic_bus=bus,
        replay_molecules=replay_mols,
        replay_targets_eV=replay_y,
        alignment_mode="auto",
        lr=1e-2,
        replay_weight=0.35,
        replay_batch_size=2,
        awr_temperature=0.45,
        awr_max_weight=6.0,
        awr_repeat_k=2,
        grad_clip_mode="value",
        ppo_epochs=2,
        seed=7,
    )
    assert trainer_fl.alignment_mode == "awr_sft"
    assert trainer_fl.reference is None
    assert trainer_fl.replay_dataset is not None and len(trainer_fl.replay_dataset) == 2

    # Check rdkit_mol admission + ValenceReconciler repair on both radical (idx=2) and net-charged (idx=3) defects
    probe_trajs = g_fl.sample(torch.zeros((4, 1)), torch.ones((4, 1), dtype=torch.bool))
    admit = bus.admit_trajectories(probe_trajs)
    assert admit["raw_clean_charge_mask"][2] == False  # noqa: E712
    assert admit["raw_clean_charge_mask"][3] == False  # noqa: E712
    assert admit["legal_mask"].all()
    assert admit["clean_charge_mask"].all()

    # Verify FlowMolReplayDataset index alignment & strict rejection of unsupported atoms (`Au`)
    bad_elem_mol = Chem.RWMol(replay_mols[0])
    bad_elem_mol.GetAtomWithIdx(0).SetAtomicNum(79)  # Au
    rds = g_fl.build_replay_dataset([replay_mols[0], bad_elem_mol.GetMol(), replay_mols[1]])
    assert len(rds) == 3
    with pytest.raises(ValueError, match="incompatible with FlowMol"):
        rds.collate([1])
    ok_batch = rds.collate([0, 2])
    assert ok_batch["mode"] == "tensor" and ok_batch["batch_size"] == 2

    # Verify AWR weighted updates monotonically reduce weighted loss on fixed positive-advantage targets
    m_pos = replay_mols[0]
    m_high = replay_mols[1]
    enc_pair, kept_pair = g_fl.encode_molecules_batch([m_pos, m_high], repeat_k=1)
    assert enc_pair is not None and kept_pair == [0, 1]
    t_pair = g_fl.normalize_targets(np.array([[-7.14], [-6.78]], dtype=np.float32))
    m_pair = torch.ones((2, 1), dtype=torch.bool)
    w_pair = torch.tensor([0.2, 0.8], dtype=torch.float32)
    with torch.no_grad():
        l0, _ = g_fl.compute_sft_loss(enc_pair, t_pair, m_pair, return_per_molecule=True)
        w_loss_before = float((l0 * w_pair).sum().item())
    opt_check = torch.optim.SGD(g_fl.parameters(), lr=5e-2)
    for _ in range(5):
        opt_check.zero_grad()
        li, _ = g_fl.compute_sft_loss(enc_pair, t_pair, m_pair, return_per_molecule=True)
        (li * w_pair).sum().backward()
        opt_check.step()
    with torch.no_grad():
        l1, _ = g_fl.compute_sft_loss(enc_pair, t_pair, m_pair, return_per_molecule=True)
        w_loss_after = float((l1 * w_pair).sum().item())
    assert w_loss_after < w_loss_before

    w_before = g_fl.backbone.proj.weight.detach().clone()

    def _sampler(_s: int, _rng: np.random.Generator):
        return np.array([[-7.10], [-6.75]], dtype=np.float32), np.ones((2, 1), dtype=bool)

    hist = trainer_fl.train_loop(total_steps=3, prompt_sampler_fn=_sampler, group_size=4, log_every=1)
    assert len(hist) == 3
    for row in hist:
        assert row["alignment_mode"] == "awr_sft"
        assert row["awr_admitted"] >= 1
        assert np.isfinite(row["loss_total"])
        assert row["loss_replay"] > 0.0
        assert row["grad_norm"] > 0.0

    assert not torch.allclose(w_before, g_fl.backbone.proj.weight.detach())
