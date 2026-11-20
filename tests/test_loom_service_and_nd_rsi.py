"""Unit tests for the unified `Uni-Loom` Self-Driving Lab service architecture:
  1. `GeneratorSlot` capability declarations & `EFFORT_TO_M` (`base/low/medium/high <-> 1/2/4/8`).
  2. `LoomUni3DARGenerator` bundled `PostTreeGraphRefiner` + `TwoStageOctreeProcessCritic` persistence
     (without polluting `state_dict`), locked 4-tier funnels, and `generate_candidates`.
  3. `LoomFlowMolGenerator` fixed-`atom_budget` `generate_candidates` parity.
  4. `LoomService` symmetric zero-leakage `calibrate_on_dev`, zero-MLIP `generate`, capability-adaptive
     `align`, and 4-quadrant `GPAtlas` epistemic confidence envelopes.
  5. Multi-objective `RSIDiagnoser.diagnose_nd`, `InnerRSIRunner` $K$-D curriculum, partial `NaN` assay
     feedback ingestion in `ResidualCalibratedAsset`, and `F_phys`-aware `fit_octree_process_critic`.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn

from loom.generator.octree_model import AtomFeatDict, build_default_uni3dar_args
from loom.vendor.uni3dar.models.uni3dar_sampler import Uni3DARSampler
from loom.weaving import (
    EFFORT_TO_M,
    AssetRegistry,
    BuiltInRuleAsset,
    CallablePredictorAsset,
    CompositeCriticBus,
    HeteroscedasticGPAtlas1D,
    IncrementalCalibrator,
    InnerRSIRunner,
    LoomFlowMolGenerator,
    LoomService,
    LoomUni3DARGenerator,
    NUM_COARSE_GEOM_FEATS,
    NUM_ELEM_CLOUD_FEATS,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    PostTreeGraphRefiner,
    RSIDiagnoser,
    ResidualCalibratedAsset,
    TaskWeavingSpec,
    TwoStageOctreeProcessCritic,
    fit_octree_process_critic,
    resolve_effort_candidates,
)


def _make_mol(smiles: str, seed: int = 7) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


class _MockFlowBackbone(nn.Module):
    is_mock_flow_backbone: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3)
        self.mols = [_make_mol("CCO", 1), _make_mol("c1ccccc1", 2), _make_mol("c1ccncc1", 3)]
        self.recorded_n_atoms: list[int] = []

    def sample_trajectories(self, targets_norm: torch.Tensor, active_mask: torch.Tensor, *, n_atoms: list[int], seed=None):
        del active_mask, seed
        self.recorded_n_atoms = list(n_atoms)
        out = []
        for i in range(targets_norm.shape[0]):
            m = Chem.Mol(self.mols[i % len(self.mols)])
            syms = [a.GetSymbol() for a in m.GetAtoms()]
            pos = np.asarray(m.GetConformer().GetPositions(), dtype=np.float32)
            out.append({"atom_symbols": syms, "positions_np": pos, "rdkit_mol": m})
        return out

    def compute_flow_loss(self, packed_batch: dict, targets_norm: torch.Tensor, active_mask: torch.Tensor, *, return_per_molecule: bool = False):
        del active_mask
        bsz = int(packed_batch["batch_size"])
        nb = packed_batch["node_batch_idx"]
        x1 = packed_batch["x_1_true"].float()
        pred = self.proj(torch.cat([x1, targets_norm[:, 0][nb].unsqueeze(-1)], dim=-1))
        mse = ((pred - x1) ** 2).mean(dim=-1)
        per_mol = torch.zeros(bsz, dtype=torch.float32).index_add(0, nb, mse) / torch.bincount(nb, minlength=bsz).float().clamp_min(1.0)
        info = {"loss_total": float(per_mol.detach().mean().item()), "per_molecule_generation_loss": per_mol}
        return (per_mol, info) if return_per_molecule else (per_mol.mean(), info)


class _CountingForbiddenPotential:
    """Mock potential slot verifying ZERO inference-time MLIP calls inside `LoomService`."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_molecules(self, mols, chunk_size: int = 32):
        del chunk_size
        self.calls += 1
        n = len(mols)
        return {
            "energy_eV": np.zeros(n),
            "energy_per_atom_eV": np.zeros(n),
            "f_max_eV_A": np.ones(n),
            "f_rms_eV_A": np.ones(n) * 0.5,
        }


def _build_tiny_uni3dar() -> LoomUni3DARGenerator:
    args = build_default_uni3dar_args(batch_size=4)
    args.emb_dim = 32
    args.num_head = 4
    args.layer = 2
    args.merge_level = 6
    args.random_rotation_prob = 0.0
    backbone = Uni3DARSampler(args, AtomFeatDict(args))
    return LoomUni3DARGenerator(
        backbone=backbone,
        target_cols=["homo", "sa_score"],
        target_units=["eV", "sa"],
        mean=[-6.5, 2.5],
        scale=[0.7, 0.6],
        num_rbf=8,
        num_freqs=4,
        num_prefix_tokens=0,
        lora_rank=4,
        lora_alpha=8.0,
        use_valence_head=True,
        non_trapping_budget=True,
    )


def _build_mock_critic_bus() -> tuple[CompositeCriticBus, _CountingForbiddenPotential]:
    reg = AssetRegistry(register_builtin_rules=True)
    reg.register_predictor(
        CallablePredictorAsset(
            asset_id="mock_homo",
            predict_fn=lambda ms: (
                np.array([[-6.8 + 0.04 * m.GetNumAtoms(), -2.4 + 0.02 * m.GetNumAtoms()] for m in ms], dtype=np.float64),
                np.full((len(ms), 2), 0.12, dtype=np.float64),
            ),
            target_names=["homo", "lumo"],
            units=["eV", "eV"],
        )
    )
    pot = _CountingForbiddenPotential()
    reg.register_potential("forbid_mlip", pot)  # type: ignore[arg-type]
    spec = TaskWeavingSpec.from_dict(
        {
            "task_name": "test_sdl_homo_lumo",
            "potential_asset_id": "forbid_mlip",
            "objectives": [
                {"name": "homo", "asset_id": "mock_homo", "asset_channel": "homo", "tolerance": 0.25},
                {"name": "lumo", "asset_id": "mock_homo", "asset_channel": "lumo", "tolerance": 0.25},
            ],
            "constraints": [
                {"name": "sa_ok", "asset_id": "builtin_rules", "asset_channel": "sa_score", "op": "leq", "threshold": 4.5, "hard_gate": True}
            ],
            "auditors": [],
        }
    )
    return CompositeCriticBus(spec=spec, registry=reg), pot


def test_effort_tiers_and_uni3dar_bundle_serialization(tmp_path: Path):
    assert EFFORT_TO_M == {"base": 1, "low": 2, "medium": 4, "high": 8}
    assert resolve_effort_candidates("high") == ("high", 8)
    assert resolve_effort_candidates(n_candidates=4) == ("medium", 4)

    gen = _build_tiny_uni3dar()
    orig_sd_keys = set(gen.state_dict().keys())

    refiner = PostTreeGraphRefiner(num_atom_classes=gen.backbone.num_atom, d_model=32, n_heads=4, n_layers=2, n_rbf=8, d_pair=32)
    s1 = OctreeProcessCritic(emb_dim=32, num_targets=2, hidden_dim=64)
    s2 = OctreeElementProcessCritic(emb_dim=32, num_targets=2, num_atom_classes=gen.backbone.num_atom, d_node=32, n_rbf=8, n_gnn_layers=2, hidden_dim=64)
    ts_critic = TwoStageOctreeProcessCritic(s1, s2)
    gen.attach_post_processors(post_tree_refiner=refiner, process_critic=ts_critic)

    # Attaching post-processors must NOT pollute `state_dict()`
    assert set(gen.state_dict().keys()) == orig_sd_keys
    assert gen.supports_exact_trajectory_logprob is True
    assert gen.supports_prefix_process_search is True
    assert gen.resolve_internal_funnel(effort="base") == ("base", 1, 1, 1)
    assert gen.resolve_internal_funnel(effort="low") == ("low", 16, 4, 2)
    assert gen.resolve_internal_funnel(effort="medium") == ("medium", 32, 8, 4)
    assert gen.resolve_internal_funnel(effort="high") == ("high", 64, 16, 8)
    assert gen.resolve_internal_funnel(effort="medium", enable_internal_tree=False) == ("medium", 4, 4, 4)

    ckpt = tmp_path / "uni3dar_bundled.pt"
    gen.save_checkpoint(ckpt)
    loaded, _ = LoomUni3DARGenerator.load_checkpoint(ckpt, device="cpu", batch_size=4)
    assert loaded.post_tree_refiner is not None
    assert loaded.process_critic is not None


def test_loom_service_symmetric_calibration_zero_mlip_and_epistemic_envelope():
    bus, pot = _build_mock_critic_bus()
    flow_gen = LoomFlowMolGenerator(
        backbone=_MockFlowBackbone(),
        atom_type_map=("H", "C", "N", "O", "F"),
        target_cols=["homo"],
        target_units=["eV"],
        mean=[-6.5],
        scale=[0.7],
    )
    assert flow_gen.supports_exact_trajectory_logprob is False
    assert flow_gen.supports_prefix_process_search is False

    # Mount a 1D GPAtlas with a clear blindspot at deep HOMO (-8.5 eV) and converged zone near -6.4 eV
    req_fit = np.concatenate([np.full(20, -6.4), np.full(20, -6.8), np.full(15, -8.5)])
    pred_fit = req_fit + np.concatenate(
        [
            np.random.default_rng(1).normal(0.0, 0.08, 20),
            np.random.default_rng(2).normal(0.02, 0.10, 20),
            np.random.default_rng(3).normal(0.08, 0.12, 15),
        ]
    )
    eval_fit = np.ones_like(req_fit, dtype=bool)
    sig_cal = np.concatenate([np.full(40, 0.09), np.full(15, 0.55)])
    atlas = HeteroscedasticGPAtlas1D.fit_from_proposals(
        req_fit,
        pred_fit,
        eval_fit,
        sigma_cal_samples=sig_cal,
        source_property_values_eV=np.linspace(-7.2, -5.8, 80),
        axis_name="homo",
        unit="eV",
    )

    svc = LoomService(flow_gen, bus, gp_atlases={"homo": atlas})
    dev_t = np.linspace(-6.8, -6.0, 24, dtype=np.float32)
    dev_n = np.full(24, 12.0, dtype=np.float32)
    cal = svc.calibrate_on_dev(dev_t, dev_atom_budget=dev_n, calib_effort="low")
    assert 0.95 <= cal.alphas[0] <= 1.08
    assert pot.calls == 0  # Zero MLIP calls during dev calibration!

    out = svc.generate(np.array([-6.4, -8.5], dtype=np.float32), atom_budget=[15.0, 22.0], effort="medium", seed=12)
    assert len(out["molecules"]) == 2
    assert len(out["candidates_grid"]) == 2 and len(out["candidates_grid"][0]) == 4
    assert flow_gen.backbone.recorded_n_atoms == [15, 15, 15, 15, 22, 22, 22, 22]
    assert pot.calls == 0  # Strictly zero inference-time DPA-4 calls!
    assert out["epistemic_envelope"] is not None
    assert len(out["epistemic_envelope"]["service_envelopes"]) == 2


def test_multi_objective_diagnose_nd_partial_nan_assays_and_fphys_critic():
    # 1. Multi-objective `diagnose_nd`: channel 1 blindspot must freeze RL (`OUTER_BLINDSPOT`)
    q_homo = {
        "axis_name": "homo",
        "unit": "eV",
        "targets": [-6.5, -6.5, -7.8],
        "bias": [0.04, 0.05, 0.08],
        "sigma_gen": [0.18, 0.55, 0.22],
        "sigma_cal": [0.10, 0.12, 0.11],
        "p_eval": [0.95, 0.92, 0.90],
        "p_hit_reward": [0.75, 0.15, 0.65],
        "p_hit_dft_calibrated": [0.72, 0.14, 0.60],
        "source_kde_density": [0.8, 0.7, 0.5],
    }
    q_lumo = {
        "axis_name": "lumo",
        "unit": "eV",
        "targets": [-2.2, -2.8, -3.9],
        "bias": [0.05, 0.12, 0.10],
        "sigma_gen": [0.19, 0.45, 0.25],
        "sigma_cal": [0.11, 0.14, 0.45],  # OOD blindspot on LUMO at idx 2!
        "p_eval": [0.95, 0.90, 0.88],
        "p_hit_reward": [0.70, 0.20, 0.60],
        "p_hit_dft_calibrated": [0.68, 0.18, 0.30],
        "source_kde_density": [0.8, 0.6, 0.4],
    }
    diagnoser = RSIDiagnoser()
    diag_nd = diagnoser.diagnose_nd([q_homo, q_lumo], tolerances=[0.25, 0.25])
    assert diag_nd["rsi_zones"] == ["CONVERGED", "INNER_UNDEREXPLORED", "OUTER_BLINDSPOT"]
    assert diag_nd["curriculum_sampling_probs"][2] == 0.0  # RL frozen on blindspot!
    assert diag_nd["blindspot_channels"][2] == [1]

    # 2. InnerRSIRunner K-D curriculum sampler
    bus, _ = _build_mock_critic_bus()
    flow_2d = LoomFlowMolGenerator(
        backbone=_MockFlowBackbone(),
        atom_type_map=("H", "C", "N", "O", "F"),
        target_cols=["homo", "lumo"],
        target_units=["eV", "eV"],
        mean=[-6.5, -2.5],
        scale=[0.7, 0.8],
    )
    runner = InnerRSIRunner(flow_2d, bus, active_channel_indices=[0, 1], alignment_mode="auto")
    sampler_fn = runner.build_curriculum_sampler(diag_nd, prompts_per_step=6)
    t_mat, m_act = sampler_fn(0, np.random.default_rng(9))
    assert t_mat.shape == (6, 2) and m_act.all()

    # 3. Partial `NaN` assay readouts in `ResidualCalibratedAsset`
    base_2d = CallablePredictorAsset(
        asset_id="base_2d",
        predict_fn=lambda ms: (
            np.array([[-6.4, -2.2] for _ in ms], dtype=np.float64),
            np.full((len(ms), 2), 0.05, dtype=np.float64),
        ),
        target_names=["homo", "lumo"],
        units=["eV", "eV"],
    )
    cal_asset = ResidualCalibratedAsset(base_2d, asset_id="cal_2d")
    mols = [_make_mol(s, i + 1) for i, s in enumerate(["CCO", "CCC", "c1ccccc1", "c1ccncc1", "CC=O", "CC#N"])]
    cal_asset.seed_in_domain_anchors(mols[:4], np.array([[-6.4, -2.2]] * 4))
    partial_labels = np.array([[-6.9, np.nan], [np.nan, -2.8], [-7.1, np.nan], [-6.8, -2.6]], dtype=np.float64)
    stats = IncrementalCalibrator.ingest_labeled_feedback(cal_asset, mols[2:], partial_labels)
    assert stats["ingested_this_round"] == 4
    assert stats["ingested_per_channel"] == [3, 2]
    pred_post = cal_asset.predict_mols(mols[:2])
    assert np.isfinite(pred_post["mean"]).all()

    # 4. `F_phys`-aware `fit_octree_process_critic`
    critic = OctreeProcessCritic(emb_dim=16, num_targets=2, hidden_dim=32)
    n_r = 32
    st = fit_octree_process_critic(
        critic,
        h_concat=torch.randn(n_r, 64),
        geom_feats=torch.randn(n_r, NUM_COARSE_GEOM_FEATS),
        targets_norm=torch.zeros(n_r, 2),
        active_mask=torch.ones(n_r, 2, dtype=torch.bool),
        y_surrogate_norm=torch.randn(n_r, 2) * 0.4,
        y_valid_mask=torch.ones(n_r, dtype=torch.bool),
        is_legal=torch.ones(n_r, dtype=torch.bool),
        is_constr_met=torch.ones(n_r, dtype=torch.bool),
        phys_scores=torch.linspace(0.2, 0.95, n_r),
        group_ids=torch.arange(n_r) // 4,
        ranking_weight=0.5,
        epochs=3,
    )
    assert np.isfinite(st.train_loss)


def test_stoichiometric_descriptor_invariance_continuous_funnel_and_spec_driven_service():
    from rdkit.Chem import rdFingerprintGenerator
    from loom.weaving.registry import _mol_static_residual_descriptor

    # 1. 2D/3D explicit-H invariance + saturation discrimination of `_mol_static_residual_descriptor`
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256, includeChirality=True)
    m_with_h = _make_mol("c1ccncc1", seed=11)
    m_no_h = Chem.RemoveHs(Chem.Mol(m_with_h))
    d_with_h = _mol_static_residual_descriptor(m_with_h, fp_gen)
    d_no_h = _mol_static_residual_descriptor(m_no_h, fp_gen)
    assert d_with_h.shape == (29,)
    assert np.allclose(d_with_h, d_no_h, atol=1e-9)

    m_aliph = _make_mol("C1CCNCC1", seed=12)
    d_aliph = _mol_static_residual_descriptor(m_aliph, fp_gen)
    # Stoichiometry & hybridization entries (`[16:20]`: N/50, H/heavy, frac_csp3, frac_arom) cleanly separate aromatic vs aliphatic
    assert not np.allclose(d_with_h[16:20], d_aliph[16:20])

    # 2. Continuous combinatorial complexity tree-branching law (`m_keep = M` locked!)
    gen = _build_tiny_uni3dar()
    s1 = OctreeProcessCritic(emb_dim=32, num_targets=2, hidden_dim=64)
    s2 = OctreeElementProcessCritic(
        emb_dim=32, num_targets=2, num_atom_classes=gen.backbone.num_atom, d_node=32, n_rbf=8, n_gnn_layers=2, hidden_dim=64
    )
    gen.attach_post_processors(process_critic=TwoStageOctreeProcessCritic(s1, s2))

    assert gen.resolve_internal_funnel(effort="high", atom_budget=20.0) == ("high", 42, 12, 8)
    assert gen.resolve_internal_funnel(effort="high", atom_budget=23.0) == ("high", 48, 14, 8)
    assert gen.resolve_internal_funnel(effort="high", atom_budget=30.4) == ("high", 64, 16, 8)
    assert gen.resolve_internal_funnel(effort="medium", atom_budget=20.0) == ("medium", 22, 6, 4)
    assert gen.resolve_internal_funnel(effort="low", atom_budget=20.0) == ("low", 12, 4, 2)

    # Verify sync-free two-stage chunking across heterogeneous budgets (`[18, 20, 34, 46]`)
    t_norm = gen.normalize_targets(np.array([[-6.3, 2.0], [-6.5, 2.0], [-6.7, 2.0], [-6.9, 2.0]], dtype=np.float32))
    m_act = torch.ones((4, 2), dtype=torch.bool)
    c_grid = gen.generate_candidates(
        t_norm, m_act, atom_budget=[18.0, 20.0, 34.0, 46.0], effort="low", seed=42, max_decode_steps=36
    )
    assert len(c_grid) == 4 and all(len(row) == 2 for row in c_grid)

    # 3. Pure Spec-driven `LoomService` (`maximize` mode + `geq` auxiliary constraint, zero `"sa_score"`)
    reg = AssetRegistry(register_builtin_rules=True)
    reg.register_predictor(
        CallablePredictorAsset(
            asset_id="mock_potency",
            predict_fn=lambda ms: (
                np.array([[6.0 + 0.15 * m.GetNumAtoms()] for m in ms], dtype=np.float64),
                np.full((len(ms), 1), 0.10, dtype=np.float64),
            ),
            target_names=["pic50"],
            units=["pIC50"],
        )
    )
    spec_max = TaskWeavingSpec.from_dict(
        {
            "task_name": "test_maximize_pic50_with_qed_constraint",
            "objectives": [
                {"name": "pic50", "asset_id": "mock_potency", "asset_channel": "pic50", "mode": "maximize", "tolerance": 0.50}
            ],
            "constraints": [
                {"name": "drug_like_qed", "asset_id": "builtin_rules", "asset_channel": "qed", "op": "geq", "threshold": 0.40, "soft_margin": 0.10}
            ],
        }
    )
    bus_max = CompositeCriticBus(spec=spec_max, registry=reg)
    flow_qed = LoomFlowMolGenerator(
        backbone=_MockFlowBackbone(),
        atom_type_map=("H", "C", "N", "O", "F"),
        target_cols=["pic50", "qed"],
        target_units=["pIC50", "unitless"],
        mean=[7.0, 0.55],
        scale=[0.8, 0.15],
    )
    svc_max = LoomService(flow_qed, bus_max)
    assert "qed" in svc_max.calibration.auxiliary_anchors
    assert "sa_score" not in svc_max.calibration.auxiliary_anchors
    cal_max = svc_max.calibrate_on_dev(np.full(12, 8.0, dtype=np.float32), dev_atom_budget=np.full(12, 14.0, dtype=np.float32))
    assert cal_max.alphas == [1.0] and cal_max.betas == [0.0]  # Directional `maximize` skips affine regression
    gen_max = svc_max.generate(np.array([8.0, 8.5], dtype=np.float32), effort="medium", seed=7)
    # `_MockFlowBackbone` cycles 9-atom CCO, 12-atom benzene, 11-atom pyridine -> `select_best_of_m` picks highest `pic50`!
    assert [m.GetNumAtoms() for m in gen_max["molecules"]] == [12, 12]



def test_standalone_teacher_forced_features_and_single_speculative(tmp_path):
    from loom.generator.octree_backend import OctreeDataset
    from loom.weaving import PostTreeGraphRefiner

    gen = _build_tiny_uni3dar()
    s1 = OctreeProcessCritic(emb_dim=32, num_targets=2, hidden_dim=64)
    s2 = OctreeElementProcessCritic(
        emb_dim=32,
        num_targets=2,
        num_atom_classes=gen.backbone.num_atom,
        d_node=32,
        n_rbf=16,
        n_gnn_layers=2,
        hidden_dim=64,
        rbf_min=0.65,
        rbf_max=5.20,
        rbf_gamma=42.0,
        edge_value_gate=True,
        equivariant_multipole=True,
        h_gate_scale=0.0,
    )
    ref = PostTreeGraphRefiner(num_atom_classes=gen.backbone.num_atom, d_model=32, n_heads=4, n_layers=1, n_rbf=8, d_pair=16)
    gen.attach_post_processors(post_tree_refiner=ref, process_critic=TwoStageOctreeProcessCritic(s1, s2))

    ckpt_p = tmp_path / "bundled_standalone.pt"
    gen.save_checkpoint(ckpt_p)
    gen_reloaded, _ = LoomUni3DARGenerator.load_checkpoint(ckpt_p, device="cpu", batch_size=4)
    assert gen_reloaded.process_critic.stage2_critic.edge_value_gate is True
    assert gen_reloaded.process_critic.stage2_critic.equivariant_multipole is True
    assert gen_reloaded.process_critic.stage2_critic.h_gate_scale == 0.0

    mols = [_make_mol("CCO", 1), _make_mol("c1ccccc1", 2), _make_mol("c1ccncc1", 3)]
    oct_ds = OctreeDataset(mols, gen_reloaded.backbone, training=False)
    packed = oct_ds.collate([0, 1, 2])

    t_norm = gen_reloaded.normalize_targets(np.array([[-6.3, 2.0], [-6.5, 2.1], [-6.7, 2.2]], dtype=np.float32))
    m_act = torch.ones((3, 2), dtype=torch.bool)
    feats = gen_reloaded.extract_teacher_forced_two_stage_features(packed, t_norm, m_act)
    assert feats["s1_h_concat"].shape == (3, 128)
    assert feats["s1_geom_feats"].shape == (3, NUM_COARSE_GEOM_FEATS)
    assert feats["s2_h_stage2"].shape == (3, 192)
    assert feats["s2_cloud_feats"].shape == (3, 42)
    assert [int(es.numel()) for es in feats["s2_elem_slices"]] == [m.GetNumAtoms() for m in mols]

    mols_k1, fun_k1 = gen_reloaded.generate_single_speculative(
        t_norm[:2], m_act[:2], speculative_scale=1, atom_budget=[18.0, 18.0], seed=11
    )
    assert len(mols_k1) == 2 and fun_k1 == (1, 1, 1)

    mols_k8, fun_k8 = gen_reloaded.generate_single_speculative(
        t_norm[:2], m_act[:2], speculative_scale=8, atom_budget=[18.0, 18.0], seed=17
    )
    assert len(mols_k8) == 2 and fun_k8 == (38, 12, 1)
