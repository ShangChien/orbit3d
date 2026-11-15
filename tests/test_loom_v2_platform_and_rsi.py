"""Comprehensive unit tests for `loom v2.0`:
  1. Exotic Zwitterion Elimination (`ValenceReconciler`, `HeavyAtomValenceHead`, Neutral Valence Guard)
  2. 3-Layer Predictor-Agnostic Weaving (`AssetRegistry`, `BuiltInRuleAsset`, `CallablePredictorAsset`,
     `ResidualCalibratedAsset`, `PhysicsConstrainedCoordRefiner`, `TaskWeavingSpec`, `CompositeCriticBus`)
  3. Generalized `GPAtlas` Gatekeepers & Grounded RSI Engine (`RSIDiagnoser`, `OuterRSIAcquisition`,
     `IncrementalCalibrator`, `InnerRSIRunner`)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from loom.chemistry import ValenceReconciler, reconcile_3d_point_cloud
from loom.generator.molecular_state import mol_to_clean_smiles
from loom.generator.octree_backend import OctreeDataset
from loom.generator.octree_model import AtomFeatDict, build_default_uni3dar_args
from loom.oracles.applicability import MorganDomain
from loom.vendor.uni3dar.models.uni3dar_sampler import Uni3DARSampler
from loom.weaving import (
    AssetRegistry,
    AuditorChannelSpec,
    BuiltInRuleAsset,
    CallablePredictorAsset,
    CompositeCriticBus,
    ConstraintChannelSpec,
    HeteroscedasticGPAtlas1D,
    HeteroscedasticGPAtlas2D,
    IncrementalCalibrator,
    LoomGRPOTrainer,
    LoomUni3DARGenerator,
    ObjectiveChannelSpec,
    OuterRSIAcquisition,
    PhysicsConstrainedCoordRefiner,
    PotentialSlot,
    RSIDiagnoser,
    RSIDiagnosticThresholds,
    ResidualCalibratedAsset,
    TaskWeavingSpec,
    formal_charge_signature,
    is_clean_organic_charge_state,
)


def _make_3d_mol(smiles: str, seed: int = 1) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if res != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _extract_point_cloud(mol: Chem.Mol) -> tuple[list[str], np.ndarray]:
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    return syms, pos


def _make_small_v2_generator(num_props: int = 2) -> LoomUni3DARGenerator:
    args = build_default_uni3dar_args(batch_size=4)
    args.emb_dim = 64
    args.num_head = 4
    args.layer = 3
    args.merge_level = 6
    args.random_rotation_prob = 0.0
    backbone = Uni3DARSampler(args, AtomFeatDict(args))
    gen = LoomUni3DARGenerator(
        backbone=backbone,
        target_cols=["homo", "lumo"][:num_props],
        target_units=["eV"] * num_props,
        mean=[-6.0, -2.0][:num_props],
        scale=[0.8, 0.9][:num_props],
        num_rbf=16,
        num_freqs=8,
        num_prefix_tokens=2,
        lora_rank=8,
        lora_alpha=16.0,
    )
    gen.enable_v20_valence_head()
    return gen


# ==============================================================================
# Phase 1 Tests: ValenceReconciler + HeavyAtomValenceHead + Neutral Valence Guard
# ==============================================================================


def test_valence_reconciler_clean_nitro_zwitterion_and_missing_h():
    reconciler = ValenceReconciler()

    # 1. Clean phenol passes through completely untouched
    phenol = _make_3d_mol("Oc1ccccc1", seed=10)
    syms_p, pos_p = _extract_point_cloud(phenol)
    res_p = reconciler.reconcile_point_cloud(syms_p, pos_p)
    assert res_p.status == "clean_raw"
    assert res_p.was_reconciled is False
    assert res_p.reconciled_legal is True
    assert res_p.reconciled_exotic_charge is False
    assert res_p.heavy_rmsd_A == 0.0

    # 2. Canonical nitrobenzene (where RDKit DetermineBonds(charge=0) fails with charge=-1)
    nitro = _make_3d_mol("O=[N+]([O-])c1ccccc1", seed=11)
    syms_n, pos_n = _extract_point_cloud(nitro)
    res_n = reconciler.reconcile_point_cloud(syms_n, pos_n)
    assert res_n.reconciled_legal is True
    assert res_n.reconciled_exotic_charge is False
    assert res_n.heavy_rmsd_A < 1e-5
    assert res_n.mol is not None
    assert is_clean_organic_charge_state(res_n.mol)
    assert formal_charge_signature(res_n.mol) == (("N", 1), ("O", -1))
    # Ensure phenyl ring did not become a strained cyclic cumulene!
    assert "=C=" not in (res_n.smiles or "")

    # 3. Separated zwitterion `[CH2-]CC[OH+]C` (shifted H from terminal C onto ether O) -> neutral `CCOC`
    ether = _make_3d_mol("CCOC", seed=12)
    rw = Chem.RWMol(ether)
    conf = rw.GetConformer()
    # Find ether O and a terminal methyl H far from O, relocate that H near O
    o_idx = next(a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "O")
    o_pos = np.array(conf.GetAtomPosition(o_idx), dtype=np.float64)
    h_indices = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "H"]
    far_h = max(
        h_indices,
        key=lambda hi: np.linalg.norm(np.array(conf.GetAtomPosition(hi), dtype=np.float64) - o_pos),
    )
    new_h_pos = o_pos + np.array([0.0, 0.0, 0.97])
    syms_zw, pos_zw = _extract_point_cloud(ether)
    pos_zw[far_h] = new_h_pos.astype(np.float32)

    res_zw = reconcile_3d_point_cloud(syms_zw, pos_zw)
    assert res_zw.raw_exotic_charge is True
    assert res_zw.reconciled_legal is True
    assert res_zw.reconciled_exotic_charge is False
    assert res_zw.was_reconciled is True
    assert res_zw.heavy_rmsd_A < 1e-5
    assert res_zw.smiles == mol_to_clean_smiles(ether)

    # 4. Missing peripheral H on benzonitrile (`N#Cc1ccccc1 - 1H`) -> recovered exact benzonitrile
    bn = _make_3d_mol("N#Cc1ccccc1", seed=13)
    syms_b, pos_b = _extract_point_cloud(bn)
    h_drop = next(i for i, s in enumerate(syms_b) if s == "H")
    syms_miss = [s for i, s in enumerate(syms_b) if i != h_drop]
    pos_miss = np.delete(pos_b, h_drop, axis=0)

    res_bn = reconciler.reconcile_point_cloud(syms_miss, pos_miss)
    assert res_bn.raw_legal is False
    assert res_bn.reconciled_legal is True
    assert res_bn.reconciled_exotic_charge is False
    assert res_bn.delta_h_count == 1
    assert res_bn.heavy_rmsd_A < 1e-5
    assert res_bn.smiles == mol_to_clean_smiles(bn)

    # 5. Unphysical pentavalent carbon skeleton -> honestly rejected (no hallucination)
    syms_bad = ["C", "C", "C", "C", "C", "C"]
    pos_bad = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.45, 0.0, 0.0],
            [-1.45, 0.0, 0.0],
            [0.0, 1.45, 0.0],
            [0.0, -1.45, 0.0],
            [0.0, 0.0, 1.45],
        ],
        dtype=np.float32,
    )
    res_bad = reconciler.reconcile_point_cloud(syms_bad, pos_bad)
    assert res_bad.reconciled_legal is False
    assert res_bad.mol is None


def test_v20_valence_head_sft_checkpoint_and_guard_logprob_match(tmp_path: Path):
    torch.manual_seed(2026)
    gen = _make_small_v2_generator(num_props=2)
    assert gen.valence_head is not None
    assert gen.loom_config["use_valence_head"] is True

    mols = [_make_3d_mol("CC=O", seed=1), _make_3d_mol("c1ccccc1", seed=2)]
    ds = OctreeDataset(mols, gen.backbone, training=True, seed=0)
    packed = ds.collate([0, 1])
    t_norm = gen.normalize_targets(torch.tensor([[-6.2, -1.5], [-5.8, -2.1]]))
    m_act = torch.ones((2, 2), dtype=torch.bool)

    gen.train()
    loss, info = gen.compute_sft_loss(packed, t_norm, m_act, cfg_dropout_prob=0.0, valence_loss_weight=0.25)
    assert torch.isfinite(loss)
    assert info["loss_valence"] > 0.0
    loss.backward()
    val_grads = [p.grad.abs().sum().item() for p in gen.valence_head.parameters() if p.grad is not None]
    assert sum(val_grads) > 0.0

    ckpt = tmp_path / "loom_v20.pt"
    gen.save_checkpoint(ckpt)
    loaded, _ = LoomUni3DARGenerator.load_checkpoint(ckpt, device="cpu")
    assert loaded.valence_head is not None
    assert loaded.loom_config["use_valence_head"] is True

    loaded.eval()
    trajs = loaded.sample(
        t_norm,
        m_act,
        cfg_scale=1.7,
        xyz_cfg_scale=1.0,
        geometric_guard=True,
        temperatures=(1.0, 1.0, 1.0),
        seed=404,
        use_bf16=False,
        electron_parity=0,
    )
    packets = loaded.trajectory_distributions(
        trajs,
        t_norm,
        m_act,
        cfg_scale=1.7,
        xyz_cfg_scale=1.0,
        geometric_guard=True,
        electron_parity=0,
    )
    for b_i, tr in enumerate(trajs):
        dist_sum = sum(
            float(log_p[s_ids == b_i].gather(-1, acts[s_ids == b_i].unsqueeze(-1)).sum().item())
            for log_p, acts, s_ids in packets
            if bool((s_ids == b_i).any().item())
        )
        sample_sum = float(tr["scores"].sum().item())
        assert abs(dist_sum - sample_sum) < max(1e-4, 2e-6 * abs(sample_sum))


# ==============================================================================
# Phase 2 Tests: 3-Layer Predictor-Agnostic Registry, Spec & CompositeCriticBus
# ==============================================================================


def test_builtin_rules_callable_predictor_and_coord_refiner():
    rules = BuiltInRuleAsset()
    benzoic = _make_3d_mol("O=C(O)c1ccccc1", seed=1)
    peroxide = _make_3d_mol("COOC", seed=2)
    cyclopropane = _make_3d_mol("C1CC1", seed=3)

    out = rules.predict_mols([benzoic, peroxide, cyclopropane, None])
    assert out["valid_mask"].tolist() == [True, True, True, False]
    sa_idx = rules.channel_index("sa_score")
    smarts_idx = rules.channel_index("smarts_alert")
    ring_idx = rules.channel_index("strained_small_rings")
    assert out["mean"][0, sa_idx] < 2.5
    assert out["mean"][0, smarts_idx] == 0.0
    assert out["mean"][1, smarts_idx] >= 1.0  # peroxide alert!
    assert out["mean"][2, ring_idx] >= 1.0    # 3-membered strained ring!

    # Differentiable 3D Callable Asset (toy radius-of-gyration property) + PhysicsConstrainedCoordRefiner
    def _rg_predict(mols):
        vals = []
        for m in mols:
            pos = m.GetConformer().GetPositions()
            c = pos - pos.mean(axis=0, keepdims=True)
            vals.append([float(np.sqrt(np.mean(np.sum(c ** 2, axis=-1))))])
        return np.asarray(vals), np.full((len(mols), 1), 0.05)

    def _rg_grad(mol: Chem.Mol, ch: int):
        assert ch == 0
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, requires_grad=True)
        c = pos - pos.mean(dim=0, keepdim=True)
        rg = torch.sqrt(torch.mean(torch.sum(c ** 2, dim=-1)))
        (g,) = torch.autograd.grad(rg, pos)
        return float(rg.item()), 0.05, g.detach().numpy().astype(np.float64)

    diff_asset = CallablePredictorAsset(
        "rg_3d_sensor", _rg_predict, target_names=["rg"], units=["A"], grad_fn=_rg_grad
    )
    assert diff_asset.capabilities.is_3d_differentiable is True

    dummy_pot = PotentialSlot.__new__(PotentialSlot)
    dummy_pot.backend_name = "dummy"
    dummy_pot.atom_types = ["H", "C", "N", "O"]
    dummy_pot.device_str = "cpu"
    dummy_pot._pack_molecules = lambda ms: (
        torch.tensor(ms[0].GetConformer().GetPositions(), dtype=torch.float32),
        torch.zeros(ms[0].GetNumAtoms(), dtype=torch.long),
        torch.zeros(ms[0].GetNumAtoms(), dtype=torch.long),
    )
    dummy_pot.engine = SimpleNamespace(
        compute_energy_and_forces=lambda coords, atypes, b_idx, **kw: (
            torch.tensor([-1.0]),
            torch.zeros_like(coords),
        )
    )
    refiner = PhysicsConstrainedCoordRefiner(dummy_pot, diff_asset, prop_weight=0.5, step_size=0.02)
    eth = _make_3d_mol("CCO", seed=5)
    init_rg, _, _ = diff_asset.grad_coords(eth, "rg")
    ref_out = refiner.refine_molecule(eth, target_value=init_rg + 0.08, channel="rg", tolerance=0.02, n_steps=6)
    assert ref_out["accepted"] is True
    assert ref_out["final_abs_err"] < ref_out["init_abs_err"]
    assert mol_to_clean_smiles(ref_out["mol"]) == mol_to_clean_smiles(eth)


def test_residual_calibrated_asset_bayesian_contraction_and_persistence(tmp_path: Path):
    # Audit-only assets must be rejected
    aud_callable = CallablePredictorAsset("aud_only", lambda ms: np.zeros((len(ms), 1)), ["y"], is_audit_only=True)
    with pytest.raises(ValueError, match="audit-only"):
        ResidualCalibratedAsset(aud_callable)

    # Simulated truncated enterprise HOMO predictor: clamps everything at >= -6.00 eV!
    in_domain_smiles = ["CCN", "CCCO", "c1ccc(O)cc1", "c1ccc(OC)cc1", "CC(C)N", "c1ccncc1", "CCOC", "CC(=O)N"]
    ood_deep_smiles = [
        "N#Cc1ccc(C#N)cc1",
        "Fc1c(F)c(F)c(C#N)c(F)c1F",
        "O=[N+]([O-])c1ccc(C#N)cc1",
        "Clc1ccc(C(=O)C#N)cc1",
        "Fc1ccc(S(=O)(=O)C(F)(F)F)cc1",
        "N#CC(=C(C#N)C#N)C#N",
    ]
    in_mols = [_make_3d_mol(s, seed=i + 1) for i, s in enumerate(in_domain_smiles)]
    ood_mols = [_make_3d_mol(s, seed=i + 20) for i, s in enumerate(ood_deep_smiles)]

    def _true_synthetic_homo(m: Chem.Mol) -> float:
        no_h = Chem.RemoveHs(Chem.Mol(m))
        n_ewg = sum(1 for a in no_h.GetAtoms() if a.GetSymbol() in ("F", "Cl", "N", "S"))
        return -5.45 - 0.25 * n_ewg

    in_true = np.array([[_true_synthetic_homo(m)] for m in in_mols], dtype=np.float64)
    ood_true = np.array([[_true_synthetic_homo(m)] for m in ood_mols], dtype=np.float64)

    # Frozen base predictor truncates at -6.00 eV (mimicking RF leaf floor!)
    base_asset = CallablePredictorAsset(
        "truncated_enterprise_qsar",
        lambda ms: (
            np.array([[max(_true_synthetic_homo(m), -6.00)] for m in ms], dtype=np.float64),
            np.full((len(ms), 1), 0.08),
        ),
        target_names=["homo"],
        units=["eV"],
    )
    cal = ResidualCalibratedAsset(base_asset, prior_blindspot_std=0.40)

    # 1. Exact zero perturbation before seeding or feedback
    pre_zero = cal.predict_mols(ood_mols)
    assert np.allclose(pre_zero["mean"], base_asset.predict_mols(ood_mols)["mean"])
    assert np.all(pre_zero["std"] > 0.40)

    # 2. Seed in-domain anchors -> low std in-domain, high std OOD
    cal.seed_in_domain_anchors(in_mols, in_true)
    in_pred = cal.predict_mols(in_mols)
    ood_pre = cal.predict_mols(ood_mols)
    assert float(np.mean(in_pred["std"])) < float(np.mean(ood_pre["std"]))
    assert float(np.mean(ood_pre["std"])) > 0.30

    # 3. Ingest 4 OOD molecules via IncrementalCalibrator; test on held-out 2 OOD molecules
    cal_stats = IncrementalCalibrator.ingest_labeled_feedback(
        cal,
        ood_mols[:4],
        ood_true[:4],
        validation_mols=ood_mols[4:],
        validation_true_targets=ood_true[4:],
        save_path=tmp_path / "res_cal.joblib",
    )
    assert cal_stats["post_calibration_blindspot_mae"][0] < cal_stats["pre_calibration_blindspot_mae"][0]
    assert cal_stats["post_calibration_mean_std"][0] < cal_stats["pre_calibration_mean_std"][0]

    # 4. Reload calibrator into a fresh wrapper and verify identical predictions
    cal_reloaded = ResidualCalibratedAsset(base_asset)
    cal_reloaded.load_calibrator(tmp_path / "res_cal.joblib")
    assert np.allclose(cal.predict_mols(ood_mols[4:])["mean"], cal_reloaded.predict_mols(ood_mols[4:])["mean"])


def test_task_spec_yaml_and_composite_critic_bus_with_grpo(tmp_path: Path, monkeypatch):
    # Verify bundled YAML configs load and round-trip cleanly
    repo_root = Path(__file__).resolve().parents[1]
    spec_synth = TaskWeavingSpec.from_file(repo_root / "configs/tasks/oe62_orbital_synthesizable.yaml")
    assert spec_synth.objective_names == ["homo", "lumo"]
    assert len(spec_synth.constraints) == 3
    roundtrip_yaml = tmp_path / "roundtrip.yaml"
    spec_synth.save(roundtrip_yaml)
    reloaded_spec = TaskWeavingSpec.from_file(roundtrip_yaml)
    assert reloaded_spec.to_dict() == spec_synth.to_dict()

    # Audit-only isolation check inside CompositeCriticBus
    reg = AssetRegistry(register_builtin_rules=True)
    reg.register_predictor(
        CallablePredictorAsset("secret_audit", lambda ms: np.zeros((len(ms), 1)), ["homo"], is_audit_only=True)
    )
    bad_spec = TaskWeavingSpec(
        task_name="illegal_leak",
        objectives=(ObjectiveChannelSpec(name="homo", asset_id="secret_audit", asset_channel="homo"),),
    )
    with pytest.raises(ValueError, match="Audit-only"):
        CompositeCriticBus(bad_spec, reg)

    # Multi-asset CompositeCriticBus with single-oracle AD + SA/SMARTS constraints + GRPO step
    reg.register_predictor(
        CallablePredictorAsset(
            "ext_homo",
            lambda ms: (np.full((len(ms), 1), -6.10), np.full((len(ms), 1), 0.10)),
            target_names=["homo"],
            units=["eV"],
        )
    )
    good_spec = TaskWeavingSpec(
        task_name="homo_clean_synth",
        objectives=(ObjectiveChannelSpec(name="homo", asset_id="ext_homo", asset_channel="homo", tolerance=0.25),),
        constraints=(
            ConstraintChannelSpec(name="sa_ok", asset_id="builtin_rules", asset_channel="sa_score", op="leq", threshold=3.5),
            ConstraintChannelSpec(name="no_peroxide", asset_id="builtin_rules", asset_channel="smarts_alert", op="leq", threshold=0.0),
        ),
        auditors=(AuditorChannelSpec(name="ad_guard", objective_name="homo", mode="single_oracle_ad"),),
    )
    clean_mol = _make_3d_mol("c1ccccc1", seed=1)       # Clean + SA<=3.5 + no alert -> Constrained Clean Hit!
    peroxide_mol = _make_3d_mol("COOC", seed=2)        # Violates hard SMARTS alert!
    exotic_mol = Chem.MolFromSmiles("[CH2-]CC[OH+]C")  # Exotic zwitterion!

    ad = MorganDomain(["c1ccccc1", "CCO"], cutoff=0.25)
    bus = CompositeCriticBus(good_spec, reg, applicability_domain=ad)
    ev = bus.evaluate_molecules(
        [clean_mol, peroxide_mol, exotic_mol, None],
        requested_targets=np.array([[-6.12], [-6.12], [-6.12], [-6.12]]),
        include_audit=True,
        for_training=True,
    )
    assert ev["constrained_clean_hits"].tolist() == [True, False, False, False]
    assert np.isfinite(ev["sigma_cal_estimate"][0, 0])
    adv = bus.compute_group_advantages(ev, group_size=4, device=torch.device("cpu"))
    # Monotonic hierarchy: Constrained Clean Hit > Hard-Constraint Violator > Exotic Zwitterion > Invalid
    assert float(adv[0]) > float(adv[1]) > float(adv[2]) > float(adv[3])

    # Verify LoomGRPOTrainer runs cleanly with `critic_bus=bus` (both 1-prop AND 2-prop generator!)
    gen = _make_small_v2_generator(num_props=1)
    trainer = LoomGRPOTrainer(
        generator=gen,
        critic_bus=bus,
        replay_molecules=[clean_mol, _make_3d_mol("CCO", seed=3)],
        replay_targets_eV=np.array([[-6.10], [-5.90]], dtype=np.float32),
        lr=5e-5,
        seed=99,
    )
    res = trainer.step(np.array([[-6.10]], dtype=np.float32), np.array([[True]]), group_size=2, step_index=0)
    assert np.isfinite(res["loss_total"])
    assert "constraints_met_rate_among_valid" in res

    gen_2prop = _make_small_v2_generator(num_props=2)
    trainer_2p = LoomGRPOTrainer(
        generator=gen_2prop,
        critic_bus=bus,
        replay_molecules=[clean_mol, _make_3d_mol("CCO", seed=3)],
        replay_targets_eV=np.array([[-6.10, -2.0], [-5.90, -2.1]], dtype=np.float32),
        lr=5e-5,
        seed=100,
    )
    res_2p = trainer_2p.step(
        np.array([[-6.10, 0.0]], dtype=np.float32), np.array([[True, False]]), group_size=2, step_index=0
    )
    assert np.isfinite(res_2p["loss_total"])


# ==============================================================================
# Phase 3 Tests: Generalized GPAtlas Gatekeepers & Grounded RSI Engine
# ==============================================================================


def test_generalized_gp_atlas_gatekeeper_and_rsi_closed_loop_modules(tmp_path: Path):
    rng = np.random.default_rng(123)

    # 1. Non-orbital 2D GPAtlas (`logP`, `tpsa`) must NOT apply `x >= y` unphysical mask!
    x_vals = np.linspace(1.0, 4.0, 4)
    y_vals = np.linspace(0.5, 3.5, 4)  # Note x >= y in half the plane, which is completely physical for (logP, scaled_tpsa)!
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")
    req_2d = np.repeat(np.stack([xx.ravel(), yy.ravel()], axis=-1), 10, axis=0)
    pred_2d = req_2d + rng.normal(0.0, 0.15, size=req_2d.shape)
    atlas_non_orb = HeteroscedasticGPAtlas2D.fit_from_proposals(
        req_2d, pred_2d, np.ones(len(req_2d), dtype=bool), axis_names=("logP", "tpsa_scaled"), units=("logP", "A2")
    )
    assert atlas_non_orb.unphysical_rule is None
    q2 = atlas_non_orb.query_grid([3.0], [1.5])
    assert q2["unphysical_mask"][0][0] is False
    assert q2["p_joint_hit_dft_calibrated"][0][0] > 0.10

    # Meanwhile canonical (`homo`, `lumo`) defaults to `x >= y` unphysical rule & Gatekeeper reroutes!
    atlas_orb = HeteroscedasticGPAtlas2D.fit_from_proposals(
        req_2d - 7.0, pred_2d - 7.0, np.ones(len(req_2d), dtype=bool), axis_names=("homo", "lumo"), units=("eV", "eV")
    )
    assert atlas_orb.unphysical_rule == "x>=y"
    gk2 = atlas_orb.recommend_target_2d((-3.0, -5.0))  # HOMO=-3.0 > LUMO=-5.0 -> unphysical!
    assert gk2["status"] == "rerouted_warning"
    assert gk2["reason"] == "unphysical_target"
    assert gk2["recommended_target"][0] < gk2["recommended_target"][1]

    # 2. 1D GPAtlas + RSIDiagnoser 4-zone classification & safe curriculum (0% weight on OUTER_BLINDSPOT!)
    # Construct 4 distinct regimes along [-7.2, -6.5, -5.8, -5.2]:
    #   -7.2 : PRIOR_CEILING (zero source density + huge bias +0.85 eV)
    #   -6.5 : OUTER_BLINDSPOT (valid molecules, but sigma_cal = 0.48 > 0.32!)
    #   -5.8 : INNER_UNDEREXPLORED (trusted sigma_cal = 0.15, but wide sigma_gen = 0.58 -> low hit rate)
    #   -5.2 : CONVERGED (trusted sigma_cal = 0.14, tight sigma_gen = 0.12, zero bias -> high hit rate)
    grid_pts = [-7.2, -6.5, -5.8, -5.2]
    req_1d = np.repeat(grid_pts, 40)
    preds_1d = np.concatenate(
        [
            rng.normal(-6.35, 0.25, 40),  # -7.2 -> pulled back by +0.85 eV!
            rng.normal(-6.45, 0.22, 40),  # -6.5 -> blindspot
            rng.normal(-5.52, 0.85, 40),  # -5.8 -> underexplored wide spread + moderate drift
            rng.normal(-5.20, 0.10, 40),  # -5.2 -> converged!
        ]
    )
    sig_cal_1d = np.concatenate(
        [
            np.full(40, 0.22),
            np.full(40, 0.50),  # high calibration uncertainty at -6.5!
            np.full(40, 0.15),
            np.full(40, 0.14),
        ]
    )
    src_vals = rng.normal(-5.5, 0.40, 400)  # zero support near -7.2!
    atlas_1d = HeteroscedasticGPAtlas1D.fit_from_proposals(
        req_1d,
        preds_1d,
        np.ones(len(req_1d), dtype=bool),
        sigma_cal_samples=sig_cal_1d,
        source_property_values_eV=src_vals,
        axis_name="homo",
        unit="eV",
    )
    gk_blind = atlas_1d.recommend_target(-6.5, tolerance=0.25, cal_sigma_cap=0.32)
    assert gk_blind["status"] == "rerouted_warning"
    gk_ok = atlas_1d.recommend_target(-5.2, tolerance=0.25, cal_sigma_cap=0.32)
    assert gk_ok["status"] == "trusted"

    diagnoser = RSIDiagnoser(RSIDiagnosticThresholds(cal_sigma_cap=0.32))
    diag = diagnoser.diagnose_1d(atlas_1d.query(grid_pts, tolerance_eV=0.25))
    assert diag["rsi_zones"] == ["PRIOR_CEILING", "OUTER_BLINDSPOT", "INNER_UNDEREXPLORED", "CONVERGED"]
    probs = diag["curriculum_sampling_probs"]
    # Zero online RL weight on PRIOR_CEILING and OUTER_BLINDSPOT!
    assert probs[0] == 0.0
    assert probs[1] == 0.0
    assert abs(probs[2] - 0.80) < 1e-6
    assert abs(probs[3] - 0.20) < 1e-6

    # 3. OuterRSIAcquisition active selection & SDF export
    cand_smiles = ["c1ccccc1", "c1ccncc1", "N#Cc1ccccc1", "Fc1ccccc1", "COc1ccccc1"]
    cand_mols = [_make_3d_mol(s, seed=i + 1) for i, s in enumerate(cand_smiles)]
    acq = OuterRSIAcquisition(max_fmax_eV_A=3.5)
    sel_mols, sel_recs = acq.select_acquisition_batch(
        cand_mols,
        requested_targets=np.array([[-6.5], [-6.5], [-6.5], [-5.2], [-6.5]]),
        base_preds=np.array([[-6.0], [-6.1], [-6.0], [-5.2], [-6.1]]),
        sigma_cal_vec=np.array([0.42, 0.45, 0.49, 0.12, 0.38]),
        blindspot_target_values=[-6.5],
        budget_k=3,
        min_sigma_cal=0.30,
    )
    assert len(sel_mols) == 3
    # Highest-uncertainty blindspot molecule (`0.49`) must be selected #1
    assert abs(sel_recs[0]["sigma_cal"] - 0.49) < 1e-6
    sdf_out = tmp_path / "acq_test.sdf"
    manifest = acq.export_sdf(sel_mols, sel_recs, sdf_out)
    assert sdf_out.exists()
    assert sdf_out.with_suffix(".manifest.json").exists()
    assert manifest["exported_count"] == 3
