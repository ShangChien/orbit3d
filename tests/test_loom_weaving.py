"""Comprehensive unit tests for `loom.weaving` (`Uni-Loom` 4-slot architecture)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from loom.generator.octree_backend import OctreeDataset
from loom.generator.octree_model import AtomFeatDict, build_default_uni3dar_args
from loom.vendor.uni3dar.models.uni3dar import DecoderInputFeat
from loom.vendor.uni3dar.models.uni3dar_sampler import Uni3DARSampler
from loom.weaving import (
    AuditOracleSlot,
    HeteroscedasticGPAtlas1D,
    HeteroscedasticGPAtlas2D,
    LoRALinear,
    LoomConditioner,
    LoomGRPOTrainer,
    LoomUni3DARGenerator,
    PotentialSlot,
    RewardOracleSlot,
    assert_not_audit_oracle,
    compute_disentangled_group_advantages,
    compute_loom_metrics,
    formal_charge_signature,
)


def _make_small_generator(num_props: int = 2, num_prefix_tokens: int = 4) -> LoomUni3DARGenerator:
    args = build_default_uni3dar_args(batch_size=4)
    args.emb_dim = 64
    args.num_head = 4
    args.layer = 3
    args.merge_level = 6
    args.random_rotation_prob = 0.0
    backbone = Uni3DARSampler(args, AtomFeatDict(args))
    target_cols = ["homo", "lumo"][:num_props]
    target_units = ["eV"] * num_props
    mean = [-6.0, -2.0][:num_props]
    scale = [0.8, 0.9][:num_props]
    return LoomUni3DARGenerator(
        backbone=backbone,
        target_cols=target_cols,
        target_units=target_units,
        mean=mean,
        scale=scale,
        num_rbf=16,
        num_freqs=8,
        num_prefix_tokens=num_prefix_tokens,
        lora_rank=8,
        lora_alpha=16.0,
    )


def _make_3d_mol(smiles: str, seed: int = 0) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_lora_zero_perturbation_and_gradient_routing():
    torch.manual_seed(0)
    base = torch.nn.Linear(32, 48, bias=True)
    x = torch.randn(5, 7, 32)
    ref = base(x).detach()

    wrapped = LoRALinear(base, rank=8, alpha=16.0)
    out = wrapped(x)
    assert torch.allclose(out, ref, atol=0.0)
    assert not wrapped.base.weight.requires_grad
    assert wrapped.lora_A.requires_grad and wrapped.lora_B.requires_grad

    loss = out.square().sum()
    loss.backward()
    assert wrapped.lora_B.grad is not None and wrapped.lora_B.grad.abs().sum() > 0


def test_loom_conditioner_zero_init_and_unclamped_modulation():
    cond = LoomConditioner(emb_dim=64, num_props=2, num_layers=3, num_rbf=16, num_freqs=8, num_prefix_tokens=4)
    vals = torch.tensor([[-1.5, 0.5], [1.8, -0.7]], dtype=torch.float32)
    mask = torch.tensor([[True, False], [True, True]])

    global_c, has_any = cond.encode_condition(vals, mask)
    assert global_c.shape == (2, 64)
    assert has_any.tolist() == [[True], [True]]

    h = torch.randn(2, 10, 64)
    # Exact identity at initialization because AdaLN-Zero weights & biases are 0
    assert torch.allclose(cond.modulate_layer(0, h, global_c.unsqueeze(1)), h, atol=0.0)
    assert torch.allclose(cond.modulate_out(h, global_c.unsqueeze(1)), h, atol=0.0)

    prefix_cond = cond.build_prefix_tokens(global_c, has_any)
    null_c, null_any = cond.encode_condition(vals, torch.zeros_like(mask))
    prefix_null = cond.build_prefix_tokens(null_c, null_any)
    assert prefix_cond is not None and prefix_cond.shape == (2, 4, 64)
    assert not torch.allclose(prefix_cond, prefix_null)


def test_parallel_sdpa_matches_stepwise_kv_cache_exactly():
    """Verify right-padded parallel SDPA matches step-by-step KV-cache decoding to <1e-5."""
    torch.manual_seed(7)
    gen = _make_small_generator(num_props=2, num_prefix_tokens=4)
    # Give non-zero weights to conditioner modulations + LoRA B so EVERY pathway is active!
    with torch.no_grad():
        gen.conditioner.root_gate.fill_(0.35)
        for mod in gen.conditioner.layer_mods:
            torch.nn.init.normal_(mod[1].weight, std=0.05)
            torch.nn.init.normal_(mod[1].bias, std=0.05)
        torch.nn.init.normal_(gen.conditioner.out_mod[1].weight, std=0.05)
        for name, param in gen.named_parameters():
            if name.endswith(".lora_B"):
                torch.nn.init.normal_(param, std=0.05)
    gen.eval()

    mols = [_make_3d_mol("CCO", seed=1), _make_3d_mol("c1ccccc1", seed=2)]
    ds = OctreeDataset(mols, gen.backbone, training=False, seed=42)
    packed = ds.collate([0, 1])

    targets_norm = torch.tensor([[-1.2, 0.4], [0.9, -0.8]], dtype=torch.float32)
    active_mask = torch.tensor([[True, False], [True, True]])

    decoder_input, _, _ = gen.backbone.get_feat_from_dataloader(packed)
    cu_lens = packed["decoder_input_len"].to(torch.int32)
    max_len = int(packed["decoder_input_max_len"].item())

    with torch.no_grad():
        parallel_out = gen.forward_sequence(decoder_input, cu_lens, max_len, targets_norm, active_mask)

        # Now run molecule 1 (benzene, odd/longer length) step-by-step via KV cache (`1` root + `2`-token steps)
        for b_idx in (0, 1):
            s = int(cu_lens[b_idx].item())
            e = int(cu_lens[b_idx + 1].item())
            seq_len = e - s
            t_b = targets_norm[b_idx : b_idx + 1]
            m_b = active_mask[b_idx : b_idx + 1]
            global_c, has_any = gen.conditioner.encode_condition(t_b, m_b)
            prefix_b = gen.conditioner.build_prefix_tokens(global_c, has_any)
            root_delta = gen.conditioner.forward_root_prefix(global_c)
            kv_cache = [[{"step": 1, "mode": "concat"} for _ in range(gen.args.layer)]]

            step_outputs = []
            pos = 0
            while pos < seq_len:
                chunk_len = min(1 if pos == 0 else 2, seq_len - pos)
                sl = slice(s + pos, s + pos + chunk_len)
                sub_in = DecoderInputFeat(
                    decoder_input.type[sl].unsqueeze(0),
                    decoder_input.xyz[sl].unsqueeze(0),
                    decoder_input.level[sl].unsqueeze(0),
                    decoder_input.phy_pos[sl].unsqueeze(0).float(),
                    decoder_input.space_index[sl].unsqueeze(0),
                    decoder_input.tree_index[sl].unsqueeze(0),
                    decoder_input.remaining_atoms[sl].unsqueeze(0),
                    decoder_input.remaining_tokens[sl].unsqueeze(0),
                    decoder_input.count[sl].unsqueeze(0),
                )
                emb = gen.backbone.input_embedding(sub_in)
                grid = sub_in.phy_pos / gen.args.xyz_resolution
                if pos == 0:
                    emb = emb + root_delta.unsqueeze(1)
                    emb = torch.cat([prefix_b, emb], dim=1)
                    grid = torch.cat([gen._prefix_grid_positions(1, emb.device), grid], dim=1)

                gen.backbone.rope.calc_and_cache(grid)
                for l_i, layer in enumerate(gen.backbone.decoder.layers):
                    emb = gen._run_layer_kv_step(layer, emb, kv_cache[0][l_i])
                    emb = gen.conditioner.modulate_layer(l_i, emb, global_c.unsqueeze(1))
                emb = gen.backbone.norm(emb)
                emb = gen.conditioner.modulate_out(emb, global_c.unsqueeze(1))
                if pos == 0:
                    step_outputs.append(emb[0, -1:, :])
                else:
                    step_outputs.append(emb[0])
                pos += chunk_len

            kv_out = torch.cat(step_outputs, dim=0)
            max_diff = float((kv_out - parallel_out[s:e]).abs().max().item())
            assert max_diff < 1e-5, f"KV cache vs parallel SDPA mismatch on mol {b_idx}: {max_diff}"


def test_sample_sft_checkpoint_and_distributions(tmp_path: Path):
    torch.manual_seed(11)
    gen = _make_small_generator(num_props=2, num_prefix_tokens=4)
    mols = [_make_3d_mol("CC=O", seed=1), _make_3d_mol("c1ccncc1", seed=2), _make_3d_mol("CC(=O)O", seed=3)]
    ds = OctreeDataset(mols, gen.backbone, training=True, seed=0)
    packed = ds.collate([0, 1, 2])

    targets_eV = torch.tensor([[-7.0, -1.0], [-5.5, -2.5], [-6.2, -1.8]])
    targets_norm = gen.normalize_targets(targets_eV)
    active_mask = torch.ones((3, 2), dtype=torch.bool)

    gen.train()
    loss, info = gen.compute_sft_loss(
        packed, targets_norm, active_mask, cfg_dropout_prob=0.0, contrast_weight=0.5, contrast_min_gap_eV=0.25
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert "inbatch_preference_rate" in info

    pref = gen.evaluate_teacher_forcing_preference(packed, targets_norm, active_mask, min_gap_eV=0.25)
    assert pref["eligible"] >= 2

    trajs = gen.sample(
        targets_norm[:2],
        active_mask[:2],
        cfg_scale=1.75,
        xyz_cfg_scale=1.0,
        seed=123,
        use_bf16=False,
        electron_parity=0,
    )
    assert len(trajs) == 2
    packets = gen.trajectory_distributions(
        trajs, targets_norm[:2], active_mask[:2], cfg_scale=1.75, xyz_cfg_scale=1.0, electron_parity=0
    )
    assert len(packets) > 0

    ckpt_path = tmp_path / "loom_gen.pt"
    gen.save_checkpoint(ckpt_path, extra_metadata={"stage": "unit_test"})
    loaded, payload = LoomUni3DARGenerator.load_checkpoint(ckpt_path, device="cpu")
    assert payload["metadata"]["stage"] == "unit_test"
    for p1, p2 in zip(gen.parameters(), loaded.parameters()):
        assert torch.allclose(p1, p2)


def test_audit_oracle_guard_blocks_training():
    dummy_audit = object.__new__(AuditOracleSlot)
    with pytest.raises(RuntimeError, match="AUDIT VIOLATION"):
        assert_not_audit_oracle(dummy_audit)


def test_disentangled_advantages_metrics_and_gp_atlas():
    # Group of 8: [0] clean hit, [1] clean worst miss, [2] exotic zwitterion, [3] invalid, [4..7] clean mid/good
    valid_scores = torch.tensor([1.0, 1.0, -0.35, -1.0, 1.0, 1.0, -1.0, 1.0])
    valid_mask = torch.tensor([True, True, True, False, True, True, False, True])
    phys_scores = torch.tensor([0.9, 0.4, 0.6, 0.0, 0.85, 0.35, 0.0, 0.7])
    eval_mask = torch.tensor([True, True, False, False, True, True, False, True])
    prop_scores = torch.tensor([0.95, 0.12, 0.0, 0.0, 0.82, 0.45, 0.0, 0.68])

    adv, parts = compute_disentangled_group_advantages(
        valid_scores, valid_mask, phys_scores, eval_mask, prop_scores
    )
    assert adv.shape == (8,)
    assert torch.isfinite(adv).all()
    # Strict monotonic ordering: Clean Hit [0] > Clean Worst Miss [1] > Exotic Zwitterion [2] > Invalid [3]
    assert float(adv[0]) > float(adv[1]) > float(adv[2]) > float(adv[3])

    zwitterion = Chem.MolFromSmiles("C[N+](C)(C)CC(=O)[O-]")
    assert formal_charge_signature(zwitterion) == (("N", 1), ("O", -1))

    rng = np.random.default_rng(42)
    grid_t = np.repeat(np.linspace(-7.5, -4.5, 7), 32)
    rew_y = 0.96 * grid_t - 0.22 + rng.normal(0.0, 0.22, size=len(grid_t))
    aud_y = rew_y + rng.normal(0.02, 0.16, size=len(grid_t))
    legal = rng.random(len(grid_t)) < 0.88
    ev = legal.copy()
    fmax = np.abs(rng.normal(0.45, 0.20, size=len(grid_t)))

    metrics = compute_loom_metrics(
        grid_t, rew_y, legal, ev, audit_predictions_eV=aud_y, audit_evaluated_mask=ev, f_max_eV_A=fmax
    )
    assert 0.85 < metrics["calibration_slope"] < 1.08
    assert metrics["within_target_std_eV"] < 0.35
    assert metrics["audit"] is not None

    atlas_1d = HeteroscedasticGPAtlas1D.fit_from_proposals(
        grid_t, rew_y, ev, audit_or_dft_preds_eV=aud_y, source_property_values_eV=rng.normal(-6.0, 0.8, size=500)
    )
    q1 = atlas_1d.query(np.linspace(-8.0, -4.0, 25))
    assert len(q1["p_hit_reward"]) == 25
    assert q1["trusted_reachable_interval_eV"] is not None

    # 2D HOMO-LUMO GP Atlas test (including spatially-varying cross-foundation audit RMSE)
    h_vals = np.linspace(-7.0, -5.0, 4)
    l_vals = np.linspace(-3.0, -1.0, 4)
    hh, ll = np.meshgrid(h_vals, l_vals, indexing="ij")
    cells_2d = np.repeat(np.stack([hh.ravel(), ll.ravel()], axis=-1), 16, axis=0)
    preds_2d = cells_2d + rng.normal(0.0, 0.20, size=cells_2d.shape)
    ood_scale = 0.14 + 0.18 * (np.abs(cells_2d[:, 0] + 6.0) + np.abs(cells_2d[:, 1] + 2.0))[:, None]
    aud_2d = preds_2d + rng.normal(0.0, 1.0, size=cells_2d.shape) * ood_scale
    ev_2d = np.ones(len(cells_2d), dtype=bool)
    src_2d = np.column_stack([rng.normal(-6.0, 0.6, 300), rng.normal(-2.0, 0.6, 300)])

    atlas_2d = HeteroscedasticGPAtlas2D.fit_from_proposals(
        cells_2d, preds_2d, ev_2d, audit_or_dft_preds_2d_eV=aud_2d, source_2d_eV=src_2d
    )
    q2 = atlas_2d.query_grid(np.linspace(-7.5, -4.5, 10), np.linspace(-3.5, -0.5, 10))
    assert np.asarray(q2["p_joint_hit_reward"]).shape == (10, 10)
    assert q2["has_heteroscedastic_cal_gp"] is True
    cal_grid = np.asarray(q2["sigma_cal_2d_eV"])
    assert cal_grid.shape == (10, 10)
    # Corner OOD disagreement should exceed central well-supported disagreement
    assert float(cal_grid[0, 0]) > float(cal_grid[5, 5])


def test_v11_xyz_heads_and_geometric_guard_logprob_match(tmp_path: Path):
    """Verify v1.1 fine-XYZ LoRA upgrade, roundtrip checkpointing, and exact log-prob match under guard+tree_decay."""
    torch.manual_seed(19)
    gen = _make_small_generator(num_props=2, num_prefix_tokens=2)
    n_v10 = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    trainable_v11 = gen.enable_v11_xyz_heads()
    n_v11 = sum(p.numel() for p in trainable_v11)
    assert n_v11 > n_v10
    assert gen.loom_config["lora_include_xyz_heads"] is True
    assert gen.loom_config["train_xyz_embeddings"] is True

    ckpt = tmp_path / "loom_v11.pt"
    gen.save_checkpoint(ckpt)
    loaded, _ = LoomUni3DARGenerator.load_checkpoint(ckpt, device="cpu")
    assert loaded.loom_config["lora_include_xyz_heads"] is True

    t_norm = torch.tensor([[-0.8, 0.5], [0.7, -0.6]], dtype=torch.float32)
    m_act = torch.ones((2, 2), dtype=torch.bool)
    loaded.eval()
    trajs = loaded.sample(
        t_norm,
        m_act,
        cfg_scale=1.8,
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.30,
        geometric_guard=True,
        temperatures=(1.0, 1.0, 1.0),
        seed=2026,
        use_bf16=False,
        electron_parity=0,
    )
    packets = loaded.trajectory_distributions(
        trajs,
        t_norm,
        m_act,
        cfg_scale=1.8,
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.30,
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
        tol = max(1e-4, 2e-6 * abs(sample_sum))
        assert abs(dist_sum - sample_sum) < tol, f"Mismatch on mol {b_i}: {dist_sum} vs {sample_sum}"

    from loom.generator.octree_backend import OctreeDataset

    ds_mol = OctreeDataset([_make_3d_mol("CCO", seed=1), _make_3d_mol("CC=O", seed=2)], loaded.backbone, training=False, seed=0)
    packed_real = ds_mol.collate([0, 1])
    orig_pack = loaded.pack_sampled_trajectories
    loaded.pack_sampled_trajectories = lambda _t, dev: {
        k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in packed_real.items()
    }
    try:
        full_real = loaded.trajectory_distributions(trajs, t_norm, m_act, topology_only_policy=False, electron_parity=0)
        topo_real = loaded.trajectory_distributions(trajs, t_norm, m_act, topology_only_policy=True, electron_parity=0)
        assert len(full_real) == 5  # tree + element + x + y + z
        assert len(topo_real) == 2  # tree + element only
    finally:
        loaded.pack_sampled_trajectories = orig_pack


def test_grpo_trainer_v11_step_with_relax_distill(monkeypatch):
    """Verify LoomGRPOTrainer.step end-to-end with v1.1 guard, sharp concentration, replay, and relax distillation."""
    torch.manual_seed(31)
    gen = _make_small_generator(num_props=2, num_prefix_tokens=2)
    gen.enable_v11_xyz_heads()

    clean_mols = [_make_3d_mol("CCO", seed=1), _make_3d_mol("CC=O", seed=2), _make_3d_mol("c1ccccc1", seed=3)]
    replay_y = np.array([[-6.2, -1.5], [-5.8, -2.1], [-6.5, -1.2]], dtype=np.float32)

    class _DummyReward(RewardOracleSlot):
        def __init__(self):
            self.target_cols = ["homo", "lumo"]

        def check_support(self, mol):
            return None

        def evaluate_candidates(self, mols, batch_size=32):
            n = len(mols)
            preds = np.column_stack([np.full(n, -6.1), np.full(n, -1.8)])
            ok = np.array([m is not None for m in mols], dtype=bool)
            return preds, np.zeros_like(preds), ok, [None] * n

    class _DummyPotential(PotentialSlot):
        def __init__(self):
            pass

        def evaluate_molecules(self, mols, chunk_size=32):
            n = len(mols)
            return {"f_rms_eV_A": np.full(n, 0.25), "f_max_eV_A": np.full(n, 0.70)}

        def relax_molecules(self, mols, n_steps=30, chunk_size=16):
            n = len(mols)
            return {
                "relaxed_mols": [Chem.Mol(m) for m in mols],
                "graph_preserved": np.ones(n, dtype=bool),
                "init_energy_eV": np.zeros(n),
                "final_energy_eV": np.full(n, -0.2),
                "final_f_max_eV_A": np.full(n, 0.12),
            }

    # Ensure trajectory_to_instance yields valid clean molecules on half the rollouts so relax_distill triggers
    import loom.weaving.grpo as grpo_mod
    from types import SimpleNamespace

    call_counter = {"i": 0}

    def _fake_traj_to_inst(tr, element_table):
        idx = call_counter["i"]
        call_counter["i"] += 1
        if idx % 2 == 0:
            m = Chem.Mol(clean_mols[idx % len(clean_mols)])
            return SimpleNamespace(graph=m, smiles=Chem.MolToSmiles(Chem.RemoveHs(m)))
        raise ValueError("synthetic invalid graph")

    monkeypatch.setattr(grpo_mod, "trajectory_to_instance", _fake_traj_to_inst)

    trainer = LoomGRPOTrainer(
        generator=gen,
        reward_oracle=_DummyReward(),
        potential_slot=_DummyPotential(),
        seen_charge_signatures=frozenset({formal_charge_signature(m) for m in clean_mols}),
        replay_molecules=clean_mols,
        replay_targets_eV=replay_y,
        lr=1e-4,
        kl_beta=0.025,
        replay_weight=0.20,
        replay_batch_size=2,
        sharp_concentration=True,
        cfg_scale=1.6,
        cfg_tree_decay=0.30,
        geometric_guard=True,
        relax_distill_weight=0.25,
        relax_distill_steps=10,
        relax_distill_max_mols=3,
        seed=7,
    )
    prompts = np.array([[-6.15, -1.85], [-6.40, -1.30]], dtype=np.float32)
    pmask = np.array([[True, False], [True, True]], dtype=bool)
    res = trainer.step(prompts, pmask, group_size=2, step_index=0)
    assert np.isfinite(res["loss_total"])
    assert np.isfinite(res["kl"])
    assert res["relax_distill_admitted"] >= 1
    assert res["loss_relax_distill"] > 0.0


def test_potential_slot_unsupported_element_graceful_nan():
    """Ensure molecules with out-of-type-map heavy elements (e.g. Te) yield NaN/False instead of crashing batches."""
    from types import SimpleNamespace

    slot = PotentialSlot.__new__(PotentialSlot)
    slot.backend_name = "dpa4"
    slot.atom_types = ["H", "C", "N", "O"]
    slot.device_str = "cpu"
    slot.engine = SimpleNamespace(
        compute_energy_and_forces=lambda coords, atypes, b_idx, **kw: (
            torch.full((int(b_idx.max().item()) + 1,) if b_idx.numel() else (0,), -1.5),
            torch.full_like(coords, 0.1),
        ),
        relax_conformers=lambda coords, atypes, b_idx, **kw: (
            coords,
            torch.full((int(b_idx.max().item()) + 1,), -1.0),
            torch.full((int(b_idx.max().item()) + 1,), -1.2),
        ),
    )

    m_ok = _make_3d_mol("CCO", seed=1)
    m_te = _make_3d_mol("C[Te]C", seed=2)
    ev = slot.evaluate_molecules([m_ok, m_te], chunk_size=4)
    assert np.isfinite(ev["f_max_eV_A"][0])
    assert np.isnan(ev["f_max_eV_A"][1])

    rel = slot.relax_molecules([m_ok, m_te], n_steps=3, chunk_size=4)
    assert bool(rel["graph_preserved"][0]) is True
    assert bool(rel["graph_preserved"][1]) is False
    assert np.isnan(rel["final_f_max_eV_A"][1])
