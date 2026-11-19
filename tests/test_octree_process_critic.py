"""Unit tests for `OctreeProcessCritic` and `LoomUni3DARGenerator.sample_with_process_critic`:
1. Masked KV-cache wait steps (`kv_valid_mask`) produce bit-exact / `<1e-5` identical outputs to unpadded KV history.
2. `sample(..., return_coarse_summary=True)` + `fit_octree_process_critic` + `sample_with_process_critic(m_init=6, m_keep=2)`
   executes cleanly under CFG, `Size-Anchored CFG`, and `geometric_guard`, returning exact grouped prompt outputs.
"""

from __future__ import annotations

import numpy as np
import torch

from loom.weaving import (
    LoomUni3DARGenerator,
    NUM_COARSE_GEOM_FEATS,
    NUM_ELEM_CLOUD_FEATS,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    TwoStageOctreeProcessCritic,
    compute_element_cloud_features,
    fit_octree_element_process_critic,
    fit_octree_process_critic,
)


def _make_tiny_generator() -> LoomUni3DARGenerator:
    backbone = LoomUni3DARGenerator._build_backbone(
        merge_level=3,
        allowed_elements=("H", "C", "N", "O", "F"),
        batch_size=4,
        emb_dim=64,
        num_head=4,
        layer=2,
        ffn_multiple=2,
    )
    return LoomUni3DARGenerator(
        backbone=backbone,
        target_cols=["homo", "n_atoms"],
        target_units=["eV", "atoms"],
        mean=[-6.5, 18.0],
        scale=[0.8, 4.0],
        num_rbf=16,
        num_freqs=8,
        num_prefix_tokens=2,
        lora_rank=0,
        use_valence_head=True,
    )


def test_kv_valid_mask_ignores_paused_wait_tokens_exactly() -> None:
    torch.manual_seed(11)
    gen = _make_tiny_generator()
    gen.eval()
    layer = gen.backbone.decoder.layers[0]

    # Reference sequence: step0 (3 tokens), step1 (2 tokens), step2 (2 tokens) -> total 7 valid tokens
    x0 = torch.randn(2, 3, 64)
    x1 = torch.randn(2, 2, 64)
    x2 = torch.randn(2, 2, 64)
    g0 = torch.randn(2, 3, 3) * 10.0
    g1 = torch.randn(2, 2, 3) * 10.0
    g2 = torch.randn(2, 2, 3) * 10.0

    kv_ref = {"step": 1, "mode": "concat"}
    gen.backbone.rope.calc_and_cache(g0)
    _ = gen._run_layer_kv_step(layer, x0, kv_ref)
    gen.backbone.rope.calc_and_cache(g1)
    _ = gen._run_layer_kv_step(layer, x1, kv_ref)
    gen.backbone.rope.calc_and_cache(g2)
    out_ref = gen._run_layer_kv_step(layer, x2, kv_ref)

    # Paused sequence on row 0: insert TWO dummy wait steps (4 tokens) between step1 and step2 with `kv_valid_mask=False` on row 0
    kv_paused = {"step": 1, "mode": "concat"}
    valid_mask = torch.ones((2, 3), dtype=torch.bool)
    gen.backbone.rope.calc_and_cache(g0)
    _ = gen._run_layer_kv_step(layer, x0, kv_paused, kv_valid_mask=valid_mask)

    valid_mask = torch.cat([valid_mask, torch.ones((2, 2), dtype=torch.bool)], dim=1)
    gen.backbone.rope.calc_and_cache(g1)
    _ = gen._run_layer_kv_step(layer, x1, kv_paused, kv_valid_mask=valid_mask)

    # Insert dummy wait step where row 0 is paused (`False`) and row 1 is active (`True`)
    x_dummy = torch.randn(2, 2, 64) * 5.0
    g_dummy = torch.randn(2, 2, 3) * 20.0
    step_valid = torch.tensor([[False, False], [True, True]], dtype=torch.bool)
    valid_mask = torch.cat([valid_mask, step_valid], dim=1)
    gen.backbone.rope.calc_and_cache(g_dummy)
    _ = gen._run_layer_kv_step(layer, x_dummy, kv_paused, kv_valid_mask=valid_mask)

    # Now wake row 0 up for step2 (`True, True`) -> row 0 output MUST match `out_ref[0]` exactly!
    valid_mask = torch.cat([valid_mask, torch.ones((2, 2), dtype=torch.bool)], dim=1)
    gen.backbone.rope.calc_and_cache(g2)
    out_paused = gen._run_layer_kv_step(layer, x2, kv_paused, kv_valid_mask=valid_mask)

    assert torch.allclose(out_ref[0], out_paused[0], atol=1e-5)


def test_octree_process_critic_fit_and_branch_prune_sampling(tmp_path) -> None:
    torch.manual_seed(2026)
    gen = _make_tiny_generator()
    gen.eval()

    prompts_raw = np.array(
        [[-6.8, 16.0], [-6.1, 19.0], [-7.2, 15.0], [-5.9, 21.0]], dtype=np.float32
    )
    t_norm = gen.normalize_targets(prompts_raw)
    cond_mask = torch.ones((4, 2), dtype=torch.bool)
    uncond_mask = torch.tensor([[False, True]] * 4, dtype=torch.bool)

    # 1. Collect rollout coarse summaries via `sample(..., return_coarse_summary=True)`
    trajs = gen.sample(
        t_norm.repeat(3, 1),
        cond_mask.repeat(3, 1),
        cfg_scale=1.75,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        seed=123,
        use_bf16=False,
        max_decode_steps=48,
        uncond_active_mask=uncond_mask.repeat(3, 1),
        return_coarse_summary=True,
    )
    assert len(trajs) == 12
    h_cat = torch.stack([tr["coarse_h_concat"] for tr in trajs], dim=0)
    geom = torch.stack([tr["coarse_geom_feats"] for tr in trajs], dim=0)
    assert h_cat.shape == (12, 4 * 64)
    assert geom.shape == (12, NUM_COARSE_GEOM_FEATS)

    # 2. Fit `OctreeProcessCritic` and test checkpoint save/load
    critic = OctreeProcessCritic(emb_dim=64, num_targets=2, hidden_dim=128)
    critic.set_normalization([-6.5, 18.0], [0.8, 4.0])

    y_surr_norm = t_norm.repeat(3, 1) + 0.15 * torch.randn(12, 2)
    y_ok = torch.tensor([i % 3 != 0 for i in range(12)], dtype=torch.bool)
    is_leg = y_ok.clone()
    is_con = torch.tensor([i % 2 == 0 for i in range(12)], dtype=torch.bool)

    stats = fit_octree_process_critic(
        critic,
        h_concat=h_cat,
        geom_feats=geom,
        targets_norm=t_norm.repeat(3, 1),
        active_mask=cond_mask.repeat(3, 1),
        y_surrogate_norm=y_surr_norm,
        y_valid_mask=y_ok,
        is_legal=is_leg,
        is_constr_met=is_con,
        epochs=6,
        batch_size=6,
    )
    assert np.isfinite(stats.train_loss)

    ckpt_file = tmp_path / "proc_critic.pt"
    critic.save_checkpoint(ckpt_file, extra={"tag": "unit_test"})
    loaded_critic, meta = OctreeProcessCritic.load_checkpoint(ckpt_file)
    assert meta["tag"] == "unit_test"

    # 3. Run mid-tree KV-cache branch-and-prune (`m_init=5 -> m_keep=2`) on 3 prompts
    obj_col_mask = torch.tensor([True, False], dtype=torch.bool)
    grouped = gen.sample_with_process_critic(
        t_norm[:3],
        cond_mask[:3],
        loaded_critic,
        m_init=5,
        m_keep=2,
        objective_col_mask=obj_col_mask,
        legal_weight=1.0,
        constr_weight=0.5,
        cfg_scale=1.75,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        seed=999,
        use_bf16=False,
        max_decode_steps=48,
        uncond_active_mask=uncond_mask[:3],
    )
    assert len(grouped) == 3
    for group in grouped:
        assert len(group) == 2
        assert group[0]["process_utility"] >= group[1]["process_utility"] - 1e-6
        for tr in group:
            assert "atom_symbols" in tr and "positions_np" in tr
            assert int((tr["decoder_feat"][:, 4] == -99).sum().item()) == 0


def test_zero_replay_kv_continuation_logprob_parity_and_leaf_budget() -> None:
    torch.manual_seed(77)
    gen = _make_tiny_generator()
    gen.eval()
    critic = OctreeProcessCritic(emb_dim=64, num_targets=2, hidden_dim=128)
    critic.set_normalization([-6.5, 18.0], [0.8, 4.0])
    critic.eval()

    prompts_raw = np.array(
        [[-6.8, 14.0], [-6.1, 22.0], [-7.2, 11.0], [-5.9, 26.0], [-6.4, 18.0]], dtype=np.float32
    )
    t_norm = gen.normalize_targets(prompts_raw)
    cond_mask = torch.ones((5, 2), dtype=torch.bool)
    uncond_mask = torch.tensor([[False, True]] * 5, dtype=torch.bool)
    budgets = torch.tensor([14.0, 22.0, 11.0, 26.0, 18.0])

    grouped = gen.sample_with_process_critic(
        t_norm,
        cond_mask,
        critic,
        m_init=5,
        m_keep=2,
        cfg_scale=1.8,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        electron_parity=0,
        uncond_active_mask=uncond_mask,
        seed=42,
        use_bf16=False,
        max_decode_steps=256,
    )
    flat_trajs = [tr for g in grouped for tr in g]
    t_rep = t_norm.repeat_interleave(2, dim=0)
    m_rep = cond_mask.repeat_interleave(2, dim=0)
    u_rep = uncond_mask.repeat_interleave(2, dim=0)
    packets = gen.trajectory_distributions(
        flat_trajs,
        t_rep,
        m_rep,
        temperatures=(1.0, 1.0, 1.0),
        cfg_scale=1.8,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        electron_parity=0,
        topology_only_policy=True,
        uncond_active_mask=u_rep,
    )
    tot_tf = torch.zeros(len(flat_trajs))
    for logp, labels, owner in packets:
        chosen = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        assert torch.isfinite(chosen).all()
        tot_tf.index_add_(0, owner, chosen)
    samp_scores = torch.tensor([float(tr["scores"].sum().item()) for tr in flat_trajs])
    assert float((tot_tf - samp_scores).abs().max().item()) < 5e-4

    # Verify leaf_atom_budget + length-sorting preserves prompt order
    grouped_b = gen.sample_with_process_critic(
        t_norm.repeat(2, 1),
        cond_mask.repeat(2, 1),
        critic,
        m_init=4,
        m_keep=2,
        cfg_scale=1.6,
        geometric_guard=True,
        electron_parity=0,
        uncond_active_mask=uncond_mask.repeat(2, 1),
        seed=99,
        use_bf16=False,
        max_decode_steps=160,
        leaf_atom_budget=budgets.repeat(2),
        leaf_budget_tol=2.0,
        sort_by_budget=True,
    )
    assert len(grouped_b) == 10 and all(len(g) == 2 for g in grouped_b)


def test_element_process_critic_so3_invariance() -> None:
    torch.manual_seed(314)
    elem_ids = torch.tensor([[5, 5, 6, 7, 0, 0, 0, 0], [5, 5, 5, 8, 0, 0, 0, 0]], dtype=torch.long)
    leaf_pos = torch.randn(2, 8, 3) * 1.6
    leaf_valid = torch.tensor(
        [[True, True, True, True, True, True, False, False], [True, True, True, True, True, True, True, False]],
        dtype=torch.bool,
    )
    coarse_geom = torch.randn(2, NUM_COARSE_GEOM_FEATS)
    h_s2 = torch.randn(2, 6 * 64)
    t_norm = torch.randn(2, 2)
    a_mask = torch.ones((2, 2), dtype=torch.bool)

    # Arbitrary proper 3D rotation Q (det = +1) + 3D translation shift
    q_mat, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q_mat) < 0:
        q_mat[:, 0] = -q_mat[:, 0]
    shift = torch.tensor([[[12.5, -7.3, 4.1]]])
    leaf_pos_rt = leaf_pos @ q_mat.T + shift

    cloud_1 = compute_element_cloud_features(elem_ids, leaf_pos, leaf_valid, coarse_geom)
    cloud_2 = compute_element_cloud_features(elem_ids, leaf_pos_rt, leaf_valid, coarse_geom)
    assert cloud_1.shape == (2, NUM_ELEM_CLOUD_FEATS)
    assert torch.allclose(cloud_1, cloud_2, atol=1e-5)

    s2_critic = OctreeElementProcessCritic(emb_dim=64, num_targets=2, d_node=64, hidden_dim=128)
    s2_critic.eval()
    u1 = s2_critic.score_utilities(h_s2, elem_ids, leaf_pos, leaf_valid, cloud_1, t_norm, a_mask)
    u2 = s2_critic.score_utilities(h_s2, elem_ids, leaf_pos_rt, leaf_valid, cloud_2, t_norm, a_mask)
    assert torch.allclose(u1, u2, atol=1e-5)


def test_two_stage_process_critic_fit_save_and_logprob_parity(tmp_path) -> None:
    torch.manual_seed(2027)
    gen = _make_tiny_generator()
    gen.eval()

    prompts_raw = np.array(
        [[-6.8, 15.0], [-6.1, 20.0], [-7.2, 13.0], [-5.9, 23.0], [-6.4, 17.0]], dtype=np.float32
    )
    t_norm = gen.normalize_targets(prompts_raw)
    cond_mask = torch.ones((5, 2), dtype=torch.bool)
    uncond_mask = torch.tensor([[False, True]] * 5, dtype=torch.bool)
    budgets = torch.tensor([15.0, 20.0, 13.0, 23.0, 17.0])

    roll = gen.sample_rollouts_with_two_stage_features(
        t_norm.repeat(3, 1),
        cond_mask.repeat(3, 1),
        cfg_scale=1.75,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        electron_parity=0,
        seed=555,
        use_bf16=False,
        max_decode_steps=96,
        uncond_active_mask=uncond_mask.repeat(3, 1),
        leaf_atom_budget=budgets.repeat(3),
    )
    assert len(roll["trajectories"]) == 15
    assert roll["s1_h_concat"].shape == (15, 4 * 64)
    assert roll["s2_h_stage2"].shape == (15, 6 * 64)
    assert roll["s2_cloud_feats"].shape == (15, NUM_ELEM_CLOUD_FEATS)

    s1 = OctreeProcessCritic(emb_dim=64, num_targets=2, hidden_dim=128)
    s1.set_normalization([-6.5, 18.0], [0.8, 4.0])
    s2 = OctreeElementProcessCritic(emb_dim=64, num_targets=2, d_node=64, hidden_dim=128)
    s2.set_normalization([-6.5, 18.0], [0.8, 4.0])

    t_rep_tr = t_norm.repeat(3, 1)
    m_rep_tr = cond_mask.repeat(3, 1)
    y_surr_norm = t_rep_tr + 0.1 * torch.randn(15, 2)
    y_ok = torch.tensor([i % 4 != 0 for i in range(15)], dtype=torch.bool)
    is_leg = y_ok.clone()
    is_con = torch.tensor([i % 2 == 0 for i in range(15)], dtype=torch.bool)

    st1 = fit_octree_process_critic(
        s1,
        h_concat=roll["s1_h_concat"],
        geom_feats=roll["s1_geom_feats"],
        targets_norm=t_rep_tr,
        active_mask=m_rep_tr,
        y_surrogate_norm=y_surr_norm,
        y_valid_mask=y_ok,
        is_legal=is_leg,
        is_constr_met=is_con,
        epochs=5,
        batch_size=8,
    )
    st2 = fit_octree_element_process_critic(
        s2,
        h_stage2=roll["s2_h_stage2"],
        elem_ids=roll["s2_elem_ids"],
        leaf_pos=roll["s2_leaf_pos"],
        leaf_valid=roll["s2_leaf_valid"],
        cloud_feats=roll["s2_cloud_feats"],
        targets_norm=t_rep_tr,
        active_mask=m_rep_tr,
        y_surrogate_norm=y_surr_norm,
        y_valid_mask=y_ok,
        is_legal=is_leg,
        is_constr_met=is_con,
        epochs=5,
        batch_size=8,
    )
    assert np.isfinite(st1.train_loss) and np.isfinite(st2.train_loss)

    two_stage = TwoStageOctreeProcessCritic(s1, s2)
    ckpt_p = tmp_path / "two_stage_critic.pt"
    two_stage.save_checkpoint(ckpt_p, extra={"step": "v4_step1"})
    loaded_ts, meta = TwoStageOctreeProcessCritic.load_checkpoint(ckpt_p)
    assert meta["step"] == "v4_step1"

    # 1. Verify vectorized GPU pause-and-gather funnel `6 -> 4 -> 2` & exact teacher-forcing logprob parity
    grouped = gen.sample_with_two_stage_process_critic(
        t_norm,
        cond_mask,
        loaded_ts,
        m_stage1=6,
        m_stage2=4,
        m_keep=2,
        objective_col_mask=torch.tensor([True, True], dtype=torch.bool),
        coarse_chunk_size=12,  # Forces multi-chunk execution (`2` prompts/chunk)
        cfg_scale=1.8,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        electron_parity=0,
        temperatures=(1.0, 1.0, 1.0),
        seed=808,
        use_bf16=False,
        max_decode_steps=192,
        uncond_active_mask=uncond_mask,
    )
    assert len(grouped) == 5
    for g in grouped:
        assert len(g) == 2
        assert g[0]["process_utility"] >= g[1]["process_utility"] - 1e-6
        for tr in g:
            assert "atom_symbols" in tr and "positions_np" in tr
            assert int((tr["decoder_feat"][:, 4] == -99).sum().item()) == 0

    flat_trajs = [tr for g in grouped for tr in g]
    packets = gen.trajectory_distributions(
        flat_trajs,
        t_norm.repeat_interleave(2, dim=0),
        cond_mask.repeat_interleave(2, dim=0),
        temperatures=(1.0, 1.0, 1.0),
        cfg_scale=1.8,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        electron_parity=0,
        topology_only_policy=True,
        uncond_active_mask=uncond_mask.repeat_interleave(2, dim=0),
    )
    tot_tf = torch.zeros(len(flat_trajs))
    for logp, labels, owner in packets:
        chosen = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        assert torch.isfinite(chosen).all()
        tot_tf.index_add_(0, owner, chosen)
    samp_scores = torch.tensor([float(tr["scores"].sum().item()) for tr in flat_trajs])
    assert float((tot_tf - samp_scores).abs().max().item()) < 5e-4

    # 2. Verify `leaf_atom_budget` + `sort_by_budget=True` preserves exact prompt order & funnel shape
    grouped_b = gen.sample_with_two_stage_process_critic(
        t_norm.repeat(2, 1),
        cond_mask.repeat(2, 1),
        loaded_ts,
        m_stage1=6,
        m_stage2=4,
        m_keep=2,
        coarse_chunk_size=18,
        cfg_scale=1.8,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        electron_parity=0,
        seed=909,
        use_bf16=False,
        max_decode_steps=160,
        uncond_active_mask=uncond_mask.repeat(2, 1),
        leaf_atom_budget=budgets.repeat(2),
        leaf_budget_tol=2.0,
        sort_by_budget=True,
    )
    assert len(grouped_b) == 10 and all(len(g) == 2 for g in grouped_b)


def test_v4_step2_scale_free_exact_count_conservation_and_expected_xyz_loss(tmp_path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    torch.manual_seed(4040)
    backbone = LoomUni3DARGenerator._build_backbone(
        merge_level=6,
        allowed_elements=("H", "C", "N", "O", "F"),
        batch_size=4,
        emb_dim=64,
        num_head=4,
        layer=2,
        ffn_multiple=2,
        dropout=0.05,
        random_rotation_prob=0.8,
    )
    gen = LoomUni3DARGenerator(
        backbone=backbone,
        target_cols=["homo", "sa_score"],
        target_units=["eV", "sa"],
        mean=[-6.5, 2.8],
        scale=[0.8, 0.7],
        num_rbf=16,
        num_freqs=8,
        num_prefix_tokens=8,
        lora_rank=0,
        use_valence_head=True,
        use_scale_free_count=True,
        rope_center_prefix=True,
    )

    # 1. Verify exact Delta N == 0 conservation across N in [9, 17, 29, 44, 95] (> 80 unbounded!)
    gen.eval()
    target_counts = [9, 17, 29, 44, 95]
    prompts = np.array([[-6.6, 2.5]] * len(target_counts), dtype=np.float32)
    t_norm = gen.normalize_targets(prompts)
    c_mask = torch.ones((len(target_counts), 2), dtype=torch.bool)
    budgets = torch.tensor(target_counts, dtype=torch.float32)

    trajs = gen.sample(
        t_norm,
        c_mask,
        cfg_scale=1.6,
        geometric_guard=True,
        electron_parity=None,
        seed=1234,
        use_bf16=False,
        max_decode_steps=400,
        leaf_atom_budget=budgets,
    )
    assert [len(tr["atom_symbols"]) for tr in trajs] == target_counts

    # Also verify exact Delta N == 0 inside TwoStage funnel (`4 -> 3 -> 2`)
    s1 = OctreeProcessCritic(emb_dim=64, num_targets=2, hidden_dim=64)
    s2 = OctreeElementProcessCritic(emb_dim=64, num_targets=2, d_node=64, hidden_dim=64)
    ts_critic = TwoStageOctreeProcessCritic(s1, s2)
    grouped = gen.sample_with_two_stage_process_critic(
        t_norm[:4],
        c_mask[:4],
        ts_critic,
        m_stage1=4,
        m_stage2=3,
        m_keep=2,
        coarse_chunk_size=8,
        cfg_scale=1.6,
        geometric_guard=True,
        electron_parity=None,
        seed=5678,
        use_bf16=False,
        max_decode_steps=200,
        leaf_atom_budget=budgets[:4],
    )
    for idx, n_goal in enumerate(target_counts[:4]):
        assert [len(tr["atom_symbols"]) for tr in grouped[idx]] == [n_goal, n_goal]

    # 2. Verify full-backbone SFT loss (`loss_count` + `loss_geom`) & SO(3) OctreeDataset epoch rotation
    mols = []
    for smi in ("c1ccccc1O", "CC(=O)Nc1ccc(F)cc1"):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=17)
        mols.append(m)

    ds = gen.build_replay_dataset(mols, seed=99)
    ds.set_grid_epoch(3)
    packed = ds.collate([0, 1])

    gen.configure_trainable(train_conditioner=True, train_lora=False, unfreeze_full_backbone=True)
    gen.train()
    loss, info = gen.compute_sft_loss(
        packed,
        t_norm[:2],
        c_mask[:2],
        cfg_dropout_prob=0.10,
        contrast_weight=0.05,
        xyz_loss_weight=0.20,
        valence_loss_weight=0.15,
        count_loss_weight=0.25,
        geom_dist_loss_weight=1.50,
    )
    loss.backward()
    assert info["loss_geom"] > 0.0
    assert info["loss_count"] >= 0.0
    assert gen.scale_free_count_head is not None
    assert gen.scale_free_count_head.mlp[0].weight.grad is not None

    ckpt = tmp_path / "v4_step2_gen.pt"
    gen.save_checkpoint(ckpt, extra_metadata={"v4_step2": True})
    loaded_gen, payload = LoomUni3DARGenerator.load_checkpoint(ckpt, device="cpu")
    assert payload["loom_config"]["use_scale_free_count"] is True
    assert payload["loom_config"]["rope_center_prefix"] is True
    assert loaded_gen.scale_free_count_head is not None


def test_post_tree_refiner_se3_equivariance_and_priors(tmp_path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from loom.chemistry.valence_reconciler import ValenceReconciler
    from loom.weaving import PostTreeGraphRefiner, fit_post_tree_graph_refiner

    torch.manual_seed(777)
    mols = []
    for smi in ("c1ccccc1", "c1ccncc1", "CC(=O)Nc1ccc(F)cc1", "c1ccc2c(c1)cccc2"):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(m)
        mols.append(m)

    refiner = PostTreeGraphRefiner(d_model=64, n_heads=4, n_layers=2, n_rbf=16, d_pair=64, max_disp_A=0.08)
    stats = fit_post_tree_graph_refiner(refiner, mols * 6, epochs=4, batch_size=8, lr=2e-3, seed=2026)
    assert stats.val_bond_dist_mae_A <= stats.val_noisy_bond_dist_mae_A + 1e-3

    # Verify exact SE(3) equivariance of refined_pos and SE(3) invariance + symmetry of bond_logits
    refiner.eval()
    elem_ids = torch.tensor([[5, 5, 6, 7, 0, 0], [5, 5, 5, 8, 0, 0]], dtype=torch.long)
    pos = torch.randn(2, 6, 3) * 1.4
    valid = torch.tensor([[True, True, True, True, True, False], [True, True, True, True, True, True]], dtype=torch.bool)

    q_mat, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q_mat) < 0:
        q_mat[:, 0] = -q_mat[:, 0]
    shift = torch.tensor([[[8.2, -5.1, 3.7]]])
    pos_rt = pos @ q_mat.T + shift

    out1 = refiner(elem_ids, pos, valid)
    out2 = refiner(elem_ids, pos_rt, valid)

    ref1_rt = out1["refined_pos"] @ q_mat.T + shift
    assert torch.allclose(ref1_rt[valid], out2["refined_pos"][valid], atol=1e-5)
    assert torch.allclose(out1["bond_logits"], out2["bond_logits"], atol=1e-5)
    assert torch.allclose(out1["bond_logits"], out1["bond_logits"].transpose(1, 2), atol=1e-5)

    ckpt = tmp_path / "post_tree_refiner.pt"
    refiner.save_checkpoint(ckpt, extra={"bond_acc": stats.val_bond_order_acc})
    loaded_ref, extra = PostTreeGraphRefiner.load_checkpoint(ckpt, device="cpu")
    assert "bond_acc" in extra

    # Verify ValenceReconciler accepts bond_order_priors cleanly
    benz = mols[0]
    syms = [a.GetSymbol() for a in benz.GetAtoms()]
    coords = np.asarray(benz.GetConformer().GetPositions(), dtype=np.float32).copy()
    # Drop one H to force reconciliation & stretch one aromatic C-C bond slightly
    syms_drop = syms[:-1]
    coords_drop = coords[:-1]
    n_rem = len(syms_drop)
    priors = np.zeros((n_rem, n_rem, 5), dtype=np.float32)
    priors[:, :, 0] = 0.90
    for b in benz.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u < n_rem and v < n_rem:
            priors[u, v] = priors[v, u] = [0.02, 0.08, 0.10, 0.0, 0.80] if b.GetIsAromatic() else [0.02, 0.95, 0.03, 0.0, 0.0]
    rec = ValenceReconciler(mmff_h_iters=16)
    res = rec.reconcile_point_cloud(syms_drop, coords_drop, potential_slot=None, bond_order_priors=priors)
    assert res.reconciled_legal and res.mol is not None


def test_vine_rollouts_exact_budget_logprob_parity_and_grpo() -> None:
    from loom.weaving.grpo import compute_vine_counterfactual_advantages

    torch.manual_seed(909)
    backbone = LoomUni3DARGenerator._build_backbone(
        merge_level=6,
        allowed_elements=("H", "C", "N", "O", "F"),
        batch_size=4,
        emb_dim=64,
        num_head=4,
        layer=2,
        ffn_multiple=2,
        dropout=0.05,
        random_rotation_prob=0.8,
    )
    gen = LoomUni3DARGenerator(
        backbone=backbone,
        target_cols=["homo", "sa_score"],
        target_units=["eV", "sa"],
        mean=[-6.5, 2.8],
        scale=[0.8, 0.7],
        num_rbf=16,
        num_freqs=8,
        num_prefix_tokens=8,
        lora_rank=0,
        use_valence_head=True,
        use_scale_free_count=True,
        rope_center_prefix=True,
    )
    gen.eval()

    prompts_raw = np.array([[-6.6, 2.40], [-5.9, 2.40]], dtype=np.float32)
    t_norm = gen.normalize_targets(prompts_raw)
    c_mask = torch.ones((2, 2), dtype=torch.bool)
    budgets = torch.tensor([12.0, 16.0], dtype=torch.float32)

    trajs = gen.sample_vine_rollouts(
        t_norm,
        c_mask,
        m_trunks=3,
        k_leaves=2,
        cfg_scale=1.6,
        geometric_guard=True,
        temperatures=(1.0, 1.0, 1.0),
        electron_parity=0,
        seed=42,
        use_bf16=False,
        leaf_atom_budget=budgets,
    )
    assert len(trajs) == 2 * 3 * 2
    assert [len(tr["atom_symbols"]) for tr in trajs[:6]] == [12] * 6
    assert [len(tr["atom_symbols"]) for tr in trajs[6:]] == [16] * 6

    # Siblings sharing a trunk (`k_leaves=2`) must share identical coarse `level >= 1` decisions!
    for t_idx in range(0, len(trajs), 2):
        df0 = trajs[t_idx]["decoder_feat"]
        df1 = trajs[t_idx + 1]["decoder_feat"]
        c0 = df0[df0[:, 4] > gen.backbone.atom_level_index]
        c1 = df1[df1[:, 4] > gen.backbone.atom_level_index]
        assert torch.equal(c0, c1)

    # Verify exact teacher-forcing logprob parity WITH `scale_free_count_head` + `leaf_atom_budget`!
    t_rep = t_norm.repeat_interleave(6, dim=0)
    m_rep = c_mask.repeat_interleave(6, dim=0)
    packets = gen.trajectory_distributions(
        trajs,
        t_rep,
        m_rep,
        temperatures=(1.0, 1.0, 1.0),
        cfg_scale=1.6,
        geometric_guard=True,
        electron_parity=0,
        topology_only_policy=True,
    )
    tot_tf = torch.zeros(len(trajs))
    for logp, labels, owner in packets:
        chosen = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        assert torch.isfinite(chosen).all()
        tot_tf.index_add_(0, owner, chosen)
    samp_scores = torch.tensor([float(tr["scores"].sum().item()) for tr in trajs])
    assert float((tot_tf - samp_scores).abs().max().item()) < 5e-4

    # Verify counterfactual Tree-Vine advantage decomposition
    fake_adv = torch.tensor([1.2, 0.8, -0.4, -0.6, -0.5, -0.5, 0.9, -0.9, 0.2, 0.1, -0.2, -0.1])
    adv_tree, adv_leaf = compute_vine_counterfactual_advantages(fake_adv, m_trunks=3, k_leaves=2)
    assert adv_tree.shape == fake_adv.shape
    assert adv_leaf.shape == fake_adv.shape
    # Sibling leaves of trunk 0 share identical `adv_tree` while differing in `adv_leaf`!
    assert float(adv_tree[0].item()) == float(adv_tree[1].item())
    assert float(adv_leaf[6].item()) > float(adv_leaf[7].item())


def test_non_trapping_global_budget_sft_and_vine_parity(tmp_path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    torch.manual_seed(2026)
    backbone = LoomUni3DARGenerator._build_backbone(
        merge_level=6,
        allowed_elements=("H", "C", "N", "O", "F"),
        batch_size=4,
        emb_dim=64,
        num_head=4,
        layer=2,
        ffn_multiple=2,
        dropout=0.0,
        random_rotation_prob=0.8,
    )
    gen = LoomUni3DARGenerator(
        backbone=backbone,
        target_cols=["homo", "sa_score"],
        target_units=["eV", "sa"],
        mean=[-6.5, 2.8],
        scale=[0.8, 0.7],
        num_rbf=16,
        num_freqs=8,
        num_prefix_tokens=0,
        lora_rank=8,
        lora_alpha=16.0,
        lora_include_ffn=True,
        lora_include_heads=True,
        lora_include_xyz_heads=False,
        train_xyz_embeddings=False,
        use_valence_head=True,
        use_scale_free_count=True,
        non_trapping_budget=True,
        rope_center_prefix=False,
    )
    assert gen.budget_emb is not None
    assert gen.budget_emb.global_only is True
    assert gen.scale_free_count_head is None

    # Train 1 step so `budget_emb` & `LoRA` become non-zero, then verify exact Vine log-prob parity!
    mols = []
    for smi in ("c1ccccc1O", "CC(=O)Nc1ccc(F)cc1"):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=23)
        mols.append(m)

    ds = gen.build_replay_dataset(mols, seed=11)
    packed = ds.collate([0, 1])
    prompts_raw = np.array([[-6.6, 2.1], [-5.8, 2.3]], dtype=np.float32)
    t_norm = gen.normalize_targets(prompts_raw)
    c_mask = torch.ones((2, 2), dtype=torch.bool)

    opt = torch.optim.AdamW([p for p in gen.parameters() if p.requires_grad], lr=5e-3)
    gen.train()
    loss, _ = gen.compute_sft_loss(packed, t_norm, c_mask, budget_dropout_prob=0.0)
    loss.backward()
    assert gen.budget_emb.net[0].weight.grad is not None
    opt.step()

    ckpt = tmp_path / "non_trapping_v4.pt"
    gen.save_checkpoint(ckpt)
    loaded, payload = LoomUni3DARGenerator.load_checkpoint(ckpt, device="cpu")
    assert payload["loom_config"]["non_trapping_budget"] is True
    assert loaded.non_trapping_budget is True
    assert loaded.scale_free_count_head is None

    budgets = torch.tensor([14.0, 22.0], dtype=torch.float32)
    trajs = loaded.sample_vine_rollouts(
        t_norm,
        c_mask,
        m_trunks=2,
        k_leaves=2,
        cfg_scale=1.8,
        geometric_guard=True,
        temperatures=(1.0, 1.0, 1.0),
        electron_parity=0,
        seed=99,
        use_bf16=False,
        max_decode_steps=48,
        leaf_atom_budget=budgets,
    )
    assert len(trajs) == 8
    assert [tr.get("budget_atoms") for tr in trajs] == [14, 14, 14, 14, 22, 22, 22, 22]

    t_rep = t_norm.repeat_interleave(4, dim=0)
    m_rep = c_mask.repeat_interleave(4, dim=0)
    packets = loaded.trajectory_distributions(
        trajs,
        t_rep,
        m_rep,
        temperatures=(1.0, 1.0, 1.0),
        cfg_scale=1.8,
        geometric_guard=True,
        electron_parity=0,
        topology_only_policy=True,
    )
    tot_tf = torch.zeros(len(trajs))
    for logp, labels, owner in packets:
        tot_tf.index_add_(0, owner, logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1))
    samp_scores = torch.tensor([float(tr["scores"].sum().item()) for tr in trajs])
    assert float((tot_tf - samp_scores).abs().max().item()) < 2e-3

    # Verify post-GRPO `enable_v11_xyz_heads` + checkpoint round-trip + Kabsch paired refiner distillation!
    from loom.weaving.post_tree_refiner import PostTreeGraphRefiner, fit_post_tree_graph_refiner, kabsch_align_coords

    loaded.enable_v11_xyz_heads(rank=None, alpha=None, train_xyz_embeddings=True)
    xyz_ckpt = tmp_path / "non_trapping_v4_xyz.pt"
    loaded.save_checkpoint(xyz_ckpt)
    loaded_xyz, payload_xyz = LoomUni3DARGenerator.load_checkpoint(xyz_ckpt, device="cpu")
    assert payload_xyz["loom_config"]["lora_include_xyz_heads"] is True
    assert loaded_xyz.non_trapping_budget is True

    # Verify Kabsch SE(3) alignment & paired_trajectories refiner training (`max_disp_A = 0.25`)
    eq_p = np.asarray(mols[0].GetConformer().GetPositions(), dtype=np.float32)
    q_r, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    if np.linalg.det(q_r) < 0:
        q_r[:, 0] = -q_r[:, 0]
    rot_p = (eq_p @ q_r.T + np.array([[4.5, -2.2, 1.8]])).astype(np.float32)
    aligned_p = kabsch_align_coords(eq_p, rot_p)
    assert np.allclose(aligned_p, rot_p, atol=1e-4)

    noisy_pair = rot_p + np.random.default_rng(9).normal(0.0, 0.12, size=rot_p.shape).astype(np.float32)
    ref4 = PostTreeGraphRefiner(d_model=64, n_heads=4, n_layers=4, n_rbf=24, d_pair=64, max_disp_A=0.25)
    st_ref4 = fit_post_tree_graph_refiner(
        ref4,
        mols * 4,
        paired_trajectories=[(noisy_pair, mols[0])] * 4,
        epochs=4,
        batch_size=8,
        lr=2e-3,
        seed=2026,
    )
    assert st_ref4.val_bond_dist_mae_A <= st_ref4.val_noisy_bond_dist_mae_A + 1e-3

    # Verify refiner_disp_rms_A tracking
    tr_sample = {"atom_symbols": [a.GetSymbol() for a in mols[0].GetAtoms()], "positions_np": noisy_pair.copy()}
    ref4.refine_trajectories_batch([tr_sample], batch_size=4)
    assert "refiner_disp_rms_A" in tr_sample and np.isfinite(tr_sample["refiner_disp_rms_A"])
    assert "refiner_disp_max_A" in tr_sample and tr_sample["refiner_disp_max_A"] >= tr_sample["refiner_disp_rms_A"] - 1e-6


def test_grouped_ranking_critic_and_stage2_mmr_diversity(tmp_path: Path) -> None:
    """Verify grouped ListNet+pairwise ranking loss (`n_gnn_layers=3`) & Stage-2 Trunk-Diverse MMR."""
    torch.manual_seed(101)
    backbone = LoomUni3DARGenerator._build_backbone(
        merge_level=6,
        allowed_elements=("H", "C", "N", "O", "F"),
        batch_size=8,
        emb_dim=64,
        num_head=4,
        layer=2,
        ffn_multiple=2,
        dropout=0.0,
    )
    gen = LoomUni3DARGenerator(
        backbone=backbone,
        target_cols=["homo", "sa_score"],
        target_units=["eV", "sa"],
        mean=[-6.5, 2.8],
        scale=[0.8, 0.7],
        num_rbf=16,
        num_freqs=8,
        num_prefix_tokens=0,
        non_trapping_budget=True,
        rope_center_prefix=False,
    )
    n_prompts = 6
    g_sz = 4
    n_tot = n_prompts * g_sz
    t_p = torch.randn(n_prompts, 2)
    m_p = torch.ones(n_prompts, 2, dtype=torch.bool)
    t_rep = t_p.repeat_interleave(g_sz, dim=0)
    m_rep = m_p.repeat_interleave(g_sz, dim=0)
    gids = torch.arange(n_prompts, dtype=torch.long).repeat_interleave(g_sz)

    roll = gen.sample_rollouts_with_two_stage_features(
        t_rep,
        m_rep,
        cfg_scale=1.5,
        cfg_tree_decay=0.20,
        geometric_guard=True,
        electron_parity=0,
        seed=303,
        use_bf16=False,
        max_decode_steps=64,
    )
    y_surr = t_rep + 0.15 * torch.randn(n_tot, 2)
    y_ok = torch.ones(n_tot, dtype=torch.bool)
    is_leg = torch.tensor([i % 5 != 0 for i in range(n_tot)], dtype=torch.bool)
    is_con = torch.tensor([i % 2 == 0 for i in range(n_tot)], dtype=torch.bool)

    s1 = OctreeProcessCritic(emb_dim=64, num_targets=2, hidden_dim=96)
    s2 = OctreeElementProcessCritic(emb_dim=64, num_targets=2, d_node=64, n_gnn_layers=3, hidden_dim=96)
    obj_m = torch.tensor([True, False], dtype=torch.bool)

    st1 = fit_octree_process_critic(
        s1,
        h_concat=roll["s1_h_concat"],
        geom_feats=roll["s1_geom_feats"],
        targets_norm=t_rep,
        active_mask=m_rep,
        y_surrogate_norm=y_surr,
        y_valid_mask=y_ok,
        is_legal=is_leg,
        is_constr_met=is_con,
        group_ids=gids,
        ranking_weight=0.50,
        ranking_obj_mask=obj_m,
        epochs=4,
        batch_size=8,
    )
    st2 = fit_octree_element_process_critic(
        s2,
        h_stage2=roll["s2_h_stage2"],
        elem_ids=roll["s2_elem_ids"],
        leaf_pos=roll["s2_leaf_pos"],
        leaf_valid=roll["s2_leaf_valid"],
        cloud_feats=roll["s2_cloud_feats"],
        targets_norm=t_rep,
        active_mask=m_rep,
        y_surrogate_norm=y_surr,
        y_valid_mask=y_ok,
        is_legal=is_leg,
        is_constr_met=is_con,
        group_ids=gids,
        ranking_weight=0.60,
        ranking_obj_mask=obj_m,
        epochs=4,
        batch_size=8,
    )
    assert np.isfinite(st1.train_loss) and np.isfinite(st2.train_loss)

    ts = TwoStageOctreeProcessCritic(s1, s2)
    ckpt = tmp_path / "ts_rank3.pt"
    ts.save_checkpoint(ckpt)
    ts_loaded, _ = TwoStageOctreeProcessCritic.load_checkpoint(ckpt)
    assert ts_loaded.stage2_critic.n_gnn_layers == 3

    grouped = gen.sample_with_two_stage_process_critic(
        t_p[:3],
        m_p[:3],
        ts_loaded,
        m_stage1=6,
        m_stage2=4,
        m_keep=2,
        objective_col_mask=obj_m,
        stage2_diversity_weight=0.45,
        cfg_scale=1.5,
        geometric_guard=True,
        electron_parity=0,
        seed=404,
        use_bf16=False,
        max_decode_steps=64,
    )
    assert len(grouped) == 3 and all(len(g) == 2 for g in grouped)

    # Verify GRPO `cfg_anchor_cols=("sa_score",)` decouples HOMO-only CFG from `prompt_atom_budgets`
    from loom.weaving.grpo import LoomGRPOTrainer

    class _DummyOracle:
        target_cols = ["homo", "sa_score"]
        capabilities = type("Cap", (), {"is_audit_only": False})()

    gen.configure_trainable(train_conditioner=True)
    tr = LoomGRPOTrainer(
        generator=gen,
        reward_oracle=_DummyOracle(),  # type: ignore[arg-type]
        size_anchored_cfg=False,
        cfg_anchor_cols=("sa_score",),
        size_adherence_weight=0.30,
    )
    assert tr.size_col_idx is None
    assert tr.cfg_anchor_indices == (1,)
    u_mask = tr._build_uncond_active_mask(torch.ones((3, 2), dtype=torch.bool))
    assert u_mask is not None and torch.equal(u_mask, torch.tensor([[False, True]] * 3))
    from rdkit import Chem
    m12 = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    _, errs, _, _, _ = tr._compute_size_metrics_and_advantages(
        [m12, m12],
        np.array([True, True]),
        np.array([[-6.5, 2.25], [-6.5, 2.25]], dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        group_size=2,
        prompt_atom_budgets=np.array([12.0], dtype=np.float32),
    )
    assert np.allclose(errs, 0.0, atol=1e-6)





