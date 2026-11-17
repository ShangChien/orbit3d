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
    OctreeProcessCritic,
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
