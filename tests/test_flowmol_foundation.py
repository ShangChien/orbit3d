"""Unit tests for the DGL-free SE(3)-equivariant general FlowMol foundation package."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from loom.flowmol import (
    ATOM_MASK_IDX,
    BOND_MASK_IDX,
    CHARGE_MASK_IDX,
    DynamicEdgeBudgetBatchSampler,
    FLOWMOL_ATOM_TYPES,
    FlatMoleculeCorpus,
    FlowMolConfig,
    FlowMolModel,
    N_ATOM_TYPES,
    N_BOND_TYPES,
    N_CHARGES,
    build_complete_graph_batch,
    collate_flowmol_batch,
    disable_lora,
    inject_flowmol_lora,
    mol_to_sparse_record,
    proper_kabsch_align_fp32,
    write_flat_corpus,
)


def _random_so3_matrix(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn((3, 3), generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    if torch.linalg.det(Q) < 0:
        Q[:, -1] = -Q[:, -1]
    assert abs(float(torch.linalg.det(Q).item()) - 1.0) < 1e-5
    return Q


def _make_sample_rdkit_mols() -> list[Chem.Mol]:
    smiles_list = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "c1ccc2c(c1)[nH]c1ccccc12",       # carbazole
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
        "O=C(O)c1ccc(S(=O)(=O)N)cc1",
        "FC(F)(F)c1ccc(Cl)cc1",
    ]
    mols = []
    for idx, smi in enumerate(smiles_list):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        params = AllChem.ETKDGv3()
        params.randomSeed = 100 + idx
        assert AllChem.EmbedMolecule(m, params) == 0
        mols.append(m)
    return mols


def test_full_periodic_table_alignment_and_legacy_lut_remapping(tmp_path: Path):
    import json
    from loom.flowmol import ATOM_TO_IDX, LEGACY_6_FORMAL_CHARGES, ORGANIC_16_ATOM_TYPES
    from loom.vendor.uni3dar.data.atom_data import atom_list as uni3dar_atoms

    # Exact 118-element slot-for-slot parity with Uni-3DAR
    assert N_ATOM_TYPES == 118
    assert ATOM_MASK_IDX == 118
    assert N_CHARGES == 11
    assert CHARGE_MASK_IDX == 11
    assert N_BOND_TYPES == 6
    assert BOND_MASK_IDX == 6
    assert FLOWMOL_ATOM_TYPES == tuple(uni3dar_atoms)
    assert ATOM_TO_IDX["H"] == 0
    assert ATOM_TO_IDX["C"] == 5
    assert ATOM_TO_IDX["La"] == 56
    assert ATOM_TO_IDX["Ir"] == 76
    assert ATOM_TO_IDX["Pt"] == 77
    assert ATOM_TO_IDX["Og"] == 117

    # Simulate a legacy 16-element / 6-charge corpus (like flowmol-corpus-v1:1) and verify LUT remapping
    legacy_to_idx = {s: i for i, s in enumerate(ORGANIC_16_ATOM_TYPES)}
    legacy_q_to_idx = {q: i for i, q in enumerate(LEGACY_6_FORMAL_CHARGES)}
    m = Chem.AddHs(Chem.MolFromSmiles("CC[O-]"))
    p = AllChem.ETKDGv3()
    p.randomSeed = 77
    assert AllChem.EmbedMolecule(m, p) == 0
    rec_legacy = mol_to_sparse_record(m, atom_to_idx=legacy_to_idx, charge_to_idx=legacy_q_to_idx)
    assert rec_legacy is not None

    write_flat_corpus(tmp_path, [rec_legacy] * 4, splits=[0, 0, 0, 1], source_ids=[0] * 4, source_names=["legacy"])
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["atom_types"] = list(ORGANIC_16_ATOM_TYPES)
    meta["n_atom_types"] = len(ORGANIC_16_ATOM_TYPES)
    meta["formal_charges"] = list(LEGACY_6_FORMAL_CHARGES)
    meta["n_charges"] = len(LEGACY_6_FORMAL_CHARGES)
    meta["element_counts"] = {"H": 20, "C": 8, "O": 4, "Li": 0}
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    ds = FlatMoleculeCorpus(tmp_path, split="train", load_in_memory=True)
    item = ds[0]
    expected_syms = [a.GetSymbol() for a in m.GetAtoms()]
    expected_qs = [int(a.GetFormalCharge()) for a in m.GetAtoms()]
    remapped_syms = [FLOWMOL_ATOM_TYPES[int(i)] for i in item["a_1"].tolist()]
    from loom.flowmol import FLOWMOL_FORMAL_CHARGES
    remapped_qs = [FLOWMOL_FORMAL_CHARGES[int(i)] for i in item["c_1"].tolist()]
    assert remapped_syms == expected_syms
    assert remapped_qs == expected_qs
    assert set(ds.active_elements) == {"H", "C", "O"}


def test_organometallic_coordinate_bond_roundtrip():
    from loom.flowmol import tensors_to_rdkit_mol

    # Build octahedral [Ir(CN)3(NH3)3] complex with Ir(3+) + 3 [C-]#N + 3 NH3 via DATIVE (L -> Ir) bonds
    rw = Chem.RWMol()
    ir_i = rw.AddAtom(Chem.Atom("Ir"))
    rw.GetAtomWithIdx(ir_i).SetFormalCharge(3)
    rw.GetAtomWithIdx(ir_i).SetNoImplicit(True)

    dirs = np.array(
        [[2.0, 0, 0], [-2.0, 0, 0], [0, 2.0, 0], [0, -2.1, 0], [0, 0, 2.1], [0, 0, -2.1]],
        dtype=np.float64,
    )
    coords = [[0.0, 0.0, 0.0]]
    # 3 cyanide ligands
    for k in range(3):
        c_i = rw.AddAtom(Chem.Atom("C"))
        n_i = rw.AddAtom(Chem.Atom("N"))
        rw.GetAtomWithIdx(c_i).SetFormalCharge(-1)
        rw.GetAtomWithIdx(c_i).SetNoImplicit(True)
        rw.GetAtomWithIdx(n_i).SetNoImplicit(True)
        rw.AddBond(c_i, n_i, Chem.BondType.TRIPLE)
        rw.AddBond(c_i, ir_i, Chem.BondType.DATIVE)
        u = dirs[k] / np.linalg.norm(dirs[k])
        coords.append((dirs[k]).tolist())
        coords.append((dirs[k] + u * 1.16).tolist())
    # 3 ammine ligands
    for k in range(3, 6):
        n_i = rw.AddAtom(Chem.Atom("N"))
        rw.GetAtomWithIdx(n_i).SetNoImplicit(True)
        rw.AddBond(n_i, ir_i, Chem.BondType.DATIVE)
        u = dirs[k] / np.linalg.norm(dirs[k])
        coords.append(dirs[k].tolist())
        perp1 = np.array([u[1], -u[0], 0.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(u, perp1)
        for ang in (0.0, 2.094, 4.188):
            h_i = rw.AddAtom(Chem.Atom("H"))
            rw.GetAtomWithIdx(h_i).SetNoImplicit(True)
            rw.AddBond(n_i, h_i, Chem.BondType.SINGLE)
            h_vec = dirs[k] + 0.4 * u + 0.94 * (np.cos(ang) * perp1 + np.sin(ang) * perp2)
            coords.append(h_vec.tolist())

    mol = rw.GetMol()
    conf = Chem.Conformer(len(coords))
    for idx, xyz in enumerate(coords):
        conf.SetAtomPosition(idx, tuple(xyz))
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    mol.UpdatePropertyCache(strict=False)

    rec = mol_to_sparse_record(mol, max_atoms=120)
    assert rec is not None
    assert (rec["bond_type"] == 5).sum() == 6  # 6 COORDINATE bonds

    na = int(rec["pos"].shape[0])
    e_up = np.zeros(na * (na - 1) // 2, dtype=np.int64)
    bu, bv = rec["bond_u"].astype(np.int64), rec["bond_v"].astype(np.int64)
    e_up[bu * na - (bu * (bu + 1)) // 2 + (bv - bu - 1)] = rec["bond_type"].astype(np.int64)

    rec_mol, diag = tensors_to_rdkit_mol(rec["pos"], rec["atom_type"], rec["charge"], e_up, sanitize=True)
    assert rec_mol is not None
    assert diag["valid"] and diag["connected"]
    assert diag["has_metal"] and diag["has_coordinate_bond"]


def test_preset_parameter_counts():
    sml = FlowMolModel(FlowMolConfig.preset("small"))
    std = FlowMolModel(FlowMolConfig.preset("standard"))
    flg = FlowMolModel(FlowMolConfig.preset("flagship"))
    n_sml = sml.count_parameters()["total"]
    n_std = std.count_parameters()["total"]
    n_flg = flg.count_parameters()["total"]
    # 3-tier geometric Scaling Law suite: ~4.2M -> ~12.7M -> ~40.2M
    assert 3_600_000 <= n_sml <= 5_000_000, f"Unexpected small params: {n_sml}"
    assert 9_000_000 <= n_std <= 16_000_000, f"Unexpected standard params: {n_std}"
    assert 38_000_000 <= n_flg <= 58_000_000, f"Unexpected flagship params: {n_flg}"
    assert 2.5 <= (n_std / n_sml) <= 3.6
    assert 2.5 <= (n_flg / n_std) <= 3.6


def test_strict_se3_equivariance_with_and_without_self_conditioning():
    torch.manual_seed(7)
    cfg = FlowMolConfig.preset("tiny", self_conditioning=True, n_cp_feats=4)
    model = FlowMolModel(cfg).eval()
    # Set sc_vec_scale non-zero so we verify the equivariant self-conditioning vector channel too
    with torch.no_grad():
        model.sc_vec_scale.fill_(0.5)

    batch = build_complete_graph_batch([5, 7], device=torch.device("cpu"))
    n_nodes = batch.x_0.shape[0]
    n_up = int(batch.upper_edge_mask.sum().item())

    x_t = batch.x_0.clone()
    a_t = torch.randint(0, N_ATOM_TYPES + 1, (n_nodes,))
    c_t = torch.randint(0, N_CHARGES + 1, (n_nodes,))
    e_t = torch.randint(0, N_BOND_TYPES + 1, (n_up,))
    t = torch.tensor([0.35, 0.72], dtype=torch.float32)

    R = _random_so3_matrix(seed=123)
    trans = torch.tensor([[1.7, -2.3, 0.9]], dtype=torch.float32)

    with torch.no_grad():
        out1 = model(batch, x_t, a_t, c_t, e_t, t, sc_dict=None, remove_com=True)
        out1_rot = model(batch, x_t @ R.T + trans, a_t, c_t, e_t, t, sc_dict=None, remove_com=True)

        assert torch.allclose(out1_rot["x"], out1["x"] @ R.T, atol=1e-4)
        assert torch.allclose(out1_rot["a"], out1["a"], atol=1e-4)
        assert torch.allclose(out1_rot["c"], out1["c"], atol=1e-4)
        assert torch.allclose(out1_rot["e"], out1["e"], atol=1e-4)

        # Now feed equivariantly transformed self-conditioning state
        sc = {
            "x": out1["x"],
            "a": torch.softmax(out1["a"], dim=-1),
            "c": torch.softmax(out1["c"], dim=-1),
            "e": torch.softmax(out1["e"], dim=-1),
        }
        sc_rot = {
            "x": out1["x"] @ R.T + trans,
            "a": sc["a"],
            "c": sc["c"],
            "e": sc["e"],
        }
        out2 = model(batch, x_t, a_t, c_t, e_t, t, sc_dict=sc, remove_com=True)
        out2_rot = model(batch, x_t @ R.T + trans, a_t, c_t, e_t, t, sc_dict=sc_rot, remove_com=True)

        assert torch.allclose(out2_rot["x"], out2["x"] @ R.T, atol=1e-4)
        assert torch.allclose(out2_rot["a"], out2["a"], atol=1e-4)
        assert torch.allclose(out2_rot["c"], out2["c"], atol=1e-4)
        assert torch.allclose(out2_rot["e"], out2["e"], atol=1e-4)


def test_proper_kabsch_preserves_chirality():
    # Chiral tetrahedron and its mirror image
    x1 = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.5]],
        dtype=torch.float32,
    )
    x1 = x1 - x1.mean(dim=0, keepdim=True)
    # Signed volume of x1
    def signed_vol(p):
        return float(torch.linalg.det(torch.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], dim=0)).item())

    x0_mirror = x1.clone()
    x0_mirror[:, 2] *= -1.0  # opposite chirality
    aligned = proper_kabsch_align_fp32(x0_mirror, x1)
    # Must NOT invert handedness to match x1
    assert np.sign(signed_vol(aligned)) == np.sign(signed_vol(x0_mirror))

    # Proper rotation of x1 must recover x1 exactly
    R = _random_so3_matrix(seed=99)
    x0_rot = x1 @ R.T
    aligned_rot = proper_kabsch_align_fp32(x0_rot, x1)
    assert torch.allclose(aligned_rot, x1, atol=1e-5)


def test_zero_init_conditioner_and_lora_disable_parity():
    torch.manual_seed(11)
    cfg = FlowMolConfig.preset("tiny", self_conditioning=False)
    model = FlowMolModel(cfg).eval()
    batch = build_complete_graph_batch([4, 6], device=torch.device("cpu"))
    n_nodes = batch.x_0.shape[0]
    n_up = int(batch.upper_edge_mask.sum().item())
    a_t = torch.full((n_nodes,), ATOM_MASK_IDX, dtype=torch.long)
    c_t = torch.full((n_nodes,), CHARGE_MASK_IDX, dtype=torch.long)
    e_t = torch.full((n_up,), BOND_MASK_IDX, dtype=torch.long)
    t = torch.tensor([0.2, 0.8], dtype=torch.float32)

    with torch.no_grad():
        base_out = model(batch, batch.x_0, a_t, c_t, e_t, t)

    # 1. Zero-initialized LoomConditioner preserves base output bit-for-bit
    model.attach_conditioner(num_props=2)
    cond_vals = torch.tensor([[-1.2, 0.5], [0.8, -0.3]], dtype=torch.float32)
    cond_mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)
    with torch.no_grad():
        cond_out = model(batch, batch.x_0, a_t, c_t, e_t, t, cond_values=cond_vals, cond_active_mask=cond_mask)
    for k in ("x", "a", "c", "e"):
        assert torch.allclose(cond_out[k], base_out[k], atol=1e-6)

    # 2. Inject LoRA -> zero init parity -> perturb lora_B -> disable_lora recovers base output
    injected = inject_flowmol_lora(model, rank=4, alpha=8.0)
    assert len(injected) > 0
    with torch.no_grad():
        lora_init_out = model(batch, batch.x_0, a_t, c_t, e_t, t)
    for k in ("x", "a", "c", "e"):
        assert torch.allclose(lora_init_out[k], base_out[k], atol=1e-6)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.fill_(0.1)
        perturbed_out = model(batch, batch.x_0, a_t, c_t, e_t, t)
        assert not torch.allclose(perturbed_out["a"], base_out["a"], atol=1e-4)
        with disable_lora(model):
            ref_out = model(batch, batch.x_0, a_t, c_t, e_t, t)
    for k in ("x", "a", "c", "e"):
        assert torch.allclose(ref_out[k], base_out[k], atol=1e-6)


def test_flat_corpus_roundtrip_bucketing_loss_and_sampling(tmp_path: Path):
    mols = _make_sample_rdkit_mols()
    records = [mol_to_sparse_record(m) for m in mols]
    assert all(r is not None for r in records)
    # Replicate to have enough records for 16-rank bucketing test
    rep_records = [records[i % len(records)] for i in range(128)]
    splits = [0] * 112 + [1] * 16
    src_ids = [i % 2 for i in range(128)]
    meta = write_flat_corpus(
        tmp_path,
        rep_records,
        splits=splits,
        source_ids=src_ids,
        source_names=["qm9", "oe62"],
    )
    assert meta["n_mols"] == 128

    corpus = FlatMoleculeCorpus(tmp_path, split="train", load_in_memory=True, seed=42)
    assert len(corpus) == 112

    # Verify exact 16-rank lockstep batching
    rank_lens = []
    rank_indices = []
    for rank in range(16):
        sampler = DynamicEdgeBudgetBatchSampler(
            corpus.n_atoms_view,
            max_edges_per_batch=1500,
            max_mols_per_batch=4,
            min_mols_per_batch=1,
            rank=rank,
            world_size=16,
            seed=7,
        )
        sampler.set_epoch(0)
        batches = list(sampler)
        rank_lens.append(len(batches))
        rank_indices.extend([idx for b in batches for idx in b])
    assert len(set(rank_lens)) == 1 and rank_lens[0] >= 1
    assert len(rank_indices) == len(set(rank_indices)), "Duplicate indices across DDP ranks!"

    # Verify compute_loss, grad checkpointing, and edge-case t~1 (zero masked tokens)
    batch = collate_flowmol_batch([corpus[i] for i in range(4)])
    cfg = FlowMolConfig.preset("tiny", self_conditioning=True, grad_checkpointing=True)
    model = FlowMolModel(cfg).train()
    loss, metrics = model.compute_loss(batch, use_self_conditioning=True, apply_distortion=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    # Edge case: t very close to 1.0 -> possibly zero masked tokens -> still finite loss
    model.zero_grad()
    t_near_one = torch.full((batch.batch_size,), 0.999999, dtype=torch.float32)
    loss_edge, _ = model.compute_loss(batch, t=t_near_one, use_self_conditioning=False, apply_distortion=False)
    assert torch.isfinite(loss_edge)
    loss_edge.backward()

    # Verify Euler and Heun sampling + allowed_elements masking + zero [MASK] leaks
    allowed = ("H", "C", "N", "O", "F")
    allowed_ids = {FLOWMOL_ATOM_TYPES.index(s) for s in allowed}
    for solver, grid in [("euler", "uniform"), ("heun", "quadratic_end")]:
        samples = model.sample(
            [6, 9],
            n_steps=6,
            solver=solver,
            grid_type=grid,
            allowed_elements=allowed,
            use_bf16=False,
            seed=42,
            build_rdkit=True,
        )
        assert len(samples) == 2
        for s in samples:
            assert s["positions"].shape[1] == 3
            assert np.allclose(s["positions"].mean(axis=0), 0.0, atol=1e-4)
            assert set(int(x) for x in s["atom_types"]).issubset(allowed_ids)
            assert not s["diag"]["has_mask_token"]


def test_ddp_grad_coverage_step_logprob_and_standalone_plus_slot_loading(tmp_path: Path):
    from loom.weaving.flowmol_slot import LoomFlowMolGenerator

    mols = _make_sample_rdkit_mols()[:4]
    recs = [mol_to_sparse_record(m) for m in mols]
    meta = write_flat_corpus(
        tmp_path / "corpus",
        recs,
        splits=[0, 0, 0, 1],
        source_ids=[0, 1, 0, 1],
        source_names=["qm9", "tmqm"],
    )
    ds = FlatMoleculeCorpus(tmp_path / "corpus", split="train", load_in_memory=True)
    batch = collate_flowmol_batch([ds[0], ds[1], ds[2]])

    cfg = FlowMolConfig.preset("tiny", self_conditioning=True)
    model = FlowMolModel(cfg).train()

    # 1. Verify EVERY trainable parameter receives a finite non-None gradient in both do_sc=False and do_sc=True,
    # both with and without BF16 autocast (prevents PyTorch autocast weight-cache + no_grad() gradient dropping!)
    for use_amp in (False, True):
        for do_sc in (False, True):
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = model.compute_loss(batch, use_self_conditioning=do_sc, apply_distortion=True)
            loss.backward()
            missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing, f"Parameters missing gradient when use_amp={use_amp}, do_sc={do_sc}: {missing}"

    # 2. Verify analytical sde_ctmc_step_logprob() returns finite differentiable per-molecule log-probs
    model.zero_grad(set_to_none=True)
    t_vec = torch.full((batch.batch_size,), 0.4, dtype=torch.float32)
    s_vec = torch.full((batch.batch_size,), 0.5, dtype=torch.float32)
    x_t, a_t, c_t, e_t = model.sample_conditional_path(batch, t_vec, apply_distortion=False)
    x_s, a_s, c_s, e_s = model.sample_conditional_path(batch, s_vec, apply_distortion=False)
    lp = model.sde_ctmc_step_logprob(
        batch, x_t, a_t, c_t, e_t, x_s, a_s, c_s, e_s, t_val=0.4, s_val=0.5
    )
    assert lp.shape == (batch.batch_size,)
    assert torch.isfinite(lp).all()
    (-lp.mean()).backward()

    # 3. Verify standalone save_pretrained / from_pretrained AND domain size sampling
    ckpt_path = tmp_path / "best_ema.pt"
    model.save_pretrained(ckpt_path, step=100, samples_seen=6400, corpus_meta=meta)
    loaded_model, loaded_meta = FlowMolModel.from_pretrained(ckpt_path, device="cpu")
    assert loaded_meta["format"] == "flowmol_foundation_v1"
    tmqm_sizes = FlowMolModel.sample_empirical_sizes(loaded_meta, n_mols=6, domain="tmqm", seed=7)
    assert len(tmqm_sizes) == 6
    standalone_samples = loaded_model.sample(tmqm_sizes[:2], n_steps=4, solver="heun", grid_type="quadratic_end", seed=9)
    assert len(standalone_samples) == 2

    # 4. Verify plug-and-play weight support in LoomFlowMolGenerator.from_pretrained_flowmol (with LoRA + SFT/AWR loss)
    slot = LoomFlowMolGenerator.from_pretrained_flowmol(
        ckpt_path,
        target_cols=["homo"],
        target_units=["eV"],
        mean=[-5.5],
        scale=[0.8],
        device="cpu",
        sample_timesteps=5,
        sft_time_draws=1,
        lora_rank=4,
    )
    targets_norm = torch.tensor([[-0.5], [0.7]], dtype=torch.float32)
    active_mask = torch.tensor([[True], [True]], dtype=torch.bool)
    slot_samples = slot.sample(targets_norm, active_mask, seed=123, cfg_scale=1.5, xyz_cfg_scale=1.2)
    assert len(slot_samples) == 2
    packed, kept = slot.encode_molecules_batch(mols[:2], seed=42)
    assert packed is not None and len(kept) == 2
    per_mol_loss, info = slot.compute_sft_loss(packed, targets_norm, active_mask, return_per_molecule=True)
    assert per_mol_loss.shape == (2,) and torch.isfinite(per_mol_loss).all()
    per_mol_loss.mean().backward()
