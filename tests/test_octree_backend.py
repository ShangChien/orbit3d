import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

pytest.importorskip("numba")
pytest.importorskip("lmdb")
from orbit3d.generator.octree_model import ConditionalUni3DAR, build_default_uni3dar_args, AtomFeatDict
from orbit3d.generator.octree_backend import OctreeDataset
from orbit3d.generator.octree_alignment import categorical_policy_loss
from orbit3d.vendor.uni3dar.models.rope import RoPEnD


def test_octree_dictionary_is_stable_and_sft_backpropagates():
    args = build_default_uni3dar_args(2)
    args.emb_dim = 96
    args.num_head = 4
    args.layer = 2
    first = AtomFeatDict(args)
    second = AtomFeatDict(args)
    assert first.atom_type == second.atom_type
    model = ConditionalUni3DAR(args, first, 2)
    # Upstream output heads start at zero; mimic the nonzero pretrained readout
    # so this checks the conditional gradient, not first-step initialization.
    with torch.no_grad():
        for parameter in model.tree_heads.parameters():
            if parameter.ndim > 1:
                parameter.normal_(0, 0.02)
    molecules = []
    for smi in ["CC", "CO"]:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        molecules.append(mol)
    data = OctreeDataset(molecules, model).collate([0, 1])
    y = torch.zeros(2, 2)
    loss, _ = model.compute_sft_and_understanding_loss(
        data, y, y - 0.2, y + 0.2, torch.ones(2, 2, dtype=torch.bool)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.interval_conditioner.parameters())


def test_rotary_positions_preserve_pair_norm_and_dot_products():
    rope = RoPEnD(64)
    position = torch.randn(7, 3)
    rope.calc_and_cache(position)
    q = torch.randn(7, 12, 64)
    k = torch.randn_like(q)
    qr, kr = rope.apply_qk(q, k)
    torch.testing.assert_close(q.square().sum(-1), qr.square().sum(-1), atol=2e-5, rtol=1e-5)
    torch.testing.assert_close((q * k).sum(-1), (qr * kr).sum(-1), atol=2e-5, rtol=1e-5)


def test_octree_policy_loss_includes_coordinate_heads_and_kl_gradient():
    owners = torch.tensor([0, 1])
    labels = torch.tensor([0, 1])
    advantages = torch.tensor([1.0, -1.0])
    raw = [torch.randn(2, 4, requires_grad=True) for _ in range(5)]
    current = [(r.log_softmax(-1), labels, owners) for r in raw]
    old = [(v.detach(), y, o) for v, y, o in current]
    reference = [(torch.zeros_like(v).log_softmax(-1), y, o) for v, y, o in current]
    loss = categorical_policy_loss(current, old, reference, advantages)
    loss.backward()
    assert all(r.grad is not None and r.grad.abs().sum() > 0 for r in raw)
