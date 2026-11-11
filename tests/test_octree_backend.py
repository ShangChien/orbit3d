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


def test_one_sided_masks_are_explicit_and_missing_bounds_are_inert():
    from orbit3d.generator.octree_model import DynamicIntervalConditioner
    from orbit3d.generator.octree_backend import interval_arrays
    from orbit3d.constraints import TargetIntervals

    intervals = TargetIntervals.parse({"homo": [None, -5], "lumo": [0, None]}, ["homo", "lumo"])
    with pytest.raises(ValueError, match="v2"):
        interval_arrays(intervals, np.zeros(2), np.ones(2), 1)
    lo, hi, flags = interval_arrays(intervals, np.zeros(2), np.ones(2), 2)
    np.testing.assert_array_equal(flags, [[False, True], [True, False]])
    conditioner = DynamicIntervalConditioner(emb_dim=32, num_props=2, num_layers=1, bound_masks=True)
    lo, hi = torch.tensor(lo)[None], torch.tensor(hi)[None]
    mask = torch.tensor(flags)[None]
    first = conditioner(lo, hi, mask)
    changed_lo, changed_hi = lo.clone(), hi.clone()
    changed_lo[0, 0] = -999
    changed_hi[0, 1] = 999
    torch.testing.assert_close(first, conditioner(changed_lo, changed_hi, mask))
    first.square().sum().backward()
    assert conditioner.bound_projection.weight.grad.abs().sum() > 0
    null = conditioner(changed_lo, changed_hi, torch.zeros_like(mask))
    assert torch.count_nonzero(null) == 0


def test_checkpoint_semantics_include_grid_units_and_scaler():
    from types import SimpleNamespace
    from orbit3d.generator.octree_backend import checkpoint_contract

    args = build_default_uni3dar_args(1)
    model = SimpleNamespace(args=args, dictionary=AtomFeatDict(args), condition_version=2)
    contract = checkpoint_contract(model, ['homo'], ['eV'], [-7.0], [0.5])
    assert checkpoint_contract(model, ['homo'], ['Ha'], [-7.0], [0.5]) != contract
    assert checkpoint_contract(model, ['homo'], ['eV'], [-6.0], [0.5]) != contract
    model.args.grid_len = 0.48
    assert checkpoint_contract(model, ['homo'], ['eV'], [-7.0], [0.5]) != contract
    with pytest.raises(ValueError, match='scaler'):
        checkpoint_contract(model, ['homo'], ['eV'], [-7.0], [0.0])


def test_smooth_condition_features_preserve_local_property_neighborhoods():
    from orbit3d.generator.octree_model import DynamicIntervalConditioner

    smooth = DynamicIntervalConditioner(emb_dim=32, num_props=2, num_layers=1,
                                        bound_masks=True, smooth_frequencies=True)
    legacy = DynamicIntervalConditioner(emb_dim=32, num_props=2, num_layers=1, bound_masks=True)
    values = torch.tensor([[-1.2, .3, -1.5, -.9], [.5, .2, .3, .7]])
    small = 1e-5
    assert (smooth._fourier_encode(values+small)-smooth._fourier_encode(values)).abs().max() < .001
    assert (legacy._fourier_encode(values+small)-legacy._fourier_encode(values)).abs().max() > .1


def test_cfg_guides_each_nonlinear_head_with_the_same_sampled_prefix():
    from orbit3d.vendor.uni3dar.models.autoregressive_prediction_head import AutoRegressivePredictionHead
    torch.manual_seed(13)
    head = AutoRegressivePredictionHead([5, 7, 7, 7], 16, emb_dropout=0).eval()
    for p in head.parameters():
        if p.ndim > 1:
            p.data.normal_(0, .15)
    conditional, unconditional = torch.randn(3, 16), torch.randn(3, 16)
    for scale in [0., 2.]:
        captured = []
        def sample(logits):
            captured.append(logits)
            return logits.argmax(-1), torch.zeros(len(logits))
        chosen, _ = head.inference(conditional, [sample]*4, [1]*4,
                                   unconditional_hidden=unconditional, cfg_scale=scale)
        targets = [chosen[:, i].long() for i in range(4)]
        expected = [u+scale*(c-u) for c, u in zip(head(conditional, targets), head(unconditional, targets))]
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


def test_terminal_parity_mask_is_a_conditional_categorical_distribution():
    from orbit3d.generator.electronic_constraints import mask_terminal_element_logits
    # H/C/N/O/F/null: only the last decision is restricted. Previous atoms
    # contribute odd/even electron counts independently for each molecule.
    probabilities=torch.tensor([[.2,.3,.1,.2,.1,.1]]).repeat(3,1)
    logits=mask_terminal_element_logits(probabilities.log(),torch.tensor([0,1,0]),
                                       torch.tensor([True,True,False]),0,torch.tensor([1,0,1,0,1,0]))
    p=logits.softmax(-1)
    torch.testing.assert_close(p[0],torch.tensor([0.,.5,0.,1/3,0.,1/6]))
    torch.testing.assert_close(p[1],torch.tensor([.5,0.,.25,0.,.25,0.]))
    torch.testing.assert_close(p[2],probabilities[2])


def test_octree_encoding_rejects_lost_atoms_and_disallowed_elements():
    from types import SimpleNamespace
    args = build_default_uni3dar_args(1)
    args.random_rotation_prob = 0
    model = SimpleNamespace(args=args, dictionary=AtomFeatDict(args))
    mol = Chem.MolFromSmiles('CC')
    mol.AddConformer(Chem.Conformer(2))  # Both atoms occupy the same leaf cell.
    dataset = OctreeDataset([mol], model, training=False)
    with pytest.raises(ValueError, match='lost atoms'):
        dataset.collate([0])
    other = Chem.MolFromSmiles('B')
    other.AddConformer(Chem.Conformer(1))
    with pytest.raises(ValueError, match='outside the generator contract'):
        OctreeDataset([other], model, training=False).collate([0])


def test_larger_octree_retains_all_atoms_at_fixed_resolution():
    from types import SimpleNamespace
    # A long linear carbon skeleton, deliberately longer than the old box.
    mol = Chem.MolFromSmiles('C' * 16)
    conf = Chem.Conformer(16)
    for i in range(16):
        conf.SetAtomPosition(i, [i * 1.5, 0., 0.])
    mol.AddConformer(conf)
    args = build_default_uni3dar_args(1)
    args.random_rotation_prob = 0
    model = SimpleNamespace(args=args, dictionary=AtomFeatDict(args))
    with pytest.raises(AssertionError, match='exceed'):
        OctreeDataset([mol], model, training=False).collate([0])
    args.merge_level = 7
    packed = OctreeDataset([mol], model, training=False).collate([0])
    atoms = (packed['decoder_level'] == 0) & packed['decoder_is_second_atom']
    coords = packed['decoder_phy_pos'][atoms].numpy()
    distance = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    expected = np.linalg.norm(conf.GetPositions()[:, None] - conf.GetPositions()[None, :], axis=-1)
    assert len(coords) == 16
    np.testing.assert_allclose(np.sort(distance.ravel()), np.sort(expected.ravel()), atol=.018)
    assert packed['decoder_level'].max() == 7


def test_expanding_allowed_elements_retains_dictionary_ids_and_changes_support():
    from orbit3d.generator.octree_backend import checkpoint_contract
    args = build_default_uni3dar_args(1)
    args.emb_dim=96;args.num_head=4;args.layer=2
    old=ConditionalUni3DAR(args,AtomFeatDict(args),2)
    import copy
    new_args=copy.copy(args)
    new_args.allow_atoms='H,C,N,O,F,B,Si,Se,As,Te'
    new=ConditionalUni3DAR(new_args,AtomFeatDict(new_args),2)
    assert old.dictionary.atom_type==new.dictionary.atom_type
    new.load_state_dict(old.state_dict(),strict=True)
    assert not old.allowed_atom_mask(torch.device('cpu'))[old.dictionary['B']]
    assert new.allowed_atom_mask(torch.device('cpu'))[new.dictionary['B']]
    assert checkpoint_contract(old,['homo','lumo'],['eV','eV'],[0,0],[1,1]) != checkpoint_contract(new,['homo','lumo'],['eV','eV'],[0,0],[1,1])


@pytest.mark.parametrize('state',[{'charge':1},{'multiplicity':3}])
def test_octree_pipeline_does_not_ignore_unsupported_electronic_state(state):
    from orbit3d.pipeline.octree_runner import run_octree_from_config
    with pytest.raises(ValueError,match='neutral closed-shell'):
        run_octree_from_config({'stage5_relax_config':state})


def test_zero_kl_objective_needs_no_reference_model_and_keeps_policy_gradients():
    torch.manual_seed(21)
    logits=torch.randn(5,4,requires_grad=True)
    labels=torch.tensor([0,1,2,3,0]);owner=torch.tensor([0,0,1,1,1])
    packet=(logits.log_softmax(-1),labels,owner)
    old=[(packet[0].detach(),labels,owner)]
    advantage=torch.tensor([1.,-.5])
    loss=categorical_policy_loss([packet],old,None,advantage,beta=0.)
    legacy_ref=[(torch.randn(5,4).log_softmax(-1),labels,owner)]
    reference_loss=categorical_policy_loss([packet],old,legacy_ref,advantage,beta=0.)
    torch.testing.assert_close(loss,reference_loss,atol=0,rtol=0)
    loss.backward()
    assert logits.grad.abs().sum()>0
    with pytest.raises(ValueError,match='reference policy'):
        categorical_policy_loss([packet],old,None,advantage,beta=.1)
