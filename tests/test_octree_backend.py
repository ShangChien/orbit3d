import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

pytest.importorskip("numba")
pytest.importorskip("lmdb")
from loom.generator.octree_model import ConditionalUni3DAR, build_default_uni3dar_args, AtomFeatDict
from loom.generator.octree_backend import OctreeDataset
from loom.generator.octree_alignment import categorical_policy_loss
from loom.vendor.uni3dar.models.rope import RoPEnD


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
    from loom.generator.octree_model import DynamicIntervalConditioner
    from loom.generator.octree_backend import interval_arrays
    from loom.constraints import TargetIntervals

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
    from loom.generator.octree_backend import checkpoint_contract

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
    from loom.generator.octree_model import DynamicIntervalConditioner

    smooth = DynamicIntervalConditioner(emb_dim=32, num_props=2, num_layers=1,
                                        bound_masks=True, smooth_frequencies=True)
    legacy = DynamicIntervalConditioner(emb_dim=32, num_props=2, num_layers=1, bound_masks=True)
    values = torch.tensor([[-1.2, .3, -1.5, -.9], [.5, .2, .3, .7]])
    small = 1e-5
    assert (smooth._fourier_encode(values+small)-smooth._fourier_encode(values)).abs().max() < .001
    assert (legacy._fourier_encode(values+small)-legacy._fourier_encode(values)).abs().max() > .1


def test_cfg_guides_each_nonlinear_head_with_the_same_sampled_prefix():
    from loom.vendor.uni3dar.models.autoregressive_prediction_head import AutoRegressivePredictionHead
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
    from loom.generator.electronic_constraints import mask_terminal_element_logits
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
    from loom.generator.octree_backend import checkpoint_contract
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
    from loom.pipeline.octree_runner import run_octree_from_config
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


def test_failed_groups_without_auxiliary_objectives_leave_policy_unchanged(monkeypatch):
    import copy
    from types import SimpleNamespace
    from loom.constraints import TargetIntervals
    from loom.generator.octree_alignment import align_octree

    class FailedPolicy(torch.nn.Module):
        condition_version = 3
        args = SimpleNamespace(allow_atoms='H,C,N,O,F')

        def __init__(self):
            super().__init__()
            self.interval_conditioner = torch.nn.Linear(2, 2)

        def generate_conditional_trajectories(self, *args, **kwargs):
            return [{'decoder_feat': torch.zeros(1, 6), 'atom_symbols': ['C'],
                     'positions_np': np.zeros((1, 3))} for _ in range(4)]

        def trajectory_distributions(self, *args, **kwargs):
            logits = self.interval_conditioner(torch.ones(4, 2))
            return [(logits.log_softmax(-1), torch.zeros(4, dtype=torch.long), torch.arange(4))]

    def invalid(*args, **kwargs):
        raise ValueError('invalid generated graph')

    monkeypatch.setattr('loom.generator.octree_alignment.trajectory_to_instance', invalid)
    adam = torch.optim.AdamW
    def optimizer_with_existing_momentum(parameters, **kwargs):
        optimizer = adam(parameters, **kwargs)
        for group in optimizer.param_groups:
            for parameter in group['params']:
                optimizer.state[parameter] = {'step': torch.tensor(1.),
                    'exp_avg': torch.ones_like(parameter) * .1,
                    'exp_avg_sq': torch.ones_like(parameter) * .01}
        return optimizer
    monkeypatch.setattr(torch.optim, 'AdamW', optimizer_with_existing_momentum)
    model = FailedPolicy()
    original = copy.deepcopy(model.state_dict())
    history = align_octree(model, [], None, np.zeros(1), np.ones(1),
                           TargetIntervals.parse((0, 1), ['target']), None, None,
                           steps=2, group_size=4, kl_beta=0, replay_weight=0)
    assert all(row['optimizer_updates'] == 0 and row['valid'] == 0 for row in history)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())


def test_element_supervision_remains_pending_after_structural_adaptation():
    from loom.generator.octree_backend import required_condition_elements

    state = {'representation_migration': {'old_elements': ['H', 'C'], 'new_elements': ['H', 'C', 'B'],
                                        'requires_adaptation': False},
             'condition_migration': {'requires_condition_adaptation': True}}
    assert required_condition_elements(state) == {'B'}
    # Expanding again must carry the earlier B label requirement into the new contract.
    pending = required_condition_elements(state)
    state['condition_migration']['required_elements'] = sorted(pending)
    state['representation_migration'] = {'old_elements': ['H', 'C', 'B'],
                                        'new_elements': ['H', 'C', 'B', 'Si'], 'requires_adaptation': True}
    assert required_condition_elements(state) == {'B', 'Si'}


@pytest.mark.parametrize('weighted', [False, True])
def test_per_molecule_training_scores_match_separate_unequal_size_molecules(weighted):
    args = build_default_uni3dar_args(2)
    args.emb_dim, args.num_head, args.layer = 96, 4, 2
    model = ConditionalUni3DAR(args, AtomFeatDict(args), 2).eval()
    with torch.no_grad():
        for parameter in list(model.tree_heads.parameters()) + list(model.atom_heads.parameters()):
            if parameter.ndim > 1:
                parameter.normal_(0, .02)
    molecules = []
    for smi in ['CO', 'c1ccccc1']:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        molecules.append(mol)
    dataset = OctreeDataset(molecules, model, training=False)
    weights = torch.ones(model.num_atom) if weighted else None
    if weighted:
        weights[model.dictionary['O']] = 10.
    target = torch.tensor([[0., 1.], [1., 0.]])
    _, stats = model.compute_sft_and_understanding_loss(
        dataset.collate([0, 1]), target, target-.25, target+.25, torch.ones(2, 2, dtype=torch.bool),
        understanding_weight=0, return_per_molecule=True, atom_class_weights=weights)
    scores = stats['per_molecule_generation_loss']
    for i in range(2):
        y = target[i:i+1]
        separate, _ = model.compute_sft_and_understanding_loss(
            dataset.collate([i]), y, y-.25, y+.25, torch.ones(1, 2, dtype=torch.bool), understanding_weight=0, atom_class_weights=weights)
        torch.testing.assert_close(scores[i], separate, atol=1e-5, rtol=1e-5)
    scores.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in model.interval_conditioner.parameters())


def test_incompatible_intervals_and_margin_use_only_valid_negative_pairs():
    from loom.generator.condition_contrast import incompatible_condition_indices, condition_margin_loss
    targets = torch.tensor([[0., 0.], [.1, .1], [.6, -.1]])
    indices, eligible = incompatible_condition_indices(targets, halfwidth=.25)
    assert eligible.all()
    assert ((targets-targets[indices]).abs() > .25).any(-1).all()
    positive = torch.tensor([1., 1.], requires_grad=True)
    negative = torch.tensor([.99, 1.1], requires_grad=True)
    loss = condition_margin_loss(positive, negative, torch.ones(2, dtype=torch.bool), margin=.02)
    assert loss.item() == pytest.approx(.015)
    loss.backward()
    torch.testing.assert_close(positive.grad, torch.tensor([.5, 0.]))
    torch.testing.assert_close(negative.grad, torch.tensor([-.5, 0.]))
    _, eligible = incompatible_condition_indices(torch.zeros(2, 2))
    assert not eligible.any()


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_sampling_temperature_rejects_invalid_values(value):
    args = build_default_uni3dar_args(1)
    args.emb_dim, args.num_head, args.layer = 96, 4, 2
    model = ConditionalUni3DAR(args, AtomFeatDict(args), 2)
    y = torch.zeros(1, 2)
    with pytest.raises(ValueError, match='temperature'):
        model.generate_conditional_trajectories(y, y, torch.ones_like(y, dtype=torch.bool), atom_temp=value)


def test_element_loss_weights_come_only_from_the_supplied_training_atoms():
    from loom.generator.supervised_weights import inverse_sqrt_element_weights
    dictionary = AtomFeatDict(build_default_uni3dar_args(1))
    weights, record = inverse_sqrt_element_weights([Chem.MolFromSmiles('CCCO')], dictionary)
    assert record['training_atom_counts'] == {'C': 3, 'O': 1}
    assert weights[dictionary['C']].item() == 1.
    assert weights[dictionary['O']].item() == pytest.approx(3**.5)
    assert weights[dictionary['[NULL]']].item() == 1.


def test_target_expansion_preserves_old_conditions_and_readout_when_new_targets_are_inactive():
    from loom.generator.octree_model import DynamicIntervalConditioner, Suffix3DPropertyHead
    from loom.generator.target_migration import expand_interval_conditioner, expand_property_readout
    torch.manual_seed(42)
    before = DynamicIntervalConditioner(emb_dim=32, num_props=2, num_layers=2, bound_masks=True, smooth_frequencies=True)
    after = DynamicIntervalConditioner(emb_dim=32, num_props=4, num_layers=2, bound_masks=True, smooth_frequencies=True)
    old_names, names = ['homo','lumo'], ['s0s1','lumo','s0t1','homo']
    expand_interval_conditioner(before, after, old_names, names)
    lower = torch.randn(3, 2);upper = lower+.7
    old_mask = torch.ones(3, 2, 2, dtype=torch.bool)
    new_lower, new_upper = torch.zeros(3, 4), torch.zeros(3, 4)
    new_mask = torch.zeros(3, 4, 2, dtype=torch.bool)
    for i, name in enumerate(old_names):
        j = names.index(name)
        new_lower[:,j], new_upper[:,j], new_mask[:,j] = lower[:,i], upper[:,i], True
    torch.testing.assert_close(before(lower, upper, old_mask), after(new_lower, new_upper, new_mask), atol=2e-6, rtol=2e-6)
    old_head, new_head = Suffix3DPropertyHead(32, 2), Suffix3DPropertyHead(32, 4)
    expand_property_readout(old_head, new_head, old_names, names)
    features = torch.randn(3, 96)
    torch.testing.assert_close(old_head.net(features), new_head.net(features)[:,[3,1]], atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(new_head.net(features)[:,[0,2]]) == 0


@pytest.mark.parametrize("version,continued,timed", [(3,False,False),(4,True,False),(4,True,True)])
def test_labelled_sft_runs_without_any_oracle_or_forcefield(tmp_path, monkeypatch, version, continued, timed):
    import json
    from loom.pipeline import octree_runner as runner
    args = build_default_uni3dar_args(2)
    args.emb_dim, args.num_head, args.layer = 96, 4, 2
    args.condition_version = version
    model = ConditionalUni3DAR(args, AtomFeatDict(args), 2)
    monkeypatch.setattr(runner, 'load_pretrained', lambda *a, **kw: model)
    def unavailable(*a, **kw):
        raise AssertionError('SDF-labelled SFT must not load an evaluator')
    monkeypatch.setattr(runner, 'load_evaluators', unavailable)
    source = tmp_path/'initial.pt'; source.write_bytes(b'fixture pretrained source')
    sdf = tmp_path/'train.sdf'
    with Chem.SDWriter(str(sdf)) as writer:
        for i, smi in enumerate(['CC', 'CO']):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(mol, randomSeed=42)
            mol.SetProp('homo', str(-6.+i)); mol.SetProp('lumo', str(-1.+i))
            writer.write(mol)
    cfg = {'device': 'cpu', 'mode': 'train', 'generator_config': {'sft_steps': 2, 'batch_size': 2},
           'oracle_config': {'target_cols': ['homo','lumo'], 'target_units': ['eV','eV']},
           'mount_paths': {'output_dir': str(tmp_path/'run'), 'checkpoint_save_dir': str(tmp_path/'save'),
                           'generator_pretrained_path': str(source), 'train_data_path': str(sdf)}}
    if continued:
        initial = {'target_cols':['homo','lumo'],'target_units':['eV','eV'],
                   'mean':np.array([-5.5,-.5],dtype=np.float64),'scale':np.array([.5,.5],dtype=np.float64)}
        monkeypatch.setattr(runner, 'load_octree_checkpoint', lambda *a, **kw: (model,initial))
        cfg['mount_paths'].pop('generator_pretrained_path')
        cfg['mount_paths']['checkpoint_load_path'] = str(source)
        cfg['generator_config']['active_targets'] = ['homo']
    if timed:
        cfg['generator_config']['sft_wall_seconds']=1e-12
        cfg['generator_config']['lr_schedule']='constant'
    result = runner.run_octree_from_config(cfg)
    state = torch.load(result['checkpoint'], weights_only=False)
    assert len(state['history']) == (1 if timed else 2) and state['label_source'] == 'sdf'
    assert state['optimizer_updates']==len(state['history'])
    assert state['requested_sft_steps']==2 and state['training_wall_seconds']>0
    manifest = json.loads((tmp_path/'run/input_manifest.json').read_text())
    assert set(manifest) == {'generator', 'adaptation_data'}
