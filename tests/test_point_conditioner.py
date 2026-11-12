import torch
from orbit3d.generator.octree_model import PointConditioner


def test_only_active_values_affect_condition_and_gradients():
    torch.manual_seed(42)
    model=PointConditioner(16,2,2)
    y=torch.tensor([[.2,.7]],requires_grad=True)
    active=torch.tensor([[[True,True],[False,False]]])
    c=model(y,y,active)
    changed=y.detach().clone();changed[0,1]=1000
    torch.testing.assert_close(c,model(changed,changed,active))
    moved=y.detach().clone();moved[0,0]=1.2
    assert not torch.allclose(c,model(moved,moved,active))
    c.square().sum().backward()
    assert y.grad[0,0].abs()>0
    assert y.grad[0,1]==0
    assert torch.count_nonzero(model(y,y,torch.zeros_like(active)))==0


def test_point_distance_changes_inside_acceptance_window_and_ignores_inactive():
    from orbit3d.constraints import TargetPoints
    points=TargetPoints.parse({'homo':-7.},{'homo':.25},['homo','lumo'])
    assert points.intervals().hit([-7.1,99.])
    assert points.distance([-7.1,99.])>points.distance([-7.,1000.])
    lo,hi,mask=points.arrays([-8.,0.],[2.,1.])
    assert lo.tolist()==[.5,0.]
    assert (lo==hi).all() and mask.tolist()==[[True,True],[False,False]]


def test_alignment_snapshots_do_not_modify_frozen_sft_directory(tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from orbit3d.pipeline import octree_runner as runner
    from orbit3d.generator import octree_alignment as alignment
    from orbit3d.generator.octree_model import ConditionalUni3DAR,build_default_uni3dar_args,AtomFeatDict
    args=build_default_uni3dar_args(1);args.emb_dim=32;args.num_head=4;args.layer=1;args.condition_version=4
    model=ConditionalUni3DAR(args,AtomFeatDict(args),2)
    source=tmp_path/'frozen';source.mkdir();(source/'sft.pt').write_bytes(b'frozen initial weights')
    state={'target_cols':['homo','lumo'],'target_units':['eV','eV'],'mean':np.zeros(2),'scale':np.ones(2)}
    monkeypatch.setattr(runner,'load_octree_checkpoint',lambda *a,**k:(model,state))
    monkeypatch.setattr(runner,'load_evaluators',lambda *a,**k:(SimpleNamespace(set_atom_type_map=lambda _:None),object(),None))
    def align(*args,**kwargs):
        history=[{'step':50}]
        kwargs['checkpoint_callback'](50,model,history)
        return history
    monkeypatch.setattr(alignment,'align_octree',align)
    cfg={'mode':'align','device':'cpu','generator_config':{'grpo_steps':50,'replay_weight':0,'physics_weight':0},
         'mount_paths':{'checkpoint_load_path':str(source/'sft.pt'),'checkpoint_save_dir':str(tmp_path/'new_weights'),
                        'output_dir':str(tmp_path/'run'),'forcefield_path':str(source/'sft.pt'),
                        'property_checkpoint_dir':str(source)},
         'oracle_config':{'target_cols':['homo','lumo'],'target_units':['eV','eV']},
         'inference_config':{'target_point':{'homo':-7},'target_tolerance':{'homo':.25}}}
    runner.run_octree_from_config(cfg)
    assert sorted(p.name for p in source.iterdir())==['sft.pt']
    assert (tmp_path/'new_weights/octree_group_50.pt').is_file()
    assert (tmp_path/'new_weights/octree_aligned.pt').is_file()
