import copy,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from orbit3d.generator.octree_model import PointConditioner

spec=importlib.util.spec_from_file_location('null_audit',Path(__file__).parents[1]/'scripts/audit_shared_null_policy.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_changed_conditioner_preserves_off_but_changed_backbone_is_rejected(tmp_path,monkeypatch):
    torch.manual_seed(1);model=PointConditioner(16,2,2)
    weights={'interval_conditioner.'+k:v.clone() for k,v in model.state_dict().items()};weights['backbone.weight']=torch.ones(3,3)
    state=dict(model_state_dict=weights,backend='uni3dar_octree_v4',target_cols=['homo','lumo'],target_units=['eV','eV'],representation_config={'merge_level':7},mean=np.zeros(2),scale=np.ones(2))
    reference=tmp_path/'reference.pt';variant=tmp_path/'variant.pt';torch.save(state,reference);changed=copy.deepcopy(state)
    for k,v in changed['model_state_dict'].items():
        if k.startswith('interval_conditioner.'):v.normal_(0,.1)
    torch.save(changed,variant);mapping=tmp_path/'models.json';mapping.write_text(json.dumps({'variant':str(variant)}));output=tmp_path/'audit.json'
    monkeypatch.setattr(sys,'argv',['audit','--checkpoints',str(mapping),'--reference',str(reference),'--output',str(output)]);module.main()
    assert json.loads(output.read_text())['models']['variant']['null_modulations_identity']
    changed['model_state_dict']['backbone.weight']+=1;torch.save(changed,variant)
    with pytest.raises(AssertionError,match='Non-conditioner'):module.main()
