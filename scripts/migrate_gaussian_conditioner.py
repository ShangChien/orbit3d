"""Migrate a trained point conditioner without changing its initial predictions."""
import argparse
import hashlib
from pathlib import Path
import torch
from loom.generator.octree_backend import build_octree, checkpoint_contract


def expand_point_weights(weights):
    updated = dict(weights)
    for key, value in weights.items():
        if key.startswith('interval_conditioner.prop_encoders.') and key.endswith('.0.weight'):
            assert value.shape[1] == 1
            updated[key] = torch.cat([value, torch.zeros_like(value)], 1)
    return updated


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert state['backend']=='uni3dar_octree_v4'
    model=build_octree('cpu',len(state['target_cols']),8,5,state['representation_config'])
    model.load_state_dict(expand_point_weights(state['model_state_dict']),strict=True)
    state.update(model_state_dict=model.state_dict(),backend='uni3dar_octree_v5',
                 condition_migration={'requires_condition_adaptation':True,'source_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                    'semantics':'Mean and log normalized preference standard deviation; new width weights zero-initialized'})
    state['contract']=checkpoint_contract(model,state['target_cols'],state['target_units'],state['mean'],state['scale'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as stream:torch.save(state,stream)


if __name__=='__main__':main()
