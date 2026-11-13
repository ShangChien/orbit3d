"""Prove a shared off policy for conditioner-only updates before reusing one off pool."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from orbit3d.generator.octree_model import PointConditioner


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);ref=torch.load(a.reference,map_location='cpu',weights_only=False,mmap=True);prefix='interval_conditioner.';base=ref['model_state_dict'];keys=[k for k in base if not k.startswith(prefix)];records={}
    for name,value in json.loads(a.checkpoints.read_text()).items():
        path=Path(value);state=torch.load(path,map_location='cpu',weights_only=False,mmap=True);weights=state['model_state_dict'];assert set(base)==set(weights)
        assert state['backend']==ref['backend']=='uni3dar_octree_v4'
        for k in ['target_cols','target_units','representation_config']:assert state[k]==ref[k]
        for k in ['mean','scale']:np.testing.assert_array_equal(state[k],ref[k])
        assert all(base[k].dtype==weights[k].dtype and torch.equal(base[k],weights[k]) for k in keys),'Non-conditioner parameters changed'
        conditioned={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
        width=conditioned['root_proj.weight'].shape[0];layers=sum(k.startswith('layer_mods.') and k.endswith('.weight') for k in conditioned)
        model=PointConditioner(width,len(state['target_cols']),layers).eval();model.load_state_dict(conditioned,strict=True)
        n=len(state['target_cols']);values=torch.arange(3*n,dtype=torch.float32).reshape(3,n)-n;mask=torch.zeros(3,n,2,dtype=torch.bool);hidden=torch.ones(3,width)
        with torch.no_grad():
            zero=model(values,values,mask);assert torch.count_nonzero(zero)==0 and torch.count_nonzero(model.forward_root_prefix(zero))==0
            assert torch.equal(model.modulate_out(hidden,zero),hidden)
            assert all(torch.equal(model.modulate_layer(i,hidden,zero),hidden) for i in range(layers))
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        records[name]={'checkpoint_sha256':digest,'unchanged_non_conditioner_tensors':len(keys),'null_prefix_zero':True,'null_modulations_identity':True}
        del state,weights,model,conditioned
    report={'scope':'All v4 non-conditioner tensors and semantic contracts match the reference. Loaded trained conditioners yield a zero prefix and identity modulation under all-off masks. Evaluation uses eval mode, identical sampling settings and candidate streams; one shared off pool is therefore reused, not counted as independent runs.','reference':str(a.reference),'models':records,'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(list(records))

if __name__=='__main__':main()
