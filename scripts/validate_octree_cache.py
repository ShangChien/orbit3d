"""Compare full-prefix/layer outputs and benchmark explicit KV storage modes."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.constraints import TargetIntervals
from orbit3d.generator.octree_backend import load_octree_checkpoint, OctreeDataset, interval_arrays
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.vendor.uni3dar.models.uni3dar import DecoderInputFeat
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment


@torch.no_grad()
def compare_prefix(model, molecule, lower, upper, active, mode, chunk):
    packed = OctreeDataset([molecule], model, training=False).collate([0])
    packed = {k: v.to(lower.device) if torch.is_tensor(v) else v for k,v in packed.items()}
    inputs, targets, _ = model.get_feat_from_dataloader(packed)
    cu = packed['decoder_input_len'].to(lower.device, dtype=torch.int32)
    length = int(packed['decoder_input_max_len'].item())
    layer_full = [[] for _ in model.decoder.layers]
    handles = [layer.register_forward_hook(lambda m, a, output, i=i: layer_full[i].append(output.clone()))
               for i,layer in enumerate(model.decoder.layers)]
    try:
        full, _ = model.forward_conditional_sequence(inputs, cu, length, lower, upper, active)
    finally:
        for handle in handles: handle.remove()
    c = model.interval_conditioner(lower, upper, active).unsqueeze(1)
    caches = [[{'step':1,'mode':mode} for _ in model.decoder.layers] for _ in range(model.args.recycle)]
    chunks=[]; layer_errors=np.zeros((model.args.recycle, len(model.decoder.layers)))
    for start in range(0, length, chunk):
        end=min(length,start+chunk)
        part=DecoderInputFeat(**{key:value[start:end].unsqueeze(0) for key,value in vars(inputs).items()})
        hidden=model.input_embedding(part)
        if start==0:
            hidden[:,0] += model.interval_conditioner.forward_root_prefix(c[:,0]).to(hidden.dtype)
        model.rope.calc_and_cache(part.phy_pos/model.args.xyz_resolution)
        for recycle in range(model.args.recycle):
            for index,layer in enumerate(model.decoder.layers):
                hidden=layer(hidden,None,None,caches[recycle][index])
                error=(hidden[0]-layer_full[index][recycle][start:end]).abs().max().item()
                layer_errors[recycle,index]=max(layer_errors[recycle,index],error)
                hidden=model.interval_conditioner.modulate_layer(index,hidden,c)
        hidden=model.interval_conditioner.modulate_out(model.norm(hidden),c)
        chunks.append(hidden[0])
    cached=torch.cat(chunks)
    tree_error=(model.tree_heads[0](full)-model.tree_heads[0](cached)).abs().max().item()
    mask=targets.is_gen_atom;truth=[v.long() for v in targets.to_list(mask)]
    heads_full=model.atom_heads[0](full[mask],truth);heads_cached=model.atom_heads[0](cached[mask],truth)
    atom_errors=[float((a-b).abs().max()) for a,b in zip(heads_full,heads_cached)]
    return {'tokens':length,'cache_mode':mode,'chunk_size':chunk,'layer_hidden_max_abs_diff':layer_errors.tolist(),
            'output_hidden_max_abs_diff':float((full-cached).abs().max()),
            'tree_logits_max_abs_diff':tree_error,'atom_xyz_logits_max_abs_diff':atom_errors,
            'cached_hidden_sha256':hashlib.sha256(cached.cpu().numpy().tobytes()).hexdigest(),
            'cached_atom_xyz_logits_sha256':[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in heads_cached]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sdf',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--records',type=int,default=2);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();freeze_run(a.output,__file__);record_input_hashes(a.output,{'checkpoint':a.checkpoint,'sdf':a.sdf,'config':a.config})
    lock_deterministic_environment(42);torch.set_num_threads(4)
    cfg=json.loads(a.config.read_text());model,state=load_octree_checkpoint(a.checkpoint,'cuda:0',a.batch_size);model.eval()
    bounds=TargetIntervals.parse(cfg['inference_config']['target_interval'],state['target_cols'])
    lower,upper,mask=interval_arrays(bounds,state['mean'],state['scale'],model.condition_version)
    tensors=[torch.tensor(x,device='cuda:0').unsqueeze(0) for x in [lower,upper,mask]]
    records=[]
    for index,molecule in enumerate(Chem.SDMolSupplier(str(a.sdf),removeHs=False)):
        if index>=a.records:break
        for mode in ['concat','buffer']:
            for chunk in [1,2,129]:
                row=compare_prefix(model,molecule,*tensors,mode,chunk);row['molecule']=index
                records.append(row);print('prefix',index,mode,chunk,row['output_hidden_max_abs_diff'],flush=True)
    (a.output/'prefix.json').write_text(json.dumps(records,indent=2)+'\n')
    # Explicit numerical threshold, distinct from exact trajectory comparisons.
    prefix_absolute_pass = all(max([r['tree_logits_max_abs_diff'],*r['atom_xyz_logits_max_abs_diff']])<2e-4 for r in records)
    storage_equal = True
    for row in records:
        if row['cache_mode'] == 'buffer':
            reference = next(x for x in records if x['cache_mode']=='concat' and x['molecule']==row['molecule'] and x['chunk_size']==row['chunk_size'])
            storage_equal &= (row['cached_hidden_sha256']==reference['cached_hidden_sha256'] and
                              row['cached_atom_xyz_logits_sha256']==reference['cached_atom_xyz_logits_sha256'])
    # The reference may already exceed an absolute full-prefix/logit tolerance.
    # Keep that failure separate from the exact buffer-versus-concat comparison.
    assert storage_equal, 'KV storage changes fixed-prefix hidden states or logits'
    low,high,active=[x.expand(a.batch_size,*x.shape[1:]) for x in tensors]
    seeds=[candidate_seed(2026,i) for i in range(a.batch_size)]
    @torch.no_grad()
    def sample(mode,guidance):
        return model.generate_conditional_trajectories(low,high,active,cfg_scale=guidance,
                evaluate_self_consistency=False,row_seeds=seeds,electron_parity=0,cache_mode=mode)
    comparisons=[]
    for guidance in [0.,1.,1.5]:
        reference=sample('concat',guidance);actual=sample('buffer',guidance)
        tokens=all(torch.equal(x['decoder_feat'],y['decoder_feat']) for x,y in zip(reference,actual))
        positions=all(np.array_equal(x['positions_np'],y['positions_np']) for x,y in zip(reference,actual))
        scores=max(float((x['scores']-y['scores']).abs().max()) for x,y in zip(reference,actual))
        comparisons.append({'cfg_scale':guidance,'trajectories':len(actual),'tokens_exact':tokens,'coordinates_exact':positions,'score_max_abs_diff':scores})
        (a.output/'generation.json').write_text(json.dumps(comparisons,indent=2)+'\n')
        print('generation',comparisons[-1],flush=True)
        assert len(actual)==len(reference)==a.batch_size and tokens and positions and scores<1e-4
    timings={'concat':[],'buffer':[]};memory={'concat':[],'buffer':[]}
    for repeat in range(a.repeats):
        for mode in (['concat','buffer'] if repeat%2==0 else ['buffer','concat']):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            sample(mode,1.);torch.cuda.synchronize()
            timings[mode].append(time.perf_counter()-start);memory[mode].append(torch.cuda.max_memory_allocated())
            print('timing',mode,timings[mode][-1],flush=True)
    report={'prefix_absolute_logit_threshold':2e-4,'prefix_absolute_logit_passed':prefix_absolute_pass,
            'buffer_vs_concat_fixed_prefix_bitwise_equal':storage_equal,'prefix':records,'generation':comparisons,'seconds':timings,'median_seconds':{k:statistics.median(v) for k,v in timings.items()},'peak_allocated_bytes':memory,
            'scope':'FP32 on shared A800; alternated mode order after separate warmup/comparison runs; numerical equivalence is not cross-device bitwise identity',
            'gpu':torch.cuda.get_device_name(),'default_cache_mode':'concat'}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('cache validation complete',flush=True)


if __name__=='__main__':main()
