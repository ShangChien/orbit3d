"""Compare inference batch sizes on a fixed point task and candidate random streams."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from orbit3d.constraints import TargetPoints
from orbit3d.generator.octree_backend import load_octree_checkpoint
from orbit3d.generator.random_streams import candidate_seed
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-sizes',nargs='+',type=int,default=[8,16,32,64])
    p.add_argument('--candidates',type=int,default=64)
    p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args();cfg=json.loads(a.config.read_text());inf=cfg['inference_config']
    freeze_run(a.output,__file__)
    checkpoint=cfg['mount_paths']['checkpoint_load_path']
    record_input_hashes(a.output,{'checkpoint':checkpoint,'config':a.config})
    lock_deterministic_environment(42);torch.set_num_threads(4);torch.cuda.set_device(a.device)
    report={'config':str(a.config),'device':a.device,'candidates':a.candidates,
            'scope':'FP32 sampling and decoding only; fixed candidate streams; no oracle or model selection',
            'batches':[]};reference=None
    for batch in a.batch_sizes:
        model,state=load_octree_checkpoint(checkpoint,a.device,batch)
        point=TargetPoints.parse(inf['target_point'],inf['target_tolerance'],state['target_cols'])
        lower,upper,active=point.arrays(state['mean'],state['scale'])
        def sample(count,offset):
            low=torch.tensor(lower,device=a.device).repeat(count,1)
            high=torch.tensor(upper,device=a.device).repeat(count,1)
            mask=torch.tensor(active,device=a.device).unsqueeze(0).expand(count,*active.shape)
            with torch.no_grad():
                return model.generate_conditional_trajectories(low,high,mask,
                    cfg_scale=inf.get('cfg_scale',1.),tree_temp=inf['tree_temperature'],
                    atom_temp=inf['atom_temperature'],xyz_temp=inf['xyz_temperature'],
                    electron_parity=inf.get('electron_parity'),evaluate_self_consistency=True,
                    cache_mode=inf.get('cache_mode','concat'),
                    row_seeds=[candidate_seed(inf['base_seed'],offset+i) for i in range(count)])
        sample(min(batch,a.candidates),0)
        seconds=[];first=None
        for repeat in range(a.repeats):
            torch.cuda.synchronize();start=time.perf_counter();rows=[]
            for begin in range(0,a.candidates,batch):rows.extend(sample(min(batch,a.candidates-begin),begin))
            torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
            arrays={key:val for i,r in enumerate(rows) for key,val in
                    [(f'tokens_{i}',r['decoder_feat'].cpu().numpy()),(f'positions_{i}',r['positions_np'])]}
            if first is None:first=arrays
            else:assert all(np.array_equal(first[k],v) for k,v in arrays.items()),'Same-batch replay changed'
        np.savez_compressed(a.output/f'batch_{batch}.npz',**first)
        if reference is None:reference=first
        report['batches'].append({'batch_size':batch,'seconds':seconds,'median_seconds':float(np.median(seconds)),
            'bitwise_identical_to_first_batch':sum(all(np.array_equal(first[f'{key}_{i}'],reference[f'{key}_{i}'])
                for key in ['tokens','positions']) for i in range(a.candidates)),
            'same_batch_repeats_bitwise_equal':True})
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(report['batches'][-1],flush=True)
        del model,state


if __name__=='__main__':main()
