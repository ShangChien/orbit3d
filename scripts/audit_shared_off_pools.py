"""Verify exact off-policy trajectory reuse before computing a shared E12 baseline."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ARMS=['point_frozen','terminal_sft','terminal_structure','terminal_physical','terminal_property']

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    hashes={};inputs={};reference=None;unique=set()
    for seed in [2081,2082]:
        hashes[str(seed)]={}
        for arm in ARMS:
            lane=0 if arm in ['terminal_sft','terminal_property'] else 1
            folder=a.root/f'evaluation_lane{lane}/eval_homo_seed42/eval_{arm}_p50_wNone_s{seed}_off/run'
            cfg=json.loads((folder/'run_config.json').read_text());symbols=json.loads((folder/'raw_symbols.json').read_text())
            assert len(symbols)==64 and cfg['inference_config']['condition_control']=='off'
            spec={'oracle':cfg['oracle_config'],'relaxation':cfg.get('stage5_relax_config',{}),'weights':{k:cfg['mount_paths'][k] for k in ['forcefield_path','property_checkpoint_dir']}}
            if reference is None:reference=spec
            assert spec==reference
            h=hashlib.sha256()
            with np.load(folder/'raw_trajectories.npz') as arrays:
                for i,atoms in enumerate(symbols):
                    xyz=arrays[f'coordinates_{i}'];data=json.dumps([atoms,xyz.shape,xyz.dtype.str]).encode()+xyz.tobytes()
                    h.update(data)
                    if arm=='point_frozen':unique.add(hashlib.sha256(data).hexdigest())
            hashes[str(seed)][arm]=h.hexdigest()
            for name in ['run_config.json','raw_symbols.json','raw_trajectories.npz']:
                inputs[str(folder/name)]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
        assert len(set(hashes[str(seed)].values()))==1
    report={'scope':'Five conditioner-only arms have identical off trajectories and evaluator settings. Use one shared 128-slot baseline, not 640 independent attempts. Coincident outputs within those 128 slots remain attempts and are reported separately.','archived_run_slots':640,'shared_proposal_slots':128,'distinct_coordinate_records':len(unique),'geometry_hashes':hashes,'evaluator_specification':reference,'source_sha256':inputs}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print('Verified shared off baseline:',128,'slots,',len(unique),'distinct coordinate records',flush=True)

if __name__=='__main__':main()
