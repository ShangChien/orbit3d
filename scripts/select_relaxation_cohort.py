"""Freeze four domain-evaluable P50 proposals per policy/property/inference seed."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from loom.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--homo',type=Path,required=True)
    p.add_argument('--lumo',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze_run(a.output,__file__)
    manifest=json.loads(a.manifest.read_text());inputs={'source_manifest':a.manifest};samples=[];arrays={}
    for target in ['homo','lumo']:
        campaign=getattr(a,target)
        for policy in ['frozen','grpo200']:
            for seed in [2026,2027,2028]:
                root=campaign/f'eval_{policy}_p50_s{seed}'
                source=json.loads((root/'raw/summary.json').read_text())
                assert source['target_point'][target]==manifest['single_target_points'][target]['50']
                candidates=sorted([r for r in source['records'] if r['valid'] and r['evaluated'] and r.get('domain') and r['domain']['in_domain']],key=lambda r:r['candidate_id'])
                if len(candidates)<4:raise ValueError(f'Fewer than four eligible proposals in {root}')
                symbols=json.loads((root/'run/raw_symbols.json').read_text())
                for name in ['raw_symbols.json','raw_trajectories.npz','input_manifest.json']:
                    inputs[f'{target}/{policy}/{seed}/{name}']=root/'run'/name
                inputs[f'{target}/{policy}/{seed}/evaluation']=root/'raw/summary.json'
                with np.load(root/'run/raw_trajectories.npz') as coords:
                    for row in candidates[:4]:
                        i=row['candidate_id'];key=f'coordinates_{len(samples)}';arrays[key]=coords[f'coordinates_{i}']
                        samples.append({'id':len(samples),'target':target,'policy':policy,'inference_seed':seed,
                            'candidate_id':i,'source_pool':str(root),'coordinates_key':key,'symbols':symbols[i],
                            'smiles':row['smiles'],'reference_prediction_eV':row['prediction'],'reference_rms_force_eVA':row['rms_force'],
                            'target_point':{target:manifest['single_target_points'][target]['50']},'target_tolerance':{target:.25},
                            'token_sha256':hashlib.sha256(coords[f'tokens_{i}'].tobytes()).hexdigest()})
    np.savez_compressed(a.output/'initial_coordinates.npz',**arrays)
    (a.output/'cohort.json').write_text(json.dumps({'selection':'First four candidate IDs with successful original DPA/domain evaluation per stratum; no property-score or independent-model selection',
        'expected_samples':48,'samples':samples,'elements':manifest['elements'],'target_names':['homo','lumo'],
        'geometry':'Original raw octree coordinates; recentered once by the normal trajectory decoder'},indent=2)+'\n')
    record_input_hashes(a.output,inputs);print('Frozen starting points:',len(samples))


if __name__=='__main__':main()
