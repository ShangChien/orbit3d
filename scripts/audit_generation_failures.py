"""Recheck saved raw coordinates; preserve acceptance bits and expose rejection causes."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.generator.spatial_box import BoxSpec3D
from orbit3d.utils.run_evidence import freeze_run
p=argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
freeze_run(a.output,__file__)
config=json.loads((a.run/'run_config.json').read_text())
box=BoxSpec3D(**config['spatial_box']) if config.get('spatial_box') else None
symbols=json.loads((a.run/'raw_symbols.json').read_text())
old=pd.read_csv(a.run/'candidates.csv')
records=[]
with np.load(a.run/'raw_trajectories.npz') as arrays:
    for i, atoms in enumerate(symbols):
        record={'candidate_id':i,'atom_count':len(atoms),'element_counts':dict(Counter(atoms))}
        try:
            inst=trajectory_to_instance({'atom_symbols':atoms,'positions_np':arrays[f'coordinates_{i}']},box)
            record.update(valid=True,smiles=inst.smiles,category='valid')
        except ValueError as exc:
            record.update(valid=False,reason=str(exc),category=str(exc).split(':',1)[0])
        assert record['valid']==bool(old.iloc[i].raw_valid), f'Acceptance changed at candidate {i}'
        records.append(record)
report={'attempted':len(records),'acceptance_bits_unchanged':True,
        'categories':dict(Counter(r['category'] for r in records)),'records':records}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
