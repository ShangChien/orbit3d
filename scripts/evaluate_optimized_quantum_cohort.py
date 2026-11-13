"""Frozen A/B predictions on computed final geometries; graph failures remain failures."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np,torch
from rdkit import Chem
from profile_dual_oracle import load_models,predict
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--profile',default='profile');p.add_argument('--output',type=Path);a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2','3'];torch.set_num_threads(2)
r=a.root/'quantum_cohort/results';assert (r/'completed.json').exists();frozen=json.loads((a.root/a.profile/'models_frozen.json').read_text());ff,models=load_models(a.root,frozen['selected_unimol'])
ms=list(Chem.SDMolSupplier(str(a.root/'quantum_cohort/cohort.sdf'),removeHs=False));selected=[];indices=[];geometry_status=[]
for i,m in enumerate(ms):
 path=r/f'job{i:02d}'/'result.json'
 if not path.exists():continue
 d=json.loads(path.read_text())
 if 'optimized_singlepoint' not in d:continue
 coords=np.load(r/f'job{i:02d}'/'optimized_coords.npy');out=Chem.Mol(m);out.RemoveAllConformers();conf=Chem.Conformer(len(coords));conf.SetPositions(coords);out.AddConformer(conf);selected.append(out);indices.append(i);geometry_status.append(d['geometry'])
values={name:predict(model,selected)[0].tolist() for name,model in models.items()}
report={'indices':indices,'models':values,'scope':'Atomic-coordinate predictions at optimized geometries. Failed graph identities are explicitly excluded from design-success counts, not silently repaired.',
 'model_lock_sha256':hashlib.sha256((a.root/a.profile/'models_frozen.json').read_bytes()).hexdigest(),'cohort_sha256':hashlib.sha256((a.root/'quantum_cohort/cohort.sdf').read_bytes()).hexdigest(),'selected_unimol':frozen['selected_unimol']}
(a.output or a.root/'quantum_cohort/optimized_model_predictions.json').write_text(json.dumps(report,indent=2)+'\n');print('evaluated optimized geometries',len(indices),flush=True)
