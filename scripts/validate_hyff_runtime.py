"""Real HyFF checkpoint: reference/native routes, electronic states and raw forces."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from orbit3d.physics.hyff_forcefield import HyFFPotentialGuidance
from orbit3d.utils.run_evidence import freeze_run
p=argparse.ArgumentParser()
p.add_argument('--weights', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--tier', default='presto')
a=p.parse_args()
freeze_run(a.output,__file__)
torch.set_num_threads(2)
ff=HyFFPotentialGuidance(tier=a.tier,weights=a.weights,atom_type_map=['C','H','N','O'],device='cuda:0',route='reference')
positions=[];types=[];owners=[];charges=[];multiplicities=[]
for i,(smiles,mult) in enumerate([('CC',1),('CO',1),('[NH4+]',1),('[CH3]',2)]):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m,randomSeed=42)==0
    positions.extend(m.GetConformer().GetPositions())
    types.extend(ff.atom_type_map.index(x.GetSymbol()) for x in m.GetAtoms())
    owners.extend([i]*m.GetNumAtoms())
    charges.append(Chem.GetFormalCharge(m));multiplicities.append(mult)
x=torch.tensor(np.array(positions),device='cuda:0',dtype=torch.float32)
t=torch.tensor(types,device='cuda:0');b=torch.tensor(owners,device='cuda:0')
kwargs=dict(node_batch_idx=b,clip_forces=False,zero_com_forces=False,charges=charges,multiplicities=multiplicities)
e_ref,f_ref=ff.compute_energy_and_forces(x,t,**kwargs)
coordinate=int(f_ref.abs().argmax());h=.002
plus,minus=x.clone(),x.clone();plus.reshape(-1)[coordinate]+=h;minus.reshape(-1)[coordinate]-=h
ep,_=ff.compute_energy_and_forces(plus,t,**kwargs)
em,_=ff.compute_energy_and_forces(minus,t,**kwargs)
fd=float(-(ep.sum()-em.sum())/(2*h));analytic=float(f_ref.reshape(-1)[coordinate])
assert abs(fd-analytic)<max(.02,.02*abs(analytic))
ff.route='native'
e_native,f_native=ff.compute_energy_and_forces(x,t,**kwargs)
report={'tier':a.tier,'weights_sha256':hashlib.sha256(Path(ff.model_path).read_bytes()).hexdigest(),
        'charges':charges,'multiplicities':multiplicities,'atom_counts':torch.bincount(b).tolist(),
        'reference_energy_eV':e_ref.cpu().tolist(),'native_energy_eV':e_native.cpu().tolist(),
        'max_energy_error_eV':float(abs(e_ref-e_native).max()),
        'max_raw_force_error_eV_A':float(abs(f_ref-f_native).max()),
        'finite_difference_force_eV_A':fd,'analytic_force_eV_A':analytic,
        'force_difference_tolerance_eV_A':.001,'energy_difference_tolerance_eV':.001,
        'scope':'Checkpoint execution only; not broad-element generation or property accuracy'}
report['passed']=report['max_energy_error_eV']<.001 and report['max_raw_force_error_eV_A']<.001
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
ff.close()
assert report['passed']
