"""Check real HyFF descriptor batching and coordinate gradients before training."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from loom.physics.factory import build_forcefield
from loom.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from loom.utils.run_evidence import freeze_run,record_input_hashes

p=argparse.ArgumentParser()
p.add_argument('--weights',required=True)
p.add_argument('--tier',default='presto')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
freeze_run(a.output,__file__)
record_input_hashes(a.output,{'forcefield':a.weights})
torch.manual_seed(42);torch.set_num_threads(2)
ff=build_forcefield('hyff',a.weights,['C','H','N','O'],'cuda:0',tier=a.tier)
molecules=[]
for smiles in ['CO','NN']:
 m=Chem.AddHs(Chem.MolFromSmiles(smiles));assert AllChem.EmbedMolecule(m,randomSeed=42)==0
 molecules.append(m)
x=torch.tensor(np.stack([m.GetConformer().GetPositions() for m in molecules]),device='cuda:0',dtype=torch.float32)
symbols=[[atom.GetSymbol() for atom in m.GetAtoms()] for m in molecules]
desc,energy,force=ff.extract_native_batch(x,symbols)
serial=[ff.extract_native_descriptor_and_forces(x[i],symbols[i]) for i in range(2)]
errors={name:float(abs(value-torch.stack([r[j] for r in serial])).max()) for j,(name,value) in enumerate([('descriptor',desc),('energy_eV',energy),('forces_eV_A',force)])}
assert energy.dtype==torch.float64
assert errors['descriptor']<1e-4 and errors['energy_eV']<1e-4 and errors['forces_eV_A']<1e-4
# This untrained head is only an autograd test, not a property predictor.
oracle=InternalDPA4PropertyOracle(ff,2,['homo','lumo'],2)
x.requires_grad_(True)
y,_=oracle.forward_differentiable_batch(x,symbols)
grad=torch.autograd.grad(y[0,0],x)[0]
assert torch.count_nonzero(grad[1])==0
coord=int(grad[0].abs().argmax());h=.002
plus,minus=x.detach().clone(),x.detach().clone()
plus[0].reshape(-1)[coord]+=h;minus[0].reshape(-1)[coord]-=h
yp=oracle.forward_differentiable_batch(plus,symbols)[0][0,0]
ym=oracle.forward_differentiable_batch(minus,symbols)[0][0,0]
finite_difference=float((yp-ym)/(2*h));analytic=float(grad[0].reshape(-1)[coord])
assert abs(finite_difference-analytic)<max(.002,.03*abs(analytic))
report={'tier':a.tier,'serial_batch_max_abs_errors':errors,'energy_dtype':str(energy.dtype),'finite_difference':finite_difference,'analytic_gradient':analytic,'cross_molecule_gradient_nonzero':int(torch.count_nonzero(grad[1])),'scope':'Descriptor and gradient contract only; random property head is not evaluated as a trained predictor','passed':True}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
