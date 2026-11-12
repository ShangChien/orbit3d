"""Initialize point-conditioned OE62 adaptation from upstream structural weights."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.octree_backend import load_pretrained,build_octree,checkpoint_contract

p=argparse.ArgumentParser();p.add_argument('--pretrained',type=Path,required=True)
p.add_argument('--sdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.manual_seed(42)
ms=list(Chem.SDMolSupplier(str(a.sdf),removeHs=False));assert ms and all(m is not None for m in ms)
elements=sorted({atom.GetSymbol() for m in ms for atom in m.GetAtoms()})
names=['homo','lumo'];units=['eV','eV']
y=np.array([[float(m.GetProp(n)) for n in names] for m in ms]);mean=y.mean(0);scale=y.std(0)
old=load_pretrained(a.pretrained,'cpu',2,8,condition_version=4)
config={'merge_level':7,'allowed_elements':elements}
new=build_octree('cpu',2,8,condition_version=4,representation_config=config)
weights=old.state_dict();levels=new.cur_level.weight.detach().clone()
levels[:len(old.cur_level.weight)]=old.cur_level.weight.detach()
levels[len(old.cur_level.weight):]=old.cur_level.weight[-1].detach();weights['cur_level.weight']=levels
new.load_state_dict(weights,strict=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
if a.output.exists():raise FileExistsError(a.output)
torch.save({'model_state_dict':new.state_dict(),'backend':'uni3dar_octree_v4','target_cols':names,'target_units':units,
            'mean':mean,'scale':scale,'representation_config':config,
            'contract':checkpoint_contract(new,names,units,mean,scale),'training_smiles':[],
            'label_theory':sorted({m.GetProp('LabelTheory') for m in ms}),
            'upstream_sha256':hashlib.sha256(a.pretrained.read_bytes()).hexdigest(),
            'training_sdf_sha256':hashlib.sha256(a.sdf.read_bytes()).hexdigest(),
            'representation_migration':{'requires_adaptation':True,'old_merge_level':old.args.merge_level,
                                        'new_merge_level':7,'new_elements':elements,'old_elements':old.args.allow_atoms.split(',')},
            'condition_migration':{'requires_condition_adaptation':True}},a.output)
print(json.dumps({'records':len(ms),'elements':elements,'mean':mean.tolist(),'scale':scale.tolist()}))
