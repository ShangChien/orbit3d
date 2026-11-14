"""Check whether the two preference channels add distinct gradients on source dev."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.octree_backend import load_octree_checkpoint, OctreeDataset
from orbit3d.generator.point_preferences import nearby_size_matched_pairs
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from orbit3d.utils.reproducibility import lock_deterministic_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--sdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=128)
    p.add_argument('--objective',choices=['hinge','softmax'],default='hinge')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__);lock_deterministic_environment(42);torch.set_num_threads(2)
    model,state=load_octree_checkpoint(a.checkpoint,'cpu');reference,_=load_octree_checkpoint(a.reference,'cpu')
    model.eval().requires_grad_(False);model.interval_conditioner.requires_grad_(True);reference.eval().requires_grad_(False)
    parameters=[v for v in model.parameters() if v.requires_grad]
    mols=list(Chem.SDMolSupplier(str(a.sdf),removeHs=False))[:a.limit]
    for m in mols:
        assert Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True) not in state['training_smiles']
    raw=np.asarray([[float(m.GetProp(n)) for n in state['target_cols']] for m in mols],dtype=np.float32)
    y=torch.tensor((raw-state['mean'])/state['scale']);column=state['target_cols'].index(state['active_targets'][0])
    data=OctreeDataset(mols,model,training=False);rows=[]
    for begin in range(0,len(mols),8):
        ids=np.arange(begin,min(begin+8,len(mols)));target=y[ids];packed=data.collate(ids)
        values=torch.tensor(raw[ids,column]);atoms=torch.tensor([mols[i].GetNumAtoms() for i in ids]);enabled=torch.ones(len(ids),dtype=torch.bool)
        partner,eligible=nearby_size_matched_pairs(values,atoms,enabled)
        if not bool(eligible.any()):continue
        mask=torch.zeros(len(ids),len(state['target_cols']),2,dtype=torch.bool);mask[:,column]=True
        def score(net,condition):
            return net.compute_sft_and_understanding_loss(packed,target,condition,condition,mask,understanding_weight=0,return_per_molecule=True)[1]['per_molecule_generation_loss']
        positive=score(model,target);negative=score(model,target[partner])
        with torch.no_grad():ref_positive=score(reference,target);ref_negative=score(reference,target[partner])
        own=ref_positive-positive;cross=ref_negative-negative
        condition_delta=own[eligible]-cross[eligible];sample_delta=own[eligible]-cross[partner][eligible]
        if a.objective=='hinge':
            condition=torch.relu(.02-condition_delta).mean();sample=torch.relu(.02-sample_delta).mean()
        else:
            condition=.02*torch.nn.functional.softplus(-condition_delta/.02).mean()
            sample=.02*torch.nn.functional.softplus(-sample_delta/.02).mean()
        def gradients(loss,retain):
            values=torch.autograd.grad(loss,parameters,retain_graph=retain,allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1) for p,v in zip(parameters,values)])
        # Accumulate the large parameter-vector reductions in float64; float32
        # dot/norm reduction order can otherwise report a cosine slightly > 1.
        first=gradients(condition,True).double();second=gradients(sample,False).double()
        rows.append(dict(eligible=int(eligible.sum()),condition_hinge_active_fraction=float((condition_delta<.02).float().mean()),
            sample_hinge_active_fraction=float((sample_delta<.02).float().mean()),
            channel_gradient_cosine=float(torch.dot(first,second)/(first.norm()*second.norm()).clamp_min(1e-20)),
            relative_gradient_difference=float((first-second).norm()/first.norm().clamp_min(1e-20)),
            max_absolute_reference_relative_score_change=float(torch.cat([own,cross]).detach().abs().max()),
            mean_condition_delta=float(condition_delta.detach().mean()),mean_sample_delta=float(sample_delta.detach().mean())))
    report=dict(scope='Source-dev structures, dropout-free FP32, no updates. Checks preference channel gradients, not generated quality.',
        structures=len(mols),objective=a.objective,rows=rows,
        means={k:float(np.average([r[k] for r in rows],weights=[r['eligible'] for r in rows])) for k in rows[0] if k!='eligible'})
    record_input_hashes(a.output,dict(checkpoint=a.checkpoint,reference=a.reference,sdf=a.sdf))
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report['means'])


if __name__=='__main__':main()
