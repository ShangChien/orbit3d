"""Measure packed-frame numerical agreement and force-call cost on fixed molecules."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from rdkit import Chem
import torch
from loom.physics.dpa_forcefield import DPAPotentialGuidance
from loom.utils.run_evidence import freeze_run, record_input_hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sdf',type=Path,required=True)
    parser.add_argument('--weight',type=Path,required=True)
    parser.add_argument('--legacy-module',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--molecules',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args();torch.set_num_threads(4);freeze_run(args.output,__file__)
    molecules=list(Chem.SDMolSupplier(str(args.sdf),removeHs=False))[:args.molecules]
    if len(molecules)!=args.molecules:raise ValueError('Fixed input does not contain the requested molecule count')
    elements=sorted({a.GetSymbol() for m in molecules for a in m.GetAtoms()})
    field=DPAPotentialGuidance(str(args.weight),atom_type_map=elements,device='cuda:0')
    spec=importlib.util.spec_from_file_location('legacy_dpa',args.legacy_module);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    methods={'legacy_float32_packing':legacy.DPAPotentialGuidance.compute_energy_and_forces,'float64_packing':DPAPotentialGuidance.compute_energy_and_forces}
    coords=[torch.tensor(m.GetConformer().GetPositions(),dtype=torch.float32,device='cuda:0') for m in molecules]
    types=[torch.tensor([elements.index(a.GetSymbol()) for a in m.GetAtoms()],dtype=torch.long,device='cuda:0') for m in molecules]

    def prepare(size):
        batches=[]
        for start in range(0,len(molecules),size):
            xx,tt=coords[start:start+size],types[start:start+size]
            batches.append((torch.cat(xx),torch.cat(tt),torch.cat([torch.full((len(x),),i,dtype=torch.long,device='cuda:0') for i,x in enumerate(xx)])))
        return batches

    def evaluate(method,batches):
        energies=[];forces=[]
        for x,t,b in batches:
            e,f=method(field,x,t,node_batch_idx=b,clip_forces=False,zero_com_forces=False)
            energies.append(e);forces.append(f)
        return torch.cat(energies),torch.cat(forces)

    results={};references={}
    for name,method in methods.items():
        single=prepare(1)
        for _ in range(2):evaluate(method,single)
        references[name]=evaluate(method,single)
        rows=[]
        for size in [1,4,16,32]:
            batches=prepare(size)
            for _ in range(2):evaluate(method,batches)
            seconds=[]
            for _ in range(args.repeats):
                torch.cuda.synchronize();start=time.perf_counter();energy,force=evaluate(method,batches);torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
            re,rf=references[name]
            rows.append(dict(batch_size=size,forcefield_calls=len(batches),median_seconds=float(np.median(seconds)),seconds=seconds,
                             max_energy_delta_vs_own_serial_eV=float((energy-re).abs().max()),
                             max_force_component_delta_vs_own_serial_eVA=float((force-rf).abs().max()),
                             rms_force_component_delta_vs_own_serial_eVA=float((force-rf).square().mean().sqrt()),
                             energy_eV=energy.cpu().tolist()))
        results[name]=rows
    le,lf=references['legacy_float32_packing'];ne,nf=references['float64_packing']
    report=dict(scope='Fixed first 32 E14 accepted endpoints, no result-based selection. Same model, raw forces, no clipping or center-of-mass force removal. Warmed CUDA-synchronized force-call timings exclude input tensor preparation, optimization, inference and chemistry. Packed versus serial agreement is measured, not presumed; call speed is not end-to-end speed.',
                atoms=[m.GetNumAtoms() for m in molecules],elements=elements,results=results,
                serial_precision_change=dict(max_energy_delta_eV=float((le-ne).abs().max()),max_force_component_delta_eVA=float((lf-nf).abs().max())))
    record_input_hashes(args.output,{'input_sdf':args.sdf,'weight':args.weight,'legacy_module':args.legacy_module})
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':main()
