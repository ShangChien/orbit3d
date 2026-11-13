"""Fixed source-geometry relaxation diagnostic; no accuracy claim for changed coordinates."""
import argparse,hashlib,json,os
from dataclasses import replace
from pathlib import Path
import numpy as np,torch
from rdkit import Chem
from experiment_relaxation_priority import Experiment
from orbit3d.constraints import TargetPoints
from orbit3d.generator.molecular_state import Mol3DInstance,mol3d_to_validated_rdkit_mol
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.physics.factory import build_forcefield
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2','3']
torch.set_num_threads(2);freeze_run(a.output,__file__)
manifest=json.loads((a.root/'data/manifest.json').read_text());elements=tuple(manifest['splits']['train']['elements'])
weight='/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt';ff=build_forcefield('dpa4',weight,elements,'cuda:0');oracle=InternalDPA4PropertyOracle.load_from_dir(str(a.root/'training/dpa_common/oracle'),ff)
ms=list(Chem.SDMolSupplier(str(a.root/'data/valid.sdf'),removeHs=False));ms=sorted(ms,key=lambda m:hashlib.sha256(('geometry_profile:'+m.GetProp('SourceID')).encode()).hexdigest())[:24]
record_input_hashes(a.output,{'valid':a.root/'data/valid.sdf','oracle':a.root/'training/dpa_common/oracle','forcefield':Path(weight)})
write_precise_sdf(a.output/'original.sdf',ms);results=[];finals=[]
for m in ms:
 symbols=[x.GetSymbol() for x in m.GetAtoms()];coords=m.GetConformer().GetPositions();inst=Mol3DInstance(coords=coords,atom_types=np.array([elements.index(s) for s in symbols]),smiles=Chem.MolToSmiles(m),graph=m,element_table=elements)
 task=TargetPoints.parse({k:float(m.GetProp(k)) for k in ['homo','lumo']},{'homo':.25,'lumo':.25},['homo','lumo']);exp=Experiment(inst,ff,oracle,task.intervals(),'cuda:0');row={'source_id':m.GetProp('SourceID'),'atoms':len(coords)}
 try:
  before=exp.evaluate(coords);relaxed,trace=exp.minimize(coords,'energy');out=mol3d_to_validated_rdkit_mol(replace(inst,coords=relaxed));out.SetProp('SourceID',row['source_id']);out.SetProp('GeometryOrigin','DPA pure-energy relaxation; new quantum labels unavailable');finals.append(out)
  row.update(status='completed',before=before['prediction'],after=trace['final']['prediction'],initial_rms_force=before['rms_force_eV_A'],final_rms_force=trace['final']['rms_force_eV_A'],converged=trace['converged'],steps=len(trace['history']))
 except (ValueError,RuntimeError,FloatingPointError) as e:row.update(status='failed',reason=str(e))
 results.append(row);(a.output/'summary.json').write_text(json.dumps({'attempted':24,'scope':'Stability diagnostic only; no labels for DPA-relaxed geometries','records':results},indent=2));print(row['source_id'],row['status'],flush=True)
write_precise_sdf(a.output/'relaxed.sdf',finals);(a.output/'completed.json').write_text(json.dumps({'completed':len(finals),'attempted':24}))
