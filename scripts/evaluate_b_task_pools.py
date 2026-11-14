"""Re-evaluate saved raw proposals under the fixed B-only task, without training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from loom.generator.octree_backend import trajectory_to_instance
from loom.generator.molecular_state import mol3d_to_validated_rdkit_mol
from loom.oracles.unimol2_task import FrozenUniMol2Task, support_exclusion, graph_set, summarize_task
from loom.utils.run_evidence import freeze_run, record_input_hashes
from loom.utils.reproducibility import lock_deterministic_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--specs',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--source-train',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();lock_deterministic_environment(42);torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    model=FrozenUniMol2Task(a.oracle,a.selection);source_graphs=graph_set(a.source_train)
    inputs={'specs':a.specs,'oracle':a.oracle,'selection':a.selection,'source_train':a.source_train}
    reports=[];started=time.perf_counter()
    for spec in json.loads(a.specs.read_text()):
        root=Path(spec['matrix']).parent;matrix=json.loads(Path(spec['matrix']).read_text())
        inputs[spec['name']+'/matrix']=Path(spec['matrix']);assets=set(source_graphs)
        for path in spec.get('asset_graph_files',[]):
            path=Path(path);assets.update(json.loads(path.read_text()));inputs[str(path)]=path
        for path in spec.get('asset_sdfs',[]):
            path=Path(path);assets.update(graph_set(path));inputs[str(path)]=path
        for entry in matrix:
            pool=root/entry['pool'];run=pool/'run';begin=time.perf_counter()
            cfg=json.loads((run/'run_config.json').read_text())
            original=json.loads((pool/'raw/summary.json').read_text())
            old={r['candidate_id']:r for r in original['records']}
            symbols=json.loads((run/'raw_symbols.json').read_text())
            rows=[];molecules=[];indices=[]
            with np.load(run/'raw_trajectories.npz') as arrays:
                for i,atoms in enumerate(symbols):
                    xyz=arrays[f'coordinates_{i}'];row=dict(candidate_id=i,atoms=len(atoms),legal=False,evaluated=False,
                        original_valid=old[i]['valid'],domain=old[i].get('domain'),
                        raw_coordinate_sha256=__import__('hashlib').sha256(xyz.tobytes()).hexdigest());rows.append(row)
                    try:
                        inst=trajectory_to_instance({'atom_symbols':atoms,'positions_np':xyz},element_table=tuple(sorted(set(atoms))))
                        mol=mol3d_to_validated_rdkit_mol(inst)
                    except ValueError as error:
                        row['rejection']=str(error);continue
                    row.update(legal=True,smiles=inst.smiles,scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(mol)))
                    exclusion=support_exclusion(mol)
                    if exclusion:row['support_exclusion']=exclusion;continue
                    molecules.append(mol);indices.append(i)
            predictions=model.predict(molecules)
            for i,values in zip(indices,predictions):rows[i].update(evaluated=True,prediction_eV=values.tolist())
            target=entry['target'];goal=float(cfg['inference_config']['target_point'][target])
            assert goal==entry['goal_eV'] and len(rows)==entry['attempted']
            report=dict(campaign=spec['name'],**entry)
            report.update(source_pool=str(pool),seconds=time.perf_counter()-begin,records=rows,
                          metrics=summarize_task(rows,target,goal,source_graphs,assets),
                          rejection_counts=dict(Counter(r.get('rejection','').split(':')[0] for r in rows if not r['legal'])),
                          size_strata={name:summarize_task([r for r in rows if lo<=r['atoms']<=hi],target,goal,source_graphs,assets)
                            for name,lo,hi in [('1-20',1,20),('21-40',21,40),('41-80',41,80),('over80',81,100000)]})
            column=['homo','lumo'].index(target);ranked=sorted([r for r in rows if r['evaluated']],key=lambda r:(abs(r['prediction_eV'][column]-goal),r['candidate_id']))
            report['top_k']={str(k):dict(requested=k,**summarize_task(ranked[:k],target,goal,source_graphs,assets)) for k in [4,8]}
            for k,selected in report['top_k'].items():
                selected['retained_hits_per_original_proposal']=selected['hits']/len(rows)
                selected['hits_per_requested_slot']=selected['hits']/int(k)
            # Selected metrics retain their own denominator; raw metrics above remain primary.
            path=a.output/(spec['name']+'__'+entry['pool']+'.json');path.write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
            for name,file in [('config',run/'run_config.json'),('raw',pool/'raw/summary.json'),('xyz',run/'raw_trajectories.npz'),('symbols',run/'raw_symbols.json')]:inputs[spec['name']+'/'+entry['pool']+'/'+name]=file
            print(spec['name'],entry['pool'],report['metrics']['legal'],report['metrics']['hits'],flush=True)
    result=dict(scope='New B-only task on historical raw pools, development diagnostic. No DPA force gate, relaxation, DFT or new proposals. Requests are original requests; off rescoring must be labelled separately. No caches from terminal coordinates reused.',
                predictor=model.contract,source_graphs=len(source_graphs),pools=reports,elapsed_seconds=time.perf_counter()-started,
                B_molecule_calls=model.molecule_calls,B_forward_calls=model.forward_calls)
    record_input_hashes(a.output,inputs);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
