"""Re-evaluate fixed XYZ pools with a new graph predictor; retain the original domain mask."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from orbit3d.constraints import TargetPoints
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.oracles.applicability import MorganDomain
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
from evaluate_point_pool import summarize_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--domain',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);freeze_run(a.output,__file__)
    (a.output/'worker_config.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
    record_input_hashes(a.output,{'independent':a.model,'normalized_domain':a.domain,'source_manifest':a.manifest})
    model=FingerprintOracle.load_from_dir(a.model);domain=MorganDomain.load(a.domain)
    elements=tuple(json.loads(a.manifest.read_text())['elements']);costs=[]
    for path in sorted(a.source.glob('eval_*/raw/summary.json')):
        started=time.perf_counter();source=path.parent.parent;dest=a.output/source.name/'raw';dest.mkdir(parents=True)
        original=json.loads(path.read_text());records=copy.deepcopy(original['records'])
        names=original['target_names'];assert model.target_cols==names
        point=TargetPoints.parse({k:v for k,v in original['target_point'].items() if v is not None},original['target_tolerance'],names)
        symbols=json.loads((source/'run/raw_symbols.json').read_text());molecules=[];ids=[]
        with np.load(source/'run/raw_trajectories.npz') as coords:
            for row in records:
                if not row['valid']:continue
                i=row['candidate_id']
                instance=trajectory_to_instance({'atom_symbols':symbols[i],'positions_np':coords[f'coordinates_{i}']},element_table=elements)
                molecules.append(instance.graph);ids.append(i)
        if molecules:
            evaluated=model.evaluate_graph_mols(molecules)
            for i,pred,error,molecule in zip(ids,evaluated['mean'],evaluated['errors'],molecules):
                if error:
                    records[i]['independent_evaluated']=False
                    records[i].pop('independent_prediction',None)
                    records[i]['independent_error']=error
                else:
                    records[i].update(independent_evaluated=True,independent_prediction=pred.tolist())
                try:records[i]['normalized_domain']=domain.assess(molecule)
                except ValueError as error:
                    records[i].update(normalized_domain=None,normalized_domain_error=str(error))
        alternative=[{**r,'domain':r.get('normalized_domain')} for r in records]
        report={**original,'scope':'Independent graph re-evaluation on identical XYZ; original domain mask retained; normalized-domain results are a separate diagnostic',
                'oracle_calls':0,'independent_calls':len(molecules),'records':records,
                'independent_prediction_count':sum(r['independent_evaluated'] for r in records),
                'metrics':summarize_records(records,point),'normalized_domain_metrics':summarize_records(alternative,point),
                'elapsed_seconds':time.perf_counter()-started,'source_pool':str(source),
                'graph_normalization':model.graph_normalization}
        (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        record_input_hashes(dest,{'original_evaluation':path,'coordinates':source/'run/raw_trajectories.npz','symbols':source/'run/raw_symbols.json'})
        costs.append({'name':source.name+'_graph_reevaluation','wall_seconds':time.perf_counter()-started,'returncode':0,'device':'cpu'})
        (a.output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
        print(source.name,len(molecules),costs[-1]['wall_seconds'],flush=True)
    print('Graph re-evaluation complete',flush=True)


if __name__=='__main__':main()
