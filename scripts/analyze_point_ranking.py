"""Compare distance and source-residual probability ranking on frozen pools."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes


def probability(mu,residual,target):
    return np.mean(np.abs(np.asarray(mu)[:,None]+residual[None,:]-target)<=.25,axis=1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--homo',type=Path,required=True)
    p.add_argument('--lumo',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();freeze_run(a.output,__file__);manifest=json.loads((a.data/'manifest.json').read_text())
    with np.load(a.calibration) as z:truth=z['truth'];prediction=z['prediction']
    ids=manifest['selection_ids']['calibration'];assert len(ids)==len(truth)
    with (a.data/'calibration.csv').open() as stream:scaffolds={r['identifier']:r['scaffold'] for r in csv.DictReader(stream)}
    folds=np.array([int(hashlib.sha256(('residual-cv:'+scaffolds[i]).encode()).hexdigest()[:16],16)%2 for i in ids])
    assert set(folds)=={0,1}
    residuals=truth-prediction;calibration={};rows=[];inputs={'source_manifest':a.data/'manifest.json','calibration_predictions':a.calibration,'calibration_groups':a.data/'calibration.csv'}
    for column,target in enumerate(['homo','lumo']):
        probabilities=[];hits=[]
        for q,center in manifest['single_target_points'][target].items():
            predicted=np.zeros(len(ids))
            for fold in [0,1]:predicted[folds==fold]=probability(prediction[folds==fold,column],residuals[folds!=fold,column],center)
            probabilities.extend(predicted.tolist());hits.extend((np.abs(truth[:,column]-center)<=.25).tolist())
        probabilities=np.array(probabilities);hits=np.array(hits,dtype=float);bins=np.minimum((probabilities*5).astype(int),4)
        reliability=[]
        for index in range(5):
            mask=bins==index
            if mask.any():reliability.append({'bin':[index/5,(index+1)/5],'n':int(mask.sum()),'mean_estimate':float(probabilities[mask].mean()),'empirical_hit_rate':float(hits[mask].mean())})
        calibration[target]={'fold_sizes':np.bincount(folds).tolist(),'unique_source_molecules':len(ids),'target_query_count':len(hits),
            'brier_score':float(np.mean((probabilities-hits)**2)),'reliability_bins':reliability,
            'ece':sum(b['n']/len(hits)*abs(b['mean_estimate']-b['empirical_hit_rate']) for b in reliability)}
        root=getattr(a,target)
        inputs[f'{target}/pool_models']=root/'input_manifest.json'
        for policy in ['frozen','continue400','grpo200','continue_wall','frozen_off']:
            for q,center in manifest['single_target_points'][target].items():
                for seed in [2026,2027,2028]:
                    name=f'eval_{policy}_s{seed}' if policy=='frozen_off' else f'eval_{policy}_p{q}_s{seed}'
                    path=root/name/'raw/summary.json';pool=json.loads(path.read_text());inputs[f'{target}/{name}']=path
                    eligible=[r for r in pool['records'] if r['valid'] and r['evaluated'] and r.get('domain') and r['domain']['in_domain']]
                    distance={r['candidate_id']:((r['prediction'][column]-center)/.25)**2 for r in eligible}
                    estimates=probability([r['prediction'][column] for r in eligible],residuals[:,column],center) if eligible else []
                    probs={r['candidate_id']:float(value) for r,value in zip(eligible,estimates)}
                    for method in ['distance','probability']:
                        order=sorted(eligible,key=lambda r:(distance[r['candidate_id']],r['candidate_id']) if method=='distance'
                                     else (-probs[r['candidate_id']],distance[r['candidate_id']],r['candidate_id']))
                        selected=[];seen=set()
                        for r in order:
                            if r['smiles'] not in seen:selected.append(r);seen.add(r['smiles'])
                        for k in [1,4,8,16]:
                            chosen=selected[:k];evaluated=[r for r in chosen if r['independent_evaluated']]
                            error=[abs(r['independent_prediction'][column]-center) for r in evaluated]
                            rows.append({'target':target,'policy':policy,'quantile':q,'inference_seed':seed,'source_pool':str(path.parent.parent),
                                'method':method,'requested_outputs':k,'attempted':pool['attempted'],'eligible':len(eligible),
                                'selected':len(chosen),'selected_ids':[r['candidate_id'] for r in chosen],
                                'independent_evaluated':len(evaluated),'independent_absolute_error_sum_eV':sum(error),
                                'independent_hits':sum(e<=.25 for e in error)})
    aggregate=[]
    for target in ['homo','lumo']:
        for policy in ['frozen','continue400','grpo200','continue_wall','frozen_off']:
            for k in [1,4,8,16]:
                item={'target':target,'policy':policy,'requested_outputs':k,'methods':{}}
                for method in ['distance','probability']:
                    selected=[r for r in rows if r['target']==target and r['policy']==policy and r['requested_outputs']==k and r['method']==method]
                    n=sum(r['independent_evaluated'] for r in selected);attempted=sum(r['attempted'] for r in selected)
                    item['methods'][method]={'task_proposal_count':attempted,'unique_raw_pools':len({r['source_pool'] for r in selected}),
                        'selected':sum(r['selected'] for r in selected),'independent_evaluated':n,
                        'independent_target_mae_eV':sum(r['independent_absolute_error_sum_eV'] for r in selected)/n if n else None,
                        'independent_hits':sum(r['independent_hits'] for r in selected),
                        'independent_hit_rate_all_task_proposals':sum(r['independent_hits'] for r in selected)/attempted}
                aggregate.append(item)
    report={'scope':'No new generation or oracle calls. Paired ranking within the same pools; off-condition pools are reused across goals and are not independent repeats.',
        'probability_fit':'Full 512 source-calibration residuals for ranking; two-fold scaffold cross-fitting for source reliability. No Gaussian or target-independence assumption.',
        'normalization':'Original domain masks; independent predictor is whatever was frozen in each supplied pool.',
        'source_reliability':calibration,'aggregate':aggregate,'records':rows}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');record_input_hashes(a.output,inputs)
    print('Paired ranking tasks:',len(rows),'source reliability:',calibration)


if __name__=='__main__':main()
