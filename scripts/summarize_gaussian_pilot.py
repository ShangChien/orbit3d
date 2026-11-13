"""Keep preference width, fixed-tolerance success and evaluator roles separate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator


def statistics(values, goal, attempted, requested_width):
    x=np.asarray(values,dtype=float);error=x-goal
    widths=sorted({.1,.25,.5,float(requested_width)}) if requested_width else [.1,.25,.5]
    return {'attempted':attempted,'evaluated':len(x),'coverage':len(x)/attempted,
            'target_mae_eV':float(np.abs(error).mean()) if len(x) else None,
            'target_rmse_eV':float(np.sqrt(np.square(error).mean())) if len(x) else None,
            'mean_prediction_eV':float(x.mean()) if len(x) else None,
            'bias_eV':float(error.mean()) if len(x) else None,
            'prediction_std_eV':float(x.std()) if len(x) else None,
            'hit_rate_all_attempts':{str(w):int((np.abs(error)<=w).sum())/attempted for w in widths}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--unimol2',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    matrix=json.loads((a.evaluation/'matrix.json').read_text());complete=json.loads((a.evaluation/'completed.json').read_text());assert complete['pools']==len(matrix)
    independent=json.loads((a.unimol2/'summary.json').read_text());bound=json.loads((a.unimol2/'input_manifest.json').read_text())
    b={r['name']:r for r in independent['pools']};raw={};hashes={}
    for row in matrix:
        path=a.evaluation/row['pool']/'raw/summary.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert bound[row['target']+'/'+row['pool']+'/original']['summary.json']==digest
        raw[row['pool']]=json.loads(path.read_text());hashes[str(path)]=digest
    arms=list(dict.fromkeys(r['arm'] for r in matrix));target=matrix[0]['target'];idx=['homo','lumo'].index(target)
    quantiles=['10','25','50','75','90'];shuffle={'10':'90','25':'75','50':'10','75':'25','90':'50'}
    goals={r['quantile']:r['goal_eV'] for r in matrix if r['control']=='correct'}
    cells=[];aggregates=[];fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    for arm in arms:
        widths=list(dict.fromkeys(r['preference_sigma_eV'] for r in matrix if r['arm']==arm and r['control']=='correct'))
        for width in widths:
            for control in ['correct','shuffled','off']:
                local=[]
                for q in quantiles:
                    matching=[r for r in matrix if r['arm']==arm and r['control']==('off' if control=='off' else 'correct')
                        and (control=='off' or (r['preference_sigma_eV']==width and r['quantile']==(shuffle[q] if control=='shuffled' else q)))]
                    assert matching
                    records=sum([raw[r['pool']]['records'] for r in matching],[])
                    values={name:[] for name in ['dpa','fingerprint','unimol2']}
                    for r in matching:
                        for record in raw[r['pool']]['records']:
                            if not (record.get('domain') and record['domain']['in_domain']):continue
                            if record['evaluated']:values['dpa'].append(record['prediction'][idx])
                            if record['independent_evaluated']:values['fingerprint'].append(record['independent_prediction'][idx])
                        values['unimol2'] += [record['unimol2_prediction_eV'][idx] for record in b[r['pool']]['records'] if record['original_in_domain'] and record['unimol2_evaluated']]
                    mols=[Chem.MolFromSmiles(r['smiles']) for r in records if r['valid']]
                    fingerprints=[fp.GetFingerprint(m) for m in mols]
                    sims=[v for i,f in enumerate(fingerprints) for v in DataStructs.BulkTanimotoSimilarity(f,fingerprints[:i])]
                    cell={'arm':arm,'preference_sigma_eV':width,'control':control,'quantile':q,'goal_eV':goals[q],
                          'attempted':len(records),'valid':len(mols),'unique_valid':len({r['smiles'] for r in records if r['valid']}),
                          'valid_atom_median':float(np.median([r['atoms'] for r in records if r['valid']])) if mols else None,
                          'internal_diversity':1-float(np.mean(sims)) if sims else None,
                          'metrics':{name:statistics(v,goals[q],len(records),width) for name,v in values.items()}}
                    cells.append(cell);local.append(cell)
                for evaluator in values:
                    metrics=[r['metrics'][evaluator] for r in local]
                    def mean(key):return float(np.mean([r[key] for r in metrics])) if all(r[key] is not None for r in metrics) else None
                    predictions=[r['mean_prediction_eV'] for r in metrics]
                    aggregates.append({'arm':arm,'preference_sigma_eV':width,'control':control,'evaluator':evaluator,
                        'target_mae_eV':mean('target_mae_eV'),'coverage':mean('coverage'),
                        'mean_prediction_std_eV':mean('prediction_std_eV'),
                        'response_slope':float(np.polyfit([goals[q] for q in quantiles],predictions,1)[0]) if all(v is not None for v in predictions) else None,
                        'hit_rate_fixed_025':float(np.mean([r['hit_rate_all_attempts']['0.25'] for r in metrics])),
                        'hit_rate_requested_width':float(np.mean([r['hit_rate_all_attempts'][str(width)] for r in metrics])) if width else None})
    report={'target':target,'training_seed':42,'actual_pools':len(matrix),'actual_proposals':sum(r['attempted'] for r in matrix),
        'scope':'Single training seed pilot; equal target-quantile means. Off/shuffled are rescoring of frozen pools, not additional proposals. Point-policy widths are absent; tolerance changes do not retrain or regenerate them. Frozen Uni-Mol2 is cross-model evidence, not quantum truth.',
        'aggregates':aggregates,'cells':cells,'raw_source_sha256':hashes,
        'independent_summary_sha256':hashlib.sha256((a.unimol2/'summary.json').read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
