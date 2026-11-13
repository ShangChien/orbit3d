"""Keep initial supply, terminal acceptance and target precision as separate outcomes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator

ARMS=['point_frozen','terminal_sft','terminal_structure','terminal_physical','terminal_property']


def target_metrics(rows,field,evaluable):
    cells=[]
    for q in [25,50,75]:
        subset=[r for r in rows if r['quantile']==q]
        chosen=[r for r in subset if evaluable(r)]
        values=[field(r) for r in chosen];goals=[r['goal_eV'] for r in chosen]
        error=np.asarray(values)-np.asarray(goals)
        cells.append({'quantile':q,'attempted':len(subset),'evaluated':len(chosen),
            'mae_eV':float(np.abs(error).mean()) if chosen else None,
            'bias_eV':float(error.mean()) if chosen else None,
            'prediction_mean_eV':float(np.mean(values)) if chosen else None,
            'prediction_std_eV':float(np.std(values)) if chosen else None,
            'goal_eV':subset[0]['goal_eV'],
            'hits':{str(t):int((np.abs(error)<=t).sum()) for t in [.1,.25,.5]}})
    complete=all(r['mae_eV'] is not None for r in cells)
    attempts=sum(r['attempted'] for r in cells)
    return {'attempted':attempts,'evaluated':sum(r['evaluated'] for r in cells),
        'coverage':sum(r['evaluated'] for r in cells)/attempts,
        'mae_equal_target_eV':float(np.mean([r['mae_eV'] for r in cells])) if complete else None,
        'mean_absolute_response_bias_eV':float(np.mean([abs(r['bias_eV']) for r in cells])) if complete else None,
        'response_slope':float(np.polyfit([r['goal_eV'] for r in cells],[r['prediction_mean_eV'] for r in cells],1)[0]) if complete else None,
        'hit_yield':{str(t):sum(r['hits'][str(t)] for r in cells)/attempts for t in [.1,.25,.5]},'cells':cells}


def structural_metrics(rows,accepted=False):
    chosen=[r for r in rows if r['physical_accepted']] if accepted else [r for r in rows if r['initial_valid']]
    fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fps=[fp.GetFingerprint(Chem.MolFromSmiles(r['smiles'])) for r in chosen]
    similarities=[v for i,f in enumerate(fps) for v in DataStructs.BulkTanimotoSimilarity(f,fps[:i])]
    return {'count':len(chosen),'rate':len(chosen)/len(rows),'unique_graphs':len({r['smiles'] for r in chosen}),
        'novel_fraction':sum(r['novel'] for r in chosen)/len(chosen) if chosen else None,
        'atoms_p10_p50_p90':np.quantile([r['atoms'] for r in chosen],[.1,.5,.9]).tolist() if chosen else None,
        'internal_diversity':1-float(np.mean(similarities)) if similarities else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reports={};inputs={}
    def read(path):
        raw=path.read_bytes();inputs[str(path)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    for arm in ARMS:
        original=read(a.root/'terminals'/arm/'summary.json');rows=original['records']
        raw_cache={}
        for row in rows:
            pool=row['source_pool']
            if pool not in raw_cache:raw_cache[pool]=read(Path(pool)/'raw/summary.json')['records']
            source=next(r for r in raw_cache[pool] if r['candidate_id']==row['candidate_id'])
            assert source['valid']==row['initial_valid']
            row['novel']=bool(source.get('novel_vs_sft_training',False))
        independent=read(a.root/'independent'/arm/(arm+'.json'));b=independent['records']
        assert len(rows)==len(b)==384
        assert [(r['quantile'],r['inference_seed'],r['candidate_id']) for r in rows]==[(r['quantile'],r['inference_seed'],r['candidate_id']) for r in b]
        record={'initial':structural_metrics(rows),'terminal':structural_metrics(rows,True),
            'dpa_raw':target_metrics(rows,lambda r:r['initial']['prediction_eV'][0],lambda r:'initial' in r),
            'dpa_terminal':target_metrics(rows,lambda r:r['final']['prediction_eV'][0],lambda r:r['physical_accepted'] and 'final' in r),
            'b_terminal':target_metrics(b,lambda r:r['prediction_eV'][0],lambda r:r['evaluated']),
            'dpa_terminal_in_domain':target_metrics(rows,lambda r:r['final']['prediction_eV'][0],lambda r:r['physical_accepted'] and 'final' in r and bool(r.get('domain') and r['domain']['in_domain'])),
            'b_terminal_in_domain':target_metrics(b,lambda r:r['prediction_eV'][0],lambda r:r['evaluated'] and r['in_domain']),
            'dpa_calls':sum(r['total_dpa_calls'] for r in rows),'terminal_seconds':sum(r['seconds'] for r in rows),
            'initial_rms_force_p10_p50_p90':np.quantile([r['initial']['rms_force_eVA'] for r in rows if 'initial' in r],[.1,.5,.9]).tolist() if any('initial' in r for r in rows) else None,
            'b_support_exclusions':sum('support_exclusion' in r for r in b)}
        matrix=read(Path(rows[0]['source_pool']).parent/'matrix.json')
        record['generation_and_raw_evaluation_seconds']=sum(r['wall_seconds'] for r in matrix if r['arm']==arm)
        record['raw_proposals_including_off']=sum(r['attempted'] for r in matrix if r['arm']==arm)
        record['independent_seconds']=independent['seconds']
        if arm!='point_frozen':record['training_costs']=read(a.root/arm.removeprefix('terminal_')/'costs.json')
        reports[arm]=record
    result={'scope':'Single training seed pilot. Primary terminal denominators retain 384 proposals per arm. Three targets equally weighted for MAE; no claim of quantum truth from DPA or frozen B. Domain subsets use the same full proposal denominators.','arms':reports,'source_sha256':inputs}
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
