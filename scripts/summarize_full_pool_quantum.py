"""Whole-proposal DFT yields, target strata and domain diagnostics for frozen pools."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from rdkit import Chem


def metric(rows, denominator, target, property_index=0, known_graphs=None):
    values=[r['singlepoint']['prediction_eV'][property_index] for r in rows if r.get('status')=='completed' and 'singlepoint' in r]
    hit_rows=[r for r in rows if r.get('status')=='completed' and 'singlepoint' in r and abs(r['singlepoint']['prediction_eV'][property_index]-target)<=.25]
    graphs={Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(r['smiles']))) for r in hit_rows if r.get('smiles')}
    all_graphs_known=all(r.get('smiles') for r in hit_rows)
    errors=np.asarray(values)-target
    hits=int((abs(errors)<=.25).sum()); n=denominator; z=1.959963984540054
    center=(hits/n+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(hits/n*(1-hits/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(proposals=n,terminal_selected=len(rows),dft_completed=len(values),hits=hits,
                unique_hit_graphs=len(graphs) if all_graphs_known else None,unique_novel_hit_graphs=len(graphs-known_graphs) if known_graphs is not None and all_graphs_known else None,
                yield_all_proposals=hits/n,wilson95_descriptive=[center-radius,center+radius],
                mae_eV=float(abs(errors).mean()) if values else None,
                prediction_mean_eV=float(np.mean(values)) if values else None,bias_eV=float(errors.mean()) if values else None,
                prediction_std_eV=float(np.std(values)) if values else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument("--target",choices=["homo","lumo"],default="homo");p.add_argument("--quantiles",type=int,nargs="+",default=[25,50,75]);p.add_argument("--training-sdf",type=Path);p.add_argument("--extra-training-sdf",type=Path,nargs="+",default=[]);a=p.parse_args();property_index=["homo","lumo"].index(a.target)
    known_graphs={Chem.MolToSmiles(Chem.RemoveHs(m)) for m in Chem.SDMolSupplier(str(a.training_sdf),removeHs=False)} if a.training_sdf else None
    native_graphs=None if known_graphs is None else known_graphs.copy()
    if a.extra_training_sdf and known_graphs is None:raise ValueError('Extra fitting structures require the native reference')
    for path in a.extra_training_sdf:
        known_graphs.update(Chem.MolToSmiles(Chem.RemoveHs(m)) for m in Chem.SDMolSupplier(str(path),removeHs=False))
    manifest=json.loads(a.manifest.read_text());points=json.loads(a.targets.read_text())['single_target_points'][a.target]
    completed=json.loads((a.results/'completed.json').read_text());assert completed['attempted']==manifest['selected']
    rows=[];hashes={}
    for item in manifest['records']:
        path=a.results/f'job{item["index"]:02d}/result.json'
        result=json.loads(path.read_text()) if path.exists() else dict(status='worker_failed')
        if path.exists():
            assert result['input_sha256']==manifest['sdf_sha256'] and result['index']==item['index']
            assert result['status']!='running'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({**item,**result})
    arms={};off=manifest.get('control')=='off'
    def size_bin(r):
        n=r.get('atoms')
        return 'unknown' if n is None else '0-25' if n<=25 else '26-40' if n<=40 else '41-80' if n<=80 else '>80'
    for arm in manifest['arms']:
        records=[r for r in rows if r['arm']==arm];n=manifest['proposal_denominators'][arm]
        cells={}
        for q in a.quantiles:
            subset=records if off else [r for r in records if r['quantile']==q]
            den=n if off else n//len(a.quantiles)
            cells[str(q)]=metric(subset,den,points[str(q)],property_index,known_graphs)
        strata={}
        for name,predicate in [('in_domain',lambda r:bool(r.get('domain') and r['domain']['in_domain'])),('out_domain',lambda r:bool(r.get('domain') and not r['domain']['in_domain'])),('domain_unavailable',lambda r:not r.get('domain'))]:
            strata[name]={str(q):metric([r for r in records if predicate(r) and (off or r['quantile']==q)],n if off else n//len(a.quantiles),points[str(q)],property_index,known_graphs) for q in a.quantiles}
        size_strata={}
        if records:
            source=Path(records[0]['source_terminal'])/'summary.json'
            raw=source.read_bytes();hashes[str(source)]=hashlib.sha256(raw).hexdigest()
            proposals=json.loads(raw)['records'];assert len(proposals)==n
            for label in ['0-25','26-40','41-80','>80','unknown']:
                size_strata[label]={}
                for q in a.quantiles:
                    den=sum(size_bin(r)==label and (off or r['quantile']==q) for r in proposals)
                    chosen=[r for r in records if size_bin(r)==label and (off or r['quantile']==q)]
                    size_strata[label][str(q)]=metric(chosen,den,points[str(q)],property_index,known_graphs) if den else None
        hits=[r for r in records if r.get('status')=='completed' and 'singlepoint' in r and
              any(abs(r['singlepoint']['prediction_eV'][property_index]-goal)<=.25 for goal in
                  ([points[str(q)] for q in a.quantiles] if off else [r['target_eV']]))]
        unique={Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(r['smiles']))) for r in hits}
        indices={r['index'] for r in records}
        task_seconds=sum(j['end']-j['start'] for j in completed['jobs'] if j['index'] in indices)
        mismatched=None
        if not off and records and len(a.quantiles)>1:
            seeds=sorted({r['inference_seed'] for r in proposals});slots=max(r['candidate_id'] for r in proposals)+1
            assert n==len(a.quantiles)*len(seeds)*slots
            differences=np.zeros((len(a.quantiles),len(seeds),slots));wrong_hits=0.;wrong_maes=[]
            for q in a.quantiles:
                observed=[r for r in records if r['quantile']==q and r.get('status')=='completed' and 'singlepoint' in r]
                other=[points[str(k)] for k in a.quantiles if k!=q]
                if observed:wrong_maes.append(float(np.mean([abs(r['singlepoint']['prediction_eV'][property_index]-goal) for r in observed for goal in other])))
                for r in observed:
                    value=r['singlepoint']['prediction_eV'][property_index]
                    wrong=float(np.mean([abs(value-goal)<=.25 for goal in other]));correct=float(abs(value-points[str(q)])<=.25)
                    wrong_hits+=wrong;differences[a.quantiles.index(q),seeds.index(r['inference_seed']),r['candidate_id']]=correct-wrong
            rng=np.random.default_rng(1520);bootstrap=[]
            for _ in range(2000):
                selected=rng.integers(0,len(seeds),size=len(seeds));ids=rng.integers(0,slots,size=(len(seeds),slots))
                bootstrap.append(float(np.concatenate([differences[:,seed,idx] for seed,idx in zip(selected,ids)],axis=1).mean()))
            mismatched=dict(scope='Supplementary target-label reassignment, not new proposals. Each active-condition pool is scored against every other requested value, equally weighted. Shared seed and candidate resampling across targets; no training-variance estimate.',
                wrong_value_hit_yield=wrong_hits/n,correct_minus_wrong_hit_yield=float(differences.mean()),
                sampling_delta_95=np.quantile(bootstrap,[.025,.975]).tolist(),
                wrong_value_mae_equal_source_eV=float(np.mean(wrong_maes)) if len(wrong_maes)==len(a.quantiles) else None)
        arms[arm]=dict(proposals=n,terminal_selected=len(records),cells=cells,domain=strata,size_strata=size_strata,correct_vs_mismatched=mismatched,
            unique_hits_across_requests=len(unique),
            unique_novel_hits_across_requests=len(unique-known_graphs) if known_graphs is not None else None,
            unique_novel_native_hits_across_requests=len(unique-native_graphs) if native_graphs is not None else None,
            quantum_task_wall_seconds=task_seconds,
            response_slope=float(np.polyfit([points[str(q)] for q in a.quantiles],[cells[str(q)]['prediction_mean_eV'] for q in a.quantiles],1)[0]) if len(a.quantiles)>1 and all(v['prediction_mean_eV'] is not None for v in cells.values()) else None,
            mean_absolute_response_bias_eV=float(np.mean([abs(v['bias_eV']) for v in cells.values()])) if all(v['bias_eV'] is not None for v in cells.values()) else None,
            yield_equal_target=float(np.mean([v['yield_all_proposals'] for v in cells.values()])),
            mae_equal_target_eV=float(np.mean([v['mae_eV'] for v in cells.values()])) if all(v['mae_eV'] is not None for v in cells.values()) else None)
    report=dict(scope='Single trained seed; Wilson intervals describe proposal sampling only. Off proposals rescored at three goals are not independent repeats. Across-request unique counts are system pool counts, not single-goal success rates. Quantum task times exclude upstream generation/relaxation/training. DFT singlepoints do not certify optimized identity.',
        target=a.target,extra_fitting_reference_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.extra_training_sdf},novelty_reference_sha256=hashlib.sha256(a.training_sdf.read_bytes()).hexdigest() if a.training_sdf else None,control=manifest.get('control','correct'),arms=arms,records=rows,source_sha256=hashes,
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),elapsed_seconds=completed['elapsed_seconds'])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(arms,indent=2))

if __name__=='__main__':main()
