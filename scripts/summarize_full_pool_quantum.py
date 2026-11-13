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
                prediction_std_eV=float(np.std(values)) if values else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument("--target",choices=["homo","lumo"],default="homo");p.add_argument("--quantiles",type=int,nargs="+",default=[25,50,75]);p.add_argument("--training-sdf",type=Path);a=p.parse_args();property_index=["homo","lumo"].index(a.target)
    known_graphs={Chem.MolToSmiles(Chem.RemoveHs(m)) for m in Chem.SDMolSupplier(str(a.training_sdf),removeHs=False)} if a.training_sdf else None
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
        arms[arm]=dict(proposals=n,terminal_selected=len(records),cells=cells,domain=strata,
            yield_equal_target=float(np.mean([v['yield_all_proposals'] for v in cells.values()])),
            mae_equal_target_eV=float(np.mean([v['mae_eV'] for v in cells.values()])) if all(v['mae_eV'] is not None for v in cells.values()) else None)
    report=dict(scope='Single trained seed; Wilson intervals describe proposal sampling only. Off proposals rescored at three goals are not independent repeats. DFT singlepoints do not certify optimized identity.',
        target=a.target,novelty_reference_sha256=hashlib.sha256(a.training_sdf.read_bytes()).hexdigest() if a.training_sdf else None,control=manifest.get('control','correct'),arms=arms,records=rows,source_sha256=hashes,
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),elapsed_seconds=completed['elapsed_seconds'])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(arms,indent=2))

if __name__=='__main__':main()
