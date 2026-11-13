"""Whole-proposal DFT yields, target strata and domain diagnostics for frozen pools."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np


def metric(rows, denominator, target, property_index=0):
    values=[r['singlepoint']['prediction_eV'][property_index] for r in rows if r.get('status')=='completed' and 'singlepoint' in r]
    errors=np.asarray(values)-target
    hits=int((abs(errors)<=.25).sum()); n=denominator; z=1.959963984540054
    center=(hits/n+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(hits/n*(1-hits/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(proposals=n,terminal_selected=len(rows),dft_completed=len(values),hits=hits,
                yield_all_proposals=hits/n,wilson95_descriptive=[center-radius,center+radius],
                mae_eV=float(abs(errors).mean()) if values else None,
                prediction_std_eV=float(np.std(values)) if values else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument("--target",choices=["homo","lumo"],default="homo");p.add_argument("--quantiles",type=int,nargs="+",default=[25,50,75]);a=p.parse_args();property_index=["homo","lumo"].index(a.target)
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
            cells[str(q)]=metric(subset,den,points[str(q)],property_index)
        strata={}
        for name,predicate in [('in_domain',lambda r:bool(r.get('domain') and r['domain']['in_domain'])),('out_domain',lambda r:not bool(r.get('domain') and r['domain']['in_domain']))]:
            strata[name]={str(q):metric([r for r in records if predicate(r) and (off or r['quantile']==q)],n if off else n//len(a.quantiles),points[str(q)],property_index) for q in a.quantiles}
        arms[arm]=dict(proposals=n,terminal_selected=len(records),cells=cells,domain=strata,
            yield_equal_target=float(np.mean([v['yield_all_proposals'] for v in cells.values()])),
            mae_equal_target_eV=float(np.mean([v['mae_eV'] for v in cells.values()])) if all(v['mae_eV'] is not None for v in cells.values()) else None)
    report=dict(scope='Single trained seed; Wilson intervals describe proposal sampling only. Off proposals rescored at three goals are not independent repeats. DFT singlepoints do not certify optimized identity.',
        control=manifest.get('control','correct'),arms=arms,records=rows,source_sha256=hashes,
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),elapsed_seconds=completed['elapsed_seconds'])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(arms,indent=2))

if __name__=='__main__':main()
