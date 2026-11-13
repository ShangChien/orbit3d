"""Compose disjoint policy reports without renumbering or rewriting quantum evidence."""
import argparse,hashlib,json
from pathlib import Path


def compose(reports):
    base=reports[0];arms={};rows=[]
    for report in reports:
        for field in ['target','control','novelty_reference_sha256','extra_fitting_reference_sha256']:
            assert report.get(field)==base.get(field),f'Mismatched {field}'
        assert not set(arms)&set(report['arms']),'Repeated policy arm'
        arms.update(report['arms']);rows.extend(report['records'])
    assert len({(r['arm'],r['index']) for r in rows})==len(rows)
    return {'scope':'Composition of disjoint policy arms. Quantum indices remain local to each arm; no re-numbering or replacement of evidence. All proposal denominators are retained. Elapsed-time sum is not concurrent wall time.','target':base['target'],'control':base['control'],'novelty_reference_sha256':base.get('novelty_reference_sha256'),'extra_fitting_reference_sha256':base.get('extra_fitting_reference_sha256',{}),'arms':arms,'records':rows,'quantum_elapsed_seconds_sum':sum(r.get('elapsed_seconds',0) for r in reports)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=compose([json.loads(path.read_text()) for path in a.reports]);report['source_reports_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in a.reports};a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
