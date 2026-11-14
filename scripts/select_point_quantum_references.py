"""Extend a source-protocol check to elements present in the frozen generated cohort."""
import argparse,hashlib,json
from pathlib import Path
from rdkit import Chem
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--valid',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--existing',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze_run(a.output,__file__)
    cohort=json.loads(a.cohort.read_text());existing=json.loads(a.existing.read_text())
    elements={s for r in cohort['records'] for s in r['symbols']};covered={s for r in existing['records'] for s in r['elements']}
    missing=sorted(elements-covered);mols=list(Chem.SDMolSupplier(str(a.valid),removeHs=False));selected=[];records=[];ids=set()
    median=sorted(len(r['symbols']) for r in cohort['records'])[len(cohort['records'])//2]
    for element in missing:
        choices=[m for m in mols if element in {at.GetSymbol() for at in m.GetAtoms()} and m.GetProp('SourceID') not in ids]
        m=min(choices,key=lambda m:(abs(m.GetNumAtoms()-median),m.GetProp('SourceID')));ids.add(m.GetProp('SourceID'));selected.append(m)
        records.append({'index':len(records),'diagnostic_element':element,'source_id':m.GetProp('SourceID'),'atoms':m.GetNumAtoms(),
            'elements':sorted({at.GetSymbol() for at in m.GetAtoms()}),'source_labels_eV':[float(m.GetProp(k)) for k in ['homo','lumo']]})
    write_precise_sdf(a.output/'source.sdf',selected)
    report={'selection':'One development source molecule per uncovered generated element, nearest to generated atom-count median then SourceID; no computed errors used.',
        'covered_elements':sorted(covered),'missing_elements':missing,'selected':len(records),'records':records,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.valid,a.cohort,a.existing]},
        'sdf_sha256':hashlib.sha256((a.output/'source.sdf').read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('Additional references:',missing)


if __name__=='__main__':main()
