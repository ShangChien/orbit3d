"""Describe terminal chemistry shift; no new selection, ranking, or causal claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski


def features(mol):
    heavy=Chem.RemoveHs(mol);atoms=[a for a in heavy.GetAtoms() if a.GetAtomicNum()!=1]
    return dict(atoms=Chem.AddHs(heavy).GetNumAtoms(),heavy_atoms=len(atoms),
                aromatic_fraction=sum(a.GetIsAromatic() for a in atoms)/len(atoms),
                fraction_csp3=Lipinski.FractionCSP3(heavy),rings=Lipinski.RingCount(heavy),
                hetero_fraction=sum(a.GetAtomicNum()!=6 for a in atoms)/len(atoms))


def summarize(rows):
    result={'n':len(rows)}
    if not rows:return result
    result['feature_p10_p50_p90']={key:np.quantile([r['features'][key] for r in rows],[.1,.5,.9]).tolist() for key in rows[0]['features']}
    result['aromatic_fraction_counts']={key:sum(aromatic_bin(r)==key for r in rows) for key in ['zero','0-0.5','>0.5']}
    for name in ['dpa','unimol2']:
        eligible=[r for r in rows if r.get('truth') is not None and r.get(name) is not None]
        if eligible:
            errors=np.asarray([np.asarray(r[name])-r['truth'] for r in eligible]);result[name]={'n':len(eligible),'mae_eV':abs(errors).mean(0).tolist(),'bias_eV':errors.mean(0).tolist()}
    return result


def aromatic_bin(row):
    x=row['features']['aromatic_fraction'];return 'zero' if x==0 else '0-0.5' if x<=.5 else '>0.5'


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--quantum',type=Path,nargs='+',required=True);p.add_argument('--oracle-errors',type=Path,nargs='+',required=True);p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert len(a.quantum)==len(a.oracle_errors)==len(a.names)
    inputs=[a.source,*a.quantum,*a.oracle_errors];groups={};rows=[]
    source_rows=[{'features':features(m)} for m in Chem.SDMolSupplier(str(a.source),removeHs=False)]
    groups['source43951']=summarize(source_rows)
    for name,path,oracle_path in zip(a.names,a.quantum,a.oracle_errors):
        quantum=json.loads(path.read_text());oracles={r['index']:r for r in json.loads(oracle_path.read_text())['records']};items=[]
        for record in quantum['records']:
            paired=oracles[record['index']]
            item=dict(cohort=name,index=record['index'],arm=record['arm'],features=features(Chem.MolFromSmiles(record['smiles'])),truth=paired['truth_eV'],dpa=paired['dpa'],unimol2=paired['unimol2'])
            assert item['features']['atoms']==record['atoms']
            items.append(item)
        groups[name]={'all':summarize(items),'aromatic_strata':{key:summarize([r for r in items if aromatic_bin(r)==key]) for key in ['zero','0-0.5','>0.5']}}
        rows.extend(items)
    report={'scope':'Post-hoc descriptive chemistry on physically accepted endpoints. Source reference is the native training set. Size and aromatic fractions are correlated, not randomized controls. SCF failures remain in chemistry counts but not oracle error pairs. No claims of proposal yield from these conditional counts. Hydrogen counts reconstructed from validated molecular graphs and checked against original atoms.','groups':groups,'records':rows,'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
