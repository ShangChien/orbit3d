"""Post hoc formal-charge support strata on frozen terminal pools; no new filter."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from orbit3d.oracles.graph_features import fingerprint_graph


def states(mol):
    graph = fingerprint_graph(mol)
    return {f'{a.GetSymbol()}:{a.GetFormalCharge():+}' for a in graph.GetAtoms()}


def summary(rows):
    result = {'n': len(rows), 'unseen_state_molecules': sum(bool(r['unseen_states']) for r in rows),
              'unseen_state_counts': dict(Counter(s for r in rows for s in r['unseen_states']))}
    for group, subset in [('seen_states', [r for r in rows if not r['unseen_states']]),
                           ('unseen_states', [r for r in rows if r['unseen_states']])]:
        metrics = {'n': len(subset)}
        for name in ['baseline', 'adapted']:
            paired = [r for r in subset if r.get('truth') is not None and r.get(name) is not None]
            if paired:
                error = np.array([np.array(r[name])-r['truth'] for r in paired])
                metrics[name] = {'n': len(paired), 'mae_eV': abs(error).mean(0).tolist(), 'bias_eV': error.mean(0).tolist()}
        result[group] = metrics
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--quantum',type=Path,required=True);p.add_argument('--transfer',type=Path,required=True);p.add_argument('--brics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reference=Counter(s for m in Chem.SDMolSupplier(str(a.source),removeHs=False) for s in states(m));known=set(reference)
    quantum=json.loads(a.quantum.read_text())['reports'];transfer=json.loads(a.transfer.read_text())
    paired={(r['cohort'],r['index']):r for r in transfer['records']};rows=[]
    for name,report in quantum.items():
        cohort=name.replace('e14_','e14_cohort_',1)
        for r in report['records']:
            data=paired[(cohort,r['index'])]
            rows.append(dict(cohort=cohort,arm=r['arm'],target=report['target'],control=report['control'],index=r['index'],unseen_states=sorted(states(Chem.MolFromSmiles(r['smiles']))-known),**{k:data[k] for k in ['truth','baseline','adapted']}))
    brics=[dict(unseen_states=sorted(states(Chem.MolFromSmiles(r['smiles']))-known)) for r in json.loads(a.brics.read_text())['records']]
    report={'scope':'Normalized element/formal-charge states versus all 43951 native training graphs. Descriptive strata on accepted endpoints only; formal charge is not an observed quantum atomic charge. This neither invalidates molecules nor changes admission, training or denominators. Chemical composition, size and geometry are confounded.',
            'training_state_counts':dict(reference),'e14_all':summary(rows),'brics_terminal':summary(brics),
            'by_pool':{f'{t}/{arm}/{control}':summary([r for r in rows if (r['target'],r['arm'],r['control'])==(t,arm,control)]) for t,arm,control in sorted({(r['target'],r['arm'],r['control']) for r in rows})},
            'records':rows,'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.source,a.quantum,a.transfer,a.brics]}}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['records','by_pool','source_sha256']},indent=2))


if __name__=='__main__':main()
