"""Describe training support for normalized element/charge states; never filter proposals."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit import Chem
from loom.oracles.graph_features import fingerprint_graph
from loom.utils.run_evidence import freeze_run,record_input_hashes


def states(mol):
    graph=fingerprint_graph(mol)
    return {f'{a.GetSymbol()}:{a.GetFormalCharge():+}' for a in graph.GetAtoms()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--hyff',type=Path,required=True)
    p.add_argument('--homo',type=Path,required=True);p.add_argument('--lumo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze_run(a.output,__file__)
    train=list(Chem.SDMolSupplier(str(a.data/'train.sdf'),removeHs=False));reference=Counter(s for m in train for s in states(m));known=set(reference)
    hyff=json.loads(a.hyff.read_text());rows=[];inputs={'source_training':a.data/'train.sdf','hyff':a.hyff}
    for pool in hyff['pools']:
        if '_off_' in pool['name']:continue
        path=getattr(a,pool['target'])/pool['name']/'raw/summary.json';original=json.loads(path.read_text());inputs[pool['target']+'/'+pool['name']]=path
        independent={r['candidate_id']:r for r in pool['records']};target=pool['target'];column=['homo','lumo'].index(target)
        for candidate in original['records']:
            row={'pool':pool['name'],'target':target,'candidate_id':candidate['candidate_id'],'valid':candidate['valid'],
                 'domain':bool(candidate.get('domain') and candidate['domain']['in_domain']),'supported':None}
            if candidate['valid']:
                try:
                    unseen=sorted(states(Chem.MolFromSmiles(candidate['smiles']))-known);row.update(supported=not unseen,unseen_states=unseen)
                except ValueError as error:row['feature_error']=str(error)
                if candidate['evaluated']:row['dpa_error_eV']=abs(candidate['prediction'][column]-pool['target_eV'])
                if candidate['independent_evaluated']:row['fingerprint_error_eV']=abs(candidate['independent_prediction'][column]-pool['target_eV'])
                h=independent[candidate['candidate_id']]
                if h['hyff_evaluated']:row['hyff_error_eV']=abs(h['hyff_prediction_eV'][column]-pool['target_eV'])
            rows.append(row)
    aggregate=[]
    for target in ['homo','lumo']:
        for arm in ['frozen','continue400','grpo200','continue_wall']:
            whole=[r for r in rows if r['target']==target and r['pool'].startswith(f'eval_{arm}_p')]
            result={'target':target,'arm':arm,'attempted':len(whole),'valid':sum(r['valid'] for r in whole),
                    'domain':sum(r['domain'] for r in whole),'feature_failures':sum('feature_error' in r for r in whole),'strata':{}}
            for key,support in [('seen_charge_states',True),('unseen_charge_states',False)]:
                subset=[r for r in whole if r['domain'] and r['supported']==support];stats={'n':len(subset)}
                for model in ['dpa','fingerprint','hyff']:
                    q_maes=[]
                    for q in ['10','25','50','75','90']:
                        error=[r[model+'_error_eV'] for r in subset if '_p'+q+'_' in r['pool'] and model+'_error_eV' in r]
                        q_maes.append(float(np.mean(error)) if error else None)
                    stats[model]={'per_quantile_mae_eV':q_maes,'equal_quantile_mae_eV':float(np.mean(q_maes)) if all(v is not None for v in q_maes) else None,
                                  'property_hits':sum(r[model+'_error_eV']<=.25 for r in subset if model+'_error_eV' in r)}
                result['strata'][key]=stats
            aggregate.append(result)
    record_input_hashes(a.output,inputs)
    report={'scope':'Descriptive support diagnostic only; no chemical-validity claim or new admission gate. Element/formal-charge pairs after canonical graph normalization; counts are molecules containing each state.',
            'training_molecules':len(train),'training_state_counts':dict(reference),'unseen_generated_state_counts':dict(Counter(s for r in rows for s in r.get('unseen_states',[]))),
            'aggregate':aggregate,'records':rows}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in aggregate:print(r['target'],r['arm'],r['domain'],{k:v['n'] for k,v in r['strata'].items()},flush=True)


if __name__=='__main__':main()
