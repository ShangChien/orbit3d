"""Freeze top1 and hash-selected retained proposals from matched formal policy pools."""
import argparse,csv,hashlib,json
from pathlib import Path
from rdkit import Chem
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=['continue_wall','grpo200'])
    p.add_argument('--campaign-prefix',default='formal_eval')
    p.add_argument('--assignments',nargs='+',default=['25:42:2026','50:43:2027','75:44:2028'],help='quantile:training_seed:inference_seed')
    p.add_argument('--hash-salt',default='OE62-E5-20261113');a=p.parse_args()
    assignments=[tuple(map(int,value.split(':'))) for value in a.assignments]
    if any(len(value)!=3 for value in assignments):p.error('Assignments require quantile:training_seed:inference_seed')
    expected=2*len(assignments)*len(a.arms)*2
    freeze_run(a.output,__file__);records=[];molecules=[];missing=[];hashes={}
    # Fixed assignment balances training/inference seeds across goals, identically for both policies.
    for target in ['homo','lumo']:
        for quantile,training_seed,inference_seed in assignments:
            for arm in a.arms:
                run=a.root/f'{a.campaign_prefix}_{target}_seed{training_seed}'/f'eval_{arm}_p{quantile}_s{inference_seed}'/'run'
                cfg=json.loads((run/'run_config.json').read_text());goal=cfg['inference_config']['target_point'][target]
                with (run/'candidates.csv').open() as stream:rows=[r for r in csv.DictReader(stream) if r['status']=='refined']
                rows.sort(key=lambda r:(float(r['oracle_rank_score']),int(r['candidate_id'])))
                mols={int(m.GetProp('CandidateID')):m for m in Chem.SDMolSupplier(str(run/'generated.sdf'),removeHs=False)}
                for name in ['run_config.json','candidates.csv','generated.sdf']:
                    hashes[str(run/name)]=hashlib.sha256((run/name).read_bytes()).hexdigest()
                selected=[]
                if rows:selected.append(('top1',rows[0]))
                if len(rows)>1:
                    selected.append(('fixed_hash',min(rows[1:],key=lambda r:hashlib.sha256(f'{a.hash_salt}/{target}/{quantile}/{training_seed}/{inference_seed}/{arm}/{r["candidate_id"]}'.encode()).hexdigest())))
                if len(selected)<2:missing.append({'source':str(run),'missing':2-len(selected)})
                for rule,row in selected:
                    i=int(row['candidate_id']);mol=mols[i];mol.SetProp('QuantumIndex',str(len(records)))
                    for name in ['homo','lumo']:
                        if mol.HasProp(name):mol.ClearProp(name)
                    records.append({'index':len(records),'target':target,'target_eV':goal,'quantile':quantile,'arm':arm,'selection':rule,
                        'training_seed':training_seed,'inference_seed':inference_seed,'candidate_id':i,'source_run':str(run),
                        'dpa_prediction_eV':json.loads(row['y_pred']),'dpa_rms_force_eVA':float(row['final_force']),
                        'symbols':[at.GetSymbol() for at in mol.GetAtoms()],'smiles':Chem.MolToSmiles(mol)})
                    molecules.append(mol)
    write_precise_sdf(a.output/'cohort.sdf',molecules)
    report={'selection':'Paired fixed policies; top DPA-ranked export and fixed SHA256 draw from remaining exports. No quantum or independent-model score used.',
        'seed_assignment':a.assignments,'arms':a.arms,'hash_salt':a.hash_salt,
        'planned_per_group':{f'{target}/{arm}':2*len(assignments) for target in ['homo','lumo'] for arm in a.arms},
        'expected':expected,'selected':len(records),'missing':missing,'records':records,'source_sha256':hashes,
        'sdf_sha256':hashlib.sha256((a.output/'cohort.sdf').read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('Frozen',len(records),'of',expected)


if __name__=='__main__':main()
