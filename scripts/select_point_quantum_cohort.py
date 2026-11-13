"""Freeze top1 and hash-selected retained proposals from matched formal policy pools."""
import argparse,csv,hashlib,json
from pathlib import Path
from rdkit import Chem
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    freeze_run(a.output,__file__);records=[];molecules=[];missing=[];hashes={}
    # Fixed assignment balances training/inference seeds across goals, identically for both policies.
    for target in ['homo','lumo']:
        for quantile,training_seed,inference_seed in [(25,42,2026),(50,43,2027),(75,44,2028)]:
            for arm in ['continue_wall','grpo200']:
                run=a.root/f'formal_eval_{target}_seed{training_seed}'/f'eval_{arm}_p{quantile}_s{inference_seed}'/'run'
                cfg=json.loads((run/'run_config.json').read_text());goal=cfg['inference_config']['target_point'][target]
                with (run/'candidates.csv').open() as stream:rows=[r for r in csv.DictReader(stream) if r['status']=='refined']
                rows.sort(key=lambda r:(float(r['oracle_rank_score']),int(r['candidate_id'])))
                mols={int(m.GetProp('CandidateID')):m for m in Chem.SDMolSupplier(str(run/'generated.sdf'),removeHs=False)}
                for name in ['run_config.json','candidates.csv','generated.sdf']:
                    hashes[str(run/name)]=hashlib.sha256((run/name).read_bytes()).hexdigest()
                selected=[]
                if rows:selected.append(('top1',rows[0]))
                if len(rows)>1:
                    selected.append(('fixed_hash',min(rows[1:],key=lambda r:hashlib.sha256(f'OE62-E5-20261113/{target}/{quantile}/{training_seed}/{inference_seed}/{arm}/{r["candidate_id"]}'.encode()).hexdigest())))
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
    report={'selection':'P25/P50/P75 paired GRPO200 vs time-matched SFT; top DPA-ranked export and fixed SHA256 draw from remaining exports. No quantum or independent-model score used.',
        'seed_assignment':'P25: train42/infer2026; P50: train43/infer2027; P75: train44/infer2028; exploratory target/seed strata, not independent target replications',
        'expected':24,'selected':len(records),'missing':missing,'records':records,'source_sha256':hashes,
        'sdf_sha256':hashlib.sha256((a.output/'cohort.sdf').read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('Frozen',len(records),'of 24')


if __name__=='__main__':main()
