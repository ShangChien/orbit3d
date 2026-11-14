"""Freeze a small terminal-geometry cohort without using independent scores."""
import argparse,hashlib,json
from pathlib import Path
from rdkit import Chem
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__);records=[];molecules=[];inputs={};missing=[]
    arms=['terminal_property','terminal_physical','terminal_sft'];assignments=[(25,2081),(50,2082),(75,2081)]
    salt='OE62-E12-terminal-20261113'
    for arm in arms:
        root=a.root/arm;report=json.loads((root/'summary.json').read_text());assert report['target']=='homo'
        inputs[arm+'/summary']=root/'summary.json';inputs[arm+'/sdf']=root/'graph_preserved.sdf'
        mols={m.GetProp('TerminalID'):m for m in Chem.SDMolSupplier(str(root/'graph_preserved.sdf'),removeHs=False)}
        for q,seed in assignments:
            rows=[r for r in report['records'] if r['quantile']==q and r['inference_seed']==seed and r['physical_accepted'] and 'final' in r]
            rows.sort(key=lambda r:(abs(r['final']['prediction_eV'][0]-r['goal_eV']),r['candidate_id']))
            selected=[]
            if rows:selected.append(('top1',rows[0]))
            if len(rows)>1:selected.append(('fixed_hash',min(rows[1:],key=lambda r:hashlib.sha256(f'{salt}/{arm}/{q}/{seed}/{r["candidate_id"]}'.encode()).hexdigest())))
            if len(selected)<2:missing.append({'arm':arm,'quantile':q,'missing':2-len(selected)})
            for rule,row in selected:
                mol=mols[row['coordinates_key']];mol.SetProp('QuantumIndex',str(len(records)))
                records.append({'index':len(records),'target':'homo','target_eV':row['goal_eV'],'quantile':q,'arm':arm,'selection':rule,'training_seed':42,'inference_seed':seed,'candidate_id':row['candidate_id'],'source_terminal':str(root),'dpa_prediction_eV':row['final']['prediction_eV'],'dpa_rms_force_eVA':row['final']['rms_force_eVA'],'symbols':[at.GetSymbol() for at in mol.GetAtoms()],'smiles':Chem.MolToSmiles(mol)})
                molecules.append(mol)
    write_precise_sdf(a.output/'cohort.sdf',molecules);record_input_hashes(a.output,inputs)
    manifest={'selection':'Predeclared terminal top1 A distance and fixed hash; no B/DFT selection or missing-slot replacement.','arms':arms,'targets':['homo'],'seed_assignment':['25:42:2081','50:42:2082','75:42:2081'],'expected':18,'selected':len(records),'missing':missing,'hash_salt':salt,'planned_per_group':{f'homo/{arm}':6 for arm in arms},'records':records,'sdf_sha256':hashlib.sha256((a.output/'cohort.sdf').read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Frozen',len(records),'of 18',flush=True)


if __name__=='__main__':main()
