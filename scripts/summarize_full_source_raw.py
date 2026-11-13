"""Summarize frozen full-source SFT raw pools before relaxation or quantum evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from analyze_terminal_chemistry import features


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--training-sdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    known={Chem.MolToSmiles(Chem.RemoveHs(m)) for m in Chem.SDMolSupplier(str(a.training_sdf),removeHs=False)}
    hashes={str(a.training_sdf):hashlib.sha256(a.training_sdf.read_bytes()).hexdigest()};results={};all_rows=[]
    for target in ['homo','lumo']:
        folder=a.root/f'evaluation_{target}/eval_{target}_seed42';matrix=json.loads((folder/'matrix.json').read_text());complete=json.loads((folder/'completed.json').read_text());assert complete['pools']==len(matrix)==36
        goals={r['quantile']:r['goal_eV'] for r in matrix if r['control']=='correct'};arms=list(dict.fromkeys(r['arm'] for r in matrix));rows=[]
        for cell in matrix:
            path=folder/cell['pool']/'raw/summary.json';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();raw=json.loads(path.read_text())
            symbols=json.loads((folder/cell['pool']/'run/raw_symbols.json').read_text());assert len(symbols)==raw['attempted']==64
            for record in raw['records']:
                row={**record,'arm':cell['arm'],'target':target,'quantile':cell['quantile'],'inference_seed':cell['seed'],'control':cell['control'],'atoms':len(symbols[record['candidate_id']])}
                if row['valid']:
                    mol=Chem.MolFromSmiles(row['smiles']);row['canonical_graph']=Chem.MolToSmiles(Chem.RemoveHs(mol));row['novel_vs_43951']=row['canonical_graph'] not in known;row['features']=features(mol)
                rows.append(row)
        results[target]={}
        for arm in arms:
            results[target][arm]={}
            for control in ['correct','off']:
                subset=[r for r in rows if r['arm']==arm and r['control']==control];valid=[r for r in subset if r['valid']];assert len(subset)==(640 if control=='correct' else 128)
                cells={}
                for q,goal in goals.items():
                    chosen=[r for r in subset if control=='off' or r['quantile']==q];pred=[r['prediction'][['homo','lumo'].index(target)] for r in chosen if r['evaluated']];errors=np.asarray(pred)-goal
                    cells[q]={'attempted':len(chosen),'dpa_evaluated':len(pred),'dpa_target_mae_eV':float(abs(errors).mean()) if pred else None,'dpa_mean_eV':float(np.mean(pred)) if pred else None,'dpa_hits':int((abs(errors)<=.25).sum())}
                graphs={r['canonical_graph'] for r in valid}
                results[target][arm][control]={'proposals':len(subset),'valid':len(valid),'in_domain':sum(bool(r.get('domain') and r['domain']['in_domain']) for r in valid),'over_80_proposals':sum(r['atoms']>80 for r in subset),'unique_valid_graphs':len(graphs),'unique_novel_graphs_vs_43951':len(graphs-known),'valid_atoms_p10_p50_p90':np.quantile([r['atoms'] for r in valid],[.1,.5,.9]).tolist() if valid else None,'median_valid_aromatic_fraction':float(np.median([r['features']['aromatic_fraction'] for r in valid])) if valid else None,'median_valid_hetero_fraction':float(np.median([r['features']['hetero_fraction'] for r in valid])) if valid else None,'cells':cells,'dpa_target_mae_equal_goal_eV':float(np.mean([r['dpa_target_mae_eV'] for r in cells.values()])) if all(r['dpa_target_mae_eV'] is not None for r in cells.values()) else None,'dpa_mean_response_slope':float(np.polyfit(list(goals.values()),[r['dpa_mean_eV'] for r in cells.values()],1)[0]) if all(r['dpa_mean_eV'] is not None for r in cells.values()) else None,'generation_and_raw_evaluation_seconds':sum(r['wall_seconds'] for r in matrix if r['arm']==arm and r['control']==control)}
        all_rows.extend(rows)
    a.output.write_text(json.dumps({'scope':'All frozen raw proposals, before ranking or terminal treatment. DPA values are developer-only diagnostics over all evaluable structures; no Morgan gate. No DFT or B claim. Off rows are scored repeatedly but their proposal counts stay unique. Common novelty reference is all 43951 source train graphs; one training seed.','groups':results,'records':all_rows,'source_sha256':hashes},indent=2)+'\n')
    print(json.dumps({t:{a:{c:{k:v for k,v in s.items() if k!='cells'} for c,s in cs.items()} for a,cs in arms.items()} for t,arms in results.items()},indent=2))

if __name__=='__main__':main()
