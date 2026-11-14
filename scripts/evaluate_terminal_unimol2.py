"""Frozen B on the exact graph-preserved terminal coordinates, retaining all proposals."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from loom.oracles.unimol2_oracle import InternalUniMol2Oracle
from loom.utils.run_evidence import freeze_run,record_input_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--terminals',type=Path,nargs='+',required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);freeze_run(a.output,__file__);selection=json.loads(a.selection.read_text())
    for name in ['model.pt','oracle_meta.json']:
        with (a.oracle/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==selection['hashes'][selection['selected_unimol']][name]
    model=InternalUniMol2Oracle.load_from_dir(a.oracle);inputs={'oracle':a.oracle,'selection':a.selection};reports=[]
    for root in a.terminals:
        started=time.perf_counter();original=json.loads((root/'summary.json').read_text());inputs[str(root)+'/summary']=root/'summary.json';inputs[str(root)+'/sdf']=root/'graph_preserved.sdf'
        mols={m.GetProp('TerminalID'):m for m in Chem.SDMolSupplier(str(root/'graph_preserved.sdf'),removeHs=False)}
        rows=[];batch=[];indices=[]
        for source in original['records']:
            row={k:source[k] for k in ['target','quantile','candidate_id','inference_seed','goal_eV','initial_valid','physical_accepted']}
            row.update(evaluated=False,in_domain=bool(source.get('domain') and source['domain']['in_domain']));rows.append(row)
            if not row['physical_accepted']:continue
            mol=mols[source['coordinates_key']]
            assert np.allclose(mol.GetConformer().GetPositions(),source['coordinates_A'],atol=1e-10,rtol=0)
            if mol.GetNumAtoms()>80:row['support_exclusion']='More than 80 source atoms';continue
            if any(at.GetSymbol()=='Te' for at in mol.GetAtoms()):row['support_exclusion']='Te absent from shared training support';continue
            batch.append(mol);indices.append(len(rows)-1)
        if batch:
            predictions,_=model.predict_3d_mols(batch,batch_size=16)
            for i,values in zip(indices,predictions):
                assert np.isfinite(values).all()
                rows[i].update(evaluated=True,prediction_eV=values.tolist())
        report={'source':str(root),'target':original['target'],'attempted':len(rows),'records':rows,'seconds':time.perf_counter()-started}
        (a.output/(root.name+'.json')).write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
        print(root.name,len(batch),flush=True)
    record_input_hashes(a.output,inputs)
    (a.output/'summary.json').write_text(json.dumps({'scope':'Frozen cross-model developer evidence on exact accepted endpoints; no ranking or training feedback. All proposals remain in denominator; B support exclusions are explicit.','runs':reports},indent=2)+'\n')


if __name__=='__main__':main()
