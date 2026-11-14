"""Apply the octree terminal gate to all frozen BRICS attempts without a Morgan gate."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from loom.generator.molecular_state import rdkit_mol_to_mol3d_instance,mol3d_to_validated_rdkit_mol
from loom.pipeline.evaluators import load_evaluators
from loom.physics.terminal_evaluation import evaluate_terminal
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.run_evidence import freeze_run,record_input_hashes
from loom.utils.reproducibility import lock_deterministic_environment
from diagnose_terminal_pool import summarize


def frozen_instance(original,mols,arrays,elements):
    key=f'{original["seed"]}:{original["candidate_id"]}'
    if key in mols:
        mol=Chem.Mol(mols[key]);center=False
    elif 'conformer_coordinates_key' in original:
        mol=Chem.AddHs(Chem.MolFromSmiles(original['smiles']));xyz=arrays[original['conformer_coordinates_key']]
        assert [at.GetSymbol() for at in mol.GetAtoms()]==original['conformer_symbols']
        conf=Chem.Conformer(len(xyz))
        for j,position in enumerate(xyz):conf.SetAtomPosition(j,position)
        mol.AddConformer(conf);center=True
    else:raise ValueError(original.get('proposal_error','no_original_coordinates'))
    if mol.GetNumAtoms()>80:raise ValueError('atom_limit')
    inst=rdkit_mol_to_mol3d_instance(mol,np.empty(0),center=center,element_table=elements)
    if inst is None:raise ValueError('geometry_admission')
    return inst


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    lock_deterministic_environment(42);torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);freeze_run(a.output,__file__)
    cfg=json.loads(a.config.read_text());cfg['device']='cuda:0';manifest=json.loads(a.manifest.read_text());elements=tuple(manifest['elements'])
    source=json.loads((a.source/'summary.json').read_text());assert len(source['records'])==192
    mols={m.GetProp('ProposalID'):m for m in Chem.SDMolSupplier(str(a.source/'raw_valid.sdf'),removeHs=False)}
    field,oracle,domain=load_evaluators(cfg,elements);records=[];exports=[]
    record_input_hashes(a.output,{'summary':a.source/'summary.json','raw_sdf':a.source/'raw_valid.sdf','raw_coordinates':a.source/'raw_conformers.npz','config':a.config,'manifest':a.manifest})
    with np.load(a.source/'raw_conformers.npz') as arrays:
        for original in source['records']:
            started=time.perf_counter();seed=original['seed'];i=original['candidate_id'];key=f'{seed}:{i}'
            row=dict(target='homo',quantile=50,inference_seed=seed,candidate_id=i,group=f'{seed}/{i//32}',goal_eV=manifest['single_target_points']['homo']['50'],initial_valid=False,physical_accepted=False,total_dpa_calls=0,atoms=original.get('proposed_atoms'),source_pool=str(a.source),historical_domain=original['domain'])
            try:
                inst=frozen_instance(original,mols,arrays,elements)
                graph=mol3d_to_validated_rdkit_mol(inst)
                row.update(initial_valid=True,atoms=len(inst.coords),smiles=inst.smiles,domain=domain.assess(graph) if domain else None)
                final,terminal=evaluate_terminal(inst,field,oracle);row.update(terminal);row['coordinates_key']=key
                info=row.get('optimization',{})
                if info.get('geometry_valid') and info.get('connectivity_preserved'):
                    mol=mol3d_to_validated_rdkit_mol(final);mol.SetProp('TerminalID',key);exports.append(mol)
            except (ValueError,RuntimeError,FloatingPointError) as error:row['error']=str(error)
            row['seconds']=time.perf_counter()-started;assert row['total_dpa_calls']<=150;records.append(row)
            (a.output/'records.json').write_text(json.dumps(records,indent=2)+'\n')
            print(seed,i,row['initial_valid'],row['physical_accepted'],flush=True)
    write_precise_sdf(a.output/'graph_preserved.sdf',exports)
    (a.output/'summary.json').write_text(json.dumps(dict(scope='192 frozen unconditional attempts, no Morgan or property selection. Same 148+2 DPA gate. One geometry per proposal reused across target requests.',target='homo',control='off',all=summarize(records),records=records),indent=2)+'\n')

if __name__=='__main__':main()
