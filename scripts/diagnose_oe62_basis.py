"""Post-hoc fixed-geometry basis check; never overwrite the primary PBE0/TZVP cohort."""
import argparse,hashlib,json,os
from pathlib import Path
from rdkit import Chem
from pyscf import gto,lib
from oe62_quantum_singlepoint import single


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);lib.num_threads(4)
    mol=list(Chem.SDMolSupplier(str(a.input),removeHs=False))[a.index];symbols=[at.GetSymbol() for at in mol.GetAtoms()];coords=mol.GetConformer().GetPositions();rows=[]
    for basis in ['def2-tzvpd','def2-qzvp']:
        system=gto.M(atom=list(zip(symbols,coords.tolist())),basis=basis,unit='Angstrom',charge=0,spin=0,verbose=4,output=str(a.output/(basis+'.log')))
        result,_=single(system);rows.append({'basis':basis,'nao':system.nao,'result':result})
    report={'scope':'Post-hoc basis sensitivity for the LUMO reference outlier and a sulfur-containing control; no offset fitting or replacement of primary values.',
        'index':a.index,'source_id':mol.GetProp('SourceID'),'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'source_labels_eV':[float(mol.GetProp(k)) for k in ['homo','lumo']],'records':rows}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
