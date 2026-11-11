"""Recover numerical evidence from a completed SCF whose graph validation failed."""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from rdkit import Chem
from pyscf.scf import chkfile
from orbit3d.quantum.ocelot import OcelotEvaluator,OcelotTheory,host_array,frontier_orbitals,inspect_optimized_graph,HARTREE_EV,BOHR_ANGSTROM
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run,record_input_hashes
p=argparse.ArgumentParser()
p.add_argument('--sdf',type=Path,required=True)
p.add_argument('--index',type=int,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--engine',choices=['cpu','gpu'],default='gpu')
a=p.parse_args();freeze_run(a.output,__file__)
record_input_hashes(a.output,{'cohort':a.sdf,'SCF_checkpoint':a.checkpoint,'SCF_record':a.checkpoint.with_suffix('.json')})
source=json.loads(a.checkpoint.with_suffix('.json').read_text());assert source['converged'] and source['charge']==0 and source['spin_2S']==0
molecule=list(Chem.SDMolSupplier(str(a.sdf),removeHs=False))[a.index]
qm,saved=chkfile.load_scf(str(a.checkpoint));atoms=[qm.atom_symbol(i) for i in range(qm.natm)]
assert atoms==[atom.GetSymbol() for atom in molecule.GetAtoms()]
coords=qm.atom_coords(unit='Angstrom');molecule.GetConformer().SetPositions(coords)
theory=OcelotTheory(engine=a.engine,xc=source['xc'],basis=source['basis'],threads=2,max_scf_cycles=200)
evaluator=OcelotEvaluator(theory,a.output)
coeff=np.asarray(saved['mo_coeff']);occ=np.asarray(saved['mo_occ'])
density=np.array([(c*o)@c.conj().T for c,o in zip(coeff,occ)])
evaluator.density_guesses[(np.asarray(coords,dtype=np.float64).tobytes(),0,0)]=density
start=time.perf_counter();mf=evaluator.scf(atoms,coords,source['omega'])
force=-host_array(mf.nuc_grad_method().kernel())*HARTREE_EV/BOHR_ANGSTROM
homo,lumo=frontier_orbitals(mf.mo_energy,mf.mo_occ)
report={'index':a.index,'role':'Fixed-geometry evidence recovery; original cohort failure is unchanged','theory':asdict(theory),'omega':source['omega'],'geometry_optimized_in_this_run':False,'energy_Ha':float(mf.e_tot),'source_energy_difference_Ha':float(mf.e_tot)-source['energy_Ha'],'homo_eV':homo*HARTREE_EV,'lumo_eV':lumo*HARTREE_EV,'rms_force_eV_A':float(np.sqrt(np.mean(np.sum(force**2,axis=-1)))),'atom_symbols':atoms,'optimized_coordinates_A':coords.tolist(),'raw_forces_eV_A':force.tolist(),'elapsed_seconds':time.perf_counter()-start,**inspect_optimized_graph(molecule)}
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
molecule.SetProp('QuantumGraphCheck',report['graph_check_status']);write_precise_sdf(a.output/'diagnostic_input_graph.sdf',[molecule])
print({k:v for k,v in report.items() if k not in ['raw_forces_eV_A','optimized_coordinates_A','atom_symbols']})
assert abs(report['source_energy_difference_Ha'])<1e-6, 'Recovered state differs from the saved SCF energy'
