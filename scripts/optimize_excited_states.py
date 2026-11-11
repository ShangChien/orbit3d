"""Optimize spin-adapted S1/T1 at the saved ground protocol and fixed tuned omega."""
import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
from rdkit import Chem
from pyscf import gto, lib
from pyscf.geomopt.geometric_solver import kernel
from orbit3d.quantum.ocelot import OcelotTheory, HARTREE_EV, BOHR_ANGSTROM, host_array
from orbit3d.quantum.excited_states import restricted_reference, response
from orbit3d.generator.octree_tokenizer import rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run

p=argparse.ArgumentParser()
p.add_argument('--ground-run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--state', choices=['S1','T1'], required=True)
p.add_argument('--engine', choices=['cpu','gpu'], default='gpu')
a=p.parse_args()
freeze_run(a.output, __file__)
r=json.loads((a.ground_run/'result.json').read_text())
theory=replace(OcelotTheory(**r['theory']), engine=a.engine)
lib.num_threads(theory.threads)
m=next(iter(Chem.SDMolSupplier(str(a.ground_run/'optimized.sdf'), removeHs=False)))
symbols=[atom.GetSymbol() for atom in m.GetAtoms()]
mol=gto.M(atom=list(zip(symbols,m.GetConformer().GetPositions().tolist())), unit='Angstrom',
          basis=theory.basis, charge=0, spin=0, verbose=0, max_memory=theory.memory_mb)
started=time.perf_counter()
mf=restricted_reference(mol,r['omega'],theory)
td,_=response(mf,a.state=='S1')
scanner=td.nuc_grad_method().as_scanner(state=1)
converged, optimized=kernel(scanner, maxsteps=theory.max_geometry_steps,
                            convergence_energy=1e-6, convergence_grms=3e-4, convergence_gmax=4.5e-4,
                            convergence_drms=1.2e-3, convergence_dmax=1.8e-3)
energy,grad=scanner(optimized)
grad=host_array(grad)
if not converged or not np.isfinite(grad).all():
    raise RuntimeError(f'{a.state} geometry optimization did not converge')
# A gradient test is necessary even when an optimizer reports step convergence.
rms=float(np.sqrt(np.mean(np.sum(grad**2,axis=-1))))
maximum=float(np.linalg.norm(grad,axis=-1).max())
if rms>3e-4 or maximum>4.5e-4:
    raise RuntimeError(f'{a.state} final gradient exceeds the stated convergence thresholds')
coordinates=optimized.atom_coords(unit='Angstrom')
for i,xyz in enumerate(coordinates): m.GetConformer().SetAtomPosition(i,xyz.tolist())
inst=rdkit_mol_to_mol3d_instance(m,np.zeros(2))
recovered=mol3d_to_validated_rdkit_mol(replace(inst,graph=None))
identity=Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True)
preserved=Chem.MolToSmiles(Chem.RemoveHs(recovered),isomericSmiles=True)==identity
result={'state':a.state,'method':'Full spin-adapted LR-TDDFT total excited energy',
        'functional':theory.xc,'basis':theory.basis,'omega':r['omega'],
        'omega_policy':'Fixed S0-tuned omega for every geometry/state',
        'state_energy_Ha':float(energy),'S0_reference_energy_Ha':r['energy_Ha'],
        'adiabatic_excitation_eV':(float(energy)-r['energy_Ha'])*HARTREE_EV,
        'vertical_deexcitation_at_state_geometry_eV':float(host_array(scanner.base.e)[0])*HARTREE_EV,
        'rms_gradient_Ha_bohr':rms,'max_gradient_Ha_bohr':maximum,
        'raw_forces_eV_A':(-grad*HARTREE_EV/BOHR_ANGSTROM).tolist(),
        'graph_preserved':preserved,'stationary_geometry':True,'minimum_verified':False,
        'ground_result_sha256':hashlib.sha256((a.ground_run/'result.json').read_bytes()).hexdigest(),
        'elapsed_seconds':time.perf_counter()-started}
write_precise_sdf(a.output/'optimized.sdf',[m])
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert preserved, 'Excited optimization changed the inferred molecular graph'
