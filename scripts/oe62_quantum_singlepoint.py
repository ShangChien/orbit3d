"""CPU/GPU PBE0 approximate reference, optionally followed by PBE-D3BJ relaxation and PBE0 recheck."""
import argparse,hashlib,json,os,time
from importlib import metadata
from pathlib import Path
import numpy as np
from rdkit import Chem
from pyscf import gto,lib

EV=27.211386245988

def array(x):return x.get() if hasattr(x,'get') else np.asarray(x)

def make_mol(symbols,coords,path):
 ecp={s:'def2-tzvp' for s in set(symbols) if Chem.GetPeriodicTable().GetAtomicNumber(s)>36}
 return gto.M(atom=list(zip(symbols,coords.tolist())),basis='def2-tzvp',ecp=ecp,unit='Angstrom',charge=0,spin=0,verbose=4,output=str(path))

def method(mol,xc,disp=None,engine="gpu"):
 if engine=='cpu':
  from pyscf.dft.rks import RKS
 else:
  from gpu4pyscf.dft.rks import RKS
 mf=RKS(mol).density_fit(auxbasis='def2-universal-jkfit');mf.xc=xc;mf.disp=disp;mf.grids.level=3;mf.conv_tol=1e-8;mf.max_cycle=100
 return mf

def single(mol,engine="gpu"):
 mf=method(mol,'pbe0',engine=engine);started=time.perf_counter();mf.kernel()
 if not mf.converged:raise RuntimeError('PBE0 SCF did not converge')
 e=array(mf.mo_energy);occ=array(mf.mo_occ);y=[float(e[occ>0].max()*EV),float(e[occ==0].min()*EV)]
 return {'prediction_eV':y,'energy_Ha':float(mf.e_tot),'converged':True,'seconds':time.perf_counter()-started},mf

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--optimize',action='store_true');p.add_argument('--engine',choices=['cpu','gpu'],default='gpu');a=p.parse_args()
 calculate(a)

def calculate(a):
 lib.num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
 ms=list(Chem.SDMolSupplier(str(a.input),removeHs=False));m=ms[a.index];assert m is not None
 syms=[x.GetSymbol() for x in m.GetAtoms()];coords=m.GetConformer().GetPositions();start=time.perf_counter()
 packages=['numpy','rdkit','pyscf']+(['pyscf-dispersion','geometric'] if a.optimize else [])
 if a.engine=='gpu':packages+=['gpu4pyscf-cuda12x','gpu4pyscf-libxc-cuda12x','cupy-cuda12x','cutensor-cu12']
 versions={}
 for name in packages:
  try:versions[name]=metadata.version(name)
  except metadata.PackageNotFoundError:versions[name]=None
 report={'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'index':a.index,'atoms':len(coords),'elements':sorted(set(syms)),
  'coordinates_A':coords.tolist(),'symbols':syms,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'engine':a.engine,'runtime_versions':versions,'scratch_directory':os.environ.get('TMPDIR'),'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'protocol':'PBE0/def2-TZVP/DF(def2-universal-jkfit), grid level3, SCF1e-8; approximate OE62 reference','status':'running'}
 if m.HasProp('homo'):report['source_labels_eV']=[float(m.GetProp(k)) for k in ['homo','lumo']]
 try:
  mol=make_mol(syms,coords,a.output/'singlepoint.log')
  report['ecp_core_electrons']={mol.atom_symbol(i):int(mol.atom_nelec_core(i)) for i in range(mol.natm) if mol.atom_nelec_core(i)}
  sp,mf=single(mol,a.engine);report['singlepoint']=sp
  (a.output/'result.json').write_text(json.dumps(report,indent=2))
  if a.optimize:
   from pyscf.geomopt.geometric_solver import kernel
   from loom.generator.molecular_state import Mol3DInstance,mol3d_to_validated_rdkit_mol
   optmol=make_mol(syms,coords,a.output/'geometry.log');optmf=method(optmol,'pbe','d3bj',engine=a.engine);trace=[]
   report['geometry']={'protocol':'PBE-D3BJ/def2-TZVP/DF, not source PBE-TS/Tier2','converged':False,'graph_preserved':False,'trace':trace}
   def callback(env):
    grad=array(env['gradients']);trace.append({'energy_Ha':float(env['energy']),'rms_gradient_Ha_Bohr':float(np.sqrt(np.mean(np.square(grad)))),'max_atom_gradient_Ha_Bohr':float(np.linalg.norm(grad,axis=1).max())})
   converged,relaxed=kernel(optmf,maxsteps=80,callback=callback,convergence_energy=1e-6,convergence_grms=1e-4,convergence_gmax=2e-4,convergence_drms=1.2e-3,convergence_dmax=1.8e-3)
   final_coords=relaxed.atom_coords(unit='Angstrom');np.save(a.output/'optimized_coords.npy',final_coords)
   report['geometry']={'protocol':'PBE-D3BJ/def2-TZVP/DF, not source PBE-TS/Tier2','converged':bool(converged),'gradient_evaluations':len(trace),'trace':trace,'coordinates_A':final_coords.tolist()}
   elements=tuple(sorted(set(syms)));inst=Mol3DInstance(coords=final_coords,atom_types=np.array([elements.index(s) for s in syms]),smiles=Chem.MolToSmiles(m),graph=m,element_table=elements)
   try:
    mol3d_to_validated_rdkit_mol(inst);report['geometry']['graph_preserved']=True
   except (ValueError,RuntimeError) as error:report['geometry'].update(graph_preserved=False,identity_error=str(error))
   final_mol=make_mol(syms,final_coords,a.output/'optimized_singlepoint.log');report['optimized_singlepoint'],_=single(final_mol,a.engine)
   report['status']='completed' if converged else 'geometry_not_converged'
  else:report['status']='completed'
 except (ValueError,RuntimeError,NotImplementedError) as error:report.update(status='failed',reason=str(error))
 if 'geometry' in report:report['geometry']['gradient_evaluations']=len(report['geometry']['trace'])
 report['elapsed_seconds']=time.perf_counter()-start;(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report['index'],report['status'],report.get('singlepoint'),flush=True)

if __name__=='__main__':main()
