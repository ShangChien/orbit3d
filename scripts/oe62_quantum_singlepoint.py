"""GPU PBE0 approximate reference, optionally followed by PBE-D3BJ relaxation and PBE0 recheck."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from rdkit import Chem
from pyscf import gto,lib
from gpu4pyscf.dft.rks import RKS

EV=27.211386245988

def array(x):return x.get() if hasattr(x,'get') else np.asarray(x)

def make_mol(symbols,coords,path):
 ecp={s:'def2-ecp' for s in set(symbols) if Chem.GetPeriodicTable().GetAtomicNumber(s)>36}
 return gto.M(atom=list(zip(symbols,coords.tolist())),basis='def2-tzvp',ecp=ecp,unit='Angstrom',charge=0,spin=0,verbose=4,output=str(path))

def method(mol,xc,disp=None):
 mf=RKS(mol).density_fit(auxbasis='def2-universal-jkfit');mf.xc=xc;mf.disp=disp;mf.grids.level=3;mf.conv_tol=1e-8;mf.max_cycle=100
 return mf

def single(mol):
 mf=method(mol,'pbe0');started=time.perf_counter();mf.kernel()
 if not mf.converged:raise RuntimeError('PBE0 SCF did not converge')
 e=array(mf.mo_energy);occ=array(mf.mo_occ);y=[float(e[occ>0].max()*EV),float(e[occ==0].min()*EV)]
 return {'prediction_eV':y,'energy_Ha':float(mf.e_tot),'converged':True,'seconds':time.perf_counter()-started},mf

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--optimize',action='store_true');a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2','3'];lib.num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
 ms=list(Chem.SDMolSupplier(str(a.input),removeHs=False));m=ms[a.index];assert m is not None
 syms=[x.GetSymbol() for x in m.GetAtoms()];coords=m.GetConformer().GetPositions();start=time.perf_counter()
 report={'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'index':a.index,'atoms':len(coords),'elements':sorted(set(syms)),
  'visible_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'protocol':'PBE0/def2-TZVP/DF(def2-universal-jkfit), grid level3, SCF1e-8; approximate OE62 reference','status':'running'}
 if m.HasProp('homo'):report['source_labels_eV']=[float(m.GetProp(k)) for k in ['homo','lumo']]
 try:
  mol=make_mol(syms,coords,a.output/'singlepoint.log');sp,mf=single(mol);report['singlepoint']=sp
  (a.output/'result.json').write_text(json.dumps(report,indent=2))
  if a.optimize:
   from pyscf.geomopt.geometric_solver import kernel
   from orbit3d.generator.molecular_state import Mol3DInstance,mol3d_to_validated_rdkit_mol
   from orbit3d.utils.molecular_identity import compare_molecular_identity
   from rdkit.Chem import rdDetermineBonds
   optmol=make_mol(syms,coords,a.output/'geometry.log');optmf=method(optmol,'pbe','d3bj');trace=[]
   def callback(env):
    grad=array(env['gradients']);trace.append({'energy_Ha':float(env['energy']),'rms_gradient_Ha_Bohr':float(np.sqrt(np.mean(np.square(grad)))),'max_atom_gradient_Ha_Bohr':float(np.linalg.norm(grad,axis=1).max())})
   converged,relaxed=kernel(optmf,maxsteps=80,callback=callback,convergence_energy=1e-6,convergence_grms=1e-4,convergence_gmax=2e-4,convergence_drms=1.2e-3,convergence_dmax=1.8e-3)
   final_coords=relaxed.atom_coords(unit='Angstrom');np.save(a.output/'optimized_coords.npy',final_coords)
   report['geometry']={'protocol':'PBE-D3BJ/def2-TZVP/DF, not source PBE-TS/Tier2','converged':bool(converged),'steps':len(trace),'trace':trace}
   elements=tuple(sorted(set(syms)));inst=Mol3DInstance(coords=final_coords,atom_types=np.array([elements.index(s) for s in syms]),smiles=Chem.MolToSmiles(m),graph=m,element_table=elements)
   try:
    valid=mol3d_to_validated_rdkit_mol(inst);xyz=str(len(syms))+'\noptimized\n'+'\n'.join(s+' '+' '.join(map(str,p)) for s,p in zip(syms,final_coords));recovered=Chem.MolFromXYZBlock(xyz);rdDetermineBonds.DetermineBonds(recovered,charge=0,allowChargedFragments=True,embedChiral=True)
    identity=compare_molecular_identity(m,recovered);report['geometry']['identity']=identity;report['geometry']['graph_preserved']=bool(identity['source_specified_stereo_preserved'])
   except (ValueError,RuntimeError) as error:report['geometry'].update(graph_preserved=False,identity_error=str(error))
   final_mol=make_mol(syms,final_coords,a.output/'optimized_singlepoint.log');report['optimized_singlepoint'],_=single(final_mol)
   report['status']='completed' if converged else 'geometry_not_converged'
  else:report['status']='completed'
 except (ValueError,RuntimeError,NotImplementedError) as error:report.update(status='failed',reason=str(error))
 report['elapsed_seconds']=time.perf_counter()-start;(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report['index'],report['status'],report.get('singlepoint'),flush=True)

if __name__=='__main__':main()
