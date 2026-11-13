"""CPU/GPU consistency check for the declared approximate OE62 protocol."""
import argparse,json,os
from pathlib import Path
import numpy as np
from pyscf import gto,dft,lib
from gpu4pyscf.dft.rks import RKS
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2','3']
lib.num_threads(4)
m=gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587',basis='def2-tzvp',verbose=0);values={}
for name,mf in [('cpu',dft.RKS(m).density_fit(auxbasis='def2-universal-jkfit')),('gpu',RKS(m).density_fit(auxbasis='def2-universal-jkfit'))]:
 mf.xc='pbe0';mf.grids.atom_grid=(99,590);mf.conv_tol=1e-9;mf.kernel();e=mf.mo_energy.get() if hasattr(mf.mo_energy,'get') else mf.mo_energy
 values[name]={'energy':float(mf.e_tot),'homo':float(e[m.nelectron//2-1]*27.211386245988),'lumo':float(e[m.nelectron//2]*27.211386245988),'converged':bool(mf.converged)};assert mf.converged
mf=RKS(m).density_fit(auxbasis='def2-universal-jkfit');mf.xc='pbe';mf.disp='d3bj';mf.grids.atom_grid=(99,590);mf.conv_tol=1e-9;mf.kernel();grad=mf.nuc_grad_method().kernel();grad=grad.get() if hasattr(grad,'get') else grad
values['pbe_d3bj']={'converged':bool(mf.converged),'gradient_finite':bool(np.isfinite(grad).all())}
a.output.write_text(json.dumps(values,indent=2));print(values)
assert max(abs(values['gpu'][k]-values['cpu'][k]) for k in ['homo','lumo'])<1e-4
assert values['pbe_d3bj']['converged'] and values['pbe_d3bj']['gradient_finite']
