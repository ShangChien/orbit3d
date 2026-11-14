"""Compare CPU/GPU tuned-LC-wHPBE on exactly the same geometry and density."""

import argparse
import json
from pathlib import Path
import cupy
from pyscf import dft, lib
from pyscf.scf.chkfile import load_scf
from loom.quantum.gpu_engine import gpu_scf
from loom.quantum.ocelot import frontier_orbitals, HARTREE_EV
from loom.utils.run_evidence import freeze_run

p = argparse.ArgumentParser()
p.add_argument('--input-checkpoint', required=True)
p.add_argument('--omega', type=float, required=True)
p.add_argument('--xc', default='HYB_GGA_XC_LC_WPBE')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
freeze_run(a.output, __file__)
lib.num_threads(2)
mol, data = load_scf(a.input_checkpoint)
cpu = dft.UKS(mol)
gpu = gpu_scf(mol)
initial = cpu.make_rdm1(data['mo_coeff'], data['mo_occ'])
for mf in [cpu, gpu]:
    mf.xc = a.xc
    mf.omega = a.omega
    mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1e-9
    mf.max_cycle = 100
cpu.kernel(dm0=initial)
gpu.kernel(dm0=cupy.asarray(initial))
hc, lc = frontier_orbitals(cpu.mo_energy, cpu.mo_occ)
hg, lg = frontier_orbitals(gpu.mo_energy, gpu.mo_occ)
report = {'converged': bool(cpu.converged and gpu.converged), 'omega': a.omega,
          'energy_delta_Ha': float(gpu.e_tot - cpu.e_tot),
          'homo_delta_eV': (hg - hc) * HARTREE_EV,
          'lumo_delta_eV': (lg - lc) * HARTREE_EV, 'density_fitting': False}
(a.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(report)
assert report['converged']
assert abs(report['energy_delta_Ha']) < 1e-6
assert max(abs(report['homo_delta_eV']), abs(report['lumo_delta_eV'])) < 1e-4
