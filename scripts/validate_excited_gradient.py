"""Finite-difference check of a total excited-state gradient at fixed theory/omega."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from pyscf import lib
from rdkit import Chem

from loom.quantum.excited_states import response, restricted_reference
from loom.quantum.ocelot import BOHR_ANGSTROM, host_array, make_quantum_molecule, theory_from_record
from loom.utils.run_evidence import freeze_run

p = argparse.ArgumentParser()
p.add_argument('--ground-run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--engine', choices=['cpu', 'gpu'], default='gpu')
p.add_argument('--triplet', action='store_true')
p.add_argument('--compare-cpu-gradient', action='store_true')
a = p.parse_args()
freeze_run(a.output, __file__)
r = json.loads((a.ground_run/'result.json').read_text())
theory = replace(theory_from_record(r['theory']), engine=a.engine)
lib.num_threads(theory.threads)
m = next(iter(Chem.SDMolSupplier(str(a.ground_run/'optimized.sdf'), removeHs=False)))
symbols = [x.GetSymbol() for x in m.GetAtoms()]
coords = m.GetConformer().GetPositions()


def state(xyz):
    mol = make_quantum_molecule(symbols, xyz, theory)
    mf = restricted_reference(mol, r['omega'], theory)
    td, energies = response(mf, not a.triplet)
    return td, float(mf.e_tot + energies[0])


td, energy = state(coords)
grad = host_array(td.nuc_grad_method().kernel(state=1))/BOHR_ANGSTROM
cpu_grad = None
if a.compare_cpu_gradient:
    lib.num_threads(16)
    cpu_grad = td.to_cpu().nuc_grad_method().kernel(state=1)/BOHR_ANGSTROM
    lib.num_threads(theory.threads)
coordinate = int(np.abs(grad).argmax())
h = .001
plus, minus = coords.copy(), coords.copy()
plus.flat[coordinate] += h
minus.flat[coordinate] -= h
_, ep = state(plus)
_, em = state(minus)
fd = (ep-em)/(2*h)
analytic = float(grad.flat[coordinate])
report = {'state': 'T1' if a.triplet else 'S1', 'energy_definition': 'RKS ground energy + full LR-TDDFT root energy',
          'functional': theory.xc, 'basis': theory.basis, 'omega': r['omega'],
          'coordinate': coordinate, 'step_A': h, 'analytic_Ha_A': analytic, 'finite_difference_Ha_A': fd,
          'absolute_error_Ha_A': abs(fd-analytic), 'tolerance_Ha_A': max(2e-5, .01*abs(analytic)),
          'total_energy_Ha': energy}
report['passed'] = report['absolute_error_Ha_A'] <= report['tolerance_Ha_A']
if cpu_grad is not None:
    report.update(cpu_analytic_Ha_A=float(cpu_grad.flat[coordinate]),
                  cpu_finite_difference_error_Ha_A=abs(float(cpu_grad.flat[coordinate])-fd),
                  cpu_gpu_gradient_max_abs_Ha_A=float(np.abs(cpu_grad-grad).max()))
np.savez(a.output/'gradients.npz', gpu_Ha_A=grad, cpu_Ha_A=cpu_grad)
(a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
assert report['passed']
