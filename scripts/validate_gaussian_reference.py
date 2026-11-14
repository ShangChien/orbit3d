"""Cross-program LC-wHPBE check using a public Cartesian Gaussian input/output pair."""

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
import numpy as np
from loom.quantum.ocelot import OcelotEvaluator, OcelotTheory, frontier_orbitals, HARTREE_EV
from loom.utils.run_evidence import freeze_run

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--log', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--engine', choices=['cpu', 'gpu'], default='cpu')
p.add_argument('--xc', default='HYB_GGA_XC_LC_WPBE_WHS')
a = p.parse_args()
text, log = a.input.read_text(), a.log.read_text()
basis = re.search(r'u?LC-wHPBE/(\S+)', text, re.I).group(1)
omega = int(re.search(r'3/107=(\d+)', text).group(1)) / 1e9
lines = text.splitlines()
start = next(i for i, s in enumerate(lines) if re.fullmatch(r'-?\d+\s+\d+', s.strip()))
charge, multiplicity = map(int, lines[start].split())
atoms, xyz = [], []
for line in lines[start + 1:]:
    if not line.strip():
        break
    symbol, *position = line.split()
    atoms.append(symbol)
    xyz.append(list(map(float, position)))
energy = float(re.findall(r'SCF Done:\s+E\(ULC-wHPBE\)\s+=\s+([-\d.]+)', log)[-1])
occupied, virtual = [], []
for line in log.splitlines():
    match = re.match(r'\s*(?:Alpha|Beta)\s+(occ\.|virt\.) eigenvalues --\s*(.*)', line)
    if match:
        values = [float(x) for x in re.findall(r'-?\d+\.\d+', match.group(2))]
        (occupied if match.group(1) == 'occ.' else virtual).extend(values)
freeze_run(a.output, __file__)
theory = OcelotTheory(basis=basis, engine=a.engine, threads=4, xc=a.xc)
mf = OcelotEvaluator(theory, a.output / 'scf').scf(atoms, np.array(xyz), omega, charge, multiplicity-1)
homo, lumo = frontier_orbitals(mf.mo_energy, mf.mo_occ)
report = {'theory': asdict(theory), 'omega': omega, 'charge': charge, 'multiplicity': multiplicity,
          'source_input_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(),
          'source_log_sha256': hashlib.sha256(a.log.read_bytes()).hexdigest(),
          'energy_delta_Ha': float(mf.e_tot - energy),
          'homo_delta_eV': (homo - max(occupied)) * HARTREE_EV,
          'lumo_delta_eV': (lumo - min(virtual)) * HARTREE_EV}
report['passed'] = abs(report['energy_delta_Ha']) < 1e-3 and max(abs(report['homo_delta_eV']), abs(report['lumo_delta_eV'])) < 5e-3
(a.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
assert report['passed'], 'LC-wHPBE cross-program parity failed'
