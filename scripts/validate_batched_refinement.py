"""Compare native batched property gradients and refinement with serial execution."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from orbit3d.generator.octree_tokenizer import ATOM_TYPES, rdkit_mol_to_mol3d_instance
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.physics.dpa_forcefield import DPAPotentialGuidance
from orbit3d.physics.conformer_relaxer import relax_conformer_iso_property
from orbit3d.physics.batched_relaxer import relax_conformers_iso_property
from orbit3d.utils.run_evidence import freeze_run
from orbit3d.utils.reproducibility import lock_deterministic_environment

p = argparse.ArgumentParser()
p.add_argument('--oracle', required=True)
p.add_argument('--sdf', required=True)
p.add_argument('--weight', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', default='cuda:0')
p.add_argument('--homogeneous', action='store_true')
a = p.parse_args()
freeze_run(a.output, __file__)
lock_deterministic_environment(42)
torch.set_num_threads(2)
ff = DPAPotentialGuidance(a.weight, atom_type_map=ATOM_TYPES, device=a.device)
oracle = InternalDPA4PropertyOracle.load_from_dir(a.oracle, ff)
mols = [m for m in Chem.SDMolSupplier(a.sdf, removeHs=False) if m is not None]
if a.homogeneous:
    counts = {m.GetNumAtoms(): sum(x.GetNumAtoms() == m.GetNumAtoms() for x in mols) for m in mols}
    size = max(counts, key=lambda n: (counts[n], -n))
    mols = [m for m in mols if m.GetNumAtoms() == size]
mols = mols[:8]
instances = [rdkit_mol_to_mol3d_instance(m, np.zeros(2)) for m in mols]
coords = torch.tensor(instances[0].coords, device=a.device, requires_grad=True)
symbols = [a.GetSymbol() for a in mols[0].GetAtoms()]
single, _ = oracle.forward_differentiable_3d(coords, symbols)
sg = torch.autograd.grad(single.sum(), coords)[0]
batch_coords = torch.stack([coords.detach(), coords.detach() + 10]).requires_grad_(True)
values, _ = oracle.forward_differentiable_batch(batch_coords, [symbols, symbols])
bg = torch.autograd.grad(values[0].sum(), batch_coords)[0]
torch.testing.assert_close(bg[0], sg, atol=1e-4, rtol=1e-4)
assert torch.count_nonzero(bg[1]) == 0
coordinate = int(bg[0].abs().argmax())
h = .002
plus, minus = batch_coords.detach().clone(), batch_coords.detach().clone()
plus[0].reshape(-1)[coordinate] += h
minus[0].reshape(-1)[coordinate] -= h
vp, _ = oracle.forward_differentiable_batch(plus, [symbols, symbols])
vm, _ = oracle.forward_differentiable_batch(minus, [symbols, symbols])
fd = float((vp[0].sum() - vm[0].sum()) / (2*h))
autograd = float(bg[0].reshape(-1)[coordinate])
assert abs(fd - autograd) <= max(.01, abs(autograd)*.1)
intervals = {'homo': [None, -7.2], 'lumo': [-.6, None]}
settings = dict(n_steps=8, step_size=.004)
start = time.perf_counter()
serial = [relax_conformer_iso_property(inst, ff, oracle, intervals, **settings) for inst in instances]
serial_seconds = time.perf_counter()-start
start = time.perf_counter()
batched = relax_conformers_iso_property(instances, ff, oracle, intervals, **settings)
batch_seconds = time.perf_counter()-start
rows = []
for i, ((x, rx), (y, ry)) in enumerate(zip(serial, batched)):
    assert y is not None, ry
    row = {'index': i, 'coordinate_max_abs_A': float(np.abs(x.coords-y.coords).max()),
           'prediction_max_abs_eV': float(np.abs(np.array(rx['y_pred'])-ry['y_pred']).max()),
           'force_max_abs_eV_A': float(np.abs(np.array(rx['final_forces_eva'])-ry['final_forces_eva']).max()),
           'acceptance_identical': [(s['accepted'], s['backtracks']) for s in rx['history']] == [(s['accepted'], s['backtracks']) for s in ry['history']],
           'hit_identical': rx['interval_hit'] == ry['interval_hit']}
    rows.append(row)
report = {'records': rows, 'gradient_serial_batch_max_abs': float((sg-bg[0]).abs().max()),
          'atom_counts': [m.GetNumAtoms() for m in mols],
          'autograd': autograd, 'finite_difference': fd,
          'serial_seconds': serial_seconds, 'batch_seconds': batch_seconds, 'timing_scope': 'Shared server, contention included'}
(a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
assert all(r['acceptance_identical'] and r['hit_identical'] and r['coordinate_max_abs_A'] < 1e-5 and r['prediction_max_abs_eV'] < 1e-5 and r['force_max_abs_eV_A'] < 1e-4 for r in rows)
