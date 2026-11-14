"""Measure native frame batching against serial trained-oracle evaluations."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from loom.generator.molecular_state import ATOM_TYPES
from loom.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from loom.physics.dpa_forcefield import DPAPotentialGuidance
from loom.utils.run_evidence import freeze_run
from loom.utils.reproducibility import lock_deterministic_environment

p = argparse.ArgumentParser()
p.add_argument('--oracle', required=True)
p.add_argument('--sdf', required=True)
p.add_argument('--weight', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
freeze_run(a.output, __file__)
lock_deterministic_environment(42)
torch.set_num_threads(2)
ff = DPAPotentialGuidance(a.weight, atom_type_map=ATOM_TYPES, device=a.device)
oracle = InternalDPA4PropertyOracle.load_from_dir(a.oracle, ff)
mols = [m for m in Chem.SDMolSupplier(a.sdf, removeHs=False) if m is not None][:64]
oracle.predict_3d_mols(mols[:2])
oracle.predict_3d_mols_batched(mols[:2])
start = time.perf_counter()
serial = oracle.predict_3d_mols(mols)
serial_seconds = time.perf_counter() - start
report = {'records': len(mols), 'serial_seconds': serial_seconds, 'device': a.device, 'batches': []}
for size in [1, 8, 16]:
    start = time.perf_counter()
    batched = oracle.predict_3d_mols_batched(mols, batch_size=size)
    elapsed = time.perf_counter() - start
    difference = max(float(np.abs(x-y).max()) for x, y in zip(serial, batched))
    report['batches'].append({'batch_size': size, 'seconds': elapsed, 'max_prediction_or_std_error_eV': difference})
    np.testing.assert_allclose(batched, serial, atol=1e-5, rtol=1e-5)
order = np.random.default_rng(43).permutation(len(mols))
permuted = oracle.predict_3d_mols_batched([mols[i] for i in order], batch_size=8)
np.testing.assert_allclose(np.asarray(permuted)[:, np.argsort(order)], serial, atol=1e-5, rtol=1e-5)
# A full frame batch of one geometry with distinct rigid translations.
mol = mols[0]
coords = torch.tensor(mol.GetConformer().GetPositions(), device=a.device, dtype=torch.float32)
symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
single = ff.extract_native_descriptor_and_forces(coords, symbols)
frames = torch.stack([coords + 10 * i for i in range(8)])
batch = ff.extract_native_batch(frames, [symbols] * 8)
report['translated_frame_max_errors'] = []
for x, y in zip(single, batch):
    error = float((x[None] - y).abs().max())
    report['translated_frame_max_errors'].append(error)
    torch.testing.assert_close(y, x[None].expand_as(y), atol=2e-3, rtol=1e-4)
report['passed'] = True
report['timing_scope'] = 'Shared server; timing is diagnostic and includes contention'
(a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
