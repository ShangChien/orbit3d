"""Verify an optimized TDDFT state using a central-difference gradient Hessian."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
from rdkit import Chem
from pyscf import gto, lib
from pyscf.hessian.thermo import harmonic_analysis
from orbit3d.quantum.ocelot import OcelotTheory, host_array
from orbit3d.quantum.excited_states import restricted_reference, response, response_gradient_scanner
from orbit3d.utils.run_evidence import freeze_run

p = argparse.ArgumentParser()
p.add_argument("--ground-run", type=Path, required=True)
p.add_argument("--state-run", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--step-bohr", type=float, default=0.005)
p.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
a = p.parse_args()
freeze_run(a.output, __file__)
ground = json.loads((a.ground_run / "result.json").read_text())
state = json.loads((a.state_run / "result.json").read_text())
assert (
    state["ground_result_sha256"] == hashlib.sha256((a.ground_run / "result.json").read_bytes()).hexdigest()
)
assert state["stationary_geometry"] and state["graph_preserved"]
theory = replace(OcelotTheory(**ground["theory"]), engine=a.engine)
lib.num_threads(theory.threads)
m = next(iter(Chem.SDMolSupplier(str(a.state_run / "optimized.sdf"), removeHs=False)))
mol = gto.M(
    atom=list(zip([x.GetSymbol() for x in m.GetAtoms()], m.GetConformer().GetPositions().tolist())),
    unit="Angstrom",
    basis=theory.basis,
    charge=0,
    spin=0,
    verbose=0,
    max_memory=theory.memory_mb,
)
base = mol.atom_coords()
mf = restricted_reference(mol, state["omega"], theory)
td, _ = response(mf, state["state"] == "S1")
scanner = response_gradient_scanner(td, state=1)
h = a.step_bohr
if h <= 0:
    raise ValueError("Finite-difference step must be positive")
hessian = np.full((base.size, base.size), np.nan)
records = []
started = time.perf_counter()
for coordinate in range(base.size):
    gradients = []
    energies = []
    for sign in [1, -1]:
        displaced = base.copy()
        displaced.flat[coordinate] += sign * h
        molecule = mol.set_geom_(displaced, unit="Bohr", inplace=False)
        e, g = scanner(molecule)
        if not scanner.converged:
            raise RuntimeError(f"Electronic convergence failed at coordinate{coordinate}, sign{sign}")
        gradients.append(host_array(g).reshape(-1))
        energies.append(float(e))
    hessian[:, coordinate] = (gradients[0] - gradients[1]) / (2 * h)
    records.append({"coordinate": coordinate, "total_energies_Ha": energies})
    np.savez(a.output / "partial_hessian.tmp.npz", hessian_Ha_bohr2=hessian)
    (a.output / "partial_hessian.tmp.npz").replace(a.output / "partial_hessian.npz")
    (a.output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Hessian {coordinate + 1}/{base.size}", flush=True)
asymmetry = float(abs(hessian - hessian.T).max())
symmetric = (hessian + hessian.T) * 0.5
block = symmetric.reshape(len(base), 3, len(base), 3).transpose(0, 2, 1, 3)
analysis = harmonic_analysis(mol, block, exclude_trans=True, exclude_rot=True)
frequency = np.asarray(analysis["freq_wavenumber"])
signed = np.where(abs(frequency.imag) > 1e-8, -abs(frequency.imag), frequency.real)
report = {
    "state": state["state"],
    "functional": theory.xc,
    "basis": theory.basis,
    "omega": state["omega"],
    "step_bohr": h,
    "method": "Central differences of total excited-state gradients; translation/rotation projected",
    "hessian_antisymmetry_max_Ha_bohr2": asymmetry,
    "antisymmetry_tolerance_Ha_bohr2": 0.001,
    "frequencies_cm-1": signed.tolist(),
    "imaginary_frequency_tolerance_cm-1": 20.0,
    "minimum_verified": bool(asymmetry < 0.001 and signed.min() >= -20),
    "state_result_sha256": hashlib.sha256((a.state_run / "result.json").read_bytes()).hexdigest(),
    "elapsed_seconds": time.perf_counter() - started,
}
np.savez(a.output / "normal_modes.npz", hessian_Ha_bohr2=block, modes=analysis["norm_mode"])
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not report["minimum_verified"]:
    raise SystemExit(1)
