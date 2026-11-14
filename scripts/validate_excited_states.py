"""Evaluate spin-labelled excitations on a previously recorded ground-state calculation."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from pyscf import lib
from rdkit import Chem

from loom.quantum.excited_states import vertical_excitations
from loom.quantum.ocelot import make_quantum_molecule, theory_from_record
from loom.utils.run_evidence import freeze_run

p = argparse.ArgumentParser()
p.add_argument('--ground-run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--engine', choices=['cpu', 'gpu'], default='cpu')
p.add_argument('--threads', type=int, default=4)
a = p.parse_args()
freeze_run(a.output, __file__)
record = json.loads((a.ground_run / 'result.json').read_text())
theory = replace(theory_from_record(record['theory']), engine=a.engine, threads=a.threads)
lib.num_threads(theory.threads)
m = next(iter(Chem.SDMolSupplier(str(a.ground_run / 'optimized.sdf'), removeHs=False)))
mol = make_quantum_molecule([x.GetSymbol() for x in m.GetAtoms()], m.GetConformer().GetPositions(), theory)
result = vertical_excitations(mol, record['omega'], theory, unrestricted_energy=record['energy_Ha'])
result['ground_result_sha256'] = hashlib.sha256((a.ground_run / 'result.json').read_bytes()).hexdigest()
(a.output / 'excitations.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
