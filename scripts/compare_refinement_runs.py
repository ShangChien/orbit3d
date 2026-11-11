"""Compare serial/batched physical paths; distinct from deterministic same-path replay."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

p = argparse.ArgumentParser()
p.add_argument('--serial', type=Path, required=True)
p.add_argument('--batched', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
left, right = [pd.read_csv(root/'candidates.csv') for root in [a.serial, a.batched]]
for name in ['candidate_id', 'token_sha256', 'status', 'joint_success', 'smiles', 'raw_valid']:
    pd.testing.assert_series_equal(left[name].fillna(''), right[name].fillna(''))
max_prediction = 0.
for x, y in zip(left.y_pred, right.y_pred):
    if not pd.isna(x):
        max_prediction = max(max_prediction, float(np.max(np.abs(np.array(json.loads(x))-json.loads(y)))))
mols = [[m for m in Chem.SDMolSupplier(str(root/'generated.sdf'), removeHs=False)] for root in [a.serial, a.batched]]
assert len(mols[0]) == len(mols[1])
max_coordinate = 0.
for x, y in zip(*mols):
    assert Chem.MolToSmiles(x) == Chem.MolToSmiles(y)
    max_coordinate = max(max_coordinate, float(np.abs(x.GetConformer().GetPositions()-y.GetConformer().GetPositions()).max()))
traces = [json.loads((root/'relaxation_traces.json').read_text()) for root in [a.serial, a.batched]]
assert len(traces[0]) == len(traces[1])
max_force = 0.
for x, y in zip(*traces):
    assert x['candidate_id'] == y['candidate_id']
    assert [(r['accepted'], r['backtracks']) for r in x['history']] == [(r['accepted'], r['backtracks']) for r in y['history']]
    for key in ['initial_forces_eva', 'final_forces_eva']:
        max_force = max(max_force, float(np.abs(np.array(x[key])-y[key]).max()))
report = {'comparison': 'serial versus native batched physical evaluation',
          'all_candidates': len(left), 'retained': len(mols[0]), 'token_and_decision_equality': True,
          'max_coordinate_error_A': max_coordinate, 'max_prediction_error_eV': max_prediction,
          'max_raw_force_error_eV_A': max_force,
          'tolerances': {'coordinates_A': 1e-5, 'properties_eV': 1e-5, 'raw_forces_eV_A': 1e-4},
          'same_path_replay_is_separate': True}
report['passed'] = max_coordinate < 1e-5 and max_prediction < 1e-5 and max_force < 1e-4
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
assert report['passed']
