"""Predeclare one lowest-ID qualified design per inference run for quantum checks."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze_run(args.output, __file__)
    if len({run.name for run in args.runs}) != len(args.runs):
        raise ValueError('Source run names must be unique within this cohort')
    selected, records, inputs = [], [], {}
    for run in args.runs:
        cfg = json.loads((run / 'run_config.json').read_text())
        summary = json.loads((run / 'summary.json').read_text())
        table = pd.read_csv(run / 'candidates.csv').set_index('candidate_id')
        eligible = table[table['joint_success']]
        if len(table) != summary['attempted'] or len(eligible) != summary['qualified'] or not table.index.is_unique:
            raise ValueError(f'Original attempt and qualification counts disagree: {run.name}')
        for name in ['run_config.json', 'summary.json', 'candidates.csv', 'generated.sdf', 'input_manifest.json']:
            inputs[f'{run.name}/{name}'] = run / name
        entry = {'source_run': run.name, 'inference_seed': cfg['inference_config']['base_seed'],
                 'attempted': len(table), 'surrogate_qualified': len(eligible), 'selected': False,
                 'target_names': cfg['oracle_config']['target_cols'], 'target_units': cfg['oracle_config']['target_units'],
                 'target_intervals': cfg['inference_config']['target_interval']}
        if len(eligible):
            candidate_id = int(eligible.index.min())
            molecules = list(Chem.SDMolSupplier(str(run / 'generated.sdf'), removeHs=False))
            if any(m is None for m in molecules):
                raise ValueError(f'Unreadable source export: {run.name}')
            choices = [m for m in molecules if int(m.GetProp('CandidateID')) == candidate_id]
            if len(choices) != 1:
                raise ValueError('Qualified candidate must identify exactly one exported structure')
            mol = choices[0]
            mol.SetProp('SourceRun', run.name)
            mol.SetProp('SourceTargetIntervals', json.dumps(entry['target_intervals']))
            mol.SetProp('SourceTargetNames', json.dumps(entry['target_names']))
            mol.SetProp('QuantumSelection', 'Lowest original candidate ID among qualified candidates in this source run')
            entry.update(selected=True, quantum_index=len(selected), candidate_id=candidate_id,
                         surrogate_prediction_eV=json.loads(mol.GetProp('PredictedValue')),
                         source_rms_force_eV_A=float(table.loc[candidate_id, 'final_force']))
            selected.append(mol)
        records.append(entry)
    if not selected:
        raise ValueError('No qualified designs in the supplied fixed inference runs')
    write_precise_sdf(args.output / 'cohort.sdf', selected)
    record_input_hashes(args.output, inputs)
    report = {'selection': 'One lowest original candidate ID per nonempty inference-seed/window stratum; no quantum values used.',
              'attempted_generation': sum(r['attempted'] for r in records),
              'surrogate_qualified': sum(r['surrogate_qualified'] for r in records),
              'quantum_selected': len(selected),
              'source_sha256': hashlib.sha256((args.output / 'cohort.sdf').read_bytes()).hexdigest(),
              'source_runs': records,
              'limitations': 'A diagnostic subset of surrogate-qualified designs. Unselected designs have unknown quantum outcomes; no all-proposal quantum success rate can be inferred.'}
    (args.output / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
