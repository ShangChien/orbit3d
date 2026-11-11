"""Keep source, selected, failed and uncomputed design denominators separate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--quantum-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text())
    quantum = json.loads((args.quantum_run / 'summary.json').read_text())
    manifest = json.loads((args.quantum_run / 'manifest.json').read_text())
    if manifest['source_sha256'] != selected['source_sha256']:
        raise ValueError('Quantum calculation used a different selected cohort')
    source = {r['quantum_index']: r for r in selected['source_runs'] if r['selected']}
    if quantum['attempted'] != len(source) or len(quantum['records']) != len(source):
        raise ValueError('The selected quantum cohort is incomplete')
    if {r['index'] for r in quantum['records']} != set(source):
        raise ValueError('Quantum results are missing or duplicate a selected index')
    rows, errors = [], {}
    for result in quantum['records']:
        original = source[result['index']]
        row = {'quantum_index': result['index'], 'source_run': original['source_run'],
               'source_candidate_id': original['candidate_id'], 'source_prediction_eV': dict(zip(original['target_names'], original['surrogate_prediction_eV'])),
               'target_intervals': original['target_intervals'], 'source_rms_force_eV_A': original['source_rms_force_eV_A'],
               'status': result['status'], 'quantum_graph_preserved_and_target_hit': False}
        if result['status'] == 'completed':
            if not result['graph_preserved'] or not result['geometry_optimized']:
                raise ValueError('Completed primary quantum check requires graph-preserving S0 optimization')
            if original['target_names'] != ['homo', 'lumo'] or original['target_units'] != ['eV', 'eV']:
                raise ValueError('This primary design check evaluates HOMO and LUMO in eV')
            values = {'homo': result['homo_eV'], 'lumo': result['lumo_eV']}
            per_target = {name: (lower is None or values[name] >= lower) and (upper is None or values[name] <= upper)
                          for name, (lower, upper) in original['target_intervals'].items()}
            disagreement = {name: abs(values[name] - row['source_prediction_eV'][name]) for name in values}
            for name, value in disagreement.items():
                errors.setdefault(name, []).append(value)
            row.update(quantum_values_eV=values, quantum_target_hits=per_target,
                       quantum_graph_preserved_and_target_hit=all(per_target.values()),
                       absolute_disagreement_eV=disagreement, omega=result['omega'],
                       quantum_rms_force_eV_A=result['rms_force_eV_A'], quantum_S2=result['S2'],
                       energy_Ha=result['energy_Ha'])
        else:
            row['failure_reason'] = result['error']
        rows.append(row)
    report = {'attempted_generation': selected['attempted_generation'],
              'source_surrogate_qualified': selected['surrogate_qualified'], 'quantum_selected': len(source),
              'qualified_but_not_quantum_selected': selected['surrogate_qualified'] - len(source),
              'quantum_completed': sum(r['status'] == 'completed' for r in rows),
              'quantum_failed': sum(r['status'] != 'completed' for r in rows),
              'quantum_graph_preserved_and_target_hits': sum(r['quantum_graph_preserved_and_target_hit'] for r in rows),
              'completed_case_mean_absolute_disagreement_eV': {name: sum(values) / len(values) for name, values in errors.items()},
              'records': rows, 'quantum_manifest': manifest,
              'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                                [('selection', args.selection), ('quantum_results', args.quantum_run / 'summary.json'),
                                 ('quantum_manifest', args.quantum_run / 'manifest.json')]},
              'limitations': ['Selected surrogate-qualified subset from one fixed SFT continuation seed, not a random sample of all generated proposals.',
                              'Unselected qualified designs have unknown quantum outcomes; no all-proposal quantum success rate is inferred.',
                              'The quantum protocol is internally consistent but does not reproduce unavailable OCELOT per-molecule geometries and omega values.',
                              'Geometry convergence and source-graph preservation do not establish vibrational minima; unrestricted S2 diagnostics are retained.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ['records', 'quantum_manifest', 'source_sha256']}, indent=2))


if __name__ == '__main__':
    main()
