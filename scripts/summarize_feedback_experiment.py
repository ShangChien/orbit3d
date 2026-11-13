"""Summarize fixed terminal-feedback arms after every proposal and DFT task finishes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_quantum_arms import compare


def request_subset(report, quantiles):
    """Keep request-level denominators, including proposals without an endpoint."""
    rows = [r for r in report['records'] if r['quantile'] in quantiles]
    arms = {}
    for arm, original in report['arms'].items():
        cells = {str(q): original['cells'][str(q)] for q in quantiles}
        goals = {r['quantile']: r['target_eV'] for r in rows if r['arm'] == arm}
        arms[arm] = dict(proposals=sum(c['proposals'] for c in cells.values()),
                         terminal_selected=sum(c['terminal_selected'] for c in cells.values()),
                         cells=cells,
                         yield_equal_target=float(np.mean([c['yield_all_proposals'] for c in cells.values()])),
                         mae_equal_target_eV=float(np.mean([c['mae_eV'] for c in cells.values()])),
                         response_slope=float(np.polyfit([goals[q] for q in quantiles], [cells[str(q)]['prediction_mean_eV'] for q in quantiles], 1)[0]))
    return dict(control='correct', target=report['target'], arms=arms, records=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    protocol = read(root/'protocol.json')
    quantum = read(root/'statistics/correct.json')
    result = dict(scope='E16 fixed single training seed. Full-proposal DFT hit yield is primary; raw chemistry, terminal supply and accepted-target quality are separate. P35/P65 are unseen GRPO requests, not unseen continuous SFT labels. B is frozen development evidence; neither B nor DFT selected checkpoints. Bootstrap intervals cover sampling only.', protocol=protocol, arms={},
                  comparisons_vs_native_critic=compare(quantum, 'native_critic'),
                  comparisons_vs_frozen=compare(quantum, 'frozen'),
                  comparisons_vs_time_sft=compare(quantum, 'continue_wall'),
                  condition_vs_off=read(root/'statistics/condition_vs_off.json'))
    for name, quantiles in [('trained_requests', [25, 50, 75]), ('interpolation_requests', [35, 65])]:
        subset = request_subset(quantum, quantiles)
        result[name] = dict(arms=subset['arms'], comparisons_vs_native_critic=compare(subset, 'native_critic'))
    for arm, summary in quantum['arms'].items():
        terminal = read(root/f'terminals/{arm}/summary.json')
        rows = terminal['records']
        b = read(root/f'independent/{arm}/summary.json')['runs'][0]
        values = [r for r in b['records'] if r['evaluated']]
        indexed = {(r['quantile'], r['inference_seed'], r['candidate_id']): r for r in quantum['records'] if r['arm'] == arm}
        errors = [np.asarray(r['prediction_eV'])-indexed[(r['quantile'], r['inference_seed'], r['candidate_id'])]['singlepoint']['prediction_eV'] for r in values if indexed[(r['quantile'], r['inference_seed'], r['candidate_id'])]['status']=='completed']
        forces = [r['initial']['max_force_eVA'] for r in rows if r.get('initial', {}).get('max_force_eVA') is not None]
        sizes = [r['atoms'] for r in rows if r['initial_valid']]
        valid_graphs = {r['smiles'] for r in rows if r['initial_valid'] and r.get('smiles')}
        item = dict(quantum=summary, terminal={k:v for k,v in terminal.items() if k!='records'},
                    initial_force_measurements=len(forces), initial_max_force_p10_p50_p90=np.quantile(forces,[.1,.5,.9]).tolist() if forces else None,
                    initial_valid_atoms_p10_p50_p90=np.quantile(sizes,[.1,.5,.9]).tolist() if sizes else None,
                    initial_valid_unique_graphs=len(valid_graphs),
                    B=dict(evaluated=len(values), support_exclusions=[r for r in b['records'] if r['physical_accepted'] and not r['evaluated']],
                           hits=sum(abs(r['prediction_eV'][0]-r['goal_eV'])<=.25 for r in values),
                           paired_dft=len(errors), mae_vs_dft_eV=np.abs(errors).mean(0).tolist() if errors else None))
        if arm!='frozen': item['training_costs']=read(root/f'training/{arm}/costs.json')
        result['arms'][arm]=item
    result['source_sha256']=inputs
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
