"""Publish version comparison, frozen-selection diversity, and square scientific plots."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def read(path):
    return json.loads(Path(path).read_text())


def mean_dict(values):
    return {key: float(np.mean([v[key] for v in values if v[key] is not None]))
            if any(v[key] is not None for v in values) else None for key in values[0]}


def square(ax, lo, hi):
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ticks = np.arange(np.ceil(lo), hi+.001, 2 if hi-lo > 6 else 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.plot([lo, hi], [lo, hi], '--', color='#777777', linewidth=1)
    ax.grid(alpha=.15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True, help='Downloaded v2 results')
    p.add_argument('--v1', type=Path, required=True, help='Downloaded v1 profile root')
    p.add_argument('--frozen', type=Path, required=True, help='Frozen candidate manifest root')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--quantum-v1', type=Path, required=True)
    p.add_argument('--quantum-v2', type=Path, required=True)
    a = p.parse_args()
    r = read(a.input/'profile_unimol2/summary.json')
    v1 = read(a.v1/'profile/summary.json')
    selection = read(a.input/'selection_unimol2/selection.json')
    selected = selection['selected_unimol']
    assert selected == read(a.input/'profile_unimol2/models_frozen.json')['selected_unimol']
    r['version_selection'] = selection
    r['evaluator'] = 'Uni-Mol2 '+selection['candidates'][selected]['model_size']
    r['training'] = {name: {'metadata': read(a.input/f'{name}_meta.json'),
                          'runtime': read(a.input/f'{name}_completed.json')}
                     for name in selection['candidates']}
    r['source_test_history'] = 'Same test previously reported for v1; v2 selected exclusively on development.'
    r['v1_test_metrics'] = v1['split_metrics']['test']['unimol']
    r['input_verification'] = read('docs/experiments/unimol2_input_verification_20261113.json')
    # Keep the canonical split manifest as the sole long ID list.
    r['dataset']['id_manifest'] = 'dual_oracle_data_manifest_20261112.json'
    for split in r['dataset']['splits'].values():
        split.pop('ids')
    a.figures.mkdir(exist_ok=True, parents=True)
    with np.load(a.input/'profile_unimol2/test_predictions.npz') as current, np.load(a.v1/'profile/test_predictions.npz') as old:
        np.testing.assert_array_equal(current['ids'], old['ids'])
        np.testing.assert_array_equal(current['labels'], old['labels'])
        np.testing.assert_allclose(current['dpa_common'], old['dpa_common'], atol=2e-5, rtol=0)
        labels = current['labels'].copy()
        predictions = [current['dpa_common'].copy(), old['unimol'].copy(), current['unimol'].copy()]
    names = ['DPA (matched training)', 'Uni-Mol v1', r['evaluator']]
    colors = ['#b64e39', '#7a7f88', '#2675a8']
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.4))
    bounds = []
    for k in range(2):
        values = np.concatenate([labels[:, k]]+[x[:, k] for x in predictions])
        bounds.append((np.floor(values.min())-1, np.ceil(values.max())+1))
    span = max(hi-lo for lo, hi in bounds)
    for k, prop in enumerate(['HOMO', 'LUMO']):
        for j, (name, pred, color) in enumerate(zip(names, predictions, colors)):
            ax = axes[k, j]
            ax.scatter(labels[:, k], pred[:, k], s=4, alpha=.18, color=color, rasterized=True)
            square(ax, bounds[k][0], bounds[k][0]+span)
            mae = np.mean(abs(labels[:, k]-pred[:, k]))
            slope = np.polyfit(labels[:, k], pred[:, k], 1)[0]
            ax.set_title(f'{name} | {prop}\nMAE {mae:.3f} eV; slope {slope:.3f}', fontsize=10)
            ax.set_xlabel('OE62 label (eV)')
            ax.set_ylabel('Prediction (eV)')
    fig.suptitle('Same 4,758 scaffold-held-out source molecules', fontsize=14)
    fig.text(.5, .025, 'Identical 8,170 training records. v1/DPA use explicit H; Uni-Mol2 uses source heavy-atom coordinates.\nModel selection uses development only; one training seed per configuration.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .065, 1, .95], h_pad=2.5)
    fig.savefig(a.figures/'oe62_unimol2_source_test.png', dpi=180)
    plt.close(fig)

    frozen = read(a.frozen/'candidate_snapshot/manifest.json')
    report = read(a.input/'reevaluation_unimol2/summary.json')
    r['geometry_diagnostic'] = report['geometry_diagnostic']
    bypool = defaultdict(dict)
    for line in (a.input/'reevaluation_unimol2/candidate_predictions.jsonl').read_text().splitlines():
        row = json.loads(line)
        bypool[row['pool']][row['candidate_id']] = row
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, scaffolds, distances = {}, {}, {}
    def diversity(rows):
        rows = [x for x in rows if x.get('valid')]
        smiles = sorted({x['smiles'] for x in rows})
        for s in smiles:
            if s not in fps:
                m = Chem.MolFromSmiles(s)
                fps[s] = fpgen.GetFingerprint(m)
                scaffolds[s] = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or '<ACYCLIC>'
        ds = []
        for i, s in enumerate(smiles):
            for t in smiles[:i]:
                key = (t, s)
                if key not in distances:
                    distances[key] = 1-DataStructs.TanimotoSimilarity(fps[t], fps[s])
                ds.append(distances[key])
        return {'molecules': len(rows), 'unique_fraction': len(smiles)/len(rows) if rows else None,
                'scaffolds': len({scaffolds[s] for s in smiles}),
                'fingerprint_distance': float(np.mean(ds)) if ds else None,
                'mean_atoms': float(np.mean([x['atoms'] for x in rows])) if rows else None}
    per_pool = []
    summaries = {x['id']: x for x in report['pools']}
    for pool in frozen['pools']:
        if not pool['group'].startswith(('response_calibration:', 'concentration:', 'main:')):
            continue
        if pool['group'].startswith('main:') and not any(f'_p{q}_' in pool['id'] for q in [25, 50, 75]):
            continue
        rows = bypool[pool['id']]
        summary = summaries[pool['id']]
        entry = {key: pool[key] for key in ['id', 'group', 'arm', 'target', 'seed', 'attempted']}
        entry['all_proposals'] = summary['diversity']
        entry['domain_raw'] = diversity([x for x in rows.values() if x.get('domain', {}).get('in_domain') and 'unimol_prediction' in x])
        entry['selections'] = {}
        for name, ids in pool['selections'].items():
            repetitions = ids if ids and isinstance(ids[0], list) else [ids]
            values = [diversity([rows[i] for i in selected_ids]) for selected_ids in repetitions]
            entry['selections'][name] = {'repetitions': len(repetitions), 'mean': mean_dict(values)}
        per_pool.append(entry)
    macro_diversity = {}
    for group, arm in sorted({(x['group'], x['arm']) for x in per_pool}):
        pools = [x for x in per_pool if (x['group'], x['arm']) == (group, arm)]
        def macro(getter):
            return mean_dict([mean_dict([getter(x) for x in pools if x['target'] == target])
                              for target in sorted({x['target'] for x in pools})])
        selected_views = set.intersection(*(set(x['selections']) for x in pools))
        macro_diversity[f'{group}/{arm}'] = {
            'attempted': sum(x['attempted'] for x in pools),
            'valid': sum(x['all_proposals']['valid'] for x in pools),
            'domain_raw': macro(lambda x: x['domain_raw']),
            'selections': {name: macro(lambda x: x['selections'][name]['mean']) for name in sorted(selected_views)},
        }
    r['selection_diversity'] = {'scope': 'All frozen selection repetitions; equal targets then seeds. Unique fraction, scaffold count and fingerprint distance reported together. No B-based selection.',
                               'macro': macro_diversity, 'per_pool': per_pool}
    # Original five-target response, unselected domain pool; uncertainty bars
    # show the molecule distribution width, not a confidence interval of its mean.
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    r['five_target_response'] = {}
    for k, prop in enumerate(['homo', 'lumo']):
        pools = [x for x in frozen['pools'] if x['group'] == f'campaign:{prop}' and x['arm'] == 'original']
        targets = sorted({x['target'] for x in pools})
        all_values = []
        responses = {}
        for field, title in [('prediction', 'DPA optimization model'), ('unimol_prediction', r['evaluator'])]:
            means, widths = [], []
            for target in targets:
                seed_values = [[x[field][k] for x in bypool[pool['id']].values()
                                if x.get('domain', {}).get('in_domain') and 'unimol_prediction' in x]
                               for pool in pools if pool['target'] == target]
                means.append(float(np.mean([np.mean(v) for v in seed_values])))
                widths.append(float(np.mean([np.std(v) for v in seed_values])))
            slope, intercept = np.polyfit(targets, means, 1)
            responses[field] = {'targets': targets, 'means': means, 'mean_within_seed_std': widths,
                                'slope': float(slope), 'intercept': float(intercept), 'label': title}
            all_values.extend(np.array(means)-widths)
            all_values.extend(np.array(means)+widths)
        lo = np.floor(min(targets+all_values)*2)/2-.25
        hi = np.ceil(max(targets+all_values)*2)/2+.25
        for j, (field, values) in enumerate(responses.items()):
            ax = axes[k, j]
            ax.errorbar(targets, values['means'], yerr=values['mean_within_seed_std'], fmt='o-', capsize=4,
                        color=colors[0 if j == 0 else 2])
            square(ax, lo, hi)
            ax.set_title(f'{prop.upper()} | {values["label"]}\nMean-response slope {values["slope"]:.3f}', fontsize=10)
            ax.set_xlabel('Requested target (eV)')
            ax.set_ylabel('Mean predicted property (eV)')
        r['five_target_response'][prop] = responses
    fig.suptitle('Fixed SFT: original five targets, three seeds each')
    fig.text(.5, .025, 'Raw domain pool; all-attempt validity reported separately. Bars: mean within-seed molecular SD, not mean-estimation uncertainty.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .055, 1, .95], h_pad=2.5)
    fig.savefig(a.figures/'oe62_unimol2_target_response.png', dpi=180)
    plt.close(fig)

    q1, q2 = read(a.quantum_v1), read(a.quantum_v2)
    oldq = {x['id']: x for x in q1['records']}
    qs = [x for x in q2['records'] if x['graph_preserved']]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.4))
    all_values = []
    plotdata = {}
    for i, stage in enumerate(['input', 'optimized']):
        quantum = np.array([x[f'quantum_{stage}'][x['property_index']] for x in qs])
        preds = [np.array([x[f'{stage}_models']['dpa_historical'][x['property_index']] for x in qs]),
                 np.array([oldq[x['id']][f'{stage}_models']['unimol'][x['property_index']] for x in qs]),
                 np.array([x[f'{stage}_models']['unimol'][x['property_index']] for x in qs])]
        plotdata[stage] = quantum, preds
        all_values.extend(quantum)
        for pred in preds:
            all_values.extend(pred)
    lo, hi = np.floor(min(all_values))-1, np.ceil(max(all_values))+1
    for i, (stage, (quantum, preds)) in enumerate(plotdata.items()):
        for j, (name, pred) in enumerate(zip(['Historical DPA A', 'Uni-Mol v1', r['evaluator']], preds)):
            ax = axes[i, j]
            for prop, marker in [('homo', 'o'), ('lumo', '^')]:
                mask = np.array([x['property'] == prop for x in qs])
                ax.scatter(quantum[mask], pred[mask], s=35, marker=marker, color=colors[j], label=prop.upper())
            square(ax, lo, hi)
            ax.set_title(f'{name} | {stage} geometry\nMAE vs quantum {np.mean(abs(pred-quantum)):.3f} eV', fontsize=10)
            ax.set_xlabel('GPU4PySCF property (eV)')
            ax.set_ylabel('Model prediction (eV)')
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Same 19 identity-preserving molecules before / after geometry optimization')
    fig.text(.5, .025, 'Exploratory frozen cohort: 24 planned, 4 identity failures, 1 optimization failure. Approximate OE62 quantum protocol.\nOnly each molecule\'s requested property is plotted; failed candidates remain in the planned denominator.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .065, 1, .95], h_pad=2.5)
    fig.savefig(a.figures/'oe62_unimol2_quantum_comparison.png', dpi=180)
    plt.close(fig)
    r['quantum_summary'] = {'scope': q2['scope'], 'all': q2['all'], 'groups': q2['groups'],
                            'full_evidence': a.quantum_v2.name}
    r['evidence_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
        a.input/'profile_unimol2/summary.json', a.input/'selection_unimol2/selection.json',
        a.input/'reevaluation_unimol2/summary.json', a.quantum_v2]}
    a.output.write_text(json.dumps(r, indent=2)+'\n')
    print('Selected:', r['evaluator'])
    print('Test MAE:', r['split_metrics']['test']['unimol']['mae'])
    print('Response slopes:', {p: {m: v['slope'] for m, v in d.items()} for p, d in r['five_target_response'].items()})


if __name__ == '__main__':
    main()
