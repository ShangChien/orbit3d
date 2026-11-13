"""Plot conditioner response and paired tradeoffs across independent training seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = ['point_frozen', 'point_sft', 'point_contrast', 'point_time']
LABELS = ['Frozen SFT', 'Continued SFT (1000)', 'Point contrast (1000)', 'Time-matched SFT']
COLORS = ['#7d858a', '#477cad', '#c57c30', '#2d8d79']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summaries', nargs='+', type=Path, required=True)
    p.add_argument('--repeats', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = [json.loads(path.read_text()) for path in a.summaries]
    repeats = [json.loads(path.read_text()) for path in a.repeats]
    targets = [r['target'] for r in repeats]
    evaluators = ['dpa', 'fingerprint', 'unimol2']
    fig, axes = plt.subplots(len(targets), 3, figsize=(13.5, 3.6 * len(targets)), squeeze=False, layout='constrained')
    for i, target in enumerate(targets):
        source = [r for r in runs if r['target'] == target]
        for j, evaluator in enumerate(evaluators):
            ax = axes[i, j]
            for arm, label, color in zip(ARMS, LABELS, COLORS):
                values = []
                for run in source:
                    cells = sorted([c for c in run['cells'] if c['control'] == 'correct' and c['arm'] == arm], key=lambda c: int(c['quantile']))
                    x = np.array([c['goal_eV'] for c in cells])
                    values.append([c['metrics'][evaluator]['mean_prediction_eV'] for c in cells])
                values = np.asarray(values)
                ax.plot(x, values.mean(0), '-o', color=color, label=label, markersize=3, linewidth=1.4)
                ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=.1)
            ax.plot([x.min(), x.max()], [x.min(), x.max()], ':', color='#aeb5b9', linewidth=1)
            ax.set_title(f'{target.upper()} / {evaluator}')
            ax.set_xlabel('Requested target (eV)'); ax.set_ylabel('Mean property prediction (eV)')
            ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.015), ncol=4, frameon=False)
    fig.suptitle('Point-condition response\nLines: training-seed means; shading: seed range, not confidence intervals', fontsize=12)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']: fig.savefig(a.output.with_name(a.output.name + '_response').with_suffix('.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    metrics = [('target_mae_eV', 'Molecule target MAE (eV)', 1),
               ('mean_absolute_response_bias_eV', 'Mean response bias (eV)', 1),
               ('mean_prediction_std_eV', 'Within-target SD (eV)', 1),
               ('hit_rate_fixed_025', 'Hit yield at ±0.25 eV (%)', 100)]
    fig, axes = plt.subplots(len(targets), 4, figsize=(15, 3.5 * len(targets)), squeeze=False, layout='constrained')
    for i, report in enumerate(repeats):
        for j, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[i, j]
            for k, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS)):
                values = np.asarray(report['arms'][arm]['evaluators']['unimol2'][metric]['values']) * scale
                ax.scatter(np.repeat(k, len(values)), values, color=color, s=22, alpha=.6)
                ax.plot([k - .18, k + .18], [values.mean()] * 2, color=color, linewidth=2.5)
            ax.set_xticks(range(4), ['Frozen', 'SFT', 'Contrast', 'Time SFT'], rotation=20)
            ax.set_title(report['target'].upper()); ax.set_ylabel(ylabel)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15)
    fig.suptitle('Frozen Uni-Mol2 / paired training-seed results\nErrors: evaluable subset, equal target weights; hit yield: all proposals', fontsize=12)
    for ext in ['png', 'pdf', 'svg']: fig.savefig(a.output.with_name(a.output.name + '_tradeoff').with_suffix('.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__': main()
