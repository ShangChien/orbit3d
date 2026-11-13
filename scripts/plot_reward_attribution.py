"""Plot paired yield attribution with training seeds as the repeat unit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries', nargs='+', type=Path, required=True)
    parser.add_argument('--evaluator', default='unimol2')
    parser.add_argument('--controls', nargs='+', default=['eligibility200', 'continue_wall'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.summaries]
    labels = {'eligibility200': 'Eligibility-only GRPO', 'continue_wall': 'Time-matched SFT',
              'grpo200': 'Original point GRPO', 'frozen': 'Frozen SFT',
              'conditional200': 'Separated-advantage GRPO'}
    measures = [('supply_component', 'Evaluable supply', '#327da8'),
                ('conditional_hit_component', 'Conditional hit fraction', '#c17936'),
                ('yield_delta', 'Total hit yield', '#273f42')]
    fig, axes = plt.subplots(len(runs), len(args.controls),
        figsize=(5.0 * len(args.controls), 3.5 * len(runs)), squeeze=False, layout='constrained')
    for i, run in enumerate(runs):
        for j, control in enumerate(args.controls):
            ax = axes[i, j]
            rows = [row for row in run['comparisons']
                    if row['evaluator'] == args.evaluator and row['control'] == control]
            if not rows or any(row['mean'][key] is None for row in rows for key, _, _ in measures):
                raise ValueError('Attribution requires nonempty evaluable subsets in every paired cell')
            values = np.array([[row['mean'][key] * 100 for key, _, _ in measures] for row in rows])
            np.testing.assert_allclose(values[:, 0] + values[:, 1], values[:, 2], atol=1e-10)
            ax.barh(range(3), values.mean(axis=0), color=[color for _, _, color in measures], alpha=.8)
            for k, row in enumerate(rows):
                offset = (k - (len(rows) - 1) / 2) * .13
                ax.scatter(values[k], np.arange(3) + offset, facecolors='white', edgecolors='#263744',
                           s=28, zorder=3, label=f"Train seed {row['training_seed']}")
            ax.axvline(0, color='#61717b', linewidth=.8)
            ax.set_yticks(range(3), [name for _, name, _ in measures])
            ax.invert_yaxis()
            ax.set_xlabel('Paired difference (percentage points)')
            ax.set_title(f"{run['target'].upper()} vs {labels.get(control, control)}")
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.grid(axis='x', alpha=.15)
    treatments = {run['treatment'] for run in runs}
    if len(treatments) != 1:
        raise ValueError('Use one treatment per figure')
    treatment = treatments.pop()
    fig.suptitle(f"{labels.get(treatment, treatment)}: where does hit yield change?\n"
                 f"{args.evaluator} evaluation; bars: training-seed means, dots: individual seeds", fontsize=12)
    fig.supxlabel('Exact accounting identity within target strata; not a causal mediation estimate.', fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(args.output.with_suffix('.' + extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
