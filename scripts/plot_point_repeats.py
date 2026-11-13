"""Show paired GRPO effects without treating inference seeds as training repeats."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--summaries',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--evaluator-label',default='fingerprint');a=p.parse_args()
    runs=[json.loads(path.read_text()) for path in a.summaries]
    fig,axes=plt.subplots(len(runs),3,figsize=(11,3.6*len(runs)),squeeze=False,layout='constrained')
    controls=['frozen','continue400','continue_wall']
    labels=['Frozen\nSFT','Continue SFT\n400 updates','Continue SFT\ntime budget']
    columns=[('independent_target_mae_eV',1.,'Target MAE difference (eV)','Lower is better'),
             ('domain_rate',100.,'Evaluable-domain coverage difference\n(percentage points)','Higher is better'),
             ('independent_hit_rate',100.,'Property-hit difference\n(percentage points)','Higher is better')]
    colors=['#3465a4','#168579','#b97b3d']
    for i,run in enumerate(runs):
        for j,(key,scale,ylabel,direction) in enumerate(columns):
            ax=axes[i,j]
            for k,row in enumerate(run['per_training_seed']):
                offset=(k-(len(run['per_training_seed'])-1)/2)*.13
                ax.scatter(np.arange(3)+offset,[row['paired_grpo_minus_control'][control][key]*scale for control in controls],
                           color=colors[k%len(colors)],label=f"Training seed {row['training_seed']}",s=34,zorder=3)
            ax.scatter(np.arange(3),[run['paired_grpo_minus_control'][control][key]['mean']*scale for control in controls],
                       color='#202c36',marker='_',s=220,linewidths=2,label='Mean paired difference',zorder=4)
            ax.axhline(0.,color='#777777',linewidth=.8,linestyle='--')
            ax.set_xticks(range(3),labels)
            ax.set_ylabel(ylabel);ax.set_title(f"{run['target'].upper()} — {direction}")
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'GRPO minus each matched control — independent {a.evaluator_label}\nOne dot per independent training seed; all proposals retained in rate denominators',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['png','pdf','svg']:fig.savefig(a.output.with_suffix('.'+extension),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
