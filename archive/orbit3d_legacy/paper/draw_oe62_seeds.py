"""Publication-size paired GRPO effects under the frozen independent HyFF head."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--homo',type=Path,required=True);p.add_argument('--lumo',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'figures/fig2_oe62_paired_effects');a=p.parse_args()
    runs=[json.loads(path.read_text()) for path in [a.homo,a.lumo]]
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.7),layout='constrained')
    controls=['frozen','continue400','continue_wall'];colors=['#3465a4','#168579','#b97b3d']
    for i,run in enumerate(runs):
        assert sorted(run['training_seeds'])==[42,43,44] and run['attempts_per_arm_per_training_seed']==1920
        for j,(metric,scale,label) in enumerate([('independent_target_mae_eV',1.,'Target MAE difference (eV)'),('independent_hit_rate',100.,'Property-hit difference\n(percentage points)')]):
            ax=axes[i,j]
            for k,row in enumerate(sorted(run['per_training_seed'],key=lambda r:r['training_seed'])):
                ax.scatter(np.arange(3)+(k-1)*.1,[row['paired_grpo_minus_control'][c][metric]*scale for c in controls],
                           color=colors[k],s=20,label=f"Seed {row['training_seed']}",zorder=3)
            ax.scatter(np.arange(3),[run['paired_grpo_minus_control'][c][metric]['mean']*scale for c in controls],
                       marker='_',s=130,color='#202c36',linewidths=1.5,label='Mean',zorder=4)
            ax.axhline(0,color='#777777',ls='--',lw=.7);ax.set_xticks(range(3),['Frozen\nSFT','Continue\nSFT','Time-budget\nSFT'])
            ax.set_ylabel(label);ax.set_title(run['target'].upper()+(' (lower is better)' if j==0 else ' (higher is better)'))
            ax.grid(axis='y',alpha=.17);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(ncol=2,loc='best',frameon=False)
    fig.suptitle('GRPO minus each control: independent HyFF assessment',fontsize=10)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(a.output.with_suffix('.'+ext),dpi=220)
    plt.close(fig)


if __name__=='__main__':main()
