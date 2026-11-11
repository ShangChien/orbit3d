"""Plot the observed training-seed variation from the complete public matrix."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

paper=Path(__file__).resolve().parent
source=paper.parent/'docs/experiments/grpo_factorial_20261112.json'
data=json.loads(source.read_text())
variants=['full','no_kl','no_physics','no_replay','uncertainty_on']
labels=['Full','No KL','No physics','No replay','Uncert. on']
colors=['#287B86','#C37439','#7A629E']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
                     'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(7.6,3.0),sharey=True,layout='constrained')
for ax,metric,title in zip(axes,['optimizing_joint','independent_joint'],['Optimizing property model','Independent fingerprint model']):
    for x,variant in enumerate(variants):
        records=data['aggregate'][variant]['per_training_seed']
        assert [r['training_seed'] for r in records]==[42,43,44]
        values=np.array([100*r[metric]/r['attempted'] for r in records])
        for i,(value,color) in enumerate(zip(values,colors)):
            ax.scatter(x+(i-1)*.11,value,s=34,color=color,edgecolors='white',linewidths=.5,
                       label=f'Train seed {42+i}' if x==0 and ax is axes[0] else None,zorder=3)
        ax.plot([x-.23,x+.23],[values.mean()]*2,color='#263746',linewidth=1.7,
                label='Mean' if x==0 and ax is axes[0] else None,zorder=2)
    ax.set_title(title,loc='left',pad=12)
    ax.set_xticks(range(5),labels,rotation=20,ha='right')
    ax.set_ylim(0,5.3);ax.set_yticks(range(6));ax.grid(axis='y',color='#E1E6EB',linewidth=.6)
    ax.set_axisbelow(True);ax.tick_params(length=0)
axes[0].set_ylabel('Joint success / all proposals (%)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=4,frameon=False,
           handlelength=1.5,columnspacing=1.6)
stem=paper/'figures/fig1_grpo_training_seeds'
for ext in ['pdf','svg','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=220)
plt.close(fig)
