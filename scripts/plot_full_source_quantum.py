"""Show data-size/step comparisons without conflating response and accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads(a.report.read_text())['reports'];arms=['subset8192_5000','full43951_5000','full43951_27000'];labels=['8,192 / 5k','43,951 / 5k','43,951 / 5k+22k'];colors=['#64748b','#d97706','#059669']
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for i,target in enumerate(['homo','lumo']):
 correct=d[f'e14_{target}_correct'];off=d[f'e14_{target}_off'];ax=axes[i,0];x=np.arange(3)
 for offset,key,label,color in [(-.18,correct,'Correct condition','#0f766e'),(.18,off,'Own condition-off pool','#94a3b8')]:
  values=[100*key['arms'][arm]['yield_equal_target'] for arm in arms];ax.bar(x+offset,values,.34,color=color,label=label)
  for pos,value in zip(x+offset,values):ax.text(pos,value+.15,f'{value:.2f}',ha='center',fontsize=9)
 ax.set_xticks(x,labels);ax.set_ylim(0,12);ax.set_ylabel('DFT hits / all proposals (%)');ax.set_title(target.upper()+': whole-pool yield');ax.legend(fontsize=8)
 ax=axes[i,1];goals=sorted({r['target_eV'] for r in correct['records']});ax.plot(goals,goals,ls='--',color='#334155',lw=1,label='Ideal mean')
 for arm,label,color in zip(arms,labels,colors):
  cells=correct['arms'][arm]['cells'];ax.plot(goals,[cells[str(q)]['prediction_mean_eV'] for q in [10,25,50,75,90]],'o-',color=color,label=label)
 ax.set_xlabel('Requested target (eV)');ax.set_ylabel('DFT endpoint mean (eV)');ax.set_title(target.upper()+': response on accepted endpoints');ax.legend(fontsize=8)
for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
fig.suptitle('E14: more data and updates do not consistently improve point accuracy',fontsize=13)
fig.savefig(a.output.with_suffix('.png'),dpi=180);fig.savefig(a.output.with_suffix('.pdf'))
