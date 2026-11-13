"""Render whole-proposal yield and DFT response with a single shared off baseline."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--correct',type=Path,required=True);p.add_argument('--off',type=Path,required=True);p.add_argument('--brics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    correct,off,brics=[json.loads(path.read_text()) for path in [a.correct,a.off,a.brics]]
    labels=['Frozen SFT','Time-matched SFT','Structure','Structure + physics','Full feedback','Shared off','BRICS off']
    arms=list(correct['arms'].values())+list(off['arms'].values())+list(brics['arms'].values())
    goals={str(r['quantile']):r['target_eV'] for r in correct['records']};qs=sorted(goals,key=int);x=[goals[q] for q in qs]
    colors=['#64748b','#3b82f6','#a855f7','#059669','#ea580c','#111827','#ca8a04']
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    ax=axes[0];positions=np.arange(len(arms));ax.barh(positions,[100*v['yield_equal_target'] for v in arms],color=colors)
    ax.set_yticks(positions,labels);ax.invert_yaxis();ax.set_xlabel('DFT hit yield / all proposals (%)');ax.set_title('Same terminal gate; target-average yield')
    for i,v in enumerate(arms):ax.text(100*v['yield_equal_target']+.25,i,f"{100*v['yield_equal_target']:.2f}%",va='center',fontsize=9)
    ax.set_xlim(0,max(100*v['yield_equal_target'] for v in arms)*1.25)
    ax=axes[1]
    for name,v,color in zip(labels,arms,colors):
        ax.plot(x,[v['cells'][q]['prediction_mean_eV'] for q in qs],marker='o',color=color,label=name,linestyle='--' if 'off' in name else '-')
    ax.plot(x,x,':',color='#9ca3af',label='Requested = mean');ax.set_xlabel('Requested HOMO (eV)');ax.set_ylabel('Mean DFT HOMO of successful endpoints (eV)');ax.set_title('Mean response is separate from hit yield');ax.legend(fontsize=7,loc='best')
    fig.suptitle('E13: one training seed; off pools reused across requests',fontsize=12)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)

if __name__=='__main__':main()
