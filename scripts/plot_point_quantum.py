"""Paired quantum target errors for every preselected policy candidate."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.summary.read_text());fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained',sharey=True)
    for row,target in enumerate(['homo','lumo']):
        k=['homo','lumo'].index(target)
        for col,arm in enumerate(['continue_wall','grpo200']):
            ax=axes[row,col];data=[v for v in r['records'] if v['target']==target and v['arm']==arm]
            for v in data:
                if 'singlepoint' not in v:continue
                e0=abs(v['singlepoint']['prediction_eV'][k]-v['target_eV'])
                color='#3465a4' if v['selection']=='top1' else '#b97b3d'
                ax.scatter(0,e0,color=color,s=20)
                if 'optimized_singlepoint' in v:
                    e1=abs(v['optimized_singlepoint']['prediction_eV'][k]-v['target_eV'])
                    accepted=v['geometry']['converged'] and v['geometry']['graph_preserved']
                    ax.plot([0,1],[e0,e1],color=color,alpha=.45,lw=.9)
                    ax.scatter(1,e1,color=color,marker='o' if accepted else 'x',s=30)
            ax.axhline(.25,color='#777',ls='--',lw=.8);ax.set_xticks([0,1],['Export geometry','DFT optimized'])
            ax.set_xlim(-.15,1.15);ax.set_ylabel('Absolute target error (eV)');ax.grid(axis='y',alpha=.15)
            ax.set_title(target.upper()+' / '+('Time-matched SFT' if arm=='continue_wall' else 'GRPO 200'))
            ax.spines[['top','right']].set_visible(False)
    for label,c in [('DPA top1','#3465a4'),('Fixed hash sample','#b97b3d')]:axes[0,0].plot([],[],color=c,label=label)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Frozen 24-candidate cohort: PBE0/def2-TZVP target error\nPBE-D3BJ geometry; crosses = nonconverged or changed graph; not a population success estimate',fontsize=10)
    for ext in ['png','pdf','svg']:fig.savefig(a.output.with_suffix('.'+ext),dpi=180)
    svg=a.output.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
