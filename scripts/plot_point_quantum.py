"""Paired quantum target errors for every preselected policy candidate."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.summary.read_text());arms=r['selection'].get('arms',list(dict.fromkeys(v['arm'] for v in r['records'])))
    optimize=r.get('optimization_requested',True)
    names={'continue_wall':'Time-matched SFT','grpo200':'Original point GRPO','eligibility200':'Eligibility-only GRPO','conditional200':'Separated-advantage GRPO'}
    fig,axes=plt.subplots(2,len(arms),figsize=(4.5*len(arms),6),layout='constrained',sharey=True,squeeze=False)
    for row,target in enumerate(['homo','lumo']):
        k=['homo','lumo'].index(target)
        for col,arm in enumerate(arms):
            ax=axes[row,col];data=[v for v in r['records'] if v['target']==target and v['arm']==arm]
            for j,v in enumerate(data):
                if 'singlepoint' not in v:continue
                e0=abs(v['singlepoint']['prediction_eV'][k]-v['target_eV'])
                color='#3465a4' if v['selection']=='top1' else '#b97b3d'
                offset=(j-(len(data)-1)/2)*min(.015,.20/max(len(data)-1,1));ax.scatter(offset,e0,color=color,s=20,zorder=3)
                if 'optimized_singlepoint' in v:
                    e1=abs(v['optimized_singlepoint']['prediction_eV'][k]-v['target_eV'])
                    accepted=v['geometry']['converged'] and v['geometry']['graph_preserved']
                    ax.plot([offset,1+offset],[e0,e1],color=color,alpha=.45,lw=.9)
                    ax.scatter(1+offset,e1,color=color,marker='o' if accepted else 'x',s=24 if accepted else 55,zorder=3 if accepted else 4)
            ax.axhline(.25,color='#777',ls='--',lw=.8)
            ax.set_xticks([0,1] if optimize else [0],['Export geometry','DFT optimized'] if optimize else ['Export geometry'])
            ax.set_xlim(-.15,1.15 if optimize else .15);ax.set_ylabel('Absolute target error (eV)');ax.grid(axis='y',alpha=.15)
            counts=r['groups'][f'{target}/{arm}'];coverage=f"SP {counts['singlepoint_completed']}/{counts['planned']}"
            if optimize:coverage+=f"; accepted optimized {counts['accepted_optimized']}/{counts['planned']}"
            ax.set_title(target.upper()+' / '+names.get(arm,arm)+'\n'+coverage,fontsize=10)
            ax.spines[['top','right']].set_visible(False)
    for label,c in [('DPA top1','#3465a4'),('Fixed hash sample','#b97b3d')]:axes[0,0].plot([],[],color=c,label=label)
    axes[0,0].legend(fontsize=8,frameon=False)
    detail='PBE-D3BJ geometry; crosses = nonconverged or changed graph' if optimize else 'Input-geometry singlepoints; optimization not requested'
    fig.suptitle(f"Frozen {r['selection']['expected']}-slot cohort: PBE0/def2-TZVP target error\n{detail}; not a population success estimate",fontsize=10)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(a.output.with_suffix('.'+ext),dpi=180)
    svg=a.output.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
