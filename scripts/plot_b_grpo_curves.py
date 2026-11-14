"""Plot frozen development curves and the fresh fixed-1024 validation comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    data = report.get('comparison', report)
    dev = data['results']['development']
    final = data['results']['validation']['1024']
    parents = data['results']['validation']['0']
    groups = [0,64,128,256,512,1024]
    colors = {'direct':'#2869a6', 'margin':'#c85c3a'}
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.titleweight':'bold', 'savefig.facecolor':'white'})
    fig, axes = plt.subplots(2,2,figsize=(12,8.3))
    for arm, label in [('direct','SFT → GRPO'), ('margin','Margin → GRPO')]:
        x = np.arange(len(groups))
        axes[0,0].plot(x,[100*dev[str(g)][arm]['hit_yield'] for g in groups],
                       'o-',color=colors[arm],label=label,lw=2)
        axes[0,1].plot(x,[dev[str(g)][arm]['mae_equal_pool_eV'] for g in groups],
                       'o-',color=colors[arm],label=label,lw=2)
    for ax in axes[0]:
        ax.set_xticks(range(len(groups)),groups);ax.set_xlabel('Cumulative GRPO groups')
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    axes[0,0].set(title='A  Development: raw hit yield',ylabel='Hits / all proposals (%)')
    axes[0,1].set(title='B  Development: target error',ylabel='MAE on evaluable molecules (eV)')
    values=[parents['direct'],final['direct'],parents['margin'],final['margin']]
    ax=axes[1,0]
    bars=ax.bar(range(4),[100*v['hit_yield'] for v in values],
                color=['#96bbdd',colors['direct'],'#edb7a5',colors['margin']],width=.65)
    for bar,v in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,
                f"{v['hits']}/{v['attempted']}\n{100*v['hit_yield']:.2f}%",ha='center',fontsize=9)
    ax.set_xticks(range(4),['Frozen\nSFT','SFT +\nGRPO','Frozen\nmargin','Margin +\nGRPO'])
    ax.set_ylim(0,max(b.get_height() for b in bars)*1.25)
    ax.set(title='C  Fresh validation at 1024 groups',ylabel='Hits / all proposals (%)')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    contrasts=[final['direct_gain'],final['margin_gain'],final['margin_minus_direct'],final['interaction']]
    labels=['GRPO gain from SFT','GRPO gain from margin','Combined vs direct GRPO','Difference of gains']
    ax=axes[1,1]
    bounds=[v for row in contrasts for v in row['paired_sampling_95_pp']]+[0.]
    padding=max(.5,(max(bounds)-min(bounds))*.25)
    ax.set_xlim(min(bounds)-padding,max(bounds)+padding)
    for i,row in enumerate(contrasts):
        point=row.get('difference_pp',row.get('difference_of_gains_pp'))
        lo,hi=row['paired_sampling_95_pp']
        ax.plot([lo,hi],[i,i],color='#454545',lw=2)
        ax.plot(point,i,'o',color=colors['margin'] if i else colors['direct'])
        ax.text(hi+padding*.1,i,f'{point:+.2f}',va='center',fontsize=9)
    ax.axvline(0,color='#999999',ls='--',lw=1)
    ax.set_yticks(range(4),labels);ax.invert_yaxis()
    ax.set(title='D  Paired differences in hit yield',xlabel='Percentage points; pointwise 95% sampling intervals')
    fig.suptitle('HOMO point control: margin pretraining and GRPO budget',fontsize=15,fontweight='bold')
    fig.text(.5,.015,'Frozen Uni-Mol2 defines reward and acceptance; ±0.25 eV. '
             '1 training seed; 7 targets × 2 sampling seeds × 128 proposals per checkpoint.\n'
             'Development selects checkpoints; panel C/D use fresh streams and fixed 1024 groups. '
             'Margin pretraining cost is additional. No new DFT.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.07,1,.95))
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:
        fig.savefig(args.output_prefix.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
