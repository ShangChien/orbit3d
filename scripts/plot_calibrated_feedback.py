"""Plot fixed E16 supply, quantum target precision and condition response separately."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=json.loads(args.report.read_text())
    arms=['frozen','continue_wall','native_critic','calibrated_critic']
    labels=['Frozen SFT','Time SFT','Native critic','DFT-cal. critic']
    colors=['#64748b','#3b82f6','#d97706','#059669'];x=np.arange(4)
    rows=[report['arms'][arm]['quantum'] for arm in arms]
    off=report['condition_vs_off']['arms']['frozen']
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    panels=[(axes[0,0],[100*r['yield_equal_target'] for r in rows],'DFT hits / all proposals (%)','Whole-proposal target yield',100*off['shared_off_yield']),
            (axes[0,1],[100*r['terminal_selected']/r['proposals'] for r in rows],'Accepted endpoints / proposals (%)','Terminal supply',None),
            (axes[1,0],[r['mae_equal_target_eV'] for r in rows],'Target MAE (eV)','Accuracy on quantum-evaluated endpoints',off['off_mae_eV'])]
    for ax,values,ylabel,title,reference in panels:
        ax.bar(x,values,color=colors,width=.65)
        top=max(values+[reference] if reference is not None else values)
        ax.set_ylim(0,max(top*1.23,.05));ax.set_xticks(x,labels,fontsize=8)
        ax.set_ylabel(ylabel);ax.set_title(title,fontsize=11)
        for pos,value in zip(x,values):ax.text(pos,value+max(top*.025,.001),f'{value:.3f}' if ylabel.startswith('Target') else f'{value:.2f}',ha='center',fontsize=9)
        if reference is not None:
            ax.axhline(reference,ls='--',lw=1.2,color='#475569',label='Shared off (rescored)');ax.legend(fontsize=8,loc='upper right')
    ax=axes[1,1];quantiles=sorted(report['protocol']['evaluation_targets'],key=int)
    goals=[report['protocol']['evaluation_targets'][q]['homo_eV'] for q in quantiles]
    ax.plot(goals,goals,'--',lw=1,color='#475569',label='Ideal mean')
    for row,label,color in zip(rows,labels,colors):
        means=[row['cells'][q]['prediction_mean_eV'] for q in quantiles]
        ax.plot(goals,means,'o-',color=color,label=label,markersize=5)
        ids=[i for i,q in enumerate(quantiles) if not report['protocol']['evaluation_targets'][q]['used_in_grpo_rotation']]
        ax.scatter(np.asarray(goals)[ids],np.asarray(means)[ids],facecolors='white',edgecolors=color,zorder=4,s=27)
    ax.set_xlabel('Requested HOMO (eV)');ax.set_ylabel('DFT endpoint mean (eV)')
    ax.set_title('Response; hollow markers: GRPO interpolation',fontsize=10)
    ax.legend(fontsize=8)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('E16: evaluator calibration and terminal-feedback generation',fontsize=13)
    fig.savefig(args.output.with_suffix('.png'),dpi=180)
    fig.savefig(args.output.with_suffix('.svg'))


if __name__=='__main__':main()
