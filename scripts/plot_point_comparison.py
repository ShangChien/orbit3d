"""Plot raw-pool target response and coverage from completed pilot summaries."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.summary.read_text());target=data['target'];rows=data['rows']
    colors={'frozen':'#3465a4','continue400':'#777777','grpo200':'#168579','continue_wall':'#c47b26'}
    labels={'frozen':'Frozen SFT','continue400':'Continue SFT (400)','grpo200':'GRPO (200 groups)','continue_wall':'Continue SFT (time budget)'}
    fig,axes=plt.subplots(2,2,figsize=(10,7.5),layout='constrained')
    def metric(row,backend,key):
        return row['metrics'][backend]['in_domain']['targets'].get(target,{}).get(key,np.nan)
    def ranges(selected,backend,key,values):
        limits=[]
        for row,value in zip(selected,values):
            seeds=[m[backend]['in_domain']['targets'].get(target,{}).get(key,np.nan)
                   for m in row['per_inference_seed_metrics']]
            limits.append([value-np.nanmin(seeds),np.nanmax(seeds)-value])
        return np.asarray(limits).T
    for arm,color in colors.items():
        selected=sorted([r for r in rows if r['condition_control']=='correct' and
                         r['group'].startswith(f'eval_{arm}_p')],key=lambda r:r['scoring_target_eV'])
        if not selected:continue
        x=[r['scoring_target_eV'] for r in selected]
        for ax,backend in [(axes[0,0],'dpa'),(axes[0,1],'fingerprint')]:
            values=[metric(r,backend,'mean_prediction_eV') for r in selected]
            ax.errorbar(x,values,yerr=ranges(selected,backend,'mean_prediction_eV',values),fmt='-o',color=color,label=labels[arm],markersize=4,capsize=2,elinewidth=.7)
        values=[metric(r,'fingerprint','mae_eV') for r in selected]
        axes[1,0].errorbar(x,values,yerr=ranges(selected,'fingerprint','mae_eV',values),fmt='-o',color=color,markersize=4,capsize=2,elinewidth=.7)
        axes[1,1].plot(x,[r['metrics']['fingerprint']['in_domain']['coverage'] for r in selected],'-o',color=color,markersize=4)
    # Common unconditional pool is scored at each task target; no extra proposals.
    off=sorted([r for r in rows if r['condition_control']=='off'],key=lambda r:r['scoring_target_eV'])
    if off:
        for ax,backend in [(axes[0,0],'dpa'),(axes[0,1],'fingerprint')]:
            ax.plot([r['scoring_target_eV'] for r in off],[metric(r,backend,'mean_prediction_eV') for r in off],
                    ':',color='#333333',label='Frozen SFT, condition off')
    xvalues=[r['scoring_target_eV'] for r in rows if r['condition_control']=='correct']
    for ax in axes[0]:
        ax.plot([min(xvalues),max(xvalues)],[min(xvalues),max(xvalues)],'--',color='#bbbbbb',linewidth=1,label='Ideal mean response')
        ax.set_ylabel('Mean prediction (eV)')
    axes[0,0].set_title('Optimization oracle: DPA')
    axes[0,1].set_title('Independent fingerprint predictor')
    axes[1,0].set_title('Independent target error')
    axes[1,0].set_ylabel('Target MAE (eV)')
    axes[1,1].set_title('Evaluation coverage')
    axes[1,1].set_ylabel('In-domain evaluable / all proposals')
    axes[1,1].set_ylim(0,1)
    for ax in axes.flat:
        ax.set_xlabel(f'Requested {target.upper()} (eV)')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=8,loc='best')
    fig.suptitle(f'{target.upper()} point control — raw pools, before ranking or relaxation\n'
                 'Pilot: one SFT seed; bars show inference-seed ranges, not confidence intervals; OOD excluded',fontsize=10)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['png','pdf','svg']:
        fig.savefig(a.output.with_suffix('.'+extension),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
