"""Raw-pool target response across independent training seeds and frozen evaluators."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--primary',nargs='+',type=Path,required=True)
    p.add_argument('--hyff',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    primary=[json.loads(p.read_text()) for p in a.primary];hyff=[json.loads(p.read_text()) for p in a.hyff]
    arms=['frozen','continue400','grpo200','continue_wall'];colors=['#3465a4','#777777','#168579','#c47b26']
    labels=['Frozen SFT','Continue SFT (400)','GRPO (200 groups)','Continue SFT (time budget)'];quantiles=['10','25','50','75','90']
    fig,axes=plt.subplots(2,3,figsize=(12,7.5),sharey='row',layout='constrained')
    seeds=set()
    for i,target in enumerate(['homo','lumo']):
        runs=[r for r in primary if r['target']==target];run_seeds={r['training_seed'] for r in runs};seeds|=run_seeds
        assert len(run_seeds)==len(runs) and run_seeds=={r['training_seed'] for r in hyff}
        for j,backend in enumerate(['dpa','fingerprint','hyff']):
            ax=axes[i,j]
            for arm,color,label in zip(arms,colors,labels):
                responses=[];goals=None
                for run in (hyff if backend=='hyff' else runs):
                    if backend=='hyff':
                        values=next(r for r in run['arms'] if r['target']==target and r['arm']==arm)['per_quantile']
                        values=sorted(values,key=lambda r:int(r['quantile']));x=[r['goal_eV'] for r in values];y=[r['mean_prediction_eV'] for r in values]
                    else:
                        values=[r for r in run['rows'] if r['condition_control']=='correct' and r['group'].startswith(f'eval_{arm}_p')]
                        values=sorted(values,key=lambda r:int(r['scoring_quantile']));assert [r['scoring_quantile'] for r in values]==quantiles
                        x=[r['scoring_target_eV'] for r in values];y=[r['metrics'][backend]['in_domain']['targets'][target]['mean_prediction_eV'] for r in values]
                    if goals is not None:assert goals==x
                    goals=x;responses.append(y)
                responses=np.array(responses);ax.plot(goals,responses.mean(0),'-o',color=color,label=label,markersize=3)
                if len(responses)>1:ax.fill_between(goals,responses.min(0),responses.max(0),color=color,alpha=.12,linewidth=0)
            off=[]
            for run in (hyff if backend=='hyff' else runs):
                if backend=='hyff':
                    values=next(r for r in run['condition_controls'] if r['target']==target and r['arm']=='frozen' and r['condition']=='off')['per_quantile']
                    off.append(values[0]['mean_prediction_eV'])
                else:
                    values=[r for r in run['rows'] if r['condition_control']=='off' and r['group']=='eval_frozen_off']
                    off.append(values[0]['metrics'][backend]['in_domain']['targets'][target]['mean_prediction_eV'])
            ax.plot([min(goals),max(goals)],[np.mean(off)]*2,':',color='#333333',label='Frozen, condition off')
            ax.plot([min(goals),max(goals)],[min(goals),max(goals)],'--',color='#bbbbbb',linewidth=1,label='Ideal mean response')
            ax.set_xlabel(f'Requested {target.upper()} (eV)');ax.set_ylabel('Mean prediction (eV)')
            ax.set_title({'dpa':'Optimization oracle: DPA','fingerprint':'Independent fingerprint','hyff':'Independent HyFF'}[backend])
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(f'Raw-pool target response — {len(seeds)} independent generator training seed(s)\nLines: seed means; shaded bands: training-seed ranges, not confidence intervals; original-domain evaluable cases',fontsize=11)
    for ext in ['svg','png','pdf']:fig.savefig(a.output.with_suffix('.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
