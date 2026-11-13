"""Display frozen pilot responses and width effects without treating them as training repeats."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--summaries',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--evaluator',default='unimol2')
    a=p.parse_args();runs=[json.loads(path.read_text()) for path in a.summaries]
    arms=['sft','contrast','hindsight','both','point_frozen','point_grpo']
    names={'sft':'Gaussian SFT','contrast':'+ Condition contrast','hindsight':'+ Hindsight','both':'+ Both',
           'point_frozen':'Point SFT reference','point_grpo':'Point GRPO reference'}
    colors=['#3465a4','#c37b28','#308774','#9262a5','#6b7176','#ad5555']
    widths=[.1,.25,.5]
    fig,axes=plt.subplots(len(runs),3,figsize=(13.5,3.5*len(runs)),squeeze=False,layout='constrained')
    for i,run in enumerate(runs):
        for j,width in enumerate(widths):
            ax=axes[i,j]
            for arm,color in zip(arms,colors):
                selected=[r for r in run['cells'] if r['arm']==arm and r['control']=='correct' and r['preference_sigma_eV']==(None if arm.startswith('point_') else width)]
                selected.sort(key=lambda r:int(r['quantile']))
                x=[r['goal_eV'] for r in selected];y=[r['metrics'][a.evaluator]['mean_prediction_eV'] for r in selected]
                ax.plot(x,y,'--' if arm.startswith('point_') else '-o',color=color,label=names[arm],markersize=3,linewidth=1.3)
            ax.plot([min(x),max(x)],[min(x),max(x)],':',color='#b4b9bc',linewidth=1)
            ax.set_title(f"{run['target'].upper()} / preference width {width:.2f} eV")
            ax.set_xlabel('Requested mean (eV)');ax.set_ylabel('Mean prediction (eV)')
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=3,fontsize=9,frameon=False)
    fig.suptitle(f'Conditional response / {a.evaluator}\nOne training seed; five fixed targets; point references have no width input',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(a.output.with_name(a.output.name+'_response').with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    metrics=[('target_mae_eV','Target MAE (eV)',1),('hit_rate_fixed_025','Hit yield at ±0.25 eV (%)',100),
             ('coverage','Evaluable coverage (%)',100),('mean_prediction_std_eV','Within-target prediction SD (eV)',1)]
    fig,axes=plt.subplots(len(runs),4,figsize=(16,3.5*len(runs)),squeeze=False,layout='constrained')
    for i,run in enumerate(runs):
        for j,(metric,label,scale) in enumerate(metrics):
            ax=axes[i,j]
            for arm,color in zip(arms,colors):
                values=[]
                for width in widths:
                    r=next(r for r in run['aggregates'] if r['arm']==arm and r['control']=='correct' and r['evaluator']==a.evaluator and r['preference_sigma_eV']==(None if arm.startswith('point_') else width))
                    values.append(None if r[metric] is None else r[metric]*scale)
                ax.plot(widths,values,'--' if arm.startswith('point_') else '-o',color=color,label=names[arm],markersize=3,linewidth=1.3)
            ax.set_title(run['target'].upper());ax.set_xlabel('Preference width (eV)');ax.set_ylabel(label)
            ax.set_xticks(widths);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=3,fontsize=9,frameon=False)
    fig.suptitle(f'Width response / {a.evaluator} / one training seed\nMAE and SD: evaluable subset, equal target weighting; hit yield and coverage: all proposals',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(a.output.with_name(a.output.name+'_width').with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
