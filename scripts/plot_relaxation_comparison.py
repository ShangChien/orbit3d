"""Plot chemical admission, force, independent error and actual optimization cost."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.summary.read_text());methods=data['methods'];records=data['records']
    labels={'none':'None','full':'Current Full','energy_current':'Current energy','lbfgs':'LBFGS','fire':'FIRE'}
    colors=['#6c757d','#bd6c3c','#3465a4','#168579','#8263a4'];x=np.arange(len(methods))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    stats=[next(r for r in data['aggregate'] if r['target']=='all' and r['method']==m) for m in methods]
    for j,(key,label,color) in enumerate([('graph_preserved','Graph preserved','#b4bcc4'),('graph_and_rms_below_0.1','Graph + RMS force <= 0.1','#3465a4'),('joint_success','Graph + force + both property models','#168579')]):
        axes[0,0].bar(x+(j-1)*.25,[r[key] for r in stats],width=.24,label=label,color=color)
    axes[0,0].set_ylim(0,data['executed_samples']*1.13);axes[0,0].set_ylabel(f'Cases / {data["executed_samples"]} frozen starts');axes[0,0].legend(fontsize=8,loc='upper right')
    for i,m in enumerate(methods):
        rows=[r for r in records if r['method']==m]
        for valid,marker in [(True,'o'),(False,'x')]:
            selected=[r for r in rows if r['graph_preserved']==valid and r['numerical_final_rms_force_eVA'] is not None]
            axes[0,1].scatter(np.full(len(selected),i)+np.linspace(-.13,.13,len(selected)),[r['numerical_final_rms_force_eVA'] for r in selected],color=colors[i],marker=marker,alpha=.65,s=16)
        errors=[r['hyff_target_error_eV']-r['hyff_initial_target_error_eV'] for r in rows if r['completed'] and r['graph_preserved']]
        axes[1,0].scatter(np.full(len(errors),i)+np.linspace(-.13,.13,len(errors)),errors,color=colors[i],s=14,alpha=.5)
        if errors:axes[1,0].plot([i-.17,i+.17],[np.mean(errors)]*2,color='black',lw=2)
        axes[1,1].bar(i,stats[i]['energy_force_calls']/stats[i]['attempted'],color=colors[i],alpha=.75)
    axes[0,1].axhline(.1,color='black',linestyle='--',lw=1);axes[0,1].set_yscale('log');axes[0,1].set_ylabel('Final DPA RMS force (eV / Å)')
    axes[0,1].set_title('All numerical endpoints; x = graph rejected')
    axes[1,0].axhline(0,color='black',linestyle='--',lw=1);axes[1,0].set_ylabel('HyFF target absolute-error change (eV)');axes[1,0].set_title('Graph-preserved evaluable cases; line = mean')
    axes[1,1].set_ylabel('Mean optimization energy/force calls');axes[1,1].set_title('Full also uses property gradients; budgets differ')
    for ax in axes.flat:
        ax.set_xticks(x,[labels[m] for m in methods],rotation=15);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle(f'Same starting geometries: force convergence, graph admission and independent target error\n{data["executed_samples"]} frozen seed42 proposals; {data["device"]}, {data["threads"]} threads; proxy predictions, not quantum validation',fontsize=11)
    for ext in ['svg','png','pdf']:fig.savefig(a.output.with_suffix('.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
