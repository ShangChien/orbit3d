"""Show the difference between force convergence and complete endpoint admission."""
import argparse,json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--summaries',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reports=[json.loads(path.read_text()) for path in a.summaries]
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained',gridspec_kw={'width_ratios':[1.6,1]})
    labels=['All proposals','Initial valid','Force converged','Full endpoint gate','DPA target hit']
    for i,(report,color) in enumerate(zip(reports,['#477cad','#c57c30'])):
        counts=[len(report['records']),report['all']['initial_valid'],sum(r.get('optimization',{}).get('force_converged',False) for r in report['records']),report['all']['physical_accepted'],report['all']['joint_hits']]
        x=np.arange(5)+(i-.5)*.36
        bars=axes[0].bar(x,counts,width=.34,color=color,label=report['target'].upper())
        axes[0].bar_label(bars,fontsize=9,padding=3)
    axes[0].set_xticks(range(5),labels,rotation=15);axes[0].set_ylabel('Count / 192 proposals per property');axes[0].legend(frameon=False)
    reasons=collections.Counter(r['optimization']['geometry_error'].split(':')[0] for report in reports for r in report['records'] if 'optimization' in r and not r['physical_accepted'])
    reasons=dict(reasons.most_common())
    names={'bond_geometry':'Original bond length','connectivity_geometry':'Connectivity','stereo_geometry':'Stereo geometry'}
    bars=axes[1].barh([names[k] for k in reasons],list(reasons.values()),color='#8b959b')
    axes[1].bar_label(bars,padding=3);axes[1].set_xlim(0,max(reasons.values())*1.15);axes[1].set_xlabel('Endpoint geometry / identity failures');axes[1].invert_yaxis()
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen, unranked 384-proposal diagnostic\nDPA force convergence, endpoint admission and predicted target hits',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(a.output.with_suffix('.'+ext),dpi=180,bbox_inches='tight')


if __name__=='__main__':main()
