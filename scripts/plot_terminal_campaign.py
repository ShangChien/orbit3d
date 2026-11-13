"""Visualize supply, endpoint target control and cost on the fixed five-arm pilot."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.summary.read_text())['arms'];arms=['point_frozen','terminal_sft','terminal_structure','terminal_physical','terminal_property']
    labels=['Frozen SFT','Time SFT','Structure','+ Physics','+ Target'];colors=['#8b959b','#477cad','#8c6ca3','#398b78','#c57c30']
    metrics=[('Initial valid (%)',lambda r:100*r['initial']['rate']),
             ('DPA endpoint accepted (%)',lambda r:100*r['terminal']['rate']),
             ('B joint hit yield / all proposals (%)',lambda r:100*r['b_terminal']['hit_yield']['0.25']),
             ('B target MAE / evaluated endpoints (eV)',lambda r:r['b_terminal']['mae_equal_target_eV']),
             ('Median atoms / all initial valid',lambda r:r['initial']['atoms_p10_p50_p90'][1] if r['initial']['atoms_p10_p50_p90'] else None),
             ('Median initial RMS force (eV/A)',lambda r:r['initial_rms_force_p10_p50_p90'][1] if r['initial_rms_force_p10_p50_p90'] else None),
             ('Terminal seconds / accepted endpoint',lambda r:r['terminal_seconds']/r['terminal']['count'] if r['terminal']['count'] else None),
             ('Additional training pipeline minutes',lambda r:r.get('training_costs',{}).get('pipeline_seconds',0)/60)]
    fig,axes=plt.subplots(2,4,figsize=(16,7.2),layout='constrained')
    for ax,(title,get) in zip(axes.flat,metrics):
        values=[get(data[arm]) for arm in arms];numeric=[np.nan if v is None else v for v in values]
        bars=ax.bar(range(5),numeric,color=colors,width=.7)
        ax.bar_label(bars,labels=['NA' if v is None else f'{v:.2f}' for v in values],padding=3,fontsize=8)
        ax.set_xticks(range(5),labels,rotation=28,ha='right',fontsize=8);ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
        lo,hi=ax.get_ylim();ax.set_ylim(lo,hi*1.12 if hi>0 else 1)
    fig.suptitle('Terminal feedback / one training seed / 384 primary proposals per arm\nSame postprocessing; B is cross-model evidence, not quantum truth; target tolerance ±0.25 eV',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(a.output.with_suffix('.'+ext),dpi=180,bbox_inches='tight')


if __name__=='__main__':main()
