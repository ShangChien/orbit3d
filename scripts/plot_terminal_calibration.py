"""Plot generated-geometry accuracy and native-source retention side by side."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert len(a.reports)==len(a.labels)
fig,axes=plt.subplots(len(a.reports),2,figsize=(10,3.8*len(a.reports)),squeeze=False,sharey=True,layout='constrained')
for i,(path,label) in enumerate(zip(a.reports,a.labels)):
 r=json.loads(path.read_text())
 for j,(dataset,title) in enumerate([('test_generated','Generated DFT holdout'),('source_test','Native source holdout')]):
  m=r['metrics'][dataset];ax=axes[i,j];x=np.arange(2)
  for k,(key,name,color) in enumerate([('baseline_mae_eV','Frozen parent','#64748b'),('affine_mae_eV','Affine calibration','#d97706'),('adapted_mae_eV','DFT-label adaptation','#059669')]):
   ax.bar(x+(k-1)*.24,m[key],width=.23,color=color,label=name)
   for pos,value in zip(x+(k-1)*.24,m[key]):ax.text(pos,value+.008,f'{value:.3f}',ha='center',fontsize=8)
  ax.set_xticks(x,['HOMO','LUMO']);ax.set_ylabel('Prediction MAE (eV)');ax.set_title(f'{label}: {title} (n={m["n"]})',fontsize=10);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
  if i==0 and j==1:ax.legend(fontsize=8,loc='upper right')
fig.suptitle('Terminal-label adaptation: accuracy gain and source retention',fontsize=13)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
