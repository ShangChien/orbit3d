"""Publish the frozen source-test profile and approximate-protocol reference comparison."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path,required=True);a=p.parse_args()
r=json.loads((a.input/'profile/summary.json').read_text());refs=json.loads((a.input/'quantum_reference/manifest.json').read_text());records=[]
for item in refs['records']:
 d=json.loads((a.input/'quantum_reference/results'/f'job{int(item["id"][1:]):02d}'/'result.json').read_text());records.append({'reference':item,**d})
valid=[d for d in records if d['status']=='completed'];e=np.array([np.array(d['singlepoint']['prediction_eV'])-d['source_labels_eV'] for d in valid]);secs=[d['singlepoint']['seconds'] for d in valid]
r['quantum_protocol_reference']={'protocol':refs['protocol'],'scope':refs['selection'],'expected':len(records),'converged':len(valid),'mae_eV':abs(e).mean(0).tolist(),'bias_eV':e.mean(0).tolist(),'rmse_eV':np.sqrt(np.square(e).mean(0)).tolist(),'max_abs_error_eV':abs(e).max(0).tolist(),'seconds_min_median_max':np.quantile(secs,[0,.5,1]).tolist(),'records':records}
r['backbone_verification']=json.loads(Path('docs/experiments/unimol_backbone_verification_20261112.json').read_text());r['quantum_engine_verification']=json.loads((a.input/'assets/quantum_engine_check.json').read_text())
a.output.write_text(json.dumps(r,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with np.load(a.input/'profile/test_predictions.npz') as z:
 y=z['labels'];preds={name:z[name] for name in ['dpa_common','unimol']}
fig,axes=plt.subplots(2,2,figsize=(9.4,10));bounds=[]
for k in range(2):
 v=np.concatenate([y[:,k],*[v[:,k] for v in preds.values()]]);bounds.append((np.floor(v.min()*2)/2-.5,np.ceil(v.max()*2)/2+.5))
span=max(hi-lo for lo,hi in bounds)
for row,name in enumerate(['homo','lumo']):
 lo=bounds[row][0]
 for col,(model,pred) in enumerate(preds.items()):
  ax=axes[row,col];m=r['split_metrics']['test'][model]
  ax.scatter(y[:,row],pred[:,row],s=4,alpha=.18,color='#c45132' if model=='dpa_common' else '#2c6caa')
  ax.plot([lo,lo+span],[lo,lo+span],'--',color='gray');ax.set_xlim(lo,lo+span);ax.set_ylim(lo,lo+span);ticks=np.arange(np.ceil(lo),lo+span+.01,2);ax.set_xticks(ticks);ax.set_yticks(ticks);ax.set_aspect('equal',adjustable='box')
  ax.set_title(f'{name.upper()} | {model}\nMAE {m["mae"][row]:.3f} eV; slope {m["slope"][row]:.3f}',fontsize=11);ax.set_xlabel('OE62 label (eV)');ax.set_ylabel('Prediction (eV)');ax.grid(alpha=.15)
fig.suptitle('Locked scaffold-held-out source test: 4,758 molecules')
fig.text(.5,.025,'Identical 8,170 training molecules; original source coordinates and hydrogens; no Te; no test-based model selection.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.055,1,.94],h_pad=3,w_pad=2);fig.savefig(a.plot,dpi=180);plt.close(fig)
print('source test',[(m,r['split_metrics']['test'][m]['mae']) for m in preds]);print('quantum source differences',r['quantum_protocol_reference']['mae_eV'])
