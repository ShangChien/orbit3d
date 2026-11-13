"""Bind calibration outcomes to frozen inputs and paired generated-scaffold diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def errors(truth,prediction):
    residual=prediction-truth
    return {'n':len(truth),'mae_eV':abs(residual).mean(0).tolist(),'bias_eV':residual.mean(0).tolist(),'absolute_error_within_025':(abs(residual)<=.25).mean(0).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=json.loads(a.summary.read_text());report['model_metadata']=json.loads(a.metadata.read_text());report['input_manifest']=json.loads(a.inputs.read_text());split=json.loads(a.split.read_text());rows=[r for r in split['records'] if r['status']=='accepted' and r['split']=='test'];values=dict(np.load(a.test));assert len(rows)==len(values['truth'])
    report['test_generated_by_cohort']={}
    for cohort in sorted({r['cohort'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['cohort']==cohort]
        report['test_generated_by_cohort'][cohort]={name:errors(values['truth'][ids],values[name][ids]) for name in ['baseline','affine','prediction']}
    scaffolds=sorted({r['scaffold'] for r in rows});groups=[np.array([i for i,r in enumerate(rows) if r['scaffold']==key]) for key in scaffolds];rng=np.random.default_rng(1515);deltas=[]
    residuals={name:abs(values[name]-values['truth']) for name in ['baseline','affine','prediction']}
    for _ in range(4000):
        ids=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))]);deltas.append((residuals['prediction'][ids]-residuals['baseline'][ids]).mean(0))
    report['generated_test_paired_sampling']={'scope':'Scaffold-block paired bootstrap on fixed models, 4000 draws, RNG1515; not training variance. Test labels do not tune models.','scaffolds':len(scaffolds),'adapted_minus_parent_mae_eV':(residuals['prediction']-residuals['baseline']).mean(0).tolist(),'delta95_eV':np.quantile(deltas,[.025,.975],axis=0).T.tolist()}
    report['generated_test_records']=[{**r,**{name:values[name][i].tolist() for name in ['truth','baseline','affine','prediction','std']}} for i,r in enumerate(rows)]
    report['summary_inputs_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.summary,a.metadata,a.inputs,a.test,a.split]}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(report['test_generated_by_cohort']);print(report['generated_test_paired_sampling'])

if __name__=='__main__':main()
