"""Summarize all four arms and verify historical baseline replay."""
import argparse
import json,hashlib,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--reference',type=Path,default=Path('docs/experiments/refinement_matched_20261112.json'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=a.input
records=json.loads((root/'records.json').read_text())
summary=json.loads((root/'summary.json').read_text())
assert len(records)==136 and summary['candidates']==34
ref=json.loads(a.reference.read_text())
old={(x['run'],x['candidate_id'],x['rule']):x for x in ref['records']}
parity={}
for arm,rule in [('full','full_acceptance'),('energy30','energy_only')]:
 errors=[];force_errors=[]
 for r in records:
  if r['arm']!=arm:continue
  o=old[(r['run'],r['candidate_id'],rule)]
  assert r['status']==o['status']
  if r['status']=='completed':
   assert r['joint_success']==o['joint_success']
   errors.append(float(np.max(np.abs(np.array(r['final']['prediction'])-o['final']['prediction']))))
   force_errors.append(abs(r['final']['rms_force_eV_A']-o['final']['rms_force_eV_A']))
 parity[arm]={'max_prediction_difference_eV':max(errors),'max_rms_difference':max(force_errors),'all_decisions_equal':True}
 assert max(errors)<1e-4 and max(force_errors)<1e-4
for arm,stats in summary['arms'].items():
 rows=[r for r in records if r['arm']==arm and r['status']=='completed']
 stats['mean_interval_distance_eV']=float(np.mean([np.sqrt(r['final']['interval_loss']) for r in rows]))
 stats['rms01_count']=sum(r['final']['rms_force_eV_A']<=.1 for r in rows)
 stats['by_seed']={seed:sum(r['joint_success'] for r in rows if r['run']==seed) for seed in sorted({r['run'] for r in rows})}
three=[r for r in records if r['arm']=='lbfgs_reverse_reopt' and len(r['phases'])==3]
transitions=collections.Counter(tuple(p['final']['property_hit'] for p in r['phases']) for r in three)
rejects=collections.Counter(reason for r in three for step in r['phases'][1]['history'] for trial in step['trials'] for reason in trial['reasons'])
summary['guidance']={'three_phase_candidates':len(three),'hit_transitions':{str(k):v for k,v in transitions.items()},'rejected_trial_reasons':dict(rejects),'accepted_guidance_steps':sum(s['accepted'] for r in three for s in r['phases'][1]['history'])}
summary['guidance']['mean_interval_distance_by_phase_eV'] = [float(np.mean([np.sqrt(r['phases'][i]['final']['interval_loss']) for r in three])) for i in range(3)]
summary['baseline_replay']=parity
summary['remote_artifacts']='/sjtu-combustion/wjw/chensq/orbit_priority_5fe5/full34'
summary['input_manifest']=json.loads((root/'input_manifest.json').read_text())
summary['run_manifest']=json.loads((root/'run_manifest.json').read_text())
summary['records_sha256']=hashlib.sha256((root/'records.json').read_bytes()).hexdigest()
compact=[]
for r in records:
 c={k:v for k,v in r.items() if k not in ['legacy_trace','phases']}
 c['phases']=[{k:v for k,v in p.items() if k!='history'} for p in r['phases']]
 for v in [c.get('initial'),c.get('final')]+[q.get(k) for q in c['phases'] for k in ['initial','final']]:
  if v is not None:v.pop('forces',None)
 compact.append(c)
summary['records']=compact
a.output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['records','input_manifest','run_manifest']},indent=2))
