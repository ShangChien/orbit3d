"""Paired continuous target-error summaries; no claim of generator retraining."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
records = json.loads((a.input/'records.json').read_text())
report = json.loads((a.input/'summary.json').read_text())
assert len(records) == 136
assert len({(r['run'],r['candidate_id'],r['target_name'],r['arm']) for r in records}) == 136
report['target_support'] = json.loads(Path('data/point_refinement_target_support.json').read_text())
report['paired_comparisons'] = {}
for name,target in report['protocol']['targets'].items():
    arms = {arm:{(r['run'],r['candidate_id']):r for r in records if r['target_name']==name and r['arm']==arm} for arm in ['interval','point']}
    assert set(arms['interval'])==set(arms['point']) and len(arms['point'])==34
    differences = collections.defaultdict(list)
    by_seed = collections.defaultdict(list)
    for key,left in arms['interval'].items():
        right = arms['point'][key]
        np.testing.assert_allclose(left['initial']['prediction'],right['initial']['prediction'],atol=1e-5,rtol=0)
        if left['status']!='completed' or right['status']!='completed':continue
        for phase in ['after_guidance','final']:
            l=np.abs(np.array(left[phase]['prediction'])-target).mean()
            r=np.abs(np.array(right[phase]['prediction'])-target).mean()
            differences[phase].append(float(r-l))
            if phase=='final':by_seed[key[0]].append(float(r-l))
    report['paired_comparisons'][name] = {phase:{'paired_count':len(ds),'mean_point_minus_interval_mae_eV':float(np.mean(ds)),
        'point_better_by_more_than_1meV':sum(v<-.001 for v in ds),'point_worse_by_more_than_1meV':sum(v>.001 for v in ds),
        'within_1meV':sum(abs(v)<=.001 for v in ds)} for phase,ds in differences.items()}
    report['paired_comparisons'][name]['mean_final_difference_by_inference_seed']={k:float(np.mean(v)) for k,v in by_seed.items()}
    for arm in ['interval','point']:
        rows=list(arms[arm].values());ok=[r for r in rows if r['status']=='completed']
        extra=report['groups'][name+'/'+arm]
        extra['evaluation_calls']=sum(r['evaluation_calls'] for r in rows)
        extra['gradient_calls']=sum(r['gradient_calls'] for r in rows)
        extra['accepted_guidance_steps']=sum(r.get('accepted_guidance_steps',0) for r in rows)
        extra['mean_l2_squared_by_phase']={phase:float(np.square(np.array([r[phase]['prediction'] for r in ok])-target).sum(1).mean()) for phase in ['initial','after_guidance','final']}
        extra['rms01_final_count']=sum(r['final']['rms_force_eV_A']<=.1 for r in ok)
        extra['hit_transitions'] = dict(collections.Counter(str(tuple(r[phase]['property_hit'] for phase in ['initial','after_guidance','final'])) for r in ok))
old=json.loads(Path('docs/experiments/relaxation_priority_20261112.json').read_text())
report['historical_rescore']={}
for name,target in report['protocol']['targets'].items():
    report['historical_rescore'][name]={}
    for arm in old['arms']:
        rows=[r for r in old['records'] if r['arm']==arm and r['status']=='completed']
        error=abs(np.array([r['final']['prediction'] for r in rows])-target)
        report['historical_rescore'][name][arm]={'mae_eV':error.mean(0).tolist(),'mean_mae_eV':float(error.mean()),'joint_within025':int((error<=.25).all(1).sum())}
report['run_manifest']=json.loads((a.input/'run_manifest.json').read_text())
report['input_manifest']=json.loads((a.input/'input_manifest.json').read_text())
assert report['target_support']['historical_oracle_training4096']['oracle_meta_sha256'] == report['input_manifest']['oracle']['oracle_meta.json']
report['raw_records_sha256']=hashlib.sha256((a.input/'records.json').read_bytes()).hexdigest()
report['remote_artifacts']='/sjtu-combustion/wjw/chensq/orbit_point_guidance_5fe5/full34'
report['records']=[]
for row in records:
    c={k:v for k,v in row.items() if k not in ['guidance_trace','reoptimization_trace']}
    for phase in ['initial','after_guidance','final']:
        if phase in c:c[phase].pop('forces',None)
    report['records'].append(c)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k in ['groups','paired_comparisons','historical_rescore']},indent=2))
