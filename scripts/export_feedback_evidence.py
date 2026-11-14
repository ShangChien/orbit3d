"""Export one E16 result report and exact concatenated quantum-input SDF blocks."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=a.root;hashes={}
    def read(path):
        raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    report=read(r/'statistics/feedback_summary.json')
    report['evaluation_schedule']=read(r/'evaluation_overlap.json')
    report['null_policy_audit']=read(r/'null_policy_audit.json')
    report['critic_transfer']=read(r/'critic_transfer/summary.json')
    report['critic_ranking']=read(r/'statistics/critic_ranking.json')
    report['critic_transfer_inputs']=read(r/'critic_transfer/input_manifest.json')
    report['training_inputs']={arm:read(r/f'training/{arm}/run/input_manifest.json') for arm in ['native_critic','calibrated_critic','continue_wall']}
    report['evaluation_costs']={};report['whole_pool_reports']={};report['proposal_records']={};report['B_records']={};report['B_inputs']={};blocks=[];offset=0;global_index=0;data=bytearray()
    qkeys=['index','arm','quantile','target_eV','inference_seed','candidate_id','smiles','atoms','status','singlepoint','domain','ecp_core_electrons','input_sha256']
    rawkeys=['target','quantile','candidate_id','group','inference_seed','goal_eV','atoms','initial_valid','physical_accepted','total_dpa_calls','smiles','domain','domain_error','scope_supported','scope_exclusion','initial','optimization','final','error','initial_property_error','seconds','coordinates_key']
    for arm in ['frozen','continue_wall','native_critic','calibrated_critic','shared_off']:
        cohort=r/f'cohort_{arm}';manifest=read(cohort/'manifest.json');raw=(cohort/'cohort.sdf').read_bytes();digest=hashlib.sha256(raw).hexdigest()
        assert digest==manifest['sdf_sha256']
        assert [v['index'] for v in manifest['records']]==list(range(manifest['selected']))
        assert raw.rstrip().endswith(b'$$$$')
        if not raw.endswith(b'\n'):raise ValueError('A complete SDF record must terminate with a newline before concatenation')
        block=dict(arm=arm,byte_offset=offset,byte_length=len(raw),first_combined_index=global_index,molecules=manifest['selected'],sha256=digest)
        blocks.append(block);data.extend(raw);offset+=len(raw)
        quantum=read(r/f'statistics/{arm}.json');records=[]
        for row in quantum['records']:
            records.append({**{k:row[k] for k in qkeys if k in row},'combined_input_index':global_index+row['index']})
        assert len(records)==manifest['selected']
        report['whole_pool_reports'][arm]={**{k:v for k,v in quantum.items() if k not in ['records','source_sha256']},'records':records}
        cross=read(r/f'independent/{arm}/summary.json')['runs'][0]
        report['B_records'][arm]=[{k:row[k] for k in ['target','quantile','inference_seed','candidate_id','evaluated','prediction_eV','support_exclusion'] if k in row} for row in cross['records']]
        report['B_inputs'][arm]=read(r/f'independent/{arm}/input_manifest.json')
        terminal=read(r/f'terminals/{arm}/summary.json')
        report['proposal_records'][arm]=[{k:row[k] for k in rawkeys if k in row} for row in terminal['records']]
        assert len(report['proposal_records'][arm])==manifest['proposal_denominators'][arm]
        matrix=read(r/f'campaigns/{arm}/eval_homo_seed42/matrix.json')
        assert sum(row['attempted'] for row in matrix)==manifest['proposal_denominators'][arm]
        report['evaluation_costs'][arm]=dict(raw_generation_and_scoring_seconds_sum=sum(row['wall_seconds'] for row in matrix),terminal_candidate_seconds_sum=terminal['all']['seconds'],energy_force_calls=terminal['all']['energy_force_calls'],quantum_queue_wall_seconds=quantum['elapsed_seconds'],B_candidate_seconds=cross['seconds'])
        global_index+=manifest['selected']
    sdf=a.output.with_name(a.output.stem+'_inputs.sdf');sdf.write_bytes(data)
    report['quantum_inputs']=dict(file=sdf.name,molecules=global_index,sha256=hashlib.sha256(data).hexdigest(),blocks=blocks,
                                 scope='Exact original SDF byte blocks; no coordinate re-serialization. Extract by byte offset/length to recover each original quantum-input SHA. Local cohort indices are scoped by arm; combined_input_index is global.')
    report['evaluation_cost_scope']='Reported component timers: raw generation plus scoring includes diagnostic fingerprint evaluation; terminal and B candidate timers exclude model loading; quantum queue timer includes its reported queue overhead. Evaluation overlaps on each GPU, so component wall times are not isolated GPU costs and cannot be summed into end-to-end elapsed time. Training process/pipeline times are reported separately.'
    report['publication_source_sha256']=hashes
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Exported',sum(len(v) for v in report['proposal_records'].values()),'proposals and',global_index,'quantum inputs')


if __name__=='__main__':main()
