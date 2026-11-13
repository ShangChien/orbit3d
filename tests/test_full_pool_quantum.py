import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('full_quantum',Path(__file__).parents[1]/'scripts/summarize_full_pool_quantum.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_failures_and_rejected_proposals_remain_in_yield_denominator():
    rows=[{'status':'completed','singlepoint':{'prediction_eV':[-6.,-1.]}},
          {'status':'failed'}, {'status':'worker_failed'}]
    result=module.metric(rows,10,-6.)
    assert result['hits']==1 and result['yield_all_proposals']==.1
    assert result['terminal_selected']==3 and result['dft_completed']==1
    assert result['mae_eV']==0


def test_empty_quantum_cell_is_missing_error_not_zero():
    result=module.metric([],64,-6.)
    assert result['hits']==0 and result['mae_eV'] is None
    assert result['prediction_std_eV'] is None


def test_lumo_uses_lumo_prediction_not_homo_column():
    rows=[{'status':'completed','singlepoint':{'prediction_eV':[-6.,-1.2]}}]
    assert module.metric(rows,64,-1.2,property_index=1)['hits']==1
    assert module.metric(rows,64,-1.2,property_index=0)['hits']==0


def test_unique_novel_hits_do_not_count_repeated_graphs_twice():
    rows=[{'status':'completed','singlepoint':{'prediction_eV':[-6.,-1.]},'smiles':s} for s in ['CC','CC','CO']]
    result=module.metric(rows,10,-6.,known_graphs={'CC'})
    assert result['hits']==3 and result['unique_hit_graphs']==2
    assert result['unique_novel_hit_graphs']==1


def test_dft_adjudication_does_not_require_successful_proxy_prediction():
    path=Path(__file__).parents[1]/'scripts/summarize_point_quantum.py'
    spec=importlib.util.spec_from_file_location('quantum_summary',path)
    summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
    row=dict(index=0,target='homo',target_eV=-6.,dpa_prediction_eV=None,
             singlepoint={'prediction_eV':[-6.,-1.]})
    result=summary.summarize([row],planned=4,optimization_requested=False)
    assert result['input_target']['hits']['0.25']==1
    assert result['dpa_target_input']['evaluated']==0
    assert result['input_target']['planned']==4


def test_target_reassignment_rejects_value_insensitive_control(tmp_path, monkeypatch):
    import json
    import sys
    source=tmp_path/'terminal';source.mkdir();results=tmp_path/'quantum';results.mkdir()
    goals={'25':-7.,'50':-6.,'75':-5.};rows=[];jobs=[]
    for q in [25,50,75]:
        for seed in [1,2]:
            for candidate in [0,1]:
                i=len(rows);rows.append(dict(index=i,arm='model',quantile=q,inference_seed=seed,candidate_id=candidate,target_eV=goals[str(q)],source_terminal=str(source),smiles='CC',atoms=2,domain={'in_domain':True}))
                job=results/f'job{i:02d}';job.mkdir();(job/'result.json').write_text(json.dumps(dict(index=i,input_sha256='fixture',status='completed',atoms=2,singlepoint={'prediction_eV':[-6.,0.]})))
                jobs.append(dict(index=i,start=0,end=1))
    (source/'summary.json').write_text(json.dumps({'records':rows}))
    (results/'completed.json').write_text(json.dumps({'attempted':12,'elapsed_seconds':12,'jobs':jobs}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(selected=12,sdf_sha256='fixture',arms=['model'],proposal_denominators={'model':12},records=rows)))
    targets=tmp_path/'targets.json';targets.write_text(json.dumps({'single_target_points':{'homo':goals}}))
    output=tmp_path/'report.json';monkeypatch.setattr(sys,'argv',['summary','--manifest',str(manifest),'--results',str(results),'--targets',str(targets),'--output',str(output)])
    module.main();report=json.loads(output.read_text())['arms']['model']
    assert report['proposals']==12
    assert report['correct_vs_mismatched']['correct_minus_wrong_hit_yield']==0
    assert report['correct_vs_mismatched']['sampling_delta_95']==[0.,0.]
    for r in rows:
        path=results/f"job{r['index']:02d}"/'result.json';data=json.loads(path.read_text());data['singlepoint']['prediction_eV'][0]=r['target_eV'];path.write_text(json.dumps(data))
    module.main();report=json.loads(output.read_text())['arms']['model']
    assert report['correct_vs_mismatched']['correct_minus_wrong_hit_yield']==1
    assert report['correct_vs_mismatched']['wrong_value_hit_yield']==0
