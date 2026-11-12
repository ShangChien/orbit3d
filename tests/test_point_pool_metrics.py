import runpy
from pathlib import Path
import pytest
from orbit3d.constraints import TargetPoints


def test_raw_pool_metrics_keep_failures_and_separate_domain():
    summarize=runpy.run_path(str(Path(__file__).parents[1]/'scripts/evaluate_point_pool.py'))['summarize_records']
    def evaluated(value,domain=True):
        return {'evaluated':True,'independent_evaluated':True,'prediction':[value,99.],
                'independent_prediction':[value,99.],'domain':{'in_domain':domain},'rms_force':.01}
    records=[evaluated(-7.1),evaluated(-7.,False),
             {'evaluated':False,'independent_evaluated':False}]
    points=TargetPoints.parse({'homo':-7},{'homo':.25},['homo','lumo'])
    result=summarize(records,points)['dpa']
    assert result['all_evaluable']['property_hit_rate_all_attempts']==2/3
    assert result['in_domain']['coverage']==1/3
    assert result['in_domain']['targets']['homo']['mae_eV']==pytest.approx(.1)
    assert result['in_domain']['joint_domain_force_hits']==1
    assert 'lumo' not in result['all_evaluable']['targets']
