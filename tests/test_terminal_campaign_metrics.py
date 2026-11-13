import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_terminal_campaign import target_metrics


def test_joint_yield_keeps_unaccepted_proposals_and_mae_does_not_impute_them():
    rows=[]
    for q in [25,50,75]:
        rows.extend([{'quantile':q,'goal_eV':q/100,'ok':True,'value':q/100+.1}, {'quantile':q,'goal_eV':q/100,'ok':False}])
    r=target_metrics(rows,lambda r:r['value'],lambda r:r['ok'])
    assert r['attempted']==6 and r['evaluated']==3 and r['hit_yield']['0.25']==.5
    assert abs(r['mae_equal_target_eV']-.1)<1e-10
    rows[0]['ok']=False
    r=target_metrics(rows,lambda r:r['value'],lambda r:r['ok'])
    assert r['mae_equal_target_eV'] is None and r['hit_yield']['0.25']==2/6
