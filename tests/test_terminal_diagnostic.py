import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_terminal_pool import summarize

def test_terminal_denominator_keeps_invalid_and_unconverged_samples():
    base={'target':'homo','goal_eV':-6.,'group':'25/0','total_dpa_calls':2,'seconds':1.}
    rows=[{**base,'initial_valid':False}, {**base,'initial_valid':True,'physical_accepted':False,'initial':{'prediction_eV':[-5.,0.]},'final':{'prediction_eV':[-6.,0.]}}, {**base,'initial_valid':True,'physical_accepted':True,'initial':{'prediction_eV':[-5.,0.]},'final':{'prediction_eV':[-6.,0.]}}]
    r=summarize(rows)
    assert r['attempted']==3 and r['initial_valid']==2 and r['physical_accepted']==1
    assert r['joint_hits']==1 and r['paired_geometry_valid_count']==2
    assert r['zero_or_one_accepted_groups']==1
