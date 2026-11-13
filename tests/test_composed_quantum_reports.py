import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('composition',Path(__file__).parents[1]/'scripts/compose_full_pool_reports.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_disjoint_arms_keep_local_indices_but_duplicate_arms_are_rejected():
    def report(arm):return dict(target='homo',control='correct',novelty_reference_sha256='native',arms={arm:{'proposals':1280}},records=[{'arm':arm,'index':0}])
    result=module.compose([report('a'),report('b')]);assert len(result['records'])==2
    assert result['arms']['a']['proposals']==1280
    with pytest.raises(AssertionError,match='Repeated policy'):module.compose([report('a'),report('a')])
