import json
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
from orbit3d.utils import replay_validation


def test_weight_packaging_never_skips_a_requested_model_or_changes_bytes(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/pack_pretrained_weights.py'
    source = tmp_path / 'source'
    source.mkdir()
    checkpoint = source / 'DPA4-Mini-OMol25-100M-v20260820.pt'
    checkpoint.write_bytes(b'opaque checkpoint bytes')
    output = tmp_path / 'weights'
    command = [sys.executable, str(script), '--dpa4-src-dir', str(source), '--out-dir', str(output)]
    missing = subprocess.run(command + ['--dpa4-variants', 'Mini', 'Plus'], capture_output=True)
    assert missing.returncode != 0 and not output.exists()
    subprocess.run(command + ['--dpa4-variants', 'Mini'], check=True, capture_output=True)
    assert (output / 'dpa4' / checkpoint.name).read_bytes() == checkpoint.read_bytes()
    manifest = json.loads((output / 'weights_manifest.json').read_text())
    assert len(manifest) == 1 and manifest[0]['transformation'] == 'none'


def test_replay_fails_when_a_later_relaxation_trace_is_missing(tmp_path, monkeypatch):
    reference = tmp_path / 'reference'
    reference.mkdir()
    pd.DataFrame({'candidate_id': [0, 1], 'status': ['retained', 'retained'],
                  'joint_success': [False, False], 'smiles': ['CC', 'CO'], 'raw_valid': [True, True]}).to_csv(reference / 'candidates.csv', index=False)
    (reference / 'generated.sdf').write_text('')
    (reference / 'raw_symbols.json').write_text(json.dumps([['C', 'C'], ['C', 'O']]))
    (reference / 'relaxation_traces.json').write_text(json.dumps([{'candidate_id': 0}, {'candidate_id': 1}]))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'generator_config': {}, 'mount_paths': {}}))

    def incomplete_run(cfg):
        destination = Path(cfg['mount_paths']['output_dir'])
        shutil.copytree(reference, destination)
        (destination / 'relaxation_traces.json').write_text(json.dumps([{'candidate_id': 0}]))
        return {}

    monkeypatch.setattr(replay_validation, 'run_octree_from_config', incomplete_run)
    monkeypatch.setattr(sys, 'argv', ['verify_e2e', '--config', str(config), '--checkpoint', 'unused',
                        '--reference-run', str(reference), '--output', str(tmp_path / 'replay')])
    with pytest.raises(AssertionError, match='trace count differs'):
        replay_validation.main()


def test_scaffold_replay_uses_numeric_tolerance_but_exact_placement_decisions():
    import copy
    first = [{'candidate_id': 0, 'property_score': 1.,
              'growth': [{'anchor_position_A': [1., 2., 3.]}]},
             {'candidate_id': 1, 'property_score': 2., 'growth': [{'port_map': 3}]}]
    second = copy.deepcopy(first)
    second[0]['property_score'] += 1e-6
    replay_validation.compare_scaffold_records(first, second)
    second[0]['growth'][0]['anchor_position_A'][0] += 1e-10
    with pytest.raises(AssertionError, match='growth differs'):
        replay_validation.compare_scaffold_records(first, second)
    second = copy.deepcopy(first)
    second[1]['growth'][0]['port_map'] = 4
    with pytest.raises(AssertionError, match='proposal 1'):
        replay_validation.compare_scaffold_records(first, second)
