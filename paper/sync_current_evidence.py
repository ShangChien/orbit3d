"""Copy selected public run summaries into the self-contained manuscript archive."""
import hashlib
import json
import shutil
from pathlib import Path
paper = Path(__file__).resolve().parent
staging = paper / "build" / "data"
staging.mkdir(parents=True, exist_ok=True)
shutil.copyfile(paper / "ancillary_readme.txt", staging / "ancillary_readme.txt")
names = [
    'oe62_formal_homo_hyff_repeats_20261113.json',
    'oe62_formal_homo_repeats_20261113.json',
    'oe62_formal_homo_seed42_20261113.json',
    'oe62_formal_homo_seed43_20261113.json',
    'oe62_formal_homo_seed44_20261113.json',
    'oe62_formal_hyff_seed42_20261113.json',
    'oe62_formal_hyff_seed43_20261113.json',
    'oe62_formal_hyff_seed44_20261113.json',
    'oe62_formal_lumo_hyff_repeats_20261113.json',
    'oe62_formal_lumo_repeats_20261113.json',
    'oe62_formal_lumo_seed42_20261113.json',
    'oe62_formal_lumo_seed43_20261113.json',
    'oe62_formal_lumo_seed44_20261113.json',
    'oe62_formal_raw_force_audit_20261113.json',
    'oe62_wheel_replay_homo_20261113.json',
    'oe62_wheel_replay_lumo_20261113.json',

    'oe62_source_holdout_20261113.json',
    'oe62_geometry_comparison_20261113.json',
    'oe62_hyff_pools_20261113.json',
    'oe62_brics_point_pools_20261113.json',
    'oe62_ranking_20261113.json',
    'oe62_charge_support_20261113.json',
    'oe62_point_homo_seed42_20261112.json',
    'oe62_point_lumo_seed42_20261112.json',
    'oe62_source_properties_20261112.json',
    'oe62_prior_comparison_20261112.json',
    'oe62_graph_feature_contract_20261112.json',
    'ranking_ablation_20261111.json',
    'generated_quantum_diagnostic_20261111.json',
    'generated_excited_minima_20261112.json',
    'boron_structural500_20261112.json',
    'boron_physics50_20261112.json',
    'refinement_matched_20261112.json',
    'stage_and_training_seeds_20261112.json',
    'ocelot_wide8192_20261112.json',
    'scaffold_cli_20261112.json',
    'element_balance_20261112.json',
    'element_temperature_20261112.json',
    'condition_contrast_20261112.json',
    'optical_matched512_20261112.json',
    'ocelot_optical8192_20261112.json',
    'fingerprint_optical20356_20261112.json',
    'grpo_factorial_20261112.json',
    'optical_generation_20261112.json',
    'optical_calibration_20261112.json',
    'boron_quantum_cohort_20261112.json',
    'boron_quantum_oracles_20261112.json',
    'boron_geometry160_20261112.json',
    'kv_buffer_20261112.json',
    'brics_comparison_20261112.json',
    'hyff_optical8192_20261112.json',
    'hyff_optical_calibration_20261112.json',
    'optical_hyff_independent_20261112.json',
    'relocated_artifact_verification_20261112.json',
    'boron_quantum_hyff_20261112.json',
    'excitation_gradient_comparison_20261112.json',
    'boron_excitation_gradients_20261112.json',
    'boron_quantum_dpa_gradients_20261112.json',
    'boron_quantum_hyff_gradients_20261112.json',
    'qualified20k_quantum_20261112.json',
    'qualified20k_quantum_cohort_20261112.json',
    'qualified20k_selection_20261112.json',
    'qualified20k_same_geometry_20261112.json',
    'qualified20k_same_geometry_inputs_20261112.json',
    'qualified20k_label_overlap_20261112.json',
    'bond_perception_20261112.json',
    'bond_perception_inputs_20261112.json',
    'sft_duration_20261112.json',
    'orbital20k_artifact_verification_20261112.json',
    'orbital20k_bundle_manifest_20261112.json',
    'orbital20k_bundle_reload_20261112.json',
    'wide_spatial_audit_20261112.json',
]
records = {}
for name in names:
    path = paper.parent/'docs/experiments'/name
    raw = path.read_text()
    records[name] = {'source': 'docs/experiments/'+name, 'sha256': hashlib.sha256(raw.encode()).hexdigest(), 'json': raw}
for name in ['wide_target_windows.json', 'wide_target_windows_protocol.json', 'optical_target_windows.json', 'oe62_split_manifest.json', 'oe62_source.json']:
    raw = (paper.parent / 'data' / name).read_text()
    records[name] = {'source': 'data/' + name, 'sha256': hashlib.sha256(raw.encode()).hexdigest(), 'json': raw}
(staging / 'unified_evidence.json').write_text(json.dumps(records, indent=2)+'\n')

quantum_input = paper.parent/'docs/experiments/boron_quantum_input_20261112.sdf'
cohort = json.loads(records['boron_quantum_cohort_20261112.json']['json'])
assert hashlib.sha256(quantum_input.read_bytes()).hexdigest() == cohort['manifest']['source_sha256']
shutil.copyfile(quantum_input, staging / 'boron_quantum_input_20261112.sdf')

qualified_input = paper.parent / 'docs/experiments/qualified20k_quantum_input_20261112.sdf'
selection = json.loads(records['qualified20k_selection_20261112.json']['json'])
assert hashlib.sha256(qualified_input.read_bytes()).hexdigest() == selection['source_sha256']
shutil.copyfile(qualified_input, staging / qualified_input.name)

gradient_inputs = json.loads(records['excitation_gradient_comparison_20261112.json']['json'])['source_sha256']
for name, key in [('boron_excitation_gradients_20261112.npz', 'quantum_gradients'),
                  ('boron_quantum_dpa_gradients_20261112.npz', 'dpa_gradients'),
                  ('boron_quantum_hyff_gradients_20261112.npz', 'hyff_gradients')]:
    source = paper.parent / 'docs/experiments' / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == gradient_inputs[key]
    shutil.copyfile(source, staging / name)

geometry=json.loads(records['oe62_geometry_comparison_20261113.json']['json'])
for name,digest in geometry['geometry_files_sha256'].items():
    source=paper.parent/'docs/experiments'/name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    shutil.copyfile(source,staging/name)
brics=json.loads(records['oe62_brics_point_pools_20261113.json']['json'])
source=paper.parent/'docs/experiments'/brics['raw_conformers_file']
assert hashlib.sha256(source.read_bytes()).hexdigest()==brics['replay_audit']['raw_conformers_sha256']
shutil.copyfile(source,staging/source.name)
