"""Copy selected public run summaries into the self-contained manuscript archive."""
import hashlib
import json
import shutil
from pathlib import Path
paper = Path(__file__).resolve().parent
names = ['ocelot_4096_20261111.json', 'ranking_ablation_20261111.json',
         'ocelot_generator_continued_20261111.json', 'generated_quantum_diagnostic_20261111.json',
         'batched_pipeline_comparison_20261111.json', 'batched_pipeline_reload_20261111.json',
         'clean_deepmd_native_20261111.json', 'conditioning_factorial_20261112.json',
         'generated_excited_minima_20261112.json', 'reward_ablation_20261112.json',
         'ocelot8192_evaluation_20261112.json', 'sampling_compaction_20261112.json',
         'boron_structural500_20261112.json', 'boron_physics50_20261112.json',
         'depth7_adaptation_20261112.json', 'refinement_matched_20261112.json',
         'stage_and_training_seeds_20261112.json', 'ocelot_wide8192_20261112.json',
         'hyff_property_workflow_20261112.json', 'hyff_full_reload_20261112.json',
         'oracle8192_full_reload_20261112.json', 'wide_target_campaign_20261112.json',
         'wide_element_likelihood_20261112.json', 'scaffold_cli_20261112.json',
         'element_balance_20261112.json', 'element_temperature_20261112.json',
         'condition_contrast_20261112.json', 'optical_matched512_20261112.json',
         'ocelot_optical8192_20261112.json', 'fingerprint_optical20356_20261112.json',
         'latest_mainline_reload_20261112.json', 'boron_unconverged6_20261112.json',
         'grpo_factorial_20261112.json', 'optical_generation_20261112.json',
         'optical_calibration_20261112.json', 'optical_wheel_reload_20261112.json',
         'boron_quantum_cohort_20261112.json', 'boron_quantum_oracles_20261112.json',
         'boron_geometry160_20261112.json', 'kv_buffer_20261112.json',
         'brics_comparison_20261112.json', 'hyff_optical8192_20261112.json',
         'hyff_optical_calibration_20261112.json', 'optical_hyff_independent_20261112.json',
         'relocated_artifact_verification_20261112.json', 'boron_quantum_hyff_20261112.json',
         'excitation_gradient_comparison_20261112.json', 'ecp_initial_failure_20261112.json',
         'ecp_tight_grid_20261112.json', 'quantum_ecp_wheel_20261112.json',
         'boron_excitation_gradients_20261112.json', 'boron_quantum_dpa_gradients_20261112.json',
         'boron_quantum_hyff_gradients_20261112.json', 'qualified20k_quantum_20261112.json',
         'qualified20k_quantum_cohort_20261112.json', 'qualified20k_selection_20261112.json',
         'qualified20k_same_geometry_20261112.json', 'qualified20k_same_geometry_inputs_20261112.json',
         'qualified20k_label_overlap_20261112.json', 'bond_perception_20261112.json',
         'bond_perception_inputs_20261112.json', 'quantum_identity_wheel_20261112.json',
         'quantum_identity_recheck_20261112.json']
records = {}
for name in names:
    path = paper.parent/'docs/experiments'/name
    raw = path.read_text()
    records[name] = {'source': 'docs/experiments/'+name, 'sha256': hashlib.sha256(raw.encode()).hexdigest(), 'json': raw}
for name in ['wide_target_windows.json', 'wide_target_windows_protocol.json', 'optical_target_windows.json']:
    raw = (paper.parent / 'data' / name).read_text()
    records[name] = {'source': 'data/' + name, 'sha256': hashlib.sha256(raw.encode()).hexdigest(), 'json': raw}
(paper/'data/unified_evidence.json').write_text(json.dumps(records, indent=2)+'\n')

quantum_input = paper.parent/'docs/experiments/boron_quantum_input_20261112.sdf'
cohort = json.loads(records['boron_quantum_cohort_20261112.json']['json'])
assert hashlib.sha256(quantum_input.read_bytes()).hexdigest() == cohort['manifest']['source_sha256']
shutil.copyfile(quantum_input, paper/'data/boron_quantum_input_20261112.sdf')

qualified_input = paper.parent / 'docs/experiments/qualified20k_quantum_input_20261112.sdf'
selection = json.loads(records['qualified20k_selection_20261112.json']['json'])
assert hashlib.sha256(qualified_input.read_bytes()).hexdigest() == selection['source_sha256']
shutil.copyfile(qualified_input, paper / 'data' / qualified_input.name)

gradient_inputs = json.loads(records['excitation_gradient_comparison_20261112.json']['json'])['source_sha256']
for name, key in [('boron_excitation_gradients_20261112.npz', 'quantum_gradients'),
                  ('boron_quantum_dpa_gradients_20261112.npz', 'dpa_gradients'),
                  ('boron_quantum_hyff_gradients_20261112.npz', 'hyff_gradients')]:
    source = paper.parent / 'docs/experiments' / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == gradient_inputs[key]
    shutil.copyfile(source, paper / 'data' / name)
