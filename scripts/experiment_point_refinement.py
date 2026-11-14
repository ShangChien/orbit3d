"""Matched point/interval guidance from frozen energy-minimized geometries."""
import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from experiment_relaxation_priority import Experiment
from loom.constraints import TargetIntervals
from loom.generator.molecular_state import ATOM_TYPES, rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol
from loom.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from loom.physics.factory import build_forcefield
from loom.utils.dataset_io import write_precise_sdf
from loom.utils.reproducibility import lock_deterministic_environment
from loom.utils.run_evidence import freeze_run, record_input_hashes

TARGETS = {'central': [-7.25, -.63], 'original_center': [-7.70, .10]}


class PointExperiment(Experiment):
    def __init__(self, *args, point, **kwargs):
        super().__init__(*args, **kwargs)
        self.point = point

    def guidance_loss(self, prediction):
        point = torch.as_tensor(self.point, device=prediction.device, dtype=prediction.dtype)
        return (prediction.reshape(-1) - point).square().sum()

    def guidance_value(self, value):
        return float(np.square(np.asarray(value['prediction']) - self.point).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-directory', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    freeze_run(a.output, __file__)
    helper = Path(__file__).with_name('experiment_relaxation_priority.py')
    shutil.copy2(helper, a.output / helper.name)
    lock_deterministic_environment(42); torch.set_num_threads(2)
    cfg = json.loads(a.config.read_text()); paths = cfg['mount_paths']
    ff = build_forcefield('dpa4', paths['forcefield_path'], ATOM_TYPES, 'cuda:0')
    oracle = InternalDPA4PropertyOracle.load_from_dir(paths['property_checkpoint_dir'], ff)
    assert list(oracle.target_cols) == ['homo', 'lumo']
    previous = json.loads(a.reference.read_text())
    selected = [r for r in previous['records'] if r['arm'] == 'lbfgs']
    assert len(selected) == 34 and all(r['status'] == 'completed' for r in selected)
    inputs = {'forcefield': paths['forcefield_path'], 'oracle': paths['property_checkpoint_dir'],
              'reference': a.reference, 'config': a.config, 'experiment_helper': helper}
    for r in selected:
        inputs[f"{r['run']}/{r['candidate_id']}"] = a.baseline_directory / r['run'] / 'lbfgs' / f"candidate_{r['candidate_id']}.sdf"
    record_input_hashes(a.output, inputs)
    protocol = {'targets': TARGETS, 'target_names': list(oracle.target_cols), 'auxiliary_tolerance_eV': .25,
        'losses': {'interval': 'sum squared violations of point +/- 0.25 eV', 'point': 'sum squared deviations from point'},
        'primary_metric': 'Mean absolute deviation from requested point, per property and average; surrogate, not DFT error.',
        'baseline_protocol': previous['protocol'], 'guidance_max_steps': 20,
        'source_generation_attempts': 192, 'fixed_cohort': 34,
        'scope': 'Frozen interval-generated/selected historical cohort. Tests geometry guidance objective, not point-conditioned generator training. GPU shared; budgets are caps, not equal compute.',
        'target_support_training8192': {'central': {'within_010': 147, 'within_025': 885},
                                       'original_center': {'within_010': 32, 'within_025': 239}},
        'limit': a.limit}
    (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    records = []
    for source in selected[:a.limit]:
        run, index = source['run'], source['candidate_id']
        path = inputs[f'{run}/{index}']
        mols = list(Chem.SDMolSupplier(str(path), removeHs=False)); assert len(mols) == 1 and mols[0] is not None
        inst = rdkit_mol_to_mol3d_instance(mols[0], np.empty(0), center=False)
        for name, point in TARGETS.items():
            intervals = TargetIntervals.parse({k: [t-.25, t+.25] for k,t in zip(oracle.target_cols, point)}, oracle.target_cols)
            for arm in ['interval', 'point']:
                t = time.perf_counter()
                cls = Experiment if arm == 'interval' else PointExperiment
                exp = cls(inst, ff, oracle, intervals, 'cuda:0', **({'point': point} if arm == 'point' else {}))
                initial = exp.evaluate(inst.coords)
                np.testing.assert_allclose(initial['prediction'], source['final']['prediction'], atol=1e-5, rtol=0)
                np.testing.assert_allclose(initial['rms_force_eV_A'], source['final']['rms_force_eV_A'], atol=1e-4, rtol=0)
                row = {'run': run, 'candidate_id': index, 'target_name': name, 'target': point, 'arm': arm, 'initial': initial}
                try:
                    coords, guide = exp.guide(inst.coords)
                    after_guide = guide['final']
                    coords, reopt = exp.minimize(coords, 'energy_final')
                    final = reopt['final']
                    row.update(status='completed', after_guidance=after_guide, final=final,
                               accepted_guidance_steps=sum(h['accepted'] for h in guide['history']),
                               guidance_trace=guide, reoptimization_trace=reopt)
                    dest = a.output / name / arm / run; dest.mkdir(parents=True, exist_ok=True)
                    write_precise_sdf(dest / f'candidate_{index}.sdf', [mol3d_to_validated_rdkit_mol(replace(inst, coords=coords))])
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    row.update(status='failed', reason=str(exc))
                row.update(evaluation_calls=exp.evaluations, gradient_calls=exp.gradients, elapsed_seconds=time.perf_counter()-t)
                records.append(row)
                (a.output / 'records.json').write_text(json.dumps(records, indent=2))
                print(run,index,name,arm,row['status'],row.get('final',{}).get('prediction'),flush=True)
    summary = {'protocol': protocol, 'groups': {}}
    for name, target in TARGETS.items():
        for arm in ['interval','point']:
            rows = [r for r in records if r['target_name']==name and r['arm']==arm]
            ok = [r for r in rows if r['status']=='completed']
            phases = {}
            for phase in ['initial','after_guidance','final']:
                error = np.abs(np.array([r[phase]['prediction'] for r in ok])-target)
                phases[phase] = {'mae_eV': error.mean(0).tolist() if ok else None,
                    'mean_mae_eV': float(error.mean()) if ok else None,
                    'joint_within025': int((error<=.25).all(1).sum()),
                    'joint_within025_and_rms01': sum(bool((e<=.25).all()) and r[phase]['rms_force_eV_A']<=.1 for e,r in zip(error,ok)),
                    'mean_rms': float(np.mean([r[phase]['rms_force_eV_A'] for r in ok])) if ok else None}
            summary['groups'][name+'/'+arm] = {'completed':len(ok),'failed':len(rows)-len(ok),'phases':phases}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)


if __name__ == '__main__': main()
