"""Four-arm, fixed-input relaxation diagnostic; never changes production defaults."""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS
from orbit3d.constraints import TargetIntervals
from orbit3d.generator.octree_backend import trajectory_to_instance
from orbit3d.generator.molecular_state import ATOM_TYPES, mol3d_to_validated_rdkit_mol
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.physics.factory import build_forcefield
from orbit3d.physics.conformer_relaxer import (
    cap_molecular_displacement, relax_conformer_energy_only, relax_conformer_iso_property,
)
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.reproducibility import lock_deterministic_environment
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes


def reverse_project(force, gradient):
    return gradient - torch.clamp((gradient * force).sum(), min=0) / (force.square().sum() + 1e-12) * force


class Experiment:
    def __init__(self, inst, ff, oracle, targets, device):
        self.inst, self.ff, self.oracle, self.targets, self.device = inst, ff, oracle, targets, device
        self.evaluations = self.gradients = 0

    def guidance_loss(self, prediction):
        return self.targets.violations(prediction).square().sum()

    def guidance_value(self, value):
        return value['interval_loss']

    def evaluate(self, coords):
        mol3d_to_validated_rdkit_mol(replace(self.inst, coords=np.asarray(coords)))
        self.evaluations += 1
        x = torch.tensor(coords, device=self.device, dtype=torch.float32)
        y, std, energy, force = self.oracle.forward_shared_force_and_property(x, self.inst.symbols)
        if not all(torch.isfinite(v).all() for v in (y, std, energy, force)):
            raise FloatingPointError('Nonfinite evaluation')
        f = force.detach().cpu().numpy()
        return {'prediction': y.detach().cpu().reshape(-1).tolist(), 'energy_eV': float(energy.detach()),
                'rms_force_eV_A': float(np.sqrt(np.square(f).sum(-1).mean())),
                'max_force_eV_A': float(np.linalg.norm(f, axis=-1).max()),
                'property_hit': bool(self.targets.hit(y)),
                'interval_loss': float(self.targets.violations(y).square().sum().detach()),
                'forces': f.tolist()}

    def minimize(self, coords, label):
        owner = self
        class PhysicalCalculator(Calculator):
            implemented_properties = ['energy', 'forces']
            def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                value = owner.evaluate(atoms.positions)
                self.results = {'energy': value['energy_eV'], 'forces': np.array(value['forces'])}
        atoms = Atoms(self.inst.symbols, positions=coords)
        atoms.calc = PhysicalCalculator()
        optimizer = LBFGS(atoms, maxstep=.025, logfile=None)
        history = []
        for converged in optimizer.irun(fmax=.1, steps=150):
            f = atoms.get_forces()
            rms = float(np.sqrt(np.square(f).sum(-1).mean()))
            history.append({'step': optimizer.nsteps, 'energy_eV': float(atoms.get_potential_energy()),
                            'rms_force_eV_A': rms, 'max_force_eV_A': float(np.linalg.norm(f, axis=-1).max())})
            if rms <= .1 and np.linalg.norm(f, axis=-1).max() <= .15:
                break
        value = self.evaluate(atoms.positions)
        return atoms.positions.copy(), {'phase': label, 'history': history, 'final': value,
                                       'converged': value['rms_force_eV_A'] <= .1 and value['max_force_eV_A'] <= .15}

    def guide(self, coords):
        x = torch.tensor(coords, device=self.device, dtype=torch.float32)
        start = value = self.evaluate(coords)
        history = []
        for step in range(20):
            if self.guidance_value(value) == 0:
                break
            self.gradients += 1
            xv = x.detach().requires_grad_(True)
            yp, _ = self.oracle.forward_differentiable_3d(xv, self.inst.symbols)
            loss = self.guidance_loss(yp)
            gradient = torch.autograd.grad(loss, xv)[0].detach()
            force = torch.tensor(value['forces'], device=self.device, dtype=x.dtype)
            direction = force - .25 * reverse_project(force, gradient)
            accepted = False
            trials = []
            for backtrack in range(11):
                candidate = x + cap_molecular_displacement(.004 * 2 ** (-backtrack) * direction, .025)
                try:
                    trial = self.evaluate(candidate.cpu().numpy())
                    reasons = []
                    if trial['energy_eV'] > start['energy_eV'] + .05: reasons.append('energy_budget')
                    if trial['rms_force_eV_A'] > .2: reasons.append('force_budget')
                    if self.guidance_value(trial) >= self.guidance_value(value): reasons.append('no_property_improvement')
                except (ValueError, FloatingPointError) as exc:
                    reasons = [str(exc)]
                trials.append({'backtrack': backtrack, 'reasons': reasons})
                if not reasons:
                    x, value, accepted = candidate.detach(), trial, True
                    break
            history.append({'step': step, 'accepted': accepted, 'trials': trials})
            if not accepted: break
        return x.cpu().numpy(), {'phase': 'reverse_guidance', 'initial': start, 'final': value, 'history': history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    freeze_run(a.output, __file__)
    lock_deterministic_environment(42)
    torch.set_num_threads(2)
    config = json.loads((a.runs[0] / 'run_config.json').read_text())
    paths = config['mount_paths']
    ff = build_forcefield('dpa4', paths['forcefield_path'], ATOM_TYPES, a.device)
    oracle = InternalDPA4PropertyOracle.load_from_dir(paths['property_checkpoint_dir'], ff)
    bounds = config['inference_config']['target_interval']
    targets = TargetIntervals.parse(bounds, oracle.target_cols)
    ref = json.loads(a.reference.read_text())
    inputs = {'forcefield': paths['forcefield_path'], 'oracle': paths['property_checkpoint_dir'], 'reference': a.reference}
    for run in a.runs:
        for filename in ['run_config.json', 'raw_trajectories.npz', 'raw_symbols.json']:
            inputs[run.name + '/' + filename] = run / filename
    record_input_hashes(a.output, inputs)
    protocol = {'targets': bounds, 'target_names': list(oracle.target_cols), 'maxstep_A': .025,
                'lbfgs_max_steps_per_phase': 150, 'convergence_rms': .1, 'convergence_max': .15,
                'guidance_max_steps': 20, 'guidance_energy_budget_eV': .05, 'guidance_rms_limit': .2,
                'guidance_step_size': .004, 'guidance_weight': .25,
                'scope': 'Fixed historical 34-candidate cohort, not new conditional generation. Unequal call budgets, shared GPU; no speedup claim.',
                'source_attempts': ref['attempted_per_rule'], 'limit': a.limit}
    (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    records = []
    count = 0
    for run in a.runs:
        other = json.loads((run / 'run_config.json').read_text())
        assert other['inference_config']['target_interval'] == bounds
        archive = np.load(run / 'raw_trajectories.npz')
        symbols = json.loads((run / 'raw_symbols.json').read_text())
        chosen = [r for r in ref['records'] if r['run'] == run.name and r['rule'] == 'no_refinement']
        for original in chosen:
            if a.limit is not None and count >= a.limit: break
            count += 1
            index = original['candidate_id']
            inst = trajectory_to_instance({'atom_symbols': symbols[index], 'positions_np': archive[f'coordinates_{index}']})
            initial = Experiment(inst, ff, oracle, targets, a.device).evaluate(inst.coords)
            np.testing.assert_allclose(initial['prediction'], original['initial']['prediction'], atol=1e-4, rtol=0)
            np.testing.assert_allclose(initial['rms_force_eV_A'], original['initial']['rms_force_eV_A'], atol=1e-4, rtol=0)
            for arm in ['full', 'energy30', 'lbfgs', 'lbfgs_reverse_reopt']:
                t = time.perf_counter()
                exp = Experiment(inst, ff, oracle, targets, a.device)
                row = {'run': run.name, 'candidate_id': index, 'arm': arm, 'initial': initial, 'phases': []}
                try:
                    if arm in ('full', 'energy30'):
                        settings = dict(n_steps=30, step_size=.004, max_atom_disp=.025, prop_grad_weight=.25,
                                        property_tolerance=0., energy_tolerance=0.)
                        if arm == 'full': out, trace = relax_conformer_iso_property(inst, ff, oracle, bounds, **settings)
                        else: out, trace = relax_conformer_energy_only(inst, ff, **settings)
                        coords = out.coords
                        exp.evaluations += trace['evaluation_calls']; exp.gradients += trace['gradient_calls']
                        row['legacy_trace'] = trace
                    else:
                        coords, phase = exp.minimize(inst.coords, 'energy_initial')
                        row['phases'].append(phase)
                        if arm == 'lbfgs_reverse_reopt':
                            if phase['converged']:
                                coords, phase = exp.guide(coords); row['phases'].append(phase)
                                coords, phase = exp.minimize(coords, 'energy_final'); row['phases'].append(phase)
                            else: row['guidance_skipped'] = 'initial_energy_not_converged'
                    final = exp.evaluate(coords)
                    row.update(status='completed', final=final,
                               joint_success=final['property_hit'] and final['rms_force_eV_A'] <= .5)
                    dest = a.output / run.name / arm; dest.mkdir(parents=True, exist_ok=True)
                    write_precise_sdf(dest / f'candidate_{index}.sdf', [mol3d_to_validated_rdkit_mol(replace(inst, coords=coords))])
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    row.update(status='failed', reason=str(exc), joint_success=False)
                row.update(evaluation_calls=exp.evaluations, gradient_calls=exp.gradients, elapsed_seconds=time.perf_counter()-t)
                records.append(row)
                (a.output / 'records.json').write_text(json.dumps(records, indent=2))
                print(run.name, index, arm, row['status'], row['joint_success'], flush=True)
    summary = {'protocol': protocol, 'candidates': count, 'arms': {}}
    for arm in ['full', 'energy30', 'lbfgs', 'lbfgs_reverse_reopt']:
        rows = [r for r in records if r['arm'] == arm]; ok = [r for r in rows if r['status'] == 'completed']
        summary['arms'][arm] = {'completed': len(ok), 'failed': len(rows)-len(ok),
            'property_hits': sum(r['final']['property_hit'] for r in ok),
            'joint_hits_rms05': sum(r['joint_success'] for r in rows),
            'joint_hits_rms01': sum(r['final']['property_hit'] and r['final']['rms_force_eV_A'] <= .1 for r in ok),
            'mean_final_rms': float(np.mean([r['final']['rms_force_eV_A'] for r in ok])) if ok else None,
            'evaluation_calls': sum(r['evaluation_calls'] for r in rows), 'gradient_calls': sum(r['gradient_calls'] for r in rows)}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
