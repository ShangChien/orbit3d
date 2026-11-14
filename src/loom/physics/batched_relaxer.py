"""Frame-batched constrained refinement with independent acceptance and failure state."""

from dataclasses import replace
import numpy as np
import torch
from .conformer_relaxer import cap_molecular_displacement
from ..constraints import TargetIntervals
from ..generator.molecular_state import mol3d_to_validated_rdkit_mol


def relax_conformers_iso_property(instances, forcefield, oracle, target_interval, n_steps=35,
                                 step_size=.008, max_atom_disp=.025, prop_grad_weight=.25,
                                 max_backtracks=10, property_tolerance=0., energy_tolerance=0.,
                                 frozen_indices=None, coordinate_bounds=None):
    if getattr(oracle, 'dpa_guidance', None) is not forcefield:
        raise ValueError('Native batched refinement requires the shared DPA property Oracle')
    if n_steps < 0 or step_size <= 0 or max_atom_disp <= 0 or property_tolerance < 0 or energy_tolerance < 0:
        raise ValueError('Invalid batched refinement settings')
    targets = TargetIntervals.parse(target_interval, oracle.target_cols)
    device = torch.device(forcefield.device_str)
    results = [None] * len(instances)
    groups = {}
    checked = {}
    for i, inst in enumerate(instances):
        try:
            mol = mol3d_to_validated_rdkit_mol(inst)
            if hasattr(oracle, 'validate_molecule'):
                oracle.validate_molecule(mol)
            checked[i] = replace(inst, graph=mol)
            groups.setdefault((len(inst.coords), inst.coords.dtype), []).append(i)
        except ValueError as exc:
            results[i] = (None, {'status': 'failed', 'reason': str(exc)})

    def violations(y):
        return torch.stack([torch.zeros_like(y[:, k]) if interval is None else
                            torch.relu(interval[0] - y[:, k]) + torch.relu(y[:, k] - interval[1])
                            for k, interval in enumerate(targets.bounds)], -1)

    for original_ids in groups.values():
        items = [checked[i] for i in original_ids]
        symbols = [inst.symbols for inst in items]
        coordinate_dtype = torch.float64 if items[0].coords.dtype == np.float64 else torch.float32
        x = torch.tensor(np.stack([m.coords for m in items]), dtype=coordinate_dtype, device=device)
        b, n, _ = x.shape
        k = len(targets.names)
        movable = torch.ones_like(x)
        if frozen_indices is not None:
            for j, i in enumerate(original_ids):
                movable[j, list(frozen_indices[i])] = 0
        evaluations = np.zeros(b, dtype=int)
        gradients = np.zeros(b, dtype=int)
        histories = [[] for _ in range(b)]
        reasons = [''] * b

        def evaluate(values, ids):
            size = len(ids)
            y = torch.full((size, k), torch.nan, device=device)
            std = torch.full_like(y, torch.nan)
            energy = torch.full((size,), torch.nan, dtype=torch.float64, device=device)
            force = torch.full_like(values, torch.nan)
            usable = []
            coordinates = values.detach().cpu().numpy()
            for j, i in enumerate(ids):
                evaluations[i] += 1
                try:
                    if coordinate_bounds is not None:
                        lo, hi = coordinate_bounds
                        if ((coordinates[j] < lo) | (coordinates[j] >= hi)).any():
                            raise ValueError('Outside spatial window')
                    mol3d_to_validated_rdkit_mol(replace(items[i], coords=coordinates[j]))
                    usable.append(j)
                except ValueError as exc:
                    reasons[i] = str(exc)
            if usable:
                with torch.no_grad():
                    yp, sp, ep, fp = oracle.forward_shared_batch(values[usable], [symbols[ids[j]] for j in usable])
                y[usable], std[usable], force[usable] = yp, sp, fp
                energy[usable] = ep.double()
            finite = torch.isfinite(y).all(-1) & torch.isfinite(std).all(-1) & torch.isfinite(energy) & torch.isfinite(force).all((-1, -2))
            return energy, force, y, std, finite

        e, f, y, std, active = evaluate(x, list(range(b)))
        failed = ~active.clone()
        initial_e, initial_f, initial_d = e.clone(), f.clone(), violations(y)
        for step in range(n_steps):
            current_ids = active.nonzero().flatten().tolist()
            if not current_ids:
                break
            grad = torch.zeros_like(x)
            recovery_ids = (active & (violations(y) > 0).any(-1)).nonzero().flatten().tolist()
            if recovery_ids:
                with torch.enable_grad():
                    xv = x[recovery_ids].detach().clone().requires_grad_(True)
                    yp, _ = oracle.forward_differentiable_batch(xv, [symbols[i] for i in recovery_ids])
                    g = torch.autograd.grad(violations(yp).square().sum(), xv)[0]
                gradients[recovery_ids] += 1
                finite = torch.isfinite(g).all((-1, -2))
                for j, i in enumerate(recovery_ids):
                    if not finite[j]:
                        failed[i], active[i], reasons[i] = True, False, 'Nonfinite property gradient'
                    else:
                        grad[i] = g[j].detach() * movable[i]
            force = f * movable
            norm2 = grad.square().sum((-1, -2))
            factor = (force * grad).sum((-1, -2)).clamp_min(0) / norm2.clamp_min(torch.finfo(x.dtype).eps)
            factor = torch.where(norm2 > torch.finfo(x.dtype).eps, factor, 0.)
            direction = (force - factor[:, None, None] * grad - prop_grad_weight * grad) * movable
            old_d = violations(y)
            pending = active.clone()
            accepted = torch.zeros(b, dtype=torch.bool, device=device)
            last_backtrack = np.full(b, max_backtracks)
            tiny = torch.zeros_like(accepted)
            for backtrack in range(max_backtracks + 1):
                ids = pending.nonzero().flatten().tolist()
                if not ids:
                    break
                disp = direction[ids] * (step_size * 2**(-backtrack))
                disp = cap_molecular_displacement(disp, max_atom_disp)
                proposed = x[ids] + disp
                en, fn, yn, sn, finite = evaluate(proposed, ids)
                dn = violations(yn)
                accept = finite & (en <= e[ids] + energy_tolerance) & (en <= initial_e[ids] + energy_tolerance)
                accept &= (dn <= old_d[ids] + property_tolerance).all(-1) & (dn <= initial_d[ids] + property_tolerance).all(-1)
                selected = [i for j, i in enumerate(ids) if accept[j]]
                if selected:
                    x[selected], e[selected], f[selected], y[selected], std[selected] = proposed[accept], en[accept], fn[accept], yn[accept], sn[accept]
                    accepted[selected], pending[selected] = True, False
                    tiny[selected] = disp[accept].abs().amax((-1, -2)) < 1e-8
                    last_backtrack[selected] = backtrack
            for i in current_ids:
                if not failed[i]:
                    histories[i].append({'step': step, 'accepted': bool(accepted[i]), 'backtracks': int(last_backtrack[i]),
                                         'energy': float(e[i]), 'violations': violations(y[i:i+1])[0].tolist()})
            active &= accepted & ~tiny & ~failed
        final_ids = (~failed).nonzero().flatten().tolist()
        if final_ids:
            en, fn, yn, sn, finite = evaluate(x[final_ids], final_ids)
            e[final_ids], f[final_ids], y[final_ids], std[final_ids] = en, fn, yn, sn
            failed[final_ids] |= ~finite
        for i, original in enumerate(original_ids):
            if failed[i]:
                results[original] = (None, {'status': 'failed', 'reason': reasons[i] or 'Nonfinite force/property evaluation',
                                           'history': histories[i], 'initial_forces_eva': initial_f[i].cpu().tolist(),
                                           'final_forces_eva': f[i].cpu().tolist(),
                                           'evaluation_calls': int(evaluations[i]), 'gradient_calls': int(gradients[i])})
                continue
            yp = y[i].tolist() if k > 1 else float(y[i, 0])
            sp = std[i].tolist() if k > 1 else float(std[i, 0])
            hit = targets.hit(y[i])
            results[original] = (replace(items[i], coords=x[i].cpu().numpy()), {
                'status': 'feasible' if hit else 'infeasible', 'interval_hit': hit,
                'y_pred': yp, 'y_std': sp, 'history': histories[i],
                'init_force_eva': float(initial_f[i].square().sum(-1).mean().sqrt()),
                'final_force_eva': float(f[i].square().sum(-1).mean().sqrt()),
                'init_max_force_eva': float(initial_f[i].norm(dim=-1).max()),
                'final_max_force_eva': float(f[i].norm(dim=-1).max()),
                'initial_forces_eva': initial_f[i].cpu().tolist(), 'final_forces_eva': f[i].cpu().tolist(),
                'initial_energy_ev': float(initial_e[i]), 'final_energy_ev': float(e[i]),
                'evaluation_calls': int(evaluations[i]), 'gradient_calls': int(gradients[i]),
            })
    return results
