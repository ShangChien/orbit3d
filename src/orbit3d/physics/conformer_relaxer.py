"""Finite-step constrained relaxation with true-oracle acceptance and backtracking."""

from dataclasses import replace
import numpy as np
import torch
from ..constraints import TargetIntervals
from ..generator.octree_tokenizer import mol3d_to_validated_rdkit_mol, mol_to_clean_smiles


def project_force(force, gradient):
    norm2 = gradient.square().sum()
    if norm2 <= torch.finfo(force.dtype).eps:
        return force
    return force - torch.clamp((force * gradient).sum(), min=0) / norm2 * gradient


def cap_molecular_displacement(displacement, max_atom_disp):
    """Bound every atom with one scale per molecule, preserving the projected direction."""
    largest = displacement.norm(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
    return displacement * torch.clamp(max_atom_disp / largest.clamp_min(1e-12), max=1)


def relax_conformer_iso_property(
    inst,
    dpa_guidance,
    oracle,
    target_interval,
    n_steps=35,
    step_size=0.008,
    max_atom_disp=0.025,
    prop_grad_weight=0.25,
    tangent_projection=True,
    frozen_indices=(),
    property_tolerance=0.0,
    energy_tolerance=0.0,
    max_backtracks=10,
    coordinate_bounds=None,
):
    if n_steps < 0 or step_size <= 0 or max_atom_disp <= 0 or property_tolerance < 0 or energy_tolerance < 0:
        raise ValueError("Invalid relaxation settings")
    initial_mol = mol3d_to_validated_rdkit_mol(inst)
    if hasattr(oracle, 'validate_molecule'):
        oracle.validate_molecule(initial_mol)
    inst = replace(inst, graph=initial_mol, smiles=mol_to_clean_smiles(initial_mol))
    device = torch.device(dpa_guidance.device_str)
    coordinate_dtype = torch.float64 if inst.coords.dtype == np.float64 else torch.float32
    x = torch.tensor(inst.coords, dtype=coordinate_dtype, device=device)
    symbols = inst.symbols
    if oracle is None and target_interval is not None:
        raise ValueError('Energy-only refinement cannot evaluate property intervals')
    targets = TargetIntervals.parse(target_interval, oracle.target_cols) if oracle is not None else None

    def violations(values):
        return targets.violations(values) if targets is not None else torch.empty(0, device=device)

    movable = torch.ones_like(x)
    if len(frozen_indices):
        movable[list(frozen_indices)] = 0
    evaluations = 0
    gradient_evaluations = 0

    def evaluate(coords):
        nonlocal evaluations
        evaluations += 1
        if coordinate_bounds is not None:
            lower, upper = [torch.as_tensor(b, dtype=coords.dtype, device=coords.device) for b in coordinate_bounds]
            if ((coords < lower) | (coords >= upper)).any():
                raise ValueError('Refinement candidate lies outside the spatial window')
        candidate = replace(inst, coords=coords.detach().cpu().numpy())
        mol = mol3d_to_validated_rdkit_mol(candidate)
        if oracle is None:
            energy, force = dpa_guidance.compute_energy_and_forces(
                coords.detach(), torch.tensor([dpa_guidance.atom_type_map.index(s) for s in symbols], device=device),
                clip_forces=False, zero_com_forces=False, charges=[inst.charge], multiplicities=[inst.multiplicity])
            yp = ys = torch.empty(0, device=device)
        elif getattr(oracle, "dpa_guidance", None) is dpa_guidance:
            yp, ys, energy, force = oracle.forward_shared_force_and_property(coords.detach(), symbols)
        else:
            energy, force = dpa_guidance.compute_energy_and_forces(
                coords.detach(), torch.tensor([dpa_guidance.atom_type_map.index(s) for s in symbols],
                                               dtype=torch.long, device=device),
                clip_forces=False, zero_com_forces=False, charges=[inst.charge], multiplicities=[inst.multiplicity]
            )
            try:
                yp, ys = oracle.predict_3d_mols([mol])
                yp = torch.as_tensor(np.asarray(yp).reshape(-1), device=device, dtype=x.dtype)
                ys = torch.as_tensor(np.asarray(ys).reshape(-1), device=device, dtype=x.dtype)
            except NotImplementedError:
                yp, ys = oracle.forward_differentiable_3d(coords, symbols, smiles=inst.smiles)
        energy = torch.as_tensor(energy).sum()
        if not all(bool(torch.isfinite(v).all()) for v in [energy, force, yp, ys]):
            raise FloatingPointError("Nonfinite force-field or property evaluation")
        return energy.detach(), force.detach(), yp.detach().reshape(-1), ys.detach().reshape(-1)

    e, f, y, std = evaluate(x)
    initial_e = e.clone()
    initial_d = violations(y)
    initial_force = float(f.square().sum(-1).mean().sqrt())
    initial_max_force = float(f.norm(dim=-1).max())
    initial_forces = f.cpu().tolist()
    history = []
    for step in range(n_steps):
        grad = torch.zeros_like(x)
        if violations(y).any():
            try:
                xv = x.detach().clone().requires_grad_(True)
                gradient_evaluations += 1
                yp, _ = oracle.forward_differentiable_3d(xv, symbols, smiles=inst.smiles)
                loss = violations(yp).square().sum()
                if loss.requires_grad:
                    g = torch.autograd.grad(loss, xv, allow_unused=True)[0]
                    if g is not None:
                        grad = g.detach() * movable
            except NotImplementedError:
                pass  # True black-box oracle still controls acceptance.
        if not torch.isfinite(grad).all():
            raise FloatingPointError("Nonfinite property gradient")
        direction = project_force(f * movable, grad) if tangent_projection else f * movable
        direction = (direction - prop_grad_weight * grad) * movable
        accepted = False
        old_d = violations(y)
        for backtrack in range(max_backtracks + 1):
            disp = direction * (step_size * 2 ** (-backtrack))
            disp = cap_molecular_displacement(disp, max_atom_disp)
            candidate = x + disp
            try:
                en, fn, yn, sn = evaluate(candidate)
                dn = violations(yn)
                # No accumulated tolerance drift: both local and initial budgets.
                accept = bool(
                    en <= e + energy_tolerance
                    and en <= initial_e + energy_tolerance
                    and (dn <= old_d + property_tolerance).all()
                    and (dn <= initial_d + property_tolerance).all()
                )
            except (ValueError, FloatingPointError):
                accept = False
            if accept:
                x, e, f, y, std = candidate.detach(), en, fn, yn, sn
                accepted = True
                break
        history.append(
            {
                "step": step,
                "accepted": accepted,
                "backtracks": backtrack,
                "energy": float(e),
                "violations": violations(y).tolist(),
            }
        )
        if not accepted or float(disp.abs().max()) < 1e-8:
            break
    result = replace(inst, coords=x.cpu().numpy())
    e, f, y, std = evaluate(x)  # Validate exact exported coordinates.
    scalar = lambda t: float(t[0]) if t.numel() == 1 else t.tolist()
    return result, {
        "init_force_eva": initial_force,
        "final_force_eva": float(f.square().sum(-1).mean().sqrt()),
        "init_max_force_eva": initial_max_force,
        "final_max_force_eva": float(f.norm(dim=-1).max()),
        "initial_forces_eva": initial_forces,
        "final_forces_eva": f.cpu().tolist(),
        "initial_energy_ev": float(initial_e),
        "final_energy_ev": float(e),
        "evaluation_calls": evaluations,
        "gradient_calls": gradient_evaluations,
        "target_names": list(targets.names) if targets else [],
        "y_pred": scalar(y) if targets else None,
        "y_std": scalar(std) if targets else None,
        "interval_hit": targets.hit(y) if targets else None,
        "history": history,
        "status": ("feasible" if targets.hit(y) else "infeasible") if targets else "energy_only",
    }


def relax_conformer_energy_only(inst, forcefield, **settings):
    """Physical baseline with graph/stereo acceptance and no property evaluator."""
    return relax_conformer_iso_property(inst, forcefield, None, None, **settings)
