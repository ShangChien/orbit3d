"""
Iso-Property Tangent-Projected 3D Conformer Relaxation (`conformer_relaxer.py`)
===============================================================================
Couples `DPA-4` physical restoring forces `F_phys = -grad_X E_DPA4(X)` with the
differentiable 3D property gradient `g_prop = -grad_X L_interval(y_oracle(X))`
via tangent-space orthogonal projection:
  F_phys_perp = F_phys - (<F_phys, g_hat> * g_hat)   (when <F_phys, g_prop> < 0)
  F_coupled   = F_phys_perp + prop_grad_weight * g_prop
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch

from .dpa_forcefield import DPAPotentialGuidance
from ..generator.octree_tokenizer import ATOM_TYPES, Mol3DInstance


def relax_conformer_iso_property(
    inst: Mol3DInstance,
    dpa_guidance: DPAPotentialGuidance,
    oracle: Any,
    target_interval: Union[Tuple[float, float], Dict[str, Tuple[float, float]], List[Tuple[float, float]]],
    n_steps: int = 35,
    step_size: float = 0.008,
    max_atom_disp: float = 0.025,
    prop_grad_weight: float = 0.25,
    tangent_projection: bool = True,
) -> Tuple[Mol3DInstance, Dict[str, Any]]:
    """
    Relaxes a generated 3D molecule on the DPA-4 potential energy surface while
    preserving or driving its 3D conformation inside `target_interval`.
    """
    device = torch.device(dpa_guidance.device_str)
    x = torch.tensor(inst.coords, dtype=torch.float32, device=device)
    atypes = torch.tensor(inst.atom_types, dtype=torch.long, device=device)
    atom_symbols = [ATOM_TYPES[int(a) % len(ATOM_TYPES)] for a in inst.atom_types]
    smiles = inst.smiles or "C"

    best_x = x.clone()
    best_obj = float("inf")
    init_force_norm = 0.0
    final_force_norm = 0.0
    best_y_pred = None
    best_y_std = None

    for step in range(n_steps):
        x_var = x.detach().clone().requires_grad_(True)

        # Shared-backbone DPA-4 oracle computes both F_phys & y_pred in a single pass
        if hasattr(oracle, "forward_shared_force_and_property"):
            y_pred, y_std, e_phys, f_phys = oracle.forward_shared_force_and_property(x_var, atom_symbols)
            loss_prop = oracle.compute_multi_target_interval_loss(y_pred, target_interval)
            if loss_prop.requires_grad:
                loss_prop.backward()
                g_prop = -x_var.grad.detach() if x_var.grad is not None else torch.zeros_like(x)
            else:
                g_prop = torch.zeros_like(x)
        else:
            e_phys_t, f_phys = dpa_guidance.compute_energy_and_forces(
                x_var.detach(), atypes, clip_forces=True, zero_com_forces=True
            )
            e_phys = e_phys_t.squeeze()
            if getattr(oracle, "has_3d_unimol", True):
                y_pred, y_std = oracle.forward_differentiable_3d(x_var, atom_symbols, smiles=smiles)
                loss_prop = oracle.compute_multi_target_interval_loss(y_pred, target_interval)
                if loss_prop.requires_grad:
                    loss_prop.backward()
                    g_prop = -x_var.grad.detach() if x_var.grad is not None else torch.zeros_like(x)
                else:
                    g_prop = torch.zeros_like(x)
            else:
                with torch.no_grad():
                    y_pred, y_std = oracle.forward_differentiable_3d(x_var.detach(), atom_symbols, smiles=smiles)
                    loss_prop = oracle.compute_multi_target_interval_loss(y_pred, target_interval)
                g_prop = torch.zeros_like(x)

        f_norm_val = float(torch.norm(f_phys, dim=-1).mean().item())
        if step == 0:
            init_force_norm = f_norm_val

        obj_val = float(loss_prop.item()) * 10.0 + 0.05 * f_norm_val
        if obj_val <= best_obj:
            best_obj = obj_val
            best_x = x.detach().clone()
            final_force_norm = f_norm_val
            best_y_pred = y_pred.detach().cpu().numpy()
            best_y_std = y_std.detach().cpu().numpy()

        g_norm = torch.norm(g_prop, dim=-1, keepdim=True) + 1e-8
        g_unit = g_prop / g_norm
        g_clipped = g_prop * torch.clamp(5.0 / g_norm, max=1.0)

        if tangent_projection and float(loss_prop.item()) > 1e-5:
            dot = (f_phys * g_unit).sum()
            if float(dot.item()) < 0.0:
                f_phys_proj = f_phys - dot * g_unit
            else:
                f_phys_proj = f_phys
            f_coupled = f_phys_proj + prop_grad_weight * g_clipped
        else:
            f_coupled = f_phys

        f_coupled = f_coupled - f_coupled.mean(dim=0, keepdim=True)
        disp = step_size * f_coupled
        disp_norm = torch.norm(disp, dim=-1, keepdim=True) + 1e-8
        disp = disp * torch.clamp(max_atom_disp / disp_norm, max=1.0)
        x = (x.detach() + disp).detach()
        x = x - x.mean(dim=0, keepdim=True)

    relaxed_inst = Mol3DInstance(
        coords=best_x.cpu().numpy().astype(np.float32),
        atom_types=inst.atom_types.copy(),
        props=inst.props.copy(),
        smiles=inst.smiles,
    )
    info = {
        "init_force_eva": init_force_norm,
        "final_force_eva": final_force_norm,
        "y_pred": float(np.squeeze(best_y_pred)) if best_y_pred is not None and best_y_pred.size == 1 else best_y_pred.tolist(),
        "y_std": float(np.squeeze(best_y_std)) if best_y_std is not None and best_y_std.size == 1 else best_y_std.tolist(),
    }
    return relaxed_inst, info
