"""
DPA-4 / DeePMD Interatomic Potential Force Field & Native Equivariant Descriptor Extractor.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class DPAPotentialGuidance(nn.Module):
    """
    Universal Machine Learning Interatomic Potential (MLIP) physical force & descriptor module
    supporting all DPA-4 scales (`Nano`, `Mini`, `Neo`, `Air`, `Plus`) and `DPA-3.1`.
    """

    def __init__(
        self,
        model_path: str,
        head: str = "Default",
        atom_type_map: Optional[List[str]] = None,
        device: str = "cuda:0",
        max_force_norm: float = 15.0,
        batch_separation_angstrom: float = 60.0,
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.head = head
        self.device_str = device
        self.max_force_norm = max_force_norm
        self.batch_separation = batch_separation_angstrom

        from deepmd.pt.model.model import get_model
        from deepmd.pt.train.wrapper import ModelWrapper

        state_dict = torch.load(self.model_path, map_location=device, weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model_params = state_dict["_extra_state"]["model_params"]

        if "model_dict" in self.model_params:
            if head not in self.model_params["model_dict"]:
                available = list(self.model_params["model_dict"].keys())
                raise ValueError(f"Head '{head}' not found in DPA model. Available heads: {available}")
            head_param = copy.deepcopy(self.model_params["model_dict"][head])
            state_dict_head = {"_extra_state": state_dict["_extra_state"]}
            for item in state_dict:
                if f"model.{head}." in item:
                    state_dict_head[item.replace(f"model.{head}.", "model.Default.")] = state_dict[
                        item
                    ].clone()
        else:
            head_param = copy.deepcopy(self.model_params)
            state_dict_head = state_dict
            if isinstance(head_param, dict) and "use_compile" in head_param:
                head_param["use_compile"] = False

        desc_cfg = head_param.get("descriptor", {}) if isinstance(head_param, dict) else {}
        fit_cfg = head_param.get("fitting_net", {}) if isinstance(head_param, dict) else {}
        prec_str = desc_cfg.get("precision", fit_cfg.get("precision", "float64"))
        self.coord_dtype = torch.float32 if prec_str in ("float32", "single", "fp32") else torch.float64

        model = get_model(head_param).to(device)
        self.wrapper = ModelWrapper(model)
        incompatible = self.wrapper.load_state_dict(state_dict_head, strict=False)
        if incompatible.missing_keys:
            raise ValueError(f"Missing forcefield parameters: {incompatible.missing_keys}")
        self.wrapper.to(device)
        self.wrapper.eval()
        self.dpa_type_map: List[str] = list(model.get_type_map())
        for p in self.wrapper.parameters():
            p.requires_grad_(False)

        if atom_type_map is None:
            atom_type_map = ["C", "H", "N", "O", "F"]
        self.set_atom_type_map(atom_type_map)

    @property
    def dpa_model(self) -> nn.Module:
        return self.wrapper.model["Default"]

    def _get_fitting_hook_target(self) -> nn.Module:
        """
        Returns the `fitting_net` submodule whose `forward(descriptor, ...)` receives
        100% of DPA-4's (`SeZMModel`) or DPA-3.1's native multi-body equivariant descriptor
        tensor `(nf, nloc, D_desc)` as its first positional argument `inp[0]`.
        """
        m = self.dpa_model
        if hasattr(m, "atomic_model") and hasattr(m.atomic_model, "fitting_net"):
            return m.atomic_model.fitting_net
        if hasattr(m, "get_fitting_net") and m.get_fitting_net() is not None:
            return m.get_fitting_net()
        raise RuntimeError(f"Cannot locate fitting_net on DPA model {type(m)}")

    def probe_native_descriptor_dim(self) -> int:
        """Probes the exact channel dimension `D_desc` of DPA-4's native equivariant descriptor."""
        captured: Dict[str, int] = {}
        target_mod = self._get_fitting_hook_target()

        def _hook(mod, inp, out):
            desc_t = inp[0]
            nloc = desc_t.shape[1]
            captured["dim"] = int(desc_t.reshape(nloc, -1).shape[-1])

        h = target_mod.register_forward_hook(_hook)
        try:
            with torch.enable_grad():
                dummy_c = torch.zeros((1, 2, 3), dtype=self.coord_dtype, device=self.device_str)
                dummy_c[0, 1, 0] = 1.4
                dummy_a = torch.zeros((1, 2), dtype=torch.long, device=self.device_str)
                self.wrapper(dummy_c.requires_grad_(True), dummy_a)
        finally:
            h.remove()
        if "dim" not in captured:
            raise RuntimeError("Failed to capture native DPA-4 descriptor dimension from fitting_net.")
        return captured["dim"]

    def extract_native_descriptor_and_forces(
        self,
        coords: torch.Tensor,
        atom_symbols: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward/backward pass through frozen DPA-4 that extracts:
          1. `native_descriptor`: `(N, D_desc)` — 100% of DPA-4's internal multi-layer equivariant
             per-atom representation (`inp[0]` passed into `SeZMEnergyFittingNet`), differentiable w.r.t. `coords`!
          2. `energy_ev`: scalar physical potential energy (eV)
          3. `forces_eva`: `(N, 3)` physical atomic restoring forces `F_phys` (eV/Å)
        """
        atypes = torch.tensor(
            [[self.dpa_type_map.index(s) for s in atom_symbols]],
            dtype=torch.long,
            device=coords.device,
        )
        coords_in = coords.to(self.coord_dtype).unsqueeze(0)
        if not coords_in.requires_grad:
            coords_in = coords_in.requires_grad_(True)

        captured: Dict[str, torch.Tensor] = {}
        target_mod = self._get_fitting_hook_target()

        def _hook(mod, inp, out):
            desc_t = inp[0]
            captured["descriptor"] = desc_t.reshape(desc_t.shape[1], -1)

        h = target_mod.register_forward_hook(_hook)
        try:
            with torch.enable_grad():
                pred_dict = self.wrapper(coords_in, atypes)[0]
        finally:
            h.remove()

        native_desc = captured["descriptor"].float()
        forces = pred_dict["force"].squeeze(0).detach().to(coords.dtype)
        energy = pred_dict["energy"].squeeze().detach().to(coords.dtype)
        if not all(torch.isfinite(v).all() for v in (native_desc, energy, forces)):
            raise FloatingPointError("Nonfinite DPA descriptor/energy/forces")
        return native_desc, energy, forces

    def extract_differentiable_descriptor(self, coords, atom_symbols):
        """SeZM embedding path retains coordinate AD; force evaluation detaches edge leaves.

        This is deliberately eager: the inference/force trace may sever upstream
        coordinate gradients. Unsupported backends must use black-box acceptance.
        """
        import inspect

        model = self.dpa_model
        if (
            not hasattr(model, "build_neighbor_list")
            or "embedding_only" not in inspect.signature(model.forward_common_lower).parameters
        ):
            raise NotImplementedError("No verified coordinate-gradient descriptor path")
        atypes = torch.tensor(
            [[self.dpa_type_map.index(s) for s in atom_symbols]], device=coords.device, dtype=torch.long
        )
        coord = coords.to(self.coord_dtype).unsqueeze(0)
        with torch.enable_grad():
            edges = model.build_neighbor_list(coord, atypes, None)
            result = model.forward_common_lower(
                edges.coord,
                edges.atype,
                edges.edge_index,
                edges.edge_vec,
                edges.edge_scatter_index,
                edges.edge_mask,
                embedding_only=True,
                use_compile=False,
            )
            descriptor = result["descriptor"].reshape(len(atom_symbols), -1).float()
        if not torch.isfinite(descriptor).all():
            raise FloatingPointError("Nonfinite differentiable descriptor")
        if coords.requires_grad:
            if (
                not descriptor.requires_grad
                or torch.autograd.grad(descriptor.sum(), coords, retain_graph=True, allow_unused=True)[0]
                is None
            ):
                raise NotImplementedError("Descriptor computation is disconnected from input coordinates")
        return descriptor

    def extract_native_batch(self, coords, atom_symbols):
        """Evaluate equal-size independent frames in one native DeePMD call."""
        batch, atoms, _ = coords.shape
        atypes = torch.tensor([[self.dpa_type_map.index(s) for s in row] for row in atom_symbols],
                              device=coords.device, dtype=torch.long)
        captured = {}

        def capture(module, inputs, output):
            captured['descriptor'] = inputs[0].reshape(batch, atoms, -1).float()

        hook = self._get_fitting_hook_target().register_forward_hook(capture)
        try:
            with torch.enable_grad():
                prediction = self.wrapper(coords.to(self.coord_dtype).requires_grad_(True), atypes)[0]
        finally:
            hook.remove()
        descriptor = captured['descriptor'].detach()
        energy = prediction['energy'].reshape(batch).detach().to(coords.dtype)
        forces = prediction['force'].reshape(batch, atoms, 3).detach().to(coords.dtype)
        if not all(torch.isfinite(v).all() for v in [descriptor, energy, forces]):
            raise FloatingPointError('Nonfinite native DPA batch')
        return descriptor, energy, forces

    def set_atom_type_map(self, atom_type_map: List[str]) -> None:
        self.atom_type_map = list(atom_type_map)
        idx_list = []
        for sym in self.atom_type_map:
            if sym in self.dpa_type_map:
                idx_list.append(self.dpa_type_map.index(sym))
            else:
                raise ValueError(f"Unsupported forcefield element: {sym}")
        self.register_buffer(
            "dataset_to_dpa_idx",
            torch.tensor(idx_list, dtype=torch.long, device=self.device_str),
            persistent=False,
        )

    def map_atom_types(self, dataset_atom_indices: torch.Tensor) -> torch.Tensor:
        if ((dataset_atom_indices < 0) | (dataset_atom_indices >= len(self.atom_type_map))).any():
            raise ValueError("Forcefield atom indices out of range")
        clamped = dataset_atom_indices.long()
        return self.dataset_to_dpa_idx.to(dataset_atom_indices.device)[clamped]

    def compute_energy_and_forces(
        self,
        coords: torch.Tensor,
        dataset_atom_indices: torch.Tensor,
        node_batch_idx: Optional[torch.Tensor] = None,
        clip_forces: bool = True,
        zero_com_forces: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = coords.device
        n_total = coords.shape[0]
        if node_batch_idx is None:
            node_batch_idx = torch.zeros(n_total, dtype=torch.long, device=device)
        batch_size = int(node_batch_idx.max().item()) + 1 if n_total > 0 else 0

        with torch.enable_grad():
            mol_counts = torch.bincount(node_batch_idx, minlength=batch_size).clamp(min=1).to(coords.dtype)
            mol_sums = torch.zeros((batch_size, 3), dtype=coords.dtype, device=device)
            mol_sums.index_add_(0, node_batch_idx, coords.detach())
            mol_coms = mol_sums / mol_counts.unsqueeze(-1)

            grid_offsets = torch.zeros((batch_size, 3), dtype=coords.dtype, device=device)
            grid_offsets[:, 0] = (
                torch.arange(batch_size, dtype=coords.dtype, device=device) * self.batch_separation
            )

            atom_shifts = (grid_offsets - mol_coms)[node_batch_idx].detach()
            coords_packed = (
                (coords.detach() + atom_shifts).to(self.coord_dtype).unsqueeze(0).requires_grad_(True)
            )
            dpa_atypes = self.map_atom_types(dataset_atom_indices).unsqueeze(0)

            pred_dict = self.wrapper(coords_packed, dpa_atypes)[0]
            forces = pred_dict["force"].squeeze(0).detach()
            if "atom_energy" in pred_dict:
                atom_energies = pred_dict["atom_energy"].squeeze(0).reshape(-1).detach()
            else:
                if batch_size > 1:
                    raise ValueError("Batched energies require atom-resolved energy output")
                atom_energies = torch.full(
                    (n_total,),
                    float(pred_dict["energy"].item()) / max(n_total, 1),
                    dtype=torch.float64,
                    device=device,
                )

        mol_energies = torch.zeros(batch_size, dtype=coords.dtype, device=device)
        mol_energies.index_add_(0, node_batch_idx, atom_energies.to(coords.dtype))
        forces = forces.to(coords.dtype)

        if not torch.isfinite(mol_energies).all() or not torch.isfinite(forces).all():
            raise FloatingPointError("Nonfinite MLIP energy/forces")

        if clip_forces and self.max_force_norm is not None and self.max_force_norm > 0:
            f_norm = torch.norm(forces, dim=-1, keepdim=True) + 1e-8
            forces = forces * torch.clamp(self.max_force_norm / f_norm, max=1.0)

        if zero_com_forces and batch_size > 0:
            mol_counts = torch.bincount(node_batch_idx, minlength=batch_size).clamp(min=1).to(forces.dtype)
            f_sums = torch.zeros((batch_size, 3), dtype=forces.dtype, device=device)
            f_sums.index_add_(0, node_batch_idx, forces)
            f_means = f_sums / mol_counts.unsqueeze(-1)
            forces = forces - f_means[node_batch_idx]

        return mol_energies, forces

    @torch.no_grad()
    def relax_conformers(
        self,
        coords: torch.Tensor,
        dataset_atom_indices: torch.Tensor,
        node_batch_idx: Optional[torch.Tensor] = None,
        n_steps: int = 15,
        step_size: float = 0.008,
        max_atom_disp: float = 0.03,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = coords.device
        n_total = coords.shape[0]
        if node_batch_idx is None:
            node_batch_idx = torch.zeros(n_total, dtype=torch.long, device=device)
        batch_size = int(node_batch_idx.max().item()) + 1 if n_total > 0 else 0

        x = coords.clone()
        best_E, forces = self.compute_energy_and_forces(
            x, dataset_atom_indices, node_batch_idx, clip_forces=True, zero_com_forces=True
        )
        init_E = best_E.clone()
        best_x = x.clone()
        lr = torch.full((batch_size,), step_size, dtype=coords.dtype, device=device)

        for _ in range(n_steps):
            disp = forces * lr[node_batch_idx].unsqueeze(-1)
            disp_norm = torch.norm(disp, dim=-1, keepdim=True) + 1e-8
            disp = disp * torch.clamp(max_atom_disp / disp_norm, max=1.0)
            x_cand = best_x + disp

            cand_E, cand_forces = self.compute_energy_and_forces(
                x_cand, dataset_atom_indices, node_batch_idx, clip_forces=True, zero_com_forces=True
            )
            improved = cand_E < best_E
            improved_nodes = improved[node_batch_idx]

            best_x = torch.where(improved_nodes.unsqueeze(-1), x_cand, best_x)
            forces = torch.where(improved_nodes.unsqueeze(-1), cand_forces, forces)
            best_E = torch.where(improved, cand_E, best_E)
            lr = torch.where(improved, (lr * 1.1).clamp(max=step_size * 2.0), lr * 0.5)

        mol_counts = torch.bincount(node_batch_idx, minlength=batch_size).clamp(min=1).to(best_x.dtype)
        x_sums = torch.zeros((batch_size, 3), dtype=best_x.dtype, device=device)
        x_sums.index_add_(0, node_batch_idx, best_x)
        x_coms = x_sums / mol_counts.unsqueeze(-1)
        best_x = best_x - x_coms[node_batch_idx]
        return best_x, init_E, best_E
