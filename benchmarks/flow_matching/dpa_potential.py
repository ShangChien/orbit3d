import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class DPAPotentialGuidance(nn.Module):
    """
    Universal Machine Learning Interatomic Potential (MLIP) physical force module
    powered by DeePMD-kit's DPA-3 (DPA-3.1-3M.pt).

    Provides:
      1. Batched evaluation of physical potential energy E_phys (eV) and atomic restoring
         forces F_phys = -grad_x E_phys (eV/A) in a single forward/backward pass via
         spatially separated non-interacting batch packing (separation >> r_cut).
      2. Soft per-atom force norm clipping to prevent short-range Pauli repulsion
         singularities during intermediate flow-matching timesteps.
      3. Terminal local conformational relaxation on the DPA-3 potential energy surface.
    """

    def __init__(
        self,
        model_path: str,
        head: str = "SPICE2",
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

        import copy
        if "model_dict" in self.model_params:
            if head not in self.model_params["model_dict"]:
                available = list(self.model_params["model_dict"].keys())
                raise ValueError(f"Head '{head}' not found in DPA model. Available heads: {available}")
            head_param = copy.deepcopy(self.model_params["model_dict"][head])
            state_dict_head = {"_extra_state": state_dict["_extra_state"]}
            for item in state_dict:
                if f"model.{head}." in item:
                    state_dict_head[item.replace(f"model.{head}.", "model.Default.")] = state_dict[item].clone()
        else:
            head_param = copy.deepcopy(self.model_params)
            state_dict_head = state_dict
            if isinstance(head_param, dict) and "use_compile" in head_param:
                head_param["use_compile"] = False

        # Detect model coordinate precision (DPA4/SeZM uses float32; DPA-3.1 uses float64)
        desc_cfg = head_param.get("descriptor", {}) if isinstance(head_param, dict) else {}
        fit_cfg = head_param.get("fitting_net", {}) if isinstance(head_param, dict) else {}
        prec_str = desc_cfg.get("precision", fit_cfg.get("precision", "float64"))
        self.coord_dtype = torch.float32 if prec_str in ("float32", "single", "fp32") else torch.float64

        model = get_model(head_param).to(device)
        self.wrapper = ModelWrapper(model)
        self.wrapper.load_state_dict(state_dict_head, strict=False)
        self.wrapper.to(device)
        self.wrapper.eval()
        self.dpa_type_map: List[str] = list(model.get_type_map())
        for p in self.wrapper.parameters():
            p.requires_grad_(False)

        if atom_type_map is None:
            atom_type_map = ["C", "H", "N", "O", "F"]
        self.set_atom_type_map(atom_type_map)

    def set_atom_type_map(self, atom_type_map: List[str]):
        """Build index mapping from dataset atom_type_map to DPA-3's 118-element type_map."""
        self.atom_type_map = list(atom_type_map)
        idx_list = []
        for sym in self.atom_type_map:
            if sym in self.dpa_type_map:
                idx_list.append(self.dpa_type_map.index(sym))
            else:
                # Fallback for dummy/mask token ('Se' or '[MASK]') -> Carbon
                idx_list.append(self.dpa_type_map.index("C"))
        self.register_buffer(
            "dataset_to_dpa_idx",
            torch.tensor(idx_list, dtype=torch.long, device=self.device_str),
            persistent=False,
        )

    def map_atom_types(self, dataset_atom_indices: torch.Tensor) -> torch.Tensor:
        """Convert dataset atom type indices (N,) to DPA-3 atom type indices (N,)."""
        clamped = dataset_atom_indices.long().clamp(0, len(self.atom_type_map) - 1)
        return self.dataset_to_dpa_idx.to(dataset_atom_indices.device)[clamped]

    def compute_energy_and_forces(
        self,
        coords: torch.Tensor,
        dataset_atom_indices: torch.Tensor,
        node_batch_idx: Optional[torch.Tensor] = None,
        clip_forces: bool = True,
        zero_com_forces: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate DPA-3 physical energy and forces on a batch of molecules.

        Args:
            coords: (N_total, 3) atomic coordinates in Angstrom.
            dataset_atom_indices: (N_total,) integer atom types in dataset order.
            node_batch_idx: (N_total,) molecule index in [0, B-1] for each atom.
                            If None, assumes a single molecule (B=1).
            clip_forces: whether to soft-clip per-atom force norms to self.max_force_norm.
            zero_com_forces: whether to subtract per-molecule mean force (Zero-CoM).

        Returns:
            mol_energies: (B,) potential energy per molecule in eV.
            forces: (N_total, 3) physical restoring forces F_phys = -grad_x E_phys in eV/A.
        """
        device = coords.device
        n_total = coords.shape[0]
        if node_batch_idx is None:
            node_batch_idx = torch.zeros(n_total, dtype=torch.long, device=device)
        batch_size = int(node_batch_idx.max().item()) + 1 if n_total > 0 else 0

        # Translate each molecule i by (i * batch_separation, 0, 0) so all molecules
        # can be evaluated in a single non-interacting super-frame (> 60 A >> r_cut = 6 A).
        with torch.enable_grad():
            # Compute per-molecule CoM to keep molecules centered near their grid slot
            mol_counts = torch.bincount(node_batch_idx, minlength=batch_size).clamp(min=1).to(coords.dtype)
            mol_sums = torch.zeros((batch_size, 3), dtype=coords.dtype, device=device)
            mol_sums.index_add_(0, node_batch_idx, coords.detach())
            mol_coms = mol_sums / mol_counts.unsqueeze(-1)

            grid_offsets = torch.zeros((batch_size, 3), dtype=coords.dtype, device=device)
            grid_offsets[:, 0] = torch.arange(batch_size, dtype=coords.dtype, device=device) * self.batch_separation

            atom_shifts = (grid_offsets - mol_coms)[node_batch_idx].detach()
            coords_packed = (coords.detach() + atom_shifts).to(self.coord_dtype).unsqueeze(0).requires_grad_(True)
            dpa_atypes = self.map_atom_types(dataset_atom_indices).unsqueeze(0)

            pred_dict = self.wrapper(coords_packed, dpa_atypes)[0]
            forces = pred_dict["force"].squeeze(0).detach()  # (N_total, 3) in eV/A
            if "atom_energy" in pred_dict:
                atom_energies = pred_dict["atom_energy"].squeeze(0).reshape(-1).detach()  # (N_total,) in eV
            else:
                atom_energies = torch.full((n_total,), float(pred_dict["energy"].item()) / max(n_total, 1), dtype=torch.float64, device=device)

        # Aggregate per-molecule energies
        mol_energies = torch.zeros(batch_size, dtype=coords.dtype, device=device)
        mol_energies.index_add_(0, node_batch_idx, atom_energies.to(coords.dtype))

        forces = forces.to(coords.dtype)

        # Replace any NaN/Inf from extreme overlaps
        forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)

        if clip_forces and self.max_force_norm is not None and self.max_force_norm > 0:
            f_norm = torch.norm(forces, dim=-1, keepdim=True) + 1e-8
            scale = torch.clamp(self.max_force_norm / f_norm, max=1.0)
            forces = forces * scale

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
        """
        Perform fast batched terminal physical relaxation on the DPA-3 potential surface
        with adaptive backtracking per molecule (only accepts steps that lower E_phys).

        Returns:
            relaxed_coords: (N_total, 3)
            init_energies: (B,) eV
            final_energies: (B,) eV
        """
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

            # Adaptive step size: slightly increase on success, halve on rejection
            lr = torch.where(improved, (lr * 1.1).clamp(max=step_size * 2.0), lr * 0.5)

        # Re-center each molecule's CoM
        mol_counts = torch.bincount(node_batch_idx, minlength=batch_size).clamp(min=1).to(best_x.dtype)
        x_sums = torch.zeros((batch_size, 3), dtype=best_x.dtype, device=device)
        x_sums.index_add_(0, node_batch_idx, best_x)
        x_coms = x_sums / mol_counts.unsqueeze(-1)
        best_x = best_x - x_coms[node_batch_idx]

        return best_x, init_E, best_E
