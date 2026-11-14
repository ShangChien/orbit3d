"""HyFF energy, force and descriptor adapter with explicit execution routes.

Native and reference engines use the same requested checkpoint. Selected-route
failures propagate; element identity is never replaced to fit the model vocabulary.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem



class HyFFPotentialGuidance(nn.Module):
    """
    Pluggable HyFF (v0.2.0) interatomic potential force field & descriptor extractor
    compatible with all `ORBIT-3D` stages (Stage 4 Gated-GRPO, Stage 5 Iso-Property Relaxation,
    Mode C Shared-Backbone Oracle, and 3D Flow Matching Orthogonal Manifold Projection).
    """

    SUPPORTED_TIERS = ("presto", "vivace", "moderato", "andante", "largo")
    backend_name = 'hyff'

    def __init__(
        self,
        tier: str = "vivace",
        weights: Optional[str] = None,
        atom_type_map: Optional[List[str]] = None,
        device: str = "cuda:0",
        max_force_norm: Optional[float] = 15.0,
        default_charge: float = 0.0,
        default_multiplicity: int = 1,
        route: str = "auto",
        min_clamped_dist: float = 0.35,
    ):
        super().__init__()
        tier_clean = tier.lower().replace("hyff-", "").replace("hyff_", "").strip()
        if tier_clean not in self.SUPPORTED_TIERS:
            raise ValueError(f"Unsupported HyFF tier '{tier}'. Choose from {self.SUPPORTED_TIERS}.")

        self.tier = tier_clean
        self.device_str = str(device)
        self.max_force_norm = max_force_norm
        self.default_charge = float(default_charge)
        self.default_multiplicity = int(default_multiplicity)
        if route not in {'auto', 'reference', 'native'}:
            raise ValueError(f'Unknown HyFF execution route: {route}')
        self.route = route
        self.min_clamped_dist = float(min_clamped_dist)

        from ..compat.hyff.registry import resolve
        from ..compat.hyff.weights import load_model

        self.selection = resolve(self.tier, weights=weights)
        self.model_path = str(self.selection.weights)
        self.hyff_device = "cuda" if self.device_str.startswith("cuda") else "cpu"
        if route == 'native' and self.hyff_device != 'cuda':
            raise ValueError('Native HyFF execution requires CUDA')

        # Load the selected checkpoint for differentiable descriptor calls.
        self.ref_model, self.model_config = load_model(self.selection.weights, self.device_str)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.cutoff = float(self.ref_model.cutoff)
        self._native_cp = None
        self._native_caps: Optional[Tuple[int, int]] = None

        pt = Chem.GetPeriodicTable()
        self._pt = pt
        if atom_type_map is None:
            atom_type_map = ["C", "H", "N", "O", "F"]
        self.set_atom_type_map(atom_type_map)

    def _get_or_create_native_cp(self, n_graphs: int, n_atoms: int):
        from ..compat.hyff.batch.collection import CollectionPotential

        g_cap = max(64, 1 << int(max(n_graphs, 1) - 1).bit_length())
        if self.tier == "largo":
            g_cap = min(max(64, g_cap), 64)
        a_cap = max(2048, 1 << int(max(n_atoms + g_cap + 64, 1024) - 1).bit_length())
        if self.tier == "largo":
            a_cap = max(2048, a_cap)

        req_caps = (g_cap, a_cap)
        if self._native_cp is None or self._native_caps != req_caps:
            if self._native_cp is not None:
                try:
                    self._native_cp.clear()
                except Exception:
                    pass
            self._native_cp = CollectionPotential.from_weights(
                self.selection.weights,
                tier=self.tier,
                device=self.device_str,
                route="native",
                graph_capacity=g_cap,
                atom_capacity=a_cap,
                edge_capacity=max(65536, a_cap * 48),
                neighbor_capacity=128,
            )
            self._native_caps = req_caps
        return self._native_cp

    def close(self) -> None:
        if self._native_cp is not None:
            try:
                self._native_cp.clear()
            except Exception:
                pass
            self._native_cp = None

    def _atomic_numbers(self, symbols):
        numbers = []
        for symbol in symbols:
            try:
                z = int(self._pt.GetAtomicNumber(str(symbol).strip()))
            except RuntimeError as exc:
                raise ValueError(f'Unknown element: {symbol}') from exc
            if not 1 <= z <= 83:
                raise ValueError(f'Element outside HyFF vocabulary: {symbol}')
            numbers.append(z)
        return numbers

    def set_atom_type_map(self, atom_type_map: List[str]) -> None:
        """Maps dataset atom symbols (e.g. `['C', 'H', 'N', 'O', 'Ir']`) to integer atomic numbers $Z \\in [1, 83]$."""
        self.atom_type_map = list(atom_type_map)
        z_list = self._atomic_numbers(self.atom_type_map)
        self.register_buffer(
            "dataset_to_atomic_num",
            torch.tensor(z_list, dtype=torch.long, device=self.device_str),
            persistent=False,
        )

    def map_atomic_numbers(self, dataset_atom_indices: torch.Tensor) -> torch.Tensor:
        if ((dataset_atom_indices < 0) | (dataset_atom_indices >= len(self.atom_type_map))).any():
            raise ValueError("Forcefield atom indices out of range")
        clamped = dataset_atom_indices.long()
        return self.dataset_to_atomic_num.to(dataset_atom_indices.device)[clamped]

    def _get_readout_hook_target(self) -> nn.Module:
        if hasattr(self.ref_model, "readout_norms") and len(self.ref_model.readout_norms) > 0:
            return self.ref_model.readout_norms[-1]
        raise RuntimeError(f"Cannot locate readout_norms on HyFF model {type(self.ref_model)}")

    def probe_native_descriptor_dim(self) -> int:
        """Probes the channel dimension `D_desc` of HyFF's final-layer normalized scalar atom features."""
        from ..compat.hyff.engines.reference import system_batch
        from ..compat.hyff.graph import radius_edges

        captured: Dict[str, int] = {}
        target_mod = self._get_readout_hook_target()

        def _hook(mod, inp, out):
            captured["dim"] = int(out.shape[-1])

        h = target_mod.register_forward_hook(_hook)
        try:
            batch = system_batch(
                np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np.float32),
                np.array([6, 6], dtype=np.int64),
                None,
                self.default_charge,
                self.default_multiplicity,
                self.device_str,
            )
            edges = radius_edges(batch, self.cutoff)
            batch["edge_index"] = edges
            with torch.no_grad():
                self.ref_model.energy(batch, edges)
        finally:
            h.remove()
        if "dim" not in captured:
            raise RuntimeError(f"Failed to probe native descriptor dim on HyFF {self.tier}")
        return captured["dim"]

    def extract_native_descriptor_and_forces(
        self,
        coords: torch.Tensor,
        atom_symbols: List[str],
        charge: Optional[float] = None,
        multiplicity: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same descriptor/energy/force contract as the homogeneous batch route."""
        desc, energy, force = self.extract_native_batch(
            coords[None], [atom_symbols],
            charges=[self.default_charge if charge is None else charge],
            multiplicities=[self.default_multiplicity if multiplicity is None else multiplicity],
        )
        return desc[0], energy[0], force[0]

    def _descriptor_batch(self, coords, symbols, charges=None, multiplicities=None, forces=True):
        """One connected PyTorch graph for a homogeneous batch of molecules.

        Descriptors use the checkpoint's reference graph. This is deliberately
        distinct from the native CUDA-graph energy/force engine, whose descriptor
        outputs do not expose an autograd graph to the property adapter.
        """
        from ..compat.hyff.graph import radius_edges

        size, atoms, _ = coords.shape
        if len(symbols) != size or any(len(row) != atoms for row in symbols):
            raise ValueError('HyFF descriptor batch requires one symbol for every coordinate')
        device = torch.device(self.device_str)
        outer_grad = coords.requires_grad
        captured = {}
        def capture(module, inputs, output):
            captured['descriptor'] = output
        with torch.enable_grad():
            positions = coords.to(device=device, dtype=torch.float32).reshape(-1, 3)
            if not outer_grad:
                positions = positions.detach().requires_grad_(forces)
            batch = {
                'positions': positions,
                'atomic_numbers': torch.tensor([z for row in symbols for z in self._atomic_numbers(row)], device=device),
                'batch': torch.arange(size, device=device).repeat_interleave(atoms),
                'ptr': torch.arange(size + 1, device=device) * atoms,
                'total_charge': torch.as_tensor(charges if charges is not None else [self.default_charge] * size,
                                               dtype=torch.float32, device=device),
                'multiplicity': torch.as_tensor(multiplicities if multiplicities is not None else [self.default_multiplicity] * size,
                                               dtype=torch.long, device=device),
            }
            edges = radius_edges(batch, self.cutoff)
            batch['edge_index'] = edges
            hook = self._get_readout_hook_target().register_forward_hook(capture)
            try:
                energy = self.ref_model.energy(batch, edges)
                force = -torch.autograd.grad(energy.sum(), positions, retain_graph=outer_grad)[0] if forces else None
            finally:
                hook.remove()
            descriptor = captured['descriptor'].reshape(size, atoms, -1)
        if not outer_grad:
            descriptor = descriptor.detach()
        return descriptor, energy.detach(), force.detach().reshape_as(coords).to(coords.dtype) if forces else None

    def extract_native_batch(self, coords, symbols, check_finite=True, charges=None, multiplicities=None):
        descriptor, energy, force = self._descriptor_batch(coords, symbols, charges, multiplicities)
        if check_finite and not all(torch.isfinite(value).all() for value in (descriptor, energy, force)):
            raise FloatingPointError('Nonfinite HyFF descriptor/energy/forces')
        return descriptor, energy, force

    def extract_differentiable_batch(self, coords, symbols):
        descriptor, _, _ = self._descriptor_batch(coords, symbols, forces=False)
        return descriptor

    def extract_differentiable_descriptor(self, coords, symbols):
        return self.extract_differentiable_batch(coords[None], [symbols])[0]

    def _compute_pytorch_gpu(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        node_batch_idx: torch.Tensor,
        batch_size: int,
        charges_t: torch.Tensor,
        mults_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ..compat.hyff.graph import radius_edges

        device = torch.device(self.device_str)
        # Ensure contiguous molecule ordering
        is_sorted = bool(coords.shape[0] <= 1 or (node_batch_idx[1:] >= node_batch_idx[:-1]).all())
        if not is_sorted:
            perm = torch.argsort(node_batch_idx, stable=True)
            inv_perm = torch.argsort(perm)
            coords_s = coords[perm]
            z_s = atomic_nums[perm]
            b_s = node_batch_idx[perm]
        else:
            perm = inv_perm = None
            coords_s = coords
            z_s = atomic_nums
            b_s = node_batch_idx

        counts = torch.bincount(b_s, minlength=batch_size)
        ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        ptr[1:] = torch.cumsum(counts, dim=0)

        pos_in = coords_s.detach().to(device=device, dtype=torch.float32).requires_grad_(True)
        batch = {
            "positions": pos_in,
            "atomic_numbers": z_s.to(device=device, dtype=torch.long),
            "batch": b_s.to(device=device, dtype=torch.long),
            "ptr": ptr,
            "total_charge": charges_t.to(device=device, dtype=torch.float32),
            "multiplicity": mults_t.to(device=device, dtype=torch.long),
        }
        edges = radius_edges(batch, self.cutoff)
        batch["edge_index"] = edges

        with torch.enable_grad():
            mol_e = self.ref_model.energy(batch, edges)
            forces_s = -torch.autograd.grad(mol_e.sum(), pos_in, create_graph=False)[0]

        forces_out = forces_s.detach()
        if inv_perm is not None:
            forces_out = forces_out[inv_perm]

        return mol_e.detach(), forces_out.to(coords.dtype)

    def compute_energy_and_forces(
        self,
        coords: torch.Tensor,
        dataset_atom_indices: torch.Tensor,
        node_batch_idx: Optional[torch.Tensor] = None,
        clip_forces: bool = True,
        zero_com_forces: bool = True,
        charges: Optional[Union[float, List[float], torch.Tensor]] = None,
        multiplicities: Optional[Union[int, List[int], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates HyFF potential energy `mol_energies` `(B,)` in eV and conservative atomic forces
        `forces` `(N_total, 3)` in eV/Å.
        """
        device = coords.device
        n_total = coords.shape[0]
        if n_total == 0:
            return torch.zeros(0, dtype=coords.dtype, device=device), torch.zeros_like(coords)

        if node_batch_idx is None:
            node_batch_idx = torch.zeros(n_total, dtype=torch.long, device=device)
        batch_size = int(node_batch_idx.max().item()) + 1

        atomic_nums = self.map_atomic_numbers(dataset_atom_indices)

        if charges is None:
            charges_t = torch.full((batch_size,), self.default_charge, dtype=torch.float32, device=device)
        elif isinstance(charges, (int, float)):
            charges_t = torch.full((batch_size,), float(charges), dtype=torch.float32, device=device)
        else:
            charges_t = torch.as_tensor(charges, dtype=torch.float32, device=device).reshape(batch_size)

        if multiplicities is None:
            mults_t = torch.full((batch_size,), self.default_multiplicity, dtype=torch.long, device=device)
        elif isinstance(multiplicities, int):
            mults_t = torch.full((batch_size,), int(multiplicities), dtype=torch.long, device=device)
        else:
            mults_t = torch.as_tensor(multiplicities, dtype=torch.long, device=device).reshape(batch_size)

        use_native = self.hyff_device == "cuda" and (
            self.route == "native"
            or (self.route == "auto" and batch_size >= 8 and self.tier in ("presto", "vivace", "moderato"))
        )

        mol_energies: Optional[torch.Tensor] = None
        forces: Optional[torch.Tensor] = None

        if use_native:
            is_sorted = bool(n_total <= 1 or (node_batch_idx[1:] >= node_batch_idx[:-1]).all())
            if not is_sorted:
                perm = torch.argsort(node_batch_idx, stable=True)
                inv_perm = torch.argsort(perm)
                coords_s = coords[perm]
                z_s = atomic_nums[perm]
                b_s = node_batch_idx[perm]
            else:
                inv_perm = None
                coords_s = coords
                z_s = atomic_nums
                b_s = node_batch_idx

            counts = torch.bincount(b_s, minlength=batch_size)
            if not bool((counts > 0).all()):
                raise ValueError('Native HyFF batches require nonempty molecule frames')
            offsets_np = np.zeros(batch_size + 1, dtype=np.int64)
            offsets_np[1:] = np.cumsum(counts.detach().cpu().numpy(), dtype=np.int64)
            pos_np = coords_s.detach().cpu().numpy().astype(np.float32, copy=False)
            z_np = z_s.detach().cpu().numpy().astype(np.int64, copy=False)
            chg_np = charges_t.detach().cpu().numpy().astype(np.float32, copy=False)
            mult_np = mults_t.detach().cpu().numpy().astype(np.int64, copy=False)

            from ..compat.hyff.batch.collection import CapacityClassLimitError

            cp = self._get_or_create_native_cp(batch_size, n_total)
            try:
                out = cp.predict(pos_np, z_np, offsets_np, chg_np, mult_np)
                mol_energies = torch.as_tensor(out["energies"], dtype=torch.float64, device=device)
                f_t = torch.as_tensor(out["forces"], dtype=coords.dtype, device=device)
                forces = f_t[inv_perm] if inv_perm is not None else f_t
            except CapacityClassLimitError:
                self.close()
                mol_energies, forces = self._compute_pytorch_gpu(
                    coords=coords,
                    atomic_nums=atomic_nums,
                    node_batch_idx=node_batch_idx,
                    batch_size=batch_size,
                    charges_t=charges_t,
                    mults_t=mults_t,
                )

        else:
            mol_energies, forces = self._compute_pytorch_gpu(
                coords=coords,
                atomic_nums=atomic_nums,
                node_batch_idx=node_batch_idx,
                batch_size=batch_size,
                charges_t=charges_t,
                mults_t=mults_t,
            )

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
        charges: Optional[Union[float, List[float], torch.Tensor]] = None,
        multiplicities: Optional[Union[int, List[int], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = coords.device
        n_total = coords.shape[0]
        if node_batch_idx is None:
            node_batch_idx = torch.zeros(n_total, dtype=torch.long, device=device)
        batch_size = int(node_batch_idx.max().item()) + 1 if n_total > 0 else 0

        x = coords.clone()
        best_E, forces = self.compute_energy_and_forces(
            x,
            dataset_atom_indices,
            node_batch_idx,
            clip_forces=True,
            zero_com_forces=True,
            charges=charges,
            multiplicities=multiplicities,
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
                x_cand,
                dataset_atom_indices,
                node_batch_idx,
                clip_forces=True,
                zero_com_forces=True,
                charges=charges,
                multiplicities=multiplicities,
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
