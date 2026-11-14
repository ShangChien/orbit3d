"""Four hot-swappable protocol slots for Uni-Loom (`loom.weaving.slots`):
  1. `GeneratorSlot`    — 3D generative prior (Autoregressive Uni-3DAR, Diffusion, or Flow Matching)
  2. `PotentialSlot`    — Universal MLIP force field (DPA-4, HyFF, etc.) for batched strain & relaxation
  3. `RewardOracleSlot` — 3D property oracle for labeling, hindsight relabeling, and RL reward
  4. `AuditOracleSlot`  — Strictly isolated cross-family calibrated oracle for anti-reward-hacking audit
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

from ..generator.molecular_state import (
    ATOM_TYPES,
    Mol3DInstance,
    mol3d_to_validated_rdkit_mol,
    mol_to_clean_smiles,
)
from ..oracles.base import PropertyOracleBase
from ..oracles.unimol2_oracle import InternalUniMol2Oracle, source_features
from ..physics.factory import build_forcefield


DEFAULT_SUPPORTED_ELEMENTS = (
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"
)


def file_sha256(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_unimol2_support(
    mol: Chem.Mol,
    supported_elements: Sequence[str] = DEFAULT_SUPPORTED_ELEMENTS,
    max_atoms: int = 80,
) -> Optional[str]:
    if mol is None:
        return "null_molecule"
    if mol.GetNumAtoms() > max_atoms:
        return f"more_than_{max_atoms}_total_atoms"
    allowed = frozenset(supported_elements)
    unseen = sorted({a.GetSymbol() for a in mol.GetAtoms()} - allowed)
    if unseen:
        return "unsupported_elements:" + ",".join(unseen)
    try:
        source_features(mol, max_atoms=max_atoms)
    except ValueError as exc:
        return str(exc)
    return None


class GeneratorSlot(nn.Module, ABC):
    """Unified contract for 3D molecular generators (autoregressive, diffusion, or flow matching)."""

    slot_role: str = "generator"

    @abstractmethod
    def sample(
        self,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        cfg_scale: float = 1.0,
        xyz_cfg_scale: float = 1.0,
        temperatures: Tuple[float, float, float] = (0.85, 0.30, 0.25),
        seed: Optional[int] = None,
        use_bf16: bool = True,
        electron_parity: Optional[int] = 0,
    ) -> List[Dict[str, Any]]:
        """Generate conditional 3D molecular trajectories."""

    @abstractmethod
    def compute_sft_loss(
        self,
        packed_batch: Dict[str, torch.Tensor],
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        return_per_molecule: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Supervised / hindsight maximum-likelihood training step."""

    @abstractmethod
    def trajectory_distributions(
        self,
        trajectories: List[Dict[str, Any]],
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        temperatures: Tuple[float, float, float] = (0.85, 0.30, 0.25),
        cfg_scale: float = 1.0,
        xyz_cfg_scale: float = 1.0,
        electron_parity: Optional[int] = 0,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return exact step/token log-probability packets `(log_probs, actions, owner_idx)` for GRPO."""


class PotentialSlot(nn.Module):
    """Hot-swappable MLIP interatomic potential slot (`dpa4`, `hyff`, etc.)."""

    slot_role: str = "potential"

    def __init__(
        self,
        backend: str,
        weights: Optional[str] = None,
        atom_types: Sequence[str] = DEFAULT_SUPPORTED_ELEMENTS,
        device: str = "cuda:0",
        *,
        tier: str = "vivace",
        route: str = "auto",
    ) -> None:
        super().__init__()
        self.backend_name = backend
        self.atom_types = list(atom_types)
        self.device_str = device
        if backend == "hyff":
            if weights:
                w_lower = str(weights).lower()
                for candidate_tier in ("presto", "vivace", "moderato", "andante", "largo"):
                    if candidate_tier in w_lower:
                        tier = candidate_tier
                        break
            if route == "auto":
                # Heterogeneous generated molecule batches vary in atom/edge count every step;
                # PyTorch GPU eager (`reference`) avoids 60s CUDA-graph recompilations and bucket caps.
                route = "reference"
        self.engine = build_forcefield(
            backend=backend,
            weights=weights,
            atom_types=self.atom_types,
            device=device,
            tier=tier,
            route=route,
        )

    def _pack_molecules(self, mols: Sequence[Chem.Mol]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.device(self.device_str)
        coords_list = []
        atype_list = []
        batch_idx_list = []
        type_to_idx = {sym: idx for idx, sym in enumerate(self.atom_types)}
        for b_idx, mol in enumerate(mols):
            pos = mol.GetConformer().GetPositions().astype(np.float32)
            syms = [a.GetSymbol() for a in mol.GetAtoms()]
            try:
                at = [type_to_idx[s] for s in syms]
            except KeyError as exc:
                raise ValueError(f"Unsupported element in PotentialSlot ({self.backend_name}): {exc}") from exc
            coords_list.append(torch.from_numpy(pos))
            atype_list.append(torch.tensor(at, dtype=torch.long))
            batch_idx_list.append(torch.full((len(syms),), b_idx, dtype=torch.long))
        if not coords_list:
            return (
                torch.zeros((0, 3), dtype=torch.float32, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )
        return (
            torch.cat(coords_list, dim=0).to(device),
            torch.cat(atype_list, dim=0).to(device),
            torch.cat(batch_idx_list, dim=0).to(device),
        )

    @torch.no_grad()
    def evaluate_molecules(self, mols: Sequence[Chem.Mol], chunk_size: int = 32) -> Dict[str, np.ndarray]:
        """Batched zero-relaxation energy, per-atom energy, `F_max`, and `F_rms` evaluation."""
        n = len(mols)
        if n == 0:
            empty = np.zeros((0,), dtype=np.float64)
            return {"energy_eV": empty, "energy_per_atom_eV": empty, "f_max_eV_A": empty, "f_rms_eV_A": empty}

        energies = np.empty(n, dtype=np.float64)
        e_per_atom = np.empty(n, dtype=np.float64)
        f_max = np.empty(n, dtype=np.float64)
        f_rms = np.empty(n, dtype=np.float64)

        for start in range(0, n, chunk_size):
            sub = mols[start : start + chunk_size]
            coords, atypes, b_idx = self._pack_molecules(sub)
            mol_e, forces = self.engine.compute_energy_and_forces(
                coords, atypes, b_idx, clip_forces=False, zero_com_forces=True
            )
            f_norms = torch.norm(forces.float(), dim=-1)
            for local_i, mol in enumerate(sub):
                g_i = start + local_i
                mask = b_idx == local_i
                n_atoms = max(mol.GetNumAtoms(), 1)
                e_val = float(mol_e[local_i].item())
                energies[g_i] = e_val
                e_per_atom[g_i] = e_val / n_atoms
                fn_i = f_norms[mask]
                f_max[g_i] = float(fn_i.max().item()) if fn_i.numel() else 0.0
                f_rms[g_i] = float(fn_i.square().mean().sqrt().item()) if fn_i.numel() else 0.0

        return {
            "energy_eV": energies,
            "energy_per_atom_eV": e_per_atom,
            "f_max_eV_A": f_max,
            "f_rms_eV_A": f_rms,
        }

    @torch.no_grad()
    def relax_molecules(
        self,
        mols: Sequence[Chem.Mol],
        n_steps: int = 15,
        step_size: float = 0.008,
        max_atom_disp: float = 0.03,
        chunk_size: int = 32,
    ) -> Dict[str, Any]:
        """Batched short-horizon MLIP relaxation with strict molecular-graph preservation check."""
        n = len(mols)
        if n == 0:
            return {
                "relaxed_mols": [],
                "graph_preserved": np.zeros((0,), dtype=bool),
                "init_energy_eV": np.zeros((0,), dtype=np.float64),
                "final_energy_eV": np.zeros((0,), dtype=np.float64),
                "final_f_max_eV_A": np.zeros((0,), dtype=np.float64),
            }

        relaxed_mols: List[Optional[Chem.Mol]] = [None] * n
        graph_preserved = np.zeros(n, dtype=bool)
        init_e = np.empty(n, dtype=np.float64)
        final_e = np.empty(n, dtype=np.float64)
        final_f_max = np.empty(n, dtype=np.float64)
        elem_tuple = tuple(self.atom_types)

        for start in range(0, n, chunk_size):
            sub = mols[start : start + chunk_size]
            coords, atypes, b_idx = self._pack_molecules(sub)
            best_x, e0, e1 = self.engine.relax_conformers(
                coords,
                atypes,
                b_idx,
                n_steps=n_steps,
                step_size=step_size,
                max_atom_disp=max_atom_disp,
            )
            _, forces_after = self.engine.compute_energy_and_forces(
                best_x, atypes, b_idx, clip_forces=False, zero_com_forces=True
            )
            f_norms = torch.norm(forces_after.float(), dim=-1)

            for local_i, orig_mol in enumerate(sub):
                g_i = start + local_i
                mask = b_idx == local_i
                init_e[g_i] = float(e0[local_i].item())
                final_e[g_i] = float(e1[local_i].item())
                final_f_max[g_i] = float(f_norms[mask].max().item())
                new_pos = best_x[mask].detach().cpu().numpy().astype(np.float32)

                orig_smi = mol_to_clean_smiles(orig_mol)
                syms = [a.GetSymbol() for a in orig_mol.GetAtoms()]
                inst = Mol3DInstance(
                    coords=new_pos,
                    atom_types=np.array([elem_tuple.index(s) for s in syms], dtype=np.int64),
                    element_table=elem_tuple,
                )
                try:
                    rebuilt = mol3d_to_validated_rdkit_mol(inst)
                    if mol_to_clean_smiles(rebuilt) == orig_smi:
                        graph_preserved[g_i] = True
                        relaxed_mols[g_i] = rebuilt
                except ValueError:
                    graph_preserved[g_i] = False

        return {
            "relaxed_mols": relaxed_mols,
            "graph_preserved": graph_preserved,
            "init_energy_eV": init_e,
            "final_energy_eV": final_e,
            "final_f_max_eV_A": final_f_max,
        }


class RewardOracleSlot(PropertyOracleBase):
    """Unified hot-swappable property reward & relabeling slot."""

    slot_role: str = "reward_oracle"
    is_audit_only: bool = False

    def __init__(
        self,
        backend: str,
        checkpoint_dir: Path | str,
        device: str = "cuda:0",
        *,
        expected_weight_sha256: Optional[str] = None,
        supported_elements: Sequence[str] = DEFAULT_SUPPORTED_ELEMENTS,
        max_atoms: int = 80,
        forcefield_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backend_name = backend
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device_str = device
        self.supported_elements = tuple(supported_elements)
        self.max_atoms = int(max_atoms)
        self.molecule_calls = 0

        if backend == "unimol2":
            if expected_weight_sha256 is not None:
                actual_sha = file_sha256(self.checkpoint_dir / "model.pt")
                if actual_sha != expected_weight_sha256:
                    raise ValueError(
                        f"Uni-Mol2 weight SHA256 mismatch: expected {expected_weight_sha256}, got {actual_sha}"
                    )
            self.oracle = InternalUniMol2Oracle.load_from_dir(self.checkpoint_dir, device=device)
            self.oracle.model.float().eval().requires_grad_(False)
            self.target_cols = list(self.oracle.target_cols)
            self.output_dim = len(self.target_cols)
        elif backend in ("dpa4", "hyff"):
            from ..oracles.dpa4_oracle import InternalDPA4PropertyOracle

            tier = "vivace"
            if backend == "hyff" and forcefield_path:
                w_lower = Path(forcefield_path).name.lower()
                for candidate_tier in ("presto", "vivace", "moderato", "andante", "largo"):
                    if candidate_tier in w_lower:
                        tier = candidate_tier
                        break
            guidance = build_forcefield(
                backend=backend,
                weights=forcefield_path,
                atom_types=list(self.supported_elements),
                device=device,
                tier=tier,
                route="auto" if backend == "hyff" else "reference",
            )
            self.oracle = InternalDPA4PropertyOracle.load_from_dir(
                str(self.checkpoint_dir),
                guidance,
            )
            self.target_cols = list(self.oracle.target_cols)
            self.output_dim = len(self.target_cols)
        else:
            raise ValueError(f"Unsupported oracle backend: {backend}")

    def check_support(self, mol: Chem.Mol) -> Optional[str]:
        if self.backend_name == "unimol2":
            return check_unimol2_support(mol, self.supported_elements, self.max_atoms)
        if mol is None:
            return "null_molecule"
        if mol.GetNumAtoms() > self.max_atoms:
            return f"more_than_{self.max_atoms}_total_atoms"
        unseen = sorted({a.GetSymbol() for a in mol.GetAtoms()} - set(self.supported_elements))
        return ("unsupported_elements:" + ",".join(unseen)) if unseen else None

    @torch.no_grad()
    def predict_3d_mols(self, mols: List[Chem.Mol], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        if not mols:
            return np.empty((0, self.output_dim), dtype=np.float32), np.empty((0, self.output_dim), dtype=np.float32)
        self.molecule_calls += len(mols)
        if self.backend_name == "unimol2":
            with torch.autocast(device_type=torch.device(self.device_str).type, enabled=False):
                return self.oracle.predict_3d_mols(mols, batch_size=batch_size)
        return self.oracle.predict_3d_mols(mols)

    @torch.no_grad()
    def evaluate_candidates(
        self, mols: Sequence[Optional[Chem.Mol]], batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Optional[str]]]:
        """Filter unsupported structures cleanly and batch-evaluate valid ones."""
        n = len(mols)
        preds = np.full((n, self.output_dim), np.nan, dtype=np.float64)
        stds = np.full((n, self.output_dim), np.nan, dtype=np.float64)
        valid_mask = np.zeros(n, dtype=bool)
        reasons: List[Optional[str]] = [None] * n

        admitted_indices: List[int] = []
        admitted_mols: List[Chem.Mol] = []
        for idx, mol in enumerate(mols):
            if mol is None:
                reasons[idx] = "invalid_3d_graph"
                continue
            err = self.check_support(mol)
            if err is not None:
                reasons[idx] = err
                continue
            admitted_indices.append(idx)
            admitted_mols.append(mol)

        if admitted_mols:
            p_arr, s_arr = self.predict_3d_mols(admitted_mols, batch_size=batch_size)
            for local_i, global_i in enumerate(admitted_indices):
                if np.isfinite(p_arr[local_i]).all():
                    preds[global_i] = p_arr[local_i]
                    stds[global_i] = s_arr[local_i]
                    valid_mask[global_i] = True
                else:
                    reasons[global_i] = "nonfinite_prediction"
        return preds, stds, valid_mask, reasons


class AuditOracleSlot(RewardOracleSlot):
    """Strictly isolated cross-family calibrated oracle for independent evaluation only."""

    slot_role: str = "audit_oracle"
    is_audit_only: bool = True


def assert_not_audit_oracle(oracle: Any) -> None:
    """Hard runtime guard called inside all SFT / Hindsight / RL training loops."""
    if getattr(oracle, "is_audit_only", False) or getattr(oracle, "slot_role", None) == "audit_oracle":
        raise RuntimeError(
            "SECURITY/AUDIT VIOLATION: AuditOracleSlot must never be passed into training, relabeling, or RL loops!"
        )
