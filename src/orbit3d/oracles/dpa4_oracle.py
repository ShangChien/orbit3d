"""Frozen DPA descriptors with independent bootstrap property adapters.

Property evaluation and native forces share the force-forward descriptor. Coordinate
derivatives use the separate embedding path, since the force path detaches edge
vectors. Equal-size frames and their property adapters support native batching.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from .base import PropertyOracleBase
from ..physics.dpa_forcefield import DPAPotentialGuidance
from ..utils.dataset_io import (
    fit_target_scalers,
    inverse_single_scaler_torch,
    transform_with_scalers,
)


class _DPA4EquivariantPropertyAdapter(nn.Module):
    """
    3D Distance-Biased Attention Adapter + Multi-Head Gated Pooling sitting directly on top
    of frozen `DPA-4` native equivariant descriptor features (`atom_repr` of shape `(N, D_desc)`).
    """

    def __init__(self, desc_dim: int, hidden_dim: int = 256, output_dim: int = 1, n_ensemble_heads: int = 5):
        super().__init__()
        self.in_proj = nn.Linear(desc_dim, hidden_dim)
        self.dist_rbf = nn.Linear(16, 4)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.gates = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_ensemble_heads)])
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, output_dim),
                )
                for _ in range(n_ensemble_heads)
            ]
        )
        self.register_buffer("rbf_centers", torch.linspace(0.5, 6.0, 16))

    def forward(self, atom_repr: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        single = atom_repr.ndim == 2
        if single:
            atom_repr, coords = atom_repr[None], coords[None]
        h = self.norm1(self.in_proj(atom_repr))
        B, N, H = h.shape
        dist = torch.cdist(coords, coords, p=2.0)
        rbf = torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.rbf_centers) / 0.4) ** 2)
        attn_bias = self.dist_rbf(rbf).permute(0, 3, 1, 2)

        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, 4, H // 4).transpose(1, 2)
        k = k.view(B, N, 4, H // 4).transpose(1, 2)
        v = v.view(B, N, 4, H // 4).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / ((H // 4) ** 0.5) + attn_bias
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, N, H)
        h = h + self.out_proj(ctx)
        h = h + self.ffn(self.norm2(h))

        mean_pool = h.mean(dim=1)
        head_preds = []
        for gate, mlp in zip(self.gates, self.heads):
            w = F.softmax(gate(h), dim=1)
            attn_pool = (h * w).sum(dim=1)
            mol_vec = torch.cat([mean_pool, attn_pool], dim=-1)
            head_preds.append(mlp(mol_vec))

        stacked = torch.stack(head_preds, dim=0)
        if single:
            stacked = stacked[:, 0]
        return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)


class _BootstrapAdapters(nn.Module):
    """Independent complete adapters sharing only a frozen descriptor provider."""

    def __init__(self, desc_dim, output_dim, members, hidden_dim=128):
        super().__init__()
        self.members = nn.ModuleList(
            [_DPA4EquivariantPropertyAdapter(desc_dim, hidden_dim, output_dim, 1) for _ in range(members)]
        )

    def forward_members(self, features, coords):
        return torch.stack([member(features, coords)[0] for member in self.members])

    def forward(self, features, coords):
        values = self.forward_members(features, coords)
        return values.mean(0), values.std(0, unbiased=False)


class InternalDPA4PropertyOracle(PropertyOracleBase):
    """
    Shared-Backbone `DPA-4` Force Field + Differentiable Property Predictor.
    Uses 100% of `DPA-4`'s native equivariant descriptor tensor (`inp[0]` of `fitting_net`).
    """

    def __init__(
        self,
        dpa_guidance: DPAPotentialGuidance,
        output_dim: int = 1,
        target_cols: Optional[List[str]] = None,
        n_ensemble_heads: int = 5,
    ):
        super().__init__()
        self.dpa_guidance = dpa_guidance
        self.output_dim = output_dim
        self.n_ensemble_heads = n_ensemble_heads
        self.target_cols = target_cols or [f"TARGET_{i}" for i in range(output_dim)]
        self.has_3d_unimol = True
        self.device = torch.device(dpa_guidance.device_str)
        self.scalers: List = []

        self.desc_dim = self.dpa_guidance.probe_native_descriptor_dim()
        self.adapter = _BootstrapAdapters(self.desc_dim, output_dim, n_ensemble_heads).to(self.device)

    @staticmethod
    def validate_molecule(mol):
        multiplicity = int(mol.GetProp('SpinMultiplicity')) if mol.HasProp('SpinMultiplicity') else 1
        if Chem.GetFormalCharge(mol) != 0 or multiplicity != 1 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            raise ValueError('OCELOT DPA property adapters support neutral closed-shell molecules only')

    def forward_shared_force_and_property(
        self,
        coords: torch.Tensor,
        atom_symbols: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-pass simultaneous evaluation of:
          1. `y_pred_real`: `(K,)` or scalar property prediction
          2. `y_std_real`:  `(K,)` or scalar epistemic uncertainty across ensemble heads
          3. `energy_ev`:   scalar physical potential energy from `DPA-4`
          4. `forces_eva`:  `(N, 3)` physical atomic forces `F_phys` from `DPA-4` (100% unchanged!)
        """
        native_desc, energy, forces = self.dpa_guidance.extract_native_descriptor_and_forces(
            coords, atom_symbols
        )
        if coords.requires_grad and hasattr(self.dpa_guidance, "extract_differentiable_descriptor"):
            native_desc = self.dpa_guidance.extract_differentiable_descriptor(coords, atom_symbols)
        y_pred_real, y_std_real = self._properties_from_descriptor(native_desc, coords)
        if self.output_dim == 1:
            return y_pred_real.squeeze(0), y_std_real.squeeze(0), energy, forces
        return y_pred_real, y_std_real, energy, forces

    def _properties_from_descriptor(self, native_desc, coords):
        normalized = self.adapter.forward_members(native_desc, coords.float())
        physical = torch.stack(
            [
                torch.stack(
                    [
                        inverse_single_scaler_torch(row[..., k], self.scalers[k] if self.scalers else None)
                        for k in range(self.output_dim)
                    ], dim=-1
                )
                for row in normalized
            ]
        )
        y_pred_real = physical.mean(0)
        y_std_real = physical.std(0, unbiased=False)

        return y_pred_real, y_std_real

    def forward_differentiable_3d(
        self,
        coords: torch.Tensor,
        atom_symbols: List[str],
        smiles: str = "C",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.dpa_guidance, "extract_differentiable_descriptor"):
            descriptor = self.dpa_guidance.extract_differentiable_descriptor(coords, atom_symbols)
        else:
            descriptor, _, _ = self.dpa_guidance.extract_native_descriptor_and_forces(coords, atom_symbols)
            if coords.requires_grad and (
                not descriptor.requires_grad
                or torch.autograd.grad(descriptor.sum(), coords, retain_graph=True, allow_unused=True)[0]
                is None
            ):
                raise NotImplementedError("Descriptor gradients are unavailable; use true-oracle acceptance")
        yp, ys = self._properties_from_descriptor(descriptor, coords)
        if self.output_dim == 1:
            return yp.squeeze(0), ys.squeeze(0)
        return yp, ys

    @torch.no_grad()
    def predict_3d_mols(self, mols: List[Chem.Mol]) -> Tuple[np.ndarray, np.ndarray]:
        self.adapter.eval()
        preds, stds = [], []
        for mol in mols:
            self.validate_molecule(mol)
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, device=self.device)
            with torch.enable_grad():
                yp, ys, _, _ = self.forward_shared_force_and_property(pos, atoms)
            preds.append(yp.detach().cpu().numpy())
            stds.append(ys.detach().cpu().numpy())
        return np.array(preds, dtype=np.float32), np.array(stds, dtype=np.float32)

    @torch.no_grad()
    def predict_3d_mols_batched(self, mols, batch_size=16):
        result = self.evaluate_3d_mols_batched(mols, batch_size)
        failed = [i for i, reason in enumerate(result['errors']) if reason]
        if failed:
            raise ValueError(f'Property evaluation failed for rows {failed}: {[result["errors"][i] for i in failed]}')
        return result['mean'], result['std']

    @torch.no_grad()
    def evaluate_3d_mols_batched(self, mols, batch_size=16):
        """Group by atom count; vectorize native descriptors and property adapters."""
        self.adapter.eval()
        groups = {}
        errors = [''] * len(mols)
        for i, mol in enumerate(mols):
            try:
                self.validate_molecule(mol)
                groups.setdefault(mol.GetNumAtoms(), []).append(i)
            except ValueError as exc:
                errors[i] = str(exc)
        mean = np.full((len(mols), self.output_dim), np.nan, dtype=np.float32)
        std = np.full_like(mean, np.nan)
        energies = np.full(len(mols), np.nan, dtype=np.float64)
        forces = [None] * len(mols)
        for ids in groups.values():
            for begin in range(0, len(ids), batch_size):
                ix = ids[begin:begin + batch_size]
                coords = torch.tensor(np.stack([mols[i].GetConformer().GetPositions() for i in ix]),
                                      dtype=torch.float32, device=self.device)
                symbols = [[atom.GetSymbol() for atom in mols[i].GetAtoms()] for i in ix]
                features, energy, force = self.dpa_guidance.extract_native_batch(coords, symbols, check_finite=False)
                yp, ys = self._properties_from_descriptor(features, coords)
                mean[ix], std[ix] = yp.cpu().numpy(), ys.cpu().numpy()
                energies[ix] = energy.cpu().numpy()
                finite = torch.isfinite(yp).all(-1) & torch.isfinite(ys).all(-1) & torch.isfinite(energy) & torch.isfinite(force).all((-1, -2))
                for j, i in enumerate(ix):
                    forces[i] = force[j].cpu().numpy()
                    if not finite[j]:
                        errors[i] = 'Nonfinite DPA descriptor/property/energy/forces'
        return {'mean': mean, 'std': std, 'energy_eV': energies, 'forces_eV_A': forces, 'errors': errors,
                'target_cols': self.target_cols, 'target_units': getattr(self, 'target_units', None)}

    def forward_shared_batch(self, coords, atom_symbols):
        descriptor, energy, force = self.dpa_guidance.extract_native_batch(coords, atom_symbols, check_finite=False)
        mean, std = self._properties_from_descriptor(descriptor, coords)
        return mean, std, energy, force

    def forward_differentiable_batch(self, coords, atom_symbols):
        descriptor = self.dpa_guidance.extract_differentiable_batch(coords, atom_symbols)
        return self._properties_from_descriptor(descriptor, coords)

    def fit_from_dataset(
        self,
        mols,
        targets_2d,
        save_dir,
        epochs=35,
        lr=2e-4,
        scaler_mode="standard",
        seed=42,
        validation_mols=None,
        validation_targets=None,
        batch_size=1,
    ):
        from sklearn.model_selection import GroupShuffleSplit

        os.makedirs(save_dir, exist_ok=True)
        if len(mols) < 2 or epochs < 1:
            raise ValueError("Need training records and positive epochs")
        y = np.asarray(targets_2d, dtype=np.float32)
        if validation_mols is None:
            groups = [Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True) for m in mols]
            tr, va = next(
                GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(y, groups=groups)
            )
            train = [mols[i] for i in tr]
            valid = [mols[i] for i in va]
            yt = y[tr]
            yv = y[va]
        else:
            train = list(mols)
            valid = list(validation_mols)
            yt = y
            yv = np.asarray(validation_targets, dtype=np.float32)
            identities = lambda ms: {Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True) for m in ms}
            if identities(train) & identities(valid):
                raise ValueError("Train/validation molecular overlap")
        if not valid or not np.isfinite(yt).all() or not np.isfinite(yv).all():
            raise ValueError("Invalid training/validation labels")
        self.scalers = fit_target_scalers(yt, mode=scaler_mode)
        norm_train = transform_with_scalers(yt, self.scalers)
        norm_valid = transform_with_scalers(yv, self.scalers)
        all_mols = train + valid
        for mol in all_mols:
            self.validate_molecule(mol)
        cached = [None] * len(all_mols)

        def batches(indices, rng=None):
            groups = {}
            for i in indices:
                groups.setdefault(all_mols[i].GetNumAtoms(), []).append(int(i))
            chunks = [ids[b:b + batch_size] for ids in groups.values() for b in range(0, len(ids), batch_size)]
            if rng is not None:
                rng.shuffle(chunks)
            return chunks

        completed = 0
        for ids in batches(range(len(all_mols))):
            symbols = [[a.GetSymbol() for a in all_mols[i].GetAtoms()] for i in ids]
            positions = torch.tensor(np.stack([all_mols[i].GetConformer().GetPositions() for i in ids]),
                                     dtype=torch.float32, device=self.device)
            if batch_size == 1:
                desc, _, _ = self.dpa_guidance.extract_native_descriptor_and_forces(positions[0], symbols[0])
                descriptors = desc[None]
            else:
                descriptors, _, _ = self.dpa_guidance.extract_native_batch(positions, symbols)
            for i, desc, pos in zip(ids, descriptors, positions):
                cached[i] = (desc.detach(), pos.detach())
            completed += len(ids)
            if completed % 100 < batch_size:
                print(f"DPA descriptors {completed}/{len(all_mols)}", flush=True)
        training_history = []
        for index in range(self.n_ensemble_heads):
            torch.manual_seed(seed + index)
            member = _DPA4EquivariantPropertyAdapter(self.desc_dim, 128, self.output_dim, 1).to(self.device)
            self.adapter.members[index] = member
            rng = np.random.default_rng(seed + index)
            bootstrap = rng.integers(0, len(train), size=len(train))
            opt = torch.optim.AdamW(member.parameters(), lr=lr, weight_decay=1e-4)
            best = float("inf")
            best_state = None
            history = []
            for epoch in range(epochs):
                member.train()
                order = rng.permutation(bootstrap)
                for ids in batches(order, rng):
                    desc = torch.stack([cached[i][0] for i in ids])
                    pos = torch.stack([cached[i][1] for i in ids])
                    target = torch.tensor(norm_train[ids], device=self.device)
                    pred, _ = member(desc, pos)
                    loss = F.smooth_l1_loss(pred, target, beta=0.1)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(), 5.0)
                    opt.step()
                member.eval()
                with torch.no_grad():
                    errors = []
                    for ids in batches(range(len(train), len(all_mols))):
                        desc = torch.stack([cached[i][0] for i in ids])
                        pos = torch.stack([cached[i][1] for i in ids])
                        target = torch.tensor(norm_valid[np.array(ids) - len(train)], device=self.device)
                        errors.extend((member(desc, pos)[0] - target).abs().mean(-1).tolist())
                score = float(np.mean(errors))
                history.append(score)
                if score < best:
                    best = score
                    best_state = {k: v.detach().cpu().clone() for k, v in member.state_dict().items()}
                print(
                    f"Bootstrap {index + 1}/{self.n_ensemble_heads} epoch {epoch + 1}/{epochs} validation normalized MAE {score:.4f}",
                    flush=True,
                )
            member.load_state_dict(best_state)
            member.eval()
            member.requires_grad_(False)
            training_history.append(
                {
                    "seed": seed + index,
                    "bootstrap_indices_sha256": hashlib.sha256(bootstrap.tobytes()).hexdigest(),
                    "validation_mae_normalized": history,
                    "best": best,
                }
            )
        self.adapter.eval()
        joblib.dump(self.scalers, os.path.join(save_dir, "target_scaler.ss"))
        checkpoint = os.path.join(save_dir, "dpa4_prop_adapter.pt")
        torch.save({"adapter_state_dict": self.adapter.state_dict()}, checkpoint)
        weight = getattr(self.dpa_guidance, "model_path", None)
        meta = {
            "schema_version": 3,
            "oracle_type": 'internal_' + getattr(self.dpa_guidance, 'backend_name', 'test_descriptor'),
            "descriptor_provider": type(self.dpa_guidance).__name__,
            "ensemble_kind": "bootstrap_adapters_v2",
            "desc_dim": self.desc_dim,
            "output_dim": self.output_dim,
            "n_ensemble_heads": self.n_ensemble_heads,
            "target_cols": self.target_cols,
            "target_units": getattr(self, 'target_units', ['unspecified'] * self.output_dim),
            "electronic_states": [{'charge': 0, 'multiplicity': 1}],
            "training_elements": sorted({atom.GetSymbol() for mol in train for atom in mol.GetAtoms()}),
            "input_geometry_method": getattr(self, 'geometry_method', None),
            "scaler_sha256": hashlib.sha256(open(os.path.join(save_dir, 'target_scaler.ss'), 'rb').read()).hexdigest(),
            "batch_size": batch_size,
            "train_records": len(train),
            "validation_records": len(valid),
            "history": training_history,
            "checkpoint_sha256": hashlib.sha256(open(checkpoint, "rb").read()).hexdigest(),
            "forcefield_sha256": hashlib.sha256(open(weight, "rb").read()).hexdigest() if weight else None,
            "uncertainty": "Independent bootstrap adapter disagreement; calibration not implied",
        }
        with open(os.path.join(save_dir, "oracle_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    @classmethod
    def load_from_dir(cls, load_dir, dpa_guidance):
        with open(os.path.join(load_dir, "oracle_meta.json")) as f:
            meta = json.load(f)
        if meta.get("ensemble_kind") != "bootstrap_adapters_v2":
            raise ValueError("Legacy shared-head oracle requires retraining")
        checkpoint = os.path.join(load_dir, "dpa4_prop_adapter.pt")
        scaler_path = os.path.join(load_dir, 'target_scaler.ss')
        if meta.get('schema_version') == 3 and hashlib.sha256(open(scaler_path, 'rb').read()).hexdigest() != meta['scaler_sha256']:
            raise ValueError('Oracle target scaler checksum mismatch')
        if hashlib.sha256(open(checkpoint, "rb").read()).hexdigest() != meta["checkpoint_sha256"]:
            raise ValueError("Oracle checkpoint checksum mismatch")
        weight = getattr(dpa_guidance, "model_path", None)
        if meta["forcefield_sha256"] and (
            not weight or hashlib.sha256(open(weight, "rb").read()).hexdigest() != meta["forcefield_sha256"]
        ):
            raise ValueError("Forcefield mismatch for property adapter")
        obj = cls(dpa_guidance, int(meta["output_dim"]), meta["target_cols"], int(meta["n_ensemble_heads"]))
        obj.scalers = joblib.load(scaler_path)
        obj.target_units = meta.get('target_units', ['unspecified'] * obj.output_dim)
        obj.geometry_method = meta.get('input_geometry_method')
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["adapter_state_dict"]
        obj.adapter.load_state_dict(state, strict=True)
        obj.adapter.eval()
        obj.adapter.requires_grad_(False)
        return obj
