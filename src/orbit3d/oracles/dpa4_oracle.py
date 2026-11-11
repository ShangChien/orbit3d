"""
DPA-4 Native Shared-Backbone Force Field & Equivariant Property Predictor (`dpa4_oracle.py`)
============================================================================================
- Freezes `DPA-4`'s `DescrptSeZM` and `SeZMEnergyFittingNet` (`requires_grad=False`) so
  physical atomic forces `F_phys = -grad_X E_DPA4(X)` are 100% preserved.
- Captures 100% of `DPA-4`'s native internal multi-layer equivariant per-atom descriptor
  tensor `descriptor` (`(N, D_desc)`) passed into `SeZMEnergyFittingNet` via forward hook.
- Attaches a trainable 3D Distance-Biased Attention Adapter + Multi-Head Gated Pooling Ensemble
  on `descriptor` so a single forward pass produces `y_pred`, `y_std`, `E_phys`, `F_phys`, and
  exact differentiable 3D coordinate gradients `d y / d X`!
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
        h = self.norm1(self.in_proj(atom_repr))
        N, H = h.shape
        dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0), p=2.0).squeeze(0)
        rbf = torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.rbf_centers) / 0.4) ** 2)
        attn_bias = self.dist_rbf(rbf).permute(2, 0, 1)

        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(N, 4, H // 4).transpose(0, 1)
        k = k.view(N, 4, H // 4).transpose(0, 1)
        v = v.view(N, 4, H // 4).transpose(0, 1)

        scores = torch.bmm(q, k.transpose(1, 2)) / ((H // 4) ** 0.5) + attn_bias
        probs = F.softmax(scores, dim=-1)
        ctx = torch.bmm(probs, v).transpose(0, 1).contiguous().view(N, H)
        h = h + self.out_proj(ctx)
        h = h + self.ffn(self.norm2(h))

        mean_pool = h.mean(dim=0)
        head_preds = []
        for gate, mlp in zip(self.gates, self.heads):
            w = F.softmax(gate(h), dim=0)
            attn_pool = (h * w).sum(dim=0)
            mol_vec = torch.cat([mean_pool, attn_pool], dim=-1)
            head_preds.append(mlp(mol_vec))

        stacked = torch.stack(head_preds, dim=0)
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
                        inverse_single_scaler_torch(row[k], self.scalers[k] if self.scalers else None)
                        for k in range(self.output_dim)
                    ]
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
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, device=self.device)
            with torch.enable_grad():
                yp, ys, _, _ = self.forward_shared_force_and_property(pos, atoms)
            preds.append(yp.detach().cpu().numpy())
            stds.append(ys.detach().cpu().numpy())
        return np.array(preds, dtype=np.float32), np.array(stds, dtype=np.float32)

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
        cached = []
        for index, mol in enumerate(train + valid):
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, device=self.device)
            desc, energy, force = self.dpa_guidance.extract_native_descriptor_and_forces(pos, atoms)
            if not all(torch.isfinite(v).all() for v in (desc, energy, force)):
                raise FloatingPointError("Nonfinite descriptor cache")
            cached.append((desc.detach(), pos.detach()))
            if index % 100 == 0:
                print(f"DPA descriptors {index + 1}/{len(train) + len(valid)}", flush=True)
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
                for i in order:
                    desc, pos = cached[i]
                    target = torch.tensor(norm_train[i], device=self.device)
                    pred, _ = member(desc, pos)
                    loss = F.smooth_l1_loss(pred, target, beta=0.1)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(), 5.0)
                    opt.step()
                member.eval()
                with torch.no_grad():
                    errors = [
                        float(
                            torch.abs(
                                member(*cached[len(train) + i])[0]
                                - torch.tensor(norm_valid[i], device=self.device)
                            ).mean()
                        )
                        for i in range(len(valid))
                    ]
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
            "oracle_type": "internal_dpa4",
            "ensemble_kind": "bootstrap_adapters_v2",
            "desc_dim": self.desc_dim,
            "output_dim": self.output_dim,
            "n_ensemble_heads": self.n_ensemble_heads,
            "target_cols": self.target_cols,
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
        if hashlib.sha256(open(checkpoint, "rb").read()).hexdigest() != meta["checkpoint_sha256"]:
            raise ValueError("Oracle checkpoint checksum mismatch")
        weight = getattr(dpa_guidance, "model_path", None)
        if meta["forcefield_sha256"] and (
            not weight or hashlib.sha256(open(weight, "rb").read()).hexdigest() != meta["forcefield_sha256"]
        ):
            raise ValueError("Forcefield mismatch for property adapter")
        obj = cls(dpa_guidance, int(meta["output_dim"]), meta["target_cols"], int(meta["n_ensemble_heads"]))
        obj.scalers = joblib.load(os.path.join(load_dir, "target_scaler.ss"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["adapter_state_dict"]
        obj.adapter.load_state_dict(state, strict=True)
        obj.adapter.eval()
        obj.adapter.requires_grad_(False)
        return obj
