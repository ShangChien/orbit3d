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
        self.adapter = _DPA4EquivariantPropertyAdapter(
            desc_dim=self.desc_dim,
            hidden_dim=256,
            output_dim=output_dim,
            n_ensemble_heads=n_ensemble_heads,
        ).to(self.device)

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
        norm_mean, norm_std = self.adapter(native_desc, coords.float())

        real_cols = []
        std_cols = []
        for k in range(self.output_dim):
            sc = self.scalers[k] if k < len(self.scalers) else None
            real_cols.append(inverse_single_scaler_torch(norm_mean[k], sc))
            scale_val = float(np.squeeze(getattr(sc, "scale_", getattr(getattr(sc, "_scaler", None), "scale_", 1.0))))
            std_cols.append(norm_std[k] * scale_val)

        y_pred_real = torch.stack(real_cols, dim=0)
        y_std_real = torch.stack(std_cols, dim=0)
        if self.output_dim == 1:
            return y_pred_real.squeeze(0), y_std_real.squeeze(0), energy, forces
        return y_pred_real, y_std_real, energy, forces

    def forward_differentiable_3d(
        self,
        coords: torch.Tensor,
        atom_symbols: List[str],
        smiles: str = "C",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        yp, ys, _, _ = self.forward_shared_force_and_property(coords, atom_symbols)
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
        mols: List[Chem.Mol],
        targets_2d: np.ndarray,
        save_dir: str,
        epochs: int = 35,
        lr: float = 2e-4,
        scaler_mode: str = "standard",
        seed: int = 42,
    ) -> Dict[str, float]:
        os.makedirs(save_dir, exist_ok=True)
        self.scalers = fit_target_scalers(targets_2d, mode=scaler_mode)
        y_norm = transform_with_scalers(targets_2d, self.scalers)
        joblib.dump(self.scalers, os.path.join(save_dir, "target_scaler.ss"))

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(mols))
        n_val = max(1, int(0.15 * len(mols)))
        va_idx, tr_idx = perm[:n_val], perm[n_val:]

        # Pre-extract frozen native DPA-4 descriptors once for fast, high-throughput training
        cached_features = []
        for mol in mols:
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, device=self.device)
            native_desc, _, _ = self.dpa_guidance.extract_native_descriptor_and_forces(pos, atoms)
            cached_features.append((native_desc.detach(), pos.detach(), atoms))

        opt = torch.optim.AdamW(self.adapter.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        best_val = float("inf")
        best_sd = None

        for _ in range(epochs):
            self.adapter.train()
            rng.shuffle(tr_idx)
            for i in tr_idx:
                native_desc_i, pos_i, _ = cached_features[i]
                y_t = torch.tensor(y_norm[i], dtype=torch.float32, device=self.device).view(self.output_dim)
                norm_mean, norm_std = self.adapter(native_desc_i, pos_i)
                loss = F.smooth_l1_loss(norm_mean, y_t, beta=0.1) + 0.05 * norm_std.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 5.0)
                opt.step()
            sched.step()

            self.adapter.eval()
            val_errs = []
            for i in va_idx:
                _, pos_i, atoms_i = cached_features[i]
                yp_real, _, _, _ = self.forward_shared_force_and_property(pos_i, atoms_i)
                y_true_real = torch.tensor(targets_2d[i], dtype=torch.float32, device=self.device).view(self.output_dim)
                val_errs.append(float(F.l1_loss(yp_real.view(self.output_dim), y_true_real).item()))
            mean_val = float(np.mean(val_errs)) if val_errs else 0.0
            if mean_val <= best_val:
                best_val = mean_val
                best_sd = {k: v.detach().cpu().clone() for k, v in self.adapter.state_dict().items()}

        if best_sd is not None:
            self.adapter.load_state_dict(best_sd)
        self.adapter.eval()
        for p in self.adapter.parameters():
            p.requires_grad_(False)
        torch.save({"adapter_state_dict": self.adapter.state_dict()}, os.path.join(save_dir, "dpa4_prop_adapter.pt"))
        meta = {
            "oracle_type": "internal_dpa4",
            "desc_dim": self.desc_dim,
            "output_dim": self.output_dim,
            "n_ensemble_heads": self.n_ensemble_heads,
            "target_cols": self.target_cols,
            "val_mae_physical": best_val,
        }
        with open(os.path.join(save_dir, "oracle_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    @classmethod
    def load_from_dir(
        cls,
        load_dir: str,
        dpa_guidance: DPAPotentialGuidance,
    ) -> "InternalDPA4PropertyOracle":
        with open(os.path.join(load_dir, "oracle_meta.json"), "r") as f:
            meta = json.load(f)
        obj = cls(
            dpa_guidance=dpa_guidance,
            output_dim=int(meta["output_dim"]),
            target_cols=meta.get("target_cols"),
            n_ensemble_heads=int(meta.get("n_ensemble_heads", 5)),
        )
        obj.scalers = joblib.load(os.path.join(load_dir, "target_scaler.ss"))
        sd = torch.load(os.path.join(load_dir, "dpa4_prop_adapter.pt"), map_location="cpu", weights_only=False)[
            "adapter_state_dict"
        ]
        obj.adapter.load_state_dict(sd, strict=True)
        obj.adapter.eval()
        for p in obj.adapter.parameters():
            p.requires_grad_(False)
        return obj
