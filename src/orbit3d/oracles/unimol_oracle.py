"""
Standalone 3D Uni-Mol Property Predictor & K-Fold Trainer (`unimol_oracle.py`).
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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from .base import PropertyOracleBase
from .uniqsar_oracle import (
    UNIMOL_BOS_IDX,
    UNIMOL_DICT_SYMBOLS,
    UNIMOL_EOS_IDX,
    UNIMOL_PAD_IDX,
    UNIMOL_SYM2IDX,
    UNIMOL_UNK_IDX,
    StandaloneUniMolModel,
)
from ..utils.dataset_io import (
    fit_target_scalers,
    inverse_single_scaler_torch,
    inverse_transform_with_scalers,
    transform_with_scalers,
)


class _UniMol3DDataset(Dataset):
    def __init__(self, mols: List[Chem.Mol], y_norm: np.ndarray, remove_hs: bool = False, max_atoms: int = 256):
        self.items = []
        dict_len = len(UNIMOL_DICT_SYMBOLS)
        for mol, y in zip(mols, y_norm):
            atoms = np.array([a.GetSymbol() for a in mol.GetAtoms()])
            coords = mol.GetConformer().GetPositions().astype(np.float32)
            if remove_hs:
                keep = atoms != "H"
                if keep.any():
                    atoms = atoms[keep]
                    coords = coords[keep]
            if len(atoms) > max_atoms:
                atoms = atoms[:max_atoms]
                coords = coords[:max_atoms]

            toks = np.array(
                [UNIMOL_BOS_IDX]
                + [UNIMOL_SYM2IDX.get(str(a), UNIMOL_UNK_IDX) for a in atoms]
                + [UNIMOL_EOS_IDX],
                dtype=np.int64,
            )
            coords_c = coords - coords.mean(axis=0, keepdims=True)
            coords_pad = np.pad(coords_c, ((1, 1), (0, 0)), mode="constant", constant_values=0.0)
            edges = toks.reshape(-1, 1) * dict_len + toks.reshape(1, -1)
            self.items.append((toks, coords_pad, edges, y.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def _collate_unimol_3d(batch):
    max_l = max(len(x[0]) for x in batch)
    bsz = len(batch)
    out_dim = len(batch[0][3])
    tok_b = torch.full((bsz, max_l), UNIMOL_PAD_IDX, dtype=torch.long)
    coord_b = torch.zeros((bsz, max_l, 3), dtype=torch.float32)
    edge_b = torch.zeros((bsz, max_l, max_l), dtype=torch.long)
    y_b = torch.zeros((bsz, out_dim), dtype=torch.float32)

    for i, (t, c, e, y) in enumerate(batch):
        l = len(t)
        tok_b[i, :l] = torch.from_numpy(t)
        coord_b[i, :l, :] = torch.from_numpy(c)
        edge_b[i, :l, :l] = torch.from_numpy(e)
        y_b[i] = torch.from_numpy(y)

    dist_b = torch.cdist(coord_b, coord_b, p=2.0)
    return tok_b, dist_b, edge_b, y_b


def load_pretrained_unimol_backbone(model: StandaloneUniMolModel, weight_path: Optional[str]) -> int:
    """
    Loads pretrained `Uni-Mol` backbone weights (supports both `fp32` and `fp16`-compressed
    `mol_pre_all_h_fp16.pt` / `mol_pre_no_h_fp16.pt` checkpoints).
    """
    if not weight_path or not os.path.exists(weight_path):
        return 0
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        clean_k = k.replace("unimol.", "")
        if clean_k in model_sd and model_sd[clean_k].shape == v.shape:
            model_sd[clean_k] = v.to(model_sd[clean_k].dtype)
            loaded += 1
    model.load_state_dict(model_sd, strict=False)
    return loaded


class InternalUniMolOracle(PropertyOracleBase):
    """
    Standalone `Uni-Mol` 3D Property Predictor & Trainer (`oracle_mode = "internal_unimol"`).
    Provides:
      - `fit_from_dataset(mols, targets_2d, save_dir, ...)` -> trains `n_folds` 3D `Uni-Mol` models
      - `predict_3d_mols(mols)` -> `(y_pred, y_std)` of shape `(N, K)`
      - `forward_differentiable_3d(coords, atom_symbols)` -> `(y_pred, y_std)` with `coords.grad`
    """

    def __init__(
        self,
        output_dim: int = 1,
        n_folds: int = 5,
        remove_hs: bool = False,
        target_cols: Optional[List[str]] = None,
        pretrain_weight_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_folds = n_folds
        self.remove_hs = remove_hs
        self.target_cols = target_cols or [f"TARGET_{i}" for i in range(output_dim)]
        self.pretrain_weight_path = pretrain_weight_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.has_3d_unimol = True
        self.folds = nn.ModuleList()
        self.scalers: List = []

    def fit_from_dataset(
        self,
        mols: List[Chem.Mol],
        targets_2d: np.ndarray,
        save_dir: str,
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 5e-5,
        scaler_mode: str = "standard",
        seed: int = 42,
    ) -> Dict[str, float]:
        os.makedirs(save_dir, exist_ok=True)
        self.scalers = fit_target_scalers(targets_2d, mode=scaler_mode)
        y_norm = transform_with_scalers(targets_2d, self.scalers)
        joblib.dump(self.scalers, os.path.join(save_dir, "target_scaler.ss"))

        dataset = _UniMol3DDataset(mols, y_norm, remove_hs=self.remove_hs)
        kf = KFold(n_splits=min(self.n_folds, len(dataset)), shuffle=True, random_state=seed)
        self.folds = nn.ModuleList()
        fold_val_losses = []

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(range(len(dataset)))):
            tr_sub = torch.utils.data.Subset(dataset, tr_idx)
            va_sub = torch.utils.data.Subset(dataset, va_idx)
            tr_loader = DataLoader(tr_sub, batch_size=batch_size, shuffle=True, collate_fn=_collate_unimol_3d)
            va_loader = DataLoader(va_sub, batch_size=batch_size, shuffle=False, collate_fn=_collate_unimol_3d)

            model = StandaloneUniMolModel(output_dim=self.output_dim).to(self.device)
            load_pretrained_unimol_backbone(model, self.pretrain_weight_path)

            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
            best_val = float("inf")
            best_sd = None

            for _ in range(epochs):
                model.train()
                for tok_b, dist_b, edge_b, y_b in tr_loader:
                    tok_b = tok_b.to(self.device)
                    dist_b = dist_b.to(self.device)
                    edge_b = edge_b.to(self.device)
                    y_b = y_b.to(self.device)
                    pred = model(tok_b, dist_b, edge_b)
                    loss = F.smooth_l1_loss(pred, y_b, beta=0.1)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                sched.step()

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for tok_b, dist_b, edge_b, y_b in va_loader:
                        pred = model(
                            tok_b.to(self.device),
                            dist_b.to(self.device),
                            edge_b.to(self.device),
                        )
                        val_losses.append(float(F.l1_loss(pred, y_b.to(self.device)).item()))
                mean_val = float(np.mean(val_losses)) if val_losses else 0.0
                if mean_val <= best_val:
                    best_val = mean_val
                    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_sd is not None:
                model.load_state_dict(best_sd)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.folds.append(model)
            fold_val_losses.append(best_val)
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(save_dir, f"unimol_fold_{fold_idx}.pth"),
            )

        meta = {
            "oracle_type": "internal_unimol",
            "output_dim": self.output_dim,
            "n_folds": len(self.folds),
            "remove_hs": self.remove_hs,
            "target_cols": self.target_cols,
            "cv_l1_norm": float(np.mean(fold_val_losses)),
        }
        with open(os.path.join(save_dir, "oracle_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    @classmethod
    def load_from_dir(cls, load_dir: str, device: str = "cuda:0") -> "InternalUniMolOracle":
        with open(os.path.join(load_dir, "oracle_meta.json"), "r") as f:
            meta = json.load(f)
        obj = cls(
            output_dim=int(meta["output_dim"]),
            n_folds=int(meta["n_folds"]),
            remove_hs=bool(meta.get("remove_hs", False)),
            target_cols=meta.get("target_cols"),
            device=device,
        )
        obj.scalers = joblib.load(os.path.join(load_dir, "target_scaler.ss"))
        for i in range(obj.n_folds):
            p = os.path.join(load_dir, f"unimol_fold_{i}.pth")
            sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
            m = StandaloneUniMolModel(output_dim=obj.output_dim)
            m.load_state_dict(sd, strict=True)
            m.eval()
            for param in m.parameters():
                param.requires_grad_(False)
            obj.folds.append(m.to(obj.device))
        return obj

    def forward_differentiable_3d(
        self,
        coords: torch.Tensor,
        atom_symbols: List[str],
        smiles: str = "C",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.remove_hs:
            keep = [i for i, s in enumerate(atom_symbols) if s != "H"]
            if len(keep) == 0:
                keep = list(range(len(atom_symbols)))
            sub_syms = [atom_symbols[i] for i in keep]
            sub_coords = coords[keep]
        else:
            sub_syms = list(atom_symbols)
            sub_coords = coords

        dict_len = len(UNIMOL_DICT_SYMBOLS)
        toks = np.array(
            [UNIMOL_BOS_IDX]
            + [UNIMOL_SYM2IDX.get(str(a), UNIMOL_UNK_IDX) for a in sub_syms]
            + [UNIMOL_EOS_IDX],
            dtype=np.int64,
        )
        edges = toks.reshape(-1, 1) * dict_len + toks.reshape(1, -1)
        tok_t = torch.from_numpy(toks).unsqueeze(0).to(coords.device)
        edge_t = torch.from_numpy(edges).unsqueeze(0).to(coords.device)

        c_centered = sub_coords - sub_coords.mean(dim=0, keepdim=True)
        zeros_1 = torch.zeros((1, 3), dtype=coords.dtype, device=coords.device)
        c_padded = torch.cat([zeros_1, c_centered, zeros_1], dim=0).unsqueeze(0)
        dist_t = torch.cdist(c_padded, c_padded, p=2.0)

        fold_preds = []
        for m in self.folds:
            out = m(tok_t, dist_t, edge_t).squeeze(0)
            fold_preds.append(out.view(self.output_dim))

        stacked = torch.stack(fold_preds, dim=0)
        norm_mean = stacked.mean(dim=0)
        norm_std = stacked.std(dim=0, unbiased=False)

        real_cols = []
        std_cols = []
        for k in range(self.output_dim):
            sc = self.scalers[k] if k < len(self.scalers) else None
            real_cols.append(inverse_single_scaler_torch(norm_mean[k], sc))
            scale_val = float(np.squeeze(getattr(sc, "scale_", getattr(getattr(sc, "_scaler", None), "scale_", 1.0))))
            std_cols.append(norm_std[k] * scale_val)

        y_pred = torch.stack(real_cols, dim=0)
        y_std = torch.stack(std_cols, dim=0)
        if self.output_dim == 1:
            return y_pred.squeeze(0), y_std.squeeze(0)
        return y_pred, y_std

    @torch.no_grad()
    def predict_3d_mols(self, mols: List[Chem.Mol]) -> Tuple[np.ndarray, np.ndarray]:
        dummy_y = np.zeros((len(mols), self.output_dim), dtype=np.float32)
        ds = _UniMol3DDataset(mols, dummy_y, remove_hs=self.remove_hs)
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=_collate_unimol_3d)
        fold_outputs = []
        for m in self.folds:
            preds = []
            for tok_b, dist_b, edge_b, _ in loader:
                p = m(tok_b.to(self.device), dist_b.to(self.device), edge_b.to(self.device))
                preds.append(p.cpu().numpy())
            fold_outputs.append(np.concatenate(preds, axis=0))
        stacked = np.stack(fold_outputs, axis=0)
        norm_mean = np.mean(stacked, axis=0)
        norm_std = np.std(stacked, axis=0)
        y_pred = inverse_transform_with_scalers(norm_mean, self.scalers)
        y_std = np.zeros_like(y_pred)
        for k, sc in enumerate(self.scalers):
            scale_val = float(np.squeeze(getattr(sc, "scale_", getattr(getattr(sc, "_scaler", None), "scale_", 1.0))))
            y_std[:, k] = norm_std[:, k] * scale_val
        return y_pred, y_std
