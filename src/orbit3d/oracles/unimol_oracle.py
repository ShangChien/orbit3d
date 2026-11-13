"""
Standalone 3D Uni-Mol Property Predictor & K-Fold Trainer (`unimol_oracle.py`).
"""

from __future__ import annotations

import json
import hashlib
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
    StandaloneUniMolModel,
)
from ..utils.dataset_io import (
    fit_target_scalers,
    inverse_single_scaler_torch,
    transform_with_scalers,
)


class _UniMol3DDataset(Dataset):
    def __init__(self, mols: List[Chem.Mol], y_norm: np.ndarray, remove_hs: bool = False, max_atoms: int = 256):
        self.items = []
        if len(mols) != len(y_norm):
            raise ValueError('Molecule and target row counts differ')
        dict_len = len(UNIMOL_DICT_SYMBOLS)
        for index, (mol, y) in enumerate(zip(mols, y_norm)):
            if mol is None or not mol.GetNumConformers() or not mol.GetNumAtoms():
                raise ValueError(f'Uni-Mol record {index} requires a nonempty 3D molecule')
            atoms = np.array([a.GetSymbol() for a in mol.GetAtoms()])
            coords = mol.GetConformer().GetPositions().astype(np.float32)
            if remove_hs:
                keep = atoms != "H"
                atoms = atoms[keep]
                coords = coords[keep]
            if not 0 < len(atoms) <= max_atoms:
                raise ValueError(f'Uni-Mol record {index} has {len(atoms)} encoded atoms; supported range is 1..{max_atoms}')
            unknown = sorted(set(atoms) - set(UNIMOL_SYM2IDX))
            if unknown:
                raise ValueError(f'Uni-Mol vocabulary does not cover elements {unknown}')
            if not np.isfinite(coords).all() or not np.isfinite(y).all():
                raise ValueError(f'Uni-Mol record {index} has nonfinite coordinates or targets')

            toks = np.array(
                [UNIMOL_BOS_IDX]
                + [UNIMOL_SYM2IDX[str(a)] for a in atoms]
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
        raise FileNotFoundError(f"Uni-Mol pretrained weights missing: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    expected = model.state_dict()
    required = {key for key in expected if not key.startswith('classification_head.')}
    source = {key.removeprefix('unimol.'): value for key, value in sd.items()}
    missing = sorted(required - set(source))
    mismatched = sorted(key for key in required & set(source) if source[key].shape != expected[key].shape)
    unexpected_backbone = sorted(key for key in source if key not in expected
        and key.startswith(('embed_tokens.', 'encoder.', 'gbf.', 'gbf_proj.'))
        and not key.startswith('encoder.final_head_layer_norm.'))
    if missing or mismatched or unexpected_backbone:
        raise ValueError(f'Incomplete or incompatible Uni-Mol backbone: missing={missing}, shape={mismatched}, extra={unexpected_backbone}')
    # The task head is initialized for the new labels, never borrowed by shape alone.
    expected.update({key: source[key].to(expected[key].dtype) for key in required})
    model.load_state_dict(expected, strict=True)
    return len(required)



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
        max_atoms: int = 256,
        target_units: Optional[List[str]] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_folds = n_folds
        self.remove_hs = remove_hs
        self.target_cols = target_cols or [f"TARGET_{i}" for i in range(output_dim)]
        self.pretrain_weight_path = pretrain_weight_path
        self.device = torch.device(device)
        self.max_atoms = int(max_atoms)
        self.target_units = target_units or ['unspecified'] * output_dim
        self.fold_scalers: List = []
        self.scaler_scope = 'fold_training_only'
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
        if not self.pretrain_weight_path or not os.path.isfile(self.pretrain_weight_path):
            raise FileNotFoundError(f'Uni-Mol pretrained weights missing: {self.pretrain_weight_path}')
        with open(self.pretrain_weight_path, 'rb') as stream:
            pretrained_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        targets = np.asarray(targets_2d, dtype=np.float32)
        if epochs <= 0 or self.n_folds < 2 or len(mols) < 2:
            raise ValueError('Uni-Mol cross-validation requires positive epochs and at least two folds/structures')
        if targets.shape != (len(mols), self.output_dim) or not np.isfinite(targets).all():
            raise ValueError('Invalid Uni-Mol target rows or dimensions')
        # Validate complete structures before fitting or writing any model artifact.
        _UniMol3DDataset(mols, targets, remove_hs=self.remove_hs, max_atoms=self.max_atoms)
        os.makedirs(save_dir, exist_ok=True)
        kf = KFold(n_splits=min(self.n_folds, len(mols)), shuffle=True, random_state=seed)
        self.folds, self.fold_scalers = nn.ModuleList(), []
        self.scaler_scope = 'fold_training_only'
        fold_val_losses, splits, fold_hashes = [], [], {}
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(range(len(mols)))):
            torch.manual_seed(seed + fold_idx)
            scalers = fit_target_scalers(targets[tr_idx], mode=scaler_mode)
            self.fold_scalers.append(scalers)
            normalized = transform_with_scalers(targets, scalers)
            tr_sub = _UniMol3DDataset([mols[i] for i in tr_idx], normalized[tr_idx], self.remove_hs, self.max_atoms)
            va_sub = _UniMol3DDataset([mols[i] for i in va_idx], normalized[va_idx], self.remove_hs, self.max_atoms)
            tr_loader = DataLoader(tr_sub, batch_size=batch_size, shuffle=True, collate_fn=_collate_unimol_3d)
            va_loader = DataLoader(va_sub, batch_size=batch_size, shuffle=False, collate_fn=_collate_unimol_3d)
            splits.append({'train_indices': tr_idx.tolist(), 'validation_indices': va_idx.tolist()})

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
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite Uni-Mol training loss')
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    opt.step()
                sched.step()

                model.eval()
                val_error, val_count = 0., 0
                with torch.no_grad():
                    for tok_b, dist_b, edge_b, y_b in va_loader:
                        pred = model(
                            tok_b.to(self.device),
                            dist_b.to(self.device),
                            edge_b.to(self.device),
                        )
                        if not torch.isfinite(pred).all():
                            raise FloatingPointError('Nonfinite Uni-Mol validation prediction')
                        val_error += float(F.l1_loss(pred, y_b.to(self.device), reduction='sum'))
                        val_count += y_b.numel()
                mean_val = val_error / val_count
                if mean_val <= best_val:
                    best_val = mean_val
                    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_sd, strict=True)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.folds.append(model)
            fold_val_losses.append(best_val)
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(save_dir, f"unimol_fold_{fold_idx}.pth"),
            )

            name = f'unimol_fold_{fold_idx}.pth'
            with open(os.path.join(save_dir, name), 'rb') as stream:
                fold_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()

        scaler_path = os.path.join(save_dir, 'target_scaler.ss')
        joblib.dump({'fold_scalers': self.fold_scalers}, scaler_path)
        with open(scaler_path, 'rb') as stream:
            scaler_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        meta = {
            "schema_version": 2,
            "oracle_type": "internal_unimol",
            "scaler_scope": self.scaler_scope,
            "fold_splits": splits,
            "seed": seed,
            "pretrained_sha256": pretrained_hash,
            "max_atoms": self.max_atoms,
            "target_units": self.target_units,
            "checkpoint_sha256": fold_hashes,
            "scaler_sha256": scaler_hash,
            "uncertainty": "Physical-unit disagreement across fitted folds; calibration not implied",
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
            max_atoms=int(meta.get('max_atoms', 256)),
            target_units=meta.get('target_units'),
        )
        scaler_path = os.path.join(load_dir, 'target_scaler.ss')
        if meta.get('schema_version', 1) == 2:
            with open(scaler_path, 'rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != meta['scaler_sha256']:
                    raise ValueError('Uni-Mol target scaler checksum mismatch')
            obj.fold_scalers = joblib.load(scaler_path)['fold_scalers']
            if len(obj.fold_scalers) != obj.n_folds:
                raise ValueError('Uni-Mol fold/scaler count mismatch')
        elif meta.get('schema_version', 1) == 1:
            obj.scalers = joblib.load(scaler_path)
            obj.fold_scalers = [obj.scalers] * obj.n_folds
            obj.scaler_scope = 'legacy_shared_scaler'
        else:
            raise ValueError('Unknown Uni-Mol checkpoint schema')
        for i in range(obj.n_folds):
            p = os.path.join(load_dir, f"unimol_fold_{i}.pth")
            if meta.get('schema_version', 1) == 2:
                with open(p, 'rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != meta['checkpoint_sha256'][f'unimol_fold_{i}.pth']:
                        raise ValueError('Uni-Mol fold checkpoint checksum mismatch')
            sd = torch.load(p, map_location="cpu", weights_only=True)["model_state_dict"]
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
            sub_syms = [atom_symbols[i] for i in keep]
            sub_coords = coords[keep]
        else:
            sub_syms = list(atom_symbols)
            sub_coords = coords

        if not 0 < len(sub_syms) <= self.max_atoms:
            raise ValueError('Uni-Mol atom count is outside the configured representation')
        unknown = sorted(set(sub_syms) - set(UNIMOL_SYM2IDX))
        if unknown:
            raise ValueError(f'Uni-Mol vocabulary does not cover elements {unknown}')
        if sub_coords.shape != (len(sub_syms), 3) or not torch.isfinite(sub_coords).all():
            raise ValueError('Invalid Uni-Mol coordinates')
        dict_len = len(UNIMOL_DICT_SYMBOLS)
        toks = np.array(
            [UNIMOL_BOS_IDX]
            + [UNIMOL_SYM2IDX[str(a)] for a in sub_syms]
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

        y_pred, y_std = self._physical_statistics(torch.stack(fold_preds, dim=0))
        if self.output_dim == 1:
            return y_pred.squeeze(0), y_std.squeeze(0)
        return y_pred, y_std

    def _physical_statistics(self, normalized):
        if len(normalized) != len(self.fold_scalers) or not len(normalized):
            raise ValueError('Uni-Mol inference requires matching trained folds and scalers')
        physical = torch.stack([torch.stack([inverse_single_scaler_torch(values[..., k], scalers[k])
                                            for k in range(self.output_dim)], dim=-1)
                                for values, scalers in zip(normalized, self.fold_scalers)])
        center = physical.mean(0)
        # Preserve the old checkpoint's point-prediction definition explicitly.
        mean = (torch.stack([inverse_single_scaler_torch(normalized.mean(0)[..., k], self.scalers[k])
                             for k in range(self.output_dim)], dim=-1)
                if self.scaler_scope == 'legacy_shared_scaler' else center)
        spread = torch.linalg.vector_norm(physical-center, dim=0) / len(physical)**.5
        return mean, spread

    @torch.no_grad()
    def predict_3d_mols(self, mols: List[Chem.Mol], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        if not mols:
            return np.empty((0, self.output_dim)), np.empty((0, self.output_dim))
        dummy_y = np.zeros((len(mols), self.output_dim), dtype=np.float32)
        ds = _UniMol3DDataset(mols, dummy_y, remove_hs=self.remove_hs, max_atoms=self.max_atoms)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_unimol_3d)
        fold_outputs = []
        for model in self.folds:
            predictions = [model(tokens.to(self.device), distances.to(self.device), edges.to(self.device))
                           for tokens, distances, edges, _ in loader]
            fold_outputs.append(torch.cat(predictions))
        mean, std = self._physical_statistics(torch.stack(fold_outputs))
        return mean.cpu().numpy(), std.cpu().numpy()
