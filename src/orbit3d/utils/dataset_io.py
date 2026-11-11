"""
3D Molecular Dataset Loader & Differentiable Target Scaler Utilities.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import PandasTools
from .conformers import ocelot_conformer
from sklearn.preprocessing import PowerTransformer, StandardScaler


def load_3d_molecular_dataset(
    data_path: str,
    smiles_col: str = "SMILES",
    target_cols: Union[str, List[str]] = "TARGET",
    dpa_guidance: Optional[Any] = None,
    seed: int = 42,
    report_path: Optional[str] = None,
) -> Tuple[List[Chem.Mol], np.ndarray, List[str]]:
    """
    Loads a dataset from `.sdf` or `.csv`:
    - If `.sdf`: reads existing 3D conformers directly (`removeHs=False`).
    - If `.csv`: embeds 3D conformers via RDKit ETKDGv3 + MMFF94 (and optional DPA-4 pre-relaxation).
    Returns `(mols_3d, targets_2d, target_col_names)`.
    """
    if isinstance(target_cols, str):
        target_col_list = [c.strip() for c in target_cols.split(",") if c.strip()]
    else:
        target_col_list = list(target_cols)

    if data_path.endswith(".sdf"):
        df = PandasTools.LoadSDF(data_path, removeHs=False)
        raw_mols = df["ROMol"].tolist()
    else:
        df = pd.read_csv(data_path)
        raw_mols = [None] * len(df)

    missing_cols = [c for c in target_col_list if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Target columns {missing_cols} not found in {data_path}. Available: {list(df.columns)}"
        )

    mols_3d: List[Chem.Mol] = []
    valid_targets: List[np.ndarray] = []

    records = []
    for idx, (source_index, row) in enumerate(df.iterrows()):
        record = {'source_index': str(source_index), 'status': 'rejected'}
        records.append(record)
        try:
            y_vals = np.array([float(row[c]) for c in target_col_list], dtype=np.float32)
            if not np.isfinite(y_vals).all():
                raise ValueError('Nonfinite target values')
            mol = raw_mols[idx]
            if mol is None or not isinstance(mol, Chem.Mol) or mol.GetNumConformers() == 0:
                mol = ocelot_conformer(str(row[smiles_col]), seed=seed)
            else:
                mol = Chem.Mol(mol)
                if not mol.HasProp('GeometryOrigin'):
                    mol.SetProp('GeometryOrigin', 'Source SDF coordinates')
            from ..generator.octree_tokenizer import rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol
            elements = tuple(dict.fromkeys(atom.GetSymbol() for atom in mol.GetAtoms()))
            inst = rdkit_mol_to_mol3d_instance(mol, y_vals, center=False, element_table=elements)
            mol3d_to_validated_rdkit_mol(inst)
            if dpa_guidance is not None:
                from ..physics.conformer_relaxer import relax_conformer_energy_only
                refined, trace = relax_conformer_energy_only(inst, dpa_guidance, n_steps=8, step_size=.006)
                mol.GetConformer().SetPositions(refined.coords.astype(np.float64))
                mol.SetProp('GeometryOrigin', mol.GetProp('GeometryOrigin') + '; explicit graph-preserving MLIP refinement')
                record['refinement'] = trace
            mol.SetProp('SourceRow', str(source_index))
            record.update(status='accepted', geometry_origin=mol.GetProp('GeometryOrigin'))
            mols_3d.append(mol)
            valid_targets.append(y_vals)
        except ValueError as exc:
            record['reason'] = str(exc)
    if report_path is not None:
        Path(report_path).write_text(json.dumps({'parsed_rows': len(df), 'accepted': len(mols_3d),
            'rejected': len(df)-len(mols_3d), 'records': records}, indent=2) + '\n')

    if not valid_targets:
        raise ValueError("No usable molecular records with finite targets")
    targets_2d = np.stack(valid_targets, axis=0)
    return mols_3d, targets_2d, target_col_list


def fit_target_scalers(targets_2d: np.ndarray, mode: str = "standard") -> List[Any]:
    """Fits per-column TargetScaler (StandardScaler or PowerTransformer) for K targets."""
    if mode not in {"standard", "power"}:
        raise ValueError(f"Unknown target scaling method: {mode}")
    num_tasks = targets_2d.shape[1]
    scalers = []
    for k in range(num_tasks):
        col = targets_2d[:, k : k + 1]
        sc = PowerTransformer(method="yeo-johnson") if mode == "power" else StandardScaler()
        sc.fit(col)
        scalers.append(sc)
    return scalers


def transform_with_scalers(targets_2d: np.ndarray, scalers: List[Any]) -> np.ndarray:
    out = np.zeros_like(targets_2d, dtype=np.float32)
    for k, sc in enumerate(scalers):
        out[:, k : k + 1] = sc.transform(targets_2d[:, k : k + 1])
    return out


def inverse_single_scaler_torch(y_col: torch.Tensor, scaler_obj: Any, feature_index=None) -> torch.Tensor:
    """Differentiable PyTorch inverse_transform for a 1D/scalar tensor `y_col`."""
    if scaler_obj is None:
        return y_col
    inner = getattr(scaler_obj, "scaler", scaler_obj)
    if isinstance(inner, list) and len(inner) == 1:
        inner = inner[0]
    if inner is None:
        return y_col
    def parameter(name):
        value = np.asarray(getattr(inner, name)).reshape(-1)
        return float(value[feature_index] if feature_index is not None else value.item())

    cls_name = type(inner).__name__
    if cls_name == "StandardScaler":
        mean = parameter("mean_") if getattr(inner, 'with_mean', True) else 0.
        scale = parameter("scale_") if getattr(inner, 'with_std', True) else 1.
        return y_col * scale + mean
    if cls_name == 'RobustScaler':
        center = parameter("center_") if inner.with_centering else 0.
        scale = parameter("scale_") if inner.with_scaling else 1.
        return y_col * scale + center
    if cls_name == 'MinMaxScaler':
        return (y_col - parameter("min_")) / parameter("scale_")
    if cls_name == 'MaxAbsScaler':
        return y_col * parameter("scale_")
    if cls_name == 'FunctionTransformer':
        inverse = inner.inverse_func
        if inverse is None:
            return y_col
        functions = {np.exp: torch.exp, np.expm1: torch.expm1, np.log: torch.log, np.log1p: torch.log1p}
        if inverse in functions:
            return functions[inverse](y_col)
        raise NotImplementedError('No verified differentiable inverse for this FunctionTransformer')
    if "PowerTransformer" in cls_name and hasattr(inner, "lambdas_"):
        lmbda = parameter("lambdas_")
        x = inverse_single_scaler_torch(y_col, inner._scaler, feature_index) if inner.standardize else y_col
        if inner.method == 'box-cox':
            return torch.exp(x) if lmbda == 0 else torch.exp(torch.log1p(lmbda * x) / lmbda)
        if inner.method != 'yeo-johnson':
            raise NotImplementedError(f'Unknown power transform: {inner.method}')
        positive = x >= 0
        # Keep inactive branches in their real domain, without clipping a genuinely
        # invalid selected inverse into an apparently finite physical prediction.
        xp = torch.where(positive, x, torch.zeros_like(x))
        xn = torch.where(positive, torch.zeros_like(x), x)
        pos_value = torch.expm1(xp) if lmbda == 0 else torch.expm1(torch.log1p(lmbda * xp) / lmbda)
        complement = 2. - lmbda
        neg_value = -torch.expm1(-xn) if complement == 0 else -torch.expm1(torch.log1p(-complement * xn) / complement)
        return torch.where(positive, pos_value, neg_value)
    raise NotImplementedError(f'No verified differentiable inverse for {cls_name}')


def inverse_transform_with_scalers(y_norm: np.ndarray, scalers: List[Any]) -> np.ndarray:
    out = np.zeros_like(y_norm, dtype=np.float32)
    for k, sc in enumerate(scalers):
        out[:, k : k + 1] = sc.inverse_transform(y_norm[:, k : k + 1])
    return out


def write_precise_sdf(path, molecules, precision=12):
    """Write V3000 coordinates precisely enough to round-trip float32 model inputs."""
    params = Chem.MolWriterParams()
    params.forceV3000 = True
    params.precision = precision
    with open(path, "w", encoding="utf-8") as output:
        for mol in molecules:
            output.write(Chem.MolToMolBlock(mol, params))
            for name in mol.GetPropNames(includePrivate=False, includeComputed=False):
                value = mol.GetProp(name)
                if any(c in name for c in "\n\r<>") or "\n\n" in value or "$$$$" in value:
                    raise ValueError("Unsupported SDF property formatting")
                output.write(f">  <{name}>\n{value}\n\n")
            output.write("$$$$\n")
