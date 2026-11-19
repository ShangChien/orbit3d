"""Layer 1: Unified Predictor & Physics Asset Hub (`src/loom/weaving/registry.py`).

Enables zero-retrain plug-and-play mounting of heterogeneous enterprise property predictors
alongside internal K-fold trainable predictors, built-in chemical constraint operators,
frozen black-box residual calibrators, and physics-constrained 3D coordinate refinement:

  - `UniMol3DAsset`           : 3D coordinate-differentiable `Uni-Mol` K-fold predictor (`grad_coords` enabled)
  - `UniMol2Asset`            : Pretrained / adapted `Uni-Mol2` 3D transformer evaluator
  - `DPA4PropertyAsset`       : `DPA-4` / `HyFF` 3D property evaluator (reward or isolated audit role)
  - `FingerprintQSARAsset`    : 2D Morgan fingerprint + descriptor ensemble QSAR model
  - `CallablePredictorAsset`  : Zero-retrain wrapper for arbitrary Python / sklearn / ONNX callables
  - `BuiltInRuleAsset`        : Built-in zero-cost chemical feasibility operators (`sa_score`, `qed`,
                                `mol_weight`, `rotatable_bonds`, `max_ring_size`, `strained_small_rings`,
                                `smarts_alert`)
  - `ResidualCalibratedAsset` : Non-destructive GP residual calibrator over any frozen black-box predictor
                                (`y_cal(M) = y_base(M) + delta_gp(M)`), updated by Grounded RSI feedback
  - `PhysicsConstrainedCoordRefiner` : Joint `MLIP Force + Differentiable 3D Oracle Gradient` conformer refiner
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import joblib
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, RDConfig, rdFingerprintGenerator, rdMolDescriptors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge

from ..generator.molecular_state import Mol3DInstance, mol3d_to_validated_rdkit_mol, mol_to_clean_smiles
from ..oracles.fingerprint_oracle import FingerprintOracle
from ..oracles.graph_features import RAW_GRAPH
from ..oracles.unimol_oracle import InternalUniMolOracle
from .slots import AuditOracleSlot, DEFAULT_SUPPORTED_ELEMENTS, PotentialSlot, RewardOracleSlot


def _load_rdkit_sascorer() -> Any:
    sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer  # type: ignore[import-not-found]

    return sascorer


_DEFAULT_REACTIVE_SMARTS: Tuple[str, ...] = (
    "[OX2][OX2]",               # Peroxides
    "[CX3](=[OX1])[F,Cl,Br,I]", # Acyl halides
    "[NX2]=[OX1]",              # Nitroso
    "[SX2][SX2][SX2]",          # Polysulfides
    "[C+]#[N-]",                # Isocyanide reversed
)


@dataclass(frozen=True)
class AssetCapabilities:
    asset_id: str
    modality: str  # "3d_differentiable" | "3d_blackbox" | "2d_graph" | "rule_analytical"
    is_3d_differentiable: bool
    has_uncertainty: bool
    target_names: Tuple[str, ...]
    units: Tuple[str, ...]
    is_audit_only: bool = False


class PredictorAsset(ABC):
    """Unified predictor-agnostic interface for Layer 1 of `Uni-Loom`."""

    capabilities: AssetCapabilities

    def channel_index(self, channel: str | int) -> int:
        if isinstance(channel, int):
            if 0 <= channel < len(self.capabilities.target_names):
                return channel
            raise IndexError(f"Channel index {channel} out of bounds for {self.capabilities.asset_id}")
        try:
            return self.capabilities.target_names.index(channel)
        except ValueError as exc:
            raise KeyError(
                f"Channel '{channel}' not found in asset '{self.capabilities.asset_id}' "
                f"(available: {self.capabilities.target_names})"
            ) from exc

    @abstractmethod
    def predict_mols(
        self, mols: Sequence[Optional[Chem.Mol]], batch_size: int = 32
    ) -> Dict[str, Any]:
        """Batch-evaluate `mols`, returning:
        `{"mean": (N, K), "std": (N, K), "valid_mask": (N,), "reasons": List[Optional[str]]}`.
        """

    def predict_channel(
        self, mols: Sequence[Optional[Chem.Mol]], channel: str | int = 0, batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_idx = self.channel_index(channel)
        out = self.predict_mols(mols, batch_size=batch_size)
        return (
            np.asarray(out["mean"][:, c_idx], dtype=np.float64),
            np.asarray(out["std"][:, c_idx], dtype=np.float64),
            np.asarray(out["valid_mask"], dtype=bool),
        )

    def grad_coords(self, mol: Chem.Mol, channel: str | int = 0) -> Tuple[float, float, np.ndarray]:
        """Return `(pred_scalar, pred_std, d_pred_d_coords)` of shape `(N_atoms, 3)` for 3D differentiable assets."""
        raise NotImplementedError(
            f"Asset '{self.capabilities.asset_id}' (modality={self.capabilities.modality}) "
            "does not expose analytical 3D coordinate gradients"
        )


class BuiltInRuleAsset(PredictorAsset):
    """Zero-cost built-in chemical constraint & synthesizability operators."""

    CHANNEL_UNITS: Dict[str, str] = {
        "sa_score": "SA(1-10)",
        "qed": "QED(0-1)",
        "mol_weight": "Da",
        "rotatable_bonds": "count",
        "max_ring_size": "atoms",
        "strained_small_rings": "count",
        "smarts_alert": "count",
    }

    def __init__(
        self,
        asset_id: str = "builtin_rules",
        channels: Sequence[str] = (
            "sa_score",
            "qed",
            "mol_weight",
            "rotatable_bonds",
            "max_ring_size",
            "strained_small_rings",
            "smarts_alert",
        ),
        alert_smarts: Sequence[str] = _DEFAULT_REACTIVE_SMARTS,
    ) -> None:
        unknown = [c for c in channels if c not in self.CHANNEL_UNITS]
        if unknown:
            raise ValueError(f"Unknown built-in rule channels: {unknown}")
        self._channels = tuple(channels)
        self._sascorer = _load_rdkit_sascorer() if "sa_score" in self._channels else None
        self._smarts_mols = [Chem.MolFromSmarts(p) for p in alert_smarts]
        self._smarts_mols = [m for m in self._smarts_mols if m is not None]
        self.capabilities = AssetCapabilities(
            asset_id=asset_id,
            modality="rule_analytical",
            is_3d_differentiable=False,
            has_uncertainty=False,
            target_names=self._channels,
            units=tuple(self.CHANNEL_UNITS[c] for c in self._channels),
            is_audit_only=False,
        )

    def predict_mols(self, mols: Sequence[Optional[Chem.Mol]], batch_size: int = 32) -> Dict[str, Any]:
        del batch_size
        n = len(mols)
        k = len(self._channels)
        mean = np.full((n, k), np.nan, dtype=np.float64)
        std = np.zeros((n, k), dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        reasons: List[Optional[str]] = [None] * n

        for i, mol in enumerate(mols):
            if mol is None:
                reasons[i] = "null_molecule"
                continue
            try:
                no_h = Chem.RemoveHs(Chem.Mol(mol))
                ring_sizes = [len(r) for r in no_h.GetRingInfo().AtomRings()]
                vals = []
                for ch in self._channels:
                    if ch == "sa_score":
                        vals.append(float(self._sascorer.calculateScore(no_h)))
                    elif ch == "qed":
                        vals.append(float(QED.qed(no_h)))
                    elif ch == "mol_weight":
                        vals.append(float(Descriptors.MolWt(no_h)))
                    elif ch == "rotatable_bonds":
                        vals.append(float(Descriptors.NumRotatableBonds(no_h)))
                    elif ch == "max_ring_size":
                        vals.append(float(max(ring_sizes) if ring_sizes else 0))
                    elif ch == "strained_small_rings":
                        vals.append(float(sum(1 for sz in ring_sizes if sz in (3, 4))))
                    elif ch == "smarts_alert":
                        vals.append(float(sum(1 for pat in self._smarts_mols if no_h.HasSubstructMatch(pat))))
                mean[i] = np.asarray(vals, dtype=np.float64)
                valid[i] = True
            except Exception as exc:
                reasons[i] = f"rule_eval_error:{exc}"

        return {"mean": mean, "std": std, "valid_mask": valid, "reasons": reasons}


class CallablePredictorAsset(PredictorAsset):
    """Zero-retrain adapter wrapping any external Python / sklearn / ONNX callable `fn(mols)`."""

    def __init__(
        self,
        asset_id: str,
        predict_fn: Callable[[Sequence[Chem.Mol]], Tuple[np.ndarray, Optional[np.ndarray]] | np.ndarray],
        target_names: Sequence[str],
        units: Optional[Sequence[str]] = None,
        *,
        modality: str = "2d_graph",
        grad_fn: Optional[Callable[[Chem.Mol, int], Tuple[float, float, np.ndarray]]] = None,
        is_audit_only: bool = False,
    ) -> None:
        t_names = tuple(target_names)
        u_names = tuple(units) if units is not None else tuple(["unspecified"] * len(t_names))
        self._predict_fn = predict_fn
        self._grad_fn = grad_fn
        self.capabilities = AssetCapabilities(
            asset_id=asset_id,
            modality="3d_differentiable" if grad_fn is not None else modality,
            is_3d_differentiable=grad_fn is not None,
            has_uncertainty=True,
            target_names=t_names,
            units=u_names,
            is_audit_only=is_audit_only,
        )

    def predict_mols(self, mols: Sequence[Optional[Chem.Mol]], batch_size: int = 32) -> Dict[str, Any]:
        del batch_size
        n = len(mols)
        k = len(self.capabilities.target_names)
        mean = np.full((n, k), np.nan, dtype=np.float64)
        std = np.zeros((n, k), dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        reasons: List[Optional[str]] = [None] * n

        idx_valid = [i for i, m in enumerate(mols) if m is not None]
        for i in range(n):
            if mols[i] is None:
                reasons[i] = "null_molecule"
        if idx_valid:
            sub = [mols[i] for i in idx_valid]  # type: ignore[misc]
            raw_out = self._predict_fn(sub)
            if isinstance(raw_out, tuple):
                p_arr, s_arr = raw_out
            else:
                p_arr, s_arr = raw_out, None
            p_np = np.asarray(p_arr, dtype=np.float64).reshape(len(sub), k)
            s_np = (
                np.asarray(s_arr, dtype=np.float64).reshape(len(sub), k)
                if s_arr is not None
                else np.zeros_like(p_np)
            )
            for loc_i, glob_i in enumerate(idx_valid):
                if np.isfinite(p_np[loc_i]).all():
                    mean[glob_i] = p_np[loc_i]
                    std[glob_i] = np.nan_to_num(s_np[loc_i], nan=0.0)
                    valid[glob_i] = True
                else:
                    reasons[glob_i] = "nonfinite_prediction"

        return {"mean": mean, "std": std, "valid_mask": valid, "reasons": reasons}

    def grad_coords(self, mol: Chem.Mol, channel: str | int = 0) -> Tuple[float, float, np.ndarray]:
        if self._grad_fn is None:
            return super().grad_coords(mol, channel)
        return self._grad_fn(mol, self.channel_index(channel))


class UniMol3DAsset(PredictorAsset):
    """3D coordinate-differentiable `Uni-Mol` K-Fold ensemble asset (`is_3d_differentiable=True`)."""

    def __init__(
        self,
        oracle: InternalUniMolOracle,
        asset_id: str = "unimol_3d",
        *,
        is_audit_only: bool = False,
    ) -> None:
        self.oracle = oracle
        self.capabilities = AssetCapabilities(
            asset_id=asset_id,
            modality="3d_differentiable",
            is_3d_differentiable=True,
            has_uncertainty=len(oracle.folds) > 1,
            target_names=tuple(oracle.target_cols),
            units=tuple(getattr(oracle, "target_units", ["unspecified"] * oracle.output_dim)),
            is_audit_only=is_audit_only,
        )

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: Path | str,
        asset_id: str = "unimol_3d",
        *,
        device: str = "cuda:0",
        is_audit_only: bool = False,
    ) -> "UniMol3DAsset":
        oracle = InternalUniMolOracle.load_from_dir(str(checkpoint_dir), device=device)
        return cls(oracle=oracle, asset_id=asset_id, is_audit_only=is_audit_only)

    @classmethod
    def train_internal(
        cls,
        mols: List[Chem.Mol],
        targets_2d: np.ndarray,
        save_dir: Path | str,
        *,
        pretrain_weight_path: str,
        target_names: Sequence[str],
        units: Optional[Sequence[str]] = None,
        n_folds: int = 3,
        epochs: int = 20,
        batch_size: int = 16,
        device: str = "cuda:0",
        asset_id: str = "unimol_3d_internal",
    ) -> "UniMol3DAsset":
        oracle = InternalUniMolOracle(
            output_dim=len(target_names),
            n_folds=n_folds,
            target_cols=list(target_names),
            pretrain_weight_path=pretrain_weight_path,
            device=device,
            target_units=list(units) if units else None,
        )
        oracle.fit_from_dataset(mols, targets_2d, str(save_dir), epochs=epochs, batch_size=batch_size)
        return cls(oracle=oracle, asset_id=asset_id)

    def predict_mols(self, mols: Sequence[Optional[Chem.Mol]], batch_size: int = 32) -> Dict[str, Any]:
        del batch_size
        n = len(mols)
        k = len(self.capabilities.target_names)
        mean = np.full((n, k), np.nan, dtype=np.float64)
        std = np.full((n, k), np.nan, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        reasons: List[Optional[str]] = [None] * n

        admitted_idx: List[int] = []
        admitted_mols: List[Chem.Mol] = []
        for i, m in enumerate(mols):
            if m is None or m.GetNumConformers() == 0:
                reasons[i] = "null_or_2d_molecule"
                continue
            admitted_idx.append(i)
            admitted_mols.append(m)

        if admitted_mols:
            # Evaluate in safe chunks; fall back per-molecule if an unsupported element appears
            try:
                p_arr, s_arr = self.oracle.predict_3d_mols(admitted_mols)
                p_np = np.asarray(p_arr, dtype=np.float64).reshape(len(admitted_mols), k)
                s_np = np.asarray(s_arr, dtype=np.float64).reshape(len(admitted_mols), k)
                for loc_i, glob_i in enumerate(admitted_idx):
                    if np.isfinite(p_np[loc_i]).all():
                        mean[glob_i] = p_np[loc_i]
                        std[glob_i] = s_np[loc_i]
                        valid[glob_i] = True
                    else:
                        reasons[glob_i] = "nonfinite_prediction"
            except Exception:
                for glob_i, m in zip(admitted_idx, admitted_mols):
                    try:
                        p1, s1 = self.oracle.predict_3d_mols([m])
                        mean[glob_i] = np.asarray(p1, dtype=np.float64).reshape(k)
                        std[glob_i] = np.asarray(s1, dtype=np.float64).reshape(k)
                        valid[glob_i] = True
                    except Exception as exc:
                        reasons[glob_i] = str(exc)

        return {"mean": mean, "std": std, "valid_mask": valid, "reasons": reasons}

    def grad_coords(self, mol: Chem.Mol, channel: str | int = 0) -> Tuple[float, float, np.ndarray]:
        c_idx = self.channel_index(channel)
        pos_np = mol.GetConformer().GetPositions().astype(np.float32)
        syms = [a.GetSymbol() for a in mol.GetAtoms()]
        coords_t = torch.tensor(pos_np, dtype=torch.float32, device=self.oracle.device, requires_grad=True)
        y_pred, y_std = self.oracle.forward_differentiable_3d(coords_t, syms)
        scalar = y_pred.reshape(-1)[c_idx]
        std_val = float(y_std.reshape(-1)[c_idx].detach().item())
        (grad_t,) = torch.autograd.grad(scalar, coords_t, retain_graph=False, create_graph=False)
        return float(scalar.detach().item()), std_val, grad_t.detach().cpu().numpy().astype(np.float64)


class SlotOracleAsset(PredictorAsset):
    """Wraps an existing `RewardOracleSlot` or `AuditOracleSlot` (`Uni-Mol2` or `DPA-4`) as a Layer 1 Asset."""

    def __init__(
        self,
        slot: RewardOracleSlot | AuditOracleSlot,
        asset_id: str,
        units: Optional[Sequence[str]] = None,
    ) -> None:
        self.slot = slot
        t_names = tuple(slot.target_cols)
        u_names = tuple(units) if units is not None else tuple(["eV"] * len(t_names))
        self.capabilities = AssetCapabilities(
            asset_id=asset_id,
            modality="3d_blackbox",
            is_3d_differentiable=False,
            has_uncertainty=slot.backend_name in ("dpa4", "hyff"),
            target_names=t_names,
            units=u_names,
            is_audit_only=bool(getattr(slot, "is_audit_only", False)),
        )

    def predict_mols(self, mols: Sequence[Optional[Chem.Mol]], batch_size: int = 32) -> Dict[str, Any]:
        preds, stds, v_mask, reasons = self.slot.evaluate_candidates(mols, batch_size=batch_size)
        return {"mean": preds, "std": stds, "valid_mask": v_mask, "reasons": reasons}


class FingerprintQSARAsset(PredictorAsset):
    """2D Morgan Fingerprint + Descriptor Bootstrap Ensemble QSAR Asset."""

    def __init__(
        self,
        oracle: FingerprintOracle,
        asset_id: str = "fingerprint_qsar",
        units: Optional[Sequence[str]] = None,
        *,
        is_audit_only: bool = False,
    ) -> None:
        self.oracle = oracle
        t_names = tuple(oracle.target_cols)
        u_names = tuple(units) if units is not None else tuple(getattr(oracle, "target_units", ["unspecified"] * len(t_names)))
        self.capabilities = AssetCapabilities(
            asset_id=asset_id,
            modality="2d_graph",
            is_3d_differentiable=False,
            has_uncertainty=True,
            target_names=t_names,
            units=u_names,
            is_audit_only=is_audit_only,
        )

    @classmethod
    def from_checkpoint_dir(
        cls, directory: Path | str, asset_id: str = "fingerprint_qsar", *, is_audit_only: bool = False
    ) -> "FingerprintQSARAsset":
        return cls(FingerprintOracle.load_from_dir(directory), asset_id=asset_id, is_audit_only=is_audit_only)

    @classmethod
    def train_internal(
        cls,
        mols: Sequence[Chem.Mol],
        targets: np.ndarray,
        save_dir: Path | str,
        *,
        target_names: Sequence[str],
        units: Optional[Sequence[str]] = None,
        n_models: int = 5,
        n_trees: int = 128,
        seed: int = 42,
        asset_id: str = "fingerprint_qsar_internal",
        graph_normalization: str = RAW_GRAPH,
    ) -> "FingerprintQSARAsset":
        oracle = FingerprintOracle(target_cols=tuple(target_names), graph_normalization=graph_normalization)
        y_mat = np.asarray(targets, dtype=np.float64).reshape(len(mols), len(target_names))
        clean_mols: List[Chem.Mol] = []
        clean_y: List[np.ndarray] = []
        for m, y_row in zip(mols, y_mat):
            if m is None or not np.isfinite(y_row).all():
                continue
            try:
                oracle.features([m])
            except Exception:
                continue
            clean_mols.append(m)
            clean_y.append(y_row)
        oracle.fit(clean_mols, np.stack(clean_y, axis=0), save_dir, seed=seed, n_models=n_models, n_trees=n_trees)
        return cls(oracle, asset_id=asset_id, units=units)

    def predict_mols(self, mols: Sequence[Optional[Chem.Mol]], batch_size: int = 32) -> Dict[str, Any]:
        del batch_size
        n = len(mols)
        k = len(self.capabilities.target_names)
        mean = np.full((n, k), np.nan, dtype=np.float64)
        std = np.full((n, k), np.nan, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        reasons: List[Optional[str]] = [None] * n

        valid_idx = [i for i, m in enumerate(mols) if m is not None]
        for i in range(n):
            if mols[i] is None:
                reasons[i] = "null_molecule"
        if valid_idx:
            sub = [mols[i] for i in valid_idx]  # type: ignore[misc]
            out = self.oracle.evaluate_graph_mols(sub)
            for loc_i, glob_i in enumerate(valid_idx):
                if out["errors"][loc_i]:
                    reasons[glob_i] = out["errors"][loc_i]
                elif np.isfinite(out["mean"][loc_i]).all():
                    mean[glob_i] = out["mean"][loc_i]
                    std[glob_i] = out["std"][loc_i]
                    valid[glob_i] = True
                else:
                    reasons[glob_i] = "nonfinite_prediction"

        return {"mean": mean, "std": std, "valid_mask": valid, "reasons": reasons}


def _mol_static_residual_descriptor(mol: Chem.Mol, fp_gen: Any) -> np.ndarray:
    """Compact 29-D folded Morgan + 2D/3D-invariant stoichiometry/hybridization + electronic + 3D shape descriptor."""
    no_h = Chem.RemoveHs(Chem.Mol(mol))
    bits = fp_gen.GetFingerprintAsNumPy(no_h).astype(np.float64)  # (256,)
    folded = bits.reshape(16, 16).mean(axis=1)
    bits_norm = 0.35 * (folded / max(np.linalg.norm(folded), 0.25))

    n_heavy = max(no_h.GetNumAtoms(), 1)
    n_h_tot = sum(int(a.GetTotalNumHs()) for a in no_h.GetAtoms())
    n_total = n_heavy + n_h_tot
    h_to_heavy = float(n_h_tot) / float(n_heavy)
    frac_csp3 = float(rdMolDescriptors.CalcFractionCSP3(no_h))
    n_arom = sum(1 for a in no_h.GetAtoms() if a.GetIsAromatic())
    frac_arom = float(n_arom) / float(n_heavy)
    n_hetero_ewg = sum(
        1 for a in no_h.GetAtoms() if a.GetSymbol() in ("N", "O", "F", "Cl", "Br", "S")
    )

    if no_h.GetNumConformers() > 0 and no_h.GetNumAtoms() > 1:
        pos = no_h.GetConformer().GetPositions().astype(np.float64)
        centered = pos - pos.mean(axis=0, keepdims=True)
        rg = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=-1))))
        cov = (centered.T @ centered) / max(len(pos), 1)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        span = float(np.max(np.linalg.norm(centered, axis=-1)))
    else:
        rg = 0.0
        eigvals = np.zeros(3, dtype=np.float64)
        span = 0.0

    stoich_hyb = 0.50 * np.array(
        [
            float(n_total) / 50.0,
            h_to_heavy / 1.5,
            frac_csp3,
            frac_arom,
        ],
        dtype=np.float64,
    )
    extra = 0.30 * np.array(
        [
            Descriptors.MolWt(no_h) / 500.0,
            Descriptors.TPSA(no_h) / 150.0,
            Descriptors.MolLogP(no_h) / 5.0,
            float(n_hetero_ewg) / float(n_heavy),
            rg / 5.0,
            span / 8.0,
            float(np.sqrt(max(eigvals[0], 0.0))) / 4.0,
            float(np.sqrt(max(eigvals[1], 0.0))) / 4.0,
            float(np.sqrt(max(eigvals[2], 0.0))) / 4.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate([bits_norm, stoich_hyb, extra])


class ResidualCalibratedAsset(PredictorAsset):
    """Non-destructive Bayesian GP Residual Calibrator wrapping any frozen `PredictorAsset`.

    Models `y_true(M) = y_base(M) + delta_gp(z(M))` with prior epistemic std `prior_blindspot_std`.
    - Before calibration (`num_feedback_samples == 0` and no anchors), returns exact `base_asset`
      mean predictions (`delta == 0`) with epistemic uncertainty `sqrt(sigma_base^2 + prior_blindspot_std^2)`.
    - Near in-domain anchors (`seed_in_domain_anchors`) or external Grounded-RSI feedback
      (`ingest_feedback`), posterior `sigma_gp(M)` contracts while `delta_gp(M)` corrects systematic
      extrapolation/truncation bias without modifying a single weight of `base_asset`.
    """

    def __init__(
        self,
        base_asset: PredictorAsset,
        *,
        asset_id: Optional[str] = None,
        prior_blindspot_std: float = 0.38,
    ) -> None:
        if base_asset.capabilities.is_audit_only:
            raise ValueError("Cannot attach training/RSI calibrator to an audit-only asset")
        self.base_asset = base_asset
        self.prior_blindspot_std = float(prior_blindspot_std)
        self._fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256, includeChirality=True)
        self._gps: List[Optional[GaussianProcessRegressor]] = [
            None for _ in base_asset.capabilities.target_names
        ]
        self._trends: List[Optional[Ridge]] = [
            None for _ in base_asset.capabilities.target_names
        ]
        self._anchor_static: List[np.ndarray] = []
        self._anchor_base_preds: List[np.ndarray] = []
        self._anchor_residuals: List[np.ndarray] = []

        self._feedback_static: List[np.ndarray] = []
        self._feedback_base_preds: List[np.ndarray] = []
        self._feedback_residuals: List[np.ndarray] = []

        self._pred_mean: Optional[np.ndarray] = None
        self._pred_scale: Optional[np.ndarray] = None

        cap = base_asset.capabilities
        self.capabilities = AssetCapabilities(
            asset_id=asset_id or f"{cap.asset_id}_calibrated",
            modality=cap.modality,
            is_3d_differentiable=cap.is_3d_differentiable,
            has_uncertainty=True,
            target_names=cap.target_names,
            units=cap.units,
            is_audit_only=False,
        )

    @property
    def num_feedback_samples(self) -> int:
        return len(self._feedback_static)

    @property
    def num_anchor_samples(self) -> int:
        return len(self._anchor_static)

    @property
    def calibrated_target_bounds(self) -> Optional[List[Tuple[float, float]]]:
        """Empirical `[min, max]` target bounds over all seeded anchors + ingested feedback (NaN-safe per channel)."""
        all_preds = self._anchor_base_preds + self._feedback_base_preds
        all_res = self._anchor_residuals + self._feedback_residuals
        if len(all_preds) < 2:
            return None
        y_obs = np.stack(all_preds, axis=0) + np.stack(all_res, axis=0)
        bounds: List[Tuple[float, float]] = []
        for j in range(y_obs.shape[1]):
            col = y_obs[:, j]
            fin = col[np.isfinite(col)]
            if len(fin) > 0:
                bounds.append((float(np.min(fin)), float(np.max(fin))))
            else:
                bounds.append((float("nan"), float("nan")))
        return bounds

    def _compose_features(self, static_mat: np.ndarray, base_preds: np.ndarray) -> np.ndarray:
        assert self._pred_mean is not None and self._pred_scale is not None
        z_pred = 0.85 * ((base_preds - self._pred_mean[None, :]) / np.maximum(self._pred_scale[None, :], 0.20))
        return np.concatenate([static_mat, z_pred], axis=-1)

    def _ensure_pred_scaler(self, base_preds_mat: np.ndarray) -> None:
        if self._pred_mean is None or self._pred_scale is None:
            self._pred_mean = np.mean(base_preds_mat, axis=0)
            self._pred_scale = np.maximum(np.std(base_preds_mat, axis=0), 0.25)

    def _refit_gps(self) -> None:
        all_static = self._anchor_static + self._feedback_static
        all_preds = self._anchor_base_preds + self._feedback_base_preds
        all_res = self._anchor_residuals + self._feedback_residuals
        if len(all_static) < 4:
            return

        preds_mat = np.stack(all_preds, axis=0)
        self._ensure_pred_scaler(preds_mat)
        X_mat = self._compose_features(np.stack(all_static, axis=0), preds_mat)
        R_mat = np.stack(all_res, axis=0)
        n_anc = len(self._anchor_static)

        k = len(self.capabilities.target_names)
        self._gps = []
        self._trends = []
        for c_i in range(k):
            r_col = R_mat[:, c_i]
            fin_mask = np.isfinite(r_col)
            if int(fin_mask.sum()) < 4:
                self._trends.append(None)
                self._gps.append(None)
                continue
            X_c = X_mat[fin_mask]
            r_vec = r_col[fin_mask]
            fb_fin_cnt = int(np.isfinite(R_mat[n_anc:, c_i]).sum()) if R_mat.shape[0] > n_anc else 0
            if fb_fin_cnt >= 3:
                trend = Ridge(alpha=6.0).fit(X_c, r_vec)
                r_detrended = r_vec - trend.predict(X_c)
                self._trends.append(trend)
            else:
                r_detrended = r_vec
                self._trends.append(None)

            prior_var = float(max(self.prior_blindspot_std ** 2, 0.04))
            kernel = ConstantKernel(
                prior_var, (prior_var * 0.88, prior_var * 1.15)
            ) * Matern(
                length_scale=0.65, length_scale_bounds=(0.45, 0.88), nu=1.5
            ) + WhiteKernel(
                noise_level=0.03, noise_level_bounds=(1e-3, 0.25)
            )
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=False, n_restarts_optimizer=2, random_state=42 + c_i
            ).fit(X_c, r_detrended)
            self._gps.append(gp)

    def seed_in_domain_anchors(
        self,
        anchor_mols: Sequence[Chem.Mol],
        anchor_true_targets: Optional[np.ndarray] = None,
        *,
        max_anchors: int = 256,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Seed in-domain reference structures so GP epistemic uncertainty is low in-domain and high OOD."""
        k = len(self.capabilities.target_names)
        n_in = len(anchor_mols)
        if n_in == 0:
            return {"seeded_anchors": 0}

        rng = np.random.default_rng(seed)
        if n_in > max_anchors:
            sel_idx = np.sort(rng.choice(n_in, size=max_anchors, replace=False))
        else:
            sel_idx = np.arange(n_in)

        sub_mols = [anchor_mols[int(i)] for i in sel_idx]
        base_out = self.base_asset.predict_mols(sub_mols)
        v_mask = base_out["valid_mask"]
        if anchor_true_targets is not None:
            y_all = np.asarray(anchor_true_targets, dtype=np.float64).reshape(n_in, k)[sel_idx]
            v_mask = v_mask & np.isfinite(y_all).all(axis=-1)
        else:
            y_all = None

        self._anchor_static = []
        self._anchor_base_preds = []
        self._anchor_residuals = []
        for loc_i, ok in enumerate(v_mask):
            if not ok:
                continue
            b_pred = np.asarray(base_out["mean"][loc_i], dtype=np.float64)
            res = (y_all[loc_i] - b_pred) if y_all is not None else np.zeros(k, dtype=np.float64)
            self._anchor_static.append(_mol_static_residual_descriptor(sub_mols[loc_i], self._fp_gen))
            self._anchor_base_preds.append(b_pred)
            self._anchor_residuals.append(res)

        if self._anchor_base_preds:
            self._pred_mean = np.mean(np.stack(self._anchor_base_preds, axis=0), axis=0)
            self._pred_scale = np.maximum(np.std(np.stack(self._anchor_base_preds, axis=0), axis=0), 0.25)
        self._refit_gps()
        return {"seeded_anchors": len(self._anchor_static)}

    def ingest_feedback(
        self,
        feedback_mols: Sequence[Chem.Mol],
        true_targets: np.ndarray,
    ) -> Dict[str, Any]:
        """Incrementally ingest external Grounded-RSI labels `(M_i, y_true_i)` (supports partial NaN assay readouts)."""
        k = len(self.capabilities.target_names)
        y_true = np.asarray(true_targets, dtype=np.float64).reshape(len(feedback_mols), k)
        base_out = self.base_asset.predict_mols(feedback_mols)
        v_mask = base_out["valid_mask"] & np.isfinite(y_true).any(axis=-1)

        added = 0
        per_channel_added = np.zeros(k, dtype=np.int64)
        pre_res_list: List[np.ndarray] = []
        for idx, ok in enumerate(v_mask):
            if not ok:
                continue
            b_pred = np.asarray(base_out["mean"][idx], dtype=np.float64)
            res = np.where(np.isfinite(y_true[idx]), y_true[idx] - b_pred, np.nan)
            self._feedback_static.append(_mol_static_residual_descriptor(feedback_mols[idx], self._fp_gen))
            self._feedback_base_preds.append(b_pred)
            self._feedback_residuals.append(res)
            pre_res_list.append(np.abs(res))
            per_channel_added += np.isfinite(res).astype(np.int64)
            added += 1

        self._refit_gps()
        base_mae_batch: Optional[List[Optional[float]]] = None
        if pre_res_list:
            rmat = np.stack(pre_res_list, axis=0)
            base_mae_batch = [
                (float(np.nanmean(rmat[:, j])) if np.isfinite(rmat[:, j]).any() else None)
                for j in range(k)
            ]
        return {
            "ingested_this_round": added,
            "ingested_per_channel": per_channel_added.tolist(),
            "total_feedback_samples": len(self._feedback_static),
            "base_uncalibrated_mae_on_batch": base_mae_batch,
        }

    def predict_mols(self, mols: Sequence[Optional[Chem.Mol]], batch_size: int = 32) -> Dict[str, Any]:
        base = self.base_asset.predict_mols(mols, batch_size=batch_size)
        mean = base["mean"].copy()
        std = np.nan_to_num(base["std"], nan=0.0).copy()
        v_mask = base["valid_mask"]

        if not any(gp is not None for gp in self._gps):
            std[v_mask] = np.sqrt(std[v_mask] ** 2 + self.prior_blindspot_std ** 2)
            return {"mean": mean, "std": std, "valid_mask": v_mask, "reasons": base["reasons"]}

        valid_indices = [i for i, ok in enumerate(v_mask) if ok and mols[i] is not None]
        if valid_indices:
            static_mat = np.stack(
                [_mol_static_residual_descriptor(mols[i], self._fp_gen) for i in valid_indices], axis=0  # type: ignore[arg-type]
            )
            feats = self._compose_features(static_mat, base["mean"][valid_indices])
            for c_i, gp in enumerate(self._gps):
                if gp is None:
                    continue
                trend_mu = (
                    self._trends[c_i].predict(feats)
                    if (c_i < len(self._trends) and self._trends[c_i] is not None)
                    else 0.0
                )
                delta_mu, delta_sd = gp.predict(feats, return_std=True)
                noise_var = float(getattr(gp.kernel_, "k2", None).noise_level) if hasattr(gp.kernel_, "k2") else 0.0
                epistemic_var = np.maximum(delta_sd ** 2 - noise_var, 0.0)
                mean[valid_indices, c_i] += trend_mu + delta_mu
                std[valid_indices, c_i] = np.sqrt(std[valid_indices, c_i] ** 2 + epistemic_var)

        return {"mean": mean, "std": std, "valid_mask": v_mask, "reasons": base["reasons"]}

    def grad_coords(self, mol: Chem.Mol, channel: str | int = 0) -> Tuple[float, float, np.ndarray]:
        val, sd, grad = self.base_asset.grad_coords(mol, channel=channel)
        c_idx = self.channel_index(channel)
        if self._gps[c_idx] is not None:
            base_vec = self.base_asset.predict_mols([mol])["mean"][0:1]
            static_mat = _mol_static_residual_descriptor(mol, self._fp_gen)[None, :]
            feat = self._compose_features(static_mat, base_vec)
            t_mu = (
                float(self._trends[c_idx].predict(feat)[0])
                if (c_idx < len(self._trends) and self._trends[c_idx] is not None)
                else 0.0
            )
            gp_c = self._gps[c_idx]
            assert gp_c is not None
            d_mu, d_sd = gp_c.predict(feat, return_std=True)
            noise_var = float(getattr(gp_c.kernel_, "k2", None).noise_level) if hasattr(gp_c.kernel_, "k2") else 0.0
            ep_var = max(float(d_sd[0]) ** 2 - noise_var, 0.0)
            val += t_mu + float(d_mu[0])
            sd = float(np.sqrt(sd ** 2 + ep_var))
        else:
            sd = float(np.sqrt(sd ** 2 + self.prior_blindspot_std ** 2))
        return val, sd, grad

    def save_calibrator(self, path: Path | str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "asset_id": self.capabilities.asset_id,
                "prior_blindspot_std": self.prior_blindspot_std,
                "pred_mean": self._pred_mean,
                "pred_scale": self._pred_scale,
                "anchor_static": self._anchor_static,
                "anchor_base_preds": self._anchor_base_preds,
                "anchor_residuals": self._anchor_residuals,
                "feedback_static": self._feedback_static,
                "feedback_base_preds": self._feedback_base_preds,
                "feedback_residuals": self._feedback_residuals,
                "trends": self._trends,
                "gps": self._gps,
            },
            out,
            compress=3,
        )

    def load_calibrator(self, path: Path | str) -> None:
        state = joblib.load(Path(path))
        self.prior_blindspot_std = float(state["prior_blindspot_std"])
        self._pred_mean = state["pred_mean"]
        self._pred_scale = state["pred_scale"]
        self._anchor_static = list(state.get("anchor_static", []))
        self._anchor_base_preds = list(state.get("anchor_base_preds", []))
        self._anchor_residuals = list(state.get("anchor_residuals", []))
        self._feedback_static = list(state.get("feedback_static", []))
        self._feedback_base_preds = list(state.get("feedback_base_preds", []))
        self._feedback_residuals = list(state.get("feedback_residuals", []))
        self._trends = list(state.get("trends", [None for _ in self.capabilities.target_names]))
        self._gps = list(state["gps"])


class PhysicsConstrainedCoordRefiner:
    """Boltzmann-Manifold Differentiable 3D Coordinate Refiner.

    Jointly optimizes `E_MLIP(R) + prop_weight * L_prop(R)` for 3D differentiable
    predictors (such as `Uni-Mol`) while projecting out adversarial bond-stretching
    property gradients whenever atomic forces exceed `force_gate_eV_A` and verifying
    exact RDKit covalent graph invariance.
    """

    def __init__(
        self,
        potential_slot: PotentialSlot,
        differentiable_asset: PredictorAsset,
        *,
        prop_weight: float = 0.35,
        step_size: float = 0.010,
        max_atom_disp: float = 0.025,
        force_gate_eV_A: float = 2.50,
    ) -> None:
        if not differentiable_asset.capabilities.is_3d_differentiable:
            raise ValueError(
                f"PhysicsConstrainedCoordRefiner requires `is_3d_differentiable=True`, "
                f"got {differentiable_asset.capabilities.asset_id}"
            )
        if differentiable_asset.capabilities.is_audit_only:
            raise ValueError("Audit-only asset cannot drive coordinate optimization")
        self.potential_slot = potential_slot
        self.asset = differentiable_asset
        self.prop_weight = float(prop_weight)
        self.step_size = float(step_size)
        self.max_atom_disp = float(max_atom_disp)
        self.force_gate_eV_A = float(force_gate_eV_A)

    def refine_molecule(
        self,
        mol: Chem.Mol,
        target_value: float,
        *,
        channel: str | int = 0,
        tolerance: float = 0.25,
        n_steps: int = 12,
    ) -> Dict[str, Any]:
        orig_smi = mol_to_clean_smiles(mol)
        syms = [a.GetSymbol() for a in mol.GetAtoms()]
        elem_tuple = tuple(self.potential_slot.atom_types)
        if any(s not in elem_tuple for s in syms):
            return {"mol": mol, "accepted": False, "reason": "unsupported_mlip_element"}

        cur_mol = Chem.Mol(mol)
        best_mol = Chem.Mol(mol)
        init_val, _, _ = self.asset.grad_coords(cur_mol, channel=channel)
        best_err = abs(init_val - float(target_value))
        best_val = init_val

        for _ in range(n_steps):
            val, _, grad_y = self.asset.grad_coords(cur_mol, channel=channel)
            err = val - float(target_value)
            if abs(err) <= 0.45 * tolerance:
                prop_force = np.zeros_like(grad_y)
            else:
                # Smooth Huber-like descent direction on |y(R) - target|
                scale = np.clip(err / max(tolerance, 1e-3), -2.0, 2.0)
                prop_force = -float(scale) * grad_y  # moves R toward target_value
                prop_force = prop_force - prop_force.mean(axis=0, keepdims=True)

            with torch.no_grad():
                coords_t, atypes_t, b_idx_t = self.potential_slot._pack_molecules([cur_mol])
                _, forces_t = self.potential_slot.engine.compute_energy_and_forces(
                    coords_t, atypes_t, b_idx_t, clip_forces=True, zero_com_forces=True
                )
                mlip_force = forces_t.detach().cpu().numpy().astype(np.float64)

            f_norms = np.linalg.norm(mlip_force, axis=-1)
            f_max = float(np.max(f_norms)) if len(f_norms) else 0.0

            # Project out property gradient components opposing restoring MLIP forces when strained
            if f_max > self.force_gate_eV_A:
                f_flat = mlip_force.reshape(-1)
                p_flat = prop_force.reshape(-1)
                denom = float(np.dot(f_flat, f_flat))
                dot_fp = float(np.dot(f_flat, p_flat))
                if denom > 1e-8 and dot_fp < 0.0:
                    prop_force = prop_force - (dot_fp / denom) * mlip_force

            total_force = mlip_force + self.prop_weight * prop_force
            disp = np.clip(self.step_size * total_force, -self.max_atom_disp, self.max_atom_disp)
            cand_pos = cur_mol.GetConformer().GetPositions() + disp

            inst = Mol3DInstance(
                coords=cand_pos.astype(np.float32),
                atom_types=np.array([elem_tuple.index(s) for s in syms], dtype=np.int64),
                element_table=elem_tuple,
            )
            try:
                rebuilt = mol3d_to_validated_rdkit_mol(inst)
                if mol_to_clean_smiles(rebuilt) != orig_smi:
                    break
            except ValueError:
                break

            cur_mol = rebuilt
            cand_val, _, _ = self.asset.grad_coords(cur_mol, channel=channel)
            cand_err = abs(cand_val - float(target_value))
            if cand_err <= best_err:
                best_err = cand_err
                best_val = cand_val
                best_mol = Chem.Mol(cur_mol)

        return {
            "mol": best_mol,
            "accepted": True,
            "init_pred": float(init_val),
            "final_pred": float(best_val),
            "init_abs_err": abs(init_val - float(target_value)),
            "final_abs_err": float(best_err),
        }


class AssetRegistry:
    """Central predictor-agnostic asset hub for `Uni-Loom`."""

    def __init__(self, *, register_builtin_rules: bool = True) -> None:
        self._predictors: Dict[str, PredictorAsset] = {}
        self._potentials: Dict[str, PotentialSlot] = {}
        if register_builtin_rules:
            self.register_predictor(BuiltInRuleAsset(asset_id="builtin_rules"))

    def register_predictor(self, asset: PredictorAsset) -> None:
        aid = asset.capabilities.asset_id
        self._predictors[aid] = asset

    def register_potential(self, potential_id: str, slot: PotentialSlot) -> None:
        self._potentials[potential_id] = slot

    def get_predictor(self, asset_id: str) -> PredictorAsset:
        if asset_id not in self._predictors:
            raise KeyError(f"PredictorAsset '{asset_id}' not registered (have: {sorted(self._predictors)})")
        return self._predictors[asset_id]

    def get_potential(self, potential_id: str) -> Optional[PotentialSlot]:
        return self._potentials.get(potential_id)

    def list_assets(self) -> Dict[str, Dict[str, Any]]:
        return {
            aid: {
                "modality": a.capabilities.modality,
                "is_3d_differentiable": a.capabilities.is_3d_differentiable,
                "has_uncertainty": a.capabilities.has_uncertainty,
                "target_names": list(a.capabilities.target_names),
                "units": list(a.capabilities.units),
                "is_audit_only": a.capabilities.is_audit_only,
            }
            for aid, a in self._predictors.items()
        }

    @classmethod
    def from_asset_configs(
        cls,
        asset_configs: Sequence[Dict[str, Any]],
        *,
        device: str = "cuda:0",
    ) -> "AssetRegistry":
        """Instantiate an `AssetRegistry` from declarative YAML/JSON asset definitions."""
        reg = cls(register_builtin_rules=True)
        for cfg in asset_configs:
            kind = str(cfg["type"])
            aid = str(cfg["id"])
            dev = str(cfg.get("device", device))

            if kind == "builtin_rules":
                channels = cfg.get("channels")
                reg.register_predictor(
                    BuiltInRuleAsset(asset_id=aid, channels=channels) if channels else BuiltInRuleAsset(asset_id=aid)
                )
            elif kind == "unimol2":
                is_aud = bool(cfg.get("audit_only", False))
                slot_cls = AuditOracleSlot if is_aud else RewardOracleSlot
                slot = slot_cls(
                    backend="unimol2",
                    checkpoint_dir=cfg["checkpoint_dir"],
                    device=dev,
                    expected_weight_sha256=cfg.get("expected_weight_sha256"),
                    supported_elements=cfg.get("supported_elements", DEFAULT_SUPPORTED_ELEMENTS),
                )
                base_asset: PredictorAsset = SlotOracleAsset(slot, asset_id=aid, units=cfg.get("units"))
                if cfg.get("residual_calibrated", False) and not is_aud:
                    base_asset = ResidualCalibratedAsset(base_asset, asset_id=aid)
                    if cfg.get("calibrator_path") and Path(cfg["calibrator_path"]).exists():
                        base_asset.load_calibrator(cfg["calibrator_path"])
                reg.register_predictor(base_asset)
            elif kind in ("dpa4_oracle", "hyff_oracle"):
                backend = "dpa4" if kind == "dpa4_oracle" else "hyff"
                is_aud = bool(cfg.get("audit_only", False))
                slot_cls = AuditOracleSlot if is_aud else RewardOracleSlot
                slot = slot_cls(
                    backend=backend,
                    checkpoint_dir=cfg["checkpoint_dir"],
                    device=dev,
                    forcefield_path=cfg.get("forcefield_path"),
                    supported_elements=cfg.get("supported_elements", DEFAULT_SUPPORTED_ELEMENTS),
                )
                base_asset = SlotOracleAsset(slot, asset_id=aid, units=cfg.get("units"))
                if cfg.get("residual_calibrated", False) and not is_aud:
                    base_asset = ResidualCalibratedAsset(base_asset, asset_id=aid)
                reg.register_predictor(base_asset)
            elif kind == "unimol_3d":
                base_asset = UniMol3DAsset.from_checkpoint_dir(
                    cfg["checkpoint_dir"],
                    asset_id=aid,
                    device=dev,
                    is_audit_only=bool(cfg.get("audit_only", False)),
                )
                if cfg.get("residual_calibrated", False) and not base_asset.capabilities.is_audit_only:
                    base_asset = ResidualCalibratedAsset(base_asset, asset_id=aid)
                reg.register_predictor(base_asset)
            elif kind == "fingerprint_qsar":
                base_asset = FingerprintQSARAsset.from_checkpoint_dir(
                    cfg["checkpoint_dir"],
                    asset_id=aid,
                    is_audit_only=bool(cfg.get("audit_only", False)),
                )
                if cfg.get("residual_calibrated", False) and not base_asset.capabilities.is_audit_only:
                    base_asset = ResidualCalibratedAsset(base_asset, asset_id=aid)
                reg.register_predictor(base_asset)
            elif kind == "potential":
                pot = PotentialSlot(
                    backend=str(cfg["backend"]),
                    weights=cfg.get("weights"),
                    device=dev,
                )
                reg.register_potential(aid, pot)
            else:
                raise ValueError(f"Unsupported declarative asset type: '{kind}'")
        return reg
