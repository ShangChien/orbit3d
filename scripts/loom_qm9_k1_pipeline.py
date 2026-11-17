"""Fair 6x single-property (K=1) QM9 benchmark pipeline for Uni-Loom.

Trains:
  1. (`--phase train_surrogate`): An internal 6-output `StandaloneUniMolModel` reward surrogate
     initialized from `mol_pre_all_h_fp16.pt` strictly on `splits["train"]` (`train_rQM_v0`,
     zero overlap with PropMolFlow's `gvp_rQM_v0` held-out test split) and validated on `dev`.
  2. (`--phase run_props`): Six dedicated K=1 Uni-Loom models (one per QM9 property:
     alpha, gap, homo, lumo, mu, cv), benchmarking BOTH:
       - Arm A (`K=1 SFT`): LoRA-16 + LoomConditioner + HeavyAtomValenceHead + v11 XYZ heads
         with tail-balanced sampling & contrastive conditioning loss.
       - Arm B (`K=1 Full`): Arm A + Stage 1 Hindsight self-training (relabeled ONLY by the
         zero-leakage `train_rQM_v0` surrogate) + Stage 2.5A TopoGRPO (scale-decoupled discrete
         policy optimization with Track A/B valence purity).
     All CFG & checkpoint selection runs blindly on `splits["dev"]` using `QM9TrainSplitSurrogateOracle`.
     Final evaluation runs one-shot on `N=1,000` official PropMolFlow test targets via frozen
     `GVPBenchmarkSuite` oracles alongside warm batched A800 inference speed (`mol/s`) and
     per-stage training speed (`mol/s`) / GPU-hour accounting.
  3. (`--phase consolidate`): Merges K=6 SFT (`qm9_conditional_benchmark.json`), direct PropMolFlow
     6-model audit (`qm9_propmolflow_audit.json`), surrogate audit, and all 6 K=1 JSONs into
     `qm9_conditional_benchmark_v2.json`.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
import torch
import torch.nn as nn
import torch.nn.functional as F

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.chemistry.valence_reconciler import ValenceReconciler  # noqa: E402
from loom.generator.octree_backend import OctreeDataset  # noqa: E402
from loom.oracles.uniqsar_oracle import (  # noqa: E402
    StandaloneUniMolModel,
    UNIMOL_BOS_IDX,
    UNIMOL_DICT_SYMBOLS,
    UNIMOL_EOS_IDX,
    UNIMOL_PAD_IDX,
    UNIMOL_SYM2IDX,
)
from loom.weaving.grpo import LoomGRPOTrainer  # noqa: E402
from loom.weaving.uni3dar_slot import LoomUni3DARGenerator  # noqa: E402
from scripts.loom_qm9_benchmark import (  # noqa: E402
    ALLOWED_QM9_ELEMENTS,
    EV_PROPS,
    GVPBenchmarkSuite,
    HA_TO_EV,
    HA_TO_MEV,
    PROPS,
    PointCloudOctreeDataset,
    _move_packed_to_device,
    load_disjoint_qm9_splits,
)
from scripts.loom_train_sft import compute_tail_balanced_weights  # noqa: E402

PROP_GEN_UNITS: Dict[str, str] = {
    "alpha": "Bohr^3",
    "gap": "eV",
    "homo": "eV",
    "lumo": "eV",
    "mu": "D",
    "cv": "cal/(mol*K)",
}
PROP_EVAL_UNITS: Dict[str, str] = {
    "alpha": "Bohr^3",
    "gap": "meV",
    "homo": "meV",
    "lumo": "meV",
    "mu": "D",
    "cv": "cal/(mol*K)",
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _records_to_gen_targets(records: Sequence[Dict[str, Any]], prop_idx: int, prop_name: str) -> np.ndarray:
    vals = np.asarray([float(r["props_ha"][prop_idx]) for r in records], dtype=np.float32)
    if prop_name in EV_PROPS:
        vals = vals * float(HA_TO_EV)
    return vals


def _records_to_all_gen_targets(records: Sequence[Dict[str, Any]]) -> np.ndarray:
    arr = np.stack([np.asarray(r["props_ha"], dtype=np.float32) for r in records], axis=0)
    for idx, p in enumerate(PROPS):
        if p in EV_PROPS:
            arr[:, idx] *= float(HA_TO_EV)
    return arr


def _symbols_coords_to_conformer_mol(symbols: Sequence[str], coords: np.ndarray) -> Chem.Mol:
    """Fast bondless RDKit Mol holding conformer coordinates for `OctreeDataset` replay."""
    rw = Chem.RWMol()
    for sym in symbols:
        rw.AddAtom(Chem.Atom(str(sym)))
    conf = Chem.Conformer(len(symbols))
    pts = np.asarray(coords, dtype=np.float64)
    for i in range(len(symbols)):
        conf.SetAtomPosition(i, Point3D(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])))
    rw.AddConformer(conf, assignId=True)
    return rw.GetMol()


def _load_propmolflow_eval_targets(propmolflow_root: Path, prop: str, eval_n: int) -> np.ndarray:
    """Load official PropMolFlow sampling target values (Hartree for gap/homo/lumo, native otherwise)."""
    npy_path = propmolflow_root / "sampling_input" / "in_distribution_sampling" / f"train_half_sampled_values_{prop}.npy"
    arr = np.load(npy_path).astype(np.float64)
    if len(arr) < eval_n:
        raise ValueError(f"Expected at least {eval_n} targets in {npy_path}, found {len(arr)}")
    return arr[:eval_n]


def _target_ha_to_gen(prop: str, vals_ha: np.ndarray) -> np.ndarray:
    if prop in EV_PROPS:
        return vals_ha.astype(np.float32) * float(HA_TO_EV)
    return vals_ha.astype(np.float32)


def _target_gen_to_ha(prop: str, vals_gen: np.ndarray) -> np.ndarray:
    if prop in EV_PROPS:
        return vals_gen.astype(np.float64) / float(HA_TO_EV)
    return vals_gen.astype(np.float64)


# ==============================================================================
# Zero-Leakage QM9 `train_rQM_v0` Surrogate Oracle (StandaloneUniMolModel)
# ==============================================================================
def _pack_unimol_tensors(
    symbols_list: Sequence[Sequence[str]],
    coords_list: Sequence[np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack variable-length geometries into `(tokens [N, L], coords [N, L, 3], lengths [N])`."""
    n = len(symbols_list)
    max_len = max(len(s) for s in symbols_list) + 2
    tokens = np.full((n, max_len), UNIMOL_PAD_IDX, dtype=np.int64)
    coords = np.zeros((n, max_len, 3), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)

    for i, (syms, xyz) in enumerate(zip(symbols_list, coords_list)):
        m = len(syms)
        lengths[i] = m + 2
        tokens[i, 0] = UNIMOL_BOS_IDX
        tokens[i, 1 : m + 1] = [UNIMOL_SYM2IDX[s] for s in syms]
        tokens[i, m + 1] = UNIMOL_EOS_IDX
        c = np.asarray(xyz, dtype=np.float32)
        c = c - c.mean(axis=0, keepdims=True)
        coords[i, 1 : m + 1, :] = c

    return torch.from_numpy(tokens), torch.from_numpy(coords), torch.from_numpy(lengths)


def _forward_unimol_batch(
    model: StandaloneUniMolModel,
    tokens: torch.Tensor,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    *,
    use_bf16: bool,
) -> torch.Tensor:
    max_l = int(lengths.max().item())
    t_sub = tokens[:, :max_l]
    c_sub = coords[:, :max_l, :]
    d_sub = torch.cdist(c_sub, c_sub, p=2.0)
    dict_len = len(UNIMOL_DICT_SYMBOLS)
    e_sub = (t_sub.unsqueeze(-1) * dict_len + t_sub.unsqueeze(1)).long()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(use_bf16 and t_sub.device.type == "cuda")):
        return model(t_sub, d_sub, c_sub, e_sub).float()


class QM9TrainSplitSurrogateOracle:
    """Train-split-only multi-target 3D Uni-Mol surrogate implementing the Loom `reward_oracle` contract."""

    is_audit_only: bool = False
    slot_role: str = "reward_oracle"
    oracle_name: str = "qm9_train_unimol_surrogate"
    target_cols: List[str] = list(PROPS)

    def __init__(self, ckpt_path: Path, device: str = "cuda:0") -> None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.mean_gen = np.asarray(payload["mean_gen"], dtype=np.float32)
        self.std_gen = np.asarray(payload["std_gen"], dtype=np.float32)
        self.split_audit = payload.get("split_audit", {})
        self.dev_maes_eval_unit = payload.get("dev_maes_eval_unit", {})

        model = StandaloneUniMolModel(output_dim=len(PROPS), dict_len=len(UNIMOL_DICT_SYMBOLS), dropout=0.0)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def evaluate_candidates(
        self,
        mols: Sequence[Optional[Chem.Mol]],
        batch_size: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        n = len(mols)
        k = len(PROPS)
        preds = np.full((n, k), np.nan, dtype=np.float32)
        stds = np.zeros((n, k), dtype=np.float32)
        eval_mask = np.zeros(n, dtype=bool)
        reasons = ["missing_mol"] * n

        valid_idx: List[int] = []
        valid_syms: List[List[str]] = []
        valid_coords: List[np.ndarray] = []
        for i, mol in enumerate(mols):
            if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
                reasons[i] = "empty_or_no_conformer"
                continue
            syms = [a.GetSymbol() for a in mol.GetAtoms()]
            if any(s not in UNIMOL_SYM2IDX for s in syms) or len(syms) > 96:
                reasons[i] = "unsupported_element_or_size"
                continue
            pts = mol.GetConformer().GetPositions().astype(np.float32)
            if not np.isfinite(pts).all():
                reasons[i] = "nonfinite_coords"
                continue
            valid_idx.append(i)
            valid_syms.append(syms)
            valid_coords.append(pts)
            eval_mask[i] = True
            reasons[i] = "ok"

        if not valid_idx:
            return preds, stds, eval_mask, reasons

        mean_t = torch.tensor(self.mean_gen, device=self.device, dtype=torch.float32)
        std_t = torch.tensor(self.std_gen, device=self.device, dtype=torch.float32)
        for s in range(0, len(valid_idx), batch_size):
            e = min(s + batch_size, len(valid_idx))
            tok, crd, ln = _pack_unimol_tensors(valid_syms[s:e], valid_coords[s:e])
            out_norm = _forward_unimol_batch(
                self.model,
                tok.to(self.device),
                crd.to(self.device),
                ln.to(self.device),
                use_bf16=(self.device.type == "cuda"),
            )
            out_gen = (out_norm * std_t + mean_t).detach().cpu().numpy().astype(np.float32)
            for j, g_idx in enumerate(valid_idx[s:e]):
                preds[g_idx] = out_gen[j]

        return preds, stds, eval_mask, reasons


def train_surrogate_phase(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    qm9_raw_dir = args.propmolflow_root / "data" / "qm9_raw"
    splits = load_disjoint_qm9_splits(args.lmdb_dir, qm9_raw_dir, dev_size=1500, holdout_size=1000, seed=args.seed)
    audit = splits["audit"]
    assert int(audit["overlap_train_with_gvp_rQM_v0"]) == 0, f"Leakage detected: {audit}"

    # Secondary defensive set overlap verification against gvp_rQM_v0.npy
    gvp_indices = set(int(x) for x in np.load(qm9_raw_dir / "gvp_rQM_v0.npy").reshape(-1))
    train_indices = {int(r["gdb9_idx"]) for r in splits["train"]}
    dev_indices = {int(r["gdb9_idx"]) for r in splits["dev"]}
    assert not (train_indices & gvp_indices), "Train split overlaps gvp_rQM_v0"
    assert not (dev_indices & gvp_indices), "Dev split overlaps gvp_rQM_v0"

    train_recs = splits["train"]
    dev_recs = splits["dev"]
    y_train_gen = _records_to_all_gen_targets(train_recs)
    y_dev_gen = _records_to_all_gen_targets(dev_recs)
    mean_gen = y_train_gen.mean(axis=0).astype(np.float32)
    std_gen = y_train_gen.std(axis=0).astype(np.float32)
    y_train_norm = torch.from_numpy((y_train_gen - mean_gen) / std_gen).to(device)
    y_dev_norm = torch.from_numpy((y_dev_gen - mean_gen) / std_gen).to(device)

    print("[Surrogate] Packing train & dev tensors...", flush=True)
    tr_tok, tr_crd, tr_len = _pack_unimol_tensors(
        [r["symbols"] for r in train_recs],
        [r["coords"] for r in train_recs],
    )
    dv_tok, dv_crd, dv_len = _pack_unimol_tensors(
        [r["symbols"] for r in dev_recs],
        [r["coords"] for r in dev_recs],
    )
    tr_tok, tr_crd, tr_len = tr_tok.to(device), tr_crd.to(device), tr_len.to(device)
    dv_tok, dv_crd, dv_len = dv_tok.to(device), dv_crd.to(device), dv_len.to(device)

    model = StandaloneUniMolModel(output_dim=len(PROPS), dict_len=len(UNIMOL_DICT_SYMBOLS), dropout=0.05)
    raw_ckpt = torch.load(args.unimol_pretrained, map_location="cpu", weights_only=False)
    pre_sd = raw_ckpt.get("model", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
    cur_sd = model.state_dict()
    load_sd = {
        k: v.float() for k, v in pre_sd.items() if k in cur_sd and tuple(v.shape) == tuple(cur_sd[k].shape)
    }
    model.load_state_dict(load_sd, strict=False)
    assert len(load_sd) > 100, f"Expected pretrained UniMol tensors, only matched {len(load_sd)}"
    print(f"[Surrogate] Loaded {len(load_sd)}/{len(cur_sd)} pretrained UniMol tensors from {args.unimol_pretrained}", flush=True)

    model = model.to(device)
    n_train = tr_tok.shape[0]
    bsz = args.surrogate_batch_size
    steps_per_epoch = math.ceil(n_train / bsz)
    total_steps = args.surrogate_epochs * steps_per_epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-4)
    warmup = min(300, total_steps // 10)

    def _lr_mult(step_i: int) -> float:
        if step_i < warmup:
            return float(step_i + 1) / float(max(1, warmup))
        prog = float(step_i - warmup) / float(max(1, total_steps - warmup))
        return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_mult)
    criterion = nn.SmoothL1Loss(beta=0.05)

    best_dev_std_mae = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_dev_eval_unit: Dict[str, float] = {}
    t_start = time.perf_counter()
    rng = torch.Generator(device="cpu").manual_seed(args.seed)

    for epoch in range(1, args.surrogate_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=rng).to(device)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, n_train, bsz):
            idx = perm[s : s + bsz]
            optimizer.zero_grad(set_to_none=True)
            pred_norm = _forward_unimol_batch(
                model,
                tr_tok[idx],
                tr_crd[idx],
                tr_len[idx],
                use_bf16=(device.type == "cuda"),
            )
            loss = criterion(pred_norm, y_train_norm[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach().item())
            n_batches += 1

        model.eval()
        dev_preds: List[torch.Tensor] = []
        with torch.no_grad():
            for s in range(0, dv_tok.shape[0], bsz):
                e = min(s + bsz, dv_tok.shape[0])
                dev_preds.append(
                    _forward_unimol_batch(
                        model,
                        dv_tok[s:e],
                        dv_crd[s:e],
                        dv_len[s:e],
                        use_bf16=(device.type == "cuda"),
                    )
                )
        dev_pred_norm = torch.cat(dev_preds, dim=0)
        abs_norm_err = (dev_pred_norm - y_dev_norm).abs().mean(dim=0).cpu().numpy()
        mean_std_mae = float(abs_norm_err.mean())

        abs_gen_err = abs_norm_err * std_gen
        eval_unit_err = {
            p: round(float(abs_gen_err[i] * (1000.0 if p in EV_PROPS else 1.0)), 4)
            for i, p in enumerate(PROPS)
        }
        if mean_std_mae < best_dev_std_mae:
            best_dev_std_mae = mean_std_mae
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_dev_eval_unit = eval_unit_err

        elapsed = time.perf_counter() - t_start
        print(
            f"[Surrogate Epoch {epoch:02d}/{args.surrogate_epochs:02d}] "
            f"loss={epoch_loss / max(n_batches, 1):.4f} dev_std_mae={mean_std_mae:.4f} "
            f"dev_mae={eval_unit_err} ({elapsed:.1f}s)",
            flush=True,
        )

    assert best_state is not None
    ckpt_out = args.surrogate_ckpt
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "mean_gen": mean_gen.tolist(),
            "std_gen": std_gen.tolist(),
            "props": list(PROPS),
            "dev_std_mae": best_dev_std_mae,
            "dev_maes_eval_unit": best_dev_eval_unit,
            "train_wall_seconds": round(time.perf_counter() - t_start, 2),
            "split_audit": audit,
            "zero_gvp_overlap_verified": True,
        },
        ckpt_out,
    )
    audit_json = args.results_dir / "qm9_surrogate_audit.json"
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        "checkpoint": str(ckpt_out),
        "n_train": len(train_recs),
        "n_dev": len(dev_recs),
        "epochs": args.surrogate_epochs,
        "best_dev_std_mae": round(best_dev_std_mae, 5),
        "dev_maes_eval_unit": best_dev_eval_unit,
        "train_wall_seconds": round(time.perf_counter() - t_start, 2),
        "split_audit": audit,
        "zero_gvp_overlap_verified": True,
    }
    audit_json.write_text(json.dumps(audit_payload, indent=2) + "\n")
    print(f"[Surrogate] Saved checkpoint to {ckpt_out} and audit to {audit_json}", flush=True)


# ==============================================================================
# Single-Property Prefetched SFT Stream (Strict `active_mask = 1`)
# ==============================================================================
class SinglePropPrefetchedSFTStream:
    """Async CPU prefetched sampler for K=1 SFT and mixed Hindsight+Train SFT."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        targets_gen: np.ndarray,
        backbone: Any,
        *,
        prop_mean: float,
        prop_std: float,
        primary_count: int,
        extra_mix_prob: float = 0.0,
        tail_power: float = 0.35,
        batch_size: int = 128,
        seed: int = 42,
        prefetch_batches: int = 4,
    ) -> None:
        self.dataset = PointCloudOctreeDataset(records, backbone, training=True, seed=seed)
        self.targets_gen = np.asarray(targets_gen, dtype=np.float32)
        self.prop_mean = float(prop_mean)
        self.prop_std = float(prop_std)
        self.primary_count = int(primary_count)
        self.extra_count = len(records) - self.primary_count
        self.extra_mix_prob = float(extra_mix_prob) if self.extra_count >= 16 else 0.0
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

        primary_weights = compute_tail_balanced_weights(
            self.targets_gen[: self.primary_count, None], [0], power=tail_power, n_bins=24
        )
        self.primary_probs = primary_weights / np.sum(primary_weights)

        self._queue: deque = deque()
        self._cond = threading.Condition()
        self._stop = False
        self._prefetch = prefetch_batches
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _build_batch(self) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        for _ in range(12):
            if self.extra_mix_prob > 0.0 and self.extra_count > 0:
                n_extra = max(1, int(round(self.batch_size * self.extra_mix_prob)))
                n_prim = self.batch_size - n_extra
                idx_p = self.rng.choice(self.primary_count, size=n_prim, replace=True, p=self.primary_probs)
                idx_e = self.primary_count + self.rng.integers(0, self.extra_count, size=n_extra)
                indices = np.concatenate([idx_p, idx_e])
                self.rng.shuffle(indices)
            else:
                indices = self.rng.choice(self.primary_count, size=self.batch_size, replace=True, p=self.primary_probs)

            out = self.dataset.collate_valid(indices.tolist())
            if out is None:
                continue
            packed, kept_local_pos = out
            kept_global = indices[kept_local_pos]
            vals = self.targets_gen[kept_global]
            t_norm = ((vals - self.prop_mean) / self.prop_std)[:, None].astype(np.float32)
            active = np.ones((len(kept_local_pos), 1), dtype=bool)
            return packed, t_norm, active
        raise RuntimeError("Failed to collate valid SFT batch after 12 attempts")

    def _worker(self) -> None:
        while not self._stop:
            batch = self._build_batch()
            with self._cond:
                while len(self._queue) >= self._prefetch and not self._stop:
                    self._cond.wait(timeout=0.05)
                if self._stop:
                    return
                self._queue.append(batch)
                self._cond.notify_all()

    def next_batch(self) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        with self._cond:
            while not self._queue and not self._stop:
                self._cond.wait(timeout=0.05)
            item = self._queue.popleft()
            self._cond.notify_all()
            return item

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=2.0)


# ==============================================================================
# Unified Evaluation Helpers (Dev Surrogate Blind Check & Held-Out GVP Suite)
# ==============================================================================
def _batched_sample_mols(
    generator: LoomUni3DARGenerator,
    reconciler: ValenceReconciler,
    targets_gen: np.ndarray,
    *,
    batch_size: int,
    cfg_scale: float,
    cfg_tree_decay: float = 0.25,
    geometric_guard: bool,
    seed: int,
    device: torch.device,
    do_warmup: bool = False,
) -> Dict[str, Any]:
    """Generate molecules across `targets_gen` with accurate GPU timing and Track A/B validation."""
    n_total = len(targets_gen)
    generator.eval()

    if do_warmup and n_total > 0:
        w_n = min(batch_size, n_total)
        w_t = generator.normalize_targets(targets_gen[:w_n, None].astype(np.float32))
        w_m = torch.ones((w_n, 1), dtype=torch.bool, device=device)
        for w_i in range(2):
            _ = generator.sample(
                w_t,
                w_m,
                cfg_scale=cfg_scale,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=cfg_tree_decay,
                geometric_guard=geometric_guard,
                temperatures=(0.85, 0.30, 0.25),
                seed=seed + 777 + w_i,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
            )
        _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    t0 = time.perf_counter()

    trajectories: List[Dict[str, Any]] = []
    for s in range(0, n_total, batch_size):
        e = min(s + batch_size, n_total)
        b_t = generator.normalize_targets(targets_gen[s:e, None].astype(np.float32))
        b_m = torch.ones((e - s, 1), dtype=torch.bool, device=device)
        trajs = generator.sample(
            b_t,
            b_m,
            cfg_scale=cfg_scale,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=cfg_tree_decay,
            geometric_guard=geometric_guard,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
        )
        trajectories.extend(trajs)

    _sync(device)
    sampling_wall_s = time.perf_counter() - t0
    peak_vram_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0

    raw_mols: List[Optional[Chem.Mol]] = []
    rec_mols: List[Optional[Chem.Mol]] = []
    raw_legal = 0
    raw_exotic = 0
    rec_legal = 0
    rec_exotic = 0
    fallback_count = 0
    max_rmsd = 0.0
    raw_clean_records: List[Dict[str, Any]] = []
    smiles_list: List[Optional[str]] = []

    for traj in trajectories:
        res = reconciler.reconcile_point_cloud(traj["atom_symbols"], traj["positions_np"], potential_slot=None)
        if res.raw_legal and res.raw_mol is not None:
            raw_legal += 1
            raw_mols.append(res.raw_mol)
            if res.raw_exotic_charge:
                raw_exotic += 1
            else:
                raw_clean_records.append(
                    {
                        "mol": res.raw_mol,
                        "symbols": list(traj["atom_symbols"]),
                        "coords": np.asarray(traj["positions_np"], dtype=np.float32),
                        "smiles": Chem.MolToSmiles(Chem.RemoveHs(res.raw_mol), isomericSmiles=True),
                    }
                )
        else:
            raw_mols.append(None)

        if res.reconciled_legal and res.mol is not None:
            rec_legal += 1
            rec_mols.append(res.mol)
            if res.reconciled_exotic_charge:
                rec_exotic += 1
            if res.was_reconciled:
                fallback_count += 1
            if math.isfinite(res.heavy_rmsd_A):
                max_rmsd = max(max_rmsd, float(res.heavy_rmsd_A))
            smiles_list.append(res.smiles)
        else:
            rec_mols.append(None)
            smiles_list.append(None)

    return {
        "n_proposals": n_total,
        "sampling_wall_seconds": round(sampling_wall_s, 3),
        "sampling_mol_per_sec": round(float(n_total) / max(sampling_wall_s, 1e-6), 2),
        "ms_per_mol": round(1000.0 * sampling_wall_s / max(n_total, 1), 3),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "raw_mols": raw_mols,
        "rec_mols": rec_mols,
        "raw_legal_count": raw_legal,
        "raw_legal_pct": round(100.0 * raw_legal / max(n_total, 1), 2),
        "raw_exotic_count": raw_exotic,
        "raw_exotic_charge_pct": round(100.0 * raw_exotic / max(raw_legal, 1), 2),
        "rec_legal_count": rec_legal,
        "rec_legal_pct": round(100.0 * rec_legal / max(n_total, 1), 2),
        "rec_exotic_count": rec_exotic,
        "rec_exotic_charge_pct": round(100.0 * rec_exotic / max(rec_legal, 1), 2),
        "reconciled_fallback_count": fallback_count,
        "max_heavy_rmsd_A": round(max_rmsd, 6),
        "smiles_list": smiles_list,
        "raw_clean_records": raw_clean_records,
    }


def _evaluate_dev_with_surrogate(
    generator: LoomUni3DARGenerator,
    surrogate: QM9TrainSplitSurrogateOracle,
    reconciler: ValenceReconciler,
    prop_idx: int,
    prop_name: str,
    prop_std: float,
    dev_targets_gen: np.ndarray,
    *,
    batch_size: int,
    cfg_scale: float,
    cfg_tree_decay: float = 0.25,
    geometric_guard: bool,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    gen_out = _batched_sample_mols(
        generator,
        reconciler,
        dev_targets_gen,
        batch_size=batch_size,
        cfg_scale=cfg_scale,
        cfg_tree_decay=cfg_tree_decay,
        geometric_guard=geometric_guard,
        seed=seed,
        device=device,
        do_warmup=False,
    )
    preds_raw, _, mask_raw, _ = surrogate.evaluate_candidates(gen_out["raw_mols"], batch_size=128)
    preds_rec, _, mask_rec, _ = surrogate.evaluate_candidates(gen_out["rec_mols"], batch_size=128)

    scale = 1000.0 if prop_name in EV_PROPS else 1.0
    if bool(mask_raw.any()):
        raw_mae_gen = float(np.mean(np.abs(preds_raw[mask_raw, prop_idx] - dev_targets_gen[mask_raw])))
        raw_r = (
            float(np.corrcoef(preds_raw[mask_raw, prop_idx], dev_targets_gen[mask_raw])[0, 1])
            if int(mask_raw.sum()) >= 4
            else float("nan")
        )
    else:
        raw_mae_gen = float("inf")
        raw_r = float("nan")

    if bool(mask_rec.any()):
        rec_mae_gen = float(np.mean(np.abs(preds_rec[mask_rec, prop_idx] - dev_targets_gen[mask_rec])))
        rec_r = (
            float(np.corrcoef(preds_rec[mask_rec, prop_idx], dev_targets_gen[mask_rec])[0, 1])
            if int(mask_rec.sum()) >= 4
            else float("nan")
        )
    else:
        rec_mae_gen = float("inf")
        rec_r = float("nan")

    raw_rate = gen_out["raw_legal_pct"] / 100.0
    raw_exotic_rate = gen_out["raw_exotic_charge_pct"] / 100.0
    # Combined blind dev selection criterion balancing normalized property accuracy + clean 3D legality
    dev_score = (rec_mae_gen / max(prop_std, 1e-6)) + 0.35 * max(0.0, 0.90 - raw_rate) + 0.15 * raw_exotic_rate
    return {
        "cfg_scale": cfg_scale,
        "geometric_guard": geometric_guard,
        "raw_legal_pct": gen_out["raw_legal_pct"],
        "raw_exotic_charge_pct": gen_out["raw_exotic_charge_pct"],
        "rec_legal_pct": gen_out["rec_legal_pct"],
        "raw_surrogate_mae_eval": round(raw_mae_gen * scale, 4),
        "rec_surrogate_mae_eval": round(rec_mae_gen * scale, 4),
        "raw_surrogate_r": round(raw_r, 4) if math.isfinite(raw_r) else None,
        "rec_surrogate_r": round(rec_r, 4) if math.isfinite(rec_r) else None,
        "dev_score": round(dev_score, 5),
    }


def _time_k1_inference_bs(
    gen: LoomUni3DARGenerator,
    targets_gen: np.ndarray,
    *,
    batch_size: int,
    n_batches: int,
    cfg_scale: float,
    cfg_tree_decay: float,
    geometric_guard: bool,
    seed: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Warm timed GPU throughput (`mol/s`, `ms/mol`) at exact requested batch size."""
    gen.eval()
    w_t = gen.normalize_targets(targets_gen[:batch_size, None].astype(np.float32))
    w_m = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
    for w_i in range(2):
        _ = gen.sample(
            w_t,
            w_m,
            cfg_scale=cfg_scale,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=cfg_tree_decay,
            geometric_guard=geometric_guard,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed + 1900 + w_i,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
        )
    _sync(device)
    t0 = time.perf_counter()
    total_n = 0
    for b_i in range(n_batches):
        s = (b_i * batch_size) % max(len(targets_gen) - batch_size + 1, 1)
        b_t = gen.normalize_targets(targets_gen[s : s + batch_size, None].astype(np.float32))
        if b_t.shape[0] < batch_size:
            b_t = w_t
        b_m = torch.ones((b_t.shape[0], 1), dtype=torch.bool, device=device)
        _ = gen.sample(
            b_t,
            b_m,
            cfg_scale=cfg_scale,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=cfg_tree_decay,
            geometric_guard=geometric_guard,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed + 2000 + b_i,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
        )
        total_n += int(b_t.shape[0])
    _sync(device)
    wall = max(time.perf_counter() - t0, 1e-6)
    return round(float(total_n) / wall, 2), round(1000.0 * wall / float(total_n), 3)


def _run_heldout_gvp_eval(
    ckpt_path: Path,
    prop: str,
    targets_ha: np.ndarray,
    gvp_suite: GVPBenchmarkSuite,
    reconciler: ValenceReconciler,
    train_smiles: set[str],
    *,
    cfg_scale: float,
    cfg_tree_decay: float = 0.25,
    geometric_guard: bool = True,
    batch_size: int = 250,
    seed: int,
    device_str: str,
) -> Dict[str, Any]:
    device = torch.device(device_str)
    gen, _ = LoomUni3DARGenerator.load_checkpoint(ckpt_path, device=device_str, batch_size=batch_size)
    targets_gen = _target_ha_to_gen(prop, targets_ha)
    gen_out = _batched_sample_mols(
        gen,
        reconciler,
        targets_gen,
        batch_size=batch_size,
        cfg_scale=cfg_scale,
        cfg_tree_decay=cfg_tree_decay,
        geometric_guard=geometric_guard,
        seed=seed,
        device=device,
        do_warmup=True,
    )
    bs128_mol_s, bs128_ms = _time_k1_inference_bs(
        gen,
        targets_gen,
        batch_size=128,
        n_batches=4,
        cfg_scale=cfg_scale,
        cfg_tree_decay=cfg_tree_decay,
        geometric_guard=geometric_guard,
        seed=seed + 500,
        device=device,
    )
    gvp_raw = gvp_suite.evaluate_mols(prop, gen_out["raw_mols"], targets_ha)
    gvp_rec = gvp_suite.evaluate_mols(prop, gen_out["rec_mols"], targets_ha)

    clean_smis: List[str] = []
    for s in gen_out["smiles_list"]:
        if s:
            m2d = Chem.MolFromSmiles(s)
            if m2d is not None:
                clean_smis.append(Chem.MolToSmiles(Chem.RemoveHs(m2d), isomericSmiles=False))
    uniq_smis = set(clean_smis)
    novel_smis = [s for s in clean_smis if s not in train_smiles]

    del gen
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint": str(ckpt_path),
        "cfg_scale": cfg_scale,
        "cfg_tree_decay": cfg_tree_decay,
        "geometric_guard": geometric_guard,
        "eval_batch_size": batch_size,
        "n_proposals": gen_out["n_proposals"],
        "sampling_wall_seconds": gen_out["sampling_wall_seconds"],
        "sampling_mol_per_sec": gen_out["sampling_mol_per_sec"],
        "sampling_mol_per_sec_bs250": gen_out["sampling_mol_per_sec"],
        "ms_per_mol": gen_out["ms_per_mol"],
        "sampling_mol_per_sec_bs128": bs128_mol_s,
        "ms_per_mol_bs128": bs128_ms,
        "peak_vram_gb": gen_out["peak_vram_gb"],
        "uniqueness_pct": round(100.0 * len(uniq_smis) / max(len(clean_smis), 1), 2),
        "novelty_pct": round(100.0 * len(novel_smis) / max(len(clean_smis), 1), 2),
        "raw": {
            "legal_count": gen_out["raw_legal_count"],
            "legal_pct": gen_out["raw_legal_pct"],
            "exotic_charge_count": gen_out["raw_exotic_count"],
            "exotic_charge_pct": gen_out["raw_exotic_charge_pct"],
            "n_evaluable": gvp_raw["n_evaluable"],
            "mae": round(gvp_raw["mae"], 4),
            "rmse": round(gvp_raw["rmse"], 4),
            "slope": round(gvp_raw["slope"], 4),
            "pearson_r": round(gvp_raw["pearson_r"], 4),
        },
        "reconciled": {
            "legal_count": gen_out["rec_legal_count"],
            "legal_pct": gen_out["rec_legal_pct"],
            "exotic_charge_count": gen_out["rec_exotic_count"],
            "exotic_charge_pct": gen_out["rec_exotic_charge_pct"],
            "fallback_count": gen_out["reconciled_fallback_count"],
            "max_heavy_rmsd_A": gen_out["max_heavy_rmsd_A"],
            "n_evaluable": gvp_rec["n_evaluable"],
            "mae": round(gvp_rec["mae"], 4),
            "rmse": round(gvp_rec["rmse"], 4),
            "slope": round(gvp_rec["slope"], 4),
            "pearson_r": round(gvp_rec["pearson_r"], 4),
        },
    }


# ==============================================================================
# Per-Property K=1 Pipeline (`SFT` -> `Hindsight` -> `TopoGRPO` -> `Blind Eval`)
# ==============================================================================
def _run_sft_loop(
    generator: LoomUni3DARGenerator,
    stream: SinglePropPrefetchedSFTStream,
    *,
    steps: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    contrast_min_gap_gen: float,
    device: torch.device,
    log_tag: str,
) -> Dict[str, Any]:
    trainable = [p for p in generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    def _lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(max(1, warmup_steps))
        prog = float(step_idx - warmup_steps) / float(max(1, steps - warmup_steps))
        ratio = min_lr / lr
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    total_examples = 0
    _sync(device)
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        packed, t_norm_np, mask_np = stream.next_batch()
        t_norm = torch.from_numpy(t_norm_np).to(device)
        active = torch.from_numpy(mask_np).to(device)
        bd = _move_packed_to_device(packed, device)

        generator.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss, info = generator.compute_sft_loss(
                bd,
                t_norm,
                active,
                cfg_dropout_prob=0.12,
                contrast_weight=0.25,
                contrast_margin=0.04,
                contrast_min_gap_eV=contrast_min_gap_gen,
                xyz_loss_weight=0.10,
                valence_loss_weight=0.15,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        total_examples += int(t_norm.shape[0])

        if step == 1 or step % 400 == 0 or step == steps:
            _sync(device)
            elapsed = time.perf_counter() - t0
            rate = float(total_examples) / max(elapsed, 1e-6)
            print(
                f"[{log_tag} {step:04d}/{steps:04d}] lr={scheduler.get_last_lr()[0]:.2e} "
                f"loss={info['loss_total']:.4f} gen={info['loss_gen']:.4f} "
                f"con={info['loss_contrast']:.4f} rate={rate:.1f} mol/s ({elapsed/60:.1f}m)",
                flush=True,
            )

    _sync(device)
    wall_s = time.perf_counter() - t0
    return {
        "steps": steps,
        "examples": total_examples,
        "wall_seconds": round(wall_s, 2),
        "mol_per_sec": round(float(total_examples) / max(wall_s, 1e-6), 2),
    }


def _train_single_property(
    prop: str,
    prop_idx: int,
    splits: Dict[str, Any],
    surrogate: QM9TrainSplitSurrogateOracle,
    reconciler: ValenceReconciler,
    gvp_suite: GVPBenchmarkSuite,
    train_smiles: set[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device_str = args.device
    device = torch.device(device_str)
    prop_seed = args.seed + prop_idx * 1009
    torch.manual_seed(prop_seed)
    np.random.seed(prop_seed)
    random.seed(prop_seed)

    run_dir = args.runs_dir / f"k1_{prop}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"qm9_k1_{prop}.json"

    train_recs = splits["train"]
    dev_recs = splits["dev"]
    train_y = _records_to_gen_targets(train_recs, prop_idx, prop)
    dev_y_all = _records_to_gen_targets(dev_recs, prop_idx, prop)
    prop_mean = float(np.mean(train_y))
    prop_std = float(np.std(train_y))
    dev_targets = dev_y_all[: args.dev_eval_n]

    p01, p99 = np.quantile(train_y, [0.01, 0.99])
    p005, p995 = np.quantile(train_y, [0.005, 0.995])
    contrast_min_gap = max(0.40, 0.65 * prop_std)
    grpo_tol = 0.22 * prop_std

    print(
        f"\n==================== Property {prop} (idx={prop_idx}, unit={PROP_GEN_UNITS[prop]}) ====================\n"
        f"[K1 {prop}] train={len(train_recs)} mean={prop_mean:.4f} std={prop_std:.4f} "
        f"[p01,p99]=[{p01:.3f},{p99:.3f}] contrast_gap={contrast_min_gap:.3f} grpo_tol={grpo_tol:.3f}",
        flush=True,
    )

    # 1. Stage 0: K=1 SFT
    generator = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_checkpoint,
        target_cols=[prop],
        target_units=[PROP_GEN_UNITS[prop]],
        mean=[prop_mean],
        scale=[prop_std],
        allowed_elements=ALLOWED_QM9_ELEMENTS,
        device=device_str,
        batch_size=args.batch_size,
        lora_rank=16,
        lora_alpha=32.0,
        lora_dropout=0.05,
        lora_include_ffn=True,
        lora_include_heads=True,
        lora_include_xyz_heads=True,
        train_xyz_embeddings=True,
        use_valence_head=True,
    )
    n_train_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in generator.parameters())
    print(f"[K1 {prop}] Trainable parameters: {n_train_params:,} / {n_total_params:,}", flush=True)

    sft_stream = SinglePropPrefetchedSFTStream(
        train_recs,
        train_y,
        generator.backbone,
        prop_mean=prop_mean,
        prop_std=prop_std,
        primary_count=len(train_recs),
        extra_mix_prob=0.0,
        tail_power=0.35,
        batch_size=args.batch_size,
        seed=prop_seed,
    )
    try:
        sft_timing = _run_sft_loop(
            generator,
            sft_stream,
            steps=args.sft_steps,
            lr=3.5e-4,
            min_lr=2.5e-5,
            warmup_steps=min(150, args.sft_steps // 10),
            contrast_min_gap_gen=contrast_min_gap,
            device=device,
            log_tag=f"K1-{prop}-SFT",
        )
    finally:
        sft_stream.close()

    sft_ckpt = run_dir / f"k1_{prop}_sft.pt"
    generator.save_checkpoint(sft_ckpt, extra_metadata={"prop": prop, "stage": "sft", "timing": sft_timing})

    # Blind Dev CFG Sweep for Arm A (`K=1 SFT`, matched protocol: cfg_tree_decay=0.25, geometric_guard=True)
    sft_dev_sweep: List[Dict[str, Any]] = []
    for c_scale in (1.5, 2.0, 2.5):
        res_dev = _evaluate_dev_with_surrogate(
            generator,
            surrogate,
            reconciler,
            prop_idx,
            prop,
            prop_std,
            dev_targets,
            batch_size=args.batch_size,
            cfg_scale=c_scale,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            seed=prop_seed + 100 + int(c_scale * 10),
            device=device,
        )
        sft_dev_sweep.append(res_dev)
        print(f"[K1 {prop} SFT Dev] {res_dev}", flush=True)
    best_sft_entry = min(sft_dev_sweep, key=lambda d: d["dev_score"])
    best_sft_cfg = float(best_sft_entry["cfg_scale"])

    # 2. Stage 1 Hindsight Self-Training (zero test leakage: relabeled ONLY by train_rQM_v0 surrogate)
    _sync(device)
    hs_t0 = time.perf_counter()
    rng = np.random.default_rng(prop_seed + 2026)
    hs_targets = rng.uniform(float(p01), float(p99), size=args.hindsight_proposals).astype(np.float32)
    hs_gen_out = _batched_sample_mols(
        generator,
        reconciler,
        hs_targets,
        batch_size=args.batch_size,
        cfg_scale=best_sft_cfg,
        geometric_guard=True,
        seed=prop_seed + 3030,
        device=device,
        do_warmup=False,
    )
    cand_recs = hs_gen_out["raw_clean_records"]
    dedup_recs: List[Dict[str, Any]] = []
    seen_smis: set[str] = set()
    for cr in cand_recs:
        smi = cr["smiles"]
        if smi and smi not in seen_smis:
            seen_smis.add(smi)
            dedup_recs.append(cr)

    hs_mols_list = [r["mol"] for r in dedup_recs]
    hs_preds, _, hs_mask, _ = surrogate.evaluate_candidates(hs_mols_list, batch_size=128)
    hs_dataset_recs: List[Dict[str, Any]] = []
    hs_labels_gen: List[float] = []
    hs_replay_mols: List[Chem.Mol] = []
    hs_replay_vals: List[float] = []

    low_bound = float(p005) - 0.5 * prop_std
    high_bound = float(p995) + 0.5 * prop_std
    for i, rec in enumerate(dedup_recs):
        if not bool(hs_mask[i]):
            continue
        val_gen = float(hs_preds[i, prop_idx])
        if not math.isfinite(val_gen) or val_gen < low_bound or val_gen > high_bound:
            continue
        props_ha = np.zeros(len(PROPS), dtype=np.float32)
        props_ha[prop_idx] = val_gen / float(HA_TO_EV) if prop in EV_PROPS else val_gen
        hs_dataset_recs.append(
            {
                "gdb9_idx": -1,
                "symbols": rec["symbols"],
                "coords": rec["coords"],
                "props_ha": props_ha,
            }
        )
        hs_labels_gen.append(val_gen)
        if len(hs_replay_mols) < 1024:
            hs_replay_mols.append(_symbols_coords_to_conformer_mol(rec["symbols"], rec["coords"]))
            hs_replay_vals.append(val_gen)

    print(
        f"[K1 {prop} Hindsight] generated={args.hindsight_proposals} clean_raw={hs_gen_out['raw_legal_count']} "
        f"unique_clean={len(dedup_recs)} kept_relabeled={len(hs_dataset_recs)}",
        flush=True,
    )

    mixed_recs = list(train_recs) + hs_dataset_recs
    mixed_y = np.concatenate([train_y, np.asarray(hs_labels_gen, dtype=np.float32)], axis=0)
    hs_stream = SinglePropPrefetchedSFTStream(
        mixed_recs,
        mixed_y,
        generator.backbone,
        prop_mean=prop_mean,
        prop_std=prop_std,
        primary_count=len(train_recs),
        extra_mix_prob=0.25 if len(hs_dataset_recs) >= 32 else 0.0,
        tail_power=0.35,
        batch_size=args.batch_size,
        seed=prop_seed + 44,
    )
    try:
        hs_sft_timing = _run_sft_loop(
            generator,
            hs_stream,
            steps=args.hindsight_sft_steps,
            lr=1.0e-4,
            min_lr=2.0e-5,
            warmup_steps=min(30, max(1, args.hindsight_sft_steps // 10)),
            contrast_min_gap_gen=contrast_min_gap,
            device=device,
            log_tag=f"K1-{prop}-HS",
        )
    finally:
        hs_stream.close()

    hs_total_wall = time.perf_counter() - hs_t0
    hs_ckpt = run_dir / f"k1_{prop}_hindsight.pt"
    generator.save_checkpoint(hs_ckpt, extra_metadata={"prop": prop, "stage": "hindsight"})

    # 3. Stage 2.5A TopoGRPO (`topology_only_policy=True`, train-only surrogate reward)
    generator.configure_trainable(train_conditioner=True, train_lora=True)
    n_train_replay = min(1024, len(train_recs))
    tr_replay_idx = rng.choice(len(train_recs), size=n_train_replay, replace=False)
    replay_mols: List[Chem.Mol] = list(hs_replay_mols)
    replay_vals: List[float] = list(hs_replay_vals)
    for ti in tr_replay_idx:
        r = train_recs[int(ti)]
        replay_mols.append(_symbols_coords_to_conformer_mol(r["symbols"], r["coords"]))
        replay_vals.append(float(train_y[int(ti)]))
    replay_targets_arr = np.asarray(replay_vals, dtype=np.float32)[:, None]

    # Build 24 quantile stratified prompt sampler on train [p01, p99]
    q_edges = np.quantile(train_y, np.linspace(0.01, 0.99, 25))

    def _prompt_sampler_fn(step_idx: int, step_rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        p_n = args.grpo_prompts
        half_strat = p_n // 2
        bins = step_rng.integers(0, 24, size=half_strat)
        t_strat = np.asarray([step_rng.uniform(q_edges[b], q_edges[b + 1]) for b in bins], dtype=np.float32)
        t_emp = train_y[step_rng.integers(0, len(train_y), size=p_n - half_strat)].astype(np.float32)
        t_cat = np.concatenate([t_strat, t_emp], axis=0)[:, None]
        step_rng.shuffle(t_cat)
        return t_cat, np.ones((p_n, 1), dtype=bool)

    grpo_trainer = LoomGRPOTrainer(
        generator,
        reward_oracle=surrogate,
        potential_slot=None,
        seen_charge_signatures=None,
        replay_molecules=replay_mols,
        replay_targets_eV=replay_targets_arr,
        lr=3e-5,
        weight_decay=1e-4,
        kl_beta=0.02,
        clip_eps=0.20,
        ppo_epochs=2,
        replay_weight=0.15,
        replay_batch_size=8,
        w_valid=0.75,
        w_phys=0.0,
        w_prop=1.0,
        tolerance_eV=grpo_tol,
        sharp_concentration=False,
        cfg_scale=1.0,
        xyz_cfg_scale=1.0,
        cfg_tree_decay=0.0,
        geometric_guard=True,
        topology_only_policy=True,
        temperatures=(0.85, 0.30, 0.25),
        electron_parity=0,
        seed=prop_seed + 505,
    )

    grpo_best_dev_score = float("inf")
    grpo_best_dev_ckpt = run_dir / f"k1_{prop}_grpo_best_dev.pt"
    grpo_final_ckpt = run_dir / f"k1_{prop}_grpo_final.pt"
    eval_every = max(1, min(40, args.grpo_steps // 2))

    def _grpo_ckpt_cb(step_num: int, cur_gen: LoomUni3DARGenerator, _hist: List[Dict[str, Any]]) -> None:
        nonlocal grpo_best_dev_score
        if step_num % eval_every == 0 or step_num == args.grpo_steps:
            d_res = _evaluate_dev_with_surrogate(
                cur_gen,
                surrogate,
                reconciler,
                prop_idx,
                prop,
                prop_std,
                dev_targets,
                batch_size=args.batch_size,
                cfg_scale=best_sft_cfg,
                geometric_guard=True,
                seed=prop_seed + 9000 + step_num,
                device=device,
            )
            print(f"[K1 {prop} TopoGRPO Dev @ {step_num}] {d_res}", flush=True)
            if d_res["dev_score"] < grpo_best_dev_score:
                grpo_best_dev_score = float(d_res["dev_score"])
                cur_gen.save_checkpoint(grpo_best_dev_ckpt, extra_metadata={"prop": prop, "step": step_num, "dev": d_res})

    _sync(device)
    grpo_t0 = time.perf_counter()
    grpo_hist = grpo_trainer.train_loop(
        args.grpo_steps,
        _prompt_sampler_fn,
        group_size=args.grpo_group_size,
        checkpoint_fn=_grpo_ckpt_cb,
        log_every=20,
    )
    _sync(device)
    grpo_wall = time.perf_counter() - grpo_t0
    generator.save_checkpoint(grpo_final_ckpt, extra_metadata={"prop": prop, "stage": "grpo_final"})
    grpo_rollouts = args.grpo_steps * args.grpo_prompts * args.grpo_group_size

    del generator, grpo_trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Blind Dev Selection across post-SFT candidate checkpoints x cfg_scale for Arm B (`K=1 Full`)
    full_dev_candidates: List[Dict[str, Any]] = []
    cand_ckpts = [
        ("hindsight", hs_ckpt),
        ("grpo_best_dev", grpo_best_dev_ckpt if grpo_best_dev_ckpt.exists() else grpo_final_ckpt),
        ("grpo_final", grpo_final_ckpt),
    ]
    for tag, cpath in cand_ckpts:
        cand_gen, _ = LoomUni3DARGenerator.load_checkpoint(cpath, device=device_str, batch_size=args.batch_size)
        for c_scale in (1.5, 2.0, 2.5):
            d_res = _evaluate_dev_with_surrogate(
                cand_gen,
                surrogate,
                reconciler,
                prop_idx,
                prop,
                prop_std,
                dev_targets,
                batch_size=args.batch_size,
                cfg_scale=c_scale,
                cfg_tree_decay=0.25,
                geometric_guard=True,
                seed=prop_seed + 6100 + int(c_scale * 10),
                device=device,
            )
            d_res["ckpt_tag"] = tag
            d_res["ckpt_path"] = str(cpath)
            full_dev_candidates.append(d_res)
        del cand_gen
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_full_entry = min(full_dev_candidates, key=lambda d: d["dev_score"])
    print(f"[K1 {prop} Blind Dev Best Full] {best_full_entry}", flush=True)

    # 5. One-shot final blind evaluation on N=1,000 PropMolFlow targets via GVPBenchmarkSuite
    targets_ha = _load_propmolflow_eval_targets(args.propmolflow_root, prop, args.eval_n)

    print(f"[K1 {prop}] Blind GVP Eval: Arm A (K=1 SFT, cfg={best_sft_cfg})...", flush=True)
    eval_k1_sft = _run_heldout_gvp_eval(
        sft_ckpt,
        prop,
        targets_ha,
        gvp_suite,
        reconciler,
        train_smiles,
        cfg_scale=best_sft_cfg,
        cfg_tree_decay=0.25,
        geometric_guard=True,
        batch_size=args.batch_size,
        seed=prop_seed + 70000,
        device_str=device_str,
    )

    print(
        f"[K1 {prop}] Blind GVP Eval: Arm B (K=1 Full={best_full_entry['ckpt_tag']}, "
        f"cfg={best_full_entry['cfg_scale']})...",
        flush=True,
    )
    eval_k1_full = _run_heldout_gvp_eval(
        Path(best_full_entry["ckpt_path"]),
        prop,
        targets_ha,
        gvp_suite,
        reconciler,
        train_smiles,
        cfg_scale=float(best_full_entry["cfg_scale"]),
        cfg_tree_decay=0.25,
        geometric_guard=True,
        batch_size=args.batch_size,
        seed=prop_seed + 80000,
        device_str=device_str,
    )
    eval_k1_full["selected_stage"] = best_full_entry["ckpt_tag"]

    total_full_train_wall = sft_timing["wall_seconds"] + hs_total_wall + grpo_wall
    payload = {
        "property": prop,
        "gen_unit": PROP_GEN_UNITS[prop],
        "eval_unit": PROP_EVAL_UNITS[prop],
        "train_stats": {
            "n_train": len(train_recs),
            "mean_gen": round(prop_mean, 6),
            "std_gen": round(prop_std, 6),
            "trainable_params": n_train_params,
            "total_params": n_total_params,
        },
        "training_cost": {
            "sft": sft_timing,
            "hindsight": {
                "proposals": args.hindsight_proposals,
                "clean_raw": hs_gen_out["raw_legal_count"],
                "unique_clean_relabeled": len(hs_dataset_recs),
                "mixed_sft": hs_sft_timing,
                "total_wall_seconds": round(hs_total_wall, 2),
            },
            "topo_grpo": {
                "steps": args.grpo_steps,
                "rollouts": grpo_rollouts,
                "wall_seconds": round(grpo_wall, 2),
                "rollout_mol_per_sec": round(float(grpo_rollouts) / max(grpo_wall, 1e-6), 2),
                "last_step_metrics": {
                    k: round(float(v), 5) if isinstance(v, (int, float)) else v
                    for k, v in (grpo_hist[-1].items() if grpo_hist else [])
                },
            },
            "k1_sft_total_wall_seconds": round(sft_timing["wall_seconds"], 2),
            "k1_full_total_wall_seconds": round(total_full_train_wall, 2),
        },
        "blind_dev_selection": {
            "sft_cfg_sweep": sft_dev_sweep,
            "selected_sft_cfg": best_sft_cfg,
            "full_dev_sweep": full_dev_candidates,
            "selected_full": best_full_entry,
        },
        "k1_sft": eval_k1_sft,
        "k1_full": eval_k1_full,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[K1 {prop} DONE] "
        f"SFT Raw MAE={eval_k1_sft['raw']['mae']} (Legal={eval_k1_sft['raw']['legal_pct']}% ExoQ={eval_k1_sft['raw']['exotic_charge_pct']}%) "
        f"Rec MAE={eval_k1_sft['reconciled']['mae']} ({eval_k1_sft['sampling_mol_per_sec']} mol/s) | "
        f"Full Raw MAE={eval_k1_full['raw']['mae']} (Legal={eval_k1_full['raw']['legal_pct']}% ExoQ={eval_k1_full['raw']['exotic_charge_pct']}%) "
        f"Rec MAE={eval_k1_full['reconciled']['mae']} (Legal={eval_k1_full['reconciled']['legal_pct']}%, {eval_k1_full['sampling_mol_per_sec']} mol/s)",
        flush=True,
    )
    return payload


def run_props_phase(args: argparse.Namespace) -> None:
    qm9_raw_dir = args.propmolflow_root / "data" / "qm9_raw"
    splits = load_disjoint_qm9_splits(args.lmdb_dir, qm9_raw_dir, dev_size=1500, holdout_size=1000, seed=args.seed)
    assert int(splits["audit"]["overlap_train_with_gvp_rQM_v0"]) == 0

    surrogate = QM9TrainSplitSurrogateOracle(args.surrogate_ckpt, device=args.device)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    gvp_suite = GVPBenchmarkSuite(args.propmolflow_root, device=args.device)

    # Precompute train canonical SMILES for novelty computation
    print("[K1] Precomputing train canonical SMILES set for novelty audit...", flush=True)
    train_smiles: set[str] = set()
    for r in splits["train"]:
        res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
        if res.smiles:
            m2d = Chem.MolFromSmiles(res.smiles)
            if m2d is not None:
                train_smiles.add(Chem.MolToSmiles(Chem.RemoveHs(m2d), isomericSmiles=False))
    print(f"[K1] Indexed {len(train_smiles)} unique canonical train SMILES.", flush=True)

    for prop in args.props:
        if prop not in PROPS:
            raise ValueError(f"Unknown property '{prop}'; expected one of {PROPS}")
        prop_idx = PROPS.index(prop)
        _train_single_property(prop, prop_idx, splits, surrogate, reconciler, gvp_suite, train_smiles, args)


def _time_k6_inference_bs128(
    k6_ckpt: Path,
    targets_alpha_gen: np.ndarray,
    *,
    device_str: str,
    seed: int,
) -> Dict[str, Any]:
    """Measure uncontended warm K=6 generation throughput at batch_size=128."""
    device = torch.device(device_str)
    bs = 128
    gen, _ = LoomUni3DARGenerator.load_checkpoint(k6_ckpt, device=device_str, batch_size=bs)
    gen.eval()
    k_cols = list(gen.target_cols)
    alpha_idx = k_cols.index("alpha")

    def _make_batch(slice_vals: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        mat = np.zeros((len(slice_vals), len(k_cols)), dtype=np.float32)
        mat[:, alpha_idx] = slice_vals.astype(np.float32)
        t_norm = gen.normalize_targets(mat)
        m_act = torch.zeros((len(slice_vals), len(k_cols)), dtype=torch.bool, device=device)
        m_act[:, alpha_idx] = True
        return t_norm, m_act

    w_t, w_m = _make_batch(targets_alpha_gen[:bs])
    for w_i in range(2):
        _ = gen.sample(
            w_t,
            w_m,
            cfg_scale=2.0,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed + 100 + w_i,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
        )
    _sync(device)
    t0 = time.perf_counter()
    total_n = 0
    for b_i in range(4):
        s = (b_i * bs) % max(len(targets_alpha_gen) - bs + 1, 1)
        b_t, b_m = _make_batch(targets_alpha_gen[s : s + bs])
        _ = gen.sample(
            b_t,
            b_m,
            cfg_scale=2.0,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed + 200 + b_i,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
        )
        total_n += int(b_t.shape[0])
    _sync(device)
    wall = max(time.perf_counter() - t0, 1e-6)
    del gen
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "checkpoint": str(k6_ckpt),
        "batch_size": bs,
        "n_timed": total_n,
        "wall_seconds": round(wall, 3),
        "mol_per_sec_bs128": round(float(total_n) / wall, 2),
        "ms_per_mol_bs128": round(1000.0 * wall / float(total_n), 3),
    }


def reeval_saved_phase(args: argparse.Namespace) -> None:
    """Re-evaluate saved K=1 checkpoints uncontended with matched decode protocol & measure K=6 bs=128 speed."""
    device = torch.device(args.device)
    qm9_raw_dir = args.propmolflow_root / "data" / "qm9_raw"
    splits = load_disjoint_qm9_splits(args.lmdb_dir, qm9_raw_dir, dev_size=1500, holdout_size=1000, seed=args.seed)
    assert int(splits["audit"]["overlap_train_with_gvp_rQM_v0"]) == 0

    surrogate = QM9TrainSplitSurrogateOracle(args.surrogate_ckpt, device=args.device)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    gvp_suite = GVPBenchmarkSuite(args.propmolflow_root, device=args.device)

    print("[Reeval] Precomputing train canonical SMILES set for novelty audit...", flush=True)
    train_smiles: set[str] = set()
    for r in splits["train"]:
        res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
        if res.smiles:
            m2d = Chem.MolFromSmiles(res.smiles)
            if m2d is not None:
                train_smiles.add(Chem.MolToSmiles(Chem.RemoveHs(m2d), isomericSmiles=False))
    print(f"[Reeval] Indexed {len(train_smiles)} unique canonical train SMILES.", flush=True)

    if args.k6_checkpoint.exists():
        print(f"[Reeval] Measuring uncontended K=6 bs=128 inference speed on {args.k6_checkpoint}...", flush=True)
        alpha_targets_ha = _load_propmolflow_eval_targets(args.propmolflow_root, "alpha", args.eval_n)
        k6_bs128_res = _time_k6_inference_bs128(
            args.k6_checkpoint,
            _target_ha_to_gen("alpha", alpha_targets_ha),
            device_str=args.device,
            seed=args.seed + 9001,
        )
        k6_bs128_path = args.results_dir / "qm9_k6_speed_bs128.json"
        k6_bs128_path.parent.mkdir(parents=True, exist_ok=True)
        k6_bs128_path.write_text(json.dumps(k6_bs128_res, indent=2) + "\n")
        print(f"[Reeval] K=6 bs=128 speed: {k6_bs128_res}", flush=True)
    else:
        print(f"[Reeval] WARNING: K=6 checkpoint not found at {args.k6_checkpoint}; skipping bs=128 timing.", flush=True)

    dev_bs = int(args.reeval_batch_size)
    dev_n = int(args.reeval_dev_n)

    for prop in args.props:
        if prop not in PROPS:
            raise ValueError(f"Unknown property '{prop}'; expected one of {PROPS}")
        prop_idx = PROPS.index(prop)
        prop_seed = args.seed + prop_idx * 1009
        run_dir = args.runs_dir / f"k1_{prop}"
        out_json = args.results_dir / f"qm9_k1_{prop}.json"
        if not out_json.exists():
            raise FileNotFoundError(f"Expected existing result JSON at {out_json} before reeval_saved")

        existing = json.loads(out_json.read_text())
        train_y = _records_to_gen_targets(splits["train"], prop_idx, prop)
        dev_y_all = _records_to_gen_targets(splits["dev"], prop_idx, prop)
        prop_std = float(np.std(train_y))
        dev_targets = dev_y_all[:dev_n]

        sft_ckpt = run_dir / f"k1_{prop}_sft.pt"
        hs_ckpt = run_dir / f"k1_{prop}_hindsight.pt"
        grpo_best_dev_ckpt = run_dir / f"k1_{prop}_grpo_best_dev.pt"
        grpo_final_ckpt = run_dir / f"k1_{prop}_grpo_final.pt"
        for req_p in (sft_ckpt, hs_ckpt, grpo_final_ckpt):
            if not req_p.exists():
                raise FileNotFoundError(f"Missing saved checkpoint: {req_p}")

        print(f"\n[Reeval {prop}] Running blind surrogate dev sweep (N={dev_n}, bs={dev_bs}, cfg_tree_decay=0.25, guard=True)...", flush=True)

        # Arm A (`K=1 SFT`) blind dev sweep
        sft_gen, _ = LoomUni3DARGenerator.load_checkpoint(sft_ckpt, device=args.device, batch_size=dev_bs)
        sft_dev_sweep: List[Dict[str, Any]] = []
        for c_scale in (1.5, 2.0, 2.5):
            res_dev = _evaluate_dev_with_surrogate(
                sft_gen,
                surrogate,
                reconciler,
                prop_idx,
                prop,
                prop_std,
                dev_targets,
                batch_size=dev_bs,
                cfg_scale=c_scale,
                cfg_tree_decay=0.25,
                geometric_guard=True,
                seed=prop_seed + 100 + int(c_scale * 10),
                device=device,
            )
            sft_dev_sweep.append(res_dev)
            print(f"[Reeval {prop} SFT Dev] {res_dev}", flush=True)
        del sft_gen
        if device.type == "cuda":
            torch.cuda.empty_cache()

        best_sft_entry = min(sft_dev_sweep, key=lambda d: d["dev_score"])
        best_sft_cfg = float(best_sft_entry["cfg_scale"])

        # Arm B (`K=1 Full`): restricted strictly to post-SFT stages
        full_dev_candidates: List[Dict[str, Any]] = []
        cand_ckpts = [
            ("hindsight", hs_ckpt),
            ("grpo_best_dev", grpo_best_dev_ckpt if grpo_best_dev_ckpt.exists() else grpo_final_ckpt),
            ("grpo_final", grpo_final_ckpt),
        ]
        for tag, cpath in cand_ckpts:
            cand_gen, _ = LoomUni3DARGenerator.load_checkpoint(cpath, device=args.device, batch_size=dev_bs)
            for c_scale in (1.5, 2.0, 2.5):
                d_res = _evaluate_dev_with_surrogate(
                    cand_gen,
                    surrogate,
                    reconciler,
                    prop_idx,
                    prop,
                    prop_std,
                    dev_targets,
                    batch_size=dev_bs,
                    cfg_scale=c_scale,
                    cfg_tree_decay=0.25,
                    geometric_guard=True,
                    seed=prop_seed + 6100 + int(c_scale * 10),
                    device=device,
                )
                d_res["ckpt_tag"] = tag
                d_res["ckpt_path"] = str(cpath)
                full_dev_candidates.append(d_res)
                print(f"[Reeval {prop} Full Dev ({tag})] {d_res}", flush=True)
            del cand_gen
            if device.type == "cuda":
                torch.cuda.empty_cache()

        best_full_entry = min(full_dev_candidates, key=lambda d: d["dev_score"])
        print(f"[Reeval {prop}] Selected Arm A cfg={best_sft_cfg} | Arm B={best_full_entry['ckpt_tag']} cfg={best_full_entry['cfg_scale']}", flush=True)

        targets_ha = _load_propmolflow_eval_targets(args.propmolflow_root, prop, args.eval_n)
        eval_k1_sft = _run_heldout_gvp_eval(
            sft_ckpt,
            prop,
            targets_ha,
            gvp_suite,
            reconciler,
            train_smiles,
            cfg_scale=best_sft_cfg,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            batch_size=dev_bs,
            seed=prop_seed + 70000,
            device_str=args.device,
        )
        eval_k1_full = _run_heldout_gvp_eval(
            Path(best_full_entry["ckpt_path"]),
            prop,
            targets_ha,
            gvp_suite,
            reconciler,
            train_smiles,
            cfg_scale=float(best_full_entry["cfg_scale"]),
            cfg_tree_decay=0.25,
            geometric_guard=True,
            batch_size=dev_bs,
            seed=prop_seed + 80000,
            device_str=args.device,
        )
        eval_k1_full["selected_stage"] = best_full_entry["ckpt_tag"]

        existing["evaluation_protocol"] = {
            "uncontended_gpu": args.device,
            "dev_eval_n": dev_n,
            "eval_n": int(args.eval_n),
            "eval_batch_size": dev_bs,
            "speed_also_measured_bs128": 128,
            "cfg_tree_decay": 0.25,
            "geometric_guard": True,
            "arm_b_candidates": ["hindsight", "grpo_best_dev", "grpo_final"],
        }
        existing["blind_dev_selection"] = {
            "sft_cfg_sweep": sft_dev_sweep,
            "selected_sft_cfg": best_sft_cfg,
            "full_dev_sweep": full_dev_candidates,
            "selected_full": best_full_entry,
        }
        existing["k1_sft"] = eval_k1_sft
        existing["k1_full"] = eval_k1_full
        out_json.write_text(json.dumps(existing, indent=2) + "\n")
        print(
            f"[Reeval {prop} DONE] "
            f"SFT Raw MAE={eval_k1_sft['raw']['mae']} (Legal={eval_k1_sft['raw']['legal_pct']}% ExoQ={eval_k1_sft['raw']['exotic_charge_pct']}%) "
            f"Rec MAE={eval_k1_sft['reconciled']['mae']} (bs250={eval_k1_sft['sampling_mol_per_sec']} bs128={eval_k1_sft['sampling_mol_per_sec_bs128']} mol/s) | "
            f"Full({eval_k1_full['selected_stage']}) Raw MAE={eval_k1_full['raw']['mae']} (Legal={eval_k1_full['raw']['legal_pct']}% ExoQ={eval_k1_full['raw']['exotic_charge_pct']}%) "
            f"Rec MAE={eval_k1_full['reconciled']['mae']} (Legal={eval_k1_full['reconciled']['legal_pct']}%, bs250={eval_k1_full['sampling_mol_per_sec']} bs128={eval_k1_full['sampling_mol_per_sec_bs128']} mol/s)",
            flush=True,
        )


def consolidate_phase(args: argparse.Namespace) -> None:
    k6_json_path = args.results_dir / "qm9_conditional_benchmark.json"
    k6_bs128_path = args.results_dir / "qm9_k6_speed_bs128.json"
    pmf_json_path = args.results_dir / "qm9_propmolflow_audit.json"
    sur_json_path = args.results_dir / "qm9_surrogate_audit.json"

    k6_data = json.loads(k6_json_path.read_text()) if k6_json_path.exists() else {}
    k6_bs128_data = json.loads(k6_bs128_path.read_text()) if k6_bs128_path.exists() else {}
    pmf_data = json.loads(pmf_json_path.read_text()) if pmf_json_path.exists() else {}
    sur_data = json.loads(sur_json_path.read_text()) if sur_json_path.exists() else {}

    per_prop_k1: Dict[str, Any] = {}
    for prop in PROPS:
        p_path = args.results_dir / f"qm9_k1_{prop}.json"
        if not p_path.exists():
            raise FileNotFoundError(f"Missing per-property result JSON: {p_path}")
        per_prop_k1[prop] = json.loads(p_path.read_text())

    # Aggregate speeds & training cost summaries
    k6_per_prop = k6_data.get("uniloom_results", {}).get("per_property", {})
    k6_speeds = [float(k6_per_prop[p]["sampling_mol_per_sec"]) for p in PROPS if p in k6_per_prop]
    k6_raw_legal = [float(k6_per_prop[p]["raw"]["legal_pct"]) for p in PROPS if p in k6_per_prop]
    k6_raw_exoq = [float(k6_per_prop[p]["raw"]["exotic_charge_pct_of_legal"]) for p in PROPS if p in k6_per_prop]
    k6_rec_legal = [float(k6_per_prop[p]["reconciled"]["legal_pct"]) for p in PROPS if p in k6_per_prop]
    k6_train_s = float(k6_data.get("train_wall_seconds", 0.0))
    k6_steps = int(k6_data.get("train_steps", 6000))
    k6_bs = int(k6_data.get("batch_size", 128))

    k1_sft_train_s = sum(per_prop_k1[p]["training_cost"]["k1_sft_total_wall_seconds"] for p in PROPS)
    k1_full_train_s = sum(per_prop_k1[p]["training_cost"]["k1_full_total_wall_seconds"] for p in PROPS)
    k1_sft_infer_bs250 = [float(per_prop_k1[p]["k1_sft"]["sampling_mol_per_sec"]) for p in PROPS]
    k1_full_infer_bs250 = [float(per_prop_k1[p]["k1_full"]["sampling_mol_per_sec"]) for p in PROPS]
    k1_sft_infer_bs128 = [
        float(per_prop_k1[p]["k1_sft"]["sampling_mol_per_sec_bs128"])
        for p in PROPS
        if "sampling_mol_per_sec_bs128" in per_prop_k1[p]["k1_sft"]
    ]
    k1_full_infer_bs128 = [
        float(per_prop_k1[p]["k1_full"]["sampling_mol_per_sec_bs128"])
        for p in PROPS
        if "sampling_mol_per_sec_bs128" in per_prop_k1[p]["k1_full"]
    ]
    k1_sft_train_rates = [float(per_prop_k1[p]["training_cost"]["sft"]["mol_per_sec"]) for p in PROPS]
    k1_hs_sft_train_rates = [float(per_prop_k1[p]["training_cost"]["hindsight"]["mixed_sft"]["mol_per_sec"]) for p in PROPS]

    summary = {
        "k6_sft": {
            "num_models": 1,
            "train_steps": k6_steps,
            "train_wall_hours": round(k6_train_s / 3600.0, 3),
            "train_mol_per_sec": round(float(k6_steps * k6_bs) / max(k6_train_s, 1e-6), 2) if k6_train_s > 0 else None,
            "infer_bs250_mol_per_sec_range": [min(k6_speeds), max(k6_speeds)] if k6_speeds else None,
            "infer_bs250_mol_per_sec_mean": round(float(np.mean(k6_speeds)), 2) if k6_speeds else None,
            "infer_bs128_mol_per_sec": k6_bs128_data.get("mol_per_sec_bs128"),
            "raw_legal_pct_range": [min(k6_raw_legal), max(k6_raw_legal)] if k6_raw_legal else None,
            "raw_exotic_q_pct_range": [min(k6_raw_exoq), max(k6_raw_exoq)] if k6_raw_exoq else None,
            "rec_legal_pct_range": [min(k6_rec_legal), max(k6_rec_legal)] if k6_rec_legal else None,
        },
        "k1_sft_6models": {
            "num_models": 6,
            "train_steps_total": sum(per_prop_k1[p]["training_cost"]["sft"]["steps"] for p in PROPS),
            "train_wall_hours_total": round(k1_sft_train_s / 3600.0, 3),
            "train_mol_per_sec_mean": round(float(np.mean(k1_sft_train_rates)), 2),
            "infer_bs250_mol_per_sec_range": [min(k1_sft_infer_bs250), max(k1_sft_infer_bs250)],
            "infer_bs250_mol_per_sec_mean": round(float(np.mean(k1_sft_infer_bs250)), 2),
            "infer_bs128_mol_per_sec_range": [min(k1_sft_infer_bs128), max(k1_sft_infer_bs128)] if k1_sft_infer_bs128 else None,
            "infer_bs128_mol_per_sec_mean": round(float(np.mean(k1_sft_infer_bs128)), 2) if k1_sft_infer_bs128 else None,
            "raw_legal_pct_range": [
                min(per_prop_k1[p]["k1_sft"]["raw"]["legal_pct"] for p in PROPS),
                max(per_prop_k1[p]["k1_sft"]["raw"]["legal_pct"] for p in PROPS),
            ],
            "raw_exotic_q_pct_range": [
                min(per_prop_k1[p]["k1_sft"]["raw"]["exotic_charge_pct"] for p in PROPS),
                max(per_prop_k1[p]["k1_sft"]["raw"]["exotic_charge_pct"] for p in PROPS),
            ],
            "rec_legal_pct_range": [
                min(per_prop_k1[p]["k1_sft"]["reconciled"]["legal_pct"] for p in PROPS),
                max(per_prop_k1[p]["k1_sft"]["reconciled"]["legal_pct"] for p in PROPS),
            ],
        },
        "k1_full_6models": {
            "num_models": 6,
            "train_wall_hours_total": round(k1_full_train_s / 3600.0, 3),
            "train_hindsight_sft_mol_per_sec_mean": round(float(np.mean(k1_hs_sft_train_rates)), 2),
            "infer_bs250_mol_per_sec_range": [min(k1_full_infer_bs250), max(k1_full_infer_bs250)],
            "infer_bs250_mol_per_sec_mean": round(float(np.mean(k1_full_infer_bs250)), 2),
            "infer_bs128_mol_per_sec_range": [min(k1_full_infer_bs128), max(k1_full_infer_bs128)] if k1_full_infer_bs128 else None,
            "infer_bs128_mol_per_sec_mean": round(float(np.mean(k1_full_infer_bs128)), 2) if k1_full_infer_bs128 else None,
            "selected_stages": {p: per_prop_k1[p]["k1_full"].get("selected_stage") for p in PROPS},
            "raw_legal_pct_range": [
                min(per_prop_k1[p]["k1_full"]["raw"]["legal_pct"] for p in PROPS),
                max(per_prop_k1[p]["k1_full"]["raw"]["legal_pct"] for p in PROPS),
            ],
            "raw_exotic_q_pct_range": [
                min(per_prop_k1[p]["k1_full"]["raw"]["exotic_charge_pct"] for p in PROPS),
                max(per_prop_k1[p]["k1_full"]["raw"]["exotic_charge_pct"] for p in PROPS),
            ],
            "rec_legal_pct_range": [
                min(per_prop_k1[p]["k1_full"]["reconciled"]["legal_pct"] for p in PROPS),
                max(per_prop_k1[p]["k1_full"]["reconciled"]["legal_pct"] for p in PROPS),
            ],
        },
        "propmolflow_6models_audit": pmf_data.get("summary", {}),
    }

    consolidated = {
        "benchmark": "QM9_50k_50k_Disjoint_Fair_K6_vs_6xK1_With_Speed_And_PropMolFlow_Audit",
        "summary": summary,
        "surrogate_audit": sur_data,
        "k6_speed_bs128": k6_bs128_data,
        "propmolflow_direct_audit": pmf_data,
        "k6_sft_benchmark": k6_data,
        "k1_per_property": per_prop_k1,
    }
    out_v2 = args.results_dir / "qm9_conditional_benchmark_v2.json"
    out_v2.write_text(json.dumps(consolidated, indent=2) + "\n")
    print(f"[Consolidate] Wrote consolidated benchmark JSON to {out_v2}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["train_surrogate", "run_props", "reeval_saved", "consolidate"], required=True)
    p.add_argument("--props", nargs="+", default=list(PROPS))
    p.add_argument("--lmdb-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9_data"))
    p.add_argument("--parent-checkpoint", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9.pt"))
    p.add_argument("--propmolflow-root", type=Path, default=Path("/sjtu-combustion/wjw/chensq/PropMolFlow"))
    p.add_argument(
        "--k6-checkpoint",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_uniloom_k6/qm9_uniloom_k6_best.pt"),
    )
    p.add_argument(
        "--unimol-pretrained",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/molgen3d/pretrained/unimol/mol_pre_all_h_fp16.pt"),
    )
    p.add_argument(
        "--surrogate-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_surrogate/qm9_train_unimol_surrogate.pt"),
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results"),
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=20261115)

    # Surrogate hyperparams
    p.add_argument("--surrogate-epochs", type=int, default=18)
    p.add_argument("--surrogate-batch-size", type=int, default=128)

    # K=1 SFT + Hindsight + TopoGRPO hyperparams
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--sft-steps", type=int, default=2800)
    p.add_argument("--hindsight-proposals", type=int, default=2048)
    p.add_argument("--hindsight-sft-steps", type=int, default=400)
    p.add_argument("--grpo-steps", type=int, default=160)
    p.add_argument("--grpo-prompts", type=int, default=8)
    p.add_argument("--grpo-group-size", type=int, default=8)
    p.add_argument("--dev-eval-n", type=int, default=64)
    p.add_argument("--eval-n", type=int, default=1000)

    # Matched uncontended re-evaluation hyperparams
    p.add_argument("--reeval-dev-n", type=int, default=250)
    p.add_argument("--reeval-batch-size", type=int, default=250)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    if args.phase == "train_surrogate":
        train_surrogate_phase(args)
    elif args.phase == "run_props":
        run_props_phase(args)
    elif args.phase == "reeval_saved":
        reeval_saved_phase(args)
    elif args.phase == "consolidate":
        consolidate_phase(args)


if __name__ == "__main__":
    main()
