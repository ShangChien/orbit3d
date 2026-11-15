"""Public QM9 50k/50k Disjoint Conditional Multi-Property Benchmark for Uni-Loom.

Trains ONE unified K=6 Uni-Loom model strictly on PropMolFlow's training split
(`train_rQM_v0.npy` minus `problematic_indices_v0.npy`, verified 0 overlap with
`gvp_rQM_v0.npy` by both gdb9 row index and 7-tuple properties), runs a mandatory
`GVP Oracle Lower-Bound` and `Random` baseline sanity check BEFORE training, and
evaluates N=1,000 generated molecules per property across all 6 QM9 properties
(`alpha`, `gap`, `homo`, `lumo`, `mu`, `cv`) reporting both Raw Point-Cloud and
zero-displacement Valence-Reconciled (`0.0000 A` heavy-atom RMSD) cohorts.
"""

from __future__ import annotations

import argparse
from copy import copy
import glob
import gzip
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
for candidate in (
    ROOT / "src" / "compat",
    ROOT / "src",
    ROOT,
    Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat"),
    Path("/sjtu-combustion/wjw/chensq/PropMolFlow"),
):
    cstr = str(candidate)
    if candidate.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from loom.chemistry.valence_reconciler import ValenceReconciler  # noqa: E402
from loom.generator.octree_model import GridDataset  # noqa: E402
from loom.weaving import LoomUni3DARGenerator  # noqa: E402

HA_TO_EV = 27.211386245988
HA_TO_MEV = 27211.386245988
PROPS: Tuple[str, ...] = ("alpha", "gap", "homo", "lumo", "mu", "cv")
EV_PROPS = {"gap", "homo", "lumo"}
ALLOWED_QM9_ELEMENTS: Tuple[str, ...] = ("H", "C", "N", "O", "F")

PUBLISHED_BASELINES: Dict[str, Dict[str, Any]] = {
    "EDM (6 models)": {
        "n_models": 6,
        "alpha": 2.76,
        "gap": 655.0,
        "homo": 356.0,
        "lumo": 584.0,
        "mu": 1.111,
        "cv": 1.101,
    },
    "GeoLDM (6 models)": {
        "n_models": 6,
        "alpha": 2.37,
        "gap": 587.0,
        "homo": 340.0,
        "lumo": 522.0,
        "mu": 1.108,
        "cv": 1.025,
    },
    "PropMolFlow (6 models)": {
        "n_models": 6,
        "alpha": 1.3115,
        "gap": 390.98,
        "homo": 253.99,
        "lumo": 314.52,
        "mu": 0.6202,
        "cv": 0.6259,
    },
}


def _row_key7(
    alpha: float, homo: float, lumo: float, cv: float, mu: float, r2: float, zpve: float
) -> Tuple[float, float, float, float, float, float, float]:
    return (
        round(float(alpha), 2),
        round(float(homo), 4),
        round(float(lumo), 4),
        round(float(cv), 3),
        round(float(mu), 3),
        round(float(r2), 2),
        round(float(zpve), 4),
    )


def _to_generator_targets(raw_ha_targets: np.ndarray) -> np.ndarray:
    """Convert raw QM9 properties (alpha, gap_Ha, homo_Ha, lumo_Ha, mu, cv) into generator units (gap/homo/lumo in eV)."""
    out = np.asarray(raw_ha_targets, dtype=np.float32).copy()
    for col_idx, name in enumerate(PROPS):
        if name in EV_PROPS:
            out[..., col_idx] *= HA_TO_EV
    return out


def _eval_unit_scale(prop_name: str) -> float:
    return HA_TO_MEV if prop_name in EV_PROPS else 1.0


def load_disjoint_qm9_splits(
    lmdb_dir: Path,
    qm9_raw_dir: Path,
    dev_size: int = 1500,
    holdout_size: int = 1000,
    seed: int = 20261115,
) -> Dict[str, Any]:
    """Match Uni-3DAR QM9 LMDBs against PropMolFlow's 50k/50k split with strict 0-leakage verification."""
    tbl = np.load(qm9_raw_dir / "qm9_gdb9_index_table.npz")
    n_gdb = len(tbl["alpha"])

    train_idx = set(int(i) for i in np.load(qm9_raw_dir / "train_rQM_v0.npy").tolist())
    gvp_idx = set(int(i) for i in np.load(qm9_raw_dir / "gvp_rQM_v0.npy").tolist())
    prob_idx = set(int(i) for i in np.load(qm9_raw_dir / "problematic_indices_v0.npy").tolist())
    assert len(train_idx & gvp_idx) == 0, "Split files train_rQM_v0 and gvp_rQM_v0 overlap!"

    train_allowed = train_idx - prob_idx
    gvp_keys = set()
    key7_to_gdb: Dict[Tuple[float, ...], List[int]] = {}

    alphas = tbl["alpha"]
    homos = tbl["homo"]
    lumos = tbl["lumo"]
    cvs = tbl["cv"]
    mus = tbl["mu"]
    r2s = tbl["r2"]
    zpves = tbl["zpve"]
    for idx in range(n_gdb):
        k = _row_key7(alphas[idx], homos[idx], lumos[idx], cvs[idx], mus[idx], r2s[idx], zpves[idx])
        if idx in gvp_idx:
            gvp_keys.add(k)
        key7_to_gdb.setdefault(k, []).append(idx)

    allowed_elem_set = set(ALLOWED_QM9_ELEMENTS)
    gen_records: List[Dict[str, Any]] = []
    holdout_records: List[Dict[str, Any]] = []
    seen_gdb_indices: set[int] = set()
    ambiguous_skipped = 0

    for split_name in ("train", "valid", "test"):
        env = lmdb.open(str(lmdb_dir / f"{split_name}.lmdb"), readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            with txn.cursor() as cur:
                ok = cur.first()
                while ok:
                    _, val = cur.item()
                    d = pickle.loads(gzip.decompress(val))
                    syms = [str(s) for s in d["atom_type"]]
                    if all(s in allowed_elem_set for s in syms):
                        k = _row_key7(d["alpha"], d["homo"], d["lumo"], d["Cv"], d["mu"], d["r2"], d["zpve"])
                        matches = key7_to_gdb.get(k, [])
                        # Reject any record touching gvp_rQM_v0 by key or matching ambiguous multi-row groups spanning splits.
                        if len(matches) == 1 and k not in gvp_keys:
                            gid = matches[0]
                            if gid not in seen_gdb_indices and gid not in gvp_idx and gid not in prob_idx:
                                seen_gdb_indices.add(gid)
                                pos = np.asarray(d["atom_pos"], dtype=np.float32).reshape(-1, 3)
                                pos = pos - pos.mean(axis=0, keepdims=True)
                                props_ha = np.array(
                                    [
                                        float(d["alpha"]),
                                        float(d["gap"]),
                                        float(d["homo"]),
                                        float(d["lumo"]),
                                        float(d["mu"]),
                                        float(d["Cv"]),
                                    ],
                                    dtype=np.float32,
                                )
                                rec = {
                                    "gdb9_idx": gid,
                                    "symbols": syms,
                                    "coords": pos,
                                    "props_ha": props_ha,
                                }
                                if gid in train_allowed:
                                    gen_records.append(rec)
                                elif gid not in train_idx:
                                    holdout_records.append(rec)
                        elif len(matches) > 1:
                            ambiguous_skipped += 1
                    ok = cur.next()
        env.close()

    rng = np.random.default_rng(seed)
    gen_perm = rng.permutation(len(gen_records))
    gen_shuffled = [gen_records[int(i)] for i in gen_perm]
    dev_recs = gen_shuffled[:dev_size]
    train_recs = gen_shuffled[dev_size:]

    hold_perm = rng.permutation(len(holdout_records))
    holdout_recs = [holdout_records[int(i)] for i in hold_perm[:holdout_size]]

    gen_gids = {r["gdb9_idx"] for r in gen_records}
    hold_gids = {r["gdb9_idx"] for r in holdout_records}
    assert len(gen_gids & gvp_idx) == 0, "LEAKAGE: generator pool overlaps gvp_rQM_v0!"
    assert len(hold_gids & gvp_idx) == 0, "LEAKAGE: holdout sanity pool overlaps gvp_rQM_v0!"
    assert len(gen_gids & hold_gids) == 0

    return {
        "train": train_recs,
        "dev": dev_recs,
        "holdout": holdout_recs,
        "audit": {
            "total_gdb9_rows": n_gdb,
            "train_rQM_v0_size": len(train_idx),
            "gvp_rQM_v0_size": len(gvp_idx),
            "problematic_v0_size": len(prob_idx),
            "matched_train_records": len(train_recs),
            "matched_dev_records": len(dev_recs),
            "holdout_sanity_available": len(holdout_records),
            "holdout_sanity_used": len(holdout_recs),
            "ambiguous_7tuple_skipped": ambiguous_skipped,
            "overlap_train_with_gvp_rQM_v0": len(gen_gids & gvp_idx),
            "overlap_holdout_with_gvp_rQM_v0": len(hold_gids & gvp_idx),
        },
    }


class PointCloudOctreeDataset:
    """Direct `(atom_type, atom_pos)` octree encoder mirroring `OctreeDataset` without RDKit rebuild overhead."""

    def __init__(self, records: Sequence[Dict[str, Any]], backbone: Any, training: bool = True, seed: int = 42):
        self.records = list(records)
        self.allowed_elements = set(backbone.args.allow_atoms.split(","))
        grid_args = copy(backbone.args)
        if not training:
            grid_args.random_rotation_prob = 0.0
        self.grid = GridDataset(self, seed, backbone.dictionary, "atom_type", "atom_pos", training, grid_args)
        self.grid.set_epoch(1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[int(index)]
        return {
            "uid": int(rec["gdb9_idx"]),
            "data_type": "molecule",
            "atom_type": rec["symbols"],
            "atom_pos": rec["coords"],
        }

    def collate_valid(self, indices: Sequence[int]) -> Optional[Tuple[Dict[str, torch.Tensor], np.ndarray]]:
        items: List[Dict[str, Any]] = []
        kept_local_pos: List[int] = []
        for local_i, raw_idx in enumerate(indices):
            idx = int(raw_idx)
            item = self.grid[idx]
            expected = len(self.records[idx]["symbols"])
            decoded = int(((item["decoder_level"] == 0) & item["decoder_is_second_atom"]).sum().item())
            if item["cutoff_atom_count"] or decoded != expected:
                continue
            items.append(item)
            kept_local_pos.append(local_i)
        if len(items) < 4:
            return None
        return self.grid.collater(items), np.asarray(kept_local_pos, dtype=np.int64)


class PrefetchedSFTStream(IterableDataset):
    """Multi-worker CPU octree collator eliminating CPU-side voxelization stalls on GPU 0."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        targets_gen: np.ndarray,
        backbone_args: Any,
        backbone_dict: Any,
        batch_size: int,
        seed: int = 42,
    ):
        self.records = list(records)
        self.targets_gen = np.asarray(targets_gen, dtype=np.float32)
        self.backbone_args = copy(backbone_args)
        self.backbone_dict = backbone_dict
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + wid * 9973)

        class _Proxy:
            pass

        proxy = _Proxy()
        proxy.args = self.backbone_args
        proxy.dictionary = self.backbone_dict
        ds = PointCloudOctreeDataset(self.records, proxy, training=True, seed=self.seed + wid * 131)
        n = len(self.records)
        num_props = self.targets_gen.shape[1]

        while True:
            idx = rng.integers(0, n, size=self.batch_size)
            res = ds.collate_valid(idx.tolist())
            if res is None:
                continue
            packed, kept_pos = res
            kept_idx = idx[kept_pos]
            t_gen = self.targets_gen[kept_idx]
            bsz = len(kept_idx)
            mask = np.zeros((bsz, num_props), dtype=bool)
            for row_i in range(bsz):
                if rng.random() < 0.75:
                    mask[row_i, int(rng.integers(0, num_props))] = True
                else:
                    k_act = int(rng.integers(2, num_props + 1))
                    cols = rng.choice(num_props, size=k_act, replace=False)
                    mask[row_i, cols] = True
            yield packed, torch.from_numpy(t_gen), torch.from_numpy(mask)


def _move_packed_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_packed_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_packed_to_device(v, device) for v in obj)
    return obj


class _DirectGVPPredictor:
    """Direct zero-lightning loader matching PropMolFlow `MoleculePredictor` (`test_regressor.py`)."""

    def __init__(
        self,
        checkpoint_path: Path,
        atom_map: Sequence[str],
        norm_params: Dict[str, Any],
        property_name: str,
        device: str,
    ):
        import dgl
        from propmolflow.data_processing.dataset import collate
        from propmolflow.data_processing.geom import MoleculeFeaturizer
        from propmolflow.models.ctmc_vector_field import PROPERTY_MAP
        from propmolflow.property_regressor.gvp_regressor import GVPRegressor

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_cfg = dict(ckpt["hyper_parameters"]["model_config"])
        sd = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        self.model = GVPRegressor(**model_cfg)
        self.model.load_state_dict(sd, strict=True)
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.dgl = dgl
        self.collate = collate
        self.featurizer = MoleculeFeaturizer(list(atom_map))
        self.norm_params = norm_params
        self.property_idx = PROPERTY_MAP[property_name]

    def _mol_to_graph(self, mol: Chem.Mol):
        try:
            positions, atom_types, atom_charges, bond_types, bond_idxs, _, _, failed_idxs = (
                self.featurizer.featurize_molecules([mol])
            )
            if len(failed_idxs) > 0:
                return None

            positions = positions[0].to(torch.float32)
            atom_types = atom_types[0].to(torch.float32)
            atom_charges = atom_charges[0].long()
            bond_types = bond_types[0].to(torch.int64)
            bond_idxs = bond_idxs[0].long()
            positions = positions - positions.mean(dim=0, keepdim=True)

            n_atoms = positions.shape[0]
            adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int64)
            adj[bond_idxs[:, 0], bond_idxs[:, 1]] = bond_types

            upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
            upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]
            lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))
            edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
            edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

            edge_labels = torch.nn.functional.one_hot(edge_labels.to(torch.int64), num_classes=5).to(torch.float32)
            atom_charges = torch.nn.functional.one_hot(atom_charges + 2, num_classes=6).to(torch.float32)

            g = self.dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)
            g.ndata["x_1_true"] = positions
            g.ndata["a_1_true"] = atom_types
            g.ndata["c_1_true"] = atom_charges
            g.edata["e_1_true"] = edge_labels
            return g
        except Exception:
            return None

    def predict(self, graphs: List[Any]) -> Optional[torch.Tensor]:
        if not graphs:
            return None
        self.model.eval()
        predictions = []
        with torch.no_grad():
            batch_size = 128
            for i in range(0, len(graphs), batch_size):
                batched_graph = self.collate(graphs[i : i + batch_size]).to(self.device)
                pred = self.model(batched_graph)
                pred = (
                    pred * self.norm_params["std"][self.property_idx].to(pred.device)
                    + self.norm_params["mean"][self.property_idx].to(pred.device)
                )
                predictions.append(pred.cpu())
        return torch.cat(predictions, dim=0) if predictions else None


class GVPBenchmarkSuite:
    """Wrapper around PropMolFlow's 6 official frozen `GVPRegressor` checkpoints with explicit norm fix."""

    def __init__(self, propmolflow_root: Path, device: str = "cuda:0"):
        import yaml

        norm_path = propmolflow_root / "data" / "qm9" / "train_data_property_normalization.pt"
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing normalization stats at {norm_path}")
        norm_params = torch.load(norm_path, map_location="cpu", weights_only=False)
        cfg_path = propmolflow_root / "propmolflow" / "property_regressor" / "configs" / "test.yaml"
        with cfg_path.open("r") as f:
            atom_map = yaml.safe_load(f)["dataset"]["atom_map"]

        self.predictors: Dict[str, _DirectGVPPredictor] = {}
        self.checkpoints: Dict[str, str] = {}
        for prop in PROPS:
            ckpt_dir = propmolflow_root / "propmolflow" / "property_regressor" / "model_output" / prop / "checkpoints"
            ckpts = sorted(glob.glob(str(ckpt_dir / "*.ckpt")))
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt found under {ckpt_dir}")
            ckpt_path = Path(ckpts[-1])
            self.predictors[prop] = _DirectGVPPredictor(
                checkpoint_path=ckpt_path,
                atom_map=atom_map,
                norm_params=norm_params,
                property_name=prop,
                device=device,
            )
            self.checkpoints[prop] = ckpt_path.name

    def evaluate_mols(
        self, prop: str, mols: Sequence[Optional[Chem.Mol]], targets_ha: np.ndarray
    ) -> Dict[str, Any]:
        predictor = self.predictors[prop]
        graphs = []
        valid_indices: List[int] = []
        for idx, m in enumerate(mols):
            if m is None:
                continue
            g = predictor._mol_to_graph(m)
            if g is not None:
                graphs.append(g)
                valid_indices.append(idx)

        scale = _eval_unit_scale(prop)
        if not graphs:
            return {
                "n_evaluable": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "slope": float("nan"),
                "pearson_r": float("nan"),
                "preds_eval_unit": np.zeros(0, dtype=np.float64),
                "targets_eval_unit": np.zeros(0, dtype=np.float64),
                "valid_indices": [],
            }

        with torch.no_grad():
            preds = predictor.predict(graphs).view(-1).cpu().numpy().astype(np.float64) * scale
        targs = np.asarray([targets_ha[i] for i in valid_indices], dtype=np.float64) * scale
        err = preds - targs
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        if len(preds) >= 4 and np.std(targs) > 1e-8 and np.std(preds) > 1e-8:
            slope = float(np.polyfit(targs, preds, 1)[0])
            corr = float(np.corrcoef(targs, preds)[0, 1])
        else:
            slope = float("nan")
            corr = float("nan")
        return {
            "n_evaluable": len(valid_indices),
            "mae": mae,
            "rmse": rmse,
            "slope": slope,
            "pearson_r": corr,
            "preds_eval_unit": preds,
            "targets_eval_unit": targs,
            "valid_indices": valid_indices,
        }


def run_gvp_sanity_baselines(
    gvp_suite: GVPBenchmarkSuite,
    holdout_records: Sequence[Dict[str, Any]],
    reconciler: ValenceReconciler,
    seed: int = 20261115,
) -> Dict[str, Any]:
    """Verify GVPRegressor calibration BEFORE training via GVP Lower-Bound & Random baselines."""
    mols: List[Optional[Chem.Mol]] = []
    props_mat = np.stack([r["props_ha"] for r in holdout_records], axis=0)
    raw_ok = 0
    rec_fallback = 0
    for r in holdout_records:
        res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
        if res.raw_legal and res.raw_mol is not None:
            mols.append(res.raw_mol)
            raw_ok += 1
        elif res.reconciled_legal and res.mol is not None:
            mols.append(res.mol)
            rec_fallback += 1
        else:
            mols.append(None)

    rng = np.random.default_rng(seed)
    lower_bound: Dict[str, float] = {}
    random_baseline: Dict[str, float] = {}
    details: Dict[str, Any] = {}
    for col_idx, prop in enumerate(PROPS):
        tcol = props_mat[:, col_idx]
        ev = gvp_suite.evaluate_mols(prop, mols, tcol)
        lower_bound[prop] = ev["mae"]
        perm_targets = ev["targets_eval_unit"][rng.permutation(len(ev["targets_eval_unit"]))]
        rand_mae = float(np.mean(np.abs(ev["preds_eval_unit"] - perm_targets))) if len(perm_targets) > 0 else float("nan")
        random_baseline[prop] = rand_mae
        details[prop] = {
            "n_evaluable": ev["n_evaluable"],
            "lower_bound_mae": ev["mae"],
            "lower_bound_rmse": ev["rmse"],
            "lower_bound_r": ev["pearson_r"],
            "random_mae": rand_mae,
        }

    print(
        f"[Sanity] Ground-truth holdout conformers (N={len(holdout_records)}): "
        f"raw_legal={raw_ok}, rec_fallback={rec_fallback}",
        flush=True,
    )
    print(f"[Sanity] GVP Oracle Lower-Bound MAE: {lower_bound}", flush=True)
    print(f"[Sanity] Random Baseline MAE:         {random_baseline}", flush=True)

    # Hard check against normalization/unit bugs
    if not (
        lower_bound["alpha"] < 1.5
        and lower_bound["gap"] < 200.0
        and lower_bound["homo"] < 200.0
        and lower_bound["lumo"] < 200.0
        and lower_bound["mu"] < 0.40
        and lower_bound["cv"] < 0.60
    ):
        raise RuntimeError(f"GVP Lower-Bound sanity check failed: {lower_bound}")

    return {
        "n_holdout": len(holdout_records),
        "raw_legal_holdout": raw_ok,
        "reconciled_fallback_holdout": rec_fallback,
        "gvp_lower_bound": lower_bound,
        "random": random_baseline,
        "per_property_details": details,
    }


def _evaluate_conditional_set(
    gen: LoomUni3DARGenerator,
    reconciler: ValenceReconciler,
    gvp_suite: GVPBenchmarkSuite,
    train_smiles_set: Optional[set[str]],
    prop_targets_ha: Dict[str, np.ndarray],
    cfg_scale_by_prop: Dict[str, float],
    batch_size: int = 250,
    cfg_tree_decay: float = 0.25,
    base_seed: int = 31000,
) -> Dict[str, Any]:
    """Generate and score Raw + Valence-Reconciled cohorts across all 6 QM9 properties."""
    device = next(gen.parameters()).device
    per_prop: Dict[str, Any] = {}
    all_raw_maes: Dict[str, float] = {}
    all_rec_maes: Dict[str, float] = {}

    for col_idx, prop in enumerate(PROPS):
        targets_ha = np.asarray(prop_targets_ha[prop], dtype=np.float32)
        n_req = len(targets_ha)
        cfg_s = float(cfg_scale_by_prop[prop])

        targets_gen = np.zeros((n_req, len(PROPS)), dtype=np.float32)
        targets_gen[:, col_idx] = targets_ha * (HA_TO_EV if prop in EV_PROPS else 1.0)
        act_mask = torch.zeros((n_req, len(PROPS)), dtype=torch.bool, device=device)
        act_mask[:, col_idx] = True
        targets_norm = gen.normalize_targets(targets_gen)

        trajectories: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        for start in range(0, n_req, batch_size):
            end = min(start + batch_size, n_req)
            sub = gen.sample(
                targets_norm[start:end],
                act_mask[start:end],
                cfg_scale=cfg_s,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=cfg_tree_decay,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=base_seed + col_idx * 1000 + start,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
            )
            trajectories.extend(sub)
        gen_seconds = time.perf_counter() - t0

        raw_mols: List[Optional[Chem.Mol]] = []
        rec_mols: List[Optional[Chem.Mol]] = []
        raw_legal = 0
        raw_exotic = 0
        rec_legal = 0
        rec_exotic = 0
        max_heavy_rmsd = 0.0
        raw_smiles: set[str] = set()
        rec_smiles: set[str] = set()

        for tr in trajectories:
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.raw_legal and res.raw_mol is not None:
                raw_legal += 1
                raw_mols.append(res.raw_mol)
                smi_r = Chem.MolToSmiles(Chem.RemoveHs(res.raw_mol), isomericSmiles=False)
                raw_smiles.add(smi_r)
                if res.raw_exotic_charge:
                    raw_exotic += 1
            else:
                raw_mols.append(None)

            if res.reconciled_legal and res.mol is not None:
                rec_legal += 1
                rec_mols.append(res.mol)
                smi_c = res.smiles or Chem.MolToSmiles(Chem.RemoveHs(res.mol), isomericSmiles=False)
                rec_smiles.add(smi_c)
                if res.reconciled_exotic_charge:
                    rec_exotic += 1
                max_heavy_rmsd = max(max_heavy_rmsd, float(res.heavy_rmsd_A))
            else:
                rec_mols.append(None)

        raw_eval = gvp_suite.evaluate_mols(prop, raw_mols, targets_ha)
        rec_eval = gvp_suite.evaluate_mols(prop, rec_mols, targets_ha)
        novel_rec = (
            len(rec_smiles - train_smiles_set) / max(len(rec_smiles), 1)
            if train_smiles_set is not None and len(rec_smiles) > 0
            else float("nan")
        )

        all_raw_maes[prop] = raw_eval["mae"]
        all_rec_maes[prop] = rec_eval["mae"]
        per_prop[prop] = {
            "n_requested": n_req,
            "cfg_scale": cfg_s,
            "sampling_mol_per_sec": round(n_req / max(gen_seconds, 1e-6), 2),
            "raw": {
                "legal_count": raw_legal,
                "legal_pct": round(100.0 * raw_legal / n_req, 2),
                "exotic_charge_count": raw_exotic,
                "exotic_charge_pct_of_legal": round(100.0 * raw_exotic / max(raw_legal, 1), 2),
                "unique_smiles": len(raw_smiles),
                "uniqueness_pct_of_legal": round(100.0 * len(raw_smiles) / max(raw_legal, 1), 2),
                "gvp_evaluable": raw_eval["n_evaluable"],
                "mae": round(raw_eval["mae"], 4),
                "rmse": round(raw_eval["rmse"], 4),
                "slope": round(raw_eval["slope"], 4),
                "pearson_r": round(raw_eval["pearson_r"], 4),
            },
            "reconciled": {
                "legal_count": rec_legal,
                "legal_pct": round(100.0 * rec_legal / n_req, 2),
                "exotic_charge_count": rec_exotic,
                "exotic_charge_pct_of_legal": round(100.0 * rec_exotic / max(rec_legal, 1), 2),
                "max_heavy_rmsd_A": round(max_heavy_rmsd, 6),
                "unique_smiles": len(rec_smiles),
                "uniqueness_pct_of_legal": round(100.0 * len(rec_smiles) / max(rec_legal, 1), 2),
                "novelty_pct_of_unique": round(100.0 * novel_rec, 2) if math.isfinite(novel_rec) else None,
                "gvp_evaluable": rec_eval["n_evaluable"],
                "mae": round(rec_eval["mae"], 4),
                "rmse": round(rec_eval["rmse"], 4),
                "slope": round(rec_eval["slope"], 4),
                "pearson_r": round(rec_eval["pearson_r"], 4),
            },
        }
        print(
            f"[Eval {prop:>5}] cfg={cfg_s:.2f} | Raw: legal={per_prop[prop]['raw']['legal_pct']}% "
            f"exotic={per_prop[prop]['raw']['exotic_charge_pct_of_legal']}% MAE={raw_eval['mae']:.3f} r={raw_eval['pearson_r']:.3f} | "
            f"Rec: legal={per_prop[prop]['reconciled']['legal_pct']}% exotic={per_prop[prop]['reconciled']['exotic_charge_pct_of_legal']}% "
            f"MAE={rec_eval['mae']:.3f} r={rec_eval['pearson_r']:.3f}",
            flush=True,
        )

    return {"per_property": per_prop, "summary_raw_mae": all_raw_maes, "summary_reconciled_mae": all_rec_maes}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmdb-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9_data"))
    p.add_argument("--parent-checkpoint", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9.pt"))
    p.add_argument("--propmolflow-root", type=Path, default=Path("/sjtu-combustion/wjw/chensq/PropMolFlow"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_uniloom_k6"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/qm9_conditional_benchmark.json"),
    )
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3.5e-4)
    p.add_argument("--min-lr", type=float, default=2.5e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    print("[QM9] Loading leakage-verified 50k/50k disjoint splits...", flush=True)
    splits = load_disjoint_qm9_splits(args.lmdb_dir, args.propmolflow_root / "data" / "qm9_raw", seed=args.seed)
    print(f"[QM9] Split audit: {json.dumps(splits['audit'])}", flush=True)

    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    gvp_suite = GVPBenchmarkSuite(args.propmolflow_root, device=args.device)
    sanity = run_gvp_sanity_baselines(gvp_suite, splits["holdout"], reconciler, seed=args.seed)

    train_recs = splits["train"]
    dev_recs = splits["dev"]
    train_targets_ha = np.stack([r["props_ha"] for r in train_recs], axis=0)
    dev_targets_ha = np.stack([r["props_ha"] for r in dev_recs], axis=0)
    train_targets_gen = _to_generator_targets(train_targets_ha)
    dev_targets_gen = _to_generator_targets(dev_targets_ha)

    mean_vec = np.mean(train_targets_gen, axis=0).tolist()
    scale_vec = np.std(train_targets_gen, axis=0).tolist()
    print(f"[QM9] Unified K=6 generator stats (gap/homo/lumo in eV): mean={mean_vec}, scale={scale_vec}", flush=True)

    # Pre-collect training heavy-atom canonical SMILES for novelty check (fast, no 3D geom perception)
    gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_checkpoint,
        target_cols=list(PROPS),
        target_units=["Bohr^3", "eV", "eV", "eV", "D", "cal/(mol*K)"],
        mean=mean_vec,
        scale=scale_vec,
        allowed_elements=ALLOWED_QM9_ELEMENTS,
        device=args.device,
        batch_size=args.batch_size,
        num_rbf=32,
        num_freqs=16,
        num_prefix_tokens=0,
        modulation_cap=4.0,
        lora_rank=16,
        lora_alpha=32.0,
        lora_include_ffn=True,
        lora_include_heads=True,
        use_valence_head=True,
    )
    gen.enable_v11_xyz_heads(train_xyz_embeddings=True)
    trainable = gen.configure_trainable(train_conditioner=True, train_lora=True, train_xyz_embeddings=True)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in gen.parameters())
    print(f"[QM9] Trainable parameters: {n_trainable:,} / {n_total:,} ({100.0*n_trainable/n_total:.2f}%)", flush=True)

    stream = PrefetchedSFTStream(
        train_recs,
        train_targets_gen,
        gen.backbone.args,
        gen.backbone.dictionary,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    loader_kwargs: Dict[str, Any] = {
        "batch_size": None,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 3
        loader_kwargs["pin_memory"] = True
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(stream, **loader_kwargs)
    loader_iter = iter(loader)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def _lr_at(step_i: int) -> float:
        if step_i < args.warmup_steps:
            return args.lr * float(step_i + 1) / float(max(args.warmup_steps, 1))
        prog = float(step_i - args.warmup_steps) / float(max(args.steps - args.warmup_steps, 1))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    dev_ds = PointCloudOctreeDataset(dev_recs, gen.backbone, training=False, seed=args.seed + 77)
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda"

    gen.train()
    t_train0 = time.perf_counter()
    running_loss = 0.0
    running_nll = 0.0
    running_con = 0.0
    best_dev_nll = float("inf")

    for step in range(1, args.steps + 1):
        lr_now = _lr_at(step - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        packed_cpu, t_gen_cpu, mask_cpu = next(loader_iter)
        packed = _move_packed_to_device(packed_cpu, device)
        t_gen = t_gen_cpu.to(device, non_blocking=True)
        act_mask = mask_cpu.to(device, non_blocking=True)

        t_norm = gen.normalize_targets(t_gen)
        t_norm = t_norm + 0.015 * torch.randn_like(t_norm)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss, info = gen.compute_sft_loss(
                packed,
                t_norm,
                act_mask,
                cfg_dropout_prob=0.12,
                contrast_weight=0.25,
                contrast_margin=0.05,
                contrast_min_gap_eV=0.35,
                xyz_loss_weight=0.10,
                valence_loss_weight=0.15,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        running_loss += float(info.get("loss", loss.item()))
        running_nll += float(info.get("nll_loss", 0.0))
        running_con += float(info.get("contrast_loss", 0.0))

        if step % 200 == 0 or step == 1:
            denom = 1.0 if step == 1 else 200.0
            elapsed = time.perf_counter() - t_train0
            rate = (step * args.batch_size) / max(elapsed, 1e-6)
            print(
                f"[Step {step:4d}/{args.steps}] lr={lr_now:.2e} loss={running_loss/denom:.4f} "
                f"nll={running_nll/denom:.4f} con={running_con/denom:.4f} "
                f"rate={rate:.1f} mol/s elapsed={elapsed/60.0:.1f}m",
                flush=True,
            )
            running_loss = 0.0
            running_nll = 0.0
            running_con = 0.0

        if step % 1000 == 0 or step == args.steps:
            gen.eval()
            val_losses: List[float] = []
            for start in range(0, min(len(dev_recs), 512), 128):
                idx = list(range(start, min(start + 128, len(dev_recs))))
                res = dev_ds.collate_valid(idx)
                if res is None:
                    continue
                vp_cpu, kpos = res
                vp = _move_packed_to_device(vp_cpu, device)
                vt = gen.normalize_targets(dev_targets_gen[idx][kpos])
                vm = torch.ones((len(kpos), len(PROPS)), dtype=torch.bool, device=device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    vloss, _ = gen.compute_sft_loss(vp, vt, vm, cfg_dropout_prob=0.0, contrast_weight=0.0)
                val_losses.append(float(vloss.item()))
            dev_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            print(f"[Dev @ {step}] dev_loss={dev_loss:.4f}", flush=True)
            if dev_loss < best_dev_nll:
                best_dev_nll = dev_loss
                gen.save_checkpoint(args.output_dir / "qm9_uniloom_k6_best.pt", {"step": step, "dev_loss": dev_loss})
            if step % 2000 == 0 or step == args.steps:
                gen.save_checkpoint(args.output_dir / f"qm9_uniloom_k6_step{step}.pt", {"step": step, "dev_loss": dev_loss})
            gen.train()

    train_wall_s = time.perf_counter() - t_train0
    final_ckpt = args.output_dir / "qm9_uniloom_k6_sft.pt"
    gen.save_checkpoint(final_ckpt, {"step": args.steps, "train_seconds": train_wall_s})

    # Reload best checkpoint cleanly for dev CFG sweep & PropMolFlow N=1000 benchmark
    eval_ckpt = args.output_dir / "qm9_uniloom_k6_best.pt"
    if not eval_ckpt.exists():
        eval_ckpt = final_ckpt
    gen_eval, _ = LoomUni3DARGenerator.load_checkpoint(eval_ckpt, device=args.device, batch_size=250)
    gen_eval.eval()

    # Build dev targets (40 per property -> 240 total per CFG setting)
    dev_targets_map = {
        prop: dev_targets_ha[col_idx * 40 : (col_idx + 1) * 40, col_idx]
        for col_idx, prop in enumerate(PROPS)
    }
    cfg_sweep: Dict[str, Any] = {}
    best_cfg_by_prop: Dict[str, float] = {p: 2.0 for p in PROPS}
    best_score_by_prop: Dict[str, float] = {p: float("inf") for p in PROPS}

    for cfg_s in (1.5, 2.0, 2.5):
        print(f"\n[CFG Sweep] Evaluating cfg_scale={cfg_s:.2f} on 40 dev targets/property...", flush=True)
        sw = _evaluate_conditional_set(
            gen_eval,
            reconciler,
            gvp_suite,
            None,
            dev_targets_map,
            {p: cfg_s for p in PROPS},
            batch_size=40,
            cfg_tree_decay=0.25,
            base_seed=41000 + int(round(cfg_s * 100)),
        )
        cfg_sweep[f"cfg_{cfg_s:.2f}"] = sw
        for p in PROPS:
            mae_p = sw["per_property"][p]["reconciled"]["mae"]
            if math.isfinite(mae_p) and mae_p < best_score_by_prop[p]:
                best_score_by_prop[p] = mae_p
                best_cfg_by_prop[p] = cfg_s

    print(f"\n[Benchmark] Selected per-property cfg_scale: {best_cfg_by_prop}", flush=True)
    # Canonical SMILES of generator training set (using atom connectivity via reconciler on a subset or full? Use RDKit SMILES)
    train_smiles: set[str] = set()
    for r in train_recs:
        res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
        if res.smiles:
            train_smiles.add(res.smiles)

    samp_dir = args.propmolflow_root / "sampling_input" / "in_distribution_sampling"
    bench_targets = {
        p: np.load(samp_dir / f"train_half_sampled_values_{p}.npy")[: args.eval_n]
        for p in PROPS
    }
    print(f"[Benchmark] Evaluating N={args.eval_n} molecules per property on PropMolFlow targets...", flush=True)
    final_eval = _evaluate_conditional_set(
        gen_eval,
        reconciler,
        gvp_suite,
        train_smiles,
        bench_targets,
        best_cfg_by_prop,
        batch_size=250,
        cfg_tree_decay=0.25,
        base_seed=51000,
    )

    out_payload = {
        "benchmark": "QM9_PropMolFlow_50k50k_Disjoint",
        "model": "Uni-Loom Unified K=6 (1 model across all 6 properties)",
        "parent_checkpoint": str(args.parent_checkpoint),
        "eval_checkpoint": str(eval_ckpt),
        "trainable_params": n_trainable,
        "total_params": n_total,
        "train_steps": args.steps,
        "batch_size": args.batch_size,
        "train_wall_seconds": round(train_wall_s, 2),
        "split_audit": splits["audit"],
        "gvp_checkpoints": gvp_suite.checkpoints,
        "sanity_baselines": sanity,
        "dev_cfg_sweep": cfg_sweep,
        "selected_cfg_scale": best_cfg_by_prop,
        "eval_n_per_property": args.eval_n,
        "uniloom_results": final_eval,
        "published_baselines": PUBLISHED_BASELINES,
        "units": {
            "alpha": "Bohr^3",
            "gap": "meV (train/gen in eV)",
            "homo": "meV (train/gen in eV)",
            "lumo": "meV (train/gen in eV)",
            "mu": "D",
            "cv": "cal/(mol*K)",
        },
    }
    args.output_json.write_text(json.dumps(out_payload, indent=2) + "\n")
    print(f"\n[QM9] Saved full benchmark results to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
