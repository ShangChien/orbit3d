"""OE62 Cross-Scale Head-to-Head Benchmark: 12-Element `PropMolFlow-OE62` vs `Uni-Loom`.

Adapts official `PropMolFlow` (`FlowMol` CTMC geometry-complete flow matching) from 5-element QM9
to the 12-element, up-to-80-atom scaffold-disjoint OE62 crystal-forming optoelectronic dataset
(`47,807` training conformers, `HOMO` target conditioning) and benchmarks side-by-side against
`Uni-Loom` (`v2.0` + `OctreeProcessCritic`) under identical blind `DPA-4` & synthesizability audits:

Phases:
  1. `preprocess`:
     Builds compact OE62 12-element flat node/edge tensors, `train_data_marginal_dists.pt`,
     `train_data_n_atoms_histogram.pt`, and `train_data_property_normalization.pt` from
     `oe62_master_train_B.sdf` and `oe62_master_valid_B.sdf`.
  2. `benchmark_speed` (run on uncontended `GPU 0` or `GPU 1`):
     Measures exact single-A800 training throughput (`mol/s`, `sec/step`, peak VRAM GB, OOM flag)
     and 100-step ODE sampling throughput across `batch_size in {16, 32, 64, 128}`.
  3. `train` (supports 1-GPU or `torchrun --nproc_per_node=2` DDP with `--resume`):
     Trains 12-element `PropMolFlow-OE62` (`HOMO`) and saves exact compute-budget checkpoints
     at `--milestones-gpuh` (default: `3.0, 6.0, 12.0` A800 GPU-hours).
  4. `eval_checkpoint` (generation timed on uncontended GPU; blind `DPA-4` + `SA<=3.5` audit):
     Evaluates a saved `PropMolFlow-OE62` checkpoint across `M in {1, 2, 4, 8}` (`1,792` proposals)
     on both `native_graph` and `coord_reconciled` (`PropMolFlow + Uni-Loom CriticBus`) tracks.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch
import torch.distributed as dist
from torch.nn.functional import one_hot
from torch.nn.parallel import DistributedDataParallel as DDP

RDLogger.DisableLog("rdApp.*")
torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[1]
PROPMOLFLOW_DEFAULT = Path("/sjtu-combustion/wjw/chensq/PropMolFlow")
MOLGEN_COMPAT_DEFAULT = Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat")

for cand in (
    ROOT / "src" / "compat",
    ROOT / "src",
    ROOT,
    MOLGEN_COMPAT_DEFAULT,
    PROPMOLFLOW_DEFAULT,
):
    cstr = str(cand)
    if cand.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)


def _install_pytorch_lightning_shim() -> None:
    try:
        import pytorch_lightning  # noqa: F401
        return
    except ImportError:
        pass
    pl = types.ModuleType("pytorch_lightning")

    class _LightningModule(torch.nn.Module):
        def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
            pass

        def log(self, *args: Any, **kwargs: Any) -> None:
            pass

        def log_dict(self, *args: Any, **kwargs: Any) -> None:
            pass

    pl.LightningModule = _LightningModule
    sys.modules["pytorch_lightning"] = pl


_install_pytorch_lightning_shim()

import dgl  # noqa: E402
from propmolflow.data_processing.priors import coupled_node_prior, edge_prior  # noqa: E402
from propmolflow.models.ctmc_vector_field import PROPERTY_MAP  # noqa: E402
from propmolflow.models.flowmol import FlowMol  # noqa: E402

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402

OE62_ATOM_MAP: List[str] = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
OE62_ATOM_SET = frozenset(OE62_ATOM_MAP)
HOMO_PROP_IDX = int(PROPERTY_MAP["homo"])  # 5


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _min_pair_dist(coords: np.ndarray) -> float:
    pts = np.asarray(coords, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 99.0 if pts.shape[0] == 1 else 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dmat, 99.0)
    return float(dmat.min())


def _default_oe62_flowmol_hparams(data_dir: Path) -> Dict[str, Any]:
    """Construct exact 12-element `FlowMol` hparams mirroring PropMolFlow's official `homo` model."""
    return {
        "atom_type_map": list(OE62_ATOM_MAP),
        "n_atoms_hist_file": str(data_dir / "train_data_n_atoms_histogram.pt"),
        "marginal_dists_file": str(data_dir / "train_data_marginal_dists.pt"),
        "n_atom_charges": 6,
        "n_bond_types": 5,
        "sample_interval": 1.0,
        "n_mols_to_sample": 64,
        "time_scaled_loss": False,
        "exclude_charges": False,
        "weight_ae": False,
        "target_blur": 0.0,
        "parameterization": "ctmc",
        "total_loss_weights": {"a": 0.4, "c": 1.0, "e": 2.0, "x": 3.0},
        "lr_scheduler_config": {
            "base_lr": 0.00025,
            "restart_interval": 0,
            "restart_type": "linear",
            "warmup_length": 0.0,
            "weight_decay": 1e-12,
        },
        "interpolant_scheduler_config": {
            "cosine_params": {"a": 2, "c": 2, "e": 1.5, "x": 1},
            "schedule_type": {"a": "linear", "c": "linear", "e": "linear", "x": "linear"},
        },
        "vector_field_config": {
            "convs_per_update": 1,
            "high_confidence_threshold": 0.0,
            "message_norm": 100,
            "n_cp_feats": 4,
            "n_hidden_edge_feats": 128,
            "n_hidden_scalars": 256,
            "n_message_gvps": 3,
            "n_molecule_updates": 8,
            "n_recycles": 1,
            "n_update_gvps": 3,
            "n_vec_channels": 16,
            "rbf_dim": 32,
            "rbf_dmax": 18,
            "separate_mol_updaters": True,
            "stochasticity": 10.0,
            "update_edge_w_distance": True,
        },
        "prior_config": {
            "a": {"align": False, "kwargs": {}, "type": "ctmc"},
            "c": {"align": False, "kwargs": {}, "type": "ctmc"},
            "e": {"align": False, "kwargs": {}, "type": "ctmc"},
            "x": {"align": True, "kwargs": {"std": 1.0}, "type": "centered-normal"},
        },
        "default_n_timesteps": 100,
        "property_embedding_dim": 256,
        "conditional_generation": True,
        "gaussian_expansion": False,
        "gaussian_start": None,
        "gaussian_stop": None,
        "n_gaussians": 5,
        "properties_handle_method": "concatenate",
    }


# ---------------------------------------------------------------------------
# Phase 1: Compact OE62 Preprocessing (`oe62_master_train_B.sdf`)
# ---------------------------------------------------------------------------
def _encode_sdf_split(sdf_path: Path, max_atoms: int = 80) -> Dict[str, Any]:
    atom_to_idx = {s: i for i, s in enumerate(OE62_ATOM_MAP)}
    pos_chunks: List[np.ndarray] = []
    a_chunks: List[np.ndarray] = []
    c_chunks: List[np.ndarray] = []
    bt_chunks: List[np.ndarray] = []
    bi_chunks: List[np.ndarray] = []
    node_ranges: List[Tuple[int, int]] = []
    edge_ranges: List[Tuple[int, int]] = []
    homo_vals: List[float] = []
    n_atoms_list: List[int] = []

    n_cursor = 0
    e_cursor = 0
    skipped = 0

    for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False):
        if mol is None:
            skipped += 1
            continue
        n_a = mol.GetNumAtoms()
        if n_a < 2 or n_a > max_atoms:
            skipped += 1
            continue
        if mol.HasProp("B_homo"):
            h_val = float(mol.GetProp("B_homo"))
        elif mol.HasProp("homo"):
            h_val = float(mol.GetProp("homo"))
        else:
            skipped += 1
            continue
        if not math.isfinite(h_val):
            skipped += 1
            continue

        syms = [a.GetSymbol() for a in mol.GetAtoms()]
        if any(s not in atom_to_idx for s in syms):
            skipped += 1
            continue
        qs = [int(a.GetFormalCharge()) for a in mol.GetAtoms()]
        if any(q < -2 or q > 3 for q in qs):
            skipped += 1
            continue

        coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
        coords = coords - coords.mean(axis=0, keepdims=True)
        a_arr = np.asarray([atom_to_idx[s] for s in syms], dtype=np.int16)
        c_arr = np.asarray([q + 2 for q in qs], dtype=np.int16)

        b_src: List[int] = []
        b_dst: List[int] = []
        b_cls: List[int] = []
        for bond in mol.GetBonds():
            bt = bond.GetBondType()
            if bt == Chem.rdchem.BondType.SINGLE:
                cls_id = 1
            elif bt == Chem.rdchem.BondType.DOUBLE:
                cls_id = 2
            elif bt == Chem.rdchem.BondType.TRIPLE:
                cls_id = 3
            elif bt == Chem.rdchem.BondType.AROMATIC:
                cls_id = 4
            else:
                continue
            u, v = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
            b_src.append(u)
            b_dst.append(v)
            b_cls.append(cls_id)

        n_b = len(b_cls)
        pos_chunks.append(coords)
        a_chunks.append(a_arr)
        c_chunks.append(c_arr)
        if n_b > 0:
            bi_chunks.append(np.stack([b_src, b_dst], axis=1).astype(np.int16))
            bt_chunks.append(np.asarray(b_cls, dtype=np.int8))
        node_ranges.append((n_cursor, n_cursor + n_a))
        edge_ranges.append((e_cursor, e_cursor + n_b))
        n_cursor += n_a
        e_cursor += n_b
        homo_vals.append(h_val)
        n_atoms_list.append(n_a)

    return {
        "positions": torch.from_numpy(np.concatenate(pos_chunks, axis=0)),
        "atom_idx": torch.from_numpy(np.concatenate(a_chunks, axis=0)),
        "charge_idx": torch.from_numpy(np.concatenate(c_chunks, axis=0)),
        "bond_idxs": torch.from_numpy(np.concatenate(bi_chunks, axis=0)) if bi_chunks else torch.zeros((0, 2), dtype=torch.int16),
        "bond_types": torch.from_numpy(np.concatenate(bt_chunks, axis=0)) if bt_chunks else torch.zeros((0,), dtype=torch.int8),
        "node_idx_array": torch.tensor(node_ranges, dtype=torch.int64),
        "edge_idx_array": torch.tensor(edge_ranges, dtype=torch.int64),
        "homo_eV": torch.tensor(homo_vals, dtype=torch.float32),
        "n_atoms": torch.tensor(n_atoms_list, dtype=torch.int32),
        "skipped": skipped,
    }


def run_preprocess(args: argparse.Namespace) -> None:
    data_dir = args.work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    tr = _encode_sdf_split(args.oe62_master_train_sdf, max_atoms=80)
    va = _encode_sdf_split(args.oe62_master_valid_sdf, max_atoms=80)

    torch.save(tr, data_dir / "train_data_processed.pt")
    torch.save(va, data_dir / "val_data_processed.pt")

    # Marginal distributions `(p_a [12], p_c [6], p_e [5], p_c_given_a [12, 6])`
    a_counts = torch.bincount(tr["atom_idx"].long(), minlength=len(OE62_ATOM_MAP)).float() + 1e-3
    p_a = a_counts / a_counts.sum()

    c_counts = torch.bincount(tr["charge_idx"].long(), minlength=6).float() + 1e-3
    p_c = c_counts / c_counts.sum()

    p_c_given_a = torch.zeros((len(OE62_ATOM_MAP), 6), dtype=torch.float32) + 1e-4
    for a_i in range(len(OE62_ATOM_MAP)):
        mask_a = tr["atom_idx"].long() == a_i
        if bool(mask_a.any().item()):
            p_c_given_a[a_i] += torch.bincount(tr["charge_idx"][mask_a].long(), minlength=6).float()
        p_c_given_a[a_i] /= p_c_given_a[a_i].sum()

    n_arr = tr["n_atoms"].long()
    total_upper_pairs = float(((n_arr * (n_arr - 1)) // 2).sum().item())
    bt_counts = torch.bincount(tr["bond_types"].long(), minlength=5).float()
    bt_counts[0] = max(1.0, total_upper_pairs - float(bt_counts[1:].sum().item()))
    p_e = (bt_counts + 1.0) / (bt_counts + 1.0).sum()

    torch.save((p_a, p_c, p_e, p_c_given_a), data_dir / "train_data_marginal_dists.pt")

    uniq_n, cnt_n = torch.unique(tr["n_atoms"].long(), sorted=True, return_counts=True)
    torch.save((uniq_n, cnt_n), data_dir / "train_data_n_atoms_histogram.pt")

    # 19-dim normalization dict compatible with `FlowMol.sample(..., property_name='homo')` (`PROPERTY_MAP['homo']=5`)
    mean_vec = torch.zeros(19, dtype=torch.float32)
    std_vec = torch.ones(19, dtype=torch.float32)
    mean_vec[HOMO_PROP_IDX] = float(tr["homo_eV"].mean().item())
    std_vec[HOMO_PROP_IDX] = float(tr["homo_eV"].std().clamp_min(1e-6).item())
    prop_names = [
        "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
        "u0", "u298", "h298", "g298", "cv", "u0_atom", "u298_atom", "h298_atom", "g298_atom",
    ]
    torch.save({"mean": mean_vec, "std": std_vec, "property_names": prop_names}, data_dir / "train_data_property_normalization.pt")

    meta = {
        "train_molecules": int(len(tr["n_atoms"])),
        "val_molecules": int(len(va["n_atoms"])),
        "train_skipped": int(tr["skipped"]),
        "mean_n_atoms": round(float(tr["n_atoms"].float().mean().item()), 2),
        "median_n_atoms": round(float(tr["n_atoms"].float().median().item()), 1),
        "max_n_atoms": int(tr["n_atoms"].max().item()),
        "mean_complete_edges_per_mol": round(float((n_arr * (n_arr - 1)).float().mean().item()), 1),
        "max_complete_edges_per_mol": int((n_arr * (n_arr - 1)).max().item()),
        "homo_mean_eV": round(float(mean_vec[HOMO_PROP_IDX].item()), 4),
        "homo_std_eV": round(float(std_vec[HOMO_PROP_IDX].item()), 4),
        "preprocess_wall_sec": round(time.perf_counter() - t0, 2),
    }
    (data_dir / "preprocess_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[PropMolFlow-OE62 Preprocess DONE] {meta}", flush=True)


class OE62FlowMolDataset(torch.utils.data.Dataset):
    """On-the-fly complete DGL graph dataset matching `MoleculeDataset.__getitem__` in PropMolFlow."""

    def __init__(self, pt_path: Path, norm_path: Path, prior_config: Dict[str, Any]) -> None:
        super().__init__()
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        norm = torch.load(norm_path, map_location="cpu", weights_only=False)
        self.positions = d["positions"]
        self.atom_idx = d["atom_idx"]
        self.charge_idx = d["charge_idx"]
        self.bond_idxs = d["bond_idxs"]
        self.bond_types = d["bond_types"]
        self.node_idx_array = d["node_idx_array"]
        self.edge_idx_array = d["edge_idx_array"]
        self.n_atoms = d["n_atoms"]
        h_mean = float(norm["mean"][HOMO_PROP_IDX].item())
        h_std = float(norm["std"][HOMO_PROP_IDX].item())
        self.homo_norm = (d["homo_eV"].float() - h_mean) / max(h_std, 1e-6)
        self.prior_config = prior_config
        self.num_atom_types = len(OE62_ATOM_MAP)

    def __len__(self) -> int:
        return int(self.node_idx_array.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        ns, ne = int(self.node_idx_array[idx, 0]), int(self.node_idx_array[idx, 1])
        es, ee = int(self.edge_idx_array[idx, 0]), int(self.edge_idx_array[idx, 1])
        n_a = ne - ns

        pos = self.positions[ns:ne].float()
        pos = pos - pos.mean(dim=0, keepdim=True)
        a_oh = one_hot(self.atom_idx[ns:ne].long(), num_classes=self.num_atom_types).float()
        c_oh = one_hot(self.charge_idx[ns:ne].long(), num_classes=6).float()

        adj = torch.zeros((n_a, n_a), dtype=torch.int64)
        if ee > es:
            bi = self.bond_idxs[es:ee].long()
            bt = self.bond_types[es:ee].long()
            adj[bi[:, 0], bi[:, 1]] = bt
            adj[bi[:, 1], bi[:, 0]] = bt

        upper = torch.triu_indices(n_a, n_a, offset=1)
        u_lbl = adj[upper[0], upper[1]]
        lower = torch.stack((upper[1], upper[0]))
        edges = torch.cat((upper, lower), dim=1)
        e_oh = one_hot(torch.cat((u_lbl, u_lbl)), num_classes=5).float()

        g = dgl.graph((edges[0], edges[1]), num_nodes=n_a)
        g.ndata["x_1_true"] = pos
        g.ndata["a_1_true"] = a_oh
        g.ndata["c_1_true"] = c_oh
        g.edata["e_1_true"] = e_oh

        prior_nodes = coupled_node_prior(dst_dict={"x": pos, "a": a_oh, "c": c_oh}, prior_config=self.prior_config)
        for feat, val in prior_nodes.items():
            g.ndata[f"{feat}_0"] = val
        upper_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        upper_mask[: upper.shape[1]] = True
        g.edata["e_0"] = edge_prior(upper_mask, self.prior_config["e"])

        return g, self.homo_norm[idx]


def _collate_flowmol(items: List[Tuple[Any, torch.Tensor]]) -> Tuple[Any, torch.Tensor]:
    bg = dgl.batch([it[0] for it in items])
    props = torch.stack([it[1] for it in items], dim=0).view(-1)
    return bg, props


# ---------------------------------------------------------------------------
# Phase 2: Clean Single-A800 Speed & Peak VRAM Benchmark (`O(N^2)` vs Octree)
# ---------------------------------------------------------------------------
def run_benchmark_speed(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
        _ = torch.zeros(1, device=device)
    data_dir = args.work_dir / "data"
    hp = _default_oe62_flowmol_hparams(data_dir)
    ds = OE62FlowMolDataset(data_dir / "train_data_processed.pt", data_dir / "train_data_property_normalization.pt", hp["prior_config"])

    # Deterministic representative 256-molecule slice from OE62 training set
    rng = np.random.default_rng(2026)
    sample_idxs = rng.choice(len(ds), size=min(256, len(ds)), replace=False).tolist()
    sample_items = [ds[i] for i in sample_idxs]
    sample_natoms = [int(ds.n_atoms[i].item()) for i in sample_idxs]

    train_bench: Dict[str, Any] = {}
    for bs in (16, 32, 64, 128):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            model = FlowMol(**hp).to(device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=2.5e-4)
            lw = dict(model.total_loss_weights)
            f_ord = list(model.canonical_feat_order)

            batches = []
            for b_i in range(4):
                sub = sample_items[(b_i * bs) % len(sample_items) : ((b_i * bs) % len(sample_items)) + bs]
                if len(sub) < bs:
                    sub = (sub + sample_items)[:bs]
                bg, pr = _collate_flowmol(sub)
                batches.append((bg.to(device), pr.to(device)))

            n_warm, n_timed = 4, 14
            t_s = 0.0
            for st in range(n_warm + n_timed):
                if st == n_warm:
                    _sync(device)
                    t_s = time.perf_counter()
                bg, pr = batches[st % len(batches)]
                bg.prop = pr.clone()
                opt.zero_grad(set_to_none=True)
                losses = model(bg)
                tot = sum(float(lw[f]) * losses[f] for f in f_ord)
                tot.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                opt.step()
            _sync(device)
            el = time.perf_counter() - t_s
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3) if device.type == "cuda" else 0.0
            train_bench[f"bs{bs}"] = {
                "status": "ok",
                "sec_per_step": round(el / n_timed, 4),
                "train_mol_per_sec": round((bs * n_timed) / max(el, 1e-6), 2),
                "peak_vram_gb": round(peak_gb, 3),
                "total_params": sum(p.numel() for p in model.parameters()),
            }
            del model, opt, batches
        except RuntimeError as exc:
            train_bench[f"bs{bs}"] = {"status": "oom_or_error", "error": str(exc)[:200]}
            if device.type == "cuda":
                torch.cuda.empty_cache()

    infer_bench: Dict[str, Any] = {}
    norm_file = str(data_dir / "train_data_property_normalization.pt")
    for bs in (16, 32, 64, 128):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            model = FlowMol(**hp).to(device)
            model.eval()
            # Warmup
            _ = model.sample_random_sizes(
                min(8, bs),
                device=device,
                n_timesteps=10,
                property_name="homo",
                normalization_file_path=norm_file,
                training_mode=False,
                properties_handle_method="concatenate",
                multilple_values_to_one_property=[-5.8] * min(8, bs),
                number_of_atoms=sample_natoms[: min(8, bs)],
            )
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t_i0 = time.perf_counter()
            n_gen = max(bs, 128)
            for s in range(0, n_gen, bs):
                cb = min(bs, n_gen - s)
                _ = model.sample_random_sizes(
                    cb,
                    device=device,
                    n_timesteps=100,
                    property_name="homo",
                    normalization_file_path=norm_file,
                    training_mode=False,
                    properties_handle_method="concatenate",
                    multilple_values_to_one_property=[-5.8] * cb,
                    number_of_atoms=sample_natoms[s : s + cb],
                )
            _sync(device)
            el_i = time.perf_counter() - t_i0
            peak_gb_i = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3) if device.type == "cuda" else 0.0
            infer_bench[f"bs{bs}"] = {
                "status": "ok",
                "n_timesteps": 100,
                "infer_mol_per_sec": round(n_gen / max(el_i, 1e-6), 2),
                "peak_vram_gb": round(peak_gb_i, 3),
            }
            del model
        except RuntimeError as exc:
            infer_bench[f"bs{bs}"] = {"status": "oom_or_error", "error": str(exc)[:200]}
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out = {
        "experiment": "PropMolFlow-OE62 (12 elements, up to 80 atoms, O(N^2) complete DGL graph) Single-A800 Speed & VRAM Benchmark",
        "preprocess_meta": json.loads((data_dir / "preprocess_meta.json").read_text()),
        "train_throughput": train_bench,
        "infer_100step_ode_throughput": infer_bench,
    }
    out_path = args.work_dir / "propmolflow_oe62_speed_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[PropMolFlow-OE62 Speed Benchmark DONE] {json.dumps(out, indent=2)}", flush=True)


# ---------------------------------------------------------------------------
# Phase 3: Single-GPU / Multi-GPU DDP Training with Exact GPU-Hour Milestones
# ---------------------------------------------------------------------------
def _save_flowmol_ckpt(
    path: Path,
    raw_model: FlowMol,
    opt: torch.optim.Optimizer,
    hparams: Dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    cum_gpu_hours: float,
    last_train_loss: float,
    val_loss: Optional[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hyper_parameters": hparams,
        "state_dict": raw_model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "cum_gpu_hours": round(cum_gpu_hours, 4),
        "last_train_loss": round(last_train_loss, 5),
        "val_loss": round(val_loss, 5) if val_loss is not None else None,
    }
    tmp = path.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    tmp.replace(path)


def run_train(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1
    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        device = torch.device(args.device)
        rank = 0

    data_dir = args.work_dir / "data"
    ckpt_dir = args.work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hp = _default_oe62_flowmol_hparams(data_dir)
    train_ds = OE62FlowMolDataset(
        data_dir / "train_data_processed.pt",
        data_dir / "train_data_property_normalization.pt",
        hp["prior_config"],
    )
    val_ds = OE62FlowMolDataset(
        data_dir / "val_data_processed.pt",
        data_dir / "train_data_property_normalization.pt",
        hp["prior_config"],
    )

    raw_model = FlowMol(**hp).to(device)
    opt = torch.optim.Adam(raw_model.parameters(), lr=float(args.lr), weight_decay=1e-12)

    start_epoch = 0
    global_step = 0
    cum_gpu_sec = 0.0
    milestones = sorted(float(x) for x in args.milestones_gpuh)
    max_gpuh = float(args.max_gpuh)

    latest_ckpt = ckpt_dir / "latest.pt"
    if args.resume and latest_ckpt.exists():
        ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["state_dict"], strict=True)
        opt.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = int(ck.get("epoch", 0))
        global_step = int(ck.get("global_step", 0))
        cum_gpu_sec = float(ck.get("cum_gpu_hours", 0.0)) * 3600.0
        if rank == 0:
            print(
                f"[PropMolFlow-OE62 Resume] epoch={start_epoch}, step={global_step}, cum_gpu_h={cum_gpu_sec/3600:.3f}h",
                flush=True,
            )

    model: torch.nn.Module = (
        DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True) if is_ddp else raw_model
    )

    sampler = (
        torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if is_ddp
        else None
    )
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(args.train_batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        collate_fn=_collate_flowmol,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate_flowmol,
    )

    lw = dict(raw_model.total_loss_weights)
    f_ord = list(raw_model.canonical_feat_order)

    def _eval_val() -> float:
        raw_model.eval()
        tot_v = 0.0
        n_v = 0
        with torch.no_grad():
            for bg_v, pr_v in val_loader:
                bg_v = bg_v.to(device)
                bg_v.prop = pr_v.to(device)
                l_v = raw_model(bg_v)
                lv = sum(float(lw[f]) * float(l_v[f].item()) for f in f_ord)
                bs_v = int(pr_v.numel())
                tot_v += lv * bs_v
                n_v += bs_v
        raw_model.train()
        return tot_v / max(n_v, 1)

    done = False
    ep = start_epoch
    last_loss = 0.0

    while not done:
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        ep_t0 = time.perf_counter()
        running_loss = 0.0
        running_steps = 0

        for bg, pr in loader:
            st_t0 = time.perf_counter()
            bg = bg.to(device)
            bg.prop = pr.to(device)

            opt.zero_grad(set_to_none=True)
            losses = model(bg)
            total_loss = torch.zeros((), device=device)
            for feat in f_ord:
                total_loss = total_loss + float(lw[feat]) * losses[feat]
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(raw_model.parameters(), 1.0)
            opt.step()

            step_dt = time.perf_counter() - st_t0
            cum_gpu_sec += step_dt * float(world_size)
            global_step += 1
            last_loss = float(total_loss.item())
            running_loss += last_loss
            running_steps += 1

            cum_gpuh = cum_gpu_sec / 3600.0
            if rank == 0:
                for m_h in milestones:
                    tag = f"ckpt_{str(m_h).replace('.', 'p')}gpuh.pt"
                    m_path = ckpt_dir / tag
                    if cum_gpuh >= m_h and not m_path.exists():
                        v_l = _eval_val()
                        _save_flowmol_ckpt(
                            m_path,
                            raw_model,
                            opt,
                            hp,
                            epoch=ep,
                            global_step=global_step,
                            cum_gpu_hours=cum_gpuh,
                            last_train_loss=running_loss / max(running_steps, 1),
                            val_loss=v_l,
                        )
                        _save_flowmol_ckpt(
                            latest_ckpt,
                            raw_model,
                            opt,
                            hp,
                            epoch=ep,
                            global_step=global_step,
                            cum_gpu_hours=cum_gpuh,
                            last_train_loss=running_loss / max(running_steps, 1),
                            val_loss=v_l,
                        )
                        print(
                            f"[PropMolFlow-OE62 MILESTONE {m_h:.2f} GPU-h] Saved {m_path} "
                            f"(epoch={ep}, step={global_step}, train_loss={running_loss/max(running_steps,1):.4f}, val_loss={v_l:.4f})",
                            flush=True,
                        )

            if global_step % 10 == 0:
                if is_ddp:
                    stop_t = torch.tensor([1 if (rank == 0 and cum_gpuh >= max_gpuh) else 0], dtype=torch.int32, device=device)
                    dist.broadcast(stop_t, src=0)
                    should_stop = bool(stop_t.item() > 0)
                else:
                    should_stop = cum_gpuh >= max_gpuh
                if should_stop:
                    done = True
                    break

        ep += 1
        if rank == 0:
            v_l = _eval_val()
            cum_gpuh = cum_gpu_sec / 3600.0
            _save_flowmol_ckpt(
                latest_ckpt,
                raw_model,
                opt,
                hp,
                epoch=ep,
                global_step=global_step,
                cum_gpu_hours=cum_gpuh,
                last_train_loss=running_loss / max(running_steps, 1),
                val_loss=v_l,
            )
            ep_sec = time.perf_counter() - ep_t0
            print(
                f"[PropMolFlow-OE62 Epoch {ep:03d}] step={global_step} cum_gpu_h={cum_gpuh:.3f}/{max_gpuh:.2f}h "
                f"ep_wall={ep_sec:.1f}s train_loss={running_loss/max(running_steps,1):.4f} val_loss={v_l:.4f}",
                flush=True,
            )
            if args.stop_flag_file is not None and args.stop_flag_file.exists():
                print(f"[PropMolFlow-OE62] Detected stop flag {args.stop_flag_file}; exiting cleanly after saving latest.pt.", flush=True)
                done = True

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Phase 4: Evaluate `PropMolFlow-OE62` Checkpoint on Held-Out OE62 Quantiles (`1,792` Proposals)
# ---------------------------------------------------------------------------
def run_eval_checkpoint(args: argparse.Namespace) -> None:
    from scripts.loom_octree_process_critic_exp import (
        _build_oe62_critic_bus,
        _score_oe62_selected_cohort,
        _select_oe62_from_pool,
    )

    device = torch.device(args.device)
    data_dir = args.work_dir / "data"
    ck = torch.load(args.eval_ckpt, map_location="cpu", weights_only=False)
    hp = dict(ck["hyper_parameters"])
    hp["n_atoms_hist_file"] = str(data_dir / "train_data_n_atoms_histogram.pt")
    hp["marginal_dists_file"] = str(data_dir / "train_data_marginal_dists.pt")

    model = FlowMol(**hp).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    norm_file = str(data_dir / "train_data_property_normalization.pt")

    # Build conditional `P(N_atoms | HOMO)` empirical sampler from OE62 training set so PropMolFlow
    # receives realistic target-matched atom counts `(t, N_atoms)` on OE62!
    tr_d = torch.load(data_dir / "train_data_processed.pt", map_location="cpu", weights_only=False)
    tr_homo = tr_d["homo_eV"].numpy()
    tr_nat = tr_d["n_atoms"].numpy()

    targets_cfg = json.loads(args.oe62_targets_json.read_text())
    q_points = targets_cfg["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    seeds = (24411, 24412)
    attempts_per_seed = 128

    req_list: List[float] = []
    for q in quantiles:
        for _sd in seeds:
            req_list.extend([float(q_points[str(q)])] * attempts_per_seed)
    n_prompts = len(req_list)  # 1,792

    sd_off = int(getattr(args, "seed_offset", 0))
    torch.manual_seed(202611 + sd_off * 10007)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(202611 + sd_off * 10007)
    rng = np.random.default_rng(202611 + sd_off * 10007)
    m_max = int(args.m_max)
    natoms_rounds: List[List[int]] = []
    for _m in range(m_max):
        r_nat: List[int] = []
        for goal in req_list:
            near = np.where(np.abs(tr_homo - goal) <= 0.35)[0]
            if len(near) < 32:
                near = np.argsort(np.abs(tr_homo - goal))[:256]
            r_nat.append(int(rng.choice(tr_nat[near])))
        natoms_rounds.append(r_nat)

    # Warmup
    _ = model.sample_random_sizes(
        16,
        device=device,
        n_timesteps=10,
        property_name="homo",
        normalization_file_path=norm_file,
        training_mode=False,
        properties_handle_method="concatenate",
        multilple_values_to_one_property=req_list[:16],
        number_of_atoms=natoms_rounds[0][:16],
    )
    _sync(device)

    infer_bs = int(args.eval_batch_size)
    sampled_by_round: List[List[Any]] = []
    round_gen_walls: List[float] = []
    for m_i in range(m_max):
        _sync(device)
        tg0 = time.perf_counter()
        cur_mols: List[Any] = []
        for s in range(0, n_prompts, infer_bs):
            e = min(s + infer_bs, n_prompts)
            sub = model.sample_random_sizes(
                e - s,
                device=device,
                n_timesteps=100,
                property_name="homo",
                normalization_file_path=norm_file,
                training_mode=False,
                properties_handle_method="concatenate",
                multilple_values_to_one_property=req_list[s:e],
                number_of_atoms=natoms_rounds[m_i][s:e],
            )
            cur_mols.extend(sub)
        _sync(device)
        rw = time.perf_counter() - tg0
        round_gen_walls.append(rw)
        sampled_by_round.append(cur_mols)
        print(
            f"[PropMolFlow-OE62 Sample Round {m_i+1}/{m_max}] wall={rw:.2f}s ({n_prompts/max(rw,1e-6):.2f} mol/s)",
            flush=True,
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reconciler = ValenceReconciler(allowed_elements=tuple(OE62_ATOM_MAP))

    # Track 1: Native co-generated graph
    tn0 = time.perf_counter()
    nat_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(n_prompts)]
    nat_leg = np.zeros((n_prompts, m_max), dtype=bool)
    nat_exo = np.zeros((n_prompts, m_max), dtype=bool)
    for i in range(n_prompts):
        for m_i in range(m_max):
            sm = sampled_by_round[m_i][i]
            pos_np = sm.positions.detach().cpu().numpy().astype(np.float32)
            if _min_pair_dist(pos_np) < 0.65 or sm.rdkit_mol is None:
                continue
            try:
                cand = Chem.Mol(sm.rdkit_mol)
                if any(a.GetSymbol() not in OE62_ATOM_SET for a in cand.GetAtoms()):
                    continue
                Chem.SanitizeMol(cand)
                if len(Chem.GetMolFrags(cand)) != 1:
                    continue
                nat_grid[i][m_i] = cand
                nat_leg[i, m_i] = True
                has_rad = any(a.GetNumRadicalElectrons() > 0 for a in cand.GetAtoms())
                nat_exo[i, m_i] = (not bool(is_clean_organic_charge_state(cand))) or has_rad
            except Exception:
                continue
    nat_san_wall = time.perf_counter() - tn0

    # Track 2: Coordinate-only + `ValenceReconciler` (`PropMolFlow + Uni-Loom Track B`)
    tr0 = time.perf_counter()
    rec_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(n_prompts)]
    rec_leg = np.zeros((n_prompts, m_max), dtype=bool)
    rec_exo = np.zeros((n_prompts, m_max), dtype=bool)
    for i in range(n_prompts):
        for m_i in range(m_max):
            sm = sampled_by_round[m_i][i]
            syms = [s for s in sm.atom_types if s in OE62_ATOM_SET]
            pos_np = sm.positions.detach().cpu().numpy().astype(np.float32)
            if len(syms) != len(sm.atom_types):
                continue
            res = reconciler.reconcile_point_cloud(syms, pos_np, potential_slot=None)
            if res.reconciled_legal and res.mol is not None:
                rec_grid[i][m_i] = res.mol
                rec_leg[i, m_i] = True
                rec_exo[i, m_i] = bool(res.reconciled_exotic_charge)
    rec_san_wall = time.perf_counter() - tr0

    bus, source_graphs = _build_oe62_critic_bus(args, include_blind_auditor=True)

    m_budgets = [b for b in (1, 2, 4, 8) if b <= m_max]
    out_tracks: Dict[str, Any] = {"native_graph": {}, "coord_reconciled": {}}

    for track_name, grid, leg_arr, exo_arr, san_wall in (
        ("native_graph", nat_grid, nat_leg, nat_exo, nat_san_wall),
        ("coord_reconciled", rec_grid, rec_leg, rec_exo, rec_san_wall),
    ):
        for mb in m_budgets:
            sub_g = [row[:mb] for row in grid]
            sel_m, sel_l, sel_x, surr_w = _select_oe62_from_pool(
                sub_g, leg_arr[:, :mb], exo_arr[:, :mb], req_list, bus, mb, use_surrogate=(mb > 1), device=device
            )
            if track_name == "coord_reconciled" and bus.potential_slot is not None:
                reconciler.batch_polish_hydrogens_mlip(
                    [m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32
                )
            gen_w = float(sum(round_gen_walls[:mb]))
            rw_scaled = san_wall * (float(mb) / float(m_max))
            sdf_out = (
                args.eval_output_json.parent
                / f"{args.eval_output_json.stem}_cohorts"
                / f"{track_name}_M{mb}.sdf"
            )
            arm_metrics = _score_oe62_selected_cohort(
                sel_m,
                sel_l,
                sel_x,
                req_list,
                bus,
                source_graphs,
                gen_w,
                rw_scaled,
                surr_w,
                export_sdf_path=sdf_out,
            )
            leg_mols = [m for m, ok in zip(sel_m, sel_l) if ok and m is not None]
            n_leg = len(leg_mols)
            n_rad = sum(1 for m in leg_mols if any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()))
            n_netq = sum(1 for m in leg_mols if Chem.GetFormalCharge(m) != 0)
            n_cs_neu = sum(
                1
                for m in leg_mols
                if Chem.GetFormalCharge(m) == 0 and not any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms())
            )
            arm_metrics["closed_shell_neutral_rate_among_legal"] = round(n_cs_neu / n_leg, 5) if n_leg else 0.0
            arm_metrics["open_shell_radical_rate_among_legal"] = round(n_rad / n_leg, 5) if n_leg else 0.0
            arm_metrics["net_charged_rate_among_legal"] = round(n_netq / n_leg, 5) if n_leg else 0.0
            out_tracks[track_name][f"M{mb}"] = arm_metrics
            blind_mae_val = (
                float("nan") if arm_metrics.get("audit_mae_eV") is None else float(arm_metrics["audit_mae_eV"])
            )
            print(
                f"[PropMolFlow-OE62 {track_name} M={mb}] Eff={arm_metrics['effective_output_mol_per_sec']:.2f} mol/s | "
                f"Legal={arm_metrics['legal_rate']*100:.1f}% | SynthCleanHits={arm_metrics['synthesizable_clean_hits']}/1792 | "
                f"BlindDPA4_CleanHits={arm_metrics['blind_dpa4_clean_hits']} | BlindMAE={blind_mae_val:.4f}",
                flush=True,
            )

    report = {
        "checkpoint": str(args.eval_ckpt),
        "epoch": ck.get("epoch"),
        "global_step": ck.get("global_step"),
        "cum_gpu_hours": ck.get("cum_gpu_hours"),
        "val_loss": ck.get("val_loss"),
        "proposals_per_arm": n_prompts,
        "eval_batch_size": infer_bs,
        "tracks": out_tracks,
    }
    args.eval_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output_json.write_text(json.dumps(report, indent=2))
    print(f"[PropMolFlow-OE62 EVAL DONE] Wrote {args.eval_output_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["preprocess", "benchmark_speed", "train", "eval_checkpoint"], required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/propmolflow_oe62"))
    p.add_argument("--oe62-master-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"))
    p.add_argument("--oe62-master-valid-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_valid_B.sdf"))
    # Train options
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--milestones-gpuh", type=float, nargs="+", default=[3.0, 6.0, 12.0])
    p.add_argument("--max-gpuh", type=float, default=12.05)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--stop-flag-file", type=Path, default=None)
    # Eval options
    p.add_argument("--eval-ckpt", type=Path, default=None)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--m-max", type=int, default=4)
    p.add_argument("--eval-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_propmolflow_eval.json"))
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument("--oe62-source-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/source_labels/train_source.sdf"))
    p.add_argument("--oe62-replay-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_B.sdf"))
    p.add_argument("--oe62-targets-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/targets_B.json"))
    p.add_argument("--oe62-reward-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"))
    p.add_argument("--oe62-audit-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_e15_calibration_20261114/full_fit/oracle"))
    p.add_argument("--oe62-potential-weights", type=Path, default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"))
    p.add_argument("--allow-tf32", action="store_true", help="Enable Ampere TF32 Tensor Core matmuls fairly.")
    p.add_argument("--seed-offset", type=int, default=0, help="Independent sampling seed offset for multi-seed CI repeats.")

    args = p.parse_args()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.phase == "preprocess":
        run_preprocess(args)
    elif args.phase == "benchmark_speed":
        run_benchmark_speed(args)
    elif args.phase == "train":
        run_train(args)
    elif args.phase == "eval_checkpoint":
        if args.eval_ckpt is None:
            raise ValueError("--eval-ckpt is required for --phase eval_checkpoint")
        run_eval_checkpoint(args)


if __name__ == "__main__":
    main()
