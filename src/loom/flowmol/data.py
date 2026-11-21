"""Compact Flat-CSR 3D molecular corpus, Dynamic Edge-Budget Bucketing, and pure-PyTorch batching."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .constants import (
    ATOM_TO_IDX,
    CHARGE_TO_IDX,
    FLOWMOL_ATOM_TYPES,
    FLOWMOL_BOND_TYPES,
    FLOWMOL_FORMAL_CHARGES,
    LEGACY_5_BOND_TYPES,
    LEGACY_6_FORMAL_CHARGES,
    N_ATOM_TYPES,
    N_BOND_TYPES,
    N_CHARGES,
)
from .priors import coord_prior_std_for_size, sample_aligned_coord_prior_np


@dataclass
class FlowMolBatch:
    """Pure-PyTorch batched complete-graph representation (zero DGL)."""

    batch_size: int
    n_atoms: torch.Tensor               # (B,) int64
    node_batch_idx: torch.Tensor        # (N_tot,) int64
    src: torch.Tensor                   # (E_tot,) int64  [first half upper (u<v), second half lower (v>u)]
    dst: torch.Tensor                   # (E_tot,) int64
    upper_edge_mask: torch.Tensor       # (E_tot,) bool
    edge_batch_idx: torch.Tensor        # (E_tot,) int64
    upper_edge_batch_idx: torch.Tensor  # (E_tot // 2,) int64
    x_0: torch.Tensor                   # (N_tot, 3) float32
    x_1: Optional[torch.Tensor] = None        # (N_tot, 3) float32
    a_1: Optional[torch.Tensor] = None        # (N_tot,) int64 in [0, N_ATOM_TYPES)
    c_1: Optional[torch.Tensor] = None        # (N_tot,) int64 in [0, N_CHARGES)
    e_1_upper: Optional[torch.Tensor] = None  # (E_tot // 2,) int64 in [0, N_BOND_TYPES)
    source_id: Optional[torch.Tensor] = None  # (B,) int64

    def to(self, device: torch.device, non_blocking: bool = True) -> "FlowMolBatch":
        return FlowMolBatch(
            batch_size=self.batch_size,
            n_atoms=self.n_atoms.to(device, non_blocking=non_blocking),
            node_batch_idx=self.node_batch_idx.to(device, non_blocking=non_blocking),
            src=self.src.to(device, non_blocking=non_blocking),
            dst=self.dst.to(device, non_blocking=non_blocking),
            upper_edge_mask=self.upper_edge_mask.to(device, non_blocking=non_blocking),
            edge_batch_idx=self.edge_batch_idx.to(device, non_blocking=non_blocking),
            upper_edge_batch_idx=self.upper_edge_batch_idx.to(device, non_blocking=non_blocking),
            x_0=self.x_0.to(device, non_blocking=non_blocking),
            x_1=self.x_1.to(device, non_blocking=non_blocking) if self.x_1 is not None else None,
            a_1=self.a_1.to(device, non_blocking=non_blocking) if self.a_1 is not None else None,
            c_1=self.c_1.to(device, non_blocking=non_blocking) if self.c_1 is not None else None,
            e_1_upper=self.e_1_upper.to(device, non_blocking=non_blocking) if self.e_1_upper is not None else None,
            source_id=self.source_id.to(device, non_blocking=non_blocking) if self.source_id is not None else None,
        )


def build_complete_graph_batch(
    n_atoms_list: Sequence[int],
    device: torch.device,
    *,
    prior_std: float = 1.15,
    size_scale_prior: bool = True,
) -> FlowMolBatch:
    """Build a fresh inference `FlowMolBatch` with size-adaptive zero-COM Gaussian prior `x_0`."""
    bsz = len(n_atoms_list)
    n_atoms_t = torch.tensor(n_atoms_list, dtype=torch.long, device=device)
    node_batch_idx = torch.repeat_interleave(torch.arange(bsz, dtype=torch.long, device=device), n_atoms_t)
    n_tot = int(n_atoms_t.sum().item())

    mol_stds = coord_prior_std_for_size(n_atoms_t, base_std=prior_std, size_scale=size_scale_prior)
    assert isinstance(mol_stds, torch.Tensor)
    node_stds = mol_stds[node_batch_idx].unsqueeze(-1)
    x_0 = torch.randn((n_tot, 3), dtype=torch.float32, device=device) * node_stds
    sums = torch.zeros((bsz, 3), dtype=torch.float32, device=device)
    sums.index_add_(0, node_batch_idx, x_0)
    x_0 = x_0 - (sums / n_atoms_t.float().clamp_min(1.0).unsqueeze(-1))[node_batch_idx]

    u_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    eb_parts: List[torch.Tensor] = []
    offset = 0
    for b_i, na in enumerate(n_atoms_list):
        na_int = int(na)
        tri = torch.triu_indices(na_int, na_int, offset=1, device=device)
        u_parts.append(tri[0] + offset)
        v_parts.append(tri[1] + offset)
        eb_parts.append(torch.full((tri.shape[1],), b_i, dtype=torch.long, device=device))
        offset += na_int

    u_all = torch.cat(u_parts, dim=0)
    v_all = torch.cat(v_parts, dim=0)
    upper_eb = torch.cat(eb_parts, dim=0)
    n_up = int(u_all.shape[0])

    src = torch.cat([u_all, v_all], dim=0)
    dst = torch.cat([v_all, u_all], dim=0)
    edge_batch_idx = torch.cat([upper_eb, upper_eb], dim=0)
    upper_edge_mask = torch.zeros((2 * n_up,), dtype=torch.bool, device=device)
    upper_edge_mask[:n_up] = True

    return FlowMolBatch(
        batch_size=bsz,
        n_atoms=n_atoms_t,
        node_batch_idx=node_batch_idx,
        src=src,
        dst=dst,
        upper_edge_mask=upper_edge_mask,
        edge_batch_idx=edge_batch_idx,
        upper_edge_batch_idx=upper_eb,
        x_0=x_0,
    )


class FlatMoleculeCorpus(Dataset):
    """Memory-resident (or mmap) Flat-CSR 3D molecular corpus for multi-GPU FlowMol pretraining."""

    def __init__(
        self,
        corpus_dir: Path | str,
        *,
        split: str = "train",
        prior_std: float = 1.15,
        size_scale_prior: bool = True,
        use_ot_permutation: bool = True,
        use_ot_rigid: bool = True,
        load_in_memory: bool = True,
        max_atoms: Optional[int] = None,
        source_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.meta: Dict[str, Any] = json.loads((self.corpus_dir / "meta.json").read_text())
        self.prior_std = float(prior_std)
        self.size_scale_prior = bool(size_scale_prior)
        self.use_ot_permutation = bool(use_ot_permutation)
        self.use_ot_rigid = bool(use_ot_rigid)
        self.seed = int(seed)

        mmap = None if load_in_memory else "r"
        self.atom_ptr = np.load(self.corpus_dir / "atom_ptr.npy", mmap_mode=mmap)
        self.bond_ptr = np.load(self.corpus_dir / "bond_ptr.npy", mmap_mode=mmap)
        self.all_n_atoms = np.load(self.corpus_dir / "n_atoms.npy", mmap_mode=mmap)
        self.all_split = np.load(self.corpus_dir / "split.npy", mmap_mode=mmap)
        self.all_source_id = np.load(self.corpus_dir / "source_id.npy", mmap_mode=mmap)

        self.pos = np.load(self.corpus_dir / "pos.npy", mmap_mode=mmap)
        self.atom_type = np.load(self.corpus_dir / "atom_type.npy", mmap_mode=mmap)
        self.charge = np.load(self.corpus_dir / "charge.npy", mmap_mode=mmap)
        self.bond_u = np.load(self.corpus_dir / "bond_u.npy", mmap_mode=mmap)
        self.bond_v = np.load(self.corpus_dir / "bond_v.npy", mmap_mode=mmap)
        self.bond_type = np.load(self.corpus_dir / "bond_type.npy", mmap_mode=mmap)

        # Remap subset-vocabulary corpora (atom types & formal charges) to the canonical full-atom contract
        corpus_vocab = tuple(self.meta.get("atom_types", FLOWMOL_ATOM_TYPES))
        if corpus_vocab != FLOWMOL_ATOM_TYPES:
            lut = np.empty(len(corpus_vocab), dtype=np.uint8)
            for old_i, sym in enumerate(corpus_vocab):
                if sym not in ATOM_TO_IDX:
                    raise ValueError(f"Corpus atom symbol '{sym}' not in canonical 118-element vocabulary")
                lut[old_i] = ATOM_TO_IDX[sym]
            self._atom_remap_lut: Optional[np.ndarray] = lut
        else:
            self._atom_remap_lut = None

        raw_chg_vocab = self.meta.get("formal_charges")
        if raw_chg_vocab is not None:
            corpus_charges = tuple(int(q) for q in raw_chg_vocab)
        elif int(self.meta.get("n_charges", N_CHARGES)) == len(LEGACY_6_FORMAL_CHARGES):
            corpus_charges = LEGACY_6_FORMAL_CHARGES
        else:
            corpus_charges = FLOWMOL_FORMAL_CHARGES

        if corpus_charges != FLOWMOL_FORMAL_CHARGES:
            clut = np.empty(len(corpus_charges), dtype=np.uint8)
            for old_i, q in enumerate(corpus_charges):
                if q not in CHARGE_TO_IDX:
                    raise ValueError(f"Corpus formal charge {q} not in canonical vocabulary {FLOWMOL_FORMAL_CHARGES}")
                clut[old_i] = CHARGE_TO_IDX[q]
            self._charge_remap_lut: Optional[np.ndarray] = clut
        else:
            self._charge_remap_lut = None

        elem_counts = self.meta.get("element_counts", {})
        active = [sym for sym, cnt in elem_counts.items() if int(cnt) > 0 and sym in ATOM_TO_IDX]
        self.active_elements: Tuple[str, ...] = tuple(active) if active else corpus_vocab

        chg_counts = self.meta.get("charge_counts", {})
        if chg_counts:
            act_q = [int(k) for k, cnt in chg_counts.items() if int(cnt) > 0 and int(k) in CHARGE_TO_IDX]
            self.active_charges: Tuple[int, ...] = tuple(act_q) if act_q else corpus_charges
        else:
            self.active_charges = corpus_charges

        split_map = {"train": 0, "val": 1, "valid": 1, "test": 2}
        if split == "all":
            mask = np.ones(len(self.all_n_atoms), dtype=bool)
        else:
            mask = self.all_split == split_map[split]
        if max_atoms is not None:
            mask &= self.all_n_atoms <= int(max_atoms)

        base_indices = np.nonzero(mask)[0].astype(np.int64)

        if source_weights and split == "train":
            src_names = self.meta.get("source_names", [])
            expanded: List[np.ndarray] = []
            for sid, sname in enumerate(src_names):
                s_idx = base_indices[self.all_source_id[base_indices] == sid]
                if len(s_idx) == 0:
                    continue
                w = float(source_weights.get(sname, 1.0))
                if w <= 0:
                    continue
                reps = int(np.floor(w))
                rem = w - reps
                for _ in range(reps):
                    expanded.append(s_idx)
                if rem > 1e-6:
                    rng = np.random.default_rng(self.seed + sid)
                    k = max(1, int(round(len(s_idx) * rem)))
                    expanded.append(rng.choice(s_idx, size=k, replace=False))
            self.indices = np.concatenate(expanded, axis=0) if expanded else base_indices
        else:
            self.indices = base_indices

        self.n_atoms_view = np.asarray(self.all_n_atoms[self.indices], dtype=np.int32)
        self._worker_rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def _get_rng(self, idx: int) -> np.random.Generator:
        if self._worker_rng is None:
            info = torch.utils.data.get_worker_info()
            w_seed = (info.seed if info is not None else self.seed) ^ (idx * 2654435761 & 0xFFFFFFFF)
            self._worker_rng = np.random.default_rng(w_seed)
        return self._worker_rng

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mol_id = int(self.indices[idx])
        a0 = int(self.atom_ptr[mol_id])
        a1 = int(self.atom_ptr[mol_id + 1])
        b0 = int(self.bond_ptr[mol_id])
        b1 = int(self.bond_ptr[mol_id + 1])
        n_a = a1 - a0

        x1 = np.array(self.pos[a0:a1], dtype=np.float32, copy=True)
        x1 -= x1.mean(axis=0, keepdims=True)
        raw_a = self.atom_type[a0:a1]
        if self._atom_remap_lut is not None:
            a_type = self._atom_remap_lut[raw_a].astype(np.int64, copy=False)
        else:
            a_type = np.array(raw_a, dtype=np.int64, copy=True)

        raw_c = self.charge[a0:a1]
        if self._charge_remap_lut is not None:
            chg = self._charge_remap_lut[raw_c].astype(np.int64, copy=False)
        else:
            chg = np.array(raw_c, dtype=np.int64, copy=True)

        n_up = n_a * (n_a - 1) // 2
        e_upper = np.zeros(n_up, dtype=np.int64)
        if b1 > b0:
            bu = np.asarray(self.bond_u[b0:b1], dtype=np.int64)
            bv = np.asarray(self.bond_v[b0:b1], dtype=np.int64)
            bt = np.asarray(self.bond_type[b0:b1], dtype=np.int64)
            up_idx = bu * n_a - (bu * (bu + 1)) // 2 + (bv - bu - 1)
            e_upper[up_idx] = bt

        rng = self._get_rng(idx)
        x0 = sample_aligned_coord_prior_np(
            x1,
            rng,
            prior_std=self.prior_std,
            size_scale_prior=self.size_scale_prior,
            use_permutation=self.use_ot_permutation,
            use_rigid=self.use_ot_rigid,
        )

        return {
            "n_atoms": n_a,
            "source_id": int(self.all_source_id[mol_id]),
            "x_0": x0,
            "x_1": x1,
            "a_1": a_type,
            "c_1": chg,
            "e_1_upper": e_upper,
        }


def collate_flowmol_batch(items: Sequence[Dict[str, Any]]) -> FlowMolBatch:
    """Collate list of sparse-reconstructed molecule dicts into a single pure-PyTorch `FlowMolBatch`."""
    bsz = len(items)
    n_atoms_list = [int(it["n_atoms"]) for it in items]
    n_atoms_t = torch.tensor(n_atoms_list, dtype=torch.long)
    source_id_t = (
        torch.tensor([int(it["source_id"]) for it in items], dtype=torch.long)
        if items and ("source_id" in items[0])
        else None
    )

    node_batch_idx = torch.repeat_interleave(torch.arange(bsz, dtype=torch.long), n_atoms_t)
    x_0 = torch.from_numpy(np.concatenate([it["x_0"] for it in items], axis=0))
    x_1 = torch.from_numpy(np.concatenate([it["x_1"] for it in items], axis=0))
    a_1 = torch.from_numpy(np.concatenate([it["a_1"] for it in items], axis=0))
    c_1 = torch.from_numpy(np.concatenate([it["c_1"] for it in items], axis=0))
    e_1_upper = torch.from_numpy(np.concatenate([it["e_1_upper"] for it in items], axis=0))

    u_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    eb_parts: List[torch.Tensor] = []
    offset = 0
    for b_i, na in enumerate(n_atoms_list):
        tri = torch.triu_indices(na, na, offset=1)
        u_parts.append(tri[0] + offset)
        v_parts.append(tri[1] + offset)
        eb_parts.append(torch.full((tri.shape[1],), b_i, dtype=torch.long))
        offset += na

    u_all = torch.cat(u_parts, dim=0)
    v_all = torch.cat(v_parts, dim=0)
    upper_eb = torch.cat(eb_parts, dim=0)
    n_up = int(u_all.shape[0])

    src = torch.cat([u_all, v_all], dim=0)
    dst = torch.cat([v_all, u_all], dim=0)
    edge_batch_idx = torch.cat([upper_eb, upper_eb], dim=0)
    upper_edge_mask = torch.zeros((2 * n_up,), dtype=torch.bool)
    upper_edge_mask[:n_up] = True

    return FlowMolBatch(
        batch_size=bsz,
        n_atoms=n_atoms_t,
        node_batch_idx=node_batch_idx,
        src=src,
        dst=dst,
        upper_edge_mask=upper_edge_mask,
        edge_batch_idx=edge_batch_idx,
        upper_edge_batch_idx=upper_eb,
        x_0=x_0,
        x_1=x_1,
        a_1=a_1,
        c_1=c_1,
        e_1_upper=e_1_upper,
        source_id=source_id_t,
    )


class DynamicEdgeBudgetBatchSampler(Sampler[List[int]]):
    """Lockstep DDP Dynamic Edge-Budget Bucketing Sampler (Roadmap Stage 2)."""

    def __init__(
        self,
        n_atoms: np.ndarray,
        *,
        max_edges_per_batch: int = 120_000,
        max_mols_per_batch: int = 192,
        min_mols_per_batch: int = 4,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        size_jitter: float = 3.0,
        shuffle: bool = True,
        skip_batches: int = 0,
    ) -> None:
        self.n_atoms = np.asarray(n_atoms, dtype=np.int32)
        self.max_edges_per_batch = int(max_edges_per_batch)
        self.max_mols_per_batch = int(max_mols_per_batch)
        self.min_mols_per_batch = int(min_mols_per_batch)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.size_jitter = float(size_jitter)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.skip_batches = int(skip_batches)
        self._batches_cache: Optional[List[List[int]]] = None

    def set_epoch(self, epoch: int, skip_batches: int = 0) -> None:
        self.epoch = int(epoch)
        self.skip_batches = max(0, int(skip_batches))
        self._batches_cache = None

    def _build_all_batches(self) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 10007)
        n = len(self.n_atoms)
        if n == 0:
            return []

        if self.shuffle:
            sort_keys = self.n_atoms.astype(np.float64) + rng.normal(0.0, self.size_jitter, size=n)
            order = np.argsort(sort_keys, kind="mergesort")
        else:
            order = np.argsort(self.n_atoms, kind="mergesort")

        batches: List[List[int]] = []
        batch_edges: List[int] = []
        cur: List[int] = []
        cur_edges = 0

        for idx in order.tolist():
            na = int(self.n_atoms[idx])
            e_cost = na * (na - 1)
            if cur and (cur_edges + e_cost > self.max_edges_per_batch or len(cur) >= self.max_mols_per_batch):
                batches.append(cur)
                batch_edges.append(cur_edges)
                cur = []
                cur_edges = 0
            cur.append(idx)
            cur_edges += e_cost
        if len(cur) >= self.min_mols_per_batch or (not batches and cur):
            batches.append(cur)
            batch_edges.append(cur_edges)

        n_groups = len(batches) // self.world_size
        if n_groups == 0:
            return batches[: self.world_size] if len(batches) >= self.world_size else batches

        by_load = np.argsort(np.asarray(batch_edges, dtype=np.int64), kind="mergesort")
        trimmed = by_load[: n_groups * self.world_size]
        groups = trimmed.reshape(n_groups, self.world_size)

        if self.shuffle:
            group_perm = rng.permutation(n_groups)
            groups = groups[group_perm]
            for g_i in range(n_groups):
                rng.shuffle(groups[g_i])

        rank_batch_ids = groups[:, self.rank].tolist()
        return [batches[b_id] for b_id in rank_batch_ids]

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._build_all_batches()
        self._batches_cache = batches
        if self.skip_batches > 0:
            return iter(batches[self.skip_batches :])
        return iter(batches)

    def __len__(self) -> int:
        if self._batches_cache is None:
            self._batches_cache = self._build_all_batches()
        return max(0, len(self._batches_cache) - self.skip_batches)

    def total_steps_in_epoch(self) -> int:
        if self._batches_cache is None:
            self._batches_cache = self._build_all_batches()
        return len(self._batches_cache)


def write_flat_corpus(
    out_dir: Path | str,
    records: Sequence[Dict[str, np.ndarray]],
    *,
    splits: Sequence[int],
    source_ids: Sequence[int],
    source_names: Sequence[str],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize a list of sparse molecular records into the compact Flat-CSR binary format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_mols = len(records)
    if n_mols == 0:
        raise ValueError("Cannot write empty molecular corpus")

    atom_ptr = np.zeros(n_mols + 1, dtype=np.int64)
    bond_ptr = np.zeros(n_mols + 1, dtype=np.int64)
    n_atoms_arr = np.empty(n_mols, dtype=np.uint16)

    for i, r in enumerate(records):
        na = int(r["pos"].shape[0])
        nb = int(r["bond_u"].shape[0])
        n_atoms_arr[i] = na
        atom_ptr[i + 1] = atom_ptr[i] + na
        bond_ptr[i + 1] = bond_ptr[i] + nb

    pos_all = np.concatenate([r["pos"].astype(np.float32, copy=False) for r in records], axis=0)
    atom_all = np.concatenate([r["atom_type"].astype(np.uint8, copy=False) for r in records], axis=0)
    chg_all = np.concatenate([r["charge"].astype(np.uint8, copy=False) for r in records], axis=0)
    bu_all = np.concatenate([r["bond_u"].astype(np.uint8, copy=False) for r in records], axis=0) if bond_ptr[-1] > 0 else np.zeros(0, dtype=np.uint8)
    bv_all = np.concatenate([r["bond_v"].astype(np.uint8, copy=False) for r in records], axis=0) if bond_ptr[-1] > 0 else np.zeros(0, dtype=np.uint8)
    bt_all = np.concatenate([r["bond_type"].astype(np.uint8, copy=False) for r in records], axis=0) if bond_ptr[-1] > 0 else np.zeros(0, dtype=np.uint8)

    split_arr = np.asarray(splits, dtype=np.uint8)
    src_arr = np.asarray(source_ids, dtype=np.uint8)

    np.save(out / "atom_ptr.npy", atom_ptr)
    np.save(out / "bond_ptr.npy", bond_ptr)
    np.save(out / "n_atoms.npy", n_atoms_arr)
    np.save(out / "split.npy", split_arr)
    np.save(out / "source_id.npy", src_arr)
    np.save(out / "pos.npy", pos_all)
    np.save(out / "atom_type.npy", atom_all)
    np.save(out / "charge.npy", chg_all)
    np.save(out / "bond_u.npy", bu_all)
    np.save(out / "bond_v.npy", bv_all)
    np.save(out / "bond_type.npy", bt_all)

    train_na = n_atoms_arr[split_arr == 0]
    hist_vals, hist_counts = np.unique(train_na if len(train_na) > 0 else n_atoms_arr, return_counts=True)
    elem_counts_np = np.bincount(atom_all.astype(np.int64), minlength=N_ATOM_TYPES)
    chg_counts_np = np.bincount(chg_all.astype(np.int64), minlength=N_CHARGES)
    bond_counts_np = np.bincount(bt_all.astype(np.int64), minlength=N_BOND_TYPES)

    elem_counts = {FLOWMOL_ATOM_TYPES[i]: int(elem_counts_np[i]) for i in range(N_ATOM_TYPES)}
    charge_counts = {str(FLOWMOL_FORMAL_CHARGES[i]): int(chg_counts_np[i]) for i in range(N_CHARGES)}
    bond_counts = {FLOWMOL_BOND_TYPES[i]: int(bond_counts_np[i]) for i in range(1, N_BOND_TYPES)}

    # Per-source empirical distributions (enables standalone domain-matched benchmarking from checkpoint metadata)
    mol_source_atom_id = np.repeat(src_arr, n_atoms_arr.astype(np.int64))
    per_source_n_atoms_hist: Dict[str, Dict[int, int]] = {}
    per_source_active_elements: Dict[str, List[str]] = {}
    split_source_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for sid, sname in enumerate(source_names):
        s_mask = src_arr == sid
        for sp_id, sp_name in ((0, "train"), (1, "val"), (2, "test")):
            split_source_counts[sp_name][sname] = int((s_mask & (split_arr == sp_id)).sum())
        s_train_na = n_atoms_arr[s_mask & (split_arr == 0)]
        s_na_ref = s_train_na if len(s_train_na) > 0 else n_atoms_arr[s_mask]
        if len(s_na_ref) > 0:
            sv, sc = np.unique(s_na_ref, return_counts=True)
            per_source_n_atoms_hist[sname] = {int(k): int(v) for k, v in zip(sv.tolist(), sc.tolist())}
        else:
            per_source_n_atoms_hist[sname] = {}
        s_atoms = atom_all[mol_source_atom_id == sid]
        s_ec = np.bincount(s_atoms.astype(np.int64), minlength=N_ATOM_TYPES)
        per_source_active_elements[sname] = [FLOWMOL_ATOM_TYPES[i] for i in range(N_ATOM_TYPES) if int(s_ec[i]) > 0]

    meta: Dict[str, Any] = {
        "atom_types": list(FLOWMOL_ATOM_TYPES),
        "formal_charges": list(FLOWMOL_FORMAL_CHARGES),
        "bond_types": list(FLOWMOL_BOND_TYPES),
        "n_atom_types": N_ATOM_TYPES,
        "n_charges": N_CHARGES,
        "n_bond_types": N_BOND_TYPES,
        "n_mols": n_mols,
        "n_atoms_total": int(atom_ptr[-1]),
        "n_bonds_total": int(bond_ptr[-1]),
        "source_names": list(source_names),
        "source_counts": {sname: int((src_arr == sid).sum()) for sid, sname in enumerate(source_names)},
        "split_source_counts": split_source_counts,
        "split_counts": {
            "train": int((split_arr == 0).sum()),
            "val": int((split_arr == 1).sum()),
            "test": int((split_arr == 2).sum()),
        },
        "n_atoms_min": int(n_atoms_arr.min()),
        "n_atoms_median": float(np.median(n_atoms_arr)),
        "n_atoms_p95": float(np.percentile(n_atoms_arr, 95)),
        "n_atoms_max": int(n_atoms_arr.max()),
        "n_atoms_hist": {int(k): int(v) for k, v in zip(hist_vals.tolist(), hist_counts.tolist())},
        "per_source_n_atoms_hist": per_source_n_atoms_hist,
        "per_source_active_elements": per_source_active_elements,
        "n_active_elements": int((elem_counts_np > 0).sum()),
        "element_counts": elem_counts,
        "charge_counts": charge_counts,
        "covalent_bond_counts": bond_counts,
        **(extra_meta or {}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta
