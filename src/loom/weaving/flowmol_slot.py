"""Hot-swappable CTMC Flow Matching (`PropMolFlow` / `FlowMol`) `GeneratorSlot` for `Uni-Loom`."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F

from .slots import GeneratorSlot, resolve_effort_candidates

DEFAULT_PROPMOLFLOW_ROOT = Path("/sjtu-combustion/wjw/chensq/PropMolFlow")
DEFAULT_COMPAT_ROOT = Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat")
OE62_DEFAULT_ATOM_MAP: Tuple[str, ...] = (
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"
)


def _install_pytorch_lightning_shim() -> None:
    try:
        import pytorch_lightning  # noqa: F401
        return
    except ImportError:
        pass
    pl = types.ModuleType("pytorch_lightning")

    class _LightningModule(nn.Module):
        def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
            pass

        def log(self, *args: Any, **kwargs: Any) -> None:
            pass

        def log_dict(self, *args: Any, **kwargs: Any) -> None:
            pass

    pl.LightningModule = _LightningModule  # type: ignore[attr-defined]
    sys.modules["pytorch_lightning"] = pl


def _ensure_propmolflow_paths(propmolflow_root: Optional[Path | str] = None) -> None:
    _install_pytorch_lightning_shim()
    roots = [
        Path(propmolflow_root) if propmolflow_root is not None else DEFAULT_PROPMOLFLOW_ROOT,
        DEFAULT_COMPAT_ROOT,
    ]
    for cand in roots:
        if cand.exists():
            cstr = str(cand)
            if cstr not in sys.path:
                sys.path.insert(0, cstr)


def _mol_to_flowmol_record(
    mol: Optional[Chem.Mol],
    atom_to_idx: Dict[str, int],
    *,
    max_atoms: int = 80,
) -> Optional[Dict[str, torch.Tensor]]:
    """Extract centered 3D coordinates, atom/charge/bond types from an RDKit molecule."""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    n_a = int(mol.GetNumAtoms())
    if n_a < 2 or n_a > max_atoms:
        return None
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    if any(s not in atom_to_idx for s in syms):
        return None
    qs = [int(a.GetFormalCharge()) for a in mol.GetAtoms()]
    if any(q < -2 or q > 3 for q in qs):
        return None

    pos = torch.from_numpy(np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32))
    pos = pos - pos.mean(dim=0, keepdim=True)
    a_idx = torch.tensor([atom_to_idx[s] for s in syms], dtype=torch.long)
    c_idx = torch.tensor([q + 2 for q in qs], dtype=torch.long)

    adj = torch.zeros((n_a, n_a), dtype=torch.long)
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
        adj[u, v] = cls_id
        adj[v, u] = cls_id

    upper = torch.triu_indices(n_a, n_a, offset=1)
    u_lbl = adj[upper[0], upper[1]]
    return {
        "n_atoms": torch.tensor(n_a, dtype=torch.long),
        "pos": pos,
        "atom_idx": a_idx,
        "charge_idx": c_idx,
        "upper_src": upper[0],
        "upper_dst": upper[1],
        "upper_bond_type": u_lbl,
    }


def _collate_records_for_flowmol(
    records: Sequence[Dict[str, torch.Tensor]],
    *,
    num_atom_types: int,
    prior_config: Optional[Dict[str, Any]],
    use_dgl: bool,
) -> Dict[str, Any]:
    """Pack molecular records into either a batched DGLGraph (real `FlowMol`) or pure tensors (CPU test mock)."""
    if use_dgl:
        import dgl
        from propmolflow.data_processing.priors import coupled_node_prior, edge_prior

        p_cfg = prior_config or {
            "a": {"align": False, "kwargs": {}, "type": "ctmc"},
            "c": {"align": False, "kwargs": {}, "type": "ctmc"},
            "e": {"align": False, "kwargs": {}, "type": "ctmc"},
            "x": {"align": True, "kwargs": {"std": 1.0}, "type": "centered-normal"},
        }
        graphs = []
        for rec in records:
            n_a = int(rec["n_atoms"].item())
            a_oh = F.one_hot(rec["atom_idx"], num_classes=num_atom_types).float()
            c_oh = F.one_hot(rec["charge_idx"], num_classes=6).float()
            u_src, u_dst = rec["upper_src"], rec["upper_dst"]
            u_lbl = rec["upper_bond_type"]
            src = torch.cat([u_src, u_dst], dim=0)
            dst = torch.cat([u_dst, u_src], dim=0)
            e_oh = F.one_hot(torch.cat([u_lbl, u_lbl], dim=0), num_classes=5).float()

            g = dgl.graph((src, dst), num_nodes=n_a)
            g.ndata["x_1_true"] = rec["pos"]
            g.ndata["a_1_true"] = a_oh
            g.ndata["c_1_true"] = c_oh
            g.edata["e_1_true"] = e_oh

            prior_nodes = coupled_node_prior(dst_dict={"x": rec["pos"], "a": a_oh, "c": c_oh}, prior_config=p_cfg)
            for feat, val in prior_nodes.items():
                g.ndata[f"{feat}_0"] = val
            upper_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
            upper_mask[: u_src.shape[0]] = True
            g.edata["e_0"] = edge_prior(upper_mask, p_cfg["e"])
            graphs.append(g)

        return {
            "mode": "dgl",
            "graph": dgl.batch(graphs),
            "batch_size": len(records),
        }

    # Pure-PyTorch tensor representation for lightweight unit tests without DGL
    node_batch_idx: List[torch.Tensor] = []
    edge_batch_idx: List[torch.Tensor] = []
    x1_list: List[torch.Tensor] = []
    a1_list: List[torch.Tensor] = []
    c1_list: List[torch.Tensor] = []
    e1_list: List[torch.Tensor] = []
    for b_i, rec in enumerate(records):
        n_a = int(rec["n_atoms"].item())
        n_ue = int(rec["upper_bond_type"].shape[0])
        node_batch_idx.append(torch.full((n_a,), b_i, dtype=torch.long))
        edge_batch_idx.append(torch.full((n_ue,), b_i, dtype=torch.long))
        x1_list.append(rec["pos"])
        a1_list.append(F.one_hot(rec["atom_idx"], num_classes=num_atom_types).float())
        c1_list.append(F.one_hot(rec["charge_idx"], num_classes=6).float())
        e1_list.append(F.one_hot(rec["upper_bond_type"], num_classes=5).float())

    return {
        "mode": "tensor",
        "batch_size": len(records),
        "node_batch_idx": torch.cat(node_batch_idx, dim=0),
        "edge_batch_idx": torch.cat(edge_batch_idx, dim=0),
        "x_1_true": torch.cat(x1_list, dim=0),
        "a_1_true": torch.cat(a1_list, dim=0),
        "c_1_true": torch.cat(c1_list, dim=0),
        "e_1_true": torch.cat(e1_list, dim=0),
    }


class FlowMolReplayDataset:
    """Index-aligned replay dataset compatible with `LoomGRPOTrainer` and `LoomFlowMolGenerator`."""

    def __init__(
        self,
        molecules: Sequence[Chem.Mol],
        atom_type_map: Sequence[str],
        *,
        prior_config: Optional[Dict[str, Any]] = None,
        max_atoms: int = 80,
        use_dgl: bool = True,
        seed: int = 42,
    ) -> None:
        self.molecules = list(molecules)
        self.atom_type_map = tuple(atom_type_map)
        self.atom_to_idx = {s: i for i, s in enumerate(self.atom_type_map)}
        self.prior_config = prior_config
        self.max_atoms = int(max_atoms)
        self.use_dgl = bool(use_dgl)
        self.seed = int(seed)
        self._records: List[Optional[Dict[str, torch.Tensor]]] = [
            _mol_to_flowmol_record(m, self.atom_to_idx, max_atoms=self.max_atoms) for m in self.molecules
        ]

    def __len__(self) -> int:
        return len(self.molecules)

    def collate(self, indices: Sequence[int]) -> Dict[str, Any]:
        chosen: List[Dict[str, torch.Tensor]] = []
        for idx in indices:
            rec = self._records[int(idx)]
            if rec is None:
                raise ValueError(f"Record {idx} is incompatible with FlowMol element/charge/size contract")
            chosen.append(rec)
        return _collate_records_for_flowmol(
            chosen,
            num_atom_types=len(self.atom_type_map),
            prior_config=self.prior_config,
            use_dgl=self.use_dgl,
        )


class LoomFlowMolGenerator(GeneratorSlot):
    """Plug-and-play CTMC Flow Matching (`PropMolFlow` / `FlowMol`) backend implementing `GeneratorSlot`."""

    backend_id: str = "loom_flowmol_v1"
    supports_exact_trajectory_logprob: bool = False
    supports_prefix_process_search: bool = False
    generator_family: str = "ctmc_flow_matching"

    def __init__(
        self,
        backbone: nn.Module,
        atom_type_map: Sequence[str],
        target_cols: Sequence[str],
        target_units: Sequence[str],
        mean: Sequence[float],
        scale: Sequence[float],
        *,
        sample_timesteps: int = 100,
        sft_time_draws: int = 2,
        max_atoms: int = 80,
        norm_file_path: Optional[str] = None,
        prop_histogram_vals: Optional[np.ndarray] = None,
        prop_histogram_natoms: Optional[np.ndarray] = None,
        prior_config: Optional[Dict[str, Any]] = None,
        hparams: Optional[Dict[str, Any]] = None,
        epoch: int = 0,
        global_step: int = 0,
        cum_gpu_hours: float = 0.0,
    ) -> None:
        super().__init__()
        if len(target_cols) != len(target_units) or len(mean) != len(target_cols) or len(scale) != len(target_cols):
            raise ValueError("target_cols, target_units, mean, and scale must have matching lengths")

        self.backbone = backbone
        self.atom_type_map = tuple(atom_type_map)
        self._allowed_elements = self.atom_type_map
        self.atom_to_idx = {s: i for i, s in enumerate(self.atom_type_map)}
        self.args = types.SimpleNamespace(allow_atoms=",".join(self.atom_type_map))

        self.target_cols = list(target_cols)
        self.target_units = list(target_units)
        self.num_props = len(self.target_cols)

        mean_t = torch.tensor(np.asarray(mean, dtype=np.float32))
        scale_t = torch.tensor(np.asarray(scale, dtype=np.float32))
        if not torch.isfinite(mean_t).all() or not torch.isfinite(scale_t).all() or (scale_t <= 0).any():
            raise ValueError("Invalid FlowMol target normalization statistics")
        self.register_buffer("target_mean", mean_t)
        self.register_buffer("target_scale", scale_t)

        self.sample_timesteps = int(sample_timesteps)
        self.sft_time_draws = max(1, int(sft_time_draws))
        self.max_atoms = int(max_atoms)
        self.norm_file_path = norm_file_path
        self.prop_histogram_vals = (
            np.asarray(prop_histogram_vals, dtype=np.float64) if prop_histogram_vals is not None else None
        )
        self.prop_histogram_natoms = (
            np.asarray(prop_histogram_natoms, dtype=np.int64) if prop_histogram_natoms is not None else None
        )
        self.prior_config = prior_config or getattr(backbone, "prior_config", None)
        self.hparams = dict(hparams) if hparams is not None else {}
        self.epoch = int(epoch)
        self.global_step = int(global_step)
        self.cum_gpu_hours = float(cum_gpu_hours)

    @property
    def _is_native_flowmol(self) -> bool:
        from ..flowmol.model import FlowMolModel

        return isinstance(self.backbone, FlowMolModel)

    @property
    def _uses_dgl(self) -> bool:
        return (not self._is_native_flowmol) and (not bool(getattr(self.backbone, "is_mock_flow_backbone", False)))

    def set_allowed_elements(self, elements: Sequence[str]) -> None:
        valid = set(self.atom_type_map)
        cleaned = tuple(str(e) for e in elements if str(e) in valid)
        if not cleaned:
            raise ValueError(f"No elements in {elements} match FlowMol vocabulary")
        self._allowed_elements = cleaned
        self.args.allow_atoms = ",".join(cleaned)

    def normalize_targets(self, targets_eV: np.ndarray | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(targets_eV, dtype=torch.float32, device=self.target_mean.device)
        return (t - self.target_mean) / self.target_scale

    def denormalize_targets(self, targets_norm: np.ndarray | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(targets_norm, dtype=torch.float32, device=self.target_mean.device)
        return t * self.target_scale + self.target_mean

    def _sample_conditional_natoms(self, goals_eV: Sequence[float], seed: Optional[int]) -> List[int]:
        if self.prop_histogram_vals is None or self.prop_histogram_natoms is None:
            if hasattr(self.backbone, "sample_n_atoms"):
                return [int(x) for x in self.backbone.sample_n_atoms(len(goals_eV)).tolist()]
            return [8] * len(goals_eV)

        rng = np.random.default_rng(seed if seed is not None else 202611)
        out: List[int] = []
        vals = self.prop_histogram_vals
        nats = self.prop_histogram_natoms
        for g in goals_eV:
            near = np.where(np.abs(vals - float(g)) <= 0.35)[0]
            if len(near) < 32:
                near = np.argsort(np.abs(vals - float(g)))[: min(256, len(vals))]
            out.append(int(rng.choice(nats[near])))
        return out

    @torch.no_grad()
    def sample(
        self,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        cfg_scale: float = 1.0,
        xyz_cfg_scale: float = 1.0,
        cfg_tree_decay: float = 0.0,
        geometric_guard: bool = False,
        temperatures: Tuple[float, float, float] = (0.85, 0.30, 0.25),
        seed: Optional[int] = None,
        use_bf16: bool = True,
        electron_parity: Optional[int] = 0,
        leaf_atom_budget: Optional[np.ndarray | Sequence[float]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        del cfg_tree_decay, geometric_guard, temperatures, electron_parity
        atom_budget_kw = kwargs.pop("atom_budget", None)
        if leaf_atom_budget is None and atom_budget_kw is not None:
            leaf_atom_budget = atom_budget_kw
        del kwargs
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            raw_targets = self.denormalize_targets(targets_norm)
            # Determine primary conditioned column (first active column if multiple)
            act_any = active_mask.bool().any(dim=0)
            prop_col = int(torch.nonzero(act_any, as_tuple=False)[0].item()) if act_any.any() else 0
            prop_name = self.target_cols[prop_col]
            goals_raw = [float(x) for x in raw_targets[:, prop_col].detach().cpu().tolist()]
            if leaf_atom_budget is not None:
                b_arr = np.asarray(leaf_atom_budget, dtype=np.float64).reshape(len(goals_raw))
                n_atoms = [max(2, int(round(float(x)))) for x in b_arr.tolist()]
            else:
                n_atoms = self._sample_conditional_natoms(goals_raw, seed=seed)

            if self._is_native_flowmol:
                solver = str(self.hparams.get("solver", "heun" if self.sample_timesteps <= 32 else "euler"))
                grid_type = str(self.hparams.get("grid_type", "quadratic_end"))
                raw_samples = self.backbone.sample(
                    n_atoms,
                    device=device,
                    n_steps=self.sample_timesteps,
                    solver=solver,
                    grid_type=grid_type,
                    allowed_elements=self.allowed_elements,
                    cond_values=targets_norm.to(device),
                    cond_active_mask=active_mask.to(device),
                    cfg_scale=float(cfg_scale),
                    xyz_cfg_scale=float(xyz_cfg_scale),
                    use_bf16=bool(use_bf16),
                    seed=seed,
                    build_rdkit=True,
                )
                vocab = self.backbone.config.atom_types
                return [
                    {
                        "atom_symbols": [str(vocab[int(i)]) for i in s["atom_types"].tolist()],
                        "positions_np": np.asarray(s["positions"], dtype=np.float32),
                        "rdkit_mol": s.get("rdkit_mol"),
                    }
                    for s in raw_samples
                ]

            if not self._uses_dgl:
                return self.backbone.sample_trajectories(
                    targets_norm, active_mask, n_atoms=n_atoms, seed=seed
                )

            devices = [device.index] if (device.type == "cuda" and device.index is not None) else []
            with torch.random.fork_rng(devices=devices, enabled=(seed is not None)):
                if seed is not None:
                    torch.manual_seed(int(seed))
                if self.norm_file_path is not None:
                    sampled_mols = self.backbone.sample_random_sizes(
                        len(goals_raw),
                        device=device,
                        n_timesteps=self.sample_timesteps,
                        property_name=prop_name,
                        normalization_file_path=self.norm_file_path,
                        training_mode=False,
                        properties_handle_method=getattr(self.backbone, "properties_handle_method", "concatenate"),
                        multilple_values_to_one_property=goals_raw,
                        number_of_atoms=n_atoms,
                    )
                else:
                    norm_list = [float(x) for x in targets_norm[:, prop_col].detach().cpu().tolist()]
                    sampled_mols = self.backbone.sample_random_sizes(
                        len(norm_list),
                        device=device,
                        n_timesteps=self.sample_timesteps,
                        property_name=prop_name,
                        normalization_file_path=None,
                        training_mode=False,
                        properties_handle_method=getattr(self.backbone, "properties_handle_method", "concatenate"),
                        multilple_values_to_one_property=norm_list,
                        number_of_atoms=n_atoms,
                    )

            allowed_set = frozenset(self.atom_type_map)
            trajs: List[Dict[str, Any]] = []
            for sm in sampled_mols:
                syms = list(sm.atom_types)
                pos_np = sm.positions.detach().cpu().numpy().astype(np.float32)
                if len(syms) == 0 or any(s not in allowed_set for s in syms):
                    # Reject un-unmasked CTMC sentinel tokens (e.g. `'Se'`) cleanly
                    trajs.append(
                        {
                            "atom_symbols": [],
                            "positions_np": np.zeros((0, 3), dtype=np.float32),
                            "rdkit_mol": None,
                        }
                    )
                else:
                    trajs.append(
                        {
                            "atom_symbols": syms,
                            "positions_np": pos_np,
                            "rdkit_mol": sm.rdkit_mol,
                        }
                    )
            return trajs
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate_candidates(
        self,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        atom_budget: Optional[np.ndarray | Sequence[float]] = None,
        n_candidates: Optional[int] = None,
        effort: str = "base",
        seed: Optional[int] = None,
        reconcile_coords: bool = True,
        max_batch_trajs: int = 256,
        **kwargs: Any,
    ) -> List[List[Optional[Chem.Mol]]]:
        """Generate `[batch_size][M]` validated 3D RDKit `Chem.Mol` candidates via `M` outer flow trajectories."""
        from ..chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state

        _, m_keep = resolve_effort_candidates(effort=effort, n_candidates=n_candidates)
        bsz = int(targets_norm.shape[0])
        if bsz == 0:
            return []

        if atom_budget is not None:
            b_1d = np.asarray(atom_budget, dtype=np.float32).reshape(bsz)
        else:
            device = next(self.parameters()).device
            raw_t = self.denormalize_targets(targets_norm.to(device))
            act_any = active_mask.bool().any(dim=0)
            p_col = int(torch.nonzero(act_any, as_tuple=False)[0].item()) if act_any.any() else 0
            goals_r = [float(x) for x in raw_t[:, p_col].detach().cpu().tolist()]
            b_1d = np.asarray(self._sample_conditional_natoms(goals_r, seed=seed), dtype=np.float32)

        reconciler = ValenceReconciler(allowed_elements=self.allowed_elements) if reconcile_coords else None
        allowed_set = frozenset(self.allowed_elements)
        out: List[List[Optional[Chem.Mol]]] = [[None] * m_keep for _ in range(bsz)]
        p_chunk = max(1, int(max_batch_trajs) // max(m_keep, 1))

        for s in range(0, bsz, p_chunk):
            e = min(s + p_chunk, bsz)
            rep_t = targets_norm[s:e].repeat_interleave(m_keep, dim=0)
            rep_m = active_mask[s:e].repeat_interleave(m_keep, dim=0)
            rep_b = np.repeat(b_1d[s:e], m_keep)
            sub_seed = (int(seed) + s * 37) if seed is not None else None
            trajs = self.sample(rep_t, rep_m, seed=sub_seed, leaf_atom_budget=rep_b, **kwargs)
            for idx, tr in enumerate(trajs):
                p_loc, k_i = divmod(idx, m_keep)
                p_glob = s + p_loc
                syms = tr.get("atom_symbols", [])
                pos_np = np.asarray(tr.get("positions_np", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
                if len(syms) < 2 or any(sym not in allowed_set for sym in syms):
                    continue
                rd_m = tr.get("rdkit_mol")
                if rd_m is not None:
                    try:
                        dmat = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=-1)
                        np.fill_diagonal(dmat, 99.0)
                        cand = Chem.Mol(rd_m)
                        if float(dmat.min()) >= 0.65:
                            Chem.SanitizeMol(cand)
                            if (
                                len(Chem.GetMolFrags(cand)) == 1
                                and is_clean_organic_charge_state(cand)
                                and not any(a.GetNumRadicalElectrons() > 0 for a in cand.GetAtoms())
                            ):
                                out[p_glob][k_i] = cand
                                continue
                    except Exception:
                        pass
                if reconciler is not None:
                    res = reconciler.reconcile_point_cloud(syms, pos_np, potential_slot=None)
                    if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
                        out[p_glob][k_i] = res.mol
        return out

    def build_replay_dataset(self, mols: Sequence[Chem.Mol], *, seed: int = 42) -> FlowMolReplayDataset:
        return FlowMolReplayDataset(
            mols,
            self.atom_type_map,
            prior_config=self.prior_config,
            max_atoms=self.max_atoms,
            use_dgl=self._uses_dgl,
            seed=seed,
        )

    def encode_molecules_batch(
        self,
        mols: Sequence[Chem.Mol],
        *,
        seed: int = 42,
        repeat_k: int = 1,
    ) -> Tuple[Optional[Dict[str, Any]], List[int]]:
        if not mols:
            return None, []
        if self._is_native_flowmol:
            from ..flowmol.constants import CHARGE_TO_IDX, mol_to_sparse_record
            from ..flowmol.data import collate_flowmol_batch
            from ..flowmol.priors import sample_aligned_coord_prior_np

            rng = np.random.default_rng(int(seed))
            a_lut = {s: i for i, s in enumerate(self.backbone.config.atom_types)}
            c_lut = {int(q): i for i, q in enumerate(self.backbone.config.formal_charges)}
            sparse_items: List[Dict[str, Any]] = []
            kept_idx: List[int] = []
            for idx, m in enumerate(mols):
                sr = mol_to_sparse_record(m, atom_to_idx=a_lut, charge_to_idx=c_lut, max_atoms=self.max_atoms)
                if sr is None:
                    continue
                na = int(sr["pos"].shape[0])
                n_up = na * (na - 1) // 2
                e_up = np.zeros(n_up, dtype=np.int64)
                if len(sr["bond_u"]) > 0:
                    bu = sr["bond_u"].astype(np.int64)
                    bv = sr["bond_v"].astype(np.int64)
                    up_i = bu * na - (bu * (bu + 1)) // 2 + (bv - bu - 1)
                    e_up[up_i] = sr["bond_type"].astype(np.int64)
                x0 = sample_aligned_coord_prior_np(
                    sr["pos"],
                    rng,
                    prior_std=self.backbone.config.prior_std,
                    size_scale_prior=self.backbone.config.size_scale_prior,
                )
                sparse_items.append(
                    {
                        "n_atoms": na,
                        "x_0": x0,
                        "x_1": sr["pos"],
                        "a_1": sr["atom_type"].astype(np.int64),
                        "c_1": sr["charge"].astype(np.int64),
                        "e_1_upper": e_up,
                    }
                )
                kept_idx.append(idx)
            if not sparse_items:
                return None, []
            rep = max(1, int(repeat_k))
            flow_batch = collate_flowmol_batch(sparse_items * rep)
            return {"mode": "native_flowmol", "batch": flow_batch}, kept_idx * rep

        records: List[Dict[str, torch.Tensor]] = []
        kept_indices: List[int] = []
        for idx, m in enumerate(mols):
            rec = _mol_to_flowmol_record(m, self.atom_to_idx, max_atoms=self.max_atoms)
            if rec is not None:
                records.append(rec)
                kept_indices.append(idx)
        if not records:
            return None, []
        rep = max(1, int(repeat_k))
        expanded_records = records * rep
        expanded_indices = kept_indices * rep
        packed = _collate_records_for_flowmol(
            expanded_records,
            num_atom_types=len(self.atom_type_map),
            prior_config=self.prior_config,
            use_dgl=self._uses_dgl,
        )
        return packed, expanded_indices

    def compute_sft_loss(
        self,
        packed_batch: Dict[str, Any],
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        cfg_dropout_prob: float = 0.0,
        return_per_molecule: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del kwargs
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = targets_norm.device
        targets_norm = targets_norm.to(device)
        if active_mask is not None:
            active_mask = active_mask.to(device)

        if self._is_native_flowmol:
            flow_batch = packed_batch["batch"].to(device)
            bsz = int(flow_batch.batch_size)
            eff_mask = active_mask
            if cfg_dropout_prob > 0.0 and self.training and eff_mask is not None:
                drop = torch.rand((bsz, 1), device=device) < float(cfg_dropout_prob)
                eff_mask = torch.where(drop, torch.zeros_like(eff_mask, dtype=torch.bool), eff_mask.bool())

            k_draws = self.sft_time_draws
            per_mol_accum = torch.zeros((bsz,), device=device, dtype=torch.float32)
            feat_means = {"x": 0.0, "a": 0.0, "c": 0.0, "e": 0.0}
            for d_idx in range(k_draws):
                u = torch.rand((bsz,), device=device, dtype=torch.float32)
                t = ((float(d_idx) + u) / float(k_draws)).clamp(0.002, 0.998)
                pm_loss, m_info = self.backbone.compute_loss(
                    flow_batch,
                    t=t,
                    cond_values=targets_norm,
                    cond_active_mask=eff_mask,
                    return_per_molecule=True,
                )
                per_mol_accum = per_mol_accum + pm_loss / float(k_draws)
                for fk in feat_means:
                    feat_means[fk] += float(m_info[f"loss_{fk}"].item()) / float(k_draws)

            info_nat: Dict[str, Any] = {f"loss_{k}": v for k, v in feat_means.items()}
            info_nat["loss_total"] = float(per_mol_accum.detach().mean().item())
            if return_per_molecule:
                info_nat["per_molecule_generation_loss"] = per_mol_accum
                return per_mol_accum, info_nat
            return per_mol_accum.mean(), info_nat

        if not self._uses_dgl:
            return self.backbone.compute_flow_loss(
                packed_batch, targets_norm, active_mask, return_per_molecule=return_per_molecule
            )

        from propmolflow.data_processing.utils import get_batch_idxs, get_upper_edge_mask

        g = packed_batch["graph"].to(device)
        bsz = int(g.batch_size)
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)
        upper_edge_mask = get_upper_edge_mask(g)
        ue_batch = edge_batch_idx[upper_edge_mask]

        prop_col = 0
        if active_mask is not None:
            act_any = active_mask.bool().any(dim=0)
            if act_any.any():
                prop_col = int(torch.nonzero(act_any, as_tuple=False)[0].item())
        prop_vals = targets_norm[:, prop_col : prop_col + 1].float().to(device)
        if hasattr(self.backbone, "property_embedder") and self.backbone.property_embedder is not None:
            self.backbone.property_embedder = self.backbone.property_embedder.to(device)

        lw = dict(self.backbone.total_loss_weights)
        f_ord = list(self.backbone.canonical_feat_order)
        k_draws = self.sft_time_draws
        per_mol_accum = torch.zeros(bsz, device=device, dtype=torch.float32)
        feat_means: Dict[str, float] = {f: 0.0 for f in f_ord}

        for d_idx in range(k_draws):
            u = torch.rand(bsz, device=device, dtype=torch.float32)
            t = ((float(d_idx) + u) / float(k_draws)).clamp(1e-4, 1.0 - 1e-4)

            with g.local_scope():
                g_t = self.backbone.vector_field.sample_conditional_path(
                    g, t, node_batch_idx, edge_batch_idx, upper_edge_mask
                )
                if getattr(self.backbone, "conditional_generation", True):
                    prop_emb = self.backbone.property_embedder(prop_vals)
                    t_combined = torch.cat([t.unsqueeze(-1), prop_emb], dim=-1)
                else:
                    t_combined = t

                vf_out = self.backbone.vector_field(
                    g_t, t_combined, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask
                )

                alpha_t_prime = self.backbone.interpolant_scheduler.alpha_t_prime(t)
                use_time_w = bool(getattr(self.backbone, "time_scaled_loss", False))
                time_w_all = self.backbone.interpolant_scheduler.loss_weights(t) if use_time_w else None

                draw_mol_loss = torch.zeros(bsz, device=device, dtype=torch.float32)
                for feat_idx, feat in enumerate(f_ord):
                    w_f = float(lw[feat])
                    if feat == "x":
                        pred_x = vf_out["x"].float()
                        if self.backbone.parameterization in ("endpoint", "dirichlet", "ctmc"):
                            tgt_x = g_t.ndata["x_1_true"].float()
                        else:
                            a_p = alpha_t_prime[:, feat_idx][node_batch_idx].unsqueeze(-1)
                            tgt_x = a_p * (g_t.ndata["x_1_true"].float() - g_t.ndata["x_0"].float())
                        err_node = ((pred_x - tgt_x) ** 2).mean(dim=-1)
                        if time_w_all is not None:
                            err_node = err_node * time_w_all[:, feat_idx][node_batch_idx].float()
                        sum_m = torch.zeros(bsz, device=device, dtype=torch.float32).index_add(0, node_batch_idx, err_node)
                        cnt_m = torch.bincount(node_batch_idx, minlength=bsz).float().clamp_min(1.0)
                        loss_m = sum_m / cnt_m
                    else:
                        logits = vf_out[feat].float()
                        if feat == "e":
                            tgt_cls = g_t.edata["e_1_true"][upper_edge_mask].argmax(dim=-1)
                            xt_cls = g_t.edata["e_t"][upper_edge_mask].argmax(dim=-1)
                            b_idx = ue_batch
                        else:
                            tgt_cls = g_t.ndata[f"{feat}_1_true"].argmax(dim=-1)
                            xt_cls = g_t.ndata[f"{feat}_t"].argmax(dim=-1)
                            b_idx = node_batch_idx

                        if self.backbone.parameterization == "ctmc":
                            mask_id = int(self.backbone.n_cat_dict[feat])
                            valid_pos = xt_cls == mask_id
                        else:
                            valid_pos = torch.ones_like(tgt_cls, dtype=torch.bool)

                        if bool(valid_pos.any().item()):
                            class_w = None
                            if feat in ("a", "e") and bool(getattr(self.backbone, "weight_ae", False)):
                                p_marg = getattr(self.backbone, f"p_{feat}", None)
                                if p_marg is not None and int(p_marg.numel()) == int(logits.shape[-1]):
                                    class_w = (1.0 - p_marg).to(device=device, dtype=torch.float32)
                            ce_all = F.cross_entropy(
                                logits[valid_pos], tgt_cls[valid_pos], weight=class_w, reduction="none"
                            )
                            b_v = b_idx[valid_pos]
                            if time_w_all is not None:
                                ce_all = ce_all * time_w_all[:, feat_idx][b_v].float()
                            sum_m = torch.zeros(bsz, device=device, dtype=torch.float32).index_add(0, b_v, ce_all)
                            cnt_m = torch.bincount(b_v, minlength=bsz).float().clamp_min(1.0)
                            loss_m = sum_m / cnt_m
                        else:
                            loss_m = torch.zeros(bsz, device=device, dtype=torch.float32)

                    draw_mol_loss = draw_mol_loss + w_f * loss_m
                    feat_means[feat] += float(loss_m.detach().mean().item()) / float(k_draws)

            per_mol_accum = per_mol_accum + draw_mol_loss / float(k_draws)

        info: Dict[str, Any] = {f"loss_{f}": feat_means[f] for f in f_ord}
        info["loss_total"] = float(per_mol_accum.detach().mean().item())
        if return_per_molecule:
            info["per_molecule_generation_loss"] = per_mol_accum
            return per_mol_accum, info
        return per_mol_accum.mean(), info

    def trajectory_distributions(
        self,
        trajectories: List[Dict[str, Any]],
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        temperatures: Tuple[float, float, float] = (0.85, 0.30, 0.25),
        cfg_scale: float = 1.0,
        xyz_cfg_scale: float = 1.0,
        cfg_tree_decay: float = 0.0,
        geometric_guard: bool = False,
        electron_parity: Optional[int] = 0,
        **kwargs: Any,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        raise NotImplementedError(
            "Deterministic ODE/CTMC flow generator does not expose exact stepwise log-probs; "
            "use LoomGRPOTrainer(alignment_mode='awr_sft') or an SDE flow-policy wrapper."
        )

    @classmethod
    def from_pretrained_flowmol(
        cls,
        checkpoint_path: Path | str,
        *,
        target_cols: Sequence[str],
        target_units: Sequence[str],
        mean: Sequence[float],
        scale: Sequence[float],
        device: str = "cuda:0",
        sample_timesteps: int = 25,
        sft_time_draws: int = 2,
        max_atoms: int = 120,
        allowed_elements: Optional[Sequence[str]] = None,
        lora_rank: int = 0,
        lora_alpha: float = 32.0,
    ) -> "LoomFlowMolGenerator":
        """Load a DGL-free `uniloom_flowmol_v1` pretrained checkpoint and attach `LoomConditioner` (+ optional LoRA)."""
        from ..flowmol.gvp import inject_flowmol_lora
        from ..flowmol.model import FlowMolConfig, FlowMolModel

        ck = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        cfg = FlowMolConfig.from_dict(ck["config"])
        backbone = FlowMolModel(cfg)
        state = ck.get("ema_state_dict") or ck.get("model_state_dict")
        backbone.load_state_dict(state, strict=False)
        if backbone.conditioner is None:
            backbone.attach_conditioner(num_props=len(target_cols))
        if lora_rank > 0:
            inject_flowmol_lora(backbone, rank=int(lora_rank), alpha=float(lora_alpha))

        # Populate empirical size sampler if checkpoint carries `n_atoms_hist`
        hist = ck.get("n_atoms_hist") or {}
        if hist:
            keys = np.asarray([int(k) for k in hist.keys()], dtype=np.int64)
            cnts = np.asarray([float(v) for v in hist.values()], dtype=np.float64)
            probs = cnts / cnts.sum()

            def _sample_n_atoms(n: int) -> torch.Tensor:
                drawn = np.random.default_rng().choice(keys, size=int(n), replace=True, p=probs)
                return torch.from_numpy(drawn)

            backbone.sample_n_atoms = _sample_n_atoms  # type: ignore[attr-defined]

        slot = cls(
            backbone=backbone,
            atom_type_map=cfg.atom_types,
            target_cols=target_cols,
            target_units=target_units,
            mean=mean,
            scale=scale,
            sample_timesteps=sample_timesteps,
            sft_time_draws=sft_time_draws,
            max_atoms=max_atoms,
            hparams={"format": "uniloom_flowmol_v1", "config": cfg.to_dict()},
            epoch=int(ck.get("epoch", 0)),
            global_step=int(ck.get("step", 0)),
        ).to(device)

        eff_allowed = allowed_elements if allowed_elements is not None else ck.get("active_elements")
        if eff_allowed:
            slot.set_allowed_elements(tuple(eff_allowed))
        return slot

    @classmethod
    def from_propmolflow_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        device: str = "cuda:0",
        data_dir: Optional[Path | str] = None,
        property_name: str = "homo",
        sample_timesteps: int = 100,
        sft_time_draws: int = 2,
        max_atoms: int = 80,
        propmolflow_root: Optional[Path | str] = None,
    ) -> "LoomFlowMolGenerator":
        _ensure_propmolflow_paths(propmolflow_root)
        from propmolflow.models.ctmc_vector_field import PROPERTY_MAP
        from propmolflow.models.flowmol import FlowMol

        ckpt_p = Path(checkpoint_path)
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        hp = copy.deepcopy(ck["hyper_parameters"])

        resolved_data = Path(data_dir) if data_dir is not None else ckpt_p.parent.parent / "data"
        norm_file = resolved_data / "train_data_property_normalization.pt"
        hist_file = resolved_data / "train_data_n_atoms_histogram.pt"
        marg_file = resolved_data / "train_data_marginal_dists.pt"
        if hist_file.exists():
            hp["n_atoms_hist_file"] = str(hist_file)
        if marg_file.exists():
            hp["marginal_dists_file"] = str(marg_file)

        model = FlowMol(**copy.deepcopy(hp)).to(device)
        model.load_state_dict(ck["state_dict"], strict=True)

        p_idx = int(PROPERTY_MAP[property_name])
        if norm_file.exists():
            nd = torch.load(norm_file, map_location="cpu", weights_only=False)
            mean_val = float(nd["mean"][p_idx].item())
            scale_val = max(float(nd["std"][p_idx].item()), 1e-6)
            norm_path_str: Optional[str] = str(norm_file)
        else:
            mean_val, scale_val, norm_path_str = 0.0, 1.0, None

        prop_vals_np: Optional[np.ndarray] = None
        prop_nat_np: Optional[np.ndarray] = None
        proc_file = resolved_data / "train_data_processed.pt"
        if proc_file.exists():
            tr_d = torch.load(proc_file, map_location="cpu", weights_only=False)
            key = f"{property_name}_eV"
            if key in tr_d and "n_atoms" in tr_d:
                prop_vals_np = tr_d[key].numpy()
                prop_nat_np = tr_d["n_atoms"].numpy()

        atom_map = tuple(hp.get("atom_type_map", OE62_DEFAULT_ATOM_MAP))
        return cls(
            backbone=model,
            atom_type_map=atom_map,
            target_cols=[property_name],
            target_units=["eV"],
            mean=[mean_val],
            scale=[scale_val],
            sample_timesteps=sample_timesteps,
            sft_time_draws=sft_time_draws,
            max_atoms=max_atoms,
            norm_file_path=norm_path_str,
            prop_histogram_vals=prop_vals_np,
            prop_histogram_natoms=prop_nat_np,
            prior_config=hp.get("prior_config"),
            hparams=hp,
            epoch=int(ck.get("epoch", 0)),
            global_step=int(ck.get("global_step", 0)),
            cum_gpu_hours=float(ck.get("cum_gpu_hours", 0.0)),
        ).to(device)

    def save_propmolflow_checkpoint(
        self,
        output_path: Path | str,
        *,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save aligned `FlowMol` weights in exact `eval_checkpoint`-compatible payload format."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "hyper_parameters": copy.deepcopy(self.hparams),
            "state_dict": {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()},
            "epoch": self.epoch,
            "global_step": self.global_step,
            "cum_gpu_hours": round(self.cum_gpu_hours, 4),
            "val_loss": None,
        }
        if extra_metadata is not None:
            payload["loom_awr_metadata"] = dict(extra_metadata)
        tmp = out_p.with_suffix(".tmp.pt")
        torch.save(payload, tmp)
        tmp.replace(out_p)
        return out_p
