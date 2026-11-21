"""Standalone DGL-free SE(3)-equivariant full-periodic-table FlowMol 3D foundation model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint as grad_ckpt

from .constants import (
    FLOWMOL_ATOM_TYPES,
    FLOWMOL_FORMAL_CHARGES,
    N_BOND_TYPES,
    N_CHARGES,
    PERIODIC_FEATURE_DIM,
    build_allowed_charge_mask,
    build_allowed_element_mask,
    build_periodic_feature_tensor,
    tensors_to_rdkit_mol,
)
from .data import FlowMolBatch, build_complete_graph_batch
from .gvp import (
    EdgeUpdate,
    GVPConv,
    NodePositionUpdate,
    _rbf_fp32,
    compute_edge_geometry_fp32,
)
from .priors import apply_train_geometry_distortion_fp32, remove_com_fp32
from .scheduler import FlowInterpolantScheduler, build_inference_tspan, sample_train_timesteps


@dataclass
class FlowMolConfig:
    """Architecture and flow-matching hyperparameters for full-atom general FlowMol pretraining."""

    atom_types: Tuple[str, ...] = FLOWMOL_ATOM_TYPES
    formal_charges: Tuple[int, ...] = FLOWMOL_FORMAL_CHARGES
    n_charges: int = N_CHARGES
    n_bond_types: int = N_BOND_TYPES

    # Backbone dimensions
    n_hidden_scalars: int = 320
    n_hidden_edge_feats: int = 192
    n_vec_channels: int = 24
    n_cp_feats: int = 6
    n_molecule_updates: int = 9
    convs_per_update: int = 1
    n_message_gvps: int = 3
    n_update_gvps: int = 3
    n_pos_update_gvps: int = 3
    use_dst_feats: bool = False
    separate_mol_updaters: bool = True
    update_edge_w_distance: bool = True
    message_norm: float | str = 100.0
    rbf_dim: int = 32
    rbf_dmax: float = 20.0
    time_emb_dim: int = 64
    dropout: float = 0.0

    # Full-periodic-table & FlowMol3 mechanisms
    use_periodic_features: bool = True
    self_conditioning: bool = True
    self_conditioning_prob: float = 0.5
    geometry_distortion_prob: float = 0.25
    geometry_distortion_max: float = 0.20
    geometry_distortion_min_t: float = 0.30

    # Schedule, prior, & per-molecule loss weights
    schedule_type: str = "cosine"
    nu_x: float = 1.0
    nu_a: float = 2.0
    nu_c: float = 2.0
    nu_e: float = 1.5
    timestep_sampling: str = "poly_mix"
    time_scaled_loss: bool = False
    loss_weight_x: float = 3.0
    loss_weight_a: float = 0.4
    loss_weight_c: float = 1.0
    loss_weight_e: float = 2.0
    prior_std: float = 1.15
    size_scale_prior: bool = True

    # Memory / training controls
    grad_checkpointing: bool = False

    @property
    def n_atom_types(self) -> int:
        return len(self.atom_types)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["atom_types"] = list(self.atom_types)
        d["formal_charges"] = list(self.formal_charges)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowMolConfig":
        kw = dict(data)
        if "atom_types" in kw:
            kw["atom_types"] = tuple(kw["atom_types"])
        if "formal_charges" in kw:
            kw["formal_charges"] = tuple(int(q) for q in kw["formal_charges"])
            kw["n_charges"] = len(kw["formal_charges"])
        elif kw.get("n_charges") == 6:
            kw["formal_charges"] = (-2, -1, 0, 1, 2, 3)
        return cls(**kw)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "FlowMolConfig":
        name_l = name.lower()
        if name_l == "tiny":
            base = dict(
                n_hidden_scalars=64,
                n_hidden_edge_feats=32,
                n_vec_channels=8,
                n_cp_feats=2,
                n_molecule_updates=2,
                convs_per_update=1,
                n_message_gvps=2,
                n_update_gvps=2,
                n_pos_update_gvps=2,
                rbf_dim=16,
                rbf_dmax=20.0,
            )
        elif name_l == "small":
            # ~4.23M parameters (FlowMol / FlowMol3 ~4.2M baseline scale; 3-tier Scaling Law anchor)
            base = dict(
                n_hidden_scalars=224,
                n_hidden_edge_feats=128,
                n_vec_channels=16,
                n_cp_feats=4,
                n_molecule_updates=6,
                convs_per_update=1,
                n_message_gvps=3,
                n_update_gvps=3,
                n_pos_update_gvps=3,
                rbf_dim=32,
                rbf_dmax=20.0,
            )
        elif name_l == "standard":
            # ~12.74M parameters (~3.0x small; Scaling Law mid-tier backbone)
            base = dict(
                n_hidden_scalars=320,
                n_hidden_edge_feats=192,
                n_vec_channels=24,
                n_cp_feats=6,
                n_molecule_updates=9,
                convs_per_update=1,
                n_message_gvps=3,
                n_update_gvps=3,
                n_pos_update_gvps=3,
                rbf_dim=32,
                rbf_dmax=20.0,
            )
        elif name_l == "flagship":
            # ~40.19M parameters (~3.15x standard, ~9.5x small; Scaling Law flagship backbone)
            base = dict(
                n_hidden_scalars=512,
                n_hidden_edge_feats=256,
                n_vec_channels=32,
                n_cp_feats=8,
                n_molecule_updates=12,
                convs_per_update=1,
                n_message_gvps=3,
                n_update_gvps=3,
                n_pos_update_gvps=3,
                rbf_dim=48,
                rbf_dmax=20.0,
            )
        else:
            raise ValueError(f"Unknown FlowMol preset '{name}'. Expected tiny|small|standard|flagship.")
        base.update(overrides)
        return cls(**base)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = int(dim)
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=t.device.type, enabled=False):
            tf = t.float().unsqueeze(-1)
            ang = tf * self.freqs.unsqueeze(0)
            emb = torch.cat([tf, torch.sin(ang), torch.cos(ang)], dim=-1)
        return emb


class _MoleculeUpdateBlock(nn.Module):
    """Single checkpointable molecule update block (K GVPConvs + PositionUpdate + EdgeUpdate)."""

    def __init__(
        self,
        convs: Sequence[GVPConv],
        pos_updater: NodePositionUpdate,
        edge_updater: EdgeUpdate,
        rbf_dmax: float,
        rbf_dim: int,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.pos_updater = pos_updater
        self.edge_updater = edge_updater
        self.rbf_dmax = float(rbf_dmax)
        self.rbf_dim = int(rbf_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        e: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        unit_diff: torch.Tensor,
        rbf: torch.Tensor,
        mod_scale: Optional[torch.Tensor] = None,
        mod_shift: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mod_scale is not None and mod_shift is not None:
            h = h * mod_scale.to(h.dtype) + mod_shift.to(h.dtype)
        for conv in self.convs:
            h, v = conv(h, v, e, src, dst, unit_diff, rbf)
        x = self.pos_updater(h, x, v)
        unit_diff, _, rbf = compute_edge_geometry_fp32(x, src, dst, rbf_dmax=self.rbf_dmax, rbf_dim=self.rbf_dim)
        e = self.edge_updater(h, e, src, dst, rbf)
        return h, x, v, e, unit_diff, rbf


class FlowMolModel(nn.Module):
    """DGL-free SE(3)-equivariant full-periodic-table CTMC + continuous coordinate Flow Matching model."""

    def __init__(self, config: FlowMolConfig) -> None:
        super().__init__()
        self.config = config
        self.n_atom_types = config.n_atom_types
        self.n_charges = config.n_charges
        self.n_bond_types = config.n_bond_types
        self.atom_mask_idx = self.n_atom_types
        self.charge_mask_idx = self.n_charges
        self.bond_mask_idx = self.n_bond_types

        self.scheduler = FlowInterpolantScheduler(
            schedule_type=config.schedule_type,
            nu_params={"x": config.nu_x, "a": config.nu_a, "c": config.nu_c, "e": config.nu_e},
        )
        self.time_embedder = SinusoidalTimeEmbedding(config.time_emb_dim)

        # Frozen periodic-table physical invariant table: (n_atom_types + 1, 34) with row[-1] = 0 for [MASK]
        self.register_buffer(
            "periodic_table_feats",
            build_periodic_feature_tensor(config.atom_types),
            persistent=True,
        )

        time_out_dim = 1 + 2 * (config.time_emb_dim // 2)
        node_in_dim = (self.n_atom_types + 1) + (self.n_charges + 1) + time_out_dim
        if config.use_periodic_features:
            node_in_dim += PERIODIC_FEATURE_DIM

        edge_in_dim = self.n_bond_types + 1
        if config.self_conditioning:
            node_in_dim += self.n_atom_types + self.n_charges
            if config.use_periodic_features:
                node_in_dim += PERIODIC_FEATURE_DIM
            edge_in_dim += self.n_bond_types + config.rbf_dim
            self.sc_vec_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        else:
            self.register_parameter("sc_vec_scale", None)

        self.scalar_embedding = nn.Sequential(
            nn.Linear(node_in_dim, config.n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(config.n_hidden_scalars, config.n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(config.n_hidden_scalars),
        )
        # Zero-initialized Stage 5 fragment/anchor conditioning hook: [is_fixed_node, is_growth_site]
        self.fragment_cond_proj = nn.Linear(2, config.n_hidden_scalars, bias=False)
        nn.init.zeros_(self.fragment_cond_proj.weight)

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_in_dim, config.n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(config.n_hidden_edge_feats, config.n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(config.n_hidden_edge_feats),
        )

        shared_pos = None if config.separate_mol_updaters else NodePositionUpdate(
            config.n_hidden_scalars,
            config.n_vec_channels,
            n_gvps=config.n_pos_update_gvps,
            n_cp_feats=config.n_cp_feats,
        )
        shared_edge = None if config.separate_mol_updaters else EdgeUpdate(
            config.n_hidden_scalars,
            config.n_hidden_edge_feats,
            update_edge_w_distance=config.update_edge_w_distance,
            rbf_dim=config.rbf_dim,
        )

        blocks: List[_MoleculeUpdateBlock] = []
        for _ in range(config.n_molecule_updates):
            convs = [
                GVPConv(
                    scalar_size=config.n_hidden_scalars,
                    vector_size=config.n_vec_channels,
                    n_cp_feats=config.n_cp_feats,
                    n_message_gvps=config.n_message_gvps,
                    n_update_gvps=config.n_update_gvps,
                    use_dst_feats=config.use_dst_feats,
                    rbf_dim=config.rbf_dim,
                    edge_feat_size=config.n_hidden_edge_feats,
                    message_norm=config.message_norm,
                    dropout=config.dropout,
                )
                for _ in range(config.convs_per_update)
            ]
            pos_upd = shared_pos if shared_pos is not None else NodePositionUpdate(
                config.n_hidden_scalars,
                config.n_vec_channels,
                n_gvps=config.n_pos_update_gvps,
                n_cp_feats=config.n_cp_feats,
            )
            edge_upd = shared_edge if shared_edge is not None else EdgeUpdate(
                config.n_hidden_scalars,
                config.n_hidden_edge_feats,
                update_edge_w_distance=config.update_edge_w_distance,
                rbf_dim=config.rbf_dim,
            )
            blocks.append(_MoleculeUpdateBlock(convs, pos_upd, edge_upd, config.rbf_dmax, config.rbf_dim))
        self.blocks = nn.ModuleList(blocks)

        self.node_output_head = nn.Sequential(
            nn.Linear(config.n_hidden_scalars, config.n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(config.n_hidden_scalars, self.n_atom_types + self.n_charges),
        )
        # Factored Periodic-Table bilinear residual head so rare/unseen elements generalize via
        # shared covalent-radius / group / period / s-p-d-f block structure
        if config.use_periodic_features:
            self.periodic_logit_proj: Optional[nn.Linear] = nn.Linear(
                config.n_hidden_scalars, PERIODIC_FEATURE_DIM, bias=False
            )
            nn.init.normal_(self.periodic_logit_proj.weight, mean=0.0, std=0.02)
        else:
            self.periodic_logit_proj = None

        self.to_edge_logits = nn.Sequential(
            nn.Linear(config.n_hidden_edge_feats, config.n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(config.n_hidden_edge_feats, self.n_bond_types),
        )

        self.conditioner: Optional[nn.Module] = None

    def attach_conditioner(
        self,
        num_props: int = 1,
        *,
        conditioner: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Attach an optional zero-initialized property conditioner (`conditioner` or default AdaLN conditioner)."""
        if conditioner is not None:
            self.conditioner = conditioner
            return conditioner
        from loom.weaving.conditioner import LoomConditioner

        cond = LoomConditioner(
            emb_dim=self.config.n_hidden_scalars,
            num_props=int(num_props),
            num_layers=self.config.n_molecule_updates,
            **kwargs,
        )
        self.conditioner = cond
        return cond

    def save_pretrained(
        self,
        checkpoint_path: Path | str,
        *,
        step: int = 0,
        samples_seen: int = 0,
        corpus_meta: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save standalone `flowmol_foundation_v1` weights + empirical corpus distributions atomically."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = corpus_meta or {}
        payload: Dict[str, Any] = {
            "format": "flowmol_foundation_v1",
            "step": int(step),
            "samples_seen": int(samples_seen),
            "config": self.config.to_dict(),
            "active_elements": [k for k, v in meta.get("element_counts", {}).items() if int(v) > 0],
            "active_charges": [int(k) for k, v in meta.get("charge_counts", {}).items() if int(v) > 0],
            "n_atoms_hist": meta.get("n_atoms_hist", {}),
            "per_source_n_atoms_hist": meta.get("per_source_n_atoms_hist", {}),
            "per_source_active_elements": meta.get("per_source_active_elements", {}),
            "source_counts": meta.get("source_counts", {}),
            "model_state_dict": self.state_dict(),
        }
        if extra:
            payload.update(extra)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Path | str,
        *,
        device: str | torch.device = "cpu",
        use_ema: bool = True,
        strict: bool = True,
    ) -> Tuple["FlowMolModel", Dict[str, Any]]:
        """Load a standalone pretrained `FlowMolModel` and its checkpoint metadata (zero external coupling)."""
        ck = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        cfg = FlowMolConfig.from_dict(ck["config"])
        model = cls(cfg)
        state = (ck.get("ema_state_dict") or ck.get("model_state_dict")) if use_ema else ck["model_state_dict"]
        model.load_state_dict(state, strict=strict)
        model = model.to(device).eval()
        meta = {k: v for k, v in ck.items() if k not in ("model_state_dict", "ema_state_dict", "optimizer_state_dict")}
        return model, meta

    @staticmethod
    def sample_empirical_sizes(
        meta: Dict[str, Any],
        n_mols: int,
        *,
        domain: Optional[str] = None,
        seed: int = 42,
        default_size: int = 30,
    ) -> List[int]:
        """Sample molecule sizes `N` from empirical `n_atoms_hist` (overall or domain-specific e.g. `'tmqm'`)."""
        hist: Dict[Any, Any] = {}
        if domain and domain != "all":
            hist = (meta.get("per_source_n_atoms_hist") or {}).get(domain, {})
        if not hist:
            hist = meta.get("n_atoms_hist") or {}
        if not hist:
            return [int(default_size)] * int(n_mols)
        keys = np.asarray([int(k) for k in hist.keys()], dtype=np.int64)
        vals = np.asarray([float(v) for v in hist.values()], dtype=np.float64)
        probs = vals / vals.sum()
        rng = np.random.default_rng(int(seed))
        return [int(x) for x in rng.choice(keys, size=int(n_mols), replace=True, p=probs).tolist()]

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}

    def forward(
        self,
        batch: FlowMolBatch,
        x_t: torch.Tensor,
        a_t_idx: torch.Tensor,
        c_t_idx: torch.Tensor,
        e_t_upper_idx: torch.Tensor,
        t: torch.Tensor,
        *,
        sc_dict: Optional[Dict[str, torch.Tensor]] = None,
        cond_values: Optional[torch.Tensor] = None,
        cond_active_mask: Optional[torch.Tensor] = None,
        node_cond_flags: Optional[torch.Tensor] = None,
        remove_com: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Predict clean endpoint `{'x': (N,3) fp32, 'a': (N,A), 'c': (N,C), 'e': (E_up,B)}`."""
        device = x_t.device
        n_nodes = x_t.shape[0]
        n_edges = batch.src.shape[0]

        a_oh = F.one_hot(a_t_idx, num_classes=self.n_atom_types + 1).float()
        c_oh = F.one_hot(c_t_idx, num_classes=self.n_charges + 1).float()
        t_emb = self.time_embedder(t)[batch.node_batch_idx]

        e_up_oh = F.one_hot(e_t_upper_idx, num_classes=self.n_bond_types + 1).float()
        e_dir_oh = torch.empty((n_edges, self.n_bond_types + 1), dtype=torch.float32, device=device)
        e_dir_oh[batch.upper_edge_mask] = e_up_oh
        e_dir_oh[~batch.upper_edge_mask] = e_up_oh

        node_parts = [a_oh, c_oh, t_emb]
        if self.config.use_periodic_features:
            node_parts.append(self.periodic_table_feats[a_t_idx])
        edge_parts = [e_dir_oh]

        v = torch.zeros((n_nodes, self.config.n_vec_channels, 3), dtype=torch.float32, device=device)

        if self.config.self_conditioning:
            if sc_dict is not None:
                a_sc = sc_dict["a"].float()
                c_sc = sc_dict["c"].float()
                e_sc_up = sc_dict["e"].float()
                e_sc_dir = torch.empty((n_edges, self.n_bond_types), dtype=torch.float32, device=device)
                e_sc_dir[batch.upper_edge_mask] = e_sc_up
                e_sc_dir[~batch.upper_edge_mask] = e_sc_up

                x_sc = sc_dict["x"].float()
                with torch.autocast(device_type=device.type, enabled=False):
                    sc_diff = x_sc[batch.src] - x_sc[batch.dst]
                    sc_dist = torch.sqrt(torch.sum(sc_diff * sc_diff, dim=-1).clamp_min(1e-8))
                    rbf_sc = _rbf_fp32(sc_dist, d_min=0.0, d_max=self.config.rbf_dmax, d_count=self.config.rbf_dim)
                    if self.sc_vec_scale is not None:
                        v[:, 0, :] = self.sc_vec_scale.float() * (x_sc - x_t.float())
                    phys_sc = (
                        a_sc @ self.periodic_table_feats[: self.n_atom_types].float()
                        if self.config.use_periodic_features
                        else None
                    )
            else:
                a_sc = torch.zeros((n_nodes, self.n_atom_types), dtype=torch.float32, device=device)
                c_sc = torch.zeros((n_nodes, self.n_charges), dtype=torch.float32, device=device)
                e_sc_dir = torch.zeros((n_edges, self.n_bond_types), dtype=torch.float32, device=device)
                rbf_sc = torch.zeros((n_edges, self.config.rbf_dim), dtype=torch.float32, device=device)
                phys_sc = (
                    torch.zeros((n_nodes, PERIODIC_FEATURE_DIM), dtype=torch.float32, device=device)
                    if self.config.use_periodic_features
                    else None
                )
                if self.sc_vec_scale is not None:
                    v[:, 0, :] = self.sc_vec_scale.float() * 0.0
            node_parts.extend([a_sc, c_sc])
            if phys_sc is not None:
                node_parts.append(phys_sc)
            edge_parts.extend([e_sc_dir, rbf_sc])

        h = self.scalar_embedding(torch.cat(node_parts, dim=-1))
        if node_cond_flags is not None:
            h = h + self.fragment_cond_proj(node_cond_flags.to(h.dtype))
        else:
            # Keep fragment_cond_proj connected with exact zero contribution for DDP safety
            h = h + (self.fragment_cond_proj.weight.sum() * 0.0).to(h.dtype)

        e = self.edge_embedding(torch.cat(edge_parts, dim=-1))
        x = x_t.float()

        unit_diff, _, rbf = compute_edge_geometry_fp32(
            x, batch.src, batch.dst, rbf_dmax=self.config.rbf_dmax, rbf_dim=self.config.rbf_dim
        )

        layer_mods: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        out_mod: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.conditioner is not None and cond_values is not None and cond_active_mask is not None:
            global_c, _ = self.conditioner.encode_condition(cond_values, cond_active_mask)
            layer_mods, out_mod = self.conditioner.precompute_modulations(global_c, h.dtype)

        use_ckpt = self.config.grad_checkpointing and self.training and torch.is_grad_enabled()
        for blk_idx, block in enumerate(self.blocks):
            ms = layer_mods[blk_idx][0][batch.node_batch_idx] if layer_mods is not None else None
            mb = layer_mods[blk_idx][1][batch.node_batch_idx] if layer_mods is not None else None
            if use_ckpt:
                h, x, v, e, unit_diff, rbf = grad_ckpt(
                    block,
                    h, x, v, e, batch.src, batch.dst, unit_diff, rbf, ms, mb,
                    use_reentrant=False,
                )
            else:
                h, x, v, e, unit_diff, rbf = block(
                    h, x, v, e, batch.src, batch.dst, unit_diff, rbf, ms, mb
                )

        if out_mod is not None:
            h = h * out_mod[0][batch.node_batch_idx].to(h.dtype) + out_mod[1][batch.node_batch_idx].to(h.dtype)

        node_logits = self.node_output_head(h)
        atom_logits = node_logits[:, : self.n_atom_types]
        charge_logits = node_logits[:, self.n_atom_types :]
        if self.periodic_logit_proj is not None:
            p_lat = self.periodic_logit_proj(h).float()
            atom_logits = atom_logits.float() + p_lat @ self.periodic_table_feats[: self.n_atom_types].float().T

        ue_feats = e[batch.upper_edge_mask]
        le_feats = e[~batch.upper_edge_mask]
        edge_logits = self.to_edge_logits(ue_feats + le_feats)

        if remove_com:
            x = remove_com_fp32(x, batch.node_batch_idx, batch.batch_size, batch.n_atoms)

        return {
            "x": x,
            "a": atom_logits,
            "c": charge_logits,
            "e": edge_logits,
        }

    def sample_conditional_path(
        self,
        batch: FlowMolBatch,
        t: torch.Tensor,
        *,
        apply_distortion: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample `(x_t, a_t_idx, c_t_idx, e_t_upper_idx)` at time `t` strictly in FP32."""
        assert batch.x_1 is not None and batch.a_1 is not None and batch.c_1 is not None and batch.e_1_upper is not None
        device = batch.x_1.device
        with torch.autocast(device_type=device.type, enabled=False):
            alpha_t = self.scheduler.alpha_t(t)
            ax = alpha_t[:, 0][batch.node_batch_idx].unsqueeze(-1)
            x_t = (1.0 - ax) * batch.x_0.float() + ax * batch.x_1.float()
            x_t = remove_com_fp32(x_t, batch.node_batch_idx, batch.batch_size, batch.n_atoms)

            if apply_distortion and self.training:
                x_t = apply_train_geometry_distortion_fp32(
                    x_t,
                    t,
                    batch.node_batch_idx,
                    batch.batch_size,
                    prob=self.config.geometry_distortion_prob,
                    max_distortion=self.config.geometry_distortion_max,
                    min_t=self.config.geometry_distortion_min_t,
                )

            aa = alpha_t[:, 1][batch.node_batch_idx]
            ac = alpha_t[:, 2][batch.node_batch_idx]
            ae = alpha_t[:, 3][batch.upper_edge_batch_idx]

            a_t = torch.where(torch.rand_like(aa) < (1.0 - aa), torch.full_like(batch.a_1, self.atom_mask_idx), batch.a_1)
            c_t = torch.where(torch.rand_like(ac) < (1.0 - ac), torch.full_like(batch.c_1, self.charge_mask_idx), batch.c_1)
            e_t = torch.where(
                torch.rand_like(ae) < (1.0 - ae),
                torch.full_like(batch.e_1_upper, self.bond_mask_idx),
                batch.e_1_upper,
            )
        return x_t, a_t, c_t, e_t

    def compute_loss(
        self,
        batch: FlowMolBatch,
        *,
        t: Optional[torch.Tensor] = None,
        use_self_conditioning: Optional[bool] = None,
        apply_distortion: Optional[bool] = None,
        cond_values: Optional[torch.Tensor] = None,
        cond_active_mask: Optional[torch.Tensor] = None,
        node_cond_flags: Optional[torch.Tensor] = None,
        return_per_molecule: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute per-molecule-balanced FlowMol endpoint + CTMC masked loss strictly in FP32."""
        assert batch.x_1 is not None and batch.a_1 is not None and batch.c_1 is not None and batch.e_1_upper is not None
        device = batch.x_1.device
        bsz = batch.batch_size
        if t is None:
            t = sample_train_timesteps(bsz, device, mode=self.config.timestep_sampling)

        do_distort = self.training if apply_distortion is None else bool(apply_distortion)
        x_t, a_t, c_t, e_t = self.sample_conditional_path(batch, t, apply_distortion=do_distort)

        sc_dict: Optional[Dict[str, torch.Tensor]] = None
        if self.config.self_conditioning:
            if use_self_conditioning is None:
                do_sc = self.training and (torch.rand((), device=device).item() < self.config.self_conditioning_prob)
            else:
                do_sc = bool(use_self_conditioning)
            if do_sc:
                # Disable autocast weight caching inside no_grad() and clear cache afterwards so the
                # subsequent grad-enabled forward() never reuses requires_grad=False cached BF16 weights!
                ac_on = bool(torch.is_autocast_enabled(device.type))
                ac_dtype = torch.get_autocast_dtype(device.type) if ac_on else torch.bfloat16
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=ac_dtype, enabled=ac_on, cache_enabled=False
                ):
                    prev = self.forward(
                        batch,
                        x_t,
                        a_t,
                        c_t,
                        e_t,
                        t,
                        sc_dict=None,
                        cond_values=cond_values,
                        cond_active_mask=cond_active_mask,
                        node_cond_flags=node_cond_flags,
                        remove_com=True,
                    )
                    sc_dict = {
                        "x": prev["x"].detach(),
                        "a": F.softmax(prev["a"].float(), dim=-1).detach(),
                        "c": F.softmax(prev["c"].float(), dim=-1).detach(),
                        "e": F.softmax(prev["e"].float(), dim=-1).detach(),
                    }
                if ac_on:
                    torch.clear_autocast_cache()

        pred = self.forward(
            batch,
            x_t,
            a_t,
            c_t,
            e_t,
            t,
            sc_dict=sc_dict,
            cond_values=cond_values,
            cond_active_mask=cond_active_mask,
            node_cond_flags=node_cond_flags,
            remove_com=True,
        )

        # Strictly FP32 per-molecule loss reductions (prevents O(N^2) large-molecule dominance)
        with torch.autocast(device_type=device.type, enabled=False):
            tw = self.scheduler.loss_weights(t) if self.config.time_scaled_loss else None

            x_err = F.mse_loss(pred["x"].float(), batch.x_1.float(), reduction="none").mean(dim=-1)
            x_sum_mol = torch.zeros((bsz,), dtype=torch.float32, device=device)
            x_sum_mol.index_add_(0, batch.node_batch_idx, x_err)
            per_mol_x = x_sum_mol / batch.n_atoms.float().clamp_min(1.0)
            if tw is not None:
                per_mol_x = per_mol_x * tw[:, 0]

            def _per_mol_masked_ce(
                logits: torch.Tensor,
                target: torch.Tensor,
                cur_state: torch.Tensor,
                mask_idx: int,
                item_batch_idx: torch.Tensor,
                mol_time_w: Optional[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                masked = cur_state == mask_idx
                n_masked = masked.sum()
                frac = n_masked.float() / max(float(target.numel()), 1.0)
                if n_masked.item() == 0:
                    zero_vec = torch.zeros((bsz,), dtype=torch.float32, device=device) + (logits.float().sum() * 0.0)
                    return zero_vec, frac
                ce = F.cross_entropy(logits[masked].float(), target[masked], reduction="none")
                b_m = item_batch_idx[masked]
                sum_m = torch.zeros((bsz,), dtype=torch.float32, device=device)
                cnt_m = torch.zeros((bsz,), dtype=torch.float32, device=device)
                sum_m.index_add_(0, b_m, ce)
                cnt_m.index_add_(0, b_m, torch.ones_like(ce))
                # Molecules with 0 masked slots at high t contribute 0 cleanly
                per_m = torch.where(cnt_m > 0, sum_m / cnt_m.clamp_min(1.0), torch.zeros_like(sum_m))
                if mol_time_w is not None:
                    per_m = per_m * mol_time_w
                # Keep all logits connected for DDP safety even when only a subset was masked
                per_m = per_m + (logits.float().sum() * 0.0)
                return per_m, frac

            per_mol_a, frac_a = _per_mol_masked_ce(
                pred["a"], batch.a_1, a_t, self.atom_mask_idx, batch.node_batch_idx, tw[:, 1] if tw is not None else None
            )
            per_mol_c, frac_c = _per_mol_masked_ce(
                pred["c"], batch.c_1, c_t, self.charge_mask_idx, batch.node_batch_idx, tw[:, 2] if tw is not None else None
            )
            per_mol_e, frac_e = _per_mol_masked_ce(
                pred["e"], batch.e_1_upper, e_t, self.bond_mask_idx, batch.upper_edge_batch_idx, tw[:, 3] if tw is not None else None
            )

            per_mol_total = (
                self.config.loss_weight_x * per_mol_x
                + self.config.loss_weight_a * per_mol_a
                + self.config.loss_weight_c * per_mol_c
                + self.config.loss_weight_e * per_mol_e
            )
            loss_x = per_mol_x.mean()
            loss_a = per_mol_a.mean()
            loss_c = per_mol_c.mean()
            loss_e = per_mol_e.mean()
            total_loss = per_mol_total.mean()

        metrics: Dict[str, Any] = {
            "loss_total": total_loss.detach(),
            "loss_x": loss_x.detach(),
            "loss_a": loss_a.detach(),
            "loss_c": loss_c.detach(),
            "loss_e": loss_e.detach(),
            "mask_frac_a": frac_a.detach(),
            "mask_frac_e": frac_e.detach(),
        }
        if return_per_molecule:
            metrics["per_molecule_generation_loss"] = per_mol_total
            return per_mol_total, metrics
        return total_loss, metrics

    @staticmethod
    def _ctmc_transition_rates(
        alpha_t: torch.Tensor,
        alpha_s: torch.Tensor,
        dt: torch.Tensor,
        stochasticity: float,
        last_step: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact finite-interval unmask & annealed remask probabilities in `[0, 1]`."""
        if last_step:
            one = torch.ones_like(alpha_t, dtype=torch.float32)
            zero = torch.zeros_like(alpha_t, dtype=torch.float32)
            return one, zero
        d_alpha = (alpha_s - alpha_t).clamp_min(0.0)
        rem_s = (1.0 - alpha_s).clamp(0.0, 1.0)
        denom = (1.0 - alpha_t).clamp_min(1e-5)
        churn = dt * float(stochasticity) * rem_s
        unmask_prob = ((d_alpha + churn * alpha_t) / denom).clamp(0.0, 1.0)
        remask_prob = churn.clamp(0.0, 1.0)
        return unmask_prob, remask_prob

    @classmethod
    def _campbell_ctmc_step(
        cls,
        probs: torch.Tensor,
        xt: torch.Tensor,
        *,
        alpha_t: torch.Tensor,
        alpha_s: torch.Tensor,
        dt: torch.Tensor,
        mask_idx: int,
        stochasticity: float,
        temperature: float,
        last_step: bool,
    ) -> torch.Tensor:
        """Exact finite-step Campbell CTMC jump with `(1 - alpha_s)`-annealed remasking."""
        p_f = probs.float().clamp_min(1e-9)
        if temperature > 0.0 and abs(temperature - 1.0) > 1e-6:
            p_sharp = F.softmax(torch.log(p_f) / float(temperature), dim=-1)
        else:
            p_sharp = p_f / p_f.sum(dim=-1, keepdim=True)

        if last_step:
            x1_draw = p_sharp.argmax(dim=-1) if temperature <= 0.1 else Categorical(probs=p_sharp).sample()
            return torch.where(xt == mask_idx, x1_draw, xt)

        unmask_prob, remask_prob = cls._ctmc_transition_rates(alpha_t, alpha_s, dt, stochasticity, last_step=False)
        x1_draw = Categorical(probs=p_sharp).sample()

        u1 = torch.rand((xt.shape[0],), device=xt.device, dtype=torch.float32)
        u2 = torch.rand((xt.shape[0],), device=xt.device, dtype=torch.float32)
        will_unmask = (u1 < unmask_prob) & (xt == mask_idx)
        will_mask = (u2 < remask_prob) & (xt != mask_idx)

        out = xt.clone()
        out[will_mask] = mask_idx
        out[will_unmask] = x1_draw[will_unmask]
        return out

    def sde_ctmc_step_logprob(
        self,
        batch: FlowMolBatch,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        c_t: torch.Tensor,
        e_t: torch.Tensor,
        x_s: torch.Tensor,
        a_s: torch.Tensor,
        c_s: torch.Tensor,
        e_s: torch.Tensor,
        *,
        t_val: float,
        s_val: float,
        sde_noise_scale: float = 0.25,
        stochasticity: float = 8.0,
        cat_temperature: float = 1.0,
        sc_dict: Optional[Dict[str, torch.Tensor]] = None,
        cond_values: Optional[torch.Tensor] = None,
        cond_active_mask: Optional[torch.Tensor] = None,
        node_cond_flags: Optional[torch.Tensor] = None,
        free_node_mask: Optional[torch.Tensor] = None,
        free_edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Joint per-molecule `(B,)` step log-probability for Stage 4 `Hybrid Flow-GRPO` strictly in FP32."""
        dev = x_t.device
        bsz = batch.batch_size
        t_vec = torch.full((bsz,), float(t_val), dtype=torch.float32, device=dev)
        s_vec = torch.full((bsz,), float(s_val), dtype=torch.float32, device=dev)
        out = self.forward(
            batch,
            x_t,
            a_t,
            c_t,
            e_t,
            t_vec,
            sc_dict=sc_dict,
            cond_values=cond_values,
            cond_active_mask=cond_active_mask,
            node_cond_flags=node_cond_flags,
            remove_com=True,
        )

        with torch.autocast(device_type=dev.type, enabled=False):
            dt = torch.tensor(max(float(s_val - t_val), 1e-6), dtype=torch.float32, device=dev)
            last_step = float(s_val) >= 0.9999
            al_t = self.scheduler.alpha_t(t_vec[:1])[0]
            al_s = self.scheduler.alpha_t(s_vec[:1])[0]

            # 1. Zero-COM Gaussian SDE log-density on R^{3(N_free - 1)}
            wx = ((al_s[0] - al_t[0]) / (1.0 - al_t[0]).clamp_min(1e-5)).clamp(0.0, 1.0)
            mu_x = remove_com_fp32(
                x_t.float() + wx * (out["x"].float() - x_t.float()),
                batch.node_batch_idx,
                bsz,
                batch.n_atoms,
            )
            var = max((float(sde_noise_scale) ** 2) * float(dt.item()), 1e-6)
            sq_err = ((x_s.float() - mu_x) ** 2).sum(dim=-1)
            if free_node_mask is not None:
                sq_err = sq_err * free_node_mask.float()
                n_free = torch.zeros((bsz,), dtype=torch.float32, device=dev)
                n_free.index_add_(0, batch.node_batch_idx, free_node_mask.float())
                dof = 3.0 * n_free.clamp_min(1.0)
            else:
                dof = 3.0 * (batch.n_atoms.float() - 1.0).clamp_min(1.0)
            mol_sq = torch.zeros((bsz,), dtype=torch.float32, device=dev)
            mol_sq.index_add_(0, batch.node_batch_idx, sq_err)
            lp_x = -0.5 * (dof * math.log(2.0 * math.pi * var) + mol_sq / var)

            # 2. Exact CTMC discrete transition log-probabilities
            def _ctmc_lp(
                logits: torch.Tensor,
                cur_idx: torch.Tensor,
                nxt_idx: torch.Tensor,
                feat_i: int,
                mask_id: int,
                b_idx: torch.Tensor,
                free_m: Optional[torch.Tensor],
            ) -> torch.Tensor:
                scaled_logits = logits.float() / max(float(cat_temperature), 1e-4)
                log_p1 = F.log_softmax(scaled_logits, dim=-1)
                p_un, p_re = self._ctmc_transition_rates(al_t[feat_i], al_s[feat_i], dt, stochasticity, last_step)
                p_un_c = p_un.clamp(1e-7, 1.0 - 1e-7) if not last_step else torch.tensor(1.0 - 1e-7, device=dev)
                p_re_c = p_re.clamp(1e-7, 1.0 - 1e-7) if not last_step else torch.tensor(1e-7, device=dev)

                was_mask = cur_idx == mask_id
                now_mask = nxt_idx == mask_id
                safe_nxt = nxt_idx.clamp(0, logits.shape[-1] - 1)
                lp_unmask_to_k = torch.log(p_un_c) + log_p1.gather(-1, safe_nxt.unsqueeze(-1)).squeeze(-1)
                lp_stay_mask = torch.log((1.0 - p_un_c).clamp_min(1e-7)).expand_as(lp_unmask_to_k)
                lp_remask = torch.log(p_re_c).expand_as(lp_unmask_to_k)
                lp_stay_unmasked = torch.log((1.0 - p_re_c).clamp_min(1e-7)).expand_as(lp_unmask_to_k)

                from_mask = torch.where(now_mask, lp_stay_mask, lp_unmask_to_k)
                from_unmask = torch.where(now_mask, lp_remask, lp_stay_unmasked)
                item_lp = torch.where(was_mask, from_mask, from_unmask)
                if free_m is not None:
                    item_lp = item_lp * free_m.float()
                mol_lp = torch.zeros((bsz,), dtype=torch.float32, device=dev)
                mol_lp.index_add_(0, b_idx, item_lp)
                return mol_lp

            lp_a = _ctmc_lp(out["a"], a_t, a_s, 1, self.atom_mask_idx, batch.node_batch_idx, free_node_mask)
            lp_c = _ctmc_lp(out["c"], c_t, c_s, 2, self.charge_mask_idx, batch.node_batch_idx, free_node_mask)
            lp_e = _ctmc_lp(out["e"], e_t, e_s, 3, self.bond_mask_idx, batch.upper_edge_batch_idx, free_edge_mask)
            return lp_x + lp_a + lp_c + lp_e

    @torch.no_grad()
    def sample(
        self,
        n_atoms_list: Sequence[int],
        *,
        device: Optional[torch.device] = None,
        n_steps: int = 100,
        solver: str = "euler",
        grid_type: str = "uniform",
        stochasticity: float = 10.0,
        cat_temperature: float = 0.05,
        allowed_elements: Optional[Sequence[str]] = None,
        allowed_charges: Optional[Sequence[int]] = None,
        cond_values: Optional[torch.Tensor] = None,
        cond_active_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        xyz_cfg_scale: float = 1.0,
        use_bf16: bool = True,
        seed: Optional[int] = None,
        build_rdkit: bool = True,
    ) -> List[Dict[str, Any]]:
        """Sample 3D all-atom molecules via overshoot-free alpha-space Euler/Heun + Self-Conditioned CTMC."""
        was_training = self.training
        self.eval()
        dev = device if device is not None else next(self.parameters()).device
        amp_enabled = bool(use_bf16 and dev.type == "cuda")

        elem_mask = build_allowed_element_mask(allowed_elements, self.config.atom_types, device=dev)
        chg_mask = build_allowed_charge_mask(allowed_charges, self.config.formal_charges, device=dev)
        use_cfg = (
            self.conditioner is not None
            and cond_values is not None
            and cond_active_mask is not None
            and bool(cond_active_mask.any().item())
            and (abs(cfg_scale - 1.0) > 1e-5 or abs(xyz_cfg_scale - 1.0) > 1e-5)
        )
        null_mask = torch.zeros_like(cond_active_mask, dtype=torch.bool) if use_cfg and cond_active_mask is not None else None

        devices = [dev.index] if (dev.type == "cuda" and dev.index is not None) else []
        try:
            with torch.random.fork_rng(devices=devices, enabled=(seed is not None)):
                if seed is not None:
                    torch.manual_seed(int(seed))

                batch = build_complete_graph_batch(
                    n_atoms_list,
                    dev,
                    prior_std=self.config.prior_std,
                    size_scale_prior=self.config.size_scale_prior,
                )
                n_nodes = batch.x_0.shape[0]
                n_up = int(batch.upper_edge_mask.sum().item())

                x_t = batch.x_0.clone()
                a_t = torch.full((n_nodes,), self.atom_mask_idx, dtype=torch.long, device=dev)
                c_t = torch.full((n_nodes,), self.charge_mask_idx, dtype=torch.long, device=dev)
                e_t = torch.full((n_up,), self.bond_mask_idx, dtype=torch.long, device=dev)

                tspan = build_inference_tspan(n_steps, dev, grid_type=grid_type)
                alphas = self.scheduler.alpha_t(tspan)  # (n_steps + 1, 4)

                sc_dict: Optional[Dict[str, torch.Tensor]] = None

                def _predict(x_in, a_in, c_in, e_in, t_val, sc_in):
                    t_vec = torch.full((batch.batch_size,), float(t_val.item()), dtype=torch.float32, device=dev)
                    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp_enabled):
                        out = self.forward(
                            batch,
                            x_in,
                            a_in,
                            c_in,
                            e_in,
                            t_vec,
                            sc_dict=sc_in,
                            cond_values=cond_values,
                            cond_active_mask=cond_active_mask,
                            remove_com=True,
                        )
                        if use_cfg:
                            out_u = self.forward(
                                batch,
                                x_in,
                                a_in,
                                c_in,
                                e_in,
                                t_vec,
                                sc_dict=sc_in,
                                cond_values=cond_values,
                                cond_active_mask=null_mask,
                                remove_com=True,
                            )
                        else:
                            out_u = None

                    if out_u is not None:
                        x_hat = remove_com_fp32(
                            out_u["x"].float() + float(xyz_cfg_scale) * (out["x"].float() - out_u["x"].float()),
                            batch.node_batch_idx,
                            batch.batch_size,
                            batch.n_atoms,
                        )
                        a_logits = out_u["a"].float() + float(cfg_scale) * (out["a"].float() - out_u["a"].float())
                        c_logits = out_u["c"].float() + float(cfg_scale) * (out["c"].float() - out_u["c"].float())
                        e_logits = out_u["e"].float() + float(cfg_scale) * (out["e"].float() - out_u["e"].float())
                    else:
                        x_hat = out["x"].float()
                        a_logits = out["a"].float()
                        c_logits = out["c"].float()
                        e_logits = out["e"].float()

                    if elem_mask is not None:
                        a_logits = a_logits.masked_fill(~elem_mask.unsqueeze(0), -1e9)
                    if chg_mask is not None:
                        c_logits = c_logits.masked_fill(~chg_mask.unsqueeze(0), -1e9)
                    return {
                        "x": x_hat,
                        "a": F.softmax(a_logits, dim=-1),
                        "c": F.softmax(c_logits, dim=-1),
                        "e": F.softmax(e_logits, dim=-1),
                    }

                for step_idx in range(n_steps):
                    t_i = tspan[step_idx]
                    s_i = tspan[step_idx + 1]
                    dt = s_i - t_i
                    last_step = step_idx == (n_steps - 1)

                    pred1 = _predict(x_t, a_t, c_t, e_t, t_i, sc_dict)

                    a_next = self._campbell_ctmc_step(
                        pred1["a"], a_t,
                        alpha_t=alphas[step_idx, 1], alpha_s=alphas[step_idx + 1, 1],
                        dt=dt, mask_idx=self.atom_mask_idx,
                        stochasticity=stochasticity, temperature=cat_temperature, last_step=last_step,
                    )
                    c_next = self._campbell_ctmc_step(
                        pred1["c"], c_t,
                        alpha_t=alphas[step_idx, 2], alpha_s=alphas[step_idx + 1, 2],
                        dt=dt, mask_idx=self.charge_mask_idx,
                        stochasticity=stochasticity, temperature=cat_temperature, last_step=last_step,
                    )
                    e_next = self._campbell_ctmc_step(
                        pred1["e"], e_t,
                        alpha_t=alphas[step_idx, 3], alpha_s=alphas[step_idx + 1, 3],
                        dt=dt, mask_idx=self.bond_mask_idx,
                        stochasticity=stochasticity, temperature=cat_temperature, last_step=last_step,
                    )

                    # Overshoot-free exact finite-step alpha-space coordinate integration
                    if last_step:
                        x_next = remove_com_fp32(pred1["x"], batch.node_batch_idx, batch.batch_size, batch.n_atoms)
                        sc_next = pred1
                    else:
                        ax_t = alphas[step_idx, 0]
                        ax_s = alphas[step_idx + 1, 0]
                        wx = ((ax_s - ax_t) / (1.0 - ax_t).clamp_min(1e-5)).clamp(0.0, 1.0)
                        x_euler = remove_com_fp32(
                            x_t + wx * (pred1["x"] - x_t), batch.node_batch_idx, batch.batch_size, batch.n_atoms
                        )
                        if solver == "heun":
                            pred2 = _predict(
                                x_euler, a_next, c_next, e_next, s_i, pred1 if self.config.self_conditioning else None
                            )
                            x_bar = 0.5 * (pred1["x"] + pred2["x"])
                            x_next = remove_com_fp32(
                                x_t + wx * (x_bar - x_t), batch.node_batch_idx, batch.batch_size, batch.n_atoms
                            )
                            sc_next = pred2
                        else:
                            x_next = x_euler
                            sc_next = pred1

                    x_t, a_t, c_t, e_t = x_next, a_next, c_next, e_next
                    sc_dict = sc_next if self.config.self_conditioning else None

                node_splits = batch.n_atoms.cpu().tolist()
                edge_splits = [na * (na - 1) // 2 for na in node_splits]
                xs = torch.split(x_t.cpu(), node_splits)
                as_ = torch.split(a_t.cpu(), node_splits)
                cs = torch.split(c_t.cpu(), node_splits)
                es = torch.split(e_t.cpu(), edge_splits)

                results: List[Dict[str, Any]] = []
                for pos_i, a_i, c_i, e_i in zip(xs, as_, cs, es):
                    item: Dict[str, Any] = {
                        "positions": pos_i.numpy(),
                        "atom_types": a_i.numpy(),
                        "charges": c_i.numpy(),
                        "upper_bond_types": e_i.numpy(),
                    }
                    if build_rdkit:
                        mol, diag = tensors_to_rdkit_mol(
                            pos_i,
                            a_i,
                            c_i,
                            e_i,
                            atom_vocab=self.config.atom_types,
                            charge_vocab=self.config.formal_charges,
                            sanitize=True,
                        )
                        item["rdkit_mol"] = mol
                        item["diag"] = diag
                    results.append(item)
                return results
        finally:
            self.train(was_training)
