"""High-throughput, calibrated `Uni-3DAR` implementation of `GeneratorSlot` for `Uni-Loom`."""

from __future__ import annotations

from copy import copy
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..generator.condition_contrast import condition_margin_loss
from ..generator.electronic_constraints import atom_electron_parities, mask_terminal_element_logits
from ..generator.octree_backend import OctreeDataset
from ..generator.octree_model import AtomFeatDict, build_default_uni3dar_args
from ..vendor.uni3dar.data.atom_data import atom_list
from ..vendor.uni3dar.data.grid_utils import subcell_orders
from ..vendor.uni3dar.models.kv_cache import append_kv, compact_kv
from ..vendor.uni3dar.models.uni3dar import DecoderInputFeat
from ..vendor.uni3dar.models.uni3dar_sampler import (
    IDX_CNT,
    IDX_LEVEL,
    IDX_SPACE,
    IDX_TREE,
    IDX_TYPE,
    NUM_FEAT,
    Uni3DARSampler,
)
from .conditioner import LoomConditioner
from .lora import LoRALinear, extract_linear_weight_bias, inject_uni3dar_lora, inject_xyz_head_lora, iter_lora_parameters
from .slots import GeneratorSlot

try:
    from torch.nn.attention.bias import causal_lower_right as _causal_lower_right
    _HAS_CAUSAL_LOWER_RIGHT = True
except Exception:  # pragma: no cover
    _causal_lower_right = None  # type: ignore[assignment]
    _HAS_CAUSAL_LOWER_RIGHT = False


_ELEMENT_RCOV_A: Dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "I": 1.39,
}

_ELEMENT_MAX_COORD: Dict[str, int] = {
    "H": 1,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "O": 2,
    "B": 3,
    "C": 4,
    "N": 3,
    "Si": 4,
    "Se": 4,
    "P": 5,
    "As": 5,
    "S": 6,
}


class HeavyAtomValenceHead(nn.Module):
    """Lightweight auxiliary head supervising `(num_attached_H in {0..4}, formal_charge in {-1,0,+1})`
    on heavy-atom hidden states so the backbone internalizes local neutral Lewis valence budgets.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.proj_num_h = nn.Linear(emb_dim, 5)
        self.proj_charge = nn.Linear(emb_dim, 3)

    def forward(self, hidden_atoms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(hidden_atoms)
        return self.proj_num_h(h), self.proj_charge(h)


@lru_cache(maxsize=32)
def _atom_geometry_tables(num_atom: int, device_str: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cached `(rcov_table, max_coord_table)` tensors of shape `(num_atom,)` on `device`."""
    device = torch.device(device_str)
    rcov = torch.full((num_atom,), 0.80, dtype=torch.float32, device=device)
    max_c = torch.full((num_atom,), 8, dtype=torch.int32, device=device)
    for idx, sym in enumerate(atom_list):
        if idx >= num_atom:
            break
        if sym in _ELEMENT_RCOV_A:
            rcov[idx] = _ELEMENT_RCOV_A[sym]
        if sym in _ELEMENT_MAX_COORD:
            max_c[idx] = _ELEMENT_MAX_COORD[sym]
    return rcov, max_c


@lru_cache(maxsize=256)
def _suffix_causal_sdpa_mask(seq_q: int, seq_k: int, device_str: str) -> torch.Tensor:
    """Boolean mask `(1, 1, seq_q, seq_k)` where `True` means allowed to attend."""
    device = torch.device(device_str)
    q_idx = torch.arange(seq_q, device=device)[:, None] + (seq_k - seq_q)
    k_idx = torch.arange(seq_k, device=device)[None, :]
    return (k_idx <= q_idx).unsqueeze(0).unsqueeze(0)


@lru_cache(maxsize=16)
def _split_popcount_table(device_str: str) -> torch.Tensor:
    """Cached `(255,)` float32 popcounts for non-empty octree split masks `1..255`."""
    device = torch.device(device_str)
    pops = [float(bin(mask_val).count("1")) for mask_val in range(1, 256)]
    return torch.tensor(pops, dtype=torch.float32, device=device)


def _select_incompatible_negatives(
    targets_eV: torch.Tensor,
    active_mask: torch.Tensor,
    min_gap_eV: float = 0.40,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select an incompatible condition `j` for each row `i` along row `i`'s active target axes."""
    bsz = targets_eV.shape[0]
    device = targets_eV.device
    act = active_mask.bool()
    has_cond = act.any(dim=-1)
    # Pairwise |y_i - y_j| masked by row i's active_mask: (B, B, P) -> (B, B)
    diff = (targets_eV.unsqueeze(1) - targets_eV.unsqueeze(0)).abs()
    masked_max_diff = (diff * act.unsqueeze(1).float()).amax(dim=-1)
    valid = (masked_max_diff > float(min_gap_eV)) & has_cond.unsqueeze(1)
    eligible = valid.any(dim=1)
    if deterministic:
        gen = torch.Generator(device=device)
        gen.manual_seed(13579 + bsz)
        scores = torch.rand((bsz, bsz), generator=gen, device=device)
    else:
        scores = torch.rand((bsz, bsz), device=device)
    scores = scores.masked_fill(~valid, -1.0)
    neg_idx = scores.argmax(dim=1)
    return neg_idx, eligible


def _gpu_multinomial_sample(
    logits: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-GPU categorical sampling with zero CPU synchronization."""
    orig_logits = logits.float()
    scaled_logits = orig_logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    log_probs = torch.log_softmax(orig_logits, dim=-1)
    preds = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    scores = log_probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
    return preds.to(torch.int32), scores


class LoomUni3DARGenerator(GeneratorSlot):
    """High-bandwidth, FlashAttention-batched conditional Uni-3DAR generator slot.

    Upgrades over legacy `ConditionalUni3DAR`:
      1. `LoomConditioner` (32-RBF + 16-Fourier + `P` causal prefix KV tokens + unclamped AdaLN-Zero).
      2. Zero-perturbation `LoRALinear` adapters across all 12 Transformer attention + FFN layers
         plus tree and element classification heads (fine XYZ heads remain untouched).
      3. Vectorized right-padded `F.scaled_dot_product_attention(..., is_causal=True)` sequence pass
         eliminating the `B`-way Python loop & `.tolist()` syncs in `flash_attn_varlen_func`.
      4. Single-pass `2 * B` batched `[cond, uncond]` Selective Logit-CFG (`cfg_scale` on tree & element
         logits; `xyz_cfg_scale=1.0` on fine `0.01 Å` coordinates) + sync-free GPU decoding.
    """

    backend_id: str = "loom_uni3dar_v1"
    supports_exact_trajectory_logprob: bool = True
    generator_family: str = "octree_autoregressive"

    def __init__(
        self,
        backbone: Uni3DARSampler,
        target_cols: Sequence[str],
        target_units: Sequence[str],
        mean: Sequence[float],
        scale: Sequence[float],
        *,
        num_rbf: int = 32,
        num_freqs: int = 16,
        num_prefix_tokens: int = 0,
        modulation_cap: float = 4.0,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        lora_include_ffn: bool = True,
        lora_include_heads: bool = True,
        lora_include_xyz_heads: bool = False,
        train_xyz_embeddings: bool = False,
        use_valence_head: bool = False,
    ) -> None:
        super().__init__()
        if len(target_cols) != len(target_units) or len(mean) != len(target_cols) or len(scale) != len(target_cols):
            raise ValueError("target_cols, target_units, mean, and scale must have identical length")

        self.backbone = backbone
        self.args = backbone.args
        self.dictionary = backbone.dictionary
        self.target_cols = list(target_cols)
        self.target_units = list(target_units)
        self.num_props = len(self.target_cols)

        mean_t = torch.tensor(np.asarray(mean, dtype=np.float32))
        scale_t = torch.tensor(np.asarray(scale, dtype=np.float32))
        if not torch.isfinite(mean_t).all() or not torch.isfinite(scale_t).all() or (scale_t <= 0).any():
            raise ValueError("Invalid target mean/scale")
        self.register_buffer("target_mean", mean_t)
        self.register_buffer("target_scale", scale_t)

        self.loom_config: Dict[str, Any] = {
            "num_rbf": int(num_rbf),
            "num_freqs": int(num_freqs),
            "num_prefix_tokens": int(num_prefix_tokens),
            "modulation_cap": float(modulation_cap),
            "lora_rank": int(lora_rank),
            "lora_alpha": float(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "lora_include_ffn": bool(lora_include_ffn),
            "lora_include_heads": bool(lora_include_heads),
            "lora_include_xyz_heads": bool(lora_include_xyz_heads),
            "train_xyz_embeddings": bool(train_xyz_embeddings),
            "use_valence_head": bool(use_valence_head),
        }

        self.conditioner = LoomConditioner(
            emb_dim=self.args.emb_dim,
            num_props=self.num_props,
            num_layers=self.args.layer,
            num_rbf=num_rbf,
            num_freqs=num_freqs,
            num_prefix_tokens=num_prefix_tokens,
            modulation_cap=modulation_cap,
        )
        self.valence_head: Optional[HeavyAtomValenceHead] = (
            HeavyAtomValenceHead(self.args.emb_dim) if use_valence_head else None
        )

        self.injected_lora_modules: List[str] = []
        if lora_rank > 0:
            self.injected_lora_modules = inject_uni3dar_lora(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                include_ffn=lora_include_ffn,
                include_heads=lora_include_heads,
                include_xyz_heads=lora_include_xyz_heads,
            )

        self.register_buffer("_subcell_shifts", torch.tensor(subcell_orders, dtype=torch.int32), persistent=False)
        self.register_buffer("_bit_shifts", (1 << torch.arange(8, dtype=torch.int32)), persistent=False)
        self._fused_eval_layers: Dict[Tuple[int, str, torch.dtype], Any] = {}
        self.configure_trainable()

    def train(self, mode: bool = True) -> "LoomUni3DARGenerator":
        self._fused_eval_layers.clear()
        return super().train(mode)

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._fused_eval_layers.clear()
        super()._load_from_state_dict(*args, **kwargs)

    # ------------------------------------------------------------------
    # Construction & Checkpoint Serialization
    # ------------------------------------------------------------------
    @staticmethod
    def _build_backbone(
        merge_level: int = 7,
        allowed_elements: Sequence[str] = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"),
        batch_size: int = 32,
        *,
        emb_dim: int = 768,
        num_head: int = 12,
        layer: int = 12,
        ffn_multiple: int = 4,
    ) -> Uni3DARSampler:
        args = build_default_uni3dar_args(batch_size=batch_size)
        args.merge_level = int(merge_level)
        args.allow_atoms = ",".join(allowed_elements)
        args.emb_dim = int(emb_dim)
        args.num_head = int(num_head)
        args.layer = int(layer)
        args.ffn_multiple = int(ffn_multiple)
        return Uni3DARSampler(args, AtomFeatDict(args))

    @classmethod
    def from_parent_checkpoint(
        cls,
        parent_ckpt_path: Path | str,
        target_cols: Sequence[str],
        target_units: Sequence[str],
        mean: Sequence[float],
        scale: Sequence[float],
        *,
        allowed_elements: Optional[Sequence[str]] = None,
        device: str = "cuda:0",
        batch_size: int = 32,
        **loom_kwargs: Any,
    ) -> "LoomUni3DARGenerator":
        raw = torch.load(parent_ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and raw.get("backend") == cls.backend_id:
            model, _ = cls.load_checkpoint(parent_ckpt_path, device=device, batch_size=batch_size)
            return model

        default_allowed = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I")
        if isinstance(raw, dict) and "ema" in raw and "params" in raw["ema"]:
            backbone_state = raw["ema"]["params"]
            depth = int(backbone_state["cur_level.weight"].shape[0]) - 2
            allowed = tuple(allowed_elements) if allowed_elements is not None else default_allowed
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            rep = raw.get("representation_config", {})
            full_sd = raw["model_state_dict"]
            depth = int(rep.get("merge_level", int(full_sd["cur_level.weight"].shape[0]) - 2))
            if allowed_elements is not None:
                allowed = tuple(allowed_elements)
            else:
                allowed = tuple(
                    rep.get(
                        "allowed_elements",
                        raw.get("contract", {})
                        .get("semantics", {})
                        .get("allowed_elements", default_allowed),
                    )
                )
            skip_prefixes = ("interval_conditioner.", "prop_understanding_head.")
            backbone_state = {k: v for k, v in full_sd.items() if not k.startswith(skip_prefixes)}
        else:
            raise ValueError(f"Unsupported parent checkpoint format at {parent_ckpt_path}")

        backbone = cls._build_backbone(merge_level=depth, allowed_elements=allowed, batch_size=batch_size)
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        if missing or unexpected:
            raise ValueError(f"Backbone state mismatch when loading {parent_ckpt_path}: missing={missing}, unexpected={unexpected}")

        gen = cls(
            backbone=backbone,
            target_cols=target_cols,
            target_units=target_units,
            mean=mean,
            scale=scale,
            **loom_kwargs,
        )
        return gen.to(device)

    def save_checkpoint(self, path: Path | str, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend_id,
            "target_cols": list(self.target_cols),
            "target_units": list(self.target_units),
            "mean": self.target_mean.detach().cpu().numpy().tolist(),
            "scale": self.target_scale.detach().cpu().numpy().tolist(),
            "representation_config": {
                "merge_level": int(self.args.merge_level),
                "allowed_elements": self.args.allow_atoms.split(","),
                "emb_dim": int(self.args.emb_dim),
                "num_head": int(self.args.num_head),
                "layer": int(self.args.layer),
                "ffn_multiple": int(self.args.ffn_multiple),
            },
            "loom_config": copy(self.loom_config),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "metadata": extra_metadata or {},
        }
        torch.save(payload, out_path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path | str,
        *,
        device: str = "cuda:0",
        batch_size: int = 32,
    ) -> Tuple["LoomUni3DARGenerator", Dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("backend") != cls.backend_id:
            raise ValueError(f"Expected backend {cls.backend_id}, got {payload.get('backend')}")
        rep = payload["representation_config"]
        backbone = cls._build_backbone(
            merge_level=int(rep["merge_level"]),
            allowed_elements=rep["allowed_elements"],
            batch_size=batch_size,
            emb_dim=int(rep.get("emb_dim", 768)),
            num_head=int(rep.get("num_head", 12)),
            layer=int(rep.get("layer", 12)),
            ffn_multiple=int(rep.get("ffn_multiple", 4)),
        )
        gen = cls(
            backbone=backbone,
            target_cols=payload["target_cols"],
            target_units=payload["target_units"],
            mean=payload["mean"],
            scale=payload["scale"],
            **payload["loom_config"],
        )
        gen.load_state_dict(payload["state_dict"], strict=True)
        gen = gen.to(device)
        gen.eval()
        return gen, payload

    def configure_trainable(
        self,
        *,
        train_conditioner: bool = True,
        train_lora: bool = True,
        train_xyz_embeddings: Optional[bool] = None,
        unfreeze_full_backbone: bool = False,
    ) -> List[nn.Parameter]:
        if train_xyz_embeddings is None:
            train_xyz_embeddings = bool(self.loom_config.get("train_xyz_embeddings", False))
        self.backbone.requires_grad_(unfreeze_full_backbone)
        if train_lora and not unfreeze_full_backbone:
            for _, param in iter_lora_parameters(self.backbone):
                param.requires_grad_(True)
        if train_xyz_embeddings and not unfreeze_full_backbone:
            for ahead in getattr(self.backbone, "atom_heads", []):
                for emb in getattr(ahead, "emb_layers", []):
                    emb.requires_grad_(True)
        self.conditioner.requires_grad_(train_conditioner)
        if self.valence_head is not None:
            self.valence_head.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]

    def enable_v20_valence_head(self) -> List[nn.Parameter]:
        """Attach `HeavyAtomValenceHead` in-place onto a `v1.0` or `v1.1` checkpoint
        with zero initial perturbation to autoregressive sampling.
        """
        if self.valence_head is None:
            self.valence_head = HeavyAtomValenceHead(self.args.emb_dim).to(self.target_mean.device)
        self.loom_config["use_valence_head"] = True
        return self.configure_trainable()

    def enable_v11_xyz_heads(
        self,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        train_xyz_embeddings: bool = True,
    ) -> List[nn.Parameter]:
        """Seamlessly upgrade a `v1.0` checkpoint in-place to `loom v1.1`:
        attaches zero-initialized LoRA adapters (`lora_B=0`) to fine `0.01 Å` `XYZ` coordinate heads
        (`atom_heads.0.pred_heads[1..3]`) and unfreezes intra-voxel coordinate embeddings.
        """
        r = int(rank if rank is not None else self.loom_config.get("lora_rank", 16))
        a = float(alpha if alpha is not None else self.loom_config.get("lora_alpha", 32.0))
        d = float(self.loom_config.get("lora_dropout", 0.0))
        added = inject_xyz_head_lora(self.backbone, rank=r, alpha=a, dropout=d)
        self.injected_lora_modules.extend(added)
        self.loom_config["lora_include_xyz_heads"] = True
        self.loom_config["train_xyz_embeddings"] = bool(train_xyz_embeddings)
        device = self.target_mean.device
        self.backbone.to(device)
        return self.configure_trainable(train_xyz_embeddings=train_xyz_embeddings)

    def _prepare_placed_geometry(
        self,
        placed_pos: torch.Tensor,
        placed_types: torch.Tensor,
        placed_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute `(p_rcov, p_saturated)` of shape `(N, K)` using bond-order-aware neutral valence budgets.

        Saturates:
          - Carbonyl `O` after a single short `C=O / N=O` bond (`d <= 0.91 * (r_i + r_j)`, order proxy = 2)
            or after 2 single bonds, preventing oxonium `[O+]` internal salts.
          - Boron `B` at 3 bonds (preventing `[B-]`), halogens `F/Cl/Br/I` at 1 bond, `H` at 1 bond.
          - Nitrogen `N` at 3 valence units unless already bonded to `O` (allowing `-NO2` / `N-oxide` up to 4).
        """
        device = placed_pos.device
        n_rows, k_atoms, _ = placed_pos.shape
        rcov_table, max_c_table = _atom_geometry_tables(self.backbone.num_atom, str(device))
        safe_types = placed_types.clamp(0, self.backbone.num_atom - 1).long()
        p_rcov = rcov_table[safe_types]
        p_maxc = max_c_table[safe_types].clone()
        if k_atoms <= 1 or not bool(placed_valid.any().item()):
            p_val_used = torch.zeros((n_rows, k_atoms), dtype=torch.int32, device=device)
        else:
            d_kk = torch.cdist(placed_pos.float(), placed_pos.float())
            not_self = ~torch.eye(k_atoms, dtype=torch.bool, device=device).unsqueeze(0)
            valid_pair = placed_valid.unsqueeze(2) & placed_valid.unsqueeze(1) & not_self
            r_pair = p_rcov.unsqueeze(2) + p_rcov.unsqueeze(1)
            is_single = valid_pair & (d_kk <= 1.26 * r_pair)
            is_double = valid_pair & (d_kk <= 0.91 * r_pair)
            is_triple = valid_pair & (d_kk <= 0.83 * r_pair)
            bo_proxy = is_single.to(torch.int32) + is_double.to(torch.int32) + is_triple.to(torch.int32)
            p_val_used = bo_proxy.sum(dim=-1)

            # Permit N up to 4 valence units only when bonded to an oxygen atom (nitro / N-oxide)
            n_id = self.dictionary.indices.get("N", -1) if hasattr(self.dictionary, "indices") else self.dictionary["N"]
            o_id = self.dictionary.indices.get("O", -1) if hasattr(self.dictionary, "indices") else self.dictionary["O"]
            is_o = (safe_types == o_id).unsqueeze(1)
            has_o_nb = (is_single & is_o).any(dim=-1)
            p_maxc = torch.where((safe_types == n_id) & has_o_nb, torch.full_like(p_maxc, 4), p_maxc)
        p_sat = placed_valid & (p_val_used >= p_maxc)
        return p_rcov, p_sat

    def _element_guard_logit_bias(
        self,
        vox_origin: torch.Tensor,
        placed_pos: torch.Tensor,
        placed_valid: torch.Tensor,
        p_rcov: torch.Tensor,
        p_sat: torch.Tensor,
    ) -> torch.Tensor:
        """Finite steric & max-coordination logit bias `(N, num_atom)` for element prediction (`axis=0`)."""
        device = vox_origin.device
        n_rows = vox_origin.shape[0]
        bias = torch.zeros((n_rows, self.backbone.num_atom), dtype=torch.float32, device=device)
        if placed_pos.shape[1] == 0 or not bool(placed_valid.any().item()):
            return bias

        rcov_table, max_c_table = _atom_geometry_tables(self.backbone.num_atom, str(device))
        allowed_ids = torch.tensor(
            [self.dictionary[s] for s in self.args.allow_atoms.split(",")], dtype=torch.long, device=device
        )
        r_allow = rcov_table[allowed_ids].view(1, 1, -1)
        v_allow = max_c_table[allowed_ids].view(1, -1)
        grid_len = float(self.args.grid_len)

        dx_max = torch.maximum(
            (vox_origin[:, None, :] - placed_pos).abs(),
            (vox_origin[:, None, :] + grid_len - placed_pos).abs(),
        )
        d_max = torch.linalg.norm(dx_max, dim=-1)
        d_cen = torch.linalg.norm((vox_origin[:, None, :] + 0.5 * grid_len) - placed_pos, dim=-1)
        r_sum = p_rcov.unsqueeze(-1) + r_allow

        clash_elem = (placed_valid.unsqueeze(-1) & (d_max.unsqueeze(-1) < 0.61 * r_sum)).any(dim=1)
        sat_intrude = (p_sat.unsqueeze(-1) & (d_cen.unsqueeze(-1) < 1.22 * r_sum)).any(dim=1)
        new_deg = (placed_valid.unsqueeze(-1) & (d_cen.unsqueeze(-1) < 1.24 * r_sum)).sum(dim=1)
        over_coord = new_deg > v_allow

        sub_bias = (
            -16.0 * clash_elem.float()
            - 12.0 * sat_intrude.float()
            - 14.0 * over_coord.float()
        )
        bias[:, allowed_ids] = sub_bias
        return bias

    def _zbin_guard_logit_bias(
        self,
        vox_origin: torch.Tensor,
        chosen_elem: torch.Tensor,
        chosen_xy_bins: torch.Tensor,
        placed_pos: torch.Tensor,
        placed_valid: torch.Tensor,
        p_rcov: torch.Tensor,
        p_sat: torch.Tensor,
        num_bins: int,
    ) -> torch.Tensor:
        """Finite steric & saturated-valence logit bias `(N, num_bins)` for exact 3D `Z` bin selection (`axis=3`)."""
        device = vox_origin.device
        n_rows = vox_origin.shape[0]
        if placed_pos.shape[1] == 0 or not bool(placed_valid.any().item()):
            return torch.zeros((n_rows, num_bins), dtype=torch.float32, device=device)

        rcov_table, _ = _atom_geometry_tables(self.backbone.num_atom, str(device))
        r_new = rcov_table[chosen_elem.clamp(0, self.backbone.num_atom - 1).long()].view(n_rows, 1, 1)
        r_sum_z = p_rcov.unsqueeze(-1) + r_new
        xyz_res = float(self.args.xyz_resolution)

        xy_a = vox_origin[:, :2] + chosen_xy_bins.float() * xyz_res
        z_a = vox_origin[:, 2:3] + torch.arange(num_bins, device=device, dtype=torch.float32).view(1, -1) * xyz_res
        d2_xy = ((xy_a[:, None, :] - placed_pos[:, :, :2]) ** 2).sum(dim=-1, keepdim=True)
        dz = z_a[:, None, :] - placed_pos[:, :, 2:3]
        d_3d = torch.sqrt((d2_xy + dz ** 2).clamp_min(1e-8))

        clash_pen = torch.where(
            placed_valid.unsqueeze(-1) & (d_3d < 0.63 * r_sum_z),
            -25.0 * (0.63 * r_sum_z - d_3d) - 8.0,
            torch.zeros_like(d_3d),
        ).sum(dim=1)
        sat_pen = torch.where(
            p_sat.unsqueeze(-1) & (d_3d < 1.24 * r_sum_z),
            -12.0 * (1.24 * r_sum_z - d_3d) - 4.0,
            torch.zeros_like(d_3d),
        ).sum(dim=1)
        return (clash_pen + sat_pen).clamp_min(-40.0)

    def normalize_targets(self, targets_eV: np.ndarray | torch.Tensor) -> torch.Tensor:
        device = self.target_mean.device
        t = torch.as_tensor(targets_eV, dtype=torch.float32, device=device)
        return (t - self.target_mean) / self.target_scale

    def denormalize_targets(self, targets_norm: np.ndarray | torch.Tensor) -> torch.Tensor:
        device = self.target_mean.device
        t = torch.as_tensor(targets_norm, dtype=torch.float32, device=device)
        return t * self.target_scale + self.target_mean

    def allowed_atom_mask(self, device: torch.device) -> torch.Tensor:
        allowed = torch.zeros(self.backbone.num_atom, dtype=torch.bool, device=device)
        for symbol in self.args.allow_atoms.split(","):
            allowed[self.dictionary[symbol]] = True
        allowed[self.backbone.atom_null_id] = True
        return allowed

    def _prefix_grid_positions(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        p = self.conditioner.num_prefix_tokens
        if p <= 0:
            return None
        cls_grid = self.backbone.get_cls_phy_pos(self.args.merge_level) / self.args.xyz_resolution
        offsets = (torch.arange(p, device=device, dtype=torch.float32) - p) * 16.0
        return (cls_grid + offsets).view(1, p, 1).expand(batch_size, p, 3).contiguous()

    # ------------------------------------------------------------------
    # Batched Fused Causal SDPA Sequence Pass (Zero Syncs)
    # ------------------------------------------------------------------
    def _get_fused_layer_weights(
        self,
        layer_idx: int,
        layer: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        key = (layer_idx, str(device), dtype)
        cached = self._fused_eval_layers.get(key)
        q_mod = layer.attn.q_proj
        lora_stale = isinstance(q_mod, LoRALinear) and (q_mod._eval_weight is None)
        if cached is not None and not lora_stale:
            return cached

        wq, bq = extract_linear_weight_bias(layer.attn.q_proj, dtype)
        wk, bk = extract_linear_weight_bias(layer.attn.k_proj, dtype)
        wv, bv = extract_linear_weight_bias(layer.attn.v_proj, dtype)
        w_qkv = torch.cat([wq, wk, wv], dim=0).contiguous()
        if bq is not None or bk is not None or bv is not None:
            z = torch.zeros(wq.shape[0], device=device, dtype=dtype)
            b_qkv: Optional[torch.Tensor] = torch.cat(
                [bq if bq is not None else z, bk if bk is not None else z, bv if bv is not None else z], dim=0
            ).contiguous()
        else:
            b_qkv = None

        w_out, b_out = extract_linear_weight_bias(layer.attn.out_proj, dtype)
        w1, b1 = extract_linear_weight_bias(layer.ffn.w1, dtype)
        w3, b3 = extract_linear_weight_bias(layer.ffn.w3, dtype)
        w_13 = torch.cat([w1, w3], dim=0).contiguous()
        if b1 is not None or b3 is not None:
            z_f = torch.zeros(w1.shape[0], device=device, dtype=dtype)
            b_13: Optional[torch.Tensor] = torch.cat(
                [b1 if b1 is not None else z_f, b3 if b3 is not None else z_f], dim=0
            ).contiguous()
        else:
            b_13 = None
        w_2, b_2 = extract_linear_weight_bias(layer.ffn.w2, dtype)

        bundle = (w_qkv, b_qkv, w_out, b_out, w_13, b_13, w_2, b_2)
        self._fused_eval_layers[key] = bundle
        return bundle

    def _run_layer_batched_sdpa(self, layer: nn.Module, x: torch.Tensor, *, layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, emb_dim = x.shape
        h = layer.attn_norm(x)
        num_heads = layer.attn.num_heads
        head_dim = layer.attn.head_dim
        fast_eval = (not self.training) and (not torch.is_grad_enabled()) and (layer_idx is not None)
        if fast_eval:
            w_qkv, b_qkv, w_out, b_out, w_13, b_13, w_2, b_2 = self._get_fused_layer_weights(
                layer_idx, layer, x.device, h.dtype
            )
            q, k, v = F.linear(h, w_qkv, b_qkv).chunk(3, dim=-1)
            q = q.reshape(bsz, seq_len, num_heads, head_dim)
            k = k.reshape(bsz, seq_len, num_heads, head_dim)
            v = v.reshape(bsz, seq_len, num_heads, head_dim)
        else:
            q = layer.attn.q_proj(h).view(bsz, seq_len, num_heads, head_dim)
            k = layer.attn.k_proj(h).view(bsz, seq_len, num_heads, head_dim)
            v = layer.attn.v_proj(h).view(bsz, seq_len, num_heads, head_dim)
        if layer.attn.rope is not None:
            q, k = layer.attn.rope.apply_qk(q, k)
        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=(layer.attn.dropout if self.training else 0.0),
            is_causal=True,
        )
        attn_flat = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, emb_dim)
        if fast_eval:
            x = x + F.linear(attn_flat, w_out, b_out)
            g1, g3 = F.linear(layer.ffn_norm(x), w_13, b_13).chunk(2, dim=-1)
            x = x + F.linear(F.silu(g1) * g3, w_2, b_2)
            return x
        attn_out = layer.attn.out_proj(attn_flat)
        x = x + layer.dropout(attn_out)
        x = x + layer.dropout(layer.ffn(layer.ffn_norm(x)))
        return x

    def _run_layer_kv_step(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        kv_cache: Dict[str, Any],
        kv_valid_mask: Optional[torch.Tensor] = None,
        *,
        layer_idx: Optional[int] = None,
        prebuilt_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_q, emb_dim = x.shape
        h = layer.attn_norm(x)
        num_heads = layer.attn.num_heads
        head_dim = layer.attn.head_dim
        fast_eval = (not self.training) and (not torch.is_grad_enabled()) and (layer_idx is not None)
        if fast_eval:
            w_qkv, b_qkv, w_out, b_out, w_13, b_13, w_2, b_2 = self._get_fused_layer_weights(
                layer_idx, layer, x.device, h.dtype
            )
            q, k, v = F.linear(h, w_qkv, b_qkv).chunk(3, dim=-1)
            q = q.reshape(bsz, seq_q, num_heads, head_dim)
            k = k.reshape(bsz, seq_q, num_heads, head_dim)
            v = v.reshape(bsz, seq_q, num_heads, head_dim)
        else:
            q = layer.attn.q_proj(h).view(bsz, seq_q, num_heads, head_dim)
            k = layer.attn.k_proj(h).view(bsz, seq_q, num_heads, head_dim)
            v = layer.attn.v_proj(h).view(bsz, seq_q, num_heads, head_dim)
        if layer.attn.rope is not None:
            q, k = layer.attn.rope.apply_qk(q, k)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        k_all, v_all = append_kv(kv_cache, k_t, v_t)
        kv_cache["k"] = k_all
        kv_cache["v"] = v_all
        kv_cache["step"] = kv_cache["step"] + 1

        seq_k = k_all.shape[2]
        if prebuilt_attn_mask is not None:
            attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all, attn_mask=prebuilt_attn_mask)
        elif kv_valid_mask is not None:
            if seq_q == 1:
                attn_mask = kv_valid_mask[:, None, None, :]
            elif seq_q == 2:
                m0 = kv_valid_mask.clone()
                m0[:, -1] = False
                attn_mask = torch.stack([m0, kv_valid_mask], dim=1).unsqueeze(1)
            else:
                tril = torch.ones((seq_q, seq_k), dtype=torch.bool, device=x.device).tril(diagonal=seq_k - seq_q)
                attn_mask = tril.unsqueeze(0).unsqueeze(0) & kv_valid_mask[:, None, None, :]
            attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all, attn_mask=attn_mask)
        elif seq_q == 1:
            attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all, is_causal=False)
        elif seq_q == seq_k:
            attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all, is_causal=True)
        elif _HAS_CAUSAL_LOWER_RIGHT and _causal_lower_right is not None:
            attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all, attn_mask=_causal_lower_right(seq_q, seq_k))
        else:
            mask = _suffix_causal_sdpa_mask(seq_q, seq_k, str(x.device))
            attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all, attn_mask=mask)

        attn_flat = attn_out.transpose(1, 2).contiguous().view(bsz, seq_q, emb_dim)
        if fast_eval:
            x = x + F.linear(attn_flat, w_out, b_out)
            g1, g3 = F.linear(layer.ffn_norm(x), w_13, b_13).chunk(2, dim=-1)
            x = x + F.linear(F.silu(g1) * g3, w_2, b_2)
            return x
        attn_out = layer.attn.out_proj(attn_flat)
        x = x + layer.dropout(attn_out)
        x = x + layer.dropout(layer.ffn(layer.ffn_norm(x)))
        return x

    def forward_sequence(
        self,
        decoder_input: DecoderInputFeat,
        cu_lens: torch.Tensor,
        max_len: int,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Right-padded parallel sequence forward returning flat `(N_tokens, emb_dim)` octree states."""
        device = targets_norm.device
        bsz = targets_norm.shape[0]
        dec_flat = self.backbone.input_embedding(decoder_input)
        global_c, has_any = self.conditioner.encode_condition(targets_norm, active_mask)

        root_indices = cu_lens[:-1].long()
        root_delta = self.conditioner.forward_root_prefix(global_c).to(dec_flat.dtype)
        dec_flat = dec_flat.clone()
        dec_flat[root_indices] = dec_flat[root_indices] + root_delta

        n_tokens = dec_flat.shape[0]
        token_idx = torch.arange(n_tokens, device=device)
        batch_ids = torch.searchsorted(cu_lens[1:].long(), token_idx, right=True)
        intra_pos = token_idx - cu_lens[batch_ids].long()

        emb_dim = dec_flat.shape[-1]
        seq_padded = dec_flat.new_zeros((bsz, max_len, emb_dim))
        pos_padded = decoder_input.phy_pos.new_zeros((bsz, max_len, 3))
        seq_padded[batch_ids, intra_pos] = dec_flat
        pos_padded[batch_ids, intra_pos] = decoder_input.phy_pos

        prefix = self.conditioner.build_prefix_tokens(global_c, has_any)
        p_len = self.conditioner.num_prefix_tokens
        seq_grid = pos_padded.float() / self.args.xyz_resolution
        if prefix is not None and p_len > 0:
            x = torch.cat([prefix.to(seq_padded.dtype), seq_padded], dim=1)
            prefix_grid = self._prefix_grid_positions(bsz, device)
            grid_pos = torch.cat([prefix_grid, seq_grid], dim=1)
        else:
            x = seq_padded
            grid_pos = seq_grid

        c_3d = global_c.unsqueeze(1)
        self.backbone.rope.calc_and_cache(grid_pos)
        for _ in range(self.args.recycle):
            for l_idx, layer in enumerate(self.backbone.decoder.layers):
                x = self._run_layer_batched_sdpa(layer, x, layer_idx=l_idx)
                x = self.conditioner.modulate_layer(l_idx, x, c_3d)

        x = self.backbone.norm(x)
        x = self.conditioner.modulate_out(x, c_3d)
        out_seq = x[:, p_len:, :]
        return out_seq[batch_ids, intra_pos]

    # ------------------------------------------------------------------
    # Supervised / Hindsight + Contrastive Margin Objective
    # ------------------------------------------------------------------
    def _compute_valence_aux_loss(
        self,
        atom_hidden: torch.Tensor,
        gts_xyz: List[torch.Tensor],
        atom_vox_phy_pos: torch.Tensor,
        cu_lens: torch.Tensor,
        is_gen_atom: torch.Tensor,
        bsz: int,
    ) -> torch.Tensor:
        """Vectorized GPU supervision of `(num_attached_H in {0..4}, formal_charge in {-1,0,+1})` on heavy atoms."""
        assert self.valence_head is not None
        device = atom_hidden.device
        elem_ids = gts_xyz[0]
        h_id = self.dictionary["H"]
        n_id = self.dictionary["N"]
        o_id = self.dictionary["O"]
        real_mask = (elem_ids >= 0) & (elem_ids < len(atom_list))
        heavy_mask = real_mask & (elem_ids != h_id)
        if not bool(heavy_mask.any().item()):
            return torch.zeros((), device=device)

        owners_all = torch.repeat_interleave(torch.arange(bsz, device=device), torch.diff(cu_lens).long())
        atom_owners = owners_all[is_gen_atom]

        vox_origins = atom_vox_phy_pos.float() - 0.5 * float(self.args.grid_len)
        xyz_bins = torch.stack(gts_xyz[1:4], dim=-1).float()
        atom_pos = vox_origins + xyz_bins * float(self.args.xyz_resolution)

        same_mol = atom_owners.unsqueeze(1) == atom_owners.unsqueeze(0)
        not_self = ~torch.eye(atom_pos.shape[0], dtype=torch.bool, device=device)
        d_aa = torch.cdist(atom_pos, atom_pos)

        is_h_col = (elem_ids == h_id).unsqueeze(0)
        h_bonded = same_mol & not_self & is_h_col & (d_aa <= 1.32)
        gt_num_h_all = h_bonded.sum(dim=-1).clamp(0, 4).long()

        # Dipolar charge proxy on heavy atoms: {-1 -> 0, 0 -> 1, +1 -> 2}
        rcov_table, _ = _atom_geometry_tables(self.backbone.num_atom, str(device))
        r_vec = rcov_table[elem_ids.clamp(0, self.backbone.num_atom - 1)]
        heavy_col = heavy_mask.unsqueeze(0)
        heavy_bonded = same_mol & not_self & heavy_col & (d_aa <= 1.26 * (r_vec.unsqueeze(1) + r_vec.unsqueeze(0)))
        heavy_deg = heavy_bonded.sum(dim=-1)
        has_o_nb = (heavy_bonded & (elem_ids == o_id).unsqueeze(0)).any(dim=-1)

        is_n_plus = (elem_ids == n_id) & (heavy_deg >= 3) & has_o_nb & (gt_num_h_all == 0)
        bonded_to_n_plus = (heavy_bonded & is_n_plus.unsqueeze(0)).any(dim=-1)
        is_o_minus = (elem_ids == o_id) & (heavy_deg == 1) & (gt_num_h_all == 0) & bonded_to_n_plus

        gt_charge_cls_all = torch.ones_like(gt_num_h_all)
        gt_charge_cls_all = torch.where(is_o_minus, torch.zeros_like(gt_charge_cls_all), gt_charge_cls_all)
        gt_charge_cls_all = torch.where(is_n_plus, torch.full_like(gt_charge_cls_all, 2), gt_charge_cls_all)

        logits_h, logits_q = self.valence_head(atom_hidden[heavy_mask])
        loss_h = F.cross_entropy(logits_h.float(), gt_num_h_all[heavy_mask])
        loss_q = F.cross_entropy(logits_q.float(), gt_charge_cls_all[heavy_mask])
        return loss_h + 0.5 * loss_q

    def _nll_from_hidden(
        self,
        dec_y: torch.Tensor,
        decoder_gen_target: Any,
        cu_lens: torch.Tensor,
        bsz: int,
        *,
        decoder_input: Optional[DecoderInputFeat] = None,
        xyz_loss_weight: float = 0.1,
        valence_loss_weight: float = 0.15,
        atom_class_weights: Optional[torch.Tensor] = None,
        return_per_molecule: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = dec_y.device
        is_gen_tree = decoder_gen_target.is_gen_tree
        is_gen_atom = decoder_gen_target.is_gen_atom

        loss_tree = torch.zeros((), device=device)
        logits_tree = None
        gt_tree = None
        if is_gen_tree.any():
            logits_tree = self.backbone.tree_heads[0](dec_y[is_gen_tree]).float()
            gt_tree = (decoder_gen_target.type[is_gen_tree] - self.backbone.tree0_split_id - 1).long()
            loss_tree = F.cross_entropy(logits_tree, gt_tree)

        loss_atom = torch.zeros((), device=device)
        loss_xyz = torch.zeros((), device=device)
        loss_valence = torch.zeros((), device=device)
        atom_logits = None
        preds_xyz = None
        gts_xyz = None
        if is_gen_atom.any():
            gts_xyz = [v.long() for v in decoder_gen_target.to_list(is_gen_atom)]
            preds_xyz = self.backbone.atom_heads[0](dec_y[is_gen_atom], gts_xyz)
            atom_logits = preds_xyz[0].float().masked_fill(~self.allowed_atom_mask(device), -torch.inf)
            loss_atom = F.cross_entropy(atom_logits, gts_xyz[0], weight=atom_class_weights)
            loss_xyz = (
                F.cross_entropy(preds_xyz[1].float(), gts_xyz[1])
                + F.cross_entropy(preds_xyz[2].float(), gts_xyz[2])
                + F.cross_entropy(preds_xyz[3].float(), gts_xyz[3])
            ) / 3.0

            if self.valence_head is not None and decoder_input is not None and valence_loss_weight > 0.0:
                loss_valence = self._compute_valence_aux_loss(
                    dec_y[is_gen_atom], gts_xyz, decoder_input.phy_pos[is_gen_atom], cu_lens, is_gen_atom, bsz
                )

        gen_loss = loss_tree + loss_atom + xyz_loss_weight * loss_xyz + valence_loss_weight * loss_valence
        info: Dict[str, Any] = {
            "loss_gen": float(gen_loss.detach().item()),
            "loss_tree": float(loss_tree.detach().item()),
            "loss_atom": float(loss_atom.detach().item()),
            "loss_xyz": float(loss_xyz.detach().item()),
            "loss_valence": float(loss_valence.detach().item()),
        }

        if return_per_molecule:
            owners = torch.repeat_interleave(torch.arange(bsz, device=device), torch.diff(cu_lens).long())
            per_mol = torch.zeros(bsz, device=device, dtype=torch.float32)

            def _avg_decisions(vals: torch.Tensor, sel: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
                ids = owners[sel]
                summed = torch.zeros_like(per_mol).index_add(0, ids, vals)
                counts = (
                    torch.bincount(ids, minlength=bsz).to(per_mol.dtype)
                    if weights is None
                    else torch.zeros_like(per_mol).index_add(0, ids, weights)
                )
                return summed / counts.clamp_min(1e-8)

            if is_gen_tree.any() and logits_tree is not None and gt_tree is not None:
                per_mol = per_mol + _avg_decisions(
                    F.cross_entropy(logits_tree, gt_tree, reduction="none"), is_gen_tree
                )
            if is_gen_atom.any() and atom_logits is not None and preds_xyz is not None and gts_xyz is not None:
                a_losses = F.cross_entropy(atom_logits, gts_xyz[0], reduction="none", weight=atom_class_weights)
                xyz_losses = (
                    F.cross_entropy(preds_xyz[1].float(), gts_xyz[1], reduction="none")
                    + F.cross_entropy(preds_xyz[2].float(), gts_xyz[2], reduction="none")
                    + F.cross_entropy(preds_xyz[3].float(), gts_xyz[3], reduction="none")
                ) / 3.0
                d_weights = atom_class_weights[gts_xyz[0]] if atom_class_weights is not None else None
                per_mol = per_mol + _avg_decisions(a_losses, is_gen_atom, d_weights) + xyz_loss_weight * _avg_decisions(
                    xyz_losses, is_gen_atom
                )
            info["per_molecule_generation_loss"] = per_mol

        return gen_loss, info

    def compute_sft_loss(
        self,
        packed_batch: Dict[str, torch.Tensor],
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        cfg_dropout_prob: float = 0.0,
        contrast_weight: float = 0.0,
        contrast_margin: float = 0.05,
        contrast_min_gap_eV: float = 0.25,
        contrast_col_mask: Optional[torch.Tensor] = None,
        xyz_loss_weight: float = 0.1,
        valence_loss_weight: float = 0.15,
        atom_class_weights: Optional[torch.Tensor] = None,
        return_per_molecule: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = targets_norm.device
        bsz = targets_norm.shape[0]
        bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in packed_batch.items()}
        bd["decoder_phy_pos"] = bd["decoder_phy_pos"].float()

        decoder_input, decoder_gen_target, _ = self.backbone.get_feat_from_dataloader(bd)
        cu_lens = bd["decoder_input_len"].to(device, dtype=torch.int32)
        max_len = int(bd["decoder_input_max_len"].item())

        train_mask = active_mask.bool().clone()
        if self.training and cfg_dropout_prob > 0.0:
            keep = torch.rand((bsz, 1), device=device) >= cfg_dropout_prob
            train_mask = train_mask & keep

        need_per_mol = return_per_molecule or (contrast_weight > 0.0)
        dec_y = self.forward_sequence(decoder_input, cu_lens, max_len, targets_norm, train_mask)
        gen_loss, info = self._nll_from_hidden(
            dec_y,
            decoder_gen_target,
            cu_lens,
            bsz,
            decoder_input=decoder_input,
            xyz_loss_weight=xyz_loss_weight,
            valence_loss_weight=valence_loss_weight,
            atom_class_weights=atom_class_weights,
            return_per_molecule=need_per_mol,
        )

        total_loss = gen_loss
        if contrast_weight > 0.0:
            targets_eV = self.denormalize_targets(targets_norm)
            if contrast_col_mask is not None:
                c_mask = contrast_col_mask.to(device=device, dtype=torch.bool)
                if c_mask.ndim == 1:
                    c_mask = c_mask.unsqueeze(0)
                select_mask = train_mask & c_mask
                eff_gap = float(contrast_min_gap_eV)
            else:
                c_mask = None
                select_mask = train_mask
                eff_gap = max(contrast_min_gap_eV, 0.40)
            neg_idx, eligible = _select_incompatible_negatives(
                targets_eV, select_mask, min_gap_eV=eff_gap, deterministic=False
            )
            if bool(eligible.any()):
                if c_mask is not None:
                    neg_targets_norm = torch.where(c_mask, targets_norm[neg_idx], targets_norm)
                else:
                    neg_targets_norm = targets_norm[neg_idx]
                dec_neg = self.forward_sequence(decoder_input, cu_lens, max_len, neg_targets_norm, train_mask)
                _, neg_info = self._nll_from_hidden(
                    dec_neg,
                    decoder_gen_target,
                    cu_lens,
                    bsz,
                    xyz_loss_weight=xyz_loss_weight,
                    atom_class_weights=atom_class_weights,
                    return_per_molecule=True,
                )
                pos_per_mol = info["per_molecule_generation_loss"]
                neg_per_mol = neg_info["per_molecule_generation_loss"]
                c_loss = condition_margin_loss(pos_per_mol, neg_per_mol, eligible, margin=contrast_margin)
                pref = float((pos_per_mol[eligible] < neg_per_mol[eligible]).float().mean().item())
                total_loss = total_loss + contrast_weight * c_loss
                info["loss_contrast"] = float(c_loss.detach().item())
                info["inbatch_preference_rate"] = pref
                info["contrast_eligible"] = int(eligible.sum().item())
            else:
                info["loss_contrast"] = 0.0
                info["inbatch_preference_rate"] = float("nan")
                info["contrast_eligible"] = 0

        info["loss_total"] = float(total_loss.detach().item())
        return total_loss, info

    def build_replay_dataset(self, mols: Sequence[Chem.Mol], *, seed: int = 42) -> OctreeDataset:
        return OctreeDataset(list(mols), self.backbone, training=True, seed=seed)

    def encode_molecules_batch(
        self,
        mols: Sequence[Chem.Mol],
        *,
        seed: int = 42,
        repeat_k: int = 1,
    ) -> Tuple[Optional[Dict[str, Any]], List[int]]:
        if not mols:
            return None, []
        device = next(self.parameters()).device
        temp_ds = OctreeDataset(list(mols), self.backbone, training=False, seed=seed)
        kept_indices: List[int] = []
        for idx in range(len(mols)):
            try:
                temp_ds.collate([idx])
                kept_indices.append(idx)
            except ValueError:
                continue
        if not kept_indices:
            return None, []
        expanded = kept_indices * max(1, int(repeat_k))
        try:
            raw_packed = temp_ds.collate(expanded)
            packed = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw_packed.items()}
            return packed, expanded
        except ValueError:
            return None, []

    @torch.no_grad()
    def evaluate_teacher_forcing_preference(
        self,
        packed_batch: Dict[str, torch.Tensor],
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        min_gap_eV: float = 0.25,
    ) -> Dict[str, Any]:
        """Evaluate held-out teacher-forcing preference rate (Stage 0 Gate: > 92%)."""
        was_training = self.training
        self.eval()
        try:
            device = targets_norm.device
            bsz = targets_norm.shape[0]
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in packed_batch.items()}
            bd["decoder_phy_pos"] = bd["decoder_phy_pos"].float()
            decoder_input, decoder_gen_target, _ = self.backbone.get_feat_from_dataloader(bd)
            cu_lens = bd["decoder_input_len"].to(device, dtype=torch.int32)
            max_len = int(bd["decoder_input_max_len"].item())

            dec_pos = self.forward_sequence(decoder_input, cu_lens, max_len, targets_norm, active_mask)
            _, pos_info = self._nll_from_hidden(
                dec_pos, decoder_gen_target, cu_lens, bsz, return_per_molecule=True
            )
            pos_nll = pos_info["per_molecule_generation_loss"]

            targets_eV = self.denormalize_targets(targets_norm)
            neg_idx, eligible = _select_incompatible_negatives(
                targets_eV, active_mask, min_gap_eV=max(min_gap_eV, 0.50), deterministic=True
            )

            if not bool(eligible.any()):
                return {"eligible": 0, "preferred": 0, "preference_rate": float("nan"), "mean_nll_gap": float("nan")}

            dec_neg = self.forward_sequence(decoder_input, cu_lens, max_len, targets_norm[neg_idx], active_mask)
            _, neg_info = self._nll_from_hidden(
                dec_neg, decoder_gen_target, cu_lens, bsz, return_per_molecule=True
            )
            neg_nll = neg_info["per_molecule_generation_loss"]
            gaps = neg_nll[eligible] - pos_nll[eligible]
            preferred = int((gaps > 0).sum().item())
            n_el = int(eligible.sum().item())
            return {
                "eligible": n_el,
                "preferred": preferred,
                "preference_rate": preferred / n_el,
                "mean_nll_gap": float(gaps.mean().item()),
            }
        finally:
            self.train(was_training)

    # ------------------------------------------------------------------
    # Fast Sync-Free Batched Sampling with Selective Logit-CFG
    # ------------------------------------------------------------------
    def _fast_tree_expand(
        self,
        raw_feat: List[torch.Tensor],
        cur_type: torch.Tensor,
        pred_type: torch.Tensor,
        next_feat: torch.Tensor,
        count_vars: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        device = next_feat.device
        next_pos = next_feat[:, :3]
        next_level = next_feat[:, IDX_LEVEL]
        next_tree_index = next_feat[:, IDX_TREE]
        next_space_index = next_feat[:, IDX_SPACE]

        is_split = (cur_type == self.backbone.atom_mask_id) & (next_level > self.backbone.atom_level_index)
        pred_type_binary = (((pred_type.unsqueeze(-1) - self.backbone.tree0_split_id) & self._bit_shifts) > 0) & is_split.unsqueeze(-1)
        counts_t = pred_type_binary.sum(dim=-1).to(torch.int32)
        count_vars["acc_tokens"] = count_vars["acc_tokens"] + counts_t
        counts_list = counts_t.tolist()  # Single D2H sync for the entire step!

        total_splits = sum(counts_list)
        if total_splits == 0:
            return raw_feat, count_vars

        next_cell_len = (1 << (next_level - 1)).view(-1, 1, 1)
        expanded_pos = next_pos.unsqueeze(1) + self._subcell_shifts * next_cell_len  # (B, 8, 3)
        flat_pos = expanded_pos[pred_type_binary]  # (M, 3)
        flat_lvl = (next_level - 1).unsqueeze(1).expand(-1, 8)[pred_type_binary]
        flat_tree = next_tree_index.unsqueeze(1).expand(-1, 8)[pred_type_binary]
        flat_space = next_space_index.unsqueeze(1).expand(-1, 8)[pred_type_binary]

        pairs = torch.zeros((total_splits, 2, NUM_FEAT), dtype=torch.int32, device=device)
        pairs[:, :, :3] = flat_pos.unsqueeze(1)
        pairs[:, 0, IDX_TYPE] = self.backbone.atom_mask_id
        pairs[:, 1, IDX_TYPE] = -1
        pairs[:, :, IDX_LEVEL] = flat_lvl.unsqueeze(1)
        pairs[:, :, IDX_TREE] = flat_tree.unsqueeze(1)
        pairs[:, :, IDX_SPACE] = flat_space.unsqueeze(1)

        chunks = torch.split(pairs.view(2 * total_splits, NUM_FEAT), [2 * c for c in counts_list], dim=0)
        new_raw_feat = [
            (torch.cat([feat, chunk], dim=0) if c > 0 else feat)
            for feat, chunk, c in zip(raw_feat, chunks, counts_list)
        ]
        return new_raw_feat, count_vars

    def _fast_update_running_stats(
        self,
        count_vars: Dict[str, torch.Tensor],
        is_new_level: torch.Tensor,
        last_level: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        cur_rem_a = count_vars["last_remaining_atoms"] - count_vars["pred_count"]
        cur_rem_t = count_vars["last_remaining_tokens"] - 1

        cur_rem_a = torch.where(is_new_level, count_vars["total_atoms"], cur_rem_a)
        cur_rem_t = torch.where(is_new_level, count_vars["acc_tokens"], cur_rem_t)
        count_vars["acc_tokens"] = torch.where(is_new_level, torch.zeros_like(count_vars["acc_tokens"]), count_vars["acc_tokens"])

        if self.backbone.num_tree > 1:
            is_new_tree = is_new_level & (last_level == self.backbone.atom_level_index - 1)
            count_vars["total_atoms"] = torch.where(is_new_tree, torch.zeros_like(count_vars["total_atoms"]), count_vars["total_atoms"])
            count_vars["acc_tokens"] = torch.where(is_new_tree, torch.zeros_like(count_vars["acc_tokens"]), count_vars["acc_tokens"])
            cur_rem_a = torch.where(is_new_tree, torch.zeros_like(cur_rem_a), cur_rem_a)
            cur_rem_t = torch.where(is_new_tree, torch.ones_like(cur_rem_t), cur_rem_t)

        pair_rem_a = torch.stack([count_vars["last_remaining_atoms"], cur_rem_a], dim=1)
        pair_rem_t = torch.stack([count_vars["last_remaining_tokens"], cur_rem_t], dim=1)
        return count_vars, pair_rem_a, pair_rem_t

    def _write_feat_by_pred_inplace(
        self,
        next_feat: torch.Tensor,
        dec_feat_buf: torch.Tensor,
        dec_pos_buf: torch.Tensor,
        write_col: int,
        pred_type: torch.Tensor,
        pred_count: Optional[torch.Tensor] = None,
        pred_xyz: Optional[torch.Tensor] = None,
    ) -> None:
        next_pos = next_feat[:, :3]
        next_level = next_feat[:, IDX_LEVEL]
        edge_len = (1 << next_level.clamp_min(0)).view(-1, 1)

        slot_f = dec_feat_buf[:, write_col, :]
        slot_f.zero_()
        slot_f[:, :3] = self.backbone.xyz_null_id
        slot_f[:, IDX_TYPE] = pred_type
        slot_f[:, IDX_LEVEL] = next_level
        slot_f[:, IDX_TREE] = next_feat[:, IDX_TREE]
        slot_f[:, IDX_SPACE] = next_feat[:, IDX_SPACE]
        if pred_count is not None:
            slot_f[:, IDX_CNT] = pred_count
        else:
            slot_f[:, IDX_CNT] = -1

        next_pos_f = next_pos.float()
        cur_phy_pos = (next_pos_f + edge_len.float() * 0.5) * float(self.args.grid_len)
        if pred_xyz is not None:
            is_atom_lvl = next_level == self.backbone.atom_level_index
            atom_pos = next_pos_f * float(self.args.grid_len) + pred_xyz.float() * float(self.args.xyz_resolution)
            cur_phy_pos = torch.where(is_atom_lvl.unsqueeze(-1), atom_pos, cur_phy_pos)
            slot_f[:, :3] = torch.where(is_atom_lvl.unsqueeze(-1), pred_xyz, slot_f[:, :3])

        is_eot = next_level == (self.backbone.atom_level_index - 1)
        cls_p = float(self.backbone.get_cls_phy_pos(self.args.merge_level))
        cur_phy_pos = torch.where(is_eot.unsqueeze(-1), torch.full_like(cur_phy_pos, cls_p), cur_phy_pos)
        dec_pos_buf[:, write_col, :] = cur_phy_pos

    def _fast_tree_expand_gpu(
        self,
        q_buf: torch.Tensor,
        q_tail: torch.Tensor,
        q_cap: int,
        cur_type: torch.Tensor,
        pred_type: torch.Tensor,
        next_feat: torch.Tensor,
        count_vars: Dict[str, torch.Tensor],
        step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, torch.Tensor]]:
        device = next_feat.device
        b_active = next_feat.shape[0]
        next_pos = next_feat[:, :3]
        next_level = next_feat[:, IDX_LEVEL]

        is_split = (cur_type == self.backbone.atom_mask_id) & (next_level > self.backbone.atom_level_index)
        pred_type_binary = (
            (((pred_type.unsqueeze(-1) - self.backbone.tree0_split_id) & self._bit_shifts) > 0)
            & is_split.unsqueeze(-1)
        )
        counts_t = pred_type_binary.sum(dim=-1).to(torch.int32)
        count_vars["acc_tokens"] = count_vars["acc_tokens"] + counts_t

        if 1 + 16 * step_idx >= q_cap:
            if bool((q_tail + 18 >= q_cap).any().item()):
                new_cap = q_cap * 2
                new_buf = torch.zeros((b_active, new_cap + 16, NUM_FEAT), dtype=torch.int32, device=device)
                new_buf[:, :q_cap, :] = q_buf[:, :q_cap, :]
                q_buf = new_buf
                q_cap = new_cap

        next_cell_len = (1 << (next_level - 1).clamp_min(0)).view(-1, 1, 1)
        expanded_pos = next_pos.unsqueeze(1) + self._subcell_shifts * next_cell_len  # (B, 8, 3)

        cand_pairs = torch.zeros((b_active, 8, 2, NUM_FEAT), dtype=torch.int32, device=device)
        cand_pairs[:, :, :, :3] = expanded_pos.unsqueeze(2)
        cand_pairs[:, :, 0, IDX_TYPE] = self.backbone.atom_mask_id
        cand_pairs[:, :, 1, IDX_TYPE] = -1
        cand_pairs[:, :, :, IDX_LEVEL] = (next_level - 1).view(-1, 1, 1)
        cand_pairs[:, :, :, IDX_TREE] = next_feat[:, IDX_TREE].view(-1, 1, 1)
        cand_pairs[:, :, :, IDX_SPACE] = next_feat[:, IDX_SPACE].view(-1, 1, 1)

        oct_rank = pred_type_binary.long().cumsum(dim=1) - 1
        base_slot = q_tail.unsqueeze(1) + 2 * oct_rank
        pair_slots = torch.stack((base_slot, base_slot + 1), dim=2)
        trash_slots = q_cap + torch.arange(16, device=device, dtype=torch.long).view(1, 8, 2)
        safe_slots = torch.where(pred_type_binary.unsqueeze(2), pair_slots, trash_slots).view(b_active, 16)

        q_buf.scatter_(1, safe_slots.unsqueeze(-1).expand(-1, -1, NUM_FEAT), cand_pairs.view(b_active, 16, NUM_FEAT))
        q_tail = q_tail + 2 * counts_t.long()
        return q_buf, q_tail, q_cap, count_vars

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
        max_decode_steps: int = 256,
        uncond_active_mask: Optional[torch.Tensor] = None,
        return_coarse_summary: bool = False,
        leaf_atom_budget: Optional[torch.Tensor] = None,
        leaf_budget_tol: float = 2.0,
        sort_by_budget: bool = True,
        _stop_at_leaf_boundary: bool = False,
        _capture_continuation: bool = False,
        _replay_coarse_actions: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """A800-optimized conditional octree trajectory generation with zero-alloc KV buffer & GPU queue."""
        was_training = self.training
        self.eval()
        try:
            device = targets_norm.device
            batch_size = int(targets_norm.shape[0])
            if batch_size == 0:
                return []
            tree_temp, atom_temp, xyz_temp = temperatures
            for name, val in (("tree", tree_temp), ("atom", atom_temp), ("xyz", xyz_temp)):
                if not math.isfinite(val) or val <= 0:
                    raise ValueError(f"{name} temperature must be positive and finite")

            # Optional transparent length-bucketing so similar-size octrees share batches
            inv_perm_list: Optional[List[int]] = None
            if (
                leaf_atom_budget is not None
                and sort_by_budget
                and batch_size > 16
                and _replay_coarse_actions is None
                and not _capture_continuation
            ):
                b_vec = torch.as_tensor(leaf_atom_budget, dtype=torch.float32, device=device).view(batch_size)
                perm = torch.argsort(b_vec, stable=True)
                inv_perm = torch.empty_like(perm)
                inv_perm[perm] = torch.arange(batch_size, device=device)
                inv_perm_list = inv_perm.detach().cpu().tolist()
                targets_norm = targets_norm[perm]
                active_mask = active_mask[perm]
                if uncond_active_mask is not None:
                    uncond_active_mask = uncond_active_mask[perm]
                leaf_atom_budget = b_vec[perm]

            has_cond = bool(active_mask.bool().any().item())
            use_cfg = has_cond and (cfg_scale != 1.0 or xyz_cfg_scale != 1.0)

            rng_gen: Optional[torch.Generator] = None
            if seed is not None:
                rng_gen = torch.Generator(device=device)
                rng_gen.manual_seed(int(seed))

            parities = atom_electron_parities(self.backbone, device) if electron_parity is not None else None
            forbidden_atom_mask = ~self.allowed_atom_mask(device)
            score_ratio = [1.0, 0.0, 0.0, 0.0] if self.args.rank_by == "atom" else [1.0, 1.0, 1.0, 1.0]

            budget_vec: Optional[torch.Tensor] = None
            if leaf_atom_budget is not None:
                budget_vec = torch.as_tensor(leaf_atom_budget, dtype=torch.float32, device=device).view(batch_size)
            pops_1d = _split_popcount_table(str(device))

            autocast_enabled = bool(use_bf16 and device.type == "cuda")
            run_dtype = torch.bfloat16 if autocast_enabled else torch.float32
            p_len = self.conditioner.num_prefix_tokens
            kv_init_cap = max(64, 2 * int(max_decode_steps) + p_len + 16)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                global_c, has_any = self.conditioner.encode_condition(targets_norm, active_mask)
                prefix_cond = self.conditioner.build_prefix_tokens(global_c, has_any)
                root_delta = self.conditioner.forward_root_prefix(global_c)

                if use_cfg:
                    null_mask = (
                        uncond_active_mask.to(device=device, dtype=torch.bool)
                        if uncond_active_mask is not None
                        else torch.zeros_like(active_mask, dtype=torch.bool)
                    )
                    global_c_uncond, has_any_uncond = self.conditioner.encode_condition(targets_norm, null_mask)
                    prefix_uncond = self.conditioner.build_prefix_tokens(global_c_uncond, has_any_uncond)
                    root_delta_uncond = self.conditioner.forward_root_prefix(global_c_uncond)
                    c_run_init = torch.cat([global_c.unsqueeze(1), global_c_uncond.unsqueeze(1)], dim=0)
                else:
                    global_c_uncond = None
                    prefix_uncond = None
                    root_delta_uncond = None
                    c_run_init = global_c.unsqueeze(1)

                layer_mods_cache, out_mod_cache = self.conditioner.precompute_modulations(c_run_init, run_dtype)

                kv_cache = [
                    [{"step": 1, "mode": "buffer", "init_capacity": kv_init_cap} for _ in range(self.args.layer)]
                    for _ in range(self.args.recycle)
                ]

                cond_vars: Dict[str, Optional[torch.Tensor]] = {
                    "global_c": global_c.unsqueeze(1),
                    "global_c_uncond": global_c_uncond.unsqueeze(1) if global_c_uncond is not None else None,
                    "orig_idx": torch.arange(batch_size, dtype=torch.long, device=device),
                }
                count_vars: Dict[str, torch.Tensor] = {
                    "known_atoms": torch.full((batch_size, self.backbone.num_tree), -1, dtype=torch.int32, device=device),
                    "total_atoms": torch.zeros(batch_size, dtype=torch.int32, device=device),
                    "acc_tokens": torch.zeros(batch_size, dtype=torch.int32, device=device),
                }

                init_dec_feat, init_dec_pos, init_raw_feat = self.backbone.calc_root_level(
                    self.args.merge_level, batch_size, device, 0, 0
                )
                dec_cap = 2 * int(max_decode_steps) + 16
                dec_feat_buf = torch.zeros((batch_size, dec_cap, NUM_FEAT), dtype=torch.int32, device=device)
                dec_pos_buf = torch.zeros((batch_size, dec_cap, 3), dtype=torch.float32, device=device)
                scores_buf = torch.zeros((batch_size, dec_cap), dtype=torch.float32, device=device)
                init_len = int(init_dec_feat.shape[1])
                dec_feat_buf[:, :init_len, :] = init_dec_feat
                dec_pos_buf[:, :init_len, :] = init_dec_pos
                dec_len = init_len

                q_cap = 2048
                q_buf = torch.zeros((batch_size, q_cap + 16, NUM_FEAT), dtype=torch.int32, device=device)
                q_head = torch.zeros((batch_size,), dtype=torch.long, device=device)
                q_tail_list = [int(r.shape[0]) for r in init_raw_feat]
                for b_i, r_t in enumerate(init_raw_feat):
                    if r_t.shape[0] > 0:
                        q_buf[b_i, : r_t.shape[0], :] = r_t
                q_tail = torch.tensor(q_tail_list, dtype=torch.long, device=device)

                spawned_leaves = torch.zeros((batch_size,), dtype=torch.float32, device=device)
                rem_l1_cells = (
                    torch.ones((batch_size,), dtype=torch.float32, device=device)
                    if self.args.merge_level == (self.backbone.atom_level_index + 1)
                    else torch.zeros((batch_size,), dtype=torch.float32, device=device)
                )

                finished_trajs: List[Dict[str, Any]] = []
                coarse_actions_rec: Optional[List[List[int]]] = (
                    [[] for _ in range(batch_size)] if (_stop_at_leaf_boundary and not _capture_continuation) else None
                )
                replay_cursors: Optional[List[int]] = (
                    [0] * batch_size if _replay_coarse_actions is not None else None
                )

                cur_feat = dec_feat_buf[:, dec_len - 1, :]
                cur_phy_pos = dec_pos_buf[:, dec_len - 1, :]
                cur_input = DecoderInputFeat(
                    cur_feat[:, IDX_TYPE],
                    cur_feat[:, :3],
                    cur_feat[:, IDX_LEVEL],
                    cur_phy_pos,
                    cur_feat[:, IDX_SPACE],
                    cur_feat[:, IDX_TREE],
                    torch.zeros_like(cur_feat[:, IDX_TYPE]).int(),
                    torch.ones_like(cur_feat[:, IDX_TYPE]).int(),
                    cur_feat[:, IDX_CNT],
                ).add_seq_dim()

                step_idx = 0
                emb_dim = int(self.args.emb_dim)
                if return_coarse_summary:
                    cs_h_root = torch.zeros((batch_size, emb_dim), dtype=torch.float32, device=device)
                    cs_h_coarse_sum = torch.zeros((batch_size, emb_dim), dtype=torch.float32, device=device)
                    cs_h_coarse_cnt = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
                    cs_h_l1_sum = torch.zeros((batch_size, emb_dim), dtype=torch.float32, device=device)
                    cs_h_l1_cnt = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
                    cs_h_l1_last = torch.zeros((batch_size, emb_dim), dtype=torch.float32, device=device)
                else:
                    cs_h_root = cs_h_coarse_sum = cs_h_coarse_cnt = cs_h_l1_sum = cs_h_l1_cnt = cs_h_l1_last = None

                next_feat: Optional[torch.Tensor] = None
                b_active = batch_size

                while b_active > 0:
                    step_emb = self.backbone.input_embedding(cur_input)
                    step_grid = cur_input.phy_pos.float() / self.args.xyz_resolution

                    if step_idx == 0:
                        emb_cond = step_emb + root_delta.unsqueeze(1).to(step_emb.dtype)
                        if prefix_cond is not None and p_len > 0:
                            emb_cond = torch.cat([prefix_cond.to(step_emb.dtype), emb_cond], dim=1)
                            pref_grid = self._prefix_grid_positions(b_active, device)
                            step_grid = torch.cat([pref_grid, step_grid], dim=1)
                        if use_cfg:
                            emb_uncond = (
                                step_emb + root_delta_uncond.unsqueeze(1).to(step_emb.dtype)
                                if root_delta_uncond is not None
                                else step_emb
                            )
                            if prefix_uncond is not None and p_len > 0:
                                emb_uncond = torch.cat([prefix_uncond.to(step_emb.dtype), emb_uncond], dim=1)
                            x_run = torch.cat([emb_cond, emb_uncond], dim=0)
                            grid_run = step_grid.repeat(2, 1, 1)
                        else:
                            x_run = emb_cond
                            grid_run = step_grid
                    else:
                        if use_cfg:
                            x_run = step_emb.repeat(2, 1, 1)
                            grid_run = step_grid.repeat(2, 1, 1)
                        else:
                            x_run = step_emb
                            grid_run = step_grid

                    self.backbone.rope.calc_and_cache(grid_run)
                    for r_i in range(self.args.recycle):
                        for l_i, layer in enumerate(self.backbone.decoder.layers):
                            x_run = self._run_layer_kv_step(layer, x_run, kv_cache[r_i][l_i], layer_idx=l_i)
                            ops_l, shf_l = layer_mods_cache[l_i]
                            x_run = torch.addcmul(shf_l, x_run, ops_l)
                    x_run = self.backbone.norm(x_run)
                    ops_o, shf_o = out_mod_cache
                    x_run = torch.addcmul(shf_o, x_run, ops_o)

                    last_hidden = x_run[:, -1, :]
                    if use_cfg:
                        dec_y_last, dec_u_last = last_hidden.chunk(2, dim=0)
                    else:
                        dec_y_last = last_hidden
                        dec_u_last = None

                    step_idx += 1
                    last_level = cur_input.level[:, -1]
                    last_type = cur_input.type[:, -1]
                    last_rem_atoms = cur_input.remaining_atoms[:, -1]
                    last_rem_tokens = cur_input.remaining_tokens[:, -1]

                    row_arange = torch.arange(b_active, device=device)
                    next_feat = q_buf[row_arange, q_head]
                    q_head = q_head + 1

                    next_type = next_feat[:, IDX_TYPE]
                    next_level = next_feat[:, IDX_LEVEL]
                    need_pred = next_type < 0
                    is_pred_atom = (
                        (last_level == self.backbone.atom_level_index)
                        & (last_type == self.backbone.atom_mask_id)
                        & need_pred
                    )
                    is_pred_tree = (next_level > self.backbone.atom_level_index) & need_pred
                    is_first_cnt = (last_level == self.args.merge_level) & (last_type == self.backbone.atom_mask_id)

                    # Single consolidated host-sync for branch flags `(has_atom, has_tree)`
                    branch_flags = torch.stack((is_pred_atom.any(), is_pred_tree.any())).tolist()
                    has_atom = bool(branch_flags[0])
                    has_tree = bool(branch_flags[1])

                    if return_coarse_summary and cs_h_root is not None:
                        orig_ids = cond_vars["orig_idx"]
                        assert orig_ids is not None
                        if step_idx == 1:
                            cs_h_root[orig_ids] = dec_y_last.float()
                        if has_tree:
                            t_orig = orig_ids[is_pred_tree]
                            t_hid = dec_y_last[is_pred_tree].float()
                            cs_h_coarse_sum[t_orig] += t_hid
                            cs_h_coarse_cnt[t_orig] += 1.0
                            is_l1 = is_pred_tree & (next_level == (self.backbone.atom_level_index + 1))
                            l1_orig = orig_ids[is_l1]
                            l1_hid = dec_y_last[is_l1].float()
                            cs_h_l1_sum[l1_orig] += l1_hid
                            cs_h_l1_cnt[l1_orig] += 1.0
                            cs_h_l1_last[l1_orig] = l1_hid

                    pred_type = next_type.clone()
                    pred_count = torch.zeros((b_active,), dtype=torch.int32, device=device)
                    pred_xyz = torch.full((b_active, 3), self.backbone.xyz_null_id, dtype=torch.int32, device=device)
                    score_temp = torch.zeros((b_active,), dtype=torch.float32, device=device)

                    if has_atom:
                        feat_c = dec_y_last[is_pred_atom]
                        feat_u = dec_u_last[is_pred_atom] if dec_u_last is not None else None
                        n_sub = feat_c.shape[0]
                        sub_xyz = torch.zeros((n_sub, 4), dtype=torch.int32, device=device)
                        sub_sc = torch.zeros((n_sub,), dtype=torch.float32, device=device)

                        dec_feat_cur = dec_feat_buf[:, :dec_len, :]
                        dec_pos_cur = dec_pos_buf[:, :dec_len, :]
                        if (electron_parity is not None and parities is not None) or geometric_guard:
                            types_all = dec_feat_cur[:, :, IDX_TYPE]
                            actual = (
                                (dec_feat_cur[:, :, IDX_LEVEL] == self.backbone.atom_level_index)
                                & (types_all < len(atom_list))
                                & (types_all >= 0)
                            )
                        else:
                            types_all = None
                            actual = None

                        if electron_parity is not None and parities is not None and actual is not None and types_all is not None:
                            safe = torch.where(actual, types_all, self.backbone.atom_null_id)
                            acc_par = (parities[safe].sum(-1) % 2)[is_pred_atom]
                            term_flags = ((q_head >= q_tail) | (step_idx >= max_decode_steps))[is_pred_atom]
                        else:
                            acc_par = None
                            term_flags = None

                        if geometric_guard and actual is not None and types_all is not None:
                            vox_origin = next_feat[is_pred_atom, :3].float() * float(self.args.grid_len)
                            sub_actual = actual[is_pred_atom]
                            any_col = sub_actual.any(dim=0)
                            if bool(any_col.any().item()):
                                p_valid = sub_actual[:, any_col]
                                p_pos = dec_pos_cur[is_pred_atom][:, any_col, :3].float()
                                p_types = types_all[is_pred_atom][:, any_col].long()
                                p_rcov, p_sat = self._prepare_placed_geometry(p_pos, p_types, p_valid)
                            else:
                                p_valid = torch.zeros((n_sub, 0), dtype=torch.bool, device=device)
                                p_pos = torch.zeros((n_sub, 0, 3), dtype=torch.float32, device=device)
                                p_rcov = torch.zeros((n_sub, 0), dtype=torch.float32, device=device)
                                p_sat = torch.zeros((n_sub, 0), dtype=torch.bool, device=device)
                        else:
                            vox_origin = p_pos = p_valid = p_rcov = p_sat = None

                        ahead = self.backbone.atom_heads[0]
                        for axis in range(4):
                            logits_c = ahead.pred_heads[axis](feat_c).float()
                            scale_axis = cfg_scale if axis == 0 else xyz_cfg_scale
                            if feat_u is not None and scale_axis != 1.0:
                                logits_u = ahead.pred_heads[axis](feat_u).float()
                                cur_logits = logits_u + scale_axis * (logits_c - logits_u)
                            else:
                                cur_logits = logits_c

                            if axis == 0:
                                if geometric_guard and vox_origin is not None and p_pos is not None and p_valid is not None and p_rcov is not None and p_sat is not None:
                                    cur_logits = cur_logits + self._element_guard_logit_bias(
                                        vox_origin, p_pos, p_valid, p_rcov, p_sat
                                    )
                                cur_logits = cur_logits.masked_fill(forbidden_atom_mask, -torch.inf)
                                if electron_parity is not None and parities is not None and acc_par is not None and term_flags is not None:
                                    cur_logits = mask_terminal_element_logits(
                                        cur_logits, acc_par, term_flags, electron_parity, parities
                                    )
                                temp_axis = atom_temp
                            else:
                                if axis == 3 and geometric_guard and vox_origin is not None and p_pos is not None and p_valid is not None and p_rcov is not None and p_sat is not None:
                                    cur_logits = cur_logits + self._zbin_guard_logit_bias(
                                        vox_origin,
                                        sub_xyz[:, 0],
                                        sub_xyz[:, 1:3],
                                        p_pos,
                                        p_valid,
                                        p_rcov,
                                        p_sat,
                                        cur_logits.shape[-1],
                                    )
                                temp_axis = xyz_temp

                            sampled_id, cur_sc = _gpu_multinomial_sample(cur_logits, temp_axis, rng_gen)
                            sub_xyz[:, axis] = sampled_id
                            sub_sc = sub_sc + cur_sc * score_ratio[axis]
                            if axis < 3:
                                step_add = ahead.emb_layers[axis](sampled_id)
                                feat_c = feat_c + step_add
                                if feat_u is not None:
                                    feat_u = feat_u + step_add

                        pred_type[is_pred_atom] = sub_xyz[:, 0]
                        pred_xyz[is_pred_atom] = sub_xyz[:, 1:]
                        score_temp[is_pred_atom] = score_temp[is_pred_atom] + sub_sc
                        pred_count[is_pred_atom] = 1

                    if has_tree:
                        tree_c = self.backbone.tree_heads[0](dec_y_last[is_pred_tree]).float()
                        tree_lvl = next_level[is_pred_tree]
                        if dec_u_last is not None and cfg_scale != 1.0:
                            tree_u = self.backbone.tree_heads[0](dec_u_last[is_pred_tree]).float()
                            if cfg_tree_decay != 0.0:
                                decay_mult = 1.0 - float(cfg_tree_decay) * ((3.0 - tree_lvl.float()).clamp(0.0, 2.0) / 2.0)
                                eff_cfg = (1.0 + (cfg_scale - 1.0) * decay_mult).unsqueeze(-1)
                                tree_logits = tree_u + eff_cfg * (tree_c - tree_u)
                            else:
                                tree_logits = tree_u + cfg_scale * (tree_c - tree_u)
                        else:
                            tree_logits = tree_c

                        # Native Uni-3DAR level == 1 leaf-slot budget guidance
                        if budget_vec is not None and _replay_coarse_actions is None:
                            is_l1_sub = (tree_lvl == (self.backbone.atom_level_index + 1)) & (budget_vec[is_pred_tree] > 0)
                            if bool(is_l1_sub.any().item()):
                                tree_rows_idx = torch.nonzero(is_pred_tree, as_tuple=False).squeeze(-1)
                                l1_rows_idx = tree_rows_idx[is_l1_sub]
                                rem_after = (rem_l1_cells[l1_rows_idx] - 1.0).clamp_min(0.0).unsqueeze(-1)
                                cur_sp = spawned_leaves[l1_rows_idx].unsqueeze(-1)
                                goal_n = budget_vec[l1_rows_idx].unsqueeze(-1)
                                pops_row = pops_1d.unsqueeze(0)
                                min_reach = cur_sp + pops_row + 1.0 * rem_after
                                max_reach = cur_sp + pops_row + 8.0 * rem_after
                                feas = (min_reach <= goal_n + float(leaf_budget_tol)) & (max_reach >= goal_n - float(leaf_budget_tol))
                                has_feas = feas.any(dim=-1, keepdim=True)
                                ideal_k = ((goal_n - cur_sp) / (rem_after + 1.0)).clamp(1.0, 8.0)
                                prior_bias = -0.35 * (pops_row - ideal_k).square()
                                infeas_pen = torch.where(
                                    feas | (~has_feas),
                                    torch.zeros_like(prior_bias),
                                    torch.full_like(prior_bias, -14.0),
                                )
                                tree_logits = tree_logits.clone()
                                tree_logits[is_l1_sub] = tree_logits[is_l1_sub] + prior_bias + infeas_pen

                        if _replay_coarse_actions is not None and replay_cursors is not None:
                            tree_orig = cond_vars["orig_idx"][is_pred_tree].tolist()  # type: ignore[index]
                            preset_ids: List[int] = []
                            all_preset = True
                            for oid in tree_orig:
                                cur_c = replay_cursors[oid]
                                if cur_c < len(_replay_coarse_actions[oid]):
                                    preset_ids.append(int(_replay_coarse_actions[oid][cur_c]))
                                    replay_cursors[oid] = cur_c + 1
                                else:
                                    all_preset = False
                                    preset_ids.append(-1)
                            if all_preset:
                                t_pred = torch.tensor(preset_ids, dtype=torch.long, device=device)
                                logp_t = F.log_softmax(tree_logits.float(), dim=-1)
                                t_sc = logp_t.gather(-1, t_pred.unsqueeze(-1)).squeeze(-1)
                            else:
                                t_pred, t_sc = _gpu_multinomial_sample(tree_logits, tree_temp, rng_gen)
                                if any(pid >= 0 for pid in preset_ids):
                                    logp_t = F.log_softmax(tree_logits.float(), dim=-1)
                                    p_ten = torch.tensor([max(pid, 0) for pid in preset_ids], dtype=torch.long, device=device)
                                    has_p = torch.tensor([pid >= 0 for pid in preset_ids], dtype=torch.bool, device=device)
                                    t_pred = torch.where(has_p, p_ten, t_pred)
                                    t_sc = torch.where(has_p, logp_t.gather(-1, t_pred.unsqueeze(-1)).squeeze(-1), t_sc)
                        else:
                            t_pred, t_sc = _gpu_multinomial_sample(tree_logits, tree_temp, rng_gen)
                        if coarse_actions_rec is not None:
                            for oid, tp_val in zip(cond_vars["orig_idx"][is_pred_tree].tolist(), t_pred.tolist()):  # type: ignore[index]
                                coarse_actions_rec[oid].append(int(tp_val))

                        if budget_vec is not None:
                            chosen_pops = pops_1d[t_pred]
                            tree_rows_all = torch.nonzero(is_pred_tree, as_tuple=False).squeeze(-1)
                            is_l2_sub = tree_lvl == (self.backbone.atom_level_index + 2)
                            is_l1_track = tree_lvl == (self.backbone.atom_level_index + 1)
                            rem_l1_cells[tree_rows_all[is_l2_sub]] += chosen_pops[is_l2_sub]
                            rem_l1_cells[tree_rows_all[is_l1_track]] = (
                                rem_l1_cells[tree_rows_all[is_l1_track]] - 1.0
                            ).clamp_min(0.0)
                            spawned_leaves[tree_rows_all[is_l1_track]] += chosen_pops[is_l1_track]

                        pred_type[is_pred_tree] = t_pred.int() + self.backbone.tree0_split_id + 1
                        score_temp[is_pred_tree] = score_temp[is_pred_tree] + t_sc

                    scores_buf[:, dec_len] = score_temp
                    count_vars["total_atoms"] = torch.where(is_first_cnt, pred_count, count_vars["total_atoms"])

                    if has_tree:
                        q_buf, q_tail, q_cap, count_vars = self._fast_tree_expand_gpu(
                            q_buf, q_tail, q_cap, last_type, pred_type, next_feat, count_vars, step_idx
                        )
                    self._write_feat_by_pred_inplace(
                        next_feat, dec_feat_buf, dec_pos_buf, dec_len, pred_type, pred_count, pred_xyz
                    )
                    dec_len += 1

                    count_vars["last_remaining_atoms"] = last_rem_atoms
                    count_vars["last_remaining_tokens"] = last_rem_tokens
                    count_vars["pred_count"] = pred_count

                    row_arange = torch.arange(b_active, device=device)
                    q_empty = q_head >= q_tail
                    peek_feat = q_buf[row_arange, q_head.clamp_max(q_cap)]
                    force_stop = step_idx >= max_decode_steps
                    if force_stop:
                        done_mask = torch.ones((b_active,), dtype=torch.bool, device=device)
                    elif _stop_at_leaf_boundary:
                        done_mask = q_empty | (peek_feat[:, IDX_LEVEL] <= self.backbone.atom_level_index)
                    else:
                        done_mask = q_empty

                    if bool(done_mask.any().item()):
                        done_cpu = done_mask.tolist()
                        empty_cpu = q_empty.tolist()
                        orig_cpu = cond_vars["orig_idx"].tolist()  # type: ignore[union-attr]
                        for i, is_done in enumerate(done_cpu):
                            if not is_done:
                                continue
                            oid_i = orig_cpu[i]
                            rec_d: Dict[str, Any] = {
                                "decoder_feat": dec_feat_buf[i, :dec_len].clone(),
                                "decoder_phy_pos": dec_pos_buf[i, :dec_len].clone(),
                                "scores": scores_buf[i, :dec_len].clone(),
                                "batch_idx": oid_i,
                            }
                            if _stop_at_leaf_boundary:
                                rec_d["empty_coarse"] = bool(empty_cpu[i])
                                if coarse_actions_rec is not None:
                                    rec_d["coarse_actions"] = coarse_actions_rec[oid_i]
                                if _capture_continuation and (not bool(empty_cpu[i])) and (not force_stop):
                                    h_i = int(q_head[i].item())
                                    t_i = int(q_tail[i].item())
                                    kv_snap: List[List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]] = []
                                    kv_len_i = int(kv_cache[0][0]["k"].shape[2])
                                    for r_i in range(self.args.recycle):
                                        r_snap = []
                                        for l_i in range(self.args.layer):
                                            kc = kv_cache[r_i][l_i]["k"][i : i + 1].clone()
                                            vc = kv_cache[r_i][l_i]["v"][i : i + 1].clone()
                                            if use_cfg:
                                                ku = kv_cache[r_i][l_i]["k"][b_active + i : b_active + i + 1].clone()
                                                vu = kv_cache[r_i][l_i]["v"][b_active + i : b_active + i + 1].clone()
                                            else:
                                                ku = vu = None
                                            r_snap.append((kc, vc, ku, vu))
                                        kv_snap.append(r_snap)
                                    rec_d["_cont_state"] = {
                                        "dec_feat": rec_d["decoder_feat"],
                                        "dec_pos": rec_d["decoder_phy_pos"],
                                        "scores": rec_d["scores"],
                                        "q_rem": q_buf[i, h_i:t_i].clone(),
                                        "count_vars_i": {k: v[i].clone() for k, v in count_vars.items()},
                                        "global_c": cond_vars["global_c"][i].clone(),  # type: ignore[index]
                                        "global_c_uncond": (
                                            cond_vars["global_c_uncond"][i].clone()  # type: ignore[index]
                                            if cond_vars["global_c_uncond"] is not None
                                            else None
                                        ),
                                        "kv_layers": kv_snap,
                                        "kv_len": kv_len_i,
                                        "step_idx": step_idx,
                                    }
                            finished_trajs.append(rec_d)

                        keep_mask = ~done_mask
                        b_next = int(keep_mask.sum().item())
                        if b_next == 0:
                            break

                        dec_feat_buf = dec_feat_buf[keep_mask]
                        dec_pos_buf = dec_pos_buf[keep_mask]
                        scores_buf = scores_buf[keep_mask]
                        q_buf = q_buf[keep_mask]
                        q_head = q_head[keep_mask]
                        q_tail = q_tail[keep_mask]
                        peek_feat = peek_feat[keep_mask]
                        spawned_leaves = spawned_leaves[keep_mask]
                        rem_l1_cells = rem_l1_cells[keep_mask]
                        if budget_vec is not None:
                            budget_vec = budget_vec[keep_mask]
                        for k in cond_vars:
                            if cond_vars[k] is not None:
                                cond_vars[k] = cond_vars[k][keep_mask]
                        for k in count_vars:
                            count_vars[k] = count_vars[k][keep_mask]
                        kv_keep = torch.cat([keep_mask, keep_mask], dim=0) if use_cfg else keep_mask
                        layer_mods_cache = [(a[kv_keep], b[kv_keep]) for a, b in layer_mods_cache]
                        out_mod_cache = (out_mod_cache[0][kv_keep], out_mod_cache[1][kv_keep])
                        for r_i in range(len(kv_cache)):
                            for l_i in range(len(kv_cache[r_i])):
                                compact_kv(kv_cache[r_i][l_i], kv_keep)
                        b_active = b_next

                    next_feat = peek_feat
                    q_head = q_head + 1

                    last_level = dec_feat_buf[:, dec_len - 1, IDX_LEVEL]
                    is_new_level = last_level != next_feat[:, IDX_LEVEL]

                    self._write_feat_by_pred_inplace(
                        next_feat, dec_feat_buf, dec_pos_buf, dec_len, next_feat[:, IDX_TYPE], None, None
                    )
                    scores_buf[:, dec_len] = 0.0
                    dec_len += 1

                    count_vars, pair_rem_a, pair_rem_t = self._fast_update_running_stats(
                        count_vars, is_new_level, last_level
                    )

                    cur_feat = dec_feat_buf[:, dec_len - 2 : dec_len, :]
                    cur_phy_pos = dec_pos_buf[:, dec_len - 2 : dec_len, :]
                    cur_input = DecoderInputFeat(
                        cur_feat[:, :, IDX_TYPE],
                        cur_feat[:, :, :3],
                        cur_feat[:, :, IDX_LEVEL],
                        cur_phy_pos,
                        cur_feat[:, :, IDX_SPACE],
                        cur_feat[:, :, IDX_TREE],
                        pair_rem_a,
                        pair_rem_t,
                        cur_feat[:, :, IDX_CNT],
                    )
                    cur_is_cls = cur_input.level == (self.backbone.atom_level_index - 1)
                    cur_input.remaining_atoms = torch.where(
                        cur_is_cls, torch.zeros_like(cur_input.remaining_atoms), cur_input.remaining_atoms
                    ).clamp_(min=0)
                    cur_input.remaining_tokens = torch.where(
                        cur_is_cls, torch.zeros_like(cur_input.remaining_tokens), cur_input.remaining_tokens
                    ).clamp_(min=0)
                    cur_input.count = torch.where(cur_is_cls, torch.zeros_like(cur_input.count), cur_input.count)
                    # Clear `next_feat`? NO: on the next iteration, `cur_input[:, -1]` IS `next_feat`, so `next_feat`
                    # is ALREADY popped and used right after Transformer forward! Keep `next_feat` set!

            finished_trajs.sort(key=lambda d: d["batch_idx"])
            if return_coarse_summary and cs_h_root is not None:
                from .octree_process_critic import compute_coarse_voxel_geometry

                c_mean = cs_h_coarse_sum / cs_h_coarse_cnt.clamp_min(1.0)
                l1_mean = torch.where(
                    cs_h_l1_cnt > 0, cs_h_l1_sum / cs_h_l1_cnt.clamp_min(1.0), c_mean
                )
                l1_last = torch.where(cs_h_l1_cnt > 0, cs_h_l1_last, c_mean)
                h_cat_gpu = torch.cat([cs_h_root, c_mean, l1_mean, l1_last], dim=-1)

                cuts: List[int] = []
                for tr_c in finished_trajs:
                    lvl_c = tr_c["decoder_feat"][:, IDX_LEVEL]
                    leaf_idxs = torch.nonzero(lvl_c == self.backbone.atom_level_index, as_tuple=False)
                    cuts.append(int(leaf_idxs[0].item()) + 1 if leaf_idxs.numel() > 0 else int(lvl_c.shape[0]))
                max_cut = max(cuts) if cuts else 1
                n_tr = len(finished_trajs)
                pad_df = torch.zeros((n_tr, max_cut, NUM_FEAT), dtype=torch.int32, device=device)
                pad_df[:, :, IDX_TYPE] = self.backbone.atom_null_id
                pad_df[:, :, IDX_LEVEL] = -99
                pad_dp = torch.zeros((n_tr, max_cut, 3), dtype=torch.float32, device=device)
                pad_sc = torch.zeros((n_tr, max_cut), dtype=torch.float32, device=device)
                for r_i, (tr_c, c_len) in enumerate(zip(finished_trajs, cuts)):
                    pad_df[r_i, max_cut - c_len :] = tr_c["decoder_feat"][:c_len]
                    pad_dp[r_i, max_cut - c_len :] = tr_c["decoder_phy_pos"][:c_len]
                    pad_sc[r_i, max_cut - c_len :] = tr_c["scores"][:c_len]

                geom_all_cpu = compute_coarse_voxel_geometry(
                    pad_df,
                    pad_dp,
                    pad_sc,
                    atom_mask_id=self.backbone.atom_mask_id,
                    atom_level_index=self.backbone.atom_level_index,
                    fine_coarse_level=self.backbone.atom_level_index + 1,
                ).detach().cpu()
                h_cat_all = h_cat_gpu.detach().cpu()
            else:
                h_cat_all = None
                geom_all_cpu = None

            for r_i, traj in enumerate(finished_trajs):
                df = traj["decoder_feat"]
                dp = traj["decoder_phy_pos"]
                sc = traj["scores"]
                lvl = df[:, IDX_LEVEL]
                tree_sc = sc[lvl > self.backbone.atom_level_index]
                atom_sc = sc[lvl == self.backbone.atom_level_index]
                traj["tree_score_ppl"] = float(torch.exp(-tree_sc.mean()).item()) if tree_sc.numel() > 0 else 10.0
                traj["atom_score_ppl"] = float(torch.exp(-atom_sc.mean()).item()) if atom_sc.numel() > 0 else 10.0
                if not _stop_at_leaf_boundary:
                    mask = (lvl == self.backbone.atom_level_index) & (df[:, IDX_TYPE] < len(atom_list))
                    atom_ids = df[mask, IDX_TYPE].tolist()
                    traj["atom_symbols"] = [atom_list[idx] for idx in atom_ids]
                    traj["positions_np"] = dp[mask, :3].float().detach().cpu().numpy()
                if h_cat_all is not None and geom_all_cpu is not None:
                    b_i = int(traj["batch_idx"])
                    traj["coarse_h_concat"] = h_cat_all[b_i]
                    traj["coarse_geom_feats"] = geom_all_cpu[r_i]
                if not _capture_continuation:
                    traj["decoder_feat"] = df.detach().cpu()
                    traj["decoder_phy_pos"] = dp.detach().cpu()
                    traj["scores"] = sc.detach().cpu()

            if inv_perm_list is not None:
                finished_trajs = [finished_trajs[idx] for idx in inv_perm_list]
                for new_b, tr in enumerate(finished_trajs):
                    tr["batch_idx"] = new_b

            return finished_trajs
        finally:
            self.train(was_training)

    @torch.no_grad()
    def _continue_leaf_from_states(
        self,
        states: List[Dict[str, Any]],
        *,
        use_cfg: bool,
        cfg_scale: float,
        xyz_cfg_scale: float,
        geometric_guard: bool,
        temperatures: Tuple[float, float, float],
        seed: Optional[int],
        use_bf16: bool,
        electron_parity: Optional[int],
        max_decode_steps: int,
    ) -> List[Dict[str, Any]]:
        """Resume leaf (`level == 0`) decoding directly from gathered coarse KV-caches (zero coarse replay)."""
        n_sub = len(states)
        if n_sub == 0:
            return []
        device = states[0]["dec_feat"].device
        _, atom_temp, xyz_temp = temperatures

        rng_gen: Optional[torch.Generator] = None
        if seed is not None:
            rng_gen = torch.Generator(device=device)
            rng_gen.manual_seed(int(seed))

        parities = atom_electron_parities(self.backbone, device) if electron_parity is not None else None
        forbidden_atom_mask = ~self.allowed_atom_mask(device)
        score_ratio = [1.0, 0.0, 0.0, 0.0] if self.args.rank_by == "atom" else [1.0, 1.0, 1.0, 1.0]

        autocast_enabled = bool(use_bf16 and device.type == "cuda")
        run_dtype = torch.bfloat16 if autocast_enabled else torch.float32

        max_dec_0 = max(int(st["dec_feat"].shape[0]) for st in states)
        min_step_0 = min(int(st["step_idx"]) for st in states)
        rem_max_steps = max(1, int(max_decode_steps) - min_step_0)
        dec_cap = max_dec_0 + 2 * rem_max_steps + 16

        dec_feat_buf = torch.zeros((n_sub, dec_cap, NUM_FEAT), dtype=torch.int32, device=device)
        dec_feat_buf[:, :max_dec_0, IDX_TYPE] = self.backbone.atom_null_id
        dec_feat_buf[:, :max_dec_0, IDX_LEVEL] = -99
        dec_pos_buf = torch.zeros((n_sub, dec_cap, 3), dtype=torch.float32, device=device)
        scores_buf = torch.zeros((n_sub, dec_cap), dtype=torch.float32, device=device)

        pad_offsets_list: List[int] = []
        max_q_0 = max(int(st["q_rem"].shape[0]) for st in states)
        q_cap = max(max_q_0 + 32, 256)
        q_buf = torch.zeros((n_sub, q_cap + 16, NUM_FEAT), dtype=torch.int32, device=device)
        q_head = torch.zeros((n_sub,), dtype=torch.long, device=device)
        q_tail_vals: List[int] = []

        cv_keys = list(states[0]["count_vars_i"].keys())
        count_vars: Dict[str, torch.Tensor] = {
            k: torch.stack([st["count_vars_i"][k] for st in states], dim=0) for k in cv_keys
        }
        global_c = torch.stack([st["global_c"] for st in states], dim=0)
        global_c_uncond = (
            torch.stack([st["global_c_uncond"] for st in states], dim=0)
            if (use_cfg and states[0]["global_c_uncond"] is not None)
            else None
        )

        max_kv_0 = max(int(st["kv_len"]) for st in states)
        all_same_kv = all(int(st["kv_len"]) == max_kv_0 for st in states)
        kv_total_cap = max_kv_0 + 2 * rem_max_steps + 16
        b_cfg = (2 * n_sub) if use_cfg else n_sub

        for i, st in enumerate(states):
            df_i = st["dec_feat"]
            l_d = int(df_i.shape[0])
            p_off = max_dec_0 - l_d
            pad_offsets_list.append(p_off)
            dec_feat_buf[i, p_off:max_dec_0, :] = df_i
            dec_pos_buf[i, p_off:max_dec_0, :] = st["dec_pos"]
            scores_buf[i, p_off:max_dec_0] = st["scores"]
            qr_i = st["q_rem"]
            q_len_i = int(qr_i.shape[0])
            q_buf[i, :q_len_i, :] = qr_i
            q_tail_vals.append(q_len_i)

        q_tail = torch.tensor(q_tail_vals, dtype=torch.long, device=device)
        pad_offsets = torch.tensor(pad_offsets_list, dtype=torch.long, device=device)
        orig_idx = torch.arange(n_sub, dtype=torch.long, device=device)
        step_vec = torch.tensor([int(st["step_idx"]) for st in states], dtype=torch.long, device=device)

        sample_k0 = states[0]["kv_layers"][0][0][0]
        n_heads = int(sample_k0.shape[1])
        head_dim = int(sample_k0.shape[3])
        kv_dtype = sample_k0.dtype

        kv_cache: List[List[Dict[str, Any]]] = []
        for r_i in range(self.args.recycle):
            r_list: List[Dict[str, Any]] = []
            for l_i in range(self.args.layer):
                k_b = torch.zeros((b_cfg, n_heads, kv_total_cap, head_dim), dtype=kv_dtype, device=device)
                v_b = torch.zeros((b_cfg, n_heads, kv_total_cap, head_dim), dtype=kv_dtype, device=device)
                for i, st in enumerate(states):
                    kc_i, vc_i, ku_i, vu_i = st["kv_layers"][r_i][l_i]
                    lk = int(st["kv_len"])
                    ko = max_kv_0 - lk
                    k_b[i, :, ko:max_kv_0, :] = kc_i[0]
                    v_b[i, :, ko:max_kv_0, :] = vc_i[0]
                    if use_cfg and ku_i is not None and vu_i is not None:
                        k_b[n_sub + i, :, ko:max_kv_0, :] = ku_i[0]
                        v_b[n_sub + i, :, ko:max_kv_0, :] = vu_i[0]
                r_list.append(
                    {
                        "step": 2,
                        "mode": "buffer",
                        "init_capacity": kv_total_cap,
                        "k_buffer": k_b,
                        "v_buffer": v_b,
                        "k": k_b[:, :, :max_kv_0, :],
                        "v": v_b[:, :, :max_kv_0, :],
                    }
                )
            kv_cache.append(r_list)
        # Immediately free coarse snapshot KV tensors now that they are copied into `kv_cache`
        for st in states:
            st.pop("kv_layers", None)

        kv_valid_buf: Optional[torch.Tensor] = None
        if not all_same_kv:
            kv_valid_buf = torch.ones((b_cfg, kv_total_cap), dtype=torch.bool, device=device)
            for i, st in enumerate(states):
                ko = max_kv_0 - int(st["kv_len"])
                if ko > 0:
                    kv_valid_buf[i, :ko] = False
                    if use_cfg:
                        kv_valid_buf[n_sub + i, :ko] = False

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            c_run_init = torch.cat([global_c, global_c_uncond], dim=0) if (use_cfg and global_c_uncond is not None) else global_c
            layer_mods_cache, out_mod_cache = self.conditioner.precompute_modulations(c_run_init, run_dtype)

            # Pop the first `level == 0` mask token and form the 2-token `cur_input`
            b_active = n_sub
            dec_len = max_dec_0
            row_arange = torch.arange(b_active, device=device)
            next_feat = q_buf[row_arange, q_head]
            q_head = q_head + 1

            last_level = dec_feat_buf[:, dec_len - 1, IDX_LEVEL]
            is_new_level = last_level != next_feat[:, IDX_LEVEL]
            self._write_feat_by_pred_inplace(
                next_feat, dec_feat_buf, dec_pos_buf, dec_len, next_feat[:, IDX_TYPE], None, None
            )
            scores_buf[:, dec_len] = 0.0
            dec_len += 1

            count_vars, pair_rem_a, pair_rem_t = self._fast_update_running_stats(count_vars, is_new_level, last_level)
            cur_feat = dec_feat_buf[:, dec_len - 2 : dec_len, :]
            cur_phy_pos = dec_pos_buf[:, dec_len - 2 : dec_len, :]
            cur_input = DecoderInputFeat(
                cur_feat[:, :, IDX_TYPE],
                cur_feat[:, :, :3],
                cur_feat[:, :, IDX_LEVEL],
                cur_phy_pos,
                cur_feat[:, :, IDX_SPACE],
                cur_feat[:, :, IDX_TREE],
                pair_rem_a,
                pair_rem_t,
                cur_feat[:, :, IDX_CNT],
            )
            cur_is_cls = cur_input.level == (self.backbone.atom_level_index - 1)
            cur_input.remaining_atoms = torch.where(
                cur_is_cls, torch.zeros_like(cur_input.remaining_atoms), cur_input.remaining_atoms
            ).clamp_(min=0)
            cur_input.remaining_tokens = torch.where(
                cur_is_cls, torch.zeros_like(cur_input.remaining_tokens), cur_input.remaining_tokens
            ).clamp_(min=0)
            cur_input.count = torch.where(cur_is_cls, torch.zeros_like(cur_input.count), cur_input.count)

            finished_trajs: List[Dict[str, Any]] = []
            cur_kv_len = max_kv_0

            while b_active > 0:
                step_emb = self.backbone.input_embedding(cur_input)
                step_grid = cur_input.phy_pos.float() / self.args.xyz_resolution
                if use_cfg:
                    x_run = step_emb.repeat(2, 1, 1)
                    grid_run = step_grid.repeat(2, 1, 1)
                else:
                    x_run = step_emb
                    grid_run = step_grid

                cur_kv_len += int(x_run.shape[1])
                if kv_valid_buf is not None:
                    b_cfg_cur = int(x_run.shape[0])
                    kv_mask_cur = kv_valid_buf[:b_cfg_cur, :cur_kv_len]
                    m0 = kv_mask_cur.clone()
                    m0[:, -1] = False
                    prebuilt_mask: Optional[torch.Tensor] = torch.stack([m0, kv_mask_cur], dim=1).unsqueeze(1)
                else:
                    prebuilt_mask = None

                self.backbone.rope.calc_and_cache(grid_run)
                for r_i in range(self.args.recycle):
                    for l_i, layer in enumerate(self.backbone.decoder.layers):
                        x_run = self._run_layer_kv_step(
                            layer,
                            x_run,
                            kv_cache[r_i][l_i],
                            layer_idx=l_i,
                            prebuilt_attn_mask=prebuilt_mask,
                        )
                        ops_l, shf_l = layer_mods_cache[l_i]
                        x_run = torch.addcmul(shf_l, x_run, ops_l)
                x_run = self.backbone.norm(x_run)
                ops_o, shf_o = out_mod_cache
                x_run = torch.addcmul(shf_o, x_run, ops_o)

                last_hidden = x_run[:, -1, :]
                if use_cfg:
                    dec_y_last, dec_u_last = last_hidden.chunk(2, dim=0)
                else:
                    dec_y_last = last_hidden
                    dec_u_last = None

                step_vec = step_vec + 1
                last_level = cur_input.level[:, -1]
                last_type = cur_input.type[:, -1]
                last_rem_atoms = cur_input.remaining_atoms[:, -1]
                last_rem_tokens = cur_input.remaining_tokens[:, -1]

                row_arange = torch.arange(b_active, device=device)
                next_feat = q_buf[row_arange, q_head]
                q_head = q_head + 1

                next_type = next_feat[:, IDX_TYPE]
                next_level = next_feat[:, IDX_LEVEL]
                need_pred = next_type < 0
                is_pred_atom = (
                    (last_level == self.backbone.atom_level_index)
                    & (last_type == self.backbone.atom_mask_id)
                    & need_pred
                )

                pred_type = next_type.clone()
                pred_count = torch.zeros((b_active,), dtype=torch.int32, device=device)
                pred_xyz = torch.full((b_active, 3), self.backbone.xyz_null_id, dtype=torch.int32, device=device)
                score_temp = torch.zeros((b_active,), dtype=torch.float32, device=device)

                if bool(is_pred_atom.any().item()):
                    feat_c = dec_y_last[is_pred_atom]
                    feat_u = dec_u_last[is_pred_atom] if dec_u_last is not None else None
                    n_at = feat_c.shape[0]
                    sub_xyz = torch.zeros((n_at, 4), dtype=torch.int32, device=device)
                    sub_sc = torch.zeros((n_at,), dtype=torch.float32, device=device)

                    dec_feat_cur = dec_feat_buf[:, :dec_len, :]
                    dec_pos_cur = dec_pos_buf[:, :dec_len, :]
                    if (electron_parity is not None and parities is not None) or geometric_guard:
                        types_all = dec_feat_cur[:, :, IDX_TYPE]
                        actual = (
                            (dec_feat_cur[:, :, IDX_LEVEL] == self.backbone.atom_level_index)
                            & (types_all < len(atom_list))
                            & (types_all >= 0)
                        )
                    else:
                        types_all = None
                        actual = None

                    if electron_parity is not None and parities is not None and actual is not None and types_all is not None:
                        safe = torch.where(actual, types_all, self.backbone.atom_null_id)
                        acc_par = (parities[safe].sum(-1) % 2)[is_pred_atom]
                        term_flags = ((q_head >= q_tail) | (step_vec >= max_decode_steps))[is_pred_atom]
                    else:
                        acc_par = None
                        term_flags = None

                    if geometric_guard and actual is not None and types_all is not None:
                        vox_origin = next_feat[is_pred_atom, :3].float() * float(self.args.grid_len)
                        sub_actual = actual[is_pred_atom]
                        any_col = sub_actual.any(dim=0)
                        if bool(any_col.any().item()):
                            p_valid = sub_actual[:, any_col]
                            p_pos = dec_pos_cur[is_pred_atom][:, any_col, :3].float()
                            p_types = types_all[is_pred_atom][:, any_col].long()
                            p_rcov, p_sat = self._prepare_placed_geometry(p_pos, p_types, p_valid)
                        else:
                            p_valid = torch.zeros((n_at, 0), dtype=torch.bool, device=device)
                            p_pos = torch.zeros((n_at, 0, 3), dtype=torch.float32, device=device)
                            p_rcov = torch.zeros((n_at, 0), dtype=torch.float32, device=device)
                            p_sat = torch.zeros((n_at, 0), dtype=torch.bool, device=device)
                    else:
                        vox_origin = p_pos = p_valid = p_rcov = p_sat = None

                    ahead = self.backbone.atom_heads[0]
                    for axis in range(4):
                        logits_c = ahead.pred_heads[axis](feat_c).float()
                        scale_axis = cfg_scale if axis == 0 else xyz_cfg_scale
                        if feat_u is not None and scale_axis != 1.0:
                            logits_u = ahead.pred_heads[axis](feat_u).float()
                            cur_logits = logits_u + scale_axis * (logits_c - logits_u)
                        else:
                            cur_logits = logits_c

                        if axis == 0:
                            if geometric_guard and vox_origin is not None and p_pos is not None and p_valid is not None and p_rcov is not None and p_sat is not None:
                                cur_logits = cur_logits + self._element_guard_logit_bias(
                                    vox_origin, p_pos, p_valid, p_rcov, p_sat
                                )
                            cur_logits = cur_logits.masked_fill(forbidden_atom_mask, -torch.inf)
                            if electron_parity is not None and parities is not None and acc_par is not None and term_flags is not None:
                                cur_logits = mask_terminal_element_logits(
                                    cur_logits, acc_par, term_flags, electron_parity, parities
                                )
                            temp_axis = atom_temp
                        else:
                            if axis == 3 and geometric_guard and vox_origin is not None and p_pos is not None and p_valid is not None and p_rcov is not None and p_sat is not None:
                                cur_logits = cur_logits + self._zbin_guard_logit_bias(
                                    vox_origin,
                                    sub_xyz[:, 0],
                                    sub_xyz[:, 1:3],
                                    p_pos,
                                    p_valid,
                                    p_rcov,
                                    p_sat,
                                    cur_logits.shape[-1],
                                )
                            temp_axis = xyz_temp

                        sampled_id, cur_sc = _gpu_multinomial_sample(cur_logits, temp_axis, rng_gen)
                        sub_xyz[:, axis] = sampled_id
                        sub_sc = sub_sc + cur_sc * score_ratio[axis]
                        if axis < 3:
                            step_add = ahead.emb_layers[axis](sampled_id)
                            feat_c = feat_c + step_add
                            if feat_u is not None:
                                feat_u = feat_u + step_add

                    pred_type[is_pred_atom] = sub_xyz[:, 0]
                    pred_xyz[is_pred_atom] = sub_xyz[:, 1:]
                    score_temp[is_pred_atom] = score_temp[is_pred_atom] + sub_sc
                    pred_count[is_pred_atom] = 1

                scores_buf[:, dec_len] = score_temp
                self._write_feat_by_pred_inplace(
                    next_feat, dec_feat_buf, dec_pos_buf, dec_len, pred_type, pred_count, pred_xyz
                )
                dec_len += 1

                count_vars["last_remaining_atoms"] = last_rem_atoms
                count_vars["last_remaining_tokens"] = last_rem_tokens
                count_vars["pred_count"] = pred_count

                row_arange = torch.arange(b_active, device=device)
                q_empty = q_head >= q_tail
                peek_feat = q_buf[row_arange, q_head.clamp_max(q_cap)]
                done_mask = q_empty | (step_vec >= max_decode_steps)

                if bool(done_mask.any().item()):
                    done_cpu = done_mask.tolist()
                    orig_cpu = orig_idx.tolist()
                    poff_cpu = pad_offsets.tolist()
                    for i, is_done in enumerate(done_cpu):
                        if not is_done:
                            continue
                        oid_i = orig_cpu[i]
                        p_off_i = poff_cpu[i]
                        finished_trajs.append(
                            {
                                "decoder_feat": dec_feat_buf[i, p_off_i:dec_len].clone(),
                                "decoder_phy_pos": dec_pos_buf[i, p_off_i:dec_len].clone(),
                                "scores": scores_buf[i, p_off_i:dec_len].clone(),
                                "batch_idx": oid_i,
                            }
                        )

                    keep_mask = ~done_mask
                    b_next = int(keep_mask.sum().item())
                    if b_next == 0:
                        break

                    dec_feat_buf = dec_feat_buf[keep_mask]
                    dec_pos_buf = dec_pos_buf[keep_mask]
                    scores_buf = scores_buf[keep_mask]
                    q_buf = q_buf[keep_mask]
                    q_head = q_head[keep_mask]
                    q_tail = q_tail[keep_mask]
                    peek_feat = peek_feat[keep_mask]
                    orig_idx = orig_idx[keep_mask]
                    pad_offsets = pad_offsets[keep_mask]
                    step_vec = step_vec[keep_mask]
                    for k in count_vars:
                        count_vars[k] = count_vars[k][keep_mask]
                    kv_keep = torch.cat([keep_mask, keep_mask], dim=0) if use_cfg else keep_mask
                    layer_mods_cache = [(a[kv_keep], b[kv_keep]) for a, b in layer_mods_cache]
                    out_mod_cache = (out_mod_cache[0][kv_keep], out_mod_cache[1][kv_keep])
                    if kv_valid_buf is not None:
                        n_k = int(kv_keep.sum().item())
                        kv_valid_buf[:n_k, :cur_kv_len].copy_(kv_valid_buf[: kv_keep.shape[0], :cur_kv_len][kv_keep])
                    for r_i in range(len(kv_cache)):
                        for l_i in range(len(kv_cache[r_i])):
                            compact_kv(kv_cache[r_i][l_i], kv_keep)
                    b_active = b_next

                next_feat = peek_feat
                q_head = q_head + 1
                last_level = dec_feat_buf[:, dec_len - 1, IDX_LEVEL]
                is_new_level = last_level != next_feat[:, IDX_LEVEL]
                self._write_feat_by_pred_inplace(
                    next_feat, dec_feat_buf, dec_pos_buf, dec_len, next_feat[:, IDX_TYPE], None, None
                )
                scores_buf[:, dec_len] = 0.0
                dec_len += 1
                count_vars, pair_rem_a, pair_rem_t = self._fast_update_running_stats(
                    count_vars, is_new_level, last_level
                )
                cur_feat = dec_feat_buf[:, dec_len - 2 : dec_len, :]
                cur_phy_pos = dec_pos_buf[:, dec_len - 2 : dec_len, :]
                cur_input = DecoderInputFeat(
                    cur_feat[:, :, IDX_TYPE],
                    cur_feat[:, :, :3],
                    cur_feat[:, :, IDX_LEVEL],
                    cur_phy_pos,
                    cur_feat[:, :, IDX_SPACE],
                    cur_feat[:, :, IDX_TREE],
                    pair_rem_a,
                    pair_rem_t,
                    cur_feat[:, :, IDX_CNT],
                )
                cur_is_cls = cur_input.level == (self.backbone.atom_level_index - 1)
                cur_input.remaining_atoms = torch.where(
                    cur_is_cls, torch.zeros_like(cur_input.remaining_atoms), cur_input.remaining_atoms
                ).clamp_(min=0)
                cur_input.remaining_tokens = torch.where(
                    cur_is_cls, torch.zeros_like(cur_input.remaining_tokens), cur_input.remaining_tokens
                ).clamp_(min=0)
                cur_input.count = torch.where(cur_is_cls, torch.zeros_like(cur_input.count), cur_input.count)

        finished_trajs.sort(key=lambda d: d["batch_idx"])
        for traj in finished_trajs:
            df = traj["decoder_feat"]
            dp = traj["decoder_phy_pos"]
            sc = traj["scores"]
            lvl = df[:, IDX_LEVEL]
            tree_sc = sc[lvl > self.backbone.atom_level_index]
            atom_sc = sc[lvl == self.backbone.atom_level_index]
            traj["tree_score_ppl"] = float(torch.exp(-tree_sc.mean()).item()) if tree_sc.numel() > 0 else 10.0
            traj["atom_score_ppl"] = float(torch.exp(-atom_sc.mean()).item()) if atom_sc.numel() > 0 else 10.0
            mask = (lvl == self.backbone.atom_level_index) & (df[:, IDX_TYPE] < len(atom_list))
            atom_ids = df[mask, IDX_TYPE].tolist()
            traj["atom_symbols"] = [atom_list[idx] for idx in atom_ids]
            traj["positions_np"] = dp[mask, :3].float().detach().cpu().numpy()
            traj["decoder_feat"] = df.detach().cpu()
            traj["decoder_phy_pos"] = dp.detach().cpu()
            traj["scores"] = sc.detach().cpu()
        return finished_trajs

    @torch.no_grad()
    def sample_with_process_critic(
        self,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        process_critic: Any,
        *,
        m_init: int = 16,
        m_keep: int = 2,
        objective_col_mask: Optional[torch.Tensor] = None,
        legal_weight: float = 1.0,
        constr_weight: float = 0.0,
        uncertainty_bonus: float = 0.0,
        coarse_chunk_size: int = 1024,
        fine_batch_size: int = 512,
        cfg_scale: float = 1.0,
        xyz_cfg_scale: float = 1.0,
        cfg_tree_decay: float = 0.0,
        geometric_guard: bool = False,
        temperatures: Tuple[float, float, float] = (0.85, 0.30, 0.25),
        seed: Optional[int] = None,
        use_bf16: bool = True,
        electron_parity: Optional[int] = 0,
        max_decode_steps: int = 256,
        uncond_active_mask: Optional[torch.Tensor] = None,
        leaf_atom_budget: Optional[torch.Tensor] = None,
        leaf_budget_tol: float = 2.0,
        sort_by_budget: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Zero-replay coarse-to-fine octree generation with direct KV-cache continuation (`m_init -> m_keep`)."""
        if m_init < 1 or m_keep < 1 or m_keep > m_init:
            raise ValueError(f"Invalid branch funnel m_init={m_init}, m_keep={m_keep}")

        n_prompts = int(targets_norm.shape[0])
        if n_prompts == 0:
            return []
        if m_init == 1 and m_keep == 1:
            flat = self.sample(
                targets_norm,
                active_mask,
                cfg_scale=cfg_scale,
                xyz_cfg_scale=xyz_cfg_scale,
                cfg_tree_decay=cfg_tree_decay,
                geometric_guard=geometric_guard,
                temperatures=temperatures,
                seed=seed,
                use_bf16=use_bf16,
                electron_parity=electron_parity,
                max_decode_steps=max_decode_steps,
                uncond_active_mask=uncond_active_mask,
                leaf_atom_budget=leaf_atom_budget,
                leaf_budget_tol=leaf_budget_tol,
                sort_by_budget=sort_by_budget,
            )
            for tr in flat:
                tr["process_utility"] = 0.0
            return [[tr] for tr in flat]

        device = targets_norm.device
        inv_perm_list: Optional[List[int]] = None
        if leaf_atom_budget is not None and sort_by_budget and n_prompts > 8:
            b_p = torch.as_tensor(leaf_atom_budget, dtype=torch.float32, device=device).view(n_prompts)
            perm = torch.argsort(b_p, stable=True)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(n_prompts, device=device)
            inv_perm_list = inv_perm.detach().cpu().tolist()
            targets_norm = targets_norm[perm]
            active_mask = active_mask[perm]
            if uncond_active_mask is not None:
                uncond_active_mask = uncond_active_mask[perm]
            if objective_col_mask is not None and objective_col_mask.ndim == 2:
                objective_col_mask = objective_col_mask[perm]
            leaf_atom_budget = b_p[perm]

        has_cond = bool(active_mask.bool().any().item())
        use_cfg = has_cond and (cfg_scale != 1.0 or xyz_cfg_scale != 1.0)

        # Bound per-chunk coarse & fine batch sizes for multi-recycle (`recycle >= 3` -> 48 KV caches)
        eff_coarse_chunk = min(int(coarse_chunk_size), 512) if int(self.args.recycle) >= 3 else int(coarse_chunk_size)
        eff_fine_bs = max(int(m_keep), min(int(fine_batch_size), 256) if int(self.args.recycle) >= 3 else int(fine_batch_size))
        prompts_per_chunk = max(1, eff_coarse_chunk // int(m_init))

        total_winners = n_prompts * int(m_keep)
        leaf_results: List[Optional[Dict[str, Any]]] = [None] * total_winners
        pending_slots: List[int] = []
        pending_states: List[Dict[str, Any]] = []
        pending_utils: List[float] = []
        fine_launched = 0
        next_winner_slot = 0

        def _flush_pending_fine(force_all: bool = False) -> None:
            nonlocal fine_launched
            while len(pending_states) >= eff_fine_bs or (force_all and len(pending_states) > 0):
                take = min(eff_fine_bs, len(pending_states))
                batch_states = pending_states[:take]
                batch_slots = pending_slots[:take]
                batch_utils = pending_utils[:take]
                del pending_states[:take]
                del pending_slots[:take]
                del pending_utils[:take]

                f_seed = (int(seed) * 97 + 100000 + fine_launched) if seed is not None else None
                fine_launched += take
                sub_done = self._continue_leaf_from_states(
                    batch_states,
                    use_cfg=use_cfg,
                    cfg_scale=cfg_scale,
                    xyz_cfg_scale=xyz_cfg_scale,
                    geometric_guard=geometric_guard,
                    temperatures=temperatures,
                    seed=f_seed,
                    use_bf16=use_bf16,
                    electron_parity=electron_parity,
                    max_decode_steps=max_decode_steps,
                )
                for k_local, tr_leaf in enumerate(sub_done):
                    tr_leaf["process_utility"] = batch_utils[k_local]
                    leaf_results[batch_slots[k_local]] = tr_leaf
                for st_done in batch_states:
                    st_done.clear()

        for p_start in range(0, n_prompts, prompts_per_chunk):
            p_end = min(p_start + prompts_per_chunk, n_prompts)
            n_p_sub = p_end - p_start
            t_sub = targets_norm[p_start:p_end].repeat_interleave(m_init, dim=0)
            m_sub = active_mask[p_start:p_end].repeat_interleave(m_init, dim=0)
            u_sub = (
                uncond_active_mask[p_start:p_end].repeat_interleave(m_init, dim=0)
                if uncond_active_mask is not None
                else None
            )
            b_sub = (
                torch.as_tensor(leaf_atom_budget[p_start:p_end], dtype=torch.float32, device=device).repeat_interleave(m_init, dim=0)
                if leaf_atom_budget is not None
                else None
            )
            obj_sub = (
                objective_col_mask[p_start:p_end].repeat_interleave(m_init, dim=0)
                if (objective_col_mask is not None and objective_col_mask.ndim == 2)
                else objective_col_mask
            )

            c_seed = (int(seed) * 31 + p_start * m_init) if seed is not None else None
            c_trajs = self.sample(
                t_sub,
                m_sub,
                cfg_scale=cfg_scale,
                xyz_cfg_scale=xyz_cfg_scale,
                cfg_tree_decay=cfg_tree_decay,
                geometric_guard=False,
                temperatures=temperatures,
                seed=c_seed,
                use_bf16=use_bf16,
                electron_parity=electron_parity,
                max_decode_steps=max_decode_steps,
                uncond_active_mask=u_sub,
                return_coarse_summary=True,
                leaf_atom_budget=b_sub,
                leaf_budget_tol=leaf_budget_tol,
                sort_by_budget=False,
                _stop_at_leaf_boundary=True,
                _capture_continuation=True,
            )

            h_cat = torch.stack([tr["coarse_h_concat"] for tr in c_trajs], dim=0).to(device)
            geom_feats = torch.stack([tr["coarse_geom_feats"] for tr in c_trajs], dim=0).to(device)
            empty_mask = torch.tensor([bool(tr.get("empty_coarse", False)) for tr in c_trajs], dtype=torch.bool, device=device)

            utilities = process_critic.score_utilities(
                h_cat,
                geom_feats,
                t_sub,
                m_sub,
                objective_col_mask=obj_sub,
                legal_weight=legal_weight,
                constr_weight=constr_weight,
                uncertainty_bonus=uncertainty_bonus,
                empty_mask=empty_mask,
            )
            util_grid = utilities.view(n_p_sub, m_init)
            topk_vals, topk_local = torch.topk(util_grid, k=m_keep, dim=1, largest=True, sorted=True)
            offsets = (torch.arange(n_p_sub, device=device) * m_init).unsqueeze(1)
            win_idx_list = (offsets + topk_local).reshape(-1).detach().cpu().tolist()
            win_util_list = topk_vals.reshape(-1).detach().cpu().tolist()

            for loc_w, u_val in zip(win_idx_list, win_util_list):
                tr_w = c_trajs[loc_w]
                cont = tr_w.pop("_cont_state", None)
                if cont is not None:
                    pending_slots.append(next_winner_slot)
                    pending_states.append(cont)
                    pending_utils.append(float(u_val))
                else:
                    df_e = tr_w["decoder_feat"]
                    dp_e = tr_w["decoder_phy_pos"]
                    lvl_e = df_e[:, IDX_LEVEL]
                    mask_e = (lvl_e == self.backbone.atom_level_index) & (df_e[:, IDX_TYPE] < len(atom_list))
                    a_ids = df_e[mask_e, IDX_TYPE].tolist()
                    tr_w["atom_symbols"] = [atom_list[idx] for idx in a_ids]
                    tr_w["positions_np"] = dp_e[mask_e, :3].float().detach().cpu().numpy()
                    tr_w["decoder_feat"] = df_e.detach().cpu()
                    tr_w["decoder_phy_pos"] = dp_e.detach().cpu()
                    tr_w["scores"] = tr_w["scores"].detach().cpu()
                    tr_w["process_utility"] = float(u_val)
                    leaf_results[next_winner_slot] = tr_w
                next_winner_slot += 1

            # Immediately drop unpicked `m_init - m_keep` coarse KV snapshots and flush ready fine batches
            del c_trajs
            _flush_pending_fine(force_all=False)

        _flush_pending_fine(force_all=True)

        grouped: List[List[Dict[str, Any]]] = [[] for _ in range(n_prompts)]
        for w_slot, tr_opt in enumerate(leaf_results):
            assert tr_opt is not None
            grouped[w_slot // m_keep].append(tr_opt)

        if inv_perm_list is not None:
            grouped = [grouped[idx] for idx in inv_perm_list]

        return grouped

    # ------------------------------------------------------------------
    # Trajectory Packing & Exact Policy Distributions for GRPO
    # ------------------------------------------------------------------
    def pack_sampled_trajectories(self, trajs: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
        cls_phy_pos = self.backbone.get_cls_phy_pos(self.args.merge_level)
        all_type, all_xyz, all_phy, all_lvl = [], [], [], []
        all_sec, all_space, all_tree, all_rem_a, all_rem_t, all_cnt, all_flag = [], [], [], [], [], [], []
        cu_lens = [0]

        for traj in trajs:
            df = traj["decoder_feat"].to(device).long()
            dp = traj["decoder_phy_pos"].to(device).float()
            length = df.shape[0]

            b_type = df[:, IDX_TYPE]
            b_xyz = df[:, :3]
            b_phy = dp
            b_lvl = df[:, IDX_LEVEL]
            b_sec = torch.zeros((length,), dtype=torch.bool, device=device)
            b_sec[1::2] = True
            b_space = df[:, IDX_SPACE].clamp(min=0)
            b_tree = df[:, IDX_TREE].clamp(min=0)
            b_rem_a = torch.zeros((length,), dtype=torch.long, device=device)
            b_rem_t = torch.zeros((length,), dtype=torch.long, device=device)
            b_cnt = df[:, IDX_CNT]
            b_flag = torch.ones((length,), dtype=torch.long, device=device)

            c_type = torch.tensor([self.backbone.atom_mask_id, self.backbone.atom_cls_id], dtype=torch.long, device=device)
            c_xyz = torch.full((2, 3), self.backbone.xyz_null_id, dtype=torch.long, device=device)
            c_phy = torch.full((2, 3), cls_phy_pos, dtype=torch.float32, device=device)
            c_lvl = torch.full((2,), self.backbone.atom_level_index - 1, dtype=torch.long, device=device)
            c_sec = torch.zeros((2,), dtype=torch.bool, device=device)
            c_zero = torch.zeros((2,), dtype=torch.long, device=device)

            all_type.extend([b_type, c_type])
            all_xyz.extend([b_xyz, c_xyz])
            all_phy.extend([b_phy, c_phy])
            all_lvl.extend([b_lvl, c_lvl])
            all_sec.extend([b_sec, c_sec])
            all_space.extend([b_space, c_zero])
            all_tree.extend([b_tree, c_zero])
            all_rem_a.extend([b_rem_a, c_zero])
            all_rem_t.extend([b_rem_t, c_zero])
            all_cnt.extend([b_cnt, c_zero])
            all_flag.extend([b_flag, c_zero])
            cu_lens.append(cu_lens[-1] + length + 2)

        cu_t = torch.tensor(cu_lens, dtype=torch.int32, device=device)
        max_len = int((cu_t[1:] - cu_t[:-1]).max().item())
        return {
            "decoder_type": torch.cat(all_type, dim=0),
            "decoder_xyz": torch.cat(all_xyz, dim=0),
            "decoder_phy_pos": torch.cat(all_phy, dim=0),
            "decoder_level": torch.cat(all_lvl, dim=0),
            "decoder_is_second_atom": torch.cat(all_sec, dim=0),
            "space_index": torch.cat(all_space, dim=0),
            "tree_index": torch.cat(all_tree, dim=0),
            "decoder_remaining_atoms": torch.cat(all_rem_a, dim=0),
            "decoder_remaining_tokens": torch.cat(all_rem_t, dim=0),
            "decoder_count": torch.cat(all_cnt, dim=0),
            "tree_loss_flag": torch.cat(all_flag, dim=0),
            "decoder_input_len": cu_t,
            "decoder_input_max_len": torch.tensor(max_len, dtype=torch.int32, device=device),
        }

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
        topology_only_policy: bool = False,
        electron_parity: Optional[int] = 0,
        uncond_active_mask: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        device = targets_norm.device
        packed = self.pack_sampled_trajectories(trajectories, device)
        decoder_input, target, _ = self.backbone.get_feat_from_dataloader(packed)
        cu = packed["decoder_input_len"]
        max_len = int(packed["decoder_input_max_len"].item())

        hidden = self.forward_sequence(decoder_input, cu, max_len, targets_norm, active_mask)
        use_cfg = bool(active_mask.bool().any().item()) and (cfg_scale != 1.0 or xyz_cfg_scale != 1.0)
        null_mask = (
            uncond_active_mask.to(device=device, dtype=torch.bool)
            if uncond_active_mask is not None
            else torch.zeros_like(active_mask, dtype=torch.bool)
        )
        uncond = (
            self.forward_sequence(decoder_input, cu, max_len, targets_norm, null_mask)
            if use_cfg
            else None
        )

        def _guide(val_c: torch.Tensor, val_u: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
            if isinstance(scale, (int, float)) and scale == 1.0:
                return val_c.float()
            return val_u.float() + scale * (val_c.float() - val_u.float())

        owner = torch.repeat_interleave(
            torch.arange(len(trajectories), device=device), (cu[1:] - cu[:-1]).long()
        )
        packets: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        tree = target.is_gen_tree
        if tree.any():
            logits_t = self.backbone.tree_heads[0](hidden[tree]).float()
            if uncond is not None and cfg_scale != 1.0:
                tree_u = self.backbone.tree_heads[0](uncond[tree]).float()
                if cfg_tree_decay != 0.0:
                    tree_lvl = decoder_input.level[tree].float()
                    decay_mult = 1.0 - float(cfg_tree_decay) * ((3.0 - tree_lvl).clamp(0.0, 2.0) / 2.0)
                    eff_cfg = (1.0 + (cfg_scale - 1.0) * decay_mult).unsqueeze(-1)
                    logits_t = _guide(logits_t, tree_u, eff_cfg)
                else:
                    logits_t = _guide(logits_t, tree_u, cfg_scale)
            logits_t = logits_t / temperatures[0]
            labels_t = (target.type[tree] - self.backbone.tree0_split_id - 1).long()
            packets.append((logits_t.log_softmax(-1), labels_t, owner[tree]))

        atom = target.is_gen_atom
        if atom.any():
            labels_a = [v.long() for v in target.to_list(atom)]
            logits_a = [t.float() for t in self.backbone.atom_heads[0](hidden[atom], labels_a)]
            if uncond is not None:
                null_logits_a = self.backbone.atom_heads[0](uncond[atom], labels_a)
                logits_a = [
                    _guide(x, u, cfg_scale if ax == 0 else xyz_cfg_scale)
                    for ax, (x, u) in enumerate(zip(logits_a, null_logits_a))
                ]

            atom_owner = owner[atom]
            if geometric_guard:
                counts_per_traj = torch.bincount(atom_owner, minlength=len(trajectories))
                max_k = int(counts_per_traj.max().item()) if counts_per_traj.numel() > 0 else 0
                vox_origins = decoder_input.phy_pos[atom].float() - 0.5 * float(self.args.grid_len)
                if max_k > 1:
                    cum_starts = torch.zeros(len(trajectories), dtype=torch.long, device=device)
                    cum_starts[1:] = counts_per_traj.cumsum(0)[:-1]
                    local_m = torch.arange(atom_owner.shape[0], device=device) - cum_starts[atom_owner]
                    xyz_bins = torch.stack(labels_a[1:4], dim=-1).float()
                    atom_pos_all = vox_origins + xyz_bins * float(self.args.xyz_resolution)

                    traj_pos = torch.zeros((len(trajectories), max_k, 3), dtype=torch.float32, device=device)
                    traj_typ = torch.zeros((len(trajectories), max_k), dtype=torch.long, device=device)
                    traj_pos[atom_owner, local_m] = atom_pos_all
                    traj_typ[atom_owner, local_m] = labels_a[0]

                    p_pos = traj_pos[atom_owner]
                    p_types = traj_typ[atom_owner]
                    p_valid = (
                        (torch.arange(max_k, device=device).unsqueeze(0) < local_m.unsqueeze(1))
                        & (p_types < len(atom_list))
                        & (p_types >= 0)
                    )
                    p_rcov, p_sat = self._prepare_placed_geometry(p_pos, p_types, p_valid)

                    logits_a[0] = logits_a[0] + self._element_guard_logit_bias(
                        vox_origins, p_pos, p_valid, p_rcov, p_sat
                    )
                    logits_a[3] = logits_a[3] + self._zbin_guard_logit_bias(
                        vox_origins,
                        labels_a[0],
                        torch.stack(labels_a[1:3], dim=-1),
                        p_pos,
                        p_valid,
                        p_rcov,
                        p_sat,
                        logits_a[3].shape[-1],
                    )

            allowed = self.allowed_atom_mask(device)
            if electron_parity is not None:
                parities = atom_electron_parities(self.backbone, device)
                terminal = torch.ones_like(atom_owner, dtype=torch.bool)
                terminal[:-1] = atom_owner[:-1] != atom_owner[1:]
                accumulated = torch.zeros(len(trajectories), dtype=torch.long, device=device)
                accumulated.index_add_(0, atom_owner[~terminal], parities[labels_a[0][~terminal]])
                logits_a[0] = mask_terminal_element_logits(
                    logits_a[0], accumulated[atom_owner] % 2, terminal, electron_parity, parities
                )
            real_atom = (labels_a[0] >= 0) & (labels_a[0] < len(atom_list))
            for axis, vals in enumerate(logits_a):
                if topology_only_policy and axis > 0:
                    continue
                vals_f = vals.float() / (temperatures[1] if axis == 0 else temperatures[2])
                if axis == 0:
                    vals_f = vals_f.masked_fill(~allowed, -torch.inf)
                    packets.append((vals_f.log_softmax(-1), labels_a[axis], atom_owner))
                elif bool(real_atom.any().item()):
                    packets.append((vals_f[real_atom].log_softmax(-1), labels_a[axis][real_atom], atom_owner[real_atom]))

        return packets
