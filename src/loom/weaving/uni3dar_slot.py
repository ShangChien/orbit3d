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

from ..generator.condition_contrast import condition_margin_loss, incompatible_condition_indices
from ..generator.electronic_constraints import atom_electron_parities, mask_terminal_element_logits
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
from .lora import inject_uni3dar_lora, iter_lora_parameters
from .slots import GeneratorSlot


@lru_cache(maxsize=256)
def _suffix_causal_sdpa_mask(seq_q: int, seq_k: int, device_str: str) -> torch.Tensor:
    """Boolean mask `(1, 1, seq_q, seq_k)` where `True` means allowed to attend."""
    device = torch.device(device_str)
    q_idx = torch.arange(seq_q, device=device)[:, None] + (seq_k - seq_q)
    k_idx = torch.arange(seq_k, device=device)[None, :]
    return (k_idx <= q_idx).unsqueeze(0).unsqueeze(0)


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
        num_prefix_tokens: int = 4,
        modulation_cap: float = 4.0,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        lora_include_ffn: bool = True,
        lora_include_heads: bool = True,
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

        self.injected_lora_modules: List[str] = []
        if lora_rank > 0:
            self.injected_lora_modules = inject_uni3dar_lora(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                include_ffn=lora_include_ffn,
                include_heads=lora_include_heads,
            )

        self.register_buffer("_subcell_shifts", torch.tensor(subcell_orders, dtype=torch.int32), persistent=False)
        self.register_buffer("_bit_shifts", (1 << torch.arange(8, dtype=torch.int32)), persistent=False)
        self.configure_trainable()

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
        device: str = "cuda:0",
        batch_size: int = 32,
        **loom_kwargs: Any,
    ) -> "LoomUni3DARGenerator":
        raw = torch.load(parent_ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and raw.get("backend") == cls.backend_id:
            model, _ = cls.load_checkpoint(parent_ckpt_path, device=device, batch_size=batch_size)
            return model

        if isinstance(raw, dict) and "ema" in raw and "params" in raw["ema"]:
            backbone_state = raw["ema"]["params"]
            depth = int(backbone_state["cur_level.weight"].shape[0]) - 2
            allowed = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I")
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            rep = raw.get("representation_config", {})
            full_sd = raw["model_state_dict"]
            depth = int(rep.get("merge_level", int(full_sd["cur_level.weight"].shape[0]) - 2))
            allowed = tuple(
                rep.get(
                    "allowed_elements",
                    raw.get("contract", {})
                    .get("semantics", {})
                    .get("allowed_elements", ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I")),
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
        unfreeze_full_backbone: bool = False,
    ) -> List[nn.Parameter]:
        self.backbone.requires_grad_(unfreeze_full_backbone)
        if train_lora and not unfreeze_full_backbone:
            for _, param in iter_lora_parameters(self.backbone):
                param.requires_grad_(True)
        self.conditioner.requires_grad_(train_conditioner)
        return [p for p in self.parameters() if p.requires_grad]

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
    def _run_layer_batched_sdpa(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, emb_dim = x.shape
        h = layer.attn_norm(x)
        num_heads = layer.attn.num_heads
        head_dim = layer.attn.head_dim
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
        attn_out = layer.attn.out_proj(attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, emb_dim))
        x = x + layer.dropout(attn_out)
        x = x + layer.dropout(layer.ffn(layer.ffn_norm(x)))
        return x

    def _run_layer_kv_step(self, layer: nn.Module, x: torch.Tensor, kv_cache: Dict[str, Any]) -> torch.Tensor:
        bsz, seq_q, emb_dim = x.shape
        h = layer.attn_norm(x)
        num_heads = layer.attn.num_heads
        head_dim = layer.attn.head_dim
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
        if seq_q == seq_k:
            attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all, is_causal=True)
        elif seq_q == 1:
            attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all, is_causal=False)
        else:
            mask = _suffix_causal_sdpa_mask(seq_q, seq_k, str(x.device))
            attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all, attn_mask=mask)

        attn_out = layer.attn.out_proj(attn_out.transpose(1, 2).contiguous().view(bsz, seq_q, emb_dim))
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
                x = self._run_layer_batched_sdpa(layer, x)
                x = self.conditioner.modulate_layer(l_idx, x, c_3d)

        x = self.backbone.norm(x)
        x = self.conditioner.modulate_out(x, c_3d)
        out_seq = x[:, p_len:, :]
        return out_seq[batch_ids, intra_pos]

    # ------------------------------------------------------------------
    # Supervised / Hindsight + Contrastive Margin Objective
    # ------------------------------------------------------------------
    def _nll_from_hidden(
        self,
        dec_y: torch.Tensor,
        decoder_gen_target: Any,
        cu_lens: torch.Tensor,
        bsz: int,
        *,
        xyz_loss_weight: float = 0.1,
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

        gen_loss = loss_tree + loss_atom + xyz_loss_weight * loss_xyz
        info: Dict[str, Any] = {
            "loss_gen": float(gen_loss.detach().item()),
            "loss_tree": float(loss_tree.detach().item()),
            "loss_atom": float(loss_atom.detach().item()),
            "loss_xyz": float(loss_xyz.detach().item()),
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
        xyz_loss_weight: float = 0.1,
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
            xyz_loss_weight=xyz_loss_weight,
            atom_class_weights=atom_class_weights,
            return_per_molecule=need_per_mol,
        )

        total_loss = gen_loss
        if contrast_weight > 0.0:
            targets_eV = self.denormalize_targets(targets_norm)
            has_cond_row = train_mask.any(dim=-1)
            # Build masked eV vector for active axes so incompatible pair selection respects active axes
            active_eV = torch.where(train_mask, targets_eV, torch.zeros_like(targets_eV))
            neg_idx, pair_ok = incompatible_condition_indices(active_eV, halfwidth=contrast_min_gap_eV)
            eligible = pair_ok & has_cond_row & has_cond_row[neg_idx]
            if bool(eligible.any()):
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
            active_eV = torch.where(active_mask.bool(), targets_eV, torch.zeros_like(targets_eV))
            neg_idx, pair_ok = incompatible_condition_indices(active_eV, halfwidth=min_gap_eV)
            eligible = pair_ok & active_mask.bool().any(dim=-1)

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

    @torch.no_grad()
    def sample(
        self,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        cfg_scale: float = 1.0,
        xyz_cfg_scale: float = 1.0,
        temperatures: Tuple[float, float, float] = (0.85, 0.30, 0.25),
        seed: Optional[int] = None,
        use_bf16: bool = True,
        electron_parity: Optional[int] = 0,
    ) -> List[Dict[str, Any]]:
        """High-throughput conditional octree trajectory generation with Selective Logit-CFG."""
        was_training = self.training
        self.eval()
        try:
            device = targets_norm.device
            batch_size = targets_norm.shape[0]
            tree_temp, atom_temp, xyz_temp = temperatures
            for name, val in (("tree", tree_temp), ("atom", atom_temp), ("xyz", xyz_temp)):
                if not math.isfinite(val) or val <= 0:
                    raise ValueError(f"{name} temperature must be positive and finite")

            has_cond = bool(active_mask.bool().any().item())
            use_cfg = has_cond and (cfg_scale != 1.0 or xyz_cfg_scale != 1.0)

            rng_gen: Optional[torch.Generator] = None
            if seed is not None:
                rng_gen = torch.Generator(device=device)
                rng_gen.manual_seed(int(seed))

            parities = atom_electron_parities(self.backbone, device) if electron_parity is not None else None
            forbidden_atom_mask = ~self.allowed_atom_mask(device)
            score_ratio = [1.0, 0.0, 0.0, 0.0] if self.args.rank_by == "atom" else [1.0, 1.0, 1.0, 1.0]

            autocast_enabled = bool(use_bf16 and device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                global_c, has_any = self.conditioner.encode_condition(targets_norm, active_mask)
                prefix_cond = self.conditioner.build_prefix_tokens(global_c, has_any)
                root_delta = self.conditioner.forward_root_prefix(global_c)

                if use_cfg:
                    null_mask = torch.zeros_like(active_mask, dtype=torch.bool)
                    global_c_uncond, has_any_uncond = self.conditioner.encode_condition(targets_norm, null_mask)
                    prefix_uncond = self.conditioner.build_prefix_tokens(global_c_uncond, has_any_uncond)
                else:
                    global_c_uncond = None
                    prefix_uncond = None

                kv_cache = [[{"step": 1, "mode": "concat"} for _ in range(self.args.layer)] for _ in range(self.args.recycle)]

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

                decoder_feat, decoder_phy_pos, raw_feat = self.backbone.calc_root_level(
                    self.args.merge_level, batch_size, device, 0, 0
                )
                scores = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
                finished_trajs: List[Dict[str, Any]] = []

                cur_feat = decoder_feat[:, -1, :]
                cur_phy_pos = decoder_phy_pos[:, -1, :]
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
                p_len = self.conditioner.num_prefix_tokens

                while len(raw_feat) > 0:
                    b_active = decoder_feat.shape[0]
                    step_emb = self.backbone.input_embedding(cur_input)
                    step_grid = cur_input.phy_pos.float() / self.args.xyz_resolution

                    if step_idx == 0:
                        emb_cond = step_emb + root_delta.unsqueeze(1).to(step_emb.dtype)
                        if prefix_cond is not None and p_len > 0:
                            emb_cond = torch.cat([prefix_cond.to(step_emb.dtype), emb_cond], dim=1)
                            pref_grid = self._prefix_grid_positions(b_active, device)
                            step_grid = torch.cat([pref_grid, step_grid], dim=1)
                        if use_cfg:
                            emb_uncond = step_emb
                            if prefix_uncond is not None and p_len > 0:
                                emb_uncond = torch.cat([prefix_uncond.to(step_emb.dtype), emb_uncond], dim=1)
                            x_run = torch.cat([emb_cond, emb_uncond], dim=0)
                            grid_run = step_grid.repeat(2, 1, 1)
                            c_run = torch.cat([cond_vars["global_c"], cond_vars["global_c_uncond"]], dim=0)
                        else:
                            x_run = emb_cond
                            grid_run = step_grid
                            c_run = cond_vars["global_c"]
                    else:
                        if use_cfg:
                            x_run = step_emb.repeat(2, 1, 1)
                            grid_run = step_grid.repeat(2, 1, 1)
                            c_run = torch.cat([cond_vars["global_c"], cond_vars["global_c_uncond"]], dim=0)
                        else:
                            x_run = step_emb
                            grid_run = step_grid
                            c_run = cond_vars["global_c"]

                    self.backbone.rope.calc_and_cache(grid_run)
                    for r_i in range(self.args.recycle):
                        for l_i, layer in enumerate(self.backbone.decoder.layers):
                            x_run = self._run_layer_kv_step(layer, x_run, kv_cache[r_i][l_i])
                            x_run = self.conditioner.modulate_layer(l_i, x_run, c_run)
                    x_run = self.backbone.norm(x_run)
                    x_run = self.conditioner.modulate_out(x_run, c_run)

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

                    next_feat = torch.cat([q[0:1] for q in raw_feat], dim=0)
                    raw_feat = [q[1:] for q in raw_feat]
                    next_type = next_feat[:, IDX_TYPE]
                    next_level = next_feat[:, IDX_LEVEL]
                    need_pred = next_type < 0
                    is_pred_atom = (last_level == self.backbone.atom_level_index) & (last_type == self.backbone.atom_mask_id) & need_pred
                    is_pred_tree = (next_level > self.backbone.atom_level_index) & need_pred
                    is_first_cnt = (last_level == self.args.merge_level) & (last_type == self.backbone.atom_mask_id)

                    pred_type = next_type.clone()
                    pred_count = torch.zeros((b_active,), dtype=torch.int32, device=device)
                    pred_xyz = torch.full((b_active, 3), self.backbone.xyz_null_id, dtype=torch.int32, device=device)
                    score_temp = torch.zeros((b_active,), dtype=torch.float32, device=device)

                    if bool(is_pred_atom.any().item()):
                        feat_c = dec_y_last[is_pred_atom]
                        feat_u = dec_u_last[is_pred_atom] if dec_u_last is not None else None
                        n_sub = feat_c.shape[0]
                        sub_xyz = torch.zeros((n_sub, 4), dtype=torch.int32, device=device)
                        sub_sc = torch.zeros((n_sub,), dtype=torch.float32, device=device)

                        if electron_parity is not None and parities is not None:
                            types_all = decoder_feat[:, :, IDX_TYPE]
                            actual = (
                                (decoder_feat[:, :, IDX_LEVEL] == self.backbone.atom_level_index)
                                & (types_all < len(atom_list))
                                & (types_all >= 0)
                            )
                            safe = torch.where(actual, types_all, self.backbone.atom_null_id)
                            acc_par = (parities[safe].sum(-1) % 2)[is_pred_atom]
                            term_flags = torch.tensor(
                                [len(q) == 0 for q in raw_feat], dtype=torch.bool, device=device
                            )[is_pred_atom]
                        else:
                            acc_par = None
                            term_flags = None

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
                                cur_logits = cur_logits.masked_fill(forbidden_atom_mask, -torch.inf)
                                if electron_parity is not None and parities is not None and acc_par is not None and term_flags is not None:
                                    cur_logits = mask_terminal_element_logits(
                                        cur_logits, acc_par, term_flags, electron_parity, parities
                                    )
                                temp_axis = atom_temp
                            else:
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

                    if bool(is_pred_tree.any().item()):
                        tree_c = self.backbone.tree_heads[0](dec_y_last[is_pred_tree]).float()
                        if dec_u_last is not None and cfg_scale != 1.0:
                            tree_u = self.backbone.tree_heads[0](dec_u_last[is_pred_tree]).float()
                            tree_logits = tree_u + cfg_scale * (tree_c - tree_u)
                        else:
                            tree_logits = tree_c
                        t_pred, t_sc = _gpu_multinomial_sample(tree_logits, tree_temp, rng_gen)
                        pred_type[is_pred_tree] = t_pred + self.backbone.tree0_split_id + 1
                        score_temp[is_pred_tree] = score_temp[is_pred_tree] + t_sc

                    scores = torch.cat([scores, score_temp.unsqueeze(1)], dim=1)
                    count_vars["total_atoms"] = torch.where(is_first_cnt, pred_count, count_vars["total_atoms"])

                    raw_feat, count_vars = self._fast_tree_expand(raw_feat, last_type, pred_type, next_feat, count_vars)
                    decoder_feat, decoder_phy_pos = self.backbone.get_feat_by_pred(
                        next_feat, decoder_feat, decoder_phy_pos, pred_type, pred_count, pred_xyz
                    )
                    count_vars["last_remaining_atoms"] = last_rem_atoms
                    count_vars["last_remaining_tokens"] = last_rem_tokens
                    count_vars["pred_count"] = pred_count

                    keep_flags: List[bool] = []
                    next_feat_list: List[torch.Tensor] = []
                    new_raw_feat: List[torch.Tensor] = []
                    orig_indices_list: Optional[List[int]] = None
                    if any(q.shape[0] == 0 for q in raw_feat):
                        orig_indices_list = cond_vars["orig_idx"].tolist()

                    for i, q in enumerate(raw_feat):
                        if q.shape[0] > 0:
                            keep_flags.append(True)
                            next_feat_list.append(q[0:1])
                            new_raw_feat.append(q[1:])
                        else:
                            keep_flags.append(False)
                            finished_trajs.append(
                                {
                                    "decoder_feat": decoder_feat[i].clone(),
                                    "decoder_phy_pos": decoder_phy_pos[i].clone(),
                                    "scores": scores[i].clone(),
                                    "batch_idx": orig_indices_list[i],  # type: ignore[index]
                                }
                            )

                    if not new_raw_feat:
                        break

                    rows_finished = len(new_raw_feat) != len(raw_feat)
                    next_feat = torch.cat(next_feat_list, dim=0)
                    raw_feat = new_raw_feat

                    if rows_finished:
                        keep_mask = torch.tensor(keep_flags, dtype=torch.bool, device=device)
                        decoder_feat = decoder_feat[keep_mask]
                        decoder_phy_pos = decoder_phy_pos[keep_mask]
                        for k in cond_vars:
                            if cond_vars[k] is not None:
                                cond_vars[k] = cond_vars[k][keep_mask]
                        for k in count_vars:
                            count_vars[k] = count_vars[k][keep_mask]
                        scores = scores[keep_mask]
                        kv_keep = torch.cat([keep_mask, keep_mask], dim=0) if use_cfg else keep_mask
                        for r_i in range(len(kv_cache)):
                            for l_i in range(len(kv_cache[r_i])):
                                compact_kv(kv_cache[r_i][l_i], kv_keep)

                    last_level = decoder_feat[:, -1, IDX_LEVEL]
                    is_new_level = last_level != next_feat[:, IDX_LEVEL]
                    if self.backbone.num_tree > 1:
                        raw_feat = self.backbone.add_new_tree(raw_feat, next_feat, is_new_level)

                    decoder_feat, decoder_phy_pos = self.backbone.get_feat_by_pred(
                        next_feat, decoder_feat, decoder_phy_pos, next_feat[:, IDX_TYPE], None, None
                    )
                    scores = torch.cat(
                        [scores, torch.zeros((next_feat.shape[0], 1), device=device, dtype=torch.float32)], dim=1
                    )
                    count_vars, pair_rem_a, pair_rem_t = self._fast_update_running_stats(
                        count_vars, is_new_level, last_level
                    )

                    cur_feat = decoder_feat[:, -2:, :]
                    cur_phy_pos = decoder_phy_pos[:, -2:, :]
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
                    cur_input.remaining_atoms = torch.where(cur_is_cls, torch.zeros_like(cur_input.remaining_atoms), cur_input.remaining_atoms).clamp_(min=0)
                    cur_input.remaining_tokens = torch.where(cur_is_cls, torch.zeros_like(cur_input.remaining_tokens), cur_input.remaining_tokens).clamp_(min=0)
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

            return finished_trajs
        finally:
            self.train(was_training)

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
        electron_parity: Optional[int] = 0,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        device = targets_norm.device
        packed = self.pack_sampled_trajectories(trajectories, device)
        decoder_input, target, _ = self.backbone.get_feat_from_dataloader(packed)
        cu = packed["decoder_input_len"]
        max_len = int(packed["decoder_input_max_len"].item())

        hidden = self.forward_sequence(decoder_input, cu, max_len, targets_norm, active_mask)
        use_cfg = bool(active_mask.bool().any().item()) and (cfg_scale != 1.0 or xyz_cfg_scale != 1.0)
        uncond = (
            self.forward_sequence(decoder_input, cu, max_len, targets_norm, torch.zeros_like(active_mask, dtype=torch.bool))
            if use_cfg
            else None
        )

        def _guide(val_c: torch.Tensor, val_u: torch.Tensor, scale: float) -> torch.Tensor:
            if scale == 1.0:
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
                logits_t = _guide(logits_t, self.backbone.tree_heads[0](uncond[tree]), cfg_scale)
            logits_t = logits_t / temperatures[0]
            labels_t = (target.type[tree] - self.backbone.tree0_split_id - 1).long()
            packets.append((logits_t.log_softmax(-1), labels_t, owner[tree]))

        atom = target.is_gen_atom
        if atom.any():
            labels_a = [v.long() for v in target.to_list(atom)]
            logits_a = self.backbone.atom_heads[0](hidden[atom], labels_a)
            if uncond is not None:
                null_logits_a = self.backbone.atom_heads[0](uncond[atom], labels_a)
                logits_a = [
                    _guide(x, u, cfg_scale if ax == 0 else xyz_cfg_scale)
                    for ax, (x, u) in enumerate(zip(logits_a, null_logits_a))
                ]
            allowed = self.allowed_atom_mask(device)
            if electron_parity is not None:
                parities = atom_electron_parities(self.backbone, device)
                atom_owner = owner[atom]
                terminal = torch.ones_like(atom_owner, dtype=torch.bool)
                terminal[:-1] = atom_owner[:-1] != atom_owner[1:]
                accumulated = torch.zeros(len(trajectories), dtype=torch.long, device=device)
                accumulated.index_add_(0, atom_owner[~terminal], parities[labels_a[0][~terminal]])
                logits_a[0] = mask_terminal_element_logits(
                    logits_a[0], accumulated[atom_owner] % 2, terminal, electron_parity, parities
                )
            for axis, vals in enumerate(logits_a):
                vals_f = vals.float() / (temperatures[1] if axis == 0 else temperatures[2])
                if axis == 0:
                    vals_f = vals_f.masked_fill(~allowed, -torch.inf)
                packets.append((vals_f.log_softmax(-1), labels_a[axis], owner[atom]))

        return packets
