#!/usr/bin/env python3
"""Export Uni-3DAR / Uni-3DAR-Mol bundled checkpoints (.pt) into compact .u3m bundles for uni3dar-rs.

Features:
  1. Auto-merges all `LoRALinear` adapters (`W_eff = W_base + (alpha / rank) * (B @ A)`) across
     Transformer Attention, SwiGLU FFN, `tree_heads`, and `atom_heads` (including optional `xyz` LoRA).
  2. Packs `LoomUni3DARGenerator` + `PostTreeGraphRefiner` + `TwoStageOctreeProcessCritic` into a
     single 64-byte-aligned `.u3m` file (`fp32`, `fp16`, `q8_0`, or `q4_0`).
  3. Keeps embeddings, RMSNorm/LayerNorm, `LoomConditioner`, `ScaleFreeBudgetEmbedding`, and geometric
     RBF/multipole weights in `FP32` while quantizing heavy `Linear` matrices (~95% of params).
  4. Optionally dumps a comprehensive PyTorch ground-truth parity fixture (`--dump-parity`) covering:
     - Multi-property `LoomConditioner` + `num_prefix_tokens` + `ScaleFreeBudgetEmbedding`
     - Step-0 `[prefix..., root_MASK]` and Step-1 2-token `[prev_pred, next_MASK]` KV-cached decoding
     - `TreeHead` & 4-axis `AutoRegressivePredictionHead`
     - `PostTreeGraphRefiner` (`refined_pos` + `bond_order_probs`)
     - `Stage-1 OctreeProcessCritic` (`14-D` coarse voxel geometry + utility)
     - `Stage-2 OctreeElementProcessCritic` (`42-D` element cloud + `64-RBF` + `12-ch` multipoles + utility)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "compat", ROOT / "src", ROOT):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MAGIC = b"U3DM\x01\x00\x00\x00"
ALIGN = 64


def _pad_bytes(raw: bytes, align: int = ALIGN) -> bytes:
    rem = len(raw) % align
    return raw if rem == 0 else raw + (b"\x00" * (align - rem))


def quantize_q8_0(w_2d: np.ndarray) -> Tuple[bytes, int, int]:
    """Symmetric block-32 INT8: 9 x u32 (36 B) per 32 weights = [f32 scale, 32 x i8]."""
    rows, cols = w_2d.shape
    pad_cols = ((cols + 31) // 32) * 32
    if pad_cols != cols:
        w_pad = np.zeros((rows, pad_cols), dtype=np.float32)
        w_pad[:, :cols] = w_2d
    else:
        w_pad = np.ascontiguousarray(w_2d, dtype=np.float32)
    n_blocks = pad_cols // 32
    blocks = w_pad.reshape(rows * n_blocks, 32)
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax > 0.0, amax / 127.0, 1.0).astype(np.float32)
    inv_scales = np.where(amax > 0.0, 127.0 / amax, 0.0).astype(np.float32)
    q = np.clip(np.rint(blocks * inv_scales[:, None]), -127, 127).astype(np.int8)

    out = np.empty((rows * n_blocks, 36), dtype=np.uint8)
    out[:, :4] = scales.view(np.uint8).reshape(-1, 4)
    out[:, 4:] = q.view(np.uint8)
    return out.tobytes(), pad_cols, n_blocks


def quantize_q4_0(w_2d: np.ndarray) -> Tuple[bytes, int, int]:
    """Symmetric block-32 signed INT4 [-7..+7]: 5 x u32 (20 B) per 32 weights = [f32 scale, 16 B packed nibbles].

    Each u32 holds 8 signed 4-bit two's-complement nibbles (nibble j at bits [4*j .. 4*j+3]),
    decoded in WGSL via `extractBits(bitcast<i32>(packed_u32), j * 4u, 4u)`.
    """
    rows, cols = w_2d.shape
    pad_cols = ((cols + 31) // 32) * 32
    if pad_cols != cols:
        w_pad = np.zeros((rows, pad_cols), dtype=np.float32)
        w_pad[:, :cols] = w_2d
    else:
        w_pad = np.ascontiguousarray(w_2d, dtype=np.float32)
    n_blocks = pad_cols // 32
    blocks = w_pad.reshape(rows * n_blocks, 32)
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax > 0.0, amax / 7.0, 1.0).astype(np.float32)
    inv_scales = np.where(amax > 0.0, 7.0 / amax, 0.0).astype(np.float32)
    q = np.clip(np.rint(blocks * inv_scales[:, None]), -7, 7).astype(np.int8)
    nibbles = (q.astype(np.int16) & 0x0F).astype(np.uint8)  # (N_blk, 32)
    low = nibbles[:, 0::2]   # even indices 0, 2, ..., 30 -> low 4 bits of byte
    high = nibbles[:, 1::2]  # odd indices 1, 3, ..., 31 -> high 4 bits of byte
    packed_bytes = (low | (high << 4)).astype(np.uint8)  # (N_blk, 16), little-endian nibble order!

    out = np.empty((rows * n_blocks, 20), dtype=np.uint8)
    out[:, :4] = scales.view(np.uint8).reshape(-1, 4)
    out[:, 4:] = packed_bytes
    return out.tobytes(), pad_cols, n_blocks


class BundleWriter:
    def __init__(self, quant_dtype: str) -> None:
        self.quant_dtype = quant_dtype
        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.blobs: List[bytes] = []
        self.offset = 0
        self.lora_merged = 0

    def _append(self, name: str, raw: bytes, meta: Dict[str, Any]) -> None:
        padded = _pad_bytes(raw, ALIGN)
        entry = {"offset": self.offset, "nbytes": len(raw), **meta}
        self.tensors[name] = entry
        self.blobs.append(padded)
        self.offset += len(padded)

    def add_f32(self, name: str, t: torch.Tensor) -> None:
        arr = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype="<f4")
        self._append(
            name,
            arr.tobytes(),
            {"dtype": "f32", "shape": list(arr.shape), "orig_cols": int(arr.shape[-1]) if arr.ndim >= 1 else 1},
        )

    def add_linear_weight(self, name: str, w_t: torch.Tensor, *, force_fp32: bool = False, min_q8: bool = False) -> None:
        w = np.ascontiguousarray(w_t.detach().cpu().float().numpy(), dtype=np.float32)
        assert w.ndim == 2, f"{name}: expected 2D weight, got {w.shape}"
        rows, cols = w.shape
        dt = "fp32" if force_fp32 else ("q8_0" if (min_q8 and self.quant_dtype == "q4_0") else self.quant_dtype)
        if dt == "fp32":
            raw = w.astype("<f4").tobytes()
            self._append(name, raw, {"dtype": "f32", "shape": [rows, cols], "orig_cols": cols, "pad_cols": cols})
        elif dt == "fp16":
            pad_cols = cols + (cols % 2)
            w_p = np.zeros((rows, pad_cols), dtype=np.float32)
            w_p[:, :cols] = w
            raw = w_p.astype("<f2").tobytes()
            self._append(name, raw, {"dtype": "f16", "shape": [rows, cols], "orig_cols": cols, "pad_cols": pad_cols})
        elif dt == "q8_0":
            raw, pad_cols, n_blocks = quantize_q8_0(w)
            self._append(
                name,
                raw,
                {"dtype": "q8_0", "shape": [rows, cols], "orig_cols": cols, "pad_cols": pad_cols, "blocks_per_row": n_blocks},
            )
        elif dt == "q4_0":
            raw, pad_cols, n_blocks = quantize_q4_0(w)
            self._append(
                name,
                raw,
                {"dtype": "q4_0", "shape": [rows, cols], "orig_cols": cols, "pad_cols": pad_cols, "blocks_per_row": n_blocks},
            )
        else:
            raise ValueError(f"Unsupported dtype: {dt}")

    def add_sd_linear(
        self,
        sd: Dict[str, torch.Tensor],
        sd_prefix: str,
        out_prefix: str,
        *,
        lora_alpha: float = 0.0,
        lora_rank: int = 0,
        force_fp32: bool = False,
        min_q8: bool = False,
    ) -> None:
        base_k = f"{sd_prefix}.base.weight"
        plain_k = f"{sd_prefix}.weight"
        if base_k in sd:
            w_base = sd[base_k].detach().cpu().float()
            la = sd[f"{sd_prefix}.lora_A"].detach().cpu().float()
            lb = sd[f"{sd_prefix}.lora_B"].detach().cpu().float()
            r = int(la.shape[0])
            scale = float(lora_alpha) / float(max(r, 1)) if lora_rank > 0 else 1.0
            w_eff = w_base + scale * (lb @ la)
            self.lora_merged += 1
            bias_t = sd.get(f"{sd_prefix}.base.bias")
        elif plain_k in sd:
            w_eff = sd[plain_k].detach().cpu().float()
            bias_t = sd.get(f"{sd_prefix}.bias")
        else:
            raise KeyError(f"Missing linear weight '{plain_k}' or '{base_k}'")

        self.add_linear_weight(f"{out_prefix}.weight", w_eff, force_fp32=force_fp32, min_q8=min_q8)
        if bias_t is not None:
            self.add_f32(f"{out_prefix}.bias", bias_t)

    def add_sd_norm(self, sd: Dict[str, torch.Tensor], sd_prefix: str, out_prefix: str) -> None:
        self.add_f32(f"{out_prefix}.weight", sd[f"{sd_prefix}.weight"])
        if f"{sd_prefix}.bias" in sd:
            self.add_f32(f"{out_prefix}.bias", sd[f"{sd_prefix}.bias"])


@torch.no_grad()
def export_uni3dar_bundle(ck: Dict[str, Any], out_path: Path, *, dtype: str = "q8_0") -> Dict[str, Any]:
    bw = BundleWriter(dtype)
    sd: Dict[str, torch.Tensor] = ck["state_dict"]
    rcfg: Dict[str, Any] = dict(ck["representation_config"])
    lcfg: Dict[str, Any] = dict(ck["loom_config"])

    n_layers = int(rcfg.get("layer", 12))
    emb_dim = int(rcfg.get("emb_dim", 768))
    num_head = int(rcfg.get("num_head", 12))
    ffn_mult = int(rcfg.get("ffn_multiple", 4))
    merge_level = int(rcfg.get("merge_level", 7))
    lora_rank = int(lcfg.get("lora_rank", 0))
    lora_alpha = float(lcfg.get("lora_alpha", 0.0))

    # 1. Backbone Embeddings & Final RMSNorm (strictly FP32!)
    bw.add_f32("bb.atom_emb", sd["backbone.atom_emb.weight"])
    for a in range(3):
        bw.add_f32(f"bb.xyz_emb.{a}", sd[f"backbone.xyz_emb.{a}.weight"])
    bw.add_f32("bb.cur_level", sd["backbone.cur_level.weight"])
    bw.add_f32("bb.norm.weight", sd["backbone.norm.weight"])

    # 2. Transformer Decoder Layers (0 .. n_layers - 1)
    for l in range(n_layers):
        lp = f"backbone.decoder.layers.{l}"
        op = f"bb.layers.{l}"
        bw.add_f32(f"{op}.attn_norm.weight", sd[f"{lp}.attn_norm.weight"])
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            bw.add_sd_linear(sd, f"{lp}.attn.{proj}", f"{op}.attn.{proj}", lora_alpha=lora_alpha, lora_rank=lora_rank)
        bw.add_f32(f"{op}.ffn_norm.weight", sd[f"{lp}.ffn_norm.weight"])
        for fn in ("w1", "w2", "w3"):
            bw.add_sd_linear(sd, f"{lp}.ffn.{fn}", f"{op}.ffn.{fn}", lora_alpha=lora_alpha, lora_rank=lora_rank)

    # 3. Tree Head & 4-Axis AutoRegressive Atom/XYZ Heads (keep dense >= Q8_0 and out in FP32!)
    bw.add_sd_linear(
        sd, "backbone.tree_heads.0.dense", "bb.tree_head.dense", lora_alpha=lora_alpha, lora_rank=lora_rank, min_q8=True
    )
    bw.add_sd_linear(
        sd, "backbone.tree_heads.0.out", "bb.tree_head.out", lora_alpha=lora_alpha, lora_rank=lora_rank, force_fp32=True
    )
    for e_idx in range(3):
        bw.add_f32(f"bb.atom_head.emb.{e_idx}", sd[f"backbone.atom_heads.0.emb_layers.{e_idx}.weight"])
    for p_idx in range(4):
        bw.add_sd_linear(
            sd,
            f"backbone.atom_heads.0.pred_heads.{p_idx}.dense",
            f"bb.atom_head.pred.{p_idx}.dense",
            lora_alpha=lora_alpha,
            lora_rank=lora_rank,
            min_q8=True,
        )
        # Keep small logit projection `out` (602 or 24 rows) in FP32 for crisp categorical logits!
        bw.add_sd_linear(
            sd,
            f"backbone.atom_heads.0.pred_heads.{p_idx}.out",
            f"bb.atom_head.pred.{p_idx}.out",
            lora_alpha=lora_alpha,
            lora_rank=lora_rank,
            force_fp32=True,
        )

    # 4. ScaleFreeBudgetEmbedding (FP32, tiny < 0.2M params)
    has_budget_emb = "budget_emb.net.0.weight" in sd
    if has_budget_emb:
        bw.add_sd_linear(sd, "budget_emb.net.0", "budget_emb.0", force_fp32=True)
        bw.add_sd_norm(sd, "budget_emb.net.1", "budget_emb.1")
        bw.add_sd_linear(sd, "budget_emb.net.3", "budget_emb.3", force_fp32=True)

    # 5. LoomConditioner (FP32, runs ONCE per prompt!)
    target_cols = list(ck.get("target_cols") or [])
    target_units = list(ck.get("target_units") or [])
    target_mean = [float(x) for x in (ck.get("mean") or sd["target_mean"].tolist())]
    target_scale = [float(x) for x in (ck.get("scale") or sd["target_scale"].tolist())]
    num_props = len(target_cols)

    cond_cfg: Optional[Dict[str, Any]] = None
    if "conditioner.rbf_centers" in sd and num_props > 0:
        num_rbf = int(sd["conditioner.rbf_centers"].numel())
        num_freqs = int(sd["conditioner.fourier_freqs"].numel())
        num_prefix = int(lcfg.get("num_prefix_tokens", 0))
        mod_cap = float(lcfg.get("modulation_cap", 4.0))
        spacing = 9.0 / float(max(num_rbf - 1, 1))
        rbf_gamma = 1.0 / (2.0 * spacing * spacing)
        cond_cfg = {
            "num_props": num_props,
            "num_rbf": num_rbf,
            "num_freqs": num_freqs,
            "num_prefix_tokens": num_prefix,
            "modulation_cap": mod_cap,
            "rbf_gamma": rbf_gamma,
            "rope_center_prefix": bool(lcfg.get("rope_center_prefix", False)),
            "root_gate": float(sd["conditioner.root_gate"].item()),
        }
        bw.add_f32("cond.rbf_centers", sd["conditioner.rbf_centers"])
        bw.add_f32("cond.fourier_freqs", sd["conditioner.fourier_freqs"])
        bw.add_f32("cond.prop_id_emb", sd["conditioner.prop_id_emb.weight"])
        bw.add_f32("cond.inactive_prop_emb", sd["conditioner.inactive_prop_emb"])
        bw.add_sd_linear(sd, "conditioner.root_proj", "cond.root_proj")
        for k in range(num_props):
            bw.add_sd_linear(sd, f"conditioner.prop_encoders.{k}.0", f"cond.prop_enc.{k}.0", force_fp32=True)
            bw.add_sd_norm(sd, f"conditioner.prop_encoders.{k}.1", f"cond.prop_enc.{k}.1")
            bw.add_sd_linear(sd, f"conditioner.prop_encoders.{k}.3", f"cond.prop_enc.{k}.3")
            bw.add_sd_norm(sd, f"conditioner.prop_encoders.{k}.4", f"cond.prop_enc.{k}.4")
        bw.add_sd_linear(sd, "conditioner.global_mlp.0", "cond.global_mlp.0")
        bw.add_sd_norm(sd, "conditioner.global_mlp.1", "cond.global_mlp.1")
        bw.add_sd_linear(sd, "conditioner.global_mlp.3", "cond.global_mlp.3")
        bw.add_sd_norm(sd, "conditioner.global_mlp.4", "cond.global_mlp.4")
        if num_prefix > 0 and "conditioner.prefix_queries" in sd:
            bw.add_f32("cond.prefix_queries", sd["conditioner.prefix_queries"])
            bw.add_f32("cond.null_prefix", sd["conditioner.null_prefix"])
            bw.add_sd_linear(sd, "conditioner.prefix_proj.0", "cond.prefix_proj.0")
            bw.add_sd_norm(sd, "conditioner.prefix_proj.1", "cond.prefix_proj.1")
            bw.add_sd_linear(sd, "conditioner.prefix_proj.3", "cond.prefix_proj.3")
        for l in range(n_layers):
            bw.add_sd_linear(sd, f"conditioner.layer_mods.{l}.1", f"cond.layer_mod.{l}")
        bw.add_sd_linear(sd, "conditioner.out_mod.1", "cond.out_mod")

    # 6. PostTreeGraphRefiner (< 0.6M params -> strictly FP32 for sub-0.01A coordinate accuracy!)
    ref_cfg: Optional[Dict[str, Any]] = None
    if ck.get("post_tree_refiner_bundle") is not None:
        rb = ck["post_tree_refiner_bundle"]
        rsd = rb["state_dict"]
        ref_cfg = {
            "num_atom_classes": int(rb["num_atom_classes"]),
            "d_model": int(rb["d_model"]),
            "n_heads": int(rb["n_heads"]),
            "n_layers": int(rb["n_layers"]),
            "n_rbf": int(rb["n_rbf"]),
            "d_pair": int(rb["d_pair"]),
            "max_disp_A": float(rb["max_disp_A"]),
            "rbf_gamma": float(rsd["rbf_gamma"].item()),
        }
        bw.add_f32("ref.rbf_centers", rsd["rbf_centers"])
        bw.add_f32("ref.elem_emb", rsd["elem_emb.weight"])
        for l in range(ref_cfg["n_layers"]):
            rlp = f"layers.{l}"
            rop = f"ref.layers.{l}"
            bw.add_sd_norm(rsd, f"{rlp}.norm1", f"{rop}.norm1")
            bw.add_sd_linear(rsd, f"{rlp}.qkv", f"{rop}.qkv", force_fp32=True)
            bw.add_sd_linear(rsd, f"{rlp}.rbf_attn_bias.0", f"{rop}.rbf_bias.0", force_fp32=True)
            bw.add_sd_linear(rsd, f"{rlp}.rbf_attn_bias.2", f"{rop}.rbf_bias.2", force_fp32=True)
            bw.add_sd_linear(rsd, f"{rlp}.out_proj", f"{rop}.out_proj", force_fp32=True)
            bw.add_sd_norm(rsd, f"{rlp}.norm2", f"{rop}.norm2")
            bw.add_sd_linear(rsd, f"{rlp}.ffn.0", f"{rop}.ffn.0", force_fp32=True)
            bw.add_sd_linear(rsd, f"{rlp}.ffn.3", f"{rop}.ffn.3", force_fp32=True)
            bw.add_sd_linear(rsd, f"{rlp}.coord_src", f"{rop}.coord_src", force_fp32=True)
            bw.add_sd_linear(rsd, f"{rlp}.coord_dst", f"{rop}.coord_dst", force_fp32=True)
            bw.add_sd_linear(rsd, f"{rlp}.coord_rbf", f"{rop}.coord_rbf", force_fp32=True)
            bw.add_sd_linear(rsd, f"{rlp}.coord_mlp.1", f"{rop}.coord_mlp.1", force_fp32=True)
            bw.add_sd_linear(rsd, f"{rlp}.coord_mlp.3", f"{rop}.coord_mlp.3", force_fp32=True)
        bw.add_sd_linear(rsd, "bond_head.proj_mul", "ref.bond.proj_mul", force_fp32=True)
        bw.add_sd_linear(rsd, "bond_head.proj_add", "ref.bond.proj_add", force_fp32=True)
        bw.add_sd_linear(rsd, "bond_head.proj_rbf", "ref.bond.proj_rbf", force_fp32=True)
        bw.add_sd_norm(rsd, "bond_head.head.0", "ref.bond.head.0")
        bw.add_sd_linear(rsd, "bond_head.head.2", "ref.bond.head.2", force_fp32=True)
        bw.add_sd_linear(rsd, "bond_head.head.4", "ref.bond.head.4", force_fp32=True)

    # 7. TwoStageOctreeProcessCritic (Stage 1 + Stage 2)
    s1_cfg: Optional[Dict[str, Any]] = None
    s2_cfg: Optional[Dict[str, Any]] = None
    if ck.get("process_critic_bundle") is not None:
        pcb = ck["process_critic_bundle"]
        if pcb.get("stage1_state_dict") is not None:
            s1c = dict(pcb["stage1_config"])
            s1sd = pcb["stage1_state_dict"]
            s1_cfg = {
                "emb_dim": int(s1c["emb_dim"]),
                "num_targets": int(s1c["num_targets"]),
                "hidden_dim": int(s1c.get("hidden_dim", 512)),
                "geom_dim": int(s1c.get("geom_dim", 14)),
                "y_mean": [float(x) for x in s1sd["y_mean"].tolist()],
                "y_scale": [float(x) for x in s1sd["y_scale"].tolist()],
            }
            bw.add_sd_norm(s1sd, "h_norm", "c1.h_norm")
            bw.add_sd_linear(s1sd, "h_proj.0", "c1.h_proj")
            bw.add_sd_linear(s1sd, "aux_proj.0", "c1.aux_proj.0", force_fp32=True)
            bw.add_sd_linear(s1sd, "aux_proj.2", "c1.aux_proj.2", force_fp32=True)
            bw.add_sd_linear(s1sd, "trunk.0", "c1.trunk.0", force_fp32=True)
            bw.add_sd_norm(s1sd, "trunk.1", "c1.trunk.1")
            bw.add_sd_linear(s1sd, "trunk.4", "c1.trunk.4", force_fp32=True)
            bw.add_sd_norm(s1sd, "trunk.5", "c1.trunk.5")
            bw.add_sd_linear(s1sd, "prop_mean_head", "c1.prop_mean", force_fp32=True)
            bw.add_sd_linear(s1sd, "prop_logvar_head", "c1.prop_logvar", force_fp32=True)
            bw.add_sd_linear(s1sd, "legal_head", "c1.legal", force_fp32=True)
            bw.add_sd_linear(s1sd, "constr_head", "c1.constr", force_fp32=True)

        if pcb.get("stage2_state_dict") is not None:
            s2c = dict(pcb["stage2_config"])
            s2sd = pcb["stage2_state_dict"]
            aux_hid = int(s2sd["aux_proj.0.weight"].shape[0])
            s2_cfg = {
                "emb_dim": int(s2c["emb_dim"]),
                "num_targets": int(s2c["num_targets"]),
                "num_atom_classes": int(s2c.get("num_atom_classes", 602)),
                "num_h_pools": int(s2c.get("num_h_pools", 6)),
                "d_node": int(s2c.get("d_node", 256)),
                "n_rbf": int(s2c.get("n_rbf", 64)),
                "n_gnn_layers": int(s2c.get("n_gnn_layers", 4)),
                "hidden_dim": int(s2c.get("hidden_dim", 512)),
                "aux_hid": aux_hid,
                "cloud_feat_dim": int(s2c.get("cloud_feat_dim", 42)),
                "rbf_min": float(s2c.get("rbf_min", float(s2sd["rbf_centers"][0].item()))),
                "rbf_max": float(s2c.get("rbf_max", float(s2sd["rbf_centers"][-1].item()))),
                "rbf_gamma": float(s2sd["rbf_gamma"].item()),
                "edge_value_gate": bool(s2c.get("edge_value_gate", "gnn_layers.0.edge_filter.0.weight" in s2sd)),
                "equivariant_multipole": bool(s2c.get("equivariant_multipole", "multipole_q.weight" in s2sd)),
                "h_gate_scale": float(s2c.get("h_gate_scale", 1.0)),
                "y_mean": [float(x) for x in s2sd["y_mean"].tolist()],
                "y_scale": [float(x) for x in s2sd["y_scale"].tolist()],
            }
            bw.add_f32("c2.rbf_centers", s2sd["rbf_centers"])
            bw.add_f32("c2.elem_emb", s2sd["elem_emb.weight"])
            for l in range(s2_cfg["n_gnn_layers"]):
                glp = f"gnn_layers.{l}"
                gop = f"c2.gnn.{l}"
                bw.add_sd_norm(s2sd, f"{glp}.norm1", f"{gop}.norm1")
                bw.add_sd_linear(s2sd, f"{glp}.qkv", f"{gop}.qkv", force_fp32=True)
                bw.add_sd_linear(s2sd, f"{glp}.edge_proj.0", f"{gop}.edge_proj.0", force_fp32=True)
                bw.add_sd_linear(s2sd, f"{glp}.edge_proj.2", f"{gop}.edge_proj.2", force_fp32=True)
                bw.add_sd_linear(s2sd, f"{glp}.out_proj", f"{gop}.out_proj", force_fp32=True)
                bw.add_sd_norm(s2sd, f"{glp}.norm2", f"{gop}.norm2")
                bw.add_sd_linear(s2sd, f"{glp}.ffn.0", f"{gop}.ffn.0", force_fp32=True)
                bw.add_sd_linear(s2sd, f"{glp}.ffn.3", f"{gop}.ffn.3", force_fp32=True)
                if s2_cfg["edge_value_gate"]:
                    bw.add_sd_linear(s2sd, f"{glp}.edge_filter.0", f"{gop}.edge_filter.0", force_fp32=True)
                    bw.add_sd_linear(s2sd, f"{glp}.edge_filter.2", f"{gop}.edge_filter.2", force_fp32=True)
                    bw.add_sd_linear(s2sd, f"{glp}.v_edge_in", f"{gop}.v_edge_in", force_fp32=True)
                    bw.add_sd_linear(s2sd, f"{glp}.v_edge_out", f"{gop}.v_edge_out", force_fp32=True)
            if s2_cfg["equivariant_multipole"]:
                bw.add_sd_linear(s2sd, "multipole_q", "c2.multipole_q", force_fp32=True)
            if s2_cfg["h_gate_scale"] != 0.0:
                bw.add_sd_norm(s2sd, "h_norm", "c2.h_norm")
                bw.add_sd_linear(s2sd, "h_proj.0", "c2.h_proj")
            bw.add_sd_norm(s2sd, "aux_norm", "c2.aux_norm")
            bw.add_sd_linear(s2sd, "aux_proj.0", "c2.aux_proj.0", force_fp32=True)
            bw.add_sd_linear(s2sd, "aux_proj.3", "c2.aux_proj.3", force_fp32=True)
            bw.add_sd_linear(s2sd, "trunk.0", "c2.trunk.0", force_fp32=True)
            bw.add_sd_norm(s2sd, "trunk.1", "c2.trunk.1")
            bw.add_sd_linear(s2sd, "trunk.4", "c2.trunk.4", force_fp32=True)
            bw.add_sd_norm(s2sd, "trunk.5", "c2.trunk.5")
            bw.add_sd_linear(s2sd, "prop_mean_head", "c2.prop_mean", force_fp32=True)
            bw.add_sd_linear(s2sd, "prop_logvar_head", "c2.prop_logvar", force_fp32=True)
            bw.add_sd_linear(s2sd, "legal_head", "c2.legal", force_fp32=True)
            bw.add_sd_linear(s2sd, "constr_head", "c2.constr", force_fp32=True)

    ffn_dim = int(sd["backbone.decoder.layers.0.ffn.w2.base.weight"].shape[1]) if (
        "backbone.decoder.layers.0.ffn.w2.base.weight" in sd
    ) else int(sd["backbone.decoder.layers.0.ffn.w2.weight"].shape[1])

    header = {
        "format": "uni3dar_mol_rs_v1",
        "quant_dtype": dtype,
        "lora_merged_count": bw.lora_merged,
        "target_cols": target_cols,
        "target_units": target_units,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "backbone": {
            "emb_dim": emb_dim,
            "num_head": num_head,
            "head_dim": emb_dim // num_head,
            "num_layers": n_layers,
            "ffn_dim": ffn_dim,
            "merge_level": merge_level,
            "grid_len": 0.24,
            "xyz_resolution": 0.01,
            "num_xyz": 24,
            "xyz_null_id": 12,
            "num_atom_vocab": int(sd["backbone.atom_emb.weight"].shape[0]),
            "atom_cls_id": 342,
            "atom_unk_id": 343,
            "atom_mask_id": 344,
            "atom_null_id": 345,
            "tree0_split_id": 346,
            "rope_theta": 10000.0,
            "allowed_elements": list(rcfg.get("allowed_elements") or []),
            "non_trapping_budget": bool(lcfg.get("non_trapping_budget", True)),
            "has_budget_emb": has_budget_emb,
        },
        "conditioner": cond_cfg,
        "refiner": ref_cfg,
        "stage1_critic": s1_cfg,
        "stage2_critic": s2_cfg,
        "tensors": bw.tensors,
    }

    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pre_len = len(MAGIC) + 8 + len(hdr_bytes)
    pad_hdr = (ALIGN - (pre_len % ALIGN)) % ALIGN
    hdr_padded = hdr_bytes + (b" " * pad_hdr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(hdr_padded)))
        f.write(hdr_padded)
        for b in bw.blobs:
            f.write(b)

    size_mb = round(out_path.stat().st_size / 1e6, 2)
    return {
        "path": str(out_path),
        "size_mb": size_mb,
        "n_tensors": len(bw.tensors),
        "lora_merged": bw.lora_merged,
        "has_refiner": ref_cfg is not None,
        "has_stage1": s1_cfg is not None,
        "has_stage2": s2_cfg is not None,
    }


@torch.no_grad()
def dump_parity_fixture(ck_path: Path, out_json: Path) -> None:
    """Run exact PyTorch forward passes across Backbone (Step 0 + Step 1 KV), Refiner, and Stage1/Stage2 Critics."""
    from loom.weaving.uni3dar_slot import LoomUni3DARGenerator, compute_global_scale_free_budget_features
    from loom.weaving.post_tree_refiner import PostTreeGraphRefiner
    from loom.weaving.octree_process_critic import (
        OctreeElementProcessCritic,
        OctreeProcessCritic,
        compute_coarse_voxel_geometry,
        compute_element_cloud_features,
    )

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    loaded = LoomUni3DARGenerator.load_checkpoint(ck_path, device="cpu")
    gen = loaded[0] if isinstance(loaded, tuple) else loaded
    gen.eval()
    bb = gen.backbone
    bb.eval()

    k_props = len(gen.target_cols)
    torch.manual_seed(20261122)
    raw_targets = torch.tensor(ck["mean"], dtype=torch.float32).unsqueeze(0) + 0.75 * torch.tensor(
        ck["scale"], dtype=torch.float32
    ).unsqueeze(0)
    act_mask = torch.tensor([[i % 2 == 0 for i in range(k_props)]], dtype=torch.bool)
    if not bool(act_mask.any()):
        act_mask[0, 0] = True
    targets_norm = (raw_targets - gen.target_mean.cpu()) / gen.target_scale.cpu()

    global_c, has_any = gen.conditioner.encode_condition(targets_norm, act_mask)
    layer_mods, out_mod = gen.conditioner.precompute_modulations(global_c, torch.float32)
    root_delta = gen.conditioner.forward_root_prefix(global_c)
    prefix_tokens = gen.conditioner.build_prefix_tokens(global_c, has_any)

    merge_level = int(bb.args.merge_level)
    cls_phy = float((1 << merge_level) * 0.24 * 0.5)
    budget_atoms = 26

    # Step 0: [prefix_0..P-1, root_MASK]
    r_type = torch.tensor([[344]], dtype=torch.long)
    r_xyz = torch.tensor([[[12, 12, 12]]], dtype=torch.long)
    r_lvl = torch.tensor([[merge_level]], dtype=torch.long)
    r_ra = torch.tensor([[budget_atoms]], dtype=torch.long)
    emb0 = bb.atom_emb(r_type)
    for a in range(3):
        emb0 = emb0 + bb.xyz_emb[a](r_xyz[..., a])
    emb0 = emb0 + bb.cur_level(r_lvl + 1)
    if gen.budget_emb is not None:
        emb0 = emb0 + gen.budget_emb.net(compute_global_scale_free_budget_features(r_lvl, r_ra))
    emb0 = emb0 + root_delta.unsqueeze(1)

    p_cnt = int(prefix_tokens.shape[1]) if prefix_tokens is not None else 0
    if p_cnt > 0:
        x0 = torch.cat([prefix_tokens, emb0], dim=1)
        offs = (torch.arange(p_cnt, dtype=torch.float32) - float(p_cnt)) * 16.0 * 0.01
        p_pos = torch.full((1, p_cnt, 3), cls_phy, dtype=torch.float32) + offs.view(1, p_cnt, 1)
        r_pos = torch.full((1, 1, 3), cls_phy, dtype=torch.float32)
        pos0 = torch.cat([p_pos, r_pos], dim=1)
    else:
        x0 = emb0
        pos0 = torch.full((1, 1, 3), cls_phy, dtype=torch.float32)

    kv_caches = [{"k": None, "v": None, "step": 0} for _ in range(len(bb.decoder.layers))]
    bb.rope.calc_and_cache(pos0 / 0.01)
    h = x0
    for li, layer in enumerate(bb.decoder.layers):
        h = layer(h, kv_cache=kv_caches[li])
        h = h * layer_mods[li][0].unsqueeze(1) + layer_mods[li][1].unsqueeze(1)
    h0_out = bb.norm(h)
    h0_out = h0_out * out_mod[0].unsqueeze(1) + out_mod[1].unsqueeze(1)
    h_step0_last = h0_out[0, -1]
    tree_logits_0 = bb.tree_heads[0](h_step0_last.unsqueeze(0))[0]

    # Step 1: 2-token [prev_pred = TREE_19 (346+19=365) at merge_level, next_MASK (344) at merge_level-1]
    t1_type = torch.tensor([[365, 344]], dtype=torch.long)
    t1_xyz = torch.tensor([[[12, 12, 12], [12, 12, 12]]], dtype=torch.long)
    t1_lvl = torch.tensor([[merge_level, merge_level - 1]], dtype=torch.long)
    t1_ra = torch.tensor([[budget_atoms, budget_atoms]], dtype=torch.long)
    child_edge = float(1 << (merge_level - 1))
    pos1 = torch.tensor(
        [[[cls_phy, cls_phy, cls_phy], [0.5 * child_edge * 0.24, 0.5 * child_edge * 0.24, 0.5 * child_edge * 0.24]]],
        dtype=torch.float32,
    )
    emb1 = bb.atom_emb(t1_type)
    for a in range(3):
        emb1 = emb1 + bb.xyz_emb[a](t1_xyz[..., a])
    emb1 = emb1 + bb.cur_level(t1_lvl + 1)
    if gen.budget_emb is not None:
        emb1 = emb1 + gen.budget_emb.net(compute_global_scale_free_budget_features(t1_lvl, t1_ra))

    bb.rope.calc_and_cache(pos1 / 0.01)
    h1 = emb1
    for li, layer in enumerate(bb.decoder.layers):
        h1 = layer(h1, kv_cache=kv_caches[li])
        h1 = h1 * layer_mods[li][0].unsqueeze(1) + layer_mods[li][1].unsqueeze(1)
    h1_out = bb.norm(h1)
    h1_out = h1_out * out_mod[0].unsqueeze(1) + out_mod[1].unsqueeze(1)
    h_step1_last = h1_out[0, -1]
    tree_logits_1 = bb.tree_heads[0](h_step1_last.unsqueeze(0))[0]

    # Also test 4-axis AutoRegressivePredictionHead on `h_step1_last` with fixed (elem=5 'C', bx=7, by=15)
    ah = bb.atom_heads[0]
    af0 = h_step1_last.unsqueeze(0)
    atom_logits_0 = ah.pred_heads[0](af0)[0]
    af1 = af0 + ah.emb_layers[0](torch.tensor([5]))
    xyz_logits_x = ah.pred_heads[1](af1)[0]
    af2 = af1 + ah.emb_layers[1](torch.tensor([7]))
    xyz_logits_y = ah.pred_heads[2](af2)[0]
    af3 = af2 + ah.emb_layers[2](torch.tensor([15]))
    xyz_logits_z = ah.pred_heads[3](af3)[0]

    # Test PostTreeGraphRefiner + Stage 1 & Stage 2 Critics on a 6-atom formaldehyde-like test cloud
    test_elems = torch.tensor([[5, 7, 6, 0, 0, 0]], dtype=torch.long)  # C, O, N, H, H, H
    test_pos = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.22, 0.02, -0.01], [-0.68, 1.15, 0.03], [-0.55, -0.94, 0.0], [-1.65, 1.18, 0.12], [-0.18, 2.01, -0.05]]],
        dtype=torch.float32,
    )
    test_valid = torch.ones((1, 6), dtype=torch.bool)

    ref_out_dict = None
    if ck.get("post_tree_refiner_bundle") is not None:
        rb = ck["post_tree_refiner_bundle"]
        ref = PostTreeGraphRefiner(
            num_atom_classes=rb["num_atom_classes"],
            d_model=rb["d_model"],
            n_heads=rb["n_heads"],
            n_layers=rb["n_layers"],
            n_rbf=rb["n_rbf"],
            d_pair=rb["d_pair"],
            max_disp_A=rb["max_disp_A"],
        )
        ref.load_state_dict(rb["state_dict"])
        ref.eval()
        ro = ref(test_elems, test_pos, test_valid)
        ref_out_dict = {
            "refined_pos": ro["refined_pos"][0].tolist(),
            "bond_logits_01": ro["bond_logits"][0, 0, 1].tolist(),
            "bond_logits_02": ro["bond_logits"][0, 0, 2].tolist(),
        }

    c1_out_dict = None
    c2_out_dict = None
    if ck.get("process_critic_bundle") is not None:
        pcb = ck["process_critic_bundle"]
        # Synthetic coarse octree state for Stage 1 geometry + critic
        fake_feat = torch.tensor(
            [[[12, 12, 12, 344, 1, 0, 0, -1], [12, 12, 12, 344, 1, 0, 0, -1], [12, 12, 12, 344, 2, 0, 0, -1]]],
            dtype=torch.long,
        )
        fake_ppos = torch.tensor([[[1.2, 2.4, 3.6], [2.8, 1.9, 4.1], [2.0, 2.0, 3.8]]], dtype=torch.float32)
        fake_scores = torch.tensor([[-0.25, -0.40, -0.15]], dtype=torch.float32)
        g14 = compute_coarse_voxel_geometry(fake_feat, fake_ppos, fake_scores, atom_mask_id=344)

        s1c = pcb["stage1_config"]
        c1 = OctreeProcessCritic(
            emb_dim=s1c["emb_dim"], num_targets=s1c["num_targets"], hidden_dim=s1c["hidden_dim"], geom_dim=s1c["geom_dim"]
        )
        c1.load_state_dict(pcb["stage1_state_dict"])
        c1.eval()
        h_s1 = torch.cat([h_step0_last, h_step1_last, h_step1_last, h_step0_last], dim=-1).unsqueeze(0)
        o1 = c1(h_s1, g14, targets_norm, act_mask)
        u1 = c1.score_utilities(h_s1, g14, targets_norm, act_mask, legal_weight=0.80, constr_weight=0.55)
        c1_out_dict = {
            "geom_14": g14[0].tolist(),
            "mu_norm": o1["mu_norm"][0].tolist(),
            "logit_legal": float(o1["logit_legal"][0].item()),
            "logit_constr": float(o1["logit_constr"][0].item()),
            "utility": float(u1[0].item()),
        }

        s2c = dict(pcb["stage2_config"])
        s2_init_keys = {
            "num_h_pools", "d_node", "n_rbf", "n_gnn_layers", "hidden_dim", "cloud_feat_dim",
            "rbf_min", "rbf_max", "rbf_gamma", "edge_value_gate", "equivariant_multipole", "h_gate_scale",
        }
        c2 = OctreeElementProcessCritic(
            emb_dim=s2c["emb_dim"],
            num_targets=s2c["num_targets"],
            num_atom_classes=s2c.get("num_atom_classes", 602),
            **{k: v for k, v in s2c.items() if k in s2_init_keys},
        )
        c2.load_state_dict(pcb["stage2_state_dict"])
        c2.eval()
        cloud42 = compute_element_cloud_features(test_elems, test_pos, test_valid, g14)
        h_s2 = torch.cat([h_step0_last, h_step1_last, h_step1_last, h_step0_last, h_step1_last, h_step0_last], dim=-1).unsqueeze(0)
        o2 = c2(h_s2, test_elems, test_pos, test_valid, cloud42, targets_norm, act_mask)
        u2 = c2.score_utilities(
            h_s2, test_elems, test_pos, test_valid, cloud42, targets_norm, act_mask, legal_weight=1.05, constr_weight=0.70
        )
        c2_out_dict = {
            "cloud_42": cloud42[0].tolist(),
            "mu_norm": o2["mu_norm"][0].tolist(),
            "logit_legal": float(o2["logit_legal"][0].item()),
            "logit_constr": float(o2["logit_constr"][0].item()),
            "utility": float(u2[0].item()),
        }

    fixture = {
        "raw_targets": raw_targets[0].tolist(),
        "active_mask": [bool(x) for x in act_mask[0].tolist()],
        "budget_atoms": budget_atoms,
        "step0_last_hidden": h_step0_last.tolist(),
        "step0_tree_logits": tree_logits_0.tolist(),
        "step1_prev_type": 365,
        "step1_last_hidden": h_step1_last.tolist(),
        "step1_tree_logits": tree_logits_1.tolist(),
        "atom_logits_0": atom_logits_0.tolist(),
        "xyz_logits_x": xyz_logits_x.tolist(),
        "xyz_logits_y": xyz_logits_y.tolist(),
        "xyz_logits_z": xyz_logits_z.tolist(),
        "test_elems": test_elems[0].tolist(),
        "test_pos": test_pos[0].tolist(),
        "refiner_out": ref_out_dict,
        "stage1_out": c1_out_dict,
        "stage2_out": c2_out_dict,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(fixture))


def main() -> None:
    p = argparse.ArgumentParser(description="Export Uni-3DAR-Mol bundled .pt checkpoint to .u3m for uni3dar-rs.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", choices=("fp32", "fp16", "q8_0", "q4_0"), default="q8_0")
    p.add_argument("--dump-parity", type=Path, default=None)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    info = export_uni3dar_bundle(ck, args.out, dtype=args.dtype)
    print(
        f"[export_uni3dar_rs] Exported {info['path']} ({info['size_mb']} MB, {info['n_tensors']} tensors, "
        f"lora_merged={info['lora_merged']}, dtype={args.dtype})"
    )
    if args.dump_parity is not None:
        dump_parity_fixture(args.checkpoint, args.dump_parity)
        print(f"[export_uni3dar_rs] Wrote ground-truth parity fixture -> {args.dump_parity}")


if __name__ == "__main__":
    main()
