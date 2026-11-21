#!/usr/bin/env python3
"""Export FlowMol-118 (.pt) checkpoints (tiny/small/standard/flagship + SFT/LoRA/GRPO/Conditioner)
into the compact, zero-copy-aligned `.fm118` inference bundle for `flowmol-rs` (Desktop + WebGPU).

Key optimizations baked in at export time (zero retraining, bit-exact algebra):
  1. Automatic `ToggleableLoRALinear` merging (`W + (alpha/rank) * B @ A`).
  2. Algebraic Node-Before-Gather weight splitting on every `_MoleculeUpdateBlock`:
     - `GVPConv.edge_message[0]`: splits node-scalar `W_h_src (S, S)` and node-vector `Wh_node / Wcp_node`
       away from per-edge `(rbf, e, sh)` so node projections run in `O(N)` instead of `O(N^2)` (-34% FLOPs!).
     - `EdgeUpdate.edge_update_fn[0]`: splits `W_src (E_dim, S)` and `W_dst (E_dim, S)` away from `W_er`.
  3. Precision-guarded quantization (`--dtype fp32 | fp16 | q8_0`):
     - Heavy `Linear` weight matrices (~99% of parameters) are stored in FP32, FP16 (2-aligned), or
       block-32 symmetric INT8 (`Q8_0`, 32-aligned: 1 fp32 scale + 32 int8 per block).
     - All geometric equivariant projections (`Wh`, `Wcp`, `Wu`), biases, `LayerNorm` weights/biases,
       `periodic_table_feats`, `sc_vec_scale`, `fragment_cond_proj`, and optional `LoomConditioner`
       are strictly preserved in FP32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from loom.flowmol import FlowMolConfig, FlowMolModel
from loom.flowmol.data import build_complete_graph_batch
from loom.flowmol.gvp import ToggleableLoRALinear

MAGIC = b"FM118\x01\x00\x00"
ALIGN = 16


def _merge_all_lora(model: torch.nn.Module) -> int:
    merged = 0
    for _, module in model.named_modules():
        for name, child in list(module.named_children()):
            if isinstance(child, ToggleableLoRALinear):
                setattr(module, name, child.merge_into_base())
                merged += 1
    return merged


def _pack_tensor_f32(arr: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    raw = a.tobytes()
    return raw, {"dtype": "f32", "shape": list(a.shape), "orig_cols": a.shape[-1] if a.ndim >= 1 else 1}


def _pack_linear_weight(arr_2d: np.ndarray, mode: str) -> Tuple[bytes, Dict[str, Any]]:
    """Pack a `(out_features, in_features)` matrix into `f32`, `f16` (pad2), or `q8_0` (block-32)."""
    assert arr_2d.ndim == 2, f"Expected 2D linear weight, got {arr_2d.shape}"
    out_f, in_f = int(arr_2d.shape[0]), int(arr_2d.shape[1])
    w = np.ascontiguousarray(arr_2d, dtype=np.float32)

    if mode == "fp32":
        return w.tobytes(), {"dtype": "f32", "shape": [out_f, in_f], "orig_cols": in_f}

    if mode == "fp16":
        pad_in = (in_f + 1) & ~1
        if pad_in != in_f:
            wp = np.zeros((out_f, pad_in), dtype=np.float32)
            wp[:, :in_f] = w
            w = wp
        w16 = np.ascontiguousarray(w, dtype=np.float16)
        return w16.tobytes(), {"dtype": "f16", "shape": [out_f, pad_in], "orig_cols": in_f}

    if mode == "q8_0":
        pad_in = (in_f + 31) & ~31
        if pad_in != in_f:
            wp = np.zeros((out_f, pad_in), dtype=np.float32)
            wp[:, :in_f] = w
            w = wp
        n_blocks_per_row = pad_in // 32
        blocks = w.reshape(out_f * n_blocks_per_row, 32)
        max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.round(blocks / scales), -127, 127).astype(np.int8)
        # Layout per block (36 bytes, 4-byte aligned!): f32 scale (4B) + 32 x i8 (32B = 8 x u32)
        rec = np.empty(
            out_f * n_blocks_per_row,
            dtype=np.dtype([("scale", "<f4"), ("qs", "i1", (32,))]),
        )
        rec["scale"] = scales[:, 0]
        rec["qs"] = q
        return rec.tobytes(), {
            "dtype": "q8_0",
            "shape": [out_f, pad_in],
            "orig_cols": in_f,
            "blocks_per_row": n_blocks_per_row,
        }

    raise ValueError(f"Unsupported quantization mode: {mode}")


class BundleWriter:
    def __init__(self, quant_mode: str) -> None:
        self.quant_mode = quant_mode
        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.blobs: List[bytes] = []
        self.offset = 0

    def _append_blob(self, name: str, raw: bytes, meta: Dict[str, Any]) -> None:
        pad = (ALIGN - (self.offset % ALIGN)) % ALIGN
        if pad:
            self.blobs.append(b"\x00" * pad)
            self.offset += pad
        meta_full = dict(meta)
        meta_full["offset"] = self.offset
        meta_full["nbytes"] = len(raw)
        self.tensors[name] = meta_full
        self.blobs.append(raw)
        self.offset += len(raw)

    def add_f32(self, name: str, t: torch.Tensor | np.ndarray) -> None:
        arr = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)
        raw, meta = _pack_tensor_f32(arr)
        self._append_blob(name, raw, meta)

    def add_linear(self, prefix: str, lin: torch.nn.Linear, *, force_fp32: bool = False) -> None:
        w = lin.weight.detach().cpu().float().numpy()
        mode = "fp32" if force_fp32 else self.quant_mode
        raw, meta = _pack_linear_weight(w, mode)
        self._append_blob(f"{prefix}.weight", raw, meta)
        if lin.bias is not None:
            self.add_f32(f"{prefix}.bias", lin.bias)

    def add_layernorm(self, prefix: str, ln: torch.nn.LayerNorm) -> None:
        self.add_f32(f"{prefix}.weight", ln.weight)
        self.add_f32(f"{prefix}.bias", ln.bias)

    def add_gvp(self, prefix: str, gvp: torch.nn.Module) -> None:
        self.add_f32(f"{prefix}.Wh", gvp.Wh)
        if gvp.Wcp is not None:
            self.add_f32(f"{prefix}.Wcp", gvp.Wcp)
        self.add_f32(f"{prefix}.Wu", gvp.Wu)
        self.add_linear(f"{prefix}.to_feats_out", gvp.to_feats_out[0])
        if gvp.scalar_to_vector_gates is not None:
            self.add_linear(f"{prefix}.vec_gate", gvp.scalar_to_vector_gates)


def export_model_bundle(
    model: FlowMolModel,
    ck_meta: Dict[str, Any],
    out_path: Path,
    *,
    dtype: str = "fp16",
) -> Dict[str, Any]:
    model = model.cpu().eval()
    n_merged = _merge_all_lora(model)
    cfg = model.config
    S = cfg.n_hidden_scalars
    E_dim = cfg.n_hidden_edge_feats
    V = cfg.n_vec_channels

    bw = BundleWriter(dtype)

    # Top-level invariant buffers & embeddings
    bw.add_f32("periodic_table_feats", model.periodic_table_feats)
    if model.sc_vec_scale is not None:
        bw.add_f32("sc_vec_scale", model.sc_vec_scale.view(1))

    bw.add_linear("scalar_embedding.0", model.scalar_embedding[0])
    bw.add_linear("scalar_embedding.2", model.scalar_embedding[2])
    bw.add_layernorm("scalar_embedding.4", model.scalar_embedding[4])
    bw.add_linear("fragment_cond_proj", model.fragment_cond_proj, force_fp32=True)

    bw.add_linear("edge_embedding.0", model.edge_embedding[0])
    bw.add_linear("edge_embedding.2", model.edge_embedding[2])
    bw.add_layernorm("edge_embedding.4", model.edge_embedding[4])

    # Molecule update blocks with Algebraic Node-Before-Gather pre-split weights
    for b_idx, blk in enumerate(model.blocks):
        bp = f"blocks.{b_idx}"
        for c_idx, conv in enumerate(blk.convs):
            cp = f"{bp}.convs.{c_idx}"
            # Split edge_message[0] into node-preprojected + edge-residual parts!
            gvp0 = conv.edge_message[0]
            # Vector split: row 0 = unit_diff, rows 1..1+V = src_vec (and optional 1+V..1+2V = dst_vec)
            bw.add_f32(f"{cp}.em0.Wh_unit", gvp0.Wh[0:1, :])
            bw.add_f32(f"{cp}.em0.Wh_src", gvp0.Wh[1 : 1 + V, :])
            if conv.use_dst_feats:
                bw.add_f32(f"{cp}.em0.Wh_dst", gvp0.Wh[1 + V : 1 + 2 * V, :])
            if gvp0.Wcp is not None:
                bw.add_f32(f"{cp}.em0.Wcp_unit", gvp0.Wcp[0:1, :])
                bw.add_f32(f"{cp}.em0.Wcp_src", gvp0.Wcp[1 : 1 + V, :])
                if conv.use_dst_feats:
                    bw.add_f32(f"{cp}.em0.Wcp_dst", gvp0.Wcp[1 + V : 1 + 2 * V, :])
            bw.add_f32(f"{cp}.em0.Wu", gvp0.Wu)

            # Scalar linear split: [h_src (S), rbf (R), e (E_dim), (opt h_dst (S)), sh (wu_in)]
            lin0: torch.nn.Linear = gvp0.to_feats_out[0]
            w_full = lin0.weight.detach().cpu().float()
            # Build clean temporary Linears for uniform quantization packing
            def _pack_slice(name: str, w_sub: torch.Tensor, b_sub: Optional[torch.Tensor] = None) -> None:
                raw, meta = _pack_linear_weight(w_sub.numpy(), dtype)
                bw._append_blob(f"{name}.weight", raw, meta)
                if b_sub is not None:
                    bw.add_f32(f"{name}.bias", b_sub)

            _pack_slice(f"{cp}.em0.s_src", w_full[:, :S])
            R = cfg.rbf_dim
            if conv.use_dst_feats:
                er_end = S + R + E_dim
                dst_end = er_end + S
                w_edge_sh = torch.cat([w_full[:, S:er_end], w_full[:, dst_end:]], dim=1)
                _pack_slice(f"{cp}.em0.s_dst", w_full[:, er_end:dst_end])
                _pack_slice(f"{cp}.em0.s_edge_sh", w_edge_sh, lin0.bias)
            else:
                _pack_slice(f"{cp}.em0.s_edge_sh", w_full[:, S:], lin0.bias)

            if gvp0.scalar_to_vector_gates is not None:
                bw.add_linear(f"{cp}.em0.vec_gate", gvp0.scalar_to_vector_gates)

            # Remaining edge_message GVPs (1..n_message_gvps)
            for m_idx in range(1, len(conv.edge_message)):
                bw.add_gvp(f"{cp}.em.{m_idx}", conv.edge_message[m_idx])

            bw.add_layernorm(f"{cp}.msg_ln", conv.message_layer_norm.feat_norm)
            for u_idx, ugvp in enumerate(conv.node_update):
                bw.add_gvp(f"{cp}.nu.{u_idx}", ugvp)
            bw.add_layernorm(f"{cp}.upd_ln", conv.update_layer_norm.feat_norm)

        # Position updater GVPs
        for p_idx, pgvp in enumerate(blk.pos_updater.gvps):
            bw.add_gvp(f"{bp}.pu.{p_idx}", pgvp)

        # Edge updater with pre-split W_src (E_dim, S), W_dst (E_dim, S), W_er (E_dim, E_dim + R)
        eu = blk.edge_updater
        eu0: torch.nn.Linear = eu.edge_update_fn[0]
        ew = eu0.weight.detach().cpu().float()
        raw_s, ms = _pack_linear_weight(ew[:, :S].numpy(), dtype)
        bw._append_blob(f"{bp}.eu.w_src.weight", raw_s, ms)
        raw_d, md = _pack_linear_weight(ew[:, S : 2 * S].numpy(), dtype)
        bw._append_blob(f"{bp}.eu.w_dst.weight", raw_d, md)
        raw_er, mer = _pack_linear_weight(ew[:, 2 * S :].numpy(), dtype)
        bw._append_blob(f"{bp}.eu.w_er.weight", raw_er, mer)
        bw.add_f32(f"{bp}.eu.w_er.bias", eu0.bias)
        bw.add_linear(f"{bp}.eu.lin1", eu.edge_update_fn[2])
        bw.add_layernorm(f"{bp}.eu.ln", eu.edge_norm)

    # Output heads
    bw.add_linear("node_output_head.0", model.node_output_head[0])
    bw.add_linear("node_output_head.2", model.node_output_head[2], force_fp32=True)
    if model.periodic_logit_proj is not None:
        bw.add_linear("periodic_logit_proj", model.periodic_logit_proj, force_fp32=True)
    bw.add_linear("to_edge_logits.0", model.to_edge_logits[0])
    bw.add_linear("to_edge_logits.2", model.to_edge_logits[2], force_fp32=True)

    # Optional property conditioner (kept strictly in FP32 for exact guidance fidelity)
    cond_cfg: Optional[Dict[str, Any]] = None
    if model.conditioner is not None:
        c = model.conditioner
        cond_cfg = {
            "emb_dim": int(c.emb_dim),
            "num_props": int(c.num_props),
            "num_layers": int(c.num_layers),
            "num_rbf": int(c.num_rbf),
            "num_freqs": int(c.num_freqs),
            "rbf_gamma": float(c.rbf_gamma),
            "modulation_cap": float(c.modulation_cap),
        }
        bw.add_f32("cond.rbf_centers", c.rbf_centers)
        bw.add_f32("cond.fourier_freqs", c.fourier_freqs)
        bw.add_f32("cond.prop_id_emb", c.prop_id_emb.weight)
        bw.add_f32("cond.inactive_prop_emb", c.inactive_prop_emb)
        for k, enc in enumerate(c.prop_encoders):
            bw.add_linear(f"cond.prop_enc.{k}.0", enc[0], force_fp32=True)
            bw.add_layernorm(f"cond.prop_enc.{k}.1", enc[1])
            bw.add_linear(f"cond.prop_enc.{k}.3", enc[3], force_fp32=True)
            bw.add_layernorm(f"cond.prop_enc.{k}.4", enc[4])
        bw.add_linear("cond.global_mlp.0", c.global_mlp[0], force_fp32=True)
        bw.add_layernorm("cond.global_mlp.1", c.global_mlp[1])
        bw.add_linear("cond.global_mlp.3", c.global_mlp[3], force_fp32=True)
        bw.add_layernorm("cond.global_mlp.4", c.global_mlp[4])
        for l_idx, lmod in enumerate(c.layer_mods):
            bw.add_linear(f"cond.layer_mod.{l_idx}", lmod[1], force_fp32=True)
        bw.add_linear("cond.out_mod", c.out_mod[1], force_fp32=True)

    header = {
        "format": "flowmol_rs_v1",
        "quant_dtype": dtype,
        "lora_merged_count": n_merged,
        "step": int(ck_meta.get("step", 0)),
        "samples_seen": int(ck_meta.get("samples_seen", 0)),
        "config": cfg.to_dict(),
        "conditioner": cond_cfg,
        "active_elements": list(ck_meta.get("active_elements", [])),
        "active_charges": [int(x) for x in ck_meta.get("active_charges", [])],
        "n_atoms_hist": {str(k): int(v) for k, v in (ck_meta.get("n_atoms_hist") or {}).items()},
        "per_source_n_atoms_hist": {
            str(dom): {str(k): int(v) for k, v in (h or {}).items()}
            for dom, h in (ck_meta.get("per_source_n_atoms_hist") or {}).items()
        },
        "per_source_active_elements": {
            str(dom): list(elems)
            for dom, elems in (ck_meta.get("per_source_active_elements") or {}).items()
        },
        "tensors": bw.tensors,
    }

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pre_len = len(MAGIC) + 8 + len(header_bytes)
    pad_hdr = (ALIGN - (pre_len % ALIGN)) % ALIGN
    header_padded = header_bytes + (b" " * pad_hdr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header_padded)))
        f.write(header_padded)
        for blob in bw.blobs:
            f.write(blob)

    total_bytes = out_path.stat().st_size
    return {
        "path": str(out_path),
        "size_mb": round(total_bytes / 1e6, 2),
        "n_tensors": len(bw.tensors),
        "lora_merged": n_merged,
        "has_conditioner": cond_cfg is not None,
    }


@torch.no_grad()
def dump_parity_fixture(model: FlowMolModel, out_json: Path, *, n_atoms: int = 12, seed: int = 12345) -> None:
    """Export a deterministic single-molecule forward-pass parity fixture (including active self-conditioning!)."""
    model = model.cpu().eval()
    torch.manual_seed(seed)
    batch = build_complete_graph_batch(
        [n_atoms], torch.device("cpu"), prior_std=model.config.prior_std, size_scale_prior=model.config.size_scale_prior
    )
    n_up = int(batch.upper_edge_mask.sum().item())
    x_t = batch.x_0.clone()
    a_t = torch.randint(0, model.n_atom_types + 1, (n_atoms,), dtype=torch.long)
    c_t = torch.randint(0, model.n_charges + 1, (n_atoms,), dtype=torch.long)
    e_t = torch.randint(0, model.n_bond_types + 1, (n_up,), dtype=torch.long)
    t1 = torch.tensor([0.37], dtype=torch.float32)

    cond_values: Optional[torch.Tensor] = None
    cond_active: Optional[torch.Tensor] = None
    node_cond_flags: Optional[torch.Tensor] = None
    if model.conditioner is not None:
        num_props = int(model.conditioner.num_props)
        cond_values = torch.randn((1, num_props), dtype=torch.float32) * 1.5
        cond_active = (torch.arange(num_props) % 2 == 0).unsqueeze(0)
        node_cond_flags = torch.zeros((n_atoms, 2), dtype=torch.float32)
        node_cond_flags[: min(3, n_atoms), 0] = 1.0
        node_cond_flags[: min(3, n_atoms), 1] = 0.75

    # Pass 1: without self-conditioning
    out1 = model.forward(
        batch,
        x_t,
        a_t,
        c_t,
        e_t,
        t1,
        sc_dict=None,
        cond_values=cond_values,
        cond_active_mask=cond_active,
        node_cond_flags=node_cond_flags,
        remove_com=True,
    )
    sc_dict = {
        "x": out1["x"],
        "a": torch.softmax(out1["a"], dim=-1),
        "c": torch.softmax(out1["c"], dim=-1),
        "e": torch.softmax(out1["e"], dim=-1),
    }
    # Pass 2: with full self-conditioning active at t=0.68
    t2 = torch.tensor([0.68], dtype=torch.float32)
    out2 = model.forward(
        batch,
        x_t,
        a_t,
        c_t,
        e_t,
        t2,
        sc_dict=sc_dict,
        cond_values=cond_values,
        cond_active_mask=cond_active,
        node_cond_flags=node_cond_flags,
        remove_com=True,
    )

    fixture = {
        "n_atoms": n_atoms,
        "x_t": x_t.tolist(),
        "a_t": a_t.tolist(),
        "c_t": c_t.tolist(),
        "e_t_upper": e_t.tolist(),
        "cond_values": cond_values[0].tolist() if cond_values is not None else None,
        "cond_active": [bool(x) for x in cond_active[0].tolist()] if cond_active is not None else None,
        "node_cond_flags": node_cond_flags.tolist() if node_cond_flags is not None else None,
        "t1": float(t1.item()),
        "out1": {k: v.tolist() for k, v in out1.items()},
        "t2": float(t2.item()),
        "out2": {k: v.tolist() for k, v in out2.items()},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(fixture))


def main() -> None:
    p = argparse.ArgumentParser(description="Export FlowMol-118 checkpoint to .fm118 bundle for flowmol-rs.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Input .pt checkpoint path")
    p.add_argument("--corpus-meta", type=Path, default=None, help="Optional corpus_meta.json fallback for histograms")
    p.add_argument("--out", type=Path, required=True, help="Output .fm118 bundle path")
    p.add_argument("--dtype", choices=("fp32", "fp16", "q8_0"), default="fp16")
    p.add_argument("--no-ema", action="store_true", help="Export raw model_state_dict instead of ema_state_dict")
    p.add_argument("--dump-parity", type=Path, default=None, help="Optional output JSON fixture for parity testing")
    p.add_argument("--parity-atoms", type=int, default=14)
    args = p.parse_args()

    model, ck_meta = FlowMolModel.from_pretrained(args.checkpoint, device="cpu", use_ema=not args.no_ema, strict=True)
    if args.corpus_meta is not None and args.corpus_meta.exists():
        cmeta = json.loads(args.corpus_meta.read_text())
        for k in ("active_elements", "active_charges", "n_atoms_hist", "per_source_n_atoms_hist", "per_source_active_elements"):
            if not ck_meta.get(k) and cmeta.get(k):
                ck_meta[k] = cmeta[k]
        if not ck_meta.get("active_elements") and cmeta.get("element_counts"):
            ck_meta["active_elements"] = [el for el, cnt in cmeta["element_counts"].items() if int(cnt) > 0]
        if not ck_meta.get("active_charges") and cmeta.get("charge_counts"):
            ck_meta["active_charges"] = [int(ch) for ch, cnt in cmeta["charge_counts"].items() if int(cnt) > 0]

    info = export_model_bundle(model, ck_meta, args.out, dtype=args.dtype)
    print(f"[export_flowmol_rs] Exported {info['path']} ({info['size_mb']} MB, {info['n_tensors']} tensors, dtype={args.dtype})")

    if args.dump_parity is not None:
        dump_parity_fixture(model, args.dump_parity, n_atoms=args.parity_atoms)
        print(f"[export_flowmol_rs] Wrote PyTorch ground-truth parity fixture -> {args.dump_parity}")


if __name__ == "__main__":
    main()
