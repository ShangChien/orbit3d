#!/usr/bin/env python3
"""
Export DeePMD DPA-4 (`SeZM`: `Nano`, `Mini`, `Neo`, `Air`, `Plus`) `.pt` checkpoints into
compact, 64-byte-aligned `.dpa4` (`DPA4RS01`) bundles for `dpa4-rs` (Desktop Metal/Vulkan/DX12 +
Browser WebGPU WASM), bundling precomputed Wigner-D quaternion monomial matrices, SO(3)
Lebedev-Haar grid projectors, FP64 atomic energy offsets (`out_bias`), and optional
`_BootstrapAdapters` 3D property oracle heads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch

MAGIC = b"DPA4RS01"
VERSION = 1


def align64(n: int) -> int:
    return (n + 63) & ~63


class BundleBuilder:
    def __init__(self, config: dict, extra: dict | None = None):
        self.config = config
        self.extra = extra or {}
        self.tensors: List[Tuple[str, str, List[int], bytes]] = []

    def add_tensor(self, name: str, arr: np.ndarray, dtype: str = "f32") -> None:
        arr = np.ascontiguousarray(arr)
        if dtype == "f32":
            raw = arr.astype("<f4", copy=False).tobytes()
        elif dtype == "f16":
            raw = arr.astype("<f2", copy=False).tobytes()
        elif dtype == "f64":
            raw = arr.astype("<f8", copy=False).tobytes()
        elif dtype == "i32":
            raw = arr.astype("<i4", copy=False).tobytes()
        else:
            raise ValueError(f"Unsupported bundle dtype: {dtype}")
        self.tensors.append((name, dtype, list(arr.shape), raw))

    def build_bytes(self) -> bytes:
        metas = []
        cur_off = 0
        for name, dtype, shape, raw in self.tensors:
            cur_off = align64(cur_off)
            metas.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "offset": cur_off,
                    "nbytes": len(raw),
                }
            )
            cur_off += len(raw)

        header_json = json.dumps(
            {
                "config": self.config,
                "tensors": metas,
                "extra": self.extra,
            },
            separators=(",", ":"),
        ).encode("utf-8")

        prefix_len = 16 + len(header_json)
        data_start = align64(prefix_len)
        pad_header = b"\x00" * (data_start - prefix_len)

        chunks = [
            MAGIC,
            struct.pack("<II", VERSION, len(header_json)),
            header_json,
            pad_header,
        ]
        written = 0
        for meta, (_, _, _, raw) in zip(metas, self.tensors):
            if meta["offset"] > written:
                chunks.append(b"\x00" * (meta["offset"] - written))
                written = meta["offset"]
            chunks.append(raw)
            written += len(raw)
        return b"".join(chunks)


def serialize_scaler(sc) -> dict:
    if sc is None:
        return {"kind": "identity"}
    inner = getattr(sc, "scaler", sc)
    if isinstance(inner, list) and len(inner) == 1:
        inner = inner[0]
    if inner is None:
        return {"kind": "identity"}
    cls_name = type(inner).__name__
    if cls_name == "StandardScaler":
        mean = float(np.asarray(inner.mean_).reshape(-1)[0]) if getattr(inner, "with_mean", True) else 0.0
        scale = float(np.asarray(inner.scale_).reshape(-1)[0]) if getattr(inner, "with_std", True) else 1.0
        return {"kind": "affine", "mean": mean, "scale": scale}
    if cls_name == "RobustScaler":
        center = float(np.asarray(inner.center_).reshape(-1)[0]) if inner.with_centering else 0.0
        scale = float(np.asarray(inner.scale_).reshape(-1)[0]) if inner.with_scaling else 1.0
        return {"kind": "affine", "mean": center, "scale": scale}
    if "PowerTransformer" in cls_name and hasattr(inner, "lambdas_"):
        lmbda = float(np.asarray(inner.lambdas_).reshape(-1)[0])
        standardize = bool(getattr(inner, "standardize", True))
        mean = float(np.asarray(inner._scaler.mean_).reshape(-1)[0]) if standardize else 0.0
        scale = float(np.asarray(inner._scaler.scale_).reshape(-1)[0]) if standardize else 1.0
        return {
            "kind": "yeo_johnson",
            "lambda": lmbda,
            "mean": mean,
            "scale": scale,
            "standardize": standardize,
        }
    raise ValueError(f"Unsupported oracle scaler type: {cls_name}")


def attach_property_oracle(builder: BundleBuilder, oracle_dir: Path) -> dict:
    meta_path = oracle_dir / "oracle_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    adapter_sd = torch.load(oracle_dir / "adapter.pt", map_location="cpu", weights_only=True)
    scalers_path = oracle_dir / "scalers.joblib"
    scalers = joblib.load(scalers_path) if scalers_path.exists() else []

    # Detect member count & dimensions from state_dict
    member_ids = sorted({int(k.split(".")[1]) for k in adapter_sd if k.startswith("members.")})
    in_w = adapter_sd["members.0.in_proj.weight"]
    hidden_dim, desc_dim = int(in_w.shape[0]), int(in_w.shape[1])
    out_w = adapter_sd["members.0.heads.0.5.weight"]
    output_dim = int(out_w.shape[0])

    for m in member_ids:
        mp = f"members.{m}"
        op = f"oracle.m{m}"
        builder.add_tensor(f"{op}.rbf_centers", adapter_sd[f"{mp}.rbf_centers"].numpy(), "f32")
        for layer in ("in_proj", "dist_rbf", "qkv", "out_proj", "norm1", "norm2"):
            builder.add_tensor(f"{op}.{layer}.weight", adapter_sd[f"{mp}.{layer}.weight"].numpy(), "f32")
            builder.add_tensor(f"{op}.{layer}.bias", adapter_sd[f"{mp}.{layer}.bias"].numpy(), "f32")
        builder.add_tensor(f"{op}.ffn0.weight", adapter_sd[f"{mp}.ffn.0.weight"].numpy(), "f32")
        builder.add_tensor(f"{op}.ffn0.bias", adapter_sd[f"{mp}.ffn.0.bias"].numpy(), "f32")
        builder.add_tensor(f"{op}.ffn1.weight", adapter_sd[f"{mp}.ffn.2.weight"].numpy(), "f32")
        builder.add_tensor(f"{op}.ffn1.bias", adapter_sd[f"{mp}.ffn.2.bias"].numpy(), "f32")

        head_ids = sorted(
            {int(k.split(".")[3]) for k in adapter_sd if k.startswith(f"{mp}.gates.")}
        )
        for hd in head_ids:
            hp = f"{op}.h{hd}"
            builder.add_tensor(f"{hp}.gate.weight", adapter_sd[f"{mp}.gates.{hd}.weight"].numpy(), "f32")
            builder.add_tensor(f"{hp}.gate.bias", adapter_sd[f"{mp}.gates.{hd}.bias"].numpy(), "f32")
            builder.add_tensor(f"{hp}.norm.weight", adapter_sd[f"{mp}.heads.{hd}.0.weight"].numpy(), "f32")
            builder.add_tensor(f"{hp}.norm.bias", adapter_sd[f"{mp}.heads.{hd}.0.bias"].numpy(), "f32")
            builder.add_tensor(f"{hp}.fc0.weight", adapter_sd[f"{mp}.heads.{hd}.1.weight"].numpy(), "f32")
            builder.add_tensor(f"{hp}.fc0.bias", adapter_sd[f"{mp}.heads.{hd}.1.bias"].numpy(), "f32")
            builder.add_tensor(f"{hp}.fc1.weight", adapter_sd[f"{mp}.heads.{hd}.3.weight"].numpy(), "f32")
            builder.add_tensor(f"{hp}.fc1.bias", adapter_sd[f"{mp}.heads.{hd}.3.bias"].numpy(), "f32")
            builder.add_tensor(f"{hp}.fc2.weight", adapter_sd[f"{mp}.heads.{hd}.5.weight"].numpy(), "f32")
            builder.add_tensor(f"{hp}.fc2.bias", adapter_sd[f"{mp}.heads.{hd}.5.bias"].numpy(), "f32")

    n_heads_per_member = len(
        {int(k.split(".")[3]) for k in adapter_sd if k.startswith("members.0.gates.")}
    )
    return {
        "desc_dim": desc_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "n_members": len(member_ids),
        "n_heads_per_member": n_heads_per_member,
        "target_names": meta.get("target_cols", [f"TARGET_{i}" for i in range(output_dim)]),
        "target_units": meta.get("target_units", [""] * output_dim),
        "scalers": [serialize_scaler(s) for s in scalers] if scalers else [{"kind": "identity"}] * output_dim,
    }


def export_dpa4_checkpoint(
    tier: str,
    pt_path: Path,
    constants: Dict[str, np.ndarray],
    out_path: Path,
    head: str = "Default",
    weight_dtype: str = "f32",
    oracle_dir: Optional[Path] = None,
) -> None:
    raw = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    sd_full = raw["model"] if "model" in raw else raw
    mp_root = sd_full["_extra_state"]["model_params"]

    if "model_dict" in mp_root:
        mp = mp_root["model_dict"][head]
        prefix = f"model.{head}."
    else:
        mp = mp_root
        prefix = "model.Default."

    desc_cfg = mp["descriptor"]
    fit_cfg = mp["fitting_net"]
    type_map = list(mp["type_map"])
    ntypes = len(type_map)
    channels = int(desc_cfg["channels"])
    lmax = int(desc_cfg["lmax"])
    mmax = int(desc_cfg.get("mmax", 1))
    kmax = int(desc_cfg.get("kmax", 1))
    n_blocks = int(desc_cfg["n_blocks"])
    mixing_layers = int(desc_cfg.get("mixing_layers", desc_cfg.get("so2_layers", 3)))
    n_focus = int(desc_cfg.get("n_focus", 1))
    radial_so2_mode = str(desc_cfg.get("radial_so2_mode", "degree_channel")).lower()
    radial_so2_rank = int(desc_cfg.get("radial_so2_rank", 1))
    ffn_raw = int(desc_cfg.get("ffn_neurons", 0))
    ffn_dim = int(32 * math.ceil((8.0 * float(channels) / 3.0) / 32.0)) if ffn_raw == 0 else ffn_raw

    to_grid = constants[f"lmax{lmax}_to_grid_mat"]
    from_grid = constants[f"lmax{lmax}_from_grid_mat"]
    n_grid = int(to_grid.shape[0])

    def get_t(suffix: str) -> np.ndarray:
        return sd_full[prefix + suffix].detach().cpu().numpy()

    version_val = float(get_t("atomic_model.descriptor.version_tensor").reshape(-1)[0])

    bundle_cfg = {
        "tier": tier,
        "type_map": type_map,
        "ntypes": ntypes,
        "rcut": float(desc_cfg.get("rcut", 6.0)),
        "env_exp": [int(x) for x in desc_cfg.get("env_exp", [7, 5])],
        "n_radial": int(desc_cfg.get("n_radial", 16)),
        "channels": channels,
        "lmax": lmax,
        "mmax": mmax,
        "kmax": kmax,
        "n_blocks": n_blocks,
        "mixing_layers": mixing_layers,
        "n_focus": n_focus,
        "radial_so2_mode": radial_so2_mode,
        "radial_so2_rank": radial_so2_rank,
        "ffn_dim": ffn_dim,
        "n_grid": n_grid,
        "eps": float(desc_cfg.get("eps", 1e-7)),
        "version": version_val,
        "deg_norm_floor": 0.25 if version_val >= 1.1 else float(desc_cfg.get("eps", 1e-7)),
        "default_chg_spin": [float(x) for x in desc_cfg.get("default_chg_spin", [0.0, 1.0])],
        "weight_dtype": weight_dtype,
    }

    extra: dict = {}
    builder = BundleBuilder(bundle_cfg, extra=extra)

    # 1. Precomputed Wigner-D & SO(3) Lebedev-Haar Grid Projector constants
    builder.add_tensor("const.to_grid_mat", to_grid, "f32")
    builder.add_tensor("const.from_grid_mat", from_grid, "f32")
    if lmax >= 2:
        builder.add_tensor("const.wigner_l2_coeff", constants["wigner_l2_coeff"], "f32")
        builder.add_tensor("const.wigner_exp_l2", constants["wigner_exp_l2"], "i32")
    if lmax == 3:
        builder.add_tensor("const.wigner_c_l3", constants["wigner_c_l3"], "f32")
        builder.add_tensor("const.wigner_exp_l3", constants["wigner_exp_l3"], "i32")
    if lmax == 4:
        builder.add_tensor("const.wigner_c_l3l4", constants["wigner_c_l3l4"], "f32")
        builder.add_tensor("const.wigner_exp_l4", constants["wigner_exp_l4"], "i32")

    # 2. Atomic energy statistics (FP64 lossless!) + Initial Embeddings
    builder.add_tensor("out_bias", get_t("atomic_model.out_bias").reshape(-1), "f64")
    builder.add_tensor("bias_atom_e", get_t("atomic_model.fitting_net.bias_atom_e").reshape(-1), "f64")

    dp = "atomic_model.descriptor."
    builder.add_tensor("type_embed", get_t(dp + "type_embedding.adam_type_embedding"), "f32")
    builder.add_tensor("chg_embed", get_t(dp + "charge_spin_embedding.charge_embedding.adam_type_embedding"), "f32")
    builder.add_tensor("spin_embed", get_t(dp + "charge_spin_embedding.spin_embedding.adam_type_embedding"), "f32")
    builder.add_tensor("cs_mix_w", get_t(dp + "charge_spin_embedding.mix_layer.matrix"), "f32")
    builder.add_tensor("cs_mix_b", get_t(dp + "charge_spin_embedding.mix_layer.bias"), "f32")

    builder.add_tensor("bessel_freqs", get_t(dp + "radial_basis.adam_freqs").reshape(-1), "f32")
    builder.add_tensor("rad_w0", get_t(dp + "radial_embedding.net.0.matrix"), "f32")
    builder.add_tensor("rad_norm_scale", get_t(dp + "radial_embedding.net.1.adam_scale").reshape(-1), "f32")
    builder.add_tensor("rad_w3", get_t(dp + "radial_embedding.net.3.matrix"), "f32")

    builder.add_tensor("env_rbf_w1", get_t(dp + "env_seed_embedding.rbf_proj_layer1.matrix"), "f32")
    builder.add_tensor("env_rbf_w2", get_t(dp + "env_seed_embedding.rbf_proj_layer2.matrix"), "f32")
    builder.add_tensor("env_type_embed", get_t(dp + "env_seed_embedding.env_type_embed.adam_type_embedding"), "f32")
    builder.add_tensor("env_g_w1", get_t(dp + "env_seed_embedding.g_layer1.matrix"), "f32")
    builder.add_tensor("env_g_w2", get_t(dp + "env_seed_embedding.g_layer2.matrix"), "f32")
    builder.add_tensor("env_out_proj", get_t(dp + "env_seed_embedding.output_proj.matrix"), "f32")
    builder.add_tensor("film_scale_norm", get_t(dp + "film_scale_norm.adam_scale").reshape(-1), "f32")
    builder.add_tensor("film_shift_norm", get_t(dp + "film_shift_norm.adam_scale").reshape(-1), "f32")
    builder.add_tensor(
        "film_strengths_log",
        np.array(
            [
                float(get_t(dp + "film_scale_strength_log").reshape(-1)[0]),
                float(get_t(dp + "film_shift_strength_log").reshape(-1)[0]),
            ],
            dtype=np.float32,
        ),
        "f32",
    )

    # 3. Interaction Blocks b = 0 .. n_blocks - 1
    for b in range(n_blocks):
        bp = f"{dp}blocks.{b}."
        sp = f"b{b}"
        builder.add_tensor(f"{sp}.post_so2_norm.scale", get_t(bp + "post_so2_norm.adam_scale"), "f32")
        builder.add_tensor(f"{sp}.post_so2_norm.bias", get_t(bp + "post_so2_norm.bias").reshape(-1), "f32")
        builder.add_tensor(f"{sp}.pre_focus_w", get_t(bp + "so2_conv.pre_focus_mix.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.post_focus_w", get_t(bp + "so2_conv.post_focus_mix.weight"), weight_dtype)

        if n_focus > 1:
            builder.add_tensor(f"{sp}.rad_hid_proj_w", get_t(bp + "so2_conv.radial_hidden_proj.weight"), "f32")
            builder.add_tensor(f"{sp}.focus_comp_norm", get_t(bp + "so2_conv.focus_compete_norm.adam_scale"), "f32")
            builder.add_tensor(f"{sp}.focus_comp_w", get_t(bp + "so2_conv.adamw_focus_compete_w"), "f32")

        if radial_so2_mode == "degree_channel":
            builder.add_tensor(f"{sp}.rad_deg_w", get_t(bp + "so2_conv.radial_degree_mixer.weight"), weight_dtype)
            builder.add_tensor(f"{sp}.rad_deg_basis", get_t(bp + "so2_conv.radial_degree_mixer.channel_basis"), "f32")

        for k in range(mixing_layers):
            builder.add_tensor(
                f"{sp}.so2_l{k}.w_m0",
                get_t(bp + f"so2_conv.so2_linears.{k}.weight_m0"),
                weight_dtype,
            )
            builder.add_tensor(
                f"{sp}.so2_l{k}.w_m1",
                get_t(bp + f"so2_conv.so2_linears.{k}.weight_m.0"),
                weight_dtype,
            )
            if k < mixing_layers - 1:
                builder.add_tensor(
                    f"{sp}.so2_l{k}.gate_w",
                    get_t(bp + f"so2_conv.non_linearities.{k}.gate_linear.weight"),
                    weight_dtype,
                )

        builder.add_tensor(f"{sp}.attn_qk_norm", get_t(bp + "so2_conv.attn_qk_norm.adam_scale"), "f32")
        builder.add_tensor(f"{sp}.attn_q_w", get_t(bp + "so2_conv.attn_q_proj.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.attn_k_w", get_t(bp + "so2_conv.attn_k_proj.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.attn_logit_w", get_t(bp + "so2_conv.adamw_attn_logit_w").reshape(-1), "f32")
        builder.add_tensor(f"{sp}.attn_z_raw", get_t(bp + "so2_conv.adamw_attn_z_bias_raw").reshape(-1), "f32")
        builder.add_tensor(f"{sp}.attn_gate_norm", get_t(bp + "so2_conv.attn_output_gate_norm.adam_scale"), "f32")
        builder.add_tensor(f"{sp}.attn_gate_w", get_t(bp + "so2_conv.adamw_attn_gate_w").reshape(-1), "f32")

        mgp = bp + "so2_conv.message_node_grid_product."
        builder.add_tensor(f"{sp}.mn_expand_w", get_t(mgp + "frame_expand.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.mn_contract_w", get_t(mgp + "frame_contract.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.mn_sc_gate_w", get_t(mgp + "scalar_gate.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.mn_res_scale", get_t(mgp + "residual_scale"), "f32")

        builder.add_tensor(f"{sp}.pre_ffn_norm.scale", get_t(bp + "pre_ffn_norms.0.adam_scale"), "f32")
        builder.add_tensor(f"{sp}.pre_ffn_norm.bias", get_t(bp + "pre_ffn_norms.0.bias").reshape(-1), "f32")
        fp = bp + "ffns.0."
        builder.add_tensor(f"{sp}.ffn_l1_w", get_t(fp + "so3_linear_1.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.ffn_sc_gate_w", get_t(fp + "act.scalar_gate.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.ffn_left_w", get_t(fp + "act.grid_op.left_proj.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.ffn_right_w", get_t(fp + "act.grid_op.right_proj.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.ffn_out_w", get_t(fp + "act.grid_op.out_proj.weight"), weight_dtype)
        builder.add_tensor(f"{sp}.ffn_l2_w", get_t(fp + "so3_linear_2.weight"), weight_dtype)

    # 4. Output SO(3) Readout FFN (`GridMLP` self-fused) + GLU Energy Fitting Net
    op = dp + "output_ffn."
    builder.add_tensor("out_ffn.l1_w", get_t(op + "so3_linear_1.weight"), weight_dtype)
    builder.add_tensor("out_ffn.sc_gate_w", get_t(op + "act.scalar_gate.weight"), weight_dtype)
    builder.add_tensor("out_ffn.left_w", get_t(op + "act.grid_op.left_proj.weight"), weight_dtype)
    builder.add_tensor("out_ffn.right_w", get_t(op + "act.grid_op.right_proj.weight"), weight_dtype)
    builder.add_tensor("out_ffn.out_w", get_t(op + "act.grid_op.out_proj.weight"), weight_dtype)
    builder.add_tensor("out_ffn.l2_w", get_t(op + "so3_linear_2.weight"), weight_dtype)

    fitp = "atomic_model.fitting_net.filter_layers.networks.0."
    builder.add_tensor("fit.glu_w", get_t(fitp + "hidden_layers.0.linear.matrix"), "f32")
    builder.add_tensor("fit.glu_b", get_t(fitp + "hidden_layers.0.linear.bias"), "f32")
    builder.add_tensor("fit.out_w", get_t(fitp + "output_layer.matrix").reshape(-1), "f32")

    if oracle_dir is not None and oracle_dir.exists():
        extra["oracle"] = attach_property_oracle(builder, oracle_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = builder.build_bytes()
    out_path.write_bytes(blob)
    print(f"[export_dpa4_rs] Wrote {out_path} ({len(blob) / (1024 * 1024):.2f} MiB, dtype={weight_dtype})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/Users/dp/project/mol_gen/pretrained/dpa4"),
    )
    parser.add_argument(
        "--constants-npz",
        type=Path,
        default=Path(__file__).resolve().parent / "dpa4_constants.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/dp/project/mol_gen/pretrained/dpa4"),
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    const_npz = dict(np.load(str(args.constants_npz)))
    tiers = [
        ("nano", "DPA4-Nano-OMol25-100M-v20260820.pt"),
        ("mini", "DPA4-Mini-OMol25-100M-v20260820.pt"),
        ("neo", "DPA4-Neo-OMol25-100M-v20260820.pt"),
        ("air", "DPA4-Air-OMol25-100M-v20260820.pt"),
        ("plus", "DPA4-Plus-OMol25-100M-v20260820.pt"),
    ]
    for tier, fname in tiers:
        export_dpa4_checkpoint(
            tier=tier,
            pt_path=args.ckpt_dir / fname,
            constants=const_npz,
            out_path=args.out_dir / f"dpa4_{tier}.dpa4",
            weight_dtype="f32",
            oracle_dir=args.oracle_dir,
        )


if __name__ == "__main__":
    main()
