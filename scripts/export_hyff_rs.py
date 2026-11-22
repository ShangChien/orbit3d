#!/usr/bin/env python3
"""
Export HyFF-L2 (`presto`, `vivace`, `moderato`) checkpoints into compact, 64-byte-aligned
`.hyff` (`HYFFRS01`) bundles for `hyff-rs` (Desktop Metal/Vulkan/DX12 + Browser WebGPU WASM),
stacking the 83 element-routed expert MLPs per block into contiguous 3D tensors for single-dispatch
GPU execution, and optionally dumping exact PyTorch reference parity fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors.torch import load_file

MAGIC = b"HYFFRS01"
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


def export_tier(
    tier: str,
    weights_path: Path,
    meta_json_path: Path,
    out_path: Path,
    weight_dtype: str = "f32",
) -> None:
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg_in = meta["model_config"]
    sd = load_file(str(weights_path), device="cpu")

    interactions = int(cfg_in["interactions"])
    n_experts = int(cfg_in.get("expert_count", 83))
    cutoff = float(cfg_in["cutoff"])
    layer_cutoffs = [float(x) for x in cfg_in["layer_cutoffs"]]

    bundle_cfg = {
        "model_id": str(cfg_in.get("model_id", "hyff-l2")),
        "tier": tier,
        "backbone": str(cfg_in.get("backbone", "l4a0")),
        "hidden": int(cfg_in["hidden"]),
        "tensor_hidden": int(cfg_in["tensor_hidden"]),
        "edge_hidden": int(cfg_in["edge_hidden"]),
        "expert_hidden": int(cfg_in["expert_hidden"]),
        "radial_hidden": int(cfg_in["radial_hidden"]),
        "interactions": interactions,
        "n_rbf": int(cfg_in.get("n_rbf", 32)),
        "expert_count": n_experts,
        "cutoff": cutoff,
        "layer_cutoffs": layer_cutoffs,
        "switch_width": float(cfg_in.get("switch_width", 1.0)),
        "condition_spin": bool(cfg_in.get("condition_spin", True)),
        "higher_invariants": bool(cfg_in.get("higher_invariants", True)),
        "weight_dtype": weight_dtype,
    }

    builder = BundleBuilder(bundle_cfg)

    # Global tensors: keep energy offsets, RBF parameters, norms, and pair_coeff in lossless FP32!
    builder.add_tensor("element_lut", sd["element_lut"].numpy(), dtype="i32")
    builder.add_tensor("embedding.weight", sd["embedding.weight"].numpy(), dtype="f32")
    builder.add_tensor("rbf.centers", sd["rbf.centers"].numpy(), dtype="f32")
    builder.add_tensor("rbf.gamma", np.array([float(sd["rbf.gamma"].item())], dtype=np.float32), dtype="f32")
    builder.add_tensor("element_offset.weight", sd["element_offset.weight"].reshape(-1).numpy(), dtype="f32")
    builder.add_tensor("pair_coeff.weight", sd["pair_coeff.weight"].numpy(), dtype="f32")
    builder.add_tensor("state_projection.weight", sd["state_projection.weight"].numpy(), dtype="f32")

    for b in range(interactions):
        p = f"blocks.{b}"
        # Norms & gains stay FP32 for layernorm_centered_fp64 accuracy
        for norm_key in (
            f"{p}.scalar_norm.weight",
            f"{p}.scalar_norm.bias",
            f"{p}.update_norm.weight",
            f"{p}.update_norm.bias",
            f"readout_norms.{b}.weight",
            f"readout_norms.{b}.bias",
        ):
            builder.add_tensor(norm_key, sd[norm_key].numpy(), dtype="f32")

        if f"{p}.vector_tensor_gain" in sd:
            builder.add_tensor(f"{p}.vector_tensor_gain", sd[f"{p}.vector_tensor_gain"].numpy(), dtype="f32")
            builder.add_tensor(f"{p}.vector_vector_gain", sd[f"{p}.vector_vector_gain"].numpy(), dtype="f32")

        # Linear projections & readouts
        linear_keys = [
            (f"{p}.scalar_message.0.weight", weight_dtype),
            (f"{p}.scalar_message.0.bias", "f32"),
            (f"{p}.scalar_message.2.weight", weight_dtype),
            (f"{p}.scalar_message.2.bias", "f32"),
            (f"{p}.radial_filter.0.weight", weight_dtype),
            (f"{p}.radial_filter.0.bias", "f32"),
            (f"{p}.radial_filter.2.weight", weight_dtype),
            (f"{p}.radial_filter.2.bias", "f32"),
            (f"{p}.vector_u.weight", weight_dtype),
            (f"{p}.vector_v.weight", weight_dtype),
            (f"{p}.tensor_u.weight", weight_dtype),
            (f"{p}.tensor_v.weight", weight_dtype),
            (f"{p}.invariant_vector_out.weight", weight_dtype),
            (f"{p}.invariant_tensor_out.weight", weight_dtype),
            (f"{p}.edge_scalar_out.weight", weight_dtype),
            (f"{p}.edge_vector_out.weight", weight_dtype),
            (f"{p}.edge_tensor_out.weight", weight_dtype),
            (f"readouts.{b}.0.weight", "f32"),
            (f"readouts.{b}.0.bias", "f32"),
            (f"readouts.{b}.2.weight", "f32"),
            (f"readouts.{b}.2.bias", "f32"),
        ]
        if b > 0:
            linear_keys.append((f"{p}.edge_vector_in.weight", weight_dtype))
            linear_keys.append((f"{p}.edge_tensor_in.weight", weight_dtype))

        for k, dt in linear_keys:
            builder.add_tensor(k, sd[k].numpy(), dtype=dt)

        # Stack 83 per-element experts into contiguous 3D arrays for 1-dispatch GPU execution!
        exp_w0 = np.stack([sd[f"{p}.update_mlp.experts.{e}.0.weight"].numpy() for e in range(n_experts)], axis=0)
        exp_b0 = np.stack([sd[f"{p}.update_mlp.experts.{e}.0.bias"].numpy() for e in range(n_experts)], axis=0)
        exp_w2 = np.stack([sd[f"{p}.update_mlp.experts.{e}.2.weight"].numpy() for e in range(n_experts)], axis=0)
        exp_b2 = np.stack([sd[f"{p}.update_mlp.experts.{e}.2.bias"].numpy() for e in range(n_experts)], axis=0)

        builder.add_tensor(f"{p}.expert_w0", exp_w0, dtype=weight_dtype)
        builder.add_tensor(f"{p}.expert_b0", exp_b0, dtype="f32")
        builder.add_tensor(f"{p}.expert_w2", exp_w2, dtype=weight_dtype)
        builder.add_tensor(f"{p}.expert_b2", exp_b2, dtype="f32")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = builder.build_bytes()
    out_path.write_bytes(blob)
    print(f"[export_hyff_rs] Wrote {out_path} ({len(blob) / (1024 * 1024):.2f} MiB, dtype={weight_dtype})")


def dump_parity_fixtures(repo_root: Path, out_json: Path) -> None:
    sys.path.insert(0, str(repo_root / "src"))
    from loom.physics.hyff_forcefield import HyFFPotentialGuidance

    test_cases = [
        {
            "name": "salicylaldehyde_neutral_singlet",
            "symbols": ["C", "C", "C", "C", "C", "C", "C", "O", "O", "H", "H", "H", "H", "H", "H"],
            "coords_a": [
                [0.0000, 1.3950, 0.0120],
                [1.2120, 0.6980, -0.0210],
                [1.2080, -0.7010, 0.0150],
                [-0.0040, -1.3980, -0.0180],
                [-1.2100, -0.6950, 0.0250],
                [-1.2090, 0.7020, -0.0110],
                [2.4850, 1.4250, 0.0450],
                [2.5600, 2.6420, -0.0320],
                [2.3820, -1.3920, -0.0280],
                [-0.0020, 2.4810, 0.0340],
                [-0.0060, -2.4840, -0.0410],
                [-2.1510, -1.2390, 0.0520],
                [-2.1480, 1.2480, -0.0290],
                [3.3950, 0.8120, 0.1280],
                [3.1250, -0.7520, -0.0650],
            ],
            "charge": 0.0,
            "multiplicity": 1,
        },
        {
            "name": "pyridinium_radical_cation_doublet",
            "symbols": ["N", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
            "coords_a": [
                [0.0000, 1.3620, 0.0100],
                [1.1850, 0.6850, -0.0150],
                [1.1920, -0.7080, 0.0120],
                [0.0000, -1.4020, -0.0200],
                [-1.1920, -0.7080, 0.0180],
                [-1.1850, 0.6850, -0.0140],
                [0.0000, 2.3810, 0.0250],
                [2.1180, 1.2420, -0.0310],
                [2.1410, -1.2460, 0.0220],
                [0.0000, -2.4890, -0.0380],
                [-2.1410, -1.2460, 0.0300],
                [-2.1180, 1.2420, -0.0270],
            ],
            "charge": 1.0,
            "multiplicity": 2,
        },
        {
            "name": "iridium_bromide_carbonyl_hydride_complex",
            "symbols": ["Ir", "Br", "P", "C", "O", "H", "H", "H", "H"],
            "coords_a": [
                [0.0150, -0.0200, 0.0100],
                [-2.4600, 0.0450, -0.0300],
                [2.3100, 0.0300, 0.0250],
                [0.0400, 1.8850, -0.0400],
                [0.0650, 3.0350, -0.0750],
                [-0.0100, -1.6150, 0.0550],
                [2.8900, 1.3100, 0.2100],
                [2.9200, -0.8100, 1.0150],
                [2.8850, -0.4950, -1.2050],
            ],
            "charge": 0.0,
            "multiplicity": 1,
        },
    ]

    fixtures = {"tiers": {}}
    for tier in ("presto", "vivace", "moderato"):
        ff = HyFFPotentialGuidance(
            tier=tier,
            device="cpu",
            max_force_norm=None,
            route="reference",
        )
        tier_cases = []
        for tc in test_cases:
            coords_t = torch.tensor(tc["coords_a"], dtype=torch.float32)
            desc, energy, forces = ff.extract_native_descriptor_and_forces(
                coords_t,
                tc["symbols"],
                charge=tc["charge"],
                multiplicity=tc["multiplicity"],
            )
            tier_cases.append(
                {
                    "name": tc["name"],
                    "symbols": tc["symbols"],
                    "coords_a": tc["coords_a"],
                    "charge": tc["charge"],
                    "multiplicity": tc["multiplicity"],
                    "energy_ev": float(energy.item()),
                    "forces_ev_a": forces.double().tolist(),
                    "desc_dim": int(desc.shape[-1]),
                    "descriptor": desc.reshape(-1).double().tolist(),
                }
            )
            print(
                f"[parity {tier}] {tc['name']}: E = {float(energy.item()):.8f} eV, "
                f"|F|_max = {float(forces.norm(dim=-1).max().item()):.6f} eV/A, desc={tuple(desc.shape)}"
            )
        fixtures["tiers"][tier] = tier_cases

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(fixtures, indent=2))
    print(f"[export_hyff_rs] Wrote ground-truth PyTorch parity fixtures -> {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "hyff" / "checkpoints",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "loom" / "compat" / "hyff" / "checkpoints",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/dp/project/mol_gen/pretrained/hyff"),
    )
    parser.add_argument(
        "--fixtures-out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    for tier in ("presto", "vivace", "moderato"):
        w_path = args.cache_dir / tier / "weights.safetensors"
        m_path = args.meta_dir / tier / "weights.json"
        if not w_path.exists():
            raise FileNotFoundError(f"Missing cached HyFF weights: {w_path}")
        export_tier(tier, w_path, m_path, args.out_dir / f"hyff_{tier}.hyff", weight_dtype="f32")

    # Also export a compact FP16-weight variant for Moderato (67 MiB vs 133 MiB) for fast browser downloads
    export_tier(
        "moderato",
        args.cache_dir / "moderato" / "weights.safetensors",
        args.meta_dir / "moderato" / "weights.json",
        args.out_dir / "hyff_moderato_fp16.hyff",
        weight_dtype="f16",
    )

    if args.fixtures_out is not None:
        dump_parity_fixtures(Path(__file__).resolve().parents[1], args.fixtures_out)


if __name__ == "__main__":
    main()
