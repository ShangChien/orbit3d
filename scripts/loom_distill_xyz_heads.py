"""Decoupled Sub-Voxel `0.01 Å` Equilibrium XYZ Head Distillation for `loom v1.1`.

Freezes 100% of `conditioner`, Transformer layers, `tree_heads`, and `element_head` (`hidden.detach()`),
attaches zero-initialized LoRA adapters (`atom_heads.0.pred_heads[1..3]`), and optimizes ONLY the sub-voxel
`X, Y, Z` `0.01 Å` coordinate heads on MLIP-relaxed (`DPA-4`) equilibrium conformers — internalizing low-strain
bond lengths & angles with ZERO perturbation to coarse topology or electronic property calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from loom.generator.octree_backend import OctreeDataset
from loom.vendor.uni3dar.data.atom_data import atom_list
from loom.weaving import LoomUni3DARGenerator
from scripts.loom_train_sft import load_sdf_with_targets, sample_active_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--relaxed-sdfs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["homo", "lumo", "joint"], default="homo")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    gen, payload = LoomUni3DARGenerator.load_checkpoint(args.checkpoint, device=args.device, batch_size=args.batch_size)
    gen.enable_v11_xyz_heads(train_xyz_embeddings=False)

    # Freeze EVERYTHING except `atom_heads.0.pred_heads[1..3]` LoRA parameters
    gen.requires_grad_(False)
    xyz_params = []
    for ax_idx in (1, 2, 3):
        phead = gen.backbone.atom_heads[0].pred_heads[ax_idx]
        for name, p in phead.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
                xyz_params.append(p)

    n_xyz = sum(p.numel() for p in xyz_params)
    print(f"[XYZ-Distill] Training ONLY sub-voxel XYZ LoRA heads: {n_xyz:,} params", flush=True)

    all_mols = []
    all_labels_list = []
    for p in args.relaxed_sdfs:
        m, y = load_sdf_with_targets(p, ["homo", "lumo"])
        all_mols.extend(m)
        all_labels_list.append(y)
    all_labels = np.concatenate(all_labels_list, axis=0)
    ds = OctreeDataset(all_mols, gen.backbone, training=True, seed=args.seed)

    optimizer = torch.optim.AdamW(xyz_params, lr=args.lr, weight_decay=args.weight_decay)
    t0 = time.perf_counter()
    init_xyz_loss = None
    running = 0.0

    for step in range(1, args.steps + 1):
        packed = None
        idx_b = None
        for _ in range(5):
            cand = rng.integers(0, len(all_mols), size=args.batch_size)
            try:
                packed = ds.collate(cand)
                idx_b = cand
                break
            except ValueError:
                continue
        if packed is None or idx_b is None:
            continue

        t_norm = gen.normalize_targets(all_labels[idx_b])
        m_act = sample_active_mask(len(idx_b), gen.num_props, args.mode, rng, device)

        bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in packed.items()}
        bd["decoder_phy_pos"] = bd["decoder_phy_pos"].float()

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            decoder_input, target, _ = gen.backbone.get_feat_from_dataloader(bd)
            cu = bd["decoder_input_len"].to(device, dtype=torch.int32)
            max_len = int(bd["decoder_input_max_len"].item())
            hidden = gen.forward_sequence(decoder_input, cu, max_len, t_norm, m_act).detach()

        atom = target.is_gen_atom
        labels_a = [v.long() for v in target.to_list(atom)]
        real_atom = (labels_a[0] >= 0) & (labels_a[0] < len(atom_list))
        null_atom = ~real_atom
        if not bool(real_atom.any().item()):
            continue

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits_a = gen.backbone.atom_heads[0](hidden[atom], labels_a)
            loss_xyz_real = (
                F.cross_entropy(logits_a[1].float()[real_atom], labels_a[1][real_atom])
                + F.cross_entropy(logits_a[2].float()[real_atom], labels_a[2][real_atom])
                + F.cross_entropy(logits_a[3].float()[real_atom], labels_a[3][real_atom])
            ) / 3.0
            if bool(null_atom.any().item()):
                loss_xyz_null = (
                    F.cross_entropy(logits_a[1].float()[null_atom], labels_a[1][null_atom])
                    + F.cross_entropy(logits_a[2].float()[null_atom], labels_a[2][null_atom])
                    + F.cross_entropy(logits_a[3].float()[null_atom], labels_a[3][null_atom])
                ) / 3.0
                loss_xyz = loss_xyz_real + 0.25 * loss_xyz_null
            else:
                loss_xyz = loss_xyz_real

        val_l = float(loss_xyz_real.detach().item())
        if init_xyz_loss is None:
            init_xyz_loss = val_l
        running += val_l

        optimizer.zero_grad(set_to_none=True)
        loss_xyz.backward()
        torch.nn.utils.clip_grad_norm_(xyz_params, 1.0)
        optimizer.step()

        if step % 50 == 0 or step == args.steps:
            avg_l = running / (50.0 if step % 50 == 0 else (step % 50))
            running = 0.0
            print(f"[XYZ-Distill {step:04d}/{args.steps:04d}] xyz_ce={avg_l:.4f} (init={init_xyz_loss:.4f}) ({time.perf_counter()-t0:.1f}s)", flush=True)

    meta = dict(payload.get("metadata", {}))
    meta["xyz_distill"] = {
        "steps": args.steps,
        "init_xyz_ce": init_xyz_loss,
        "final_xyz_ce": val_l,
        "wall_seconds": time.perf_counter() - t0,
    }
    gen.save_checkpoint(args.output_checkpoint, extra_metadata=meta)
    print(json.dumps(meta["xyz_distill"], indent=2), flush=True)


if __name__ == "__main__":
    main()
