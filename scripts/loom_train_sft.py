"""Stage 0 & Stage 1 Trainer for `Uni-Loom`:
  - Stage 0: High-capacity `LoomConditioner` + `LoRA (r=16)` Continuous Tail-Balanced SFT + Contrastive Margin.
  - Stage 1: Multi-round Tail-Debiased Hindsight Relabeling Self-Training with explicit pool mixing ratio.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from rdkit import Chem

from loom.generator.octree_backend import OctreeDataset
from loom.weaving import LoomUni3DARGenerator


def load_sdf_with_targets(sdf_path: Path, target_cols: Sequence[str]) -> Tuple[List[Chem.Mol], np.ndarray]:
    mols: List[Chem.Mol] = []
    labels: List[List[float]] = []
    for m in Chem.SDMolSupplier(str(sdf_path), removeHs=False):
        if m is None:
            continue
        row = []
        ok = True
        for col in target_cols:
            if m.HasProp(f"B_{col}"):
                val = float(m.GetProp(f"B_{col}"))
            elif m.HasProp(col):
                val = float(m.GetProp(col))
            else:
                ok = False
                break
            if not math.isfinite(val):
                ok = False
                break
            row.append(val)
        if ok:
            mols.append(m)
            labels.append(row)
    return mols, np.asarray(labels, dtype=np.float32)


def compute_tail_balanced_weights(labels: np.ndarray, col_indices: Sequence[int], power: float = 0.65, n_bins: int = 24) -> np.ndarray:
    """Inverse-density histogram reweighting so tail quantiles (`P05..P15`, `P85..P95`) are not starved."""
    n = len(labels)
    if power <= 0.0 or n < n_bins:
        return np.ones(n, dtype=np.float64) / max(n, 1)

    acc_w = np.zeros(n, dtype=np.float64)
    for c in col_indices:
        vals = labels[:, c]
        lo, hi = np.percentile(vals, [0.5, 99.5])
        edges = np.linspace(lo - 1e-4, hi + 1e-4, n_bins + 1)
        bin_ids = np.clip(np.digitize(vals, edges) - 1, 0, n_bins - 1)
        counts = np.bincount(bin_ids, minlength=n_bins).astype(np.float64)
        median_c = np.median(counts[counts > 0])
        bin_w = (counts + 0.15 * median_c) ** (-power)
        sample_w = bin_w[bin_ids]
        sample_w = np.clip(sample_w / np.mean(sample_w), 0.25, 4.5)
        acc_w += sample_w
    acc_w /= max(len(col_indices), 1)
    return acc_w / acc_w.sum()


def sample_active_mask(batch_size: int, num_props: int, mode: str, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    mask = np.zeros((batch_size, num_props), dtype=bool)
    if mode == "homo":
        mask[:, 0] = True
    elif mode == "lumo":
        mask[:, 1] = True
    elif mode == "joint":
        draws = rng.random(batch_size)
        for i, u in enumerate(draws):
            if u < 0.35:
                mask[i, 0] = True
            elif u < 0.70:
                mask[i, 1] = True
            else:
                mask[i, :] = True
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def evaluate_validation(
    gen: LoomUni3DARGenerator,
    val_ds: OctreeDataset,
    val_labels: np.ndarray,
    mode: str,
    batch_size: int = 32,
) -> Dict[str, float]:
    device = next(gen.parameters()).device
    rng = np.random.default_rng(12345)
    n = len(val_ds)
    order = np.arange(n)
    rng.shuffle(order)

    total_nll = 0.0
    n_batches = 0
    total_el = 0
    total_pref = 0
    total_gap = 0.0

    axes_to_check = ["homo"] if mode == "homo" else (["lumo"] if mode == "lumo" else ["homo", "lumo", "joint"])
    by_axis: Dict[str, Any] = {}

    for ax in axes_to_check:
        ax_el = 0
        ax_pref = 0
        ax_gap = 0.0
        ax_nll = 0.0
        ax_b = 0
        for start in range(0, min(n, 512), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 4:
                continue
            try:
                packed = val_ds.collate(idx)
            except ValueError:
                continue
            t_norm = gen.normalize_targets(val_labels[idx])
            m = sample_active_mask(len(idx), gen.num_props, ax if ax != "joint" else "joint", rng, device)
            if ax == "joint":
                m = torch.ones((len(idx), gen.num_props), dtype=torch.bool, device=device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss_v, _ = gen.compute_sft_loss(packed, t_norm, m, cfg_dropout_prob=0.0, contrast_weight=0.0)
                pref = gen.evaluate_teacher_forcing_preference(packed, t_norm, m, min_gap_eV=0.25)
            ax_nll += float(loss_v.item())
            ax_b += 1
            if pref["eligible"] > 0:
                ax_el += pref["eligible"]
                ax_pref += pref["preferred"]
                ax_gap += pref["mean_nll_gap"] * pref["eligible"]

        pref_rate = (ax_pref / ax_el) if ax_el > 0 else float("nan")
        mean_gap = (ax_gap / ax_el) if ax_el > 0 else float("nan")
        mean_nll = (ax_nll / ax_b) if ax_b > 0 else float("nan")
        by_axis[ax] = {
            "val_nll": mean_nll,
            "preference_rate": pref_rate,
            "mean_nll_gap": mean_gap,
            "eligible_pairs": ax_el,
        }
        total_nll += ax_nll
        n_batches += ax_b
        total_el += ax_el
        total_pref += ax_pref
        total_gap += ax_gap

    return {
        "val_nll": total_nll / max(n_batches, 1),
        "preference_rate": total_pref / max(total_el, 1),
        "mean_nll_gap": total_gap / max(total_el, 1),
        "by_axis": by_axis,  # type: ignore[dict-item]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--train-sdf", type=Path, required=True)
    parser.add_argument("--valid-sdf", type=Path, required=True)
    parser.add_argument("--extra-hindsight-sdfs", type=Path, nargs="*", default=[])
    parser.add_argument("--extra-mix-prob", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["homo", "lumo", "joint"], default="homo")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--min-lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--enable-v11-xyz-heads", action="store_true")
    parser.add_argument("--num-prefix-tokens", type=int, default=0)
    parser.add_argument("--cfg-dropout", type=float, default=0.15)
    parser.add_argument("--contrast-weight", type=float, default=0.35)
    parser.add_argument("--contrast-margin", type=float, default=0.06)
    parser.add_argument("--tail-power", type=float, default=0.65)
    parser.add_argument("--label-jitter-eV", type=float, default=0.015)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_cols = ["homo", "lumo"]
    print(f"[Loom-SFT] Loading base training SDF: {args.train_sdf}", flush=True)
    base_mols, base_labels = load_sdf_with_targets(args.train_sdf, target_cols)
    val_mols, val_labels = load_sdf_with_targets(args.valid_sdf, target_cols)

    extra_mols: List[Chem.Mol] = []
    extra_labels_list: List[np.ndarray] = []
    for p in args.extra_hindsight_sdfs:
        em, el = load_sdf_with_targets(p, target_cols)
        extra_mols.extend(em)
        extra_labels_list.append(el)
    extra_labels = np.concatenate(extra_labels_list, axis=0) if extra_labels_list else np.zeros((0, 2), dtype=np.float32)

    mean_vec = np.mean(base_labels, axis=0).tolist()
    scale_vec = np.std(base_labels, axis=0).tolist()
    print(
        f"[Loom-SFT] Base train={len(base_mols)}, Extra hindsight={len(extra_mols)}, Valid={len(val_mols)} | "
        f"mean={mean_vec}, scale={scale_vec}",
        flush=True,
    )

    gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_checkpoint,
        target_cols=target_cols,
        target_units=["eV", "eV"],
        mean=mean_vec,
        scale=scale_vec,
        device=args.device,
        batch_size=args.batch_size,
        num_rbf=32,
        num_freqs=16,
        num_prefix_tokens=args.num_prefix_tokens,
        modulation_cap=4.0,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_include_ffn=True,
        lora_include_heads=True,
    )
    if args.enable_v11_xyz_heads:
        trainable = gen.enable_v11_xyz_heads()
    else:
        trainable = gen.configure_trainable(train_conditioner=True, train_lora=True)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in gen.parameters())
    print(f"[Loom-SFT] Trainable parameters: {n_trainable:,} / {n_total:,} ({100.0*n_trainable/n_total:.2f}%)", flush=True)

    active_cols = [0] if args.mode == "homo" else ([1] if args.mode == "lumo" else [0, 1])
    base_probs = compute_tail_balanced_weights(base_labels, active_cols, power=args.tail_power)
    extra_probs = (
        compute_tail_balanced_weights(extra_labels, active_cols, power=args.tail_power)
        if len(extra_mols) > 0
        else None
    )

    base_ds = OctreeDataset(base_mols, gen.backbone, training=True, seed=args.seed)
    extra_ds = OctreeDataset(extra_mols, gen.backbone, training=True, seed=args.seed + 7) if len(extra_mols) > 0 else None
    val_ds = OctreeDataset(val_mols, gen.backbone, training=False, seed=args.seed + 13)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    def _lr_at_step(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(args.warmup_steps, 1)
        prog = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    history: List[Dict[str, Any]] = []
    best_score = -1e9
    t_start = time.perf_counter()

    # Initial zero-step validation check
    init_val = evaluate_validation(gen, val_ds, val_labels, args.mode, batch_size=args.batch_size)
    print(f"[Loom-SFT Step 0000] Init val_nll={init_val['val_nll']:.4f}, pref={init_val['preference_rate']*100:.2f}%", flush=True)
    history.append({"step": 0, "phase": "val", **init_val})

    running_loss = 0.0
    running_pref = 0.0
    running_pref_n = 0

    for step in range(1, args.steps + 1):
        cur_lr = _lr_at_step(step - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Sample tail-balanced batch (with explicit hindsight mixing if provided)
        packed = None
        batch_y = None
        for _retry in range(5):
            try:
                if extra_ds is not None and extra_probs is not None and args.extra_mix_prob > 0.0:
                    n_ex = int(round(args.batch_size * args.extra_mix_prob))
                    n_bs = args.batch_size - n_ex
                    idx_b = rng.choice(len(base_mols), size=n_bs, replace=True, p=base_probs)
                    idx_e = rng.choice(len(extra_mols), size=n_ex, replace=True, p=extra_probs)
                    items = [base_ds.grid[int(i)] for i in idx_b] + [extra_ds.grid[int(j)] for j in idx_e]
                    # Verify no atom loss
                    valid_items = True
                    for item, m_exp in zip(
                        items,
                        [base_mols[int(i)] for i in idx_b] + [extra_mols[int(j)] for j in idx_e],
                    ):
                        dec = int(((item["decoder_level"] == 0) & item["decoder_is_second_atom"]).sum().item())
                        if item["cutoff_atom_count"] or dec != m_exp.GetNumAtoms():
                            valid_items = False
                            break
                    if not valid_items:
                        continue
                    packed = base_ds.grid.collater(items)
                    batch_y = np.concatenate([base_labels[idx_b], extra_labels[idx_e]], axis=0)
                else:
                    idx_b = rng.choice(len(base_mols), size=args.batch_size, replace=True, p=base_probs)
                    packed = base_ds.collate(idx_b)
                    batch_y = base_labels[idx_b]
                break
            except ValueError:
                continue

        if packed is None or batch_y is None:
            continue

        if args.label_jitter_eV > 0.0:
            batch_y = batch_y + rng.normal(0.0, args.label_jitter_eV, size=batch_y.shape).astype(np.float32)

        targets_norm = gen.normalize_targets(batch_y)
        active_mask = sample_active_mask(len(batch_y), gen.num_props, args.mode, rng, device)

        gen.train()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss, info = gen.compute_sft_loss(
                packed,
                targets_norm,
                active_mask,
                cfg_dropout_prob=args.cfg_dropout,
                contrast_weight=args.contrast_weight,
                contrast_margin=args.contrast_margin,
                contrast_min_gap_eV=0.25,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
        optimizer.step()

        running_loss += info["loss_total"]
        if math.isfinite(info.get("inbatch_preference_rate", float("nan"))):
            running_pref += info["inbatch_preference_rate"]
            running_pref_n += 1

        if step % 100 == 0:
            elapsed = time.perf_counter() - t_start
            avg_l = running_loss / 100.0
            avg_p = (running_pref / running_pref_n) if running_pref_n > 0 else float("nan")
            print(
                f"[Loom-SFT {step:05d}/{args.steps:05d}] loss={avg_l:.4f} "
                f"(gen={info['loss_gen']:.4f}, contrast={info.get('loss_contrast', 0.0):.4f}) "
                f"inbatch_pref={avg_p*100:.1f}% lr={cur_lr:.2e} grad={grad_norm:.2f} ({elapsed:.0f}s)",
                flush=True,
            )
            history.append({
                "step": step,
                "phase": "train",
                "loss_avg100": avg_l,
                "inbatch_pref_avg100": avg_p,
                "lr": cur_lr,
                "grad_norm": grad_norm,
                "elapsed_sec": elapsed,
            })
            running_loss = 0.0
            running_pref = 0.0
            running_pref_n = 0

        if step % args.eval_every == 0 or step == args.steps:
            val_res = evaluate_validation(gen, val_ds, val_labels, args.mode, batch_size=args.batch_size)
            score = val_res["preference_rate"] - 0.15 * val_res["val_nll"]
            print(
                f"  >>> [VAL Step {step:05d}] pref={val_res['preference_rate']*100:.2f}% "
                f"(gap={val_res['mean_nll_gap']:.4f}), val_nll={val_res['val_nll']:.4f}",
                flush=True,
            )
            history.append({"step": step, "phase": "val", **val_res})
            (args.output_dir / "train_log.json").write_text(json.dumps(history, indent=2) + "\n")

            meta = {
                "mode": args.mode,
                "step": step,
                "val": val_res,
                "args": {k: str(v) for k, v in vars(args).items()},
            }
            if score > best_score:
                best_score = score
                gen.save_checkpoint(args.output_dir / "best_checkpoint.pt", extra_metadata=meta)
            if step == args.steps:
                gen.save_checkpoint(args.output_dir / "final_checkpoint.pt", extra_metadata=meta)


if __name__ == "__main__":
    main()
