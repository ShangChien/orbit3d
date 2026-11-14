"""Stage 2 Disentangled Multi-Channel Online GRPO & `PotentialSlot` Hot-Swap Runner for `Uni-Loom`.

Supports hot-swapping `--potential-backend {dpa4, hyff, none}` with zero changes to generator or
reward-oracle code, saving periodic checkpoints (`step_0064.pt`, `step_0128.pt`, `step_0192.pt`, `step_0256.pt`)
so we can trace the Anti-Reward-Hacking Curve (`RewardOracle` vs `AuditOracle` across RL budgets)!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Tuple
import numpy as np
from rdkit import Chem
import torch

from loom.oracles.unimol2_task import B_WEIGHT_SHA256
from loom.weaving import (
    LoomGRPOTrainer,
    LoomUni3DARGenerator,
    PotentialSlot,
    RewardOracleSlot,
    assert_not_audit_oracle,
    formal_charge_signature,
)
from scripts.loom_train_sft import load_sdf_with_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-sdf", type=Path, required=True)
    parser.add_argument("--reward-oracle-dir", type=Path, required=True)
    parser.add_argument("--potential-backend", choices=["dpa4", "hyff", "none"], default="dpa4")
    parser.add_argument("--potential-weights", type=Path, default=None)
    parser.add_argument("--mode", choices=["homo", "lumo", "joint"], default="homo")
    parser.add_argument("--steps", type=int, default=256, help="256 steps × 4 prompts/step = 1,024 groups (8,192 rollouts)")
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--kl-beta", type=float, default=0.025)
    parser.add_argument("--replay-weight", type=float, default=0.15)
    parser.add_argument("--w-valid", type=float, default=0.75)
    parser.add_argument("--w-phys", type=float, default=0.30)
    parser.add_argument("--w-prop", type=float, default=1.00)
    parser.add_argument("--cfg-scale", type=float, default=1.65)
    parser.add_argument("--save-every", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    gen, _ = LoomUni3DARGenerator.load_checkpoint(
        args.init_checkpoint, device=args.device, batch_size=args.prompts_per_step * args.group_size
    )
    gen.configure_trainable(train_conditioner=True, train_lora=True)

    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )
    assert_not_audit_oracle(reward_slot)

    pot_slot = (
        PotentialSlot(backend=args.potential_backend, weights=str(args.potential_weights), device=args.device)
        if args.potential_backend != "none" and args.potential_weights is not None
        else None
    )

    replay_mols, replay_labels = load_sdf_with_targets(args.replay_sdf, ["homo", "lumo"])
    seen_charges = frozenset(formal_charge_signature(m) for m in replay_mols)

    # Quantiles for continuous stratified prompt sampling across [P06 .. P94]
    h_p06, h_p94 = np.percentile(replay_labels[:, 0], [6.0, 94.0])
    l_p06, l_p94 = np.percentile(replay_labels[:, 1], [6.0, 94.0])

    def _sample_prompts(step_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        p = args.prompts_per_step
        targets = np.zeros((p, 2), dtype=np.float32)
        # Stratified across P intervals so EVERY step spans low, mid, and high quantiles!
        strata = (np.arange(p, dtype=np.float64) + rng.random(p)) / p
        rng.shuffle(strata)
        targets[:, 0] = (h_p06 + strata * (h_p94 - h_p06)).astype(np.float32)
        strata_l = (np.arange(p, dtype=np.float64) + rng.random(p)) / p
        rng.shuffle(strata_l)
        targets[:, 1] = (l_p06 + strata_l * (l_p94 - l_p06)).astype(np.float32)

        mask = np.zeros((p, 2), dtype=bool)
        if args.mode == "homo":
            mask[:, 0] = True
        elif args.mode == "lumo":
            mask[:, 1] = True
        else:
            for i in range(p):
                u = rng.random()
                if u < 0.35:
                    mask[i, 0] = True
                elif u < 0.70:
                    mask[i, 1] = True
                else:
                    mask[i, :] = True
                    if targets[i, 0] > targets[i, 1] - 0.8:
                        src_i = int(rng.integers(0, len(replay_labels)))
                        targets[i] = replay_labels[src_i]
        return targets, mask

    trainer = LoomGRPOTrainer(
        generator=gen,
        reward_oracle=reward_slot,
        potential_slot=pot_slot,
        seen_charge_signatures=seen_charges,
        replay_molecules=replay_mols,
        replay_targets_eV=replay_labels,
        lr=args.lr,
        kl_beta=args.kl_beta,
        replay_weight=args.replay_weight,
        w_valid=args.w_valid,
        w_phys=args.w_phys,
        w_prop=args.w_prop,
        cfg_scale=args.cfg_scale,
        xyz_cfg_scale=1.0,
        seed=args.seed,
    )

    t0 = time.perf_counter()

    def _ckpt_cb(step_done: int, model: LoomUni3DARGenerator, hist: list) -> None:  # type: ignore[type-arg]
        if step_done % args.save_every == 0 or step_done == args.steps:
            groups_done = step_done * args.prompts_per_step
            ckpt_file = args.output_dir / f"grpo_groups_{groups_done:04d}.pt"
            model.save_checkpoint(
                ckpt_file,
                extra_metadata={
                    "stage": "stage2_grpo",
                    "potential_backend": args.potential_backend,
                    "step": step_done,
                    "groups_completed": groups_done,
                    "mode": args.mode,
                },
            )
            (args.output_dir / "grpo_history.json").write_text(json.dumps(hist, indent=2) + "\n")

    history = trainer.train_loop(
        total_steps=args.steps,
        prompt_sampler_fn=_sample_prompts,
        group_size=args.group_size,
        checkpoint_fn=_ckpt_cb,  # type: ignore[arg-type]
        log_every=8,
    )
    gen.save_checkpoint(
        args.output_dir / "final_grpo_checkpoint.pt",
        extra_metadata={
            "stage": "stage2_grpo_final",
            "potential_backend": args.potential_backend,
            "groups_completed": args.steps * args.prompts_per_step,
            "wall_seconds": time.perf_counter() - t0,
        },
    )
    (args.output_dir / "grpo_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[Loom-GRPO] Completed {args.steps * args.prompts_per_step} groups in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
