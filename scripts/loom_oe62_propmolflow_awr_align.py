"""Online Advantage-Weighted Flow SFT (`awr_sft`) Alignment for `PropMolFlow-OE62` via `Uni-Loom`.

Demonstrates plug-and-play base `GeneratorSlot` hot-swapping (`Uni-3DAR` <-> `PropMolFlow`):
  - Instantiates `LoomFlowMolGenerator` via `build_generator_slot("propmolflow", ...)`.
  - Mounts the exact OE62 declarative `CompositeCriticBus` (`Uni-Mol2` HOMO + `SA_Score <= 3.5`
    + `smarts_alert <= 0` + `mol_weight <= 550` + `DPA-4` `F_phys`, strictly WITHOUT `AuditOracleSlot`).
  - Runs `LoomGRPOTrainer(alignment_mode="auto")` which automatically resolves to
    `Online Advantage-Weighted Flow SFT` (`awr_sft`) with strict native-legality gating,
    20-step `DPA-4` coordinate relaxation + post-relaxation critic re-verification, and a
    curated `SA_Score <= 3.5`, `F_max <= 2.2 eV/A` equilibrium replay anchor
    (default `56` steps x `4` prompts x `8` rollouts = `224` groups).
  - Saves `ckpt_52p0gpuh_loom_awr.pt` compatible with `loom_oe62_propmolflow_head2head.py --phase eval_checkpoint`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch

RDLogger.DisableLog("rdApp.*")
torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.chemistry.valence_reconciler import is_clean_organic_charge_state
from loom.oracles.unimol2_task import B_WEIGHT_SHA256
from loom.weaving import (
    AssetRegistry,
    BuiltInRuleAsset,
    CompositeCriticBus,
    LoomFlowMolGenerator,
    LoomGRPOTrainer,
    PotentialSlot,
    RewardOracleSlot,
    SlotOracleAsset,
    TaskWeavingSpec,
    build_generator_slot,
    formal_charge_signature,
)
from scripts.loom_train_sft import load_sdf_with_targets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--init-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/propmolflow_oe62/checkpoints/ckpt_52p0gpuh.pt"),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/propmolflow_oe62/data"),
    )
    p.add_argument("--task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument(
        "--replay-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"),
    )
    p.add_argument(
        "--reward-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"),
    )
    p.add_argument(
        "--potential-weights",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"),
    )
    p.add_argument("--propmolflow-root", type=Path, default=Path("/sjtu-combustion/wjw/chensq/PropMolFlow"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--awr-steps", type=int, default=56)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.8e-5)
    p.add_argument("--replay-weight", type=float, default=0.45)
    p.add_argument("--replay-batch-size", type=int, default=32)
    p.add_argument("--replay-max-fmax", type=float, default=2.2)
    p.add_argument("--relax-distill-weight", type=float, default=0.0)
    p.add_argument("--w-valid", type=float, default=1.00)
    p.add_argument("--w-phys", type=float, default=0.55)
    p.add_argument("--w-prop", type=float, default=1.05)
    p.add_argument("--grad-clip-mode", type=str, default="norm")
    p.add_argument("--awr-temperature", type=float, default=0.45)
    p.add_argument("--awr-max-weight", type=float, default=5.0)
    p.add_argument("--awr-repeat-k", type=int, default=2)
    p.add_argument("--awr-require-native-legal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--awr-relax-steps", type=int, default=20)
    p.add_argument("--awr-max-fmax", type=float, default=2.2)
    p.add_argument("--sft-time-draws", type=int, default=4)
    p.add_argument("--sample-timesteps", type=int, default=100)
    p.add_argument("--max-replay-pool", type=int, default=4096)
    p.add_argument(
        "--output-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/propmolflow_oe62_awr/ckpt_52p0gpuh_loom_awr.pt"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_propmolflow_52gpuh_awr_train_meta.json"),
    )
    args = p.parse_args()

    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(4)

    # 1. Plug-and-play `GeneratorSlot` instantiation for PropMolFlow
    gen = build_generator_slot(
        "propmolflow",
        args.init_ckpt,
        device=args.device,
        data_dir=args.data_dir,
        property_name="homo",
        sample_timesteps=int(args.sample_timesteps),
        sft_time_draws=int(args.sft_time_draws),
        propmolflow_root=args.propmolflow_root,
    )
    assert isinstance(gen, LoomFlowMolGenerator)
    allowed_set = frozenset(gen.allowed_elements)
    trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(
        f"[PropMolFlow-AWR] Loaded {args.init_ckpt} via build_generator_slot('propmolflow') | "
        f"family={gen.generator_family} | params={trainable_params:,} | base_gpu_h={gen.cum_gpu_hours:.3f}",
        flush=True,
    )

    # 2. Declarative CompositeCriticBus strictly WITHOUT blind AuditOracleSlot
    spec = TaskWeavingSpec.from_file(args.task_spec)
    registry = AssetRegistry(register_builtin_rules=True)
    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )
    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.potential_weights), device=args.device)
    registry.register_predictor(SlotOracleAsset(reward_slot, asset_id="ext_unimol2", units=["eV", "eV"]))
    registry.register_predictor(BuiltInRuleAsset(asset_id="builtin_rules"))
    registry.register_potential("mlip_dpa4", pot_slot)

    replay_mols_all, _ = load_sdf_with_targets(args.replay_sdf, ["homo"])
    # Restrict replay admission & reconciler to PropMolFlow's 12-element alphabet & clean SA<=3.5 equilibrium conformers
    replay_mols_12 = [
        m
        for m in replay_mols_all
        if m is not None
        and 2 <= m.GetNumAtoms() <= 80
        and is_clean_organic_charge_state(m)
        and all(a.GetSymbol() in allowed_set for a in m.GetAtoms())
        and all(-2 <= a.GetFormalCharge() <= 3 for a in m.GetAtoms())
    ]
    seen_charges = frozenset(formal_charge_signature(m) for m in replay_mols_12)
    rule_filter = BuiltInRuleAsset(asset_id="builtin_rules_prefilter")
    r_out = rule_filter.predict_mols(replay_mols_12, batch_size=256)
    sa_col = r_out["mean"][:, rule_filter.channel_index("sa_score")]
    mw_col = r_out["mean"][:, rule_filter.channel_index("mol_weight")]
    sm_col = r_out["mean"][:, rule_filter.channel_index("smarts_alert")]
    replay_mols_12 = [
        replay_mols_12[k]
        for k in range(len(replay_mols_12))
        if (sa_col[k] <= 3.5) and (mw_col[k] <= 550.0) and (sm_col[k] <= 0.0)
    ]
    if int(args.max_replay_pool) > 0 and len(replay_mols_12) > int(args.max_replay_pool):
        sub_rng = np.random.default_rng(2026)
        pick_idx = sub_rng.choice(len(replay_mols_12), size=int(args.max_replay_pool), replace=False)
        replay_mols_12 = [replay_mols_12[int(i)] for i in pick_idx]
    bus = CompositeCriticBus(
        spec=spec,
        registry=registry,
        seen_charge_signatures=seen_charges,
        allowed_elements=gen.allowed_elements,
    )

    replay_mols, replay_labels_2d, relabel_summary = bus.relabel_conformer_pool(
        replay_mols_12, max_fmax_eV_A=float(args.replay_max_fmax)
    )
    homo_obj_idx = bus.spec.objective_names.index("homo")
    replay_labels_homo = replay_labels_2d[:, [homo_obj_idx]].astype(np.float32)
    print(f"[PropMolFlow-AWR] Declarative Replay Admission: {relabel_summary}", flush=True)

    h_p06, h_p94 = np.percentile(replay_labels_homo[:, 0], [6.0, 94.0])

    def _sample_prompts(step_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        del step_idx
        p = int(args.prompts_per_step)
        t_mat = np.zeros((p, 1), dtype=np.float32)
        strata = (np.arange(p, dtype=np.float64) + rng.random(p)) / p
        rng.shuffle(strata)
        t_mat[:, 0] = (h_p06 + strata * (h_p94 - h_p06)).astype(np.float32)
        mask = np.ones((p, 1), dtype=bool)
        return t_mat, mask

    # 3. Unified `LoomGRPOTrainer` automatically resolves `alignment_mode='auto'` -> `'awr_sft'`
    trainer = LoomGRPOTrainer(
        generator=gen,
        critic_bus=bus,
        replay_molecules=replay_mols,
        replay_targets_eV=replay_labels_homo,
        alignment_mode="auto",
        lr=float(args.lr),
        kl_beta=0.0,
        ppo_epochs=int(args.ppo_epochs),
        replay_weight=float(args.replay_weight),
        replay_batch_size=int(args.replay_batch_size),
        w_valid=float(args.w_valid),
        w_phys=float(args.w_phys),
        w_prop=float(args.w_prop),
        sharp_concentration=True,
        relax_distill_weight=float(args.relax_distill_weight),
        relax_distill_steps=20,
        relax_distill_max_mols=8,
        relax_distill_max_fmax=2.0,
        awr_temperature=float(args.awr_temperature),
        awr_max_weight=float(args.awr_max_weight),
        awr_repeat_k=int(args.awr_repeat_k),
        awr_hindsight_fallback=True,
        awr_require_native_legal=bool(args.awr_require_native_legal),
        awr_relax_steps=int(args.awr_relax_steps),
        awr_max_fmax=float(args.awr_max_fmax),
        grad_clip_mode=str(args.grad_clip_mode),
        seed=2026,
    )
    assert trainer.alignment_mode == "awr_sft"

    t0 = time.perf_counter()
    hist = trainer.train_loop(
        total_steps=int(args.awr_steps),
        prompt_sampler_fn=_sample_prompts,
        group_size=int(args.group_size),
        log_every=6,
    )
    wall_sec = time.perf_counter() - t0
    awr_gpu_h = wall_sec / 3600.0
    gen.cum_gpu_hours = float(gen.cum_gpu_hours) + awr_gpu_h
    gen.global_step = int(gen.global_step) + int(args.awr_steps) * int(args.ppo_epochs)

    meta: Dict[str, Any] = {
        "experiment": "OE62 PropMolFlow Online Advantage-Weighted Flow SFT (Uni-Loom Plug-and-Play Post-Training)",
        "init_checkpoint": str(args.init_ckpt),
        "saved_checkpoint": str(args.output_ckpt),
        "generator_family": gen.generator_family,
        "alignment_mode": trainer.alignment_mode,
        "groups_completed": int(args.awr_steps) * int(args.prompts_per_step),
        "total_online_rollouts": int(args.awr_steps) * int(args.prompts_per_step) * int(args.group_size),
        "awr_wall_seconds": round(wall_sec, 2),
        "awr_gpu_hours": round(awr_gpu_h, 4),
        "total_cum_gpu_hours": round(gen.cum_gpu_hours, 4),
        "hyperparams": {
            "lr": float(args.lr),
            "ppo_epochs": int(args.ppo_epochs),
            "replay_weight": float(args.replay_weight),
            "replay_batch_size": int(args.replay_batch_size),
            "replay_max_fmax": float(args.replay_max_fmax),
            "relax_distill_weight": float(args.relax_distill_weight),
            "w_valid": float(args.w_valid),
            "w_phys": float(args.w_phys),
            "w_prop": float(args.w_prop),
            "grad_clip_mode": str(args.grad_clip_mode),
            "awr_temperature": float(args.awr_temperature),
            "awr_max_weight": float(args.awr_max_weight),
            "awr_repeat_k": int(args.awr_repeat_k),
            "awr_require_native_legal": bool(args.awr_require_native_legal),
            "awr_relax_steps": int(args.awr_relax_steps),
            "awr_max_fmax": float(args.awr_max_fmax),
            "sft_time_draws": int(args.sft_time_draws),
            "sample_timesteps": int(args.sample_timesteps),
        },
        "replay_admission": relabel_summary,
        "history": hist,
    }
    gen.save_propmolflow_checkpoint(args.output_ckpt, extra_metadata=meta)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[PropMolFlow-AWR DONE] Saved {args.output_ckpt} and {args.output_json} "
        f"({wall_sec:.1f}s = {awr_gpu_h:.4f} GPU-h)",
        flush=True,
    )


if __name__ == "__main__":
    main()
