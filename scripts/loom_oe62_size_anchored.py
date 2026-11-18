"""Size-Anchored Uni-Loom (`["homo", "n_atoms"]`) Pipeline on OE62 (`HOMO + SA <= 3.5`).

Addresses small-molecule reward hacking by conditioning jointly on electronic target (`homo`)
and total atom budget (`n_atoms`), paired with:
  1. `train_sft`: Structured CFG (`75% [T,T] / 15% [F,T] / 10% [F,F]`) + size-matched contrastive
     margin (`contrast_col_mask = [True, False]`) on full OE62 (`47.8k` mols, `mean = 39.0` atoms).
  2. `align_v20`: Online v2.0 synthesizable GRPO with size-anchored CFG (`uncond = [False, True]`)
     and per-group size-adherence advantage (`size_adherence_weight = 0.85`).
  3. `train_critic`: Joint `[homo, n_atoms]` + legality + `SA<=3.5` `OctreeProcessCritic` training.
  4. `eval_all`: Held-out `1,792`-prompt benchmark (`7` HOMO quantiles x size-matched OE62 `n_atoms`)
     reporting `Outcome M={1,2,4,8}` and `Process {8->1, 8->2, 16->2, 16->4}`, including full
     atom-count distribution, size-stratified `RewSynth`/`BlindSynth`, uniqueness, novelty, and
     Murcko scaffold diversity.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
for cand in (ROOT / "src" / "compat", ROOT / "src", ROOT):
    cstr = str(cand)
    if cand.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402
from loom.generator.molecular_state import mol_to_clean_smiles  # noqa: E402
from loom.generator.octree_backend import OctreeDataset  # noqa: E402
from loom.oracles.unimol2_task import B_WEIGHT_SHA256  # noqa: E402
from loom.weaving import (  # noqa: E402
    AssetRegistry,
    AuditOracleSlot,
    BuiltInRuleAsset,
    CompositeCriticBus,
    LoomGRPOTrainer,
    LoomUni3DARGenerator,
    OctreeProcessCritic,
    PotentialSlot,
    RewardOracleSlot,
    SlotOracleAsset,
    TaskWeavingSpec,
    compute_loom_metrics,
    fit_octree_process_critic,
    formal_charge_signature,
)
from scripts.loom_train_sft import load_sdf_with_targets  # noqa: E402


SIZE_BINS: Tuple[Tuple[int, int], ...] = ((0, 15), (15, 25), (25, 35), (35, 45), (45, 999))

_ATOMIC_Z: Dict[str, float] = {
    "H": 1.0,
    "B": 5.0,
    "C": 6.0,
    "N": 7.0,
    "O": 8.0,
    "F": 9.0,
    "Si": 14.0,
    "P": 15.0,
    "S": 16.0,
    "Cl": 17.0,
    "As": 33.0,
    "Se": 34.0,
    "Br": 35.0,
    "I": 53.0,
}


def _canonicalize_coords_pca(symbols: Sequence[str], coords: np.ndarray) -> np.ndarray:
    """Deterministic SO(3) inertial + polarity frame (`det(R)=+1`, exact 0.000000 A internal geometry)."""
    pts = np.asarray(coords, dtype=np.float64)
    n = pts.shape[0]
    if n <= 2:
        return (pts - pts.mean(axis=0, keepdims=True)).astype(np.float32)

    centered = pts - pts.mean(axis=0, keepdims=True)
    z = np.asarray([_ATOMIC_Z.get(str(s), 6.0) for s in symbols], dtype=np.float64)
    cov = (centered * z[:, None]).T @ centered
    _, evecs = np.linalg.eigh(cov)

    v1 = evecs[:, 2].copy()
    v2 = evecs[:, 1].copy()
    dz = z - float(np.mean(z))
    for vk in (v1, v2):
        proj = centered @ vk
        pol = float(np.dot(dz, proj))
        if abs(pol) < 1e-4:
            pol = float(np.dot(z, proj**3))
        if pol < 0.0:
            vk *= -1.0

    v3 = np.cross(v1, v2)
    n3 = float(np.linalg.norm(v3))
    if n3 > 1e-8:
        v3 /= n3
    else:
        v3 = evecs[:, 0].copy()
        if np.linalg.det(np.stack([v1, v2, v3], axis=1)) < 0.0:
            v3 *= -1.0
    rot = np.stack([v1, v2, v3], axis=1)
    return (centered @ rot).astype(np.float32)


def _canonicalize_mol_pca(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of `mol` rotated into its canonical SO(3) inertial + polarity frame."""
    cm = Chem.Mol(mol)
    conf = cm.GetConformer()
    syms = [a.GetSymbol() for a in cm.GetAtoms()]
    pos = np.asarray(conf.GetPositions(), dtype=np.float64)
    can_pos = _canonicalize_coords_pca(syms, pos)
    for idx in range(cm.GetNumAtoms()):
        x, y, z = float(can_pos[idx, 0]), float(can_pos[idx, 1]), float(can_pos[idx, 2])
        conf.SetAtomPosition(idx, (x, y, z))
    return cm


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        core = MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(Chem.Mol(mol)))
        smi = Chem.MolToSmiles(core, canonical=True) if core is not None and core.GetNumAtoms() > 0 else ""
        return smi
    except Exception:
        return ""


def _load_mols_with_homo_and_natoms(sdf_path: Path) -> Tuple[List[Chem.Mol], np.ndarray]:
    """Load SDF molecules with `(N, 2)` targets `[:, 0] = homo (eV)`, `[:, 1] = total n_atoms`."""
    mols, homo_2d = load_sdf_with_targets(sdf_path, ["homo"])
    natoms = np.asarray([float(m.GetNumAtoms()) for m in mols], dtype=np.float32)
    targets = np.stack([homo_2d[:, 0].astype(np.float32), natoms], axis=-1)
    return mols, targets


def _sample_structured_sft_masks(batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    """Structured `[homo, n_atoms]` CFG mask: `75% [T,T]`, `15% [F,T]`, `10% [F,F]`."""
    u = rng.random(size=batch_size)
    mask = np.ones((batch_size, 2), dtype=bool)
    mask[(u >= 0.75) & (u < 0.90), 0] = False
    mask[u >= 0.90] = False
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def _build_oe62_bus(
    args: argparse.Namespace,
    *,
    include_blind_auditor: bool,
    seen_charges: Optional[frozenset] = None,
) -> CompositeCriticBus:
    spec = TaskWeavingSpec.from_file(args.oe62_task_spec)
    registry = AssetRegistry(register_builtin_rules=True)
    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.oe62_reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )
    registry.register_predictor(SlotOracleAsset(reward_slot, asset_id="ext_unimol2", units=["eV", "eV"]))
    registry.register_predictor(BuiltInRuleAsset(asset_id="builtin_rules"))
    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.oe62_potential_weights), device=args.device)
    registry.register_potential("mlip_dpa4", pot_slot)

    if include_blind_auditor:
        audit_slot = AuditOracleSlot(
            backend="dpa4",
            checkpoint_dir=args.oe62_audit_oracle_dir,
            device=args.device,
            forcefield_path=str(args.oe62_potential_weights),
        )
        registry.register_predictor(SlotOracleAsset(audit_slot, asset_id="audit_dpa4", units=["eV", "eV"]))

    if seen_charges is None:
        replay_mols_all, _ = load_sdf_with_targets(args.oe62_replay_sdf, ["homo", "lumo"])
        seen_charges = frozenset(formal_charge_signature(m) for m in replay_mols_all)
    return CompositeCriticBus(spec=spec, registry=registry, seen_charge_signatures=seen_charges)


def _get_clean_oe62_pool(
    args: argparse.Namespace,
    ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Chem.Mol], np.ndarray]:
    if ctx is not None and "clean_pool" in ctx:
        return ctx["clean_pool"]
    if ctx is not None and "master_data" in ctx:
        master_mols, master_targets = ctx["master_data"]
    else:
        master_mols, master_targets = _load_mols_with_homo_and_natoms(args.oe62_master_train_sdf)
        if ctx is not None:
            ctx["master_data"] = (master_mols, master_targets)
    clean_pool = _prefilter_clean_oe62_pool(master_mols, master_targets)
    if ctx is not None:
        ctx["clean_pool"] = clean_pool
    return clean_pool


def _get_source_novelty_sets(args: argparse.Namespace) -> Tuple[set[str], set[str]]:
    source_mols = [m for m in Chem.SDMolSupplier(str(args.oe62_source_train_sdf), removeHs=False) if m is not None]
    source_graphs = {mol_to_clean_smiles(m) for m in source_mols}
    source_scaffolds = {sc for sc in (_murcko_scaffold_smiles(m) for m in source_mols) if sc}
    return source_graphs, source_scaffolds


def _prefilter_clean_oe62_pool(
    mols: Sequence[Chem.Mol],
    targets_2d: np.ndarray,
    *,
    sa_cap: float = 3.5,
    mw_cap: float = 550.0,
) -> Tuple[List[Chem.Mol], np.ndarray]:
    """Filter OE62 molecules cleanly by RDKit rules (`SA <= 3.5`, `MW <= 550`, `SMARTS == 0`, clean charge)."""
    rule_asset = BuiltInRuleAsset(asset_id="builtin_rules_filter")
    valid_idx = [i for i, m in enumerate(mols) if is_clean_organic_charge_state(m)]
    cand_mols = [mols[i] for i in valid_idx]
    out = rule_asset.predict_mols(cand_mols, batch_size=256)
    sa = out["mean"][:, rule_asset.channel_index("sa_score")]
    mw = out["mean"][:, rule_asset.channel_index("mol_weight")]
    sm = out["mean"][:, rule_asset.channel_index("smarts_alert")]
    keep_local = [
        k for k in range(len(cand_mols)) if (sa[k] <= sa_cap) and (mw[k] <= mw_cap) and (sm[k] <= 0.0)
    ]
    kept_idx = [valid_idx[k] for k in keep_local]
    return [mols[i] for i in kept_idx], targets_2d[kept_idx]


def _stratified_sample_by_size(
    mols: Sequence[Chem.Mol],
    targets_2d: np.ndarray,
    *,
    max_total: int,
    seed: int = 2026,
) -> Tuple[List[Chem.Mol], np.ndarray]:
    """Subsample across OE62 size bins so small/medium/large scaffolds remain balanced."""
    if len(mols) <= max_total:
        return list(mols), targets_2d
    rng = np.random.default_rng(seed)
    nat = targets_2d[:, 1]
    bins = ((0, 22), (22, 32), (32, 42), (42, 55), (55, 999))
    chosen: List[int] = []
    per_bin = max(1, max_total // len(bins))
    for lo, hi in bins:
        ids = np.where((nat >= lo) & (nat < hi))[0]
        if len(ids) == 0:
            continue
        k = min(len(ids), per_bin)
        chosen.extend(rng.choice(ids, size=k, replace=False).tolist())
    rem = max_total - len(chosen)
    if rem > 0:
        pool = np.setdiff1d(np.arange(len(mols)), np.asarray(chosen, dtype=np.int64))
        if len(pool) > 0:
            chosen.extend(rng.choice(pool, size=min(rem, len(pool)), replace=False).tolist())
    rng.shuffle(chosen)
    chosen = chosen[:max_total]
    return [mols[i] for i in chosen], targets_2d[chosen]


def _sample_conditional_natoms(
    homo_vals: np.ndarray,
    ref_targets: np.ndarray,
    rng: np.random.Generator,
    *,
    window_eV: float = 0.30,
    clamp_range: Tuple[float, float] = (16.0, 68.0),
) -> np.ndarray:
    """Draw OE62-faithful `n_atoms` conditioned within `±window_eV` around requested `homo`."""
    ref_h = ref_targets[:, 0]
    ref_n = ref_targets[:, 1]
    out = np.empty(len(homo_vals), dtype=np.float32)
    lo_n, hi_n = clamp_range
    for i, hv in enumerate(homo_vals):
        near = np.where(np.abs(ref_h - float(hv)) <= window_eV)[0]
        if len(near) < 32:
            order = np.argsort(np.abs(ref_h - float(hv)))[:64]
            near = order
        pick = int(rng.choice(near))
        out[i] = float(np.clip(ref_n[pick], lo_n, hi_n))
    return out


def _paths(args: argparse.Namespace) -> Dict[str, Path]:
    run_root = args.work_dir / args.run_name
    return {
        "run_root": run_root,
        "sft_dir": run_root / "stage0_sft",
        "sft_ckpt": run_root / "stage0_sft" / "oe62_size_sft_best.pt",
        "v20_dir": run_root / "v20_aligned",
        "v20_ckpt": run_root / "v20_aligned" / "oe62_size_v20_aligned.pt",
        "critic_dir": run_root / "process_critic",
        "critic_ckpt": run_root / "process_critic" / "oe62_size_process_critic.pt",
        "eval_dir": run_root / "eval",
    }


def _safe_collate_indices(
    dataset: OctreeDataset,
    indices: Sequence[int],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """Collate `indices` reliably, resampling if an occasional rotation exceeds cutoff."""
    idx_list = [int(i) for i in indices]
    try:
        return dataset.collate(idx_list), np.asarray(idx_list, dtype=np.int64)
    except ValueError:
        local_rng = rng if rng is not None else np.random.default_rng(12345)
        batch_len = max(len(idx_list), 1)
        for attempt in range(8):
            k = max(8, batch_len // (2 if attempt >= 4 else 1))
            cand = local_rng.integers(0, len(dataset), size=min(k, len(dataset))).tolist()
            try:
                return dataset.collate(cand), np.asarray(cand, dtype=np.int64)
            except ValueError:
                continue
        return None, None


def _evaluate_sft_validation(
    gen: LoomUni3DARGenerator,
    val_ds: OctreeDataset,
    val_targets: np.ndarray,
    *,
    max_eval: int = 384,
    batch_size: int = 64,
) -> float:
    gen.eval()
    device = next(gen.parameters()).device
    n_use = min(max_eval, len(val_ds))
    losses: List[float] = []
    with torch.no_grad():
        for s in range(0, n_use, batch_size):
            e = min(s + batch_size, n_use)
            idx_list = list(range(s, e))
            try:
                sub_batches = [(val_ds.collate(idx_list), idx_list)]
            except ValueError:
                sub_batches = []
                for ss in range(s, e, 16):
                    sub_idx = list(range(ss, min(ss + 16, e)))
                    try:
                        sub_batches.append((val_ds.collate(sub_idx), sub_idx))
                    except ValueError:
                        continue
            for packed, kept in sub_batches:
                t_norm = gen.normalize_targets(val_targets[kept])
                m_act = torch.ones((len(kept), 2), dtype=torch.bool, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss, _ = gen.compute_sft_loss(packed, t_norm, m_act, cfg_dropout_prob=0.0, contrast_weight=0.0)
                if torch.isfinite(loss):
                    losses.append(float(loss.item()))
    gen.train()
    return float(np.mean(losses)) if losses else float("nan")


def _compute_clean_sa_flags(
    mols: Sequence[Chem.Mol],
    *,
    sa_cap: float = 3.5,
    mw_cap: float = 550.0,
) -> np.ndarray:
    """Return boolean mask `(len(mols),)` for `CleanCharge & SA <= sa_cap & MW <= mw_cap & SMARTS == 0`."""
    n = len(mols)
    flags = np.zeros(n, dtype=bool)
    valid_idx = [i for i, m in enumerate(mols) if m is not None and is_clean_organic_charge_state(m)]
    if not valid_idx:
        return flags
    rule_asset = BuiltInRuleAsset(asset_id="builtin_rules_flag")
    out = rule_asset.predict_mols([mols[i] for i in valid_idx], batch_size=256)
    sa = out["mean"][:, rule_asset.channel_index("sa_score")]
    mw = out["mean"][:, rule_asset.channel_index("mol_weight")]
    sm = out["mean"][:, rule_asset.channel_index("smarts_alert")]
    for loc_k, orig_i in enumerate(valid_idx):
        if (sa[loc_k] <= sa_cap) and (mw[loc_k] <= mw_cap) and (sm[loc_k] <= 0.0):
            flags[orig_i] = True
    return flags


def run_train_sft(args: argparse.Namespace, ctx: Optional[Dict[str, Any]] = None) -> None:
    torch.manual_seed(2026)
    np.random.seed(2026)
    device = torch.device(args.device)
    p = _paths(args)
    p["sft_dir"].mkdir(parents=True, exist_ok=True)
    v3_mode = bool(getattr(args, "v3_mode", False))

    if bool(args.sft_clean_only) and not v3_mode:
        train_mols, train_targets = _get_clean_oe62_pool(args, ctx)
        raw_val_mols, raw_val_targets = _load_mols_with_homo_and_natoms(args.oe62_master_valid_sdf)
        valid_mols, valid_targets = _prefilter_clean_oe62_pool(raw_val_mols, raw_val_targets)
    else:
        if ctx is not None and "master_data" in ctx:
            train_mols, train_targets = ctx["master_data"]
        else:
            train_mols, train_targets = _load_mols_with_homo_and_natoms(args.oe62_master_train_sdf)
            if ctx is not None:
                ctx["master_data"] = (train_mols, train_targets)
        raw_val_mols, raw_val_targets = _load_mols_with_homo_and_natoms(args.oe62_master_valid_sdf)
        if v3_mode:
            valid_mols, valid_targets = _prefilter_clean_oe62_pool(raw_val_mols, raw_val_targets)
        else:
            valid_mols, valid_targets = raw_val_mols, raw_val_targets

    sample_probs: Optional[np.ndarray] = None
    if v3_mode:
        t_pca0 = time.perf_counter()
        clean_flags = _compute_clean_sa_flags(train_mols)
        weights = np.where(clean_flags, 4.0, 1.0).astype(np.float64)
        # Mildly upweight >= 28-atom clean scaffolds so 30-55 atom synthesizability gets strong SFT support
        large_clean = clean_flags & (train_targets[:, 1] >= 28.0)
        weights[large_clean] = 5.5
        sample_probs = weights / weights.sum()
        train_mols = [_canonicalize_mol_pca(m) for m in train_mols]
        valid_mols = [_canonicalize_mol_pca(m) for m in valid_mols]
        print(
            f"[SFT-v3] Canonicalized {len(train_mols)} train + {len(valid_mols)} valid conformers in "
            f"{time.perf_counter() - t_pca0:.1f}s | clean_sa_frac={clean_flags.mean()*100:.1f}% "
            f"(effective batch clean share={(sample_probs[clean_flags].sum())*100:.1f}%)",
            flush=True,
        )

    t_mean = [float(train_targets[:, 0].mean()), float(train_targets[:, 1].mean())]
    t_scale = [float(max(train_targets[:, 0].std(), 1e-4)), float(max(train_targets[:, 1].std(), 1e-4))]
    print(
        f"[SFT] Loaded train={len(train_mols)} valid={len(valid_mols)} (v3={v3_mode}, clean_only={args.sft_clean_only}) | "
        f"homo=({t_mean[0]:.4f}±{t_scale[0]:.4f}) n_atoms=({t_mean[1]:.2f}±{t_scale[1]:.2f})",
        flush=True,
    )

    raw_parent = torch.load(args.parent_ckpt, map_location="cpu", weights_only=False)
    if (not v3_mode) and isinstance(raw_parent, dict) and raw_parent.get("backend") == LoomUni3DARGenerator.backend_id:
        gen, _ = LoomUni3DARGenerator.load_checkpoint(
            args.parent_ckpt, device=args.device, batch_size=args.sft_batch_size
        )
        if gen.num_props != 2:
            raise ValueError(f"Expected 2-prop warm-start Loom checkpoint, got target_cols={gen.target_cols}")
        gen.target_cols = ["homo", "n_atoms"]
        gen.target_units = ["eV", "atoms"]
        with torch.no_grad():
            gen.target_mean[1] = float(t_mean[1])
            gen.target_scale[1] = float(t_scale[1])
        for mod in gen.conditioner.prop_encoders[1].modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
        gen.enable_v20_valence_head()
        print(f"[SFT] Warm-started 2-col Size-Anchored generator from aligned v2.0 checkpoint: {args.parent_ckpt}", flush=True)
    else:
        gen = LoomUni3DARGenerator.from_parent_checkpoint(
            args.parent_ckpt,
            target_cols=["homo", "n_atoms"],
            target_units=["eV", "atoms"],
            mean=t_mean,
            scale=t_scale,
            device=args.device,
            batch_size=args.sft_batch_size,
            num_rbf=(48 if v3_mode else 32),
            num_freqs=(24 if v3_mode else 16),
            num_prefix_tokens=(4 if v3_mode else 0),
            lora_rank=(32 if v3_mode else 16),
            lora_alpha=(64.0 if v3_mode else 32.0),
            lora_include_ffn=True,
            lora_include_heads=True,
            lora_include_xyz_heads=v3_mode,
            train_xyz_embeddings=v3_mode,
            use_valence_head=True,
        )
    del raw_parent
    train_ds = OctreeDataset(train_mols, gen.backbone, training=(not v3_mode), seed=2026)
    valid_ds = OctreeDataset(valid_mols, gen.backbone, training=False, seed=2027)

    trainable_params = [param for param in gen.parameters() if param.requires_grad]
    n_trainable = sum(param.numel() for param in trainable_params)
    print(f"[SFT] Trainable parameters: {n_trainable:,} (v3={v3_mode})", flush=True)

    total_steps = int(args.sft_steps)
    opt = torch.optim.AdamW(trainable_params, lr=float(args.sft_lr), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=float(args.sft_lr) * 0.1)
    c_col_mask = torch.tensor([True, False], dtype=torch.bool, device=device)
    rng = np.random.default_rng(2026)

    best_val = _evaluate_sft_validation(gen, valid_ds, valid_targets, batch_size=args.sft_batch_size)
    print(f"[SFT step=0/{total_steps}] init_val_loss={best_val:.4f}", flush=True)
    gen.save_checkpoint(p["sft_ckpt"], extra_metadata={"sft_step": 0, "best_val_loss": best_val, "v3_mode": v3_mode})

    # Multi-worker async CPU octree collation prefetch so A800 steps run with zero CPU stalls
    n_workers = 4 if v3_mode else 1
    q_depth = 8 if v3_mode else 3

    def _collate_batch(step_seed: int) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        local_rng = np.random.default_rng(step_seed)
        if sample_probs is not None:
            cand = local_rng.choice(len(train_ds), size=args.sft_batch_size, replace=True, p=sample_probs)
        else:
            cand = local_rng.integers(0, len(train_ds), size=args.sft_batch_size)
        return _safe_collate_indices(train_ds, cand.tolist(), rng=local_rng)

    t0 = time.perf_counter()
    loss_win: List[float] = []
    hist: List[Dict[str, Any]] = []
    val_every = max(1, min(600 if v3_mode else 400, max(total_steps // 5, 1)))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        init_n = min(q_depth, total_steps)
        prefetch = [pool.submit(_collate_batch, 100000 + i) for i in range(init_n)]
        next_submit = init_n
        for step_idx in range(total_steps):
            slot = step_idx % len(prefetch)
            packed, kept = prefetch[slot].result()
            if next_submit < total_steps:
                prefetch[slot] = pool.submit(_collate_batch, 100000 + next_submit)
                next_submit += 1
            if packed is None or kept is None:
                sched.step()
                continue

            cur_b = len(kept)
            t_norm = gen.normalize_targets(train_targets[kept])
            m_act = _sample_structured_sft_masks(cur_b, rng, device)
            gen.train()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, _info = gen.compute_sft_loss(
                    packed,
                    t_norm,
                    m_act,
                    cfg_dropout_prob=0.0,
                    valence_loss_weight=0.25,
                    contrast_weight=float(args.sft_contrast_weight),
                    contrast_col_mask=c_col_mask,
                )
            if torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                loss_win.append(float(loss.detach().item()))
            sched.step()

            if ((step_idx + 1) % 200 == 0) or ((step_idx + 1) == total_steps):
                avg_l = float(np.mean(loss_win[-200:])) if loss_win else float("nan")
                elapsed = time.perf_counter() - t0
                print(
                    f"[SFT step={step_idx + 1}/{total_steps}] loss={avg_l:.4f} lr={sched.get_last_lr()[0]:.2e} ({elapsed:.1f}s)",
                    flush=True,
                )

            if ((step_idx + 1) % val_every == 0) or ((step_idx + 1) == total_steps):
                v_loss = _evaluate_sft_validation(gen, valid_ds, valid_targets, batch_size=args.sft_batch_size)
                rec = {"step": step_idx + 1, "val_loss": round(v_loss, 5), "wall_s": round(time.perf_counter() - t0, 1)}
                hist.append(rec)
                if math.isfinite(v_loss) and (not math.isfinite(best_val) or v_loss <= best_val):
                    best_val = v_loss
                    gen.save_checkpoint(
                        p["sft_ckpt"],
                        extra_metadata={"sft_step": step_idx + 1, "best_val_loss": best_val, "v3_mode": v3_mode},
                    )
                    print(f"[SFT SAVED BEST step={step_idx + 1}] val_loss={best_val:.4f}", flush=True)

    # Diagnostic size-adherence check at Stage 0 completion
    gen_best, _ = LoomUni3DARGenerator.load_checkpoint(p["sft_ckpt"], device=args.device, batch_size=64)
    gen_best.eval()
    val_probe = valid_targets[:64]
    tn_p = gen_best.normalize_targets(val_probe)
    ma_p = torch.ones((len(val_probe), 2), dtype=torch.bool, device=device)
    ua_p = torch.tensor([[False, True]] * len(val_probe), dtype=torch.bool, device=device)
    trajs_p = gen_best.sample(
        tn_p,
        ma_p,
        uncond_active_mask=ua_p,
        cfg_scale=1.8,
        cfg_tree_decay=0.30,
        geometric_guard=True,
        seed=99,
        use_bf16=(device.type == "cuda"),
        leaf_atom_budget=(val_probe[:, 1] if v3_mode else None),
    )
    reconciler = ValenceReconciler(allowed_elements=tuple(gen_best.args.allow_atoms.split(",")))
    probe_atoms = []
    probe_errs = []
    for k, tr in enumerate(trajs_p):
        r = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
        if r.reconciled_legal and r.mol is not None:
            na = float(r.mol.GetNumAtoms())
            probe_atoms.append(na)
            probe_errs.append(abs(na - float(val_probe[k, 1])))
    sft_meta = {
        "v3_mode": v3_mode,
        "best_val_loss": best_val,
        "sft_steps": total_steps,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "probe_legal": len(probe_atoms),
        "probe_mean_target_atoms": round(float(val_probe[:, 1].mean()), 2),
        "probe_mean_gen_atoms": round(float(np.mean(probe_atoms)), 2) if probe_atoms else None,
        "probe_natoms_mae": round(float(np.mean(probe_errs)), 2) if probe_errs else None,
        "val_history": hist,
    }
    (p["sft_dir"] / "sft_meta.json").write_text(json.dumps(sft_meta, indent=2) + "\n")
    print(f"[SFT DONE] {sft_meta}", flush=True)
    del gen, gen_best, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _relax_and_relabel_clean_pool(
    mols: Sequence[Chem.Mol],
    bus: CompositeCriticBus,
    *,
    n_steps: int = 28,
    max_fmax: float = 2.2,
    canonical_pca: bool = True,
) -> Tuple[List[Chem.Mol], np.ndarray]:
    """Relax `mols` with `PotentialSlot('dpa4')`, enforce graph preservation + `Fmax <= max_fmax`,
    canonicalize PCA frame, and relabel `(homo_unimol2, n_atoms)` strictly without `audit_dpa4`!
    """
    if not mols:
        return [], np.zeros((0, 2), dtype=np.float32)
    cand_mols = list(mols)
    if bus.potential_slot is not None and n_steps > 0:
        rel = bus.potential_slot.relax_molecules(
            cand_mols, n_steps=n_steps, step_size=0.008, max_atom_disp=0.03, chunk_size=64
        )
        kept_rel: List[Chem.Mol] = []
        for rm, gp, fm in zip(rel["relaxed_mols"], rel["graph_preserved"], rel["final_f_max_eV_A"]):
            if rm is not None and bool(gp) and np.isfinite(fm) and float(fm) <= max_fmax:
                kept_rel.append(_canonicalize_mol_pca(rm) if canonical_pca else rm)
        cand_mols = kept_rel
    elif canonical_pca:
        cand_mols = [_canonicalize_mol_pca(m) for m in cand_mols]

    if not cand_mols:
        return [], np.zeros((0, 2), dtype=np.float32)

    admitted_mols, homo_mat, _ = bus.relabel_conformer_pool(cand_mols, max_fmax_eV_A=max_fmax + 0.3, batch_size=64)
    if not admitted_mols:
        return [], np.zeros((0, 2), dtype=np.float32)
    obj_idx = bus.spec.objective_names.index("homo")
    nat = np.asarray([float(m.GetNumAtoms()) for m in admitted_mols], dtype=np.float32)
    t_out = np.stack([homo_mat[:, obj_idx].astype(np.float32), nat], axis=-1)
    return admitted_mols, t_out


def _run_v3_online_hindsight_awr(
    gen: LoomUni3DARGenerator,
    bus: CompositeCriticBus,
    anchor_mols: List[Chem.Mol],
    anchor_targets: np.ndarray,
    clean_ref_targets: np.ndarray,
    *,
    rounds: int,
    proposals_per_round: int,
    steps_per_round: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Tuple[List[Chem.Mol], np.ndarray, List[Dict[str, Any]]]:
    """Online Clean-SA Hindsight + Advantage-Weighted SFT (`AWR`) strictly using `ext_unimol2 + rules + dpa4_relax`."""
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    rule_asset = bus.registry.get_predictor("builtin_rules")
    seen_smis = {mol_to_clean_smiles(m) for m in anchor_mols}
    disc_mols: List[Chem.Mol] = []
    disc_targets_list: List[np.ndarray] = []
    disc_weights_list: List[float] = []
    round_logs: List[Dict[str, Any]] = []

    h_p04, h_p96 = np.percentile(clean_ref_targets[:, 0], [4.0, 96.0])
    rng = np.random.default_rng(202633)
    c_col_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

    for r_idx in range(rounds):
        tr0 = time.perf_counter()
        # Oversample [28..56] atoms (72%) where synthesis compliance is hardest
        n_prop = int(proposals_per_round)
        req_h = rng.uniform(h_p04, h_p96, size=n_prop).astype(np.float32)
        req_n = _sample_conditional_natoms(req_h, clean_ref_targets, rng, window_eV=0.30, clamp_range=(18.0, 60.0))
        large_sel = rng.random(size=n_prop) < 0.65
        req_n[large_sel] = np.maximum(req_n[large_sel], rng.uniform(28.0, 54.0, size=int(large_sel.sum())).astype(np.float32))
        prompts = np.stack([req_h, req_n], axis=-1)

        gen.eval()
        cand_clean_mols: List[Chem.Mol] = []
        cand_req_rows: List[np.ndarray] = []
        gen_bs = 512 if device.type == "cuda" else 64
        for s in range(0, n_prop, gen_bs):
            e = min(s + gen_bs, n_prop)
            sub_p = prompts[s:e]
            tn = gen.normalize_targets(sub_p)
            ma = torch.ones((len(sub_p), 2), dtype=torch.bool, device=device)
            ua = torch.tensor([[False, True]] * len(sub_p), dtype=torch.bool, device=device)
            trajs = gen.sample(
                tn,
                ma,
                uncond_active_mask=ua,
                cfg_scale=1.95,
                cfg_tree_decay=0.30,
                geometric_guard=True,
                temperatures=(0.88, 0.32, 0.25),
                seed=810000 + r_idx * 50000 + s,
                use_bf16=(device.type == "cuda"),
                leaf_atom_budget=sub_p[:, 1],
                leaf_budget_tol=2.5,
                sort_by_budget=True,
            )
            for loc_i, tr in enumerate(trajs):
                res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
                if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
                    cand_clean_mols.append(res.mol)
                    cand_req_rows.append(sub_p[loc_i])

        # Filter strictly by fast 2D rules BEFORE MLIP relaxation
        if cand_clean_mols:
            r_out = rule_asset.predict_mols(cand_clean_mols, batch_size=256)
            sa_v = r_out["mean"][:, rule_asset.channel_index("sa_score")]
            mw_v = r_out["mean"][:, rule_asset.channel_index("mol_weight")]
            sm_v = r_out["mean"][:, rule_asset.channel_index("smarts_alert")]
            pre_idx = [
                k for k in range(len(cand_clean_mols))
                if (sa_v[k] <= 3.5) and (mw_v[k] <= 550.0) and (sm_v[k] <= 0.0)
            ]
        else:
            pre_idx = []

        synth_cands = [cand_clean_mols[k] for k in pre_idx]
        synth_reqs = np.stack([cand_req_rows[k] for k in pre_idx], axis=0) if pre_idx else np.zeros((0, 2), dtype=np.float32)

        rel_mols, rel_targets = _relax_and_relabel_clean_pool(
            synth_cands, bus, n_steps=28, max_fmax=2.2, canonical_pca=True
        )
        added_r = 0
        # Match back by graph or add all unique novel relaxed synthesizable conformers
        for m_rel, t_rel in zip(rel_mols, rel_targets):
            smi = mol_to_clean_smiles(m_rel)
            if not smi or smi in seen_smis:
                continue
            seen_smis.add(smi)
            disc_mols.append(m_rel)
            disc_targets_list.append(t_rel)
            # Upweight 30+ atom synthesizable discoveries for AWR
            w_m = 2.5 if float(t_rel[1]) >= 30.0 else 1.3
            disc_weights_list.append(w_m)
            added_r += 1

        # Build mixed AWR SFT dataset (`anchor + discovered`)
        comb_mols = list(anchor_mols) + list(disc_mols)
        if disc_targets_list:
            comb_targets = np.concatenate([anchor_targets, np.stack(disc_targets_list, axis=0)], axis=0)
            w_anc = np.where(anchor_targets[:, 1] >= 30.0, 1.4, 1.0).astype(np.float64)
            w_dis = np.asarray(disc_weights_list, dtype=np.float64) * (len(anchor_mols) / max(len(disc_mols), 1)) * 1.1
            w_all = np.concatenate([w_anc, np.clip(w_dis, 1.0, 8.0)], axis=0)
        else:
            comb_targets = anchor_targets
            w_all = np.where(anchor_targets[:, 1] >= 30.0, 1.4, 1.0).astype(np.float64)
        p_all = w_all / w_all.sum()

        awr_ds = OctreeDataset(comb_mols, gen.backbone, training=False, seed=3030 + r_idx)
        trainable = gen.configure_trainable()
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps_per_round, 1), eta_min=lr * 0.2)

        def _col_awr(sd: int) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
            lrng = np.random.default_rng(sd)
            c_idx = lrng.choice(len(awr_ds), size=batch_size, replace=True, p=p_all)
            return _safe_collate_indices(awr_ds, c_idx.tolist(), rng=lrng)

        losses_r: List[float] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            q_n = min(6, steps_per_round)
            futs = [pool.submit(_col_awr, 500000 + r_idx * 10000 + i) for i in range(q_n)]
            nxt = q_n
            for st_i in range(steps_per_round):
                sl = st_i % len(futs)
                pk, kp = futs[sl].result()
                if nxt < steps_per_round:
                    futs[sl] = pool.submit(_col_awr, 500000 + r_idx * 10000 + nxt)
                    nxt += 1
                if pk is None or kp is None:
                    sched.step()
                    continue
                tn_b = gen.normalize_targets(comb_targets[kp])
                ma_b = _sample_structured_sft_masks(len(kp), rng, device)
                gen.train()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss, _ = gen.compute_sft_loss(
                        pk,
                        tn_b,
                        ma_b,
                        cfg_dropout_prob=0.0,
                        valence_loss_weight=0.25,
                        contrast_weight=0.20,
                        contrast_col_mask=c_col_mask,
                    )
                if torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    losses_r.append(float(loss.detach().item()))
                sched.step()

        r_sec = round(time.perf_counter() - tr0, 2)
        r_info = {
            "round": r_idx + 1,
            "proposals": n_prop,
            "clean_legal_count": len(cand_clean_mols),
            "pre_sa35_count": len(synth_cands),
            "relaxed_added_unique": added_r,
            "total_discovered_pool": len(disc_mols),
            "mean_awr_loss": round(float(np.mean(losses_r)), 4) if losses_r else None,
            "wall_sec": r_sec,
        }
        round_logs.append(r_info)
        print(f"[Size-v3 Hindsight/AWR Round {r_idx + 1}/{rounds}] {r_info}", flush=True)

    final_mols = list(anchor_mols) + list(disc_mols)
    final_targets = (
        np.concatenate([anchor_targets, np.stack(disc_targets_list, axis=0)], axis=0)
        if disc_targets_list
        else anchor_targets
    )
    return final_mols, final_targets, round_logs


def _run_v3_detached_xyz_distillation(
    gen: LoomUni3DARGenerator,
    equil_mols: Sequence[Chem.Mol],
    equil_targets: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Stage 2.5B: Freeze 100% of Transformer, conditioner, tree_heads & element_head;
    distill ONLY sub-voxel `0.01 A` `X, Y, Z` heads (`pred_heads[1..3]` + `emb_layers[0..2]`)
    on `DPA-4`-relaxed equilibrium 30-60 atom conformers (`hidden.detach()`).
    """
    from loom.vendor.uni3dar.data.atom_data import atom_list
    import torch.nn.functional as F

    t0 = time.perf_counter()
    gen.enable_v11_xyz_heads(rank=32, alpha=64.0, train_xyz_embeddings=True)
    gen.requires_grad_(False)
    xyz_params: List[torch.nn.Parameter] = []
    ahead = gen.backbone.atom_heads[0]
    for ax_idx in (1, 2, 3):
        for p in ahead.pred_heads[ax_idx].parameters():
            p.requires_grad_(True)
            xyz_params.append(p)
    for emb_idx in (0, 1, 2):
        for p in ahead.emb_layers[emb_idx].parameters():
            p.requires_grad_(True)
            xyz_params.append(p)

    n_xyz = sum(p.numel() for p in xyz_params)
    print(f"[Stage-2.5B XYZ-Distill] Detached sub-voxel equilibrium training: {n_xyz:,} params on {len(equil_mols)} mols", flush=True)

    ds = OctreeDataset(list(equil_mols), gen.backbone, training=False, seed=202699)
    opt = torch.optim.AdamW(xyz_params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
    rng = np.random.default_rng(202699)

    def _col_xyz(sd: int) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        lrng = np.random.default_rng(sd)
        c_idx = lrng.integers(0, len(ds), size=batch_size)
        return _safe_collate_indices(ds, c_idx.tolist(), rng=lrng)

    init_loss: Optional[float] = None
    losses: List[float] = []
    gen.eval()  # Keep backbone dropout off and merged eval weights active during detached hidden extraction!

    with ThreadPoolExecutor(max_workers=4) as pool:
        qn = min(6, steps)
        futs = [pool.submit(_col_xyz, 900000 + i) for i in range(qn)]
        nxt = qn
        for st in range(1, steps + 1):
            sl = (st - 1) % len(futs)
            pk, kp = futs[sl].result()
            if nxt < steps:
                futs[sl] = pool.submit(_col_xyz, 900000 + nxt)
                nxt += 1
            if pk is None or kp is None:
                sched.step()
                continue

            tn = gen.normalize_targets(equil_targets[kp])
            ma = _sample_structured_sft_masks(len(kp), rng, device)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in pk.items()}
            bd["decoder_phy_pos"] = bd["decoder_phy_pos"].float()

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                dec_in, tgt, _ = gen.backbone.get_feat_from_dataloader(bd)
                cu = bd["decoder_input_len"].to(device, dtype=torch.int32)
                max_l = int(bd["decoder_input_max_len"].item())
                hid_atom = gen.forward_sequence(dec_in, cu, max_l, tn, ma)[tgt.is_gen_atom].detach()

            labels_a = [v.long() for v in tgt.to_list(tgt.is_gen_atom)]
            real_atom = (labels_a[0] >= 0) & (labels_a[0] < len(atom_list))
            null_atom = ~real_atom
            if not bool(real_atom.any().item()):
                sched.step()
                continue

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits_a = ahead(hid_atom, labels_a)
                loss_real = (
                    F.cross_entropy(logits_a[1].float()[real_atom], labels_a[1][real_atom])
                    + F.cross_entropy(logits_a[2].float()[real_atom], labels_a[2][real_atom])
                    + F.cross_entropy(logits_a[3].float()[real_atom], labels_a[3][real_atom])
                ) / 3.0
                if bool(null_atom.any().item()):
                    loss_null = (
                        F.cross_entropy(logits_a[1].float()[null_atom], labels_a[1][null_atom])
                        + F.cross_entropy(logits_a[2].float()[null_atom], labels_a[2][null_atom])
                        + F.cross_entropy(logits_a[3].float()[null_atom], labels_a[3][null_atom])
                    ) / 3.0
                    loss_xyz = loss_real + 0.25 * loss_null
                else:
                    loss_xyz = loss_real

            if torch.isfinite(loss_xyz):
                opt.zero_grad(set_to_none=True)
                loss_xyz.backward()
                torch.nn.utils.clip_grad_norm_(xyz_params, 1.0)
                opt.step()
                lv = float(loss_real.detach().item())
                if init_loss is None:
                    init_loss = lv
                losses.append(lv)
            sched.step()

            if (st % 200 == 0) or (st == steps):
                avg_l = float(np.mean(losses[-200:])) if losses else float("nan")
                print(
                    f"[XYZ-Distill {st:04d}/{steps:04d}] xyz_ce={avg_l:.4f} (init={init_loss}) ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )

    gen.configure_trainable()
    gen.eval()
    return {
        "steps": steps,
        "pool_size": len(equil_mols),
        "init_xyz_ce": round(float(init_loss), 4) if init_loss is not None else None,
        "final_xyz_ce": round(float(np.mean(losses[-100:])), 4) if losses else None,
        "wall_sec": round(time.perf_counter() - t0, 2),
    }


def _select_best_leaf_budget_tol_on_dev(
    gen: LoomUni3DARGenerator,
    bus: CompositeCriticBus,
    valid_targets: np.ndarray,
    device: torch.device,
) -> Optional[float]:
    """Select optimal `leaf_budget_tol` (`None` vs `3.0` vs `2.0`) strictly on held-out dev prompts (`0` audit access)."""
    probe = valid_targets[:128]
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    obj_idx = bus.spec.objective_names.index("homo")
    req_mat = np.zeros((len(probe), len(bus.spec.objectives)), dtype=np.float64)
    req_mat[:, obj_idx] = probe[:, 0].astype(np.float64)
    act_mat = np.zeros((len(probe), len(bus.spec.objectives)), dtype=bool)
    act_mat[:, obj_idx] = True

    tn = gen.normalize_targets(probe)
    ma = torch.ones((len(probe), 2), dtype=torch.bool, device=device)
    ua = torch.tensor([[False, True]] * len(probe), dtype=torch.bool, device=device)

    best_tol: Optional[float] = None
    best_score = -1e9
    for cand_tol in (None, 3.0, 2.0):
        trajs = gen.sample(
            tn,
            ma,
            uncond_active_mask=ua,
            cfg_scale=2.00,
            cfg_tree_decay=0.30,
            geometric_guard=True,
            seed=777,
            use_bf16=(device.type == "cuda"),
            leaf_atom_budget=(probe[:, 1] if cand_tol is not None else None),
            leaf_budget_tol=(float(cand_tol) if cand_tol is not None else 2.0),
        )
        mols_c: List[Optional[Chem.Mol]] = []
        errs_n: List[float] = []
        for k, tr in enumerate(trajs):
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
                mols_c.append(res.mol)
                errs_n.append(abs(float(res.mol.GetNumAtoms()) - float(probe[k, 1])))
            else:
                mols_c.append(None)
        ev = bus.evaluate_molecules(
            mols_c, requested_targets=req_mat, active_mask=act_mat, include_audit=False, for_training=False, batch_size=64
        )
        hp = ev["objective_preds"][:, obj_idx]
        cm = ev["constraints_met"].astype(bool)
        clean_ok = np.asarray([m is not None for m in mols_c], dtype=bool) & np.isfinite(hp)
        synth_hits = int((clean_ok & cm & (np.abs(hp - probe[:, 0]) <= 0.25)).sum())
        n_mae = float(np.mean(errs_n)) if errs_n else 99.0
        # Reward synthesizable hits while penalizing atom-count drift > 3.5 atoms
        sc = float(synth_hits) / len(probe) - 0.015 * max(0.0, n_mae - 3.0)
        print(f"[Dev Budget Probe tol={cand_tol}] synth_hits={synth_hits}/{len(probe)} natoms_mae={n_mae:.2f} score={sc:.4f}", flush=True)
        if sc > best_score:
            best_score = sc
            best_tol = cand_tol
    print(f"[Dev Budget Selected] leaf_budget_tol={best_tol}", flush=True)
    return best_tol


def run_align_v20(args: argparse.Namespace, ctx: Optional[Dict[str, Any]] = None) -> None:
    torch.manual_seed(2026)
    np.random.seed(2026)
    device = torch.device(args.device)
    p = _paths(args)
    p["v20_dir"].mkdir(parents=True, exist_ok=True)
    v3_mode = bool(getattr(args, "v3_mode", False))

    # Strictly NO audit oracle during alignment!
    if ctx is not None and "train_bus" in ctx:
        bus = ctx["train_bus"]
    else:
        bus = _build_oe62_bus(args, include_blind_auditor=False)
        if ctx is not None:
            ctx["train_bus"] = bus
    clean_mols, clean_targets = _get_clean_oe62_pool(args, ctx)
    print(
        f"[Size-v2.0] Clean SA<=3.5 OE62 pool: {len(clean_mols)} | mean_atoms={clean_targets[:, 1].mean():.2f} (v3={v3_mode})",
        flush=True,
    )

    t0 = time.perf_counter()
    strat_mols, _ = _stratified_sample_by_size(clean_mols, clean_targets, max_total=args.replay_pool_cap, seed=2026)
    if v3_mode:
        replay_mols, replay_targets = _relax_and_relabel_clean_pool(
            strat_mols, bus, n_steps=28, max_fmax=2.4, canonical_pca=True
        )
        relabel_summary: Dict[str, Any] = {
            "input_pool_size": len(strat_mols),
            "dpa4_relaxed_admitted_count": len(replay_mols),
            "admitted_rate": round(len(replay_mols) / max(len(strat_mols), 1), 4),
        }
        replay_natoms = replay_targets[:, 1]
    else:
        replay_mols, replay_homo_1d, relabel_summary = bus.relabel_conformer_pool(strat_mols, max_fmax_eV_A=4.5)
        replay_natoms = np.asarray([float(m.GetNumAtoms()) for m in replay_mols], dtype=np.float32)
        replay_targets = np.stack([replay_homo_1d[:, 0].astype(np.float32), replay_natoms], axis=-1)
    print(
        f"[Size-v2.0] Replay admission: {relabel_summary} | replay_mean_atoms={replay_natoms.mean():.2f}",
        flush=True,
    )

    gen, _ = LoomUni3DARGenerator.load_checkpoint(p["sft_ckpt"], device=args.device, batch_size=args.eval_batch_size)
    gen.enable_v20_valence_head()
    if (not v3_mode) and float(args.relax_distill_weight) > 0.0 and not bool(gen.loom_config.get("lora_include_xyz_heads", False)):
        gen.enable_v11_xyz_heads()

    hindsight_logs: List[Dict[str, Any]] = []
    equil_mols = list(replay_mols)
    equil_targets = replay_targets
    if v3_mode and int(args.v3_hindsight_rounds) > 0:
        equil_mols, equil_targets, hindsight_logs = _run_v3_online_hindsight_awr(
            gen,
            bus,
            replay_mols,
            replay_targets,
            clean_targets,
            rounds=int(args.v3_hindsight_rounds),
            proposals_per_round=int(args.v3_hindsight_mols),
            steps_per_round=int(args.v3_hindsight_steps),
            batch_size=int(args.sft_batch_size),
            lr=8.0e-5,
            device=device,
        )

    h_p04, h_p96 = np.percentile(clean_targets[:, 0], [4.0, 96.0])
    n_emp = len(equil_targets)

    def _sample_prompts(step_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        del step_idx
        k_p = int(args.prompts_per_step)
        t_mat = np.zeros((k_p, 2), dtype=np.float32)
        for j in range(k_p):
            if j % 2 == 0:
                idx = int(rng.integers(0, n_emp))
                th = float(equil_targets[idx, 0] + rng.normal(0.0, 0.08))
                tn = float(np.clip(equil_targets[idx, 1] + rng.integers(-2, 3), 16.0, 68.0))
            else:
                th = float(rng.uniform(h_p04, h_p96))
                tn = float(_sample_conditional_natoms(np.array([th], dtype=np.float32), clean_targets, rng)[0])
            t_mat[j, 0] = th
            t_mat[j, 1] = tn
        mask = np.ones((k_p, 2), dtype=bool)
        return t_mat, mask

    t_grpo0 = time.perf_counter()
    trainer = LoomGRPOTrainer(
        generator=gen,
        critic_bus=bus,
        replay_molecules=equil_mols,
        replay_targets_eV=equil_targets,
        lr=(1.2e-5 if v3_mode else 1.5e-5),
        kl_beta=(0.04 if v3_mode else 0.05),
        replay_weight=(0.22 if v3_mode else 0.18),
        replay_batch_size=(24 if v3_mode else 16),
        w_valid=0.80,
        w_phys=float(args.w_phys),
        w_prop=(1.15 if v3_mode else 1.05),
        sharp_concentration=True,
        cfg_scale=2.00,
        cfg_tree_decay=0.30,
        geometric_guard=True,
        topology_only_policy=True,
        relax_distill_weight=(0.0 if v3_mode else float(args.relax_distill_weight)),
        relax_distill_steps=int(args.relax_distill_steps),
        relax_distill_max_mols=int(args.relax_distill_max_mols),
        relax_distill_max_fmax=float(args.relax_distill_max_fmax),
        size_anchored_cfg=True,
        size_col_name="n_atoms",
        size_adherence_weight=args.size_adherence_weight,
        size_tolerance_atoms=4.0,
        size_tolerance_frac=0.12,
        seed=2026,
    )
    grpo_hist = trainer.train_loop(
        total_steps=int(args.grpo_steps),
        prompt_sampler_fn=_sample_prompts,
        group_size=int(args.group_size),
        log_every=max(1, min(10, int(args.grpo_steps))),
    )
    grpo_sec = round(time.perf_counter() - t_grpo0, 2)

    xyz_distill_meta: Optional[Dict[str, Any]] = None
    selected_leaf_tol: Optional[float] = None
    if v3_mode:
        if int(args.v3_xyz_distill_steps) > 0:
            xyz_distill_meta = _run_v3_detached_xyz_distillation(
                gen,
                equil_mols,
                equil_targets,
                steps=int(args.v3_xyz_distill_steps),
                batch_size=int(args.sft_batch_size),
                lr=1.5e-4,
                device=device,
            )
        raw_val_mols, raw_val_targets = _load_mols_with_homo_and_natoms(args.oe62_master_valid_sdf)
        _, clean_val_targets = _prefilter_clean_oe62_pool(raw_val_mols, raw_val_targets)
        selected_leaf_tol = _select_best_leaf_budget_tol_on_dev(gen, bus, clean_val_targets, device)

    align_sec = time.perf_counter() - t0
    gen.save_checkpoint(
        p["v20_ckpt"],
        extra_metadata={
            "version": ("3.0.0-size-anchored-a800" if v3_mode else "2.0.0-size-anchored"),
            "v3_mode": v3_mode,
            "selected_leaf_budget_tol": selected_leaf_tol,
            "task_spec": bus.spec.to_dict(),
            "groups_completed": int(args.grpo_steps) * int(args.prompts_per_step),
            "align_wall_seconds": round(align_sec, 2),
        },
    )
    meta = {
        "v3_mode": v3_mode,
        "selected_leaf_budget_tol": selected_leaf_tol,
        "init_sft_ckpt": str(p["sft_ckpt"]),
        "saved_v20_ckpt": str(p["v20_ckpt"]),
        "replay_admission": relabel_summary,
        "replay_mean_atoms": round(float(equil_targets[:, 1].mean()), 2),
        "equil_pool_total": len(equil_mols),
        "hindsight_rounds": hindsight_logs,
        "grpo_wall_seconds": grpo_sec,
        "xyz_distill": xyz_distill_meta,
        "align_wall_seconds": round(align_sec, 2),
        "grpo_history_tail": grpo_hist[-8:],
    }
    (p["v20_dir"] / "v20_align_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[Size-v2.0 ALIGN DONE] Saved {p['v20_ckpt']} ({align_sec:.1f}s)", flush=True)
    del gen, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_train_critic(args: argparse.Namespace, ctx: Optional[Dict[str, Any]] = None) -> None:
    device = torch.device(args.device)
    p = _paths(args)
    p["critic_dir"].mkdir(parents=True, exist_ok=True)

    # Strictly NO audit oracle during ProcessCritic training!
    if ctx is not None and "train_bus" in ctx:
        bus = ctx["train_bus"]
    else:
        bus = _build_oe62_bus(args, include_blind_auditor=False)
        if ctx is not None:
            ctx["train_bus"] = bus
    gen, v20_payload = LoomUni3DARGenerator.load_checkpoint(p["v20_ckpt"], device=args.device, batch_size=args.eval_batch_size)
    gen.eval()
    v20_meta = v20_payload.get("metadata") or {}
    leaf_tol: Optional[float] = (
        None
        if bool(getattr(args, "disable_eval_leaf_budget", False))
        else (
            float(args.eval_leaf_budget_tol)
            if getattr(args, "eval_leaf_budget_tol", None) is not None
            else v20_meta.get("selected_leaf_budget_tol")
        )
    )

    _, clean_targets = _get_clean_oe62_pool(args, ctx)

    h_p04, h_p96 = np.percentile(clean_targets[:, 0], [4.0, 96.0])
    rng = np.random.default_rng(202619)
    n_rollouts = int(args.critic_rollouts)
    half = n_rollouts // 2
    emp_idx = rng.choice(len(clean_targets), size=half, replace=(half > len(clean_targets)))
    emp_goals = clean_targets[emp_idx].astype(np.float32)
    uni_h = rng.uniform(h_p04, h_p96, size=n_rollouts - half).astype(np.float32)
    uni_n = _sample_conditional_natoms(uni_h, clean_targets, rng)
    goals = np.concatenate([emp_goals, np.stack([uni_h, uni_n], axis=-1)], axis=0)
    rng.shuffle(goals, axis=0)

    h_idx = gen.target_cols.index("homo")
    n_idx = gen.target_cols.index("n_atoms")
    obj_idx = bus.spec.objective_names.index("homo")
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))

    all_h: List[torch.Tensor] = []
    all_geom: List[torch.Tensor] = []
    all_t_norm: List[torch.Tensor] = []
    all_m_act: List[torch.Tensor] = []
    rec_mols: List[Optional[Chem.Mol]] = []
    rec_clean: List[bool] = []
    rec_natoms = np.zeros(n_rollouts, dtype=np.float32)

    t0 = time.perf_counter()
    for s in range(0, n_rollouts, args.eval_batch_size):
        e = min(s + args.eval_batch_size, n_rollouts)
        cb = e - s
        t_mat = goals[s:e]
        t_n = gen.normalize_targets(t_mat)
        m_a = torch.ones((cb, 2), dtype=torch.bool, device=device)
        u_a = torch.tensor([[False, True]] * cb, dtype=torch.bool, device=device)
        trajs = gen.sample(
            t_n,
            m_a,
            uncond_active_mask=u_a,
            cfg_scale=2.00,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.30,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=550000 + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            return_coarse_summary=True,
            leaf_atom_budget=(t_mat[:, n_idx] if leaf_tol is not None else None),
            leaf_budget_tol=(float(leaf_tol) if leaf_tol is not None else 2.0),
        )
        all_t_norm.append(t_n.detach().cpu())
        all_m_act.append(m_a.detach().cpu())
        for loc_i, tr in enumerate(trajs):
            g_i = s + loc_i
            all_h.append(tr["coarse_h_concat"])
            all_geom.append(tr["coarse_geom_feats"])
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and res.mol is not None:
                rec_mols.append(res.mol)
                rec_clean.append(not bool(res.reconciled_exotic_charge))
                rec_natoms[g_i] = float(res.mol.GetNumAtoms())
            else:
                rec_mols.append(None)
                rec_clean.append(False)

        if (e % 1024 == 0) or (e == n_rollouts):
            print(f"[Size ProcCritic Rollout] {e}/{n_rollouts} ({time.perf_counter() - t0:.1f}s)", flush=True)

    req_mat = np.zeros((n_rollouts, len(bus.spec.objectives)), dtype=np.float64)
    req_mat[:, obj_idx] = goals[:, h_idx].astype(np.float64)
    act_mat = np.zeros((n_rollouts, len(bus.spec.objectives)), dtype=bool)
    act_mat[:, obj_idx] = True

    ev = bus.evaluate_molecules(
        rec_mols, requested_targets=req_mat, active_mask=act_mat, include_audit=False, for_training=False, batch_size=64
    )
    homo_pred = ev["objective_preds"][:, obj_idx].astype(np.float32)
    constr_met = ev["constraints_met"].astype(bool)

    y_full = np.zeros((n_rollouts, 2), dtype=np.float32)
    y_full[:, h_idx] = np.where(np.isfinite(homo_pred), homo_pred, 0.0)
    y_full[:, n_idx] = rec_natoms

    critic = OctreeProcessCritic(
        emb_dim=int(gen.args.emb_dim), num_targets=2, hidden_dim=512, dropout=0.10
    ).to(device)
    critic.set_normalization(gen.target_mean.detach().cpu(), gen.target_scale.detach().cpu())

    clean_arr = np.asarray(rec_clean, dtype=bool)
    valid_y = clean_arr & np.isfinite(homo_pred)
    clean_and_synth = clean_arr & constr_met

    fit_t0 = time.perf_counter()
    stats = fit_octree_process_critic(
        critic,
        h_concat=torch.stack(all_h, dim=0),
        geom_feats=torch.stack(all_geom, dim=0),
        targets_norm=torch.cat(all_t_norm, dim=0),
        active_mask=torch.cat(all_m_act, dim=0),
        y_surrogate_norm=critic.normalize_y(torch.from_numpy(y_full)).detach().cpu(),
        y_valid_mask=torch.from_numpy(valid_y),
        is_legal=torch.from_numpy(clean_arr),
        is_constr_met=torch.from_numpy(clean_and_synth),
        epochs=24,
        batch_size=256,
        lr=1.5e-3,
        seed=2026,
    )
    fit_sec = time.perf_counter() - fit_t0
    meta = {
        "total_rollouts": n_rollouts,
        "selected_leaf_budget_tol": leaf_tol,
        "clean_rate": round(float(clean_arr.mean()), 4),
        "clean_and_synthesizable_rate": round(float(clean_and_synth.mean()), 4),
        "rollout_mean_clean_atoms": round(float(rec_natoms[clean_arr].mean()), 2) if clean_arr.any() else None,
        "critic_fit_wall_sec": round(fit_sec, 2),
        "total_wall_sec": round(time.perf_counter() - t0, 2),
        "train_stats": stats.__dict__,
    }
    critic.save_checkpoint(p["critic_ckpt"], extra=meta)
    (p["critic_dir"] / "process_critic_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[Size ProcCritic TRAIN DONE] Saved {p['critic_ckpt']} | meta={meta}", flush=True)


def _select_oe62_size_from_pool(
    mols_grid: List[List[Optional[Chem.Mol]]],
    legal_grid: np.ndarray,
    exotic_grid: np.ndarray,
    req_targets: np.ndarray,
    bus: CompositeCriticBus,
    m_keep: int,
    use_surrogate: bool,
    device: torch.device,
) -> Tuple[List[Optional[Chem.Mol]], np.ndarray, np.ndarray, float]:
    n_prompts = len(req_targets)
    if not use_surrogate or m_keep == 1:
        sel_m = [mols_grid[i][0] for i in range(n_prompts)]
        return sel_m, legal_grid[:, 0].copy(), exotic_grid[:, 0].copy(), 0.0

    flat_mols = [mols_grid[i][k] for i in range(n_prompts) for k in range(m_keep)]
    _sync(device)
    ts0 = time.perf_counter()
    pred_asset = bus.registry.get_predictor("ext_unimol2")
    rule_asset = bus.registry.get_predictor("builtin_rules")
    u_out = pred_asset.predict_mols(flat_mols, batch_size=64)
    r_out = rule_asset.predict_mols(flat_mols, batch_size=256)
    _sync(device)
    surr_wall = time.perf_counter() - ts0

    homo_p = u_out["mean"][:, pred_asset.channel_index("homo")].reshape(n_prompts, m_keep)
    sa_p = r_out["mean"][:, rule_asset.channel_index("sa_score")].reshape(n_prompts, m_keep)
    mw_p = r_out["mean"][:, rule_asset.channel_index("mol_weight")].reshape(n_prompts, m_keep)
    sm_p = r_out["mean"][:, rule_asset.channel_index("smarts_alert")].reshape(n_prompts, m_keep)

    sel_mols: List[Optional[Chem.Mol]] = []
    sel_leg = np.zeros(n_prompts, dtype=bool)
    sel_exo = np.zeros(n_prompts, dtype=bool)
    for i in range(n_prompts):
        goal_h = float(req_targets[i, 0])
        goal_n = float(req_targets[i, 1])
        best_k = 0
        best_key = (99, float("inf"))
        for k in range(m_keep):
            mol_k = mols_grid[i][k]
            if not bool(legal_grid[i, k]) or mol_k is None:
                ck = (4, float(k))
            else:
                cln = not bool(exotic_grid[i, k])
                c_ok = (sa_p[i, k] <= 3.5) and (sm_p[i, k] <= 0.0) and (mw_p[i, k] <= 550.0)
                err_h = abs(float(homo_p[i, k]) - goal_h) if math.isfinite(float(homo_p[i, k])) else 99.0
                err_n = abs(float(mol_k.GetNumAtoms()) - goal_n)
                tol_n = max(4.0, 0.12 * goal_n)
                size_pen = max(0.0, err_n - tol_n) * 0.04 + err_n * 0.015
                sa_pen = max(0.0, float(sa_p[i, k]) - 3.5) * 0.25
                tier = 0 if (cln and c_ok) else (1 if cln else 2)
                ck = (tier, err_h + sa_pen + size_pen)
            if ck < best_key:
                best_key = ck
                best_k = k
        sel_mols.append(mols_grid[i][best_k] if bool(legal_grid[i, best_k]) else None)
        sel_leg[i] = bool(legal_grid[i, best_k])
        sel_exo[i] = bool(exotic_grid[i, best_k])
    return sel_mols, sel_leg, sel_exo, surr_wall


def _score_oe62_size_cohort(
    selected_mols: List[Optional[Chem.Mol]],
    selected_legal: np.ndarray,
    selected_exotic: np.ndarray,
    req_targets: np.ndarray,
    bus: CompositeCriticBus,
    source_graphs: set[str],
    source_scaffolds: set[str],
    gen_wall_sec: float,
    reconcile_wall_sec: float,
    surrogate_wall_sec: float,
    *,
    export_sdf_path: Optional[Path] = None,
) -> Dict[str, Any]:
    n_total = len(selected_mols)
    obj_idx = bus.spec.objective_names.index("homo")
    req_mat = np.zeros((n_total, len(bus.spec.objectives)), dtype=np.float64)
    req_mat[:, obj_idx] = req_targets[:, 0].astype(np.float64)
    act_mat = np.zeros((n_total, len(bus.spec.objectives)), dtype=bool)
    act_mat[:, obj_idx] = True

    ev = bus.evaluate_molecules(
        selected_mols,
        requested_targets=req_mat,
        active_mask=act_mat,
        include_audit=True,
        for_training=False,
        batch_size=64,
    )
    clean_mask = selected_legal & (~selected_exotic)
    preds_1d = ev["objective_preds"][:, obj_idx]
    eval_1d = clean_mask & ev["evaluated_mask"] & np.isfinite(preds_1d)
    aud_preds_dict = ev.get("audit_preds") or {}
    aud_p = aud_preds_dict.get("homo")

    metrics = compute_loom_metrics(
        requested_targets_eV=req_targets[:, 0],
        reward_predictions_eV=preds_1d,
        legal_mask=selected_legal,
        evaluated_mask=eval_1d,
        audit_predictions_eV=aud_p,
        f_max_eV_A=ev["f_max_eV_A"],
        unseen_charge_mask=selected_exotic,
        tolerance_eV=0.25,
    )

    sa_vals = ev["constraint_values"].get("synthesizable_sa", np.full(n_total, np.nan))
    if aud_p is None:
        aud_p = np.full(n_total, np.nan, dtype=np.float64)
    aud_ok = clean_mask & np.isfinite(aud_p)
    rew_hit = eval_1d & (np.abs(preds_1d - req_targets[:, 0]) <= 0.25)
    aud_hit = aud_ok & (np.abs(aud_p - req_targets[:, 0]) <= 0.25)
    constr = ev["constraints_met"].astype(bool)

    tot_atoms = np.full(n_total, np.nan, dtype=np.float64)
    heav_atoms = np.full(n_total, np.nan, dtype=np.float64)
    nat_err = np.full(n_total, np.nan, dtype=np.float64)
    smis: List[str] = []
    scafs: List[str] = []
    acyclic = 0
    for i, m in enumerate(selected_mols):
        if m is not None and bool(selected_legal[i]):
            na = float(m.GetNumAtoms())
            nh = float(m.GetNumHeavyAtoms())
            tot_atoms[i] = na
            heav_atoms[i] = nh
            nat_err[i] = abs(na - float(req_targets[i, 1]))
            smis.append(mol_to_clean_smiles(m))
            sc = _murcko_scaffold_smiles(m)
            if sc:
                scafs.append(sc)
            else:
                acyclic += 1

    uniq_smis = set(smis)
    uniq_scafs = set(scafs)
    synth_smis = {mol_to_clean_smiles(selected_mols[i]) for i in np.where(rew_hit & constr)[0] if selected_mols[i] is not None}

    size_bins_out: Dict[str, Any] = {}
    for lo, hi in SIZE_BINS:
        label = f"[{lo},{hi})"
        bm = selected_legal & np.isfinite(tot_atoms) & (tot_atoms >= lo) & (tot_atoms < hi)
        nb = int(bm.sum())
        if nb == 0:
            size_bins_out[label] = {"n": 0, "pct": 0.0}
            continue
        b_smis = [mol_to_clean_smiles(selected_mols[i]) for i in np.where(bm)[0] if selected_mols[i] is not None]
        b_scafs = [sc for sc in (_murcko_scaffold_smiles(selected_mols[i]) for i in np.where(bm)[0] if selected_mols[i] is not None) if sc]
        b_rew_hit = int((bm & rew_hit & constr).sum())
        b_aud_hit = int((bm & aud_hit & constr).sum())
        size_bins_out[label] = {
            "n": nb,
            "pct": round(100.0 * nb / n_total, 2),
            "uniq_smi": len(set(b_smis)),
            "uniq_scaf": len(set(b_scafs)),
            "rew_synth": b_rew_hit,
            "rew_synth_pct": round(100.0 * b_rew_hit / nb, 2),
            "blind_synth": b_aud_hit,
            "blind_synth_pct": round(100.0 * b_aud_hit / nb, 2),
            "rew_mae": round(float(np.mean(np.abs(preds_1d[bm & eval_1d] - req_targets[bm & eval_1d, 0]))), 4)
            if (bm & eval_1d).any()
            else None,
            "blind_mae": round(float(np.mean(np.abs(aud_p[bm & aud_ok] - req_targets[bm & aud_ok, 0]))), 4)
            if (bm & aud_ok).any()
            else None,
        }

    if export_sdf_path is not None:
        export_sdf_path.parent.mkdir(parents=True, exist_ok=True)
        fmax_vals = ev["f_max_eV_A"]
        writer = Chem.SDWriter(str(export_sdf_path))
        for i, m in enumerate(selected_mols):
            if m is None or not bool(selected_legal[i]):
                continue
            cm = Chem.Mol(m)
            t_s = f"{float(req_targets[i, 0]):.6f}"
            sa_s = f"{float(sa_vals[i]):.4f}" if math.isfinite(float(sa_vals[i])) else "nan"
            uh_s = f"{float(preds_1d[i]):.6f}" if math.isfinite(float(preds_1d[i])) else "nan"
            dh_s = f"{float(aud_p[i]):.6f}" if math.isfinite(float(aud_p[i])) else "nan"
            fm_s = (
                f"{float(fmax_vals[i]):.4f}"
                if (fmax_vals is not None and math.isfinite(float(fmax_vals[i])))
                else "nan"
            )
            synth_s = "1" if bool(rew_hit[i] and constr[i]) else "0"
            cm.SetProp("PromptIndex", str(i))
            cm.SetProp("RequestedHOMO_eV", t_s)
            cm.SetProp("RequestedNAtoms", f"{float(req_targets[i, 1]):.1f}")
            cm.SetProp("ActualNAtoms", str(cm.GetNumAtoms()))
            cm.SetProp("SA_Score", sa_s)
            cm.SetProp("RewardHOMO_eV", uh_s)
            cm.SetProp("BlindDPA4_HOMO_eV", dh_s)
            # Lowercase properties required by scripts/loom_oe62_head2head_dft_anchor.py
            cm.SetProp("prompt_index", str(i))
            cm.SetProp("homo_target_eV", t_s)
            cm.SetProp("unimol2_homo_eV", uh_s)
            cm.SetProp("dpa4_homo_eV", dh_s)
            cm.SetProp("sa_score", sa_s)
            cm.SetProp("f_max_eV_A", fm_s)
            cm.SetProp("is_synthesizable_clean_hit", synth_s)
            writer.write(cm)
        writer.close()

    end_wall = gen_wall_sec + reconcile_wall_sec + surrogate_wall_sec
    leg_nat = tot_atoms[selected_legal & np.isfinite(tot_atoms)]
    leg_heav = heav_atoms[selected_legal & np.isfinite(heav_atoms)]
    leg_err = nat_err[selected_legal & np.isfinite(nat_err)]
    leg_req_n = req_targets[selected_legal & np.isfinite(nat_err), 1]

    metrics.update(
        {
            "synthesizable_clean_hits": int((rew_hit & constr).sum()),
            "synthesizable_clean_hit_yield": round(float((rew_hit & constr).sum()) / n_total, 5),
            "blind_dpa4_clean_hits": int(aud_hit.sum()),
            "blind_dpa4_synthesizable_clean_hits": int((aud_hit & constr).sum()),
            "blind_dpa4_synthesizable_clean_hit_yield": round(float((aud_hit & constr).sum()) / n_total, 5),
            "unique_synthesizable_hit_graphs": len(synth_smis),
            "novel_unique_synthesizable_hit_graphs": len(synth_smis - source_graphs),
            "constraints_met_rate_among_legal": round(float(constr[selected_legal].mean()), 5) if selected_legal.any() else 0.0,
            "median_sa_score_among_legal": round(float(np.nanmedian(sa_vals[selected_legal])), 4) if selected_legal.any() else None,
            "sa_leq_3_5_rate_among_legal": round(float(np.mean(sa_vals[selected_legal] <= 3.5)), 5) if selected_legal.any() else 0.0,
            "audit_mae_eV": (metrics["audit"]["audit_mae_eV"] if metrics.get("audit") else None),
            "audit_minus_reward_mae_gap_eV": (
                metrics["audit"]["audit_minus_reward_mae_gap_eV"] if metrics.get("audit") else None
            ),
            "mean_total_atoms": round(float(np.mean(leg_nat)), 2) if len(leg_nat) else None,
            "median_total_atoms": round(float(np.median(leg_nat)), 1) if len(leg_nat) else None,
            "p10_total_atoms": round(float(np.percentile(leg_nat, 10)), 1) if len(leg_nat) else None,
            "p90_total_atoms": round(float(np.percentile(leg_nat, 90)), 1) if len(leg_nat) else None,
            "mean_heavy_atoms": round(float(np.mean(leg_heav)), 2) if len(leg_heav) else None,
            "mean_natoms_abs_err": round(float(np.mean(leg_err)), 2) if len(leg_err) else None,
            "natoms_within_4_rate": round(float(np.mean(leg_err <= 4.0)), 4) if len(leg_err) else None,
            "natoms_within_15pct_rate": round(float(np.mean(leg_err <= 0.15 * np.maximum(leg_req_n, 1.0))), 4) if len(leg_err) else None,
            "unique_smiles_count": len(uniq_smis),
            "uniqueness_pct": round(100.0 * len(uniq_smis) / max(len(smis), 1), 2),
            "novel_smiles_pct": round(100.0 * len(uniq_smis - source_graphs) / max(len(uniq_smis), 1), 2),
            "unique_murcko_scaffolds": len(uniq_scafs),
            "novel_murcko_scaffolds": len(uniq_scafs - source_scaffolds),
            "acyclic_pct": round(100.0 * acyclic / max(len(smis), 1), 2),
            "size_stratified": size_bins_out,
            "gen_wall_seconds": round(gen_wall_sec, 3),
            "reconcile_wall_seconds": round(reconcile_wall_sec, 3),
            "surrogate_wall_seconds": round(surrogate_wall_sec, 3),
            "end_to_end_wall_seconds": round(end_wall, 3),
            "effective_output_mol_per_sec": round(float(n_total) / max(end_wall, 1e-6), 2),
        }
    )
    return metrics


def run_eval_all(args: argparse.Namespace, ctx: Optional[Dict[str, Any]] = None) -> None:
    device = torch.device(args.device)
    p = _paths(args)
    p["eval_dir"].mkdir(parents=True, exist_ok=True)
    sdf_dir = p["eval_dir"] / "cohorts_sdf"

    if ctx is not None and "train_bus" in ctx:
        ctx.pop("train_bus", None)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bus = _build_oe62_bus(args, include_blind_auditor=True)
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    gen, v20_payload = LoomUni3DARGenerator.load_checkpoint(p["v20_ckpt"], device=args.device, batch_size=args.eval_batch_size)
    gen.eval()
    v20_meta = v20_payload.get("metadata") or {}
    leaf_tol: Optional[float] = (
        None
        if bool(getattr(args, "disable_eval_leaf_budget", False))
        else (
            float(args.eval_leaf_budget_tol)
            if getattr(args, "eval_leaf_budget_tol", None) is not None
            else v20_meta.get("selected_leaf_budget_tol")
        )
    )
    critic, critic_meta = OctreeProcessCritic.load_checkpoint(p["critic_ckpt"], device=args.device)
    critic.eval()

    # Build deterministic 1,792 size-matched prompts across the 7 held-out OE62 HOMO quantiles
    _, clean_targets = _get_clean_oe62_pool(args, ctx)

    targets_cfg = json.loads(args.oe62_targets_json.read_text())
    q_points = targets_cfg["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    per_quantile = int(args.eval_per_quantile)
    prompt_rng = np.random.default_rng(20261117)
    req_rows: List[Tuple[float, float]] = []
    for q in quantiles:
        qh = np.full(per_quantile, float(q_points[str(q)]), dtype=np.float32)
        qn = _sample_conditional_natoms(qh, clean_targets, prompt_rng, window_eV=0.25, clamp_range=(16.0, 66.0))
        req_rows.extend(zip(qh.tolist(), qn.tolist()))
    req_targets = np.asarray(req_rows, dtype=np.float32)
    n_prompts = len(req_targets)  # 1,792
    print(
        f"[Size Eval] Built {n_prompts} prompts | req_natoms mean={req_targets[:, 1].mean():.2f}, "
        f"med={np.median(req_targets[:, 1]):.1f}, p10={np.percentile(req_targets[:, 1], 10):.1f}, "
        f"p90={np.percentile(req_targets[:, 1], 90):.1f} | leaf_tol={leaf_tol}",
        flush=True,
    )

    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))

    def _encode_batch(rows: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tn = gen.normalize_targets(rows)
        ma = torch.ones((len(rows), 2), dtype=torch.bool, device=device)
        ua = torch.tensor([[False, True]] * len(rows), dtype=torch.bool, device=device)
        return tn, ma, ua

    # Warmup
    wt, wm, wu = _encode_batch(req_targets[:16])
    _ = gen.sample(
        wt,
        wm,
        uncond_active_mask=wu,
        cfg_scale=2.0,
        cfg_tree_decay=0.30,
        geometric_guard=True,
        seed=11,
        use_bf16=(device.type == "cuda"),
        leaf_atom_budget=(req_targets[:16, 1] if leaf_tol is not None else None),
        leaf_budget_tol=(float(leaf_tol) if leaf_tol is not None else 2.0),
    )
    _sync(device)

    results_arms: Dict[str, Any] = {}

    sd_off = int(getattr(args, "seed_offset", 0)) * 1000003
    # 1. Outcome Best-of-M (`M in {1, 2, 4, 8}`)
    m_max = 8
    round_trajs: List[List[Dict[str, Any]]] = []
    round_gen_walls: List[float] = []
    for m_i in range(m_max):
        _sync(device)
        tg0 = time.perf_counter()
        cur_tr: List[Dict[str, Any]] = []
        for s in range(0, n_prompts, args.eval_batch_size):
            e = min(s + args.eval_batch_size, n_prompts)
            bt, bm, bu = _encode_batch(req_targets[s:e])
            cur_tr.extend(
                gen.sample(
                    bt,
                    bm,
                    uncond_active_mask=bu,
                    cfg_scale=2.00,
                    xyz_cfg_scale=1.0,
                    cfg_tree_decay=0.30,
                    geometric_guard=True,
                    temperatures=(0.85, 0.30, 0.25),
                    seed=sd_off + 2441100 + m_i * 9973 + s,
                    use_bf16=(device.type == "cuda"),
                    electron_parity=0,
                    leaf_atom_budget=(req_targets[s:e, 1] if leaf_tol is not None else None),
                    leaf_budget_tol=(float(leaf_tol) if leaf_tol is not None else 2.0),
                )
            )
        _sync(device)
        round_gen_walls.append(time.perf_counter() - tg0)
        round_trajs.append(cur_tr)

    tr0 = time.perf_counter()
    out_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(n_prompts)]
    out_leg = np.zeros((n_prompts, m_max), dtype=bool)
    out_exo = np.zeros((n_prompts, m_max), dtype=bool)
    for i in range(n_prompts):
        for m_i in range(m_max):
            tr = round_trajs[m_i][i]
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and res.mol is not None:
                out_grid[i][m_i] = res.mol
                out_leg[i, m_i] = True
                out_exo[i, m_i] = bool(res.reconciled_exotic_charge)
    rec_wall_total = time.perf_counter() - tr0

    for mb in (1, 2, 4, 8):
        sub_grid = [row[:mb] for row in out_grid]
        sel_m, sel_l, sel_x, surr_w = _select_oe62_size_from_pool(
            sub_grid, out_leg[:, :mb], out_exo[:, :mb], req_targets, bus, mb, use_surrogate=(mb > 1), device=device
        )
        if bus.potential_slot is not None:
            reconciler.batch_polish_hydrogens_mlip([m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
        gen_w = float(sum(round_gen_walls[:mb]))
        rec_w = rec_wall_total * (float(mb) / float(m_max))
        arm_res = _score_oe62_size_cohort(
            sel_m,
            sel_l,
            sel_x,
            req_targets,
            bus,
            source_graphs,
            source_scaffolds,
            gen_w,
            rec_w,
            surr_w,
            export_sdf_path=(sdf_dir / f"Outcome_M{mb}.sdf"),
        )
        results_arms[f"Outcome_M{mb}"] = arm_res
        bmae = float("nan") if arm_res.get("audit_mae_eV") is None else float(arm_res["audit_mae_eV"])
        print(
            f"[Size Outcome_M{mb}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
            f"Uniq={arm_res['uniqueness_pct']:.1f}% Scaf={arm_res['unique_murcko_scaffolds']} | "
            f"RewSynth={arm_res['synthesizable_clean_hits']}/{n_prompts} ({arm_res['synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} ({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindMAE={bmae:.4f}",
            flush=True,
        )

    # 2. Octree Process Critic Funnels (`8->1`, `8->2`, `16->2`, `16->4`)
    obj_col_mask = torch.tensor([True, True], dtype=torch.bool, device=device)
    funnels: List[Tuple[str, int, int, bool]] = [
        ("Process_8_to_1_nosurr", 8, 1, False),
        ("Process_8_to_2", 8, 2, True),
        ("Process_16_to_2", 16, 2, True),
        ("Process_16_to_4", 16, 4, True),
    ]
    for f_name, m_init, m_keep, use_surr in funnels:
        _sync(device)
        tg0 = time.perf_counter()
        groups_all: List[List[Dict[str, Any]]] = []
        for s in range(0, n_prompts, args.eval_batch_size):
            e = min(s + args.eval_batch_size, n_prompts)
            bt, bm, bu = _encode_batch(req_targets[s:e])
            groups_all.extend(
                gen.sample_with_process_critic(
                    bt,
                    bm,
                    critic,
                    m_init=m_init,
                    m_keep=m_keep,
                    objective_col_mask=obj_col_mask,
                    legal_weight=0.80,
                    constr_weight=0.55,
                    cfg_scale=2.00,
                    xyz_cfg_scale=1.0,
                    cfg_tree_decay=0.30,
                    geometric_guard=True,
                    seed=sd_off + 770000 + m_init * 313 + m_keep * 37 + s,
                    use_bf16=(device.type == "cuda"),
                    uncond_active_mask=bu,
                    coarse_chunk_size=int(args.coarse_chunk_size),
                    fine_batch_size=int(args.fine_batch_size),
                    leaf_atom_budget=(req_targets[s:e, 1] if leaf_tol is not None else None),
                    leaf_budget_tol=(float(leaf_tol) if leaf_tol is not None else 2.0),
                )
            )
        _sync(device)
        gen_w = time.perf_counter() - tg0

        tr0 = time.perf_counter()
        p_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_keep for _ in range(n_prompts)]
        p_leg = np.zeros((n_prompts, m_keep), dtype=bool)
        p_exo = np.zeros((n_prompts, m_keep), dtype=bool)
        for i in range(n_prompts):
            for k in range(m_keep):
                tr = groups_all[i][k]
                res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
                if res.reconciled_legal and res.mol is not None:
                    p_grid[i][k] = res.mol
                    p_leg[i, k] = True
                    p_exo[i, k] = bool(res.reconciled_exotic_charge)
        rec_w = time.perf_counter() - tr0

        sel_m, sel_l, sel_x, surr_w = _select_oe62_size_from_pool(
            p_grid, p_leg, p_exo, req_targets, bus, m_keep, use_surrogate=use_surr, device=device
        )
        if bus.potential_slot is not None:
            reconciler.batch_polish_hydrogens_mlip([m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
        arm_res = _score_oe62_size_cohort(
            sel_m,
            sel_l,
            sel_x,
            req_targets,
            bus,
            source_graphs,
            source_scaffolds,
            gen_w,
            rec_w,
            surr_w,
            export_sdf_path=(sdf_dir / f"{f_name}.sdf"),
        )
        results_arms[f_name] = arm_res
        bmae = float("nan") if arm_res.get("audit_mae_eV") is None else float(arm_res["audit_mae_eV"])
        print(
            f"[Size {f_name}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
            f"Uniq={arm_res['uniqueness_pct']:.1f}% Scaf={arm_res['unique_murcko_scaffolds']} | "
            f"RewSynth={arm_res['synthesizable_clean_hits']}/{n_prompts} ({arm_res['synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} ({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindMAE={bmae:.4f}",
            flush=True,
        )

    payload = {
        "experiment": "OE62 Size-Anchored Uni-Loom ([homo, n_atoms] + size-anchored CFG + v2.0 GRPO + OctreeProcessCritic)",
        "v3_mode": bool(v20_meta.get("v3_mode", getattr(args, "v3_mode", False))),
        "allow_tf32": bool(getattr(args, "allow_tf32", False)),
        "eval_batch_size": int(args.eval_batch_size),
        "coarse_chunk_size": int(args.coarse_chunk_size),
        "fine_batch_size": int(args.fine_batch_size),
        "selected_leaf_budget_tol": leaf_tol,
        "proposals_per_arm": n_prompts,
        "requested_natoms_summary": {
            "mean": round(float(req_targets[:, 1].mean()), 2),
            "median": round(float(np.median(req_targets[:, 1])), 1),
            "p10": round(float(np.percentile(req_targets[:, 1], 10)), 1),
            "p90": round(float(np.percentile(req_targets[:, 1], 90)), 1),
        },
        "v20_checkpoint": str(p["v20_ckpt"]),
        "v20_checkpoint_metadata": v20_meta,
        "process_critic_meta": critic_meta,
        "arms": results_arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Size Eval ALL DONE] Wrote {args.output_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["train_sft", "align_v20", "train_critic", "eval_all", "all"], required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="oe62_size_anchored")
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument(
        "--parent-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_e13_e14_20261113/full_homo_27000/weights/octree_conditional.pt"),
    )
    p.add_argument(
        "--oe62-master-train-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"),
    )
    p.add_argument(
        "--oe62-master-valid-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_valid_B.sdf"),
    )
    p.add_argument(
        "--oe62-source-train-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/source_labels/train_source.sdf"),
    )
    p.add_argument(
        "--oe62-replay-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_B.sdf"),
    )
    p.add_argument(
        "--oe62-targets-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/targets_B.json"),
    )
    p.add_argument(
        "--oe62-reward-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"),
    )
    p.add_argument(
        "--oe62-audit-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_e15_calibration_20261114/full_fit/oracle"),
    )
    p.add_argument(
        "--oe62-potential-weights",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"),
    )
    p.add_argument("--sft-steps", type=int, default=1600)
    p.add_argument("--sft-batch-size", type=int, default=64)
    p.add_argument("--sft-lr", type=float, default=2.5e-4)
    p.add_argument("--sft-clean-only", action="store_true")
    p.add_argument("--sft-contrast-weight", type=float, default=0.10)
    p.add_argument("--replay-pool-cap", type=int, default=4096)
    p.add_argument("--grpo-steps", type=int, default=96)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--w-phys", type=float, default=0.35)
    p.add_argument("--relax-distill-weight", type=float, default=0.0)
    p.add_argument("--relax-distill-steps", type=int, default=25)
    p.add_argument("--relax-distill-max-mols", type=int, default=8)
    p.add_argument("--relax-distill-max-fmax", type=float, default=2.0)
    p.add_argument("--size-adherence-weight", type=float, default=0.85)
    p.add_argument("--critic-rollouts", type=int, default=8192)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--eval-per-quantile", type=int, default=256)
    p.add_argument("--v3-mode", action="store_true", help="Enable v3 canonical PCA + Rank-32 SFT + Clean-SA Hindsight AWR + detached XYZ distillation")
    p.add_argument("--allow-tf32", action="store_true", help="Enable standard Ampere TF32 matmul/cuDNN kernels")
    p.add_argument("--v3-hindsight-rounds", type=int, default=2)
    p.add_argument("--v3-hindsight-mols", type=int, default=4096)
    p.add_argument("--v3-hindsight-steps", type=int, default=550)
    p.add_argument("--v3-xyz-distill-steps", type=int, default=900)
    p.add_argument("--coarse-chunk-size", type=int, default=1024)
    p.add_argument("--fine-batch-size", type=int, default=512)
    p.add_argument("--eval-leaf-budget-tol", type=float, default=None)
    p.add_argument("--disable-eval-leaf-budget", action="store_true")
    p.add_argument("--seed-offset", type=int, default=0, help="Independent sampling seed offset for multi-seed CI repeats.")
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_size_anchored_eval.json"),
    )
    args = p.parse_args()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ctx: Dict[str, Any] = {}
    if args.phase in ("train_sft", "all"):
        run_train_sft(args, ctx)
    if args.phase in ("align_v20", "all"):
        run_align_v20(args, ctx)
    if args.phase in ("train_critic", "all"):
        run_train_critic(args, ctx)
    if args.phase in ("eval_all", "all"):
        run_eval_all(args, ctx)


if __name__ == "__main__":
    main()
