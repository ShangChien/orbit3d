"""Public QM9 50k/50k Disjoint 6-Property Benchmark for `Uni-3DAR-Mol` + `Uni-Loom`.

Benchmarks the upgraded `Uni-3DAR-Mol` architecture against `PropMolFlow` (6 models, 155.97 GPU-h)
on the exact official PropMolFlow `50k/50k` disjoint QM9 benchmark (`alpha`, `gap`, `homo`, `lumo`,
`mu`, `cv`; `N = 1,000` test prompts per property), enforcing strict zero-leakage rules:
  1. Generator (`qm9.pt` parent), `PostTreeGraphRefiner`, `TwoStageOctreeProcessCritic`,
     `QM9TrainSplitSurrogateOracle`, and `29-D` stoichiometric `IncrementalCalibrator` touch ONLY
     `splits["train"]` (`44,453` conformers) and `splits["dev"]` (`1,500` conformers) from
     `train_rQM_v0` (`0` overlap with `gvp_rQM_v0` verified by both `gdb9_idx` and 7-tuple hash).
  2. Full `SO(3)` equivariance (`random_rotation_prob = 0.80`, zero PCA canonicalization) +
     non-trapping 6-D `ScaleFreeBudgetEmbedding` (zero `n_atoms` in `LoomConditioner`) +
     `L5->L1` packing priors (`leaf_atom_budget = train_half_sampled_no_atoms_{prop}.npy`).
  3. Differentiable expected-`XYZ` 1-2/1-3 bonded-distance supervision (`geom_dist_loss_weight = 1.5`)
     + `< 0.4 ms` 4-layer `SE(3)` `PostTreeGraphRefiner` + zero-heavy-displacement `ValenceReconciler`.
  4. Speculative two-stage octree funnel (`M in {1, 2, 4, 8, 16}` -> `1->1->1`, `10->4->2`,
     `18->6->4`, `38->12->8`, `74->22->16`) pruned on-GPU by `TwoStageOctreeProcessCritic`.
  5. Frozen `GVPBenchmarkSuite` (`gvp_rQM_v0`) is called strictly ONCE on the single selected
     Top-1 molecule per test target (`N = 1,000` / property).
"""

from __future__ import annotations

import argparse
from copy import copy, deepcopy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch
from torch.utils.data import DataLoader, IterableDataset

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
for _cand in (
    ROOT / "src" / "compat",
    ROOT / "src",
    ROOT,
    Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat"),
    Path("/sjtu-combustion/wjw/chensq/PropMolFlow"),
    Path("/sjtu-combustion/chensq/mol_gen/src/compat"),
    Path("/sjtu-combustion/chensq/PropMolFlow"),
):
    if _cand.exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402
from loom.generator.octree_alignment import categorical_policy_loss  # noqa: E402
from loom.weaving import (  # noqa: E402
    LoomUni3DARGenerator,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    PostTreeGraphRefiner,
    ResidualCalibratedAsset,
    SlotOracleAsset,
    TwoStageOctreeProcessCritic,
    fit_octree_element_process_critic,
    fit_octree_process_critic,
    fit_post_tree_graph_refiner,
)
from scripts.loom_qm9_benchmark import (  # noqa: E402
    ALLOWED_QM9_ELEMENTS,
    EV_PROPS,
    GVPBenchmarkSuite,
    HA_TO_EV,
    PUBLISHED_BASELINES,
    PROPS,
    PointCloudOctreeDataset,
    _move_packed_to_device,
    _to_generator_targets,
    load_disjoint_qm9_splits,
    run_gvp_sanity_baselines,
)
from scripts.loom_qm9_best_of_m import (  # noqa: E402
    _canonical_smiles_2d,
    _load_propmolflow_eval_natoms,
    _min_pair_dist,
)
from scripts.loom_qm9_k1_pipeline import (  # noqa: E402
    QM9TrainSplitSurrogateOracle,
    _load_propmolflow_eval_targets,
    _target_ha_to_gen,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _QM9Uni3DARMolSFTStream(IterableDataset):
    """Multi-worker full-SO(3) (`random_rotation_prob=0.80`) SFT stream emphasizing single-property queries."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        targets_gen: np.ndarray,
        backbone_args: Any,
        backbone_dict: Any,
        batch_size: int,
        seed: int = 20261120,
    ) -> None:
        self.records = list(records)
        self.targets_gen = np.asarray(targets_gen, dtype=np.float32)
        self.backbone_args = copy(backbone_args)
        self.backbone_args.random_rotation_prob = 0.80
        self.backbone_dict = backbone_dict
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def __iter__(self):
        w_info = torch.utils.data.get_worker_info()
        wid = w_info.id if w_info is not None else 0
        rng = np.random.default_rng(self.seed + wid * 9973)

        class _Proxy:
            pass

        proxy = _Proxy()
        proxy.args = self.backbone_args
        proxy.dictionary = self.backbone_dict
        ds = PointCloudOctreeDataset(self.records, proxy, training=True, seed=self.seed + wid * 131)
        n = len(self.records)
        num_props = self.targets_gen.shape[1]

        while True:
            idx = rng.integers(0, n, size=self.batch_size)
            res = ds.collate_valid(idx.tolist())
            if res is None:
                continue
            packed, kept_pos = res
            kept_idx = idx[kept_pos]
            t_gen = self.targets_gen[kept_idx]
            bsz = len(kept_idx)
            mask = np.zeros((bsz, num_props), dtype=bool)
            for row_i in range(bsz):
                if rng.random() < 0.85:
                    mask[row_i, int(rng.integers(0, num_props))] = True
                else:
                    k_act = int(rng.integers(2, num_props + 1))
                    cols = rng.choice(num_props, size=k_act, replace=False)
                    mask[row_i, cols] = True
            yield packed, torch.from_numpy(t_gen), torch.from_numpy(mask)


def run_train_sft(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda"
    out_dir = args.work_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "uni3dar_mol_qm9_k6_sft_best.pt"
    if args.reuse_checkpoints and best_ckpt.exists():
        print(f"[QM9 SFT] Reusing {best_ckpt}", flush=True)
        return best_ckpt

    splits = load_disjoint_qm9_splits(args.lmdb_dir, args.propmolflow_root / "data" / "qm9_raw", seed=args.seed)
    assert int(splits["audit"]["overlap_train_with_gvp_rQM_v0"]) == 0
    train_recs = splits["train"]
    dev_recs = splits["dev"]
    train_targets_gen = _to_generator_targets(np.stack([r["props_ha"] for r in train_recs], axis=0))
    dev_targets_gen = _to_generator_targets(np.stack([r["props_ha"] for r in dev_recs], axis=0))
    mean_vec = np.mean(train_targets_gen, axis=0).tolist()
    scale_vec = np.std(train_targets_gen, axis=0).tolist()

    gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_checkpoint,
        target_cols=list(PROPS),
        target_units=["Bohr^3", "eV", "eV", "eV", "D", "cal/(mol*K)"],
        mean=mean_vec,
        scale=scale_vec,
        allowed_elements=ALLOWED_QM9_ELEMENTS,
        device=args.device,
        batch_size=args.sft_batch_size,
        num_rbf=48,
        num_freqs=24,
        num_prefix_tokens=4,
        modulation_cap=4.5,
        lora_rank=64,
        lora_alpha=128.0,
        lora_include_ffn=True,
        lora_include_heads=True,
        use_valence_head=True,
        non_trapping_budget=True,
    )
    gen.enable_v11_xyz_heads(train_xyz_embeddings=True)
    trainable = gen.configure_trainable(train_conditioner=True, train_lora=True, train_xyz_embeddings=True)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in gen.parameters())
    print(f"[QM9 Uni-3DAR-Mol SFT] Trainable: {n_trainable:,} / {n_total:,} ({100.0*n_trainable/n_total:.2f}%)", flush=True)

    stream = _QM9Uni3DARMolSFTStream(
        train_recs,
        train_targets_gen,
        gen.backbone.args,
        gen.backbone.dictionary,
        batch_size=args.sft_batch_size,
        seed=args.seed,
    )
    loader = DataLoader(
        stream,
        batch_size=None,
        num_workers=4,
        prefetch_factor=3,
        pin_memory=True,
        multiprocessing_context="spawn",
    )
    loader_iter = iter(loader)
    dev_ds = PointCloudOctreeDataset(dev_recs, gen.backbone, training=False, seed=args.seed + 77)

    optimizer = torch.optim.AdamW(trainable, lr=4.0e-4, weight_decay=0.01)
    warmup = 250
    total_steps = int(args.sft_steps)

    def _lr_at(st: int) -> float:
        if st < warmup:
            return 4.0e-4 * float(st + 1) / float(warmup)
        prog = float(st - warmup) / float(max(total_steps - warmup, 1))
        return 2.5e-5 + 0.5 * (4.0e-4 - 2.5e-5) * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    _sync(device)
    t0 = time.perf_counter()
    best_dev = float("inf")
    run_loss = 0.0
    gen.train()

    for step in range(1, total_steps + 1):
        lr_now = _lr_at(step - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        packed_cpu, t_gen_cpu, mask_cpu = next(loader_iter)
        packed = _move_packed_to_device(packed_cpu, device)
        t_gen = t_gen_cpu.to(device, non_blocking=True)
        act_mask = mask_cpu.to(device, non_blocking=True)
        t_norm = gen.normalize_targets(t_gen) + 0.012 * torch.randn((t_gen.shape[0], len(PROPS)), device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss, info = gen.compute_sft_loss(
                packed,
                t_norm,
                act_mask,
                cfg_dropout_prob=0.12,
                contrast_weight=0.30,
                contrast_margin=0.06,
                contrast_min_gap_eV=0.35,
                xyz_loss_weight=0.12,
                valence_loss_weight=0.15,
                geom_dist_loss_weight=1.50,
                budget_dropout_prob=0.12,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        run_loss += float(info.get("loss", loss.item()))

        if step == 1 or step % 400 == 0 or step == total_steps:
            denom = 1.0 if step == 1 else 400.0
            el = time.perf_counter() - t0
            rate = (step * args.sft_batch_size) / max(el, 1e-6)
            print(f"[QM9 SFT {step:4d}/{total_steps}] lr={lr_now:.2e} loss={run_loss/denom:.4f} rate={rate:.1f} mol/s ({el/60.0:.1f}m)", flush=True)
            run_loss = 0.0

        if step % 1500 == 0 or step == total_steps:
            gen.eval()
            v_losses: List[float] = []
            for s in range(0, min(len(dev_recs), 640), 128):
                idx_v = list(range(s, min(s + 128, len(dev_recs))))
                rv = dev_ds.collate_valid(idx_v)
                if rv is None:
                    continue
                vp_cpu, kpos = rv
                vp = _move_packed_to_device(vp_cpu, device)
                vt = gen.normalize_targets(dev_targets_gen[idx_v][kpos])
                vm = torch.zeros((len(kpos), len(PROPS)), dtype=torch.bool, device=device)
                for r_i in range(len(kpos)):
                    vm[r_i, r_i % len(PROPS)] = True
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    vl, _ = gen.compute_sft_loss(vp, vt, vm, cfg_dropout_prob=0.0, contrast_weight=0.0, budget_dropout_prob=0.0)
                v_losses.append(float(vl.item()))
            dev_l = float(np.mean(v_losses)) if v_losses else float("nan")
            el_gpuh = round((time.perf_counter() - t0) / 3600.0, 4)
            print(f"[QM9 Dev@{step}] dev_single_prop_nll={dev_l:.4f} | gpuh={el_gpuh}", flush=True)
            meta = {"sft_steps": step, "sft_gpuh": el_gpuh, "dev_loss": dev_l, "n_trainable": n_trainable, "n_total": n_total}
            if dev_l < best_dev:
                best_dev = dev_l
                gen.save_checkpoint(best_ckpt, extra_metadata=meta)
            gen.train()

    _sync(device)
    sft_gpuh = round((time.perf_counter() - t0) / 3600.0, 4)
    final_ckpt = out_dir / "uni3dar_mol_qm9_k6_sft_last.pt"
    gen.save_checkpoint(final_ckpt, extra_metadata={"sft_steps": total_steps, "sft_gpuh": sft_gpuh, "best_dev_loss": best_dev, "n_trainable": n_trainable, "n_total": n_total})
    if best_ckpt.exists():
        # Stamp total sft_gpuh onto best_ckpt metadata
        g_b, p_b = LoomUni3DARGenerator.load_checkpoint(best_ckpt, device="cpu")
        m_b = dict(p_b.get("metadata") or {})
        m_b["sft_gpuh"] = sft_gpuh
        g_b.save_checkpoint(best_ckpt, extra_metadata=m_b)
    print(f"[QM9 SFT DONE] {best_ckpt} | sft_gpuh={sft_gpuh}", flush=True)
    return best_ckpt


def run_train_refiner(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    out_dir = args.work_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_ckpt = out_dir / "uni3dar_mol_qm9_refiner.pt"
    if args.reuse_checkpoints and ref_ckpt.exists():
        print(f"[QM9 Refiner] Reusing {ref_ckpt}", flush=True)
        return ref_ckpt

    splits = load_disjoint_qm9_splits(args.lmdb_dir, args.propmolflow_root / "data" / "qm9_raw", seed=args.seed)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    rng = np.random.default_rng(args.seed + 55)
    sub_idx = rng.choice(len(splits["train"]), size=min(18000, len(splits["train"])), replace=False)
    ref_mols: List[Chem.Mol] = []
    for i in sub_idx:
        r = splits["train"][int(i)]
        res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
        if res.raw_legal and (not res.raw_exotic_charge) and res.raw_mol is not None:
            ref_mols.append(res.raw_mol)
        elif res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None:
            ref_mols.append(res.mol)

    tmp_gen = LoomUni3DARGenerator.from_parent_checkpoint(
        args.parent_checkpoint,
        target_cols=list(PROPS),
        target_units=["Bohr^3", "eV", "eV", "eV", "D", "cal/(mol*K)"],
        mean=[0.0] * 6,
        scale=[1.0] * 6,
        allowed_elements=ALLOWED_QM9_ELEMENTS,
        device="cpu",
        lora_rank=0,
    )
    num_atom_classes = int(tmp_gen.backbone.num_atom)
    del tmp_gen

    _sync(device)
    t0 = time.perf_counter()
    refiner = PostTreeGraphRefiner(
        num_atom_classes=num_atom_classes,
        d_model=192,
        n_heads=4,
        n_layers=4,
        n_rbf=24,
        d_pair=128,
        max_disp_A=0.22,
    ).to(device)
    stats = fit_post_tree_graph_refiner(
        refiner,
        ref_mols,
        epochs=14,
        batch_size=128,
        lr=1.4e-3,
        seed=202671,
    )
    refiner.eval()
    _sync(device)
    ref_gpuh = round((time.perf_counter() - t0) / 3600.0, 4)
    extra = dict(stats.__dict__)
    extra["refiner_gpuh"] = ref_gpuh
    refiner.save_checkpoint(ref_ckpt, extra=extra)
    print(f"[QM9 Refiner DONE] {ref_ckpt} | stats={extra}", flush=True)
    return ref_ckpt


def run_train_align(args: argparse.Namespace) -> Path:
    """Online Multi-Property Hindsight AWR + Scale-Decoupled `Tree-Vine TopoGRPO` (zero GVP leakage)."""
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda"
    out_dir = args.work_dir / args.run_name
    sft_ckpt = out_dir / "uni3dar_mol_qm9_k6_sft_best.pt"
    aligned_ckpt = out_dir / "uni3dar_mol_qm9_k6_aligned.pt"
    if args.reuse_checkpoints and aligned_ckpt.exists():
        print(f"[QM9 Align] Reusing {aligned_ckpt}", flush=True)
        return aligned_ckpt

    splits = load_disjoint_qm9_splits(args.lmdb_dir, args.propmolflow_root / "data" / "qm9_raw", seed=args.seed)
    train_recs = splits["train"]
    train_targets_gen = _to_generator_targets(np.stack([r["props_ha"] for r in train_recs], axis=0))
    train_natoms = np.array([len(r["symbols"]) for r in train_recs], dtype=np.float32)
    prop_stds = np.std(train_targets_gen, axis=0).astype(np.float32)

    surrogate = QM9TrainSplitSurrogateOracle(args.surrogate_ckpt, device=args.device)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    gen, sft_payload = LoomUni3DARGenerator.load_checkpoint(sft_ckpt, device=args.device, batch_size=256)
    sft_meta = dict(sft_payload.get("metadata") or {})

    # ---------------- 1. Online Multi-Property Hindsight Relabeling (AWR) ----------------
    _sync(device)
    t_awr0 = time.perf_counter()
    rng = np.random.default_rng(args.seed + 303)
    gen.eval()
    hind_recs: List[Dict[str, Any]] = []
    hind_t_list: List[np.ndarray] = []

    for p_idx in range(len(PROPS)):
        q_idx = rng.choice(len(train_recs), size=512, replace=True)
        q_t = np.zeros((512, len(PROPS)), dtype=np.float32)
        q_t[:, p_idx] = train_targets_gen[q_idx, p_idx]
        q_n = train_natoms[q_idx]
        q_m = torch.zeros((512, len(PROPS)), dtype=torch.bool, device=device)
        q_m[:, p_idx] = True
        t_norm_q = gen.normalize_targets(q_t)
        trajs_q: List[Dict[str, Any]] = []
        for s in range(0, 512, 256):
            e = min(s + 256, 512)
            trajs_q.extend(
                gen.sample(
                    t_norm_q[s:e],
                    q_m[s:e],
                    cfg_scale=2.0,
                    xyz_cfg_scale=1.0,
                    cfg_tree_decay=0.25,
                    geometric_guard=True,
                    temperatures=(0.85, 0.30, 0.25),
                    seed=61000 + p_idx * 997 + s,
                    use_bf16=use_bf16,
                    electron_parity=0,
                    leaf_atom_budget=q_n[s:e],
                    budget_guidance_scale=0.55,
                )
            )
        cand_mols: List[Optional[Chem.Mol]] = []
        for tr in trajs_q:
            if _min_pair_dist(tr["positions_np"]) < 0.72:
                cand_mols.append(None)
                continue
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None and len(Chem.GetMolFrags(res.mol)) == 1:
                cand_mols.append(res.mol)
            else:
                cand_mols.append(None)
        preds_s, _, ok_s, _ = surrogate.evaluate_candidates(cand_mols, batch_size=256)
        for j, m_c in enumerate(cand_mols):
            if m_c is not None and bool(ok_s[j]) and np.isfinite(preds_s[j]).all():
                hind_recs.append(
                    {
                        "gdb9_idx": 9000000 + len(hind_recs),
                        "symbols": [a.GetSymbol() for a in m_c.GetAtoms()],
                        "coords": m_c.GetConformer().GetPositions().astype(np.float32),
                        "props_ha": np.zeros(6, dtype=np.float32),
                    }
                )
                hind_t_list.append(preds_s[j].astype(np.float32))

    print(f"[QM9 Hindsight] Admitted {len(hind_recs)} / {512*len(PROPS)} clean surrogate-relabeled conformers", flush=True)
    if len(hind_recs) >= 128:
        hind_targets_gen = np.stack(hind_t_list, axis=0)
        anchor_ds = PointCloudOctreeDataset(train_recs, gen.backbone, training=True, seed=args.seed + 401)
        hind_ds = PointCloudOctreeDataset(hind_recs, gen.backbone, training=True, seed=args.seed + 402)
        trainable_awr = gen.configure_trainable(train_conditioner=True, train_lora=True, train_xyz_embeddings=True)
        opt_awr = torch.optim.AdamW(trainable_awr, lr=1.2e-4, weight_decay=0.01)
        gen.train()
        for st_a in range(1, 501):
            i_tr = rng.integers(0, len(train_recs), size=64).tolist()
            i_hd = rng.integers(0, len(hind_recs), size=64).tolist()
            r_tr = anchor_ds.collate_valid(i_tr)
            r_hd = hind_ds.collate_valid(i_hd)
            if r_tr is None or r_hd is None:
                continue
            for pk_cpu, kp, tg_pool in ((r_tr[0], r_tr[1], train_targets_gen[i_tr]), (r_hd[0], r_hd[1], hind_targets_gen[i_hd])):
                pk = _move_packed_to_device(pk_cpu, device)
                tg = torch.from_numpy(tg_pool[kp]).to(device)
                bs_k = len(kp)
                mk = torch.zeros((bs_k, len(PROPS)), dtype=torch.bool, device=device)
                for rr in range(bs_k):
                    mk[rr, int(rng.integers(0, len(PROPS)))] = True
                opt_awr.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    l_a, _ = gen.compute_sft_loss(
                        pk, gen.normalize_targets(tg), mk,
                        cfg_dropout_prob=0.10, contrast_weight=0.25, geom_dist_loss_weight=1.20, budget_dropout_prob=0.10,
                    )
                l_a.backward()
                torch.nn.utils.clip_grad_norm_(trainable_awr, 1.0)
                opt_awr.step()
    _sync(device)
    awr_gpuh = round((time.perf_counter() - t_awr0) / 3600.0, 4)

    # ---------------- 2. Scale-Decoupled Multi-Property `Tree-Vine TopoGRPO` ----------------
    _sync(device)
    t_grpo0 = time.perf_counter()
    reference = deepcopy(gen).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    trainable_grpo = gen.configure_trainable(train_conditioner=True, train_lora=True, train_xyz_embeddings=False)
    opt_grpo = torch.optim.AdamW(trainable_grpo, lr=5.5e-5, weight_decay=0.01)
    anchor_grpo_ds = PointCloudOctreeDataset(train_recs, gen.backbone, training=True, seed=args.seed + 505)

    grpo_steps = int(args.grpo_steps)
    prompts_per_step = 8
    group_size = 8
    tot_b = prompts_per_step * group_size

    for step in range(1, grpo_steps + 1):
        p_idx = (step - 1) % len(PROPS)
        p_name = PROPS[p_idx]
        p_std = float(prop_stds[p_idx])

        pick = rng.choice(len(train_recs), size=prompts_per_step, replace=False)
        rep_val = np.repeat(train_targets_gen[pick, p_idx], group_size).astype(np.float32)
        rep_nat = np.repeat(train_natoms[pick], group_size).astype(np.float32)
        t_mat = np.zeros((tot_b, len(PROPS)), dtype=np.float32)
        t_mat[:, p_idx] = rep_val
        t_norm = gen.normalize_targets(t_mat)
        c_mask = torch.zeros((tot_b, len(PROPS)), dtype=torch.bool, device=device)
        c_mask[:, p_idx] = True

        gen.eval()
        with torch.no_grad():
            trajs = gen.sample(
                t_norm,
                c_mask,
                cfg_scale=2.0,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=810000 + step * 73,
                use_bf16=use_bf16,
                electron_parity=0,
                leaf_atom_budget=rep_nat,
                budget_guidance_scale=0.55,
            )
            old_pkts = gen.trajectory_distributions(
                trajs, t_norm, c_mask, temperatures=(0.85, 0.30, 0.25),
                cfg_scale=2.0, xyz_cfg_scale=1.0, cfg_tree_decay=0.25,
                geometric_guard=True, topology_only_policy=True, electron_parity=0,
            )
            ref_pkts = reference.trajectory_distributions(
                trajs, t_norm, c_mask, temperatures=(0.85, 0.30, 0.25),
                cfg_scale=2.0, xyz_cfg_scale=1.0, cfg_tree_decay=0.25,
                geometric_guard=True, topology_only_policy=True, electron_parity=0,
            )

        rec_mols: List[Optional[Chem.Mol]] = []
        tier_f: List[int] = []
        c_nats = np.zeros(tot_b, dtype=np.int32)
        for i, tr in enumerate(trajs):
            if _min_pair_dist(tr["positions_np"]) < 0.72:
                rec_mols.append(None)
                tier_f.append(0)
                continue
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if not res.reconciled_legal or res.mol is None:
                rec_mols.append(None)
                tier_f.append(0)
            elif res.reconciled_exotic_charge or len(Chem.GetMolFrags(res.mol)) != 1:
                rec_mols.append(res.mol)
                c_nats[i] = int(res.mol.GetNumAtoms())
                tier_f.append(1)
            else:
                rec_mols.append(res.mol)
                c_nats[i] = int(res.mol.GetNumAtoms())
                tier_f.append(2)

        preds_s, _, ok_s, _ = surrogate.evaluate_candidates(rec_mols, batch_size=128)
        rew_np = np.zeros(tot_b, dtype=np.float32)
        for i in range(tot_b):
            if tier_f[i] == 0 or not bool(ok_s[i]):
                rew_np[i] = -1.0
            else:
                err = abs(float(preds_s[i, p_idx]) - float(rep_val[i]))
                if tier_f[i] == 1:
                    rew_np[i] = -0.35 + 0.25 * math.exp(-err / max(p_std, 1e-6))
                else:
                    tight_r = math.exp(-err / max(0.14 * p_std, 1e-6))
                    wide_r = math.exp(-err / max(0.50 * p_std, 1e-6))
                    sz_pen = 0.05 * min(abs(int(c_nats[i]) - int(rep_nat[i])), 8)
                    rew_np[i] = 0.20 + 0.58 * tight_r + 0.22 * wide_r - sz_pen

        adv_np = np.zeros(tot_b, dtype=np.float32)
        for g_i in range(prompts_per_step):
            gs, ge = g_i * group_size, (g_i + 1) * group_size
            grp = rew_np[gs:ge]
            sd = float(np.std(grp))
            if sd > 1e-5:
                adv_np[gs:ge] = np.clip((grp - float(np.mean(grp))) / (sd + 1e-6), -3.0, 3.0)
        adv_t = torch.from_numpy(adv_np).to(device)

        a_idx = rng.integers(0, len(train_recs), size=64).tolist()
        a_res = anchor_grpo_ds.collate_valid(a_idx)
        gen.train()
        opt_grpo.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            cur_pkts = gen.trajectory_distributions(
                trajs, t_norm, c_mask, temperatures=(0.85, 0.30, 0.25),
                cfg_scale=2.0, xyz_cfg_scale=1.0, cfg_tree_decay=0.25,
                geometric_guard=True, topology_only_policy=True, electron_parity=0,
            )
            p_loss = categorical_policy_loss(cur_pkts, old_pkts, ref_pkts, adv_t, beta=0.025, clip=0.20)
            if a_res is not None:
                apk = _move_packed_to_device(a_res[0], device)
                atg = gen.normalize_targets(train_targets_gen[a_idx][a_res[1]])
                amk = torch.zeros((len(a_res[1]), len(PROPS)), dtype=torch.bool, device=device)
                amk[:, p_idx] = True
                anc_l, _ = gen.compute_sft_loss(apk, atg, amk, cfg_dropout_prob=0.0, contrast_weight=0.20, budget_dropout_prob=0.05)
                tot_l = p_loss + 0.22 * anc_l
            else:
                tot_l = p_loss
        tot_l.backward()
        torch.nn.utils.clip_grad_norm_(trainable_grpo, 1.0)
        opt_grpo.step()

        if step == 1 or step % 30 == 0 or step == grpo_steps:
            cln_pct = 100.0 * sum(1 for x in tier_f if x == 2) / float(tot_b)
            print(f"[QM9 TopoGRPO {step:3d}/{grpo_steps} ({p_name})] rew={float(np.mean(rew_np)):.3f} clean={cln_pct:.1f}% ploss={float(p_loss.item()):.4f}", flush=True)

    _sync(device)
    grpo_gpuh = round((time.perf_counter() - t_grpo0) / 3600.0, 4)
    meta_out = dict(sft_meta)
    meta_out["hindsight_awr_gpuh"] = awr_gpuh
    meta_out["topogrpo_gpuh"] = grpo_gpuh
    gen.save_checkpoint(aligned_ckpt, extra_metadata=meta_out)
    print(f"[QM9 Align DONE] {aligned_ckpt} | awr_gpuh={awr_gpuh} grpo_gpuh={grpo_gpuh}", flush=True)
    return aligned_ckpt


def run_rollout_critic(args: argparse.Namespace) -> None:
    """Collect sharded 6-property rollouts (`group_size=8`) for `TwoStageOctreeProcessCritic`."""
    device = torch.device(args.device)
    out_dir = args.work_dir / args.run_name
    shard_path = out_dir / f"qm9_critic_rollout_shard_{args.shard_id}.pt"
    if args.reuse_checkpoints and shard_path.exists():
        print(f"[QM9 Critic Rollout] Reusing {shard_path}", flush=True)
        return

    splits = load_disjoint_qm9_splits(args.lmdb_dir, args.propmolflow_root / "data" / "qm9_raw", seed=args.seed)
    train_recs = splits["train"]
    train_targets_gen = _to_generator_targets(np.stack([r["props_ha"] for r in train_recs], axis=0))
    train_natoms = np.array([len(r["symbols"]) for r in train_recs], dtype=np.float32)

    gen, _ = LoomUni3DARGenerator.load_checkpoint(out_dir / "uni3dar_mol_qm9_k6_aligned.pt", device=args.device, batch_size=256)
    gen.eval()
    refiner, _ = PostTreeGraphRefiner.load_checkpoint(out_dir / "uni3dar_mol_qm9_refiner.pt", device=args.device)
    refiner.eval()
    surrogate = QM9TrainSplitSurrogateOracle(args.surrogate_ckpt, device=args.device)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)

    g_sz = 8
    n_tot_shard = int(args.critic_rollouts) // int(args.num_shards)
    n_groups = n_tot_shard // g_sz
    rng = np.random.default_rng(args.seed + 7000 + int(args.shard_id) * 911)

    t_mat = np.zeros((n_tot_shard, len(PROPS)), dtype=np.float32)
    act_mat = np.zeros((n_tot_shard, len(PROPS)), dtype=bool)
    bud_vec = np.zeros(n_tot_shard, dtype=np.float32)
    gids = np.zeros(n_tot_shard, dtype=np.int64)
    gid_base = int(args.shard_id) * 100000

    for g_i in range(n_groups):
        p_c = g_i % len(PROPS)
        r_pick = int(rng.integers(0, len(train_recs)))
        s_i, e_i = g_i * g_sz, (g_i + 1) * g_sz
        t_mat[s_i:e_i, p_c] = train_targets_gen[r_pick, p_c]
        act_mat[s_i:e_i, p_c] = True
        bud_vec[s_i:e_i] = train_natoms[r_pick]
        gids[s_i:e_i] = gid_base + g_i

    _sync(device)
    t0 = time.perf_counter()
    s1_h_l, s1_g_l, s2_h_l, s2_c_l = [], [], [], []
    s2_e_sl, s2_p_sl = [], []
    roll_mols: List[Optional[Chem.Mol]] = []
    is_leg_arr = np.zeros(n_tot_shard, dtype=bool)
    phys_arr = np.zeros(n_tot_shard, dtype=np.float32)

    for s in range(0, n_tot_shard, 256):
        e = min(s + 256, n_tot_shard)
        tn = gen.normalize_targets(t_mat[s:e])
        am = torch.from_numpy(act_mat[s:e]).to(device)
        roll = gen.sample_rollouts_with_two_stage_features(
            tn,
            am,
            cfg_scale=2.0,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=930000 + int(args.shard_id) * 50000 + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            leaf_atom_budget=bud_vec[s:e],
            budget_guidance_scale=0.55,
        )
        s1_h_l.append(roll["s1_h_concat"])
        s1_g_l.append(roll["s1_geom_feats"])
        s2_h_l.append(roll["s2_h_stage2"])
        s2_c_l.append(roll["s2_cloud_feats"])
        tr_sub = roll["trajectories"]
        refiner.refine_trajectories_batch(tr_sub, batch_size=256)
        for loc_i, tr in enumerate(tr_sub):
            s2_e_sl.append(tr["stage2_elem_ids"])
            s2_p_sl.append(tr["stage2_leaf_pos"])
            d_min = _min_pair_dist(tr["positions_np"])
            res = reconciler.reconcile_point_cloud(
                tr["atom_symbols"], tr["positions_np"], potential_slot=None, bond_order_priors=tr.get("bond_order_probs")
            )
            if res.reconciled_legal and (not res.reconciled_exotic_charge) and res.mol is not None and len(Chem.GetMolFrags(res.mol)) == 1 and d_min >= 0.72:
                roll_mols.append(res.mol)
                is_leg_arr[s + loc_i] = True
                phys_arr[s + loc_i] = float(np.clip((d_min - 0.72) / 0.28, 0.25, 1.0))
            else:
                roll_mols.append(None)

    preds_s, _, ok_s, _ = surrogate.evaluate_candidates(roll_mols, batch_size=256)
    t_mean_np = gen.target_mean.detach().cpu().numpy()
    t_scale_np = gen.target_scale.detach().cpu().numpy()
    y_norm = np.where(np.isfinite(preds_s), (preds_s - t_mean_np[None, :]) / np.maximum(t_scale_np[None, :], 1e-6), np.nan).astype(np.float32)

    _sync(device)
    wall_s = time.perf_counter() - t0
    torch.save(
        {
            "s1_h_concat": torch.cat(s1_h_l, dim=0),
            "s1_geom_feats": torch.cat(s1_g_l, dim=0),
            "s2_h_stage2": torch.cat(s2_h_l, dim=0),
            "s2_cloud_feats": torch.cat(s2_c_l, dim=0),
            "s2_elem_slices": s2_e_sl,
            "s2_pos_slices": s2_p_sl,
            "targets_norm": gen.normalize_targets(t_mat).detach().cpu(),
            "active_mask": torch.from_numpy(act_mat),
            "y_surrogate_norm": torch.from_numpy(y_norm),
            "y_valid_mask": torch.from_numpy(ok_s & is_leg_arr),
            "is_legal": torch.from_numpy(is_leg_arr.astype(np.float32)),
            "is_constr_met": torch.from_numpy(is_leg_arr.astype(np.float32)),
            "phys_scores": torch.from_numpy(phys_arr),
            "group_ids": torch.from_numpy(gids),
            "wall_sec": wall_s,
        },
        shard_path,
    )
    print(f"[QM9 Critic Shard {args.shard_id} SAVED] {shard_path} | legal={is_leg_arr.mean()*100:.1f}% | wall={wall_s:.1f}s", flush=True)


def run_fit_critic_and_bundle(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    out_dir = args.work_dir / args.run_name
    bundled_ckpt = out_dir / "uni3dar_mol_qm9_k6_bundled.pt"

    shards = [torch.load(out_dir / f"qm9_critic_rollout_shard_{i}.pt", map_location="cpu", weights_only=False) for i in range(int(args.num_shards))]
    roll_gpuh = sum(float(sh["wall_sec"]) for sh in shards) / 3600.0
    s1_h = torch.cat([sh["s1_h_concat"] for sh in shards], dim=0)
    s1_g = torch.cat([sh["s1_geom_feats"] for sh in shards], dim=0)
    s2_h = torch.cat([sh["s2_h_stage2"] for sh in shards], dim=0)
    s2_cloud = torch.cat([sh["s2_cloud_feats"] for sh in shards], dim=0)
    t_all = torch.cat([sh["targets_norm"] for sh in shards], dim=0)
    m_all = torch.cat([sh["active_mask"] for sh in shards], dim=0)
    y_surr = torch.cat([sh["y_surrogate_norm"] for sh in shards], dim=0)
    y_ok = torch.cat([sh["y_valid_mask"] for sh in shards], dim=0)
    is_leg = torch.cat([sh["is_legal"] for sh in shards], dim=0)
    is_con = torch.cat([sh["is_constr_met"] for sh in shards], dim=0)
    phys_sc = torch.cat([sh["phys_scores"] for sh in shards], dim=0)
    gids = torch.cat([sh["group_ids"] for sh in shards], dim=0)

    all_e = [es for sh in shards for es in sh["s2_elem_slices"]]
    all_p = [ps for sh in shards for ps in sh["s2_pos_slices"]]
    n_tot = len(all_e)
    max_k = max(1, max((int(es.numel()) for es in all_e), default=1))
    pad_e = torch.zeros((n_tot, max_k), dtype=torch.long)
    pad_p = torch.zeros((n_tot, max_k, 3), dtype=torch.float32)
    pad_v = torch.zeros((n_tot, max_k), dtype=torch.bool)
    for i, (es, ps) in enumerate(zip(all_e, all_p)):
        ki = int(es.numel())
        if ki > 0:
            pad_e[i, :ki] = es
            pad_p[i, :ki] = ps
            pad_v[i, :ki] = True
    del shards

    gen, gen_payload = LoomUni3DARGenerator.load_checkpoint(out_dir / "uni3dar_mol_qm9_k6_aligned.pt", device=args.device, batch_size=256)
    refiner, ref_payload = PostTreeGraphRefiner.load_checkpoint(out_dir / "uni3dar_mol_qm9_refiner.pt", device=args.device)
    emb_d = int(gen.args.emb_dim)
    num_atom = int(gen.backbone.num_atom)
    t_mean_np = gen.target_mean.detach().cpu().numpy()
    t_scale_np = gen.target_scale.detach().cpu().numpy()

    _sync(device)
    t_f0 = time.perf_counter()
    s1_c = OctreeProcessCritic(emb_dim=emb_d, num_targets=len(PROPS), hidden_dim=512).to(device)
    s1_c.set_normalization(t_mean_np, t_scale_np)
    st1 = fit_octree_process_critic(
        s1_c,
        h_concat=s1_h,
        geom_feats=s1_g,
        targets_norm=t_all,
        active_mask=m_all,
        y_surrogate_norm=y_surr,
        y_valid_mask=y_ok,
        is_legal=is_leg,
        is_constr_met=is_con,
        phys_scores=phys_sc,
        phys_weight=0.50,
        group_ids=gids,
        ranking_weight=0.65,
        ranking_obj_mask=None,
        epochs=24,
        batch_size=256,
        lr=1.4e-3,
        seed=202695,
    )
    s2_c = OctreeElementProcessCritic(
        emb_dim=emb_d,
        num_targets=len(PROPS),
        num_atom_classes=num_atom,
        d_node=192,
        n_rbf=24,
        n_gnn_layers=3,
        hidden_dim=512,
    ).to(device)
    s2_c.set_normalization(t_mean_np, t_scale_np)
    st2 = fit_octree_element_process_critic(
        s2_c,
        h_stage2=s2_h,
        elem_ids=pad_e,
        leaf_pos=pad_p,
        leaf_valid=pad_v,
        cloud_feats=s2_cloud,
        targets_norm=t_all,
        active_mask=m_all,
        y_surrogate_norm=y_surr,
        y_valid_mask=y_ok,
        is_legal=is_leg,
        is_constr_met=is_con,
        phys_scores=phys_sc,
        phys_weight=0.55,
        group_ids=gids,
        ranking_weight=0.70,
        ranking_obj_mask=None,
        epochs=28,
        batch_size=256,
        lr=1.1e-3,
        seed=202696,
    )
    _sync(device)
    fit_gpuh = (time.perf_counter() - t_f0) / 3600.0
    crit_gpuh = round(roll_gpuh + fit_gpuh, 4)

    ts_critic = TwoStageOctreeProcessCritic(s1_c, s2_c).to(device).eval()
    refiner.eval()
    gen.attach_post_processors(post_tree_refiner=refiner, process_critic=ts_critic)

    g_meta = dict(gen_payload.get("metadata") or {})
    r_meta = dict((ref_payload.get("extra") if isinstance(ref_payload, dict) else {}) or {})
    tot_gpuh = round(
        float(g_meta.get("sft_gpuh", 0.0))
        + float(g_meta.get("hindsight_awr_gpuh", 0.0))
        + float(g_meta.get("topogrpo_gpuh", 0.0))
        + float(r_meta.get("refiner_gpuh", 0.05))
        + crit_gpuh,
        4,
    )
    bundle_meta = {
        **g_meta,
        "refiner_gpuh": float(r_meta.get("refiner_gpuh", 0.05)),
        "critic_gpuh": crit_gpuh,
        "total_training_gpuh": tot_gpuh,
        "stage1_critic_stats": st1.__dict__,
        "stage2_critic_stats": st2.__dict__,
    }
    gen.save_checkpoint(bundled_ckpt, extra_metadata=bundle_meta)
    print(f"[QM9 Bundled Checkpoint SAVED] {bundled_ckpt} | total_gpuh={tot_gpuh} | meta={bundle_meta}", flush=True)
    return bundled_ckpt


def _build_calibrated_qm9_surrogate_asset(
    dev_recs: Sequence[Dict[str, Any]],
    surrogate: QM9TrainSplitSurrogateOracle,
    reconciler: ValenceReconciler,
    calib_cache: Path,
) -> ResidualCalibratedAsset:
    """Wrap `QM9TrainSplitSurrogateOracle` in `29-D` `ResidualCalibratedAsset(stoich_v2)` fitted on `splits['dev']` (`0` GVP overlap)."""
    if not hasattr(surrogate, "backend_name"):
        surrogate.backend_name = "unimol"  # type: ignore[attr-defined]
    base_asset = SlotOracleAsset(
        surrogate,
        asset_id="qm9_train_unimol_surrogate",
        units=["Bohr^3", "eV", "eV", "eV", "D", "cal/(mol*K)"],
    )
    cal_asset = ResidualCalibratedAsset(base_asset, prior_blindspot_std=0.25)
    if calib_cache.exists():
        try:
            cal_asset.load_calibrator(calib_cache)
            return cal_asset
        except Exception:
            pass

    clean_mols: List[Chem.Mol] = []
    clean_y: List[np.ndarray] = []
    true_gen = _to_generator_targets(np.stack([r["props_ha"] for r in dev_recs[:1000]], axis=0))
    for i, r in enumerate(dev_recs[:1000]):
        res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
        if res.reconciled_legal and res.mol is not None:
            clean_mols.append(res.mol)
            clean_y.append(true_gen[i])
    y_mat = np.stack(clean_y, axis=0)
    cal_asset.seed_in_domain_anchors(clean_mols[:300], y_mat[:300], max_anchors=256, seed=20261120)
    cal_asset.ingest_feedback(clean_mols[300:], y_mat[300:])
    cal_asset.save_calibrator(calib_cache)
    print(f"[QM9 Stoich-v2 ResidualCalibratedAsset] Saved {calib_cache} ({len(clean_mols)} dev_rQM_v0 conformers, 0 GVP overlap)", flush=True)
    return cal_asset


def run_eval_qm9(args: argparse.Namespace) -> None:
    """Evaluate `Uni-3DAR-Mol` across `M in {1, 2, 4, 8, 16}` on official PropMolFlow `N=1,000` test prompts."""
    device = torch.device(args.device)
    out_dir = args.work_dir / args.run_name
    bundled_ckpt = out_dir / "uni3dar_mol_qm9_k6_bundled.pt"

    splits = load_disjoint_qm9_splits(args.lmdb_dir, args.propmolflow_root / "data" / "qm9_raw", seed=args.seed)
    assert int(splits["audit"]["overlap_train_with_gvp_rQM_v0"]) == 0
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    surrogate = QM9TrainSplitSurrogateOracle(args.surrogate_ckpt, device=args.device)
    gvp_suite = GVPBenchmarkSuite(args.propmolflow_root, device=args.device)

    cal_cache = out_dir / "qm9_dev_stoich_v2_residual_cal.pt"
    cal_asset = _build_calibrated_qm9_surrogate_asset(splits["dev"], surrogate, reconciler, cal_cache)

    train_smiles: set[str] = set()
    smi_cache = out_dir / "qm9_train_smiles.json"
    if smi_cache.exists():
        train_smiles = set(json.loads(smi_cache.read_text()))
    else:
        for r in splits["train"]:
            res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
            if res.smiles:
                m2 = Chem.MolFromSmiles(res.smiles)
                if m2 is not None:
                    train_smiles.add(Chem.MolToSmiles(Chem.RemoveHs(m2), isomericSmiles=False))
        smi_cache.write_text(json.dumps(sorted(train_smiles)))

    gen, gen_payload = LoomUni3DARGenerator.load_checkpoint(bundled_ckpt, device=args.device, batch_size=250)
    gen.args.max_seq_len = 144
    gen.eval()
    g_meta = dict(gen_payload.get("metadata") or {})

    dev_tail = splits["dev"][1000:]
    dev_tail_gen = _to_generator_targets(np.stack([r["props_ha"] for r in dev_tail], axis=0))
    dev_tail_nat = np.array([len(r["symbols"]) for r in dev_tail], dtype=np.float32)
    train_targets_gen = _to_generator_targets(np.stack([r["props_ha"] for r in splits["train"]], axis=0))
    prop_stds = np.std(train_targets_gen, axis=0).astype(np.float64)

    props_to_run = [p for p in args.props if p in PROPS]
    per_prop_out: Dict[str, Any] = {}

    # Warmup GPU kernels
    wt = torch.zeros((16, len(PROPS)), dtype=torch.float32, device=device)
    wm = torch.zeros((16, len(PROPS)), dtype=torch.bool, device=device)
    wm[:, 0] = True
    gen.generate_candidates(wt, wm, atom_budget=np.full(16, 18.0, dtype=np.float32), effort="low", cfg_scale=2.0, seed=11)
    _sync(device)

    part_prefix = "eval_free_part" if args.free_atom_budget else "eval_part"
    for prop in props_to_run:
        part_json = out_dir / f"{part_prefix}_{prop}.json"
        if args.reuse_checkpoints and part_json.exists():
            per_prop_out[prop] = json.loads(part_json.read_text())
            print(f"[QM9 Eval] Reusing {part_json}", flush=True)
            continue
        p_idx = PROPS.index(prop)
        p_std = float(prop_stds[p_idx])

        # 1. Zero-leakage 128-dev affine calibration (`alpha_p, beta_p`) + CFG check on `dev_tail`
        d_sub_y = dev_tail_gen[:128, p_idx]
        d_sub_n = None if args.free_atom_budget else dev_tail_nat[:128]
        best_cfg = 2.0
        best_alpha, best_beta = 1.0, 0.0
        best_dev_err = float("inf")
        center_p = float(np.mean(train_targets_gen[:, p_idx]))

        for cfg_try in (1.75, 2.20):
            dm = np.zeros((len(d_sub_y), len(PROPS)), dtype=np.float32)
            dm[:, p_idx] = d_sub_y
            d_mask = torch.zeros((len(d_sub_y), len(PROPS)), dtype=torch.bool, device=device)
            d_mask[:, p_idx] = True
            c_grid_d = gen.generate_candidates(
                gen.normalize_targets(dm), d_mask, atom_budget=d_sub_n, effort="low",
                cfg_scale=cfg_try, cfg_tree_decay=0.25, seed=31000 + p_idx * 97,
            )
            flat_d = [c for row in c_grid_d for c in row]
            cal_d = cal_asset.predict_mols(flat_d, batch_size=256)
            cal_mu_d = cal_d["mean"][:, p_idx]
            ok_d = cal_d["valid_mask"]
            top_preds: List[float] = []
            top_reqs: List[float] = []
            for r_i in range(len(d_sub_y)):
                b_sc, b_v = 1e18, float("nan")
                for k_i in range(len(c_grid_d[r_i])):
                    f_i = r_i * len(c_grid_d[r_i]) + k_i
                    if flat_d[f_i] is not None and bool(ok_d[f_i]) and math.isfinite(float(cal_mu_d[f_i])):
                        sc = abs(float(cal_mu_d[f_i]) - float(d_sub_y[r_i])) / p_std
                        if sc < b_sc:
                            b_sc, b_v = sc, float(cal_mu_d[f_i])
                if math.isfinite(b_v):
                    top_preds.append(b_v)
                    top_reqs.append(float(d_sub_y[r_i]))
            if len(top_preds) >= 24:
                x_c = np.asarray(top_reqs) - center_p
                y_c = np.asarray(top_preds) - center_p
                slp, bias_f = np.polyfit(x_c, y_c, 1)
                alp = float(np.clip(1.0 + 0.35 * (1.0 / max(float(slp), 0.65) - 1.0), 0.94, 1.06))
                bet = float(np.clip(-float(bias_f) * 0.65, -0.25 * p_std, 0.25 * p_std))
                dev_e = float(np.mean(np.abs((center_p + alp * y_c + bet) - np.asarray(top_reqs))))
                if dev_e < best_dev_err:
                    best_dev_err = dev_e
                    best_cfg, best_alpha, best_beta = cfg_try, round(alp, 4), round(bet, 4)

        print(f"[QM9 DevCal {prop:>5}] cfg={best_cfg} alpha={best_alpha} beta={best_beta}", flush=True)

        # 2. Official `N = 1,000` PropMolFlow test prompts across `M in {1, 2, 4, 8, 16}`
        targets_ha = _load_propmolflow_eval_targets(args.propmolflow_root, prop, args.eval_n)
        targets_gen = _target_ha_to_gen(prop, targets_ha)
        natoms_targets = _load_propmolflow_eval_natoms(args.propmolflow_root, prop, args.eval_n).astype(np.float32)
        eval_atom_budget = None if args.free_atom_budget else natoms_targets

        cal_targets = center_p + best_alpha * (targets_gen - center_p) + best_beta
        t_mat_eval = np.zeros((args.eval_n, len(PROPS)), dtype=np.float32)
        t_mat_eval[:, p_idx] = cal_targets.astype(np.float32)
        t_norm_eval = gen.normalize_targets(t_mat_eval)
        m_act_eval = torch.zeros((args.eval_n, len(PROPS)), dtype=torch.bool, device=device)
        m_act_eval[:, p_idx] = True

        tiers_res: Dict[str, Any] = {}
        for m_val in args.m_budgets:
            eff_tier = {1: "base", 2: "low", 4: "medium", 8: "high"}.get(int(m_val), "high")
            n_cand_ov = None if int(m_val) in (1, 2, 4, 8) else int(m_val)

            _sync(device)
            t_g0 = time.perf_counter()
            c_grid = gen.generate_candidates(
                t_norm_eval,
                m_act_eval,
                atom_budget=eval_atom_budget,
                effort=eff_tier,
                n_candidates=n_cand_ov,
                cfg_scale=best_cfg,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                seed=770000 + p_idx * 10000 + int(m_val) * 131,
            )
            _sync(device)
            gen_wall = time.perf_counter() - t_g0

            t_s0 = time.perf_counter()
            if int(m_val) == 1:
                sel_mols = [row[0] for row in c_grid]
                surr_wall = 0.0
            else:
                flat_m = [c for row in c_grid for c in row]
                cal_out = cal_asset.predict_mols(flat_m, batch_size=512)
                cal_mu = cal_out["mean"][:, p_idx]
                cal_sd = cal_out["std"][:, p_idx]
                ok_m = cal_out["valid_mask"]
                _sync(device)
                surr_wall = time.perf_counter() - t_s0

                sel_mols = []
                for i in range(args.eval_n):
                    t_star = float(targets_gen[i])
                    n_star = float(natoms_targets[i])
                    best_k = 0
                    best_key = (9, 1e18)
                    for k_i in range(int(m_val)):
                        f_idx = i * int(m_val) + k_i
                        mc = flat_m[f_idx]
                        if mc is None:
                            continue
                        is_cln = is_clean_organic_charge_state(mc) and len(Chem.GetMolFrags(mc)) == 1
                        if bool(ok_m[f_idx]) and math.isfinite(float(cal_mu[f_idx])):
                            sd_v = float(cal_sd[f_idx]) if math.isfinite(float(cal_sd[f_idx])) else 0.0
                            err_n = (abs(float(cal_mu[f_idx]) - t_star) + 0.12 * sd_v) / max(p_std, 1e-6)
                            sz_err = 0.0 if args.free_atom_budget else (0.05 * abs(float(mc.GetNumAtoms()) - n_star))
                            cand_key = (0 if is_cln else 1, err_n + sz_err)
                        else:
                            cand_key = (2 if is_cln else 3, abs(float(mc.GetNumAtoms()) - n_star))
                        if cand_key < best_key:
                            best_key = cand_key
                            best_k = k_i
                    sel_mols.append(c_grid[i][best_k])

            # Single-pass held-out GVP oracle evaluation (`gvp_rQM_v0`)
            gvp_ev = gvp_suite.evaluate_mols(prop, sel_mols, targets_ha)
            sel_leg = sum(1 for m in sel_mols if m is not None)
            sel_exo = sum(1 for m in sel_mols if m is not None and not is_clean_organic_charge_state(m))
            nat_errs = [abs(float(m.GetNumAtoms()) - float(natoms_targets[i])) for i, m in enumerate(sel_mols) if m is not None]
            smis = [s for s in (_canonical_smiles_2d(m) for m in sel_mols) if s]
            uniq_s = set(smis)
            nov_s = [s for s in uniq_s if s not in train_smiles]
            _, ms1, ms2, mk = gen.resolve_internal_funnel(effort=eff_tier, n_candidates=n_cand_ov, atom_budget=eval_atom_budget)
            tot_wall = gen_wall + surr_wall

            tiers_res[f"M{m_val}"] = {
                "m_budget": int(m_val),
                "internal_funnel": {"m_stage1": ms1, "m_stage2": ms2, "m_keep": mk},
                "legal_pct": round(100.0 * sel_leg / max(args.eval_n, 1), 2),
                "exotic_charge_pct": round(100.0 * sel_exo / max(sel_leg, 1), 2),
                "uniqueness_pct": round(100.0 * len(uniq_s) / max(len(smis), 1), 2),
                "novelty_pct": round(100.0 * len(nov_s) / max(len(uniq_s), 1), 2),
                "mean_natoms_abs_err": round(float(np.mean(nat_errs)), 3) if nat_errs else None,
                "gen_wall_sec": round(gen_wall, 3),
                "surrogate_wall_sec": round(surr_wall, 3),
                "end_to_end_wall_sec": round(tot_wall, 3),
                "effective_output_mol_per_sec": round(float(args.eval_n) / max(tot_wall, 1e-6), 2),
                "gvp": {
                    "n_evaluable": int(gvp_ev["n_evaluable"]),
                    "mae": round(float(gvp_ev["mae"]), 4),
                    "rmse": round(float(gvp_ev["rmse"]), 4),
                    "slope": round(float(gvp_ev["slope"]), 4),
                    "pearson_r": round(float(gvp_ev["pearson_r"]), 4),
                },
            }
            print(
                f"[QM9 Uni-3DAR-Mol {prop:>5} M={m_val:2d} ({ms1}->{ms2}->{mk})] "
                f"MAE={tiers_res[f'M{m_val}']['gvp']['mae']:.4f} r={tiers_res[f'M{m_val}']['gvp']['pearson_r']:.4f} | "
                f"Legal={tiers_res[f'M{m_val}']['legal_pct']:.1f}% | |dN|={tiers_res[f'M{m_val}']['mean_natoms_abs_err']} | "
                f"Eff={tiers_res[f'M{m_val}']['effective_output_mol_per_sec']:.2f} mol/s",
                flush=True,
            )

        per_prop_out[prop] = {
            "dev_calibration": {"cfg_scale": best_cfg, "alpha": best_alpha, "beta": best_beta},
            "tiers": tiers_res,
        }
        part_json = out_dir / f"{part_prefix}_{prop}.json"
        part_json.write_text(json.dumps(per_prop_out[prop], indent=2) + "\n")

    if len(props_to_run) == len(PROPS):
        _merge_qm9_eval_parts(args, g_meta)


def _merge_qm9_eval_parts(args: argparse.Namespace, g_meta: Optional[Dict[str, Any]] = None) -> None:
    out_dir = args.work_dir / args.run_name
    part_prefix = "eval_free_part" if args.free_atom_budget else "eval_part"
    if g_meta is None:
        _, gp = LoomUni3DARGenerator.load_checkpoint(out_dir / "uni3dar_mol_qm9_k6_bundled.pt", device="cpu")
        g_meta = dict(gp.get("metadata") or {})
    merged_props: Dict[str, Any] = {}
    for p in PROPS:
        merged_props[p] = json.loads((out_dir / f"{part_prefix}_{p}.json").read_text())

    summary_table: Dict[str, Dict[str, Any]] = {}
    for m_val in args.m_budgets:
        mk = f"M{m_val}"
        summary_table[mk] = {
            p: merged_props[p]["tiers"][mk]["gvp"]["mae"] for p in PROPS if mk in merged_props[p]["tiers"]
        }
        effs = [merged_props[p]["tiers"][mk]["effective_output_mol_per_sec"] for p in PROPS if mk in merged_props[p]["tiers"]]
        summary_table[mk]["mean_effective_mol_per_sec"] = round(float(np.mean(effs)), 2) if effs else None

    payload = {
        "benchmark": "QM9_PropMolFlow_50k50k_Disjoint_6Property",
        "model": "Uni-3DAR-Mol + Uni-Loom (Unified 1 x K=6, Full SO(3), ScaleFreeBudget, PostTreeGraphRefiner, TwoStageOctreeProcessCritic)",
        "zero_gvp_overlap_verified": True,
        "total_training_gpuh": g_meta.get("total_training_gpuh"),
        "training_metadata": g_meta,
        "published_baselines": PUBLISHED_BASELINES,
        "summary_mae_by_m": summary_table,
        "per_property": merged_props,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[QM9 Uni-3DAR-Mol MERGED] {args.output_json}\n{json.dumps(summary_table, indent=2)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["train_sft", "train_refiner", "train_align", "rollout_critic", "fit_critic", "eval", "merge_eval"],
        required=True,
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--reuse-checkpoints", action="store_true")
    p.add_argument("--lmdb-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9_data"))
    p.add_argument("--parent-checkpoint", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9.pt"))
    p.add_argument("--propmolflow-root", type=Path, default=Path("/sjtu-combustion/wjw/chensq/PropMolFlow"))
    p.add_argument(
        "--surrogate-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_surrogate/qm9_train_unimol_surrogate.pt"),
    )
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--run-name", type=str, default="uni3dar_mol_qm9_k6")
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/qm9_uni3dar_mol_benchmark.json"),
    )
    p.add_argument("--sft-steps", type=int, default=7500)
    p.add_argument("--sft-batch-size", type=int, default=160)
    p.add_argument("--grpo-steps", type=int, default=180)
    p.add_argument("--critic-rollouts", type=int, default=12288)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--props", type=str, nargs="*", default=list(PROPS))
    p.add_argument("--free-atom-budget", action="store_true")
    p.add_argument("--m-budgets", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261120)
    args = p.parse_args()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.phase == "train_sft":
        run_train_sft(args)
    elif args.phase == "train_refiner":
        run_train_refiner(args)
    elif args.phase == "train_align":
        run_train_align(args)
    elif args.phase == "rollout_critic":
        run_rollout_critic(args)
    elif args.phase == "fit_critic":
        run_fit_critic_and_bundle(args)
    elif args.phase == "eval":
        run_eval_qm9(args)
    elif args.phase == "merge_eval":
        _merge_qm9_eval_parts(args)


if __name__ == "__main__":
    main()
