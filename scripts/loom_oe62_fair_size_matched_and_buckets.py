"""Fair Fixed-Prompt 30-Atom Head-to-Head (`8->4` Funnel) & Multi-Size Bucket Scaling (`N=20..60`).

Implements strictly fair cross-paradigm OE62 evaluation rules:
  1. `unanchored_8to4`:
     Evaluates `Process_8_to_4` on unanchored `Uni-Loom [Uni-3DAR]` (`1,792` prompts) and merges
     with `M=1..8, 8->1*, 8->2, 16->4` (eliminating the narrow `16->2` bottleneck).
  2. `uni3dar_30atom` & `pmf_30atom`:
     Constructs a deterministic `1,792`-prompt `(homo_i, n_atoms_i)` benchmark (`mean = 30.0` atoms,
     `n_atoms_i in [18, 42]`). Crucially, prompt `i` keeps the EXACT SAME integer `n_atoms_i` across
     ALL `M in {1..8}` candidates and across BOTH `Uni-3DAR` and `PropMolFlow`, with identical
     size-consistent surrogate candidate selection.
  3. `atom_buckets`:
     Controlled fixed-atom-count sweep across `N in {20, 30, 40, 50, 60}` (`280` prompts = `7 x 40`
     per bucket) comparing `Uni-3DAR (Size-Anchored v3: M=1, 8->4, 16->4)` against
     `PropMolFlow 52h` and `52.12h + Loom-AWR` (`M=1, M=8`) in wall-clock latency, Blind `DPA-4`
     HOMO MAE, legality, and synthesizable hit yield.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROPMOLFLOW_DEFAULT = Path("/sjtu-combustion/wjw/chensq/PropMolFlow")
MOLGEN_COMPAT_DEFAULT = Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat")
for cand in (ROOT / "src" / "compat", ROOT / "src", ROOT, MOLGEN_COMPAT_DEFAULT, PROPMOLFLOW_DEFAULT):
    cstr = str(cand)
    if cand.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)

from loom.chemistry.valence_reconciler import ValenceReconciler  # noqa: E402
from loom.weaving import LoomUni3DARGenerator, OctreeProcessCritic  # noqa: E402
from scripts.loom_octree_process_critic_exp import (  # noqa: E402
    _build_oe62_critic_bus,
    _score_oe62_selected_cohort,
    _select_oe62_from_pool,
)
from scripts.loom_oe62_propmolflow_head2head import (  # noqa: E402
    FlowMol,
    _flowmol_to_rdkit_native,
)
from scripts.loom_oe62_size_anchored import (  # noqa: E402
    _build_oe62_bus,
    _get_clean_oe62_pool,
    _get_source_novelty_sets,
    _score_oe62_size_cohort,
    _select_oe62_size_from_pool,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_fair_30atom_prompts(
    clean_targets: np.ndarray,
    targets_json: Path,
    per_quantile: int = 256,
    seed: int = 20261118,
) -> np.ndarray:
    """Build deterministic `1,792` `(homo, n_atoms)` prompts with exact integer `n_atoms` (`mean ~ 30.0`)."""
    q_points = json.loads(targets_json.read_text())["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    rng = np.random.default_rng(seed)

    # Restrict empirical conditional pool to `[18, 42]` atoms (`PMF M=8` p10..p90 support, centered ~30 atoms)
    sub_mask = (clean_targets[:, 1] >= 18.0) & (clean_targets[:, 1] <= 42.0)
    pool = clean_targets[sub_mask]
    pool_h = pool[:, 0]
    pool_n = np.round(pool[:, 1]).astype(np.int32)

    rows: List[Tuple[float, float]] = []
    for q in quantiles:
        gh = float(q_points[str(q)])
        near = np.where(np.abs(pool_h - gh) <= 0.30)[0]
        if len(near) < 48:
            near = np.argsort(np.abs(pool_h - gh))[:256]
        picked_n = rng.choice(pool_n[near], size=per_quantile, replace=True)
        for ni in picked_n.tolist():
            rows.append((gh, float(int(ni))))

    arr = np.asarray(rows, dtype=np.float32)
    # Shift integer atom counts by a tiny deterministic global offset if needed so mean is within `30.0 +- 0.3`
    shift = int(round(30.0 - float(arr[:, 1].mean())))
    if shift != 0:
        arr[:, 1] = np.clip(np.round(arr[:, 1]) + shift, 18.0, 42.0)
    return arr


# ---------------------------------------------------------------------------
# Phase 1: Add `Process_8_to_4` to Unanchored `Uni-Loom [Uni-3DAR]`
# ---------------------------------------------------------------------------
def run_unanchored_8to4(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt_critic = args.work_dir / "octree_process_critic" / "oe62_v20_process_critic.pt"
    critic, critic_meta = OctreeProcessCritic.load_checkpoint(ckpt_critic, device=args.device)
    critic.eval()

    bus, source_graphs = _build_oe62_critic_bus(args, include_blind_auditor=True)
    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.oe62_v20_ckpt, device=args.device, batch_size=512)
    gen.eval()

    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    req_list: List[float] = []
    for q in (10, 25, 35, 50, 65, 75, 90):
        for _sd in (24411, 24412):
            req_list.extend([float(q_points[str(q)])] * 128)
    n_prompts = len(req_list)

    prop_idx = gen.target_cols.index("homo")
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))
    obj_col_mask = torch.zeros((gen.num_props,), dtype=torch.bool, device=device)
    obj_col_mask[prop_idx] = True

    def _encode_oe62(vals_1d: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        nb = len(vals_1d)
        mat = np.zeros((nb, gen.num_props), dtype=np.float32)
        mat[:, prop_idx] = np.asarray(vals_1d, dtype=np.float32)
        tn = gen.normalize_targets(mat)
        ma = torch.zeros((nb, gen.num_props), dtype=torch.bool, device=device)
        ma[:, prop_idx] = True
        return tn, ma

    wt, wm = _encode_oe62(req_list[:16])
    _ = gen.sample(wt, wm, cfg_scale=2.0, cfg_tree_decay=0.30, geometric_guard=True, seed=11, use_bf16=True)
    _sync(device)

    m_init, m_keep = 8, 4
    _sync(device)
    tg0 = time.perf_counter()
    groups_all: List[List[Dict[str, Any]]] = []
    for s in range(0, n_prompts, 128):
        e = min(s + 128, n_prompts)
        bt, bm = _encode_oe62(req_list[s:e])
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
                seed=770000 + m_init * 313 + m_keep * 37 + s,
                use_bf16=True,
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

    sel_m, sel_l, sel_x, surr_w = _select_oe62_from_pool(
        p_grid, p_leg, p_exo, req_list, bus, m_keep, use_surrogate=True, device=device
    )
    if bus.potential_slot is not None:
        reconciler.batch_polish_hydrogens_mlip([m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
    arm_8_4 = _score_oe62_selected_cohort(sel_m, sel_l, sel_x, req_list, bus, source_graphs, gen_w, rec_w, surr_w)
    print(
        f"[OE62 Unanchored Process_8_to_4] Eff={arm_8_4['effective_output_mol_per_sec']:.2f} mol/s | "
        f"Atoms={arm_8_4.get('mean_total_atoms')} Scaf={arm_8_4.get('unique_murcko_scaffolds')} | "
        f"RewSynth={arm_8_4['synthesizable_clean_hits']}/1792 ({arm_8_4['synthesizable_clean_hit_yield']*100:.1f}%) | "
        f"BlindSynth={arm_8_4['blind_dpa4_synthesizable_clean_hits']} ({arm_8_4['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
        f"BlindMAE={arm_8_4['audit_mae_eV']:.4f}",
        flush=True,
    )

    base_json = json.loads(args.base_unanchored_json.read_text())
    arms_merged: Dict[str, Any] = {}
    for k in ("Outcome_M1", "Outcome_M2", "Outcome_M4", "Outcome_M8", "Process_8_to_1_nosurr", "Process_8_to_2"):
        if k in base_json["arms"]:
            arms_merged[k] = base_json["arms"][k]
    arms_merged["Process_8_to_4"] = arm_8_4
    if "Process_16_to_4" in base_json["arms"]:
        arms_merged["Process_16_to_4"] = base_json["arms"]["Process_16_to_4"]

    out = {
        "experiment": "OE62 Unanchored Uni-Loom [Uni-3DAR] with Monotone ProcessCritic Funnel (8->1*, 8->2, 8->4, 16->4)",
        "proposals_per_arm": n_prompts,
        "generator_checkpoint": str(args.oe62_v20_ckpt),
        "process_critic_meta": critic_meta,
        "arms": arms_merged,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[Unanchored 8->4 DONE] Wrote {args.output_json}", flush=True)


# ---------------------------------------------------------------------------
# Phase 2A: Fair 30-Atom Matched-Prompt `Uni-Loom [Uni-3DAR v3]` (`8->4` & `16->4`)
# ---------------------------------------------------------------------------
def run_uni3dar_30atom(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    bus = _build_oe62_bus(args, include_blind_auditor=True)
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)
    req_targets = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
    n_prompts = len(req_targets)

    v3_root = args.work_dir / "oe62_size_anchored_v3_a800"
    v20_ckpt = v3_root / "v20_aligned" / "oe62_size_v20_aligned.pt"
    critic_ckpt = v3_root / "process_critic" / "oe62_size_process_critic.pt"
    sdf_dir = v3_root / "eval_30atom" / "cohorts_sdf"

    gen, v20_payload = LoomUni3DARGenerator.load_checkpoint(v20_ckpt, device=args.device, batch_size=512)
    gen.eval()
    v20_meta = v20_payload.get("metadata") or {}
    leaf_tol = v20_meta.get("selected_leaf_budget_tol")
    critic, critic_meta = OctreeProcessCritic.load_checkpoint(critic_ckpt, device=args.device)
    critic.eval()

    print(
        f"[Fair 30-Atom Uni-3DAR] Prompts={n_prompts} | req_natoms mean={req_targets[:, 1].mean():.2f}, "
        f"med={np.median(req_targets[:, 1]):.1f}, p10={np.percentile(req_targets[:, 1], 10):.1f}, "
        f"p90={np.percentile(req_targets[:, 1], 90):.1f}",
        flush=True,
    )
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))

    def _encode_batch(rows: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tn = gen.normalize_targets(rows)
        ma = torch.ones((len(rows), 2), dtype=torch.bool, device=device)
        ua = torch.tensor([[False, True]] * len(rows), dtype=torch.bool, device=device)
        return tn, ma, ua

    wt, wm, wu = _encode_batch(req_targets[:16])
    _ = gen.sample(wt, wm, uncond_active_mask=wu, cfg_scale=2.0, cfg_tree_decay=0.30, geometric_guard=True, seed=11, use_bf16=True)
    _sync(device)

    results_arms: Dict[str, Any] = {}
    m_max = 8
    round_trajs: List[List[Dict[str, Any]]] = []
    round_gen_walls: List[float] = []
    for m_i in range(m_max):
        _sync(device)
        tg0 = time.perf_counter()
        cur_tr: List[Dict[str, Any]] = []
        for s in range(0, n_prompts, 512):
            e = min(s + 512, n_prompts)
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
                    seed=3441100 + m_i * 9973 + s,
                    use_bf16=True,
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
    del round_trajs

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
            sel_m, sel_l, sel_x, req_targets, bus, source_graphs, source_scaffolds, gen_w, rec_w, surr_w,
            export_sdf_path=(sdf_dir / f"Outcome_M{mb}.sdf"),
        )
        results_arms[f"Outcome_M{mb}"] = arm_res
        print(
            f"[30-Atom Uni-3DAR Outcome_M{mb}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
            f"RewSynth={arm_res['synthesizable_clean_hits']}/{n_prompts} ({arm_res['synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} ({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindMAE={arm_res['audit_mae_eV']:.4f}",
            flush=True,
        )

    obj_col_mask = torch.tensor([True, True], dtype=torch.bool, device=device)
    funnels: List[Tuple[str, int, int, bool]] = [
        ("Process_8_to_1_nosurr", 8, 1, False),
        ("Process_8_to_2", 8, 2, True),
        ("Process_8_to_4", 8, 4, True),
        ("Process_16_to_4", 16, 4, True),
    ]
    for f_name, m_init, m_keep, use_surr in funnels:
        _sync(device)
        tg0 = time.perf_counter()
        groups_all: List[List[Dict[str, Any]]] = []
        for s in range(0, n_prompts, 512):
            e = min(s + 512, n_prompts)
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
                    seed=880000 + m_init * 313 + m_keep * 37 + s,
                    use_bf16=True,
                    uncond_active_mask=bu,
                    coarse_chunk_size=512,
                    fine_batch_size=256,
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
            sel_m, sel_l, sel_x, req_targets, bus, source_graphs, source_scaffolds, gen_w, rec_w, surr_w,
            export_sdf_path=(sdf_dir / f"{f_name}.sdf"),
        )
        results_arms[f_name] = arm_res
        print(
            f"[30-Atom Uni-3DAR {f_name}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"Atoms={arm_res['mean_total_atoms']} (err={arm_res['mean_natoms_abs_err']}) | "
            f"RewSynth={arm_res['synthesizable_clean_hits']}/{n_prompts} ({arm_res['synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindSynth={arm_res['blind_dpa4_synthesizable_clean_hits']} ({arm_res['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindMAE={arm_res['audit_mae_eV']:.4f}",
            flush=True,
        )

    payload = {
        "experiment": "OE62 Strictly Fair Fixed-Prompt 30-Atom Matched Benchmark — Uni-Loom [Uni-3DAR Size-Anchored v3]",
        "fairness_protocol": "Identical per-prompt (homo_i, n_atoms_i) held fixed across all M in {1..8} and ProcessCritic funnels (8->1*, 8->2, 8->4, 16->4)",
        "proposals_per_arm": n_prompts,
        "requested_natoms_summary": {
            "mean": round(float(req_targets[:, 1].mean()), 2),
            "median": round(float(np.median(req_targets[:, 1])), 1),
            "p10": round(float(np.percentile(req_targets[:, 1], 10)), 1),
            "p90": round(float(np.percentile(req_targets[:, 1], 90)), 1),
        },
        "v20_checkpoint": str(v20_ckpt),
        "process_critic_meta": critic_meta,
        "arms": results_arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Fair 30-Atom Uni-3DAR DONE] Wrote {args.output_json}", flush=True)


# ---------------------------------------------------------------------------
# Phase 2B: Fair 30-Atom Fixed-Per-Prompt `PropMolFlow` Evaluation (`M in {1..8}`)
# ---------------------------------------------------------------------------
def run_pmf_30atom(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    bus = _build_oe62_bus(args, include_blind_auditor=True)
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    _, clean_targets = _get_clean_oe62_pool(args, None)
    req_targets = build_fair_30atom_prompts(clean_targets, args.oe62_targets_json, per_quantile=256, seed=20261118)
    n_prompts = len(req_targets)

    req_homo = req_targets[:, 0].tolist()
    fixed_natoms_int = [int(round(float(x))) for x in req_targets[:, 1].tolist()]

    data_dir = args.pmf_work_dir / "data"
    ck = torch.load(args.eval_ckpt, map_location="cpu", weights_only=False)
    hp = dict(ck["hyper_parameters"])
    hp["n_atoms_hist_file"] = str(data_dir / "train_data_n_atoms_histogram.pt")
    hp["marginal_dists_file"] = str(data_dir / "train_data_marginal_dists.pt")
    model = FlowMol(**hp).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    norm_file = str(data_dir / "train_data_property_normalization.pt")

    torch.manual_seed(20261118)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261118)

    _ = model.sample_random_sizes(
        16,
        device=device,
        n_timesteps=10,
        property_name="homo",
        normalization_file_path=norm_file,
        training_mode=False,
        properties_handle_method="concatenate",
        multilple_values_to_one_property=req_homo[:16],
        number_of_atoms=fixed_natoms_int[:16],
    )
    _sync(device)

    infer_bs = int(args.eval_batch_size)
    m_max = 8
    sampled_by_round: List[List[Any]] = []
    round_gen_walls: List[float] = []
    for m_i in range(m_max):
        _sync(device)
        tg0 = time.perf_counter()
        cur_mols: List[Any] = []
        for s in range(0, n_prompts, infer_bs):
            e = min(s + infer_bs, n_prompts)
            cur_mols.extend(
                model.sample_random_sizes(
                    e - s,
                    device=device,
                    n_timesteps=100,
                    property_name="homo",
                    normalization_file_path=norm_file,
                    training_mode=False,
                    properties_handle_method="concatenate",
                    multilple_values_to_one_property=req_homo[s:e],
                    number_of_atoms=fixed_natoms_int[s:e],
                )
            )
        _sync(device)
        rw = time.perf_counter() - tg0
        round_gen_walls.append(rw)
        sampled_by_round.append(cur_mols)
        print(f"[Fair 30-Atom PMF Round {m_i+1}/{m_max}] wall={rw:.2f}s ({n_prompts/max(rw,1e-6):.2f} mol/s)", flush=True)

    reconciler = ValenceReconciler(allowed_elements=tuple(hp["atom_type_map"]))

    tn0 = time.perf_counter()
    nat_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(n_prompts)]
    nat_leg = np.zeros((n_prompts, m_max), dtype=bool)
    nat_exo = np.zeros((n_prompts, m_max), dtype=bool)
    for i in range(n_prompts):
        for m_i in range(m_max):
            rd_m, is_leg, is_exo = _flowmol_to_rdkit_native(sampled_by_round[m_i][i])
            if is_leg and rd_m is not None:
                nat_grid[i][m_i] = rd_m
                nat_leg[i, m_i] = True
                nat_exo[i, m_i] = is_exo
    nat_build_wall = time.perf_counter() - tn0

    tc0 = time.perf_counter()
    crd_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(n_prompts)]
    crd_leg = np.zeros((n_prompts, m_max), dtype=bool)
    crd_exo = np.zeros((n_prompts, m_max), dtype=bool)
    for i in range(n_prompts):
        for m_i in range(m_max):
            sm = sampled_by_round[m_i][i]
            pos_np = sm.positions.detach().cpu().numpy() if torch.is_tensor(sm.positions) else np.asarray(sm.positions)
            res = reconciler.reconcile_point_cloud(list(sm.atom_types), pos_np, potential_slot=None)
            if res.reconciled_legal and res.mol is not None:
                crd_grid[i][m_i] = res.mol
                crd_leg[i, m_i] = True
                crd_exo[i, m_i] = bool(res.reconciled_exotic_charge)
    crd_build_wall = time.perf_counter() - tc0

    tracks_out: Dict[str, Dict[str, Any]] = {"native_graph": {}, "coord_reconciled": {}}
    for trk_name, grid, leg, exo, b_wall, polish_h in (
        ("native_graph", nat_grid, nat_leg, nat_exo, nat_build_wall, False),
        ("coord_reconciled", crd_grid, crd_leg, crd_exo, crd_build_wall, True),
    ):
        for mb in (1, 2, 4, 8):
            sub_g = [row[:mb] for row in grid]
            sel_m, sel_l, sel_x, surr_w = _select_oe62_size_from_pool(
                sub_g, leg[:, :mb], exo[:, :mb], req_targets, bus, mb, use_surrogate=(mb > 1), device=device
            )
            pol_w = 0.0
            if polish_h and bus.potential_slot is not None:
                _sync(device)
                tp0 = time.perf_counter()
                reconciler.batch_polish_hydrogens_mlip([m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
                _sync(device)
                pol_w = time.perf_counter() - tp0
            gen_w = float(sum(round_gen_walls[:mb]))
            rec_w = b_wall * (float(mb) / float(m_max)) + pol_w
            arm_m = _score_oe62_size_cohort(
                sel_m, sel_l, sel_x, req_targets, bus, source_graphs, source_scaffolds, gen_w, rec_w, surr_w
            )
            tracks_out[trk_name][f"M{mb}"] = arm_m
            print(
                f"[Fair 30-Atom PMF {trk_name} M={mb}] Eff={arm_m['effective_output_mol_per_sec']:.2f} mol/s | "
                f"Atoms={arm_m['mean_total_atoms']} (err={arm_m['mean_natoms_abs_err']}) | "
                f"RewSynth={arm_m['synthesizable_clean_hits']}/1792 ({arm_m['synthesizable_clean_hit_yield']*100:.1f}%) | "
                f"BlindSynth={arm_m['blind_dpa4_synthesizable_clean_hits']} ({arm_m['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%) | "
                f"BlindMAE={arm_m['audit_mae_eV']:.4f}",
                flush=True,
            )

    out = {
        "experiment": "OE62 Strictly Fair Fixed-Prompt 30-Atom Matched Benchmark — PropMolFlow",
        "fairness_protocol": "Identical per-prompt (homo_i, n_atoms_i) held fixed across all M in {1..8}",
        "checkpoint": str(args.eval_ckpt),
        "cum_gpu_hours": ck.get("cum_gpu_hours"),
        "proposals_per_arm": n_prompts,
        "requested_natoms_summary": {
            "mean": round(float(req_targets[:, 1].mean()), 2),
            "median": round(float(np.median(req_targets[:, 1])), 1),
        },
        "tracks": tracks_out,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[Fair 30-Atom PMF DONE] Wrote {args.output_json}", flush=True)


# ---------------------------------------------------------------------------
# Phase 3: Multi-Atom-Count Bucket Scaling (`N in {20, 30, 40, 50, 60}`, 280 prompts/bucket)
# ---------------------------------------------------------------------------
def run_atom_buckets(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    bus = _build_oe62_bus(args, include_blind_auditor=True)
    source_graphs, source_scaffolds = _get_source_novelty_sets(args)
    q_points = json.loads(args.oe62_targets_json.read_text())["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    per_q = int(args.bucket_per_quantile)  # default 40 -> 280 prompts/bucket

    v3_root = args.work_dir / "oe62_size_anchored_v3_a800"
    gen, _ = LoomUni3DARGenerator.load_checkpoint(
        v3_root / "v20_aligned" / "oe62_size_v20_aligned.pt", device=args.device, batch_size=280
    )
    gen.eval()
    critic, _ = OctreeProcessCritic.load_checkpoint(
        v3_root / "process_critic" / "oe62_size_process_critic.pt", device=args.device
    )
    critic.eval()
    reconciler = ValenceReconciler(allowed_elements=tuple(gen.args.allow_atoms.split(",")))

    data_dir = args.pmf_work_dir / "data"
    norm_file = str(data_dir / "train_data_property_normalization.pt")

    def _load_pmf(ckpt_p: Path) -> FlowMol:
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        hp = dict(ck["hyper_parameters"])
        hp["n_atoms_hist_file"] = str(data_dir / "train_data_n_atoms_histogram.pt")
        hp["marginal_dists_file"] = str(data_dir / "train_data_marginal_dists.pt")
        m = FlowMol(**hp).to(device)
        m.load_state_dict(ck["state_dict"], strict=True)
        m.eval()
        return m

    pmf_base = _load_pmf(args.pmf_52h_ckpt)
    pmf_awr = _load_pmf(args.pmf_52h_awr_ckpt)

    buckets = (20, 30, 40, 50, 60)
    bucket_results: Dict[str, Any] = {}

    # Warmup both backbones
    w_req = np.array([[-6.2, 30.0]] * 16, dtype=np.float32)
    _ = gen.sample(
        gen.normalize_targets(w_req),
        torch.ones((16, 2), dtype=torch.bool, device=device),
        uncond_active_mask=torch.tensor([[False, True]] * 16, dtype=torch.bool, device=device),
        cfg_scale=2.0,
        cfg_tree_decay=0.30,
        geometric_guard=True,
        seed=7,
        use_bf16=True,
    )
    _ = pmf_base.sample_random_sizes(
        16,
        device=device,
        n_timesteps=10,
        property_name="homo",
        normalization_file_path=norm_file,
        training_mode=False,
        properties_handle_method="concatenate",
        multilple_values_to_one_property=[-6.2] * 16,
        number_of_atoms=[30] * 16,
    )
    _sync(device)

    obj_col_mask = torch.tensor([True, True], dtype=torch.bool, device=device)

    for n_exact in buckets:
        req_rows = [(float(q_points[str(q)]), float(n_exact)) for q in quantiles for _ in range(per_q)]
        req_t = np.asarray(req_rows, dtype=np.float32)
        np_b = len(req_t)
        tn = gen.normalize_targets(req_t)
        ma = torch.ones((np_b, 2), dtype=torch.bool, device=device)
        ua = torch.tensor([[False, True]] * np_b, dtype=torch.bool, device=device)

        b_entry: Dict[str, Any] = {"target_n_atoms": n_exact, "n_prompts": np_b, "arms": {}}

        # 1. Uni-3DAR M=1
        _sync(device)
        t0 = time.perf_counter()
        tr_m1 = gen.sample(
            tn, ma, uncond_active_mask=ua, cfg_scale=2.00, xyz_cfg_scale=1.0, cfg_tree_decay=0.30,
            geometric_guard=True, temperatures=(0.85, 0.30, 0.25), seed=5000 + n_exact, use_bf16=True, electron_parity=0,
        )
        _sync(device)
        gw_m1 = time.perf_counter() - t0
        r0 = time.perf_counter()
        m_list: List[Optional[Chem.Mol]] = []
        l_arr = np.zeros(np_b, dtype=bool)
        x_arr = np.zeros(np_b, dtype=bool)
        for i, tr in enumerate(tr_m1):
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and res.mol is not None:
                m_list.append(res.mol)
                l_arr[i] = True
                x_arr[i] = bool(res.reconciled_exotic_charge)
            else:
                m_list.append(None)
        if bus.potential_slot is not None:
            reconciler.batch_polish_hydrogens_mlip([m for m in m_list if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
        rw_m1 = time.perf_counter() - r0
        b_entry["arms"]["Uni3DAR_M1"] = _score_oe62_size_cohort(
            m_list, l_arr, x_arr, req_t, bus, source_graphs, source_scaffolds, gw_m1, rw_m1, 0.0
        )

        # 2. Uni-3DAR Process 8->4 & 16->4
        for tag, mi, mk in (("Uni3DAR_Process_8_to_4", 8, 4), ("Uni3DAR_Process_16_to_4", 16, 4)):
            _sync(device)
            t0 = time.perf_counter()
            grps = gen.sample_with_process_critic(
                tn, ma, critic, m_init=mi, m_keep=mk, objective_col_mask=obj_col_mask,
                legal_weight=0.80, constr_weight=0.55, cfg_scale=2.00, xyz_cfg_scale=1.0, cfg_tree_decay=0.30,
                geometric_guard=True, seed=9000 + mi * 31 + n_exact, use_bf16=True, uncond_active_mask=ua,
                coarse_chunk_size=512, fine_batch_size=256,
            )
            _sync(device)
            gw = time.perf_counter() - t0
            r0 = time.perf_counter()
            pg: List[List[Optional[Chem.Mol]]] = [[None] * mk for _ in range(np_b)]
            pl = np.zeros((np_b, mk), dtype=bool)
            px = np.zeros((np_b, mk), dtype=bool)
            for i in range(np_b):
                for k in range(mk):
                    res = reconciler.reconcile_point_cloud(grps[i][k]["atom_symbols"], grps[i][k]["positions_np"], potential_slot=None)
                    if res.reconciled_legal and res.mol is not None:
                        pg[i][k] = res.mol
                        pl[i, k] = True
                        px[i, k] = bool(res.reconciled_exotic_charge)
            rw = time.perf_counter() - r0
            sm, sl, sx, sw = _select_oe62_size_from_pool(pg, pl, px, req_t, bus, mk, use_surrogate=True, device=device)
            if bus.potential_slot is not None:
                reconciler.batch_polish_hydrogens_mlip([m for m in sm if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
            b_entry["arms"][tag] = _score_oe62_size_cohort(
                sm, sl, sx, req_t, bus, source_graphs, source_scaffolds, gw, rw, sw
            )

        # 3. PropMolFlow 52h & 52h+Loom-AWR (`M=1` and `M=8`, fixed `N_atoms = n_exact`)
        req_h_list = req_t[:, 0].tolist()
        nat_list = [n_exact] * np_b
        pmf_bs = 140 if n_exact >= 50 else 280
        for p_tag, p_model in (("PMF_52h", pmf_base), ("PMF_52h_LoomAWR", pmf_awr)):
            torch.manual_seed(7000 + n_exact)
            rounds_sm: List[List[Any]] = []
            r_walls: List[float] = []
            for m_i in range(8):
                _sync(device)
                t0 = time.perf_counter()
                cur_s: List[Any] = []
                for s in range(0, np_b, pmf_bs):
                    e = min(s + pmf_bs, np_b)
                    cur_s.extend(
                        p_model.sample_random_sizes(
                            e - s, device=device, n_timesteps=100, property_name="homo",
                            normalization_file_path=norm_file, training_mode=False,
                            properties_handle_method="concatenate",
                            multilple_values_to_one_property=req_h_list[s:e],
                            number_of_atoms=nat_list[s:e],
                        )
                    )
                _sync(device)
                r_walls.append(time.perf_counter() - t0)
                rounds_sm.append(cur_s)

            r0 = time.perf_counter()
            cg: List[List[Optional[Chem.Mol]]] = [[None] * 8 for _ in range(np_b)]
            cl = np.zeros((np_b, 8), dtype=bool)
            cx = np.zeros((np_b, 8), dtype=bool)
            for i in range(np_b):
                for m_i in range(8):
                    sm_obj = rounds_sm[m_i][i]
                    p_np = sm_obj.positions.detach().cpu().numpy() if torch.is_tensor(sm_obj.positions) else np.asarray(sm_obj.positions)
                    res = reconciler.reconcile_point_cloud(list(sm_obj.atom_types), p_np, potential_slot=None)
                    if res.reconciled_legal and res.mol is not None:
                        cg[i][m_i] = res.mol
                        cl[i, m_i] = True
                        cx[i, m_i] = bool(res.reconciled_exotic_charge)
            cw_tot = time.perf_counter() - r0

            for mb in (1, 8):
                sub_g = [row[:mb] for row in cg]
                sm, sl, sx, sw = _select_oe62_size_from_pool(
                    sub_g, cl[:, :mb], cx[:, :mb], req_t, bus, mb, use_surrogate=(mb > 1), device=device
                )
                if bus.potential_slot is not None:
                    reconciler.batch_polish_hydrogens_mlip([m for m in sm if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)
                b_entry["arms"][f"{p_tag}_M{mb}"] = _score_oe62_size_cohort(
                    sm, sl, sx, req_t, bus, source_graphs, source_scaffolds,
                    float(sum(r_walls[:mb])), cw_tot * (mb / 8.0), sw
                )

        bucket_results[str(n_exact)] = b_entry
        u16 = b_entry["arms"]["Uni3DAR_Process_16_to_4"]
        p8 = b_entry["arms"]["PMF_52h_M8"]
        pa8 = b_entry["arms"]["PMF_52h_LoomAWR_M8"]
        print(
            f"[Bucket N={n_exact}] Uni3DAR 16->4: {u16['effective_output_mol_per_sec']:.2f} mol/s, "
            f"Blind={u16['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%, MAE={u16['audit_mae_eV']:.3f} | "
            f"PMF 52h M8: {p8['effective_output_mol_per_sec']:.2f} mol/s, "
            f"Blind={p8['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%, MAE={p8['audit_mae_eV']:.3f} | "
            f"PMF+AWR M8: {pa8['effective_output_mol_per_sec']:.2f} mol/s, "
            f"Blind={pa8['blind_dpa4_synthesizable_clean_hit_yield']*100:.1f}%, MAE={pa8['audit_mae_eV']:.3f}",
            flush=True,
        )

    out = {
        "experiment": "OE62 Controlled Multi-Atom-Count Scaling (N in {20, 30, 40, 50, 60}): Uni-3DAR vs PropMolFlow Speed, MAE, and Hit Yield",
        "buckets": bucket_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[Atom Buckets DONE] Wrote {args.output_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["unanchored_8to4", "uni3dar_30atom", "pmf_30atom", "atom_buckets"], required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    p.add_argument("--pmf-work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/propmolflow_oe62"))
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument("--oe62-v20-ckpt", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/v20_synthesizable_homo/loom_v2_synthesizable_homo.pt"))
    p.add_argument("--base-unanchored-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_octree_process_critic_scaf_s0.json"))
    p.add_argument("--eval-ckpt", type=Path, default=None)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--bucket-per-quantile", type=int, default=40)
    p.add_argument("--pmf-52h-ckpt", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/propmolflow_oe62/checkpoints/ckpt_52p0gpuh.pt"))
    p.add_argument("--pmf-52h-awr-ckpt", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/propmolflow_oe62_awr/ckpt_52p0gpuh_loom_awr.pt"))
    p.add_argument("--oe62-master-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"))
    p.add_argument("--oe62-master-valid-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_valid_B.sdf"))
    p.add_argument("--oe62-source-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/source_labels/train_source.sdf"))
    p.add_argument("--oe62-replay-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_B.sdf"))
    p.add_argument("--oe62-targets-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/targets_B.json"))
    p.add_argument("--oe62-reward-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"))
    p.add_argument("--oe62-audit-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_e15_calibration_20261114/full_fit/oracle"))
    p.add_argument("--oe62-potential-weights", type=Path, default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"))
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.phase == "unanchored_8to4":
        run_unanchored_8to4(args)
    elif args.phase == "uni3dar_30atom":
        run_uni3dar_30atom(args)
    elif args.phase == "pmf_30atom":
        if args.eval_ckpt is None:
            raise ValueError("--eval-ckpt required for pmf_30atom")
        run_pmf_30atom(args)
    elif args.phase == "atom_buckets":
        run_atom_buckets(args)


if __name__ == "__main__":
    main()
