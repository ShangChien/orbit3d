"""Scheme 1: Zero-Leakage Test-Time Critic-Guided `Best-of-M` Weaving on Public QM9.

Evaluates saved `1 x K=6 SFT` and `6 x K=1 Full` (`SFT + Hindsight + TopoGRPO`) checkpoints
across candidate budgets `M in {1, 2, 4, 8, 16}` on the `N = 1,000` official PropMolFlow
test targets per property.

Strict Zero-Leakage Guarantee:
  - Candidate ranking uses ONLY:
      (a) Deterministic Track B `ValenceReconciler` (closed-shell valence & formal-charge purity),
      (b) Intrinsic octree `atom_score_ppl` & interatomic clash check (`d_min >= 0.72 A`),
      (c) `QM9TrainSplitSurrogateOracle` trained exclusively on `train_rQM_v0`
          (`0` overlap with `gvp_rQM_v0`), and optionally
      (d) Official empirical atom-count input `train_half_sampled_no_atoms_{prop}.npy`.
  - `GVPBenchmarkSuite` (`gvp_rQM_v0`) is called strictly ONCE on the single selected Top-1
    molecule per target to report final held-out MAE, RMSE, Slope, and Pearson `r`.
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

from loom.chemistry.valence_reconciler import ValenceReconciler
from loom.weaving import LoomUni3DARGenerator

from scripts.loom_qm9_benchmark import (
    ALLOWED_QM9_ELEMENTS,
    EV_PROPS,
    GVPBenchmarkSuite,
    PROPS,
    load_disjoint_qm9_splits,
)
from scripts.loom_qm9_k1_pipeline import (
    QM9TrainSplitSurrogateOracle,
    _load_propmolflow_eval_targets,
    _records_to_gen_targets,
    _target_ha_to_gen,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_propmolflow_eval_natoms(propmolflow_root: Path, prop: str, eval_n: int) -> np.ndarray:
    npy_path = (
        propmolflow_root
        / "sampling_input"
        / "in_distribution_sampling"
        / f"train_half_sampled_no_atoms_{prop}.npy"
    )
    arr = np.load(npy_path).astype(np.int32)
    if len(arr) < eval_n:
        raise ValueError(f"Expected >= {eval_n} entries in {npy_path}, found {len(arr)}")
    return arr[:eval_n]


def _min_pair_dist(coords: np.ndarray) -> float:
    pts = np.asarray(coords, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 99.0 if pts.shape[0] == 1 else 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dmat, 99.0)
    return float(dmat.min())


def _canonical_smiles_2d(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)
    except Exception:
        return None


def select_best_of_m_and_score(
    *,
    prop: str,
    prop_idx: int,
    prop_std: float,
    targets_ha: np.ndarray,
    targets_gen: np.ndarray,
    natoms_targets: np.ndarray,
    m_budget: int,
    m_max: int,
    mode: str,
    track: str,
    mols_grid: List[List[Optional[Chem.Mol]]],
    legal_grid: np.ndarray,
    exotic_grid: np.ndarray,
    clean_connected_grid: np.ndarray,
    natoms_grid: np.ndarray,
    ppl_grid: np.ndarray,
    surr_preds_grid: np.ndarray,
    surr_ok_grid: np.ndarray,
    round_gen_walls: Sequence[float],
    reconcile_wall_total: float,
    surrogate_wall_total: float,
    gvp_suite: GVPBenchmarkSuite,
    train_smiles: set[str],
) -> Dict[str, Any]:
    """Select 1 candidate out of `0..m_budget-1` per target using zero-leakage surrogate & evaluate on GVP."""
    n_targets = len(targets_gen)
    selected_mols: List[Optional[Chem.Mol]] = []
    sel_legal = 0
    sel_exotic = 0
    sel_natoms_err: List[float] = []

    for i in range(n_targets):
        t_val = float(targets_gen[i])
        n_star = int(natoms_targets[i])
        best_m = 0
        best_key = (99, float("inf"))

        for m in range(m_budget):
            is_leg = bool(legal_grid[i, m])
            if not is_leg or mols_grid[i][m] is None:
                cand_key = (4, float(m))
            else:
                is_cln = bool(clean_connected_grid[i, m])
                has_surr = bool(surr_ok_grid[i, m]) and math.isfinite(float(surr_preds_grid[i, m]))
                ppl_ok = float(ppl_grid[i, m]) <= 4.5
                if has_surr:
                    norm_prop_err = abs(float(surr_preds_grid[i, m]) - t_val) / max(prop_std, 1e-6)
                    if mode == "target_plus_natoms":
                        score = norm_prop_err + 0.08 * abs(int(natoms_grid[i, m]) - n_star)
                    else:
                        score = norm_prop_err
                else:
                    score = 100.0 + float(ppl_grid[i, m])

                if is_cln and has_surr and ppl_ok:
                    tier = 0
                elif is_cln and has_surr:
                    tier = 1
                elif is_leg and not bool(exotic_grid[i, m]):
                    tier = 2
                else:
                    tier = 3
                cand_key = (tier, score)

            if cand_key < best_key:
                best_key = cand_key
                best_m = m

        chosen_mol = mols_grid[i][best_m] if bool(legal_grid[i, best_m]) else None
        selected_mols.append(chosen_mol)
        if chosen_mol is not None:
            sel_legal += 1
            if bool(exotic_grid[i, best_m]):
                sel_exotic += 1
            sel_natoms_err.append(abs(int(natoms_grid[i, best_m]) - n_star))

    gvp_res = gvp_suite.evaluate_mols(prop, selected_mols, targets_ha)

    smis = [s for s in (_canonical_smiles_2d(m) for m in selected_mols) if s]
    uniq = set(smis)
    novel = [s for s in smis if s not in train_smiles]

    gen_wall_m = float(sum(round_gen_walls[:m_budget]))
    frac = float(m_budget) / float(m_max)
    # Surrogate + reconciler run once per candidate pool (shared between raw/rec audit; count 1x per track)
    end_to_end_wall = gen_wall_m + frac * (0.5 * reconcile_wall_total + 0.5 * surrogate_wall_total)

    return {
        "m_budget": m_budget,
        "mode": mode,
        "track": track,
        "legal_count": sel_legal,
        "legal_pct": round(100.0 * sel_legal / max(n_targets, 1), 2),
        "exotic_charge_count": sel_exotic,
        "exotic_charge_pct": round(100.0 * sel_exotic / max(sel_legal, 1), 2),
        "uniqueness_pct": round(100.0 * len(uniq) / max(len(smis), 1), 2),
        "novelty_pct": round(100.0 * len(novel) / max(len(smis), 1), 2),
        "mean_natoms_abs_err": round(float(np.mean(sel_natoms_err)), 3) if sel_natoms_err else None,
        "gen_wall_seconds": round(gen_wall_m, 3),
        "end_to_end_wall_seconds": round(end_to_end_wall, 3),
        "raw_proposal_mol_per_sec": round(float(n_targets * m_budget) / max(gen_wall_m, 1e-6), 2),
        "effective_output_mol_per_sec": round(float(n_targets) / max(end_to_end_wall, 1e-6), 2),
        "gvp": {
            "n_evaluable": gvp_res["n_evaluable"],
            "mae": round(gvp_res["mae"], 4),
            "rmse": round(gvp_res["rmse"], 4),
            "slope": round(gvp_res["slope"], 4),
            "pearson_r": round(gvp_res["pearson_r"], 4),
        },
    }


def evaluate_family_best_of_m(
    *,
    family_name: str,
    ckpt_map: Dict[str, Tuple[Path, float]],
    is_k6: bool,
    m_budgets: Sequence[int],
    batch_size: int,
    eval_n: int,
    splits: Dict[str, Any],
    surrogate: QM9TrainSplitSurrogateOracle,
    reconciler: ValenceReconciler,
    gvp_suite: GVPBenchmarkSuite,
    train_smiles: set[str],
    propmolflow_root: Path,
    device_str: str,
    seed: int,
) -> Dict[str, Any]:
    device = torch.device(device_str)
    m_max = int(max(m_budgets))
    out_per_prop: Dict[str, Any] = {}

    shared_gen: Optional[LoomUni3DARGenerator] = None
    if is_k6:
        k6_ckpt = next(iter(ckpt_map.values()))[0]
        shared_gen, _ = LoomUni3DARGenerator.load_checkpoint(k6_ckpt, device=device_str, batch_size=batch_size)
        shared_gen.eval()

    for prop_idx, prop in enumerate(PROPS):
        ckpt_path, cfg_scale = ckpt_map[prop]
        gen = shared_gen
        if gen is None:
            gen, _ = LoomUni3DARGenerator.load_checkpoint(ckpt_path, device=device_str, batch_size=batch_size)
            gen.eval()

        train_y = _records_to_gen_targets(splits["train"], prop_idx, prop)
        prop_std = float(np.std(train_y))
        targets_ha = _load_propmolflow_eval_targets(propmolflow_root, prop, eval_n)
        targets_gen = _target_ha_to_gen(prop, targets_ha)
        natoms_targets = _load_propmolflow_eval_natoms(propmolflow_root, prop, eval_n)

        def _encode_batch(vals_1d: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
            n_b = len(vals_1d)
            if is_k6:
                mat = np.zeros((n_b, len(gen.target_cols)), dtype=np.float32)
                c_i = list(gen.target_cols).index(prop)
                mat[:, c_i] = vals_1d.astype(np.float32)
                t_n = gen.normalize_targets(mat)
                m_a = torch.zeros((n_b, len(gen.target_cols)), dtype=torch.bool, device=device)
                m_a[:, c_i] = True
                return t_n, m_a
            t_n = gen.normalize_targets(vals_1d[:, None].astype(np.float32))
            m_a = torch.ones((n_b, 1), dtype=torch.bool, device=device)
            return t_n, m_a

        # Warmup 1 batch
        w_n = min(batch_size, eval_n)
        w_t, w_m = _encode_batch(targets_gen[:w_n])
        _ = gen.sample(
            w_t,
            w_m,
            cfg_scale=cfg_scale,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=seed + prop_idx * 777 + 1,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
        )
        _sync(device)

        # Round-major generation: `m_max` independent passes of `eval_n` targets
        trajs_by_round: List[List[Dict[str, Any]]] = []
        round_gen_walls: List[float] = []
        for m_i in range(m_max):
            _sync(device)
            t_r0 = time.perf_counter()
            r_trajs: List[Dict[str, Any]] = []
            for s in range(0, eval_n, batch_size):
                e = min(s + batch_size, eval_n)
                b_t, b_m = _encode_batch(targets_gen[s:e])
                sub = gen.sample(
                    b_t,
                    b_m,
                    cfg_scale=cfg_scale,
                    xyz_cfg_scale=1.0,
                    cfg_tree_decay=0.25,
                    geometric_guard=True,
                    temperatures=(0.85, 0.30, 0.25),
                    seed=seed + prop_idx * 100000 + m_i * 1009 + s,
                    use_bf16=(device.type == "cuda"),
                    electron_parity=0,
                )
                r_trajs.extend(sub)
            _sync(device)
            round_gen_walls.append(time.perf_counter() - t_r0)
            trajs_by_round.append(r_trajs)

        if not is_k6:
            del gen
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Reconcile all `eval_n * m_max` candidates once
        t_rec0 = time.perf_counter()
        raw_mols_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(eval_n)]
        rec_mols_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(eval_n)]
        raw_legal_grid = np.zeros((eval_n, m_max), dtype=bool)
        raw_exotic_grid = np.zeros((eval_n, m_max), dtype=bool)
        raw_clean_grid = np.zeros((eval_n, m_max), dtype=bool)
        rec_legal_grid = np.zeros((eval_n, m_max), dtype=bool)
        rec_exotic_grid = np.zeros((eval_n, m_max), dtype=bool)
        rec_clean_grid = np.zeros((eval_n, m_max), dtype=bool)
        raw_natoms_grid = np.zeros((eval_n, m_max), dtype=np.int32)
        rec_natoms_grid = np.zeros((eval_n, m_max), dtype=np.int32)
        ppl_grid = np.full((eval_n, m_max), 10.0, dtype=np.float32)

        flat_raw_mols: List[Optional[Chem.Mol]] = []
        flat_rec_mols: List[Optional[Chem.Mol]] = []

        for i in range(eval_n):
            for m_i in range(m_max):
                tr = trajs_by_round[m_i][i]
                ppl_grid[i, m_i] = float(tr.get("atom_score_ppl", 10.0))
                d_min = _min_pair_dist(tr["positions_np"])
                clash_free = d_min >= 0.72
                res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)

                if res.raw_legal and res.raw_mol is not None:
                    raw_mols_grid[i][m_i] = res.raw_mol
                    raw_legal_grid[i, m_i] = True
                    raw_exotic_grid[i, m_i] = bool(res.raw_exotic_charge)
                    raw_natoms_grid[i, m_i] = int(res.raw_mol.GetNumAtoms())
                    conn = len(Chem.GetMolFrags(res.raw_mol)) == 1
                    raw_clean_grid[i, m_i] = (not res.raw_exotic_charge) and conn and clash_free
                    flat_raw_mols.append(res.raw_mol)
                else:
                    flat_raw_mols.append(None)

                if res.reconciled_legal and res.mol is not None:
                    rec_mols_grid[i][m_i] = res.mol
                    rec_legal_grid[i, m_i] = True
                    rec_exotic_grid[i, m_i] = bool(res.reconciled_exotic_charge)
                    rec_natoms_grid[i, m_i] = int(res.mol.GetNumAtoms())
                    conn_r = len(Chem.GetMolFrags(res.mol)) == 1
                    rec_clean_grid[i, m_i] = (not res.reconciled_exotic_charge) and conn_r and clash_free
                    flat_rec_mols.append(res.mol)
                else:
                    flat_rec_mols.append(None)

        reconcile_wall = time.perf_counter() - t_rec0

        # Score all candidates strictly on `train_rQM_v0` Uni-Mol surrogate (ZERO GVP leakage)
        _sync(device)
        t_sur0 = time.perf_counter()
        raw_preds_all, _, raw_ok_all, _ = surrogate.evaluate_candidates(flat_raw_mols, batch_size=256)
        rec_preds_all, _, rec_ok_all, _ = surrogate.evaluate_candidates(flat_rec_mols, batch_size=256)
        _sync(device)
        surrogate_wall = time.perf_counter() - t_sur0

        raw_surr_grid = raw_preds_all[:, prop_idx].reshape(eval_n, m_max)
        raw_ok_grid = raw_ok_all.reshape(eval_n, m_max)
        rec_surr_grid = rec_preds_all[:, prop_idx].reshape(eval_n, m_max)
        rec_ok_grid = rec_ok_all.reshape(eval_n, m_max)

        budgets_out: Dict[str, Any] = {}
        for m_b in m_budgets:
            b_key = f"M{m_b}"
            budgets_out[b_key] = {}
            for mode in ("target_only", "target_plus_natoms"):
                r_raw = select_best_of_m_and_score(
                    prop=prop,
                    prop_idx=prop_idx,
                    prop_std=prop_std,
                    targets_ha=targets_ha,
                    targets_gen=targets_gen,
                    natoms_targets=natoms_targets,
                    m_budget=m_b,
                    m_max=m_max,
                    mode=mode,
                    track="raw",
                    mols_grid=raw_mols_grid,
                    legal_grid=raw_legal_grid,
                    exotic_grid=raw_exotic_grid,
                    clean_connected_grid=raw_clean_grid,
                    natoms_grid=raw_natoms_grid,
                    ppl_grid=ppl_grid,
                    surr_preds_grid=raw_surr_grid,
                    surr_ok_grid=raw_ok_grid,
                    round_gen_walls=round_gen_walls,
                    reconcile_wall_total=reconcile_wall,
                    surrogate_wall_total=surrogate_wall,
                    gvp_suite=gvp_suite,
                    train_smiles=train_smiles,
                )
                r_rec = select_best_of_m_and_score(
                    prop=prop,
                    prop_idx=prop_idx,
                    prop_std=prop_std,
                    targets_ha=targets_ha,
                    targets_gen=targets_gen,
                    natoms_targets=natoms_targets,
                    m_budget=m_b,
                    m_max=m_max,
                    mode=mode,
                    track="reconciled",
                    mols_grid=rec_mols_grid,
                    legal_grid=rec_legal_grid,
                    exotic_grid=rec_exotic_grid,
                    clean_connected_grid=rec_clean_grid,
                    natoms_grid=rec_natoms_grid,
                    ppl_grid=ppl_grid,
                    surr_preds_grid=rec_surr_grid,
                    surr_ok_grid=rec_ok_grid,
                    round_gen_walls=round_gen_walls,
                    reconcile_wall_total=reconcile_wall,
                    surrogate_wall_total=surrogate_wall,
                    gvp_suite=gvp_suite,
                    train_smiles=train_smiles,
                )
                budgets_out[b_key][mode] = {"raw": r_raw, "reconciled": r_rec}

        m1_rec = budgets_out["M1"]["target_only"]["reconciled"]
        m8_to = budgets_out.get("M8", budgets_out[f"M{m_max}"])["target_only"]["reconciled"]
        m16_tn = budgets_out[f"M{m_max}"]["target_plus_natoms"]["reconciled"]
        print(
            f"[Scheme1 {family_name} {prop}] "
            f"M=1 RecMAE={m1_rec['gvp']['mae']} ({m1_rec['effective_output_mol_per_sec']} mol/s) -> "
            f"M=8(T) RecMAE={m8_to['gvp']['mae']} (r={m8_to['gvp']['pearson_r']}, Leg={m8_to['legal_pct']}%, {m8_to['effective_output_mol_per_sec']} mol/s) -> "
            f"M={m_max}(T+N) RecMAE={m16_tn['gvp']['mae']} (r={m16_tn['gvp']['pearson_r']}, Leg={m16_tn['legal_pct']}%, {m16_tn['effective_output_mol_per_sec']} mol/s)",
            flush=True,
        )

        out_per_prop[prop] = {
            "checkpoint": str(ckpt_path),
            "cfg_scale": cfg_scale,
            "m_max_pool": m_max,
            "gen_wall_total_seconds": round(float(sum(round_gen_walls)), 3),
            "reconcile_wall_total_seconds": round(reconcile_wall, 3),
            "surrogate_wall_total_seconds": round(surrogate_wall, 3),
            "budgets": budgets_out,
        }

    if shared_gen is not None:
        del shared_gen
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out_per_prop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmdb-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9_data"))
    p.add_argument("--propmolflow-root", type=Path, default=Path("/sjtu-combustion/wjw/chensq/PropMolFlow"))
    p.add_argument(
        "--surrogate-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_surrogate/qm9_train_unimol_surrogate.pt"),
    )
    p.add_argument(
        "--k6-checkpoint",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_uniloom_k6/qm9_uniloom_k6_best.pt"),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results"),
    )
    p.add_argument("--m-budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=20261116)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    qm9_raw_dir = args.propmolflow_root / "data" / "qm9_raw"
    splits = load_disjoint_qm9_splits(args.lmdb_dir, qm9_raw_dir, dev_size=1500, holdout_size=1000, seed=20261115)
    assert int(splits["audit"]["overlap_train_with_gvp_rQM_v0"]) == 0

    surrogate = QM9TrainSplitSurrogateOracle(args.surrogate_ckpt, device=args.device)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    gvp_suite = GVPBenchmarkSuite(args.propmolflow_root, device=args.device)

    print("[Scheme1] Indexing canonical train SMILES for novelty check...", flush=True)
    train_smiles: set[str] = set()
    for r in splits["train"]:
        res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
        if res.smiles:
            m2d = Chem.MolFromSmiles(res.smiles)
            if m2d is not None:
                train_smiles.add(Chem.MolToSmiles(Chem.RemoveHs(m2d), isomericSmiles=False))
    print(f"[Scheme1] Indexed {len(train_smiles)} unique train SMILES.", flush=True)

    k6_json = json.loads((args.results_dir / "qm9_conditional_benchmark.json").read_text())
    k6_cfg = k6_json.get("selected_cfg_scale", {p: 2.0 for p in PROPS})
    k6_map = {p: (args.k6_checkpoint, float(k6_cfg.get(p, 2.0))) for p in PROPS}

    k1_full_map: Dict[str, Tuple[Path, float]] = {}
    for p in PROPS:
        d_p = json.loads((args.results_dir / f"qm9_k1_{p}.json").read_text())
        k1_full_map[p] = (Path(d_p["k1_full"]["checkpoint"]), float(d_p["k1_full"]["cfg_scale"]))

    print("\n=== Running Scheme 1 Best-of-M on Unified K=6 SFT ===", flush=True)
    k6_res = evaluate_family_best_of_m(
        family_name="k6_sft",
        ckpt_map=k6_map,
        is_k6=True,
        m_budgets=args.m_budgets,
        batch_size=args.batch_size,
        eval_n=args.eval_n,
        splits=splits,
        surrogate=surrogate,
        reconciler=reconciler,
        gvp_suite=gvp_suite,
        train_smiles=train_smiles,
        propmolflow_root=args.propmolflow_root,
        device_str=args.device,
        seed=args.seed,
    )

    print("\n=== Running Scheme 1 Best-of-M on Dedicated 6 x K=1 Full ===", flush=True)
    k1_res = evaluate_family_best_of_m(
        family_name="k1_full",
        ckpt_map=k1_full_map,
        is_k6=False,
        m_budgets=args.m_budgets,
        batch_size=args.batch_size,
        eval_n=args.eval_n,
        splits=splits,
        surrogate=surrogate,
        reconciler=reconciler,
        gvp_suite=gvp_suite,
        train_smiles=train_smiles,
        propmolflow_root=args.propmolflow_root,
        device_str=args.device,
        seed=args.seed + 500000,
    )

    payload = {
        "experiment": "QM9_Scheme1_ZeroLeakage_BestOfM_Weaving",
        "zero_gvp_overlap_verified": True,
        "m_budgets": list(args.m_budgets),
        "batch_size": args.batch_size,
        "eval_n": args.eval_n,
        "k6_sft": k6_res,
        "k1_full": k1_res,
    }
    out_path = args.results_dir / "qm9_scheme1_best_of_m.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[Scheme1 DONE] Saved full Pareto table to {out_path}", flush=True)


if __name__ == "__main__":
    main()
