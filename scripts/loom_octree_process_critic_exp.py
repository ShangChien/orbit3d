"""Octree Process Critic (`loom v2.0`) Experiment Runner: QM9 (6 Properties) & OE62 (`HOMO + SA<=3.5`).

Phases:
  1. `train_qm9`  (speed-insensitive; can run on any GPU):
     Collects zero-leakage on-policy octree coarse summaries on `train_rQM_v0` targets from
     `1 x K=6 SFT` (`qm9_uniloom_k6_best.pt`), scores outcomes strictly with `QM9TrainSplitSurrogateOracle`
     (`0` overlap with `gvp_rQM_v0`), and trains `OctreeProcessCritic`.
  2. `eval_qm9`   (speed-critical; run on uncontended `GPU 0` or `GPU 1`):
     Benchmarks coarse-to-fine KV-cache branch-and-prune funnels (`8->1`, `16->1` zero-surrogate,
     `8->2`, `16->2`, `16->4`, `32->4`) across all 6 QM9 properties (`N=1,000` test targets/property),
     scoring only the final Top-1 molecule per target once on frozen `GVPRegressor`.
  3. `train_oe62` (speed-insensitive; can run on any GPU):
     Collects on-policy coarse octree summaries on training `HOMO` targets from `loom_v2_synthesizable_homo.pt`,
     supervises `OctreeProcessCritic` strictly with training-side `ext_unimol2` + `BuiltInRuleAsset`
     (`Clean & SA_Score <= 3.5 & SMARTS == 0`), never touching blind `audit_dpa4`.
  4. `eval_oe62`  (speed-critical; run on uncontended `GPU 0` or `GPU 1`):
     Benchmarks `M=1`, Outcome `Best-of-{2,4,8}`, and Process Funnels (`8->1`, `8->2`, `16->2`, `16->4`)
     over the 7 held-out OE62 quantiles (`1,792` deliverable proposals/arm), audited blindly by `DPA-4`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
for cand in (ROOT / "src" / "compat", ROOT / "src", ROOT):
    cstr = str(cand)
    if cand.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402
from loom.generator.molecular_state import mol_to_clean_smiles  # noqa: E402
from loom.oracles.unimol2_task import B_WEIGHT_SHA256  # noqa: E402
from loom.weaving import (  # noqa: E402
    AssetRegistry,
    AuditOracleSlot,
    BuiltInRuleAsset,
    CompositeCriticBus,
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


# ---------------------------------------------------------------------------
# Part A: Public QM9 (`1 x K=6 SFT` + `OctreeProcessCritic`)
# ---------------------------------------------------------------------------
def run_train_qm9(args: argparse.Namespace) -> None:
    from scripts.loom_qm9_benchmark import ALLOWED_QM9_ELEMENTS, PROPS, load_disjoint_qm9_splits
    from scripts.loom_qm9_k1_pipeline import QM9TrainSplitSurrogateOracle, _records_to_gen_targets

    device = torch.device(args.device)
    out_dir = args.work_dir / "octree_process_critic"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = out_dir / "qm9_k6_process_critic.pt"

    qm9_raw_dir = args.propmolflow_root / "data" / "qm9_raw"
    splits = load_disjoint_qm9_splits(args.lmdb_dir, qm9_raw_dir, dev_size=1500, holdout_size=1000, seed=20261115)
    assert int(splits["audit"]["overlap_train_with_gvp_rQM_v0"]) == 0
    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.qm9_k6_ckpt, device=args.device, batch_size=args.batch_size)
    gen.eval()
    surrogate = QM9TrainSplitSurrogateOracle(args.qm9_surrogate_ckpt, device=args.device)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)

    k6_bench_p = args.qm9_output_json.parent / "qm9_conditional_benchmark.json"
    k6_cfg_map = json.loads(k6_bench_p.read_text()).get("selected_cfg_scale", {}) if k6_bench_p.exists() else {}

    rng = np.random.default_rng(202605)
    rollouts_per_prop = int(args.qm9_rollouts_per_prop)
    num_props = len(gen.target_cols)

    all_h: List[torch.Tensor] = []
    all_geom: List[torch.Tensor] = []
    all_t_norm: List[torch.Tensor] = []
    all_m_act: List[torch.Tensor] = []
    all_mols: List[Optional[Chem.Mol]] = []
    all_clean: List[bool] = []

    t0 = time.perf_counter()
    for p_idx, prop in enumerate(PROPS):
        col_i = list(gen.target_cols).index(prop)
        train_vals = _records_to_gen_targets(splits["train"], p_idx, prop)
        chosen = rng.choice(train_vals, size=rollouts_per_prop, replace=(rollouts_per_prop > len(train_vals)))

        for s in range(0, rollouts_per_prop, args.batch_size):
            e = min(s + args.batch_size, rollouts_per_prop)
            cur_b = e - s
            mat = np.zeros((cur_b, num_props), dtype=np.float32)
            mat[:, col_i] = chosen[s:e].astype(np.float32)
            t_n = gen.normalize_targets(mat)
            m_a = torch.zeros((cur_b, num_props), dtype=torch.bool, device=device)
            m_a[:, col_i] = True

            trajs = gen.sample(
                t_n,
                m_a,
                cfg_scale=float(k6_cfg_map.get(prop, args.qm9_cfg_scale)),
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.25,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=880000 + p_idx * 10000 + s,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
                return_coarse_summary=True,
            )
            all_t_norm.append(t_n.detach().cpu())
            all_m_act.append(m_a.detach().cpu())
            for tr in trajs:
                all_h.append(tr["coarse_h_concat"])
                all_geom.append(tr["coarse_geom_feats"])
                d_min = _min_pair_dist(tr["positions_np"])
                res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
                if res.reconciled_legal and res.mol is not None:
                    conn = len(Chem.GetMolFrags(res.mol)) == 1
                    cln = (not res.reconciled_exotic_charge) and conn and (d_min >= 0.72)
                    all_mols.append(res.mol)
                    all_clean.append(bool(cln))
                else:
                    all_mols.append(None)
                    all_clean.append(False)

        print(f"[QM9 ProcCritic Rollout] {prop}: collected {rollouts_per_prop} trajectories ({time.perf_counter()-t0:.1f}s)", flush=True)

    del gen
    if device.type == "cuda":
        torch.cuda.empty_cache()

    preds_gen, _, ok_mask, _ = surrogate.evaluate_candidates(all_mols, batch_size=256)
    gen_tmp, _ = LoomUni3DARGenerator.load_checkpoint(args.qm9_k6_ckpt, device="cpu", batch_size=16)
    t_mean = gen_tmp.target_mean.detach().cpu().numpy()
    t_scale = gen_tmp.target_scale.detach().cpu().numpy()
    t_cols = list(gen_tmp.target_cols)
    emb_dim = int(gen_tmp.args.emb_dim)
    del gen_tmp

    # Align `preds_gen` (`PROPS` order) to `gen.target_cols` order
    col_order = [PROPS.index(c) for c in t_cols]
    preds_ordered = preds_gen[:, col_order].astype(np.float32)

    critic = OctreeProcessCritic(emb_dim=emb_dim, num_targets=num_props, hidden_dim=512, dropout=0.10).to(device)
    critic.set_normalization(t_mean, t_scale)

    h_cat_t = torch.stack(all_h, dim=0)
    geom_t = torch.stack(all_geom, dim=0)
    t_norm_t = torch.cat(all_t_norm, dim=0)
    m_act_t = torch.cat(all_m_act, dim=0)
    y_surr_norm = critic.normalize_y(torch.from_numpy(preds_ordered)).detach().cpu()
    y_ok_t = torch.from_numpy(ok_mask.astype(bool)) & torch.tensor(all_clean, dtype=torch.bool)
    clean_t = torch.tensor(all_clean, dtype=torch.bool)

    fit_t0 = time.perf_counter()
    stats = fit_octree_process_critic(
        critic,
        h_concat=h_cat_t,
        geom_feats=geom_t,
        targets_norm=t_norm_t,
        active_mask=m_act_t,
        y_surrogate_norm=y_surr_norm,
        y_valid_mask=y_ok_t,
        is_legal=clean_t,
        is_constr_met=clean_t,
        epochs=24,
        batch_size=256,
        lr=1.5e-3,
        seed=2026,
    )
    fit_sec = time.perf_counter() - fit_t0
    n_params = sum(p.numel() for p in critic.parameters())
    meta = {
        "total_rollouts": len(all_mols),
        "clean_rate": round(float(clean_t.float().mean().item()), 4),
        "critic_params": n_params,
        "rollout_and_surrogate_wall_sec": round(time.perf_counter() - t0 - fit_sec, 2),
        "critic_fit_wall_sec": round(fit_sec, 2),
        "train_stats": stats.__dict__,
    }
    critic.save_checkpoint(ckpt_out, extra=meta)
    (out_dir / "qm9_k6_process_critic_train.json").write_text(json.dumps(meta, indent=2))
    print(f"[QM9 ProcCritic TRAIN DONE] Saved {ckpt_out} | meta={meta}", flush=True)


def run_eval_qm9(args: argparse.Namespace) -> None:
    from scripts.loom_qm9_benchmark import ALLOWED_QM9_ELEMENTS, GVPBenchmarkSuite, PROPS, load_disjoint_qm9_splits
    from scripts.loom_qm9_k1_pipeline import (
        QM9TrainSplitSurrogateOracle,
        _load_propmolflow_eval_targets,
        _records_to_gen_targets,
        _target_ha_to_gen,
    )
    from scripts.loom_qm9_best_of_m import _load_propmolflow_eval_natoms

    device = torch.device(args.device)
    ckpt_critic = args.work_dir / "octree_process_critic" / "qm9_k6_process_critic.pt"
    critic, critic_meta = OctreeProcessCritic.load_checkpoint(ckpt_critic, device=args.device)
    critic.eval()

    qm9_raw_dir = args.propmolflow_root / "data" / "qm9_raw"
    splits = load_disjoint_qm9_splits(args.lmdb_dir, qm9_raw_dir, dev_size=1500, holdout_size=1000, seed=20261115)
    assert int(splits["audit"]["overlap_train_with_gvp_rQM_v0"]) == 0
    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.qm9_k6_ckpt, device=args.device, batch_size=args.batch_size)
    gen.eval()
    surrogate = QM9TrainSplitSurrogateOracle(args.qm9_surrogate_ckpt, device=args.device)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)

    k6_bench_p = args.qm9_output_json.parent / "qm9_conditional_benchmark.json"
    k6_cfg_map = json.loads(k6_bench_p.read_text()).get("selected_cfg_scale", {}) if k6_bench_p.exists() else {}
    gvp_suite = GVPBenchmarkSuite(args.propmolflow_root, device=args.device)

    train_smiles: set[str] = set()
    for rec in splits["train"]:
        r = reconciler.reconcile_point_cloud(rec["symbols"], rec["coords"], potential_slot=None)
        s = _canonical_smiles_2d(r.mol if r.reconciled_legal else r.raw_mol)
        if s:
            train_smiles.add(s)

    # Funnels: `(name, m_init, m_keep, use_terminal_surrogate)`
    funnels: List[Tuple[str, int, int, bool]] = [
        ("P_8_to_1_nosurr", 8, 1, False),
        ("P_16_to_1_nosurr", 16, 1, False),
        ("P_8_to_2", 8, 2, True),
        ("P_16_to_2", 16, 2, True),
        ("P_16_to_4", 16, 4, True),
        ("P_32_to_4", 32, 4, True),
    ]

    eval_n = int(args.eval_n)
    num_props = len(gen.target_cols)
    per_prop_out: Dict[str, Any] = {}

    for p_idx, prop in enumerate(PROPS):
        col_i = list(gen.target_cols).index(prop)
        train_y = _records_to_gen_targets(splits["train"], p_idx, prop)
        prop_std = float(np.std(train_y))
        targets_ha = _load_propmolflow_eval_targets(args.propmolflow_root, prop, eval_n)
        targets_gen = _target_ha_to_gen(prop, targets_ha)
        natoms_targets = _load_propmolflow_eval_natoms(args.propmolflow_root, prop, eval_n)

        def _encode_prompts(vals_1d: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
            nb = len(vals_1d)
            mat = np.zeros((nb, num_props), dtype=np.float32)
            mat[:, col_i] = vals_1d.astype(np.float32)
            tn = gen.normalize_targets(mat)
            ma = torch.zeros((nb, num_props), dtype=torch.bool, device=device)
            ma[:, col_i] = True
            return tn, ma

        prop_cfg = float(k6_cfg_map.get(prop, args.qm9_cfg_scale))
        # Warmup
        wt, wm = _encode_prompts(targets_gen[:16])
        _ = gen.sample_with_process_critic(
            wt,
            wm,
            critic,
            m_init=4,
            m_keep=1,
            legal_weight=0.85,
            cfg_scale=prop_cfg,
            cfg_tree_decay=0.25,
            geometric_guard=True,
            seed=701 + p_idx,
            use_bf16=(device.type == "cuda"),
        )
        _sync(device)

        prop_funnels: Dict[str, Any] = {}
        for f_name, m_init, m_keep, use_surr in funnels:
            prompt_bs = 250
            _sync(device)
            tg0 = time.perf_counter()
            all_groups: List[List[Dict[str, Any]]] = []
            for s in range(0, eval_n, prompt_bs):
                e = min(s + prompt_bs, eval_n)
                bt, bm = _encode_prompts(targets_gen[s:e])
                grp = gen.sample_with_process_critic(
                    bt,
                    bm,
                    critic,
                    m_init=m_init,
                    m_keep=m_keep,
                    legal_weight=0.85,
                    cfg_scale=prop_cfg,
                    cfg_tree_decay=0.25,
                    geometric_guard=True,
                    seed=910000 + p_idx * 10000 + m_init * 137 + s,
                    use_bf16=(device.type == "cuda"),
                )
                all_groups.extend(grp)
            _sync(device)
            gen_wall = time.perf_counter() - tg0

            # Reconcile ONLY the `eval_n * m_keep` surviving leaf trajectories!
            tr0 = time.perf_counter()
            rec_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_keep for _ in range(eval_n)]
            leg_grid = np.zeros((eval_n, m_keep), dtype=bool)
            exo_grid = np.zeros((eval_n, m_keep), dtype=bool)
            cln_grid = np.zeros((eval_n, m_keep), dtype=bool)
            nat_grid = np.zeros((eval_n, m_keep), dtype=np.int32)
            ppl_grid = np.full((eval_n, m_keep), 10.0, dtype=np.float32)
            flat_mols: List[Optional[Chem.Mol]] = []

            for i in range(eval_n):
                for k_i in range(m_keep):
                    tr = all_groups[i][k_i]
                    ppl_grid[i, k_i] = float(tr.get("atom_score_ppl", 10.0))
                    d_min = _min_pair_dist(tr["positions_np"])
                    res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
                    if res.reconciled_legal and res.mol is not None:
                        rec_grid[i][k_i] = res.mol
                        leg_grid[i, k_i] = True
                        exo_grid[i, k_i] = bool(res.reconciled_exotic_charge)
                        nat_grid[i, k_i] = int(res.mol.GetNumAtoms())
                        conn = len(Chem.GetMolFrags(res.mol)) == 1
                        cln_grid[i, k_i] = (not res.reconciled_exotic_charge) and conn and (d_min >= 0.72)
                        flat_mols.append(res.mol)
                    else:
                        flat_mols.append(None)
            reconcile_wall = time.perf_counter() - tr0

            if use_surr and m_keep > 1:
                _sync(device)
                ts0 = time.perf_counter()
                preds_all, _, ok_all, _ = surrogate.evaluate_candidates(flat_mols, batch_size=256)
                _sync(device)
                surrogate_wall = time.perf_counter() - ts0
                surr_grid = preds_all[:, p_idx].reshape(eval_n, m_keep)
                ok_grid = ok_all.reshape(eval_n, m_keep)
            else:
                surrogate_wall = 0.0
                surr_grid = np.zeros((eval_n, m_keep), dtype=np.float32)
                ok_grid = np.zeros((eval_n, m_keep), dtype=bool)

            end_wall = gen_wall + reconcile_wall + surrogate_wall

            modes_dict: Dict[str, Any] = {}
            modes_to_eval = ("target_only", "target_plus_natoms") if (use_surr and m_keep > 1) else ("target_only",)
            for mode in modes_to_eval:
                sel_mols: List[Optional[Chem.Mol]] = []
                sel_leg = 0
                sel_exo = 0
                sel_nerr: List[float] = []
                for i in range(eval_n):
                    t_val = float(targets_gen[i])
                    n_star = int(natoms_targets[i])
                    best_k = 0
                    best_key = (99, float("inf"))
                    for k_i in range(m_keep):
                        if not bool(leg_grid[i, k_i]) or rec_grid[i][k_i] is None:
                            ck = (4, float(k_i))
                        elif use_surr and m_keep > 1 and bool(ok_grid[i, k_i]):
                            n_err = abs(float(surr_grid[i, k_i]) - t_val) / max(prop_std, 1e-6)
                            sc = n_err + (0.08 * abs(int(nat_grid[i, k_i]) - n_star) if mode == "target_plus_natoms" else 0.0)
                            tier = 0 if (bool(cln_grid[i, k_i]) and float(ppl_grid[i, k_i]) <= 4.5) else (1 if bool(cln_grid[i, k_i]) else 2)
                            ck = (tier, sc)
                        else:
                            tier = 0 if bool(cln_grid[i, k_i]) else 2
                            ck = (tier, float(k_i))
                        if ck < best_key:
                            best_key = ck
                            best_k = k_i
                    chosen = rec_grid[i][best_k] if bool(leg_grid[i, best_k]) else None
                    sel_mols.append(chosen)
                    if chosen is not None:
                        sel_leg += 1
                        if bool(exo_grid[i, best_k]):
                            sel_exo += 1
                        sel_nerr.append(abs(int(nat_grid[i, best_k]) - n_star))

                gvp_res = gvp_suite.evaluate_mols(prop, sel_mols, targets_ha)
                smis = [s for s in (_canonical_smiles_2d(m) for m in sel_mols) if s]
                modes_dict[mode] = {
                    "m_init": m_init,
                    "m_keep": m_keep,
                    "use_terminal_surrogate": use_surr,
                    "legal_pct": round(100.0 * sel_leg / eval_n, 2),
                    "exotic_charge_pct": round(100.0 * sel_exo / max(sel_leg, 1), 2),
                    "uniqueness_pct": round(100.0 * len(set(smis)) / max(len(smis), 1), 2),
                    "novelty_pct": round(100.0 * sum(1 for s in smis if s not in train_smiles) / max(len(smis), 1), 2),
                    "mean_natoms_abs_err": round(float(np.mean(sel_nerr)), 3) if sel_nerr else None,
                    "gen_wall_seconds": round(gen_wall, 3),
                    "reconcile_wall_seconds": round(reconcile_wall, 3),
                    "surrogate_wall_seconds": round(surrogate_wall, 3),
                    "end_to_end_wall_seconds": round(end_wall, 3),
                    "effective_output_mol_per_sec": round(float(eval_n) / max(end_wall, 1e-6), 2),
                    "gvp": {
                        "n_evaluable": gvp_res["n_evaluable"],
                        "mae": round(gvp_res["mae"], 4),
                        "rmse": round(gvp_res["rmse"], 4),
                        "slope": round(gvp_res["slope"], 4),
                        "pearson_r": round(gvp_res["pearson_r"], 4),
                    },
                }

            prop_funnels[f_name] = modes_dict
            m0 = modes_dict["target_only"]
            print(
                f"[QM9 ProcCritic {prop} | {f_name}] MAE={m0['gvp']['mae']:.4f} (r={m0['gvp']['pearson_r']:.3f}) "
                f"Leg={m0['legal_pct']:.1f}% Eff={m0['effective_output_mol_per_sec']:.2f} mol/s",
                flush=True,
            )

        per_prop_out[prop] = prop_funnels

    summary = {
        "experiment": "QM9 Octree Process Critic Coarse-to-Fine Tree-Pruning Pareto",
        "zero_gvp_overlap_verified": True,
        "generator_checkpoint": str(args.qm9_k6_ckpt),
        "process_critic_meta": critic_meta,
        "eval_n": eval_n,
        "per_property": per_prop_out,
    }
    args.qm9_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.qm9_output_json.write_text(json.dumps(summary, indent=2))
    print(f"[QM9 ProcCritic EVAL DONE] Wrote {args.qm9_output_json}", flush=True)


# ---------------------------------------------------------------------------
# Part B: OE62 Synthesizable HOMO (`loom_v2_synthesizable_homo.pt` + `OctreeProcessCritic`)
# ---------------------------------------------------------------------------
def _build_oe62_critic_bus(args: argparse.Namespace, *, include_blind_auditor: bool) -> Tuple[CompositeCriticBus, set[str]]:
    from scripts.loom_train_sft import load_sdf_with_targets

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

    source_graphs = {
        mol_to_clean_smiles(m)
        for m in Chem.SDMolSupplier(str(args.oe62_source_train_sdf), removeHs=False)
        if m is not None
    }
    replay_mols_all, _ = load_sdf_with_targets(args.oe62_replay_sdf, ["homo", "lumo"])
    seen_charges = frozenset(formal_charge_signature(m) for m in replay_mols_all)
    bus = CompositeCriticBus(spec=spec, registry=registry, seen_charge_signatures=seen_charges)
    return bus, source_graphs


def run_train_oe62(args: argparse.Namespace) -> None:
    from scripts.loom_train_sft import load_sdf_with_targets

    device = torch.device(args.device)
    out_dir = args.work_dir / "octree_process_critic"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = out_dir / "oe62_v20_process_critic.pt"

    # Strictly NO `audit_dpa4` during Process Critic training!
    bus, _ = _build_oe62_critic_bus(args, include_blind_auditor=False)
    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.oe62_v20_ckpt, device=args.device, batch_size=args.batch_size)
    gen.eval()

    _, train_labels = load_sdf_with_targets(args.oe62_master_train_sdf, ["homo"])
    homo_train = train_labels[:, 0]
    p04, p96 = np.percentile(homo_train, [4.0, 96.0])
    rng = np.random.default_rng(202609)

    n_rollouts = int(args.oe62_rollouts)
    # 50% empirical training targets + 50% uniform quantile targets across `[P04, P96]`
    half = n_rollouts // 2
    goals = np.concatenate(
        [
            rng.choice(homo_train, size=half, replace=False),
            rng.uniform(p04, p96, size=n_rollouts - half),
        ]
    ).astype(np.float32)
    rng.shuffle(goals)

    prop_idx = gen.target_cols.index("homo")
    obj_idx = bus.spec.objective_names.index("homo")
    allowed_elems = tuple(gen.args.allow_atoms.split(","))
    reconciler = ValenceReconciler(allowed_elements=allowed_elems)

    all_h: List[torch.Tensor] = []
    all_geom: List[torch.Tensor] = []
    all_t_norm: List[torch.Tensor] = []
    all_m_act: List[torch.Tensor] = []
    rec_mols: List[Optional[Chem.Mol]] = []
    rec_clean: List[bool] = []

    t0 = time.perf_counter()
    for s in range(0, n_rollouts, args.batch_size):
        e = min(s + args.batch_size, n_rollouts)
        cb = e - s
        t_mat = np.zeros((cb, gen.num_props), dtype=np.float32)
        t_mat[:, prop_idx] = goals[s:e]
        t_n = gen.normalize_targets(t_mat)
        m_a = torch.zeros((cb, gen.num_props), dtype=torch.bool, device=device)
        m_a[:, prop_idx] = True

        trajs = gen.sample(
            t_n,
            m_a,
            cfg_scale=2.00,
            xyz_cfg_scale=1.0,
            cfg_tree_decay=0.30,
            geometric_guard=True,
            temperatures=(0.85, 0.30, 0.25),
            seed=550000 + s,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
            return_coarse_summary=True,
        )
        all_t_norm.append(t_n.detach().cpu())
        all_m_act.append(m_a.detach().cpu())
        for tr in trajs:
            all_h.append(tr["coarse_h_concat"])
            all_geom.append(tr["coarse_geom_feats"])
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and res.mol is not None:
                rec_mols.append(res.mol)
                rec_clean.append(not bool(res.reconciled_exotic_charge))
            else:
                rec_mols.append(None)
                rec_clean.append(False)

        if (e % 1024 == 0) or (e == n_rollouts):
            print(f"[OE62 ProcCritic Rollout] {e}/{n_rollouts} ({time.perf_counter()-t0:.1f}s)", flush=True)

    req_mat = np.zeros((n_rollouts, len(bus.spec.objectives)), dtype=np.float64)
    req_mat[:, obj_idx] = goals.astype(np.float64)
    act_mat = np.zeros((n_rollouts, len(bus.spec.objectives)), dtype=bool)
    act_mat[:, obj_idx] = True

    ev = bus.evaluate_molecules(
        rec_mols, requested_targets=req_mat, active_mask=act_mat, include_audit=False, for_training=False, batch_size=64
    )
    homo_pred = ev["objective_preds"][:, obj_idx].astype(np.float32)
    constr_met = ev["constraints_met"].astype(bool)

    y_full = np.zeros((n_rollouts, gen.num_props), dtype=np.float32)
    y_full[:, prop_idx] = np.where(np.isfinite(homo_pred), homo_pred, 0.0)

    critic = OctreeProcessCritic(
        emb_dim=int(gen.args.emb_dim), num_targets=gen.num_props, hidden_dim=512, dropout=0.10
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
        "clean_rate": round(float(clean_arr.mean()), 4),
        "clean_and_synthesizable_rate": round(float(clean_and_synth.mean()), 4),
        "critic_params": sum(p.numel() for p in critic.parameters()),
        "rollout_and_surrogate_wall_sec": round(time.perf_counter() - t0 - fit_sec, 2),
        "critic_fit_wall_sec": round(fit_sec, 2),
        "train_stats": stats.__dict__,
    }
    critic.save_checkpoint(ckpt_out, extra=meta)
    (out_dir / "oe62_v20_process_critic_train.json").write_text(json.dumps(meta, indent=2))
    print(f"[OE62 ProcCritic TRAIN DONE] Saved {ckpt_out} | meta={meta}", flush=True)


def _score_oe62_selected_cohort(
    selected_mols: List[Optional[Chem.Mol]],
    selected_legal: np.ndarray,
    selected_exotic: np.ndarray,
    req_list: Sequence[float],
    bus: CompositeCriticBus,
    source_graphs: set[str],
    gen_wall_sec: float,
    reconcile_wall_sec: float,
    surrogate_wall_sec: float,
    *,
    export_sdf_path: Optional[Path] = None,
) -> Dict[str, Any]:
    n_total = len(selected_mols)
    obj_idx = bus.spec.objective_names.index("homo")
    req_mat = np.zeros((n_total, len(bus.spec.objectives)), dtype=np.float64)
    req_mat[:, obj_idx] = np.asarray(req_list, dtype=np.float64)
    act_mat = np.zeros((n_total, len(bus.spec.objectives)), dtype=bool)
    act_mat[:, obj_idx] = True

    ev = bus.evaluate_molecules(
        selected_mols,
        requested_targets=req_mat,
        active_mask=act_mat,
        include_audit=True,
        for_training=False,
        batch_size=32,
    )
    metrics = compute_loom_metrics(
        requested_targets_eV=req_list,
        reward_predictions_eV=ev["objective_preds"][:, obj_idx],
        legal_mask=selected_legal,
        evaluated_mask=np.isfinite(ev["objective_preds"][:, obj_idx]) & selected_legal,
        audit_predictions_eV=ev["audit_preds"].get("homo"),
        f_max_eV_A=ev["f_max_eV_A"],
        unseen_charge_mask=selected_exotic,
        tolerance_eV=0.25,
    )
    sa_vals = ev["constraint_values"].get("synthesizable_sa", np.full(n_total, np.nan))
    aud_preds = ev["audit_preds"].get("homo")
    clean_mask = selected_legal & (~selected_exotic)
    reward_clean_hits = int(
        np.sum(clean_mask & np.isfinite(ev["objective_preds"][:, obj_idx]) & (np.abs(ev["objective_preds"][:, obj_idx] - req_mat[:, obj_idx]) <= 0.25))
    )
    synth_clean_hits = int(np.sum(ev["constrained_clean_hits"]))
    audit_clean_hits = int(
        np.sum(clean_mask & np.isfinite(aud_preds) & (np.abs(aud_preds - req_mat[:, obj_idx]) <= 0.25))
        if aud_preds is not None
        else 0
    )
    audit_synth_clean_hits = int(
        np.sum(
            clean_mask
            & ev["constraints_met"]
            & np.isfinite(aud_preds)
            & (np.abs(aud_preds - req_mat[:, obj_idx]) <= 0.25)
        )
        if aud_preds is not None
        else 0
    )
    synth_smiles = {
        mol_to_clean_smiles(selected_mols[i])
        for i in range(n_total)
        if ev["constrained_clean_hits"][i] and selected_mols[i] is not None
    }
    end_wall = gen_wall_sec + reconcile_wall_sec + surrogate_wall_sec
    if export_sdf_path is not None:
        export_sdf_path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(export_sdf_path))
        for i in range(n_total):
            if not bool(selected_legal[i]) or selected_mols[i] is None:
                continue
            cm = Chem.Mol(selected_mols[i])
            uh = float(ev["objective_preds"][i, obj_idx])
            dh = float(aud_preds[i]) if (aud_preds is not None and np.isfinite(aud_preds[i])) else float("nan")
            sa_i = float(sa_vals[i]) if np.isfinite(sa_vals[i]) else float("nan")
            fm_i = float(ev["f_max_eV_A"][i]) if np.isfinite(ev["f_max_eV_A"][i]) else float("nan")
            cm.SetProp("prompt_index", str(i))
            cm.SetProp("homo_target_eV", f"{float(req_list[i]):.6f}")
            cm.SetProp("unimol2_homo_eV", f"{uh:.6f}" if np.isfinite(uh) else "nan")
            cm.SetProp("dpa4_homo_eV", f"{dh:.6f}" if np.isfinite(dh) else "nan")
            cm.SetProp("sa_score", f"{sa_i:.4f}" if np.isfinite(sa_i) else "nan")
            cm.SetProp("f_max_eV_A", f"{fm_i:.4f}" if np.isfinite(fm_i) else "nan")
            cm.SetProp("is_clean_closed_shell", str(int(bool(clean_mask[i]))))
            cm.SetProp("is_synthesizable_clean_hit", str(int(bool(ev["constrained_clean_hits"][i]))))
            cm.SetProp(
                "is_blind_dpa4_clean_hit",
                str(int(bool(clean_mask[i]) and np.isfinite(dh) and abs(dh - float(req_list[i])) <= 0.25)),
            )
            writer.write(cm)
        writer.close()
    metrics.update(
        {
            "clean_hits": reward_clean_hits,
            "clean_hit_yield": round(reward_clean_hits / n_total, 5),
            "synthesizable_clean_hits": synth_clean_hits,
            "synthesizable_clean_hit_yield": round(synth_clean_hits / n_total, 5),
            "blind_dpa4_clean_hits": audit_clean_hits,
            "blind_dpa4_clean_hit_yield": round(audit_clean_hits / n_total, 5),
            "blind_dpa4_synthesizable_clean_hits": audit_synth_clean_hits,
            "blind_dpa4_synthesizable_clean_hit_yield": round(audit_synth_clean_hits / n_total, 5),
            "unique_synthesizable_hit_graphs": len(synth_smiles),
            "novel_unique_synthesizable_hit_graphs": len(synth_smiles - source_graphs),
            "constraints_met_rate_among_legal": round(float(ev["constraints_met"][selected_legal].mean()), 5) if selected_legal.any() else 0.0,
            "median_sa_score_among_legal": round(float(np.nanmedian(sa_vals[selected_legal])), 4) if selected_legal.any() and np.isfinite(sa_vals[selected_legal]).any() else None,
            "sa_leq_3_5_rate_among_legal": round(float(np.mean(sa_vals[selected_legal] <= 3.5)), 5) if selected_legal.any() else 0.0,
            "audit_mae_eV": (metrics["audit"]["audit_mae_eV"] if metrics.get("audit") else None),
            "audit_minus_reward_mae_gap_eV": (
                metrics["audit"]["audit_minus_reward_mae_gap_eV"] if metrics.get("audit") else None
            ),
            "gen_wall_seconds": round(gen_wall_sec, 3),
            "reconcile_wall_seconds": round(reconcile_wall_sec, 3),
            "surrogate_wall_seconds": round(surrogate_wall_sec, 3),
            "end_to_end_wall_seconds": round(end_wall, 3),
            "effective_output_mol_per_sec": round(float(n_total) / max(end_wall, 1e-6), 2),
        }
    )
    return metrics


def _select_oe62_from_pool(
    mols_grid: List[List[Optional[Chem.Mol]]],
    legal_grid: np.ndarray,
    exotic_grid: np.ndarray,
    req_list: Sequence[float],
    bus: CompositeCriticBus,
    m_keep: int,
    use_surrogate: bool,
    device: torch.device,
) -> Tuple[List[Optional[Chem.Mol]], np.ndarray, np.ndarray, float]:
    n_prompts = len(req_list)
    if not use_surrogate or m_keep == 1:
        sel_m = [mols_grid[i][0] for i in range(n_prompts)]
        return sel_m, legal_grid[:, 0].copy(), exotic_grid[:, 0].copy(), 0.0

    obj_idx = bus.spec.objective_names.index("homo")
    flat_mols = [mols_grid[i][k] for i in range(n_prompts) for k in range(m_keep)]
    req_rep = np.repeat(np.asarray(req_list, dtype=np.float64), m_keep)
    req_mat = np.zeros((len(flat_mols), len(bus.spec.objectives)), dtype=np.float64)
    req_mat[:, obj_idx] = req_rep
    act_mat = np.zeros((len(flat_mols), len(bus.spec.objectives)), dtype=bool)
    act_mat[:, obj_idx] = True

    _sync(device)
    ts0 = time.perf_counter()
    # Surrogate selection uses ONLY training-side `ext_unimol2` + `builtin_rules`, NEVER `include_audit=True` or `potential_slot`!
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
        goal = float(req_list[i])
        best_k = 0
        best_key = (99, float("inf"))
        for k in range(m_keep):
            if not bool(legal_grid[i, k]) or mols_grid[i][k] is None:
                ck = (4, float(k))
            else:
                cln = not bool(exotic_grid[i, k])
                c_ok = (sa_p[i, k] <= 3.5) and (sm_p[i, k] <= 0.0) and (mw_p[i, k] <= 550.0)
                err = abs(float(homo_p[i, k]) - goal) if math.isfinite(float(homo_p[i, k])) else 99.0
                sa_pen = max(0.0, float(sa_p[i, k]) - 3.5) * 0.25
                if cln and c_ok:
                    tier = 0
                elif cln:
                    tier = 1
                else:
                    tier = 2
                ck = (tier, err + sa_pen)
            if ck < best_key:
                best_key = ck
                best_k = k
        sel_mols.append(mols_grid[i][best_k] if bool(legal_grid[i, best_k]) else None)
        sel_leg[i] = bool(legal_grid[i, best_k])
        sel_exo[i] = bool(exotic_grid[i, best_k])

    return sel_mols, sel_leg, sel_exo, surr_wall


def run_eval_oe62(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt_critic = args.work_dir / "octree_process_critic" / "oe62_v20_process_critic.pt"
    critic, critic_meta = OctreeProcessCritic.load_checkpoint(ckpt_critic, device=args.device)
    critic.eval()

    bus, source_graphs = _build_oe62_critic_bus(args, include_blind_auditor=True)
    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.oe62_v20_ckpt, device=args.device, batch_size=args.batch_size)
    gen.eval()

    targets_cfg = json.loads(args.oe62_targets_json.read_text())
    q_points = targets_cfg["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    seeds = (24411, 24412)
    attempts_per_seed = 128

    req_list: List[float] = []
    for q in quantiles:
        for _sd in seeds:
            req_list.extend([float(q_points[str(q)])] * attempts_per_seed)
    n_prompts = len(req_list)  # 1,792 prompts

    prop_idx = gen.target_cols.index("homo")
    allowed_elems = tuple(gen.args.allow_atoms.split(","))
    reconciler = ValenceReconciler(allowed_elements=allowed_elems)

    def _encode_oe62(vals_1d: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        nb = len(vals_1d)
        mat = np.zeros((nb, gen.num_props), dtype=np.float32)
        mat[:, prop_idx] = np.asarray(vals_1d, dtype=np.float32)
        tn = gen.normalize_targets(mat)
        ma = torch.zeros((nb, gen.num_props), dtype=torch.bool, device=device)
        ma[:, prop_idx] = True
        return tn, ma

    # Warmup
    wt, wm = _encode_oe62(req_list[:16])
    _ = gen.sample(wt, wm, cfg_scale=2.0, cfg_tree_decay=0.30, geometric_guard=True, seed=11, use_bf16=(device.type == "cuda"))
    _sync(device)

    results_arms: Dict[str, Any] = {}

    # 1. Outcome Best-of-M pool (`M_max = 8`, slicing `M in {1, 2, 4, 8}`)
    m_max = 8
    round_trajs: List[List[Dict[str, Any]]] = []
    round_gen_walls: List[float] = []
    for m_i in range(m_max):
        _sync(device)
        tg0 = time.perf_counter()
        cur_tr: List[Dict[str, Any]] = []
        for s in range(0, n_prompts, 128):
            e = min(s + 128, n_prompts)
            bt, bm = _encode_oe62(req_list[s:e])
            sub = gen.sample(
                bt,
                bm,
                cfg_scale=2.00,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.30,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=2441100 + m_i * 9973 + s,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
            )
            cur_tr.extend(sub)
        _sync(device)
        round_gen_walls.append(time.perf_counter() - tg0)
        round_trajs.append(cur_tr)

    tr0 = time.perf_counter()
    out_mols_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(n_prompts)]
    out_leg_grid = np.zeros((n_prompts, m_max), dtype=bool)
    out_exo_grid = np.zeros((n_prompts, m_max), dtype=bool)
    for i in range(n_prompts):
        for m_i in range(m_max):
            tr = round_trajs[m_i][i]
            res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
            if res.reconciled_legal and res.mol is not None:
                out_mols_grid[i][m_i] = res.mol
                out_leg_grid[i, m_i] = True
                out_exo_grid[i, m_i] = bool(res.reconciled_exotic_charge)
    rec_wall_total = time.perf_counter() - tr0

    for mb in (1, 2, 4, 8):
        sub_grid = [row[:mb] for row in out_mols_grid]
        sel_m, sel_l, sel_x, surr_w = _select_oe62_from_pool(
            sub_grid, out_leg_grid[:, :mb], out_exo_grid[:, :mb], req_list, bus, mb, use_surrogate=(mb > 1), device=device
        )
        # Polish hydrogens on selected repaired molecules (exact v2.0 Track B protocol)
        if bus.potential_slot is not None:
            reconciler.batch_polish_hydrogens_mlip(
                [m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32
            )
        gen_w = float(sum(round_gen_walls[:mb]))
        rec_w = rec_wall_total * (float(mb) / float(m_max))
        arm_res = _score_oe62_selected_cohort(
            sel_m, sel_l, sel_x, req_list, bus, source_graphs, gen_w, rec_w, surr_w
        )
        results_arms[f"Outcome_M{mb}"] = arm_res
        blind_mae_val = float("nan") if arm_res.get("audit_mae_eV") is None else float(arm_res["audit_mae_eV"])
        print(
            f"[OE62 Outcome M={mb}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"SynthCleanHits={arm_res['synthesizable_clean_hits']}/1792 ({arm_res['synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindDPA4_CleanHits={arm_res['blind_dpa4_clean_hits']} | BlindMAE={blind_mae_val:.4f}",
            flush=True,
        )

    # 2. Octree Process Critic Funnels (`8->1` zero-surrogate, `8->2`, `16->2`, `16->4`)
    proc_funnels: List[Tuple[str, int, int, bool]] = [
        ("Process_8_to_1_nosurr", 8, 1, False),
        ("Process_8_to_2", 8, 2, True),
        ("Process_16_to_2", 16, 2, True),
        ("Process_16_to_4", 16, 4, True),
    ]
    obj_col_mask = torch.zeros((gen.num_props,), dtype=torch.bool, device=device)
    obj_col_mask[prop_idx] = True

    for f_name, m_init, m_keep, use_surr in proc_funnels:
        prompt_bs = 128
        _sync(device)
        tg0 = time.perf_counter()
        groups_all: List[List[Dict[str, Any]]] = []
        for s in range(0, n_prompts, prompt_bs):
            e = min(s + prompt_bs, n_prompts)
            bt, bm = _encode_oe62(req_list[s:e])
            grp = gen.sample_with_process_critic(
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
                use_bf16=(device.type == "cuda"),
            )
            groups_all.extend(grp)
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
            p_grid, p_leg, p_exo, req_list, bus, m_keep, use_surrogate=use_surr, device=device
        )
        if bus.potential_slot is not None:
            reconciler.batch_polish_hydrogens_mlip(
                [m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32
            )
        arm_res = _score_oe62_selected_cohort(
            sel_m, sel_l, sel_x, req_list, bus, source_graphs, gen_w, rec_w, surr_w
        )
        results_arms[f_name] = arm_res
        blind_mae_val = float("nan") if arm_res.get("audit_mae_eV") is None else float(arm_res["audit_mae_eV"])
        print(
            f"[OE62 {f_name}] Eff={arm_res['effective_output_mol_per_sec']:.2f} mol/s | "
            f"SynthCleanHits={arm_res['synthesizable_clean_hits']}/1792 ({arm_res['synthesizable_clean_hit_yield']*100:.1f}%) | "
            f"BlindDPA4_CleanHits={arm_res['blind_dpa4_clean_hits']} | BlindMAE={blind_mae_val:.4f}",
            flush=True,
        )

    out_payload = {
        "experiment": "OE62 Synthesizable HOMO Octree Process Critic vs Outcome Best-of-M",
        "proposals_per_arm": n_prompts,
        "generator_checkpoint": str(args.oe62_v20_ckpt),
        "process_critic_meta": critic_meta,
        "arms": results_arms,
    }
    args.oe62_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.oe62_output_json.write_text(json.dumps(out_payload, indent=2))
    print(f"[OE62 ProcCritic EVAL DONE] Wrote {args.oe62_output_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["train_qm9", "eval_qm9", "train_oe62", "eval_oe62"], required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--work-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs"))
    # QM9 options
    p.add_argument("--lmdb-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9_data"))
    p.add_argument("--propmolflow-root", type=Path, default=Path("/sjtu-combustion/wjw/chensq/PropMolFlow"))
    p.add_argument("--qm9-k6-ckpt", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_uniloom_k6/qm9_uniloom_k6_best.pt"))
    p.add_argument("--qm9-surrogate-ckpt", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_surrogate/qm9_train_unimol_surrogate.pt"))
    p.add_argument("--qm9-cfg-scale", type=float, default=1.75)
    p.add_argument("--qm9-rollouts-per-prop", type=int, default=3000)
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--qm9-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/qm9_octree_process_critic.json"))
    # OE62 options
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument("--oe62-v20-ckpt", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/v20_synthesizable_homo/loom_v2_synthesizable_homo.pt"))
    p.add_argument("--oe62-master-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/oe62_master_train_B.sdf"))
    p.add_argument("--oe62-source-train-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/source_labels/train_source.sdf"))
    p.add_argument("--oe62-replay-sdf", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/data/oe62_consolidated/quantum_train_B.sdf"))
    p.add_argument("--oe62-targets-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_b_task_20261114/targets_B.json"))
    p.add_argument("--oe62-reward-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"))
    p.add_argument("--oe62-audit-oracle-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/orbit_e15_calibration_20261114/full_fit/oracle"))
    p.add_argument("--oe62-potential-weights", type=Path, default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"))
    p.add_argument("--oe62-rollouts", type=int, default=8192)
    p.add_argument("--oe62-output-json", type=Path, default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_octree_process_critic.json"))
    p.add_argument("--allow-tf32", action="store_true", help="Enable Ampere TF32 Tensor Core matmuls fairly.")

    args = p.parse_args()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.phase == "train_qm9":
        run_train_qm9(args)
    elif args.phase == "eval_qm9":
        run_eval_qm9(args)
    elif args.phase == "train_oe62":
        run_train_oe62(args)
    elif args.phase == "eval_oe62":
        run_eval_oe62(args)


if __name__ == "__main__":
    main()
