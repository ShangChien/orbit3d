"""`loom v2.0` Phase 3 End-to-End Grounded RSI Closed-Loop Runner (`loom_run_rsi_cycle.py`).

Executes a complete 2-round Grounded Recursive Self-Improvement (RSI) closed loop:
  - Round 0 : Fit `HeteroscedasticGPAtlas1D` + `RSIDiagnoser` (`CONVERGED`, `INNER_UNDEREXPLORED`,
              `OUTER_BLINDSPOT`, `PRIOR_CEILING`) + `Pre-Generation Gatekeeper` check.
  - Round 1A: Zero-external-cost `InnerRSIRunner` focusing curriculum on `INNER_UNDEREXPLORED` while
              strictly freezing online RL weight on uncalibrated `OUTER_BLINDSPOT` intervals.
  - Round 1B: `OuterRSIAcquisition` exports scaffold-diverse `acquisition_batch_r1.sdf` from
              `OUTER_BLINDSPOT`; `IncrementalCalibrator` ingests isolated external real `DFT` feedback
              into `ResidualCalibratedAsset` (zero base-model weight modification; `quantum_test_dft.sdf`
              and `AuditOracle` kept 100% blind!).
  - Round 2 : Re-diagnose with calibrated predictor (unlocking `OUTER_BLINDSPOT -> INNER_UNDEREXPLORED`),
              run Round 2 `InnerRSIRunner`, and verify monotonic improvement on blind `DFT` & `AuditOracle`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from loom.generator.molecular_state import mol_to_clean_smiles
from loom.oracles.applicability import MorganDomain
from loom.oracles.graph_features import RAW_GRAPH
from loom.weaving import (
    AssetRegistry,
    AuditOracleSlot,
    AuditorChannelSpec,
    BuiltInRuleAsset,
    CompositeCriticBus,
    ConstraintChannelSpec,
    FingerprintQSARAsset,
    HeteroscedasticGPAtlas1D,
    IncrementalCalibrator,
    InnerRSIRunner,
    LoomUni3DARGenerator,
    ObjectiveChannelSpec,
    OuterRSIAcquisition,
    PotentialSlot,
    RSIDiagnoser,
    RSIDiagnosticThresholds,
    ResidualCalibratedAsset,
    SlotOracleAsset,
    TaskWeavingSpec,
    compute_loom_metrics,
    formal_charge_signature,
)
from scripts.loom_train_sft import load_sdf_with_targets


def evaluate_rsi_stage(
    gen: LoomUni3DARGenerator,
    bus: CompositeCriticBus,
    audit_slot: AuditOracleSlot,
    q_points: Dict[str, float],
    source_homo_values: np.ndarray,
    diagnoser: RSIDiagnoser,
    *,
    quantiles: Sequence[int] = (10, 25, 35, 50, 65, 75, 90),
    seeds: Sequence[int] = (24411, 24412),
    attempts_per_seed: int = 128,
    batch_size: int = 128,
    cfg_scale: float = 1.90,
) -> Tuple[Dict[str, Any], HeteroscedasticGPAtlas1D, Dict[str, Any], Dict[str, Any]]:
    device = next(gen.parameters()).device
    prop_idx = gen.target_cols.index("homo")
    aud_col_idx = audit_slot.target_cols.index("homo")

    req_list: List[float] = []
    trajs_all: List[Dict[str, Any]] = []
    for q in quantiles:
        goal = float(q_points[str(q)])
        for sd in seeds:
            rem = attempts_per_seed
            offset = 0
            while rem > 0:
                cur_b = min(batch_size, rem)
                t_mat = np.zeros((cur_b, gen.num_props), dtype=np.float32)
                t_mat[:, prop_idx] = goal
                t_norm = gen.normalize_targets(t_mat)
                m_act = torch.zeros((cur_b, gen.num_props), dtype=torch.bool, device=device)
                m_act[:, prop_idx] = True
                trajs = gen.sample(
                    t_norm,
                    m_act,
                    cfg_scale=cfg_scale,
                    xyz_cfg_scale=1.0,
                    cfg_tree_decay=0.30,
                    geometric_guard=True,
                    temperatures=(0.85, 0.30, 0.25),
                    seed=int(sd) * 10007 + int(q) * 131 + offset,
                    use_bf16=(device.type == "cuda"),
                    electron_parity=0,
                )
                for tr in trajs:
                    req_list.append(goal)
                    trajs_all.append(tr)
                rem -= cur_b
                offset += cur_b

    n_total = len(trajs_all)
    admit = bus.admit_trajectories(trajs_all, reconcile_valence=True)
    mols = admit["mols"]
    req_mat = np.zeros((n_total, len(bus.spec.objectives)), dtype=np.float64)
    req_mat[:, 0] = np.asarray(req_list, dtype=np.float64)
    act_mat = np.zeros((n_total, len(bus.spec.objectives)), dtype=bool)
    act_mat[:, 0] = True

    ev = bus.evaluate_molecules(
        mols, requested_targets=req_mat, active_mask=act_mat, include_audit=True, for_training=False, batch_size=32
    )

    # Strictly isolated blind cross-foundation AuditOracle (`E15 DPA-4`) check
    aud_full, _, aud_ok, _ = audit_slot.evaluate_candidates(mols, batch_size=32)
    aud_1d = aud_full[:, aud_col_idx]

    preds_1d = ev["objective_preds"][:, 0]
    sigma_cal_1d = ev["sigma_cal_estimate"][:, 0]
    exotic_mask = admit["legal_mask"] & (~admit["clean_charge_mask"])

    loom_m = compute_loom_metrics(
        requested_targets_eV=req_list,
        reward_predictions_eV=preds_1d,
        legal_mask=admit["legal_mask"],
        evaluated_mask=ev["evaluated_mask"],
        audit_predictions_eV=aud_1d,
        audit_evaluated_mask=aud_ok,
        f_max_eV_A=ev["f_max_eV_A"],
        unseen_charge_mask=exotic_mask,
        tolerance_eV=0.25,
    )

    # Blind AuditOracle constrained clean hits (`|y_audit - t| <= 0.25` & CleanCharge & `SA_Score <= 3.8`)
    req_arr = np.asarray(req_list, dtype=np.float64)
    blind_aud_clean_hits = (
        admit["legal_mask"]
        & admit["clean_charge_mask"]
        & ev["constraints_met"]
        & aud_ok
        & np.isfinite(aud_1d)
        & (np.abs(aud_1d - req_arr) <= 0.25)
    )
    deep_mask = req_arr <= float(q_points["25"]) + 1e-4
    mid_mask = ~deep_mask

    loom_m.update(
        {
            "constrained_clean_hits": int(ev["constrained_clean_hits"].sum()),
            "constrained_clean_hit_yield": float(ev["constrained_clean_hits"].mean()),
            "constraints_met_rate_among_legal": (
                float(ev["constraints_met"][admit["legal_mask"]].mean()) if admit["legal_mask"].any() else 0.0
            ),
            "blind_audit_constrained_clean_hits": int(blind_aud_clean_hits.sum()),
            "blind_audit_constrained_clean_hit_yield": float(blind_aud_clean_hits.mean()),
            "deep_blindspot_audit_clean_hit_yield": (
                float(blind_aud_clean_hits[deep_mask].mean()) if deep_mask.any() else 0.0
            ),
            "mid_underexplored_audit_clean_hit_yield": (
                float(blind_aud_clean_hits[mid_mask].mean()) if mid_mask.any() else 0.0
            ),
            "deep_blindspot_audit_mae_eV": (
                float(np.mean(np.abs(aud_1d[deep_mask & aud_ok] - req_arr[deep_mask & aud_ok])))
                if (deep_mask & aud_ok).any()
                else None
            ),
            "mean_sigma_cal_proxy_eV": (
                float(np.nanmean(sigma_cal_1d[ev["evaluated_mask"]])) if ev["evaluated_mask"].any() else None
            ),
            "raw_legal_rate": float(admit["raw_legal_mask"].mean()),
            "raw_exotic_charge_rate": (
                float(1.0 - admit["raw_clean_charge_mask"][admit["raw_legal_mask"]].mean())
                if admit["raw_legal_mask"].any()
                else 0.0
            ),
        }
    )

    # Sample an empirical out-of-chemistry ceiling probe (`t_ceil = -8.55 eV`, 64 proposals)
    # so `GPAtlas` directly measures generator pullback at the prior ceiling without altering the 1,792 benchmark!
    ceil_target = -8.55
    ceil_trajs: List[Dict[str, Any]] = []
    for sd_i, sd in enumerate(seeds):
        cur_b = 32
        t_mat = np.zeros((cur_b, gen.num_props), dtype=np.float32)
        t_mat[:, prop_idx] = ceil_target
        t_norm = gen.normalize_targets(t_mat)
        m_act = torch.zeros((cur_b, gen.num_props), dtype=torch.bool, device=device)
        m_act[:, prop_idx] = True
        ceil_trajs.extend(
            gen.sample(
                t_norm,
                m_act,
                cfg_scale=cfg_scale,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.30,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=int(sd) * 10007 + 991 + sd_i,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
            )
        )
    ceil_admit = bus.admit_trajectories(ceil_trajs, reconcile_valence=True)
    ceil_req_mat = np.full((len(ceil_trajs), len(bus.spec.objectives)), ceil_target, dtype=np.float64)
    ceil_act_mat = np.ones((len(ceil_trajs), len(bus.spec.objectives)), dtype=bool)
    ceil_ev = bus.evaluate_molecules(
        ceil_admit["mols"],
        requested_targets=ceil_req_mat,
        active_mask=ceil_act_mat,
        include_audit=True,
        for_training=False,
        batch_size=32,
    )

    atlas = HeteroscedasticGPAtlas1D.fit_from_proposals(
        requested_targets_eV=list(req_list) + [ceil_target] * len(ceil_trajs),
        reward_preds_eV=np.concatenate([preds_1d, ceil_ev["objective_preds"][:, 0]]),
        evaluated_mask=np.concatenate([ev["evaluated_mask"], ceil_ev["evaluated_mask"]]),
        sigma_cal_samples=np.concatenate([sigma_cal_1d, ceil_ev["sigma_cal_estimate"][:, 0]]),
        source_property_values_eV=source_homo_values,
        axis_name="homo",
        unit="eV",
    )
    query_grid = sorted([float(q_points[str(q)]) for q in quantiles] + [ceil_target])
    atlas_q = atlas.query(query_grid, tolerance_eV=0.25)
    diagnosis = diagnoser.diagnose_1d(atlas_q)

    raw_bundle = {
        "mols": mols,
        "req_mat": req_mat,
        "preds_1d": preds_1d,
        "sigma_cal_1d": sigma_cal_1d,
        "constraints_met": ev["constraints_met"],
        "f_max_eV_A": ev["f_max_eV_A"],
        "atlas_query": atlas_q,
    }
    return loom_m, atlas, diagnosis, raw_bundle


def match_external_dft_lab_pool(
    acquisition_mols: Sequence[Chem.Mol],
    external_dft_mols: Sequence[Chem.Mol],
    external_dft_targets: np.ndarray,
    *,
    budget: int = 64,
) -> Tuple[List[Chem.Mol], np.ndarray, Dict[str, Any]]:
    """Query an isolated held-out real `DFT` pool (`quantum_valid_dft.sdf`) for the closest molecules
    to the exported `OuterRSIAcquisition` batch (simulating external high-fidelity DFT execution
    on blindspot chemical space while keeping `quantum_test_dft.sdf` 100% untouched!).
    """
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    acq_fps = [fp_gen.GetFingerprint(Chem.RemoveHs(Chem.Mol(m))) for m in acquisition_mols]
    pool_fps = [fp_gen.GetFingerprint(Chem.RemoveHs(Chem.Mol(m))) for m in external_dft_mols]

    used_pool: set[int] = set()
    matched_mols: List[Chem.Mol] = []
    matched_targets: List[np.ndarray] = []
    sims_selected: List[float] = []

    for q_fp in acq_fps:
        if len(matched_mols) >= budget:
            break
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(q_fp, pool_fps), dtype=np.float64)
        for u_i in used_pool:
            sims[u_i] = -1.0
        best_i = int(np.argmax(sims))
        if sims[best_i] < 0.0:
            continue
        used_pool.add(best_i)
        matched_mols.append(external_dft_mols[best_i])
        matched_targets.append(external_dft_targets[best_i])
        sims_selected.append(float(sims[best_i]))

    return (
        matched_mols,
        np.stack(matched_targets, axis=0),
        {
            "mode": "pool_nearest_neighbor",
            "matched_dft_molecules": len(matched_mols),
            "mean_tanimoto_to_acquisition_batch": float(np.mean(sims_selected)) if sims_selected else 0.0,
        },
    )


SUPPORTED_ALL_ELECTRON_TZVP_Z: frozenset[int] = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35})

_LIVE_DFT_WORKER_CODE = """
import json, sys, time
from pathlib import Path
import numpy as np
from rdkit import Chem
from gpu4pyscf.dft.rks import RKS
from pyscf import gto, lib

HARTREE_TO_EV = 27.211386245988
lib.num_threads(4)
in_sdf = Path(sys.argv[1])
work_dir = Path(sys.argv[2])
out_json = Path(sys.argv[3])
work_dir.mkdir(parents=True, exist_ok=True)
ptable = Chem.GetPeriodicTable()

def _arr(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)

def _build_mf(mol):
    mf = RKS(mol).density_fit(auxbasis="def2-universal-jkfit")
    mf.xc = "pbe0"
    mf.grids.level = 3
    mf.conv_tol = 1e-8
    mf.max_cycle = 100
    return mf

mols = list(Chem.SDMolSupplier(str(in_sdf), removeHs=False))
rows = []
for idx, m in enumerate(mols):
    if m is None:
        rows.append({"idx": idx, "status": "invalid_mol"})
        continue
    sp_dir = work_dir / f"mol_{idx:04d}"
    res_p = sp_dir / "result.json"
    if res_p.exists():
        prev = json.loads(res_p.read_text())
        if prev.get("status") == "completed":
            rows.append(prev)
            continue
    sp_dir.mkdir(parents=True, exist_ok=True)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    coords = m.GetConformer().GetPositions()
    t0 = time.perf_counter()
    try:
        ecp = {s: "def2-tzvp" for s in set(syms) if ptable.GetAtomicNumber(s) > 36}
        gmol = gto.M(
            atom=list(zip(syms, coords.tolist())),
            basis="def2-tzvp",
            ecp=ecp,
            unit="Angstrom",
            charge=0,
            spin=0,
            verbose=4,
            output=str(sp_dir / "singlepoint.log"),
        )
        mf = _build_mf(gmol)
        mf.kernel()
        used_newton = False
        if not mf.converged:
            used_newton = True
            dm0 = _arr(mf.make_rdm1())
            mf = _build_mf(gmol).newton()
            mf.max_cycle = 80
            mf.conv_tol = 1e-8
            mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError("SCF did not converge")
        e = _arr(mf.mo_energy)
        occ = _arr(mf.mo_occ)
        occ_idx = np.where(occ > 0.5)[0]
        homo_ev = float(e[int(occ_idx[-1])] * HARTREE_TO_EV)
        lumo_ev = float(e[int(occ_idx[-1]) + 1] * HARTREE_TO_EV) if (int(occ_idx[-1]) + 1) < len(e) else None
        dt = round(time.perf_counter() - t0, 3)
        rec = {
            "idx": idx,
            "status": "completed",
            "n_atoms": len(syms),
            "homo_eV": homo_ev,
            "lumo_eV": lumo_ev,
            "used_newton": used_newton,
            "wall_sec": dt,
        }
        res_p.write_text(json.dumps(rec, indent=2) + "\\n")
        rows.append(rec)
        if (idx + 1) % 8 == 0 or (idx + 1) == len(mols):
            print(f"[Live-RSI-DFT {idx+1}/{len(mols)}] homo={homo_ev:.4f} eV atoms={len(syms)} ({dt:.2f}s)", flush=True)
    except Exception as exc:
        rec = {"idx": idx, "status": "failed", "error": str(exc), "wall_sec": round(time.perf_counter() - t0, 3)}
        res_p.write_text(json.dumps(rec, indent=2) + "\\n")
        rows.append(rec)
out_json.write_text(json.dumps(rows, indent=2) + "\\n")
"""


def execute_live_gpu4pyscf_dft_batch(
    acquisition_mols: Sequence[Chem.Mol],
    pot_slot: PotentialSlot,
    dft_python: Path,
    work_dir: Path,
) -> Tuple[List[Chem.Mol], np.ndarray, Dict[str, Any]]:
    """Relax exported `OuterRSIAcquisition` molecules on `DPA-4` and run live `GPU4PySCF PBE0/def2-TZVP/DF`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    relaxed_mols: List[Chem.Mol] = []
    relax_res = pot_slot.relax_molecules(list(acquisition_mols), n_steps=25)
    for orig_m, rm, gp in zip(
        acquisition_mols, relax_res["relaxed_mols"], relax_res["graph_preserved"]
    ):
        relaxed_mols.append(Chem.Mol(rm) if (bool(gp) and rm is not None) else Chem.Mol(orig_m))

    rel_sdf = work_dir / "acquisition_batch_r1_relaxed.sdf"
    w = Chem.SDWriter(str(rel_sdf))
    for m in relaxed_mols:
        w.write(m)
    w.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    res_json = work_dir / "live_pbe0_results.json"
    subprocess.run(
        [str(dft_python), "-u", "-c", _LIVE_DFT_WORKER_CODE, str(rel_sdf), str(work_dir / "sp"), str(res_json)],
        check=True,
    )
    rows = json.loads(res_json.read_text())
    conv_mols: List[Chem.Mol] = []
    conv_targets: List[List[float]] = []
    scf_walls: List[float] = []
    for idx, rec in enumerate(rows):
        if rec.get("status") == "completed" and rec.get("homo_eV") is not None:
            conv_mols.append(relaxed_mols[idx])
            conv_targets.append([float(rec["homo_eV"])])
            scf_walls.append(float(rec.get("wall_sec", 0.0)))

    total_wall = round(time.perf_counter() - t0, 2)
    return (
        conv_mols,
        np.asarray(conv_targets, dtype=np.float64),
        {
            "mode": "live_gpu4pyscf_pbe0_def2_tzvp_df",
            "attempted_dft_molecules": len(relaxed_mols),
            "scf_converged_molecules": len(conv_mols),
            "scf_convergence_rate": round(len(conv_mols) / max(len(relaxed_mols), 1), 4),
            "mean_scf_wall_sec": round(float(np.mean(scf_walls)), 3) if scf_walls else 0.0,
            "total_dft_stage_wall_sec": total_wall,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True, help="e.g. stage1_homo_r2_sft checkpoint")
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--quantum-train-dft-sdf", type=Path, required=True)
    parser.add_argument("--quantum-valid-dft-sdf", type=Path, required=True, help="Isolated External DFT Lab Pool")
    parser.add_argument("--quantum-test-dft-sdf", type=Path, required=True, help="Strictly untouched blind DFT test set")
    parser.add_argument("--audit-oracle-dir", type=Path, required=True)
    parser.add_argument("--potential-weights", type=Path, required=True)
    parser.add_argument("--inner-steps", type=int, default=48, help="48 steps * 4 prompts = 192 groups per RSI round")
    parser.add_argument("--acquisition-budget", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--live-gpu4pyscf", action="store_true", help="Run live GPU4PySCF PBE0/def2-TZVP/DF on exported acquisition molecules")
    parser.add_argument(
        "--gpu4pyscf-python",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit3d_quantum_20261111/gpu_venv/bin/python"),
    )
    parser.add_argument("--reuse-frozen-qsar-dir", type=Path, default=None)
    parser.add_argument("--uniform-trusted-anchor", action="store_true", help="Stratify 22% anchor budget uniformly across all trusted quantiles")
    parser.add_argument("--allow-tf32", action="store_true")
    args = parser.parse_args()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3030)
    np.random.seed(3030)
    torch.set_num_threads(4)

    # 1. Load DFT datasets with strict split isolation:
    #    - Historical Enterprise Predictor sees ONLY `quantum_train_dft` with `HOMO > -6.92 eV`
    #      (placing `P35..P90` in-domain and isolating `P10=-7.45 eV, P25=-7.05 eV` as an authentic
    #       industrial deep-acceptor OOD blindspot!)
    #    - `quantum_valid_dft` (`HOMO <= -6.92 eV`) serves as the External DFT Lab Pool for Outer-RSI acquisition
    #    - `quantum_test_dft` + `E15 DPA-4 AuditOracle` remain 100% blind throughout!
    train_mols, train_dft = load_sdf_with_targets(args.quantum_train_dft_sdf, ["homo", "lumo"])
    valid_mols, valid_dft = load_sdf_with_targets(args.quantum_valid_dft_sdf, ["homo", "lumo"])
    test_mols, test_dft = load_sdf_with_targets(args.quantum_test_dft_sdf, ["homo", "lumo"])

    blindspot_cutoff_eV = -6.92
    generator_prior_homo = train_dft[:, 0].copy()
    hist_mask = train_dft[:, 0] > blindspot_cutoff_eV
    hist_mols = [train_mols[i] for i in range(len(train_mols)) if hist_mask[i]]
    hist_targets = train_dft[hist_mask, 0:1]

    ext_lab_mask = valid_dft[:, 0] <= blindspot_cutoff_eV
    ext_lab_mols = [valid_mols[i] for i in range(len(valid_mols)) if ext_lab_mask[i]]
    ext_lab_targets = valid_dft[ext_lab_mask, 0:1]

    blind_test_deep_mask = test_dft[:, 0] <= blindspot_cutoff_eV
    blind_test_deep_mols = [test_mols[i] for i in range(len(test_mols)) if blind_test_deep_mask[i]]
    blind_test_deep_targets = test_dft[blind_test_deep_mask, 0:1]

    print(
        f"[RSI-Setup] Enterprise historical predictor train={len(hist_mols)} (HOMO > {blindspot_cutoff_eV} eV); "
        f"External Lab DFT pool={len(ext_lab_mols)}; Blind DFT deep test={len(blind_test_deep_mols)}",
        flush=True,
    )

    # Train frozen historical enterprise QSAR asset once (base weights NEVER modified afterward!)
    base_qsar_dir = args.output_dir / "frozen_enterprise_qsar"
    if args.reuse_frozen_qsar_dir is not None and (args.reuse_frozen_qsar_dir / "metadata.json").exists() and not (base_qsar_dir / "metadata.json").exists():
        shutil.copytree(args.reuse_frozen_qsar_dir, base_qsar_dir)
    if (base_qsar_dir / "metadata.json").exists():
        frozen_base_asset = FingerprintQSARAsset.from_checkpoint_dir(
            base_qsar_dir, asset_id="enterprise_qsar_base"
        )
    else:
        frozen_base_asset = FingerprintQSARAsset.train_internal(
            hist_mols,
            hist_targets,
            base_qsar_dir,
            target_names=["homo"],
            units=["eV"],
            n_models=5,
            n_trees=128,
            seed=42,
            asset_id="enterprise_qsar_base",
        )
    calibrated_asset = ResidualCalibratedAsset(
        frozen_base_asset, asset_id="enterprise_homo_cal", prior_blindspot_std=0.40
    )
    anchor_info = calibrated_asset.seed_in_domain_anchors(hist_mols, hist_targets, max_anchors=256, seed=42)
    print(f"[RSI-Setup] Seeded in-domain residual GP anchors: {anchor_info}", flush=True)
    hist_valid_smiles = [
        s for s in (mol_to_clean_smiles(m) for m in hist_mols) if Chem.MolFromSmiles(s) is not None
    ]
    morgan_ad = MorganDomain(hist_valid_smiles, cutoff=0.28, graph_normalization=RAW_GRAPH)

    pot_slot = PotentialSlot(backend="dpa4", weights=str(args.potential_weights), device=args.device)
    audit_slot = AuditOracleSlot(
        backend="dpa4",
        checkpoint_dir=args.audit_oracle_dir,
        device=args.device,
        forcefield_path=str(args.potential_weights),
    )

    registry = AssetRegistry(register_builtin_rules=True)
    registry.register_predictor(calibrated_asset)
    registry.register_predictor(BuiltInRuleAsset(asset_id="builtin_rules"))
    registry.register_potential("mlip_dpa4", pot_slot)

    spec = TaskWeavingSpec(
        task_name="rsi_enterprise_homo_synthesizable",
        description="2-Round Grounded RSI with frozen enterprise predictor + ResidualCalibratedAsset + SA_Score<=3.8",
        potential_asset_id="mlip_dpa4",
        reconcile_valence=True,
        objectives=(
            ObjectiveChannelSpec(
                name="homo", asset_id="enterprise_homo_cal", asset_channel="homo", mode="point", tolerance=0.25, unit="eV"
            ),
        ),
        constraints=(
            ConstraintChannelSpec(
                name="synthesizable_sa", asset_id="builtin_rules", asset_channel="sa_score", op="leq", threshold=3.8, hard_gate=True
            ),
            ConstraintChannelSpec(
                name="reactive_smarts_clean", asset_id="builtin_rules", asset_channel="smarts_alert", op="leq", threshold=0.0, hard_gate=True
            ),
        ),
        auditors=(
            AuditorChannelSpec(
                name="single_oracle_ad_guard", objective_name="homo", mode="single_oracle_ad", tanimoto_cutoff=0.28
            ),
        ),
    )

    seen_charges = frozenset(formal_charge_signature(m) for m in hist_mols)
    bus = CompositeCriticBus(
        spec=spec, registry=registry, applicability_domain=morgan_ad, seen_charge_signatures=seen_charges
    )

    targets_cfg = json.loads(args.targets_json.read_text())
    q_points = targets_cfg["single_target_points"]["homo"]
    diagnoser = RSIDiagnoser(
        RSIDiagnosticThresholds(
            hit_floor=0.24,
            bias_cap=0.30,
            cal_sigma_cap=0.35,
            spread_cap=0.46,
            eval_floor=0.35,
            ceiling_bias_cap=0.52,
        ),
        uniform_trusted_anchor=bool(args.uniform_trusted_anchor),
    )

    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.init_checkpoint, device=args.device, batch_size=128)
    gen.enable_v20_valence_head()

    # ==================================================================
    # Round 0: Initial Weak-Zone Diagnostic & Pre-Generation Gatekeeper
    # ==================================================================
    print("[RSI Round 0] Running initial 1,792-proposal GPAtlas diagnostic...", flush=True)
    r0_metrics, r0_atlas, r0_diag, r0_bundle = evaluate_rsi_stage(
        gen, bus, audit_slot, q_points, generator_prior_homo, diagnoser
    )
    gate_extreme = r0_atlas.recommend_target(-8.55, tolerance=0.25, cal_sigma_cap=0.35)
    gate_blindspot = r0_atlas.recommend_target(float(q_points["10"]), tolerance=0.25, cal_sigma_cap=0.35)
    gate_trusted = r0_atlas.recommend_target(float(q_points["65"]), tolerance=0.25, cal_sigma_cap=0.35)
    print(f"[RSI Round 0] Zones: {r0_diag['zone_counts']} | BlindAuditCleanHits={r0_metrics['blind_audit_constrained_clean_hits']}/1792", flush=True)

    # ==================================================================
    # Round 1A: Inner-Loop RSI (Curriculum on INNER_UNDEREXPLORED, 0% on OUTER_BLINDSPOT)
    # ==================================================================
    print("[RSI Round 1A] Running Inner-Loop RSI on trusted INNER_UNDEREXPLORED intervals...", flush=True)
    r1_replay_mols, r1_replay_1d, _ = bus.relabel_conformer_pool(hist_mols, max_fmax_eV_A=4.5)
    r1_replay_2d = np.zeros((len(r1_replay_mols), gen.num_props), dtype=np.float32)
    r1_replay_2d[:, 0] = r1_replay_1d[:, 0]

    inner_runner = InnerRSIRunner(
        generator=gen,
        critic_bus=bus,
        prop_channel_idx=0,
        lr=2.5e-5,
        kl_beta=0.03,
        replay_weight=0.20,
        cfg_scale=1.65,
        seed=3031,
    )
    r1_inner_out = inner_runner.run_inner_cycle(
        r0_diag, replay_mols=r1_replay_mols, replay_targets=r1_replay_2d, steps=args.inner_steps
    )
    r1_metrics, r1_atlas, r1_diag, r1_bundle = evaluate_rsi_stage(
        gen, bus, audit_slot, q_points, generator_prior_homo, diagnoser
    )
    r1_gate = inner_runner.acceptance_gate(r0_metrics, r1_metrics, r1_inner_out["backup_state_dict"])
    print(
        f"[RSI Round 1A] GateAccepted={r1_gate['accepted']} | "
        f"MidZoneAuditYield: {r0_metrics['mid_underexplored_audit_clean_hit_yield']*100:.2f}% -> "
        f"{r1_metrics['mid_underexplored_audit_clean_hit_yield']*100:.2f}% | "
        f"BlindAuditCleanHits={r1_metrics['blind_audit_constrained_clean_hits']}/1792",
        flush=True,
    )
    gen.save_checkpoint(args.output_dir / "rsi_round1_inner.pt", extra_metadata={"rsi_round": 1})

    # ==================================================================
    # Round 1B: Outer-Loop Active Acquisition + Isolated External DFT Lab Feedback
    # ==================================================================
    print("[RSI Round 1B] Exporting active acquisition batch from OUTER_BLINDSPOT...", flush=True)
    acq_engine = OuterRSIAcquisition(max_fmax_eV_A=3.5)
    cand_mols_r1: List[Optional[Chem.Mol]] = list(r1_bundle["mols"])
    if args.live_gpu4pyscf:
        ptable = Chem.GetPeriodicTable()
        cand_mols_r1 = [
            m
            if (
                m is not None
                and Chem.GetFormalCharge(m) == 0
                and not any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms())
                and all(ptable.GetAtomicNumber(a.GetSymbol()) in SUPPORTED_ALL_ELECTRON_TZVP_Z for a in m.GetAtoms())
            )
            else None
            for m in cand_mols_r1
        ]
    acq_mols, acq_records = acq_engine.select_acquisition_batch(
        cand_mols_r1,
        r1_bundle["req_mat"],
        r1_bundle["preds_1d"],
        r1_bundle["sigma_cal_1d"],
        constraints_met=r1_bundle["constraints_met"],
        f_max_eV_A=r1_bundle["f_max_eV_A"],
        blindspot_target_values=r1_diag["outer_blindspot_targets"],
        budget_k=args.acquisition_budget,
        min_sigma_cal=0.24,
    )
    acq_sdf_path = args.output_dir / "acquisition_batch_r1.sdf"
    acq_manifest = acq_engine.export_sdf(acq_mols, acq_records, acq_sdf_path)

    if args.live_gpu4pyscf:
        dft_fb_mols, dft_fb_targets, match_info = execute_live_gpu4pyscf_dft_batch(
            acq_mols, pot_slot, args.gpu4pyscf_python, args.output_dir / "live_dft_r1"
        )
    else:
        dft_fb_mols, dft_fb_targets, match_info = match_external_dft_lab_pool(
            acq_mols, ext_lab_mols, ext_lab_targets, budget=min(args.acquisition_budget, len(ext_lab_mols))
        )
    cal_stats = IncrementalCalibrator.ingest_labeled_feedback(
        calibrated_asset,
        dft_fb_mols,
        dft_fb_targets,
        validation_mols=blind_test_deep_mols,
        validation_true_targets=blind_test_deep_targets,
        save_path=args.output_dir / "residual_calibrator_r1.joblib",
    )
    # Expand MorganDomain applicability coverage with newly calibrated DFT structures!
    fb_valid_smiles = [
        s for s in (mol_to_clean_smiles(m) for m in dft_fb_mols) if Chem.MolFromSmiles(s) is not None
    ]
    expanded_ref_smiles = list(morgan_ad.reference_smiles) + fb_valid_smiles
    bus.applicability_domain = MorganDomain(expanded_ref_smiles, cutoff=0.28, graph_normalization=RAW_GRAPH)
    print(f"[RSI Round 1B] Calibration update on blind DFT deep test: {cal_stats}", flush=True)

    # Re-diagnose immediately after ResidualCalibratedAsset update to verify blindspot unlocking!
    post_cal_metrics, post_cal_atlas, post_cal_diag, post_cal_bundle = evaluate_rsi_stage(
        gen, bus, audit_slot, q_points, generator_prior_homo, diagnoser
    )
    print(f"[RSI Post-Calibration] Unlocked Zones: {post_cal_diag['zone_counts']}", flush=True)

    # ==================================================================
    # Round 2: Post-Calibration Inner-Loop RSI across Newly Unlocked Deep-HOMO Zone
    # ==================================================================
    print("[RSI Round 2] Running Round 2 Inner-RSI across newly unlocked deep-HOMO intervals...", flush=True)
    combined_pool_mols = list(hist_mols) + list(dft_fb_mols)
    r2_replay_mols, r2_replay_1d, _ = bus.relabel_conformer_pool(combined_pool_mols, max_fmax_eV_A=4.5)
    r2_replay_2d = np.zeros((len(r2_replay_mols), gen.num_props), dtype=np.float32)
    r2_replay_2d[:, 0] = r2_replay_1d[:, 0]

    inner_runner_r2 = InnerRSIRunner(
        generator=gen,
        critic_bus=bus,
        prop_channel_idx=0,
        lr=2.5e-5,
        kl_beta=0.03,
        replay_weight=0.22,
        cfg_scale=1.65,
        seed=3032,
    )
    r2_inner_out = inner_runner_r2.run_inner_cycle(
        post_cal_diag, replay_mols=r2_replay_mols, replay_targets=r2_replay_2d, steps=args.inner_steps
    )
    r2_metrics, r2_atlas, r2_diag, r2_bundle = evaluate_rsi_stage(
        gen, bus, audit_slot, q_points, generator_prior_homo, diagnoser
    )
    r2_gate = inner_runner_r2.acceptance_gate(post_cal_metrics, r2_metrics, r2_inner_out["backup_state_dict"])
    gen.save_checkpoint(args.output_dir / "rsi_round2_final.pt", extra_metadata={"rsi_round": 2})

    print(
        f"[RSI Round 2 Final] GateAccepted={r2_gate['accepted']} | "
        f"DeepBlindspotAuditYield: {r0_metrics['deep_blindspot_audit_clean_hit_yield']*100:.2f}% (R0) -> "
        f"{r1_metrics['deep_blindspot_audit_clean_hit_yield']*100:.2f}% (R1) -> "
        f"{r2_metrics['deep_blindspot_audit_clean_hit_yield']*100:.2f}% (R2) | "
        f"TotalBlindAuditCleanHits: {r0_metrics['blind_audit_constrained_clean_hits']} -> "
        f"{r1_metrics['blind_audit_constrained_clean_hits']} -> "
        f"{r2_metrics['blind_audit_constrained_clean_hits']}/1792",
        flush=True,
    )

    report = {
        "wall_seconds": time.perf_counter() - t_start,
        "blindspot_cutoff_eV": blindspot_cutoff_eV,
        "historical_train_count": len(hist_mols),
        "external_dft_lab_pool_count": len(ext_lab_mols),
        "blind_test_deep_count": len(blind_test_deep_mols),
        "pre_generation_gatekeeper_checks": {
            "extreme_ceiling_query": gate_extreme,
            "blindspot_query_pre_rsi": gate_blindspot,
            "trusted_query": gate_trusted,
            "blindspot_query_post_rsi": r2_atlas.recommend_target(float(q_points["10"]), tolerance=0.25, cal_sigma_cap=0.35),
        },
        "round_0_initial": {
            "metrics": r0_metrics,
            "diagnosis": r0_diag,
            "atlas_curve": r0_bundle["atlas_query"],
        },
        "round_1_inner_rsi": {
            "acceptance_gate": r1_gate,
            "metrics": r1_metrics,
            "diagnosis": r1_diag,
            "atlas_curve": r1_bundle["atlas_query"],
        },
        "round_1b_outer_acquisition_and_calibration": {
            "acquisition_manifest_summary": {
                "exported_count": acq_manifest["exported_count"],
                "unique_murcko_scaffolds": acq_manifest["unique_murcko_scaffolds"],
                "mean_sigma_cal": acq_manifest["mean_sigma_cal"],
                "sdf_path": acq_manifest["sdf_path"],
            },
            "external_dft_lab_match": match_info,
            "residual_calibration_blind_test": cal_stats,
            "post_calibration_diagnosis": post_cal_diag,
            "post_calibration_atlas_curve": post_cal_bundle["atlas_query"],
        },
        "round_2_post_calibration_inner_rsi": {
            "acceptance_gate": r2_gate,
            "metrics": r2_metrics,
            "diagnosis": r2_diag,
            "atlas_curve": r2_bundle["atlas_query"],
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[RSI Complete] Saved 2-round Grounded RSI report to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
