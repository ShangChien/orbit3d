"""Matched `PBE0/def2-TZVP` Ab Initio DFT Single-Point Anchor Verification (`Uni-Loom` vs `PropMolFlow-OE62`).

Aligns strictly with the `atlas2d_dft_anchor_eval.json` protocol (`PBE0/def2-TZVP/DF(def2-universal-jkfit)`,
grid level 3, SCF `1e-8` with SOSCF Newton fallback in `GPU4PySCF`):
  1. `--phase export_uniloom_cohort`: Exports the 1,792-proposal `Uni-Loom v2.0 (Process_16_to_4)` cohort SDF
     with `Uni-Mol2`, `DPA-4`, `SA_Score`, and `F_max` annotations.
  2. `--phase select_cohorts`: Stratified selection across all 7 held-out OE62 `HOMO` quantiles (`6` unique
     closed-shell neutral `def2-TZVP`-supported molecules per quantile = `42` molecules per arm) using strictly
     blind selection (`Uni-Mol2` + `SA <= 3.5` only, zero `DPA-4` or `DFT` peeking) for:
       - `UniLoom_v20_Process_16_to_4`
       - `PropMolFlow_native_M1`
       - `PropMolFlow_native_M8`
       - `PropMolFlow_coord_reconciled_M8`
  3. `--phase run_dft`: Runs persistent single-process `GPU4PySCF` `PBE0/def2-TZVP/DF` Kohn-Sham calculations
     with per-molecule disk caching (`sp/<arm_name>/mol_XXXX/result.json`).
  4. `--phase summarize`: Computes raw and basis-shift-centered `|HOMO_PBE0 - Target|` MAE/RMSE, Pearson `r`,
     Spearman `rho`, plus `|Uni-Mol2 - PBE0|` surrogate-exploitation shift and `|DPA-4 - PBE0|` audit fidelity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HA_TO_EV = 27.211386245988
SUPPORTED_TZVP_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    n = len(x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def _pair_metrics(pred: np.ndarray, ref: np.ndarray, *, ref_shift: Optional[float] = None) -> Dict[str, Any]:
    mask = np.isfinite(pred) & np.isfinite(ref)
    p = pred[mask]
    r = ref[mask]
    if len(p) == 0:
        return {"n": 0, "mae_eV": None, "rmse_eV": None, "centered_mae_eV": None, "pearson_r": None, "spearman_rho": None}
    diff = p - r
    bias = float(np.mean(diff))
    shift_used = bias if ref_shift is None else float(ref_shift)
    cdiff = diff - shift_used
    pr = float(np.corrcoef(p, r)[0, 1]) if (len(p) >= 4 and np.std(p) > 1e-9 and np.std(r) > 1e-9) else float("nan")
    sr = _spearman_r(p, r)
    return {
        "n": int(len(p)),
        "mae_eV": round(float(np.mean(np.abs(diff))), 4),
        "rmse_eV": round(float(np.sqrt(np.mean(diff ** 2))), 4),
        "mean_bias_eV": round(bias, 4),
        "centered_mae_eV": round(float(np.mean(np.abs(cdiff))), 4),
        "centered_rmse_eV": round(float(np.sqrt(np.mean(cdiff ** 2))), 4),
        "hit_rate_025eV_raw": round(float(np.mean(np.abs(diff) <= 0.25)), 4),
        "hit_rate_025eV_centered": round(float(np.mean(np.abs(cdiff) <= 0.25)), 4),
        "pearson_r": round(pr, 4) if np.isfinite(pr) else None,
        "spearman_rho": round(sr, 4) if np.isfinite(sr) else None,
    }


def run_export_uniloom_cohort(args: argparse.Namespace) -> None:
    import torch
    from loom.chemistry import ValenceReconciler
    from loom.weaving import LoomUni3DARGenerator, OctreeProcessCritic
    from scripts.loom_octree_process_critic_exp import (
        _build_oe62_critic_bus,
        _score_oe62_selected_cohort,
        _select_oe62_from_pool,
    )

    device = torch.device(args.device)
    critic, _ = OctreeProcessCritic.load_checkpoint(args.oe62_critic_ckpt, device=args.device)
    critic.eval()
    bus, source_graphs = _build_oe62_critic_bus(args, include_blind_auditor=True)
    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.oe62_v20_ckpt, device=args.device, batch_size=128)
    gen.eval()

    targets_cfg = json.loads(args.oe62_targets_json.read_text())
    q_points = targets_cfg["single_target_points"]["homo"]
    quantiles = (10, 25, 35, 50, 65, 75, 90)
    seeds = (24411, 24412)
    req_list: List[float] = []
    for q in quantiles:
        for _sd in seeds:
            req_list.extend([float(q_points[str(q)])] * 128)
    n_prompts = len(req_list)

    prop_idx = gen.target_cols.index("homo")
    allowed_elems = tuple(gen.args.allow_atoms.split(","))
    reconciler = ValenceReconciler(allowed_elements=allowed_elems)
    obj_col_mask = torch.zeros((gen.num_props,), dtype=torch.bool, device=device)
    obj_col_mask[prop_idx] = True

    m_init, m_keep = 16, 4
    groups_all: List[List[Dict[str, Any]]] = []
    for s in range(0, n_prompts, 128):
        e = min(s + 128, n_prompts)
        mat = np.zeros((e - s, gen.num_props), dtype=np.float32)
        mat[:, prop_idx] = np.asarray(req_list[s:e], dtype=np.float32)
        bt = gen.normalize_targets(mat)
        bm = torch.zeros((e - s, gen.num_props), dtype=torch.bool, device=device)
        bm[:, prop_idx] = True
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

    sel_m, sel_l, sel_x, surr_w = _select_oe62_from_pool(
        p_grid, p_leg, p_exo, req_list, bus, m_keep, use_surrogate=True, device=device
    )
    if bus.potential_slot is not None:
        reconciler.batch_polish_hydrogens_mlip([m for m in sel_m if m is not None], bus.potential_slot, n_steps=5, chunk_size=32)

    _ = _score_oe62_selected_cohort(
        sel_m, sel_l, sel_x, req_list, bus, source_graphs, 0.0, 0.0, surr_w, export_sdf_path=args.uniloom_sdf
    )
    print(f"[DFT-Anchor] Exported Uni-Loom Process_16_to_4 cohort SDF to {args.uniloom_sdf}", flush=True)


def _select_stratified_from_sdf(sdf_path: Path, per_quantile: int = 6, max_atoms: int = 80) -> List[Dict[str, Any]]:
    from loom.generator.molecular_state import mol_to_clean_smiles

    ptable = Chem.GetPeriodicTable()
    sup = list(Chem.SDMolSupplier(str(sdf_path), removeHs=False))
    by_target: Dict[str, List[Tuple[Tuple[int, float, int], Chem.Mol, Dict[str, Any]]]] = {}

    for m in sup:
        if m is None or m.GetNumAtoms() > max_atoms or m.GetNumAtoms() < 2:
            continue
        if Chem.GetFormalCharge(m) != 0 or any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()):
            continue
        if any(ptable.GetAtomicNumber(a.GetSymbol()) not in SUPPORTED_TZVP_Z for a in m.GetAtoms()):
            continue
        t_str = m.GetProp("homo_target_eV")
        t_val = float(t_str)
        uh_str = m.GetProp("unimol2_homo_eV")
        uh_val = float(uh_str) if uh_str != "nan" else float("nan")
        if not math.isfinite(uh_val):
            continue
        dh_str = m.GetProp("dpa4_homo_eV")
        dh_val = float(dh_str) if dh_str != "nan" else float("nan")
        sa_str = m.GetProp("sa_score")
        sa_val = float(sa_str) if sa_str != "nan" else 9.9
        fm_str = m.GetProp("f_max_eV_A")
        fm_val = float(fm_str) if fm_str != "nan" else float("nan")
        is_synth_hit = int(m.GetProp("is_synthesizable_clean_hit")) == 1
        err_u = abs(uh_val - t_val)
        tier = 0 if is_synth_hit else (1 if sa_val <= 3.5 else 2)
        p_idx = int(m.GetProp("prompt_index"))
        key_q = f"{t_val:.4f}"
        by_target.setdefault(key_q, []).append(
            (
                (tier, err_u, p_idx),
                m,
                {
                    "prompt_index": p_idx,
                    "smiles": mol_to_clean_smiles(m),
                    "n_atoms": int(m.GetNumAtoms()),
                    "homo_target_eV": t_val,
                    "unimol2_homo_eV": uh_val,
                    "dpa4_homo_eV": dh_val,
                    "sa_score": sa_val,
                    "f_max_eV_A": fm_val,
                    "is_tier0_synth_hit": is_synth_hit,
                },
            )
        )

    selected: List[Dict[str, Any]] = []
    seen_smi: set[str] = set()
    for key_q in sorted(by_target.keys(), key=lambda x: float(x)):
        items = sorted(by_target[key_q], key=lambda tup: tup[0])
        picked_q = 0
        # First pass: unique SMILES
        for _, m, info in items:
            if info["smiles"] and info["smiles"] not in seen_smi:
                seen_smi.add(info["smiles"])
                selected.append({"mol": m, **info})
                picked_q += 1
                if picked_q >= per_quantile:
                    break
        # Fallback if fewer than `per_quantile` unique graphs exist
        if picked_q < per_quantile:
            used_p = {r["prompt_index"] for r in selected}
            for _, m, info in items:
                if info["prompt_index"] not in used_p:
                    selected.append({"mol": m, **info})
                    picked_q += 1
                    if picked_q >= per_quantile:
                        break
    return selected


def _select_unbiased_fixed_prompts_from_sdf(
    sdf_path: Path, per_quantile: int = 6, max_atoms: int = 80
) -> Tuple[List[Dict[str, Any]], int]:
    from loom.generator.molecular_state import mol_to_clean_smiles

    ptable = Chem.GetPeriodicTable()
    sup = list(Chem.SDMolSupplier(str(sdf_path), removeHs=False))
    by_pidx: Dict[int, Chem.Mol] = {}
    for m in sup:
        if m is not None and m.HasProp("prompt_index"):
            by_pidx[int(m.GetProp("prompt_index"))] = m

    fixed_pindices = [q * 256 + k * 42 for q in range(7) for k in range(per_quantile)]
    selected: List[Dict[str, Any]] = []
    for p_idx in fixed_pindices:
        m = by_pidx.get(p_idx)
        if m is None or m.GetNumAtoms() > max_atoms or m.GetNumAtoms() < 2:
            continue
        if Chem.GetFormalCharge(m) != 0 or any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()):
            continue
        if any(ptable.GetAtomicNumber(a.GetSymbol()) not in SUPPORTED_TZVP_Z for a in m.GetAtoms()):
            continue
        t_val = float(m.GetProp("homo_target_eV"))
        uh_str = m.GetProp("unimol2_homo_eV")
        uh_val = float(uh_str) if uh_str != "nan" else float("nan")
        dh_str = m.GetProp("dpa4_homo_eV")
        dh_val = float(dh_str) if dh_str != "nan" else float("nan")
        sa_str = m.GetProp("sa_score")
        sa_val = float(sa_str) if sa_str != "nan" else 9.9
        fm_str = m.GetProp("f_max_eV_A")
        fm_val = float(fm_str) if fm_str != "nan" else float("nan")
        is_synth_hit = int(m.GetProp("is_synthesizable_clean_hit")) == 1
        selected.append(
            {
                "mol": m,
                "prompt_index": p_idx,
                "smiles": mol_to_clean_smiles(m),
                "n_atoms": int(m.GetNumAtoms()),
                "homo_target_eV": t_val,
                "unimol2_homo_eV": uh_val,
                "dpa4_homo_eV": dh_val,
                "sa_score": sa_val,
                "f_max_eV_A": fm_val,
                "is_tier0_synth_hit": is_synth_hit,
            }
        )
    return selected, len(fixed_pindices)


def run_select_cohorts(args: argparse.Namespace) -> None:
    args.dft_work_dir.mkdir(parents=True, exist_ok=True)
    arm_sources: List[Tuple[str, Path]] = [
        ("UniLoom_v20_Process_16_to_4", args.uniloom_sdf),
    ]
    if args.pmf_cohort_dir is not None and args.pmf_cohort_dir.exists():
        for tag, fname in (
            ("PropMolFlow_native_M1", "native_graph_M1.sdf"),
            ("PropMolFlow_native_M8", "native_graph_M8.sdf"),
            ("PropMolFlow_coord_reconciled_M8", "coord_reconciled_M8.sdf"),
        ):
            fp = args.pmf_cohort_dir / fname
            if fp.exists():
                arm_sources.append((tag, fp))

    manifest: Dict[str, Any] = {
        "selection_mode": args.selection_mode,
        "per_quantile": int(args.per_quantile),
        "arms": {},
    }
    for arm_name, sdf_in in arm_sources:
        if args.selection_mode == "unbiased_fixed_prompts":
            sel, n_prompts_total = _select_unbiased_fixed_prompts_from_sdf(sdf_in, per_quantile=int(args.per_quantile))
        else:
            sel = _select_stratified_from_sdf(sdf_in, per_quantile=int(args.per_quantile))
            n_prompts_total = len(sel)
        arm_dir = args.dft_work_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        out_sdf = arm_dir / "selected_cohort.sdf"
        w = Chem.SDWriter(str(out_sdf))
        meta_rows = []
        for idx, row in enumerate(sel):
            cm = Chem.Mol(row["mol"])
            cm.SetProp("dft_index", str(idx))
            cm.SetProp("arm_name", arm_name)
            w.write(cm)
            meta_rows.append({"dft_index": idx, **{k: v for k, v in row.items() if k != "mol"}})
        w.close()
        manifest["arms"][arm_name] = {
            "source_sdf": str(sdf_in),
            "selected_sdf": str(out_sdf),
            "n_prompts_attempted": n_prompts_total,
            "n_selected": len(meta_rows),
            "n_tier0_synth_hits": sum(1 for r in meta_rows if r["is_tier0_synth_hit"]),
            "rows": meta_rows,
        }
        print(
            f"[DFT-Select:{args.selection_mode}] {arm_name}: admitted {len(meta_rows)}/{n_prompts_total} molecules "
            f"(tier0_synth_hits={manifest['arms'][arm_name]['n_tier0_synth_hits']})",
            flush=True,
        )

    (args.dft_work_dir / "dft_cohort_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_dft(args: argparse.Namespace) -> None:
    from gpu4pyscf.dft.rks import RKS
    from pyscf import gto, lib

    lib.num_threads(4)
    manifest = json.loads((args.dft_work_dir / "dft_cohort_manifest.json").read_text())
    ptable = Chem.GetPeriodicTable()

    def _arr(x: Any) -> np.ndarray:
        return x.get() if hasattr(x, "get") else np.asarray(x)

    def _make_mol(syms: Sequence[str], coords: np.ndarray, log_path: Path) -> gto.Mole:
        ecp = {s: "def2-tzvp" for s in set(syms) if ptable.GetAtomicNumber(s) > 36}
        return gto.M(
            atom=list(zip(syms, coords.tolist())),
            basis="def2-tzvp",
            ecp=ecp,
            unit="Angstrom",
            charge=0,
            spin=0,
            verbose=4,
            output=str(log_path),
        )

    def _build_mf(mol: gto.Mole):
        mf = RKS(mol).density_fit(auxbasis="def2-universal-jkfit")
        mf.xc = "pbe0"
        mf.grids.level = 3
        mf.conv_tol = 1e-8
        mf.max_cycle = 100
        return mf

    target_arms = set(args.only_arms) if args.only_arms else set(manifest["arms"].keys())
    for arm_name, arm_info in manifest["arms"].items():
        if arm_name not in target_arms:
            continue
        sdf_p = Path(arm_info["selected_sdf"])
        mols = list(Chem.SDMolSupplier(str(sdf_p), removeHs=False))
        sp_root = args.dft_work_dir / arm_name / "sp"
        sp_root.mkdir(parents=True, exist_ok=True)
        print(f"[DFT-Run] Arm={arm_name} ({len(mols)} molecules)...", flush=True)
        for idx, m in enumerate(mols):
            out_dir = sp_root / f"mol_{idx:04d}"
            res_p = out_dir / "result.json"
            if res_p.exists():
                prev = json.loads(res_p.read_text())
                if prev.get("status") == "completed":
                    continue
            out_dir.mkdir(parents=True, exist_ok=True)
            syms = [a.GetSymbol() for a in m.GetAtoms()]
            coords = m.GetConformer().GetPositions()
            t0 = time.perf_counter()
            rep: Dict[str, Any] = {
                "dft_index": idx,
                "arm_name": arm_name,
                "atoms": len(syms),
                "elements": sorted(set(syms)),
                "protocol": "PBE0/def2-TZVP/DF(def2-universal-jkfit), grid level3, SCF1e-8",
                "status": "running",
            }
            try:
                gmol = _make_mol(syms, coords, out_dir / "singlepoint.log")
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
                    raise RuntimeError("PBE0 SCF did not converge after DIIS + SOSCF Newton fallback")
                e = _arr(mf.mo_energy)
                occ = _arr(mf.mo_occ)
                homo_eV = float(e[occ > 0].max() * HA_TO_EV)
                lumo_eV = float(e[occ == 0].min() * HA_TO_EV)
                rep["singlepoint"] = {
                    "homo_eV": homo_eV,
                    "lumo_eV": lumo_eV,
                    "energy_Ha": float(mf.e_tot),
                    "used_newton_fallback": used_newton,
                }
                rep["status"] = "completed"
            except Exception as exc:
                rep["status"] = "failed"
                rep["reason"] = str(exc)
            rep["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
            res_p.write_text(json.dumps(rep, indent=2) + "\n")
            print(
                f"[{arm_name} {idx+1:2d}/{len(mols)}] status={rep['status']} atoms={len(syms)} "
                f"dt={rep['elapsed_seconds']:.1f}s homo={rep.get('singlepoint', {}).get('homo_eV')}",
                flush=True,
            )


def run_summarize(args: argparse.Namespace) -> None:
    manifest = json.loads((args.dft_work_dir / "dft_cohort_manifest.json").read_text())
    summary_arms: Dict[str, Any] = {}

    for arm_name, arm_info in manifest["arms"].items():
        rows = arm_info["rows"]
        sp_root = args.dft_work_dir / arm_name / "sp"
        pbe0_h = np.full(len(rows), np.nan, dtype=np.float64)
        pbe0_l = np.full(len(rows), np.nan, dtype=np.float64)
        wall_s: List[float] = []
        n_conv = 0
        for idx in range(len(rows)):
            rp = sp_root / f"mol_{idx:04d}" / "result.json"
            if rp.exists():
                rd = json.loads(rp.read_text())
                if rd.get("status") == "completed":
                    n_conv += 1
                    pbe0_h[idx] = float(rd["singlepoint"]["homo_eV"])
                    pbe0_l[idx] = float(rd["singlepoint"]["lumo_eV"])
                    wall_s.append(float(rd.get("elapsed_seconds", 0.0)))

        targ_h = np.array([float(r["homo_target_eV"]) for r in rows], dtype=np.float64)
        uni_h = np.array([float(r["unimol2_homo_eV"]) for r in rows], dtype=np.float64)
        dpa_h = np.array([float(r["dpa4_homo_eV"]) for r in rows], dtype=np.float64)
        sa_arr = np.array([float(r["sa_score"]) for r in rows], dtype=np.float64)
        fm_arr = np.array([float(r["f_max_eV_A"]) for r in rows], dtype=np.float64)
        nat_arr = np.array([int(r["n_atoms"]) for r in rows], dtype=np.int32)

        n_attempted = int(arm_info.get("n_prompts_attempted", len(rows)))
        pbe0_valid = np.isfinite(pbe0_h)
        sa_ok = sa_arr <= 3.5
        raw_pbe0_hit = pbe0_valid & (np.abs(pbe0_h - targ_h) <= 0.25)
        raw_pbe0_synth_hit = raw_pbe0_hit & sa_ok
        bias_val = float(np.mean((pbe0_h - targ_h)[pbe0_valid])) if pbe0_valid.any() else 0.0
        cen_pbe0_synth_hit = pbe0_valid & (np.abs((pbe0_h - bias_val) - targ_h) <= 0.25) & sa_ok

        summary_arms[arm_name] = {
            "n_prompts_attempted": n_attempted,
            "n_selected": len(rows),
            "n_tier0_synth_hits": int(arm_info["n_tier0_synth_hits"]),
            "n_scf_converged": n_conv,
            "scf_convergence_pct": round(100.0 * n_conv / max(len(rows), 1), 2),
            "sa_leq_3_5_count": int(np.sum(sa_ok)),
            "end_to_end_pbe0_synth_clean_hits_raw": int(np.sum(raw_pbe0_synth_hit)),
            "end_to_end_pbe0_synth_clean_yield_raw": round(float(np.sum(raw_pbe0_synth_hit)) / max(n_attempted, 1), 4),
            "end_to_end_pbe0_synth_clean_hits_centered": int(np.sum(cen_pbe0_synth_hit)),
            "end_to_end_pbe0_synth_clean_yield_centered": round(float(np.sum(cen_pbe0_synth_hit)) / max(n_attempted, 1), 4),
            "mean_n_atoms": round(float(np.mean(nat_arr)), 1) if len(nat_arr) else None,
            "median_sa_score": round(float(np.nanmedian(sa_arr)), 4) if len(sa_arr) else None,
            "median_f_max_eV_A": round(float(np.nanmedian(fm_arr)), 4) if np.isfinite(fm_arr).any() else None,
            "mean_scf_wall_sec": round(float(np.mean(wall_s)), 2) if wall_s else None,
            "pbe0_vs_target_homo": _pair_metrics(pbe0_h, targ_h),
            "unimol2_vs_pbe0_homo": _pair_metrics(uni_h, pbe0_h),
            "dpa4_vs_pbe0_homo": _pair_metrics(dpa_h, pbe0_h),
            "unimol2_vs_target_homo": _pair_metrics(uni_h, targ_h),
            "dpa4_vs_target_homo": _pair_metrics(dpa_h, targ_h),
        }

    out = {
        "experiment": "OE62 Head-to-Head Matched PBE0/def2-TZVP Ab Initio Single-Point Verification",
        "dft_protocol": "PBE0/def2-TZVP/DF(def2-universal-jkfit), grid level3, SCF 1e-8 (gas phase GPU4PySCF)",
        "arms": summary_arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[DFT-Summarize DONE] Wrote {args.output_json}\n{json.dumps(out, indent=2)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["export_uniloom_cohort", "select_cohorts", "run_dft", "summarize"], required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--dft-work-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/head2head_dft_anchor"),
    )
    p.add_argument(
        "--uniloom-sdf",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/uniloom_v20_process_16_to_4.sdf"),
    )
    p.add_argument("--pmf-cohort-dir", type=Path, default=None)
    p.add_argument("--selection-mode", choices=["top_hits", "unbiased_fixed_prompts"], default="top_hits")
    p.add_argument("--per-quantile", type=int, default=6)
    p.add_argument("--only-arms", type=str, nargs="*", default=None)
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/oe62_head2head_dft_anchor_eval.json"),
    )
    # Needed for export_uniloom_cohort
    p.add_argument("--oe62-task-spec", type=Path, default=Path("configs/tasks/oe62_orbital_synthesizable.yaml"))
    p.add_argument(
        "--oe62-v20-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/v20_synthesizable_homo/loom_v2_synthesizable_homo.pt"),
    )
    p.add_argument(
        "--oe62-critic-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/octree_process_critic/oe62_v20_process_critic.pt"),
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
    args = p.parse_args()
    if args.phase == "export_uniloom_cohort":
        run_export_uniloom_cohort(args)
    elif args.phase == "select_cohorts":
        run_select_cohorts(args)
    elif args.phase == "run_dft":
        run_dft(args)
    elif args.phase == "summarize":
        run_summarize(args)


if __name__ == "__main__":
    main()
