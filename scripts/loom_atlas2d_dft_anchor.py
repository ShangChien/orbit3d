"""Stratified 2D GP-Atlas PBE0/def2-TZVP Quantum Anchor Verification.

Three clean phases:
  --phase sample     (GPU / checkout venv): Select 8 Lowest-Divergence and 8 Highest-Divergence
                     physical 2D GP-Atlas cells, sample 5 unique clean DPA-4-relaxed molecules
                     per cell (40 + 40 = 80) + stratified 16-molecule subset (8 + 8) for full
                     PBE-D3BJ/def2-TZVP/DF geometry relaxation.
  --phase run_dft    (GPU / GPU4PySCF 1.9.1): Persistent single-process GPU4PySCF execution for
                     all 80 PBE0/def2-TZVP/DF single points (with SOSCF newton fallback) and 16
                     PBE-D3BJ/def2-TZVP/DF geometry optimizations + PBE0 rechecks.
  --phase summarize  (checkout venv): Aggregate overall and Low/High-Divergence stratified MAE,
                     RMSE, systematic shift, shift-corrected MAE/RMSE, Pearson r, Spearman rho,
                     empirical cross-oracle |Uni-Mol2 - DPA-4|, relaxation graph preservation,
                     heavy/all-atom Kabsch RMSD, and adiabatic PBE0 orbital shift into JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

HA_TO_EV = 27.211386245988
SUPPORTED_TZVP_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}


def _kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    cov = a.T @ b
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0]) @ vt
    aligned = a @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=-1))))


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
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _pair_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(ref)
    p = pred[mask]
    r = ref[mask]
    if len(p) == 0:
        return {
            "n": 0,
            "mae_eV": float("nan"),
            "rmse_eV": float("nan"),
            "mean_bias_eV": float("nan"),
            "centered_mae_eV": float("nan"),
            "centered_rmse_eV": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
        }
    diff = p - r
    bias = float(np.mean(diff))
    cdiff = diff - bias
    pr = float(np.corrcoef(p, r)[0, 1]) if (len(p) >= 4 and np.std(p) > 1e-9 and np.std(r) > 1e-9) else float("nan")
    sr = _spearman_r(p, r)
    return {
        "n": int(len(p)),
        "mae_eV": round(float(np.mean(np.abs(diff))), 4),
        "rmse_eV": round(float(np.sqrt(np.mean(diff ** 2))), 4),
        "mean_bias_eV": round(bias, 4),
        "centered_mae_eV": round(float(np.mean(np.abs(cdiff))), 4),
        "centered_rmse_eV": round(float(np.sqrt(np.mean(cdiff ** 2))), 4),
        "pearson_r": round(pr, 4) if np.isfinite(pr) else float("nan"),
        "spearman_rho": round(sr, 4) if np.isfinite(sr) else float("nan"),
    }


def _mol_with_props(mol: Chem.Mol, props: Dict[str, Any]) -> Chem.Mol:
    copy_m = Chem.Mol(mol)
    for k, v in props.items():
        copy_m.SetProp(str(k), str(v))
    return copy_m


def run_phase_sample(args: argparse.Namespace) -> None:
    import torch
    from loom.chemistry.valence_reconciler import ValenceReconciler
    from loom.generator.molecular_state import mol_to_clean_smiles
    from loom.weaving import AuditOracleSlot, LoomUni3DARGenerator, PotentialSlot, RewardOracleSlot

    args.work_dir.mkdir(parents=True, exist_ok=True)
    atlas = json.loads(args.atlas_json.read_text())
    cells_all = atlas["joint_2d_atlas"]["empirical_cells"]
    phys_cells = [
        c
        for c in cells_all
        if float(c["homo_target_eV"]) < float(c["lumo_target_eV"]) - 1.5 and int(c["evaluated"]) >= 10
    ]
    phys_sorted = sorted(phys_cells, key=lambda c: float(c["cal_rmse_2d_eV"]))
    if len(phys_sorted) < 16:
        raise RuntimeError(f"Expected >=16 physical 2D cells, found {len(phys_sorted)}")

    low_cells = [dict(c, stratum="low_divergence", cell_rank=i) for i, c in enumerate(phys_sorted[:8])]
    high_cells = [dict(c, stratum="high_divergence", cell_rank=i) for i, c in enumerate(phys_sorted[-8:])]
    selected_cells = low_cells + high_cells

    print(
        f"[Anchor-Sample] Selected 8 Low-Div cells (cal_rmse {low_cells[0]['cal_rmse_2d_eV']:.3f}..{low_cells[-1]['cal_rmse_2d_eV']:.3f} eV) "
        f"and 8 High-Div cells (cal_rmse {high_cells[0]['cal_rmse_2d_eV']:.3f}..{high_cells[-1]['cal_rmse_2d_eV']:.3f} eV)",
        flush=True,
    )

    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.joint_checkpoint, device=args.device, batch_size=48)
    gen.eval()
    device = torch.device(args.device)
    allowed_elems = tuple(gen.args.allow_atoms.split(","))
    reconciler = ValenceReconciler(allowed_elements=allowed_elems)
    pot_slot = PotentialSlot(backend="dpa4", weights=args.potential_weights, device=args.device)
    rew_slot = RewardOracleSlot(backend="unimol2", checkpoint_dir=args.reward_oracle_dir, device=args.device)
    aud_slot = AuditOracleSlot(
        backend="dpa4", checkpoint_dir=args.audit_oracle_dir, device=args.device, forcefield_path=args.potential_weights
    )

    seen_smiles: set[str] = set()
    cohort: List[Dict[str, Any]] = []
    ptable = Chem.GetPeriodicTable()

    for c_idx, cell in enumerate(selected_cells):
        t_h = float(cell["homo_target_eV"])
        t_l = float(cell["lumo_target_eV"])
        collected_cell: List[Dict[str, Any]] = []
        for round_i in range(8):
            if len(collected_cell) >= args.per_cell:
                break
            batch_n = 32
            t_mat = np.repeat(np.array([[t_h, t_l]], dtype=np.float32), batch_n, axis=0)
            m_act = torch.ones((batch_n, 2), dtype=torch.bool, device=device)
            trajs = gen.sample(
                gen.normalize_targets(t_mat),
                m_act,
                cfg_scale=2.0,
                xyz_cfg_scale=1.0,
                cfg_tree_decay=0.30,
                geometric_guard=True,
                temperatures=(0.85, 0.30, 0.25),
                seed=88001 + c_idx * 97 + round_i * 19,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
            )
            candidates: List[Chem.Mol] = []
            cand_raw_exotic: List[bool] = []
            for tr in trajs:
                if len(tr["atom_symbols"]) > args.max_atoms:
                    continue
                if any(ptable.GetAtomicNumber(s) not in SUPPORTED_TZVP_Z for s in tr["atom_symbols"]):
                    continue
                res = reconciler.reconcile_point_cloud(tr["atom_symbols"], tr["positions_np"], potential_slot=None)
                if res.reconciled_legal and not res.reconciled_exotic_charge and res.mol is not None:
                    candidates.append(res.mol)
                    cand_raw_exotic.append(bool(res.raw_exotic_charge))
            if not candidates:
                continue
            rel = pot_slot.relax_molecules(candidates, n_steps=20, chunk_size=32)
            for j, ok_rel in enumerate(rel["graph_preserved"]):
                if not ok_rel or rel["relaxed_mols"][j] is None:
                    continue
                rmol = rel["relaxed_mols"][j]
                smi = mol_to_clean_smiles(rmol)
                if not smi or smi in seen_smiles:
                    continue
                seen_smiles.add(smi)
                collected_cell.append(
                    {
                        "mol": rmol,
                        "smiles": smi,
                        "n_atoms": int(rmol.GetNumAtoms()),
                        "n_heavy": int(rmol.GetNumHeavyAtoms()),
                        "stratum": cell["stratum"],
                        "cell_id": c_idx,
                        "cell_cal_rmse_2d_eV": float(cell["cal_rmse_2d_eV"]),
                        "homo_target_eV": t_h,
                        "lumo_target_eV": t_l,
                        "raw_exotic_repaired": cand_raw_exotic[j],
                        "dpa4_final_fmax_eV_A": float(rel["final_f_max_eV_A"][j]),
                    }
                )
                if len(collected_cell) >= args.per_cell:
                    break
        print(
            f"[Anchor-Sample] Cell {c_idx:2d}/16 ({cell['stratum']:>15s}) target=({t_h:.2f}, {t_l:.2f}) eV -> "
            f"collected {len(collected_cell)}/{args.per_cell}",
            flush=True,
        )
        cohort.extend(collected_cell)

    if len(cohort) < 16 * args.per_cell:
        raise RuntimeError(f"Only collected {len(cohort)}/{16 * args.per_cell} clean anchor molecules")

    mols_80 = [item["mol"] for item in cohort]
    rew_preds, _, rew_valid, _ = rew_slot.evaluate_candidates(mols_80, batch_size=32)
    aud_preds, _, aud_valid, _ = aud_slot.evaluate_candidates(mols_80, batch_size=32)
    h_idx_r = rew_slot.target_cols.index("homo")
    l_idx_r = rew_slot.target_cols.index("lumo")
    h_idx_a = aud_slot.target_cols.index("homo")
    l_idx_a = aud_slot.target_cols.index("lumo")

    sdf80_path = args.work_dir / "atlas2d_dft_anchor_80.sdf"
    w80 = Chem.SDWriter(str(sdf80_path))
    meta_rows: List[Dict[str, Any]] = []
    for i, item in enumerate(cohort):
        uh = float(rew_preds[i, h_idx_r]) if rew_valid[i] else float("nan")
        ul = float(rew_preds[i, l_idx_r]) if rew_valid[i] else float("nan")
        dh = float(aud_preds[i, h_idx_a]) if aud_valid[i] else float("nan")
        dl = float(aud_preds[i, l_idx_a]) if aud_valid[i] else float("nan")
        item.update(
            {
                "index": i,
                "unimol2_homo_eV": uh,
                "unimol2_lumo_eV": ul,
                "dpa4_homo_eV": dh,
                "dpa4_lumo_eV": dl,
            }
        )
        m_prop = _mol_with_props(
            item["mol"],
            {
                "anchor_index": i,
                "stratum": item["stratum"],
                "cell_id": item["cell_id"],
                "homo_target_eV": round(item["homo_target_eV"], 4),
                "lumo_target_eV": round(item["lumo_target_eV"], 4),
                "unimol2_homo_eV": round(uh, 4),
                "unimol2_lumo_eV": round(ul, 4),
                "dpa4_homo_eV": round(dh, 4),
                "dpa4_lumo_eV": round(dl, 4),
            },
        )
        w80.write(m_prop)
        meta_rows.append({k: v for k, v in item.items() if k != "mol"})
    w80.close()

    # Select 1 compact molecule per cell (16 total: 8 low + 8 high) for full PBE-D3BJ/def2-TZVP optimization
    relax16_meta: List[Dict[str, Any]] = []
    sdf16_path = args.work_dir / "atlas2d_dft_relax16.sdf"
    w16 = Chem.SDWriter(str(sdf16_path))
    for c_idx in range(16):
        cell_items = [row for row in cohort if row["cell_id"] == c_idx]
        chosen = min(cell_items, key=lambda r: r["n_atoms"])
        orig_i = chosen["index"]
        r_idx = len(relax16_meta)
        m16 = _mol_with_props(
            chosen["mol"],
            {
                "relax_index": r_idx,
                "orig_idx_in_80": orig_i,
                "stratum": chosen["stratum"],
                "cell_id": c_idx,
            },
        )
        w16.write(m16)
        relax16_meta.append(
            {
                "relax_index": r_idx,
                "orig_idx_in_80": orig_i,
                "stratum": chosen["stratum"],
                "cell_id": c_idx,
                "n_atoms": chosen["n_atoms"],
                "smiles": chosen["smiles"],
                "dpa4_coords_A": chosen["mol"].GetConformer().GetPositions().tolist(),
                "symbols": [a.GetSymbol() for a in chosen["mol"].GetAtoms()],
            }
        )
    w16.close()

    meta_payload = {
        "n_cohort": len(meta_rows),
        "n_relax16": len(relax16_meta),
        "selected_cells": selected_cells,
        "cohort": meta_rows,
        "relax16": relax16_meta,
    }
    (args.work_dir / "atlas2d_dft_anchor_meta.json").write_text(json.dumps(meta_payload, indent=2) + "\n")
    print(f"[Anchor-Sample] Exported {sdf80_path} and {sdf16_path}", flush=True)


def run_phase_dft(args: argparse.Namespace) -> None:
    """Single-process persistent GPU4PySCF execution for 80 PBE0 SP + 16 PBE-D3BJ relaxations."""
    from gpu4pyscf.dft.rks import RKS
    from pyscf import gto, lib
    from pyscf.geomopt.geometric_solver import kernel as geometric_kernel

    lib.num_threads(4)
    work_dir = args.work_dir
    sp_root = work_dir / "sp80"
    rel_root = work_dir / "relax16"
    sp_root.mkdir(parents=True, exist_ok=True)
    rel_root.mkdir(parents=True, exist_ok=True)

    def _arr(x: Any) -> np.ndarray:
        return x.get() if hasattr(x, "get") else np.asarray(x)

    def _make_mol(syms: Sequence[str], coords: np.ndarray, log_path: Path) -> gto.Mole:
        ptable = Chem.GetPeriodicTable()
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

    def _build_mf(mol: gto.Mole, xc: str, disp: Optional[str] = None):
        mf = RKS(mol).density_fit(auxbasis="def2-universal-jkfit")
        mf.xc = xc
        mf.disp = disp
        mf.grids.level = 3
        mf.conv_tol = 1e-8
        mf.max_cycle = 100
        return mf

    def _solve_pbe0(mol: gto.Mole) -> Tuple[Dict[str, Any], Any]:
        mf = _build_mf(mol, "pbe0", disp=None)
        t0 = time.perf_counter()
        mf.kernel()
        used_newton = False
        if not mf.converged:
            used_newton = True
            dm0 = _arr(mf.make_rdm1())
            mf = _build_mf(mol, "pbe0", disp=None).newton()
            mf.max_cycle = 80
            mf.conv_tol = 1e-8
            mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError("PBE0 SCF did not converge after DIIS + SOSCF newton fallback")
        e = _arr(mf.mo_energy)
        occ = _arr(mf.mo_occ)
        homo_eV = float(e[occ > 0].max() * HA_TO_EV)
        lumo_eV = float(e[occ == 0].min() * HA_TO_EV)
        return (
            {
                "prediction_eV": [homo_eV, lumo_eV],
                "energy_Ha": float(mf.e_tot),
                "converged": True,
                "used_newton_fallback": used_newton,
                "seconds": round(time.perf_counter() - t0, 3),
            },
            mf,
        )

    # Part 1: 80 PBE0/def2-TZVP/DF single points
    sdf80_path = work_dir / "atlas2d_dft_anchor_80.sdf"
    mols80 = list(Chem.SDMolSupplier(str(sdf80_path), removeHs=False))
    print(f"[Anchor-DFT] Running PBE0/def2-TZVP/DF single-points on {len(mols80)} molecules...", flush=True)
    for idx, m in enumerate(mols80):
        out_dir = sp_root / f"mol_{idx:04d}"
        res_path = out_dir / "result.json"
        if res_path.exists():
            prev = json.loads(res_path.read_text())
            if prev.get("status") == "completed":
                continue
        out_dir.mkdir(parents=True, exist_ok=True)
        syms = [a.GetSymbol() for a in m.GetAtoms()]
        coords = m.GetConformer().GetPositions()
        report: Dict[str, Any] = {
            "index": idx,
            "atoms": len(syms),
            "elements": sorted(set(syms)),
            "stratum": m.GetProp("stratum") if m.HasProp("stratum") else "",
            "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "protocol": "PBE0/def2-TZVP/DF(def2-universal-jkfit), grid level3, SCF1e-8",
            "status": "running",
        }
        t0 = time.perf_counter()
        try:
            mol = _make_mol(syms, coords, out_dir / "singlepoint.log")
            sp, _ = _solve_pbe0(mol)
            report["singlepoint"] = sp
            report["status"] = "completed"
        except Exception as exc:
            report["status"] = "failed"
            report["reason"] = str(exc)
        report["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        res_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"[SP80 {idx+1:2d}/{len(mols80)}] status={report['status']} atoms={len(syms)} "
            f"dt={report['elapsed_seconds']:.1f}s pred={report.get('singlepoint', {}).get('prediction_eV')}",
            flush=True,
        )

    # Part 2: 16 PBE-D3BJ/def2-TZVP/DF geometry optimizations + PBE0/def2-TZVP/DF recheck
    sdf16_path = work_dir / "atlas2d_dft_relax16.sdf"
    mols16 = list(Chem.SDMolSupplier(str(sdf16_path), removeHs=False))
    print(f"[Anchor-DFT] Running PBE-D3BJ/def2-TZVP/DF optimization + PBE0 recheck on {len(mols16)} molecules...", flush=True)
    for idx, m in enumerate(mols16):
        out_dir = rel_root / f"mol_{idx:04d}"
        res_path = out_dir / "result.json"
        if res_path.exists():
            prev = json.loads(res_path.read_text())
            if prev.get("status") == "completed":
                continue
        out_dir.mkdir(parents=True, exist_ok=True)
        syms = [a.GetSymbol() for a in m.GetAtoms()]
        coords = m.GetConformer().GetPositions()
        orig_i = int(m.GetProp("orig_idx_in_80"))
        report16: Dict[str, Any] = {
            "relax_index": idx,
            "orig_idx_in_80": orig_i,
            "atoms": len(syms),
            "stratum": m.GetProp("stratum") if m.HasProp("stratum") else "",
            "status": "running",
        }
        t0 = time.perf_counter()
        try:
            optmol = _make_mol(syms, coords, out_dir / "geometry.log")
            optmf = _build_mf(optmol, "pbe", disp="d3bj")
            trace: List[Dict[str, float]] = []

            def _cb(env: Dict[str, Any]) -> None:
                grad = _arr(env["gradients"])
                trace.append(
                    {
                        "energy_Ha": float(env["energy"]),
                        "rms_gradient_Ha_Bohr": float(np.sqrt(np.mean(np.square(grad)))),
                        "max_atom_gradient_Ha_Bohr": float(np.linalg.norm(grad, axis=1).max()),
                    }
                )

            converged, relaxed = geometric_kernel(
                optmf,
                maxsteps=60,
                callback=_cb,
                convergence_energy=1e-6,
                convergence_grms=1e-4,
                convergence_gmax=2e-4,
                convergence_drms=1.2e-3,
                convergence_dmax=1.8e-3,
            )
            final_coords = relaxed.atom_coords(unit="Angstrom")
            report16["geometry"] = {
                "protocol": "PBE-D3BJ/def2-TZVP/DF",
                "converged": bool(converged),
                "gradient_evaluations": len(trace),
                "coordinates_A": final_coords.tolist(),
                "final_energy_Ha": trace[-1]["energy_Ha"] if trace else None,
            }
            final_mol = _make_mol(syms, final_coords, out_dir / "optimized_singlepoint.log")
            opt_sp, _ = _solve_pbe0(final_mol)
            report16["optimized_singlepoint"] = opt_sp
            report16["status"] = "completed" if converged else "geometry_not_converged"
        except Exception as exc:
            report16["status"] = "failed"
            report16["reason"] = str(exc)
        report16["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        res_path.write_text(json.dumps(report16, indent=2) + "\n")
        print(
            f"[Relax16 {idx+1:2d}/{len(mols16)}] status={report16['status']} atoms={len(syms)} "
            f"steps={report16.get('geometry', {}).get('gradient_evaluations')} dt={report16['elapsed_seconds']:.1f}s",
            flush=True,
        )


def run_phase_summarize(args: argparse.Namespace) -> None:
    from loom.generator.molecular_state import Mol3DInstance, mol3d_to_validated_rdkit_mol, mol_to_clean_smiles

    work_dir = args.work_dir
    meta = json.loads((work_dir / "atlas2d_dft_anchor_meta.json").read_text())
    cohort = meta["cohort"]

    records: List[Dict[str, Any]] = []
    for row in cohort:
        idx = int(row["index"])
        sp_path = work_dir / "sp80" / f"mol_{idx:04d}" / "result.json"
        rec = dict(row)
        if sp_path.exists():
            sp = json.loads(sp_path.read_text())
            rec["dft_status"] = sp.get("status")
            if sp.get("status") == "completed":
                pbe0_h, pbe0_l = sp["singlepoint"]["prediction_eV"]
                rec["pbe0_homo_eV"] = float(pbe0_h)
                rec["pbe0_lumo_eV"] = float(pbe0_l)
                rec["pbe0_energy_Ha"] = float(sp["singlepoint"]["energy_Ha"])
                rec["used_newton_fallback"] = bool(sp["singlepoint"].get("used_newton_fallback", False))
        else:
            rec["dft_status"] = "missing"
        records.append(rec)

    def _summarize_subset(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        conv = [r for r in sub if r.get("dft_status") == "completed"]
        uh = np.array([r["unimol2_homo_eV"] for r in conv], dtype=np.float64)
        ul = np.array([r["unimol2_lumo_eV"] for r in conv], dtype=np.float64)
        dh = np.array([r["dpa4_homo_eV"] for r in conv], dtype=np.float64)
        dl = np.array([r["dpa4_lumo_eV"] for r in conv], dtype=np.float64)
        ph = np.array([r["pbe0_homo_eV"] for r in conv], dtype=np.float64)
        pl = np.array([r["pbe0_lumo_eV"] for r in conv], dtype=np.float64)

        cross_2d = np.sqrt(0.5 * ((uh - dh) ** 2 + (ul - dl) ** 2))
        return {
            "n_total": len(sub),
            "n_scf_converged": len(conv),
            "scf_convergence_pct": round(100.0 * len(conv) / max(len(sub), 1), 2),
            "mean_empirical_cross_oracle_2d_diff_eV": round(float(np.mean(cross_2d)), 4) if len(conv) else None,
            "mean_empirical_cross_oracle_homo_mae_eV": round(float(np.mean(np.abs(uh - dh))), 4) if len(conv) else None,
            "mean_empirical_cross_oracle_lumo_mae_eV": round(float(np.mean(np.abs(ul - dl))), 4) if len(conv) else None,
            "unimol2_vs_pbe0": {
                "homo": _pair_metrics(uh, ph),
                "lumo": _pair_metrics(ul, pl),
                "joint_2d_mae_eV": round(float(0.5 * (np.mean(np.abs(uh - ph)) + np.mean(np.abs(ul - pl)))), 4) if len(conv) else None,
            },
            "dpa4_vs_pbe0": {
                "homo": _pair_metrics(dh, ph),
                "lumo": _pair_metrics(dl, pl),
                "joint_2d_mae_eV": round(float(0.5 * (np.mean(np.abs(dh - ph)) + np.mean(np.abs(dl - pl)))), 4) if len(conv) else None,
            },
        }

    low_sub = [r for r in records if r["stratum"] == "low_divergence"]
    high_sub = [r for r in records if r["stratum"] == "high_divergence"]

    # Summarize 16-molecule PBE-D3BJ/def2-TZVP relaxation cohort
    sp_by_idx = {int(r["index"]): r for r in records}
    relax_rows: List[Dict[str, Any]] = []
    heavy_rmsds: List[float] = []
    all_rmsds: List[float] = []
    adiabatic_h_shifts: List[float] = []
    adiabatic_l_shifts: List[float] = []
    n_opt_conv = 0
    n_graph_pres = 0

    for item16 in meta["relax16"]:
        r_idx = int(item16["relax_index"])
        orig_i = int(item16["orig_idx_in_80"])
        r_path = work_dir / "relax16" / f"mol_{r_idx:04d}" / "result.json"
        if not r_path.exists():
            continue
        rj = json.loads(r_path.read_text())
        geom = rj.get("geometry", {})
        opt_conv = bool(geom.get("converged", False))
        if opt_conv:
            n_opt_conv += 1

        graph_ok = False
        h_rmsd = float("nan")
        a_rmsd = float("nan")
        if "coordinates_A" in geom:
            dpa_coords = np.asarray(item16["dpa4_coords_A"], dtype=np.float64)
            dft_coords = np.asarray(geom["coordinates_A"], dtype=np.float64)
            syms = list(item16["symbols"])
            heavy_mask = np.array([s != "H" for s in syms], dtype=bool)
            a_rmsd = _kabsch_rmsd(dpa_coords, dft_coords)
            h_rmsd = _kabsch_rmsd(dpa_coords[heavy_mask], dft_coords[heavy_mask]) if heavy_mask.any() else a_rmsd
            all_rmsds.append(a_rmsd)
            heavy_rmsds.append(h_rmsd)

            elems = tuple(sorted(set(syms)))
            inst = Mol3DInstance(
                coords=dft_coords.astype(np.float32),
                atom_types=np.array([elems.index(s) for s in syms], dtype=np.int64),
                element_table=elems,
            )
            try:
                rebuilt = mol3d_to_validated_rdkit_mol(inst)
                graph_ok = mol_to_clean_smiles(rebuilt) == item16["smiles"]
            except ValueError:
                graph_ok = False
            if graph_ok:
                n_graph_pres += 1

        sp_init = sp_by_idx.get(orig_i, {})
        opt_sp = rj.get("optimized_singlepoint", {})
        dh_shift = float("nan")
        dl_shift = float("nan")
        if "prediction_eV" in opt_sp and "pbe0_homo_eV" in sp_init:
            dh_shift = abs(float(opt_sp["prediction_eV"][0]) - float(sp_init["pbe0_homo_eV"]))
            dl_shift = abs(float(opt_sp["prediction_eV"][1]) - float(sp_init["pbe0_lumo_eV"]))
            adiabatic_h_shifts.append(dh_shift)
            adiabatic_l_shifts.append(dl_shift)

        relax_rows.append(
            {
                "relax_index": r_idx,
                "orig_idx_in_80": orig_i,
                "stratum": item16["stratum"],
                "smiles": item16["smiles"],
                "opt_converged": opt_conv,
                "graph_preserved": graph_ok,
                "gradient_evaluations": geom.get("gradient_evaluations"),
                "heavy_kabsch_rmsd_A": round(h_rmsd, 4) if np.isfinite(h_rmsd) else None,
                "all_atom_kabsch_rmsd_A": round(a_rmsd, 4) if np.isfinite(a_rmsd) else None,
                "adiabatic_pbe0_homo_shift_eV": round(dh_shift, 4) if np.isfinite(dh_shift) else None,
                "adiabatic_pbe0_lumo_shift_eV": round(dl_shift, 4) if np.isfinite(dl_shift) else None,
            }
        )

    out = {
        "experiment": "Stratified 2D GP-Atlas PBE0/def2-TZVP Quantum Anchor Verification",
        "dft_singlepoint_protocol": "PBE0/def2-TZVP/DF(def2-universal-jkfit), grid level3, SCF 1e-8 (gas phase GPU4PySCF)",
        "dft_relaxation_protocol": "PBE-D3BJ/def2-TZVP/DF geomeTRIC + PBE0/def2-TZVP/DF recheck",
        "singlepoint_summary": {
            "overall": _summarize_subset(records),
            "low_divergence_stratum": _summarize_subset(low_sub),
            "high_divergence_stratum": _summarize_subset(high_sub),
        },
        "relaxation_16_summary": {
            "n_total": len(meta["relax16"]),
            "n_opt_converged": n_opt_conv,
            "opt_convergence_pct": round(100.0 * n_opt_conv / max(len(meta["relax16"]), 1), 2),
            "n_graph_preserved": n_graph_pres,
            "graph_preserved_pct": round(100.0 * n_graph_pres / max(len(meta["relax16"]), 1), 2),
            "mean_heavy_kabsch_rmsd_A": round(float(np.mean(heavy_rmsds)), 4) if heavy_rmsds else None,
            "median_heavy_kabsch_rmsd_A": round(float(np.median(heavy_rmsds)), 4) if heavy_rmsds else None,
            "mean_all_atom_kabsch_rmsd_A": round(float(np.mean(all_rmsds)), 4) if all_rmsds else None,
            "median_all_atom_kabsch_rmsd_A": round(float(np.median(all_rmsds)), 4) if all_rmsds else None,
            "mean_adiabatic_pbe0_homo_shift_eV": round(float(np.mean(adiabatic_h_shifts)), 4) if adiabatic_h_shifts else None,
            "mean_adiabatic_pbe0_lumo_shift_eV": round(float(np.mean(adiabatic_l_shifts)), 4) if adiabatic_l_shifts else None,
            "per_molecule": relax_rows,
        },
        "per_molecule_80": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[Anchor-Summarize] Wrote summary to {args.output_json}", flush=True)
    print(json.dumps(out["singlepoint_summary"], indent=2), flush=True)
    print(json.dumps({k: v for k, v in out["relaxation_16_summary"].items() if k != "per_molecule"}, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["sample", "run_dft", "summarize"], required=True)
    p.add_argument("--atlas-json", type=Path, default=ROOT / "docs" / "experiments" / "gp_atlas.json")
    p.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/stage25_joint_v11_grpo/grpo_groups_0256.pt"),
    )
    p.add_argument(
        "--reward-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_dual_oracle_5fe5/training/unimol2_11b/oracle"),
    )
    p.add_argument(
        "--audit-oracle-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/orbit_e15_calibration_20261114/full_fit/oracle"),
    )
    p.add_argument(
        "--potential-weights",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/pretrained/DPA4-Mini-OMol25-100M-v20260820.pt"),
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/atlas2d_dft_anchor"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/atlas2d_dft_anchor_eval.json"),
    )
    p.add_argument("--per-cell", type=int, default=5)
    p.add_argument("--max-atoms", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    if args.phase == "sample":
        run_phase_sample(args)
    elif args.phase == "run_dft":
        run_phase_dft(args)
    else:
        run_phase_summarize(args)


if __name__ == "__main__":
    main()
