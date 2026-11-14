"""Consolidate ALL historical same-protocol (`PBE0/def2-TZVP`) quantum calculations into OE62.

Sweeps all historical experiment directories (`E8, E11, E12, E13, E13-G, E14, E16, dual_oracle, oe62_point`),
verifies basis/ECP/protocol consistency, excludes official OE62 holdout scaffolds (`valid/calibration/test`)
and existing `train_source.sdf` graphs, deduplicates to 1 geometry per canonical molecular graph, labels all
admitted structures with `Uni-Mol2 1.1B` (`B_homo`, `B_lumo`) alongside `DFT_homo`, `DFT_lumo`, evaluates
`AuditOracleSlot` (`E15` calibrated `DPA-4`) against `DFT` on quantum `valid`/`test`, and builds the unified
`oe62_master_train_B.sdf` dataset for `Uni-Loom`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from loom.generator.molecular_state import Mol3DInstance, mol3d_to_validated_rdkit_mol, mol_to_clean_smiles
from loom.oracles.unimol2_task import B_WEIGHT_SHA256, graph_set
from loom.utils.dataset_io import write_precise_sdf
from loom.weaving.slots import AuditOracleSlot, DEFAULT_SUPPORTED_ELEMENTS, RewardOracleSlot

OE62_DFT_PROTOCOL = "PBE0/def2-TZVP/DF(def2-universal-jkfit), grid level3, SCF1e-8; approximate OE62 reference"


def murcko_scaffold(mol: Chem.Mol) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(mol), includeChirality=False) or "<ACYCLIC>"


def split_for_scaffold(scaf: str) -> str:
    val = int(hashlib.sha256(("E17-existing-DFT:" + scaf).encode()).hexdigest()[:16], 16) / (2 ** 64)
    return "train" if val < 0.8 else ("valid" if val < 0.9 else "test")


def discover_result_jsons(search_root: Path) -> List[Path]:
    found: List[Path] = []
    prune_dirs = {".venv", ".git", "scratch", "__pycache__", "node_modules"}
    for entry in sorted(search_root.glob("orbit*")):
        if not entry.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(entry):
            dirnames[:] = sorted(d for d in dirnames if d not in prune_dirs and not d.startswith("."))
            if "result.json" in filenames:
                found.append(Path(dirpath) / "result.json")
    return sorted(found)


def build_validated_mol(symbols: List[str], coords_list: List[List[float]]) -> Optional[Chem.Mol]:
    allowed = tuple(DEFAULT_SUPPORTED_ELEMENTS)
    allowed_set = set(allowed)
    if not symbols or len(symbols) > 80 or any(s not in allowed_set for s in symbols):
        return None
    coords = np.asarray(coords_list, dtype=np.float32)
    if coords.ndim != 2 or coords.shape != (len(symbols), 3) or not np.isfinite(coords).all():
        return None
    atom_types = np.array([allowed.index(s) for s in symbols], dtype=np.int64)
    inst = Mol3DInstance(coords=coords, atom_types=atom_types, element_table=allowed)
    try:
        return mol3d_to_validated_rdkit_mol(inst)
    except (ValueError, RuntimeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, default=Path("/sjtu-combustion/wjw/chensq"))
    parser.add_argument("--source-splits-dir", type=Path, required=True, help="Directory with valid.csv, calibration.csv, test.csv")
    parser.add_argument("--source-train-sdf", type=Path, required=True, help="Official B-relabeled train_source.sdf")
    parser.add_argument("--e17-manifest", type=Path, default=None, help="Optional legacy E17 manifest for delta comparison")
    parser.add_argument("--reward-oracle-dir", type=Path, required=True)
    parser.add_argument("--audit-oracle-dir", type=Path, default=None)
    parser.add_argument("--dpa4-forcefield", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Consolidate] Loading source graphs from {args.source_train_sdf}...", flush=True)
    source_mols = [m for m in Chem.SDMolSupplier(str(args.source_train_sdf), removeHs=False) if m is not None]
    source_graphs = {mol_to_clean_smiles(m) for m in source_mols}

    heldout_scaffolds = set()
    for split_name in ("valid", "calibration", "test"):
        csv_path = args.source_splits_dir / f"{split_name}.csv"
        df = pd.read_csv(csv_path)
        heldout_scaffolds.update(df["scaffold"].dropna().astype(str).tolist())
    print(
        f"[Consolidate] Source train molecules={len(source_mols)} (unique graphs={len(source_graphs)}), "
        f"heldout scaffolds={len(heldout_scaffolds)}",
        flush=True,
    )

    e17_graphs = set()
    if args.e17_manifest is not None and args.e17_manifest.exists():
        e17_data = json.loads(args.e17_manifest.read_text())
        e17_graphs = {r["smiles"] for r in e17_data.get("records", []) if r.get("status") == "accepted" and "smiles" in r}

    result_paths = discover_result_jsons(args.search_root)
    print(f"[Consolidate] Discovered {len(result_paths)} historical result.json files across {args.search_root}", flush=True)

    exclusion_counts: Counter = Counter()
    cohort_raw_counts: Counter = Counter()
    cohort_valid_candidates: Counter = Counter()
    # candidate tuple: (priority_is_unopt, geom_sha256, graph_smiles, scaf, mol, dft_pred, geom_origin, cohort_name, result_path)
    candidates: List[Tuple[int, str, str, str, Chem.Mol, List[float], str, str, str]] = []

    for r_path in result_paths:
        cohort_key = str(r_path.parent.parent.relative_to(args.search_root))
        cohort_raw_counts[cohort_key] += 1
        try:
            q = json.loads(r_path.read_text())
        except Exception:
            exclusion_counts["corrupt_json"] += 1
            continue

        if not isinstance(q, dict) or "protocol" not in q or "symbols" not in q:
            exclusion_counts["not_singlepoint_schema"] += 1
            continue
        if q.get("protocol") != OE62_DFT_PROTOCOL:
            exclusion_counts["non_oe62_protocol"] += 1
            continue

        syms = q.get("symbols", [])
        if "I" in syms and q.get("ecp_core_electrons", {}).get("I") != 28:
            exclusion_counts["missing_iodine_ecp"] += 1
            continue

        # Prefer converged, graph-preserved DFT-optimized conformer if available; otherwise converged input singlepoint
        chosen_coords = None
        chosen_dft = None
        geom_origin = ""
        priority_unopt = 1

        opt_sp = q.get("optimized_singlepoint")
        geom_info = q.get("geometry", {})
        if (
            isinstance(opt_sp, dict)
            and opt_sp.get("converged") is True
            and geom_info.get("converged") is True
            and geom_info.get("graph_preserved") is True
            and "coordinates_A" in geom_info
        ):
            chosen_coords = geom_info["coordinates_A"]
            chosen_dft = opt_sp["prediction_eV"]
            geom_origin = "DFT_PBE_D3BJ_optimized_conformer"
            priority_unopt = 0
        else:
            sp = q.get("singlepoint")
            if isinstance(sp, dict) and sp.get("converged") is True and "coordinates_A" in q:
                chosen_coords = q["coordinates_A"]
                chosen_dft = sp["prediction_eV"]
                geom_origin = "frozen_3d_input_conformer"
                priority_unopt = 1

        if chosen_coords is None or chosen_dft is None or len(chosen_dft) != 2 or not np.isfinite(chosen_dft).all():
            exclusion_counts[f"scf_or_status_{q.get('status', 'incomplete')}"] += 1
            continue

        mol = build_validated_mol(syms, chosen_coords)
        if mol is None:
            exclusion_counts["invalid_or_unsupported_3d_graph"] += 1
            continue

        graph = mol_to_clean_smiles(mol)
        scaf = murcko_scaffold(mol)
        if scaf in heldout_scaffolds:
            exclusion_counts["source_holdout_scaffold"] += 1
            continue
        if graph in source_graphs:
            exclusion_counts["already_source_training_graph"] += 1
            continue

        geom_hash = hashlib.sha256(
            (graph + json.dumps(chosen_coords, separators=(",", ":"))).encode()
        ).hexdigest()
        cohort_valid_candidates[cohort_key] += 1
        candidates.append((priority_unopt, geom_hash, graph, scaf, mol, [float(chosen_dft[0]), float(chosen_dft[1])], geom_origin, cohort_key, str(r_path)))

    # Deterministic deduplication: prefer DFT-optimized (priority 0) then lowest geometry SHA256
    candidates.sort(key=lambda item: (item[0], item[1]))
    seen_graphs = set()
    split_mols_dft: Dict[str, List[Chem.Mol]] = {"train": [], "valid": [], "test": []}
    accepted_cohort_counts: Counter = Counter()
    newly_recovered_beyond_e17 = 0

    for prio, geom_hash, graph, scaf, mol, dft_vals, geom_origin, cohort_key, r_path_str in candidates:
        if graph in seen_graphs:
            exclusion_counts["duplicate_graph"] += 1
            continue
        seen_graphs.add(graph)
        sp_name = split_for_scaffold(scaf)
        accepted_cohort_counts[cohort_key] += 1
        if e17_graphs and graph not in e17_graphs:
            newly_recovered_beyond_e17 += 1

        out_mol = Chem.Mol(mol)
        out_mol.SetProp("SourceID", f"ConsolidatedDFT:{geom_hash[:24]}")
        out_mol.SetProp("OriginType", "historical_quantum")
        out_mol.SetProp("DFTSourceCohort", cohort_key)
        out_mol.SetProp("DFTResultPath", r_path_str)
        out_mol.SetProp("DFTProtocol", OE62_DFT_PROTOCOL)
        out_mol.SetProp("GeometryOrigin", geom_origin)
        out_mol.SetProp("MurckoScaffold", scaf)
        out_mol.SetProp("DFT_homo", f"{dft_vals[0]:.8f}")
        out_mol.SetProp("DFT_lumo", f"{dft_vals[1]:.8f}")
        out_mol.SetProp("homo", f"{dft_vals[0]:.8f}")
        out_mol.SetProp("lumo", f"{dft_vals[1]:.8f}")
        split_mols_dft[sp_name].append(out_mol)

    print(
        f"[Consolidate] Unique DFT-verified generated graphs: "
        f"train={len(split_mols_dft['train'])}, valid={len(split_mols_dft['valid'])}, test={len(split_mols_dft['test'])} "
        f"(newly recovered beyond E17: +{newly_recovered_beyond_e17})",
        flush=True,
    )

    # Evaluate RewardOracleSlot (`Uni-Mol2 1.1B`) across all splits
    print(f"[Consolidate] Labeling consolidated quantum splits with Uni-Mol2 1.1B on {args.device}...", flush=True)
    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )

    audit_slot: Optional[AuditOracleSlot] = None
    if args.audit_oracle_dir is not None and args.audit_oracle_dir.exists():
        audit_slot = AuditOracleSlot(
            backend="dpa4",
            checkpoint_dir=args.audit_oracle_dir,
            device=args.device,
            forcefield_path=str(args.dpa4_forcefield) if args.dpa4_forcefield else None,
        )

    split_mols_B: Dict[str, List[Chem.Mol]] = {"train": [], "valid": [], "test": []}
    calibration_stats: Dict[str, Any] = {}

    for sp_name, m_list in split_mols_dft.items():
        write_precise_sdf(args.output_dir / f"quantum_{sp_name}_dft.sdf", m_list)
        b_preds, _, b_mask, b_reasons = reward_slot.evaluate_candidates(m_list, batch_size=32)

        aud_preds = None
        aud_mask = None
        if audit_slot is not None and sp_name in ("valid", "test"):
            aud_preds, _, aud_mask, _ = audit_slot.evaluate_candidates(m_list, batch_size=32)

        dft_arr = np.array([[float(m.GetProp("DFT_homo")), float(m.GetProp("DFT_lumo"))] for m in m_list], dtype=np.float64)
        admitted_b_mols: List[Chem.Mol] = []
        for i, m in enumerate(m_list):
            if b_mask[i]:
                bm = Chem.Mol(m)
                bh, bl = float(b_preds[i, 0]), float(b_preds[i, 1])
                bm.SetProp("B_homo", f"{bh:.8f}")
                bm.SetProp("B_lumo", f"{bl:.8f}")
                # Primary training/task properties use B_homo/B_lumo while preserving DFT_homo/DFT_lumo
                bm.SetProp("homo", f"{bh:.8f}")
                bm.SetProp("lumo", f"{bl:.8f}")
                admitted_b_mols.append(bm)
        split_mols_B[sp_name] = admitted_b_mols
        write_precise_sdf(args.output_dir / f"quantum_{sp_name}_B.sdf", admitted_b_mols)

        sp_stat: Dict[str, Any] = {
            "dft_total": len(m_list),
            "B_admitted": len(admitted_b_mols),
            "B_excluded_reasons": dict(Counter(r for r in b_reasons if r is not None)),
        }
        if b_mask.any():
            diff_b = b_preds[b_mask] - dft_arr[b_mask]
            sp_stat["B_vs_DFT"] = {
                "homo_mae_eV": float(np.mean(np.abs(diff_b[:, 0]))),
                "homo_rmse_eV": float(np.sqrt(np.mean(diff_b[:, 0] ** 2))),
                "lumo_mae_eV": float(np.mean(np.abs(diff_b[:, 1]))),
                "lumo_rmse_eV": float(np.sqrt(np.mean(diff_b[:, 1] ** 2))),
            }
        if aud_preds is not None and aud_mask is not None and aud_mask.any():
            diff_a = aud_preds[aud_mask] - dft_arr[aud_mask]
            sp_stat["Audit_DPA4_vs_DFT"] = {
                "evaluated": int(aud_mask.sum()),
                "homo_mae_eV": float(np.mean(np.abs(diff_a[:, 0]))),
                "homo_rmse_eV": float(np.sqrt(np.mean(diff_a[:, 0] ** 2))),
                "lumo_mae_eV": float(np.mean(np.abs(diff_a[:, 1]))),
                "lumo_rmse_eV": float(np.sqrt(np.mean(diff_a[:, 1] ** 2))),
            }
            both = b_mask & aud_mask
            if both.any():
                diff_ba = b_preds[both] - aud_preds[both]
                sp_stat["B_vs_Audit_DPA4"] = {
                    "paired": int(both.sum()),
                    "homo_mae_eV": float(np.mean(np.abs(diff_ba[:, 0]))),
                    "homo_rmse_eV": float(np.sqrt(np.mean(diff_ba[:, 0] ** 2))),
                    "lumo_mae_eV": float(np.mean(np.abs(diff_ba[:, 1]))),
                    "lumo_rmse_eV": float(np.sqrt(np.mean(diff_ba[:, 1] ** 2))),
                }
        calibration_stats[sp_name] = sp_stat

    # Build unified master OE62 training SDF (`oe62_master_train_B.sdf`)
    master_train: List[Chem.Mol] = []
    for sm in source_mols:
        cm = Chem.Mol(sm)
        cm.SetProp("OriginType", "oe62_source")
        master_train.append(cm)
    master_train.extend(split_mols_B["train"])
    write_precise_sdf(args.output_dir / "oe62_master_train_B.sdf", master_train)

    manifest = {
        "protocol": OE62_DFT_PROTOCOL,
        "source_train_count": len(source_mols),
        "heldout_scaffolds_excluded_count": len(heldout_scaffolds),
        "total_result_jsons_scanned": len(result_paths),
        "newly_recovered_unique_graphs_beyond_e17": newly_recovered_beyond_e17,
        "consolidated_quantum_dft_counts": {k: len(v) for k, v in split_mols_dft.items()},
        "consolidated_quantum_B_admitted_counts": {k: len(v) for k, v in split_mols_B.items()},
        "oe62_master_train_B_total": len(master_train),
        "exclusion_breakdown": dict(exclusion_counts),
        "raw_jobs_by_cohort": dict(cohort_raw_counts),
        "accepted_unique_graphs_by_cohort": dict(accepted_cohort_counts),
        "oracle_vs_dft_calibration": calibration_stats,
    }
    (args.output_dir / "consolidation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in (
        "source_train_count",
        "total_result_jsons_scanned",
        "newly_recovered_unique_graphs_beyond_e17",
        "consolidated_quantum_dft_counts",
        "consolidated_quantum_B_admitted_counts",
        "oe62_master_train_B_total",
        "oracle_vs_dft_calibration",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
