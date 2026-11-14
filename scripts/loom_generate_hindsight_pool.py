"""Stage 1 Tail-Debiased Hindsight Relabeling Pool Generator for `Uni-Loom`.

Generates 3D candidates across a tail-enriched continuous target grid, filters out
unseen formal charges, high-strain clashes (`PotentialSlot`), holdout Murcko scaffolds,
and duplicate molecular graphs, then relabels accepted structures with their ACTUAL
`RewardOracleSlot` (`Uni-Mol2 1.1B`) predictions (`B_homo`, `B_lumo`).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

from loom.generator.molecular_state import mol_to_clean_smiles
from loom.generator.octree_backend import trajectory_to_instance
from loom.oracles.unimol2_task import B_WEIGHT_SHA256
from loom.utils.dataset_io import write_precise_sdf
from loom.weaving import (
    LoomUni3DARGenerator,
    PotentialSlot,
    RewardOracleSlot,
    assert_not_audit_oracle,
    formal_charge_signature,
)


def murcko_scaffold(mol: Chem.Mol) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(mol), includeChirality=False) or "<ACYCLIC>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-train-sdf", type=Path, required=True)
    parser.add_argument("--source-splits-dir", type=Path, required=True)
    parser.add_argument("--exclude-sdfs", type=Path, nargs="*", default=[])
    parser.add_argument("--reward-oracle-dir", type=Path, required=True)
    parser.add_argument("--potential-backend", choices=["none", "dpa4", "hyff"], default="dpa4")
    parser.add_argument("--potential-weights", type=Path, default=None)
    parser.add_argument("--max-fmax-eV-A", type=float, default=3.5)
    parser.add_argument("--mode", choices=["homo", "lumo", "joint"], default="homo")
    parser.add_argument("--total-proposals", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cfg-scale", type=float, default=1.75)
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.seed in (24411, 24412):
        raise ValueError("Hindsight generation must NEVER use validation seeds 24411/24412!")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    gen, _ = LoomUni3DARGenerator.load_checkpoint(args.checkpoint, device=args.device, batch_size=args.batch_size)
    gen.eval()

    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )
    assert_not_audit_oracle(reward_slot)

    pot_slot = (
        PotentialSlot(backend=args.potential_backend, weights=str(args.potential_weights), device=args.device)
        if args.potential_backend != "none" and args.potential_weights is not None
        else None
    )

    seen_graphs = set()
    seen_charges = set()
    src_labels = []
    for m in Chem.SDMolSupplier(str(args.source_train_sdf), removeHs=False):
        if m is not None:
            seen_graphs.add(mol_to_clean_smiles(m))
            seen_charges.add(formal_charge_signature(m))
            h = float(m.GetProp("B_homo") if m.HasProp("B_homo") else m.GetProp("homo"))
            l = float(m.GetProp("B_lumo") if m.HasProp("B_lumo") else m.GetProp("lumo"))
            src_labels.append([h, l])
    src_arr = np.asarray(src_labels, dtype=np.float64)

    heldout_scaffolds = set()
    for sp in ("valid", "calibration", "test"):
        df = pd.read_csv(args.source_splits_dir / f"{sp}.csv")
        heldout_scaffolds.update(df["scaffold"].dropna().astype(str).tolist())
    for ex_path in args.exclude_sdfs:
        if ex_path.exists():
            for m in Chem.SDMolSupplier(str(ex_path), removeHs=False):
                if m is not None:
                    heldout_scaffolds.add(murcko_scaffold(m))

    # Build tail-enriched request distribution (50% from tails [P03..P22] U [P78..P97], 50% uniform [P05..P95])
    rng = np.random.default_rng(args.seed)
    h_q = np.percentile(src_arr[:, 0], [3, 22, 78, 97, 5, 95])
    l_q = np.percentile(src_arr[:, 1], [3, 22, 78, 97, 5, 95])

    def _draw_tail_1d(q_vec: np.ndarray, size: int) -> np.ndarray:
        out = np.empty(size, dtype=np.float32)
        for i in range(size):
            u = rng.random()
            if u < 0.25:
                out[i] = rng.uniform(q_vec[0], q_vec[1])
            elif u < 0.50:
                out[i] = rng.uniform(q_vec[2], q_vec[3])
            else:
                out[i] = rng.uniform(q_vec[4], q_vec[5])
        return out

    counts: Counter = Counter()
    admitted_mols: List[Chem.Mol] = []
    allowed_elems = tuple(gen.args.allow_atoms.split(","))
    device = torch.device(args.device)
    t0 = time.perf_counter()

    for start in range(0, args.total_proposals, args.batch_size):
        cur_b = min(args.batch_size, args.total_proposals - start)
        t_req = np.zeros((cur_b, 2), dtype=np.float32)
        t_req[:, 0] = _draw_tail_1d(h_q, cur_b)
        t_req[:, 1] = _draw_tail_1d(l_q, cur_b)
        # Ensure physical HOMO < LUMO - 0.8 eV when joint
        bad_gap = t_req[:, 0] > (t_req[:, 1] - 0.8)
        if np.any(bad_gap):
            idx_src = rng.integers(0, len(src_arr), size=int(bad_gap.sum()))
            t_req[bad_gap] = src_arr[idx_src] + rng.normal(0.0, 0.20, size=(int(bad_gap.sum()), 2))

        mask_np = np.zeros((cur_b, 2), dtype=bool)
        if args.mode == "homo":
            mask_np[:, 0] = True
        elif args.mode == "lumo":
            mask_np[:, 1] = True
        else:
            for i in range(cur_b):
                r = rng.random()
                if r < 0.35:
                    mask_np[i, 0] = True
                elif r < 0.70:
                    mask_np[i, 1] = True
                else:
                    mask_np[i, :] = True

        trajs = gen.sample(
            gen.normalize_targets(t_req),
            torch.as_tensor(mask_np, dtype=torch.bool, device=device),
            cfg_scale=args.cfg_scale,
            xyz_cfg_scale=1.0,
            seed=args.seed * 1009 + start,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
        )

        batch_candidates: List[Tuple[Chem.Mol, str, np.ndarray]] = []
        for i, tr in enumerate(trajs):
            counts["attempted"] += 1
            try:
                inst = trajectory_to_instance(tr, element_table=allowed_elems)
                mol = inst.graph
            except ValueError:
                counts["invalid_3d_graph"] += 1
                continue

            if formal_charge_signature(mol) not in seen_charges:
                counts["unseen_formal_charge"] += 1
                continue
            scaf = murcko_scaffold(mol)
            if scaf in heldout_scaffolds:
                counts["heldout_scaffold"] += 1
                continue
            if inst.smiles in seen_graphs:
                counts["duplicate_or_source_graph"] += 1
                continue
            if reward_slot.check_support(mol) is not None:
                counts["unsupported_by_oracle"] += 1
                continue

            seen_graphs.add(inst.smiles)
            batch_candidates.append((mol, inst.smiles, t_req[i]))

        if not batch_candidates:
            continue

        c_mols = [c[0] for c in batch_candidates]
        keep_phys = np.ones(len(c_mols), dtype=bool)
        fmax_vals = np.zeros(len(c_mols), dtype=np.float64)
        if pot_slot is not None:
            p_out = pot_slot.evaluate_molecules(c_mols, chunk_size=32)
            fmax_vals = p_out["f_max_eV_A"]
            keep_phys = np.isfinite(fmax_vals) & (fmax_vals <= args.max_fmax_eV_A)
            counts["high_strain_rejected"] += int((~keep_phys).sum())

        phys_mols = [c_mols[i] for i in range(len(c_mols)) if keep_phys[i]]
        phys_reqs = [batch_candidates[i][2] for i in range(len(c_mols)) if keep_phys[i]]
        phys_fmax = [float(fmax_vals[i]) for i in range(len(c_mols)) if keep_phys[i]]
        if not phys_mols:
            continue

        preds, _, ev_ok, _ = reward_slot.evaluate_candidates(phys_mols, batch_size=32)
        for i, m in enumerate(phys_mols):
            if not ev_ok[i]:
                counts["nonfinite_oracle"] += 1
                continue
            bh, bl = float(preds[i, 0]), float(preds[i, 1])
            om = Chem.Mol(m)
            om.SetProp("OriginType", "loom_hindsight")
            om.SetProp("B_homo", f"{bh:.8f}")
            om.SetProp("B_lumo", f"{bl:.8f}")
            om.SetProp("homo", f"{bh:.8f}")
            om.SetProp("lumo", f"{bl:.8f}")
            om.SetProp("OriginalRequest_eV", json.dumps(phys_reqs[i].tolist()))
            om.SetProp("ZeroStepFmax_eV_A", f"{phys_fmax[i]:.4f}")
            admitted_mols.append(om)
            counts["admitted"] += 1

    sdf_out = args.output_dir / "hindsight_train.sdf"
    write_precise_sdf(sdf_out, admitted_mols)
    summary = {
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "cfg_scale": args.cfg_scale,
        "max_fmax_eV_A": args.max_fmax_eV_A,
        "counts": dict(counts),
        "admitted_unique_molecules": len(admitted_mols),
        "wall_seconds": time.perf_counter() - t0,
        "sdf_path": str(sdf_out),
    }
    (args.output_dir / "hindsight_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
