"""Zero-Leakage `Best-of-M` (`M = 1, 2, 4, 8, 16`) Benchmark on Official PropMolFlow (6 Models).

Evaluates all 6 official `PropMolFlow` `in-distribution` QM9 checkpoints across candidate budgets
`M in {1, 2, 4, 8, 16}` using the EXACT SAME zero-leakage `train_rQM_v0` `Uni-Mol` 3D surrogate
(`QM9TrainSplitSurrogateOracle`, `0` overlap with `gvp_rQM_v0`) and `GVPBenchmarkSuite` protocol
applied to `Uni-Loom`.

Both tracks are audited across `M = 1, 2, 4, 8, 16`:
  1. `native_graph`: PropMolFlow's co-generated discrete bond/charge RDKit molecular graph,
     prioritizing clean connected neutral closed-shell candidates and ranking by
     `|y_surr(mol) - target| / prop_std` (zero GVP access during selection).
  2. `coord_reconciled`: pure 3D coordinate perception + Track B `ValenceReconciler`
     (`0.0000 A` heavy-atom RMSD) + `QM9TrainSplitSurrogateOracle` ranking, enabling
     both apples-to-apples Pareto comparison and Generator-Agnostic Weaving validation.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import types
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
import torch

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROPMOLFLOW_DEFAULT = Path("/sjtu-combustion/wjw/chensq/PropMolFlow")
MOLGEN_COMPAT_DEFAULT = Path("/sjtu-combustion/wjw/chensq/mol_gen/src/compat")

for candidate in (
    ROOT / "src" / "compat",
    ROOT / "src",
    ROOT,
    MOLGEN_COMPAT_DEFAULT,
    PROPMOLFLOW_DEFAULT,
):
    cstr = str(candidate)
    if candidate.exists() and cstr not in sys.path:
        sys.path.insert(0, cstr)


def _install_pytorch_lightning_shim() -> None:
    try:
        import pytorch_lightning  # noqa: F401
        return
    except ImportError:
        pass
    pl = types.ModuleType("pytorch_lightning")

    class _LightningModule(torch.nn.Module):
        def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
            pass

        def log(self, *args: Any, **kwargs: Any) -> None:
            pass

        def log_dict(self, *args: Any, **kwargs: Any) -> None:
            pass

    pl.LightningModule = _LightningModule
    sys.modules["pytorch_lightning"] = pl


_install_pytorch_lightning_shim()

from propmolflow.models.flowmol import FlowMol  # noqa: E402

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402
from scripts.loom_qm9_benchmark import (  # noqa: E402
    ALLOWED_QM9_ELEMENTS,
    GVPBenchmarkSuite,
    PROPS,
    load_disjoint_qm9_splits,
)
from scripts.loom_qm9_best_of_m import (  # noqa: E402
    _min_pair_dist,
    select_best_of_m_and_score,
)
from scripts.loom_qm9_k1_pipeline import (  # noqa: E402
    QM9TrainSplitSurrogateOracle,
    _records_to_gen_targets,
    _target_ha_to_gen,
)
from scripts.loom_qm9_propmolflow_audit import PROPMOLFLOW_OFFICIAL_SPECS  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _find_checkpoint(propmolflow_root: Path, prop: str) -> Path:
    ckpt_dir = propmolflow_root / "checkpoints" / "in-distribution" / prop
    matches = sorted(glob.glob(str(ckpt_dir / "*.ckpt")))
    if not matches:
        raise FileNotFoundError(f"Missing official PropMolFlow checkpoint under {ckpt_dir}")
    return Path(matches[-1])


def _load_flowmol(ckpt_path: Path, device: torch.device) -> FlowMol:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    model = FlowMol(**hparams)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model


def run_propmolflow_best_of_m_property(
    prop: str,
    prop_idx: int,
    splits: Dict[str, Any],
    surrogate: QM9TrainSplitSurrogateOracle,
    reconciler: ValenceReconciler,
    gvp_suite: GVPBenchmarkSuite,
    train_smiles: set[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = torch.device(args.device)
    spec = PROPMOLFLOW_OFFICIAL_SPECS[prop]
    ckpt_path = _find_checkpoint(args.propmolflow_root, prop)
    model = _load_flowmol(ckpt_path, device)
    handle_method = str(getattr(model, "properties_handle_method", spec["handle_method"]))
    norm_file = str(args.propmolflow_root / "data" / "qm9" / "train_data_property_normalization.pt")

    samp_dir = args.propmolflow_root / "sampling_input" / "in_distribution_sampling"
    eval_n = int(args.eval_n)
    m_budgets = [int(x) for x in args.m_budgets]
    m_max = int(max(m_budgets))
    max_bs = int(args.max_batch_size)
    n_timesteps = int(args.n_timesteps)

    targets_ha = np.load(samp_dir / f"train_half_sampled_values_{prop}.npy")[:eval_n].astype(np.float64)
    natoms_targets = np.load(samp_dir / f"train_half_sampled_no_atoms_{prop}.npy")[:eval_n].astype(np.int32)
    targets_gen = _target_ha_to_gen(prop, targets_ha)
    train_y = _records_to_gen_targets(splits["train"], prop_idx, prop)
    prop_std = float(np.std(train_y))

    print(
        f"\n=== [PropMolFlow Best-of-M] {prop} (ckpt={ckpt_path.name}, M_max={m_max}, bs={max_bs}, steps={n_timesteps}) ===",
        flush=True,
    )

    # Warmup call before timing
    with torch.no_grad():
        _ = model.sample_random_sizes(
            min(32, eval_n),
            device=device,
            n_timesteps=10,
            property_name=prop,
            normalization_file_path=norm_file,
            training_mode=False,
            properties_handle_method=handle_method,
            multilple_values_to_one_property=targets_ha[: min(32, eval_n)].tolist(),
            number_of_atoms=natoms_targets[: min(32, eval_n)].tolist(),
        )
    _sync(device)

    # Round-major generation: `m_max` independent passes of `eval_n` targets
    sampled_by_round: List[List[Any]] = []
    round_gen_walls: List[float] = []
    for m_i in range(m_max):
        r_seed = args.seed + prop_idx * 100003 + m_i * 1009
        torch.manual_seed(r_seed)
        np.random.seed(r_seed % (2**31 - 1))
        random.seed(r_seed)
        _sync(device)
        t_r0 = time.perf_counter()
        r_mols: List[Any] = []
        with torch.no_grad():
            for start in range(0, eval_n, max_bs):
                end = min(start + max_bs, eval_n)
                b_vals = targets_ha[start:end].tolist()
                b_nat = [int(x) for x in natoms_targets[start:end]]
                b_out = model.sample_random_sizes(
                    end - start,
                    device=device,
                    n_timesteps=n_timesteps,
                    property_name=prop,
                    normalization_file_path=norm_file,
                    training_mode=False,
                    properties_handle_method=handle_method,
                    multilple_values_to_one_property=b_vals,
                    number_of_atoms=b_nat,
                )
                r_mols.extend(b_out)
        _sync(device)
        r_wall = time.perf_counter() - t_r0
        round_gen_walls.append(r_wall)
        sampled_by_round.append(r_mols)
        if (m_i + 1) in (1, 4, 8, 12, m_max):
            print(
                f"[PMF {prop} Round {m_i+1:02d}/{m_max:02d}] round_wall={r_wall:.2f}s "
                f"({eval_n / max(r_wall, 1e-6):.2f} mol/s)",
                flush=True,
            )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 1. Native graph sanitization pass (`eval_n x m_max`)
    t_nat0 = time.perf_counter()
    nat_mols_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(eval_n)]
    nat_legal_grid = np.zeros((eval_n, m_max), dtype=bool)
    nat_exotic_grid = np.zeros((eval_n, m_max), dtype=bool)
    nat_clean_grid = np.zeros((eval_n, m_max), dtype=bool)
    nat_natoms_grid = np.zeros((eval_n, m_max), dtype=np.int32)
    dummy_ppl_grid = np.ones((eval_n, m_max), dtype=np.float32)
    flat_nat_mols: List[Optional[Chem.Mol]] = []

    for i in range(eval_n):
        for m_i in range(m_max):
            sm = sampled_by_round[m_i][i]
            pos_np = sm.positions.detach().cpu().numpy().astype(np.float32)
            clash_free = _min_pair_dist(pos_np) >= 0.72
            mol_out: Optional[Chem.Mol] = None
            if sm.rdkit_mol is not None:
                try:
                    cand = Chem.Mol(sm.rdkit_mol)
                    Chem.SanitizeMol(cand)
                    mol_out = cand
                except Exception:
                    mol_out = None
            if mol_out is not None:
                nat_mols_grid[i][m_i] = mol_out
                nat_legal_grid[i, m_i] = True
                clean_q = bool(is_clean_organic_charge_state(mol_out))
                net_q = int(Chem.GetFormalCharge(mol_out))
                conn = len(Chem.GetMolFrags(mol_out)) == 1
                nat_exotic_grid[i, m_i] = not clean_q
                nat_natoms_grid[i, m_i] = int(mol_out.GetNumAtoms())
                nat_clean_grid[i, m_i] = clean_q and (net_q == 0) and conn and clash_free
                flat_nat_mols.append(mol_out)
            else:
                flat_nat_mols.append(None)
    native_sanitize_wall = time.perf_counter() - t_nat0

    _sync(device)
    t_nsur0 = time.perf_counter()
    nat_preds_all, _, nat_ok_all, _ = surrogate.evaluate_candidates(flat_nat_mols, batch_size=256)
    _sync(device)
    native_surr_wall = time.perf_counter() - t_nsur0
    nat_surr_grid = nat_preds_all[:, prop_idx].reshape(eval_n, m_max)
    nat_ok_grid = nat_ok_all.reshape(eval_n, m_max)

    # 2. Coordinate-only `ValenceReconciler` pass (`eval_n x m_max`)
    t_rec0 = time.perf_counter()
    rec_mols_grid: List[List[Optional[Chem.Mol]]] = [[None] * m_max for _ in range(eval_n)]
    rec_legal_grid = np.zeros((eval_n, m_max), dtype=bool)
    rec_exotic_grid = np.zeros((eval_n, m_max), dtype=bool)
    rec_clean_grid = np.zeros((eval_n, m_max), dtype=bool)
    rec_natoms_grid = np.zeros((eval_n, m_max), dtype=np.int32)
    flat_rec_mols: List[Optional[Chem.Mol]] = []

    for i in range(eval_n):
        for m_i in range(m_max):
            sm = sampled_by_round[m_i][i]
            pos_np = sm.positions.detach().cpu().numpy().astype(np.float32)
            clash_free = _min_pair_dist(pos_np) >= 0.72
            res = reconciler.reconcile_point_cloud(list(sm.atom_types), pos_np, potential_slot=None)
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
    coord_rec_wall = time.perf_counter() - t_rec0

    _sync(device)
    t_csur0 = time.perf_counter()
    rec_preds_all, _, rec_ok_all, _ = surrogate.evaluate_candidates(flat_rec_mols, batch_size=256)
    _sync(device)
    coord_surr_wall = time.perf_counter() - t_csur0
    rec_surr_grid = rec_preds_all[:, prop_idx].reshape(eval_n, m_max)
    rec_ok_grid = rec_ok_all.reshape(eval_n, m_max)

    # Evaluate each M budget on GVPBenchmarkSuite
    # Note: pass 2 * wall into select_best_of_m_and_score since that helper multiplies shared wall by 0.5
    budgets_out: Dict[str, Any] = {}
    for m_b in m_budgets:
        nat_res = select_best_of_m_and_score(
            prop=prop,
            prop_idx=prop_idx,
            prop_std=prop_std,
            targets_ha=targets_ha,
            targets_gen=targets_gen,
            natoms_targets=natoms_targets,
            m_budget=m_b,
            m_max=m_max,
            mode="target_only",
            track="native_graph",
            mols_grid=nat_mols_grid,
            legal_grid=nat_legal_grid,
            exotic_grid=nat_exotic_grid,
            clean_connected_grid=nat_clean_grid,
            natoms_grid=nat_natoms_grid,
            ppl_grid=dummy_ppl_grid,
            surr_preds_grid=nat_surr_grid,
            surr_ok_grid=nat_ok_grid,
            round_gen_walls=round_gen_walls,
            reconcile_wall_total=2.0 * native_sanitize_wall,
            surrogate_wall_total=2.0 * native_surr_wall,
            gvp_suite=gvp_suite,
            train_smiles=train_smiles,
        )
        rec_res = select_best_of_m_and_score(
            prop=prop,
            prop_idx=prop_idx,
            prop_std=prop_std,
            targets_ha=targets_ha,
            targets_gen=targets_gen,
            natoms_targets=natoms_targets,
            m_budget=m_b,
            m_max=m_max,
            mode="target_only",
            track="coord_reconciled",
            mols_grid=rec_mols_grid,
            legal_grid=rec_legal_grid,
            exotic_grid=rec_exotic_grid,
            clean_connected_grid=rec_clean_grid,
            natoms_grid=rec_natoms_grid,
            ppl_grid=dummy_ppl_grid,
            surr_preds_grid=rec_surr_grid,
            surr_ok_grid=rec_ok_grid,
            round_gen_walls=round_gen_walls,
            reconcile_wall_total=2.0 * coord_rec_wall,
            surrogate_wall_total=2.0 * coord_surr_wall,
            gvp_suite=gvp_suite,
            train_smiles=train_smiles,
        )
        budgets_out[f"M{m_b}"] = {
            "native_graph": nat_res,
            "coord_reconciled": rec_res,
        }

    m1_n = budgets_out["M1"]["native_graph"]
    m8_n = budgets_out.get("M8", budgets_out[f"M{m_max}"])["native_graph"]
    m16_n = budgets_out[f"M{m_max}"]["native_graph"]
    print(
        f"[PMF Best-of-M {prop} DONE] "
        f"M=1 NativeMAE={m1_n['gvp']['mae']} ({m1_n['effective_output_mol_per_sec']} mol/s) -> "
        f"M=8 NativeMAE={m8_n['gvp']['mae']} ({m8_n['effective_output_mol_per_sec']} mol/s) -> "
        f"M={m_max} NativeMAE={m16_n['gvp']['mae']} (r={m16_n['gvp']['pearson_r']}, Leg={m16_n['legal_pct']}%, ExoQ={m16_n['exotic_charge_pct']}%, {m16_n['effective_output_mol_per_sec']} mol/s)",
        flush=True,
    )

    payload = {
        "property": prop,
        "checkpoint": ckpt_path.name,
        "published_mae": spec["published_mae"],
        "n_timesteps": n_timesteps,
        "max_batch_size": max_bs,
        "m_max_pool": m_max,
        "gen_wall_total_seconds": round(float(sum(round_gen_walls)), 3),
        "native_sanitize_wall_seconds": round(native_sanitize_wall, 3),
        "native_surrogate_wall_seconds": round(native_surr_wall, 3),
        "coord_reconcile_wall_seconds": round(coord_rec_wall, 3),
        "coord_surrogate_wall_seconds": round(coord_surr_wall, 3),
        "budgets": budgets_out,
    }
    out_json = args.results_dir / f"qm9_pmf_best_of_m_{prop}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_worker_phase(args: argparse.Namespace) -> None:
    qm9_raw_dir = args.propmolflow_root / "data" / "qm9_raw"
    splits = load_disjoint_qm9_splits(args.lmdb_dir, qm9_raw_dir, dev_size=1500, holdout_size=1000, seed=20261115)
    assert int(splits["audit"]["overlap_train_with_gvp_rQM_v0"]) == 0

    surrogate = QM9TrainSplitSurrogateOracle(args.surrogate_ckpt, device=args.device)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    gvp_suite = GVPBenchmarkSuite(args.propmolflow_root, device=args.device)

    train_smiles: set[str] = set()
    for r in splits["train"]:
        res = reconciler.reconcile_point_cloud(r["symbols"], r["coords"], potential_slot=None)
        if res.smiles:
            m2d = Chem.MolFromSmiles(res.smiles)
            if m2d is not None:
                train_smiles.add(Chem.MolToSmiles(Chem.RemoveHs(m2d), isomericSmiles=False))

    if args.queue_dir is not None:
        args.queue_dir.mkdir(parents=True, exist_ok=True)

    for prop in args.props:
        if prop not in PROPS:
            raise ValueError(f"Unknown property '{prop}'")
        out_json = args.results_dir / f"qm9_pmf_best_of_m_{prop}.json"
        if out_json.exists() and not args.overwrite:
            print(f"[PMF Queue] {out_json} already exists; skipping {prop}.", flush=True)
            continue
        if args.queue_dir is not None:
            claim_dir = args.queue_dir / f"claim_{prop}"
            try:
                os.mkdir(claim_dir)
                (claim_dir / "owner.txt").write_text(f"pid={os.getpid()} device={args.device}\n")
                print(f"[PMF Queue] Claimed {prop} on {args.device} (pid={os.getpid()})", flush=True)
            except FileExistsError:
                print(f"[PMF Queue] {prop} already claimed by peer GPU worker; moving to next.", flush=True)
                continue
        prop_idx = PROPS.index(prop)
        run_propmolflow_best_of_m_property(
            prop, prop_idx, splits, surrogate, reconciler, gvp_suite, train_smiles, args
        )


def consolidate_phase(args: argparse.Namespace) -> None:
    per_prop: Dict[str, Any] = {}
    for p in PROPS:
        pp = args.results_dir / f"qm9_pmf_best_of_m_{p}.json"
        if pp.exists():
            per_prop[p] = json.loads(pp.read_text())
    payload = {
        "experiment": "QM9_PropMolFlow_6Models_ZeroLeakage_UniMol_BestOfM",
        "zero_gvp_overlap_verified": True,
        "surrogate_checkpoint": str(args.surrogate_ckpt),
        "m_budgets": list(args.m_budgets),
        "completed_properties": sorted(per_prop.keys()),
        "per_property": per_prop,
    }
    dst = args.results_dir / "qm9_propmolflow_best_of_m.json"
    dst.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[PMF Consolidate] Wrote {dst} ({len(per_prop)}/6 properties)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["run_worker", "consolidate"], default="run_worker")
    p.add_argument("--props", nargs="+", default=list(PROPS))
    p.add_argument("--queue-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--lmdb-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9_data"))
    p.add_argument("--propmolflow-root", type=Path, default=PROPMOLFLOW_DEFAULT)
    p.add_argument(
        "--surrogate-ckpt",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/runs/qm9_surrogate/qm9_train_unimol_surrogate.pt"),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results"),
    )
    p.add_argument("--m-budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--max-batch-size", type=int, default=128)
    p.add_argument("--n-timesteps", type=int, default=100)
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=20261116)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    if args.phase == "run_worker":
        run_worker_phase(args)
    elif args.phase == "consolidate":
        consolidate_phase(args)


if __name__ == "__main__":
    main()
