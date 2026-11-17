"""Direct official PropMolFlow 6-model QM9 audit: chemistry legality, exotic formal charges, inference & training speed.

Loads PropMolFlow's 6 official `in-distribution` QM9 checkpoints, samples N=1,000 molecules
per property under the official sampling protocol on one A800 GPU, verifies GVPRegressor MAE,
audits both native discrete-bond/charge legality & exotic formal charges and 3D coordinate-only
bond-perception legality (`ValenceReconciler`), and benchmarks exact single-A800 training throughput.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
import time
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch
from torch.nn.functional import one_hot

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
    """Install lightweight PyTorch Lightning shim if full `pytorch_lightning` package is absent."""
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

import dgl  # noqa: E402
from propmolflow.data_processing.priors import coupled_node_prior, edge_prior  # noqa: E402
from propmolflow.models.flowmol import FlowMol  # noqa: E402

from loom.chemistry.valence_reconciler import ValenceReconciler, is_clean_organic_charge_state  # noqa: E402
from scripts.loom_qm9_benchmark import (  # noqa: E402
    ALLOWED_QM9_ELEMENTS,
    GVPBenchmarkSuite,
    HA_TO_MEV,
    PROPS,
    load_disjoint_qm9_splits,
)

# Official `in_distribution_best_combo.csv` metadata from PropMolFlow repository
PROPMOLFLOW_OFFICIAL_SPECS: Dict[str, Dict[str, Any]] = {
    "alpha": {
        "handle_method": "sum",
        "epochs": 1846,
        "steps": 721785,
        "published_mae": 1.3115,
    },
    "gap": {
        "handle_method": "concatenate_sum",
        "epochs": 1868,
        "steps": 730387,
        "published_mae": 390.98,
    },
    "homo": {
        "handle_method": "concatenate",
        "epochs": 1928,
        "steps": 753691,
        "published_mae": 253.99,
    },
    "lumo": {
        "handle_method": "concatenate_sum",
        "epochs": 1931,
        "steps": 754942,
        "published_mae": 314.52,
    },
    "mu": {
        "handle_method": "multiply",
        "epochs": 1869,
        "steps": 730778,
        "published_mae": 0.6202,
    },
    "cv": {
        "handle_method": "sum",
        "epochs": 1736,
        "steps": 678541,
        "published_mae": 0.6259,
    },
}


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


def _build_flowmol_training_graphs(
    records: Sequence[Dict[str, Any]],
    atom_type_map: Sequence[str],
    prior_config: Dict[str, Any],
    norm_params: Dict[str, Any],
    prop_col_idx: int,
    n_mols: int,
) -> List[Tuple[Any, torch.Tensor]]:
    """Construct exact `MoleculeDataset`-compatible DGL graphs from `train_rQM_v0` conformers."""
    atom_to_idx = {sym: i for i, sym in enumerate(atom_type_map)}
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    graphs_and_props: List[Tuple[Any, torch.Tensor]] = []
    mean_v = float(norm_params["mean"][prop_col_idx].item())
    std_v = float(norm_params["std"][prop_col_idx].item())

    for rec in records:
        res = reconciler.reconcile_point_cloud(rec["symbols"], rec["coords"], potential_slot=None)
        mol = res.mol if (res.reconciled_legal and res.mol is not None) else res.raw_mol
        if mol is None:
            continue
        syms = [a.GetSymbol() for a in mol.GetAtoms()]
        if any(s not in atom_to_idx for s in syms):
            continue
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        pos = pos - pos.mean(dim=0, keepdim=True)
        a_idx = torch.tensor([atom_to_idx[s] for s in syms], dtype=torch.long)
        a_oh = one_hot(a_idx, num_classes=len(atom_type_map)).float()
        q_vals = torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()], dtype=torch.long)
        if q_vals.min() < -2 or q_vals.max() > 3:
            continue
        c_oh = one_hot(q_vals + 2, num_classes=6).float()

        n_atoms = mol.GetNumAtoms()
        adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int64)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            if bt == Chem.rdchem.BondType.SINGLE:
                b_cls = 1
            elif bt == Chem.rdchem.BondType.DOUBLE:
                b_cls = 2
            elif bt == Chem.rdchem.BondType.TRIPLE:
                b_cls = 3
            elif bt == Chem.rdchem.BondType.AROMATIC:
                b_cls = 4
            else:
                b_cls = 0
            adj[i, j] = b_cls
            adj[j, i] = b_cls

        upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
        upper_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]
        lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))
        edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
        edge_labels = torch.cat((upper_labels, upper_labels))
        e_oh = one_hot(edge_labels, num_classes=5).float()

        g = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)
        g.edata["e_1_true"] = e_oh
        g.ndata["x_1_true"] = pos
        g.ndata["a_1_true"] = a_oh
        g.ndata["c_1_true"] = c_oh

        dst_dict = {"x": pos, "a": a_oh, "c": c_oh}
        prior_nodes = coupled_node_prior(dst_dict=dst_dict, prior_config=prior_config)
        for feat, val in prior_nodes.items():
            g.ndata[f"{feat}_0"] = val

        upper_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        upper_mask[: upper_edge_idxs.shape[1]] = True
        g.edata["e_0"] = edge_prior(upper_mask, prior_config["e"])

        prop_val = (float(rec["props_ha"][prop_col_idx]) - mean_v) / max(std_v, 1e-8)
        graphs_and_props.append((g, torch.tensor(prop_val, dtype=torch.float32)))
        if len(graphs_and_props) >= n_mols:
            break

    return graphs_and_props


def _benchmark_training_speed(
    ckpt_path: Path,
    train_records: Sequence[Dict[str, Any]],
    propmolflow_root: Path,
    device: torch.device,
    batch_size: int = 128,
    n_warmup: int = 5,
    n_timed: int = 20,
) -> Dict[str, Any]:
    """Measure exact uncontended single-A800 forward+backward+AdamW training throughput for PropMolFlow."""
    model = _load_flowmol(ckpt_path, device)
    model.train()
    norm_params = torch.load(
        propmolflow_root / "data" / "qm9" / "train_data_property_normalization.pt",
        map_location="cpu",
        weights_only=False,
    )
    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # `alpha` index in PropertyRegressor/FlowMol PROPERTY_MAP is 0
    from propmolflow.models.ctmc_vector_field import PROPERTY_MAP

    graphs_props = _build_flowmol_training_graphs(
        train_records,
        atom_type_map=model.atom_type_map,
        prior_config=model.prior_config,
        norm_params=norm_params,
        prop_col_idx=int(PROPERTY_MAP["alpha"]),
        n_mols=batch_size * 4,
    )
    if len(graphs_props) < batch_size:
        raise RuntimeError(f"Insufficient training graphs constructed for PropMolFlow timing: {len(graphs_props)}")

    batches: List[Tuple[Any, torch.Tensor]] = []
    for b_idx in range(4):
        sub = graphs_props[b_idx * batch_size : (b_idx + 1) * batch_size]
        if len(sub) < batch_size:
            sub = graphs_props[:batch_size]
        bg = dgl.batch([item[0] for item in sub]).to(device)
        prop_1d = torch.stack([item[1] for item in sub]).to(device).view(-1)
        batches.append((bg, prop_1d))

    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    loss_weights = dict(model.total_loss_weights)
    feat_order = list(model.canonical_feat_order)

    total_steps = n_warmup + n_timed
    t_start = 0.0
    for step_i in range(total_steps):
        if step_i == n_warmup:
            _sync(device)
            t_start = time.perf_counter()
        bg, prop_1d = batches[step_i % len(batches)]
        # `FlowMol.forward` mutates `g.prop = g.prop.unsqueeze(-1)` in place.
        bg.prop = prop_1d.clone()
        optimizer.zero_grad(set_to_none=True)
        losses = model(bg)
        total_loss = torch.zeros((), device=device)
        for feat in feat_order:
            total_loss = total_loss + float(loss_weights[feat]) * losses[feat]
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

    _sync(device)
    elapsed = time.perf_counter() - t_start
    sec_per_step = elapsed / float(n_timed)
    steps_per_sec = float(n_timed) / max(elapsed, 1e-9)
    mol_per_sec = float(batch_size) * steps_per_sec

    total_official_steps = sum(int(v["steps"]) for v in PROPMOLFLOW_OFFICIAL_SPECS.values())
    mean_official_steps = total_official_steps / float(len(PROPMOLFLOW_OFFICIAL_SPECS))

    del model, optimizer, batches
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "batch_size": batch_size,
        "n_warmup_steps": n_warmup,
        "n_timed_steps": n_timed,
        "total_params": n_total_params,
        "trainable_params": n_trainable_params,
        "sec_per_step": round(sec_per_step, 5),
        "steps_per_sec": round(steps_per_sec, 3),
        "train_mol_per_sec": round(mol_per_sec, 2),
        "official_mean_steps_per_model": round(mean_official_steps, 1),
        "official_total_steps_6_models": total_official_steps,
        "estimated_a800_hours_per_model": round((mean_official_steps * sec_per_step) / 3600.0, 2),
        "estimated_a800_hours_6_models": round((total_official_steps * sec_per_step) / 3600.0, 2),
    }


def _audit_single_property(
    prop: str,
    propmolflow_root: Path,
    gvp_suite: GVPBenchmarkSuite,
    reconciler: ValenceReconciler,
    device: torch.device,
    eval_n: int,
    max_batch_size: int,
    n_timesteps: int,
) -> Dict[str, Any]:
    spec = PROPMOLFLOW_OFFICIAL_SPECS[prop]
    ckpt_path = _find_checkpoint(propmolflow_root, prop)
    model = _load_flowmol(ckpt_path, device)
    handle_method = str(getattr(model, "properties_handle_method", spec["handle_method"]))
    norm_file = str(propmolflow_root / "data" / "qm9" / "train_data_property_normalization.pt")

    samp_dir = propmolflow_root / "sampling_input" / "in_distribution_sampling"
    targets_raw = np.load(samp_dir / f"train_half_sampled_values_{prop}.npy")[:eval_n]
    natoms_raw = np.load(samp_dir / f"train_half_sampled_no_atoms_{prop}.npy")[:eval_n]

    # Note: In PropMolFlow CSV/numpy files, gap/homo/lumo are stored in Hartree (same unit used by `GVPBenchmarkSuite.evaluate_mols`).
    # Warmup sampling call before timing
    with torch.no_grad():
        _ = model.sample_random_sizes(
            min(32, eval_n),
            device=device,
            n_timesteps=10,
            property_name=prop,
            normalization_file_path=norm_file,
            training_mode=False,
            properties_handle_method=handle_method,
            multilple_values_to_one_property=targets_raw[: min(32, eval_n)].tolist(),
            number_of_atoms=natoms_raw[: min(32, eval_n)].tolist(),
        )
    _sync(device)

    sampled_mols = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, eval_n, max_batch_size):
            end = min(start + max_batch_size, eval_n)
            batch_vals = targets_raw[start:end].tolist()
            batch_nat = [int(x) for x in natoms_raw[start:end]]
            batch_out = model.sample_random_sizes(
                end - start,
                device=device,
                n_timesteps=n_timesteps,
                property_name=prop,
                normalization_file_path=norm_file,
                training_mode=False,
                properties_handle_method=handle_method,
                multilple_values_to_one_property=batch_vals,
                number_of_atoms=batch_nat,
            )
            sampled_mols.extend(batch_out)
    _sync(device)
    sampling_wall_s = time.perf_counter() - t0
    peak_vram_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0

    # 1. Native discrete bond/charge graph audit
    native_sanitized_mols: List[Optional[Chem.Mol]] = []
    native_conn_neutral_mols: List[Optional[Chem.Mol]] = []
    native_sanitized_count = 0
    native_connected_neutral_count = 0
    native_disconnected_count = 0
    native_nonzero_charge_count = 0
    native_exotic_total_of_sanitized = 0
    native_exotic_zwitterion_neutral = 0
    native_smiles: set[str] = set()

    # 2. Pure 3D coordinate-only perception audit (`ValenceReconciler`, stripping predicted bonds & charges)
    coord_raw_mols: List[Optional[Chem.Mol]] = []
    coord_raw_legal = 0
    coord_raw_exotic = 0
    coord_rec_mols: List[Optional[Chem.Mol]] = []
    coord_rec_legal = 0
    coord_rec_exotic = 0
    coord_max_rmsd = 0.0

    for sm in sampled_mols:
        mol_out: Optional[Chem.Mol] = None
        if sm.rdkit_mol is not None:
            try:
                cand = Chem.Mol(sm.rdkit_mol)
                Chem.SanitizeMol(cand)
                mol_out = cand
            except Exception:
                mol_out = None

        if mol_out is not None:
            native_sanitized_count += 1
            native_sanitized_mols.append(mol_out)
            try:
                native_smiles.add(Chem.MolToSmiles(Chem.RemoveHs(mol_out), isomericSmiles=False))
            except Exception:
                pass
            n_frags = len(Chem.GetMolFrags(mol_out))
            net_q = int(Chem.GetFormalCharge(mol_out))
            clean_q = bool(is_clean_organic_charge_state(mol_out))
            if n_frags > 1:
                native_disconnected_count += 1
            if net_q != 0:
                native_nonzero_charge_count += 1
            if not clean_q:
                native_exotic_total_of_sanitized += 1
                if net_q == 0:
                    native_exotic_zwitterion_neutral += 1
            if n_frags == 1 and net_q == 0:
                native_connected_neutral_count += 1
                native_conn_neutral_mols.append(mol_out)
            else:
                native_conn_neutral_mols.append(None)
        else:
            native_sanitized_mols.append(None)
            native_conn_neutral_mols.append(None)

        pos_np = sm.positions.detach().cpu().numpy().astype(np.float32)
        res = reconciler.reconcile_point_cloud(list(sm.atom_types), pos_np, potential_slot=None)
        if res.raw_legal and res.raw_mol is not None:
            coord_raw_legal += 1
            coord_raw_mols.append(res.raw_mol)
            if res.raw_exotic_charge:
                coord_raw_exotic += 1
        else:
            coord_raw_mols.append(None)

        if res.reconciled_legal and res.mol is not None:
            coord_rec_legal += 1
            coord_rec_mols.append(res.mol)
            if res.reconciled_exotic_charge:
                coord_rec_exotic += 1
            coord_max_rmsd = max(coord_max_rmsd, float(res.heavy_rmsd_A))
        else:
            coord_rec_mols.append(None)

    gvp_san = gvp_suite.evaluate_mols(prop, native_sanitized_mols, targets_raw)
    gvp_conn = gvp_suite.evaluate_mols(prop, native_conn_neutral_mols, targets_raw)
    gvp_coord_raw = gvp_suite.evaluate_mols(prop, coord_raw_mols, targets_raw)
    gvp_coord_rec = gvp_suite.evaluate_mols(prop, coord_rec_mols, targets_raw)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    res_dict = {
        "checkpoint": ckpt_path.name,
        "properties_handle_method": handle_method,
        "published_epochs": spec["epochs"],
        "published_steps": spec["steps"],
        "published_mae": spec["published_mae"],
        "n_requested": eval_n,
        "n_timesteps": n_timesteps,
        "max_batch_size": max_batch_size,
        "sampling_wall_seconds": round(sampling_wall_s, 3),
        "sampling_mol_per_sec": round(float(eval_n) / max(sampling_wall_s, 1e-6), 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "native_graph": {
            "sanitized_count": native_sanitized_count,
            "sanitized_pct": round(100.0 * native_sanitized_count / eval_n, 2),
            "connected_neutral_count": native_connected_neutral_count,
            "connected_neutral_pct": round(100.0 * native_connected_neutral_count / eval_n, 2),
            "disconnected_count": native_disconnected_count,
            "nonzero_net_charge_count": native_nonzero_charge_count,
            "exotic_charge_total_count": native_exotic_total_of_sanitized,
            "exotic_charge_pct_of_sanitized": round(
                100.0 * native_exotic_total_of_sanitized / max(native_sanitized_count, 1), 2
            ),
            "exotic_zwitterion_neutral_count": native_exotic_zwitterion_neutral,
            "unique_smiles": len(native_smiles),
            "uniqueness_pct_of_sanitized": round(
                100.0 * len(native_smiles) / max(native_sanitized_count, 1), 2
            ),
            "gvp_sanitized": {
                "n_evaluable": gvp_san["n_evaluable"],
                "mae": round(gvp_san["mae"], 4),
                "rmse": round(gvp_san["rmse"], 4),
                "slope": round(gvp_san["slope"], 4),
                "pearson_r": round(gvp_san["pearson_r"], 4),
            },
            "gvp_connected_neutral": {
                "n_evaluable": gvp_conn["n_evaluable"],
                "mae": round(gvp_conn["mae"], 4),
                "rmse": round(gvp_conn["rmse"], 4),
                "slope": round(gvp_conn["slope"], 4),
                "pearson_r": round(gvp_conn["pearson_r"], 4),
            },
        },
        "coord_only_perception": {
            "raw_legal_count": coord_raw_legal,
            "raw_legal_pct": round(100.0 * coord_raw_legal / eval_n, 2),
            "raw_exotic_charge_count": coord_raw_exotic,
            "raw_exotic_charge_pct_of_legal": round(100.0 * coord_raw_exotic / max(coord_raw_legal, 1), 2),
            "raw_mae": round(gvp_coord_raw["mae"], 4),
            "raw_pearson_r": round(gvp_coord_raw["pearson_r"], 4),
            "reconciled_legal_count": coord_rec_legal,
            "reconciled_legal_pct": round(100.0 * coord_rec_legal / eval_n, 2),
            "reconciled_exotic_charge_count": coord_rec_exotic,
            "reconciled_exotic_charge_pct_of_legal": round(100.0 * coord_rec_exotic / max(coord_rec_legal, 1), 2),
            "max_heavy_rmsd_A": round(coord_max_rmsd, 6),
            "reconciled_mae": round(gvp_coord_rec["mae"], 4),
            "reconciled_pearson_r": round(gvp_coord_rec["pearson_r"], 4),
        },
    }
    print(
        f"[PropMolFlow {prop:>5}] speed={res_dict['sampling_mol_per_sec']:.2f} mol/s | "
        f"NativeSan={res_dict['native_graph']['sanitized_pct']:.1f}% "
        f"NativeConnNeut={res_dict['native_graph']['connected_neutral_pct']:.1f}% "
        f"NativeExoticQ={res_dict['native_graph']['exotic_charge_pct_of_sanitized']:.2f}% "
        f"GVP_MAE={res_dict['native_graph']['gvp_sanitized']['mae']:.4f} (pub {spec['published_mae']}) | "
        f"CoordRaw={res_dict['coord_only_perception']['raw_legal_pct']:.1f}% "
        f"CoordExoticQ={res_dict['coord_only_perception']['raw_exotic_charge_pct_of_legal']:.2f}%",
        flush=True,
    )
    return res_dict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmdb-dir", type=Path, default=Path("/sjtu-combustion/wjw/chensq/uni3dar_grpo/qm9_data"))
    p.add_argument("--propmolflow-root", type=Path, default=PROPMOLFLOW_DEFAULT)
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/sjtu-combustion/wjw/chensq/loom_oe62_20261115/results/qm9_propmolflow_audit.json"),
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--max-batch-size", type=int, default=128)
    p.add_argument("--n-timesteps", type=int, default=100)
    p.add_argument("--train-bench-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261115)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)

    splits = load_disjoint_qm9_splits(args.lmdb_dir, args.propmolflow_root / "data" / "qm9_raw", seed=args.seed)
    reconciler = ValenceReconciler(allowed_elements=ALLOWED_QM9_ELEMENTS)
    gvp_suite = GVPBenchmarkSuite(args.propmolflow_root, device=args.device)

    print("[Audit] Measuring PropMolFlow single-A800 training throughput on train_rQM_v0 conformers...", flush=True)
    alpha_ckpt = _find_checkpoint(args.propmolflow_root, "alpha")
    train_speed = _benchmark_training_speed(
        ckpt_path=alpha_ckpt,
        train_records=splits["train"],
        propmolflow_root=args.propmolflow_root,
        device=device,
        batch_size=128,
        n_warmup=5,
        n_timed=args.train_bench_steps,
    )
    print(f"[Audit] PropMolFlow training speed: {json.dumps(train_speed)}", flush=True)

    per_prop: Dict[str, Any] = {}
    for prop in PROPS:
        per_prop[prop] = _audit_single_property(
            prop=prop,
            propmolflow_root=args.propmolflow_root,
            gvp_suite=gvp_suite,
            reconciler=reconciler,
            device=device,
            eval_n=args.eval_n,
            max_batch_size=args.max_batch_size,
            n_timesteps=args.n_timesteps,
        )

    speeds = [per_prop[p]["sampling_mol_per_sec"] for p in PROPS]
    nat_san = [per_prop[p]["native_graph"]["sanitized_pct"] for p in PROPS]
    nat_conn = [per_prop[p]["native_graph"]["connected_neutral_pct"] for p in PROPS]
    nat_exo = [per_prop[p]["native_graph"]["exotic_charge_pct_of_sanitized"] for p in PROPS]
    coord_raw = [per_prop[p]["coord_only_perception"]["raw_legal_pct"] for p in PROPS]
    coord_exo = [per_prop[p]["coord_only_perception"]["raw_exotic_charge_pct_of_legal"] for p in PROPS]
    coord_rec = [per_prop[p]["coord_only_perception"]["reconciled_legal_pct"] for p in PROPS]

    summary = {
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "sampling_mol_per_sec_min": min(speeds),
        "sampling_mol_per_sec_max": max(speeds),
        "sampling_mol_per_sec_mean": round(float(np.mean(speeds)), 2),
        "native_sanitized_pct_range": [min(nat_san), max(nat_san)],
        "native_connected_neutral_pct_range": [min(nat_conn), max(nat_conn)],
        "native_exotic_charge_pct_range": [min(nat_exo), max(nat_exo)],
        "coord_raw_legal_pct_range": [min(coord_raw), max(coord_raw)],
        "coord_raw_exotic_pct_range": [min(coord_exo), max(coord_exo)],
        "coord_reconciled_legal_pct_range": [min(coord_rec), max(coord_rec)],
    }
    payload = {
        "audit": "PropMolFlow_Official_6Models_Direct_A800",
        "training_benchmark": train_speed,
        "summary": summary,
        "per_property": per_prop,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Audit] Saved PropMolFlow direct audit to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
