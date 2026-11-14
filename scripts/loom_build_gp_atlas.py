"""Stage 3: 1D (`HOMO`, `LUMO`) & 2D `(HOMO, LUMO)` Heteroscedastic GP Reachability & Dual-Uncertainty Atlas Builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from loom.generator.octree_backend import trajectory_to_instance
from loom.oracles.unimol2_task import B_WEIGHT_SHA256
from loom.weaving import (
    AuditOracleSlot,
    HeteroscedasticGPAtlas1D,
    HeteroscedasticGPAtlas2D,
    LoomUni3DARGenerator,
    PotentialSlot,
    RewardOracleSlot,
    is_clean_organic_charge_state,
)
from scripts.loom_train_sft import load_sdf_with_targets


def _prepare_clean_relaxed_mols(mols_all: list, pot_slot: PotentialSlot | None) -> list:
    clean_idx = [
        i for i, m in enumerate(mols_all) if m is not None and is_clean_organic_charge_state(m)
    ]
    out_mols = [None] * len(mols_all)
    if not clean_idx:
        return out_mols
    if pot_slot is None:
        for i in clean_idx:
            out_mols[i] = mols_all[i]
        return out_mols
    sub_mols = [mols_all[i] for i in clean_idx]
    rel = pot_slot.relax_molecules(sub_mols, n_steps=15, chunk_size=32)
    for loc_i, glob_i in enumerate(clean_idx):
        if rel["graph_preserved"][loc_i] and rel["relaxed_mols"][loc_i] is not None:
            out_mols[glob_i] = rel["relaxed_mols"][loc_i]
    return out_mols


def run_1d_grid(
    gen: LoomUni3DARGenerator,
    reward_slot: RewardOracleSlot,
    audit_slot: AuditOracleSlot | None,
    pot_slot: PotentialSlot | None,
    prop_name: str,
    grid_eV: np.ndarray,
    proposals_per_point: int,
    cfg_scale: float,
    source_vals_eV: np.ndarray,
    cal_sigma_eV: float,
    seed: int,
) -> dict:
    device = next(gen.parameters()).device
    prop_idx = gen.target_cols.index(prop_name)
    rew_idx = reward_slot.target_cols.index(prop_name)
    aud_idx = audit_slot.target_cols.index(prop_name) if audit_slot is not None else 0
    allowed_elems = tuple(gen.args.allow_atoms.split(","))

    req_all, mols_all = [], []
    for g_idx, t_val in enumerate(grid_eV):
        t_mat = np.zeros((proposals_per_point, gen.num_props), dtype=np.float32)
        t_mat[:, prop_idx] = float(t_val)
        m_act = torch.zeros((proposals_per_point, gen.num_props), dtype=torch.bool, device=device)
        m_act[:, prop_idx] = True
        trajs = gen.sample(
            gen.normalize_targets(t_mat),
            m_act,
            cfg_scale=cfg_scale,
            xyz_cfg_scale=1.0,
            seed=seed + g_idx * 97,
            use_bf16=(device.type == "cuda"),
            electron_parity=0,
        )
        for tr in trajs:
            req_all.append(float(t_val))
            try:
                mols_all.append(trajectory_to_instance(tr, element_table=allowed_elems).graph)
            except ValueError:
                mols_all.append(None)

    eval_mols = _prepare_clean_relaxed_mols(mols_all, pot_slot)
    rew_full, _, rew_ev, _ = reward_slot.evaluate_candidates(eval_mols, batch_size=32)
    aud_1d = None
    if audit_slot is not None:
        aud_full, _, _, _ = audit_slot.evaluate_candidates(eval_mols, batch_size=32)
        aud_1d = aud_full[:, aud_idx]

    atlas = HeteroscedasticGPAtlas1D.fit_from_proposals(
        req_all,
        rew_full[:, rew_idx],
        rew_ev,
        audit_or_dft_preds_eV=aud_1d,
        source_property_values_eV=source_vals_eV,
        constant_cal_sigma_eV=cal_sigma_eV,
    )
    dense_q = np.linspace(float(grid_eV.min()) - 0.3, float(grid_eV.max()) + 0.3, 120)
    return atlas.query(dense_q, tolerance_eV=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homo-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, default=None)
    parser.add_argument("--source-train-sdf", type=Path, required=True)
    parser.add_argument("--reward-oracle-dir", type=Path, required=True)
    parser.add_argument("--audit-oracle-dir", type=Path, default=None)
    parser.add_argument("--dpa4-forcefield", type=Path, default=None)
    parser.add_argument("--potential-backend", choices=["none", "dpa4", "hyff"], default="dpa4")
    parser.add_argument("--potential-weights", type=Path, default=None)
    parser.add_argument("--proposals-per-1d-point", type=int, default=64)
    parser.add_argument("--proposals-per-2d-cell", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=1.75)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    t0 = time.perf_counter()
    _, src_labels = load_sdf_with_targets(args.source_train_sdf, ["homo", "lumo"])

    reward_slot = RewardOracleSlot(
        backend="unimol2",
        checkpoint_dir=args.reward_oracle_dir,
        device=args.device,
        expected_weight_sha256=B_WEIGHT_SHA256,
    )
    audit_slot = (
        AuditOracleSlot(
            backend="dpa4",
            checkpoint_dir=args.audit_oracle_dir,
            device=args.device,
            forcefield_path=str(args.dpa4_forcefield) if args.dpa4_forcefield else None,
        )
        if args.audit_oracle_dir is not None and args.audit_oracle_dir.exists()
        else None
    )

    pot_weights = args.potential_weights or args.dpa4_forcefield
    pot_slot = (
        PotentialSlot(backend=args.potential_backend, weights=str(pot_weights), device=args.device)
        if args.potential_backend != "none" and pot_weights is not None
        else None
    )

    gen_h, _ = LoomUni3DARGenerator.load_checkpoint(args.homo_checkpoint, device=args.device, batch_size=128)
    gen_h.eval()

    # 1D HOMO Reachability & Dual-Uncertainty Atlas (including slight OOD extrapolation tails!)
    h_p02, h_p98 = np.percentile(src_labels[:, 0], [1.5, 98.5])
    homo_grid = np.linspace(h_p02 - 0.35, h_p98 + 0.35, 16)
    print(f"[GP-Atlas] Evaluating 1D HOMO grid ({len(homo_grid)} pts × {args.proposals_per_1d_point} mols)...", flush=True)
    homo_1d = run_1d_grid(
        gen_h, reward_slot, audit_slot, pot_slot, "homo", homo_grid, args.proposals_per_1d_point,
        args.cfg_scale, src_labels[:, 0], cal_sigma_eV=0.19, seed=71001
    )

    out_data: dict = {
        "homo_1d_atlas": homo_1d,
        "lumo_1d_atlas": None,
        "joint_2d_atlas": None,
    }

    if args.joint_checkpoint is not None and args.joint_checkpoint.exists():
        gen_j, _ = LoomUni3DARGenerator.load_checkpoint(args.joint_checkpoint, device=args.device, batch_size=128)
        gen_j.eval()
        device = torch.device(args.device)
        allowed_elems = tuple(gen_j.args.allow_atoms.split(","))

        l_p02, l_p98 = np.percentile(src_labels[:, 1], [1.5, 98.5])
        lumo_grid = np.linspace(l_p02 - 0.35, l_p98 + 0.35, 16)
        print(f"[GP-Atlas] Evaluating 1D LUMO grid ({len(lumo_grid)} pts × {args.proposals_per_1d_point} mols)...", flush=True)
        out_data["lumo_1d_atlas"] = run_1d_grid(
            gen_j, reward_slot, audit_slot, pot_slot, "lumo", lumo_grid, args.proposals_per_1d_point,
            args.cfg_scale, src_labels[:, 1], cal_sigma_eV=0.21, seed=72001
        )

        # 2D (HOMO, LUMO) 7×7 grid = 49 cells (including OOD & near-diagonal cells!)
        h_2d = np.linspace(np.percentile(src_labels[:, 0], 4), np.percentile(src_labels[:, 0], 96), 7)
        l_2d = np.linspace(np.percentile(src_labels[:, 1], 4), np.percentile(src_labels[:, 1], 96), 7)
        hh, ll = np.meshgrid(h_2d, l_2d, indexing="ij")
        cells = np.stack([hh.ravel(), ll.ravel()], axis=-1)

        print(f"[GP-Atlas] Evaluating 2D (HOMO, LUMO) grid ({len(cells)} cells × {args.proposals_per_2d_cell} mols)...", flush=True)
        req_2d_list = []
        mols_2d = []
        for c_idx, cell in enumerate(cells):
            if cell[0] >= cell[1] - 0.2:
                # Physically forbidden ground-state zone
                continue
            k_n = args.proposals_per_2d_cell
            t_mat = np.repeat(cell[None, :].astype(np.float32), k_n, axis=0)
            m_act = torch.ones((k_n, 2), dtype=torch.bool, device=device)
            trajs = gen_j.sample(
                gen_j.normalize_targets(t_mat),
                m_act,
                cfg_scale=args.cfg_scale,
                xyz_cfg_scale=1.0,
                seed=73001 + c_idx * 53,
                use_bf16=(device.type == "cuda"),
                electron_parity=0,
            )
            for tr in trajs:
                req_2d_list.append(cell)
                try:
                    mols_2d.append(trajectory_to_instance(tr, element_table=allowed_elems).graph)
                except ValueError:
                    mols_2d.append(None)

        eval_mols_2d = _prepare_clean_relaxed_mols(mols_2d, pot_slot)
        rew_2d, _, ev_2d, _ = reward_slot.evaluate_candidates(eval_mols_2d, batch_size=32)
        atlas_2d = HeteroscedasticGPAtlas2D.fit_from_proposals(
            np.asarray(req_2d_list, dtype=np.float64),
            rew_2d[:, [reward_slot.target_cols.index("homo"), reward_slot.target_cols.index("lumo")]],
            ev_2d,
            source_2d_eV=src_labels,
            constant_cal_sigma_eV=(0.19, 0.21),
        )
        out_data["joint_2d_atlas"] = atlas_2d.query_grid(
            np.linspace(h_2d.min() - 0.3, h_2d.max() + 0.3, 50),
            np.linspace(l_2d.min() - 0.3, l_2d.max() + 0.3, 50),
            tolerance_eV=0.25,
        )

    out_data["wall_seconds"] = time.perf_counter() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_data, indent=2) + "\n")
    print(f"[GP-Atlas] Saved complete 1D/2D GP Reachability Atlas to {args.output} ({out_data['wall_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
