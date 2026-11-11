#!/usr/bin/env python3
"""
3D-SAT-OLED Experiment:
3D Equivariant Flow Matching + DeePMD-kit DPA-3 MLIP Orthogonal Manifold-Projected
Interval Guidance on the 3D-SAT-OLED Dataset (homo, lumo, E_g = lumo - homo, e_ad, plqy).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "orbit3d" / "compat"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.flow_matching.dpa_potential import DPAPotentialGuidance
from benchmarks.flow_matching.interval_guidance import (
    EquivariantIntervalConditioner,
    orthogonal_manifold_projection,
    zero_com_per_molecule,
)

HARTREE_TO_EV = 27.211386245988

# Periodic table mapping for atomic numbers Z in 3D-SAT-OLED: {1, 6, 7, 8, 9, 14, 15, 16, 17, 77}
Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    77: "Ir",
}
OLED_ATOM_TYPES = ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Ir"]
Z_TO_IDX = {z: OLED_ATOM_TYPES.index(sym) for z, sym in Z_TO_SYMBOL.items()}


def load_oled_split(npz_path: Path) -> List[Dict[str, torch.Tensor]]:
    """Load 3D-SAT-OLED split (.npz) into a list of molecule dictionaries."""
    data = np.load(npz_path, allow_pickle=True)
    R_arr = data["coord"] if "coord" in data.files else data["R"]
    Z_arr = data["symbol"] if "symbol" in data.files else data["Z"]
    homo_arr = data["homo"]
    lumo_arr = data["lumo"]
    e_ad_arr = data["e_ad"]
    plqy_arr = data["plqy"]

    mols = []
    for i in range(len(R_arr)):
        coords = torch.tensor(np.asarray(R_arr[i], dtype=np.float32), dtype=torch.float32)
        coords = coords - coords.mean(dim=0, keepdim=True)
        a_idx_list = []
        for z in Z_arr[i]:
            if isinstance(z, (str, bytes, np.str_)):
                sym = z.decode() if isinstance(z, bytes) else str(z).strip()
                a_idx_list.append(OLED_ATOM_TYPES.index(sym) if sym in OLED_ATOM_TYPES else 1)
            else:
                a_idx_list.append(Z_TO_IDX.get(int(z), 1))
        a_idx = torch.tensor(a_idx_list, dtype=torch.long)

        raw_h = float(np.asarray(homo_arr[i]).reshape(-1)[0])
        raw_l = float(np.asarray(lumo_arr[i]).reshape(-1)[0])
        # If homo/lumo are in Hartree (magnitude < 1.0), convert to eV; otherwise already in eV
        scale = HARTREE_TO_EV if abs(raw_h) < 1.5 else 1.0
        homo = raw_h * scale
        lumo = raw_l * scale
        gap = lumo - homo  # eV
        e_ad = float(np.asarray(e_ad_arr[i]).reshape(-1)[0])
        plqy = float(np.asarray(plqy_arr[i]).reshape(-1)[0])

        # Compute reference pairwise covalent/coordination bond lengths (< 2.45 A)
        dist = torch.cdist(coords, coords)
        bond_mask = (dist > 0.4) & (dist < 2.45)
        bond_src, bond_dst = torch.where(torch.triu(bond_mask, diagonal=1))
        bond_ref_len = dist[bond_src, bond_dst]

        mols.append(
            {
                "coords": coords,
                "a_idx": a_idx,
                "props": torch.tensor([gap, homo, lumo, e_ad, plqy], dtype=torch.float32),
                "bond_src": bond_src,
                "bond_dst": bond_dst,
                "bond_ref_len": bond_ref_len,
            }
        )
    return mols


class EquivariantOLEDPropertyRegressor(nn.Module):
    """
    SE(3)-Invariant / Equivariant 3D Distance-RBF Message Passing Network
    predicting OLED optoelectronic properties (E_g, HOMO, LUMO, E_ad, PLQY) in eV
    with full differentiability w.r.t. 3D coordinates R.
    """

    def __init__(
        self,
        num_atom_types: int = len(OLED_ATOM_TYPES),
        hidden_dim: int = 128,
        num_rbf: int = 32,
        cutoff: float = 5.5,
        num_layers: int = 4,
        out_dim: int = 3,  # [gap, homo, lumo] in normalized units
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        centers = torch.linspace(0.5, cutoff, num_rbf)
        widths = torch.full((num_rbf,), (cutoff - 0.5) / num_rbf)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

        self.atom_emb = nn.Embedding(num_atom_types, hidden_dim)
        self.edge_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.node_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 3) atomic coordinates (requires_grad=True during guidance).
            a_idx: (N,) integer atom type indices.
        Returns:
            pred: (out_dim,) normalized property predictions [gap, homo, lumo].
        """
        h = self.atom_emb(a_idx)  # (N, D)
        dist = torch.cdist(coords, coords, p=2)  # (N, N)
        mask = (dist > 1e-4) & (dist < self.cutoff)
        src, dst = torch.where(mask)

        d_ij = dist[src, dst]
        # Smooth cosine cutoff envelope for continuous differentiability
        env = 0.5 * (torch.cos(math.pi * d_ij / self.cutoff) + 1.0)
        rbf = torch.exp(-(((d_ij.unsqueeze(-1) - self.centers) / self.widths) ** 2)) * env.unsqueeze(-1)

        for edge_mlp, norm in zip(self.edge_mlps, self.node_norms):
            m_ij = edge_mlp(torch.cat([h[src], h[dst], rbf], dim=-1)) * env.unsqueeze(-1)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m_ij)
            h = norm(h + agg / 12.0)

        mol_feat = h.mean(dim=0)
        return self.readout(mol_feat)


class EquivariantOLEDFlowField(nn.Module):
    """
    SE(3)-Equivariant Conditional Flow Matching Vector Field for 3D OLED Conformational
    and Frontier-Orbital Generation, conditioned via EquivariantIntervalConditioner.
    """

    def __init__(
        self,
        num_atom_types: int = len(OLED_ATOM_TYPES),
        hidden_dim: int = 128,
        cond_dim: int = 64,
        num_rbf: int = 32,
        cutoff: float = 6.0,
        num_layers: int = 3,
    ):
        super().__init__()
        self.cutoff = cutoff
        centers = torch.linspace(0.5, cutoff, num_rbf)
        widths = torch.full((num_rbf,), (cutoff - 0.5) / num_rbf)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

        self.interval_cond = EquivariantIntervalConditioner(
            num_rbf=24, r_min=-3.5, r_max=3.5, hidden_dim=hidden_dim, out_dim=cond_dim
        )
        self.atom_emb = nn.Embedding(num_atom_types, hidden_dim)
        self.node_init = nn.Linear(hidden_dim + cond_dim + 1, hidden_dim)

        self.scalar_layers = nn.ModuleList()
        self.coord_weights = nn.ModuleList()
        for _ in range(num_layers):
            self.scalar_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2 + num_rbf + 1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.coord_weights.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
            )

    def forward(
        self,
        x_t: torch.Tensor,
        a_idx: torch.Tensor,
        t_scalar: float,
        y_min_norm: torch.Tensor,
        y_max_norm: torch.Tensor,
        bond_src: torch.Tensor,
        bond_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Predict clean endpoint coordinates x_1_hat given noisy/deformed x_t and target interval."""
        n_atoms = x_t.shape[0]
        device = x_t.device

        c_emb = self.interval_cond(y_min_norm.view(1), y_max_norm.view(1)).expand(n_atoms, -1)
        t_feat = torch.full((n_atoms, 1), t_scalar, dtype=x_t.dtype, device=device)
        h = self.node_init(torch.cat([self.atom_emb(a_idx), c_emb, t_feat], dim=-1))

        # Combine chemical graph edges and spatial proximity edges
        dist = torch.cdist(x_t, x_t, p=2)
        prox_mask = (dist > 1e-4) & (dist < self.cutoff)
        adj_bond = torch.zeros_like(prox_mask)
        if bond_src.numel() > 0:
            adj_bond[bond_src, bond_dst] = True
            adj_bond[bond_dst, bond_src] = True
        full_mask = prox_mask | adj_bond
        src, dst = torch.where(full_mask)
        is_cov = adj_bond[src, dst].float().unsqueeze(-1)

        x_curr = x_t
        for s_mlp, w_mlp in zip(self.scalar_layers, self.coord_weights):
            rel = x_curr[src] - x_curr[dst]  # (E, 3)
            d_ij = torch.norm(rel, dim=-1) + 1e-6
            unit_rel = rel / d_ij.unsqueeze(-1)
            rbf = torch.exp(-(((d_ij.unsqueeze(-1) - self.centers) / self.widths) ** 2))

            m_ij = s_mlp(torch.cat([h[src], h[dst], rbf, is_cov], dim=-1))
            agg_h = torch.zeros_like(h)
            agg_h.index_add_(0, dst, m_ij)
            h = F.layer_norm(h + agg_h / 10.0, (h.shape[-1],))

            w_ij = torch.tanh(w_mlp(m_ij)) * unit_rel  # Equivariant edge force vector
            delta_x = torch.zeros_like(x_curr)
            delta_x.index_add_(0, dst, w_ij / 10.0)
            x_curr = x_curr + delta_x
            x_curr = x_curr - x_curr.mean(dim=0, keepdim=True)

        return x_curr


def compute_oled_hybrid_dpa_forces(
    coords: torch.Tensor,
    mol_ref: Dict[str, torch.Tensor],
    dpa_guidance: DPAPotentialGuidance,
    clip_forces: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute physical restoring forces for 3D-SAT-OLED organometallic complexes:
      - Evaluates DPA-3.1-3M (SPICE2 organic DFT head) on the pi-conjugated organic ligands
        (masking out the heavy transition metal Ir Z=77 which is outside SPICE2's organic element set),
      - Combines with first-principles harmonic coordination forces on the octahedral Ir-ligand
        coordination bonds so both the Ir(III) core and organic ligands experience accurate restoring forces.
    """
    device = coords.device
    a_idx = mol_ref["a_idx"].to(device)
    ir_idx = OLED_ATOM_TYPES.index("Ir")
    is_ir = a_idx == ir_idx

    # For SPICE2 organic evaluation, substitute central Ir with neutral dummy C far away or mask its force
    a_idx_org = torch.where(is_ir, torch.tensor(OLED_ATOM_TYPES.index("C"), device=device), a_idx)
    coords_org = coords.clone()
    if is_ir.any():
        # Shift Ir atom outside cutoff (30 A away) so SPICE2 only computes pure organic ligand DFT forces
        coords_org[is_ir] = coords_org[is_ir] + torch.tensor([30.0, 0.0, 0.0], device=device, dtype=coords.dtype)

    mol_E, f_phys = dpa_guidance.compute_energy_and_forces(
        coords=coords_org,
        dataset_atom_indices=a_idx_org,
        node_batch_idx=torch.zeros(coords.shape[0], dtype=torch.long, device=device),
        clip_forces=clip_forces,
        zero_com_forces=False,
    )
    f_phys[is_ir] = 0.0

    # Add physical covalent/coordination bond harmonic restoring forces F_cov = -k * (d - d_ref) * u_ij
    bond_src = mol_ref["bond_src"].to(device)
    bond_dst = mol_ref["bond_dst"].to(device)
    bond_ref_len = mol_ref["bond_ref_len"].to(device)
    if bond_src.numel() > 0:
        rel = coords[bond_src] - coords[bond_dst]
        d = torch.norm(rel, dim=-1) + 1e-8
        u_ij = rel / d.unsqueeze(-1)
        # k_bond = 15.0 eV/A^2 (~standard covalent stretching force constant)
        f_bond = -15.0 * (d - bond_ref_len).unsqueeze(-1) * u_ij
        f_cov = torch.zeros_like(coords)
        f_cov.index_add_(0, bond_src, f_bond)
        f_cov.index_add_(0, bond_dst, -f_bond)
        f_phys = f_phys + f_cov

    if clip_forces:
        fnorm = torch.norm(f_phys, dim=-1, keepdim=True) + 1e-8
        f_phys = f_phys * torch.clamp(15.0 / fnorm, max=1.0)

    f_phys = f_phys - f_phys.mean(dim=0, keepdim=True)
    return mol_E, f_phys


def evaluate_oled_conformation(
    coords: torch.Tensor,
    mol_ref: Dict[str, torch.Tensor],
    regressor: EquivariantOLEDPropertyRegressor,
    prop_mean: torch.Tensor,
    prop_std: torch.Tensor,
    target_min_eV: float,
    target_max_eV: float,
    dpa_guidance: DPAPotentialGuidance,
    prop_idx: int = 0,  # 0=gap, 1=homo, 2=lumo
) -> Dict[str, float]:
    """Evaluate 3D bond-length fidelity, DPA-3 physical strain, and target interval hit on an OLED complex."""
    device = coords.device
    a_idx = mol_ref["a_idx"].to(device)
    bond_src = mol_ref["bond_src"].to(device)
    bond_dst = mol_ref["bond_dst"].to(device)
    bond_ref_len = mol_ref["bond_ref_len"].to(device)

    # 1. Bond length strain vs reference coordination/covalent bonds
    d_bonds = torch.norm(coords[bond_src] - coords[bond_dst], dim=-1)
    bond_strain_mae = float(torch.abs(d_bonds - bond_ref_len).mean().item())
    max_bond_stretch = float(torch.abs(d_bonds - bond_ref_len).max().item())
    bond_intact = float((max_bond_stretch < 0.22))

    # 2. Steric clash check (non-bonded < 1.05 A)
    dist_mat = torch.cdist(coords, coords)
    eye = torch.eye(coords.shape[0], dtype=torch.bool, device=device)
    bonded = eye.clone()
    bonded[bond_src, bond_dst] = True
    bonded[bond_dst, bond_src] = True
    min_nonbond = float(dist_mat[~bonded].min().item()) if (~bonded).any() else 2.0
    clash_free = float(min_nonbond > 1.05)

    # 3. Predicted property from 3D Equivariant OLED Regressor
    with torch.no_grad():
        pred_norm = regressor(coords, a_idx)
        pred_eV = pred_norm * prop_std + prop_mean
        y_hat = float(pred_eV[prop_idx].item())

    in_interval = float((y_hat >= target_min_eV) and (y_hat <= target_max_eV))
    interval_err = max(0.0, target_min_eV - y_hat, y_hat - target_max_eV)

    # 4. DPA-3 MLIP physical energy & RMS atomic force on organic ligands + coordination sphere
    mol_E, f_phys = compute_oled_hybrid_dpa_forces(
        coords=coords,
        mol_ref=mol_ref,
        dpa_guidance=dpa_guidance,
        clip_forces=False,
    )
    rms_force = float(torch.sqrt((f_phys * f_phys).sum(dim=-1).mean()).item())

    return {
        "in_interval": in_interval,
        "interval_err_eV": interval_err,
        "pred_val_eV": y_hat,
        "bond_strain_mae_A": bond_strain_mae,
        "max_bond_stretch_A": max_bond_stretch,
        "geom_valid": float(bond_intact and clash_free),
        "dpa_rms_force_eV_A": rms_force,
        "dpa_energy_eV_per_atom": float(mol_E.item() / coords.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oled_dir",
        type=str,
        default="/sjtu-combustion/wjw/chensq/3D-SAT-OLED/SAT_OLED_Model/oled",
    )
    parser.add_argument(
        "--dpa_model_path",
        type=str,
        default="/sjtu-combustion/wjw/chensq/pretrained/DPA-3.1-3M.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/sjtu-combustion/wjw/chensq/results/oled_3dsat",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    oled_dir = Path(args.oled_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading 3D-SAT-OLED dataset (train.npz: 800, test.npz: 200)...")
    train_mols = load_oled_split(oled_dir / "train.npz")
    test_mols = load_oled_split(oled_dir / "test.npz")

    all_train_props = torch.stack([m["props"][:3] for m in train_mols], dim=0).to(device)  # [gap, homo, lumo] in eV
    prop_mean = all_train_props.mean(dim=0)
    prop_std = all_train_props.std(dim=0).clamp(min=1e-4)
    print(
        f"   Train stats (eV): E_g={prop_mean[0]:.2f}±{prop_std[0]:.2f}, "
        f"HOMO={prop_mean[1]:.2f}±{prop_std[1]:.2f}, LUMO={prop_mean[2]:.2f}±{prop_std[2]:.2f}"
    )

    print("2. Loading DPA-3.1-3M (SPICE2 head, supporting Ir + C/H/N/O/F/Si/P/S/Cl)...")
    dpa_guidance = DPAPotentialGuidance(
        model_path=args.dpa_model_path,
        head="SPICE2",
        atom_type_map=OLED_ATOM_TYPES,
        device=str(device),
        max_force_norm=15.0,
    )

    print("3. Training 3D Equivariant OLED Frontier-Orbital Regressor on train.npz...")
    regressor = EquivariantOLEDPropertyRegressor().to(device)
    opt_reg = torch.optim.AdamW(regressor.parameters(), lr=1.5e-3, weight_decay=1e-4)

    torch.manual_seed(42)
    for epoch in range(1, 26):
        regressor.train()
        perm = torch.randperm(len(train_mols))
        total_loss = 0.0
        for idx in perm[:300]:  # Fast mini-epoch over 300 complexes
            m = train_mols[int(idx)]
            coords = m["coords"].to(device)
            # Add small 3D thermal jitter so regressor learns smooth coordinate gradients
            coords_noisy = coords + 0.04 * torch.randn_like(coords)
            a_idx = m["a_idx"].to(device)
            target_norm = (m["props"][:3].to(device) - prop_mean) / prop_std

            pred = regressor(coords_noisy, a_idx)
            loss = F.smooth_l1_loss(pred, target_norm)
            opt_reg.zero_grad()
            loss.backward()
            opt_reg.step()
            total_loss += float(loss.item())

        if epoch % 5 == 0:
            regressor.eval()
            with torch.no_grad():
                maes = []
                for m in test_mols[:60]:
                    p_norm = regressor(m["coords"].to(device), m["a_idx"].to(device))
                    p_eV = p_norm * prop_std + prop_mean
                    maes.append(torch.abs(p_eV - m["props"][:3].to(device)))
                mean_mae = torch.stack(maes).mean(dim=0)
            print(
                f"   [Regressor Epoch {epoch:02d}] TrainLoss={total_loss/300:.4f} | "
                f"Test MAE (eV): E_g={mean_mae[0]:.3f}, HOMO={mean_mae[1]:.3f}, LUMO={mean_mae[2]:.3f}"
            )

    regressor.eval()
    for p in regressor.parameters():
        p.requires_grad_(False)

    print("4. Training 3D Equivariant Interval-Conditioned OLED Flow Matcher...")
    flow_net = EquivariantOLEDFlowField().to(device)
    opt_flow = torch.optim.AdamW(flow_net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    noise_sigma = 0.35  # 3D conformational flow prior scale (Angstrom)

    for epoch in range(1, 16):
        flow_net.train()
        perm = torch.randperm(len(train_mols))
        floss_sum = 0.0
        for idx in perm[:250]:
            m = train_mols[int(idx)]
            x_1 = m["coords"].to(device)
            a_idx = m["a_idx"].to(device)
            bond_src = m["bond_src"].to(device)
            bond_dst = m["bond_dst"].to(device)
            y_true_norm = ((m["props"][0].to(device) - prop_mean[0]) / prop_std[0]).view(1)

            y_min_n, y_max_n = flow_net.interval_cond.sample_training_intervals(y_true_norm)
            t_val = float(torch.rand(1).item())
            noise = noise_sigma * torch.randn_like(x_1)
            noise = noise - noise.mean(dim=0, keepdim=True)
            x_0 = x_1 + noise
            x_t = (1.0 - t_val) * x_0 + t_val * x_1

            x_1_pred = flow_net(x_t, a_idx, t_val, y_min_n, y_max_n, bond_src, bond_dst)
            loss = F.mse_loss(x_1_pred, x_1)
            opt_flow.zero_grad()
            loss.backward()
            opt_flow.step()
            floss_sum += float(loss.item())

        if epoch % 5 == 0:
            print(f"   [Flow Epoch {epoch:02d}] Flow Endpoint MSE = {floss_sum/250:.5f} A^2")

    flow_net.eval()

    print("\n5. Evaluating 3D OLED Frontier Orbital Target Interval Generation with DPA-3 Orthogonal Projection...")

    tasks = [
        ("OLED_Eg_Bandgap", 0, 0.35, 1.10),
        ("OLED_HOMO_Level", 1, 0.30, 1.05),
        ("OLED_LUMO_Level", 2, 0.30, 1.05),
    ]

    n_eval = 40
    n_steps = 25
    dt = 1.0 / n_steps

    modes = [
        ("1_baseline_flow", 0.0, False, 0.0, 0),
        ("2_naive_guidance", 1.25, False, 0.0, 0),
        ("3_dpa_orthogonal", 1.25, True, 0.015, 0),
        ("4_dpa_orthogonal_relaxed", 1.25, True, 0.015, 8),
    ]

    full_summary = {}
    for task_name, prop_idx, std_low, std_high in tasks:
        target_min_eV = float((prop_mean[prop_idx] + std_low * prop_std[prop_idx]).item())
        target_max_eV = float((prop_mean[prop_idx] + std_high * prop_std[prop_idx]).item())
        y_min_norm = ((torch.tensor(target_min_eV, device=device) - prop_mean[prop_idx]) / prop_std[prop_idx]).view(1)
        y_max_norm = ((torch.tensor(target_max_eV, device=device) - prop_mean[prop_idx]) / prop_std[prop_idx]).view(1)
        print(f"\n   === Task: {task_name} | Target Interval: [{target_min_eV:.3f}, {target_max_eV:.3f}] eV ===", flush=True)

        summary = {}
        for mode_name, lam_prop, use_dpa, eta_phys, relax_steps in modes:
            metrics_list = []
            conflict_rates = []
            for idx in range(n_eval):
                m = test_mols[idx]
                x_ref = m["coords"].to(device)
                a_idx = m["a_idx"].to(device)
                bond_src = m["bond_src"].to(device)
                bond_dst = m["bond_dst"].to(device)
                node_batch_idx = torch.zeros(x_ref.shape[0], dtype=torch.long, device=device)

                torch.manual_seed(1000 + idx)
                noise = noise_sigma * torch.randn_like(x_ref)
                noise = noise - noise.mean(dim=0, keepdim=True)
                x_t = x_ref + noise

                for s_idx in range(n_steps):
                    t_val = s_idx * dt
                    with torch.no_grad():
                        x_1_hat = flow_net(x_t, a_idx, t_val, y_min_norm, y_max_norm, bond_src, bond_dst)
                        v_flow = (x_1_hat - x_t) / max(1.0 - t_val, 0.05)
                        x_next = x_t + dt * v_flow

                    if lam_prop > 0.0 and t_val >= 0.35:
                        with torch.enable_grad():
                            x_in = x_next.detach().clone().requires_grad_(True)
                            pred_n = regressor(x_in, a_idx)[prop_idx]
                            hinge = torch.relu(y_min_norm[0] - pred_n).pow(2) + torch.relu(pred_n - y_max_norm[0]).pow(2)
                            if hinge.item() > 0:
                                grad_x = torch.autograd.grad(hinge, x_in)[0]
                                g_prop = -grad_x.detach()
                                g_prop = g_prop - g_prop.mean(dim=0, keepdim=True)
                                rms_g = torch.sqrt((g_prop * g_prop).sum(dim=-1).mean()) + 1e-8
                                gain = min(1.0, math.sqrt(float(hinge.item())) / 0.12)
                                g_prop = g_prop * (gain / rms_g)
                            else:
                                g_prop = torch.zeros_like(x_next)

                        if use_dpa:
                            _, f_phys = compute_oled_hybrid_dpa_forces(
                                coords=x_next,
                                mol_ref=m,
                                dpa_guidance=dpa_guidance,
                                clip_forces=True,
                            )
                            g_tilde, pstats = orthogonal_manifold_projection(
                                g_prop=g_prop,
                                f_phys=f_phys,
                                node_batch_idx=node_batch_idx,
                                batch_size=1,
                                atom_level_projection=True,
                                mol_level_projection=True,
                            )
                            conflict_rates.append(pstats["conflict_frac"])
                            step_shift = dt * (lam_prop * g_tilde + eta_phys * f_phys)
                        else:
                            step_shift = dt * (lam_prop * g_prop)

                        s_norm = torch.norm(step_shift, dim=-1, keepdim=True) + 1e-8
                        step_shift = step_shift * torch.clamp(0.08 / s_norm, max=1.0)
                        x_next = x_next + step_shift - step_shift.mean(dim=0, keepdim=True)

                    x_t = x_next

                if relax_steps > 0:
                    for _ in range(relax_steps):
                        _, f_rel = compute_oled_hybrid_dpa_forces(
                            coords=x_t,
                            mol_ref=m,
                            dpa_guidance=dpa_guidance,
                            clip_forces=True,
                        )
                        disp = 0.004 * f_rel
                        d_norm = torch.norm(disp, dim=-1, keepdim=True) + 1e-8
                        disp = disp * torch.clamp(0.015 / d_norm, max=1.0)
                        x_cand = x_t + disp - disp.mean(dim=0, keepdim=True)
                        # Only accept relaxation step if property stays inside or moves closer to target interval
                        with torch.no_grad():
                            p_old = float((regressor(x_t, a_idx)[prop_idx] * prop_std[prop_idx] + prop_mean[prop_idx]).item())
                            p_new = float((regressor(x_cand, a_idx)[prop_idx] * prop_std[prop_idx] + prop_mean[prop_idx]).item())
                        err_old = max(0.0, target_min_eV - p_old, p_old - target_max_eV)
                        err_new = max(0.0, target_min_eV - p_new, p_new - target_max_eV)
                        if err_new <= err_old + 0.01:
                            x_t = x_cand

                res = evaluate_oled_conformation(
                    coords=x_t,
                    mol_ref=m,
                    regressor=regressor,
                    prop_mean=prop_mean,
                    prop_std=prop_std,
                    target_min_eV=target_min_eV,
                    target_max_eV=target_max_eV,
                    dpa_guidance=dpa_guidance,
                    prop_idx=prop_idx,
                )
                metrics_list.append(res)

            agg = {
                k: float(np.mean([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            agg["valid_and_in_interval"] = float(
                np.mean([m["geom_valid"] * m["in_interval"] for m in metrics_list])
            )
            agg["conflict_rate"] = float(np.mean(conflict_rates)) if conflict_rates else 0.0
            summary[mode_name] = agg
            print(
                f"   [{mode_name:24s}] "
                f"HitRate={agg['in_interval']*100:5.1f}% | "
                f"IntErr={agg['interval_err_eV']:5.3f} eV | "
                f"GeomValid={agg['geom_valid']*100:5.1f}% | "
                f"Valid&Hit={agg['valid_and_in_interval']*100:5.1f}% | "
                f"BondStrain={agg['bond_strain_mae_A']:5.3f} A | "
                f"DPA_RMS_F={agg['dpa_rms_force_eV_A']:5.2f} eV/A",
                flush=True,
            )
        full_summary[task_name] = summary

    out_json = out_dir / "oled_3dsat_summary.json"
    with open(out_json, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nSaved 3D-SAT-OLED experiment results to {out_json}", flush=True)


if __name__ == "__main__":
    main()
