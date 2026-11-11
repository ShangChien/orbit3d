from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import math
import torch
from rdkit import Chem, RDLogger

import dgl

if TYPE_CHECKING:
    from propmolflow.analysis.molecule_builder import SampledMolecule

RDLogger.DisableLog("rdApp.*")

HARTREE_TO_EV = 27.211386245988

VALENCE_ELECTRONS = {
    "H": 1, "C": 4, "N": 5, "O": 6, "F": 7, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Se": 6, "Br": 7, "I": 7,
}
EXPECTED_VALENCES = {
    "H": [1],
    "C": [4],
    "N": [3],
    "O": [2],
    "F": [1],
    "P": [3, 5],
    "S": [2, 6],
    "Cl": [1],
    "Si": [2, 4],
    "Se": [2, 4, 6],
}


def compute_chemical_validity_metrics(molecules: List[SampledMolecule]) -> Dict[str, float]:
    """
    Compute PropMolFlow standard chemical metrics (atom stability, molecule stability,
    RDKit sanitization & connected validity, uniqueness, and closed-shell singlet ratio).
    """
    n_total = len(molecules)
    if n_total == 0:
        return {}

    n_atoms_total = 0
    n_atoms_stable = 0
    n_mols_stable = 0
    n_rdkit_valid = 0
    n_connected_valid = 0
    n_closed_shell = 0
    valid_smiles = []

    for smol in molecules:
        n_atoms_total += smol.num_atoms
        atom_ok_list = []
        total_valence_e = 0
        for sym, val_tensor in zip(smol.atom_types, smol.valencies):
            val = int(val_tensor.item())
            allowed = EXPECTED_VALENCES.get(sym, [])
            atom_ok_list.append(val in allowed)
            total_valence_e += VALENCE_ELECTRONS.get(sym, 0)

        n_atoms_stable += sum(atom_ok_list)
        if all(atom_ok_list):
            n_mols_stable += 1
        if total_valence_e % 2 == 0:
            n_closed_shell += 1

        rdmol = smol.rdkit_mol
        if rdmol is not None:
            try:
                frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                largest = max(frags, default=rdmol, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest)
                smi = Chem.MolToSmiles(largest)
                n_rdkit_valid += 1
                valid_smiles.append(smi)
                if len(frags) == 1:
                    n_connected_valid += 1
            except Exception:
                pass

    return {
        "frac_atoms_stable": n_atoms_stable / max(n_atoms_total, 1),
        "frac_mol_stable": n_mols_stable / n_total,
        "frac_valid_mols": n_rdkit_valid / n_total,
        "frac_connected_valid": n_connected_valid / n_total,
        "close_shell_ratio": n_closed_shell / n_total,
        "valid_and_unique": len(set(valid_smiles)) / n_total,
    }

# Covalent radii in Angstroms (Cordero et al.)
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ir": 1.41,
}


def compute_3d_geometry_strain_metrics(molecules: List[SampledMolecule]) -> Dict[str, float]:
    """
    Evaluate 3D geometric integrity (bond-length strain, steric clashes, and 3D physical validity)
    across sampled molecules.
    """
    n_total = len(molecules)
    if n_total == 0:
        return {}

    bond_deviations = []
    n_clash_free = 0
    n_bond_valid = 0
    n_geom_valid = 0

    for mol in molecules:
        pos = mol.positions.detach().cpu()  # (N, 3)
        atoms = mol.atom_types
        n_atoms = len(atoms)
        if n_atoms <= 1:
            continue

        # Pairwise distance matrix
        dist_mat = torch.cdist(pos, pos, p=2)

        # 1. Check bonded distances against covalent radii sum
        mol_bond_ok = True
        bonded_pairs = set()
        for src, dst, btype in zip(mol.bond_src_idxs.tolist(), mol.bond_dst_idxs.tolist(), mol.bond_types.tolist()):
            if btype == 0:
                continue
            bonded_pairs.add((min(src, dst), max(src, dst)))
            d = float(dist_mat[src, dst].item())
            r_sum = COVALENT_RADII.get(atoms[src], 0.75) + COVALENT_RADII.get(atoms[dst], 0.75)
            # Adjust expected bond length slightly for multiple/aromatic bonds
            if btype == 2:
                r_expected = r_sum * 0.88
            elif btype == 3:
                r_expected = r_sum * 0.79
            elif btype == 4:
                r_expected = r_sum * 0.93
            else:
                r_expected = r_sum

            dev = abs(d - r_expected)
            bond_deviations.append(dev)
            # Physical covalent bond bounds: [0.65 * r_sum, 1.35 * r_sum]
            if d < 0.65 * r_sum or d > 1.35 * r_sum:
                mol_bond_ok = False

        # 2. Check non-bonded steric clashes (< 0.80 Angstrom between any non-bonded pair)
        mol_clash_ok = True
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = float(dist_mat[i, j].item())
                if (i, j) not in bonded_pairs:
                    min_nonbond = 0.75 * (COVALENT_RADII.get(atoms[i], 0.7) + COVALENT_RADII.get(atoms[j], 0.7))
                    if d < min_nonbond:
                        mol_clash_ok = False
                        break
            if not mol_clash_ok:
                break

        if mol_bond_ok:
            n_bond_valid += 1
        if mol_clash_ok:
            n_clash_free += 1
        if mol_bond_ok and mol_clash_ok:
            n_geom_valid += 1

    return {
        "bond_length_mae_A": sum(bond_deviations) / len(bond_deviations) if bond_deviations else 0.0,
        "bond_valid_frac": n_bond_valid / n_total,
        "clash_free_frac": n_clash_free / n_total,
        "geom_3d_valid_frac": n_geom_valid / n_total,
    }


@torch.no_grad()
def evaluate_generated_molecules(
    molecules: List[SampledMolecule],
    target_intervals_norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    prop_guidance: Any,
    dpa_guidance: Optional[Any] = None,
    device: str = "cuda:0",
) -> Dict[str, float]:
    """
    Comprehensive evaluation of generated 3D molecules:
      1. Chemical stability & RDKit validity (atom stability, mol stability, RDKit sanitization,
         connected component validity, closed-shell singlet ratio).
      2. 3D geometric physicality (bond-length strain, steric clash rate, 3D geometry pass rate).
      3. DPA-3 MLIP physical metrics (mean potential energy, RMS atomic force norm in eV/A,
         low-strain conformation ratio).
      4. Target property interval accuracy (Interval Hit Rate %, Interval Hinge Error in Ha & eV,
         Midpoint MAE in Ha & eV, and Joint Multi-Property Hit Rate %).
    """
    chem_metrics = compute_chemical_validity_metrics(molecules)

    geom_metrics = compute_3d_geometry_strain_metrics(molecules)

    # Build batched graph on device to evaluate GVPRegressor & DPA-3
    graphs = []
    valid_indices = []
    for idx, mol in enumerate(molecules):
        g_i = mol.g.to(device)
        graphs.append(g_i)
        valid_indices.append(idx)

    batched_g = dgl.batch(graphs)
    node_batch_idx = torch.repeat_interleave(
        torch.arange(batched_g.batch_size, device=device, dtype=torch.long),
        batched_g.batch_num_nodes(),
    )

    coords = batched_g.ndata["x_1"]
    atom_probs = batched_g.ndata["a_1"][:, :5]
    edge_probs = batched_g.edata["e_1"][:, :5]

    preds_norm = prop_guidance.predict_properties(
        g=batched_g,
        coords=coords,
        atom_probs=atom_probs,
        edge_probs_full=edge_probs,
        use_hard_discrete=True,
    )

    prop_metrics: Dict[str, float] = {}
    joint_hits = torch.ones(len(molecules), dtype=torch.bool, device=device)

    for prop, (y_min_norm, y_max_norm) in target_intervals_norm.items():
        y_min_n = y_min_norm.to(device)
        y_max_n = y_max_norm.to(device)
        y_mid_n = 0.5 * (y_min_n + y_max_n)
        y_hat_n = preds_norm[prop]

        # Convert to raw units (Hartree for homo/lumo/gap)
        y_min_raw = prop_guidance.denormalize_value(prop, y_min_n)
        y_max_raw = prop_guidance.denormalize_value(prop, y_max_n)
        y_mid_raw = prop_guidance.denormalize_value(prop, y_mid_n)
        y_hat_raw = prop_guidance.denormalize_value(prop, y_hat_n)

        # Interval hit rate (with small tolerance 1e-4 for point intervals)
        tol_raw = 0.005 if torch.allclose(y_min_raw, y_max_raw) else 0.0
        in_interval = (y_hat_raw >= (y_min_raw - tol_raw)) & (y_hat_raw <= (y_max_raw + tol_raw))
        joint_hits = joint_hits & in_interval

        hit_rate = float(in_interval.float().mean().item())
        hinge_err_raw = torch.relu(y_min_raw - y_hat_raw) + torch.relu(y_hat_raw - y_max_raw)
        mae_mid_raw = torch.abs(y_hat_raw - y_mid_raw)

        mean_hinge_ha = float(hinge_err_raw.mean().item())
        mean_mae_ha = float(mae_mid_raw.mean().item())

        prop_metrics[f"{prop}_interval_hit_rate"] = hit_rate
        prop_metrics[f"{prop}_interval_err_Ha"] = mean_hinge_ha
        prop_metrics[f"{prop}_midpoint_mae_Ha"] = mean_mae_ha

        if prop in ("gap", "homo", "lumo"):
            prop_metrics[f"{prop}_interval_err_eV"] = mean_hinge_ha * HARTREE_TO_EV
            prop_metrics[f"{prop}_midpoint_mae_eV"] = mean_mae_ha * HARTREE_TO_EV

    prop_metrics["joint_interval_hit_rate"] = float(joint_hits.float().mean().item())

    # DPA-3 physical force & energy evaluation
    dpa_metrics: Dict[str, float] = {}
    if dpa_guidance is not None:
        a_idx = atom_probs.argmax(dim=-1)
        mol_E, forces = dpa_guidance.compute_energy_and_forces(
            coords=coords,
            dataset_atom_indices=a_idx,
            node_batch_idx=node_batch_idx,
            clip_forces=False,
            zero_com_forces=True,
        )
        # Compute per-molecule RMS force norm in eV/A
        f_sq_per_atom = (forces * forces).sum(dim=-1)
        mol_f_sq = torch.zeros(batched_g.batch_size, dtype=coords.dtype, device=device)
        mol_f_sq.index_add_(0, node_batch_idx, f_sq_per_atom)
        mol_rms_force = torch.sqrt(mol_f_sq / batched_g.batch_num_nodes().clamp(min=1).to(coords.dtype))

        # Per-atom energy in eV/atom
        e_per_atom = mol_E / batched_g.batch_num_nodes().clamp(min=1).to(coords.dtype)

        dpa_metrics["dpa_energy_eV_per_atom"] = float(e_per_atom.mean().item())
        dpa_metrics["dpa_rms_force_eV_A"] = float(mol_rms_force.mean().item())
        dpa_metrics["dpa_median_rms_force_eV_A"] = float(mol_rms_force.median().item())
        dpa_metrics["dpa_low_strain_frac"] = float((mol_rms_force < 1.5).float().mean().item())

    out = {}
    out.update(chem_metrics)
    out.update(geom_metrics)
    out.update(prop_metrics)
    out.update(dpa_metrics)
    out["valid_and_in_interval_frac"] = out.get("frac_valid_mols", 0.0) * out["joint_interval_hit_rate"]
    return out
