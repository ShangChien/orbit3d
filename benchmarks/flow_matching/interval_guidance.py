from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import torch
import torch.nn as nn

import dgl


def zero_com_per_molecule(vec: torch.Tensor, node_batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Subtract the per-molecule mean vector to remain strictly in the Zero-CoM subspace."""
    if batch_size <= 0 or vec.numel() == 0:
        return vec
    counts = torch.bincount(node_batch_idx, minlength=batch_size).clamp(min=1).to(vec.dtype)
    sums = torch.zeros((batch_size, 3), dtype=vec.dtype, device=vec.device)
    sums.index_add_(0, node_batch_idx, vec)
    means = sums / counts.unsqueeze(-1)
    return vec - means[node_batch_idx]


def orthogonal_manifold_projection(
    g_prop: torch.Tensor,
    f_phys: torch.Tensor,
    node_batch_idx: torch.Tensor,
    batch_size: int,
    atom_level_projection: bool = True,
    mol_level_projection: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Project property gradient g_prop onto the tangent half-space of the DPA-3
    iso-potential manifold where <g_prop, F_phys> >= 0.

    When g_prop opposes the physical restoring force F_phys (<g_prop, F_phys> < 0),
    the component parallel to -F_phys (which would stretch/compress covalent bonds or
    induce steric clashes) is projected out:
        g_tilde = g_prop - (min(0, <g_prop, F_phys>) / (||F_phys||^2 + eps)) * F_phys

    Args:
        g_prop: (N_total, 3) property guidance gradient (-grad_x E_interval).
        f_phys: (N_total, 3) physical restoring force from DPA-3 (-grad_x E_DPA).
        node_batch_idx: (N_total,) molecule batch indices in [0, B-1].
        batch_size: number of molecules B in batch.
        atom_level_projection: project out opposing components per atom first.
        mol_level_projection: project out opposing components in 3N-D molecule space.
        eps: numerical stability constant.

    Returns:
        g_projected: (N_total, 3) Zero-CoM projected property guidance vector.
        stats: diagnostic dictionary (conflict fraction, cosine similarity, etc.).
    """
    g_proj = zero_com_per_molecule(g_prop, node_batch_idx, batch_size)
    f_clean = zero_com_per_molecule(f_phys, node_batch_idx, batch_size)

    # Diagnostic before projection
    dot_per_atom = (g_proj * f_clean).sum(dim=-1)  # (N_total,)
    mol_dot = torch.zeros(batch_size, dtype=g_proj.dtype, device=g_proj.device)
    mol_dot.index_add_(0, node_batch_idx, dot_per_atom)

    mol_g_norm_sq = torch.zeros(batch_size, dtype=g_proj.dtype, device=g_proj.device)
    mol_g_norm_sq.index_add_(0, node_batch_idx, (g_proj * g_proj).sum(dim=-1))

    mol_f_norm_sq = torch.zeros(batch_size, dtype=g_proj.dtype, device=g_proj.device)
    mol_f_norm_sq.index_add_(0, node_batch_idx, (f_clean * f_clean).sum(dim=-1))

    cos_sim = mol_dot / (torch.sqrt(mol_g_norm_sq * mol_f_norm_sq) + eps)
    conflict_frac = float((mol_dot < 0).float().mean().item()) if batch_size > 0 else 0.0

    # 1. Atom-level orthogonal projection (prevents localized bond-breaking spikes)
    if atom_level_projection:
        atom_dot = (g_proj * f_clean).sum(dim=-1, keepdim=True)  # (N_total, 1)
        atom_f_sq = (f_clean * f_clean).sum(dim=-1, keepdim=True) + eps
        opposing_coeff = torch.clamp(atom_dot, max=0.0) / atom_f_sq
        g_proj = g_proj - opposing_coeff * f_clean

    # 2. Molecule-level 3N-dimensional manifold projection
    if mol_level_projection and batch_size > 0:
        dot_after = (g_proj * f_clean).sum(dim=-1)
        mol_dot_after = torch.zeros(batch_size, dtype=g_proj.dtype, device=g_proj.device)
        mol_dot_after.index_add_(0, node_batch_idx, dot_after)

        mol_coeff = torch.clamp(mol_dot_after, max=0.0) / (mol_f_norm_sq + eps)
        g_proj = g_proj - mol_coeff[node_batch_idx].unsqueeze(-1) * f_clean

    # 3. Restore strict Zero-CoM invariance
    g_proj = zero_com_per_molecule(g_proj, node_batch_idx, batch_size)

    stats = {
        "conflict_frac": conflict_frac,
        "mean_cos_sim": float(cos_sim.mean().item()) if batch_size > 0 else 0.0,
    }
    return g_proj, stats


class EquivariantIntervalConditioner(nn.Module):
    """
    Dynamic Interval Conditioning module with Gaussian RBF expansion.
    Supports both training-time dynamic interval sampling [y - d1, y + d2]
    and inference-time arbitrary interval [y_min, y_max] or point [y*, y*] conditioning.
    """

    def __init__(
        self,
        num_rbf: int = 32,
        r_min: float = -3.5,
        r_max: float = 3.5,
        hidden_dim: int = 128,
        out_dim: int = 64,
        max_half_width: float = 0.5,
        point_prob: float = 0.2,
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.max_half_width = max_half_width
        self.point_prob = point_prob

        centers = torch.linspace(r_min, r_max, num_rbf)
        widths = torch.full((num_rbf,), (r_max - r_min) / num_rbf)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

        # Input features: RBF(y_min), RBF(y_max), RBF(y_mid), and interval width (y_max - y_min)
        in_dim = num_rbf * 3 + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def rbf_expand(self, val: torch.Tensor) -> torch.Tensor:
        """Expand scalar tensor (B,) into Gaussian RBF features (B, num_rbf)."""
        diff = val.unsqueeze(-1) - self.centers.view(1, -1)
        return torch.exp(-((diff / self.widths.view(1, -1)) ** 2))

    def sample_training_intervals(self, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamically sample intervals [y_min, y_max] containing y_true during training."""
        d1 = torch.rand_like(y_true) * self.max_half_width
        d2 = torch.rand_like(y_true) * self.max_half_width
        is_point = (torch.rand_like(y_true) < self.point_prob).float()
        y_min = y_true - d1 * (1.0 - is_point)
        y_max = y_true + d2 * (1.0 - is_point)
        return y_min, y_max

    def forward(self, y_min: torch.Tensor, y_max: torch.Tensor) -> torch.Tensor:
        """Encode interval [y_min, y_max] of shape (B,) into conditioning embedding (B, out_dim)."""
        y_mid = 0.5 * (y_min + y_max)
        width = (y_max - y_min).unsqueeze(-1)
        feat = torch.cat(
            [self.rbf_expand(y_min), self.rbf_expand(y_max), self.rbf_expand(y_mid), width],
            dim=-1,
        )
        return self.mlp(feat)


class HingeIntervalPropertyGuidance(nn.Module):
    """
    Differentiable 3D Equivariant Property Interval Guidance using PropMolFlow's
    pretrained GVPRegressor models.

    Computes the Hinge-Squared Interval Potential:
        E_interval(X) = sum_k w_k * [ max(0, y_min^(k) - y_hat^(k)(X))^2
                                    + max(0, y_hat^(k)(X) - y_max^(k))^2 ]
    and returns the property gradient force:
        g_prop(X) = -grad_X E_interval(X).
    """

    def __init__(
        self,
        propmolflow_root: Union[str, Path],
        property_names: List[str],
        device: str = "cuda:0",
    ):
        super().__init__()
        self.propmolflow_root = Path(propmolflow_root)
        self.property_names = list(property_names)
        self.device_str = device

        from propmolflow.models.ctmc_vector_field import PROPERTY_MAP
        from propmolflow.property_regressor.train_regressor import GVPRegressorModule

        self.property_map = PROPERTY_MAP
        norm_path = self.propmolflow_root / "data/qm9/train_data_property_normalization.pt"
        self.norm_params = torch.load(norm_path, map_location="cpu", weights_only=False)

        self.regressors = nn.ModuleDict()
        for prop in self.property_names:
            model_dir = self.propmolflow_root / "propmolflow/property_regressor/model_output" / prop
            ckpt_files = [
                p for p in (model_dir / "checkpoints").glob("*.ckpt")
                if not p.name.startswith("._")
            ]
            if not ckpt_files:
                raise FileNotFoundError(f"No GVPRegressor checkpoint found in {model_dir / 'checkpoints'}")
            ckpt_path = ckpt_files[0]

            reg = GVPRegressorModule.load_from_checkpoint(
                str(ckpt_path),
                map_location=device,
                weights_only=False,
            )
            reg.to(device)
            reg.eval()
            for p in reg.parameters():
                p.requires_grad_(False)
            self.regressors[prop] = reg

    def normalize_value(self, prop: str, raw_val: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        idx = self.property_map[prop]
        mean = float(self.norm_params["mean"][idx].item())
        std = float(self.norm_params["std"][idx].item())
        return (raw_val - mean) / std

    def denormalize_value(self, prop: str, norm_val: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        idx = self.property_map[prop]
        mean = float(self.norm_params["mean"][idx].item())
        std = float(self.norm_params["std"][idx].item())
        return norm_val * std + mean

    def predict_properties(
        self,
        g: dgl.DGLGraph,
        coords: torch.Tensor,
        atom_probs: torch.Tensor,
        edge_probs_full: torch.Tensor,
        use_hard_discrete: bool = True,
        straight_through_discrete: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate normalized property predictions y_hat(coords) on a DGL graph batch.
        Maintains full differentiability w.r.t. coords (and w.r.t. atom_probs/edge_probs_full
        via Straight-Through Estimator when straight_through_discrete=True).
        """
        if straight_through_discrete:
            a_soft = atom_probs[:, :5]
            a_idx = a_soft.argmax(dim=-1)
            a_hard = torch.nn.functional.one_hot(a_idx, num_classes=5).to(coords.dtype)
            a_feat = a_soft + (a_hard - a_soft).detach()

            e_soft = edge_probs_full[:, :5]
            e_idx = e_soft.argmax(dim=-1)
            e_hard = torch.nn.functional.one_hot(e_idx, num_classes=5).to(coords.dtype)
            e_feat = e_soft + (e_hard - e_soft).detach()
        elif use_hard_discrete:
            # Hard discrete clean atom/bond one-hot features (5 classes)
            a_idx = atom_probs[:, :5].argmax(dim=-1)
            a_feat = torch.nn.functional.one_hot(a_idx, num_classes=5).to(coords.dtype)
            e_idx = edge_probs_full[:, :5].argmax(dim=-1)
            e_feat = torch.nn.functional.one_hot(e_idx, num_classes=5).to(coords.dtype)
        else:
            a_feat = atom_probs[:, :5]
            e_feat = edge_probs_full[:, :5]

        preds = {}
        with g.local_scope():
            g.ndata["x_1_true"] = coords
            g.ndata["a_1_true"] = a_feat
            g.edata["e_1_true"] = e_feat
            for prop, reg in self.regressors.items():
                out = reg(g)
                if out.ndim > 1:
                    out = out.squeeze(-1)
                preds[prop] = out
        return preds

    def compute_hinge_interval_gradient(
        self,
        g: dgl.DGLGraph,
        coords: torch.Tensor,
        atom_probs: torch.Tensor,
        edge_probs_full: torch.Tensor,
        target_intervals_norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        node_batch_idx: torch.Tensor,
        property_weights: Optional[Dict[str, float]] = None,
        max_grad_norm: float = 5.0,
        return_discrete_grads: bool = False,
        return_midpoint_grad: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """
        Compute the Hinge-Squared Interval Energy and its negative gradient w.r.t. coords:
            g_prop = -grad_{coords} E_interval(coords)

        Optionally also computes:
          - Straight-Through discrete simplex gradients g_atom = -grad_{p_A} E_interval,
            g_edge = -grad_{p_E} E_interval (when return_discrete_grads=True).
          - Midpoint attraction gradient j_mid = -grad_{coords} sum_k w_k (y_hat^(k) - y_mid^(k))^2
            for Iso-Property Manifold Tangent Projection during terminal relaxation
            (when return_midpoint_grad=True).
        """
        batch_size = g.batch_size
        extra_grads: Dict[str, torch.Tensor] = {}
        with torch.enable_grad():
            x_in = coords.detach().clone().requires_grad_(True)
            if return_discrete_grads:
                a_in = atom_probs[:, :5].detach().clone().requires_grad_(True)
                e_in = edge_probs_full[:, :5].detach().clone().requires_grad_(True)
                preds_norm = self.predict_properties(
                    g, x_in, a_in, e_in, straight_through_discrete=True
                )
            else:
                a_in = atom_probs.detach()
                e_in = edge_probs_full.detach()
                preds_norm = self.predict_properties(
                    g, x_in, a_in, e_in, use_hard_discrete=True
                )

            hinge_per_mol = torch.zeros(batch_size, dtype=coords.dtype, device=coords.device)
            mid_loss_per_mol = torch.zeros(batch_size, dtype=coords.dtype, device=coords.device)
            for prop, (y_min, y_max) in target_intervals_norm.items():
                y_hat = preds_norm[prop]
                w = 1.0 if property_weights is None else property_weights.get(prop, 1.0)
                below = torch.relu(y_min - y_hat)
                above = torch.relu(y_hat - y_max)
                loss_prop = w * (below.pow(2) + above.pow(2))
                hinge_per_mol = hinge_per_mol + loss_prop
                if return_midpoint_grad:
                    y_mid = 0.5 * (y_min + y_max)
                    mid_loss_per_mol = mid_loss_per_mol + w * (y_hat - y_mid).pow(2)

            total_hinge = hinge_per_mol.sum()
            if return_midpoint_grad:
                total_mid = mid_loss_per_mol.sum()
                grad_mid = torch.autograd.grad(
                    total_mid, x_in, retain_graph=(total_hinge.item() > 0.0), create_graph=False
                )[0]
                j_mid = -grad_mid.detach()
                j_mid = torch.nan_to_num(j_mid, nan=0.0, posinf=0.0, neginf=0.0)
                j_mid = zero_com_per_molecule(j_mid, node_batch_idx, batch_size)
                extra_grads["j_mid"] = j_mid

            if total_hinge.item() > 0.0:
                if return_discrete_grads:
                    grad_x, grad_a, grad_e = torch.autograd.grad(
                        total_hinge, [x_in, a_in, e_in], create_graph=False
                    )
                    g_prop = -grad_x.detach()
                    extra_grads["g_atom"] = -torch.nan_to_num(grad_a.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                    extra_grads["g_edge"] = -torch.nan_to_num(grad_e.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    grad_x = torch.autograd.grad(total_hinge, x_in, create_graph=False)[0]
                    g_prop = -grad_x.detach()
            else:
                g_prop = torch.zeros_like(coords)
                if return_discrete_grads:
                    extra_grads["g_atom"] = torch.zeros_like(atom_probs[:, :5])
                    extra_grads["g_edge"] = torch.zeros_like(edge_probs_full[:, :5])

        g_prop = torch.nan_to_num(g_prop, nan=0.0, posinf=0.0, neginf=0.0)
        g_prop = zero_com_per_molecule(g_prop, node_batch_idx, batch_size)

        # Normalize g_prop per molecule by its RMS atomic gradient norm so lambda_prop
        # has a consistent physical velocity scale (Angstrom / unit time), modulated
        # smoothly by the square-root hinge distance to the target interval.
        if batch_size > 0 and g_prop.numel() > 0:
            mol_counts = torch.bincount(node_batch_idx, minlength=batch_size).clamp(min=1).to(g_prop.dtype)
            mol_g_sq = torch.zeros(batch_size, dtype=g_prop.dtype, device=g_prop.device)
            mol_g_sq.index_add_(0, node_batch_idx, (g_prop * g_prop).sum(dim=-1))
            mol_rms = torch.sqrt(mol_g_sq / mol_counts) + 1e-8
            # Smooth proportional gain near interval boundary (saturates at 1.0 when dist >= 0.25 std)
            hinge_dist = torch.sqrt(hinge_per_mol.detach() + 1e-12)
            gain = torch.clamp(hinge_dist / 0.20, max=1.0)
            scale_per_mol = torch.where(hinge_per_mol.detach() > 0.0, gain / mol_rms, torch.zeros_like(mol_rms))
            g_prop = g_prop * scale_per_mol[node_batch_idx].unsqueeze(-1)

        if max_grad_norm is not None and max_grad_norm > 0:
            gnorm = torch.norm(g_prop, dim=-1, keepdim=True) + 1e-8
            g_prop = g_prop * torch.clamp(max_grad_norm / gnorm, max=1.0)

        g_prop = zero_com_per_molecule(g_prop, node_batch_idx, batch_size)
        preds_detached = {k: v.detach() for k, v in preds_norm.items()}
        if return_discrete_grads or return_midpoint_grad:
            return g_prop, preds_detached, hinge_per_mol.detach(), extra_grads
        return g_prop, preds_detached, hinge_per_mol.detach()
