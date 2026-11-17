"""Octree Process Critic (`loom v2.0`): Coarse-to-Fine Inference-Time Process Weaving.

Bridges `Uni-3DAR`'s hierarchical coarse-to-fine octree expansion (`level = merge_level .. 1`)
with `Uni-Loom`'s `CompositeCriticBus` without modifying generator weights or architecture:
  1. Extracts pooled 24-layer conditioned Transformer hidden states (`h_root`, `h_coarse_mean`,
     `h_l1_mean`, `h_l1_last`) and 3D coarse voxel envelope geometry (leaf-slot count `K_1`,
     `K_2`, spatial spans `(dx, dy, dz)`, radius of gyration, inertial eigenvalues `lambda_1..3`,
     planarity/linearity ratios, and coarse tree log-perplexity) at the `level = 1 -> 0` boundary
     right before expensive leaf atom & `0.01 A` coordinate decoding (`65%-80%` of compute).
  2. Predicts calibrated conditional property expectation & epistemic/generative spread
     `(mu_proc, log_var_proc)` plus chemical legality (`logit_legal`) and declarative constraint
     feasibility (`logit_constr`, e.g., `Clean + SA_Score <= 3.5`).
  3. Enables mid-generation KV-cache branch-and-prune (`M_init -> M_keep`, e.g., `16 -> 2` or
     `32 -> 4`) before leaf decoding and final `CriticBus` Top-1 selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_COARSE_GEOM_FEATS = 14


def compute_coarse_voxel_geometry(
    decoder_feat: torch.Tensor,
    decoder_phy_pos: torch.Tensor,
    scores: torch.Tensor,
    *,
    atom_mask_id: int,
    atom_level_index: int = 0,
    fine_coarse_level: int = 1,
) -> torch.Tensor:
    """Compute `(B, 14)` 3D spatial & topological descriptors from coarse octree prefixes.

    Vectorized in explicit `float32` (autocast-disabled so batched `eigvalsh` runs natively on CUDA).
    """
    device = decoder_feat.device
    bsz = decoder_feat.shape[0]
    with torch.autocast(device_type=device.type, enabled=False):
        out = torch.zeros((bsz, NUM_COARSE_GEOM_FEATS), dtype=torch.float32, device=device)
        if bsz == 0 or decoder_feat.shape[1] == 0:
            return out

        lvl = decoder_feat[:, :, 4]
        typ = decoder_feat[:, :, 3]
        is_l1_mask = (lvl == fine_coarse_level) & (typ == atom_mask_id)
        is_l2_mask = (lvl == (fine_coarse_level + 1)) & (typ == atom_mask_id)

        k1 = is_l1_mask.sum(dim=1).float()
        k2 = is_l2_mask.sum(dim=1).float()
        out[:, 0] = k1 / 50.0
        out[:, 1] = k2 / 25.0
        out[:, 2] = torch.log1p(k1)

        has_l1 = is_l1_mask.any(dim=1, keepdim=True)
        use_mask = torch.where(has_l1, is_l1_mask, is_l2_mask)
        cnt = use_mask.sum(dim=1)
        ge2 = cnt >= 2
        if bool(ge2.any().item()):
            pts = decoder_phy_pos[:, :, :3].float()
            w = use_mask.float()
            cnt_f = cnt.float().clamp_min(1.0)

            p_max = torch.where(use_mask.unsqueeze(-1), pts, torch.full_like(pts, -1e9)).max(dim=1).values
            p_min = torch.where(use_mask.unsqueeze(-1), pts, torch.full_like(pts, 1e9)).min(dim=1).values
            spans = torch.where(ge2.unsqueeze(-1), (p_max - p_min).clamp_min(0.0), torch.zeros_like(p_max))
            spans_sorted = torch.sort(spans, descending=True, dim=-1).values

            out[:, 3:6] = spans_sorted / 10.0
            out[:, 6] = torch.linalg.norm(spans, dim=-1) / 15.0
            out[:, 7] = (spans[:, 0] * spans[:, 1] * spans[:, 2]) / 500.0

            mean_p = (pts * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / cnt_f.view(-1, 1, 1)
            centered = (pts - mean_p) * w.unsqueeze(-1)
            rg = torch.sqrt(((centered.pow(2).sum(dim=-1) * w).sum(dim=1) / cnt_f).clamp_min(1e-8))
            out[:, 8] = torch.where(ge2, rg / 5.0, torch.zeros_like(rg))

            cov = torch.einsum("bli,blj->bij", centered, centered) / cnt_f.view(-1, 1, 1)
            evals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
            evals_desc = torch.flip(evals, dims=[-1])
            out[:, 9:12] = torch.where(ge2.unsqueeze(-1), evals_desc / 10.0, torch.zeros_like(evals_desc))
            planarity = evals_desc[:, 2] / (evals_desc[:, 0] + 1e-4)
            out[:, 12] = torch.where(ge2, planarity, torch.zeros_like(planarity))

        sc_len = min(scores.shape[1], lvl.shape[1])
        if sc_len > 0:
            sc_s = scores[:, :sc_len].float()
            t_mask = (lvl[:, :sc_len] > atom_level_index) & (sc_s != 0.0)
            t_cnt = t_mask.sum(dim=1)
            mean_neg = (-(sc_s * t_mask.float()).sum(dim=1) / t_cnt.float().clamp_min(1.0)).clamp(0.0, 5.0)
            out[:, 13] = torch.where(t_cnt > 0, mean_neg, torch.zeros_like(mean_neg))

        return out


class OctreeProcessCritic(nn.Module):
    """Lightweight multi-head value & feasibility critic operating on coarse octree states."""

    def __init__(
        self,
        emb_dim: int,
        num_targets: int,
        *,
        hidden_dim: int = 512,
        geom_dim: int = NUM_COARSE_GEOM_FEATS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_targets = int(num_targets)
        self.hidden_dim = int(hidden_dim)
        self.geom_dim = int(geom_dim)

        # Input: 4 pooled Transformer vectors (`h_root`, `h_coarse_mean`, `h_l1_mean`, `h_l1_last`)
        #        + `geom_feats` (14) + `targets_norm` (K) + `active_mask` (K)
        self.h_norm = nn.LayerNorm(4 * self.emb_dim)
        self.h_proj = nn.Sequential(
            nn.Linear(4 * self.emb_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        aux_in = self.geom_dim + 2 * self.num_targets
        self.aux_proj = nn.Sequential(
            nn.Linear(aux_in, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.hidden_dim + 128, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.SiLU(),
        )
        feat_d = self.hidden_dim // 2
        # Predicts normalized property mean & log-variance `(B, num_targets)`
        self.prop_mean_head = nn.Linear(feat_d, self.num_targets)
        self.prop_logvar_head = nn.Linear(feat_d, self.num_targets)
        # Predicts chemical legality (`Clean` non-exotic reconciled) and constraint satisfaction (`SA<=3.5` etc.)
        self.legal_head = nn.Linear(feat_d, 1)
        self.constr_head = nn.Linear(feat_d, 1)

        # Target normalization buffers (mirrors generator `target_mean` / `target_scale` by default)
        self.register_buffer("y_mean", torch.zeros(self.num_targets, dtype=torch.float32))
        self.register_buffer("y_scale", torch.ones(self.num_targets, dtype=torch.float32))

    def set_normalization(self, mean: Sequence[float] | np.ndarray | torch.Tensor, scale: Sequence[float] | np.ndarray | torch.Tensor) -> None:
        m = torch.as_tensor(mean, dtype=torch.float32, device=self.y_mean.device).view(self.num_targets)
        s = torch.as_tensor(scale, dtype=torch.float32, device=self.y_scale.device).view(self.num_targets).clamp_min(1e-6)
        self.y_mean.copy_(m)
        self.y_scale.copy_(s)

    def normalize_y(self, y_raw: torch.Tensor) -> torch.Tensor:
        return (y_raw.to(self.y_mean.device, dtype=torch.float32) - self.y_mean) / self.y_scale

    def denormalize_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm.to(self.y_mean.device, dtype=torch.float32) * self.y_scale + self.y_mean

    def forward(
        self,
        h_concat: torch.Tensor,
        geom_feats: torch.Tensor,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.autocast(device_type=h_concat.device.type, enabled=False):
            h_vec = self.h_proj(self.h_norm(h_concat.float()))
            aux_vec = self.aux_proj(
                torch.cat([geom_feats.float(), targets_norm.float(), active_mask.float()], dim=-1)
            )
            z = self.trunk(torch.cat([h_vec, aux_vec], dim=-1))
            mu_norm = self.prop_mean_head(z)
            logvar_norm = self.prop_logvar_head(z).clamp(-6.0, 3.0)
            logit_legal = self.legal_head(z).squeeze(-1)
            logit_constr = self.constr_head(z).squeeze(-1)
            return {
                "mu_norm": mu_norm,
                "logvar_norm": logvar_norm,
                "std_norm": torch.exp(0.5 * logvar_norm),
                "logit_legal": logit_legal,
                "prob_legal": torch.sigmoid(logit_legal),
                "logit_constr": logit_constr,
                "prob_constr": torch.sigmoid(logit_constr),
            }

    @torch.no_grad()
    def score_utilities(
        self,
        h_concat: torch.Tensor,
        geom_feats: torch.Tensor,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        objective_col_mask: Optional[torch.Tensor] = None,
        legal_weight: float = 1.0,
        constr_weight: float = 0.0,
        uncertainty_bonus: float = 0.0,
        empty_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scalar branch utility `(B,)` for coarse-to-fine Top-`M_keep` pruning."""
        was_training = self.training
        self.eval()
        try:
            out = self.forward(h_concat, geom_feats, targets_norm, active_mask)
            mu_n = out["mu_norm"]
            std_n = out["std_norm"]
            eval_mask = (
                objective_col_mask.to(device=mu_n.device, dtype=torch.bool)
                if objective_col_mask is not None
                else active_mask.to(device=mu_n.device, dtype=torch.bool)
            )
            if eval_mask.ndim == 1:
                eval_mask = eval_mask.unsqueeze(0).expand_as(mu_n)

            abs_err = torch.abs(mu_n - targets_norm.float())
            eff_err = abs_err + 0.12 * std_n - float(uncertainty_bonus) * std_n
            denom = eval_mask.float().sum(dim=-1).clamp_min(1.0)
            mean_err = (eff_err * eval_mask.float()).sum(dim=-1) / denom

            prob_leg = out["prob_legal"]
            prob_con = out["prob_constr"]
            # Lexicographic-style soft gate: penalize low-legality skeletons (`< 0.70`), while letting
            # target accuracy `|mu_n - t|` dominate among chemically viable coarse octrees (`>= 0.70`).
            leg_shortfall = F.relu(0.70 - prob_leg)
            con_shortfall = F.relu(0.55 - prob_con)
            utility = (
                -mean_err
                - float(legal_weight) * (3.0 * leg_shortfall + 0.08 * (-F.logsigmoid(out["logit_legal"])))
                - float(constr_weight) * (2.5 * con_shortfall + 0.10 * (-F.logsigmoid(out["logit_constr"])))
            )
            # Penalize pathological ultra-tiny leaf counts (`K_1 < 2`)
            tiny_leaf = geom_feats[:, 0] * 50.0 < 1.5
            utility = torch.where(tiny_leaf, utility - 50.0, utility)
            if empty_mask is not None:
                utility = torch.where(empty_mask.to(utility.device), torch.full_like(utility, -1e9), utility)
            return utility
        finally:
            self.train(was_training)

    def save_checkpoint(self, path: str | Any, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "emb_dim": self.emb_dim,
            "num_targets": self.num_targets,
            "hidden_dim": self.hidden_dim,
            "geom_dim": self.geom_dim,
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str | Any, device: str | torch.device = "cpu") -> Tuple["OctreeProcessCritic", Dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        critic = cls(
            emb_dim=int(payload["emb_dim"]),
            num_targets=int(payload["num_targets"]),
            hidden_dim=int(payload.get("hidden_dim", 512)),
            geom_dim=int(payload.get("geom_dim", NUM_COARSE_GEOM_FEATS)),
        ).to(device)
        critic.load_state_dict(payload["state_dict"], strict=True)
        critic.eval()
        return critic, payload.get("extra", {})


@dataclass
class ProcessCriticTrainStats:
    epochs: int
    train_loss: float
    val_prop_mae_norm: float
    val_legal_acc: float
    val_constr_acc: float


def fit_octree_process_critic(
    critic: OctreeProcessCritic,
    *,
    h_concat: torch.Tensor,
    geom_feats: torch.Tensor,
    targets_norm: torch.Tensor,
    active_mask: torch.Tensor,
    y_surrogate_norm: torch.Tensor,
    y_valid_mask: torch.Tensor,
    is_legal: torch.Tensor,
    is_constr_met: Optional[torch.Tensor] = None,
    epochs: int = 18,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.10,
    seed: int = 2026,
) -> ProcessCriticTrainStats:
    """Train `OctreeProcessCritic` on zero-leakage rollout coarse prefixes + surrogate outcomes."""
    device = next(critic.parameters()).device
    n_total = h_concat.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_val = max(1, min(n_total // 5, int(round(n_total * val_fraction)))) if n_total >= 10 else 0
    val_idx = torch.as_tensor(perm[:n_val], dtype=torch.long) if n_val > 0 else torch.arange(n_total, dtype=torch.long)
    tr_idx = torch.as_tensor(perm[n_val:], dtype=torch.long) if n_val > 0 else torch.arange(n_total, dtype=torch.long)

    if is_constr_met is None:
        is_constr_met = is_legal.clone()

    opt = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.05)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_metric = float("inf")
    last_tr_loss = 0.0

    for _ep in range(epochs):
        critic.train()
        shuf = tr_idx[torch.randperm(tr_idx.numel())]
        ep_loss_sum = 0.0
        ep_n = 0
        for s in range(0, shuf.numel(), batch_size):
            b_ids = shuf[s : s + batch_size]
            bh = h_concat[b_ids].to(device, non_blocking=True)
            bg = geom_feats[b_ids].to(device, non_blocking=True)
            bt = targets_norm[b_ids].to(device, non_blocking=True)
            bm = active_mask[b_ids].to(device, non_blocking=True)
            by = y_surrogate_norm[b_ids].to(device, non_blocking=True)
            by_ok = y_valid_mask[b_ids].to(device, non_blocking=True)
            bleg = is_legal[b_ids].to(device, dtype=torch.float32, non_blocking=True)
            bcon = is_constr_met[b_ids].to(device, dtype=torch.float32, non_blocking=True)

            out = critic(bh, bg, bt, bm)
            loss_leg = F.binary_cross_entropy_with_logits(out["logit_legal"], bleg)
            loss_con = F.binary_cross_entropy_with_logits(out["logit_constr"], bcon)

            # Heteroscedastic Huber + Gaussian NLL on valid legal rows & active columns
            prop_mask = by_ok.unsqueeze(-1) & bm.bool() & torch.isfinite(by)
            if bool(prop_mask.any().item()):
                diff = out["mu_norm"] - torch.where(prop_mask, by, torch.zeros_like(by))
                huber = F.smooth_l1_loss(
                    out["mu_norm"][prop_mask], by[prop_mask], beta=0.25, reduction="mean"
                )
                nll = 0.5 * (
                    out["logvar_norm"][prop_mask]
                    + diff[prop_mask].pow(2) * torch.exp(-out["logvar_norm"][prop_mask])
                ).mean()
                loss_prop = huber + 0.25 * nll
            else:
                loss_prop = torch.zeros((), device=device)

            loss = loss_prop + 0.50 * loss_leg + 0.35 * loss_con
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 2.0)
            opt.step()

            bsz_cur = b_ids.numel()
            ep_loss_sum += float(loss.item()) * bsz_cur
            ep_n += bsz_cur

        sched.step()
        last_tr_loss = ep_loss_sum / max(ep_n, 1)

        critic.eval()
        with torch.no_grad():
            vh = h_concat[val_idx].to(device)
            vg = geom_feats[val_idx].to(device)
            vt = targets_norm[val_idx].to(device)
            vm = active_mask[val_idx].to(device)
            vy = y_surrogate_norm[val_idx].to(device)
            vy_ok = y_valid_mask[val_idx].to(device)
            vleg = is_legal[val_idx].to(device, dtype=torch.float32)
            vcon = is_constr_met[val_idx].to(device, dtype=torch.float32)

            vout = critic(vh, vg, vt, vm)
            pmask = vy_ok.unsqueeze(-1) & vm.bool() & torch.isfinite(vy)
            val_mae = (
                float(torch.abs(vout["mu_norm"][pmask] - vy[pmask]).mean().item())
                if bool(pmask.any().item())
                else 0.0
            )
            leg_acc = float(((vout["prob_legal"] >= 0.5).float() == vleg).float().mean().item())
            con_acc = float(((vout["prob_constr"] >= 0.5).float() == vcon).float().mean().item())
            score_cur = val_mae + 0.25 * (1.0 - leg_acc)
            if score_cur <= best_val_metric:
                best_val_metric = score_cur
                best_state = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}

    if best_state is not None:
        critic.load_state_dict(best_state, strict=True)
    critic.eval()
    return ProcessCriticTrainStats(
        epochs=epochs,
        train_loss=round(last_tr_loss, 5),
        val_prop_mae_norm=round(val_mae, 5),
        val_legal_acc=round(leg_acc, 4),
        val_constr_acc=round(con_acc, 4),
    )
