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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vendor.uni3dar.data.atom_data import atom_list


NUM_COARSE_GEOM_FEATS = 14
NUM_ELEM_CLOUD_FEATS = 42

# 12 OE62 elements in `atom_list` (`0-indexed`: Z - 1)
_OE62_ELEM_IDS: Tuple[int, ...] = tuple(
    atom_list.index(sym) for sym in ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I")
)
_DONOR_ELEM_IDS: Tuple[int, ...] = tuple(atom_list.index(sym) for sym in ("N", "O", "S", "P"))
_ACCEPTOR_ELEM_IDS: Tuple[int, ...] = tuple(atom_list.index(sym) for sym in ("B", "F", "Cl", "Br", "I"))
_HALOGEN_ELEM_IDS: Tuple[int, ...] = tuple(atom_list.index(sym) for sym in ("F", "Cl", "Br", "I"))
_ID_H = atom_list.index("H")
_ID_C = atom_list.index("C")
_ID_N = atom_list.index("N")
_ID_P = atom_list.index("P")


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


def _grouped_utility_ranking_loss(
    out: Dict[str, torch.Tensor],
    targets_norm: torch.Tensor,
    active_mask: torch.Tensor,
    y_surrogate_norm: torch.Tensor,
    y_valid_mask: torch.Tensor,
    is_legal: torch.Tensor,
    is_constr_met: torch.Tensor,
    group_size: int,
    obj_mask: Optional[torch.Tensor] = None,
    *,
    legal_weight: float = 0.90,
    constr_weight: float = 0.65,
    temp: float = 0.35,
) -> torch.Tensor:
    """Group-aligned ListNet KL + Pairwise Logistic Ranking loss on exact utility ordering within each prompt."""
    bsz = int(targets_norm.shape[0])
    if group_size < 2 or bsz < group_size or (bsz % group_size) != 0:
        return torch.zeros((), device=targets_norm.device, dtype=torch.float32)

    mu_n = out["mu_norm"]
    std_n = out["std_norm"]
    eval_m = (
        obj_mask.to(device=mu_n.device, dtype=torch.bool)
        if obj_mask is not None
        else active_mask.to(device=mu_n.device, dtype=torch.bool)
    )
    if eval_m.ndim == 1:
        eval_m = eval_m.unsqueeze(0).expand_as(mu_n)
    denom = eval_m.float().sum(dim=-1).clamp_min(1.0)

    pred_err = ((torch.abs(mu_n - targets_norm.float()) + 0.10 * std_n) * eval_m.float()).sum(dim=-1) / denom
    u_pred = (
        -pred_err
        - float(legal_weight) * (1.8 * F.relu(0.75 - out["prob_legal"]) + 0.15 * (-F.logsigmoid(out["logit_legal"])))
        - float(constr_weight) * (1.8 * F.relu(0.60 - out["prob_constr"]) + 0.20 * (-F.logsigmoid(out["logit_constr"])))
    )

    y_safe = torch.where(torch.isfinite(y_surrogate_norm), y_surrogate_norm.float(), torch.zeros_like(mu_n))
    true_err_raw = (torch.abs(y_safe - targets_norm.float()) * eval_m.float()).sum(dim=-1) / denom
    ok_row = y_valid_mask.bool() & (is_legal >= 0.5) & torch.isfinite(true_err_raw)
    true_err = torch.where(ok_row, true_err_raw.clamp_max(3.5), torch.full_like(true_err_raw, 3.5))
    u_true = -true_err + 1.25 * is_legal.float() + 1.15 * is_constr_met.float()

    n_grp = bsz // group_size
    up_g = u_pred.view(n_grp, group_size)
    ut_g = u_true.view(n_grp, group_size)

    # 1. ListNet soft-top-k KL divergence
    tau = max(float(temp), 0.05)
    p_true = torch.softmax(ut_g / tau, dim=1)
    log_p_pred = F.log_softmax(up_g / tau, dim=1)
    loss_listnet = (p_true * (torch.log(p_true.clamp_min(1e-7)) - log_p_pred)).sum(dim=1).mean()

    # 2. Pairwise margin logistic loss over pairs separated by > 0.15 utility
    du_true = ut_g.unsqueeze(2) - ut_g.unsqueeze(1)  # (G_b, K, K)
    du_pred = up_g.unsqueeze(2) - up_g.unsqueeze(1)
    pair_mask = du_true > 0.15
    if bool(pair_mask.any().item()):
        loss_pair = F.softplus(-2.0 * du_pred[pair_mask]).mean()
    else:
        loss_pair = torch.zeros((), device=targets_norm.device, dtype=torch.float32)

    return 0.50 * loss_listnet + 0.50 * loss_pair


def _build_group_aligned_splits(
    n_total: int,
    group_ids: Optional[torch.Tensor],
    val_fraction: float,
    rng: np.random.Generator,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Return `(tr_groups, val_groups, group_size)` where each tensor has shape `(N_grp, group_size)`, or `(None, None, 0)`."""
    if group_ids is None or n_total < 8:
        return None, None, 0
    g_np = group_ids.detach().cpu().numpy().reshape(-1)
    buckets: Dict[int, List[int]] = {}
    for idx, gid in enumerate(g_np.tolist()):
        buckets.setdefault(int(gid), []).append(idx)
    sizes = [len(v) for v in buckets.values()]
    if not sizes:
        return None, None, 0
    # Keep only intact groups matching the modal group size >= 2
    vals, cnts = np.unique(sizes, return_counts=True)
    g_sz = int(vals[int(np.argmax(cnts))])
    if g_sz < 2:
        return None, None, 0
    intact = [v for v in buckets.values() if len(v) == g_sz]
    if len(intact) < 4:
        return None, None, 0
    grp_arr = np.asarray(intact, dtype=np.int64)
    perm_g = rng.permutation(len(grp_arr))
    n_val_g = max(1, min(len(grp_arr) // 5, int(round(len(grp_arr) * val_fraction))))
    val_g = torch.as_tensor(grp_arr[perm_g[:n_val_g]], dtype=torch.long)
    tr_g = torch.as_tensor(grp_arr[perm_g[n_val_g:]], dtype=torch.long)
    return tr_g, val_g, g_sz


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
    group_ids: Optional[torch.Tensor] = None,
    ranking_weight: float = 0.0,
    ranking_obj_mask: Optional[torch.Tensor] = None,
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
    tr_grp, val_grp, g_sz = _build_group_aligned_splits(n_total, group_ids, val_fraction, rng)
    use_rank = (tr_grp is not None) and (g_sz >= 2) and (float(ranking_weight) > 0.0)
    if tr_grp is not None and val_grp is not None:
        tr_idx = tr_grp.reshape(-1)
        val_idx = val_grp.reshape(-1)
    else:
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
    val_mae = 0.0
    leg_acc = 0.0
    con_acc = 0.0

    for _ep in range(epochs):
        critic.train()
        ep_loss_sum = 0.0
        ep_n = 0
        if use_rank and tr_grp is not None:
            g_perm = tr_grp[torch.randperm(tr_grp.shape[0])]
            g_step = max(1, batch_size // g_sz)
            batch_iter = [g_perm[s : s + g_step].reshape(-1) for s in range(0, g_perm.shape[0], g_step)]
        else:
            shuf = tr_idx[torch.randperm(tr_idx.numel())]
            batch_iter = [shuf[s : s + batch_size] for s in range(0, shuf.numel(), batch_size)]

        for b_ids in batch_iter:
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
            if use_rank:
                loss_rk = _grouped_utility_ranking_loss(
                    out, bt, bm, by, by_ok, bleg, bcon, g_sz, obj_mask=ranking_obj_mask
                )
                loss = loss + float(ranking_weight) * loss_rk
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


def compute_element_cloud_features(
    elem_ids: torch.Tensor,
    leaf_pos: torch.Tensor,
    leaf_valid: torch.Tensor,
    coarse_geom_feats: torch.Tensor,
) -> torch.Tensor:
    """Compute `(B, 42)` SO(3)-invariant 3D chemical composition & coordination features.

    Combines exact element counts/fractions across OE62 elements, ring-and-double-bond
    equivalents (RDBE), closed-shell electron parity, donor/acceptor ratios, 3D voxel
    coordination/clash statistics, and coarse spatial envelope descriptors.
    """
    device = elem_ids.device
    bsz = elem_ids.shape[0]
    with torch.autocast(device_type=device.type, enabled=False):
        out = torch.zeros((bsz, NUM_ELEM_CLOUD_FEATS), dtype=torch.float32, device=device)
        if bsz == 0:
            return out
        out[:, :NUM_COARSE_GEOM_FEATS] = coarse_geom_feats.to(device=device, dtype=torch.float32)
        if elem_ids.shape[1] == 0:
            return out

        valid = leaf_valid.to(device=device, dtype=torch.bool)
        v_f = valid.float()
        e_safe = torch.where(valid, elem_ids.long().clamp_min(0), torch.zeros_like(elem_ids, dtype=torch.long))
        # Atomic number Z = elem_id + 1 on valid slots
        z_num = torch.where(valid, e_safe + 1, torch.zeros_like(e_safe))

        n_tot = v_f.sum(dim=1)
        is_h = valid & (e_safe == _ID_H)
        n_h = is_h.float().sum(dim=1)
        n_heavy = (n_tot - n_h).clamp_min(0.0)
        denom_tot = n_tot.clamp_min(1.0)
        denom_hvy = n_heavy.clamp_min(1.0)

        out[:, 14] = n_tot / 50.0
        out[:, 15] = n_heavy / 35.0
        out[:, 16] = n_h / 35.0
        out[:, 17] = n_h / denom_hvy

        # Closed-shell total electron parity (1.0 = even electrons, 0.0 = odd radical)
        even_elec = ((z_num.sum(dim=1) % 2) == 0).float()
        out[:, 18] = even_elec

        # Element fractions across 12 OE62 elements -> cols 19..30
        for j, eid in enumerate(_OE62_ELEM_IDS):
            c_j = (valid & (e_safe == eid)).float().sum(dim=1)
            out[:, 19 + j] = c_j / denom_tot

        n_c = (valid & (e_safe == _ID_C)).float().sum(dim=1)
        n_n = (valid & (e_safe == _ID_N)).float().sum(dim=1)
        n_p = (valid & (e_safe == _ID_P)).float().sum(dim=1)
        n_hal = torch.zeros_like(n_tot)
        for hid in _HALOGEN_ELEM_IDS:
            n_hal = n_hal + (valid & (e_safe == hid)).float().sum(dim=1)
        # Ring-plus-double-bond equivalents (RDBE): direct measure of pi-conjugation & SA complexity
        rdbe = (2.0 * n_c + 2.0 + n_n + n_p - n_h - n_hal) * 0.5
        out[:, 31] = rdbe.clamp(-4.0, 40.0) / 15.0
        out[:, 32] = (rdbe / denom_hvy).clamp(-0.5, 2.0)

        n_don = torch.zeros_like(n_tot)
        for did in _DONOR_ELEM_IDS:
            n_don = n_don + (valid & (e_safe == did)).float().sum(dim=1)
        n_acc = torch.zeros_like(n_tot)
        for aid in _ACCEPTOR_ELEM_IDS:
            n_acc = n_acc + (valid & (e_safe == aid)).float().sum(dim=1)
        out[:, 33] = n_don / denom_hvy
        out[:, 34] = n_acc / denom_hvy
        out[:, 35] = (n_don - n_acc) / denom_hvy

        # SO(3)-invariant 3D pairwise distance / coordination diagnostics on leaf voxel centers
        k_len = elem_ids.shape[1]
        if k_len >= 2:
            pos = leaf_pos.to(device=device, dtype=torch.float32)
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
            diag_mask = torch.eye(k_len, dtype=torch.bool, device=device).unsqueeze(0)
            pair_valid = pair_valid & (~diag_mask)

            dmat = torch.cdist(pos, pos, p=2.0)
            d_masked = torch.where(pair_valid, dmat, torch.full_like(dmat, 99.0))
            nn_dist = d_masked.min(dim=-1).values  # (B, K)
            nn_valid = valid & (n_tot.unsqueeze(1) >= 2.0)

            clash_cnt = (nn_valid & (nn_dist < 0.75)).float().sum(dim=1)
            iso_cnt = (nn_valid & (nn_dist > 2.25)).float().sum(dim=1)
            cov_deg = (pair_valid & (dmat <= 1.95)).float().sum(dim=-1)  # (B, K)
            mean_deg = (cov_deg * v_f).sum(dim=1) / denom_tot

            hvy_mask = valid & (~is_h)
            hvy_pair = hvy_mask.unsqueeze(2) & hvy_mask.unsqueeze(1) & (~diag_mask)
            hvy_deg = (hvy_pair & (dmat <= 1.95)).float().sum(dim=-1)
            mean_hvy_deg = (hvy_deg * hvy_mask.float()).sum(dim=1) / denom_hvy
            over_coord = (hvy_mask & (hvy_deg > 4.2)).float().sum(dim=1) / denom_hvy
            mean_nn = (torch.where(nn_valid, nn_dist, torch.zeros_like(nn_dist)).sum(dim=1)) / denom_tot

            out[:, 36] = clash_cnt / denom_tot
            out[:, 37] = iso_cnt / denom_tot
            out[:, 38] = mean_deg / 4.0
            out[:, 39] = mean_hvy_deg / 4.0
            out[:, 40] = over_coord
            out[:, 41] = mean_nn / 2.0

        return out


class _RBFElementGraphLayer(nn.Module):
    """SO(3)-invariant 3D RBF distance-biased attention block over decoded leaf elements."""

    def __init__(self, d_model: int = 160, n_heads: int = 4, n_rbf: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.norm1 = nn.LayerNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.edge_proj = nn.Sequential(
            nn.Linear(n_rbf, self.n_heads * 2),
            nn.SiLU(),
            nn.Linear(self.n_heads * 2, self.n_heads),
        )
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
        )

    def forward(self, x: torch.Tensor, rbf_ij: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        bsz, k_len, _ = x.shape
        if k_len == 0:
            return x
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        edge_bias = self.edge_proj(rbf_ij).permute(0, 3, 1, 2)  # (B, H, K, K)
        logits = logits + edge_bias

        key_ok = valid_mask[:, None, None, :]
        has_any = valid_mask.any(dim=-1, keepdim=True)[:, :, None, None]
        logits = torch.where(key_ok | (~has_any), logits, torch.full_like(logits, -1e9))
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, k_len, self.d_model)
        x = x + self.out_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x * valid_mask.unsqueeze(-1).float()


class OctreeElementProcessCritic(nn.Module):
    """Stage-2 Element-First Process Critic (`level == 0` `pred_heads[0]` 3D element cloud).

    Evaluates candidate branches AFTER `pred_heads[0]` reveals all leaf chemical elements
    `(Z_1..Z_K)` at their 3D octree leaf centers `(r_1..r_K)`, BEFORE expensive sub-voxel
    `0.01 A` `XYZ` coordinate refinement, `ValenceReconciler`, and `1.1B` `Uni-Mol2`.
    """

    def __init__(
        self,
        emb_dim: int,
        num_targets: int,
        *,
        num_atom_classes: int = 128,
        num_h_pools: int = 6,
        d_node: int = 160,
        n_rbf: int = 16,
        n_gnn_layers: int = 2,
        hidden_dim: int = 512,
        cloud_feat_dim: int = NUM_ELEM_CLOUD_FEATS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_targets = int(num_targets)
        self.num_atom_classes = int(num_atom_classes)
        self.num_h_pools = int(num_h_pools)
        self.d_node = int(d_node)
        self.n_rbf = int(n_rbf)
        self.n_gnn_layers = max(1, int(n_gnn_layers))
        self.hidden_dim = int(hidden_dim)
        self.cloud_feat_dim = int(cloud_feat_dim)

        self.elem_emb = nn.Embedding(self.num_atom_classes, self.d_node)
        self.register_buffer("rbf_centers", torch.linspace(0.70, 5.50, self.n_rbf, dtype=torch.float32))
        self.register_buffer("rbf_gamma", torch.tensor(6.0, dtype=torch.float32))
        self.gnn_layers = nn.ModuleList(
            [
                _RBFElementGraphLayer(self.d_node, n_heads=4, n_rbf=self.n_rbf, dropout=dropout)
                for _ in range(self.n_gnn_layers)
            ]
        )

        h_in = self.num_h_pools * self.emb_dim
        self.h_norm = nn.LayerNorm(h_in)
        self.h_proj = nn.Sequential(
            nn.Linear(h_in, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # Graph readout: all-atom mean + heavy-atom mean + max pool -> `3 * d_node`
        aux_in = 3 * self.d_node + self.cloud_feat_dim + 2 * self.num_targets
        self.aux_norm = nn.LayerNorm(aux_in)
        self.aux_proj = nn.Sequential(
            nn.Linear(aux_in, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.hidden_dim + 256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.SiLU(),
        )
        feat_d = self.hidden_dim // 2
        self.prop_mean_head = nn.Linear(feat_d, self.num_targets)
        self.prop_logvar_head = nn.Linear(feat_d, self.num_targets)
        self.legal_head = nn.Linear(feat_d, 1)
        self.constr_head = nn.Linear(feat_d, 1)

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
        h_stage2: torch.Tensor,
        elem_ids: torch.Tensor,
        leaf_pos: torch.Tensor,
        leaf_valid: torch.Tensor,
        cloud_feats: torch.Tensor,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = h_stage2.device
        bsz = h_stage2.shape[0]
        with torch.autocast(device_type=device.type, enabled=False):
            valid = leaf_valid.to(device=device, dtype=torch.bool)
            k_len = elem_ids.shape[1]
            if k_len > 0:
                e_clamped = torch.where(
                    valid,
                    elem_ids.to(device=device, dtype=torch.long).clamp(0, self.num_atom_classes - 1),
                    torch.zeros((bsz, k_len), dtype=torch.long, device=device),
                )
                x_node = self.elem_emb(e_clamped) * valid.unsqueeze(-1).float()
                pos = leaf_pos.to(device=device, dtype=torch.float32)
                dmat = torch.cdist(pos, pos, p=2.0).clamp(0.0, 8.0)
                rbf_ij = torch.exp(-self.rbf_gamma * (dmat.unsqueeze(-1) - self.rbf_centers).pow(2))
                for g_layer in self.gnn_layers:
                    x_node = g_layer(x_node, rbf_ij, valid)

                v_f = valid.float()
                cnt_all = v_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                pool_all = (x_node * v_f.unsqueeze(-1)).sum(dim=1) / cnt_all

                hvy_m = (valid & (e_clamped != _ID_H)).float()
                cnt_hvy = hvy_m.sum(dim=1, keepdim=True).clamp_min(1.0)
                pool_hvy = (x_node * hvy_m.unsqueeze(-1)).sum(dim=1) / cnt_hvy

                has_v = valid.any(dim=1, keepdim=True)
                x_masked = torch.where(valid.unsqueeze(-1), x_node, torch.full_like(x_node, -1e4))
                pool_max = torch.where(has_v, x_masked.max(dim=1).values, torch.zeros_like(pool_all))
                g_pool = torch.cat([pool_all, pool_hvy, pool_max], dim=-1)
            else:
                g_pool = torch.zeros((bsz, 3 * self.d_node), dtype=torch.float32, device=device)

            h_vec = self.h_proj(self.h_norm(h_stage2.float()))
            aux_in = torch.cat(
                [g_pool, cloud_feats.float(), targets_norm.float(), active_mask.float()], dim=-1
            )
            aux_vec = self.aux_proj(self.aux_norm(aux_in))
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
        h_stage2: torch.Tensor,
        elem_ids: torch.Tensor,
        leaf_pos: torch.Tensor,
        leaf_valid: torch.Tensor,
        cloud_feats: torch.Tensor,
        targets_norm: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        objective_col_mask: Optional[torch.Tensor] = None,
        legal_weight: float = 1.0,
        constr_weight: float = 0.6,
        uncertainty_bonus: float = 0.0,
        empty_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            out = self.forward(h_stage2, elem_ids, leaf_pos, leaf_valid, cloud_feats, targets_norm, active_mask)
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
            eff_err = abs_err + 0.10 * std_n - float(uncertainty_bonus) * std_n
            denom = eval_mask.float().sum(dim=-1).clamp_min(1.0)
            mean_err = (eff_err * eval_mask.float()).sum(dim=-1) / denom

            prob_leg = out["prob_legal"]
            prob_con = out["prob_constr"]
            leg_shortfall = F.relu(0.75 - prob_leg)
            con_shortfall = F.relu(0.60 - prob_con)
            utility = (
                -mean_err
                - float(legal_weight) * (3.2 * leg_shortfall + 0.08 * (-F.logsigmoid(out["logit_legal"])))
                - float(constr_weight) * (2.6 * con_shortfall + 0.12 * (-F.logsigmoid(out["logit_constr"])))
            )
            # Penalize odd-electron radicals (`cloud_feats[:, 18] < 0.5`) & ultra-tiny leaves
            odd_rad = cloud_feats[:, 18] < 0.5
            tiny_leaf = cloud_feats[:, 14] * 50.0 < 2.5
            utility = torch.where(odd_rad, utility - 25.0, utility)
            utility = torch.where(tiny_leaf, utility - 50.0, utility)
            if empty_mask is not None:
                utility = torch.where(empty_mask.to(utility.device), torch.full_like(utility, -1e9), utility)
            return utility
        finally:
            self.train(was_training)

    def get_config(self) -> Dict[str, Any]:
        return {
            "emb_dim": self.emb_dim,
            "num_targets": self.num_targets,
            "num_atom_classes": self.num_atom_classes,
            "num_h_pools": self.num_h_pools,
            "d_node": self.d_node,
            "n_rbf": self.n_rbf,
            "n_gnn_layers": self.n_gnn_layers,
            "hidden_dim": self.hidden_dim,
            "cloud_feat_dim": self.cloud_feat_dim,
        }

    def save_checkpoint(self, path: str | Any, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            **self.get_config(),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str | Any, device: str | torch.device = "cpu") -> Tuple["OctreeElementProcessCritic", Dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        critic = cls(
            emb_dim=int(payload["emb_dim"]),
            num_targets=int(payload["num_targets"]),
            num_atom_classes=int(payload.get("num_atom_classes", 128)),
            num_h_pools=int(payload.get("num_h_pools", 6)),
            d_node=int(payload.get("d_node", 160)),
            n_rbf=int(payload.get("n_rbf", 16)),
            n_gnn_layers=int(payload.get("n_gnn_layers", 2)),
            hidden_dim=int(payload.get("hidden_dim", 512)),
            cloud_feat_dim=int(payload.get("cloud_feat_dim", NUM_ELEM_CLOUD_FEATS)),
        ).to(device)
        critic.load_state_dict(payload["state_dict"], strict=True)
        critic.eval()
        return critic, payload.get("extra", {})


class TwoStageOctreeProcessCritic(nn.Module):
    """Two-stage Element-First Octree Process Critic (`m_stage1 -> m_stage2 -> m_keep`, e.g., `32 -> 8 -> 2`).

    - `stage1_critic`: `OctreeProcessCritic` at `level 2 -> 1` coarse envelope boundary (`32 -> 8`).
    - `stage2_critic`: `OctreeElementProcessCritic` at `level 0` element-decoded boundary (`8 -> 2`).
    """

    def __init__(self, stage1_critic: OctreeProcessCritic, stage2_critic: OctreeElementProcessCritic) -> None:
        super().__init__()
        self.stage1_critic = stage1_critic
        self.stage2_critic = stage2_critic

    def save_checkpoint(self, path: str | Any, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "stage1_config": {
                "emb_dim": self.stage1_critic.emb_dim,
                "num_targets": self.stage1_critic.num_targets,
                "hidden_dim": self.stage1_critic.hidden_dim,
                "geom_dim": self.stage1_critic.geom_dim,
            },
            "stage1_state_dict": self.stage1_critic.state_dict(),
            "stage2_config": self.stage2_critic.get_config(),
            "stage2_state_dict": self.stage2_critic.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str | Any, device: str | torch.device = "cpu") -> Tuple["TwoStageOctreeProcessCritic", Dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        c1 = payload["stage1_config"]
        s1 = OctreeProcessCritic(
            emb_dim=int(c1["emb_dim"]),
            num_targets=int(c1["num_targets"]),
            hidden_dim=int(c1.get("hidden_dim", 512)),
            geom_dim=int(c1.get("geom_dim", NUM_COARSE_GEOM_FEATS)),
        ).to(device)
        s1.load_state_dict(payload["stage1_state_dict"], strict=True)
        s1.eval()

        c2 = payload["stage2_config"]
        s2 = OctreeElementProcessCritic(
            emb_dim=int(c2["emb_dim"]),
            num_targets=int(c2["num_targets"]),
            num_atom_classes=int(c2.get("num_atom_classes", 128)),
            num_h_pools=int(c2.get("num_h_pools", 6)),
            d_node=int(c2.get("d_node", 160)),
            n_rbf=int(c2.get("n_rbf", 16)),
            n_gnn_layers=int(c2.get("n_gnn_layers", 2)),
            hidden_dim=int(c2.get("hidden_dim", 512)),
            cloud_feat_dim=int(c2.get("cloud_feat_dim", NUM_ELEM_CLOUD_FEATS)),
        ).to(device)
        s2.load_state_dict(payload["stage2_state_dict"], strict=True)
        s2.eval()
        return cls(s1, s2), payload.get("extra", {})


def fit_octree_element_process_critic(
    critic: OctreeElementProcessCritic,
    *,
    h_stage2: torch.Tensor,
    elem_ids: torch.Tensor,
    leaf_pos: torch.Tensor,
    leaf_valid: torch.Tensor,
    cloud_feats: torch.Tensor,
    targets_norm: torch.Tensor,
    active_mask: torch.Tensor,
    y_surrogate_norm: torch.Tensor,
    y_valid_mask: torch.Tensor,
    is_legal: torch.Tensor,
    is_constr_met: Optional[torch.Tensor] = None,
    group_ids: Optional[torch.Tensor] = None,
    ranking_weight: float = 0.0,
    ranking_obj_mask: Optional[torch.Tensor] = None,
    epochs: int = 24,
    batch_size: int = 256,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.10,
    seed: int = 2026,
) -> ProcessCriticTrainStats:
    """Train Stage-2 `OctreeElementProcessCritic` on zero-leakage leaf-element clouds + calibrated surrogate."""
    device = next(critic.parameters()).device
    n_total = h_stage2.shape[0]
    rng = np.random.default_rng(seed)
    tr_grp, val_grp, g_sz = _build_group_aligned_splits(n_total, group_ids, val_fraction, rng)
    use_rank = (tr_grp is not None) and (g_sz >= 2) and (float(ranking_weight) > 0.0)
    if tr_grp is not None and val_grp is not None:
        tr_idx = tr_grp.reshape(-1)
        val_idx = val_grp.reshape(-1)
    else:
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
    val_mae = 0.0
    leg_acc = 0.0
    con_acc = 0.0

    for _ep in range(epochs):
        critic.train()
        ep_loss_sum = 0.0
        ep_n = 0
        if use_rank and tr_grp is not None:
            g_perm = tr_grp[torch.randperm(tr_grp.shape[0])]
            g_step = max(1, batch_size // g_sz)
            batch_iter = [g_perm[s : s + g_step].reshape(-1) for s in range(0, g_perm.shape[0], g_step)]
        else:
            shuf = tr_idx[torch.randperm(tr_idx.numel())]
            batch_iter = [shuf[s : s + batch_size] for s in range(0, shuf.numel(), batch_size)]

        for b_ids in batch_iter:
            bh = h_stage2[b_ids].to(device, non_blocking=True)
            be = elem_ids[b_ids].to(device, non_blocking=True)
            bp = leaf_pos[b_ids].to(device, non_blocking=True)
            bv = leaf_valid[b_ids].to(device, non_blocking=True)
            bc = cloud_feats[b_ids].to(device, non_blocking=True)
            bt = targets_norm[b_ids].to(device, non_blocking=True)
            bm = active_mask[b_ids].to(device, non_blocking=True)
            by = y_surrogate_norm[b_ids].to(device, non_blocking=True)
            by_ok = y_valid_mask[b_ids].to(device, non_blocking=True)
            bleg = is_legal[b_ids].to(device, dtype=torch.float32, non_blocking=True)
            bcon = is_constr_met[b_ids].to(device, dtype=torch.float32, non_blocking=True)

            out = critic(bh, be, bp, bv, bc, bt, bm)
            loss_leg = F.binary_cross_entropy_with_logits(out["logit_legal"], bleg)
            loss_con = F.binary_cross_entropy_with_logits(out["logit_constr"], bcon)

            prop_mask = by_ok.unsqueeze(-1) & bm.bool() & torch.isfinite(by)
            if bool(prop_mask.any().item()):
                diff = out["mu_norm"] - torch.where(prop_mask, by, torch.zeros_like(by))
                huber = F.smooth_l1_loss(out["mu_norm"][prop_mask], by[prop_mask], beta=0.22, reduction="mean")
                nll = 0.5 * (
                    out["logvar_norm"][prop_mask]
                    + diff[prop_mask].pow(2) * torch.exp(-out["logvar_norm"][prop_mask])
                ).mean()
                loss_prop = huber + 0.25 * nll
            else:
                loss_prop = torch.zeros((), device=device)

            loss = loss_prop + 0.45 * loss_leg + 0.40 * loss_con
            if use_rank:
                loss_rk = _grouped_utility_ranking_loss(
                    out, bt, bm, by, by_ok, bleg, bcon, g_sz, obj_mask=ranking_obj_mask
                )
                loss = loss + float(ranking_weight) * loss_rk
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
            val_loss_mae_sum = 0.0
            val_prop_cnt = 0
            val_leg_ok = 0.0
            val_con_ok = 0.0
            for vs in range(0, val_idx.numel(), batch_size):
                v_ids = val_idx[vs : vs + batch_size]
                vout = critic(
                    h_stage2[v_ids].to(device),
                    elem_ids[v_ids].to(device),
                    leaf_pos[v_ids].to(device),
                    leaf_valid[v_ids].to(device),
                    cloud_feats[v_ids].to(device),
                    targets_norm[v_ids].to(device),
                    active_mask[v_ids].to(device),
                )
                vy = y_surrogate_norm[v_ids].to(device)
                vy_ok = y_valid_mask[v_ids].to(device)
                vm = active_mask[v_ids].to(device)
                vleg = is_legal[v_ids].to(device, dtype=torch.float32)
                vcon = is_constr_met[v_ids].to(device, dtype=torch.float32)
                pmask = vy_ok.unsqueeze(-1) & vm.bool() & torch.isfinite(vy)
                if bool(pmask.any().item()):
                    val_loss_mae_sum += float(torch.abs(vout["mu_norm"][pmask] - vy[pmask]).sum().item())
                    val_prop_cnt += int(pmask.sum().item())
                val_leg_ok += float(((vout["prob_legal"] >= 0.5).float() == vleg).float().sum().item())
                val_con_ok += float(((vout["prob_constr"] >= 0.5).float() == vcon).float().sum().item())

            val_mae = val_loss_mae_sum / max(val_prop_cnt, 1)
            leg_acc = val_leg_ok / max(val_idx.numel(), 1)
            con_acc = val_con_ok / max(val_idx.numel(), 1)
            score_cur = val_mae + 0.20 * (1.0 - leg_acc) + 0.15 * (1.0 - con_acc)
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
