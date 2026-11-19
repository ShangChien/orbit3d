"""Single-Pass SE(3)-Equivariant Post-Tree Bidirectional Refiner & Bilinear Bond-Order Head (`v4 Step 3`).

Closes autoregressive octree Morton-seam ring distortions (`<= 0.08 A` radial-tanh bounded equivariant shift)
and predicts explicit symmetric bond-order probabilities `{none, single, double, triple, aromatic}` in a
single `< 0.5 ms` batched forward pass (zero iterative MLIP / diffusion loops at inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vendor.uni3dar.data.atom_data import atom_list

BOND_CLASS_NONE = 0
BOND_CLASS_SINGLE = 1
BOND_CLASS_DOUBLE = 2
BOND_CLASS_TRIPLE = 3
BOND_CLASS_AROMATIC = 4
NUM_BOND_CLASSES = 5

_SYMBOL_TO_ELEM_ID: Dict[str, int] = {sym: idx for idx, sym in enumerate(atom_list)}


class _EquivariantBiLayer(nn.Module):
    """One non-causal bidirectional 3D-RBF attention + SE(3)-equivariant radial force update block."""

    def __init__(self, d_model: int = 160, n_heads: int = 4, n_rbf: int = 24, dropout: float = 0.05) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads

        self.norm1 = nn.LayerNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.rbf_attn_bias = nn.Sequential(
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

        # Pairwise scalar force magnitude along `(r_j - r_i) / ||r_j - r_i||` (zero-init at step 0)
        self.coord_src = nn.Linear(self.d_model, 64)
        self.coord_dst = nn.Linear(self.d_model, 64)
        self.coord_rbf = nn.Linear(n_rbf, 64)
        self.coord_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        valid_mask: torch.Tensor,
        rbf_centers: torch.Tensor,
        rbf_gamma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, k_len, _ = h.shape
        if k_len <= 1:
            return h, pos

        # Pairwise displacements `u_ij = pos_j - pos_i` -> `(B, K, K, 3)`
        u_ij = pos.unsqueeze(1) - pos.unsqueeze(2)
        d_ij = torch.linalg.norm(u_ij, dim=-1)
        rbf_ij = torch.exp(-rbf_gamma * (d_ij.clamp(0.0, 6.0).unsqueeze(-1) - rbf_centers).pow(2))

        # 1. Non-causal bidirectional 3D-RBF attention update on `h`
        xn = self.norm1(h)
        q, k, v = self.qkv(xn).chunk(3, dim=-1)
        q = q.view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        logits = logits + self.rbf_attn_bias(rbf_ij).permute(0, 3, 1, 2)

        key_ok = valid_mask[:, None, None, :]
        has_any = valid_mask.any(dim=-1, keepdim=True)[:, :, None, None]
        logits = torch.where(key_ok | (~has_any), logits, torch.full_like(logits, -1e9))
        attn = torch.softmax(logits, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, k_len, self.d_model)

        h = h + self.out_proj(attn_out)
        h = h + self.ffn(self.norm2(h))
        h = h * valid_mask.unsqueeze(-1).float()

        # 2. SE(3)-equivariant coordinate update on local covalent/geminal/ring neighborhood (`0.15 < d_ij < 3.80 A`)
        not_self = ~torch.eye(k_len, dtype=torch.bool, device=h.device).unsqueeze(0)
        pair_ok = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1) & not_self & (d_ij > 0.15) & (d_ij < 3.80)

        edge_feat = self.coord_src(h).unsqueeze(2) + self.coord_dst(h).unsqueeze(1) + self.coord_rbf(rbf_ij)
        w_ij = self.coord_mlp(edge_feat).squeeze(-1) * pair_ok.float()
        unit_ij = u_ij / d_ij.clamp_min(1e-4).unsqueeze(-1)
        denom = pair_ok.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        delta_pos = (w_ij.unsqueeze(-1) * unit_ij).sum(dim=2) / denom
        pos_next = pos + delta_pos * valid_mask.unsqueeze(-1).float()
        return h, pos_next


class BilinearBondHead(nn.Module):
    """SE(3)-invariant symmetric bilinear + 3D-RBF explicit bond-order classifier (`5` classes)."""

    def __init__(self, d_model: int = 160, n_rbf: int = 24, d_pair: int = 128) -> None:
        super().__init__()
        self.proj_mul = nn.Linear(d_model, d_pair, bias=False)
        self.proj_add = nn.Linear(d_model, d_pair)
        self.proj_rbf = nn.Linear(n_rbf, d_pair, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.SiLU(),
            nn.Linear(d_pair, d_pair // 2),
            nn.SiLU(),
            nn.Linear(d_pair // 2, NUM_BOND_CLASSES),
        )

    def forward(self, h: torch.Tensor, rbf_ij: torch.Tensor) -> torch.Tensor:
        hm = self.proj_mul(h)
        ha = self.proj_add(h)
        # Symmetric in `(i, j)` by construction!
        z_ij = hm.unsqueeze(2) * hm.unsqueeze(1) + ha.unsqueeze(2) + ha.unsqueeze(1) + self.proj_rbf(rbf_ij)
        return self.head(z_ij)


class PostTreeGraphRefiner(nn.Module):
    """Single post-tree bidirectional 3D coordinate & explicit bond-order refiner."""

    def __init__(
        self,
        *,
        num_atom_classes: int = 128,
        d_model: int = 160,
        n_heads: int = 4,
        n_layers: int = 2,
        n_rbf: int = 24,
        d_pair: int = 128,
        max_disp_A: float = 0.08,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_atom_classes = int(num_atom_classes)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.n_rbf = int(n_rbf)
        self.d_pair = int(d_pair)
        self.max_disp_A = float(max_disp_A)

        self.elem_emb = nn.Embedding(self.num_atom_classes, self.d_model)
        self.register_buffer("rbf_centers", torch.linspace(0.65, 4.50, self.n_rbf, dtype=torch.float32))
        self.register_buffer("rbf_gamma", torch.tensor(10.0, dtype=torch.float32))

        self.layers = nn.ModuleList(
            [_EquivariantBiLayer(self.d_model, n_heads=self.n_heads, n_rbf=self.n_rbf, dropout=dropout) for _ in range(self.n_layers)]
        )
        self.bond_head = BilinearBondHead(self.d_model, n_rbf=self.n_rbf, d_pair=self.d_pair)

    def forward(
        self,
        elem_ids: torch.Tensor,
        pos: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = pos.device
        with torch.autocast(device_type=device.type, enabled=False):
            valid = valid_mask.to(device=device, dtype=torch.bool)
            pos_0 = pos.to(device=device, dtype=torch.float32)
            bsz, k_len = elem_ids.shape
            if k_len == 0:
                empty_b = torch.zeros((bsz, 0, 0, NUM_BOND_CLASSES), dtype=torch.float32, device=device)
                return {"refined_pos": pos_0, "delta_pos": torch.zeros_like(pos_0), "bond_logits": empty_b}

            e_safe = torch.where(
                valid,
                elem_ids.to(device=device, dtype=torch.long).clamp(0, self.num_atom_classes - 1),
                torch.zeros((bsz, k_len), dtype=torch.long, device=device),
            )
            h = self.elem_emb(e_safe) * valid.unsqueeze(-1).float()
            cur_pos = pos_0
            for layer in self.layers:
                h, cur_pos = layer(h, cur_pos, valid, self.rbf_centers, self.rbf_gamma)

            raw_delta = (cur_pos - pos_0) * valid.unsqueeze(-1).float()
            # Radial-tanh norm bound preserves exact SO(3) rotational equivariance!
            d_norm = torch.linalg.norm(raw_delta, dim=-1, keepdim=True)
            scale = self.max_disp_A * torch.tanh(d_norm / max(self.max_disp_A, 1e-4)) / d_norm.clamp_min(1e-6)
            bounded_delta = raw_delta * scale * valid.unsqueeze(-1).float()
            refined_pos = pos_0 + bounded_delta

            d_ref = torch.cdist(refined_pos, refined_pos, p=2.0).clamp(0.0, 6.0)
            rbf_ref = torch.exp(-self.rbf_gamma * (d_ref.unsqueeze(-1) - self.rbf_centers).pow(2))
            bond_logits = self.bond_head(h, rbf_ref)
            return {
                "refined_pos": refined_pos,
                "delta_pos": bounded_delta,
                "bond_logits": bond_logits,
            }

    @torch.no_grad()
    def refine_trajectories_batch(
        self,
        trajs: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 256,
    ) -> None:
        """In-place batched GPU coordinate refinement + bond-order prior attachment (`<0.5 ms` per batch)."""
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            n_tot = len(trajs)
            for s in range(0, n_tot, batch_size):
                sub = trajs[s : s + batch_size]
                k_lens = [len(tr["atom_symbols"]) for tr in sub]
                max_k = max(k_lens, default=0)
                if max_k == 0:
                    continue
                bsz = len(sub)
                pad_e = torch.zeros((bsz, max_k), dtype=torch.long, device=device)
                pad_p = torch.zeros((bsz, max_k, 3), dtype=torch.float32, device=device)
                pad_v = torch.zeros((bsz, max_k), dtype=torch.bool, device=device)
                for i, tr in enumerate(sub):
                    ki = k_lens[i]
                    if ki == 0:
                        continue
                    e_list = [_SYMBOL_TO_ELEM_ID.get(sym, 5) for sym in tr["atom_symbols"]]
                    pad_e[i, :ki] = torch.tensor(e_list, dtype=torch.long, device=device)
                    pad_p[i, :ki] = torch.as_tensor(tr["positions_np"], dtype=torch.float32, device=device)
                    pad_v[i, :ki] = True

                out = self.forward(pad_e, pad_p, pad_v)
                ref_pos_cpu = out["refined_pos"].detach().cpu().numpy()
                probs_cpu = torch.softmax(out["bond_logits"], dim=-1).detach().cpu().numpy()

                for i, tr in enumerate(sub):
                    ki = k_lens[i]
                    if ki > 0:
                        tr["positions_np"] = ref_pos_cpu[i, :ki].astype(np.float32)
                        tr["bond_order_probs"] = probs_cpu[i, :ki, :ki].astype(np.float32)
        finally:
            self.train(was_training)

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_atom_classes": self.num_atom_classes,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "n_rbf": self.n_rbf,
            "d_pair": self.d_pair,
            "max_disp_A": self.max_disp_A,
        }

    def save_checkpoint(self, path: str | Any, extra: Optional[Dict[str, Any]] = None) -> None:
        torch.save(
            {
                **self.get_config(),
                "state_dict": self.state_dict(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Any, device: str | torch.device = "cpu") -> Tuple["PostTreeGraphRefiner", Dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            num_atom_classes=int(payload.get("num_atom_classes", 128)),
            d_model=int(payload.get("d_model", 160)),
            n_heads=int(payload.get("n_heads", 4)),
            n_layers=int(payload.get("n_layers", 2)),
            n_rbf=int(payload.get("n_rbf", 24)),
            d_pair=int(payload.get("d_pair", 128)),
            max_disp_A=float(payload.get("max_disp_A", 0.08)),
        ).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        return model, payload.get("extra", {})


def extract_mol_bond_targets(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract `(elem_ids, eq_pos, bond_class_mat)` from a sanitized 3D `Chem.Mol`."""
    n = mol.GetNumAtoms()
    elem_ids = np.array([_SYMBOL_TO_ELEM_ID.get(a.GetSymbol(), 5) for a in mol.GetAtoms()], dtype=np.int64)
    eq_pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    bmat = np.zeros((n, n), dtype=np.int64)
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = b.GetBondType()
        if b.GetIsAromatic() or bt == Chem.BondType.AROMATIC:
            cls_id = BOND_CLASS_AROMATIC
        elif bt == Chem.BondType.TRIPLE:
            cls_id = BOND_CLASS_TRIPLE
        elif bt == Chem.BondType.DOUBLE:
            cls_id = BOND_CLASS_DOUBLE
        else:
            cls_id = BOND_CLASS_SINGLE
        bmat[u, v] = bmat[v, u] = cls_id
    return elem_ids, eq_pos, bmat


def kabsch_align_coords(src_pos: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
    """Superimpose `src_pos` onto `ref_pos` via proper SO(3) rotation + translation (`det(R) == +1`)."""
    p = np.asarray(src_pos, dtype=np.float64)
    q = np.asarray(ref_pos, dtype=np.float64)
    if p.shape[0] < 3:
        return (p - p.mean(axis=0, keepdims=True) + q.mean(axis=0, keepdims=True)).astype(np.float32)
    pc = p.mean(axis=0, keepdims=True)
    qc = q.mean(axis=0, keepdims=True)
    p0 = p - pc
    q0 = q - qc
    cov = p0.T @ q0
    u_mat, _, vt_mat = np.linalg.svd(cov)
    d_sign = 1.0 if np.linalg.det(u_mat @ vt_mat) >= 0.0 else -1.0
    diag = np.diag([1.0, 1.0, d_sign])
    r_opt = u_mat @ diag @ vt_mat
    return (p0 @ r_opt + qc).astype(np.float32)


def simulate_morton_seam_perturbation(
    eq_pos: np.ndarray,
    rng: np.random.Generator,
    seam_disp_std_A: Optional[float] = None,
    atom_jitter_std_A: Optional[float] = None,
) -> np.ndarray:
    """Apply multi-scale octree `0.01 A` quantization + octant Morton-seam & ring-closure shift."""
    s_std = float(seam_disp_std_A) if seam_disp_std_A is not None else float(rng.uniform(0.030, 0.125))
    a_std = float(atom_jitter_std_A) if atom_jitter_std_A is not None else float(rng.uniform(0.018, 0.070))
    center = eq_pos.mean(axis=0, keepdims=True)
    rel = eq_pos - center
    # Random SO(3) orientation for octant seam plane
    q_mat, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q_mat) < 0:
        q_mat[:, 0] = -q_mat[:, 0]
    rel_rot = rel @ q_mat
    octant_id = ((rel_rot[:, 0] >= 0).astype(np.int32) * 4 + (rel_rot[:, 1] >= 0).astype(np.int32) * 2 + (rel_rot[:, 2] >= 0).astype(np.int32))
    octant_shifts = rng.normal(0.0, s_std, size=(8, 3)).astype(np.float32)
    octant_shifts -= octant_shifts.mean(axis=0, keepdims=True)
    noisy_rot = rel_rot + octant_shifts[octant_id] + rng.normal(0.0, a_std, size=eq_pos.shape)
    quant_rot = np.round(noisy_rot / 0.01) * 0.01
    noisy_out = (quant_rot @ q_mat.T + center).astype(np.float32)
    return noisy_out


@dataclass
class PostTreeRefinerTrainStats:
    epochs: int
    train_loss: float
    val_bond_dist_mae_A: float
    val_noisy_bond_dist_mae_A: float
    val_bond_order_acc: float


def fit_post_tree_graph_refiner(
    refiner: PostTreeGraphRefiner,
    mols: Sequence[Chem.Mol],
    *,
    paired_trajectories: Optional[Sequence[Tuple[np.ndarray, Chem.Mol]]] = None,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1.5e-3,
    val_fraction: float = 0.08,
    seed: int = 2026,
) -> PostTreeRefinerTrainStats:
    """Distill `PostTreeGraphRefiner` on real `(raw_sampled -> Kabsch DPA-4 relaxed)` pairs + multi-scale seam noise."""
    device = next(refiner.parameters()).device
    rng = np.random.default_rng(seed)

    # Record tuple: `(elem_ids, eq_pos, bond_mat, optional_real_noisy_pos)`
    records: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    for m in mols:
        if m is None or m.GetNumConformers() == 0 or m.GetNumAtoms() < 3 or m.GetNumAtoms() > 80:
            continue
        ei, eqi, bi = extract_mol_bond_targets(m)
        records.append((ei, eqi, bi, None))

    if paired_trajectories:
        for raw_pos, eq_mol in paired_trajectories:
            if eq_mol is None or eq_mol.GetNumConformers() == 0:
                continue
            k_m = eq_mol.GetNumAtoms()
            raw_arr = np.asarray(raw_pos, dtype=np.float32)
            if k_m < 3 or k_m > 80 or raw_arr.shape != (k_m, 3) or not np.isfinite(raw_arr).all():
                continue
            ei, eqi, bi = extract_mol_bond_targets(eq_mol)
            eq_aligned = kabsch_align_coords(eqi, raw_arr)
            # Duplicate real on-policy pairs 2x so real autoregressive ring/seam errors dominate distillation!
            records.append((ei, eq_aligned, bi, raw_arr))
            records.append((ei, eq_aligned, bi, raw_arr))

    n_tot = len(records)
    perm = rng.permutation(n_tot)
    n_val = max(1, min(512, int(round(n_tot * val_fraction)))) if n_tot >= 16 else 0
    val_idx = perm[:n_val].tolist() if n_val > 0 else perm.tolist()
    tr_idx = perm[n_val:].tolist() if n_val > 0 else perm.tolist()

    class_weights = torch.tensor([0.12, 1.0, 2.2, 3.0, 1.8], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.05)

    def _pack_subset(indices: List[int], ep_seed: int, *, augment_so3: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sub_rng = np.random.default_rng(ep_seed)
        max_k = max(len(records[i][0]) for i in indices)
        bsz = len(indices)
        pe = np.zeros((bsz, max_k), dtype=np.int64)
        p_noisy = np.zeros((bsz, max_k, 3), dtype=np.float32)
        p_eq = np.zeros((bsz, max_k, 3), dtype=np.float32)
        pv = np.zeros((bsz, max_k), dtype=bool)
        pb = np.zeros((bsz, max_k, max_k), dtype=np.int64)
        for loc_i, idx in enumerate(indices):
            ei, eqi, bi, real_ni = records[idx]
            ki = len(ei)
            if real_ni is not None:
                ni = real_ni.copy()
                eq_use = eqi.copy()
                if augment_so3:
                    q_rot, _ = np.linalg.qr(sub_rng.normal(size=(3, 3)))
                    if np.linalg.det(q_rot) < 0:
                        q_rot[:, 0] = -q_rot[:, 0]
                    cen = ni.mean(axis=0, keepdims=True)
                    ni = ((ni - cen) @ q_rot.T + cen).astype(np.float32)
                    eq_use = ((eq_use - cen) @ q_rot.T + cen).astype(np.float32)
            else:
                ni = simulate_morton_seam_perturbation(eqi, sub_rng)
                eq_use = kabsch_align_coords(eqi, ni)
            pe[loc_i, :ki] = ei
            p_eq[loc_i, :ki] = eq_use
            p_noisy[loc_i, :ki] = ni
            pv[loc_i, :ki] = True
            pb[loc_i, :ki, :ki] = bi
        return (
            torch.from_numpy(pe).to(device),
            torch.from_numpy(p_noisy).to(device),
            torch.from_numpy(p_eq).to(device),
            torch.from_numpy(pv).to(device),
            torch.from_numpy(pb).to(device),
        )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    last_loss = 0.0
    val_mae_ref = 0.0
    val_mae_noisy = 0.0
    val_b_acc = 0.0

    for ep in range(1, epochs + 1):
        refiner.train()
        tr_shuf = rng.permutation(tr_idx).tolist()
        ep_loss_sum = 0.0
        ep_n = 0
        for s in range(0, len(tr_shuf), batch_size):
            b_ids = tr_shuf[s : s + batch_size]
            be, b_noisy, b_eq, bv, bb = _pack_subset(b_ids, seed * 31 + ep * 997 + s, augment_so3=True)
            out = refiner(be, b_noisy, bv)
            ref_p = out["refined_pos"]
            blogits = out["bond_logits"]

            k_cur = be.shape[1]
            upper = torch.triu(torch.ones((k_cur, k_cur), dtype=torch.bool, device=device), diagonal=1).unsqueeze(0)
            pair_ok = bv.unsqueeze(2) & bv.unsqueeze(1) & upper

            d_eq = torch.cdist(b_eq, b_eq, p=2.0)
            d_ref = torch.sqrt(((ref_p.unsqueeze(2) - ref_p.unsqueeze(1)).square().sum(dim=-1)).clamp_min(1e-8))

            # Supervise 1-2 covalent, 1-3 geminal angle, and 1-4 ring closure distances (`d_eq < 3.60 A`) + Kabsch Huber
            local_m = pair_ok & ((bb > 0) | (d_eq < 3.60))
            if bool(local_m.any().item()):
                is_bond = bb[local_m] > 0
                d_err = F.smooth_l1_loss(d_ref[local_m], d_eq[local_m], beta=0.015, reduction="none")
                w_d = torch.where(is_bond, torch.full_like(d_err, 4.0), torch.full_like(d_err, 1.35))
                loss_dist = (d_err * w_d).sum() / w_d.sum().clamp_min(1.0)
            else:
                loss_dist = torch.zeros((), device=device)

            loss_coord = F.smooth_l1_loss(ref_p[bv], b_eq[bv], beta=0.025)

            # Explicit bond-order cross-entropy on close pairs (`d_eq < 3.4 A` so non-bonds near covalent cutoff are crisp)
            cand_edge = pair_ok & ((bb > 0) | (d_eq < 3.40))
            if bool(cand_edge.any().item()):
                loss_bond = F.cross_entropy(blogits[cand_edge], bb[cand_edge], weight=class_weights)
            else:
                loss_bond = torch.zeros((), device=device)

            loss = 2.8 * loss_dist + 1.0 * loss_coord + 0.5 * loss_bond
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 2.0)
            opt.step()

            ep_loss_sum += float(loss.item()) * len(b_ids)
            ep_n += len(b_ids)

        sched.step()
        last_loss = ep_loss_sum / max(ep_n, 1)

        refiner.eval()
        with torch.no_grad():
            ve, v_noisy, v_eq, vv, vb = _pack_subset(val_idx, seed * 101 + 7)
            vout = refiner(ve, v_noisy, vv)
            vp = vout["refined_pos"]
            vbl = vout["bond_logits"]
            vk = ve.shape[1]
            v_up = torch.triu(torch.ones((vk, vk), dtype=torch.bool, device=device), diagonal=1).unsqueeze(0)
            v_pair = vv.unsqueeze(2) & vv.unsqueeze(1) & v_up
            vd_eq = torch.cdist(v_eq, v_eq, p=2.0)
            vd_noisy = torch.cdist(v_noisy, v_noisy, p=2.0)
            vd_ref = torch.cdist(vp, vp, p=2.0)
            vbond_m = v_pair & (vb > 0)
            if bool(vbond_m.any().item()):
                val_mae_noisy = float(torch.abs(vd_noisy[vbond_m] - vd_eq[vbond_m]).mean().item())
                val_mae_ref = float(torch.abs(vd_ref[vbond_m] - vd_eq[vbond_m]).mean().item())
                val_b_acc = float((vbl[vbond_m].argmax(dim=-1) == vb[vbond_m]).float().mean().item())
            if val_mae_ref <= best_val:
                best_val = val_mae_ref
                best_state = {k: t.detach().cpu().clone() for k, t in refiner.state_dict().items()}

    if best_state is not None:
        refiner.load_state_dict(best_state, strict=True)
    refiner.eval()
    return PostTreeRefinerTrainStats(
        epochs=epochs,
        train_loss=round(last_loss, 5),
        val_bond_dist_mae_A=round(val_mae_ref, 5),
        val_noisy_bond_dist_mae_A=round(val_mae_noisy, 5),
        val_bond_order_acc=round(val_b_acc, 4),
    )
