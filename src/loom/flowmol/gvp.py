"""Pure-PyTorch (DGL-free) SE(3)-equivariant GVP blocks with Ampere FP32 geometry whitelist & LoRA hooks."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_no_nan_fp32(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True,
) -> torch.Tensor:
    """Clamped L2 norm strictly evaluated in FP32 to prevent BF16 underflow / NaN gradients."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.float()
        sq = torch.sum(xf * xf, dim=dim, keepdim=keepdim).clamp_min(eps)
        return torch.sqrt(sq) if sqrt else sq


def _rbf_fp32(
    d: torch.Tensor,
    d_min: float = 0.0,
    d_max: float = 16.0,
    d_count: int = 32,
) -> torch.Tensor:
    """Gaussian radial basis expansion strictly evaluated in FP32."""
    with torch.autocast(device_type=d.device.type, enabled=False):
        df = d.float()
        mu = torch.linspace(d_min, d_max, d_count, dtype=torch.float32, device=df.device).view(1, -1)
        sigma = (d_max - d_min) / float(d_count)
        return torch.exp(-(((df.unsqueeze(-1) - mu) / sigma) ** 2))


def compute_edge_geometry_fp32(
    pos: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    rbf_dmax: float = 16.0,
    rbf_dim: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `(unit_diff, dist, rbf)` on directed edges `src -> dst` strictly in FP32."""
    with torch.autocast(device_type=pos.device.type, enabled=False):
        pf = pos.float()
        diff = pf[src] - pf[dst]
        dist = _norm_no_nan_fp32(diff, dim=-1, keepdim=True, eps=1e-8, sqrt=True)
        unit_diff = diff / (dist + 1e-8)
        rbf = _rbf_fp32(dist.squeeze(-1), d_min=0.0, d_max=rbf_dmax, d_count=rbf_dim)
        return unit_diff, dist.squeeze(-1), rbf


class ToggleableLoRALinear(nn.Module):
    """Zero-initialized LoRA wrapper supporting `disable_lora()` shared-backbone KL evaluation."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_disabled: bool = False

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty((self.rank, self.in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.rank), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.lora_disabled:
            return base_out
        xd = self.dropout(x)
        delta = F.linear(F.linear(xd, self.lora_A.to(xd.dtype)), self.lora_B.to(xd.dtype)) * self.scaling
        return base_out + delta.to(base_out.dtype)

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        """Return the base `nn.Linear` with LoRA weights permanently merged for zero-overhead inference."""
        self.base.weight.add_((self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scaling)
        return self.base


@contextmanager
def disable_lora(model: nn.Module) -> Iterator[None]:
    """Temporarily bypass all `ToggleableLoRALinear` adapters (for shared-backbone KL reference pass)."""
    mods: List[Tuple[ToggleableLoRALinear, bool]] = []
    for mod in model.modules():
        if isinstance(mod, ToggleableLoRALinear):
            mods.append((mod, mod.lora_disabled))
            mod.lora_disabled = True
    try:
        yield
    finally:
        for mod, prev in mods:
            mod.lora_disabled = prev


def inject_flowmol_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    include_edge_updates: bool = True,
    include_heads: bool = False,
) -> List[str]:
    """Attach zero-initialized `ToggleableLoRALinear` adapters onto GVP scalar & update linear layers."""
    injected: List[str] = []

    def _wrap_in_parent(parent: nn.Module, attr_name: str, full_name: str) -> None:
        sub = getattr(parent, attr_name, None)
        if isinstance(sub, nn.Linear) and not isinstance(sub, ToggleableLoRALinear):
            setattr(parent, attr_name, ToggleableLoRALinear(sub, rank=rank, alpha=alpha, dropout=dropout))
            injected.append(full_name)

    for full_name, mod in list(model.named_modules()):
        if isinstance(mod, GVP):
            if isinstance(mod.to_feats_out, nn.Sequential) and isinstance(mod.to_feats_out[0], nn.Linear):
                mod.to_feats_out[0] = ToggleableLoRALinear(mod.to_feats_out[0], rank=rank, alpha=alpha, dropout=dropout)
                injected.append(f"{full_name}.to_feats_out.0")
            if isinstance(mod.scalar_to_vector_gates, nn.Linear):
                _wrap_in_parent(mod, "scalar_to_vector_gates", f"{full_name}.scalar_to_vector_gates")
        elif include_edge_updates and isinstance(mod, EdgeUpdate):
            for idx, layer in enumerate(mod.edge_update_fn):
                if isinstance(layer, nn.Linear):
                    mod.edge_update_fn[idx] = ToggleableLoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
                    injected.append(f"{full_name}.edge_update_fn.{idx}")
        elif include_heads and full_name in ("node_output_head", "to_edge_logits") and isinstance(mod, nn.Sequential):
            for idx, layer in enumerate(mod):
                if isinstance(layer, nn.Linear):
                    mod[idx] = ToggleableLoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
                    injected.append(f"{full_name}.{idx}")
    return injected


class GVP(nn.Module):
    """SO(3)-equivariant Geometric Vector Perceptron with cross-product channels and FP32 vector math."""

    def __init__(
        self,
        dim_vectors_in: int,
        dim_vectors_out: int,
        dim_feats_in: int,
        dim_feats_out: int,
        *,
        n_cp_feats: int = 0,
        hidden_vectors: Optional[int] = None,
        feats_activation: Optional[nn.Module] = None,
        vectors_activation: Optional[nn.Module] = None,
        vector_gating: bool = True,
        zero_init_vector_gate: bool = False,
    ) -> None:
        super().__init__()
        self.dim_vectors_in = int(dim_vectors_in)
        self.dim_vectors_out = int(dim_vectors_out)
        self.dim_feats_in = int(dim_feats_in)
        self.dim_feats_out = int(dim_feats_out)
        self.n_cp_feats = int(n_cp_feats)
        dim_h = max(self.dim_vectors_in, self.dim_vectors_out) if hidden_vectors is None else int(hidden_vectors)

        wh_k = 1.0 / math.sqrt(max(self.dim_vectors_in, 1))
        self.Wh = nn.Parameter(torch.empty((self.dim_vectors_in, dim_h), dtype=torch.float32).uniform_(-wh_k, wh_k))

        if self.n_cp_feats > 0:
            self.Wcp = nn.Parameter(
                torch.empty((self.dim_vectors_in, self.n_cp_feats * 2), dtype=torch.float32).uniform_(-wh_k, wh_k)
            )
            wu_in_dim = dim_h + self.n_cp_feats
        else:
            self.register_parameter("Wcp", None)
            wu_in_dim = dim_h

        wu_k = 1.0 / math.sqrt(max(wu_in_dim, 1))
        self.Wu = nn.Parameter(torch.empty((wu_in_dim, self.dim_vectors_out), dtype=torch.float32).uniform_(-wu_k, wu_k))

        self.to_feats_out = nn.Sequential(
            nn.Linear(wu_in_dim + self.dim_feats_in, self.dim_feats_out),
            feats_activation if feats_activation is not None else nn.SiLU(),
        )
        self.vectors_activation = vectors_activation if vectors_activation is not None else nn.Sigmoid()

        if vector_gating:
            self.scalar_to_vector_gates: Optional[nn.Linear] = nn.Linear(self.dim_feats_out, self.dim_vectors_out)
            if zero_init_vector_gate:
                nn.init.zeros_(self.scalar_to_vector_gates.weight)
                nn.init.zeros_(self.scalar_to_vector_gates.bias)
        else:
            self.scalar_to_vector_gates = None

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, vectors = data
        # Equivariant vector contractions, cross products, and norms strictly in FP32
        with torch.autocast(device_type=vectors.device.type, enabled=False):
            vf = vectors.float()
            Vh = torch.einsum("nvc,vh->nhc", vf, self.Wh.float())
            if self.n_cp_feats > 0 and self.Wcp is not None:
                Vcp = torch.einsum("nvc,vp->npc", vf, self.Wcp.float())
                cp_src, cp_dst = torch.split(Vcp, self.n_cp_feats, dim=1)
                cp = torch.linalg.cross(cp_src, cp_dst, dim=-1)
                Vh = torch.cat((Vh, cp), dim=1)
            Vu = torch.einsum("nhc,hu->nuc", Vh, self.Wu.float())
            sh = _norm_no_nan_fp32(Vh, dim=-1, keepdim=False, eps=1e-8, sqrt=True)

        # Heavy scalar MLP runs in AMP (BF16 Tensor Cores on A100)
        s_in = torch.cat((feats, sh.to(feats.dtype)), dim=-1)
        feats_out = self.to_feats_out(s_in)

        if self.scalar_to_vector_gates is not None:
            gating = self.scalar_to_vector_gates(feats_out)
            with torch.autocast(device_type=vectors.device.type, enabled=False):
                gate_f = self.vectors_activation(gating.float()).unsqueeze(-1)
                vectors_out = gate_f * Vu
        else:
            with torch.autocast(device_type=vectors.device.type, enabled=False):
                gate_f = self.vectors_activation(_norm_no_nan_fp32(Vu, dim=-1, keepdim=True, eps=1e-8, sqrt=True))
                vectors_out = gate_f * Vu

        return feats_out, vectors_out


class _VDropout(nn.Module):
    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.drop_rate = float(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0.0:
            return x
        keep = 1.0 - self.drop_rate
        mask = (torch.rand(x.shape[:-1], device=x.device, dtype=torch.float32) < keep).to(x.dtype).unsqueeze(-1)
        return mask * x / keep


class GVPDropout(nn.Module):
    def __init__(self, rate: float = 0.0) -> None:
        super().__init__()
        self.feat_dropout = nn.Dropout(rate) if rate > 0.0 else nn.Identity()
        self.vector_dropout = _VDropout(rate)

    def forward(self, feats: torch.Tensor, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.feat_dropout(feats), self.vector_dropout(vectors)


class GVPLayerNorm(nn.Module):
    """Standard LayerNorm for scalars + FP32 RMS equivariant norm for vectors."""

    def __init__(self, feats_h_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, feats: torch.Tensor, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed_feats = self.feat_norm(feats)
        with torch.autocast(device_type=vectors.device.type, enabled=False):
            vf = vectors.float()
            vn = _norm_no_nan_fp32(vf, dim=-1, keepdim=True, eps=1e-8, sqrt=False)
            denom = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True) + self.eps) + self.eps
            normed_vectors = vf / denom
        return normed_feats, normed_vectors


class GVPConv(nn.Module):
    """Pure-PyTorch GVP message passing convolution over directed edges `(src, dst)`."""

    def __init__(
        self,
        scalar_size: int = 256,
        vector_size: int = 16,
        *,
        n_cp_feats: int = 4,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        use_dst_feats: bool = False,
        rbf_dim: int = 32,
        edge_feat_size: int = 128,
        message_norm: float | str = 100.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scalar_size = int(scalar_size)
        self.vector_size = int(vector_size)
        self.use_dst_feats = bool(use_dst_feats)
        self.edge_feat_size = int(edge_feat_size)
        self.message_norm = message_norm

        msg_layers: List[nn.Module] = []
        for i in range(n_message_gvps):
            v_in = self.vector_size
            s_in = self.scalar_size
            if i == 0:
                v_in += 1
                s_in += rbf_dim + self.edge_feat_size
                if self.use_dst_feats:
                    v_in += self.vector_size
                    s_in += self.scalar_size
            msg_layers.append(
                GVP(
                    dim_vectors_in=v_in,
                    dim_vectors_out=self.vector_size,
                    dim_feats_in=s_in,
                    dim_feats_out=self.scalar_size,
                    n_cp_feats=n_cp_feats,
                    feats_activation=nn.SiLU(),
                    vectors_activation=nn.Sigmoid(),
                    vector_gating=True,
                )
            )
        self.edge_message = nn.Sequential(*msg_layers)

        upd_layers: List[nn.Module] = []
        for _ in range(n_update_gvps):
            upd_layers.append(
                GVP(
                    dim_vectors_in=self.vector_size,
                    dim_vectors_out=self.vector_size,
                    dim_feats_in=self.scalar_size,
                    dim_feats_out=self.scalar_size,
                    n_cp_feats=n_cp_feats,
                    feats_activation=nn.SiLU(),
                    vectors_activation=nn.Sigmoid(),
                    vector_gating=True,
                )
            )
        self.node_update = nn.Sequential(*upd_layers)
        self.dropout = GVPDropout(dropout)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)
        self.update_layer_norm = GVPLayerNorm(self.scalar_size)

    def forward(
        self,
        scalar_feats: torch.Tensor,
        vec_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        unit_diff: torch.Tensor,
        rbf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_nodes = scalar_feats.shape[0]

        vec_in_parts = [unit_diff.unsqueeze(1), vec_feats[src]]
        if self.use_dst_feats:
            vec_in_parts.append(vec_feats[dst])
        vec_in = torch.cat(vec_in_parts, dim=1)

        s_in_parts = [scalar_feats[src], rbf.to(scalar_feats.dtype)]
        if self.edge_feat_size > 0:
            s_in_parts.append(edge_feats)
        if self.use_dst_feats:
            s_in_parts.append(scalar_feats[dst])
        s_in = torch.cat(s_in_parts, dim=-1)

        s_msg, v_msg = self.edge_message((s_in, vec_in))

        # FP32 scatter-add aggregation into dst nodes for exact degree stability
        with torch.autocast(device_type=scalar_feats.device.type, enabled=False):
            s_agg = torch.zeros((n_nodes, self.scalar_size), dtype=torch.float32, device=scalar_feats.device)
            v_agg = torch.zeros((n_nodes, self.vector_size, 3), dtype=torch.float32, device=vec_feats.device)
            s_agg.index_add_(0, dst, s_msg.float())
            v_agg.index_add_(0, dst, v_msg.float())

            if self.message_norm == "mean":
                deg = torch.zeros((n_nodes,), dtype=torch.float32, device=scalar_feats.device)
                deg.index_add_(0, dst, torch.ones((dst.shape[0],), dtype=torch.float32, device=scalar_feats.device))
                denom = deg.clamp_min(1.0)
                s_agg = s_agg / denom.unsqueeze(-1)
                v_agg = v_agg / denom.view(-1, 1, 1)
            elif self.message_norm != "sum":
                z = float(self.message_norm)
                s_agg = s_agg / z
                v_agg = v_agg / z

        s_agg_t, v_agg_t = self.dropout(s_agg.to(scalar_feats.dtype), v_agg)
        h = scalar_feats + s_agg_t
        v = vec_feats + v_agg_t
        h, v = self.message_layer_norm(h, v)

        h_res, v_res = self.node_update((h, v))
        h_res, v_res = self.dropout(h_res, v_res)
        h = h + h_res
        v = v + v_res
        h, v = self.update_layer_norm(h, v)
        return h, v


class NodePositionUpdate(nn.Module):
    """SE(3)-equivariant 3D coordinate update strictly applied in FP32."""

    def __init__(
        self,
        n_scalars: int,
        n_vec_channels: int,
        *,
        n_gvps: int = 3,
        n_cp_feats: int = 4,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(n_gvps):
            is_last = i == n_gvps - 1
            layers.append(
                GVP(
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=1 if is_last else n_vec_channels,
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    n_cp_feats=n_cp_feats,
                    feats_activation=nn.SiLU(),
                    vectors_activation=nn.Identity() if is_last else nn.Sigmoid(),
                    vector_gating=True,
                )
            )
        self.gvps = nn.Sequential(*layers)

    def forward(self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        _, delta = self.gvps((scalars, vectors))
        with torch.autocast(device_type=positions.device.type, enabled=False):
            return positions.float() + delta.float().squeeze(1)


class EdgeUpdate(nn.Module):
    """Residual edge representation update using incident node scalars and pairwise RBF distance."""

    def __init__(
        self,
        n_node_scalars: int,
        n_edge_feats: int,
        *,
        update_edge_w_distance: bool = True,
        rbf_dim: int = 32,
    ) -> None:
        super().__init__()
        self.update_edge_w_distance = bool(update_edge_w_distance)
        in_dim = n_node_scalars * 2 + n_edge_feats + (rbf_dim if self.update_edge_w_distance else 0)
        self.edge_update_fn = nn.Sequential(
            nn.Linear(in_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )
        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(
        self,
        node_scalars: torch.Tensor,
        edge_feats: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        rbf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [node_scalars[src], node_scalars[dst], edge_feats]
        if self.update_edge_w_distance:
            assert rbf is not None
            parts.append(rbf.to(edge_feats.dtype))
        return self.edge_norm(edge_feats + self.edge_update_fn(torch.cat(parts, dim=-1)))
