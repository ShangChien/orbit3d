"""Conditional octree core extracted from the project's original research engine.

Uses pinned Uni-3DAR, with no GVP/PropMolFlow/DGL orchestration dependency.
"""

import argparse
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..vendor.uni3dar.data import AtomFeatDict as AtomFeatDict, GridDataset as GridDataset
from ..vendor.uni3dar.data.atom_data import atom_list
from ..vendor.uni3dar.models.uni3dar import DecoderInputFeat
from ..vendor.uni3dar.models.kv_cache import compact_kv
from ..vendor.uni3dar.models.uni3dar_sampler import (
    Uni3DARSampler,
    IDX_CNT,
    IDX_LEVEL,
    IDX_SPACE,
    IDX_TREE,
    IDX_TYPE,
)
from .random_streams import RowRandomStreams
from .electronic_constraints import atom_electron_parities, mask_terminal_element_logits


class DynamicIntervalConditioner(nn.Module):
    """
    Zero-Perturbation Root-Prefix + 12-Layer AdaLN-Zero Interval Conditioner for Uni-3DAR.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        num_props: int = 3,
        num_layers: int = 12,
        num_freqs: int = 16,
        bound_masks: bool = False,
        smooth_frequencies: bool = False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_props = num_props
        self.num_layers = num_layers
        self.num_freqs = num_freqs

        freqs = (torch.linspace(0.25, 4.0, num_freqs) if smooth_frequencies else
                 2.0 ** torch.arange(num_freqs, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs)

        in_dim = 4 + 4 * 2 * num_freqs
        self.prop_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, emb_dim),
                    nn.LayerNorm(emb_dim),
                    nn.SiLU(),
                    nn.Linear(emb_dim, emb_dim),
                )
                for _ in range(num_props)
            ]
        )
        self.prop_id_emb = nn.Embedding(num_props, emb_dim)
        self.bound_projection = nn.Linear(2, emb_dim, bias=False) if bound_masks else None
        if self.bound_projection is not None:
            nn.init.zeros_(self.bound_projection.weight)

        self.global_mlp = nn.Sequential(
            nn.Linear(num_props * emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
        )

        # Root prefix projection (injected at Token 0, bias=False so c=0 -> 0)
        self.root_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        nn.init.normal_(self.root_proj.weight, mean=0.0, std=0.01)
        self.root_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Per-layer AdaLN-Zero projections (bias=False and zero-initialized)
        self.layer_mods = nn.ModuleList(
            [nn.Linear(emb_dim, 2 * emb_dim, bias=False) for _ in range(num_layers)]
        )
        for mod in self.layer_mods:
            nn.init.zeros_(mod.weight)

        # Final output norm AdaLN-Zero projection (bias=False and zero-initialized)
        self.out_mod = nn.Linear(emb_dim, 2 * emb_dim, bias=False)
        nn.init.zeros_(self.out_mod.weight)

    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        angles = x.unsqueeze(-1) * self.freqs
        return torch.cat([x, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)], dim=-1)

    def forward(
        self,
        y_min: torch.Tensor,
        y_max: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz = y_min.shape[0]
        device = y_min.device
        bound_mask = active_mask if active_mask.ndim == 3 else active_mask.unsqueeze(-1).expand(-1, -1, 2)
        if self.bound_projection is None and (bound_mask[..., 0] != bound_mask[..., 1]).any():
            raise ValueError("One-sided intervals require a v2 conditioner trained with bound masks")
        active = bound_mask.any(-1)
        mask_f = active.float().unsqueeze(-1)
        y_min = torch.where(bound_mask[..., 0], y_min, 0)
        y_max = torch.where(bound_mask[..., 1], y_max, 0)

        c = 0.5 * (y_min + y_max)
        w = 0.5 * (y_max - y_min).clamp(min=1e-4)

        per_prop_list = []
        for k in range(self.num_props):
            raw_k = torch.stack([c[:, k], w[:, k], y_min[:, k], y_max[:, k]], dim=-1)
            feat_k = self._fourier_encode(raw_k)
            id_k = self.prop_id_emb(torch.full((bsz,), k, dtype=torch.long, device=device))
            emb_k = self.prop_encoders[k](feat_k) + id_k
            if self.bound_projection is not None:
                emb_k = emb_k + self.bound_projection(bound_mask[:, k].float())
            emb_k = emb_k * mask_f[:, k]
            per_prop_list.append(emb_k)

        concat_emb = torch.cat(per_prop_list, dim=-1)
        global_c = self.global_mlp(concat_emb)
        has_any = active.any(dim=-1, keepdim=True).to(global_c.dtype)
        return global_c * has_any

    def forward_root_prefix(self, global_c: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.root_gate) * self.root_proj(global_c.float())

    def modulate_layer(self, layer_idx: int, h: torch.Tensor, global_c: torch.Tensor) -> torch.Tensor:
        scale_shift = self.layer_mods[layer_idx](global_c.float()).to(h.dtype)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = 0.08 * torch.tanh(scale)
        shift = 0.08 * torch.tanh(shift)
        return h * (1.0 + scale) + shift

    def modulate_out(self, h: torch.Tensor, global_c: torch.Tensor) -> torch.Tensor:
        scale_shift = self.out_mod(global_c.float()).to(h.dtype)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = 0.15 * torch.tanh(scale)
        shift = 0.15 * torch.tanh(shift)
        return h * (1.0 + scale) + shift


class PointConditioner(nn.Module):
    """Normalized scalar values and explicit activation; no interval-width features."""
    def __init__(self, emb_dim, num_props, num_layers):
        super().__init__()
        self.prop_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(1, emb_dim), nn.LayerNorm(emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
            for _ in range(num_props)])
        self.prop_id_emb = nn.Embedding(num_props, emb_dim)
        self.global_mlp = nn.Sequential(nn.Linear(num_props * emb_dim, emb_dim), nn.LayerNorm(emb_dim),
                                        nn.SiLU(), nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim), nn.SiLU())
        self.root_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        nn.init.normal_(self.root_proj.weight, std=0.01)
        self.root_gate = nn.Parameter(torch.tensor(0.0))
        self.layer_mods = nn.ModuleList([nn.Linear(emb_dim, 2 * emb_dim, bias=False) for _ in range(num_layers)])
        self.out_mod = nn.Linear(emb_dim, 2 * emb_dim, bias=False)
        for mod in [*self.layer_mods, self.out_mod]:
            nn.init.zeros_(mod.weight)

    def forward(self, values, unused_upper, active_mask):
        active = active_mask.any(-1) if active_mask.ndim == 3 else active_mask
        values = torch.where(active, values, 0)
        features = [(encoder(values[:, k:k+1]) + self.prop_id_emb.weight[k]) * active[:, k:k+1]
                    for k, encoder in enumerate(self.prop_encoders)]
        return self.global_mlp(torch.cat(features, -1)) * active.any(-1, keepdim=True)

    forward_root_prefix = DynamicIntervalConditioner.forward_root_prefix
    modulate_layer = DynamicIntervalConditioner.modulate_layer
    modulate_out = DynamicIntervalConditioner.modulate_out


class Suffix3DPropertyHead(nn.Module):
    """
    3D Self-Consistent Property Understanding Head pooling:
      [h_CLS, mean_pool(h_Level0_atoms), max_pool(h_Level0_atoms)] -> (gap, homo, lumo).
    """

    def __init__(self, emb_dim: int = 768, num_props: int = 3):
        super().__init__()
        in_dim = 3 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.LayerNorm(emb_dim // 2),
            nn.SiLU(),
            nn.Linear(emb_dim // 2, num_props),
        )

    def forward(
        self,
        dec_y: torch.Tensor,
        decoder_input: DecoderInputFeat,
        cu_lens: torch.Tensor,
        atom_level_index: int,
        atom_mask_id: int,
        atom_null_id: int,
    ) -> torch.Tensor:
        bsz = cu_lens.shape[0] - 1
        y_f = dec_y.float()
        pooled_list = []
        for b in range(bsz):
            s = int(cu_lens[b].item())
            e = int(cu_lens[b + 1].item())
            h_seq = y_f[s:e]
            h_cls = h_seq[-1]

            lvl_b = decoder_input.level[s:e]
            typ_b = decoder_input.type[s:e]
            atom_mask = (
                (lvl_b == atom_level_index)
                & (typ_b != atom_mask_id)
                & (typ_b != atom_null_id)
                & (typ_b < len(atom_list))
            )
            if atom_mask.any():
                h_atoms = h_seq[atom_mask]
                h_mean = h_atoms.mean(dim=0)
                h_max = h_atoms.max(dim=0).values
            else:
                h_mean = h_cls
                h_max = h_cls
            pooled_list.append(torch.cat([h_cls, h_mean, h_max], dim=-1))

        mol_repr = torch.stack(pooled_list, dim=0)
        return self.net(mol_repr)


class ConditionalUni3DAR(Uni3DARSampler):
    """
    Unified Conditional Generation + 3D Suffix Understanding + Gated-GRPO Model built on Uni-3DAR.
    """

    def __init__(self, args, dictionary, num_props: int = 3):
        super().__init__(args, dictionary)
        self.num_props = num_props
        self.condition_version = getattr(args, "condition_version", 1)
        if self.condition_version == 4:
            self.interval_conditioner = PointConditioner(args.emb_dim, num_props, args.layer)
        else:
            self.interval_conditioner = DynamicIntervalConditioner(
                emb_dim=args.emb_dim,
                num_props=num_props,
                num_layers=args.layer,
                bound_masks=self.condition_version >= 2,
                smooth_frequencies=self.condition_version >= 3,
            )
        self.prop_understanding_head = Suffix3DPropertyHead(
            emb_dim=args.emb_dim,
            num_props=num_props,
        )

    def forward_conditional_sequence(
        self,
        decoder_input: DecoderInputFeat,
        cu_lens: torch.Tensor,
        max_len: int,
        y_min: torch.Tensor,
        y_max: torch.Tensor,
        active_mask: torch.Tensor,
        compute_understanding: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dec_y = self.input_embedding(decoder_input)
        global_c = self.interval_conditioner(y_min, y_max, active_mask)

        root_indices = cu_lens[:-1].long()
        root_delta = self.interval_conditioner.forward_root_prefix(global_c).to(dec_y.dtype)
        dec_y = dec_y.clone()
        dec_y[root_indices] = dec_y[root_indices] + root_delta

        seq_lens = (cu_lens[1:] - cu_lens[:-1]).long()
        c_expanded = torch.repeat_interleave(global_c, seq_lens, dim=0)

        self.rope.calc_and_cache(decoder_input.phy_pos / self.args.xyz_resolution)
        for _ in range(self.args.recycle):
            for l_idx, layer in enumerate(self.decoder.layers):
                dec_y = layer(dec_y, cu_lens, max_len, None)
                dec_y = self.interval_conditioner.modulate_layer(l_idx, dec_y, c_expanded)

        dec_y = self.norm(dec_y)
        dec_y = self.interval_conditioner.modulate_out(dec_y, c_expanded)

        cls_preds = None
        if compute_understanding:
            cls_preds = self.prop_understanding_head(
                dec_y=dec_y,
                decoder_input=decoder_input,
                cu_lens=cu_lens,
                atom_level_index=self.atom_level_index,
                atom_mask_id=self.atom_mask_id,
                atom_null_id=self.atom_null_id,
            )
        return dec_y, cls_preds

    def compute_sft_and_understanding_loss(
        self,
        batched_data: Dict[str, torch.Tensor],
        y_true_norm: torch.Tensor,
        y_min_norm: torch.Tensor,
        y_max_norm: torch.Tensor,
        active_mask: torch.Tensor,
        understanding_weight: float = 1.0,
        return_per_molecule: bool = False,
        atom_class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        device = y_true_norm.device
        bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batched_data.items()}
        bd["decoder_phy_pos"] = bd["decoder_phy_pos"].float()

        decoder_input, decoder_gen_target, _ = self.get_feat_from_dataloader(bd)
        cu_lens = bd["decoder_input_len"].to(device, dtype=torch.int32)
        max_len = int(bd["decoder_input_max_len"].item())

        dec_y_cond, _ = self.forward_conditional_sequence(
            decoder_input=decoder_input,
            cu_lens=cu_lens,
            max_len=max_len,
            y_min=y_min_norm,
            y_max=y_max_norm,
            active_mask=active_mask,
            compute_understanding=False,
        )

        is_gen_tree = decoder_gen_target.is_gen_tree
        is_gen_atom = decoder_gen_target.is_gen_atom

        loss_tree = torch.tensor(0.0, device=device)
        if is_gen_tree.any():
            logits_tree = self.tree_heads[0](dec_y_cond[is_gen_tree]).float()
            gt_tree = (decoder_gen_target.type[is_gen_tree] - self.tree0_split_id - 1).long()
            loss_tree = F.cross_entropy(logits_tree, gt_tree)

        loss_atom = torch.tensor(0.0, device=device)
        loss_xyz = torch.tensor(0.0, device=device)
        if is_gen_atom.any():
            gts_xyz = decoder_gen_target.to_list(is_gen_atom)
            gts_xyz = [v.long() for v in gts_xyz]
            preds_xyz = self.atom_heads[0](dec_y_cond[is_gen_atom], gts_xyz)
            atom_logits = preds_xyz[0].float().masked_fill(~self.allowed_atom_mask(device), -torch.inf)
            loss_atom = F.cross_entropy(atom_logits, gts_xyz[0], weight=atom_class_weights)
            loss_xyz = (
                F.cross_entropy(preds_xyz[1].float(), gts_xyz[1])
                + F.cross_entropy(preds_xyz[2].float(), gts_xyz[2])
                + F.cross_entropy(preds_xyz[3].float(), gts_xyz[3])
            ) / 3.0

        gen_loss = loss_tree + loss_atom + 0.1 * loss_xyz

        und_loss = torch.zeros((), device=device)
        if understanding_weight:
            _, cls_preds_uncond = self.forward_conditional_sequence(
                decoder_input=decoder_input,
                cu_lens=cu_lens,
                max_len=max_len,
                y_min=y_min_norm,
                y_max=y_max_norm,
                active_mask=torch.zeros_like(active_mask, dtype=torch.bool),
                compute_understanding=True,
            )
            und_loss = F.smooth_l1_loss(cls_preds_uncond, y_true_norm, beta=0.1)

        total_loss = gen_loss + understanding_weight * und_loss
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_gen": float(gen_loss.item()),
            "loss_und": float(und_loss.item()),
            "root_gate": float(torch.tanh(self.interval_conditioner.root_gate).item()),
        }
        if return_per_molecule:
            owners = torch.repeat_interleave(torch.arange(len(y_true_norm), device=device),
                                             torch.diff(cu_lens).long())
            per_molecule = torch.zeros(len(y_true_norm), device=device)

            def average_decisions(values, selected, decision_weights=None):
                ids = owners[selected]
                summed = torch.zeros_like(per_molecule).index_add(0, ids, values)
                counts = (torch.bincount(ids, minlength=len(per_molecule)) if decision_weights is None else
                          torch.zeros_like(per_molecule).index_add(0, ids, decision_weights))
                return summed / counts.clamp_min(torch.finfo(summed.dtype).tiny)

            if is_gen_tree.any():
                per_molecule = per_molecule + average_decisions(
                    F.cross_entropy(logits_tree, gt_tree, reduction='none'), is_gen_tree)
            if is_gen_atom.any():
                atom_losses = F.cross_entropy(atom_logits, gts_xyz[0], reduction='none', weight=atom_class_weights)
                coordinate_losses = sum(F.cross_entropy(preds_xyz[axis].float(), gts_xyz[axis], reduction='none')
                                        for axis in range(1, 4)) / 3
                decision_weights = atom_class_weights[gts_xyz[0]] if atom_class_weights is not None else None
                per_molecule = (per_molecule + average_decisions(atom_losses, is_gen_atom, decision_weights)
                                + .1 * average_decisions(coordinate_losses, is_gen_atom))
            metrics['per_molecule_generation_loss'] = per_molecule
        return total_loss, metrics

    def generate_conditional_trajectories(
        self,
        y_min: torch.Tensor,
        y_max: torch.Tensor,
        active_mask: torch.Tensor,
        cfg_scale: float = 1.0,
        tree_temp: Optional[float] = None,
        atom_temp: Optional[float] = None,
        xyz_temp: Optional[float] = None,
        evaluate_self_consistency: bool = True,
        row_seeds=None,
        spatial_box=None,
        electron_parity=None,
        cache_mode="concat",
    ) -> List[Dict]:
        if cache_mode not in {"concat", "buffer"}:
            raise ValueError("KV cache mode must be concat or buffer")
        self.eval()
        device = y_min.device
        batch_size = y_min.shape[0]
        if spatial_box is not None:
            spatial_box.validate_model(self.args)
        has_cond = bool((active_mask.float().sum() > 0).item())
        if not math.isfinite(cfg_scale) or cfg_scale < 0:
            raise ValueError('CFG scale must be finite and nonnegative')
        use_cfg = has_cond and cfg_scale != 1.0
        guidance = {'hidden': None, 'scale': cfg_scale} if use_cfg else None
        electron_state = {}
        parities = atom_electron_parities(self, device) if electron_parity is not None else None
        def mask_atom(logits, selected):
            return mask_terminal_element_logits(logits, electron_state['accumulated'][selected],
                                                electron_state['terminal'][selected], electron_parity, parities)

        for name, value in [('tree', tree_temp), ('atom', atom_temp), ('xyz', xyz_temp)]:
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} sampling temperature must be finite and positive')

        orig_tree_t = list(self.tree_temperature)
        orig_atom_t = list(self.atom_temperature)
        orig_xyz_t = list(self.xyz_temperature)
        if tree_temp is not None:
            self.tree_temperature = [tree_temp] * self.num_tree
        if atom_temp is not None:
            self.atom_temperature = [atom_temp] * self.num_tree
        if xyz_temp is not None:
            self.xyz_temperature = [xyz_temp] * self.num_tree

        try:
            with torch.no_grad():
                kv_cache = [[{"step": 1, "mode": cache_mode} for _ in range(self.args.layer)] for _ in range(self.args.recycle)]
                global_c = self.interval_conditioner(y_min, y_max, active_mask).unsqueeze(1)

                if use_cfg:
                    kv_cache_uncond = [
                        [{"step": 1, "mode": cache_mode} for _ in range(self.args.layer)] for _ in range(self.args.recycle)
                    ]
                    null_mask = torch.zeros_like(active_mask, dtype=torch.bool)
                    global_c_uncond = self.interval_conditioner(y_min, y_max, null_mask).unsqueeze(1)
                else:
                    kv_cache_uncond = None
                    global_c_uncond = None

                target_tree_index, target_space_index = 0, 0
                cond_vars = {
                    "global_c": global_c,
                    "global_c_uncond": global_c_uncond,
                    "orig_idx": torch.arange(batch_size, dtype=torch.long, device=device),
                }
                streams = RowRandomStreams(row_seeds) if row_seeds is not None else None

                def sample_uniforms(selected):
                    rows = cond_vars["orig_idx"][selected].tolist()
                    return streams.draw(rows, device)
                count_vars = {
                    "known_atoms": torch.full(
                        (batch_size, self.num_tree), -1, dtype=torch.int32, device=device
                    ),
                    "total_atoms": torch.zeros(batch_size, dtype=torch.int32, device=device),
                    "acc_tokens": torch.zeros(batch_size, dtype=torch.int32, device=device),
                }

                decoder_feat, decoder_phy_pos, raw_feat = self.calc_root_level(
                    self.args.merge_level,
                    batch_size,
                    device,
                    target_tree_index,
                    target_space_index,
                )
                scores = torch.zeros((batch_size, 1), device=device)
                finished_trajs = []

                atom_predict_funcs, cond_vars = self.get_atom_predict_funcs(
                    batch_size, device, None, cond_vars, sample_uniforms if streams else None, guidance=guidance,
                    atom_logit_mask=mask_atom if electron_parity is not None else None,
                )
                def mask_tree(logits, selected):
                    return spatial_box.mask_tree_logits(logits, next_feat[selected, :3], next_feat[selected, IDX_LEVEL])

                tree_predict_funcs = self.get_tree_predict_funcs(
                    sample_uniforms if streams else None, mask_tree if spatial_box else None, guidance=guidance
                )

                cur_feat = decoder_feat[:, -1, :]
                cur_phy_pos = decoder_phy_pos[:, -1, :]
                cur_input = DecoderInputFeat(
                    cur_feat[:, IDX_TYPE],
                    cur_feat[:, :3],
                    cur_feat[:, IDX_LEVEL],
                    cur_phy_pos,
                    cur_feat[:, IDX_SPACE],
                    cur_feat[:, IDX_TREE],
                    torch.zeros_like(cur_feat[:, IDX_TYPE]).int(),
                    torch.ones_like(cur_feat[:, IDX_TYPE]).int(),
                    cur_feat[:, IDX_CNT],
                ).add_seq_dim()

                step_idx = 0
                while len(raw_feat) > 0:
                    dec_y = self.input_embedding(cur_input)
                    if step_idx == 0:
                        dec_y = dec_y + self.interval_conditioner.forward_root_prefix(
                            cond_vars["global_c"]
                        ).to(dec_y.dtype)

                    self.rope.calc_and_cache(cur_input.phy_pos / self.args.xyz_resolution)
                    for r_i in range(self.args.recycle):
                        for l_i, layer in enumerate(self.decoder.layers):
                            dec_y = layer(dec_y, None, None, kv_cache[r_i][l_i])
                            dec_y = self.interval_conditioner.modulate_layer(
                                l_i, dec_y, cond_vars["global_c"]
                            )
                    dec_y = self.norm(dec_y)
                    dec_y = self.interval_conditioner.modulate_out(dec_y, cond_vars["global_c"])
                    dec_y_last = dec_y[:, -1, :]

                    if use_cfg:
                        dec_u = self.input_embedding(cur_input)
                        self.rope.calc_and_cache(cur_input.phy_pos / self.args.xyz_resolution)
                        for r_i in range(self.args.recycle):
                            for l_i, layer in enumerate(self.decoder.layers):
                                dec_u = layer(dec_u, None, None, kv_cache_uncond[r_i][l_i])
                        dec_u = self.norm(dec_u)
                        dec_u_last = dec_u[:, -1, :]

                        # Guide each tree/element/XYZ categorical distribution.
                        # Mixing normalized hidden states is not logit-space CFG
                        # through the nonlinear autoregressive prediction heads.
                        guidance['hidden'] = dec_u_last

                    step_idx += 1
                    last_level = cur_input.level[:, -1]
                    last_type = cur_input.type[:, -1]
                    last_remaining_atoms = cur_input.remaining_atoms[:, -1]
                    last_remaining_tokens = cur_input.remaining_tokens[:, -1]

                    next_feat, raw_feat = self.pop_next_feat(raw_feat)
                    next_type = next_feat[:, IDX_TYPE]
                    is_atom_layer = last_level == self.atom_level_index
                    is_pred_atom = is_atom_layer & (last_type == self.atom_mask_id) & (next_type < 0)
                    is_first_cnt = (last_level == self.args.merge_level) & (last_type == self.atom_mask_id)
                    if electron_parity is not None and is_pred_atom.any():
                        types = decoder_feat[:, :, IDX_TYPE]
                        actual = (decoder_feat[:, :, IDX_LEVEL] == self.atom_level_index) & (types < len(atom_list)) & (types >= 0)
                        safe = torch.where(actual, types, self.atom_null_id)
                        electron_state['accumulated'] = parities[safe].sum(-1) % 2
                        electron_state['terminal'] = torch.tensor([len(queue) == 0 for queue in raw_feat], device=device)

                    pred_type, pred_count, pred_xyz, scores = self.predict_and_sample(
                        is_pred_atom,
                        next_feat,
                        is_first_cnt,
                        count_vars,
                        dec_y_last,
                        scores,
                        tree_predict_funcs,
                        atom_predict_funcs,
                    )
                    if is_first_cnt.any():
                        count_vars["total_atoms"][is_first_cnt] = pred_count[is_first_cnt]
                    raw_feat, count_vars = self.tree_expand(
                        raw_feat, last_type, pred_type, next_feat, count_vars
                    )
                    decoder_feat, decoder_phy_pos = self.get_feat_by_pred(
                        next_feat, decoder_feat, decoder_phy_pos, pred_type, pred_count, pred_xyz
                    )
                    count_vars["last_remaining_atoms"] = last_remaining_atoms
                    count_vars["last_remaining_tokens"] = last_remaining_tokens
                    count_vars["pred_count"] = pred_count

                    is_delete = torch.zeros(decoder_feat.shape[0], device=device, dtype=torch.bool)
                    next_feat_list, new_raw_feat = [], []
                    empty = True
                    for i, feat in enumerate(raw_feat):
                        if feat.shape[0] > 0:
                            next_feat_list.append(feat[0:1])
                            new_raw_feat.append(feat[1:])
                            empty = False
                        else:
                            is_delete[i] = True
                            finished_trajs.append(
                                {
                                    "decoder_feat": decoder_feat[i].clone(),
                                    "decoder_phy_pos": decoder_phy_pos[i].clone(),
                                    "scores": scores[i].clone(),
                                    "batch_idx": int(cond_vars["orig_idx"][i].item()),
                                }
                            )
                    if empty:
                        break

                    rows_finished = len(new_raw_feat) != len(raw_feat)
                    next_feat = torch.cat(next_feat_list, dim=0)
                    raw_feat = new_raw_feat
                    # Identity boolean indexing copies every KV tensor and
                    # synchronizes dynamic row counts. Compact only when a row
                    # actually finishes; Python queue lengths already tell us.
                    if rows_finished:
                        decoder_feat = decoder_feat[~is_delete]
                        decoder_phy_pos = decoder_phy_pos[~is_delete]
                        for k in cond_vars:
                            if cond_vars[k] is not None:
                                cond_vars[k] = cond_vars[k][~is_delete]
                        for k in count_vars:
                            if count_vars[k] is not None:
                                count_vars[k] = count_vars[k][~is_delete]
                        scores = scores[~is_delete]
                        for cache_obj in [kv_cache, kv_cache_uncond]:
                            if cache_obj is None:
                                continue
                            for r_i in range(len(cache_obj)):
                                for l_i in range(len(cache_obj[r_i])):
                                    compact_kv(cache_obj[r_i][l_i], ~is_delete)

                    last_level = decoder_feat[:, -1, IDX_LEVEL]
                    is_new_level = last_level != next_feat[:, IDX_LEVEL]
                    raw_feat = self.add_new_tree(raw_feat, next_feat, is_new_level)
                    decoder_feat, decoder_phy_pos = self.get_feat_by_pred(
                        next_feat, decoder_feat, decoder_phy_pos, next_feat[:, IDX_TYPE], None, None
                    )
                    scores = torch.cat([scores, torch.zeros((next_feat.shape[0], 1), device=device)], dim=1)
                    count_vars, pair_rem_a, pair_rem_t = self.update_running_stats(
                        count_vars, is_new_level, last_level
                    )
                    cur_feat = decoder_feat[:, -2:, :]
                    cur_phy_pos = decoder_phy_pos[:, -2:, :]
                    cur_input = DecoderInputFeat(
                        cur_feat[:, :, IDX_TYPE],
                        cur_feat[:, :, :3],
                        cur_feat[:, :, IDX_LEVEL],
                        cur_phy_pos,
                        cur_feat[:, :, IDX_SPACE],
                        cur_feat[:, :, IDX_TREE],
                        pair_rem_a,
                        pair_rem_t,
                        cur_feat[:, :, IDX_CNT],
                    )
                    cur_is_cls = cur_input.level == (self.atom_level_index - 1)
                    cur_input.remaining_atoms[cur_is_cls] = 0
                    cur_input.remaining_tokens[cur_is_cls] = 0
                    cur_input.count[cur_is_cls] = 0
                    cur_input.remaining_atoms.clamp_(min=0)
                    cur_input.remaining_tokens.clamp_(min=0)

                finished_trajs.sort(key=lambda d: d["batch_idx"])

                for traj in finished_trajs:
                    df = traj["decoder_feat"]
                    dp = traj["decoder_phy_pos"]
                    sc = traj["scores"]
                    lvl = df[:, IDX_LEVEL]
                    tree_sc = sc[lvl > self.atom_level_index]
                    atom_sc = sc[lvl == self.atom_level_index]
                    traj["tree_score_ppl"] = (
                        float(torch.exp(-tree_sc.mean()).item()) if tree_sc.numel() > 0 else 10.0
                    )
                    traj["atom_score_ppl"] = (
                        float(torch.exp(-atom_sc.mean()).item()) if atom_sc.numel() > 0 else 10.0
                    )

                    mask = (lvl == self.atom_level_index) & (df[:, IDX_TYPE] < len(atom_list))
                    atom_ids = df[mask, IDX_TYPE].tolist()
                    traj["atom_symbols"] = [atom_list[idx] for idx in atom_ids]
                    traj["positions_np"] = dp[mask, :3].float().detach().cpu().numpy()

                if evaluate_self_consistency and len(finished_trajs) > 0:
                    self_preds = self.evaluate_suffix_understanding_on_trajs(finished_trajs, device)
                    for traj, sp in zip(finished_trajs, self_preds):
                        traj["self_pred_norm"] = sp.detach()

                return finished_trajs
        finally:
            self.tree_temperature = orig_tree_t
            self.atom_temperature = orig_atom_t
            self.xyz_temperature = orig_xyz_t

    def pack_sampled_trajectories(
        self,
        trajs: List[Dict],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        cls_phy_pos = self.get_cls_phy_pos(self.args.merge_level)

        all_type, all_xyz, all_phy, all_lvl = [], [], [], []
        all_sec, all_space, all_tree, all_rem_a, all_rem_t, all_cnt, all_flag = ([], [], [], [], [], [], [])
        cu_lens = [0]

        for traj in trajs:
            df = traj["decoder_feat"].to(device).long()
            dp = traj["decoder_phy_pos"].to(device).float()
            L = df.shape[0]

            b_type = df[:, IDX_TYPE]
            b_xyz = df[:, :3]
            b_phy = dp
            b_lvl = df[:, IDX_LEVEL]
            b_sec = torch.zeros((L,), dtype=torch.bool, device=device)
            b_sec[1::2] = True
            b_space = df[:, IDX_SPACE].clamp(min=0)
            b_tree = df[:, IDX_TREE].clamp(min=0)
            b_rem_a = torch.zeros((L,), dtype=torch.long, device=device)
            b_rem_t = torch.zeros((L,), dtype=torch.long, device=device)
            b_cnt = df[:, IDX_CNT]
            b_flag = torch.ones((L,), dtype=torch.long, device=device)

            c_type = torch.tensor([self.atom_mask_id, self.atom_cls_id], dtype=torch.long, device=device)
            c_xyz = torch.full((2, 3), self.xyz_null_id, dtype=torch.long, device=device)
            c_phy = torch.full((2, 3), cls_phy_pos, dtype=torch.float32, device=device)
            c_lvl = torch.full((2,), self.atom_level_index - 1, dtype=torch.long, device=device)
            c_sec = torch.zeros((2,), dtype=torch.bool, device=device)
            c_zero = torch.zeros((2,), dtype=torch.long, device=device)

            all_type.extend([b_type, c_type])
            all_xyz.extend([b_xyz, c_xyz])
            all_phy.extend([b_phy, c_phy])
            all_lvl.extend([b_lvl, c_lvl])
            all_sec.extend([b_sec, c_sec])
            all_space.extend([b_space, c_zero])
            all_tree.extend([b_tree, c_zero])
            all_rem_a.extend([b_rem_a, c_zero])
            all_rem_t.extend([b_rem_t, c_zero])
            all_cnt.extend([b_cnt, c_zero])
            all_flag.extend([b_flag, c_zero])
            cu_lens.append(cu_lens[-1] + L + 2)

        cu_t = torch.tensor(cu_lens, dtype=torch.int32, device=device)
        max_len = int((cu_t[1:] - cu_t[:-1]).max().item())

        return {
            "decoder_type": torch.cat(all_type, dim=0),
            "decoder_xyz": torch.cat(all_xyz, dim=0),
            "decoder_phy_pos": torch.cat(all_phy, dim=0),
            "decoder_level": torch.cat(all_lvl, dim=0),
            "decoder_is_second_atom": torch.cat(all_sec, dim=0),
            "space_index": torch.cat(all_space, dim=0),
            "tree_index": torch.cat(all_tree, dim=0),
            "decoder_remaining_atoms": torch.cat(all_rem_a, dim=0),
            "decoder_remaining_tokens": torch.cat(all_rem_t, dim=0),
            "decoder_count": torch.cat(all_cnt, dim=0),
            "tree_loss_flag": torch.cat(all_flag, dim=0),
            "decoder_input_len": cu_t,
            "decoder_input_max_len": torch.tensor(max_len, dtype=torch.int32, device=device),
        }

    @torch.no_grad()
    def evaluate_suffix_understanding_on_trajs(
        self,
        trajs: List[Dict],
        device: torch.device,
    ) -> torch.Tensor:
        bsz = len(trajs)
        packed = self.pack_sampled_trajectories(trajs, device)
        decoder_input, _, _ = self.get_feat_from_dataloader(packed)
        dummy_y = torch.zeros((bsz, self.num_props), dtype=torch.float32, device=device)
        null_mask = torch.zeros((bsz, self.num_props), dtype=torch.bool, device=device)

        _, cls_preds = self.forward_conditional_sequence(
            decoder_input=decoder_input,
            cu_lens=packed["decoder_input_len"],
            max_len=int(packed["decoder_input_max_len"].item()),
            y_min=dummy_y,
            y_max=dummy_y,
            active_mask=null_mask,
            compute_understanding=True,
        )
        return cls_preds

    def allowed_atom_mask(self, device):
        allowed = torch.zeros(self.num_atom, dtype=torch.bool, device=device)
        for symbol in self.args.allow_atoms.split(","):
            allowed[self.dictionary[symbol]] = True
        allowed[self.atom_null_id] = True
        return allowed

    def trajectory_distributions(self, trajs, y_min, y_max, active_mask, temperatures=(0.85, 0.3, 0.25), cfg_scale=1.0, electron_parity=None):
        """Categorical distributions for every sampled tree, atom and XYZ decision."""
        packed = self.pack_sampled_trajectories(trajs, y_min.device)
        decoder_input, target, _ = self.get_feat_from_dataloader(packed)
        cu = packed["decoder_input_len"]
        hidden, _ = self.forward_conditional_sequence(
            decoder_input, cu, int(packed["decoder_input_max_len"]), y_min, y_max, active_mask
        )
        uncond = None
        if cfg_scale != 1.0:
            uncond, _ = self.forward_conditional_sequence(
                decoder_input, cu, int(packed['decoder_input_max_len']), y_min, y_max, torch.zeros_like(active_mask)
            )
        def guided(values, null_values):
            return null_values.float() + cfg_scale * (values.float() - null_values.float())
        owner = torch.repeat_interleave(
            torch.arange(len(trajs), device=y_min.device), (cu[1:] - cu[:-1]).long()
        )
        packets = []
        tree = target.is_gen_tree
        if tree.any():
            logits = self.tree_heads[0](hidden[tree]).float()
            if uncond is not None:
                logits = guided(logits, self.tree_heads[0](uncond[tree]))
            logits = logits / temperatures[0]
            labels = (target.type[tree] - self.tree0_split_id - 1).long()
            packets.append((logits.log_softmax(-1), labels, owner[tree]))
        atom = target.is_gen_atom
        if atom.any():
            labels = [v.long() for v in target.to_list(atom)]
            logits = self.atom_heads[0](hidden[atom], labels)
            if uncond is not None:
                null_logits = self.atom_heads[0](uncond[atom], labels)
                logits = [guided(x, u) for x, u in zip(logits, null_logits)]
            allowed = self.allowed_atom_mask(y_min.device)
            if electron_parity is not None:
                parities = atom_electron_parities(self, y_min.device)
                atom_owner = owner[atom]
                terminal = torch.ones_like(atom_owner, dtype=torch.bool)
                terminal[:-1] = atom_owner[:-1] != atom_owner[1:]
                accumulated = torch.zeros(len(trajs), dtype=torch.long, device=y_min.device)
                accumulated.index_add_(0, atom_owner[~terminal], parities[labels[0][~terminal]])
                logits[0] = mask_terminal_element_logits(logits[0], accumulated[atom_owner] % 2, terminal,
                                                        electron_parity, parities)
            for axis, values in enumerate(logits):
                values = values.float() / (temperatures[1] if axis == 0 else temperatures[2])
                if axis == 0:
                    values = values.masked_fill(~allowed, -torch.inf)
                packets.append((values.log_softmax(-1), labels[axis], owner[atom]))
        return packets

    def compute_trajectory_log_probs(self, trajs, y_min, y_max, active_mask):
        packets = self.trajectory_distributions(trajs, y_min, y_max, active_mask)
        total = torch.zeros(len(trajs), device=y_min.device)
        count = torch.zeros_like(total)
        for logp, labels, owner in packets:
            total.index_add_(0, owner, logp.gather(1, labels[:, None]).squeeze(1))
            count.index_add_(0, owner, torch.ones_like(owner, dtype=total.dtype))
        return total / count.clamp_min(1), None


def build_default_uni3dar_args(batch_size: int = 64) -> argparse.Namespace:
    return argparse.Namespace(
        data_type="molecule",
        max_num_atom=-1,
        crystal_pxrd=0,
        crystal_component=0,
        grid_len=0.24,
        xyz_resolution=0.01,
        merge_level=6,
        emb_dim=768,
        num_head=12,
        layer=12,
        recycle=1,
        rope_theta=10000,
        residual_dropout=0.0,
        ffn_multiple=4,
        emb_dropout=0.0,
        attn_dropout=0.0,
        head_dropout=0.0,
        pooler_dropout=0.0,
        pooler_activation_fn="tanh",
        finetune=True,
        checkpoint_activation_threshold=100000,
        atom_head_type="ar",
        loss_ratio_tree="1.0",
        loss_ratio_atom="1.0",
        loss_ratio_xyz="0.1",
        loss_ratio_count="1.0",
        loss_ratio_atom_target="1.0",
        loss_ratio_mol_target="1.0",
        tree_temperature="0.85",
        atom_temperature="0.3",
        xyz_temperature="0.25",
        count_temperature="1.0",
        mol_target_key=None,
        atom_target_key=None,
        task_name="qm9",
        allow_atoms="H,C,N,O,F",
        rank_by="atom",
        rank_ratio=0.1,
        batch_size=batch_size,
        random_rotation_prob=0.5,
        H_prob=1.0,
        enable_cutoff=False,
        tree_delete_ratio=0.0,
        tree_delete_start_layer=1,
        seed=42,
    )
